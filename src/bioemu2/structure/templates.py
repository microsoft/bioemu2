"""Residue and ligand templates from CCD, InChI, and RDKit molecules."""

import hashlib
import json
import logging
import os
import uuid
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import mdtraj
import numpy as np
import requests
import torch
from pdbecif.mmcif import CIFWrapper
from pdbecif.mmcif_io import CifFileReader
from rdkit.Chem import AddHs, Mol, MolFromInchi, MolFromSmiles, RemoveAllHs, SDMolSupplier
from rdkit.Chem.AllChem import (
    CanonicalRankAtoms,
    EmbedMolecule,
    ETKDGv3,
    UFFOptimizeMolecule,
)

from bioemu2.data_interface.const import (
    BondType,
    get_bondtype_from_pdbx_bondtype,
    get_bondtype_from_rdkit_bondtype,
)
from bioemu2.paths import ASSETS_DIR
from bioemu2.structure import AllAtomStructure, encode_atom_name

logger = logging.getLogger(__name__)


def is_leaving_atom_donor(
    ccd_code: str,
    bond_atom_names: Collection[str],
) -> bool:
    """Check whether a residue donates a leaving atom in a bond.

    Returns ``True`` if any of *bond_atom_names* is adjacent (in the CCD bond
    graph) to a heavy leaving atom for the given *ccd_code*.

    Args:
        ccd_code:  CCD component code.
        bond_atom_names: Atom names involved in the bond (e.g., ``["C1"]``).

    Returns:
        ``True`` if leaving atoms should be removed for this residue.
    """
    if isinstance(bond_atom_names, str):
        bond_atom_names = [bond_atom_names]

    bond_atom_set = set(bond_atom_names)
    if not bond_atom_set:
        return False

    cif_wrapper = _load_ccd_cifwrapper(ccd_code)
    atom_block = cif_wrapper._chem_comp_atom
    bond_block = cif_wrapper._chem_comp_bond

    atom_ids = atom_block.atom_id
    leaving_flags = atom_block.pdbx_leaving_atom_flag
    elements = atom_block.type_symbol

    heavy_leaving = {
        aid
        for aid, flag, elem in zip(atom_ids, leaving_flags, elements)
        if flag == "Y" and elem != "H"
    }
    if not heavy_leaving or bond_block is None:
        return False

    for a1, a2 in zip(bond_block.atom_id_1, bond_block.atom_id_2):
        if (a1 in bond_atom_set and a2 in heavy_leaving) or (
            a2 in bond_atom_set and a1 in heavy_leaving
        ):
            return True

    return False


_CCD_URL = "https://files.rcsb.org/ligands/download/{ccd_code}.cif"


def _get_ccd_path(ccd_code: str) -> Path:
    """Return a locally cached public CCD component file."""
    cache_root = Path(
        os.environ.get("BIOEMU2_CCD_CACHE", Path.home() / ".cache" / "bioemu2" / "ccd")
    ).expanduser()
    path = cache_root / f"{ccd_code}.cif"
    if path.exists():
        return path

    response = requests.get(_CCD_URL.format(ccd_code=ccd_code), timeout=60)
    if response.status_code == 404:
        raise CCDTemplateError(f"CCD component {ccd_code!r} was not found at RCSB")
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        partial.write_bytes(response.content)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


@lru_cache(maxsize=1024)
def _load_ccd_cifwrapper(ccd_code: str) -> CIFWrapper:
    """Load and parse a public CCD component CIF file.

    Raises:
        CCDTemplateError: If the CIF file for *ccd_code* is not found.
    """
    try:
        path = _get_ccd_path(ccd_code)
        parser = CifFileReader()
        cif_obj = parser.read(str(path), output="cif_wrapper")
        return list(cif_obj.values())[0]
    except FileNotFoundError as e:
        raise CCDTemplateError(f"CCD CIF for {ccd_code} was not found") from e


def template_from_cifwrapper(
    cif_wrapper: CIFWrapper, remove_leaving_atoms: bool = False
) -> AllAtomStructure:
    """Extract structure template information from a ``CIFWrapper`` object.

    The ``CIFWrapper`` object is usually obtained from the chemical compound dictionary
    (https://www.wwpdb.org/data/ccd). Template information includes atom names, atomic numbers,
    template positions, template bonds (edges) and template atom charges.

    Args:
        cif_wrapper: ``CIFWrapper`` object.
        remove_leaving_atoms: Whether to remove heavy atoms annotated with
            ``pdbx_leaving_atom_flag == 'Y'``. Warns if more than one heavy
            leaving atom is removed.

    Returns:
        ``AllAtomStructure`` template.
    """
    atom_block = cif_wrapper._chem_comp_atom
    if "X" in atom_block.type_symbol:
        logger.warning(
            "CCD %s contains an unknown element X; representing it as a virtual site.",
            cif_wrapper._chem_comp.id[0],
        )
    elements = np.array(
        [
            mdtraj.element.virtual_site.number
            if symbol == "X"
            else mdtraj.element.Element._elements_by_symbol[symbol.upper()].number
            for symbol in atom_block.type_symbol
        ]
    )
    atom_ids = atom_block.atom_id
    # NOTE: position units in CCD components are Angstrom.
    try:
        positions = np.array(
            [
                atom_block.pdbx_model_Cartn_x_ideal,
                atom_block.pdbx_model_Cartn_y_ideal,
                atom_block.pdbx_model_Cartn_z_ideal,
            ],
            dtype=float,
        ).T
    except ValueError:
        # cif_wrapper._chem_comp_atom.pdbx_model_Cartn_{}_ideal can sometimes contain missing
        # coordinates. These are usually encoded as '?'. Sometimes, these can be present
        # in the model_Cartn_{} field so we attempt to fetch those as a fallback
        logger.warning(
            f"Missing PDBx coordinate from CIF wrapper with comp_id {cif_wrapper._chem_comp.id[0]}. Attempting alternative coordinates."
        )
        positions = np.array(
            [
                atom_block.model_Cartn_x,
                atom_block.model_Cartn_y,
                atom_block.model_Cartn_z,
            ],
            dtype=float,
        ).T

    if cif_wrapper._chem_comp_bond is not None:
        edge_index = np.array(
            [
                [atom_ids.index(i) for i in cif_wrapper._chem_comp_bond.atom_id_1],
                [atom_ids.index(i) for i in cif_wrapper._chem_comp_bond.atom_id_2],
            ]
        )
        bond_order = cif_wrapper._chem_comp_bond.value_order
        is_bond_arom = cif_wrapper._chem_comp_bond.pdbx_aromatic_flag

        # if is_bond_arom is not boolean
        if isinstance(is_bond_arom[0], str):
            is_bond_arom = np.array([b == "Y" for b in is_bond_arom], dtype=bool)

        bond_type = np.array(
            [
                (
                    BondType.AROMATIC.value
                    if is_arom
                    else get_bondtype_from_pdbx_bondtype(border).value
                )
                for border, is_arom in zip(bond_order, is_bond_arom, strict=True)
            ],
            dtype=int,
        )
    else:
        # Single-atom ligand
        edge_index = np.empty((2, 0), dtype=int)
        bond_type = np.array([], dtype=int)

    atom_charges = np.array(
        [0.0 if charge in {"?", "."} else float(charge) for charge in atom_block.charge]
    )

    # Build atom keep mask: always filter hydrogens, optionally filter leaving atoms.
    keep = elements != 1
    if remove_leaving_atoms:
        leaving_flags = atom_block.pdbx_leaving_atom_flag
        is_not_leaving = np.array([flag != "Y" for flag in leaving_flags], dtype=bool)
        heavy_leaving_atoms = np.array(atom_ids)[keep & ~is_not_leaving]
        if len(heavy_leaving_atoms) > 1:
            logger.warning(
                "Removing %d heavy leaving atoms (%s) for component %s according to "
                "pdbx_leaving_atom_flag. This may result in incomplete templates.",
                len(heavy_leaving_atoms),
                ", ".join(heavy_leaving_atoms),
                cif_wrapper._chem_comp.id[0],
            )
        keep = keep & is_not_leaving

    # Remap atom indices after filtering.
    old_to_new = np.full(len(atom_ids), -1, dtype=int)
    old_to_new[keep] = np.arange(keep.sum())

    # Filter bonds: keep only bonds where both atoms survive.
    if edge_index.size > 0:
        bond_keep = keep[edge_index[0]] & keep[edge_index[1]]
        edge_index_filtered = old_to_new[edge_index[:, bond_keep]]
        bond_type_filtered = bond_type[bond_keep]
    else:
        edge_index_filtered = np.empty((2, 0), dtype=int)
        bond_type_filtered = np.array([], dtype=int)

    atom_ids_kept = np.array(atom_ids)[keep]
    positions_kept = positions[keep]
    positions_kept -= np.mean(positions_kept, axis=0)
    elements_kept = elements[keep]
    charges_kept = atom_charges[keep]
    name_encoding = np.array([encode_atom_name(aid) for aid in atom_ids_kept])

    return AllAtomStructure(
        num_nodes=int(keep.sum()),
        name_encoding=torch.from_numpy(name_encoding).long(),
        element=torch.from_numpy(elements_kept).long(),
        template_coords=torch.from_numpy(positions_kept).float(),
        bond_index=torch.from_numpy(edge_index_filtered).long(),
        bond_type=torch.from_numpy(bond_type_filtered).long(),
        charge=torch.from_numpy(charges_kept).float(),
    )


_POLYMER_TEMPLATE_PATH = ASSETS_DIR / "polymer_templates.pt"


@lru_cache
def load_polymer_templates() -> dict[str, AllAtomStructure]:
    """Load the pre-packaged polymer template cache."""
    if not _POLYMER_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Polymer template cache not found at {_POLYMER_TEMPLATE_PATH}. "
            "This file should be packaged with BioEmu2."
        )
    return torch.load(_POLYMER_TEMPLATE_PATH, weights_only=False)


def _assign_mol_atom_names(mol: Mol) -> None:
    """
    Assign canonical atom names to ligand atoms by RDKit rank. Operates in-place.
    """
    canonical_order = CanonicalRankAtoms(mol)
    element_counter: Counter = Counter()
    for can_idx in canonical_order:
        atom = mol.GetAtomWithIdx(can_idx)
        element = atom.GetSymbol().upper()
        element_counter[element] += 1
        atom_name = f"{element}{element_counter[element]}"
        assert len(atom_name) <= 4, "Canonical atom name cannot be longer than 4 chars."
        atom.SetProp("name", atom_name)


def _template_from_rdmol(mol: Mol, max_iters: int = 100, **kwargs) -> AllAtomStructure:
    """
    Generates a ligand template from an RDKit Mol object

    Args:
        mol: RDKit Mol object representing the molecule.
        max_iters: Maximum number of iterations during conformer generation.
        kwargs: passed through to RDKit's EmbedMolecule function.
    """

    generator = ETKDGv3()
    mol_w_hs = AddHs(mol)  # Add hydrogens before optim.
    _assign_mol_atom_names(mol_w_hs)  # Assign atom names
    conf_id = EmbedMolecule(mol_w_hs, generator, **kwargs)  # Get initial conformer
    if conf_id == -1:
        # Distance-geometry embedding can fail for complex molecules (e.g. large
        # tannins like GGP). Fall back to random coordinates + UFF optimization.
        mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"{mol.GetNumAtoms()} atoms"
        logger.warning(f"Default embedding failed for {mol_id}, retrying with useRandomCoords=True")
        generator.useRandomCoords = True
        conf_id = EmbedMolecule(mol_w_hs, generator, **kwargs)
    if conf_id == -1:
        raise RuntimeError("rdkit failed to generate conformer")
    # Optimize conformer with UFF
    is_not_converged = bool(UFFOptimizeMolecule(mol_w_hs, confId=conf_id, maxIters=max_iters))

    if is_not_converged:
        logger.warning("UFF optimization not converged, consider increasing max_iters arg.")

    mol_wo_hs = RemoveAllHs(mol_w_hs)  # Remove all hydrogens after optim.
    num_atoms = mol_wo_hs.GetNumAtoms()
    conformer = mol_wo_hs.GetConformer(conf_id)
    template_coords = np.array(
        [conformer.GetAtomPosition(idx) for idx in range(num_atoms)], dtype=np.float32
    )
    template_bonds = [bond for bond in mol_wo_hs.GetBonds()]
    if template_bonds:
        edge_index = np.array(
            [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in template_bonds],
            dtype=int,
        ).T
    else:
        edge_index = np.empty((2, 0), dtype=int)
    bond_type = np.array(
        [get_bondtype_from_rdkit_bondtype(bond.GetBondType()).value for bond in template_bonds],
        dtype=int,
    )

    name_encoding = np.array(
        [encode_atom_name(n) for n in [a.GetProp("name") for a in mol_wo_hs.GetAtoms()]],
        dtype=np.int8,
    )
    elements = np.array([a.GetAtomicNum() for a in mol_wo_hs.GetAtoms()], dtype=np.int64)
    charges = np.array([a.GetFormalCharge() for a in mol_wo_hs.GetAtoms()], dtype=np.float32)
    return AllAtomStructure(
        num_nodes=num_atoms,
        template_coords=torch.from_numpy(template_coords),
        bond_index=torch.from_numpy(edge_index),
        bond_type=torch.from_numpy(bond_type),
        name_encoding=torch.from_numpy(name_encoding),
        element=torch.from_numpy(elements),
        charge=torch.from_numpy(charges),
    )


def _template_from_inchi(inchi: str, max_iters: int = 100, **kwargs) -> AllAtomStructure:
    """
    Generates a ligand template from an InChi string

    Args:
        inchi: InChi string
        max_iters: Maximum number of iterations during conformer generation
    """
    mol = MolFromInchi(inchi)
    if mol is None:
        raise ValueError(f"Failed to parse InChI: {inchi}")
    return _template_from_rdmol(mol, max_iters=max_iters, **kwargs)


class CCDTemplateError(Exception):
    pass


def _save_template_to_npz(template: AllAtomStructure, output_file: Path) -> None:
    assert template.template_coords is not None  # Narrow Optional for mypy.

    np.savez_compressed(
        output_file,
        template_coords=template.template_coords.numpy(),
        bond_index=template.bond_index.numpy(),
        bond_type=template.bond_type.numpy(),
        name_encoding=template.name_encoding.numpy(),
        element=template.element.numpy(),
        charge=template.charge.numpy(),
    )


def _ligand_cache_root() -> Path:
    return Path(
        os.environ.get("BIOEMU2_TEMPLATE_CACHE", Path.home() / ".cache" / "bioemu2" / "templates")
    ).expanduser()


def _get_or_create_ligand_npz(key_fields: dict, create_fn) -> Path:
    """Return path to a cached template.npz, creating it if absent."""
    fp = hashlib.md5(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()
    cache_dir = _ligand_cache_root() / fp
    npz_path = cache_dir / "template.npz"
    if not npz_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        create_fn(cache_dir)
    return npz_path


@dataclass(frozen=True)
class LigandTemplateCache:
    """Cached ligand template with on-demand generation via RDKit.

    Use the classmethods to construct from different sources::

        LigandTemplateCache.from_ccd("ATP")
        LigandTemplateCache.from_inchi("InChI=1S/...")
        LigandTemplateCache.from_smiles("CCO")
        LigandTemplateCache.from_sdf("/path/to/ligand.sdf")
    """

    source: str
    kind: Literal["ccd", "inchi", "smiles", "sdf"]
    max_iters: int = 1000

    @classmethod
    def from_ccd(cls, ccd_code: str, *, max_iters: int = 1000) -> "LigandTemplateCache":
        return cls(source=ccd_code, kind="ccd", max_iters=max_iters)

    @classmethod
    def from_inchi(cls, inchi: str, *, max_iters: int = 1000) -> "LigandTemplateCache":
        return cls(source=inchi, kind="inchi", max_iters=max_iters)

    @classmethod
    def from_smiles(cls, smiles: str, *, max_iters: int = 1000) -> "LigandTemplateCache":
        return cls(source=smiles, kind="smiles", max_iters=max_iters)

    @classmethod
    def from_sdf(cls, sdf_path: str, *, max_iters: int = 1000) -> "LigandTemplateCache":
        return cls(source=sdf_path, kind="sdf", max_iters=max_iters)

    @property
    def npz_file(self) -> Path:
        def _create(output_dir: Path) -> None:
            if self.kind == "ccd":
                template = template_from_cifwrapper(_load_ccd_cifwrapper(self.source))
            elif self.kind == "inchi":
                template = _template_from_inchi(self.source, max_iters=self.max_iters)
            elif self.kind == "smiles":
                mol = MolFromSmiles(self.source)
                if mol is None:
                    raise ValueError(f"Failed to parse SMILES: {self.source}")
                template = _template_from_rdmol(mol, max_iters=self.max_iters)
            elif self.kind == "sdf":
                supplier = SDMolSupplier(self.source, removeHs=True)
                try:
                    mol = next(iter(supplier))
                except StopIteration as exc:
                    raise ValueError(f"Failed to parse SDF: {self.source}") from exc
                if mol is None:
                    raise ValueError(f"Failed to parse SDF: {self.source}")
                template = _template_from_rdmol(mol, max_iters=self.max_iters)
            else:
                raise ValueError(f"Unknown kind: {self.kind}")
            _save_template_to_npz(template, output_file=output_dir / "template.npz")

        return _get_or_create_ligand_npz(
            key_fields={
                "kind": self.kind,
                "source": self.source,
                "max_iters": self.max_iters,
            },
            create_fn=_create,
        )
