"""Utilities for writing CIF files from AllAtomStructure with entity metadata."""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import biotite.structure as bst
import biotite.structure.io.pdbx as pdbx
import numpy as np
import torch

from bioemu2.data_interface.const import BondType, MolType
from bioemu2.data_interface.ligand_validators import ALL_CCD_RESIDUES
from bioemu2.data_interface.structure_hierarchy import Branched, Ligand, _Polymer
from bioemu2.structure.inference_inputs import topology_from_structure

if TYPE_CHECKING:
    from bioemu2.data_interface import System
    from bioemu2.structure import AllAtomStructure, ChainMetadata

logger = logging.getLogger(__name__)


class EntityType(Enum):
    POLYMER = "polymer"
    NON_POLYMER = "non-polymer"
    BRANCHED = "branched"


class PolymerType(Enum):
    POLYPEPTIDE_L = "polypeptide(L)"
    RNA = "polyribonucleotide"
    DNA = "polydeoxyribonucleotide"


# Mapping from MolType to mmCIF _entity.type
_MOLTYPE_TO_ENTITY_TYPE: dict[MolType, EntityType] = {
    MolType.PROTEIN: EntityType.POLYMER,
    MolType.RNA: EntityType.POLYMER,
    MolType.DNA: EntityType.POLYMER,
    MolType.NONPOLYMER: EntityType.NON_POLYMER,
}

# Mapping from MolType to mmCIF _entity_poly.type
_MOLTYPE_TO_POLY_TYPE: dict[MolType, PolymerType] = {
    MolType.PROTEIN: PolymerType.POLYPEPTIDE_L,
    MolType.RNA: PolymerType.RNA,
    MolType.DNA: PolymerType.DNA,
}

_BOND_TYPE_TO_BIOTITE: dict[BondType, bst.BondType] = {
    BondType.OTHER_OR_UNK: bst.BondType.ANY,
    BondType.SINGLE: bst.BondType.SINGLE,
    BondType.DOUBLE: bst.BondType.DOUBLE,
    BondType.TRIPLE: bst.BondType.TRIPLE,
    BondType.AROMATIC: bst.BondType.AROMATIC,
}


@torch.no_grad()
def save_structure_to_cif(
    structure: AllAtomStructure,
    path: str | os.PathLike,
    renamed_asym_ids: Sequence[str] | None = None,
    rename_residues: dict[int, str] | None = None,
    system: System | None = None,
    asym_id_to_chain_metadata: dict[str, ChainMetadata] | None = None,
) -> None:
    """Save an AllAtomStructure to a CIF file with proper entity information.

    Builds a biotite ``AtomArray`` from the structure and writes it directly to
    mmCIF format.  When *system* and *asym_id_to_chain_metadata* are
    provided, the CIF is enriched with ``_entity``, ``_entity_poly``, and
    ``_struct_asym`` categories so that downstream evaluation tools (e.g.
    OpenStructure) can correctly identify chain types.

    Args:
        structure: The AllAtomStructure to write.
        path: Output CIF file path.
        renamed_asym_ids: Chain identifiers to use in the output.
        rename_residues: Mapping of residue indices to custom residue names.
            Unmapped single-residue ligands retain recognized CCD labels; others
            use LIG.
        system: System defining the molecular system.  When
            provided together with *asym_id_to_chain_metadata*, entity
            metadata tables are added to the CIF.
        asym_id_to_chain_metadata: Mapping from chain asym_id strings to
            ChainMetadata (as returned by ``get_asym_id_to_chain_metadata``).
    """
    path = Path(path)
    topology = topology_from_structure(
        structure,
        renamed_asym_ids=renamed_asym_ids,
        rename_residues=rename_residues,
    )

    n_atoms = structure.element.shape[0]
    atoms = bst.AtomArray(n_atoms)
    atoms.coord = structure.coords.detach().cpu().numpy()

    residue_to_chain = dict(zip(structure.residue_idx.tolist(), structure.allatom_asym_id.tolist()))
    single_residue_chains = {
        chain_id for chain_id, count in Counter(residue_to_chain.values()).items() if count == 1
    }

    for i, atom in enumerate(topology.atoms):
        atoms.chain_id[i] = atom.residue.chain.chain_id or str(atom.residue.chain.index)
        atoms.res_id[i] = atom.residue.resSeq
        residue_name = atom.residue.name
        if (
            (rename_residues is None or structure.residue_idx[i].item() not in rename_residues)
            and structure.mol_type[i].item() == MolType.NONPOLYMER.value
            and structure.allatom_asym_id[i].item() in single_residue_chains
            and (residue_name == "UNK" or residue_name not in ALL_CCD_RESIDUES)
        ):
            residue_name = "LIG"
        atoms.res_name[i] = residue_name
        atoms.atom_name[i] = atom.name
        atoms.element[i] = atom.element.symbol

    if topology.n_bonds > 0:
        explicit_bond_types = {
            tuple(sorted((int(atom1), int(atom2)))): _BOND_TYPE_TO_BIOTITE[BondType(int(value))]
            for (atom1, atom2), value in zip(
                structure.bond_index.T.tolist(), structure.bond_type.tolist(), strict=True
            )
        }
        atoms.bonds = bst.BondList(
            n_atoms,
            np.array(
                [
                    (
                        atom1.index,
                        atom2.index,
                        explicit_bond_types.get(
                            tuple(sorted((atom1.index, atom2.index))), bst.BondType.ANY
                        ).value,
                    )
                    for atom1, atom2 in topology.bonds
                ],
                dtype=int,
            ),
        )

    cif_file = pdbx.CIFFile()
    data_block = path.stem
    pdbx.set_structure(cif_file, atoms, data_block=data_block)
    charges = structure.charge.detach().cpu().numpy()
    if (
        charges.shape != (n_atoms,)
        or not np.all(np.isfinite(charges))
        or not np.all(charges == np.round(charges))
    ):
        raise ValueError("Atom formal charges must be finite integers, one per output atom.")
    cif_file[data_block]["atom_site"]["pdbx_formal_charge"] = [
        str(int(charge)) for charge in charges
    ]

    if system is not None and asym_id_to_chain_metadata is not None:
        _add_entity_metadata(cif_file[data_block], system, asym_id_to_chain_metadata)

    cif_file.write(str(path))
    logger.info("Saved CIF to %s", path)


def _add_entity_metadata(
    block: pdbx.CIFBlock,
    system: System,
    asym_id_to_chain_metadata: dict[str, ChainMetadata],
) -> None:
    """Add ``_entity``, ``_entity_poly``, and ``_struct_asym`` categories to *block*."""
    entity_ids: list[str] = []
    entity_types: list[str] = []
    poly_entity_ids: list[str] = []
    poly_types: list[str] = []
    poly_seqs: list[str] = []
    nonpoly_entity_ids: list[str] = []
    nonpoly_ccd_codes: list[str] = []

    for idx, entity in enumerate(system.entities):
        eid = str(idx + 1)
        entity_ids.append(eid)
        entity_types.append(
            EntityType.BRANCHED.value
            if isinstance(entity, Branched)
            else _MOLTYPE_TO_ENTITY_TYPE[entity.ENTITY_TYPE].value
        )

        if isinstance(entity, _Polymer):
            poly_entity_ids.append(eid)
            poly_types.append(_MOLTYPE_TO_POLY_TYPE[entity.ENTITY_TYPE].value)
            poly_seqs.append(entity.sequence)
        elif isinstance(entity, Ligand) and entity.ccd_code is not None:
            nonpoly_entity_ids.append(eid)
            nonpoly_ccd_codes.append(entity.ccd_code)

    asym_id_to_entity_id: dict[str, str] = {
        asym_id: str(meta.entity_id + 1) for asym_id, meta in asym_id_to_chain_metadata.items()
    }

    block["entity"] = pdbx.CIFCategory({"id": entity_ids, "type": entity_types})

    if poly_entity_ids:
        block["entity_poly"] = pdbx.CIFCategory(
            {
                "entity_id": poly_entity_ids,
                "type": poly_types,
                "pdbx_seq_one_letter_code_can": poly_seqs,
            }
        )

    if nonpoly_entity_ids:
        block["pdbx_entity_nonpoly"] = pdbx.CIFCategory(
            {
                "entity_id": nonpoly_entity_ids,
                "comp_id": nonpoly_ccd_codes,
            }
        )

    struct_asym_ids = list(asym_id_to_entity_id.keys())
    struct_entity_ids = [asym_id_to_entity_id[a] for a in struct_asym_ids]
    block["struct_asym"] = pdbx.CIFCategory(
        {
            "id": struct_asym_ids,
            "entity_id": struct_entity_ids,
        }
    )

    if "atom_site" in block:
        label_asym_col = block["atom_site"]["label_asym_id"].as_array(str)
        entity_id_col = np.array(
            [asym_id_to_entity_id.get(asym_id, "?") for asym_id in label_asym_col]
        )
        block["atom_site"]["label_entity_id"] = entity_id_col
