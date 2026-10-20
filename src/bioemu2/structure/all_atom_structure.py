"""Tensor-based all-atom structure representations and system conversion."""

import copy
import typing
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, Self

import numpy as np
import torch
from biotite.structure import AtomArray
from rdkit import Chem

from bioemu2.conditioning.contact import contact_conditioning_info
from bioemu2.conditioning.method import SamplingMode, method_types_ids
from bioemu2.data_interface.const import MolType, Token
from bioemu2.data_interface.structure_hierarchy import (
    Bond,
    Branched,
    Contact,
    ContactToken,
    Ligand,
    Polymer,
)
from bioemu2.structure.scatter import scatter

if TYPE_CHECKING:
    from bioemu2.data_interface.system import System

type ModificationKind = Literal["ccd", "inchi"]
type PolymerModification = tuple[int, ModificationKind, str]


class TensorStore:
    """Dynamic attribute container with dict-like access and device movement."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def __getattr__(self, key: str) -> Any:
        # Only invoked when normal lookup fails (fields live in __dict__). Enables
        # transforms to attach dynamic fields (e.g. `pocket_atom_mask`) and tells
        # the type checker these containers accept arbitrary attributes, while
        # declared annotations keep their static types.
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        # Mirror __getattr__: allow transforms to attach dynamic fields and tell
        # the type checker these containers accept arbitrary attribute assignment.
        self.__dict__[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __getitem__(self, key: str) -> Any:
        try:
            return self.__dict__[key]
        except KeyError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow dict of all stored fields."""
        return dict(self.__dict__)

    def save(self, file: str | Path | BinaryIO) -> None:
        """Save fields without pickling the tensor container's Python class."""
        torch.save(self.to_dict(), file)

    @classmethod
    def from_dict(cls, fields: dict[str, Any]) -> Self:
        """Construct this tensor container from its stored fields."""
        if not all(isinstance(key, str) for key in fields):
            raise TypeError("Expected string keys for TensorStore fields")
        return cls(**fields)

    @classmethod
    def load(
        cls,
        file: str | Path | BinaryIO,
        map_location: str | torch.device | None = None,
    ) -> Self:
        """Load fields with weights-only unpickling and construct this container class."""
        fields = torch.load(file, map_location=map_location, weights_only=True)
        if not isinstance(fields, dict):
            raise TypeError("Expected a dictionary of TensorStore fields")
        return cls.from_dict(fields)

    def get(self, key: str, default: Any = None) -> Any:
        return self.__dict__.get(key, default)

    def update(self, mapping: dict[str, Any]) -> Self:
        """Update fields in-place from a dict, returning self."""
        self.__dict__.update(mapping)
        return self

    def replace(self, **kwargs: Any) -> Self:
        """Return a shallow copy with the given fields overridden."""
        out = self.__class__.__new__(self.__class__)
        out.__dict__.update(self.__dict__)
        out.__dict__.update(kwargs)
        return out

    def clone(self) -> Self:
        """Return a deep copy (tensors cloned, other values shallow-copied)."""
        out = self.__class__.__new__(self.__class__)
        out.__dict__.update(
            {
                k: v.clone() if isinstance(v, torch.Tensor) else copy.copy(v)
                for k, v in self.__dict__.items()
            }
        )
        return out

    def to(self, device: Any) -> Self:
        """Move all tensor fields to the given device, returning a new instance."""
        out = self.__class__.__new__(self.__class__)
        out.__dict__.update(
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in self.__dict__.items()
            }
        )
        return out

    def cpu(self) -> Self:
        return self.to("cpu")


class AllAtomStructure(TensorStore):
    """
    Basic data structure for all atom systems.

    Construct inference inputs from a System with ``from_system``.

    All length-bearing fields (`coords`, `template_coords`) are in angstroms.

    Attributes:
        coords: Cartesian coordinates (in Angstrom). Shape [num_atoms, 3] or [num_atoms, num_frames, 3].
        element: Element chemical type. [0-128]. Shape [num_atoms].
        residue_idx: Increasing residue index assigning an atom to its residue. Does not reset after
          a chain and should be unique for each `asym_id`. Shape [num_atoms].
        residue_type: Residue type, shape [num_atoms].
        num_nodes: Number of atoms.
        num_residues: Number of residues.
        num_tokens: Number of tokens. This can deviate from `num_residues` if ligands are present,
          since each ligand atom is counted as a token, while the ligand is a single residue.
        edge_index: Indices of inter atom edges to consider in model. Shape: [2, n_edges].
        edge_is_bonded: Whether atoms indicated by edge share a bond. Shape: [n_edges,1].
        template_coords: Cartesian coordinates of template atoms (in Angstrom). Shape [BS, num_atoms, 3].
          These are CCD reference positions, not structure-prediction templates.
          They can be accessed from the feature dictionary via `feats["ref_pos"]`
          in the score model.
        bond_index: Indices of atoms involved in a bond. Shape [2, num_bonds].
        name_encoding: ASCII atom-name encoding. Defined as ord(c) - 32, padded to length 4.
          Shape [num_atoms, 4]
        charge: Formal chemical charge of atom. Shape [num_atoms].
        atom_mask: Whether each atom participates in coordinate operations. When
          sampling from a System, every topology atom participates. Shape [num_atoms].
        asym_id: Asymmetric unit identifier (i.e., unique chain id). Shape [num_tokens].
        entity_id: Entity identifier (i.e unique index for each individual sequence). Shape [num_tokens].
        sym_id: Unique index within chains of this sequence. Shape [num_tokens].
        mol_type: Type of molecule this atom belongs to. Shape [num_atoms]. Possible types are:
          (0: protein, 1: rna, 2:dna, 3: ligand)
        sequence: One-letter token sequences associated with this graph. Asym IDs. separated
          by colon.
        system_id: Optional identifier (e.g., PDB Id.)
        token_center_atom_index: Index mapping tokens to their center atom index [num_tokens].
        token_idx_at_atom: Index useful for mapping token-level to atom-level properties [num_atoms]
        distogram_index: Index mapping atom dimension with shape [num_tokens]. This differs slightly
          from the `token_center_atom_index` since e.g. the CB atoms are used for distogram computation instead
          of the CA which form the token centers.
        interfaces: Optional array for defining preferred chain interfaces for subsampling. Chains
          are indicated by their `asym_id`. Shape is [num_interfaces x 2], where the first column is
          the index of the first chain in the interface and the second the index of the second.
        residue_labels: List of PDB residue label strings for each atoms. Only used for outputs,
          does not affect training.
        coords_cond: Optional conditioning coordinates for MD-sequential (ITO) training. When
          present, ``coords`` is the diffusion target at time ``t + dt`` and ``coords_cond`` is the
          conditioning frame at the earlier time ``t`` (in angstroms). ``None`` for unconditioned
          (Boltzmann) samples. Shape same as coords.
        dt: Optional time lag in picoseconds between ``coords_cond`` and ``coords``. Shape [1] for
          one shared lag or [num_frames] when coords already carries a frame dimension.
    """

    coords: torch.Tensor
    element: torch.LongTensor
    residue_idx: torch.LongTensor
    residue_type: torch.LongTensor
    num_nodes: int  # Number of atoms.
    num_residues: int  # Number of residues. A ligand is counted as one residue.
    num_tokens: int  # Number of tokens (different from `num_residues` in presence of ligands).
    edge_index: torch.LongTensor  # Shape: (2, n_edges), n_edges = n_atom^2
    # Bonding information.
    edge_is_bonded: torch.Tensor
    template_coords: torch.Tensor | None
    bond_index: torch.LongTensor
    bond_type: torch.LongTensor
    # Atom reference features.
    name_encoding: torch.LongTensor
    charge: torch.LongTensor
    atom_mask: torch.BoolTensor
    asym_id: torch.LongTensor
    entity_id: torch.LongTensor
    sym_id: torch.LongTensor
    mol_type: torch.LongTensor  # [num_atoms]
    sequence: str
    system_id: str | None
    token_center_atom_index: torch.LongTensor
    token_idx_at_atom: torch.LongTensor
    distogram_index: torch.LongTensor
    interfaces: torch.LongTensor | None
    residue_labels: list[str]
    # Optional MD-sequential fields. Present only for ITO samples from ``DynamicsWeightedData``.
    coords_cond: torch.Tensor | None
    dt: torch.Tensor | None

    def upsample_to_atoms(self, token_tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Upsample a token-level tensor to atom-level by repeating the values of the token center atom.

        Args:
            token_tensor: Tensor of shape [..., num_tokens, ...] to upsample.
            dim: Dimension along which to repeat the values. (default: 0)

        Returns:
            Upsampled tensor of shape [..., num_atoms, ...].
        """
        return token_tensor.index_select(dim=dim, index=self.token_idx_at_atom)

    def downsample_to_tokens(self, atom_tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Downsample an atom-level tensor to token level by slicing along the atom indices of the
        token center atoms.

        Args:
            atom_tensor: Tensor of shape [..., num_atoms, ...] to downsample.
            dim: Dimension along which to extract the values. (default: 0)

        Returns:
            Downsampled tensor of shape [..., num_tokens, ...].
        """
        return atom_tensor.index_select(dim=dim, index=self.token_center_atom_index)

    def downsample_to_residues(self, atom_tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Downsample an atom-level tensor to residue level by scattering the values
        to the corresponding residue indices.

        Args:
            atom_tensor: Tensor of shape [..., num_atoms, ...] to downsample.
            dim: Dimension along which to scatter the values. (default: 0)

        Returns:
            Downsampled tensor of shape [..., num_residues, ...].
        """
        return scatter(atom_tensor, self.residue_idx, dim, reduce="mean")

    def name_encoding_is_atom(self, atom_name: str) -> torch.BoolTensor:
        """
        Compute a boolean mask indicating where the current `name_encoding` matches the provided
        atom name.

        Args:
            atom_name: String with atom name (e.g. "CA").

        Returns:
            Mask with true values at matching locations.
        """
        return get_atom_name_mask_for_name_encoding(self.name_encoding, atom_name)

    @property
    def allatom_asym_id(self) -> torch.LongTensor:
        """
        Get asymmetric unit identifier (i.e., unique chain id) at the atom level.

        Returns:
            Asymmetric unit identifier for each atom.
        """
        return typing.cast(torch.LongTensor, self.upsample_to_atoms(self.asym_id, dim=0))

    @classmethod
    def from_system(
        cls,
        system: "System",
        ref_struct_or_fpath: AtomArray | Path | str | None = None,
        *,
        mode: SamplingMode | None = None,
        contacts: list[Contact] | None = None,
    ) -> "AllAtomStructure":
        """
        Create an AllAtomStructure from a System containing the chain and residue information
        needed to construct the structure, but no coordinates.
        Optionally, when a reference structure file is provided, the coords from the reference
        structure will be loaded onto the output AllAtomStructure, which is suitable for
        overfitting or evaluation purposes.

        Args:
            system: The System instance.
            ref_struct_or_fpath: Optional path to a reference structure file (e.g., PDB or CIF) or
                an AtomArray instance. If provided, the reference coordinates will be parsed and
                mapped over to the corresponding atoms. The asym IDs in the reference structure
                must match those in the System. The checks are strict, so a ValueError will
                be raised when there are extra or mismatching residues and atoms from the reference
                structure.
            mode: Optional sampling mode applied to every token.
            contacts: Optional contact conditioning validated against the System.

        Returns:
            An AllAtomStructure representing the data in the System.
        """
        return _get_structure_from_system(
            system,
            ref_struct_or_fpath=ref_struct_or_fpath,
            mode=mode,
            contacts=contacts,
        )

    @property
    def atom_names(self) -> list[str]:
        """Utility for getting the decoded atom name strings (mainly for analysis)."""
        return [decode_atom_name(an) for an in self.name_encoding.tolist()]

    def __repr__(self) -> str:
        """String representation of the AllAtomStructure in pretty print format."""
        items = list(self.items())
        if not items:
            return "AllAtomStructure()"
        max_key_len = max(len(str(key)) for key, _ in items) + 3
        lines = []
        for key, value in sorted(items):
            if isinstance(value, torch.Tensor):
                lines.append(
                    f"{key:<{max_key_len}}: shape={str(tuple(value.shape)):<20} dtype={str(value.dtype):<20} device={str(value.device):<20}"
                )
            elif isinstance(value, list):
                lines.append(f"{key:<{max_key_len}}: {value[:2]}... (len={len(value)})")
            else:
                lines.append(f"{key:<{max_key_len}}: {value}")
        return "\n".join(lines)


def _get_ligand_structure(ligand: Ligand) -> AllAtomStructure:
    """Convert a Ligand entity to an AllAtomStructure."""
    # Deferred: these modules import AllAtomStructure from this module.
    from bioemu2.structure.inference_inputs import (  # noqa: PLC0415
        inference_input_structure_from_ccd,
        inference_input_structure_from_inchi,
    )

    if ligand.smiles is not None:
        mol = Chem.MolFromSmiles(ligand.smiles)
        return inference_input_structure_from_inchi(Chem.MolToInchi(mol))
    elif ligand.inchi is not None:
        return inference_input_structure_from_inchi(ligand.inchi)
    elif ligand.ccd_code is not None:
        return inference_input_structure_from_ccd(ligand.ccd_code)
    else:
        # this won't happen due to the validator
        raise ValueError("Ligand entity must contain either 'smiles', 'inchi', or 'ccd_code' key.")


@dataclass(frozen=True)
class ChainMetadata:
    """Metadata for a chain in the AllAtomStructure."""

    entity_id: int
    structure: AllAtomStructure


def _get_branched_structure(
    branched: Branched, bonds: Collection[Bond] | None = None
) -> AllAtomStructure:
    """Convert a Branched entity to an AllAtomStructure.

    Delegates to :func:`~bioemu2.structure.inference_inputs.inference_input_structure_from_branched_entity`,
    which loads each monomer from its CCD template and concatenates them with intra-residue bonds
    only. Inter-residue (glycosidic) and cross-entity bonds are added later by
    :func:`_add_extra_covalent_bonds` from ``System.bonds``.

    Leaving-atom removal is derived from ``bonds``: for each residue that participates in a bond,
    :func:`~bioemu2.structure.templates.is_leaving_atom_donor` checks CCD bond-graph adjacency
    to decide whether the residue is a donor (e.g., C1 adjacent to leaving O1 in a glycosidic
    bond). Only donor residues have their leaving atoms removed.

    Args:
        branched: ``Branched`` entity definition with ``mon_seq`` of CCD codes.
        bonds: Optional list of ``Bond`` from ``System.bonds``. Used to derive which
            residues are donors (and thus need leaving-atom removal). If ``None`` or empty,
            no leaving atoms are removed.
    """
    # Deferred: these modules import AllAtomStructure from this module.
    from bioemu2.structure.inference_inputs import (  # noqa: PLC0415
        inference_input_structure_from_branched_entity,
    )
    from bioemu2.structure.templates import is_leaving_atom_donor  # noqa: PLC0415

    ccd_codes = [mod.ccd_code for mod in branched.mon_seq]
    num_residues = len(ccd_codes)

    # Resolve per-residue leaving-atom removal: check bond-donor adjacency
    # in the CCD bond graph for each residue that participates in a bond.
    remove_leaving_atoms: list[bool] = [False] * num_residues

    if bonds is not None:
        asym_set = set(branched.asym_ids)
        bond_atoms_per_residue: dict[int, list[str]] = {}
        for bond in bonds:
            for atom in (bond.atom1, bond.atom2):
                if atom.asym_id in asym_set:
                    res_idx = atom.res_id - 1  # convert 1-based to 0-based
                    if 0 <= res_idx < num_residues:
                        bond_atoms_per_residue.setdefault(res_idx, []).append(atom.name)

        for res_idx, atom_names in bond_atoms_per_residue.items():
            res_ccd = ccd_codes[res_idx]
            assert res_ccd is not None, (
                f"CCD code must be provided for branched monomers. Missing for residue index {res_idx}."
            )
            remove_leaving_atoms[res_idx] = is_leaving_atom_donor(res_ccd, atom_names)

    return inference_input_structure_from_branched_entity(
        ccd_codes=ccd_codes, remove_leaving_atoms=remove_leaving_atoms
    )


def _get_polymer_structure(polymer: Polymer) -> AllAtomStructure:
    """Convert a Polymer entity to an AllAtomStructure."""
    # Deferred: these modules import AllAtomStructure from this module.
    from bioemu2.structure.inference_inputs import (  # noqa: PLC0415
        inference_input_structure_from_dna_sequence,
        inference_input_structure_from_protein_sequence,
        inference_input_structure_from_rna_sequence,
    )
    from bioemu2.structure.templates import load_polymer_templates  # noqa: PLC0415

    mods_by_res_id = polymer.mods_by_res_id
    modifications: list[PolymerModification] = []
    for position, mod in mods_by_res_id.items():
        if mod.inchi is not None:
            modifications.append((position, "inchi", mod.inchi))
        elif mod.ccd_code is not None:
            modifications.append((position, "ccd", mod.ccd_code))

    # Methods for converting to AllAtomStructure representation
    CONVERT_POLYMER_TO_AAS = {
        MolType.PROTEIN: inference_input_structure_from_protein_sequence,
        MolType.RNA: inference_input_structure_from_rna_sequence,
        MolType.DNA: inference_input_structure_from_dna_sequence,
    }

    return CONVERT_POLYMER_TO_AAS[polymer.ENTITY_TYPE](
        sequence=polymer.sequence,
        template_cache=load_polymer_templates(),
        modifications=modifications,
    )


def get_asym_id_to_chain_metadata(
    system: "System",
) -> dict[str, ChainMetadata]:
    """
    Util to get a map between asym ids and ChainMetadata instances from a System
    """
    asym_id_to_chain_metadata: dict[str, ChainMetadata] = {}
    for sym_id, entity in enumerate(system.entities):
        if isinstance(entity, Ligand):
            parsed_chain = _get_ligand_structure(entity)
        elif isinstance(entity, Branched):
            parsed_chain = _get_branched_structure(entity, bonds=system.bonds)
        else:
            parsed_chain = _get_polymer_structure(entity)

        for asym_id in entity.asym_ids:
            asym_id_to_chain_metadata[asym_id] = ChainMetadata(
                entity_id=sym_id,
                structure=parsed_chain,
            )
    return asym_id_to_chain_metadata


def _add_extra_covalent_bonds(
    structure: AllAtomStructure,
    bonds: list[Bond],
    asym_id_to_chain_metadata: dict[str, ChainMetadata],
) -> AllAtomStructure:
    """
    Add extra covalent bonds defined in the System to the AllAtomStructure. This is needed for handling user-defined cross-links
    between chains, as well as covalently bound ligands

    Arguments:
        structure: The AllAtomStructure to which the extra bonds will be added.
        bonds: A list of Bond instances defining the extra covalent bonds to be added.
        asym_id_to_chain_metadata: A mapping from asym_id to ChainMetadata, which contains the entity_id and the corresponding AllAtomStructure
                                for each chain. This is used for locating the correct atom indices for the bond definitions.

    Returns:
        An updated AllAtomStructure with the extra covalent bonds added.
    """
    # Add additional bonds from system definition to joint structure
    letter_asym_id_to_num_asym_id = {
        asym_id: i for i, asym_id in enumerate(asym_id_to_chain_metadata)
    }

    asym_id_atom_shaped = structure.upsample_to_atoms(structure.asym_id)
    decoded_atom_names = np.array([decode_atom_name(nm.tolist()) for nm in structure.name_encoding])

    def _get_asym_id_specific_residue_idx(
        asym_id_atom_shaped: torch.Tensor, residue_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Get residue indices that are specific to each asym_id by resetting the residue index to 0
        """
        current_asym_id = asym_id_atom_shaped[0].item()
        asym_id_specific_residue_idx = []
        current_offset = 0
        for asym_id, global_res_idx in zip(asym_id_atom_shaped, residue_idx):
            if asym_id.item() != current_asym_id:
                current_offset = global_res_idx.item()
                current_asym_id = asym_id.item()
            asym_id_specific_residue_idx.append(global_res_idx.item() - current_offset)
        return torch.tensor(asym_id_specific_residue_idx, dtype=torch.long)

    asym_id_specific_residue_idx = _get_asym_id_specific_residue_idx(
        asym_id_atom_shaped, structure.residue_idx
    )

    extra_bond_idxs: list[tuple[int, int]] = []

    for bond in bonds:
        num_asym_id_1 = letter_asym_id_to_num_asym_id[bond.atom1.asym_id]
        num_asym_id_2 = letter_asym_id_to_num_asym_id[bond.atom2.asym_id]

        res_id_1 = bond.atom1.res_id - 1  # convert to 0-based
        res_id_2 = bond.atom2.res_id - 1  # convert to 0-based

        mask_atom_name_1 = torch.from_numpy(decoded_atom_names == bond.atom1.name)
        mask_atom_name_2 = torch.from_numpy(decoded_atom_names == bond.atom2.name)

        # Build boolean masks for selecting the correct atoms based on the asym_id, res_id, and atom names
        # of the atoms involved in the bond.
        mask_1 = (
            (asym_id_atom_shaped == num_asym_id_1)
            & (asym_id_specific_residue_idx == res_id_1)
            & mask_atom_name_1
        )
        mask_2 = (
            (asym_id_atom_shaped == num_asym_id_2)
            & (asym_id_specific_residue_idx == res_id_2)
            & mask_atom_name_2
        )

        if mask_1.sum() != 1 or mask_2.sum() != 1:
            raise ValueError(
                f"Bond definition {bond} does not match exactly one atom in the structure. Please check the definition and the reference structure (if provided) for consistency."
            )
        atom_idx_1 = torch.where(mask_1)[0][0].item()
        atom_idx_2 = torch.where(mask_2)[0][0].item()

        extra_bond_idxs.append((atom_idx_1, atom_idx_2))

    extra_bond_idxs_tensor = torch.tensor(
        extra_bond_idxs, dtype=torch.long
    ).t()  # [2, num_extra_bonds]
    extra_bond_type_tensor = torch.tensor(
        [bond.bond_type.value for bond in bonds], dtype=torch.long
    )  # [num_extra_bonds]

    # Add the extra bonds to the structure
    return structure.replace(
        bond_index=torch.cat([structure.bond_index, extra_bond_idxs_tensor], dim=1),
        bond_type=torch.cat([structure.bond_type, extra_bond_type_tensor], dim=0),
    )


def _add_contact_conditioning(
    structure: AllAtomStructure,
    contacts: list[Contact],
    asym_id_to_chain_metadata: dict[str, ChainMetadata],
) -> AllAtomStructure:
    """
    Resolve user-specified Contacts from a System into contact_conditioning /
    contact_threshold token-pair features, so that inference can be conditioned on
    desired contacts.

    Non-contact pairs are set to UNSELECTED (conditioning is active, but this pair is not a
    contact) when contacts are present, rather than UNSPECIFIED (conditioning not active at
    all).

    Arguments:
        structure: The AllAtomStructure to condition.
        contacts: A non-empty list of Contact instances defining the desired contacts.
        asym_id_to_chain_metadata: A mapping from asym_id to ChainMetadata, used for locating the
                                   correct token index for each ContactToken.

    Returns:
        An updated AllAtomStructure with contact_conditioning and contact_threshold set.
    """
    letter_asym_id_to_num_asym_id = {
        asym_id: i for i, asym_id in enumerate(asym_id_to_chain_metadata)
    }
    device = structure.coords.device
    num_tokens = structure.num_tokens

    asym_id_tok = structure.asym_id.to(torch.long)  # [num_tokens]
    residue_idx_tok_global = structure.residue_idx[structure.token_center_atom_index].to(
        torch.long
    )  # [num_tokens], increasing across chains (see AllAtomStructure.residue_idx docstring)
    decoded_atom_names_tok = np.array(
        [
            decode_atom_name(nm.tolist())
            for nm in structure.name_encoding[structure.token_center_atom_index]
        ]
    )

    # Reset residue_idx to be chain-local (0-based) by subtracting each chain's minimum.
    num_asym_ids = len(asym_id_to_chain_metadata)
    min_residue_idx_per_asym = torch.full(
        (num_asym_ids,), torch.iinfo(torch.long).max, dtype=torch.long, device=device
    )
    min_residue_idx_per_asym.scatter_reduce_(
        0, asym_id_tok, residue_idx_tok_global, reduce="amin", include_self=True
    )
    residue_idx_tok_local = residue_idx_tok_global - min_residue_idx_per_asym[asym_id_tok]

    def _resolve_token(token: ContactToken) -> int:
        num_asym_id = letter_asym_id_to_num_asym_id[token.asym_id]
        mask = asym_id_tok == num_asym_id
        mask = mask & (residue_idx_tok_local == token.res_id - 1)
        if token.atom_name is not None:
            # Ligands, branched monomers and modified residues can have one token per atom.
            atom_name_mask = torch.from_numpy(decoded_atom_names_tok == token.atom_name).to(device)
            mask = mask & atom_name_mask
        elif mask.sum() > 1:
            # A modified polymer residue has multiple tokens; use its backbone anchor.
            default_atom_mask = torch.from_numpy(np.isin(decoded_atom_names_tok, ("CA", "C1'"))).to(
                device
            )
            mask = mask & default_atom_mask

        if mask.sum() != 1:
            raise ValueError(
                f"Contact token {token} does not match exactly one token in the structure. Please "
                "check the definition and the reference structure (if provided) for consistency."
            )
        return int(torch.where(mask)[0][0].item())

    contact_conditioning = torch.full(
        (num_tokens, num_tokens),
        contact_conditioning_info["UNSELECTED"],
        dtype=torch.long,
        device=device,
    )
    contact_threshold = torch.zeros((num_tokens, num_tokens), dtype=torch.float32, device=device)

    for contact in contacts:
        i = _resolve_token(contact.token1)
        j = _resolve_token(contact.token2)
        contact_conditioning[i, j] = contact_conditioning_info["CONTACT"]
        contact_conditioning[j, i] = contact_conditioning_info["CONTACT"]
        contact_threshold[i, j] = contact.max_distance
        contact_threshold[j, i] = contact.max_distance

    contact_conditioning_oh = torch.nn.functional.one_hot(
        contact_conditioning, num_classes=len(contact_conditioning_info)
    ).to(torch.long)

    return structure.replace(
        contact_conditioning=contact_conditioning_oh,
        contact_threshold=contact_threshold,
    )


def add_mode_conditioning(structure: "AllAtomStructure", mode: str) -> "AllAtomStructure":
    """Label every token of ``structure`` with the given sampling mode."""
    # Public sampling modes map onto the training-provenance vocabulary.
    key = {"equilibrium": "md", "structure": "non-md"}.get(mode.lower(), mode.lower())
    if key not in method_types_ids:
        raise ValueError(f"mode={mode!r} is not a known mode (known: {sorted(method_types_ids)}).")
    method_feature = torch.full(
        (int(structure.num_tokens),),
        method_types_ids[key],
        dtype=torch.long,
        device=structure.coords.device,
    )
    return structure.update(dict(method_feature=method_feature))


@torch.no_grad()
def _get_structure_from_system(
    system: "System",
    ref_struct_or_fpath: AtomArray | Path | str | None = None,
    *,
    mode: SamplingMode | None = None,
    contacts: list[Contact] | None = None,
) -> AllAtomStructure:
    # Deferred: these modules import AllAtomStructure from this module.
    from bioemu2.structure.builder import StructureBuilder  # noqa: PLC0415
    from bioemu2.structure.ref_struct import (  # noqa: PLC0415
        ensure_loaded_ref_struct,
        get_chain_residue_maps_from_atom_array,
        merge_ref_struct_to_allatom_chain,
    )

    asym_id_to_chain_metadata = get_asym_id_to_chain_metadata(system)
    if ref_struct_or_fpath is not None:
        ref_pdb = ensure_loaded_ref_struct(ref_struct_or_fpath)
        chain_residue_maps = get_chain_residue_maps_from_atom_array(ref_pdb)
        if sorted(chain_residue_maps) != sorted(asym_id_to_chain_metadata):
            raise ValueError(
                f"Asym IDs in reference structure {sorted(chain_residue_maps)} do not match asym "
                f"IDs in system {sorted(asym_id_to_chain_metadata)}."
            )
    builder = StructureBuilder()

    for _asym_id, metadata in asym_id_to_chain_metadata.items():
        if ref_struct_or_fpath is not None:
            residue_map = chain_residue_maps[_asym_id]
            entity = system.entities[metadata.entity_id]
            chain_struct = merge_ref_struct_to_allatom_chain(
                metadata.structure,
                residue_map,
                ref_pdb,
                ligand_without_ccd=isinstance(entity, Ligand) and entity.ccd_code is None,
            )
        else:
            chain_struct = metadata.structure
        builder.add_chain(
            structure=chain_struct,
            entity_id=metadata.entity_id,
        )

    structure = builder.build()

    # Add extra covalent bonds to structure if present in system
    if system.bonds:
        structure = _add_extra_covalent_bonds(
            structure=structure,
            bonds=system.bonds,
            asym_id_to_chain_metadata=asym_id_to_chain_metadata,
        )

    if contacts:
        system.validate_contacts(contacts)
        structure = _add_contact_conditioning(
            structure=structure,
            contacts=contacts,
            asym_id_to_chain_metadata=asym_id_to_chain_metadata,
        )

    if mode is not None:
        structure = add_mode_conditioning(structure, mode)

    return structure


class BatchAllAtomStructure(TensorStore):
    """
    Basic data structure for dense all atom systems.

    In contrast to AllAtomStructure, this structure batches data in the leading dimension [BS, ...]
    instead of concatenating the data in a sparse data format.

    Length-bearing fields (`coords`, `template_coords`) are in angstroms, inherited from
    `AllAtomStructure` via `data/batch.py`.

    Attributes:
        asym_id: Asymmetric unit identifier (i.e., unique chain id). Shape [BS, num_tokens].
        atom_mask: Whether each atom participates in coordinate operations. When
          sampling from a System, every topology atom participates. Shape [BS, num_atoms].
        atom_pad_mask: Padding masked used for atom dimension [BS, num_atoms].
        atom_types: Atom type indices [BS, num_atoms].
        batch_size: Number of Samples in the batch. Shape [1,]. Multiplicity makes coord.shape[0] ambiguous.
        bond_index: Indices of atoms involved in a bond. Shape [BS, 2, num_bonds].
        bond_pad_mask: Whether a atom bond exists. Shape [BS, num_bonds]
        charge: Formal chemical charge of atom. Shape [BS, num_atoms].
        coords: Cartesian coordinates (in Angstrom). Shape [BS, num_atoms, 3] or [BS, num_atoms, num_frames, 3].
        deletion_mean: Mean deletion value for each residue
        deletion_value: Deletion count featurized with some arctan magic
        distogram_index: Index for atom used in pairwise distances. Shape [BS, num_tokens].
        element: Element chemical type. [0-128]. Shape [BS, num_atoms].
        entity_id: Entity identifier (i.e unique index for each individual sequence). Shape [BS, num_tokens].
        mol_type: Type of molecule this atom belongs to. Shape [BS, num_atoms]. Possible types are:
        (0: protein, 1: rna, 2:dna, 3: ligand)
        msa: One-hot MSA features. Shape [BS, num_tokens, num_msa, num_msa_tokens].
        msa_mask: True for non-padding MSA entries. Shape [BS, num_tokens, num_msa].
        msa_paired: Paired MSA features. Shape [BS, num_tokens, num_msa].
        name_encoding: ASCII atom-name encoding. Defined as ord(c) - 32, padded to length 4.
          Shape [BS, num_atoms, 4]
        num_nodes: Number of atoms. [BS, 1]
        num_residues: Number of residues.
        num_tokens: Number of tokens. This can deviate from `num_residues` if ligands are present,
          since each ligand atom is counted as a token, while the ligand is a single residue.
        num_tokens: Number of tokens.
        residue_idx: Increasing residue index assigning an atom to its residue. Does not reset after
          a chain and should be unique for each `asym_id`. Shape [BS, num_atoms].
        residue_type: Residue type, shape [BS, num_atoms].
        sequences: One-letter token sequences associated with this graph. Asym IDs. separated
          by colon.
        sym_id: Unique index within chains of this sequence. Shape [BS, num_tokens].
        system_id: Optional identifier (e.g., PDB Id.)
        template_bond_index: Indices of template atoms involved in a bond. Shape [BS, 2, num_bonds].
        template_coords: Cartesian coordinates of template atoms (in Angstrom). Shape [BS, num_atoms, 3].
          These are CCD reference positions, not structure-prediction templates.
          They can be accessed from the feature dictionary via `feats["ref_pos"]`
          in the score model.
        token_pad_mask: Whether a residue token exists. Shape [BS, num_tokens]
        token_center_atom_index: Index mapping tokens to their center atom index [BS, num_tokens].
        token_idx_at_atom: Index useful for mapping token-level to atom-level properties [BS, num_atoms]
        distogram_index: Index mapping atom dimension with shape [BS, num_tokens]. This differs slightly
          from the `token_center_atom_index` since e.g. the CB atoms are used for distogram computation instead
          of the CA which form the token centers.
        coords_cond: Conditioning coordinates for MD-sequential (ITO) conditioning.
          Unconditioned samples carry a zeroed placeholder, which the ``is_dynamics_conditioned`` masks out.
          Shape [BS, num_atoms, 3] when shared by all samples of a system, or
          [BS * multiplicity, num_atoms, 3] for one conditioning frame per sample.
        dt: Time lag in picoseconds between ``coords_cond`` and ``coords``. Zero for non-ITO samples.
          Shape [BS, 1] or [BS, num_frames].
        is_dynamics_conditioned: Boolean flag marking which samples carry structure (ITO) conditioning.
          Shape matches ``dt``.
    """

    asym_id: torch.LongTensor
    atom_mask: torch.BoolTensor
    atom_pad_mask: torch.BoolTensor
    bond_index: torch.LongTensor
    bond_pad_mask: torch.BoolTensor
    charge: torch.LongTensor
    coords: torch.Tensor
    element: torch.LongTensor
    entity_id: torch.LongTensor
    mol_type: torch.LongTensor  # [num_atoms]
    name_encoding: torch.LongTensor
    num_nodes: torch.LongTensor  # Number of atoms.
    num_residues: torch.LongTensor
    num_tokens: torch.LongTensor
    residue_idx: torch.LongTensor
    residue_type: torch.LongTensor
    sequence: list[str]
    sym_id: torch.LongTensor
    system_id: str | None
    template_bond_index: torch.LongTensor
    template_coords: torch.Tensor | None
    token_pad_mask: torch.BoolTensor
    token_center_atom_index: torch.LongTensor
    token_idx_at_atom: torch.LongTensor
    distogram_index: torch.LongTensor
    coords_cond: torch.Tensor
    dt: torch.Tensor
    is_dynamics_conditioned: torch.BoolTensor

    @property
    def batch_size(self) -> int:
        return self.atom_pad_mask.shape[0]  # multiplicity can repeat the same structure

    def upsample_to_atoms(self, token_related_idx: torch.Tensor) -> torch.Tensor:
        """Repeat per-token values to atoms via token_idx_at_atom; pad atoms (-1) gather zero."""
        bs, num_atoms = self.token_idx_at_atom.shape
        num_tokens = token_related_idx.shape[1]
        feature_dims = token_related_idx.shape[2:]
        # Pad-routing from broadcast_tokens_to_atoms: route pad atoms to an appended zero token row.
        padded = torch.nn.functional.pad(token_related_idx, [0, 0] * len(feature_dims) + [0, 1])
        routed_ids = self.token_idx_at_atom.masked_fill(~self.atom_pad_mask, num_tokens)
        atom_gather_idx = routed_ids.view(bs, num_atoms, *([1] * len(feature_dims))).expand(
            bs, num_atoms, *feature_dims
        )
        return torch.gather(padded, 1, atom_gather_idx)

    def __repr__(self) -> str:
        """String representation of the BatchAllAtomStructure in pretty print format."""
        items = list(self.items())
        if not items:
            return "BatchAllAtomStructure()"
        max_key_len = max(len(str(key)) for key, _ in items) + 3
        lines = []
        for key, value in sorted(items):
            if isinstance(value, torch.Tensor):
                lines.append(
                    f"{key:<{max_key_len}}: shape={str(tuple(value.shape)):<20} {str(value.dtype):<20} {str(value.device):<20}"
                )
            elif isinstance(value, list):
                lines.append(f"{key:<{max_key_len}}: {value[:2]}... (len={len(value)})")
            else:
                lines.append(f"{key:<{max_key_len}}: {value}")
        return "\n".join(lines)


def resname_to_index(resname: str) -> int:
    """Convert three letter residue name to index."""
    return Token[resname].value


def index_to_resname(index: int) -> str:
    """Convert index to three letter residue name."""
    return Token(index).name


def letter_to_resname(letter: str) -> str:
    "Convert one letter residue name to three letter name."
    return Token.from_single_letter(letter, moltype=MolType.PROTEIN).name


def index_to_pdb_resname(index: int) -> str:
    """
    Convert `Token` indices to PDB conform labels. This should be used when producing e.g. output
    topologies for properly mapping DNA and RNA residues to their PDB format labels.

    Args:
        index: Residue index mapping to `Token` definitions.

    Returns:
        Label converted to format suitable for PDB output.
    """
    token_resname = index_to_resname(index)

    # Remove type suffix from unknown tokens (chain and atoms should be enough info).
    if token_resname.startswith("UNK"):
        return "UNK"
    # For RNA, PDB uses single letter nucleotide codes.
    elif token_resname.startswith("RNA"):
        return token_resname[-1:]
    # For DNA, PDB uses single letter codes prefixed by `D`.
    elif token_resname.startswith("DNA"):
        return f"D{token_resname[-1:]}"
    else:
        return token_resname


# Atom-name encoding follows the Boltz feature representation. See
# ``NOTICES.md`` and ``licenses/LICENSE.BOLTZ-MIT``.


def encode_atom_name(name: str, max_pad: int = 4) -> list[int]:
    """Encode an atom name as four offset ASCII code points.

    Args:
        name: Atom name string to be encoded.
        max_pad: Length the integer encoding should be padded to.

    Returns:
        Encoded atom-name characters.
    """
    return [ord(x) - 32 for x in name] + [0] * (max_pad - len(name))


def decode_atom_name(name_encoding: Iterable[int]) -> str:
    """Decode offset ASCII code points to an atom name.

    Zeros correspond to whitespace, so ``strip`` removes the padding.

    Args:
        name_encoding: List of integers encoding atom name.

    Returns:
        String of atom name.
    """
    return "".join([chr(x + 32) for x in name_encoding]).strip()


def get_atom_name_mask_for_name_encoding(
    name_encoding: torch.Tensor, atom_name: str
) -> torch.Tensor:
    """
    Utility routine for checking where a name encoding tensor corresponds to a certain atom name.

    Args:
        name_encoding: Encoded atom names [num_atoms x 4].
        atom_name: String indicating atom name (e.g. "CA")

    Returns:
        Mask that is true where the encoding is that of the given atom name [num_atoms].
    """
    atom_name_encoding = torch.LongTensor(encode_atom_name(atom_name)).to(name_encoding.device)
    return torch.all(name_encoding == atom_name_encoding, dim=-1)


ATOM_LEVEL_PROPERTIES = [
    "coords",
    "coords_cond",
    "element",
    "residue_idx",
    "residue_type",
    "name_encoding",
    "charge",
    "atom_mask",
    "mol_type",
    "template_coords",
    "token_idx_at_atom",
    "pocket_atom_mask",
]


TOKEN_LEVEL_PROPERTIES = [
    "asym_id",
    "entity_id",
    "sym_id",
    "msa",
    "msa_paired",
    "deletion_value",
    "has_deletion",
    "deletion_mean",
    "profile",
    "msa_mask",
    "token_center_atom_index",
    "ca_class",
    "cat_mask",
]

# Indexing levels used by different index arrays (for remapping after subsampling).
ATOM_LEVEL_INDEX = [
    "token_center_atom_index",
    "distogram_index",
]
RESIDUE_LEVEL_INDEX = [
    "residue_idx",
]
TOKEN_LEVEL_INDEX = [
    "token_idx_at_atom",
]


def _get_index_map(mask: torch.BoolTensor) -> torch.LongTensor:
    """
    Get a map to convert between indices of masked and unmasked structure.

    Args:
        mask: Boolean mask indicating which entries are present or not.

    Returns:
        Tensor of indices yielding new indices if applied to unmasked ones
        (e.g. `new_idx = index_map[old_idx]`).
    """
    map = torch.full(mask.shape, -1, dtype=torch.long, device=mask.device)
    map[mask] = torch.arange(mask.sum(), device=mask.device)
    return map


def _mask_edge_indices(
    edges: torch.LongTensor, mask: torch.BoolTensor
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """
    Mask an array of edges given a node mask.

    Args:
        edges: Edge indices with the shape [2, num_edges].
        mask: Node mask of the shape [num_nodes].

    Returns:
        Masked edge indices ([2, num_masked_edges]) and edge mask [num_masked_edges].
    """
    edge_mask = torch.logical_and(mask[edges[0, :]], mask[edges[1, :]])
    return edges[:, edge_mask], edge_mask


def extract_substructure(
    structure: AllAtomStructure,
    token_mask: torch.BoolTensor | None,
    atom_mask: torch.BoolTensor | None = None,
) -> AllAtomStructure:
    """
    Extract a substructure from a structure given a token mask tensor or atom mask tensor. This will
    remove all masked token and atom entries. If present, edge information where one or more atoms
    are masked will be removed. Continuous indices will be updated to match the substructure. If
    only an atom mask is provided, only tokens with all atoms masked will be considered masked as
    well.

    Args:
        structure: Structure to which mask should be applied.
        token_mask: Token level boolean mask. True indicates a token is present. Shape is
          [num_tokens]. If this is not provided an atom mask needs to be given instead.
        atom_mask: Optional atom mask. If not provided, the token mask will be used to infer an atom
          mask, assuming all atoms associated with a masked token are masked.

    Returns:
        Substructure with masked residues and associated atoms, edges etc removed.
    """
    assert token_mask is not None or atom_mask is not None, (
        "Either atom mask or token mask need to be provided."
    )

    if token_mask is None:
        # Only tokens with all atoms masked should be fully masked.
        token_mask = scatter(atom_mask, structure.token_idx_at_atom, 0, reduce="sum")

    if atom_mask is None:
        # Expand token mask to atom mask.
        atom_mask = structure.upsample_to_atoms(token_mask)

    assert token_mask.shape == (structure.num_tokens,)
    assert atom_mask.shape == (structure.num_nodes,)

    # Create residue mask as well (for reindexing the residue index).
    # NOTE: In this case we need to be careful, since multiple tokens might map to a single residue.
    # For reindexing, the residue should only be considered fully masked, if all belonging atoms are
    # gone. We get this by aggregating the atom masks along the residue index. The `or` operation
    # corresponds to `sum` reduction on Boolean tensors in the `scatter` function.
    residue_mask = scatter(atom_mask, structure.residue_idx, 0, reduce="sum")

    # Create index maps for mapping between unmasked and masked indices.
    map_atom_idx = _get_index_map(atom_mask)
    map_token_idx = _get_index_map(token_mask)
    map_residue_idx = _get_index_map(residue_mask)

    # Apply atom mask to all atom level fields.
    fields_to_replace = {
        k: v[atom_mask, ...] for k, v in structure.items() if k in ATOM_LEVEL_PROPERTIES
    }

    # Apply residue mask to all token level fields.
    fields_to_replace = fields_to_replace | {
        k: v[token_mask, ...] for k, v in structure.items() if k in TOKEN_LEVEL_PROPERTIES
    }

    # Distogram is special.
    if "distogram_index" in structure:
        distogram_mask = atom_mask[structure.distogram_index]
        fields_to_replace["distogram_index"] = structure.distogram_index[distogram_mask]

    # Map everything involving atom indices.
    for atom_index_field in ATOM_LEVEL_INDEX:
        fields_to_replace[atom_index_field] = map_atom_idx[fields_to_replace[atom_index_field]]

    # If edges and edge features are present, apply masks and remap indices.
    if "edge_index" in structure:
        edge_index, edge_mask = _mask_edge_indices(structure.edge_index, atom_mask)
        fields_to_replace["edge_index"] = map_atom_idx[edge_index]

        if "edge_is_bonded" in structure:
            fields_to_replace["edge_is_bonded"] = structure.edge_is_bonded[edge_mask]

    # Same for bond indices.
    if "bond_index" in structure and structure.bond_index is not None:
        bond_index, bond_mask = _mask_edge_indices(structure.bond_index, atom_mask)
        fields_to_replace["bond_index"] = map_atom_idx[bond_index]

        if "bond_type" in structure and structure.bond_type is not None:
            fields_to_replace["bond_type"] = structure.bond_type[bond_mask]

    # Don't forget to subsample residue labels.
    if "residue_labels" in structure:
        fields_to_replace["residue_labels"] = [
            res for res, mask in zip(structure.residue_labels, atom_mask) if mask
        ]

    # Map token index properties.
    for token_index_field in TOKEN_LEVEL_INDEX:
        fields_to_replace[token_index_field] = map_token_idx[fields_to_replace[token_index_field]]

    # Reindex residue index quantities.
    for residue_index_field in RESIDUE_LEVEL_INDEX:
        fields_to_replace[residue_index_field] = map_residue_idx[
            fields_to_replace[residue_index_field]
        ]

    # NOTE: we can ignore remapping of `asym_id`, `sym_id` and `entity_id` for now, since cropping
    # is contiguous along the sequence.

    # Extract substructure.
    sub_structure = structure.replace(**fields_to_replace)

    # Update number of atoms and residues.
    sub_structure.num_nodes = atom_mask.sum().item()
    sub_structure.num_residues = residue_mask.sum().item()
    sub_structure.num_tokens = token_mask.sum().item()

    # If present, extract subsequence.
    if structure.sequence is not None:
        # Split sequence along chain token and use chunk size to split residue mask in the same
        # manner.
        chain_sequences = structure.sequence.split(":")
        chain_sequence_masks = torch.split(residue_mask, [len(c) for c in chain_sequences])

        # Iterate over chain sequences, check for residue presence, merge and aggregate.
        sub_structure_sequence = []
        for chain_seq, chain_mask in zip(chain_sequences, chain_sequence_masks, strict=True):
            sub_structure_sequence.append(
                "".join([aa for aa, present in zip(chain_seq, chain_mask, strict=True) if present])
            )
        # Add back chain split identifiers.
        sub_structure.sequence = ":".join(sub_structure_sequence)

    return sub_structure
