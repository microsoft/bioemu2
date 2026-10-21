"""Potential feature builder and FK math helpers shared by both steered samplers.

This file holds two unrelated-but-small concerns that the two sampler files
need:

1. `build_potential_features` creates the feature dictionary from a
   `BatchAllAtomStructure`;
   `potential_features_from_system` builds that batch from a `System` so every
   `FeatsPotential.prepare(system)` is self-contained. Three tiers:

   - Tier 0: keys produced by the steering feature converter.
   - Tier 1: three connectivity indices derived from `batch.bond_index` and
     `batch.entity_id`.
   - Tier 2: RDKit-derived per-residue constraints (`rdkit_bounds_*`,
     `chiral_*`, `stereo_*`, `planar_*`). Computed for every non-polymer
     residue when a map from chain index to CCD code or RDKit molecule is supplied; left empty
     otherwise. Building blocks: `steering/rdkit_constraints.py` —
     `compute_geometry_constraints`, `compute_chiral_atom_constraints`,
     `compute_stereo_bond_constraints`, `compute_flatness_constraints`
     (ported from `boltz/data/parse/schema.py`).

2. `fk_log_increment`, `grouped_softmax`, `systematic_resample`,
   `girsanov_ll_difference` — small math helpers for FK/FKC resampling.
   Used by `edm_steered.py` and `edm_fkc.py`.
"""

import functools
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from rdkit import Chem
from torch.nn import functional as F
from torch.nn.functional import one_hot

from bioemu2.data_interface import Ligand, System
from bioemu2.data_interface.const import NUM_TOKENS, MolType
from bioemu2.denoising.steering.rdkit_constraints import (
    compute_chiral_atom_constraints,
    compute_flatness_constraints,
    compute_geometry_constraints,
    compute_stereo_bond_constraints,
)
from bioemu2.structure import AllAtomStructure, BatchAllAtomStructure, decode_atom_name
from bioemu2.structure.batch import REQUIRES_MSA_PADDING, batch_collate
from bioemu2.structure.templates import _assign_mol_atom_names

_CCD_REPO = "boltz-community/boltz-1"
MAX_ELEMENTS: int = 128
NUM_BINS_ATOM: int = 64
NUM_TOKEN_TYPES: int = NUM_TOKENS


def convert_densebatch_to_potential_features(
    dense_batch: BatchAllAtomStructure,
) -> dict[str, torch.Tensor]:
    """Convert a dense ``BatchAllAtomStructure`` to potential features.

    Atom-dimension tensors are expected to be already padded to a window-aligned
    size by :func:`~bioemu2.structure.batch.batch_collate`.
    Model-only dense token-pair conditioning and bond features are omitted.

    Args:
        dense_batch: A dense batch of structures.

    Returns:
        The batched feature dictionary.
    """

    # NOTE: Everything that is not one-hot encoding, padding or float conversion here, we should
    # consider putting into the base structure and corresponding get function.

    max_tokens = dense_batch.num_tokens.max().item()

    features = {}
    features["asym_id"] = dense_batch.asym_id
    features["atom_pad_mask"] = dense_batch.atom_pad_mask.bool()
    features["atom_resolved_mask"] = dense_batch.atom_mask.bool()
    # Per-atom token indices use -1 for padded atoms. Downstream atom<->token
    # operations mask or clamp these indices; see ``bioemu2.biomoa.ops.atom_token_ops``.
    features["atom_to_token_ids"] = dense_batch.token_idx_at_atom
    features["coords"] = dense_batch.coords
    features["entity_id"] = dense_batch.entity_id

    if "method_feature" in dense_batch:
        features["method_feature"] = dense_batch.method_feature

    features["pocket_feature"] = one_hot(torch.zeros_like(dense_batch.token_pad_mask).long(), 4)
    features["ref_atom_name_chars"] = one_hot(dense_batch.name_encoding, NUM_BINS_ATOM)
    features["ref_charge"] = dense_batch.charge
    features["ref_element"] = one_hot(dense_batch.element.long(), MAX_ELEMENTS)

    # Zero out one-hot vectors for padded atoms so they don't leak signal into atom_feats.
    atom_pad_mask = dense_batch.atom_pad_mask  # [BS, max_atoms]
    features["ref_atom_name_chars"][~atom_pad_mask] = 0
    features["ref_element"][~atom_pad_mask] = 0
    features["ref_pos"] = dense_batch.template_coords
    # Reference-space uid (an index): padded atoms carry the -1 sentinel. The atom-attention
    # encoder re-masks these to its own pad uid, so the pad value is only a debugging aid here.
    features["ref_space_uid"] = dense_batch.residue_idx

    features["coords_cond"] = dense_batch.coords_cond
    features["dt"] = dense_batch.dt
    features["is_dynamics_conditioned"] = dense_batch.is_dynamics_conditioned

    # Auxiliary index for advanced indexing.
    batch_aux_idx = torch.arange(dense_batch.batch_size, device=dense_batch.asym_id.device)[:, None]

    # Build residue type by slicing atom wise residues.
    features["res_type"] = dense_batch.residue_type[
        batch_aux_idx, dense_batch.token_center_atom_index
    ]
    features["res_type"][~dense_batch.token_pad_mask] = 0
    features["res_type"] = one_hot(features["res_type"], NUM_TOKEN_TYPES)

    # Build residue index by slicing in the same way again.
    features["residue_index"] = dense_batch.residue_idx[
        batch_aux_idx, dense_batch.token_center_atom_index
    ]
    features["residue_index"][~dense_batch.token_pad_mask] = 0
    features["token_center_atom_index"] = dense_batch.token_center_atom_index

    features["sym_id"] = dense_batch.sym_id

    # No special padding needed — token_pad_mask masks out invalid positions downstream.
    features["token_index"] = (
        torch.arange(max_tokens, device=dense_batch.coords.device)
        .unsqueeze(0)
        .expand(int(dense_batch.batch_size), -1)
    )
    features["token_pad_mask"] = dense_batch.token_pad_mask.bool()

    # Add MSA features if present.
    # NOTE: these can be taken from the batch directly.
    if "msa" in dense_batch:
        for msa_feature in [
            "msa",
            "msa_paired",
            "deletion_value",
            "has_deletion",
            "deletion_mean",
            "profile",
            "msa_mask",
        ]:
            # The batch stores token-major MSA features; the trunk expects MSA depth first.
            if msa_feature in REQUIRES_MSA_PADDING:
                features[msa_feature] = dense_batch[msa_feature].transpose(1, 2)
            else:
                features[msa_feature] = dense_batch[msa_feature]

    features = {key: value.to(dense_batch.coords.device) for key, value in features.items()}
    return features


@functools.lru_cache(maxsize=1)
def _load_ccd() -> dict:
    """Load the cached CCD reference data used for ligand steering."""
    ccd_path = Path(hf_hub_download(repo_id=_CCD_REPO, filename="ccd.pkl"))
    with ccd_path.open("rb") as file:
        return pickle.load(file)


def potential_features_from_system(system: System) -> dict[str, torch.Tensor]:
    """Build topology-only potential features for ``system``.

    The features are stored on CPU and right-padded to the collation window.
    Potentials read only topology keys, so mode, contacts, and MSA conditioning
    are irrelevant; ``mode="structure"`` matches ``System.topology``.
    """
    batch = batch_collate([AllAtomStructure.from_system(system, mode="structure")])
    ligand_refs: dict[int, str | Chem.Mol] = dict(ligand_ccd_codes_from_system(system))
    chain_index = 0
    for entity in system.entities:
        if isinstance(entity, Ligand) and entity.ccd_code is None:
            inchi = (
                Chem.MolToInchi(Chem.MolFromSmiles(entity.smiles))
                if entity.smiles is not None
                else entity.inchi
            )
            # Match the template's atom naming without generating a conformer.
            mol = Chem.AddHs(Chem.MolFromInchi(inchi))
            _assign_mol_atom_names(mol)
            mol = Chem.RemoveAllHs(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            for index in range(chain_index, chain_index + len(entity.asym_ids)):
                ligand_refs[index] = mol
        chain_index += len(entity.asym_ids)
    return build_potential_features(batch, ligand_refs)


def ligand_ccd_codes_from_system(
    system: System | None,
) -> dict[int, str]:
    """Map ``{int_asym_id: ccd_code}`` for the CCD-specified ligand chains.

    Walks the system's entities, assigning a sequential integer asym_id to each
    chain (matching ``batch.asym_id`` ordering), and records the CCD code for
    ``Ligand`` entities that carry one. SMILES/InChI ligands (no ``ccd_code``)
    are skipped.
    """
    if system is None:
        return {}
    ccd: dict[int, str] = {}
    int_asym_id = 0
    for entity in system.entities:
        for _ in entity.asym_ids:
            if isinstance(entity, Ligand) and entity.ccd_code is not None:
                ccd[int_asym_id] = entity.ccd_code
            int_asym_id += 1
    return ccd


def build_potential_features(
    batch: BatchAllAtomStructure,
    ligand_ccd_codes: Mapping[int, str | Chem.Mol] | None = None,
) -> dict[str, torch.Tensor]:
    """Build the feature dictionary consumed by physicality potentials.

    Args:
        batch: dense-batched structure (as built by `batch_collate`).
        ligand_ccd_codes: Map from chain index to CCD code or named RDKit molecule for the ligand
            chains in the batch. When non-empty, Tier-2 (RDKit per-residue
            constraints) is populated for each entry; PoseBusters, chiral,
            stereo, and planar potentials then evaluate those constraints.
            When ``None`` (no ligand information), Tier 2 contains empty
            tensors with the canonical shapes and those potentials are no-ops.
    """
    feats = convert_densebatch_to_potential_features(batch)

    # Tier 1.
    feats["symmetric_chain_index"] = _symmetric_chain_index(feats, batch)

    # Tier 2.
    if ligand_ccd_codes:
        feats.update(_compute_rdkit_constraint_feats(batch, feats, ligand_ccd_codes))
    else:
        feats.update(_empty_rdkit_feats(batch))

    return feats


def _symmetric_chain_index(feats, batch: BatchAllAtomStructure) -> torch.Tensor:
    """
    Pairs of chains sharing the same `entity_id` (homo-multimer copies).
    Every chain in the batch is paired with any other chain that shares the same `entity_id`.
    #asym >= #entity_id, there should be equal or more chains than entity IDs.
    """
    asym = feats["asym_id"]  # [B, #tokens]
    eid = batch.entity_id  # [B, #tokens]
    per_batch = []
    for b in range(asym.shape[0]):  # iterate over batches
        chains = torch.unique(
            torch.stack([asym[b], eid[b]], dim=1), dim=0
        )  # (asym, entity) mapping
        pairs = []
        for i in range(chains.shape[0]):
            for j in range(i + 1, chains.shape[0]):
                if (
                    chains[i, 1] == chains[j, 1]
                ):  # iterate over all pairs of (asym, entity) mappings
                    pairs.append([chains[i, 0].item(), chains[j, 0].item()])
        per_batch.append(
            torch.tensor(pairs, dtype=torch.long, device=asym.device).T
            if pairs
            else torch.empty(2, 0, dtype=torch.long, device=asym.device)
        )
    return _pad_dim1(per_batch, device=asym.device)


def _pad_dim1(tensors: list[torch.Tensor], device) -> torch.Tensor:
    """Stack `[2, K_b]` tensors into `[B, 2, K_max]` with right-zero-padding."""
    K = max((t.shape[1] for t in tensors), default=0)
    if K == 0:
        return torch.empty(len(tensors), 2, 0, dtype=torch.long, device=device)
    out = torch.zeros(len(tensors), 2, K, dtype=torch.long, device=device)
    for b, t in enumerate(tensors):
        out[b, :, : t.shape[1]] = t
    return out


# Tier 2: RDKit-derived per-residue constraints


def _empty_rdkit_feats(batch: BatchAllAtomStructure) -> dict[str, torch.Tensor]:
    """Build empty constraint tensors with the shapes expected by potentials."""
    B = batch.batch_size
    d = batch.coords.device
    return {
        "rdkit_bounds_index": torch.empty(B, 2, 0, dtype=torch.long, device=d),
        "rdkit_lower_bounds": torch.empty(B, 0, dtype=torch.float32, device=d),
        "rdkit_upper_bounds": torch.empty(B, 0, dtype=torch.float32, device=d),
        "rdkit_bounds_bond_mask": torch.empty(B, 0, dtype=torch.bool, device=d),
        "rdkit_bounds_angle_mask": torch.empty(B, 0, dtype=torch.bool, device=d),
        "chiral_atom_index": torch.empty(B, 4, 0, dtype=torch.long, device=d),
        "chiral_atom_orientations": torch.empty(B, 0, dtype=torch.bool, device=d),
        "stereo_bond_index": torch.empty(B, 4, 0, dtype=torch.long, device=d),
        "stereo_bond_orientations": torch.empty(B, 0, dtype=torch.bool, device=d),
        "planar_bond_index": torch.empty(B, 6, 0, dtype=torch.long, device=d),
    }


def _compute_rdkit_constraint_feats(
    batch: BatchAllAtomStructure,
    feats: dict[str, torch.Tensor],
    ligand_ccd_codes: Mapping[int, str | Chem.Mol],
) -> dict[str, torch.Tensor]:
    """Build batched RDKit constraints for every non-polymer residue.

    Atoms in canonical amino-acid residues are skipped.
    """
    ccd = _load_ccd() if any(isinstance(ref, str) for ref in ligand_ccd_codes.values()) else {}
    B = batch.batch_size
    # Per-atom chain index (asym_id broadcast onto atoms; window-padded -> chain 0).
    atom_chain = batch.upsample_to_atoms(
        feats["asym_id"]
    ).long()  # [B, #atoms] chain index per atom

    per_batch: list[_RdkitAcc] = [_RdkitAcc() for _ in range(B)]

    for b in range(B):
        for atom_lo, atom_hi in _iter_ligand_residues(batch, b, atom_chain[b]):
            chain_int = atom_chain[b, atom_lo].item()  # asym_id/chain_idx
            ref = ligand_ccd_codes.get(chain_int)
            ref_mol = ccd.get(ref) if isinstance(ref, str) else ref
            if ref_mol is None:
                continue
            idx_map = _atom_name_idx_map(batch, b, atom_lo, atom_hi, ref_mol)
            if not idx_map:
                continue

            geo = compute_geometry_constraints(ref_mol, idx_map)
            chir = compute_chiral_atom_constraints(ref_mol, idx_map)
            ster = compute_stereo_bond_constraints(ref_mol, idx_map)
            plan, _r5, _r6 = compute_flatness_constraints(ref_mol, idx_map)
            per_batch[b].accumulate(geo, chir, ster, plan)

    return _pack_rdkit(per_batch, B, batch.coords.device)


def _iter_ligand_residues(batch: BatchAllAtomStructure, b: int, atom_chain_b):
    """Yield `(atom_lo, atom_hi)` half-open atom slices for every non-polymer
    residue in batch element b.

    A "residue" here is a maximal contiguous run of atoms sharing the same
    `residue_idx`. We only yield runs whose first atom is in a non-polymer
    chain (i.e. its mol_type is NONPOLYMER)."""
    residue_idx = batch.residue_idx[b]
    mol_type = batch.mol_type[b]
    atom_mask = batch.atom_mask[b].bool()
    N = residue_idx.shape[0]

    lo = 0
    while lo < N:
        if not atom_mask[lo]:
            lo += 1
            continue
        hi = lo + 1
        while (
            hi < N
            and atom_mask[hi]
            and residue_idx[hi] == residue_idx[lo]
            and atom_chain_b[hi] == atom_chain_b[lo]
        ):
            hi += 1
        if mol_type[lo].item() == MolType.NONPOLYMER.value:
            yield lo, hi
        lo = hi


def _atom_name_idx_map(batch, b, atom_lo, atom_hi, ref_mol):
    """Build {ref_mol_atom_idx → global_atom_idx} by matching atom names.

    Atom names on the batch are encoded in `name_encoding[b, k]` (4 bytes
    packed into a small int); `decode_atom_name` reverses the encoding to a
    string. `ref_mol` atoms expose their PDB-style name via the "name"
    property in the CCD reference data; we fall back to the element symbol when
    the property is missing."""
    name_to_global: dict[str, int] = {}
    name_enc = batch.name_encoding[b, atom_lo:atom_hi].cpu()
    for k, enc in enumerate(name_enc.tolist(), start=atom_lo):
        nm = decode_atom_name(enc).strip()
        name_to_global[nm] = k

    idx_map: dict[int, int] = {}
    for ref_idx, ref_atom in enumerate(ref_mol.GetAtoms()):
        ref_name = ref_atom.GetPropsAsDict().get("name", ref_atom.GetSymbol())
        ref_name = str(ref_name).strip()
        if ref_name in name_to_global:
            idx_map[ref_idx] = name_to_global[ref_name]
    return idx_map


class _RdkitAcc:
    """Per-batch-element accumulator. Each list holds python rows from
    `compute_*` outputs; concatenated and padded by `_pack_rdkit`."""

    def __init__(self):
        self.bounds_idxs: list[list[int]] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.is_bond: list[bool] = []
        self.is_angle: list[bool] = []
        self.chiral_idxs: list[list[int]] = []
        self.chiral_is_r: list[bool] = []
        self.stereo_idxs: list[list[int]] = []
        self.stereo_is_e: list[bool] = []
        self.planar_idxs: list[list[int]] = []

    def accumulate(self, geo, chir, ster, plan):
        # Each input contains parsed constraint dataclasses from the
        # corresponding compute function.
        for c in geo:
            self.bounds_idxs.append(list(c.atom_idxs))
            self.lower.append(float(c.lower_bound))
            self.upper.append(float(c.upper_bound))
            self.is_bond.append(bool(c.is_bond))
            self.is_angle.append(bool(c.is_angle))
        for c in chir:
            self.chiral_idxs.append(list(c.atom_idxs))
            self.chiral_is_r.append(bool(c.is_r))
        for c in ster:
            self.stereo_idxs.append(list(c.atom_idxs))
            self.stereo_is_e.append(bool(c.is_e))
        for c in plan:
            self.planar_idxs.append(list(c.atom_idxs))


def _pack_rdkit(per_batch: list[_RdkitAcc], B: int, device) -> dict[str, torch.Tensor]:
    """Per-batch accumulators → batched tensors in the canonical shapes
    `process_residue_constraint_features` (`featurizer.py:1002-1046`) emits."""

    def pad_pairs(rows_per_batch: list[list[list[int]]], dim_size: int):
        K = max((len(rows) for rows in rows_per_batch), default=0)
        if K == 0:
            return torch.empty(B, dim_size, 0, dtype=torch.long, device=device)
        out = torch.zeros(B, dim_size, K, dtype=torch.long, device=device)
        for b, rows in enumerate(rows_per_batch):
            if rows:
                t = torch.tensor(rows, dtype=torch.long, device=device).T  # [dim, K_b]
                out[b, :, : t.shape[1]] = t
        return out

    def pad_flat(vals_per_batch: list[list], dtype):
        K = max((len(v) for v in vals_per_batch), default=0)
        if K == 0:
            return torch.empty(B, 0, dtype=dtype, device=device)
        out = torch.zeros(B, K, dtype=dtype, device=device)
        for b, v in enumerate(vals_per_batch):
            if v:
                out[b, : len(v)] = torch.tensor(v, dtype=dtype, device=device)
        return out

    return {
        "rdkit_bounds_index": pad_pairs([a.bounds_idxs for a in per_batch], 2),
        "rdkit_lower_bounds": pad_flat([a.lower for a in per_batch], torch.float32),
        "rdkit_upper_bounds": pad_flat([a.upper for a in per_batch], torch.float32),
        "rdkit_bounds_bond_mask": pad_flat([a.is_bond for a in per_batch], torch.bool),
        "rdkit_bounds_angle_mask": pad_flat([a.is_angle for a in per_batch], torch.bool),
        "chiral_atom_index": pad_pairs([a.chiral_idxs for a in per_batch], 4),
        "chiral_atom_orientations": pad_flat([a.chiral_is_r for a in per_batch], torch.bool),
        "stereo_bond_index": pad_pairs([a.stereo_idxs for a in per_batch], 4),
        "stereo_bond_orientations": pad_flat([a.stereo_is_e for a in per_batch], torch.bool),
        "planar_bond_index": pad_pairs([a.planar_idxs for a in per_batch], 6),
    }


# FK resampling and propagation conditioning.


def grouped_softmax(log_weights: torch.Tensor, group_size: int | None) -> torch.Tensor:
    """Normalize flat `log_weights` within consecutive groups of `group_size` (whole batch if None)."""
    chunks = log_weights.split(group_size or len(log_weights))
    return torch.cat([F.softmax(chunk, dim=0) for chunk in chunks])


def systematic_resample(weights: torch.Tensor, group_size: int | None) -> torch.Tensor:
    """Systematic resampling within consecutive groups of `group_size`; returns flat indices.

    `weights` are flat, normalized within each group (see `grouped_softmax`); the
    last group is smaller when `group_size` does not divide the batch. Each group
    draws one shared uniform offset and reads the group CDF at `n` evenly spaced
    positions, so a particle with weight `w` gets `floor(n w)` or `ceil(n w)`
    offspring — far less variance than multinomial draws.
    """
    idx, offset = [], 0
    for w in weights.split(group_size or len(weights)):
        n = len(w)
        positions = (
            torch.arange(n, device=w.device, dtype=w.dtype)
            + torch.rand((), device=w.device, dtype=w.dtype)
        ) / n
        idx.append(offset + torch.searchsorted(w.cumsum(0), positions).clamp(max=n - 1))
        offset += n
    return torch.cat(idx)


def fk_log_increment(energy_now: torch.Tensor, energy_prev: torch.Tensor | None) -> torch.Tensor:
    """Telescoping FK potential: `log G_t = E_{t-1} - E_t` (or `-E_0` at t=0)."""
    return -energy_now if energy_prev is None else energy_prev - energy_now


def girsanov_ll_difference(
    eps: torch.Tensor, scaled_guidance_update: torch.Tensor, noise_var: torch.Tensor
) -> torch.Tensor:
    """Importance correction for guidance-shifted Gaussian transitions.

    `Δℓ = (‖ε‖² - ‖ε + Δ‖²) / (2 σ²)`, summed over coord dims.
    """
    sq_diff = (eps.pow(2) - (eps + scaled_guidance_update).pow(2)).sum(dim=(-1, -2))
    return sq_diff / (2.0 * noise_var)


def _pack_to_cpu(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Prepare metrics for saving on the CPU without copying each GPU tensor separately.

    Detach and convert tensors to float32, flatten and join those on each non-CPU
    device for one CPU copy, then restore their original names and shapes.
    """
    cpu_values = {}
    device_values: dict[torch.device, dict[str, torch.Tensor]] = {}
    for key, value in values.items():
        value = value.detach()
        if value.device.type == "cpu":
            cpu_values[key] = value.to(torch.float32)
        else:
            device_values.setdefault(value.device, {})[key] = value
    for group in device_values.values():
        packed = torch.cat([value.reshape(-1).to(torch.float32) for value in group.values()]).cpu()
        chunks = packed.split([value.numel() for value in group.values()])
        for (key, value), chunk in zip(group.items(), chunks, strict=True):
            cpu_values[key] = chunk.reshape(value.shape)
    return cpu_values


class Tracker:
    """Save sampling diagnostics to disk, with one float32 ``.npy`` file per metric.

    Call ``register(key, shape)`` before logging a metric. The first dimension
    of ``shape`` is the number of rows to reserve; the rest is the shape of one
    value. A NumPy memory map lets each value be written directly into its file.

    Pass a dictionary of tensors to ``log`` to write the next row for each named
    metric. Each metric has its own row counter, so metrics can be recorded at
    different steps. Tensors are detached from autograd, and values on the same
    non-CPU device are copied to CPU together before writing.

    Registering a metric creates ``{key}{filename_suffix}.npy`` in ``directory``,
    overwriting any existing file at that path. Use a different directory or
    suffix for each sampling batch to keep its files separate. Rows that have
    not been written remain zero.
    """

    def __init__(
        self,
        directory: Path,
        *,
        filename_suffix: str = "",
    ) -> None:
        self._directory = directory
        self._filename_suffix = filename_suffix
        self._arrays: dict[str, np.memmap] = {}
        self._steps: dict[str, int] = {}

    def register(self, key: str, shape: tuple[int, ...]) -> None:
        """Create a metric file with ``shape[0]`` rows of values shaped ``shape[1:]``."""
        if key in self._arrays:
            raise ValueError(f"Tracker key {key!r} is already registered")
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{key}{self._filename_suffix}.npy"
        self._arrays[key] = np.lib.format.open_memmap(path, dtype="float32", mode="w+", shape=shape)
        self._steps[key] = 0

    def log(self, values: Mapping[str, torch.Tensor]) -> None:
        """Save one value for each named metric.

        Convert the supplied tensors to CPU float32 and write each at that metric's next row.
        """
        for key, value in _pack_to_cpu(values).items():
            self._arrays[key][self._steps[key]] = value.numpy()
            self._steps[key] += 1
