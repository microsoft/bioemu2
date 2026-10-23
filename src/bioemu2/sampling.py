"""Structure prediction and equilibrium sampling with physicality filtering."""

import logging
import shutil
from collections import Counter
from collections.abc import Collection, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import mdtraj
import numpy as np
import torch
import yaml
from tqdm.auto import trange

from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.conditioning.method import SamplingMode
from bioemu2.data_interface import Contact, System
from bioemu2.denoising.samplers import EDMSampler, FKCSampler, GuidedFKSampler, Sampler
from bioemu2.denoising.steering.feats import Tracker
from bioemu2.denoising.steering.potentials import Potential
from bioemu2.model import InferenceModel
from bioemu2.msa.alignment import get_msa
from bioemu2.physicality import (
    PhysicalityFilterMode,
    PhysicalityReport,
    _select_expected_extra_bonds,
    check_allatom_physicality,
    get_protein_atom_indices,
    normalize_physicality_mode,
)
from bioemu2.structure import (
    AllAtomStructure,
    BatchAllAtomStructure,
    get_asym_id_to_chain_metadata,
)
from bioemu2.structure.cif import save_structure_to_cif
from bioemu2.structure.preparation import prepare_sampling_inputs
from bioemu2.trajectory_io import coords_angstrom_to_trajectory

logger = logging.getLogger(__name__)

SamplerPreset = Literal["edm", "edm_gd", "edm_fk", "edm_fkc"]


def sampler_from_preset(
    preset: str,
    *,
    mode: SamplingMode,
    num_steps: int | None = None,
    potentials: Sequence[Potential] | None = None,
) -> Sampler:
    step_scale = 1.5 if mode == "structure" else 1.0
    sampler: Sampler
    if preset == "edm_fkc":
        sampler = FKCSampler()
    elif preset == "edm":
        sampler = EDMSampler(step_scale=step_scale)
    elif preset == "edm_gd":
        sampler = GuidedFKSampler(step_scale=step_scale, resampling_interval=None)
    elif preset == "edm_fk":
        sampler = GuidedFKSampler(step_scale=step_scale)
    else:
        raise ValueError(
            f"Unknown sampler preset: {preset}. Choose from: edm, edm_gd, edm_fk, edm_fkc"
        )
    if num_steps is not None:
        if num_steps <= 0:
            raise ValueError("num_steps must be a positive integer.")
        sampler.num_steps = num_steps
    if potentials is not None:
        if not isinstance(sampler, GuidedFKSampler | FKCSampler):
            raise ValueError("potentials requires a steering sampler (edm_gd, edm_fk, edm_fkc)")
        sampler.potentials = tuple(potentials)
    return sampler


def warn_batch_size(num_tokens: int, num_samples: int, batch_size: int) -> None:
    """Advise on default or poorly sized batches without changing sampling."""
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not torch.cuda.is_available():
        return
    # rough estimate
    total_mem = torch.cuda.get_device_properties().total_memory
    max_batch_size = max(1, int(0.8 * total_mem / (0.3e6 * num_tokens) - num_tokens / 75))
    if (batch_size == 1 and min(num_samples, max_batch_size) >= 2) or (
        num_tokens < 128 and num_samples > 64 and batch_size <= 64
    ):
        concern = "may underutilize the GPU; consider a larger batch"
    # generate_samples never runs more than num_samples at once.
    elif min(batch_size, num_samples) > max_batch_size:
        concern = f"may exceed GPU memory; try a batch_size of {max_batch_size} or lower"
    else:
        return
    logger.warning(
        "batch_size=%d for %d tokens and num_samples=%d. This %s. "
        "Read the documentation docs/speed-and-capacity.md for guidance.",
        batch_size,
        num_tokens,
        num_samples,
        concern,
    )


def generate_samples(
    num_samples: int,
    batch_size: int,
    score_model: InferenceModel[TrunkEmbeddings],
    sampler: Sampler,
    num_atoms: int,
    system: System,
    batch: BatchAllAtomStructure,
    trunk_embeddings: TrunkEmbeddings,
    progress: bool = True,
    run_dir: Path | None = None,
    partial_dir: Path | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Run each batch and return centered coordinates in angstroms.

    With ``run_dir`` set, steered samplers write diagnostics in ``run_dir/metrics``
    with zero-based, end-exclusive sample ranges appended to the filenames, such
    as ``x0_000000-000002.npy``. A fresh Tracker is created only for batches
    that need sampling. Completed batches loaded from ``partial_dir`` leave
    existing diagnostics untouched and do not reconstruct missing diagnostics.
    """
    partial_coords: list[torch.Tensor] = []
    if partial_dir is not None:
        partial_dir.mkdir(parents=True, exist_ok=True)
    for potential in getattr(sampler, "potentials", ()):
        potential.prepare(system)
    for start in trange(0, num_samples, batch_size, disable=not progress, desc="Sampling"):
        size = min(batch_size, num_samples - start)
        batch_seed = seed + start // batch_size if seed is not None else None
        if partial_dir is not None:
            batch_path = partial_dir / f"samples_{start:06d}-{start + size:06d}.npy"
            completion_marker = batch_path.with_suffix(".done")
            if batch_path.is_file() and completion_marker.is_file():
                partial_coords.append(
                    torch.from_numpy(np.load(batch_path, allow_pickle=False)).to(
                        batch.coords.device
                    )
                )
                continue
            completion_marker.unlink(missing_ok=True)
        tracker = (
            Tracker(
                run_dir / "metrics",
                filename_suffix=f"_{start:06d}-{start + size:06d}",
            )
            if run_dir is not None
            else None
        )
        with rng_seed(batch_seed), torch.no_grad():
            sampled = sampler(
                batch=batch,
                score_model=score_model,
                trunk_embeddings=trunk_embeddings,
                batch_size=size,
                tracker=tracker,
            )

        xyz = sampled.coords.view(size, -1, 3)
        xyz = xyz[:, :num_atoms]
        xyz = xyz - xyz.mean(dim=1, keepdim=True)
        if partial_dir is not None:
            np.save(batch_path, xyz.cpu().numpy(), allow_pickle=False)
            # An interrupted write leaves no marker, so the batch is regenerated on retry.
            completion_marker.touch()
        partial_coords.append(xyz)

    return torch.cat(partial_coords)


def save_topology_cif(
    structure: AllAtomStructure,
    system: System,
    coords_angstrom: np.ndarray | torch.Tensor,
    path: Path,
) -> None:
    """Write a CIF with the ``_entity``/``_struct_asym`` tables mdtraj's writer lacks."""
    coords = torch.as_tensor(coords_angstrom, dtype=structure.coords.dtype)
    save_structure_to_cif(
        structure.replace(coords=coords),
        path,
        renamed_asym_ids=system.chain_ids,
        system=system,
        asym_id_to_chain_metadata=get_asym_id_to_chain_metadata(system),
    )


def expected_extra_bonds_from_system(
    system: System, topology: mdtraj.Topology
) -> set[tuple[int, int]]:
    """Resolve declared bond endpoints to atom indices in a sampling topology."""
    if not system.bonds:
        return set()
    atoms = {
        (chain.chain_id, residue_number, atom.name): atom.index
        for chain in topology.chains
        for residue_number, residue in enumerate(chain.residues, start=1)
        for atom in residue.atoms
    }
    return {
        (
            atoms[(bond.atom1.asym_id, bond.atom1.res_id, bond.atom1.name)],
            atoms[(bond.atom2.asym_id, bond.atom2.res_id, bond.atom2.name)],
        )
        for bond in system.bonds
    }


def apply_physicality_filter(
    trajectory: mdtraj.Trajectory,
    mode: PhysicalityFilterMode | str,
    *,
    expected_extra_bonds: Collection[tuple[int, int]] = (),
) -> tuple[mdtraj.Trajectory, mdtraj.Trajectory]:
    """Split frames by protein physicality; extra bond indices use the input topology."""
    if normalize_physicality_mode(mode) is PhysicalityFilterMode.NO_FILTER:
        return trajectory, trajectory[:0]
    report = check_physicality(trajectory, mode, expected_extra_bonds=expected_extra_bonds)
    mask, reasons = report.passed, report.reasons
    kept, rejected = trajectory[mask], trajectory[~mask]
    if len(rejected) > 0:
        reason_counts = Counter(r for r in reasons if r is not None)
        logger.info(
            f"Protein physicality filter: kept {len(kept)}/{len(trajectory)} samples "
            f"(rejected: {reason_counts}; non-protein chains are not checked)"
        )
    if len(rejected) > len(trajectory) / 2:
        logger.warning(
            f"More than half of samples ({len(rejected)}/{len(trajectory)}) failed "
            f"protein physicality checks — the checkpoint or system may need review."
        )
    return kept, rejected


def check_physicality(
    trajectory: mdtraj.Trajectory,
    mode: PhysicalityFilterMode | str = PhysicalityFilterMode.HIGH_QUALITY,
    *,
    system: System | None = None,
    expected_extra_bonds: Collection[tuple[int, int]] = (),
) -> PhysicalityReport:
    """Check protein geometry without changing frames and return saveable results.

    Pass the original system to account for its declared extra bonds, or supply
    extra bond indices in the input topology with ``expected_extra_bonds``.
    Results correspond to the trajectory's frame order. Non-protein chains are not checked.
    With no eligible protein atoms, results are marked not checked even when a check was requested.
    """
    mode = normalize_physicality_mode(mode)
    if mode is PhysicalityFilterMode.NO_FILTER:
        return PhysicalityReport(
            passed=np.ones(len(trajectory), dtype=bool),
            reasons=[None] * len(trajectory),
            mode=mode,
            checked=False,
        )

    protein_indices = get_protein_atom_indices(trajectory.topology)
    if len(protein_indices) == 0:
        logger.warning("No eligible protein atoms; protein physicality was not checked.")
        return PhysicalityReport(
            passed=np.ones(len(trajectory), dtype=bool),
            reasons=[None] * len(trajectory),
            mode=mode,
            checked=False,
        )
    if system is not None:
        expected_extra_bonds = set(expected_extra_bonds) | expected_extra_bonds_from_system(
            system, trajectory.topology
        )
    protein_traj = trajectory.atom_slice(protein_indices)
    passed, _stats, reasons = check_allatom_physicality(
        protein_traj,
        mode=mode,
        expected_extra_bonds=_select_expected_extra_bonds(
            expected_extra_bonds, protein_indices, trajectory.n_atoms
        ),
    )
    return PhysicalityReport(passed=passed, reasons=reasons, mode=mode)


@contextmanager
def rng_seed(seed: int | None) -> Generator[None, None, None]:
    """Temporarily seed CPU and all visible CUDA generators, restoring their states.

    Args:
        seed: Seed to apply, or ``None`` to leave RNG state unchanged.

    Yields:
        Control while the requested seed is active.
    """
    if seed is None:
        yield
        return
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def sample_with_embeddings(
    system: System,
    trunk_embeddings: TrunkEmbeddings,
    *,
    model: InferenceModel[TrunkEmbeddings],
    num_samples: int = 8,
    batch_size: int | None = None,
    sampler: Sampler,
    seed: int | None = None,
    progress: bool = True,
    partial_dir: Path | None = None,
) -> mdtraj.Trajectory:
    """Sample with precomputed embeddings and an explicit sampler configuration.

    The sampler configuration is used unchanged. All generated frames are
    returned; call ``check_physicality`` separately to check their geometry.

    Args:
        system: Molecular system represented by ``trunk_embeddings``.
        trunk_embeddings: Reusable output from ``model.embed``.
        model: Inference model used to create the embeddings and denoise structures.
        num_samples: Number of output conformations.
        batch_size: Maximum samples generated per denoising call.
        sampler: Sampling algorithm used unchanged; steering potentials are configured on it.
        seed: Optional Torch random seed. When supplied, batch ``i`` uses ``seed + i``
            for reproducible retries. Otherwise, sampling does not reseed the RNG,
            and retries need not match an uninterrupted run.
        progress: Whether to display sampling progress.
        partial_dir: Optional directory for saving and reusing completed raw batches.
            Keep inputs and settings fixed and use only one writer per directory.
            Retained after return; remove it after saving the returned trajectory.

    Returns:
        Sampled trajectory in nanometers.
    """
    if batch_size is None:
        # For batch-size tuning, see docs/speed-and-capacity.md.
        batch_size = num_samples
    topology, batch = prepare_sampling_inputs(system, device=model.device)
    warn_batch_size(int(batch.num_tokens.item()), num_samples, batch_size)

    with rng_seed(seed):
        coords = generate_samples(
            num_samples=num_samples,
            batch_size=batch_size,
            score_model=model,
            sampler=sampler,
            num_atoms=topology.n_atoms,
            system=system,
            batch=batch,
            trunk_embeddings=trunk_embeddings,
            progress=progress,
            partial_dir=partial_dir,
            seed=seed,
        )
    trajectory = coords_angstrom_to_trajectory(coords, topology)
    return trajectory


def _embed_and_sample(
    system: System,
    *,
    model: InferenceModel[TrunkEmbeddings],
    mode: SamplingMode,
    contacts: list[Contact] | None,
    num_samples: int,
    batch_size: int | None,
    sampler: SamplerPreset,
    num_steps: int | None,
    potentials: Sequence[Potential] | None,
    seed: int | None,
    progress: bool,
    partial_dir: Path | None,
    output_dir: str | Path | None,
    physicality_mode: PhysicalityFilterMode | str,
) -> mdtraj.Trajectory:
    physicality_mode = normalize_physicality_mode(physicality_mode)
    configured_sampler = sampler_from_preset(
        sampler, mode=mode, num_steps=num_steps, potentials=potentials
    )
    out = Path(output_dir) if output_dir is not None else None
    managed_partial_dir = out / "partial" if out is not None and partial_dir is None else None
    if out is not None:
        write_sampling_system(system, mode=mode, contacts=contacts, out=out)
    msa = get_msa(system)
    trunk_embeddings = model.embed(system, msa=msa, mode=mode, contacts=contacts)
    trajectory = sample_with_embeddings(
        system,
        trunk_embeddings,
        model=model,
        num_samples=num_samples,
        batch_size=batch_size,
        sampler=configured_sampler,
        seed=seed,
        progress=progress,
        partial_dir=managed_partial_dir if managed_partial_dir is not None else partial_dir,
    )
    if out is not None:
        write_sampling_output(
            trajectory,
            system=system,
            mode=mode,
            contacts=contacts,
            out=out,
            physicality_filter=physicality_mode,
        )
        if managed_partial_dir is not None:
            shutil.rmtree(managed_partial_dir)
    return trajectory


def predict_structure(
    system: System,
    *,
    model: InferenceModel[TrunkEmbeddings],
    contacts: list[Contact] | None = None,
    num_samples: int = 8,
    batch_size: int | None = None,
    sampler: SamplerPreset = "edm",
    num_steps: int | None = None,
    potentials: Sequence[Potential] | None = None,
    seed: int | None = None,
    progress: bool = True,
    partial_dir: Path | None = None,
    output_dir: str | Path | None = None,
    physicality_mode: PhysicalityFilterMode | str = PhysicalityFilterMode.HIGH_QUALITY,
) -> mdtraj.Trajectory:
    """Predict structures with a loaded model and automatically computed MSA.

    Args:
        system: Molecular system to predict.
        model: Loaded inference model.
        contacts: Optional contact conditioning.
        num_samples: Number of structures to generate.
        batch_size: Maximum structures generated per denoising call.
        sampler: Sampling preset (``"edm"``, ``"edm_gd"``, ``"edm_fk"``, or
            ``"edm_fkc"``). For custom settings, use ``model.embed`` and
            ``sample_with_embeddings``.
        num_steps: Optional denoising step count instead of the preset default.
        potentials: Optional steering potentials instead of the preset defaults.
            Requires ``"edm_gd"``, ``"edm_fk"``, or ``"edm_fkc"``.
        seed: Optional Torch random seed.
        progress: Whether to display sampling progress.
        partial_dir: Directory for saving and reusing completed raw batches.
            Reuse only with identical inputs and settings and one writer.
        output_dir: Optional directory for all samples, topology, system, and
            physicality.json. When set, checking and saving are automatic.
            Completed batches in output_dir/partial are reused and removed after
            successful saving unless an explicit partial_dir is supplied.
            Reuse only with identical inputs and settings and one writer.
        physicality_mode: Check thresholds for saved outputs; no frames are removed.
            Without output_dir, no checks are run and no final outputs are written.

    Returns:
        Predicted structures as an MDTraj trajectory in nanometers.
    """
    return _embed_and_sample(
        system,
        model=model,
        mode="structure",
        contacts=contacts,
        num_samples=num_samples,
        batch_size=batch_size,
        sampler=sampler,
        num_steps=num_steps,
        potentials=potentials,
        seed=seed,
        progress=progress,
        partial_dir=partial_dir,
        output_dir=output_dir,
        physicality_mode=physicality_mode,
    )


def sample_equilibrium(
    system: System,
    *,
    model: InferenceModel[TrunkEmbeddings],
    contacts: list[Contact] | None = None,
    num_samples: int = 8,
    batch_size: int | None = None,
    sampler: SamplerPreset = "edm",
    num_steps: int | None = None,
    potentials: Sequence[Potential] | None = None,
    seed: int | None = None,
    progress: bool = True,
    partial_dir: Path | None = None,
    output_dir: str | Path | None = None,
    physicality_mode: PhysicalityFilterMode | str = PhysicalityFilterMode.HIGH_QUALITY,
) -> mdtraj.Trajectory:
    """Sample equilibrium conformations with a loaded model and automatically computed MSA.

    Args:
        system: Molecular system to sample.
        model: Loaded inference model.
        contacts: Optional contact conditioning.
        num_samples: Number of conformations to generate.
        batch_size: Maximum conformations generated per denoising call.
        sampler: Sampling preset (``"edm"``, ``"edm_gd"``, ``"edm_fk"``, or
            ``"edm_fkc"``). For custom settings, use ``model.embed`` and
            ``sample_with_embeddings``.
        num_steps: Optional denoising step count instead of the preset default.
        potentials: Optional steering potentials instead of the preset defaults.
            Requires ``"edm_gd"``, ``"edm_fk"``, or ``"edm_fkc"``.
        seed: Optional Torch random seed.
        progress: Whether to display sampling progress.
        partial_dir: Directory for saving and reusing completed raw batches.
            Reuse only with identical inputs and settings and one writer.
        output_dir: Optional directory for all samples, topology, system, and
            physicality.json. When set, checking and saving are automatic.
            Completed batches in output_dir/partial are reused and removed after
            successful saving unless an explicit partial_dir is supplied.
            Reuse only with identical inputs and settings and one writer.
        physicality_mode: Check thresholds for saved outputs; no frames are removed.
            Without output_dir, no checks are run and no final outputs are written.

    Returns:
        Equilibrium conformations as an MDTraj trajectory in nanometers.
    """
    return _embed_and_sample(
        system,
        model=model,
        mode="equilibrium",
        contacts=contacts,
        num_samples=num_samples,
        batch_size=batch_size,
        sampler=sampler,
        num_steps=num_steps,
        potentials=potentials,
        seed=seed,
        progress=progress,
        partial_dir=partial_dir,
        output_dir=output_dir,
        physicality_mode=physicality_mode,
    )


def write_sampling_system(
    system: System,
    *,
    mode: SamplingMode,
    contacts: list[Contact] | None,
    out: Path,
) -> None:
    """Write the resolved sampling system and conditions."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "system.yaml").write_text(
        yaml.safe_dump(
            {
                **system.model_dump(mode="json", exclude_none=True),
                "mode": mode,
                **({"contacts": [c.model_dump(mode="json") for c in contacts]} if contacts else {}),
            }
        )
    )


def write_sampling_output(
    trajectory: mdtraj.Trajectory,
    *,
    system: System,
    mode: SamplingMode,
    contacts: list[Contact] | None,
    out: Path,
    physicality_filter: PhysicalityFilterMode,
) -> Path:
    """Write all frames and their physicality results in original order."""
    structure = AllAtomStructure.from_system(system, mode=mode, contacts=contacts)
    report = check_physicality(
        trajectory,
        physicality_filter,
        expected_extra_bonds=expected_extra_bonds_from_system(system, trajectory.topology),
    )
    save_topology_cif(structure, system, trajectory.xyz[0] * 10.0, out / "topology.cif")
    trajectory.save_xtc(str(out / "samples.xtc"))
    report.save(out / "physicality.json")
    failed = len(trajectory) - int(report.passed.sum())
    if failed:
        logger.warning(
            "%d of %d samples failed physicality checks.",
            failed,
            len(trajectory),
        )
    logger.info("Wrote %d samples and physicality results to %s", len(trajectory), out)
    return out / "samples.xtc"
