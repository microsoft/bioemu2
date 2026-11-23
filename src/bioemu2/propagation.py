"""Trajectory simulation by iterative propagation with the dynamics model."""

import logging
import math
from collections import Counter
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import mdtraj
import numpy as np
import torch
from tqdm.auto import trange

from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.data_interface import Contact, System
from bioemu2.denoising.rigid_align import weighted_rigid_align
from bioemu2.denoising.samplers import EDMSampler, GuidedFKSampler, Sampler
from bioemu2.model import InferenceModel
from bioemu2.msa.alignment import get_msa
from bioemu2.physicality import (
    PhysicalityFilterMode,
    _select_expected_extra_bonds,
    check_allatom_physicality,
    normalize_physicality_mode,
)
from bioemu2.sampling import (
    BATCH_DIR,
    SamplerPreset,
    expected_extra_bonds_from_system,
    generate_samples,
    sampler_from_preset,
    write_sampling_output,
)
from bioemu2.structure.preparation import prepare_sampling_inputs
from bioemu2.trajectory_io import (
    coords_angstrom_to_trajectory,
    map_cif_atom_order,
    trajectory_to_coords_angstrom,
)

logger = logging.getLogger(__name__)


@cache
def _warn_custom_step_scale(step_scale: float) -> None:
    logger.warning(
        "Propagation normally uses step_scale=1.0; using the supplied "
        "step_scale=%s unchanged. This warning will not be shown again for this scale.",
        step_scale,
    )


def simulate_trajectory(
    system: System | None = None,
    *,
    model: InferenceModel[TrunkEmbeddings],
    x0: mdtraj.Trajectory | str | Path,
    lag_time_ns: float,
    num_trajectory_steps: int,
    contacts: list[Contact] | None = None,
    batch_size: int | None = None,
    sampler: SamplerPreset = "edm",
    seed: int | None = None,
    display_propagation_progress: bool = True,
    output_dir: str | Path | None = None,
    physicality_mode: PhysicalityFilterMode | str = PhysicalityFilterMode.HIGH_QUALITY,
) -> list[mdtraj.Trajectory]:
    """Simulate time-ordered trajectories with a loaded model and acquired MSA.

    Each starting frame produces an independent trajectory including its
    initial frame. Pass a CIF path as ``x0`` to infer the system, or pass a
    ``System`` with an MDTraj trajectory. The CIF must contain all system atoms;
    inferred systems are intended for simple BioEmu2 IID sampling outputs.
    ColabFold MSA results are cached on disk. To customize the sampler or
    supply an explicit MSA, embed the system and use ``simulate_trajectory_with_embeddings``.

    Args:
        system: Molecular system; inferred from a CIF path when omitted.
        model: Loaded inference model with propagation support.
        x0: Starting frames with the system's topology, or a CIF path.
        lag_time_ns: Positive time lag in nanoseconds.
        num_trajectory_steps: Positive number of propagation steps.
        contacts: Optional contact conditioning.
        batch_size: Maximum trajectories per denoiser call.
        sampler: Sampling preset; ``"edm"`` or ``"edm_gd"``. Resampling
            presets are not supported for multi-step trajectories.
        seed: Optional Torch random seed. Batch ``i`` of step ``t`` uses
            ``seed + t * num_batches + i``. Without it, each step draws a seed from
            the global RNG.
        display_propagation_progress: Whether to display propagation progress.
        output_dir: Optional directory for ``samples_000000.xtc`` and
            ``physicality_000000.json``, ... (one each per trajectory), topology.cif
            and system.yaml. Raw batches
            are kept as ``output_dir/batches/step_{t}/samples_{start}-{stop}.npy``; a
            rerun with the same directory continues from them after checking the
            inputs recorded in ``run.json`` and the initial ``x0``. Without it,
            nothing is written.
        physicality_mode: Check thresholds for saved outputs; no frames are removed.

    Returns:
        One time-ordered MDTraj trajectory per starting frame, in input order.
    """
    if num_trajectory_steps < 1:
        raise ValueError("num_trajectory_steps must be a positive integer.")
    configured_sampler = sampler_from_preset(sampler, mode="equilibrium")
    if (
        getattr(configured_sampler, "resampling_interval", None) is not None
        or getattr(configured_sampler, "ess_threshold", None) is not None
    ):
        raise ValueError(
            "Multi-step trajectory simulation does not support samplers with resampling enabled."
        )
    if isinstance(x0, str | Path):
        if system is None:
            system = System.from_cif(x0)
        x0 = map_cif_atom_order(system, x0)
    elif system is None:
        raise ValueError("system is required when x0 is a trajectory; pass a CIF path to infer it.")
    if not model.supports_propagation:
        raise ValueError("The loaded checkpoint does not support propagation.")
    msa = get_msa(system)
    trunk_embeddings = model.embed(system, msa=msa, mode="equilibrium", contacts=contacts)
    return simulate_trajectory_with_embeddings(
        system,
        trunk_embeddings,
        x0=x0,
        lag_time_ns=lag_time_ns,
        num_trajectory_steps=num_trajectory_steps,
        model=model,
        batch_size=batch_size,
        sampler=configured_sampler,
        seed=seed,
        display_propagation_progress=display_propagation_progress,
        output_dir=output_dir,
        physicality_mode=physicality_mode,
        contacts=contacts,
    )


def simulate_trajectory_with_embeddings(
    system: System,
    trunk_embeddings: TrunkEmbeddings,
    *,
    x0: mdtraj.Trajectory,
    lag_time_ns: float,
    num_trajectory_steps: int,
    model: InferenceModel[TrunkEmbeddings],
    batch_size: int | None = None,
    sampler: Sampler,
    seed: int | None = None,
    display_propagation_progress: bool = True,
    output_dir: str | Path | None = None,
    physicality_mode: PhysicalityFilterMode | str = PhysicalityFilterMode.HIGH_QUALITY,
    contacts: list[Contact] | None = None,
) -> list[mdtraj.Trajectory]:
    """Simulate independent time-lagged trajectories with reusable embeddings.

    Each frame of ``x0`` starts one independent trajectory. Every endpoint is
    aligned to its own preceding frame, and the returned trajectories include
    their starting frame at time zero. ``lag_time_ns`` is specified in
    nanoseconds because trajectory lags are typically on that scale; returned
    frame times follow MDTraj's convention and are stored in picoseconds.
    Each returned ``mdtraj.Trajectory`` contains frames in time order, rather
    than an unordered collection of samples.

    At the end of the rollout, all frames are checked for physicality: without
    ``output_dir`` against the ``"high_quality"`` thresholds, with the result
    logged; with it, against ``physicality_mode``, saved per trajectory to
    ``physicality_000000.json``, ... No
    frames are removed and physicality failures do not interrupt propagation.

    Args:
        system: Molecular system.
        trunk_embeddings: Reusable output from ``model.embed`` for the system.
        x0: One or more starting conformations with the system's topology.
        lag_time_ns: Positive time lag between consecutive frames in nanoseconds.
            We recommend 1 to 1000 (1 ns to 1 µs), the range the model mostly
            saw during training. Lags longer than a few milliseconds effectively
            disable conditioning on ``x0``; use ``sample_equilibrium()`` instead.
        num_trajectory_steps: Positive number of time-lagged propagation steps.
        model: Loaded inference model.
        batch_size: Maximum trajectories per denoiser call.
        sampler: Preconfigured sampling algorithm, including its step scale.
            Samplers with resampling enabled are not supported.
        seed: Optional Torch random seed. Batch ``i`` of step ``t`` uses
            ``seed + t * num_batches + i``. Without it, each step draws a seed from
            the global RNG.
        display_propagation_progress: Whether to display propagation progress.
        output_dir: Optional directory for ``samples_000000.xtc`` and
            ``physicality_000000.json``, ... (one each per trajectory), topology.cif
            and system.yaml. Raw batches
            are kept as ``output_dir/batches/step_{t}/samples_{start}-{stop}.npy``; a
            rerun with the same directory continues from them after checking the
            inputs recorded in ``run.json`` and the initial ``x0``. Without it,
            nothing is written.
        physicality_mode: Check thresholds for saved outputs; no frames are removed.
        contacts: Contacts used for ``trunk_embeddings``, recorded in system.yaml.

    Returns:
        One time-ordered MDTraj trajectory per starting frame, in input order.

    Raises:
        ValueError: If ``num_trajectory_steps`` is not positive, propagation
            inputs are invalid, or ``output_dir/batches`` holds batches with
            different inputs recorded in ``run.json`` or a different ``x0``.
    """
    if num_trajectory_steps < 1:
        raise ValueError("num_trajectory_steps must be a positive integer.")
    if not model.supports_propagation:
        raise ValueError("The loaded checkpoint does not support propagation.")
    if (
        getattr(sampler, "resampling_interval", None) is not None
        or getattr(sampler, "ess_threshold", None) is not None
    ):
        raise ValueError(
            "Multi-step trajectory simulation does not support samplers with resampling enabled."
        )
    out = Path(output_dir) if output_dir is not None else None
    physicality_mode = normalize_physicality_mode(physicality_mode)
    # Seeds rise linearly over the rollout: batch i of step t uses seed + t * num_batches + i.
    if len(x0) == 0:
        raise ValueError("x0 must be a non-empty mdtraj.Trajectory.")
    num_batches = len(range(0, len(x0), batch_size or len(x0)))
    # Each step holds one frame per starting configuration in the same order.
    step_frames: list[mdtraj.Trajectory] = [x0]
    if out is not None:
        _check_x0(out / BATCH_DIR / "step_000000", x0)
    for step in trange(
        num_trajectory_steps, desc="Propagating", disable=not display_propagation_progress
    ):
        step_frames.append(
            _propagate_with_embeddings(
                system,
                trunk_embeddings,
                x0=step_frames[-1],
                lag_time_ns=lag_time_ns,
                model=model,
                batch_size=batch_size,
                sampler=sampler,
                seed=None if seed is None else seed + step * num_batches,
                progress=False,
                batch_dir=out / BATCH_DIR / f"step_{step:06d}" if out is not None else None,
                contacts=contacts,
            )
        )

    trajectories = join_trajectory_steps(step_frames, lag_time_ns=lag_time_ns)
    if out is None:
        _report_trajectory_physicality(trajectories, system)
    else:
        for i, trajectory in enumerate(trajectories):
            write_sampling_output(
                trajectory,
                system=system,
                mode="equilibrium",
                contacts=contacts,
                out=out,
                physicality_filter=physicality_mode,
                suffix=f"_{i:06d}",
            )
    return trajectories


def join_trajectory_steps(
    step_frames: Sequence[mdtraj.Trajectory],
    *,
    lag_time_ns: float,
) -> list[mdtraj.Trajectory]:
    """Reorganize step-batched frames into individual time-ordered trajectories.

    Input is ordered as ``[step][trajectory]``: each step contains one frame
    per trajectory in the same order, starting with the initial frames.
    Output is ordered as ``[trajectory][step]``.

    Args:
        step_frames: Starting frames followed by endpoint frames for each step.
        lag_time_ns: Positive time between consecutive steps in nanoseconds.

    Returns:
        One trajectory per frame in each step. MDTraj frame times are in
        picoseconds.

    Raises:
        ValueError: If no frames are provided, step sizes differ, or the lag is
            not finite and positive.
    """
    if len(step_frames) == 0 or step_frames[0].n_frames == 0:
        raise ValueError("step_frames must contain at least one frame.")
    num_trajectories = step_frames[0].n_frames
    if any(frames.n_frames != num_trajectories for frames in step_frames):
        counts = Counter(frames.n_frames for frames in step_frames)
        raise ValueError(
            f"Every element of step_frames must contain {num_trajectories} frames; "
            f"observed counts: {dict(counts)}."
        )
    if not math.isfinite(lag_time_ns) or lag_time_ns <= 0:
        raise ValueError("lag_time_ns must be finite and positive.")

    times_ps = 1000 * lag_time_ns * np.arange(len(step_frames))
    trajectories = [
        mdtraj.join([frames[i] for frames in step_frames]) for i in range(num_trajectories)
    ]
    for trajectory in trajectories:
        trajectory.time = times_ps
    return trajectories


def _report_trajectory_physicality(trajectories: list[mdtraj.Trajectory], system: System) -> None:
    topology = trajectories[0].topology
    protein_atom_indices = topology.select("protein")
    if len(protein_atom_indices) == 0:
        logger.info(
            "Trajectory physicality report skipped because the system has no protein atoms."
        )
        return
    expected_extra_bonds = _select_expected_extra_bonds(
        expected_extra_bonds_from_system(system, topology), protein_atom_indices, topology.n_atoms
    )

    num_physical = 0
    num_frames = 0
    violation_totals: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    for trajectory in trajectories:
        protein_trajectory = trajectory.atom_slice(protein_atom_indices)
        protein_trajectory.topology.create_standard_bonds()
        is_physical, stats, reasons = check_allatom_physicality(
            protein_trajectory,
            mode=PhysicalityFilterMode.HIGH_QUALITY,
            expected_extra_bonds=expected_extra_bonds,
        )
        num_physical += int(is_physical.sum())
        num_frames += len(trajectory)
        violation_totals.update(
            {name: int(values.sum()) for name, values in stats.items() if values.any()}
        )
        failure_reasons.update(reason for reason in reasons if reason is not None)

    message = (
        f"Trajectory physicality report: {num_physical}/{num_frames} frames passed "
        f"high_quality thresholds; violation totals: {dict(violation_totals)}"
    )
    if num_physical == num_frames:
        logger.info(message)
    else:
        logger.warning(
            f"{message}; first failure reasons: {dict(failure_reasons)}. All frames are retained."
        )


def propagate(
    system: System | None = None,
    *,
    model: InferenceModel[TrunkEmbeddings],
    x0: mdtraj.Trajectory | str | Path,
    lag_time_ns: float,
    num_samples: int | None = None,
    contacts: list[Contact] | None = None,
    batch_size: int | None = None,
    sampler: SamplerPreset = "edm",
    seed: int | None = None,
    progress: bool = True,
    output_dir: str | Path | None = None,
    physicality_mode: PhysicalityFilterMode | str = PhysicalityFilterMode.HIGH_QUALITY,
) -> mdtraj.Trajectory:
    """Sample time-lagged endpoints with a loaded model and acquired MSA.

    Pass a CIF path as ``x0`` to load and map its coordinates into the system's
    atom order. If ``system`` is omitted, it is inferred with ``System.from_cif``.
    CIF-only inference is intended for simple BioEmu2 IID sampling outputs,
    not arbitrary PDB structures or MD frames. Supply an explicit system for
    modified residues, SMILES ligands, branched entities, or explicit bonds,
    and pass contacts separately. Every atom required by the system must have
    coordinates; supplying a system does not repair an incomplete CIF.

    ``x0`` is the conditioning set: each endpoint is sampled conditional on a
    starting frame and the lag, then rigidly superposed on that frame. A CIF
    path loads one starting conformation. A trajectory produces one endpoint
    per frame. For a CIF path or a single-frame trajectory, pass ``num_samples``
    to draw several endpoints from that frame. Embed the system and use
    ``propagate_with_embeddings`` to configure the sampler or reuse embeddings.

    Args:
        system: Molecular system. Inferred from a CIF path when omitted;
            required when ``x0`` is an MDTraj trajectory.
        model: Loaded inference model with propagation support.
        x0: Starting conformations with the system's topology, or a CIF path.
        lag_time_ns: Positive time lag in nanoseconds. We recommend 1 to 1000
            (1 ns to 1 µs), the range the model mostly saw during training.
            Lags longer than a few milliseconds effectively disable conditioning
            on ``x0``; use ``sample_equilibrium()`` instead.
        num_samples: Number of endpoints to draw from a CIF path or single-frame
            trajectory. Omit for one endpoint per starting frame.
        contacts: Optional contact conditioning.
        batch_size: Maximum frames per denoiser call.
        sampler: Sampling preset (``"edm"``, ``"edm_gd"``, ``"edm_fk"``, or
            ``"edm_fkc"``). Resampling presets require identical starting frames.
        seed: Optional Torch random seed.
        progress: Whether to display sampling progress.
        output_dir: Optional directory for samples.xtc (endpoints only, without
            ``x0``), topology.cif, system.yaml and physicality.json; see
            ``sample_with_embeddings`` for raw batches. Without it, nothing is written.
        physicality_mode: Check thresholds for saved outputs; no frames are removed.

    Returns:
        One endpoint per starting frame, or ``num_samples`` endpoints from a
        single starting frame, as an MDTraj trajectory in nanometers.
    """
    if num_samples is not None and num_samples < 1:
        raise ValueError("num_samples must be a positive integer.")
    if isinstance(x0, str | Path):
        if system is None:
            system = System.from_cif(x0)
        x0 = map_cif_atom_order(system, x0)
    elif system is None:
        raise ValueError("system is required when x0 is a trajectory; pass a CIF path to infer it.")
    if num_samples is not None:
        if x0.n_frames != 1:
            raise ValueError("num_samples requires a CIF path or a single-frame trajectory.")
        x0 = x0[[0] * num_samples]
    configured_sampler = sampler_from_preset(sampler, mode="equilibrium")
    if not model.supports_propagation:
        raise ValueError("The loaded checkpoint does not support propagation.")
    msa = get_msa(system)
    trunk_embeddings = model.embed(system, msa=msa, mode="equilibrium", contacts=contacts)
    return propagate_with_embeddings(
        system,
        trunk_embeddings,
        x0=x0,
        lag_time_ns=lag_time_ns,
        model=model,
        batch_size=batch_size,
        sampler=configured_sampler,
        seed=seed,
        progress=progress,
        output_dir=output_dir,
        physicality_mode=physicality_mode,
        contacts=contacts,
    )


def _check_x0(batch_dir: Path, x0: mdtraj.Trajectory) -> None:
    """Keep the starting frames next to ``run.json``; refuse batches from other ones."""
    path = batch_dir / "x0.npy"
    if any(batch_dir.glob("samples_*.done")):
        if not path.is_file():
            raise ValueError(
                f"{batch_dir} holds batches but no x0.npy; use a new output directory or delete it."
            )
        if not np.array_equal(np.load(path), x0.xyz):
            raise ValueError(
                f"{batch_dir} holds batches with a different x0; "
                "use a new output directory or delete it."
            )
    else:
        batch_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, x0.xyz, allow_pickle=False)


def propagate_with_embeddings(
    system: System,
    trunk_embeddings: TrunkEmbeddings,
    *,
    x0: mdtraj.Trajectory,
    lag_time_ns: float,
    model: InferenceModel[TrunkEmbeddings],
    batch_size: int | None = None,
    sampler: Sampler,
    seed: int | None = None,
    progress: bool = True,
    output_dir: str | Path | None = None,
    physicality_mode: PhysicalityFilterMode | str = PhysicalityFilterMode.HIGH_QUALITY,
    contacts: list[Contact] | None = None,
) -> mdtraj.Trajectory:
    """Propagate starting conformations using precomputed embeddings.

    Use this operation when reusing one embedding across many propagation calls.
    ``propagate`` is the simpler entry point when embedding reuse is unnecessary.
    Each endpoint is aligned to its corresponding ``x0`` frame. Resampling
    requires identical frames within each resampling group (all frames if the
    group size is ``None``). Batch size must align with group boundaries.

    With ``output_dir``, endpoints (without ``x0``) are saved as samples.xtc with
    topology.cif, system.yaml and physicality.json; completed raw batches in
    ``output_dir/batches`` are kept so a rerun with the same directory continues.
    Without it, nothing is written. ``physicality_mode`` selects the check
    thresholds; no frames are removed. ``contacts`` are recorded in system.yaml.
    """
    out = Path(output_dir) if output_dir is not None else None
    physicality_mode = normalize_physicality_mode(physicality_mode)
    batch_dir = out / BATCH_DIR if out is not None else None
    if batch_dir is not None:
        _check_x0(batch_dir, x0)
    endpoints = _propagate_with_embeddings(
        system,
        trunk_embeddings,
        x0=x0,
        lag_time_ns=lag_time_ns,
        model=model,
        batch_size=batch_size,
        sampler=sampler,
        seed=seed,
        progress=progress,
        batch_dir=batch_dir,
        contacts=contacts,
    )
    if out is not None:
        saved = endpoints[:]
        saved.time = np.full(saved.n_frames, lag_time_ns * 1000.0)
        write_sampling_output(
            saved,
            system=system,
            mode="equilibrium",
            contacts=contacts,
            out=out,
            physicality_filter=physicality_mode,
        )
    return endpoints


def _propagate_with_embeddings(
    system: System,
    trunk_embeddings: TrunkEmbeddings,
    *,
    x0: mdtraj.Trajectory,
    lag_time_ns: float,
    model: InferenceModel[TrunkEmbeddings],
    batch_size: int | None,
    sampler: Sampler,
    seed: int | None,
    progress: bool,
    batch_dir: Path | None,
    contacts: list[Contact] | None,
) -> mdtraj.Trajectory:
    """Propagate ``x0`` by one lag step; save raw batches only, never final outputs.

    Shared by ``propagate_with_embeddings``, which runs one step, and
    ``simulate_trajectory_with_embeddings``, which runs one per rollout step.
    ``propagate_with_embeddings`` generates its final .xtc and .cif file after a single
    call to ``_propagate_with_embeddings``. ``simulate_trajectory_with_embeddings``
    calls ``_propagate_with_embeddings`` in a loop and generates its final trajectory
    file after the loop.

    Validates the inputs, conditions each sample on its ``x0`` frame, samples
    with resumable batches in ``batch_dir``, and superposes each endpoint on its
    starting frame.
    """
    if not model.supports_propagation:
        raise ValueError("The loaded checkpoint does not support propagation.")
    topology, batch = prepare_sampling_inputs(system, device=model.device)
    if len(x0) == 0:
        raise ValueError("x0 must be a non-empty mdtraj.Trajectory.")
    if x0.topology != topology:
        raise ValueError("x0 topology must exactly match the embedded system topology.")
    if not math.isfinite(lag_time_ns) or lag_time_ns <= 0:
        raise ValueError("lag_time_ns must be finite and positive.")
    input_coords = trajectory_to_coords_angstrom(x0, device=model.device, dtype=torch.float32)
    if not torch.isfinite(input_coords).all():
        raise ValueError("x0 coordinates must be finite.")
    if lag_time_ns * 1000.0 < torch.finfo(torch.float32).tiny:
        raise ValueError("lag_time_ns underflows the model's float32 lag.")
    if batch_size is None:
        batch_size = len(x0)
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(sampler, EDMSampler | GuidedFKSampler) and sampler.step_scale != 1.0:
        _warn_custom_step_scale(sampler.step_scale)
    if (
        getattr(sampler, "resampling_interval", None) is not None
        or getattr(sampler, "ess_threshold", None) is not None
    ):
        # Resampling copies particles within groups of `resampling_group_size`
        # consecutive samples; every group must share one conditioning frame.
        # None resamples a whole step batch, so then all of x0 must be identical.
        group = getattr(sampler, "resampling_group_size", None) or len(x0)
        if group < len(x0) and batch_size < len(x0) and batch_size % group:
            raise ValueError(
                f"batch_size={batch_size} must be a multiple of resampling_group_size={group}."
            )
        # Check that the conditions in ``x0`` are identical and thus safe for group resampling
        for start in range(0, len(x0), group):
            block = input_coords[start : start + group]
            if not torch.equal(block, block[:1].expand_as(block)):
                raise ValueError(
                    f"x0 frames {start}:{start + len(block)} differ; with resampling every "
                    f"block of {group} consecutive frames must be identical "
                    "(all frames when resampling_group_size is None)."
                )

    # Conditions live on the sample axis: pad each frame to the model's atom dimension.
    coords_cond = input_coords.new_zeros(len(input_coords), batch.atom_pad_mask.shape[1], 3)
    coords_cond[:, : topology.n_atoms] = input_coords - input_coords.mean(dim=1, keepdim=True)

    n = len(x0)
    coords = generate_samples(
        num_samples=n,
        batch_size=batch_size,
        score_model=model,
        sampler=sampler,
        num_atoms=topology.n_atoms,
        system=system,
        batch=batch.replace(
            coords_cond=coords_cond,
            is_dynamics_conditioned=torch.ones((1, n), dtype=torch.bool, device=coords_cond.device),
            dt=torch.full((1, n), lag_time_ns * 1000.0, device=coords_cond.device),
        ),
        trunk_embeddings=trunk_embeddings,
        progress=progress,
        batch_dir=batch_dir,
        seed=seed,
        mode="equilibrium",
        contacts=contacts,
    )
    trajectory = coords_angstrom_to_trajectory(coords, topology)

    # Superpose each endpoint on its own input frame.
    mask = torch.ones(x0.xyz.shape[:2])
    aligned, _, _ = weighted_rigid_align(
        torch.as_tensor(trajectory.xyz), torch.as_tensor(x0.xyz), mask, mask
    )
    trajectory.xyz = aligned.numpy()
    return trajectory
