"""Multi-GPU Python sampling operations, to be launched with ``torchrun``.

See ``docs/context-parallel-sampling.md`` for usage instructions.
"""

import warnings
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import mdtraj
import torch
import torch.distributed as dist
from tqdm import trange

from bioemu2 import HAS_TRITON
from bioemu2.biomoa.models.edm_schedule import EDMSchedule
from bioemu2.biomoa.models.score_model import DenoiseOutput
from bioemu2.biomoa.models.score_model import (
    TrunkEmbeddings as SerialTrunkEmbeddings,
)
from bioemu2.biomoa.modules.diffusion import (
    StaticContext as SerialStaticContext,
)
from bioemu2.conditioning.method import SamplingMode
from bioemu2.context_parallel.diagnostics import attach_trace_hooks
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.features import (
    build_shard_msa_feats,
    build_shard_pair_feats,
    pad_feats_for_cp,
)
from bioemu2.context_parallel.models import wrap_score_model_for_cp
from bioemu2.context_parallel.models.score_model import (
    BioMoaCDistributed,
    CPTrunkEmbeddings,
)
from bioemu2.context_parallel.modules.diffusion import CPStaticContext
from bioemu2.context_parallel.utils.padding import make_cp_pad_sizes
from bioemu2.context_parallel.utils.session import cp_session as session
from bioemu2.data_interface import Contact, System
from bioemu2.denoising import cp_sync
from bioemu2.denoising.samplers import EDMSampler, FKCSampler, GuidedFKSampler, Sampler
from bioemu2.denoising.steering.potentials import Potential
from bioemu2.model import _InferenceModel
from bioemu2.model import load_model as serial_load_model
from bioemu2.msa.alignment import MSA
from bioemu2.physicality import PhysicalityFilterMode, normalize_physicality_mode
from bioemu2.precision import autocast_ctx, get_autocast_dtype, matmul_precision_ctx
from bioemu2.sampling import (
    SamplerPreset,
    sampler_from_preset,
    warn_batch_size,
    write_sampling_output,
    write_sampling_system,
)
from bioemu2.structure import AllAtomStructure, BatchAllAtomStructure
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.inference_inputs import topology_from_structure
from bioemu2.structure.preparation import prepare_structure
from bioemu2.trajectory_io import coords_angstrom_to_trajectory

__all__ = [
    "ModelWithSettings",
    "load_model",
    "predict_structure",
    "propagate",
    "propagate_with_embeddings",
    "sample_equilibrium",
    "sample_with_embeddings",
    "session",
]


@dataclass(frozen=True)
class ModelWithSettings:
    """CP weights and inference settings, owned by one active process group.

    Sampling calls the contained
    ``BioMoaCDistributed`` score model with its prepared CP context.
    """

    score_model: BioMoaCDistributed = field(repr=False, compare=False)
    device: torch.device
    _autocast_dtype: torch.dtype | None = field(repr=False, compare=False)
    process_group: dist.ProcessGroup = field(repr=False, compare=False)
    num_steps_recycle: int = 3
    matmul_precision: str | None = field(default=None, repr=False, compare=False)

    def embed(
        self,
        system: System,
        *,
        msa: MSA | None,
        mode: SamplingMode,
        contacts: list[Contact] | None = None,
    ) -> CPTrunkEmbeddings:
        """Compute reusable CP sequence-trunk embeddings for a system.

        Call on every rank inside ``session()`` with the same molecular system and contacts.
        Only rank 0 needs ``msa``;
        other ranks can pass ``None``.
        """
        _validate_model(self)
        contacts = [] if contacts is None else contacts
        structures: list[AllAtomStructure | None] = [None]
        if dist.get_rank() == 0:
            if msa is None:
                raise ValueError("msa is required on rank 0.")
            structures[0] = prepare_structure(system, msa=msa, mode=mode, contacts=contacts)
        dist.broadcast_object_list(structures, src=0, group=DistributedManager().group["cpu"])
        structure = structures[0]
        assert structure is not None
        structure = structure.replace(coords=torch.zeros_like(structure.coords))
        sizes = make_cp_pad_sizes(
            n_token=structure.num_tokens,
            n_atom=structure.num_nodes,
            n_msa=structure.msa.shape[1],
            atoms_per_window=self.score_model.atoms_per_window_queries,
            mesh=DistributedManager().device_mesh_subgroups,
        )
        batch = batch_collate([structure], window_size=sizes.n_atom_final)
        # Slice each rank's pair and MSA slabs on the host so the dense (B, N, N, *) contact
        # tensors and the padded MSA are never allocated on the accelerator.
        pair_feats = build_shard_pair_feats(
            batch, sizes, DistributedManager().device_mesh_subgroups, device=self.device
        )
        pair_feats.update(
            build_shard_msa_feats(
                batch, sizes, DistributedManager().device_mesh_subgroups, device=self.device
            )
        )
        for key in pair_feats:
            if key in batch:
                delattr(batch, key)
        batch = batch.to(self.device)
        batch.update(pad_feats_for_cp(batch, sizes))
        batch.update(pair_feats)
        return self._run_trunk(batch)

    @property
    def edm_schedule(self) -> EDMSchedule:
        return self.score_model.edm_schedule

    @property
    def supports_propagation(self) -> bool:
        return False

    def _run_trunk(
        self,
        batch: BatchAllAtomStructure,
    ) -> CPTrunkEmbeddings:
        score = self.score_model
        score.sequence_embedder.to(self.device)
        with (
            torch.no_grad(),
            autocast_ctx(device_type=self.device.type, dtype=self._autocast_dtype),
        ):
            trunk = score.compute_trunk_embeddings(
                res_type=batch.res_type,
                profile=batch.profile,
                deletion_mean=batch.deletion_mean,
                pocket_feature=batch.pocket_feature,
                asym_id=batch.asym_id,
                entity_id=batch.entity_id,
                residue_index=batch.residue_index,
                sym_id=batch.sym_id,
                token_pad_mask=batch.token_pad_mask,
                token_bonds=batch.token_bonds,
                template_coords=batch.template_coords,
                charge=batch.charge,
                element=batch.element,
                name_encoding=batch.name_encoding,
                ref_space_uid=batch.ref_space_uid,
                atom_pad_mask=batch.atom_pad_mask,
                token_idx_at_atom=batch.token_idx_at_atom,
                msa=batch.msa,
                has_deletion=batch.has_deletion,
                deletion_value=batch.deletion_value,
                msa_mask=batch.msa_mask,
                msa_paired=getattr(batch, "msa_paired", None),
                method_feature=getattr(batch, "method_feature", None),
                contact_conditioning=getattr(batch, "contact_conditioning", None),
                contact_threshold=getattr(batch, "contact_threshold", None),
                num_steps_recycle=self.num_steps_recycle,
            )
            dist.barrier()
            score.sequence_embedder.to("cpu")
            torch.cuda.empty_cache()
            dist.barrier()
        return trunk

    @torch.no_grad()
    def compute_static_context(
        self,
        batch: BatchAllAtomStructure,
        trunk_embeddings: SerialTrunkEmbeddings | CPTrunkEmbeddings,
        multiplicity: int,
    ) -> SerialStaticContext | CPStaticContext:
        if not isinstance(trunk_embeddings, CPTrunkEmbeddings):
            raise TypeError(
                f"Expected type {CPTrunkEmbeddings}, got type {type(trunk_embeddings)}."
            )
        return self.score_model.compute_static_context(
            trunk_embeddings,
            template_coords=batch.template_coords,
            charge=batch.charge,
            element=batch.element,
            name_encoding=batch.name_encoding,
            ref_space_uid=batch.ref_space_uid,
            atom_pad_mask=batch.atom_pad_mask,
            atom_mask=batch.atom_mask,
            token_idx_at_atom=batch.token_idx_at_atom,
            token_pad_mask=batch.token_pad_mask,
            token_center_atom_index=batch.token_center_atom_index,
            multiplicity=multiplicity,
            coords_cond=batch.coords_cond,
            is_dynamics_conditioned=batch.is_dynamics_conditioned,
            dt=batch.dt,
        )

    def denoise(
        self,
        context: SerialStaticContext | CPStaticContext,
        trunk_embeddings: SerialTrunkEmbeddings | CPTrunkEmbeddings,
        coords: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> DenoiseOutput:
        if not isinstance(context, CPStaticContext):
            raise TypeError(f"Expected type {CPStaticContext}, got type {type(context)}.")
        if not isinstance(trunk_embeddings, CPTrunkEmbeddings):
            raise TypeError(
                f"Expected type {CPTrunkEmbeddings}, got type {type(trunk_embeddings)}."
            )
        with matmul_precision_ctx(self.matmul_precision):
            return self.score_model.denoise(context, trunk_embeddings, coords, sigmas)


def _validate_model(model: ModelWithSettings) -> None:
    if model.process_group is not dist.group.WORLD:
        raise ValueError("This CP model belongs to a different or closed session.")


@contextmanager
def _rng_seed(seed: int | None, device: torch.device) -> Generator[None, None, None]:
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[device.index]):
        torch.random.default_generator.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        yield


def load_model(
    checkpoint: str | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    precision: str = "bf16-mixed",
    matmul_precision: str | None = "high",
) -> ModelWithSettings:
    """CP analogue of ``bioemu2.model.load_model``.

    Call on every rank inside ``session()`` with identical arguments. Local rank 0
    on each node loads first to populate that node's checkpoint cache. After a
    global barrier, the other ranks load locally.
    The session chooses each rank's CUDA device; there is no ``device`` argument.
    Checkpoint arguments, ``cache_dir``, ``precision``, and ``matmul_precision`` have
    the same meaning as in the serial API. ``matmul_precision`` applies to the denoiser's
    float32 torch matmul but specialized kernels and ops might have local overrides.
    """
    if not HAS_TRITON:
        raise RuntimeError("Context-parallel model loading requires Triton on Linux with CUDA.")
    if (precision != "bf16-mixed") and (matmul_precision != "high"):
        warnings.warn(
            "Using precision other than 'bf16-mixed' with matmul_precision other than 'high' "
            "may lead to suboptimal performance."
        )
    manager = DistributedManager()
    device = manager.device
    # Every node's local rank 0 reaches this barrier after warming its cache.
    if manager.local_rank != 0:
        dist.barrier()
    serial = serial_load_model(
        checkpoint,
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        device=device,
        precision=precision,
        matmul_precision=matmul_precision,
    )
    assert isinstance(serial, _InferenceModel)
    if manager.local_rank == 0:
        dist.barrier()
    score_model = cast(BioMoaCDistributed, wrap_score_model_for_cp(serial.score_model, manager))
    score_model.eval().requires_grad_(False)
    attach_trace_hooks(score_model)
    assert dist.group.WORLD is not None
    return ModelWithSettings(
        score_model,
        device,
        get_autocast_dtype(precision),
        dist.group.WORLD,
        serial.num_steps_recycle,
        matmul_precision,
    )


def _prepare_sampling_inputs(
    system: System,
    model: ModelWithSettings,
) -> tuple[mdtraj.Topology, BatchAllAtomStructure]:
    structure = AllAtomStructure.from_system(system)
    structure = structure.replace(coords=torch.zeros_like(structure.coords))
    topology = topology_from_structure(structure, renamed_asym_ids=system.chain_ids)
    sizes = make_cp_pad_sizes(
        n_token=structure.num_tokens,
        n_atom=structure.num_nodes,
        n_msa=0,
        atoms_per_window=model.score_model.atoms_per_window_queries,
        mesh=DistributedManager().device_mesh_subgroups,
    )
    batch = batch_collate([structure], window_size=sizes.n_atom_final).to(model.device)
    batch.update(pad_feats_for_cp(batch, sizes))
    return topology, batch


def sample_with_embeddings(
    system: System,
    trunk_embeddings: CPTrunkEmbeddings,
    *,
    model: ModelWithSettings,
    num_samples: int = 8,
    batch_size: int | None = None,
    sampler: Sampler,
    seed: int | None = None,
    progress: bool = True,
) -> mdtraj.Trajectory:
    """CP analogue of ``bioemu2.sampling.sample_with_embeddings``.

    Call on every rank inside ``session()`` with the corresponding trunk shards
    and model. Returns the same structures in nanometers on every rank.
    All frames are retained; call ``bioemu2.check_physicality`` separately on rank
    zero to check their geometry. ``seed`` temporarily seeds Torch without
    changing the caller's RNG state.
    """
    _validate_model(model)
    topology, batch = _prepare_sampling_inputs(system, model)

    batch_size = num_samples if batch_size is None else batch_size
    warn_batch_size(int(batch.num_tokens.item()), num_samples, batch_size)

    if isinstance(sampler, FKCSampler):
        raise NotImplementedError(
            "CP FKC integration is not implemented yet: sampling randomness and resampling "
            "must be synchronized, and guidance weights need distributed-gradient validation."
        )

    if type(sampler) not in (EDMSampler, GuidedFKSampler):
        raise ValueError("CP supports only EDMSampler and GuidedFKSampler (not FKCSampler).")
    sampler = cast(EDMSampler | GuidedFKSampler, sampler)
    if isinstance(sampler, GuidedFKSampler) and cp_sync.is_rank0():
        # CP sampler potentials run only on rank 0, so other ranks need not prepare their caches.
        for potential in sampler.potentials:
            potential.prepare(system)
    parts: list[torch.Tensor] = []
    with _rng_seed(seed, model.device), torch.no_grad():
        for start in trange(
            0, num_samples, batch_size, disable=not progress or dist.get_rank() != 0
        ):
            current_batch_size = min(batch_size, num_samples - start)
            result = sampler(
                batch=batch,
                score_model=model,
                trunk_embeddings=trunk_embeddings,
                batch_size=current_batch_size,
            )
            coords = result.coords.reshape(current_batch_size, -1, 3)[:, : topology.n_atoms]
            coords = coords - coords.mean(dim=1, keepdim=True)
            parts.append(coords.cpu())
    trajectory = coords_angstrom_to_trajectory(torch.cat(parts), topology)
    return trajectory


def _embed_and_sample(
    system: System,
    *,
    model: ModelWithSettings,
    msa: MSA | None,
    mode: SamplingMode,
    contacts: list[Contact] | None,
    num_samples: int,
    batch_size: int | None,
    sampler: SamplerPreset,
    num_steps: int | None,
    potentials: Sequence[Potential] | None,
    seed: int | None,
    progress: bool,
    output_dir: str | Path | None,
    physicality_mode: PhysicalityFilterMode | str,
) -> mdtraj.Trajectory:
    physicality_mode = normalize_physicality_mode(physicality_mode)
    configured = sampler_from_preset(sampler, mode=mode, num_steps=num_steps, potentials=potentials)
    if isinstance(configured, FKCSampler):
        raise ValueError("CP does not support the edm_fkc preset.")
    out = Path(output_dir) if output_dir is not None else None
    trunk_embeddings = model.embed(system, msa=msa, mode=mode, contacts=contacts)
    trajectory = sample_with_embeddings(
        system,
        trunk_embeddings,
        model=model,
        num_samples=num_samples,
        batch_size=batch_size,
        sampler=configured,
        seed=seed,
        progress=progress,
    )
    if out is not None:
        output_error: list[str | None] = [None]
        cause: OSError | ValueError | RuntimeError | None = None
        if dist.get_rank() == 0:
            try:
                write_sampling_system(system, mode=mode, contacts=contacts, out=out)
                write_sampling_output(
                    trajectory,
                    system=system,
                    mode=mode,
                    contacts=contacts,
                    out=out,
                    physicality_filter=physicality_mode,
                )
            except (OSError, ValueError, RuntimeError) as error:
                cause = error
                output_error[0] = f"Could not save CP sampling outputs to {out}: {error}"
        dist.broadcast_object_list(output_error, src=0, group=DistributedManager().group["cpu"])
        if output_error[0] is not None:
            raise RuntimeError(output_error[0]) from cause
    return trajectory


def sample_equilibrium(
    system: System,
    *,
    model: ModelWithSettings,
    msa: MSA | None,
    contacts: list[Contact] | None = None,
    num_samples: int = 8,
    batch_size: int | None = None,
    sampler: SamplerPreset = "edm",
    num_steps: int | None = None,
    potentials: Sequence[Potential] | None = None,
    seed: int | None = None,
    progress: bool = True,
    output_dir: str | Path | None = None,
    physicality_mode: PhysicalityFilterMode | str = PhysicalityFilterMode.HIGH_QUALITY,
) -> mdtraj.Trajectory:
    """CP analogue of ``bioemu2.sampling.sample_equilibrium``.

    See ``docs/context-parallel-sampling.md`` for collective calls, rank-zero
    MSA acquisition and sampling batch sizes.

    ``sampler`` accepts only the presets ``"edm"``, ``"edm_gd"``, and ``"edm_fk"``.
    Use ``num_steps`` to override the preset's denoising step count and
    ``potentials`` to replace its steering potentials. For configured samplers
    or custom step scales, use ``model.embed`` and ``sample_with_embeddings``.

    With ``output_dir``, rank zero checks physicality and saves all frames,
    topology, system, and physicality.json. ``physicality_mode`` selects the
    check thresholds; failed frames are not removed. Without ``output_dir``,
    no checks are run and no sampling outputs are written. All ranks must pass the same
    output settings; saving failures are raised on every rank.
    """
    return _embed_and_sample(
        system,
        model=model,
        msa=msa,
        contacts=contacts,
        mode="equilibrium",
        num_samples=num_samples,
        batch_size=batch_size,
        sampler=sampler,
        num_steps=num_steps,
        potentials=potentials,
        seed=seed,
        progress=progress,
        output_dir=output_dir,
        physicality_mode=physicality_mode,
    )


def predict_structure(
    system: System,
    *,
    model: ModelWithSettings,
    msa: MSA | None,
    contacts: list[Contact] | None = None,
    num_samples: int = 8,
    batch_size: int | None = None,
    sampler: SamplerPreset = "edm",
    num_steps: int | None = None,
    potentials: Sequence[Potential] | None = None,
    seed: int | None = None,
    progress: bool = True,
    output_dir: str | Path | None = None,
    physicality_mode: PhysicalityFilterMode | str = PhysicalityFilterMode.HIGH_QUALITY,
) -> mdtraj.Trajectory:
    """CP analogue of ``bioemu2.sampling.predict_structure``.

    See ``docs/context-parallel-sampling.md`` for collective calls, rank-zero
    MSA acquisition and sampling batch sizes.

    ``sampler`` accepts only the presets ``"edm"``, ``"edm_gd"``, and ``"edm_fk"``.
    Use ``num_steps`` to override the preset's denoising step count and
    ``potentials`` to replace its steering potentials. For configured samplers
    or custom step scales, use ``model.embed`` and ``sample_with_embeddings``.

    With ``output_dir``, rank zero checks physicality and saves all frames,
    topology, system, and physicality.json. ``physicality_mode`` selects the
    check thresholds; failed frames are not removed. Without ``output_dir``,
    no checks are run and no sampling outputs are written. All ranks must pass the same
    output settings; saving failures are raised on every rank.
    """
    return _embed_and_sample(
        system,
        model=model,
        msa=msa,
        contacts=contacts,
        mode="structure",
        num_samples=num_samples,
        batch_size=batch_size,
        sampler=sampler,
        num_steps=num_steps,
        potentials=potentials,
        seed=seed,
        progress=progress,
        output_dir=output_dir,
        physicality_mode=physicality_mode,
    )


def propagate(
    system: System,
    *,
    x0: mdtraj.Trajectory,
    lag_time_ns: float,
    model: ModelWithSettings,
    msa: MSA,
    contacts: list[Contact] | None = None,
    batch_size: int | None = None,
    sampler: Sampler | None = None,
    seed: int | None = None,
    progress: bool = True,
) -> mdtraj.Trajectory:
    """CP counterpart of ``bioemu2.propagation.propagate``; not implemented yet.

    The distributed denoiser already accepts time-lagged conditioning. The
    operation-level validation, batching and endpoint alignment are not wired
    up. Raises ``NotImplementedError`` before loading or computing anything.
    """
    raise NotImplementedError(
        "CP propagation integration is not implemented yet; the denoiser supports "
        "conditioning, but the public operation still needs validation and batching."
    )


def propagate_with_embeddings(
    system: System,
    trunk_embeddings: CPTrunkEmbeddings,
    *,
    x0: mdtraj.Trajectory,
    lag_time_ns: float,
    model: ModelWithSettings,
    batch_size: int | None = None,
    sampler: Sampler,
    seed: int | None = None,
    progress: bool = True,
) -> mdtraj.Trajectory:
    """CP counterpart of ``bioemu2.propagation.propagate_with_embeddings``.

    Not implemented: raises ``NotImplementedError`` immediately on every rank.
    This is an integration gap,
    not a limitation of context parallelism.
    """
    raise NotImplementedError(
        "CP propagation with embeddings is not implemented yet; per-frame conditioning "
        "and partial batches must preserve the prepared multiplicity and frame mapping."
    )
