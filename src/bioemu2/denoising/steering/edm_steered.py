"""EDM-Heun sampling with optional FK resampling and gradient guidance."""

import logging
import math
from collections.abc import Sequence
from dataclasses import replace

import torch
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.biomoa.modules.diffusion import StaticContext
from bioemu2.context_parallel.models.score_model import CPTrunkEmbeddings
from bioemu2.context_parallel.modules.diffusion import CPStaticContext
from bioemu2.denoising import cp_sync
from bioemu2.denoising.augmentation import augment_coordinates
from bioemu2.denoising.edm import (
    edm_sample_schedule,
)
from bioemu2.denoising.rigid_align import weighted_rigid_align
from bioemu2.denoising.steering.conditioning import resample_conditioning
from bioemu2.denoising.steering.feats import (
    Tracker,
    fk_log_increment,
    girsanov_ll_difference,
    grouped_softmax,
    systematic_resample,
)
from bioemu2.denoising.steering.potentials import Potential
from bioemu2.model import InferenceModel
from bioemu2.structure import BatchAllAtomStructure
from bioemu2.structure.inference_inputs import broadcast_multiplicity

logger = logging.getLogger(__name__)


def _micro_batch_apply(fn, coords: torch.Tensor, micro_batch_size: int | None) -> torch.Tensor:
    """Apply ``fn`` over dim-0 chunks and concatenate its outputs."""
    if micro_batch_size is None or micro_batch_size >= coords.shape[0]:
        return fn(coords)
    return torch.cat([fn(c) for c in coords.split(micro_batch_size)], dim=0)


def _validate_fk_kwargs(
    *,
    potentials: Sequence[Potential],
    resampling_interval: int | None,
    resampling_group_size: int | None,
) -> None:
    if not potentials:
        logger.warning("edm_steered_sampler called without potentials; use plain EDM instead")
    if resampling_interval is not None and resampling_interval < 1:
        raise ValueError(f"edm_steered_sampler got resampling_interval={resampling_interval} < 1")
    if resampling_group_size is not None and resampling_group_size < 2:
        raise ValueError(
            f"edm_steered_sampler got resampling_group_size={resampling_group_size}; use at "
            "least 2, or resampling_interval=None to disable resampling"
        )


def _slice_conditioning(
    context: StaticContext | CPStaticContext, start: int, stop: int
) -> StaticContext | CPStaticContext:
    """Keep propagation starting coordinates and lags aligned with each denoising micro-batch."""
    if context.coords_cond is None or context.coords_cond.shape[0] == 1:
        return context
    assert context.is_dynamics_conditioned is not None and context.dt is not None
    return replace(
        context,
        coords_cond=context.coords_cond[start:stop],
        is_dynamics_conditioned=(
            context.is_dynamics_conditioned[:, start:stop]
            if context.is_dynamics_conditioned.shape[1] > 1
            else context.is_dynamics_conditioned
        ),
        dt=context.dt[:, start:stop] if context.dt.shape[1] > 1 else context.dt,
    )


def edm_steered_sampler(
    *,
    batch: BatchAllAtomStructure,
    num_steps: int,
    score_model: InferenceModel[TrunkEmbeddings | CPTrunkEmbeddings],
    trunk_embeddings: TrunkEmbeddings | CPTrunkEmbeddings,
    step_scale: float,
    noise_scale: float,
    rho: int = 7,
    gamma_0: float = 0.8,
    gamma_min: float = 1.0,
    multiplicity: int = 1,
    alignment_reverse_diff: bool = True,
    potentials: Sequence[Potential] = (),
    resampling_interval: int | None = 3,
    resampling_group_size: int | None = None,
    fk_lambda: float = 4.0,
    num_gd_steps: int = 20,
    tracker: Tracker | None = None,
    denoise_micro_batch_size: int | None = None,
) -> BatchAllAtomStructure:
    """Sample with EDM Heun updates and optional FK resampling and gradient guidance.

    `batch` holds one structure; `multiplicity` conformations are sampled for it
    and form the FK resampling population: every `resampling_interval` steps
    they are resampled by their FK weights, within independent groups of
    `resampling_group_size` (all of them when `None`). `resampling_interval=None`
    disables FK resampling (gradient guidance only).

    During serial sampling, ``denoise_micro_batch_size`` splits score-model
    denoising and potential energy/gradient evaluation over the particle batch.
    During CP sampling, it splits only rank-0 potential energy/gradient
    evaluation; score-model denoising uses the full prepared particle batch.

    When `potentials` is empty or neither FK nor GD is enabled, the loop reduces
    inline to the unsteered EDM Heun path and is byte-identical to
    `edm_sampler` (same RNG draws, same returned tensor). The reduction is
    enforced by the parity tests in
    ``tests/denoising/steering/test_samplers_parity.py``.

    CP static context contains token masks, atom-to-token indices, and center
    indices expanded to the sampling multiplicity and sharded as
    ``DTensor`` objects. Passing a smaller particle batch to ``denoise`` would
    not match those tensors. Therefore, CP score-model denoising does not
    support micro-batching.

    During CP sampling, every rank holds the full coordinate batch; only the
    pair representations are sharded across the two-dimensional device mesh.
    Potential evaluation, gradient guidance, and metric writes run on rank 0.
    Guided coordinates and resampling ancestry are broadcast before the next
    denoising step. These ``cp_sync`` calls are no-ops during single-device
    sampling.

    Pass a ``tracker`` to save ``x_t``, ``x0_hat``, and per-potential energies
    before resampling, or ``None`` to disable tracking. Coordinates are in
    angstroms and recorded every step, with ``x0_hat`` saved after guidance.
    Energies are evaluated before guidance and recorded only on resampling
    steps, so their row numbers need not match coordinate rows; unused rows
    at the end of each energy file remain zero.
    """
    potentials = tuple(potentials)
    _validate_fk_kwargs(
        potentials=potentials,
        resampling_interval=resampling_interval,
        resampling_group_size=resampling_group_size,
    )
    context = score_model.compute_static_context(batch, trunk_embeddings, multiplicity)
    is_ito = context.is_dynamics_conditioned is not None and bool(
        context.is_dynamics_conditioned.any()
    )
    gd_enabled = num_gd_steps > 0
    distributed_sampling = cp_sync.is_distributed_sampling()
    is_rank0 = cp_sync.is_rank0()
    device = batch.coords.device
    schedule = score_model.edm_schedule

    sigmas = edm_sample_schedule(
        num_steps,
        sigma_min=schedule.sigma_min,
        sigma_max=schedule.sigma_max,
        sigma_data=schedule.sigma_data,
        rho=rho,
        device=device,
    )
    gammas = torch.where(sigmas > gamma_min, gamma_0, 0.0)
    pairs = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

    # Broadcast coords by multiplicity. Cache atom_mask similarly.
    if batch.coords.shape[0] != 1:
        # Resampling across several conditions (structures or ITO starting
        # frames) is undefined; the per-condition population is the
        # multiplicity axis.
        raise NotImplementedError("Steered samplers take one structure per call.")
    batch = batch.replace(coords=broadcast_multiplicity(batch.coords, multiplicity))
    atom_mask = batch.atom_mask.repeat_interleave(multiplicity, 0)
    shape = (*atom_mask.shape, 3)
    if distributed_sampling:
        assert isinstance(trunk_embeddings.pair, DTensor), (
            "trunk_embeddings.pair must be a DTensor (sharded) for context-parallel sampling."
        )

    # CUDA RNG is not bit-identical across devices, so rank 0 generates initial
    # noise and broadcasts it during CP sampling.
    atom_coords = sigmas[0] * cp_sync.randn(
        batch.coords.shape, device=device, dtype=batch.coords.dtype
    )
    scaled_guidance_update = torch.zeros_like(atom_coords, dtype=torch.float32)
    energy_traj: list[torch.Tensor] = []

    guidance_energy_keys: dict[int, str] = {}
    if is_rank0 and tracker is not None:
        for index, p in enumerate(potentials):
            tracker.register(
                f"{index}_{type(p).__name__}",
                (len(pairs), shape[0]),
            )
            # Energy at each GUIDANCE step, captured from the evaluation the gradient is already taken on,
            # so it costs nothing extra. The FK metric above is written only when resampling fires, which
            # never happens for guidance-only runs; this one is written whenever guidance runs.
            guidance_energy_keys[id(p)] = f"guidance_energy_{index}_{type(p).__name__}"
            tracker.register(
                guidance_energy_keys[id(p)],
                (len(pairs) * max(1, num_gd_steps), shape[0]),
            )
        tracker.register("x_t", (len(pairs), *shape))
        tracker.register("x0_hat", (len(pairs), *shape))

    # Synchronize CP ranks before entering the sampling loop.
    cp_sync.barrier()

    last_step = len(pairs) - 1
    for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(pairs):
        # Rank 0 generates the random augmentation and broadcasts it during CP sampling.
        R, tr = cp_sync.random_augmentation(multiplicity, device=device, dtype=atom_coords.dtype)
        atom_coords = augment_coordinates(atom_coords, R, tr, atom_mask)
        if is_ito:
            assert context.coords_cond is not None
            context = replace(
                context, coords_cond=augment_coordinates(context.coords_cond, R, tr, atom_mask)
            )
        # Keep the carried guidance update in the same frame as the noise we'll
        # use for the importance correction.
        if is_rank0:
            scaled_guidance_update = torch.einsum(
                "bmd,bds->bms", scaled_guidance_update.float(), R.float()
            )

        sigma_tm_v, sigma_t_v, gamma_v = sigma_tm.item(), sigma_t.item(), gamma.item()
        # Parameter schedules use normalized t in [0, 1], with t=1 at peak noise.
        t_norm = 1.0 - (step_idx + 1) / num_steps
        t_hat = sigma_tm_v * (1 + gamma_v)
        noise_var = (noise_scale**2) * max(t_hat**2 - sigma_tm_v**2, 0.0)
        noise_std = math.sqrt(noise_var) if noise_var > 0 else 0.0
        # Rank 0 generates per-step churn noise and broadcasts it during CP sampling.
        eps = noise_std * cp_sync.randn(shape, device=device, dtype=atom_coords.dtype)
        atom_coords_noisy = atom_coords + eps

        with torch.no_grad():
            if (
                distributed_sampling
                or denoise_micro_batch_size is None
                or denoise_micro_batch_size >= shape[0]
            ):
                x0_hat = score_model.denoise(
                    context=context,
                    trunk_embeddings=trunk_embeddings,
                    coords=atom_coords_noisy,
                    sigmas=torch.full((shape[0],), t_hat, device=device),
                ).denoised_coords
            else:

                def _denoise(sub: torch.Tensor, start: int) -> torch.Tensor:
                    return score_model.denoise(
                        context=_slice_conditioning(context, start, start + sub.shape[0]),
                        trunk_embeddings=trunk_embeddings,
                        coords=sub,
                        sigmas=torch.full((sub.shape[0],), t_hat, device=device),
                    ).denoised_coords

                x0_hat = torch.cat(
                    [
                        _denoise(sub, start)
                        for start, sub in zip(
                            range(0, shape[0], denoise_micro_batch_size),
                            atom_coords_noisy.split(denoise_micro_batch_size),
                            strict=True,
                        )
                    ]
                )

        # FK weights use the pre-guidance x0_hat and the carried guidance update.
        resample_now = resampling_interval is not None and (
            step_idx % resampling_interval == 0 or step_idx == last_step
        )
        w = None
        metrics: dict[str, torch.Tensor] = {}
        if is_rank0 and resample_now and potentials:
            energy = torch.zeros(x0_hat.shape[0], device=device, dtype=torch.float32)
            for index, p in enumerate(potentials):
                params = p.compute_parameters(t_norm) or {}
                # Pairwise potentials build O(n_pairs) stacks over the particle
                # dimension, so rank 0 can micro-batch their energy evaluation.
                e_i = _micro_batch_apply(
                    lambda c, p=p: p(c, t=t_norm).float(),
                    x0_hat.float(),
                    denoise_micro_batch_size,
                )
                energy = energy + params.get("resampling_weight", 1.0) * e_i
                # Per-potential (unweighted) energy for opt-in tracking.
                if tracker is not None:
                    metrics[f"{index}_{type(p).__name__}"] = e_i
            log_G = fk_log_increment(energy, energy_traj[-1] if energy_traj else None)
            energy_traj.append(energy.detach())
            if gd_enabled and noise_var > 0:
                ll_diff = girsanov_ll_difference(
                    eps.float(),
                    scaled_guidance_update.float(),
                    eps.new_tensor(noise_var),
                )
            else:
                ll_diff = torch.zeros_like(energy)
            unnormalized = ll_diff + fk_lambda * log_G
            # A non-finite energy would poison its whole group's softmax; clamp it
            # to a weight that survives normalization but loses every resample.
            unnormalized = torch.where(
                torch.isfinite(unnormalized), unnormalized, unnormalized.new_full((), -1e4)
            )
            w = grouped_softmax(unnormalized, resampling_group_size)

        # Rank 0 evaluates potential gradients and retains the update for the FK
        # likelihood correction. x0_hat is detached, so no gradient reaches the
        # score model.
        if gd_enabled and step_idx < last_step and potentials:
            x0_hat = x0_hat.float()
            if is_rank0:
                coords_unpad = x0_hat.detach()
                update = torch.zeros_like(coords_unpad)
                for guidance_step in range(num_gd_steps):
                    # Potentials active this inner step (guidance-weight + interval gate).
                    active = []
                    for p in potentials:
                        params = p.compute_parameters(t_norm) or {}
                        gw = params.get("guidance_weight", 0.0)
                        if gw != 0.0 and guidance_step % params.get("guidance_interval", 1) == 0:
                            active.append((p, gw))
                    # No potential contributed this inner step → nothing to descend.
                    if not active:
                        continue

                    captured: dict[int, list[torch.Tensor]] = {}

                    def _grad(coords_chunk, active=active, captured=captured):
                        with torch.enable_grad():
                            c = coords_chunk.detach().requires_grad_(True)
                            energy = c.new_zeros(())
                            for p, gw in active:
                                e_p = p(c, t=t_norm)
                                if guidance_energy_keys:
                                    # Detached, so recording cannot perturb the run.
                                    captured.setdefault(id(p), []).append(e_p.detach().float())
                                energy = energy + gw * e_p.sum()
                            # Every active potential can still be inactive on *this*
                            # structure (empty index → detached zeros), leaving energy
                            # graph-free; autograd.grad would raise, so return zeros.
                            if not energy.requires_grad:
                                return torch.zeros_like(c)
                            (grad_c,) = torch.autograd.grad(energy, c)
                        return grad_c

                    grad = _micro_batch_apply(
                        _grad, coords_unpad + update, denoise_micro_batch_size
                    )
                    if tracker is not None and captured:
                        # `_micro_batch_apply` walks dim-0 chunks in order, so concatenating restores
                        # the sample order.
                        tracker.log(
                            {
                                guidance_energy_keys[key]: torch.cat(values)
                                for key, values in captured.items()
                            }
                        )
                    update = (update - grad).detach()
                x0_hat = x0_hat + update
                scaled_guidance_update = update * (
                    -step_scale * (sigma_t_v - t_hat) / max(t_hat, 1e-12)
                )
            x0_hat = cp_sync.broadcast_from_rank0(x0_hat.contiguous())

        # Track pre-resample coordinates.
        if is_rank0 and tracker is not None:
            metrics.update({"x_t": atom_coords_noisy, "x0_hat": x0_hat})
            tracker.log(metrics)

        # Resample after FK weighting and gradient guidance, then broadcast ancestry.
        if resample_now and potentials:
            if is_rank0:
                assert w is not None
                idx = systematic_resample(w, resampling_group_size)
            else:
                idx = torch.empty(x0_hat.shape[0], device=device, dtype=torch.long)
            cp_sync.broadcast_from_rank0(idx)
            atom_coords = atom_coords[idx]
            atom_coords_noisy = atom_coords_noisy[idx]
            x0_hat = x0_hat[idx]
            atom_mask = atom_mask[idx]
            if is_ito:
                context = resample_conditioning(context, idx)
            if is_rank0:
                scaled_guidance_update = scaled_guidance_update[idx]
            if energy_traj:
                energy_traj[-1] = energy_traj[-1][idx]

        # Heun step.
        if alignment_reverse_diff:
            with torch.autocast("cuda", enabled=False):
                atom_coords_noisy, _, _ = weighted_rigid_align(
                    atom_coords_noisy.float(),
                    x0_hat.float(),
                    atom_mask.float(),
                    atom_mask.float(),
                )
            atom_coords_noisy = atom_coords_noisy.to(x0_hat)
        denoised_over_sigma = (atom_coords_noisy - x0_hat) / max(t_hat, 1e-12)
        atom_coords = atom_coords_noisy + step_scale * (sigma_t_v - t_hat) * denoised_over_sigma

    return batch.replace(coords=atom_coords)
