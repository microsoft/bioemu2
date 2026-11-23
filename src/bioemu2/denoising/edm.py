# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""EDM noise schedule and sampler for diffusion-based structure sampling."""

import math
from dataclasses import replace

import torch
import torch.nn.functional as F

from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.context_parallel.models.score_model import (
    CPTrunkEmbeddings as DistributedTrunkEmbeddings,
)
from bioemu2.denoising import cp_sync
from bioemu2.denoising.augmentation import augment_coordinates
from bioemu2.denoising.rigid_align import weighted_rigid_align
from bioemu2.model import InferenceModel
from bioemu2.structure import BatchAllAtomStructure
from bioemu2.structure.inference_inputs import broadcast_multiplicity


def edm_sample_schedule(
    num_steps: int,
    device: torch.device,
    sigma_min: float = 0.0004,
    sigma_max: float = 160.0,
    sigma_data: float = 16.0,  # angstrom
    rho: int = 7,
) -> torch.Tensor:
    """Sigma schedule for EDM sampling; see Karras et al.

    Args:
        num_steps: Number of sampling steps.
        sigma_min: Minimum value for sigma. dimensionless
        sigma_max: Maximal value for sigma. dimensionless
        sigma_data: Standard deviation scale of data, unit in angstroms
        rho: Control shortening of steps close to sigma_min vs lengthening close to sigma_max.
        device: Computational device.

    Returns:
        Sigma values for the specified sampling schedule, in angstroms.

    References:
        Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of
          diffusion-based generative models. Advances in neural information processing systems, 35,
          26565-26577.
    """
    if num_steps < 2:
        raise ValueError(f"num_steps must be at least 2, got {num_steps}.")
    inv_rho = 1 / rho

    steps = torch.arange(num_steps, device=device, dtype=torch.float32)
    sigmas = (
        sigma_max**inv_rho + steps / (num_steps - 1) * (sigma_min**inv_rho - sigma_max**inv_rho)
    ) ** rho

    sigmas = sigmas * sigma_data

    sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
    return sigmas


def edm_sampler(
    *,
    batch: BatchAllAtomStructure,
    num_steps: int,
    score_model: InferenceModel[TrunkEmbeddings | DistributedTrunkEmbeddings],
    trunk_embeddings: TrunkEmbeddings | DistributedTrunkEmbeddings,
    step_scale: float,
    noise_scale: float,
    rho: int = 7,
    gamma_0: float = 0.8,
    gamma_min: float = 1.0,
    multiplicity: int = 1,
    alignment_reverse_diff: bool = True,
) -> BatchAllAtomStructure:
    """Sample from the score model using the EDM schedule of Karras et al.

    Args:
        batch: Current batch.
        num_steps: Number of sampling steps.
        score_model: Score model. Should predict denoised coordinates.
        trunk_embeddings: Reusable sequence-trunk outputs. Static context is
            prepared once for this sampler invocation.
        step_scale: heuristic sharpening factor. Recommended values: 1.0 to sample varied structures, 1.5 for accurate single structure prediction.
        noise_scale: Scale sampling noise.
        rho: Control shortening of steps close to sigma_min vs lengthening close to sigma_max.
        gamma_0: Base value for determining gamma in Karras et al. Algorithm 2.
        gamma_min: Minimum value for determining gamma in Karras et al. Algorithm 2.
        multiplicity: Number of times structure in batch should be repeated for structure module
          operations.
        alignment_reverse_diff: Align noisy coordinates with denoised coordinates before each
          update.

    Returns:
        Sampled structures.
    """
    context = score_model.compute_static_context(batch, trunk_embeddings, multiplicity)
    is_ito = context.is_dynamics_conditioned is not None and bool(
        context.is_dynamics_conditioned.any()
    )
    device = batch.coords.device
    schedule = score_model.edm_schedule
    atom_mask = batch.atom_mask.repeat_interleave(multiplicity, 0)

    shape = (*atom_mask.shape, 3)

    # Sequence of noise levels to step through during denoising.
    sigmas = edm_sample_schedule(
        num_steps,
        sigma_min=schedule.sigma_min,
        sigma_max=schedule.sigma_max,
        sigma_data=schedule.sigma_data,
        rho=rho,
        device=device,
    )
    gammas = torch.where(sigmas > gamma_min, gamma_0, 0.0)
    sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

    if batch.coords.shape[0] != 1:
        # Per-sample noise and augmentation are sized by `multiplicity`, not B * multiplicity.
        raise NotImplementedError("edm_sampler takes one structure per call.")
    batch = batch.replace(coords=broadcast_multiplicity(batch.coords, multiplicity))

    # Sample random initial atom positions.  Synchronize these over the CP group.
    init_sigma = sigmas[0]
    atom_coords = init_sigma * cp_sync.randn(
        batch.coords.shape, device=device, dtype=batch.coords.dtype
    )

    cp_sync.barrier()

    # gradually denoise
    for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
        # Randomly rotate and translate the structure at each denoising step.
        random_R, random_tr = cp_sync.random_augmentation(
            multiplicity, device=device, dtype=atom_coords.dtype
        )
        atom_coords = augment_coordinates(atom_coords, random_R, random_tr, atom_mask)

        if is_ito:
            assert context.coords_cond is not None
            context = replace(
                context,
                coords_cond=augment_coordinates(
                    context.coords_cond, random_R, random_tr, atom_mask
                ),
            )

        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

        t_hat = sigma_tm * (1 + gamma)
        noise_std = noise_scale * math.sqrt(t_hat**2 - sigma_tm**2)
        # Churn: add noise to each atom position. Keep atom positions synced over the CP group.
        eps = noise_std * cp_sync.randn(shape, device=device)
        atom_coords_noisy = atom_coords + eps

        with torch.no_grad():
            sigmas_step = torch.full((atom_coords_noisy.shape[0],), t_hat, device=device)
            atom_coords_denoised = score_model.denoise(
                context=context,
                trunk_embeddings=trunk_embeddings,
                coords=atom_coords_noisy,
                sigmas=sigmas_step,
            ).denoised_coords

        if alignment_reverse_diff:
            # Align noisy coordinates to denoised coordinates before the update.
            with torch.autocast("cuda", enabled=False):
                atom_coords_noisy, _, _ = weighted_rigid_align(
                    atom_coords_noisy.float(),
                    atom_coords_denoised.float(),
                    atom_mask.float(),
                    atom_mask.float(),
                )

            atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
        atom_coords_next = atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma

        atom_coords = atom_coords_next

    return batch.replace(coords=atom_coords)
