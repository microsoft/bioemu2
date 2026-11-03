"""Public sampling algorithms."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.context_parallel.models.score_model import CPTrunkEmbeddings
from bioemu2.denoising.edm import edm_sampler
from bioemu2.denoising.steering.edm_fkc import Reheat, WeightUpdate, edm_fkc_sampler
from bioemu2.denoising.steering.edm_steered import edm_steered_sampler
from bioemu2.denoising.steering.feats import Tracker
from bioemu2.denoising.steering.potentials import Potential, get_default_potentials
from bioemu2.model import InferenceModel
from bioemu2.structure import BatchAllAtomStructure


def _validate_edm_scales(step_scale: float, noise_scale: float) -> None:
    if not math.isfinite(step_scale) or step_scale <= 0:
        raise ValueError("step_scale must be finite and positive.")
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale must be finite and nonnegative.")


@dataclass(kw_only=True)
class Sampler(ABC):
    """Sampling algorithm driving ``InferenceModel.denoise``.

    Steering samplers add a ``potentials`` field; ``EDMSampler`` has none.
    Their calls accept an optional caller-owned ``tracker`` for diagnostics.
    Steered samplers register their own metric keys and shapes; the caller
    configures output paths and batch-specific filenames. EDM ignores the tracker.
    """

    num_steps: int

    @abstractmethod
    def __call__(
        self,
        *,
        batch: BatchAllAtomStructure,
        score_model: InferenceModel[TrunkEmbeddings | CPTrunkEmbeddings],
        trunk_embeddings: TrunkEmbeddings | CPTrunkEmbeddings,
        batch_size: int,
        tracker: Tracker | None = None,
    ) -> BatchAllAtomStructure: ...


@dataclass(kw_only=True)
class EDMSampler(Sampler):
    """Unsteered EDM sampler backed by ``edm_sampler``."""

    num_steps: int = 200
    step_scale: float = 1.0
    noise_scale: float = 1.0

    def __post_init__(self) -> None:
        _validate_edm_scales(self.step_scale, self.noise_scale)

    def __call__(
        self,
        *,
        batch: BatchAllAtomStructure,
        score_model: InferenceModel[TrunkEmbeddings | CPTrunkEmbeddings],
        trunk_embeddings: TrunkEmbeddings | CPTrunkEmbeddings,
        batch_size: int,
        tracker: Tracker | None = None,
    ) -> BatchAllAtomStructure:
        return edm_sampler(
            batch=batch,
            score_model=score_model,
            trunk_embeddings=trunk_embeddings,
            multiplicity=batch_size,
            **vars(self),
        )


@dataclass(kw_only=True)
class GuidedFKSampler(Sampler):
    """FK particle resampling + gradient guidance, backed by ``edm_steered_sampler``.

    See ``docs/steering.md`` for more information.

    The resampling population is the current sampling batch (``batch_size``),
    optionally split into independent groups of ``resampling_group_size``.

    Args:
        potentials: Steering potentials. Defaults to the physicality potential
            set. An explicit sequence replaces that set; an empty sequence disables
            potentials.
        resampling_group_size: Particles per independent resampling group;
            ``None`` resamples the whole batch as one group.
        resampling_interval: Steps between systematic resamples. ``None``
            disables both FK weight computation and resampling (guidance only).
        num_gd_steps: Gradient-descent guidance steps on ``x0_hat`` per denoising step.
        denoise_micro_batch_size: Particles per micro-batch when applying the score
            model and evaluating potential energies and gradients; ``None`` applies
            the whole batch at once. Trades wall time for peak memory at large
            ``batch_size``. Each micro-batch calls a potential separately, so a
            potential that draws fresh data per call scores each micro-batch on a
            different draw; that is harmless for a gradient, and harmless for FK
            weights provided no micro-batch boundary falls inside a resampling group
            (weights are normalized and resampled within a group only). Keep it a
            multiple of ``resampling_group_size``; with ``resampling_group_size=None``
            the whole batch is one group, so any smaller micro-batch splits it.
    """

    num_steps: int = 200
    step_scale: float = 1.0
    noise_scale: float = 1.0
    potentials: Sequence[Potential] = field(default_factory=get_default_potentials)
    resampling_interval: int | None = 3
    resampling_group_size: int | None = None
    fk_lambda: float = 4.0
    num_gd_steps: int = 20
    denoise_micro_batch_size: int | None = None

    def __post_init__(self) -> None:
        _validate_edm_scales(self.step_scale, self.noise_scale)

    def __call__(
        self,
        *,
        batch: BatchAllAtomStructure,
        score_model: InferenceModel[TrunkEmbeddings | CPTrunkEmbeddings],
        trunk_embeddings: TrunkEmbeddings | CPTrunkEmbeddings,
        batch_size: int,
        tracker: Tracker | None = None,
    ) -> BatchAllAtomStructure:
        return edm_steered_sampler(
            batch=batch,
            score_model=score_model,
            trunk_embeddings=trunk_embeddings,
            multiplicity=batch_size,
            tracker=tracker,
            **vars(self),
        )


@dataclass(kw_only=True)
class FKCSampler(Sampler):
    """Feynman-Kac corrector sampler backed by ``edm_fkc_sampler``.

    See ``docs/steering.md`` for more information.

    The resampling population is the current sampling batch (``batch_size``),
    optionally split into independent groups of ``resampling_group_size``.

    Args:
        potentials: Steering potentials. Defaults to the physicality potential
            set. An explicit sequence replaces that set; an empty sequence disables
            potentials.
        resampling_group_size: Particles per independent resampling group;
            ``None`` resamples the whole batch as one group.
        resampling_interval: Steps between systematic resamples.
        ess_threshold: Effective-sample-size fraction that triggers a resample.
    """

    num_steps: int = 200
    potentials: Sequence[Potential] = field(default_factory=get_default_potentials)
    resampling_interval: int | None = None
    resampling_group_size: int | None = None
    ess_threshold: float | None = 0.9
    weight_update: WeightUpdate = "guidance"
    beta_power: float = 4.0
    reheat: Reheat = "sigma"
    gamma_0: float = 0.8
    gamma_min: float = 0.0

    def __call__(
        self,
        *,
        batch: BatchAllAtomStructure,
        score_model: InferenceModel[TrunkEmbeddings | CPTrunkEmbeddings],
        trunk_embeddings: TrunkEmbeddings | CPTrunkEmbeddings,
        batch_size: int,
        tracker: Tracker | None = None,
    ) -> BatchAllAtomStructure:
        if isinstance(trunk_embeddings, CPTrunkEmbeddings):
            raise NotImplementedError("CP FKC integration is not implemented yet.")
        return edm_fkc_sampler(
            batch=batch,
            score_model=score_model,
            trunk_embeddings=trunk_embeddings,
            multiplicity=batch_size,
            tracker=tracker,
            **vars(self),
        )


__all__ = ["EDMSampler", "FKCSampler", "GuidedFKSampler", "Sampler"]
