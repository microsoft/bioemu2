"""Numerical-correctness tests for `edm_steered_sampler` against an analytic
Gaussian-mixture ground truth.

The sampler is protein-geometry specialized — every step applies an SE(3) random
augmentation (`compute_random_augmentation` + an inline center-of-mass removal)
and an optional rigid alignment. We neutralize both purely via monkeypatch (no
production change), use `alignment_reverse_diff=False`, and pass trivial
precomputed features so `_pad_coords` is an identity and a coordinate-only
potential runs.

The one op the monkeypatch can't remove is the inline COM removal, so we use a
**2-atom antipodal** system: COM removal forces the pair to ``(+s, -s)``, leaving
the COM-invariant ``u = (x0 - x1)/sqrt(2) ∈ R³`` as the only DOF. The per-atom
noise projects onto ``u`` with variance exactly ``sigma**2`` (common-mode noise
cancels), so the ground-truth denoiser uses the sampler's own sigma. With the
rotation neutralized the three components of ``u`` are independent, so each axis
carries its own 1-D GMM. We set ``sigma_data=1.0`` so the Karras sigmas are raw.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal

from bioemu2.biomoa.models.edm_schedule import EDMSchedule
from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.denoising import cp_sync
from bioemu2.denoising.steering import edm_steered_sampler
from bioemu2.denoising.steering.feats import Tracker
from bioemu2.model import InferenceModel
from bioemu2.structure import BatchAllAtomStructure

_SQRT2 = math.sqrt(2.0)
_DUMMY_CONTEXT = SimpleNamespace(
    coords_cond=None,
    is_dynamics_conditioned=None,
    dt=None,
)
_DUMMY_TRUNK = cast(TrunkEmbeddings, object())
# Per-axis 1-D GMM: 3 well-separated modes with asymmetric weights.
_MODES = torch.tensor([-2.0, 0.0, 2.0])
_WEIGHTS = torch.tensor([0.5, 0.2, 0.3])
_STD = 0.5


def _u(coords: torch.Tensor) -> torch.Tensor:
    """COM-invariant coordinate u = (x0 - x1)/sqrt(2), shape [P, 3]."""
    return (coords[:, 0] - coords[:, 1]) / _SQRT2


class _Batch:
    """Minimal batch surface the geometry-neutralized sampler touches."""

    def __init__(self, coords: torch.Tensor, atom_mask: torch.Tensor):
        self.coords, self.atom_mask = coords, atom_mask

    def replace(self, **kw) -> _Batch:
        new = _Batch(self.coords, self.atom_mask)
        new.__dict__.update(self.__dict__, **kw)
        return new


class _GMM:
    """Exact Tweedie denoiser for the axis-factorized GMM on ``u``: the posterior
    mean ``u + sigma**2 * d/du log p_sigma(u)``, scored from `MixtureSameFamily`
    via autograd (re-enabled inside the sampler's `torch.no_grad()` block)."""

    edm_schedule = EDMSchedule(sigma_min=1e-2, sigma_max=40.0, sigma_data=1.0)

    def eval(self):
        return self

    def compute_static_context(self, batch, trunk_embeddings, multiplicity):
        assert trunk_embeddings is _DUMMY_TRUNK
        return _DUMMY_CONTEXT

    def denoise(self, context, trunk_embeddings, coords, sigmas):
        assert trunk_embeddings is _DUMMY_TRUNK
        u0 = _u(coords)
        sigma = float(sigmas.reshape(-1)[0])
        mix = MixtureSameFamily(
            Categorical(probs=_WEIGHTS), Normal(_MODES, math.sqrt(_STD**2 + sigma**2))
        )
        with torch.enable_grad():
            u = u0.detach().requires_grad_(True)
            (score,) = torch.autograd.grad(mix.log_prob(u).sum(), u)  # axis-independent
        x = (u0 + sigma**2 * score) / _SQRT2
        return SimpleNamespace(denoised_coords=torch.stack([x, -x], dim=1))  # antipodal


class _AxisWell:
    """Quadratic well on one axis of ``u`` (low energy near ``target``). FK
    reweighting upweights low-energy particles; autograd through ``__call__`` drives
    the GD inner loop (when ``guidance_weight > 0``). Duck-typed to the steering
    `Potential` interface the sampler calls."""

    def __init__(
        self, axis: int = 0, target: float = 2.0, beta: float = 1.0, guidance_weight: float = 0.0
    ):
        self.axis, self.target, self.beta, self.gw = axis, target, beta, guidance_weight

    def compute_parameters(self, t):
        return {"resampling_weight": 1.0, "guidance_weight": self.gw, "guidance_interval": 1}

    def __call__(self, coords, *, t):
        return self.beta * (_u(coords)[:, self.axis] - self.target) ** 2


def _identity_aug(m, *args, **kwargs):
    return torch.eye(3).expand(m, 3, 3).clone(), torch.zeros(m, 1, 3)


def _prepared_batch(score_model) -> _Batch:
    return _Batch(torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool))


def _run(
    multiplicity: int,
    num_steps: int = 200,
    seed: int = 0,
    denoise_micro_batch_size: int | None = None,
    tracker: Tracker | None = None,
    **steering_kwargs,
) -> torch.Tensor:
    """Seed, run the (geometry-neutralized) sampler, return final ``u`` [P, 3]."""
    torch.manual_seed(seed)

    score_model = _GMM()
    batch = _prepared_batch(score_model)
    out = edm_steered_sampler(
        batch=cast(BatchAllAtomStructure, batch),
        multiplicity=multiplicity,
        num_steps=num_steps,
        score_model=cast(InferenceModel, score_model),
        trunk_embeddings=_DUMMY_TRUNK,
        step_scale=1.0,
        noise_scale=1.0,
        alignment_reverse_diff=False,  # disable weighted_rigid_align
        denoise_micro_batch_size=denoise_micro_batch_size,
        tracker=tracker,
        **steering_kwargs,
    )
    return _u(out.coords)


def _draw_gmm(
    means: torch.Tensor, std: float, weights: torch.Tensor, n: int, seed: int
) -> torch.Tensor:
    """Draw ``n`` samples from the 1-D GMM (means, shared std, weights)."""
    torch.manual_seed(seed)
    mix = MixtureSameFamily(Categorical(probs=weights), Normal(means, std))
    return mix.sample((n,))


def _w1(a: torch.Tensor, b: torch.Tensor) -> float:
    """1-D Wasserstein-1 distance between two equal-size sample sets: the mean
    absolute gap between their sorted order statistics."""
    return float((torch.sort(a).values - torch.sort(b).values).abs().mean())


def _tilted_gmm(target: float, a: float) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Analytic tilt of the GMM by ``exp(-a (u - target)^2)``. Completing the
    square, each Gaussian stays Gaussian; the new (means, shared std, weights)
    are found in closed form and the weights renormalized (the tilt's Z)."""
    A = 1.0 / (2 * _STD**2) + a
    B = _MODES / _STD**2 + 2 * a * target
    log_scale = B**2 / (4 * A) - _MODES**2 / (2 * _STD**2) - a * target**2
    means = B / (2 * A)
    std = math.sqrt(1.0 / (2 * A))
    weights = torch.softmax(torch.log(_WEIGHTS) + log_scale, dim=0)
    return means, std, weights


# W1 thresholds (coordinate units, set by us, with margin above observed values).
# Recovery (unsteered): observed ~0.13-0.16; floor GMM-vs-GMM ~0.04.
# Tilt: FK-only W1 varies strongly with fk_lambda (weak tilt hard to capture,
# strong tilt collapses to a point so drawn ref spread matters). Observed:
#   lambda=0.1 -> ~0.24, lambda=1 -> ~0.14, lambda=5 -> ~0.34.


@pytest.fixture
def neutralized(monkeypatch):
    """Identity SE(3) augmentation for coordinate-only potentials."""
    monkeypatch.setattr(cp_sync, "random_augmentation", _identity_aug)


def test_unsteered_recovers_gmm(neutralized):
    """`edm_steered_sampler(steering=None)` reproduces the ground-truth GMM."""

    u = _run(multiplicity=4000)
    for axis in range(3):
        ref = _draw_gmm(_MODES, _STD, _WEIGHTS, u.shape[0], seed=100 + axis)
        w1 = _w1(u[:, axis], ref)
        assert w1 < _W1_RECOVER, f"axis {axis}: W1 to GMM = {w1:.3f} (> {_W1_RECOVER})"


@pytest.mark.parametrize("micro_batch_size", [1, 3], ids=["mb-1", "mb-3"])
def test_micro_batched_forward_matches_batched(neutralized, micro_batch_size):
    """`denoise_micro_batch_size` only splits the deterministic score-model
    forward over the particle batch dim — all RNG (augmentation, eps) and the
    FK/GD/resample ops stay on the full batch. So a micro-batched run must be
    exactly equal to the single-batched run. Exercises both steered branches
    (FK + GD) with multiplicity = 4, so micro-batch sizes 1 and 3 both
    straddle/don't-divide the batch."""
    steering_kwargs = dict(
        potentials=[_AxisWell(axis=0, target=2.0, beta=1.0, guidance_weight=0.1)],
        fk_lambda=1.0,
        resampling_interval=1,
        num_gd_steps=2,
    )
    batched = _run(multiplicity=4, num_steps=20, **steering_kwargs)
    micro_batched = _run(
        multiplicity=4,
        num_steps=20,
        denoise_micro_batch_size=micro_batch_size,
        **steering_kwargs,
    )
    torch.testing.assert_close(micro_batched, batched, rtol=0, atol=0)


def test_tracking_records_values_without_changing_sampling(neutralized, tmp_path):
    """Tracking preserves samples/RNG and records the coordinates underlying FK energies."""
    kwargs = dict(
        multiplicity=3,
        num_steps=4,
        potentials=[_AxisWell(axis=0, target=2.0, beta=1.0)],
        resampling_interval=2,
        num_gd_steps=0,
    )
    untracked = _run(**kwargs)
    untracked_rng = torch.get_rng_state()
    tracked = _run(tracker=Tracker(tmp_path / "metrics"), **kwargs)
    torch.testing.assert_close(tracked, untracked, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), untracked_rng, rtol=0, atol=0)

    metrics = tmp_path / "metrics"
    x0 = np.load(metrics / "x0_hat.npy")
    energy = np.load(metrics / "0__AxisWell.npy")
    energy_steps = [0, 2, 3]
    expected = (_u(torch.from_numpy(x0[energy_steps]).reshape(-1, 2, 3))[:, 0] - 2.0).square()
    np.testing.assert_allclose(
        energy[: len(energy_steps)],
        expected.reshape(len(energy_steps), 3).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(energy[len(energy_steps) :], 0)


_W1_RECOVER = 0.25
_W1_TILT = {0.1: 0.35, 1: 0.25, 5: 0.45, 10: 0.55}


@pytest.mark.parametrize("fk_lambda", [0.1, 1, 5, 10])
def test_steered_recovers_tilted_gmm(neutralized, fk_lambda):
    """A quadratic-well potential steers the sampler to the analytically tilted
    GMM ``p0 * exp(-lambda E) / Z``."""

    axis, target, beta = 0, 2.0, 1.0
    u_steered = _run(
        multiplicity=32000,
        potentials=[_AxisWell(axis, target, beta)],
        fk_lambda=fk_lambda,
        resampling_interval=1,
        num_gd_steps=0,
    )[:, axis]
    t_means, t_std, t_weights = _tilted_gmm(target, fk_lambda * beta)
    ref = _draw_gmm(t_means, t_std, t_weights, u_steered.shape[0], seed=200)
    w1 = _w1(u_steered, ref)

    threshold = _W1_TILT[fk_lambda]
    assert w1 < threshold, f"lambda={fk_lambda}: W1 to tilted GMM = {w1:.3f} (> {threshold})"


def test_edm_steered_sampler_validates_kwargs() -> None:
    """The sampler validates its resampling kwargs up front, before touching
    ``batch``/``score_model``."""
    batch = _prepared_batch(_GMM())

    def _call(**kwargs):
        edm_steered_sampler(
            batch=cast(BatchAllAtomStructure, batch),
            num_steps=2,
            score_model=cast(InferenceModel, None),
            trunk_embeddings=_DUMMY_TRUNK,
            step_scale=1.0,
            noise_scale=1.0,
            **kwargs,
        )

    with pytest.raises(ValueError, match="resampling_interval=0"):
        _call(resampling_interval=0)
    with pytest.raises(ValueError, match="resampling_group_size=1"):
        _call(resampling_group_size=1)
