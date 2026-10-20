"""Analytic intermediate marginals for the EDM-FKC parameterizations.

`edm_fkc_sampler` defines its tilt on the *denoised* sample, so the analytic
target is ``p_sigma(x) exp(-beta U(D_sigma(x)))`` rather than a tilt of the
noisy marginal.

Each parameterization is checked twice, against the *same* analytic reference,
differing only in which degree of freedom carries the tilt:

- **Invariant**: the denoiser and potential read the COM-invariant coordinate
  ``u = (x0 - x1)/sqrt(2)``, so they are blind to the frame changes the sampler
  makes each step. The antipodal 2-atom system is what makes this exact — COM
  removal forces the pair to ``(+s, -s)``, leaving ``u`` as the only degree of
  freedom, and per-atom noise projects onto it with variance exactly ``sigma^2``.
- **Absolute**: the denoiser and potential read raw atom coordinates, so the
  steered quantity is *not* frame-invariant. COM removal is monkeypatched out
  alongside the augmentation, leaving every coordinate to evolve independently
  under the same 1-D GMM diffusion.

Together they separate two failure modes that the invariant test alone cannot:
whether the Feynman-Kac bookkeeping is right, and whether it survives a
potential whose value the sampler's own geometry handling can move.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal

from bioemu2.biomoa.models.edm_schedule import EDMSchedule
from bioemu2.data_interface import Ligand, Protein, System
from bioemu2.denoising.steering import edm_fkc as _edm_fkc_module
from bioemu2.denoising.steering import edm_fkc_sampler
from bioemu2.denoising.steering.edm_fkc import Reheat, WeightUpdate
from bioemu2.denoising.steering.feats import Tracker
from bioemu2.denoising.steering.potentials import (
    BondAnglePotential,
    BondLengthPotential,
    PolymerLocalClashPotential,
    Potential,
)
from bioemu2.model import InferenceModel
from bioemu2.structure import AllAtomStructure, BatchAllAtomStructure
from bioemu2.structure.batch import batch_collate

from .test_edm_steered_gmm import (
    _DUMMY_CONTEXT,
    _DUMMY_TRUNK,
    _MODES,
    _SQRT2,
    _STD,
    _WEIGHTS,
    _identity_aug,
    _prepared_batch,
    _u,
)
from .test_samplers_parity import _DUMMY_TRUNK as _GAUSSIAN_TRUNK
from .test_samplers_parity import _GaussianScoreModel

_SIGMA_MIN = 0.0004
_SIGMA_MAX = 160.0
_RHO = 7
_NUM_PARTICLES = 20_000
_NUM_STEPS = 1_000
# Nine intermediate marginals plus the terminal transition, for any schedule
# length. That last one is the payoff check: at sigma=0 the denoiser is the
# identity and beta(0) = 1, so the analytic target reduces to the true tilted
# distribution p_0(x) exp(-U(x)).
_NUM_MARGINALS = 10
_MARGINAL_STRIDE = _NUM_STEPS // _NUM_MARGINALS
_MARGINAL_STEPS = list(range(_MARGINAL_STRIDE - 1, _NUM_STEPS - 1, _MARGINAL_STRIDE)) + [
    _NUM_STEPS - 1
]
_MARGINAL_METRICS = {"x_t", "weights", "sigma", "beta"}


class _MarginalTracker(Tracker):
    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
        self._step = 0

    def register(self, key: str, shape: tuple[int, ...]) -> None:
        if key in _MARGINAL_METRICS:
            super().register(key, (len(_MARGINAL_STEPS), *shape[1:]))

    def log(self, values: Mapping[str, torch.Tensor]) -> None:
        if self._step in _MARGINAL_STEPS:
            super().log({key: value for key, value in values.items() if key in _MARGINAL_METRICS})
        self._step += 1


def _weighted_wasserstein1(
    samples: torch.Tensor,
    weights: torch.Tensor,
    grid: torch.Tensor,
    density: torch.Tensor,
) -> float:
    """Weighted empirical-to-density Wasserstein-1 distance.

    Inverts the target CDF at the particles' own quantile levels (midpoints of their weight
    intervals), so the distance is read off matched quantiles rather than a shared grid.
    """
    sorted_samples, order = torch.sort(samples)
    sorted_weights = weights[order]
    sorted_weights = sorted_weights / sorted_weights.sum()

    dx = grid[1] - grid[0]
    target_cdf = torch.cumsum(density, dim=0) * dx
    target_cdf = target_cdf / target_cdf[-1]
    quantile_levels = torch.cumsum(sorted_weights, dim=0) - 0.5 * sorted_weights
    grid_indices = torch.searchsorted(target_cdf, quantile_levels).clamp(1, target_cdf.shape[0] - 1)
    cdf_lower = target_cdf[grid_indices - 1]
    cdf_upper = target_cdf[grid_indices]
    grid_lower = grid[grid_indices - 1]
    grid_upper = grid[grid_indices]
    fraction = (quantile_levels - cdf_lower) / (cdf_upper - cdf_lower).clamp_min(1e-12)
    target_quantiles = grid_lower + fraction * (grid_upper - grid_lower)
    return float((sorted_weights * (sorted_samples - target_quantiles).abs()).sum())


def _tweedie(x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    """Exact posterior mean of the 1-D GMM, applied elementwise.

    Coordinate-agnostic: the invariant denoiser feeds it ``u``, the absolute one
    feeds it raw atom coordinates, and `_analytic_marginal` feeds it a grid.
    Differentiable w.r.t. both ``x`` and ``sigma`` — the latter matters because
    the guidance weight reads ``dU/dsigma`` off a sigma-leaf threaded through
    the denoiser, exactly as the production score model allows.
    """
    sigma_t = sigma if isinstance(sigma, torch.Tensor) else torch.tensor(float(sigma))
    with torch.enable_grad():
        leaf = x if x.requires_grad else x.detach().requires_grad_(True)
        component_scale = torch.sqrt(_STD**2 + sigma_t**2).unsqueeze(-1)
        component_log_prob = (
            Normal(_MODES.to(leaf), component_scale).log_prob(leaf.unsqueeze(-1))
            + _WEIGHTS.to(leaf).log()
        )
        log_prob = torch.logsumexp(component_log_prob, dim=-1)
        (score,) = torch.autograd.grad(log_prob.sum(), leaf, create_graph=True)
    return x + sigma_t**2 * score


class _InvariantGMM:
    """Tweedie denoiser acting on the COM-invariant coordinate ``u``.

    Unlike the shared `test_edm_steered_gmm._GMM` the score keeps its graph, so the
    guidance parameterization can backpropagate ``U(D(x))`` through to ``x``.
    """

    edm_schedule = EDMSchedule(sigma_min=_SIGMA_MIN, sigma_max=_SIGMA_MAX, sigma_data=1.0)

    def compute_static_context(self, batch, trunk_embeddings, multiplicity):
        assert trunk_embeddings is _DUMMY_TRUNK
        return _DUMMY_CONTEXT

    def denoise(self, context, trunk_embeddings, coords, sigmas):
        assert trunk_embeddings is _DUMMY_TRUNK
        x = _tweedie(_u(coords), sigmas.reshape(-1, 1)) / _SQRT2
        return SimpleNamespace(denoised_coords=torch.stack([x, -x], dim=1))  # antipodal


class _InvariantAxisWell:
    """Quadratic well on one axis of ``u``, evaluated on denoised coordinates."""

    def __init__(self, center: float, strength: float) -> None:
        self.center = center
        self.strength = strength

    def __call__(self, coords: torch.Tensor, *, t: float) -> torch.Tensor:
        return self.strength * (_u(coords)[:, 0] - self.center).square()


class _GMM:
    """Tweedie denoiser acting on absolute atom coordinates.

    With centering and augmentation neutralized, every coordinate component
    diffuses independently under the same 1-D GMM, so no antipodal pairing is
    imposed and each component's marginal is ``p_sigma``.
    """

    edm_schedule = EDMSchedule(sigma_min=_SIGMA_MIN, sigma_max=_SIGMA_MAX, sigma_data=1.0)

    def compute_static_context(self, batch, trunk_embeddings, multiplicity):
        assert trunk_embeddings is _DUMMY_TRUNK
        return _DUMMY_CONTEXT

    def denoise(self, context, trunk_embeddings, coords, sigmas):
        assert trunk_embeddings is _DUMMY_TRUNK
        denoised = _tweedie(coords, sigmas.reshape(-1, 1, 1))
        return SimpleNamespace(denoised_coords=denoised)


class _AxisWell:
    """Quadratic well on the x component of atom 0, in absolute coordinates.

    Deliberately *not* SE(3) invariant: its value moves under the frame changes
    the production sampler applies each step.
    """

    def __init__(self, center: float, strength: float) -> None:
        self.center = center
        self.strength = strength

    def __call__(self, coords: torch.Tensor, *, t: float) -> torch.Tensor:
        return self.strength * (coords[:, 0, 0] - self.center).square()


def _invariant_coordinate(coords: torch.Tensor) -> torch.Tensor:
    """Steered scalar for the invariant run: the tilted axis of ``u``."""
    return _u(coords)[:, 0]


def _absolute_coordinate(coords: torch.Tensor) -> torch.Tensor:
    """Steered scalar for the absolute run: the x component of atom 0."""
    return coords[:, 0, 0]


def _analytic_marginal(
    grid: torch.Tensor,
    sigma: float,
    beta: float,
    reward_center: float,
    reward_strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the base marginal and the denoised-reward-tilted density.

    Shared by both runs: in each case the steered scalar has the same 1-D GMM
    prior and the same tilt, so only the sampler's geometry handling differs.
    """
    base_mix = MixtureSameFamily(
        Categorical(probs=_WEIGHTS), Normal(_MODES, math.sqrt(_STD**2 + sigma**2))
    )
    log_base = base_mix.log_prob(grid)
    denoised = _tweedie(grid.clone(), sigma).detach()
    log_target = log_base - beta * reward_strength * (denoised - reward_center).square()
    target = (log_target - log_target.max()).exp()
    return log_base.exp(), target / torch.trapezoid(target, grid)


@pytest.fixture
def neutralized(monkeypatch):
    """Fix the frame so the invariant coordinate ``u`` is directly comparable."""
    monkeypatch.setattr(_edm_fkc_module, "compute_random_augmentation", _identity_aug)


@pytest.fixture
def neutralized_absolute(monkeypatch):
    """Also disable COM removal, so absolute atom coordinates are preserved.

    Without this the sampler would re-center every step and the absolute
    coordinate the potential reads would carry no stable meaning.
    """
    monkeypatch.setattr(_edm_fkc_module, "compute_random_augmentation", _identity_aug)
    monkeypatch.setattr(
        _edm_fkc_module,
        "augment_coordinates",
        lambda coords, rotation, translation, mask: coords,
    )


def _unpack(
    metrics: Path, project: Callable[[torch.Tensor], torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """Project the tracked coordinates onto the steered scalar."""
    assert sum(path.stat().st_size for path in metrics.glob("*.npy")) < 10_000_000
    coords = torch.from_numpy(np.load(metrics / "x_t.npy"))
    values = project(coords.reshape(-1, *coords.shape[2:])).reshape(coords.shape[:2])
    return (
        values,
        torch.from_numpy(np.load(metrics / "weights.npy")),
        torch.from_numpy(np.load(metrics / "sigma.npy")),
        torch.from_numpy(np.load(metrics / "beta.npy")),
    )


@pytest.fixture
def marginal_metrics(tmp_path: Path) -> Generator[Path, None, None]:
    with TemporaryDirectory(dir=tmp_path) as directory:
        yield Path(directory)


@pytest.mark.parametrize(
    ("reheat", "weight_update"),
    [
        pytest.param("gamma", "smc", id="gamma-smc"),
        pytest.param("sigma", "smc", id="sigma-smc"),
        pytest.param("sigma", "guidance", id="sigma-guidance"),
    ],
)
@pytest.mark.parametrize(
    "resampling_interval,ess_threshold",
    [
        pytest.param(None, 0.9, id="ess0.9"),
        # A cadence longer than the run resamples only on the terminal step.
        pytest.param(_NUM_STEPS + 1, None, id="final-only"),
    ],
)
def test_invariant_weighted_intermediate_marginals(
    neutralized,
    marginal_metrics: Path,
    reheat: Reheat,
    weight_update: WeightUpdate,
    resampling_interval: int | None,
    ess_threshold: float | None,
) -> None:
    """
    Check an invariant sampler steered with an invariant potential against the analytic marginals of the tilted distribution
    for multiple intermediate t's
    This is in contrast to absolute coordinates in the test below.
    """
    center, strength = -2.0, 0.5
    torch.manual_seed(0)
    score_model = _InvariantGMM()
    batch = _prepared_batch(score_model)
    edm_fkc_sampler(
        batch=cast(BatchAllAtomStructure, batch),
        num_steps=_NUM_STEPS,
        score_model=cast(InferenceModel, score_model),
        trunk_embeddings=_DUMMY_TRUNK,
        multiplicity=_NUM_PARTICLES,
        rho=_RHO,
        gamma_0=0.8,
        gamma_min=1.0,
        potentials=[cast(Potential, _InvariantAxisWell(center, strength))],
        resampling_interval=resampling_interval,
        ess_threshold=ess_threshold,
        weight_update=weight_update,
        beta_power=2.0,
        reheat=reheat,
        tracker=_MarginalTracker(marginal_metrics),
    )
    values, weights, sigmas, betas = _unpack(marginal_metrics, _invariant_coordinate)

    for row, step_index in enumerate(_MARGINAL_STEPS):
        sigma = float(sigmas[row])
        radius = max(5.0, 2.0 + 6.0 * math.sqrt(_STD**2 + sigma**2))
        grid = torch.linspace(-radius, radius, 3_201)
        _, target = _analytic_marginal(
            grid,
            sigma=sigma,
            beta=float(betas[row]),
            reward_center=center,
            reward_strength=strength,
        )
        w1 = _weighted_wasserstein1(values[row], weights[row], grid, target)
        tolerance = 0.12 * (1.0 + sigma)
        assert w1 < tolerance, (
            f"{reheat}-{weight_update} step={step_index + 1} sigma={sigma:.4g}: "
            f"W1={w1:.4f} >= {tolerance:.4f}"
        )


@pytest.mark.parametrize(
    ("reheat", "weight_update"),
    [
        pytest.param("gamma", "smc", id="gamma-smc"),
        pytest.param("sigma", "smc", id="sigma-smc"),
        pytest.param("sigma", "guidance", id="sigma-guidance"),
    ],
)
@pytest.mark.parametrize(
    "resampling_interval,ess_threshold",
    [
        # pytest.param(None, 0.9, id="ess0.9"),
        # A cadence longer than the run resamples only on the terminal step.
        pytest.param(_NUM_STEPS + 1, None, id="final-only"),
    ],
)
def test_absolutecoordinate_weighted_intermediate_marginals(
    neutralized_absolute,
    marginal_metrics: Path,
    reheat: Reheat,
    weight_update: WeightUpdate,
    resampling_interval: int | None,
    ess_threshold: float | None,
) -> None:
    center, strength = -2.0, 0.5
    torch.manual_seed(0)
    score_model = _GMM()
    batch = _prepared_batch(score_model)
    edm_fkc_sampler(
        batch=cast(BatchAllAtomStructure, batch),
        num_steps=_NUM_STEPS,
        score_model=cast(InferenceModel, score_model),
        trunk_embeddings=_DUMMY_TRUNK,
        multiplicity=_NUM_PARTICLES,
        rho=_RHO,
        gamma_0=0.8,
        gamma_min=1.0,
        potentials=[cast(Potential, _AxisWell(center, strength))],
        resampling_interval=resampling_interval,
        ess_threshold=ess_threshold,
        weight_update=weight_update,
        beta_power=1.0,
        reheat=reheat,
        tracker=_MarginalTracker(marginal_metrics),
    )
    values, weights, sigmas, betas = _unpack(marginal_metrics, _absolute_coordinate)

    for row, step_index in enumerate(_MARGINAL_STEPS):
        sigma = float(sigmas[row])
        radius = max(5.0, 2.0 + 6.0 * math.sqrt(_STD**2 + sigma**2))
        grid = torch.linspace(-radius, radius, 3_201)
        _, target = _analytic_marginal(
            grid,
            sigma=sigma,
            beta=float(betas[row]),
            reward_center=center,
            reward_strength=strength,
        )
        w1 = _weighted_wasserstein1(values[row], weights[row], grid, target)
        tolerance = 0.12 * (1.0 + sigma)
        assert w1 < tolerance, (
            f"{reheat}-{weight_update} step={step_index + 1} sigma={sigma:.4g}: "
            f"W1={w1:.4f} >= {tolerance:.4f}"
        )


def test_guidance_requires_sigma_reheating() -> None:
    """The sampler validates the steering policy against its denoiser proposal."""
    with pytest.raises(ValueError, match="requires reheat='sigma'"):
        batch = _prepared_batch(_InvariantGMM())
        edm_fkc_sampler(
            batch=cast(BatchAllAtomStructure, batch),
            num_steps=2,
            score_model=cast(InferenceModel, None),
            trunk_embeddings=_DUMMY_TRUNK,
            weight_update="guidance",
            reheat="gamma",
        )


def test_edm_fkc_sampler_validates_kwargs() -> None:
    """The sampler validates its resampling/particle/weight-update kwargs up front,
    before touching ``batch``/``score_model``."""
    batch = _prepared_batch(_InvariantGMM())

    def _call(**kwargs):
        edm_fkc_sampler(
            batch=cast(BatchAllAtomStructure, batch),
            num_steps=2,
            score_model=cast(InferenceModel, None),
            trunk_embeddings=_DUMMY_TRUNK,
            **kwargs,
        )

    with pytest.raises(ValueError, match="unknown reheat"):
        _call(reheat="bogus")
    with pytest.raises(ValueError, match="at most one resampling trigger"):
        _call(resampling_interval=3, ess_threshold=0.5)
    with pytest.raises(ValueError, match="got neither"):
        _call(resampling_interval=None, ess_threshold=None)
    with pytest.raises(ValueError, match="resampling_interval=0"):
        _call(resampling_interval=0, ess_threshold=None)
    with pytest.raises(ValueError, match="resampling_group_size=1"):
        _call(resampling_interval=1, resampling_group_size=1, ess_threshold=None)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        _call(ess_threshold=-0.1)
    with pytest.raises(ValueError, match="unknown weight_update"):
        _call(weight_update="invalid")


@pytest.mark.parametrize(
    ("values", "weight_update", "message"),
    [
        ((0.3, math.nan), "smc", r"potential 1 \(ConstantEnergy\)"),
        ((0.3, math.inf), "smc", r"potential 1 \(ConstantEnergy\)"),
        ((0.3, -math.inf), "smc", r"potential 1 \(ConstantEnergy\)"),
        ((0.3, math.nan), "guidance", r"potential 1 \(ConstantEnergy\)"),
        ((2e38, 2e38), "smc", "Non-finite total potential energy"),
    ],
)
def test_edm_fkc_rejects_invalid_energies(
    values: tuple[float, float], weight_update: WeightUpdate, message: str
) -> None:
    class ConstantEnergy:
        def __init__(self, value: float) -> None:
            self.value = value

        def __call__(self, coords: torch.Tensor, *, t: float) -> torch.Tensor:
            return coords.new_full((coords.shape[0],), self.value)

    score_model = _InvariantGMM()
    with pytest.raises(ValueError, match=message):
        edm_fkc_sampler(
            batch=cast(BatchAllAtomStructure, _prepared_batch(score_model)),
            num_steps=2,
            score_model=cast(InferenceModel, score_model),
            trunk_embeddings=_DUMMY_TRUNK,
            potentials=[cast(Potential, ConstantEnergy(value)) for value in values],
            reheat="sigma",
            weight_update=weight_update,
        )


def test_edm_fkc_rejects_nonfinite_guidance_weights() -> None:
    class SingularEnergy:
        def __call__(self, coords: torch.Tensor, *, t: float) -> torch.Tensor:
            return (coords - coords.detach()).square().sum(dim=(1, 2)).sqrt()

    score_model = _InvariantGMM()
    with pytest.raises(ValueError, match="Non-finite particle log-weight at step 0"):
        edm_fkc_sampler(
            batch=cast(BatchAllAtomStructure, _prepared_batch(score_model)),
            num_steps=2,
            score_model=cast(InferenceModel, score_model),
            trunk_embeddings=_DUMMY_TRUNK,
            potentials=[cast(Potential, SingularEnergy())],
            reheat="sigma",
            weight_update="guidance",
        )


@pytest.mark.parametrize(
    ("reheat", "weight_update", "use_potentials"),
    [
        ("gamma", "smc", False),
        ("gamma", "smc", True),
        ("sigma", "guidance", True),
    ],
)
def test_edm_fkc_sampler_records_destination_state(
    tmp_path: Path, reheat: Reheat, weight_update: WeightUpdate, use_potentials: bool
) -> None:
    class TimeWell(_AxisWell):
        def __call__(self, coords: torch.Tensor, *, t: float) -> torch.Tensor:
            return (1.0 + t) * super().__call__(coords, t=t)

    torch.manual_seed(0)
    score_model = _InvariantGMM()
    batch = _prepared_batch(score_model)
    potentials = (
        [TimeWell(center=1.0, strength=0.3), TimeWell(center=-1.0, strength=0.2)]
        if use_potentials
        else []
    )
    result = edm_fkc_sampler(
        batch=cast(BatchAllAtomStructure, batch),
        num_steps=5,
        score_model=cast(InferenceModel, score_model),
        trunk_embeddings=_DUMMY_TRUNK,
        multiplicity=3,
        potentials=cast(list[Potential], potentials),
        reheat=reheat,
        weight_update=weight_update,
        tracker=Tracker(tmp_path / "metrics"),
    )
    metrics = tmp_path / "metrics"
    coords = torch.from_numpy(np.load(metrics / "x_t.npy"))
    x0 = torch.from_numpy(np.load(metrics / "x0.npy"))
    sigmas = torch.from_numpy(np.load(metrics / "sigma.npy"))
    expected = score_model.denoise(
        context=_DUMMY_CONTEXT,
        trunk_embeddings=_DUMMY_TRUNK,
        coords=coords[:-1].reshape(-1, 2, 3),
        sigmas=sigmas[:-1].repeat_interleave(3),
    ).denoised_coords
    torch.testing.assert_close(x0[:-1], expected.reshape(x0[:-1].shape))
    torch.testing.assert_close(x0[-1], coords[-1], rtol=0, atol=0)
    for index, potential in enumerate(potentials):
        energy = np.load(metrics / f"{index}_TimeWell.npy")
        expected_energy = torch.stack(
            [
                potential(step_coords, t=1.0 - (step + 1) / len(x0))
                for step, step_coords in enumerate(x0)
            ]
        )
        np.testing.assert_allclose(energy, expected_energy.numpy(), rtol=1e-6, atol=1e-6)
    if not potentials:
        torch.testing.assert_close(coords[-1], result.coords, rtol=0, atol=0)


@pytest.mark.parametrize("potential_type", [BondLengthPotential, BondAnglePotential])
def test_edm_fkc_guidance_with_empty_bonded_restraints(potential_type, tmp_path: Path) -> None:
    """Test that a single atom with no bonds or angles has zero bonded energy
    and gives exactly the same FKC samples as running without potentials.
    """
    system = System(entities=[Ligand(smiles="C", asym_ids="A")])
    batch = batch_collate([AllAtomStructure.from_system(system, mode="structure")])
    potential = potential_type(sigma_tolerance=10.0)
    potential.prepare(system)
    results = []
    for potentials, tracker in (([potential], Tracker(tmp_path / "metrics")), ([], None)):
        torch.manual_seed(0)
        results.append(
            edm_fkc_sampler(
                batch=batch,
                num_steps=3,
                score_model=cast(InferenceModel, _GaussianScoreModel()),
                trunk_embeddings=_GAUSSIAN_TRUNK,
                multiplicity=2,
                potentials=potentials,
                weight_update="guidance",
                reheat="sigma",
                tracker=tracker,
            )
        )
    guided, unguided = results
    assert torch.isfinite(guided.coords).all()
    assert not np.load(tmp_path / "metrics" / f"0_{potential_type.__name__}.npy").any()
    torch.testing.assert_close(guided.coords, unguided.coords, rtol=0, atol=0)


def test_edm_fkc_guidance_with_empty_local_clash_potential(tmp_path: Path) -> None:
    """Test that FKC samples are unchanged when there are no local atom pairs to penalize."""
    system = System(entities=[Protein(sequence="G", asym_ids="A")])
    batch = batch_collate([AllAtomStructure.from_system(system, mode="structure")])
    potential = PolymerLocalClashPotential(parameters={"minimum_distance": 2.0})
    potential.prepare(system)
    torch.manual_seed(0)
    result = edm_fkc_sampler(
        batch=batch,
        num_steps=3,
        score_model=cast(InferenceModel, _GaussianScoreModel()),
        trunk_embeddings=_GAUSSIAN_TRUNK,
        multiplicity=2,
        potentials=[potential],
        weight_update="guidance",
        reheat="sigma",
        tracker=Tracker(tmp_path / "metrics"),
    )
    assert torch.isfinite(result.coords).all()
    assert not np.load(tmp_path / "metrics" / "0_PolymerLocalClashPotential.npy").any()
    torch.manual_seed(0)
    unguided = edm_fkc_sampler(
        batch=batch,
        num_steps=3,
        score_model=cast(InferenceModel, _GaussianScoreModel()),
        trunk_embeddings=_GAUSSIAN_TRUNK,
        multiplicity=2,
        potentials=[],
        weight_update="guidance",
        reheat="sigma",
    )
    torch.testing.assert_close(result.coords, unguided.coords, rtol=0, atol=0)


def test_public_exports_importable():
    # Import inside the test so a broken public export fails this test rather than collection.
    from bioemu2 import Potential as ApiPotential  # noqa: PLC0415
    from bioemu2.denoising.steering import (  # noqa: F401, PLC0415
        Potential,
        edm_fkc_sampler,
        edm_steered_sampler,
    )

    assert ApiPotential is Potential
