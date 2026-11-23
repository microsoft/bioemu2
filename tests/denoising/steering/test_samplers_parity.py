"""CPU parity test for the steered EDM sampler.

Asserts that `edm_steered_sampler` without potentials produces bit-exact
output identical to `edm_sampler`. Uses a stub score model so it runs
on CPU with no checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from bioemu2.biomoa.models.edm_schedule import EDMSchedule
from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.data_interface import Protein, System
from bioemu2.denoising.edm import edm_sampler
from bioemu2.denoising.samplers import GuidedFKSampler
from bioemu2.denoising.steering.edm_steered import edm_steered_sampler
from bioemu2.denoising.steering.feats import Tracker
from bioemu2.denoising.steering.potentials import (
    BondAnglePotential,
    BondLengthPotential,
    Potential,
)
from bioemu2.model import InferenceModel
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.batch import batch_collate

_DUMMY_CONTEXT = SimpleNamespace(
    coords_cond=None,
    is_dynamics_conditioned=None,
    dt=None,
)
_DUMMY_TRUNK = cast(TrunkEmbeddings, object())


class _GaussianScoreModel:
    """Optimal denoiser for a unit-Gaussian prior p(x₀) = N(0, I).

    E[x₀ | xₜ, σ] = xₜ / (σ² + 1)

    This makes the update non-trivial (coords shrink toward 0 each step),
    so the parity test exercises the full loop rather than a no-op.
    """

    edm_schedule = EDMSchedule()

    def eval(self):
        return self

    def compute_static_context(self, batch, trunk_embeddings, multiplicity):
        assert trunk_embeddings is _DUMMY_TRUNK
        return _DUMMY_CONTEXT

    def denoise(self, context, trunk_embeddings, coords, sigmas):
        assert trunk_embeddings is _DUMMY_TRUNK
        denoised = coords / (sigmas[:, None, None] ** 2 + 1)
        return SimpleNamespace(denoised_coords=denoised)


def _kwargs(num_steps=5):
    return dict(
        num_steps=num_steps,
        step_scale=1.0,
        noise_scale=1.0,
    )


def test_edm_steered_no_steering_is_bit_exact(test_structure):
    """`edm_steered_sampler` with no potentials must match `edm_sampler` exactly,
    on non-zero inputs and producing non-zero outputs."""
    # Seed the init draw so the starting coords are deterministic + non-zero —
    # an all-zeros tensor would let trivial shape/dtype bugs slip through.
    # The sampler-internal manual_seed(0) calls below reset RNG for parity.
    torch.manual_seed(42)
    test_structure = test_structure.replace(coords=torch.randn(test_structure.num_nodes, 3))
    assert test_structure.coords.abs().sum() > 0, "input coords must be non-zero"
    batch = batch_collate([test_structure])
    score_model = _GaussianScoreModel()
    torch.manual_seed(0)
    ref = edm_sampler(
        batch=batch,
        score_model=cast(InferenceModel, score_model),
        trunk_embeddings=_DUMMY_TRUNK,
        **_kwargs(),
    )
    torch.manual_seed(0)
    steered = edm_steered_sampler(
        batch=batch,
        score_model=cast(InferenceModel, score_model),
        trunk_embeddings=_DUMMY_TRUNK,
        **_kwargs(),
    )

    assert ref.coords.abs().sum() > 0, "edm_sampler produced all-zero coords"
    assert steered.coords.abs().sum() > 0, "edm_steered_sampler produced all-zero coords"
    assert torch.equal(ref.coords, steered.coords)


def test_edm_steered_rejects_batch_with_multiplicity(test_structure):
    batch_size = 2
    multiplicity = 3
    batch = batch_collate([test_structure] * batch_size)

    with pytest.raises(NotImplementedError, match="Steered samplers take one structure per call"):
        edm_steered_sampler(
            batch=batch,
            score_model=cast(InferenceModel, _GaussianScoreModel()),
            trunk_embeddings=_DUMMY_TRUNK,
            multiplicity=multiplicity,
            **_kwargs(num_steps=2),
        )


@pytest.mark.parametrize("potential_type", [BondLengthPotential, BondAnglePotential])
def test_default_protein_bond_guidance_changes_samples(potential_type):
    """Test that each default bonded potential changes protein samples from the
    same-seed unsteered run without producing NaN or infinite coordinates.
    """
    system = System(entities=[Protein(sequence="AAA", asym_ids="A")])
    batch = batch_collate([AllAtomStructure.from_system(system, mode="structure")])
    guided_potential = next(
        p for p in GuidedFKSampler().potentials if isinstance(p, potential_type)
    )
    guided_potential.prepare(system)

    def run(potentials):
        torch.manual_seed(17)
        return edm_steered_sampler(
            batch=batch,
            score_model=cast(InferenceModel, _GaussianScoreModel()),
            trunk_embeddings=_DUMMY_TRUNK,
            potentials=potentials,
            resampling_interval=None,
            num_gd_steps=2,
            alignment_reverse_diff=False,
            **_kwargs(num_steps=6),
        ).coords

    guided = run([guided_potential])
    unsteered = run([])
    assert torch.isfinite(guided).all()
    assert not torch.allclose(guided, unsteered)


def test_default_bond_guidance_starts_at_schedule_boundary(tmp_path):
    """Test that default bond guidance leaves early steps unchanged and starts in the final 20%.

    Run the real 200-step sampler on a tiny protein with a stub denoiser and the same random
    seed for guided and unguided sampling. Record calls to the default bond-length potential
    and compare predicted clean coordinates at every step, using one GD iteration per step.
    """

    class RecordingPotential(Potential):
        def __init__(self, wrapped: BondLengthPotential):
            self.wrapped = wrapped
            self.times: list[float] = []

        def prepare(self, system: System) -> None:
            self.wrapped.prepare(system)

        def compute_parameters(self, t: float) -> dict[str, Any] | None:
            return self.wrapped.compute_parameters(t)

        def __call__(self, coords_angstrom: torch.Tensor, *, t: float) -> torch.Tensor:
            self.times.append(t)
            return self.wrapped(coords_angstrom, t=t)

    system = System(entities=[Protein(sequence="AAA", asym_ids="A")])
    potential = RecordingPotential(
        next(p for p in GuidedFKSampler().potentials if isinstance(p, BondLengthPotential))
    )
    potential.prepare(system)
    batch = batch_collate([AllAtomStructure.from_system(system, mode="structure")])
    sampler = GuidedFKSampler(
        num_steps=200,
        potentials=(potential,),
        resampling_interval=None,
        num_gd_steps=1,
    )
    score_model = cast(InferenceModel, _GaussianScoreModel())

    def run(sampler, name):
        tracker = Tracker(tmp_path / name)
        torch.manual_seed(17)
        result = sampler(
            batch=batch,
            score_model=score_model,
            trunk_embeddings=_DUMMY_TRUNK,
            batch_size=1,
            tracker=tracker,
        )
        return result.coords, np.load(tmp_path / name / "x0_hat.npy")

    guided, guided_path = run(sampler, "guided")
    unsteered, unsteered_path = run(
        GuidedFKSampler(num_steps=200, potentials=(), resampling_interval=None), "unsteered"
    )
    expected_times = [1.0 - (step + 1) / 200 for step in range(159, 199)]
    # With t = 1 - (step + 1) / 200, t <= 0.2 activates steps 159-198; the final step skips GD.
    assert potential.times == pytest.approx(expected_times)
    # Zero guidance weights must leave the entire early trajectory bit-identical.
    np.testing.assert_array_equal(guided_path[:159], unsteered_path[:159])
    # Guidance must change coordinates at its first active step, not just evaluate the potential.
    assert not np.array_equal(guided_path[159], unsteered_path[159])
    assert torch.isfinite(guided).all()
    assert not torch.equal(guided, unsteered)
