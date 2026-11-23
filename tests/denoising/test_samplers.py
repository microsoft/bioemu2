"""Behavioural tests for the public sampler classes and their configuration."""

from collections.abc import Callable
from typing import Any

import pytest
import torch

import bioemu2
from bioemu2 import cli, denoising
from bioemu2.data_interface import Protein, System
from bioemu2.denoising.samplers import EDMSampler, FKCSampler, GuidedFKSampler, Sampler
from bioemu2.denoising.steering.potentials import (
    BondAnglePotential,
    BondLengthPotential,
    IntraChainVDWOverlapPotential,
    PolymerLocalClashPotential,
    SymmetricChainCOMPotential,
)
from bioemu2.structure import AllAtomStructure

_PRESETS: list[tuple[str, type[Sampler], Callable[[Any], bool]]] = [
    (
        "edm_fk",
        GuidedFKSampler,
        lambda s: s.resampling_interval == 3 and len(s.potentials) > 0,
    ),
    (
        "edm_gd",
        GuidedFKSampler,
        lambda s: s.resampling_interval is None and s.num_gd_steps > 0,
    ),
    (
        "edm_fkc",
        FKCSampler,
        lambda s: s.weight_update == "guidance" and s.reheat == "sigma",
    ),
]


@pytest.mark.parametrize("overlay,expected_type,check", _PRESETS, ids=[p[0] for p in _PRESETS])
def test_steering_presets_instantiate_samplers(
    overlay: str, expected_type: type[Sampler], check: Callable[[Any], bool]
) -> None:
    """CLI steering presets select the intended algorithms and default physicality potentials."""
    sampler = cli._sampler(overlay, None)
    assert isinstance(sampler, expected_type)
    assert isinstance(sampler, GuidedFKSampler | FKCSampler)
    assert len(sampler.potentials) == 10
    assert [type(p) for p in sampler.potentials[-2:]] == [BondLengthPotential, BondAnglePotential]
    for potential in sampler.potentials[-2:]:
        assert isinstance(potential, BondLengthPotential | BondAnglePotential)
        early_parameters = potential.compute_parameters(0.205)
        assert early_parameters is not None
        assert early_parameters["guidance_weight"] == 0.0
        assert potential.compute_parameters(0.2) == {
            "guidance_interval": 1,
            "guidance_weight": 0.05,
            "resampling_weight": 0.1,
        }
    assert check(sampler)


def test_default_config_is_plain_edm() -> None:
    """The default CLI sampler remains unsteered EDM."""
    sampler = cli._sampler("edm", None)
    assert isinstance(sampler, EDMSampler)
    assert sampler.num_steps == 200
    assert sampler.step_scale == 1.0
    assert not hasattr(sampler, "potentials")


def test_unknown_sampler_preset_lists_valid_choices() -> None:
    """An invalid preset reports the supported sampler names."""
    with pytest.raises(ValueError) as error:
        cli._sampler("unknown", None)
    assert str(error.value) == (
        "Unknown sampler preset: unknown. Choose from: edm, edm_gd, edm_fk, edm_fkc"
    )


@pytest.mark.parametrize(
    ("sampler_type", "preset"),
    [(GuidedFKSampler, "edm_fk"), (FKCSampler, "edm_fkc")],
)
def test_cli_and_python_default_potentials_have_matching_energies_and_gradients(
    sampler_type: type[GuidedFKSampler] | type[FKCSampler], preset: str
) -> None:
    """CLI and Python defaults produce identical physicality penalties and steering gradients."""
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A", "B"])])
    coords = torch.randn(
        2, system.topology.n_atoms, 3, generator=torch.Generator().manual_seed(42)
    ).requires_grad_()
    python_sampler = sampler_type()
    cli_sampler = cli._sampler(preset, None)
    assert isinstance(cli_sampler, GuidedFKSampler | FKCSampler)
    energies = []
    gradients = []
    for sampler in (python_sampler, cli_sampler):
        for potential in sampler.potentials:
            potential.prepare(system)
        energy = torch.stack([potential(coords, t=0.0) for potential in sampler.potentials])
        energies.append(energy)
        gradients.append(torch.autograd.grad(energy.sum(), coords)[0])

    assert energies[0].sum() > 0
    assert gradients[0].abs().sum() > 0
    torch.testing.assert_close(energies[0], energies[1])
    torch.testing.assert_close(gradients[0], gradients[1])


@pytest.mark.parametrize("sampler_type", [GuidedFKSampler, FKCSampler])
@pytest.mark.parametrize(
    "potential_type",
    [
        BondLengthPotential,
        BondAnglePotential,
        IntraChainVDWOverlapPotential,
        PolymerLocalClashPotential,
    ],
)
def test_default_potentials_keep_separate_prepared_state(
    sampler_type: type[GuidedFKSampler] | type[FKCSampler],
    potential_type: (
        type[BondLengthPotential]
        | type[BondAnglePotential]
        | type[IntraChainVDWOverlapPotential]
        | type[PolymerLocalClashPotential]
    ),
) -> None:
    """Preparing a second sampler for another system cannot alter the first sampler's energies."""
    first, second = sampler_type(), sampler_type()
    first_system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    second_system = System(entities=[Protein(sequence="AA", asym_ids=["A"])])
    first_potential = next(p for p in first.potentials if isinstance(p, potential_type))
    second_potential = next(p for p in second.potentials if isinstance(p, potential_type))
    first_potential.prepare(first_system)
    coords = torch.randn(
        1, first_system.topology.n_atoms, 3, generator=torch.Generator().manual_seed(42)
    )
    expected = first_potential(coords, t=0.0)
    assert expected.item() > 0

    second_potential.prepare(second_system)
    torch.testing.assert_close(first_potential(coords, t=0.0), expected)


@pytest.mark.parametrize("t", [0.2, 0.205, 0.6, 0.61])
@pytest.mark.parametrize(
    "potential_type", [IntraChainVDWOverlapPotential, PolymerLocalClashPotential]
)
def test_default_polymer_sterics_apply_scheduled_weights(potential_type, t) -> None:
    """Test that default steric energies and gradients receive the expected weights at each time."""
    system = System(entities=[Protein(sequence="AAA", asym_ids="A")])
    coords = torch.randn(
        2, system.topology.n_atoms, 3, generator=torch.Generator().manual_seed(42)
    ).requires_grad_()
    potential = next(p for p in GuidedFKSampler().potentials if isinstance(p, potential_type))
    is_intrachain = potential_type is IntraChainVDWOverlapPotential
    direct = potential_type(
        parameters={"buffer": 0.225} if is_intrachain else {"minimum_distance": 2.0}
    )
    potential.prepare(system)
    direct.prepare(system)
    energy = potential(coords, t=t)
    reference_energy = direct(coords, t=t)
    gradient = torch.autograd.grad(energy.sum(), coords)[0]
    reference_gradient = torch.autograd.grad(reference_energy.sum(), coords)[0]
    assert reference_energy.sum() > 0
    assert reference_gradient.abs().sum() > 0
    torch.testing.assert_close(energy, reference_energy)
    torch.testing.assert_close(gradient, reference_gradient)
    parameters = potential.compute_parameters(t)
    assert parameters["guidance_interval"] == (5 if is_intrachain else 1)
    torch.testing.assert_close(
        parameters["guidance_weight"] * gradient,
        (0.125 if t <= 0.2 else 0.0) * reference_gradient,
    )
    resampling_weight = (0.01 if t <= 0.6 else 0.0) if is_intrachain else 0.1
    torch.testing.assert_close(
        parameters["resampling_weight"] * energy, resampling_weight * reference_energy
    )


def test_default_com_keeps_varying_buffer_while_guidance_is_delayed() -> None:
    """Test that delaying chain-center guidance preserves its changing distance buffer."""
    system = System(entities=[Protein(sequence="AG", asym_ids=["A", "B"])])
    structure = AllAtomStructure.from_system(system, mode="structure")
    potential = next(
        p for p in GuidedFKSampler().potentials if isinstance(p, SymmetricChainCOMPotential)
    )
    potential.prepare(system)
    coords = torch.zeros(1, structure.num_nodes, 3)
    coords[:, structure.allatom_asym_id == 1, 0] = 2.0

    early, boundary, final = (potential(coords, t=t) for t in (0.8, 0.2, 0.0))
    assert early.item() > boundary.item() > final.item() == 0.0
    early_parameters = potential.compute_parameters(0.8)
    boundary_parameters = potential.compute_parameters(0.2)
    assert early_parameters is not None and boundary_parameters is not None
    assert early_parameters["guidance_weight"] == 0.0
    assert boundary_parameters["guidance_weight"] == 0.5
    assert early_parameters["resampling_weight"] == 0.5
    assert boundary_parameters["guidance_interval"] == 4


def test_samplers_are_public() -> None:
    """Sampler classes are available through both public import paths."""
    for name in ("Sampler", "EDMSampler", "GuidedFKSampler", "FKCSampler"):
        assert getattr(bioemu2, name) is getattr(denoising, name)
