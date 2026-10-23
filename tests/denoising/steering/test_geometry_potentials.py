"""Bonded-potential angle numerics and system preparation."""

import math

import pytest
import torch

from bioemu2.data_interface import DNA, RNA, Ligand, Protein, System
from bioemu2.denoising.steering.potentials import (
    BondAnglePotential,
    BondLengthPotential,
    PlanarBondPotential,
)


def _evaluated(potential, coords):
    coords = coords.clone().requires_grad_(True)
    energy = potential(coords, t=0.5)
    return energy.detach(), torch.autograd.grad(energy.sum(), coords)[0]


@pytest.mark.parametrize("cls", [BondLengthPotential, BondAnglePotential])
@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_bonded_tolerance_must_be_finite_and_nonnegative(cls, tolerance):
    """Test that negative, NaN, and infinite tolerances are rejected before preparing bounds."""
    with pytest.raises(ValueError, match="sigma_tolerance"):
        cls(sigma_tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [0.0, 10.0])
def test_prepared_reference_bounds_use_requested_tolerance(tolerance):
    """Test that bond and angle bounds equal the reference mean plus/minus tolerance times sigma."""
    system = System(entities=[Protein(sequence="A", asym_ids="A")])
    atoms = {atom.name: atom.index for atom in system.topology.atoms}
    for cls, prefix, names, mean, sigma in (
        (BondLengthPotential, "bond", ("N", "CA"), 1.482, 0.0101),
        (BondAnglePotential, "angle", ("N", "C", "CA"), math.radians(109.627), math.radians(1.5)),
    ):
        potential = cls(sigma_tolerance=tolerance)
        potential.prepare(system)
        index = torch.tensor([atoms[name] for name in names])
        selected = (potential.feats[f"{prefix}_index"][0].T == index).all(dim=-1)
        assert selected.sum() == 1
        bounds = [
            potential.feats[f"{prefix}_{side}"][0, selected].item() for side in ("lower", "upper")
        ]
        assert bounds == pytest.approx(
            (mean - tolerance * sigma, mean + tolerance * sigma), abs=1e-6
        )


def _angle_coords(degrees, first=1.0, second=1.0):
    theta = math.radians(degrees)
    return torch.tensor(
        [[first, 0, 0], [0, 0, 0], [second * math.cos(theta), second * math.sin(theta), 0]],
        dtype=torch.float64,
    ).unsqueeze(0)


def _angle_feats(dtype=torch.float64):
    return {
        "angle_index": torch.tensor([[[0], [2], [1]]]),
        "angle_lower": torch.tensor([[math.radians(100)]], dtype=dtype),
        "angle_upper": torch.tensor([[math.radians(140)]], dtype=dtype),
    }


@pytest.mark.parametrize("angle,expected", [(90, 10), (100, 0), (120, 0), (140, 0), (150, 10)])
@pytest.mark.parametrize("first,second", [(1, 1), (0.7, 2.5), (4, 0.8)])
def test_angle_hinge_is_in_radians_and_arm_length_invariant(angle, expected, first, second):
    """Test radian penalties at and outside the angle bounds for different bond
    lengths, and check coordinate gradients against numerical derivatives.
    """
    coords = _angle_coords(angle, first, second).requires_grad_(True)
    potential = BondAnglePotential(sigma_tolerance=10.0)
    assert potential.compute(coords, _angle_feats(), {}).item() == pytest.approx(
        math.radians(expected), abs=1e-12
    )
    if expected:
        assert torch.autograd.gradcheck(
            lambda positions: potential.compute(positions, _angle_feats(), {}), (coords,)
        )


@pytest.mark.parametrize(
    "first,second", [(0, 0), (1, 2), (1, -2)], ids=["coincident", "collinear", "antiparallel"]
)
def test_degenerate_angle_has_finite_energy_and_gradient(first, second):
    """Test that coincident (undefined angle), parallel (0), and antiparallel (pi)
    atom triples produce neither NaN nor infinite energies or coordinate gradients.
    """
    coords = torch.tensor(
        [[[first, 0, 0], [0, 0, 0], [second, 0, 0]]], dtype=torch.float32, requires_grad=True
    )
    energy = BondAnglePotential(sigma_tolerance=10.0).compute(
        coords, _angle_feats(torch.float32), {}
    )
    assert torch.isfinite(energy).all()
    assert torch.isfinite(torch.autograd.grad(energy.sum(), coords)[0]).all()


@pytest.mark.parametrize("cls", [BondLengthPotential, BondAnglePotential])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_potentials_reject_unsupported_coordinate_dtypes(
    cls: type[BondLengthPotential | BondAnglePotential],
    dtype: torch.dtype,
) -> None:
    system = System(entities=[Ligand(smiles="CCO", asym_ids="L")])
    potential = cls(sigma_tolerance=10.0)
    potential.prepare(system)
    coords = torch.zeros(1, system.topology.n_atoms, 3, dtype=dtype)
    with pytest.raises(ValueError, match=f"require float32 or float64 coordinates; got {dtype}"):
        potential(coords, t=0.5)


@pytest.mark.parametrize("cls", [BondLengthPotential, BondAnglePotential])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_potentials_preserve_supported_coordinate_dtypes(
    cls: type[BondLengthPotential | BondAnglePotential],
    dtype: torch.dtype,
) -> None:
    potential = cls(sigma_tolerance=10.0)
    potential.prepare(System(entities=[Ligand(smiles="CCO", asym_ids="L")]))
    coords = torch.tensor(
        [[[0.1, 0, 0], [0, 0, 0], [0.1, 0.01, 0]]], dtype=dtype, requires_grad=True
    )
    energy = potential(coords, t=0.5)
    (gradient,) = torch.autograd.grad(energy.sum(), coords)
    assert energy.dtype == dtype
    assert energy.item() > 0
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0


@pytest.mark.parametrize(
    "points",
    [
        [(0, 1, 0), (0, 0, 0), (0, 0, 0), (1, 1, 0)],
        [(0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0)],
        [(0, 1, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0)],
    ],
    ids=["coincident-center", "vanishing-first-plane", "vanishing-second-plane"],
)
def test_undefined_dihedral_has_zero_angle_and_coordinate_gradient(points):
    """Undefined torsions must mask atan2 inputs, not merely its non-finite output."""
    coords = torch.tensor([points], dtype=torch.float32, requires_grad=True)
    angle = PlanarBondPotential().compute_variable(coords, torch.tensor([[0], [1], [2], [3]]))
    (gradient,) = torch.autograd.grad(angle.sum(), coords)
    torch.testing.assert_close(angle, torch.zeros_like(angle))
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_mixed_dihedral_batch_preserves_valid_angles_and_gradients() -> None:
    coords = torch.tensor(
        [
            [(0, 1, 0.4), (0, 0, 0), (1, 0, 0), (1, 1, -0.3)],
            [(0, 1, 0), (0, 0, 0), (0, 0, 0), (1, 1, 0)],
            [(0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0)],
            [(0, 1, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0)],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    index = torch.tensor([[0], [1], [2], [3]])
    potential = PlanarBondPotential()
    valid_coords = coords[:1].detach().clone().requires_grad_(True)
    expected_angle = potential.compute_variable(valid_coords, index)
    (expected_gradient,) = torch.autograd.grad(expected_angle.sum(), valid_coords)

    angle = potential.compute_variable(coords, index)
    (gradient,) = torch.autograd.grad(angle.sum(), coords)

    assert expected_angle.abs().sum() > 0
    assert expected_gradient.abs().sum() > 0
    torch.testing.assert_close(angle[:1], expected_angle)
    torch.testing.assert_close(gradient[:1], expected_gradient)
    torch.testing.assert_close(angle[1:], torch.zeros_like(angle[1:]))
    torch.testing.assert_close(gradient[1:], torch.zeros_like(gradient[1:]))


def test_nonplanar_dihedral_gradient_matches_finite_differences():
    """Guarding undefined planes must leave well-defined torsion gradients intact."""
    coords = torch.tensor(
        [[(0.0, 1.0, 0.4), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, -0.3)]],
        dtype=torch.float64,
        requires_grad=True,
    )
    index = torch.tensor([[0], [1], [2], [3]])
    potential = PlanarBondPotential()
    assert torch.autograd.gradcheck(lambda x: potential.compute_variable(x, index), (coords,))


def test_ligand_angle_fallback_allows_linear_but_penalizes_folded_geometry():
    """Test that a linear C-C-O angle has zero penalty, while a folded angle has
    a positive penalty and a nonzero, finite coordinate gradient.
    """
    potential = BondAnglePotential(sigma_tolerance=10.0)
    potential.prepare(System(entities=[Ligand(smiles="CCO", asym_ids="L")]))
    linear = torch.tensor([[[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0]]])
    folded = torch.tensor([[[1.5, 0, 0], [0, 0, 0], [1.5, 0.15, 0]]])
    assert not potential(linear, t=0.5).any()
    energy, gradient = _evaluated(potential, folded)
    assert energy.item() > 0 and torch.isfinite(gradient).all() and gradient.abs().max() > 0


@pytest.mark.parametrize("cls", [BondLengthPotential, BondAnglePotential])
def test_bonded_repreparation_and_padding_preserve_energies_and_gradients(cls):
    """Test that preparing a reused potential matches a fresh one for each system,
    and that padding atoms changes neither energies nor real-atom gradients.
    """
    potential = cls(sigma_tolerance=10.0)
    for polymer, sequence in ((Protein, "AXG"), (DNA, "AC"), (RNA, "GU"), (Protein, "A")):
        system = System(entities=[polymer(sequence=sequence, asym_ids="A")])
        potential.prepare(system)
        fresh = cls(sigma_tolerance=10.0)
        fresh.prepare(system)
        coords = torch.randn(
            2, system.topology.n_atoms, 3, generator=torch.Generator().manual_seed(83)
        )
        expected_energy, expected_gradient = _evaluated(fresh, coords)
        assert expected_energy.min() > 0 and expected_gradient.abs().max() > 0
        # Exercise both collate-window padding and additional context-parallel padding.
        padding = potential.feats["atom_pad_mask"].shape[1] + 5 - coords.shape[1]
        padded = torch.cat((coords, torch.full((2, padding, 3), 1000.0)), dim=1)
        energy, gradient = _evaluated(potential, padded)
        torch.testing.assert_close(energy, expected_energy, rtol=0, atol=0)
        torch.testing.assert_close(
            gradient[:, : coords.shape[1]], expected_gradient, rtol=0, atol=0
        )
        assert not gradient[:, coords.shape[1] :].any()
