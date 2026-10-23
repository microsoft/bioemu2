# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""Inference-time steering potentials.

See ``docs/potentials.md`` and ``docs/steering.md`` for more information.

Single-system topology assumption: indices are taken from ``feats[...][0]``
(the first batch element). Every particle or replicate in the broadcast batch
shares one topology.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import functional as F

from bioemu2.biomoa.ops.atom_token_ops import broadcast_tokens_to_atoms
from bioemu2.data_interface import System
from bioemu2.denoising.rigid_align import weighted_rigid_align
from bioemu2.denoising.steering.feats import (
    convert_densebatch_to_potential_features,
    potential_features_from_system,
)
from bioemu2.denoising.steering.geometry_topology import (
    BondedGeometryGraph,
    StereoTables,
    build_bonded_geometry_graph,
    build_polymer_steric_graph,
    compute_geometry_bounds,
    get_angle_reference,
    get_bond_reference,
    load_stereo_tables,
    pack_angle_constraints,
    pack_bond_constraints,
)
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.batch import batch_collate


def _connected_zero(coords: torch.Tensor) -> torch.Tensor:
    """Per-particle zero with a differentiable path to coordinates, including empty atoms."""
    return coords[..., :0, :].sum(dim=(-2, -1))


def _atom_chain_id(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-atom chain id ``(N,)`` for the topology-shared batch element 0."""
    return (
        broadcast_tokens_to_atoms(
            feats["asym_id"].unsqueeze(-1),  # (B, T) -> (B, T, 1)
            feats["atom_to_token_ids"],  # (B, N)
            ~feats["atom_pad_mask"].bool(),  # (B, N)
        )  # (B, N, 1)
        .squeeze(-1)  # (B, N)
        .long()[0]  # (N,) topology-shared batch element 0
    )


def _atom_vdw_radius(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-atom VDW radii ``[N]`` for the topology-shared batch element 0."""
    radii = torch.zeros(num_elements, dtype=torch.float32, device=feats["ref_element"].device)
    radii[1:119] = torch.tensor(vdw_radii, dtype=torch.float32, device=radii.device)
    return (feats["ref_element"].float() @ radii[:, None]).squeeze(-1)[0]  # [N]


def _steric_overlap_penalty(
    coords: torch.Tensor,  # [B, N, 3]
    index: torch.Tensor,  # [2, P]
    minimum_distance: torch.Tensor | float,  # [P] or scalar
) -> torch.Tensor:  # [B]
    """Return overlap energies with distances smoothed at 1e-6 angstroms.

    Use ``sqrt(||x_i - x_j||**2 + eps**2)`` to soften gradients near coincident atoms.
    """
    i, j = index
    # Promote FP16 so eps**2 cannot underflow to zero.
    if coords.dtype == torch.float16:
        coords = coords.float()
    eps = 1e-6  # angstroms
    delta = coords.index_select(-2, i) - coords.index_select(-2, j)  # [B, P, 3]
    distance = torch.sqrt(delta.square().sum(dim=-1) + eps**2)  # [B, P]
    return torch.relu(minimum_distance - distance).sum(dim=-1)


def compute_atom_pairs_within_cutoff(
    coords: torch.Tensor,  # [B, N, 3]
    atom_mask: torch.Tensor,  # [N]
    atom_chain_id: torch.Tensor,  # [N]
    allowed_chain_pairs: torch.Tensor,  # [C, C]
    cutoff: float,
    block_size: int,
) -> torch.Tensor:  # [2, P]
    """Find allowed pairs close in any particle, using detached blockwise distances.

    For each pair, the search tests ``min_b ||x[b, i] - x[b, j]|| < cutoff``. Callers
    recompute differentiable distances only for returned pairs. Pairs satisfy ``i < j``.
    Distances use the direct, non-matmul path to avoid cancellation for nearby atoms
    far from the origin. FP16/BF16 search coordinates are promoted to FP32.

    Args:
        coords: Coordinates of shape ``[B, N, 3]`` in angstroms.
        atom_mask: Boolean mask of shape ``[N]``; ``True`` includes an atom in the search.
        atom_chain_id: Integer chain indices of shape ``[N]``, shared across particles.
        allowed_chain_pairs: Boolean mask of shape ``[C, C]`` indexed by chain indices;
            ``True`` permits pairs between those chains, including within a chain if
            its diagonal entry is ``True``.
        cutoff: Strict distance threshold in angstroms.
        block_size: Number of candidate first atoms processed per distance block.

    Returns:
        Atom-index pairs of shape ``[2, P]`` into the original ``N``-atom coordinates,
        shared across all particles. Each pair is within the cutoff in at least one
        particle and passes both masks.
    """
    atoms = atom_mask.nonzero().squeeze(-1)  # [M]
    coords = coords.detach()[:, atoms]  # [B, M, 3]
    if coords.dtype in (torch.float16, torch.bfloat16):
        coords = coords.float()
    chain = atom_chain_id[atoms]  # [M]
    pairs = [atoms.new_zeros((2, 0))]
    # The last atom has no later partner, so the blocks stop before it.
    for start in range(0, len(atoms) - 1, block_size):
        stop = min(start + block_size, len(atoms))
        with torch.autocast(device_type=coords.device.type, enabled=False):
            closest = torch.cdist(
                coords[:, start:stop],
                coords[:, start:],
                compute_mode="donot_use_mm_for_euclid_dist",
            ).amin(0)  # [R, M-start]
        should_check = allowed_chain_pairs[
            chain[start:stop, None], chain[None, start:]
        ]  # [R, M-start]
        row, col = torch.triu((closest < cutoff) & should_check, diagonal=1).nonzero(as_tuple=True)
        pairs.append(torch.stack([atoms[start + row], atoms[start + col]]))  # [2, P_block]
    return torch.cat(pairs, dim=1)  # [2, P]


class ParameterSchedule(ABC):
    def compute(self, t: float) -> float:
        raise NotImplementedError


class ExponentialInterpolation(ParameterSchedule):
    def __init__(self, start, end, alpha):
        self.start = start
        self.end = end
        self.alpha = alpha

    def compute(self, t: float) -> float:
        if self.alpha != 0:
            return self.start + (self.end - self.start) * (math.exp(self.alpha * t) - 1) / (
                math.exp(self.alpha) - 1
            )
        else:
            return self.start + (self.end - self.start) * t


class PiecewiseStepFunction(ParameterSchedule):
    def __init__(self, thresholds, values):
        self.thresholds = thresholds
        self.values = values

    def compute(self, t: float) -> float:
        assert len(self.thresholds) > 0
        assert len(self.values) == len(self.thresholds) + 1

        idx = 0
        while idx < len(self.thresholds) and t > self.thresholds[idx]:
            idx += 1
        return self.values[idx]


class LinearInterpolation(ParameterSchedule):
    def __init__(self, start_t, end_t, start_val, end_val):
        self.start_t = start_t
        self.end_t = end_t
        self.start_val = start_val
        self.end_val = end_val

    def compute(self, t: float) -> float:
        if (t > self.start_t) or (t < self.end_t):
            return 0.0
        else:
            # Linear interpolation between start and end values
            return self.start_val - (self.start_val - self.end_val) * (self.start_t - t) / (
                self.start_t - self.end_t
            )


# ``vdw_radii`` is indexed by atomic number 1..118.
num_elements = 128
vdw_radii = [
    1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54,
    2.4, 2.2, 2.1, 2.1, 1.95, 1.8, 1.8, 1.88, 2.8, 2.4,
    2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0, 2.0, 2.0, 2.1,
    2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3,
    2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25,
    2.2, 2.1, 2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47, 2.45,
    2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32, 2.3, 2.28,
    2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05,
    2.2, 2.3, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4,
    2.0, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
]  # fmt: skip


class Potential(ABC):
    """Public interface for an inference-time steering potential."""

    @abstractmethod
    def prepare(self, system: System) -> None:
        """Cache System-derived state (atom indices, features) once per sampling run."""

    def __call__(
        self,
        coords_angstrom: torch.Tensor,
        *,
        t: float,
    ) -> torch.Tensor:
        """Return unweighted per-sample energy at normalised diffusion time ``t``.

        The output must be differentiable with respect to ``coords_angstrom`` when used
        with gradient-based steering, such as FK gradient guidance or FKC's
        ``weight_update="guidance"`` mode.

        Args:
            coords_angstrom: Float32 or float64 coordinates in angstroms,
                shape ``[particles, atoms, 3]``.
                Padding may add trailing atoms; use the indices from ``prepare(system)``
                rather than assuming that every atom is real.
            t: Normalised diffusion time, from 1 at high noise to 0 at the clean endpoint.

        Returns:
            Energies of shape ``[particles]`` on the coordinates' device.
        """
        raise NotImplementedError

    def compute_parameters(self, t: float) -> dict[str, Any] | None:
        """Return parameters controlling whether/how strongly this potential is
        applied at diffusion time ``t`` (e.g. ``resampling_weight``,
        ``guidance_weight``, ``guidance_interval``)."""
        return None


class FeatsPotential(Potential):
    """Potential evaluated from a topology feature dictionary.

    Subclasses either split evaluation into ``_compute_args``, ``compute_variable``,
    and ``compute_function``, following the Boltz potential design, or override
    ``compute`` directly.
    """

    def __init__(
        self,
        parameters: dict[str, ParameterSchedule | float | int | bool] | None = None,
    ):
        self.parameters = parameters

    def prepare(self, system: System) -> None:
        """Build the topology feature cache used by subsequent evaluations."""
        self.feats = potential_features_from_system(system)

    def __call__(
        self,
        coords_angstrom: torch.Tensor,
        *,
        t: float,
    ) -> torch.Tensor:
        """Return differentiable, unweighted per-sample energy.

        ``t`` resolves parameter schedules at normalized diffusion time.
        """
        if coords_angstrom.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "Steering potentials require float32 or float64 coordinates; "
                f"got {coords_angstrom.dtype}."
            )
        parameters = self.compute_parameters(t) or {}
        self.feats = {name: value.to(coords_angstrom.device) for name, value in self.feats.items()}
        # Atom padding is trailing on both sides: truncate (CP pads wider than the
        # collate window) then pad (unpadded coords) to the feats width.
        n = self.feats["atom_pad_mask"].shape[1]
        coords = F.pad(coords_angstrom[:, :n], (0, 0, 0, max(0, n - coords_angstrom.shape[1])))
        return self.compute(coords, self.feats, parameters)

    def compute(
        self,
        coords: torch.Tensor,
        feats: dict[str, torch.Tensor],
        parameters: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Computes the value of the potential for the given coordinates.
        Args:
            coords: The coordinates of the sample atoms, shape (batch_size, n_atoms, 3).
            feats: The features of the input data.
            parameters: Additional parameters for the potential.
        Returns:
            A tensor containing the computed potential value, of shape (batch_size, ).
        """
        index, args, com_args = self._compute_args(feats, parameters)
        return self._compute_from_args(coords, index, args, com_args)

    def _compute_from_args(
        self,
        coords: torch.Tensor,
        index: torch.Tensor,
        args: tuple,
        com_args: tuple | None,
    ) -> torch.Tensor:
        if index.shape[1] == 0:
            return _connected_zero(coords)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = coords[..., atom_pad_mask, :]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
                dtype=coords.dtype,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
                include_self=False,  # true COM: divide by N atoms, not N + 1
            )
        value = self.compute_variable(coords, index)
        energy = self.compute_function(value, *args)
        return energy.sum(dim=-1)  # type: ignore[union-attr]

    def compute_parameters(self, t: float) -> dict[str, Any] | None:
        """
        Computes the parameters for applying potential at a given timestep,
        e.g. the weight of a guidance step.
        Args:
            t: The current diffusion timestep.
        Returns:
            A dictionary of parameters for the potential, or None if no parameters are set.
        """
        if self.parameters is None:
            return None
        parameters = {
            name: parameter
            if not isinstance(parameter, ParameterSchedule)
            else parameter.compute(t)
            for name, parameter in self.parameters.items()
        }
        return parameters

    def compute_function(self, value: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Computes the potential function given an input value, e.g. compute
        flat-bottom potentials given bond lengths.
        Args:
            value: The input value for the potential function, of shape (batch_size, n_values).
            *args: Additional arguments for the potential function.
        Returns:
            A tensor containing the computed potential, of shape (batch_size, n_values).
        """
        raise NotImplementedError

    def compute_variable(self, coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantity whose value will be used in the potential function,
        e.g. bond lengths or angles.
        Args:
            coords: The coordinates of the sample atoms, shape (batch_size, n_atoms, 3).
            index: The indices of the atoms involved in the potential.
        Returns:
            A tensor containing the computed variable, of shape (batch_size, n_values).
        """
        raise NotImplementedError

    def _compute_args(
        self, feats: dict, parameters: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, tuple, tuple | None]:
        """
        Computes the arguments required for the potential computation.
        Args:
            feats: The features of the input data.
            parameters: Additional parameters for the potential.
        Returns:
            A tuple containing:
                - index: The indices of the atoms involved in the potential.
                - args: A tuple of additional arguments required for the potential computation.
                - com_args: Optional arguments for center of mass calculations.
        """
        raise NotImplementedError


class FlatBottomPotential(FeatsPotential):
    """Linear hinge outside a lower/upper interval.

    For each measured value ``v``, the energy is
    ``k * (relu(lower - v) + relu(v - upper))``. A missing bound is treated as
    negative or positive infinity, respectively. This is linear, not quadratic.
    """

    def compute_function(self, value, k, lower_bounds, upper_bounds):
        if lower_bounds is None:
            lower_bounds = torch.full_like(value, float("-inf"))
        if upper_bounds is None:
            upper_bounds = torch.full_like(value, float("inf"))
        return k * (torch.relu(lower_bounds - value) + torch.relu(value - upper_bounds))


class DistancePotential(FeatsPotential):
    def compute_variable(self, coords, index):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        return torch.linalg.norm(r_ij, dim=-1)


class DihedralPotential(FeatsPotential):
    _DIHEDRAL_EPS = 1e-8

    def compute_variable(self, coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])

        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)

        # atan2 form instead of sign(.) * arccos(clamp(.)): clamping to 1 - 1e-8 is
        # a no-op in float32, so an exactly planar dihedral (phi in {0, pi}) gives
        # arccos'(1) = inf, times sign(0) = 0 -> NaN under autograd. atan2 is exact
        # there with finite gradients.
        bond_length = torch.linalg.norm(r_kj, dim=-1)
        valid = (
            (bond_length > self._DIHEDRAL_EPS)
            & (torch.linalg.norm(n_ijk, dim=-1) > self._DIHEDRAL_EPS)
            & (torch.linalg.norm(n_jkl, dim=-1) > self._DIHEDRAL_EPS)
        )
        sin_phi = (r_kj * torch.cross(n_ijk, n_jkl, dim=-1)).sum(-1) / bond_length.clamp_min(
            self._DIHEDRAL_EPS
        )
        cos_phi = (n_ijk * n_jkl).sum(-1)
        return torch.atan2(
            torch.where(valid, sin_phi, 0.0),
            torch.where(valid, cos_phi, 1.0),
        )


class AbsDihedralPotential(DihedralPotential):
    def compute_variable(self, coords, index):
        return torch.abs(super().compute_variable(coords, index))


class PoseBustersPotential(FlatBottomPotential, DistancePotential):
    def _compute_args(self, feats, parameters):
        pair_index = feats["rdkit_bounds_index"][0]
        lower_bounds = feats["rdkit_lower_bounds"][0].clone()
        upper_bounds = feats["rdkit_upper_bounds"][0].clone()
        bond_mask = feats["rdkit_bounds_bond_mask"][0]
        angle_mask = feats["rdkit_bounds_angle_mask"][0]

        lower_bounds[bond_mask * ~angle_mask] *= 1.0 - parameters["bond_buffer"]
        upper_bounds[bond_mask * ~angle_mask] *= 1.0 + parameters["bond_buffer"]
        lower_bounds[~bond_mask * angle_mask] *= 1.0 - parameters["angle_buffer"]
        upper_bounds[~bond_mask * angle_mask] *= 1.0 + parameters["angle_buffer"]
        lower_bounds[bond_mask * angle_mask] *= 1.0 - min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        upper_bounds[bond_mask * angle_mask] *= 1.0 + min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        lower_bounds[~bond_mask * ~angle_mask] *= 1.0 - parameters["clash_buffer"]
        upper_bounds[~bond_mask * ~angle_mask] = float("inf")

        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None


class InterChainVDWOverlapPotential(FeatsPotential):
    """Clash penalty on atom pairs in different chains.

    Pairs at most ``NONBONDED_EXCLUSION_GRAPH_DISTANCE`` covalent bonds apart, which only
    occur across declared cross-chain bonds such as inter-chain disulfides, are excluded.
    For an inter-chain disulfide, the SG-SG bond and its immediate neighbours (atom pairs
    up to ``NONBONDED_EXCLUSION_GRAPH_DISTANCE`` bonds apart through it) are dropped from
    the inter-chain clash potential. Across chains such a path must pass through a declared
    bond, so only pairs next to it are skipped. For an inter-chain disulfide these are
    SG–SG′, SG–CB′, CB–SG′, CB–CB′, SG–CA′ and CA–SG′; all other pairs between the two
    chains are scored. :class:`PolymerLocalClashPotential` scores the skipped pairs other
    than SG–SG′.
    The clash distance is the sum of the two VDW radii, reduced by ``buffer``. At every
    evaluation, score a pair only if it is within ``cutoff`` in any particle, so all
    atom pairs are never held at once. The cutoff must exceed every clash distance so
    that skipped pairs would have zero penalty and gradient.
    """

    cutoff: float = 10.0  # angstroms; must exceed all clash distances
    block_size: int = 512  # atoms per search block, limiting peak memory

    def prepare(self, system: System) -> None:
        """Cache atom radii, interchain eligibility and bonded cross-chain pairs for the system."""
        super().prepare(system)
        feats = self.feats
        n_atoms = feats["atom_pad_mask"].shape[1]  # N, padded atom count
        chain = _atom_chain_id(feats)  # [N]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()  # [N]
        chain_sizes = torch.bincount(chain[atom_pad_mask])  # [C]
        steric_atom = atom_pad_mask & (chain_sizes > 1)[chain]  # [N]
        atom_radius = _atom_vdw_radius(feats)  # [N]

        # [C, C]; False on the diagonal, so only pairs of different chains are scored.
        interchain = ~torch.eye(
            len(chain_sizes), dtype=torch.bool, device=chain.device
        )  # [C, C]:bool ; false on diagonal
        graph = build_bonded_geometry_graph(AllAtomStructure.from_system(system, mode="structure"))

        # graph.nonbonded_exclusions(): Return atom pairs (i, j) separated by at most ``max_bonds`` covalent bonds.
        # Compute bonds we exclude from inter-chain VDW scoring: those separated by at most NONBONDED_EXCLUSION_GRAPH_DISTANCE covalent bonds.
        excluded = (
            torch.tensor(
                sorted(graph.nonbonded_exclusions()), dtype=torch.long, device=chain.device
            )  # [E_all, 2]
            .reshape(-1, 2)  # [E_all, 2], also when there are no exclusions
            .T
        )  # [2, E_all]
        # Filter excluded bonds to only keep excluded cross-chain pairs
        excluded = excluded[:, chain[excluded[0]] != chain[excluded[1]]]  # [2, E], cross-chain

        # Largest VDW radius per chain. Summed over scored chain pairs, it bounds every clash distance,
        # so compute() can check that pairs skipped by the cutoff search could not clash.
        chain_radius = atom_radius.new_zeros(len(chain_sizes)).scatter_reduce(
            0, chain[atom_pad_mask], atom_radius[atom_pad_mask], "amax", include_self=False
        )  # [C]
        allowed = interchain & (chain_sizes > 1)[:, None] & (chain_sizes > 1)[None, :]  # [C, C]
        radius_sum = (chain_radius[:, None] + chain_radius[None, :])[allowed]  # [A] allowed pairs
        # Keep the bound on the host, with the same float32 arithmetic as clash distances.
        self._max_interchain_radius_sum: torch.Tensor | None = (
            radius_sum.max().cpu() if radius_sum.numel() else None
        )  # [] scalar, or None without allowed pairs
        # The interchain bonded pair id's are stored as single numbers, first * N + second, for quick lookup during compute().
        feats.update(
            atom_chain_id=chain.unsqueeze(0),  # [1, N]
            atom_vdw_radius=atom_radius.unsqueeze(0),  # [1, N]
            interchain_steric_atom_mask=steric_atom.unsqueeze(0),  # [1, N]
            interchain_pair_mask=interchain.unsqueeze(0),  # [1, C, C]
            interchain_bonded_pair_key=(excluded[0] * n_atoms + excluded[1]).unsqueeze(0),  # [1, E]
        )

    def compute(self, coords, feats, parameters=None):
        """Return ``E_b = sum_p [d_min,p - ||x_bi - x_bj||]_+``."""
        assert parameters is not None and "buffer" in parameters, (
            "InterChainVDWOverlapPotential requires parameters['buffer']."
        )
        # Require pair discovery to include every pair that can have positive energy.
        if self._max_interchain_radius_sum is not None:
            maximum = (self._max_interchain_radius_sum * (1.0 - parameters["buffer"])).item()
            if maximum >= self.cutoff:
                raise ValueError(
                    f"cutoff ({self.cutoff}) must exceed every clash distance (maximum {maximum})"
                )
        # Atom pairs from different chains that are closer than the cutoff in any particle.
        pairs = compute_atom_pairs_within_cutoff(  # [2, P_found]: first atoms, second atoms
            coords,
            atom_mask=feats["interchain_steric_atom_mask"][0],
            atom_chain_id=feats["atom_chain_id"][0],
            allowed_chain_pairs=feats["interchain_pair_mask"][0],
            cutoff=self.cutoff,
            block_size=self.block_size,
        )
        first, second = pairs

        # Skip pairs within three bonds of a declared cross-chain bond, such as the two sulfurs
        # of an inter-chain disulfide: the bond sets their distance, so it is not a clash.
        # prepare() stored these pairs as one number each, first * N + second.
        n_atoms = feats["atom_vdw_radius"].shape[1]  # N
        near_bond = torch.isin(first * n_atoms + second, feats["interchain_bonded_pair_key"][0])
        pairs = pairs[:, ~near_bond]  # [2, P]
        first, second = pairs

        # A pair clashes when it is closer than its summed VDW radii, shrunk by the buffer.
        radius = feats["atom_vdw_radius"][0]  # [N]
        clash_distance = (radius[first] + radius[second]) * (1.0 - parameters["buffer"])  # [P]
        return _steric_overlap_penalty(coords, pairs, clash_distance)  # [B]


class IntraChainVDWOverlapPotential(FeatsPotential):
    """Clash penalty on nonbonded protein and nucleic-acid atoms within one chain.

    Includes canonical protein/DNA/RNA atoms and unknown protein residue X atoms,
    excluding modified residues, unknown nucleotides and ligands. Nearby covalent
    neighbors are excluded; :class:`PolymerLocalClashPotential` covers clashes
    between such neighbors in different residues.

    Pass ``buffer`` through ``parameters`` to scale the summed VDW radii by
    ``1 - buffer``. The ``cutoff`` in angstroms must exceed every clash distance.
    """

    cutoff: float = 10.0  # angstroms; must exceed all clash distances
    block_size: int = 512  # atoms per search block, limiting peak memory

    def prepare(self, system: System) -> None:
        """Cache eligible atoms and same-chain pair filters for the system.

        Exclude pairs separated by at most ``NONBONDED_EXCLUSION_GRAPH_DISTANCE``
        covalent bonds, including explicit links. Atom eligibility and connectivity
        are defined by :func:`build_polymer_steric_graph`.
        """
        super().prepare(system)
        structure = AllAtomStructure.from_system(system, mode="structure")
        graph, eligible = build_polymer_steric_graph(structure)
        feats = self.feats
        n_atoms = feats["atom_pad_mask"].shape[1]
        chain = _atom_chain_id(feats)
        atom_radius = _atom_vdw_radius(feats)
        steric_atom = torch.zeros(n_atoms, dtype=torch.bool, device=chain.device)
        steric_atom[: structure.num_nodes] = eligible
        excluded = (
            torch.tensor(
                sorted(graph.nonbonded_exclusions()), dtype=torch.long, device=chain.device
            )
            .reshape(-1, 2)
            .T
        )  # [2, E]
        keys = excluded[0] * n_atoms + excluded[1]  # [E]
        feats.update(
            atom_chain_id=chain.unsqueeze(0),  # [1, N]
            atom_vdw_radius=atom_radius.unsqueeze(0),  # [1, N]
            polymer_steric_atom_mask=steric_atom.unsqueeze(0),  # [1, N]; True means eligible
            polymer_bonded_pair_key=keys.unsqueeze(0),  # [1, E]
            intrachain_pair_mask=torch.eye(
                int(chain.max()) + 1, dtype=torch.bool, device=chain.device
            ).unsqueeze(0),  # [1, C, C]; True permits same-chain pairs
        )

        eligible_radii = atom_radius[steric_atom]
        # Twice the largest eligible radius bounds all pair sums, even if no nonbonded
        # pairs remain. Retain this float32 bound on the host to avoid GPU synchronization.
        self._max_intrachain_radius_sum = (
            2 * eligible_radii.max().cpu() if eligible_radii.numel() else None
        )

    def compute(self, coords, feats, parameters=None):
        """Return the overlap penalty for eligible nonbonded same-chain pairs."""
        assert parameters is not None and "buffer" in parameters, (
            "IntraChainVDWOverlapPotential requires parameters['buffer']."
        )
        # Validate the buffer-scaled bound before searching; skip only if no atoms are eligible.
        if self._max_intrachain_radius_sum is not None:
            maximum = (self._max_intrachain_radius_sum * (1.0 - parameters["buffer"])).item()
            if maximum >= self.cutoff:
                raise ValueError(
                    f"cutoff ({self.cutoff}) must exceed every clash distance "
                    f"(conservative bound {maximum})"
                )
        steric_atom = feats["polymer_steric_atom_mask"][0]  # [N]
        chain = feats["atom_chain_id"][0]  # [N]
        index = compute_atom_pairs_within_cutoff(  # [2, P_cutoff]
            coords,
            atom_mask=steric_atom,
            atom_chain_id=chain,
            allowed_chain_pairs=feats["intrachain_pair_mask"][0],
            cutoff=self.cutoff,
            block_size=self.block_size,
        )
        i, j = index
        nonbonded = ~torch.isin(  # [P_cutoff]
            i * len(steric_atom) + j, feats["polymer_bonded_pair_key"][0]
        )
        index = index[:, nonbonded]  # [2, P]
        i, j = index
        radius = feats["atom_vdw_radius"][0]  # [N]
        clash_distance = (radius[i] + radius[j]) * (1.0 - parameters["buffer"])  # [P]
        return _steric_overlap_penalty(coords, index, clash_distance)


class PolymerLocalClashPotential(FeatsPotential):
    """Penalize short-range clashes between protein or nucleic-acid residues.

    Uses the same eligible atoms as :class:`IntraChainVDWOverlapPotential` and
    complements it and :class:`InterChainVDWOverlapPotential` by covering nearby
    covalent neighbors excluded from those terms, including across declared bonds.
    Pass the minimum allowed distance in angstroms through
    ``parameters={"minimum_distance": 2.0}``; it can also be a ``ParameterSchedule``.
    This distance is a lower bound, not a VDW radius or an equilibrium target.
    """

    def prepare(self, system: System) -> None:
        """Cache inter-residue pairs two or three covalent bonds apart."""
        super().prepare(system)
        structure = AllAtomStructure.from_system(system, mode="structure")
        graph, eligible = build_polymer_steric_graph(structure)
        # Starting from all pairs separated by at most three covalent bonds, remove direct
        # bonds and same-residue pairs. The result contains the inter-residue 1-3 and 1-4
        # pairs scored by the local-clash potential; cross-chain pairs arise only across
        # declared bonds, such as inter-chain disulfides.
        direct = set(graph.bonds)
        index = (
            torch.tensor(
                sorted(graph.nonbonded_exclusions() - direct),
                dtype=torch.long,
                device=structure.coords.device,
            )
            .reshape(-1, 2)
            .T
        )  # [2, P]
        i, j = index
        keep = (
            eligible[i] & eligible[j] & (structure.residue_idx[i] != structure.residue_idx[j])
        )  # [P]; True means eligible inter-residue endpoints
        self.feats["polymer_local_clash_index"] = index[:, keep].unsqueeze(0)  # [1, 2, P_local]

    def compute(self, coords, feats, parameters=None):
        """Return the overlap penalty using ``parameters["minimum_distance"]`` in angstroms."""
        assert parameters is not None and "minimum_distance" in parameters, (
            "PolymerLocalClashPotential requires parameters['minimum_distance']."
        )
        return _steric_overlap_penalty(
            coords, feats["polymer_local_clash_index"][0], parameters["minimum_distance"]
        )


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def _compute_args(self, feats, parameters):
        atom_chain_id = _atom_chain_id(feats)
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = chain_sizes > 1

        pair_index = feats["symmetric_chain_index"][0]
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]
        pair_index = pair_index[:, pair_ion_mask]
        lower_bounds = torch.full(
            (pair_index.shape[1],),
            parameters["buffer"],
            dtype=torch.float32,
            device=pair_index.device,
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return (
            pair_index,
            (k, lower_bounds, upper_bounds),
            (atom_chain_id, atom_pad_mask),
        )


class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def _compute_args(self, feats, parameters):
        stereo_bond_index = feats["stereo_bond_index"][0]
        stereo_bond_orientations = feats["stereo_bond_orientations"][0].bool()

        lower_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        upper_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        lower_bounds[stereo_bond_orientations] = torch.pi - parameters["buffer"]
        upper_bounds[stereo_bond_orientations] = float("inf")
        lower_bounds[~stereo_bond_orientations] = float("-inf")
        upper_bounds[~stereo_bond_orientations] = parameters["buffer"]

        k = torch.ones_like(lower_bounds)

        return stereo_bond_index, (k, lower_bounds, upper_bounds), None


class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def _compute_args(self, feats, parameters):
        chiral_atom_index = feats["chiral_atom_index"][0]
        chiral_atom_orientations = feats["chiral_atom_orientations"][0].bool()

        lower_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        upper_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        lower_bounds[chiral_atom_orientations] = parameters["buffer"]
        upper_bounds[chiral_atom_orientations] = float("inf")
        upper_bounds[~chiral_atom_orientations] = -1 * parameters["buffer"]
        lower_bounds[~chiral_atom_orientations] = float("-inf")

        k = torch.ones_like(lower_bounds)
        return chiral_atom_index, (k, lower_bounds, upper_bounds), None


class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def _compute_args(self, feats, parameters):
        double_bond_index = feats["planar_bond_index"][0].T
        double_bond_improper_index = torch.tensor(
            [
                [1, 2, 3, 0],
                [4, 5, 0, 3],
            ],
            device=double_bond_index.device,
        ).T
        improper_index = (
            double_bond_index[:, double_bond_improper_index].swapaxes(0, 1).flatten(start_dim=1)
        )
        lower_bounds = None
        upper_bounds = torch.full(
            (improper_index.shape[1],),
            parameters["buffer"],
            device=improper_index.device,
        )
        k = torch.ones_like(upper_bounds)

        return improper_index, (k, lower_bounds, upper_bounds), None


class ContactPotential(FlatBottomPotential, DistancePotential):
    """Flat-bottom upper on the distance between pairs of atoms.

    Reads feats (single-system topology, indexed at ``[0]``):

    - ``contact_pair_index``  ``[B, 2, K]`` int64 — atom pair indices
    - ``contact_thresholds``  ``[B, K]`` float32 — per-pair upper bound

    Supports independent atom pairs. It does not implement union semantics for
    multi-atom pockets.
    """

    def _compute_args(self, feats, parameters):
        pair_index = feats["contact_pair_index"][0]
        upper_bounds = feats["contact_thresholds"][0].clone()
        k = torch.ones_like(upper_bounds)
        return pair_index, (k, None, upper_bounds), None


class TemplateReferencePotential(FlatBottomPotential):
    """Flat-bottom upper on per-token distance between sampled coords and
    a rigid-aligned reference template.

    Reads feats (single-template, single-system topology; leading dim = 1):

    - ``template_cb``               ``[1, T, 3]`` float32 — reference Cβ coords
    - ``template_mask_cb``          ``[1, T]`` bool — valid ref atoms
    - ``template_force_threshold``  scalar-shaped float — flat-bottom upper
    - ``token_center_atom_index``   ``[1, T]`` int64 — rep-atom index per token
      using direct integer indices.

    Supports one reference template. Overrides :meth:`compute` directly because
    reference coordinates and masks do not fit the scalar-potential argument
    contract used by :meth:`_compute_args`.
    """

    def _compute_args(self, feats, parameters):
        raise NotImplementedError("TemplateReferencePotential overrides compute() directly")

    def compute_variable(self, coords, index):
        raise NotImplementedError("TemplateReferencePotential overrides compute() directly")

    def compute(
        self,
        coords: torch.Tensor,
        feats: dict[str, torch.Tensor],
        parameters: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        # coords: [B_particles, N_atoms_pad, 3]. Single-system topology → feats
        # tensors carry a leading len-1 dim that we drop with [0].
        ref_coords = feats["template_cb"][0].float()  # [T, 3]
        ref_mask = feats["template_mask_cb"][0].bool()  # [T]

        if not ref_mask.any():
            return _connected_zero(coords)

        # Rep-atom index per token (Cβ for protein tokens; the per-atom token for ligand).
        rep_atom_idx = feats["token_center_atom_index"][0]  # [T]

        # Gather sampled coords at each token's rep atom.
        sampled = coords[..., rep_atom_idx, :]  # [B_particles, T, 3]

        B = sampled.shape[0]
        ref_bc = ref_coords[None].expand(B, -1, -1).contiguous()  # [B, T, 3]
        mask_bc = ref_mask[None].float().expand(B, -1).contiguous()  # [B, T]

        # weighted_rigid_align requires float32 and autocast off (see edm_steered.py:351).
        with torch.autocast("cuda", enabled=False):
            aligned_ref, _, _ = weighted_rigid_align(
                ref_bc.float(), sampled.float(), mask_bc, mask_bc
            )
        aligned_ref = aligned_ref.to(sampled.dtype)

        # Per-token Euclidean distance between sample and rigid-aligned ref.
        d = torch.linalg.norm(sampled - aligned_ref, dim=-1)  # [B, T]

        # Flat-bottom upper only where the ref token is valid; elsewhere unbounded.
        threshold = feats["template_force_threshold"].float().reshape(-1)[0]
        upper = torch.where(
            ref_mask[None].expand_as(d),
            threshold,
            torch.full_like(d, float("inf")),
        )
        k = torch.ones_like(d)
        energy = self.compute_function(d, k, None, upper)
        return energy.sum(dim=-1)  # [B_particles]


class AnglePotential(FeatsPotential):
    """Compute unsigned bond angles in radians, between zero and pi.

    Index rows are the two outer atoms followed by their shared center.
    """

    _ANGLE_EPS = 1e-6

    def compute_variable(self, coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        first = coords.index_select(-2, index[0]) - coords.index_select(-2, index[2])  # [B, P, 3]
        second = coords.index_select(-2, index[1]) - coords.index_select(-2, index[2])  # [B, P, 3]
        # Avoid division by zero when an outer atom coincides with the center.
        first = first / torch.linalg.norm(first, dim=-1, keepdim=True).clamp_min(self._ANGLE_EPS)
        second = second / torch.linalg.norm(second, dim=-1, keepdim=True).clamp_min(self._ANGLE_EPS)
        # Keep acos gradients finite for nearly straight or folded-back bonds.
        cosine = (first * second).sum(dim=-1).clamp(-1.0 + self._ANGLE_EPS, 1.0 - self._ANGLE_EPS)
        return torch.acos(cosine)


class BondPotential(FlatBottomPotential):
    """Shared geometry preparation for the bond-length and bond-angle restraints.

    Args:
        sigma_tolerance: Required finite, nonnegative number of reference standard
            deviations used to expand the bounds. This is dimensionless, not a
            distance or angle, and must be fixed because bounds are prepared once.
        parameters: Steering weights and schedules, separate from the fixed tolerance.

    See :func:`bioemu2.denoising.steering.geometry_topology.get_bond_reference`
    and :func:`bioemu2.denoising.steering.geometry_topology.get_angle_reference`
    for reference selection, and
    :func:`bioemu2.denoising.steering.geometry_topology.compute_geometry_bounds`
    for tolerance scaling and clipping.
    """

    def __init__(
        self,
        *,
        sigma_tolerance: float,
        parameters: dict[str, ParameterSchedule | float | int | bool] | None = None,
    ) -> None:
        super().__init__(parameters=parameters)
        if not math.isfinite(sigma_tolerance) or sigma_tolerance < 0:
            raise ValueError("sigma_tolerance must be finite and nonnegative.")
        self.sigma_tolerance = sigma_tolerance

    def _prepare_geometry(
        self, system: System
    ) -> tuple[AllAtomStructure, BondedGeometryGraph, StereoTables]:
        structure = AllAtomStructure.from_system(system, mode="structure")
        batch = batch_collate([structure])
        self.feats = convert_densebatch_to_potential_features(batch)
        bond_graph = build_bonded_geometry_graph(structure)
        tables = load_stereo_tables()
        return structure, bond_graph, tables


class BondLengthPotential(BondPotential, DistancePotential):
    """Penalize bond lengths outside reference or heuristic intervals.

    Reference values, stored bounds, and linear length violations are in angstroms.
    """

    def prepare(self, system: System) -> None:
        """Build bond topology, resolve reference values, and store indices and lower/upper bounds."""
        structure, bond_graph, tables = self._prepare_geometry(system)
        constraints = []
        pooled: dict[tuple[tuple[str, ...], bool], tuple[float, float, float] | None] = {}
        for i, j in bond_graph.bonds:
            reference = get_bond_reference(bond_graph, i, j, tables, pooled)
            bounds = compute_geometry_bounds(reference, self.sigma_tolerance)
            constraints.append((i, j, bounds))
        index, lower, upper = pack_bond_constraints(constraints, structure.coords.device)
        # One system: indices [1, 2, P] and lower/upper bounds [1, P], with P bonds.
        self.feats.update(bond_index=index, bond_lower=lower, bond_upper=upper)

    def _compute_args(
        self, feats: dict[str, torch.Tensor], parameters: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor], None]:
        """Return prepared bond indices, unit penalty slopes, and bounds for flat-bottom evaluation."""
        index = feats["bond_index"][0]
        lower, upper = feats["bond_lower"][0], feats["bond_upper"][0]
        return index, (torch.ones_like(lower), lower, upper), None


class BondAnglePotential(BondPotential, AnglePotential):
    """Penalize bond angles outside reference or heuristic intervals.

    Reference values are in degrees; stored bounds and linear angular violations
    are in radians.
    """

    def prepare(self, system: System) -> None:
        """Build angle topology, resolve reference values, and store indices and radian bounds."""
        structure, bond_graph, tables = self._prepare_geometry(system)
        constraints = []
        pooled: dict[tuple[tuple[str, ...], bool], tuple[float, float, float] | None] = {}
        for i, k, center in bond_graph.angles:
            reference = get_angle_reference(bond_graph, i, center, k, tables, pooled)
            bounds = compute_geometry_bounds(reference, self.sigma_tolerance, angle=True)
            constraints.append((i, k, center, bounds))
        index, lower, upper = pack_angle_constraints(constraints, structure.coords.device)
        # One system: indices [1, 3, P] and lower/upper bounds [1, P], with P angles.
        self.feats.update(angle_index=index, angle_lower=lower, angle_upper=upper)

    def _compute_args(
        self, feats: dict[str, torch.Tensor], parameters: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor], None]:
        """Return prepared angle indices, unit penalty slopes, and bounds for flat-bottom evaluation."""
        index = feats["angle_index"][0]
        lower, upper = feats["angle_lower"][0], feats["angle_upper"][0]
        return index, (torch.ones_like(lower), lower, upper), None


def get_default_potentials() -> Sequence[Potential]:
    return (
        SymmetricChainCOMPotential(
            parameters={
                "guidance_interval": 4,
                "guidance_weight": 0.5,
                "resampling_weight": 0.5,
                "buffer": ExponentialInterpolation(start=1.0, end=5.0, alpha=-2.0),
            }
        ),
        InterChainVDWOverlapPotential(
            parameters={
                "guidance_interval": 5,
                "guidance_weight": PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0]),
                "resampling_weight": PiecewiseStepFunction(thresholds=[0.6], values=[0.01, 0.0]),
                "buffer": 0.225,
            }
        ),
        PoseBustersPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 0.10,
                "resampling_weight": 0.1,
                "bond_buffer": 0.20,
                "angle_buffer": 0.20,
                "clash_buffer": 0.15,
            }
        ),
        ChiralAtomPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 0.20,
                "resampling_weight": 1.0,
                "buffer": 0.52360,
            }
        ),
        StereoBondPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 0.10,
                "resampling_weight": 1.0,
                "buffer": 0.52360,
            }
        ),
        PlanarBondPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 0.10,
                "resampling_weight": 1.0,
                "buffer": 0.26180,
            }
        ),
        IntraChainVDWOverlapPotential(
            parameters={
                "guidance_interval": 5,
                "guidance_weight": PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0]),
                "resampling_weight": PiecewiseStepFunction(thresholds=[0.6], values=[0.01, 0.0]),
                "buffer": 0.225,
            }
        ),
        PolymerLocalClashPotential(
            parameters={
                "guidance_interval": 1,
                "guidance_weight": PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0]),
                "resampling_weight": 0.1,
                "minimum_distance": 2.0,
            }
        ),
        BondLengthPotential(
            sigma_tolerance=10.0,
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 0.05,
                "resampling_weight": 0.1,
            },
        ),
        BondAnglePotential(
            sigma_tolerance=10.0,
            parameters={
                "guidance_interval": 1,
                "guidance_weight": 0.05,
                "resampling_weight": 0.1,
            },
        ),
    )
