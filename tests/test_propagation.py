from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import mdtraj
import numpy as np
import pytest
import torch
from huggingface_hub import get_token

import bioemu2.propagation as inference_api
from bioemu2 import (
    MSA,
    EDMSampler,
    FKCSampler,
    GuidedFKSampler,
    join_trajectory_steps,
    load_model,
    load_system_and_contacts,
    propagate_with_embeddings,
    sample_with_embeddings,
    simulate_trajectory,
    simulate_trajectory_with_embeddings,
)
from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.data_interface import Atom, Bond, Protein, System
from bioemu2.denoising.steering.edm_steered import _slice_conditioning
from bioemu2.paths import EXAMPLES_DIR
from bioemu2.physicality import PhysicalityFilterMode, check_allatom_physicality
from bioemu2.sampling import rng_seed
from bioemu2.structure import BatchAllAtomStructure


@dataclass(frozen=True)
class _StructureContext:
    coords_cond: torch.Tensor
    is_dynamics_conditioned: torch.Tensor
    dt: torch.Tensor
    atom_pad_mask: torch.Tensor


def _three_atom_topology() -> mdtraj.Topology:
    topology = mdtraj.Topology()
    residue = topology.add_residue("ALA", topology.add_chain())
    for name in ("N", "CA", "CB"):
        topology.add_atom(name, mdtraj.element.carbon, residue)
    return topology


def _fake_inputs(
    topology: mdtraj.Topology,
) -> tuple[Any, TrunkEmbeddings, BatchAllAtomStructure]:
    batch = BatchAllAtomStructure(
        coords_cond=torch.zeros(1, 4, 3),
        is_dynamics_conditioned=torch.zeros(1, 1, dtype=torch.bool),
        dt=torch.zeros(1, 1),
        atom_pad_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    system: Any = SimpleNamespace(topology=topology, bonds=[])
    trunk = cast(TrunkEmbeddings, SimpleNamespace())
    return system, trunk, batch


def test_propagate_conditions_each_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    topology = _three_atom_topology()
    system, trunk_embeddings, batch = _fake_inputs(topology)
    model: Any = SimpleNamespace(device=torch.device("cpu"), supports_propagation=True)
    calls: list[dict[str, Any]] = []

    def sample(**kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        coords_cond = kwargs["batch"].coords_cond
        assert coords_cond is not None
        # Scale by 2 and rotate 90 degrees about z; superposition must undo the rotation.
        rotated = (
            2
            * coords_cond[:, :3]
            @ torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        )
        return rotated

    monkeypatch.setattr(inference_api, "generate_samples", sample)
    monkeypatch.setattr(
        inference_api, "prepare_sampling_inputs", lambda *_args, **_kwargs: (topology, batch)
    )
    x0 = mdtraj.Trajectory(
        xyz=np.array(
            [
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
                [[1.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.0, 0.1, 0.0]],
                [[0.0, 1.0, 0.0], [0.3, 1.0, 0.0], [0.0, 1.4, 0.0]],
            ],
            dtype=np.float32,
        ),
        topology=topology,
    )

    result = propagate_with_embeddings(
        system,
        trunk_embeddings,
        x0=x0,
        lag_time_ns=1.0,
        model=model,
        batch_size=2,
        sampler=EDMSampler(step_scale=1.5),
        seed=7,
    )

    # One call; generate_samples batches it. Each frame has its own centered, padded condition.
    [call] = calls
    assert (call["num_samples"], call["batch_size"], call["seed"]) == (3, 2, 7)
    centered = 10 * (x0.xyz - x0.xyz.mean(axis=1, keepdims=True))
    call_batch = call["batch"]
    assert call_batch.coords_cond.shape == (3, 4, 3)
    torch.testing.assert_close(call_batch.coords_cond[:, :3], torch.as_tensor(centered))
    assert call_batch.coords_cond[:, 3].abs().sum() == 0
    assert torch.equal(call_batch.is_dynamics_conditioned, torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(call_batch.dt, torch.full((1, 3), 1000.0))
    # The caller's sampling inputs are untouched.
    original = batch.coords_cond
    assert original is not None and original.shape == (1, 4, 3)
    # One endpoint per frame, superposed on its own input frame: rotation undone,
    # centroid restored, scale kept.
    assert result.n_frames == 3
    expected = x0.xyz.mean(axis=1, keepdims=True) + 2 * (
        x0.xyz - x0.xyz.mean(axis=1, keepdims=True)
    )
    np.testing.assert_allclose(result.xyz, expected, atol=1e-5)


def test_propagate_preserves_independent_trajectories_across_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct first-step shapes remain each slot's condition in the next step.

    The fake sampler scales each of four identical starts by a different factor,
    then doubles every resulting shape on the next step. Each reconstructed
    trajectory must retain its own scale across both steps.
    """
    topology = _three_atom_topology()
    system, trunk_embeddings, batch = _fake_inputs(topology)
    model: Any = SimpleNamespace(supports_propagation=True, device=torch.device("cpu"))
    initial = mdtraj.Trajectory(
        xyz=np.array([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]], dtype=np.float32),
        topology=topology,
    )
    received: list[torch.Tensor] = []
    step = 0

    def sample(**kwargs: Any) -> torch.Tensor:
        coords_cond = kwargs["batch"].coords_cond
        assert coords_cond is not None
        received.append(coords_cond.clone())
        if step == 0:
            # Four different endpoint shapes; the second propagation must use
            # each shape rather than restarting from initial.
            return coords_cond[:, :3] * torch.tensor([1.25, 1.5, 1.75, 2.0])[:, None, None]
        return coords_cond[:, :3] * 2

    monkeypatch.setattr(inference_api, "generate_samples", sample)
    monkeypatch.setattr(
        inference_api, "prepare_sampling_inputs", lambda *_args, **_kwargs: (topology, batch)
    )
    frames = [initial[[0] * 4]]
    for step in range(2):
        frames.append(
            propagate_with_embeddings(
                system,
                trunk_embeddings,
                x0=frames[-1],
                lag_time_ns=1.0,
                model=model,
                batch_size=2,
                sampler=EDMSampler(),
                seed=step,
            )
        )

    assert [len(conditions) for conditions in received] == [4, 4]
    expected_conditions = 10 * (frames[1].xyz - frames[1].xyz.mean(axis=1, keepdims=True))
    torch.testing.assert_close(received[1][:, :3], torch.as_tensor(expected_conditions))
    for i, scale in enumerate([1.25, 1.5, 1.75, 2.0]):
        trajectory = mdtraj.join([step_frames[i] for step_frames in frames])
        assert trajectory.n_frames == 3
        np.testing.assert_allclose(
            trajectory.xyz[:, 1, 0] - trajectory.xyz[:, 0, 0],
            [0.1, 0.1 * scale, 0.2 * scale],
            atol=1e-5,
        )


@pytest.mark.parametrize("step_scale", [1.0, 1.5])
def test_simulate_trajectory_returns_every_frame_and_reports_physicality(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, step_scale: float
) -> None:
    inference_api._warn_custom_step_scale.cache_clear()
    topology = _three_atom_topology()
    system, trunk_embeddings, batch = _fake_inputs(topology)
    embed_calls = 0

    def embed(*_args: Any, **_kwargs: Any) -> TrunkEmbeddings:
        nonlocal embed_calls
        embed_calls += 1
        return trunk_embeddings

    model: Any = SimpleNamespace(
        supports_propagation=True,
        device=torch.device("cpu"),
        embed=embed,
    )
    x0 = mdtraj.Trajectory(
        xyz=np.array(
            [
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0]],
            ],
            dtype=np.float32,
        ),
        topology=topology,
    )
    sample_scales: list[float] = []

    def sample(**kwargs: Any) -> torch.Tensor:
        sample_scales.append(kwargs["sampler"].step_scale)
        coords_cond = kwargs["batch"].coords_cond
        assert coords_cond is not None
        return 2 * coords_cond[:, :3]

    monkeypatch.setattr(inference_api, "generate_samples", sample)
    monkeypatch.setattr(
        inference_api, "prepare_sampling_inputs", lambda *_args, **_kwargs: (topology, batch)
    )
    monkeypatch.setattr(
        inference_api,
        "check_allatom_physicality",
        lambda trajectory, **_kwargs: (
            np.zeros(len(trajectory), dtype=bool),
            {"bad_bonds_backbone": np.ones(len(trajectory), dtype=np.int64)},
            ["bad_bonds_backbone"] * len(trajectory),
        ),
    )

    embeddings = model.embed(system, msa=cast(MSA, SimpleNamespace()), mode="equilibrium")
    trajectories = simulate_trajectory_with_embeddings(
        system,
        embeddings,
        x0=x0,
        lag_time_ns=1.0,
        num_trajectory_steps=2,
        model=model,
        batch_size=2,
        sampler=EDMSampler(step_scale=step_scale),
        seed=7,
        display_propagation_progress=False,
    )

    assert embed_calls == 1
    assert len(trajectories) == 2
    for start, trajectory in zip(x0, trajectories, strict=True):
        assert trajectory.n_frames == 3
        np.testing.assert_allclose(trajectory.time, [0.0, 1000.0, 2000.0])
        np.testing.assert_allclose(
            trajectory.xyz - trajectory.xyz.mean(axis=1, keepdims=True),
            np.array([1.0, 2.0, 4.0])[:, None, None]
            * (start.xyz - start.xyz.mean(axis=1, keepdims=True)),
            atol=1e-5,
        )
    assert "0/6 frames passed" in caplog.text
    assert "All frames are retained" in caplog.text
    assert sample_scales == [step_scale] * 2
    if step_scale != 1.0:
        assert caplog.text.count("This warning will not be shown again") == 1


def test_trajectory_physicality_report_expects_declared_bonds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the trajectory report passes declared bonds, so a disulfide is no mismatch."""
    sulfurs = [Atom(asym_id="A", res_id=r, res_name="CYS", name="SG") for r in (1, 2)]
    system = System(
        entities=[Protein(sequence="CC", asym_ids="A")],
        bonds=[Bond(atom1=sulfurs[0], atom2=sulfurs[1])],
    )
    trajectory = mdtraj.Trajectory(np.zeros((1, system.topology.n_atoms, 3)), system.topology)
    calls = []

    def check(samples: mdtraj.Trajectory, **kwargs: Any) -> tuple:
        calls.append(kwargs["expected_extra_bonds"])
        return np.ones(len(samples), dtype=bool), {}, [None] * len(samples)

    monkeypatch.setattr(inference_api, "check_allatom_physicality", check)
    inference_api._report_trajectory_physicality([trajectory], system)
    sg = sorted(a.index for a in system.topology.atoms if a.name == "SG")
    assert calls == [{tuple(sg)}]


def test_simulate_trajectory_reuses_embedding_with_loaded_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = _three_atom_topology()
    system, trunk_embeddings, batch = _fake_inputs(topology)
    msa = cast(MSA, SimpleNamespace())
    embed_calls = 0

    def embed(*_args: Any, **_kwargs: Any) -> TrunkEmbeddings:
        nonlocal embed_calls
        embed_calls += 1
        return trunk_embeddings

    model: Any = SimpleNamespace(supports_propagation=True, device=torch.device("cpu"), embed=embed)
    x0 = mdtraj.Trajectory(
        xyz=np.array([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]]),
        topology=topology,
    )
    monkeypatch.setattr(inference_api, "get_msa", lambda _system: msa)
    monkeypatch.setattr(
        inference_api, "prepare_sampling_inputs", lambda *_args, **_kwargs: (topology, batch)
    )
    monkeypatch.setattr(inference_api, "_report_trajectory_physicality", lambda *_: None)
    scales: list[float] = []

    def sample(**kwargs: Any) -> torch.Tensor:
        scales.append(kwargs["sampler"].step_scale)
        return kwargs["batch"].coords_cond[:, :3]

    monkeypatch.setattr(inference_api, "generate_samples", sample)
    trajectories = simulate_trajectory(
        system,
        model=model,
        x0=x0,
        lag_time_ns=1.0,
        num_trajectory_steps=2,
        sampler="edm_gd",
        display_propagation_progress=False,
    )

    assert len(trajectories) == 1
    assert trajectories[0].n_frames == 3
    assert embed_calls == 1
    assert scales == [1.0, 1.0]


def test_simulate_trajectory_rejects_empty_x0() -> None:
    model: Any = SimpleNamespace(supports_propagation=True)
    x0 = mdtraj.Trajectory(np.zeros((0, 3, 3), dtype=np.float32), _three_atom_topology())
    with pytest.raises(ValueError, match="x0 must be a non-empty"):
        simulate_trajectory_with_embeddings(
            cast(System, SimpleNamespace()),
            cast(TrunkEmbeddings, SimpleNamespace()),
            x0=x0,
            lag_time_ns=1.0,
            num_trajectory_steps=1,
            model=model,
            sampler=EDMSampler(),
            display_propagation_progress=False,
        )


@pytest.mark.parametrize("preset", ["edm_fk", "edm_fkc"])
def test_simulate_trajectory_rejects_resampling_presets_before_msa(
    monkeypatch: pytest.MonkeyPatch, preset: Any
) -> None:
    monkeypatch.setattr(inference_api, "get_msa", lambda _system: pytest.fail("queried MSA"))
    model: Any = SimpleNamespace()
    with pytest.raises(ValueError, match="does not support samplers with resampling enabled"):
        simulate_trajectory(
            cast(System, SimpleNamespace()),
            model=model,
            x0=cast(mdtraj.Trajectory, SimpleNamespace()),
            lag_time_ns=1.0,
            num_trajectory_steps=2,
            sampler=preset,
        )


def test_join_trajectory_steps_sets_times_and_preserves_trajectory_order() -> None:
    topology = _three_atom_topology()
    start = mdtraj.Trajectory(
        xyz=np.array(
            [
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
                [[1.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.0, 0.3, 0.0]],
            ],
            dtype=np.float32,
        ),
        topology=topology,
    )
    endpoint = mdtraj.Trajectory(xyz=start.xyz + 0.5, topology=topology)

    trajectories = join_trajectory_steps([start, endpoint], lag_time_ns=1.0)

    assert len(trajectories) == 2
    for i, trajectory in enumerate(trajectories):
        np.testing.assert_allclose(trajectory.xyz, [start.xyz[i], endpoint.xyz[i]])
        np.testing.assert_allclose(trajectory.time, [0.0, 1000.0])

    with pytest.raises(ValueError, match=r"must contain 2 frames; observed counts: \{2: 1, 1: 1\}"):
        join_trajectory_steps([start, endpoint[:1]], lag_time_ns=1.0)


def test_simulate_trajectory_seeds_rise_linearly_and_restore_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = _three_atom_topology()
    system, trunk_embeddings, batch = _fake_inputs(topology)
    model: Any = SimpleNamespace(
        supports_propagation=True,
        device=torch.device("cpu"),
        embed=lambda *_args, **_kwargs: trunk_embeddings,
    )
    x0 = mdtraj.Trajectory(
        xyz=np.array(
            [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]],
            dtype=np.float32,
        ),
        topology=topology,
    )
    monkeypatch.setattr(
        inference_api, "prepare_sampling_inputs", lambda *_args, **_kwargs: (topology, batch)
    )
    monkeypatch.setattr(inference_api, "_report_trajectory_physicality", lambda *_: None)
    draws: list[torch.Tensor] = []

    class NoisySampler(EDMSampler):
        """Adds Gaussian noise to each conditioning frame through the real generate_samples."""

        def __call__(self, *, batch: BatchAllAtomStructure, **_kwargs: Any) -> Any:
            assert batch.coords_cond is not None
            draws.append(torch.randn_like(batch.coords_cond))
            return SimpleNamespace(coords=batch.coords_cond + draws[-1])

    def run(seed: int) -> np.ndarray:
        torch.manual_seed(1234)
        rng_state = torch.random.get_rng_state()
        trajectories = simulate_trajectory_with_embeddings(
            system,
            trunk_embeddings,
            x0=x0[[0, 0, 0]],
            lag_time_ns=1.0,
            num_trajectory_steps=2,
            model=model,
            batch_size=2,
            sampler=NoisySampler(),
            seed=seed,
            display_propagation_progress=False,
        )
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        return np.stack([trajectory.xyz for trajectory in trajectories])

    first = run(7)
    # Three frames in batches of two: batch j of step t uses 7 + 2 * t + j.
    expected = []
    for seed, size in zip((7, 8, 9, 10), (2, 1, 2, 1), strict=True):
        with rng_seed(seed):
            expected.append(torch.randn(size, 4, 3))
    assert len(draws) == len(expected)
    for draw, reference in zip(draws, expected, strict=True):
        torch.testing.assert_close(draw, reference, rtol=0, atol=0)
    np.testing.assert_array_equal(run(7), first)
    assert not np.allclose(run(8), first)


@pytest.mark.parametrize(
    "sampler",
    [
        GuidedFKSampler(),
        GuidedFKSampler(resampling_interval=1, resampling_group_size=1),
        GuidedFKSampler(resampling_interval=1, resampling_group_size=2),
        FKCSampler(),
    ],
)
def test_simulate_trajectory_rejects_resampling(
    sampler: GuidedFKSampler | FKCSampler,
) -> None:
    model: Any = SimpleNamespace(supports_propagation=True)
    with pytest.raises(ValueError, match="does not support samplers with resampling enabled"):
        simulate_trajectory_with_embeddings(
            cast(System, SimpleNamespace()),
            cast(TrunkEmbeddings, SimpleNamespace()),
            x0=cast(mdtraj.Trajectory, SimpleNamespace()),
            lag_time_ns=1.0,
            num_trajectory_steps=2,
            model=model,
            sampler=sampler,
            display_propagation_progress=False,
        )


@pytest.mark.parametrize(
    ("blocks", "group_size", "batch_size", "error"),
    [
        ("aabb", 2, None, None),
        ("aabbc", 2, None, None),
        ("abab", 2, None, "frames 0:2 differ"),
        ("aabb", 2, 3, "multiple of"),
        ("aabbc", 2, 4, None),
        ("aaaa", None, 2, None),
        ("aabb", None, 2, "frames 0:4 differ"),
        ("aabb", None, None, "frames 0:4 differ"),
    ],
)
def test_propagate_resampling_needs_identical_blocks(
    monkeypatch: pytest.MonkeyPatch,
    blocks: str,
    group_size: int | None,
    batch_size: int | None,
    error: str | None,
) -> None:
    topology = _three_atom_topology()
    system, trunk_embeddings, batch = _fake_inputs(topology)
    model: Any = SimpleNamespace(device=torch.device("cpu"), supports_propagation=True)

    def sample(**kwargs: Any) -> torch.Tensor:
        coords_cond = kwargs["batch"].coords_cond
        return coords_cond[:, :3]

    monkeypatch.setattr(inference_api, "generate_samples", sample)
    monkeypatch.setattr(
        inference_api, "prepare_sampling_inputs", lambda *_args, **_kwargs: (topology, batch)
    )
    frame = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32)
    xyz = np.stack([frame + 0.5 * "abc".index(c) for c in blocks])
    x0 = mdtraj.Trajectory(xyz=xyz, topology=topology)
    kwargs: dict[str, Any] = dict(
        x0=x0,
        lag_time_ns=1.0,
        model=model,
        batch_size=batch_size,
        sampler=GuidedFKSampler(resampling_interval=1, resampling_group_size=group_size),
    )

    if error is None:
        assert propagate_with_embeddings(system, trunk_embeddings, **kwargs).n_frames == len(blocks)
    else:
        with pytest.raises(ValueError, match=error):
            propagate_with_embeddings(system, trunk_embeddings, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"batch_size": 0}, "batch_size must be a positive integer"),
        ({"batch_size": -1}, "batch_size must be a positive integer"),
        ({"lag_time_ns": 1e-300}, "underflows"),
    ],
)
def test_propagate_validates_arguments(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], error: str
) -> None:
    topology = _three_atom_topology()
    system, trunk_embeddings, batch = _fake_inputs(topology)
    model: Any = SimpleNamespace(device=torch.device("cpu"), supports_propagation=True)
    monkeypatch.setattr(inference_api, "generate_samples", lambda **_kwargs: pytest.fail("sampled"))
    monkeypatch.setattr(
        inference_api, "prepare_sampling_inputs", lambda *_args, **_kwargs: (topology, batch)
    )
    xyz = np.array([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]] * 2, dtype=np.float32)
    x0 = mdtraj.Trajectory(xyz=xyz, topology=topology)

    with pytest.raises(ValueError, match=error):
        propagate_with_embeddings(
            system,
            trunk_embeddings,
            **{"x0": x0, "lag_time_ns": 1.0, "model": model, "sampler": EDMSampler(), **kwargs},
        )


@pytest.mark.parametrize("reuse_embeddings", [False, True])
def test_propagate_rejects_unsupported_checkpoint(
    monkeypatch: pytest.MonkeyPatch, reuse_embeddings: bool
) -> None:
    topology = _three_atom_topology()
    system, trunk_embeddings, _ = _fake_inputs(topology)
    x0 = mdtraj.Trajectory(xyz=np.zeros((1, 3, 3), dtype=np.float32), topology=topology)
    model: Any = SimpleNamespace(
        supports_propagation=False, embed=lambda *_args, **_kwargs: pytest.fail("embedded")
    )
    monkeypatch.setattr(
        inference_api, "prepare_sampling_inputs", lambda *_args, **_kwargs: pytest.fail("prepared")
    )

    with pytest.raises(ValueError, match="checkpoint does not support propagation"):
        if reuse_embeddings:
            propagate_with_embeddings(
                system,
                trunk_embeddings,
                x0=x0,
                lag_time_ns=1.0,
                model=model,
                sampler=EDMSampler(),
            )
        else:
            propagate_with_embeddings(
                system,
                trunk_embeddings,
                x0=x0,
                lag_time_ns=1.0,
                model=model,
                sampler=EDMSampler(),
            )


def test_slice_conditioning_follows_micro_batches() -> None:
    # Micro-batched denoising must slice propagate's per-sample conditioning alongside coords.
    per_sample: Any = _StructureContext(
        coords_cond=torch.arange(4.0).view(4, 1, 1).expand(4, 2, 3),
        is_dynamics_conditioned=torch.ones(1, 4, dtype=torch.bool),
        dt=torch.arange(4.0).view(1, 4),
        atom_pad_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    sliced: Any = _slice_conditioning(per_sample, 1, 3)
    torch.testing.assert_close(sliced.coords_cond, per_sample.coords_cond[1:3])
    assert sliced.is_dynamics_conditioned.shape == (1, 2)
    torch.testing.assert_close(sliced.dt, torch.tensor([[1.0, 2.0]]))

    shared: Any = _StructureContext(
        coords_cond=torch.zeros(1, 2, 3),
        is_dynamics_conditioned=torch.zeros(1, 1, dtype=torch.bool),
        dt=torch.zeros(1, 1),
        atom_pad_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert _slice_conditioning(shared, 1, 3) is shared


def test_rng_seed_forks_all_cuda_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    forked_devices: list[list[int]] = []
    seeded: list[int] = []

    @contextmanager
    def fork_rng(*, devices: list[int]) -> Generator[None, None, None]:
        forked_devices.append(devices)
        yield

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.random, "fork_rng", fork_rng)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: seeded.append(seed))

    with rng_seed(7):
        pass

    assert forked_devices == [[0, 1, 2]]
    assert seeded == [7]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.serial
def test_chignolin_inference_api_integration(gpu_device: torch.device) -> None:
    """Run sampling and propagation end to end for Chignolin."""
    if get_token() is None:
        pytest.skip("Requires Hugging Face authentication for microsoft/bioemu2")

    system, _ = load_system_and_contacts(EXAMPLES_DIR / "systems/chignolin.yaml")
    msa = MSA({"A": ">query\nGYDPETGTWG\n"})
    model = load_model(
        "bioemu2-preview",
        device=gpu_device,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed",
        matmul_precision="high" if torch.cuda.is_bf16_supported() else "highest",
    )

    structure_embeddings = model.embed(system, msa=msa, mode="structure")
    equilibrium_embeddings = model.embed(system, msa=msa, mode="equilibrium")
    structures = sample_with_embeddings(
        system,
        structure_embeddings,
        model=model,
        num_samples=2,
        batch_size=2,
        sampler=EDMSampler(num_steps=20),
        seed=0,
        progress=False,
    )

    equilibrium = sample_with_embeddings(
        system,
        equilibrium_embeddings,
        model=model,
        num_samples=2,
        batch_size=2,
        sampler=EDMSampler(num_steps=20),
        seed=1,
        progress=False,
    )
    propagated = propagate_with_embeddings(
        system,
        equilibrium_embeddings,
        x0=equilibrium,
        lag_time_ns=1.0,
        model=model,
        batch_size=2,
        sampler=EDMSampler(num_steps=20),
        seed=2,
        progress=False,
    )

    for name, trajectory in {
        "structures": structures,
        "equilibrium": equilibrium,
    }.items():
        assert trajectory.n_frames == 2
        assert trajectory.topology == system.topology
        assert np.isfinite(trajectory.xyz).all()
        is_physical, _stats, reasons = check_allatom_physicality(
            trajectory, mode=PhysicalityFilterMode.HIGH_QUALITY_PER_RESIDUE
        )
        assert is_physical.all(), f"{name} failed physicality checks: {reasons}"

    assert propagated.n_frames == equilibrium.n_frames
    assert propagated.topology == system.topology
    assert np.isfinite(propagated.xyz).all()
    is_physical, _stats, reasons = check_allatom_physicality(
        propagated, mode=PhysicalityFilterMode.HIGH_QUALITY_PER_RESIDUE
    )
    assert is_physical.all(), f"propagated failed physicality checks: {reasons}"
