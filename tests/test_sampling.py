import errno
import json
import logging
import weakref
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import mdtraj
import numpy as np
import pytest
import torch
from biotite.structure.io import pdbx
from click.testing import CliRunner
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.optim.swa_utils import AveragedModel

import bioemu2.inference as inference_api
import bioemu2.propagation as propagation_api
import bioemu2.sampling as sampling_api
from bioemu2 import (
    EmuSession,
    check_physicality,
    cli,
    load_model,
    paths,
    predict_structure,
    propagate,
    propagate_with_embeddings,
    sample_equilibrium,
    sample_with_embeddings,
    save_frames,
    simulate_trajectory,
)
from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.biomoa.modules.diffusion import StaticContext
from bioemu2.conditioning.method import SamplingMode
from bioemu2.data_interface import (
    DNA,
    RNA,
    Contact,
    Ligand,
    Protein,
    System,
    load_system_and_contacts,
)
from bioemu2.denoising.samplers import EDMSampler, FKCSampler, GuidedFKSampler, Sampler
from bioemu2.denoising.steering.feats import Tracker
from bioemu2.denoising.steering.potentials import BondLengthPotential
from bioemu2.model import _InferenceModel
from bioemu2.msa.alignment import MSA
from bioemu2.physicality import PhysicalityFilterMode, PhysicalityReport
from bioemu2.precision import autocast_ctx, get_autocast_dtype
from bioemu2.sampling import (
    SamplerPreset,
    apply_physicality_filter,
    generate_samples,
    rng_seed,
    save_topology_cif,
)
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.preparation import prepare_sampling_inputs, prepare_structure
from bioemu2.trajectory_io import map_cif_atom_order

EXAMPLES_DIR = paths.EXAMPLES_DIR / "systems"


def _two_frames() -> mdtraj.Trajectory:
    topology = mdtraj.Topology()
    residue = topology.add_residue("ALA", topology.add_chain())
    topology.add_atom("CA", mdtraj.element.carbon, residue)
    return mdtraj.Trajectory(np.zeros((2, 1, 3)), topology)


@pytest.mark.parametrize("mode", [PhysicalityFilterMode.NO_FILTER, "no_filter"])
def test_apply_physicality_filter_no_filter_is_passthrough(
    mode: PhysicalityFilterMode | str,
) -> None:
    trajectory = _two_frames()
    kept, rejected = apply_physicality_filter(trajectory, mode=mode)
    assert kept is trajectory
    assert len(rejected) == 0


def test_apply_physicality_filter_rejects_unknown_mode() -> None:
    trajectory = _two_frames()
    with pytest.raises(ValueError, match="not a valid PhysicalityFilterMode"):
        apply_physicality_filter(trajectory, "invalid")


def test_apply_physicality_filter_rejects_none() -> None:
    trajectory = _two_frames()
    with pytest.raises(ValueError, match="not a valid PhysicalityFilterMode"):
        apply_physicality_filter(trajectory, None)  # type: ignore[arg-type]


def test_apply_physicality_filter_ignores_unknown_ligand() -> None:
    system = System(
        entities=[Protein(sequence="AA", asym_ids=["A"]), Ligand(smiles="O", asym_ids=["B"])]
    )
    topology = system.topology
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)

    kept, rejected = apply_physicality_filter(trajectory, PhysicalityFilterMode.HIGH_QUALITY)
    assert len(kept) == 0
    assert len(rejected) == 1


def test_apply_physicality_filter_keeps_modified_residue_in_protein_chain() -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    topology = system.topology
    topology.residue(1).name = "UNK"
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)

    kept, rejected = apply_physicality_filter(trajectory, PhysicalityFilterMode.HIGH_QUALITY)
    assert len(kept) == 0
    assert len(rejected) == 1


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "model.ckpt"
    score_model_config = OmegaConf.load(Path(__file__).parent / "assets" / "tiny_config.yaml")
    OmegaConf.save({"score_model": score_model_config}, config_path)
    torch.save(
        {"model_ema": AveragedModel(instantiate(score_model_config)).state_dict()}, checkpoint_path
    )
    return checkpoint_path


@pytest.mark.parametrize(
    "system",
    [
        System(entities=[DNA(sequence="ACGT", asym_ids=["D"])]),
        System(entities=[RNA(sequence="ACGU", asym_ids=["R"])]),
        System(entities=[Ligand(smiles="CCO", asym_ids=["L"])]),
        System(
            entities=[
                RNA(sequence="ACGU", asym_ids=["R"]),
                Ligand(smiles="CCO", asym_ids=["L"]),
            ]
        ),
    ],
    ids=["dna", "rna", "ligand-only", "rna-ligand"],
)
def test_predict_structure_without_protein(
    tiny_checkpoint: Path, tmp_path: Path, system: System
) -> None:
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    out = tmp_path / "prediction"

    trajectory = predict_structure(
        system,
        model=model,
        num_samples=1,
        num_steps=2,
        seed=7,
        progress=False,
        output_dir=out,
    )

    assert trajectory.n_frames == 1
    assert trajectory.topology == system.topology
    assert np.isfinite(trajectory.xyz).all()
    assert (out / "samples.xtc").is_file()
    assert (out / "topology.cif").is_file()
    assert json.loads((out / "physicality.json").read_text()) == [
        {"frame": 0, "status": "not_checked", "reason": None}
    ]


@pytest.mark.parametrize("mode", ["equilibrium", "structure"])
@pytest.mark.parametrize("physicality_mode", ["high_quality", "no_filter"])
def test_high_level_sampling_saves_all_frames_and_physicality(
    tiny_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    physicality_mode: str,
) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    monkeypatch.setattr(sampling_api, "get_msa", lambda system: MSA({"A": ">query\nAAA\n"}))
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    sample = predict_structure if mode == "structure" else sample_equilibrium
    out = tmp_path / "output"
    samples = sample(
        system,
        model=model,
        num_samples=2,
        batch_size=1,
        num_steps=2,
        seed=7,
        progress=False,
        output_dir=str(out),
        physicality_mode=physicality_mode,
    )

    assert samples.n_frames == 2
    assert {path.name for path in out.iterdir()} == {
        "samples.xtc",
        "topology.cif",
        "system.yaml",
        "physicality.json",
    }
    reloaded = mdtraj.load_xtc(str(out / "samples.xtc"), top=str(out / "topology.cif"))
    np.testing.assert_allclose(reloaded.xyz, samples.xyz, atol=1e-3, rtol=0)
    report = check_physicality(samples, mode=physicality_mode, system=system)
    assert report.checked == (physicality_mode != "no_filter")
    if physicality_mode == "high_quality":
        assert not report.passed.all()
    assert json.loads((out / "physicality.json").read_text()) == [
        {
            "frame": frame,
            "status": (
                "not_checked"
                if physicality_mode == "no_filter"
                else "passed"
                if report.passed[frame]
                else "rejected"
            ),
            "reason": report.reasons[frame],
        }
        for frame in range(2)
    ]
    config = OmegaConf.load(out / "system.yaml")
    assert config["mode"] == mode
    assert config["entities"][0]["asym_ids"] == ["A"]


def test_high_level_output_preserves_caller_owned_partial_batches(
    tiny_checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    monkeypatch.setattr(sampling_api, "get_msa", lambda system: MSA({"A": ">query\nAAA\n"}))
    raw = tmp_path / "caller-owned"
    raw.mkdir()
    (raw / "keep.txt").write_text("caller data")
    sample_equilibrium(
        system,
        model=load_model(checkpoint_path=tiny_checkpoint, device="cpu"),
        num_samples=1,
        num_steps=2,
        progress=False,
        partial_dir=raw,
        output_dir=tmp_path / "output",
        physicality_mode="no_filter",
    )
    assert (raw / "keep.txt").read_text() == "caller data"
    assert (raw / "samples_000000-000001.npy").is_file()
    assert (raw / "samples_000000-000001.done").is_file()


@pytest.mark.parametrize("mode", ["equilibrium", "structure"])
def test_high_level_sampling_applies_step_and_potential_overrides(
    tiny_checkpoint: Path, monkeypatch: pytest.MonkeyPatch, mode: SamplingMode
) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAA\n"})
    monkeypatch.setattr(sampling_api, "get_msa", lambda system: msa)
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    sample = predict_structure if mode == "structure" else sample_equilibrium
    samples = sample(
        system,
        model=model,
        num_samples=2,
        sampler="edm_gd",
        num_steps=2,
        potentials=(),
        seed=7,
        progress=False,
    )
    embeddings = model.embed(system, msa=msa, mode=mode)
    expected = sample_with_embeddings(
        system,
        embeddings,
        model=model,
        num_samples=2,
        seed=7,
        progress=False,
        sampler=GuidedFKSampler(
            num_steps=2,
            step_scale=1.5 if mode == "structure" else 1.0,
            resampling_interval=None,
            potentials=(),
        ),
    )
    np.testing.assert_array_equal(samples.xyz, expected.xyz)


@pytest.mark.parametrize("workflow", ["session", "equilibrium", "structure", "embeddings"])
def test_sampling_retains_failed_frames_with_separate_saveable_physicality(
    tiny_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAA\n"})
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    monkeypatch.setattr(sampling_api, "get_msa", lambda system: msa)
    monkeypatch.setattr(
        sampling_api,
        "generate_samples",
        lambda **kwargs: torch.zeros(kwargs["num_samples"], kwargs["num_atoms"], 3),
    )
    if workflow == "session":
        samples = EmuSession(system, model=model, mode="equilibrium", msa=msa).sample(
            num_samples=2, progress=False
        )
    elif workflow == "embeddings":
        samples = sample_with_embeddings(
            system,
            model.embed(system, msa=msa, mode="equilibrium"),
            model=model,
            num_samples=2,
            sampler=EDMSampler(num_steps=2),
            progress=False,
        )
    else:
        sample = sample_equilibrium if workflow == "equilibrium" else predict_structure
        samples = sample(system, model=model, num_samples=2, progress=False)

    assert samples.n_frames == 2
    original_coords = samples.xyz.copy()
    report = check_physicality(samples, system=system)
    assert not report.passed.any()
    assert all(reason is not None for reason in report.reasons)
    np.testing.assert_array_equal(samples.xyz, original_coords)
    path = save_frames(samples, tmp_path / workflow, physicality=report)
    reloaded = mdtraj.load_xtc(str(path), top=str(path.parent / "topology.cif"))
    np.testing.assert_allclose(reloaded.xyz, original_coords, atol=1e-3)
    assert json.loads((path.parent / "physicality.json").read_text()) == [
        {"frame": frame, "status": "rejected", "reason": report.reasons[frame]}
        for frame in range(2)
    ]


@pytest.fixture
def embeds_cache(tiny_checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = torch.load(tiny_checkpoint, weights_only=True)
    state["model_fingerprint"] = "test-model"
    torch.save(state, tiny_checkpoint)
    cache = tmp_path / "embeddings"
    monkeypatch.setenv("BIOEMU2_EMBEDS_CACHE", str(cache))
    return cache


@pytest.mark.parametrize(
    ("interruption", "seed"),
    [("sampling", 0), ("writing", 23), ("before-marker", 23), ("sampling", None)],
)
def test_resume_completed_batches(
    tiny_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: str,
    seed: int | None,
) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    msa = MSA({"A": ">query\nAAA\n"})
    sample = partial(
        sample_with_embeddings,
        system,
        model=model,
        trunk_embeddings=model.embed(system, msa=msa, mode="equilibrium"),
        num_samples=5,
        batch_size=2,
        sampler=EDMSampler(num_steps=2),
        seed=seed,
        progress=False,
    )
    expected = sample() if seed is not None else None
    partial_dir = tmp_path / "interrupted"
    sizes: list[int] = []
    interrupt = True
    denoise = EDMSampler.__call__
    save = np.save

    def sample_batch(self, **kwargs):
        sizes.append(kwargs["batch_size"])
        if interrupt and interruption == "sampling" and len(sizes) == 2:
            raise RuntimeError("interrupted")
        return denoise(self, **kwargs)

    def save_batch(path, coordinates, **kwargs):
        if interrupt and Path(path).name == "samples_000002-000004.npy":
            if interruption == "writing":
                Path(path).write_bytes(b"incomplete")
                raise RuntimeError("interrupted")
            if interruption == "before-marker":
                save(path, coordinates, **kwargs)
                raise RuntimeError("interrupted")
        return save(path, coordinates, **kwargs)

    monkeypatch.setattr(EDMSampler, "__call__", sample_batch)
    monkeypatch.setattr(np, "save", save_batch)
    with pytest.raises(RuntimeError, match="interrupted"):
        sample(partial_dir=partial_dir)
    assert (partial_dir / "samples_000000-000002.done").is_file()
    assert not (partial_dir / "samples_000002-000004.done").exists()
    saved_first = np.load(partial_dir / "samples_000000-000002.npy")

    interrupt = False
    sizes.clear()
    resumed = sample(partial_dir=partial_dir)
    assert sizes == [2, 1]
    if expected is not None:
        np.testing.assert_array_equal(resumed.xyz, expected.xyz)
    else:
        np.testing.assert_allclose(resumed.xyz[:2], saved_first / 10.0, rtol=1e-6, atol=1e-7)
        assert not np.array_equal(resumed.xyz[:2], resumed.xyz[2:4])

    sizes.clear()
    repeated = sample(partial_dir=partial_dir)
    assert sizes == []
    np.testing.assert_array_equal(repeated.xyz, resumed.xyz)


@pytest.mark.parametrize("sampler_type", [FKCSampler, GuidedFKSampler])
def test_sampling_preserves_each_batch_diagnostics(
    tiny_checkpoint: Path, tmp_path: Path, sampler_type: type[FKCSampler | GuidedFKSampler]
) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    trunk = model.embed(system, msa=MSA({"A": ">query\nAAA\n"}), mode="equilibrium")
    topology, batch = prepare_sampling_inputs(system, device=model.device)
    potential = BondLengthPotential(sigma_tolerance=10.0)
    sampler = sampler_type(num_steps=3, potentials=[potential])
    if isinstance(sampler, GuidedFKSampler):
        sampler.num_gd_steps = 0
        sampler.resampling_interval = 1
    run_dir = tmp_path / "batched"
    partial_dir = tmp_path / "partial"
    sample = partial(
        generate_samples,
        num_samples=3,
        batch_size=2,
        score_model=model,
        sampler=sampler,
        num_atoms=topology.n_atoms,
        system=system,
        batch=batch,
        trunk_embeddings=trunk,
        progress=False,
        run_dir=run_dir,
        partial_dir=partial_dir,
        seed=13,
    )
    expected_coords = sample()

    reference_dir = tmp_path / "reference"
    expected_files = set()
    for start, stop in ((0, 2), (2, 3)):
        with rng_seed(13 + start // 2), torch.no_grad():
            sampler(
                batch=batch,
                score_model=model,
                trunk_embeddings=trunk,
                batch_size=stop - start,
                tracker=Tracker(reference_dir / "metrics"),
            )
        for reference in (reference_dir / "metrics").glob("*.npy"):
            name = f"{reference.stem}_{start:06d}-{stop:06d}.npy"
            expected_files.add(name)
            np.testing.assert_array_equal(np.load(run_dir / "metrics" / name), np.load(reference))
    assert expected_files
    assert {path.name for path in (run_dir / "metrics").iterdir()} == expected_files

    metrics_dir = run_dir / "metrics"
    original = {path.name: path.read_bytes() for path in metrics_dir.iterdir()}
    missing = metrics_dir / "x_t_000000-000002.npy"
    missing.unlink()
    skipped_mtimes = {
        path.name: path.stat().st_mtime_ns for path in metrics_dir.glob("*_000000-000002.npy")
    }
    for path in metrics_dir.glob("*_000002-000003.npy"):
        path.write_bytes(b"incomplete")
    (partial_dir / "samples_000002-000003.done").unlink()

    resumed_coords = sample()
    torch.testing.assert_close(resumed_coords, expected_coords, rtol=0, atol=0)
    assert {path.name for path in metrics_dir.iterdir()} == expected_files - {missing.name}
    for path in metrics_dir.iterdir():
        assert path.read_bytes() == original[path.name]
    for name, mtime in skipped_mtimes.items():
        assert (metrics_dir / name).stat().st_mtime_ns == mtime


def test_load_model_rejects_conflicting_checkpoints(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Pass only one"):
        load_model("bioemu2-preview", checkpoint_path=tmp_path / "model.ckpt")


def test_load_model_rejects_cache_dir_with_local_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cache_dir applies only"):
        load_model(checkpoint_path=tmp_path / "model.ckpt", cache_dir=tmp_path / "cache")


@pytest.mark.parametrize(
    ("arch_list", "capability", "supported"),
    [
        (["sm_75", "sm_80", "sm_90", "sm_120"], (7, 0), False),
        (["sm_75", "sm_80", "sm_90", "sm_120"], (8, 9), True),
        (["sm_70", "sm_80"], (7, 0), True),
        (["sm_80", "compute_80"], (9, 0), True),
        (["sm_86"], (8, 0), False),
    ],
)
def test_load_model_checks_gpu_against_pytorch_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arch_list: list[str],
    capability: tuple[int, int],
    supported: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: arch_list)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")
    missing_checkpoint = tmp_path / "missing.ckpt"
    if supported:
        with pytest.raises(FileNotFoundError):
            load_model(checkpoint_path=missing_checkpoint, device="cuda")
    else:
        with pytest.raises(RuntimeError, match=r"\.\[cuda12\]"):
            load_model(checkpoint_path=missing_checkpoint, device="cuda")


def test_load_model_rejects_empty_checkpoint_name() -> None:
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        load_model("")


def test_emu_session_reuses_one_embedding_for_sampling_and_propagation(
    tiny_checkpoint: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    msa = MSA({"A": ">query\nAAA\n"})
    embed_calls: list[SamplingMode] = []
    embedding_refs: list[weakref.ReferenceType[TrunkEmbeddings]] = []
    original_embed = model.embed

    def embed(system: System, *, msa: MSA, mode: SamplingMode, contacts=None):
        embed_calls.append(mode)
        embeddings = original_embed(system, msa=msa, mode=mode, contacts=contacts)
        embedding_refs.append(weakref.ref(embeddings))
        return embeddings

    def generate(**kwargs):
        batch = kwargs["batch"]
        if batch.coords_cond is not None and batch.is_dynamics_conditioned.any():
            return batch.coords_cond[:, : kwargs["num_atoms"]].clone()
        return torch.zeros(kwargs["num_samples"], kwargs["num_atoms"], 3)

    monkeypatch.setattr(model, "embed", embed)
    monkeypatch.setattr(sampling_api, "generate_samples", generate)
    monkeypatch.setattr(propagation_api, "generate_samples", generate)
    emu = EmuSession(system, model=model, mode="equilibrium", msa=msa)

    first = emu.sample(num_samples=2, progress=False)
    second = emu.sample(num_samples=1, progress=False)
    endpoints = emu.propagate(first[:1], lag_time_ns=1.0, progress=False)
    assert first.n_frames == 2
    assert second.n_frames == endpoints.n_frames == 1
    assert embed_calls == ["equilibrium"]
    assert embedding_refs[0]() is not None

    with pytest.raises(AttributeError, match="has no setter"):
        setattr(emu, "mode", "structure")

    structure_emu = EmuSession(system, model=model, mode="structure", msa=msa)
    structure_emu.sample(num_samples=1, progress=False)
    assert embed_calls == ["equilibrium", "structure"]
    with caplog.at_level(logging.WARNING, logger="bioemu2.inference"):
        structure_endpoints = structure_emu.propagate(first[:1], lag_time_ns=1.0, progress=False)
    assert structure_endpoints.n_frames == 1
    assert (
        "Propagation normally uses equilibrium embeddings; using mode=structure unchanged."
        in caplog.text
    )
    assert embed_calls == ["equilibrium", "structure"]

    other = EmuSession(
        system, model=model, mode="equilibrium", msa=MSA({"A": ">query\nAAA\n>homolog\nAAA\n"})
    )
    other.sample(num_samples=1, progress=False)
    assert embed_calls == ["equilibrium", "structure", "equilibrium"]

    contacts: list[Contact] = []
    conditioned = EmuSession(system, model=model, mode="equilibrium", msa=msa, contacts=contacts)
    conditioned.sample(num_samples=1, progress=False)
    contacts.append(
        Contact(
            token1={"asym_id": "A", "res_id": 1},
            token2={"asym_id": "A", "res_id": 3},
        )
    )
    with pytest.raises(ValueError, match="contacts have changed; create a new EmuSession"):
        conditioned.sample(num_samples=1, progress=False)
    assert len(embed_calls) == 4

    emu.cache_clear()
    assert embedding_refs[0]() is None
    emu.sample(num_samples=1, progress=False)
    assert len(embed_calls) == 5
    other.sample(num_samples=1, progress=False)
    assert len(embed_calls) == 5

    system.asym_id_to_author_id["A"] = "Z"
    with pytest.raises(ValueError, match="system has changed; create a new EmuSession"):
        _ = emu.msa
    with pytest.raises(ValueError, match="system has changed; create a new EmuSession"):
        emu.sample(num_samples=1, progress=False)
    with pytest.raises(ValueError, match="system has changed; create a new EmuSession"):
        emu.propagate(first[:1], lag_time_ns=1.0, progress=False)
    emu.cache_clear()
    with pytest.raises(ValueError, match="system has changed; create a new EmuSession"):
        emu.sample(num_samples=1, progress=False)
    assert len(embed_calls) == 5


def test_emu_session_rejects_system_changed_before_first_embedding(
    tiny_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    emu = EmuSession(system, model=model, mode="equilibrium")
    system.asym_id_to_author_id["A"] = "Z"
    monkeypatch.setattr(model, "embed", lambda *args, **kwargs: pytest.fail("Unexpected embedding"))
    monkeypatch.setattr(
        inference_api, "get_msa", lambda system: pytest.fail("Unexpected MSA computation")
    )

    with pytest.raises(ValueError, match="system has changed; create a new EmuSession"):
        _ = emu.msa
    with pytest.raises(ValueError, match="system has changed; create a new EmuSession"):
        emu.sample(num_samples=1, progress=False)


def test_emu_session_rejects_contacts_changed_before_first_embedding(
    tiny_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    contacts: list[Contact] = []
    emu = EmuSession(
        system, model=model, mode="equilibrium", msa=MSA({"A": ">query\nAAA\n"}), contacts=contacts
    )
    contacts.append(
        Contact(
            token1={"asym_id": "A", "res_id": 1},
            token2={"asym_id": "A", "res_id": 3},
        )
    )
    monkeypatch.setattr(model, "embed", lambda *args, **kwargs: pytest.fail("Unexpected embedding"))

    with pytest.raises(ValueError, match="contacts have changed; create a new EmuSession"):
        emu.sample(num_samples=1, progress=False)


def test_emu_session_computes_msa_on_first_read(
    tiny_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    msa = MSA({"A": ">query\nAAA\n"})
    calls: list[System] = []

    def get_msa(query: System) -> MSA:
        calls.append(query)
        return msa

    monkeypatch.setattr(inference_api, "get_msa", get_msa)
    emu = EmuSession(system, model=model, mode="equilibrium")
    assert calls == []
    assert emu.msa is msa
    assert emu.msa is msa
    assert calls == [system]


@pytest.mark.parametrize(
    ("mode", "sampler", "scale", "warns"),
    [
        ("equilibrium", EDMSampler(step_scale=1.2), 1.2, True),
        ("structure", EDMSampler(step_scale=1.0), 1.0, True),
        (
            "equilibrium",
            GuidedFKSampler(step_scale=1.2, resampling_interval=None),
            1.2,
            True,
        ),
        ("equilibrium", EDMSampler(step_scale=1.0), 1.0, False),
        ("structure", GuidedFKSampler(step_scale=1.5, resampling_interval=None), 1.5, False),
        ("equilibrium", FKCSampler(), 2.0, False),
        ("structure", FKCSampler(), 2.0, False),
        ("equilibrium", "edm", 1.0, False),
        ("structure", "edm", 1.5, False),
    ],
)
def test_emu_session_warns_on_nonstandard_sampling_scale(
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mode: SamplingMode,
    sampler: SamplerPreset | Sampler,
    scale: float,
    warns: bool,
) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    emu = EmuSession(system, model=model, mode=mode, msa=MSA({"A": ">query\nAAA\n"}))

    def generate(**kwargs):
        coords = torch.zeros(kwargs["num_samples"], kwargs["num_atoms"], 3)
        coords[:, 0, 0] = getattr(kwargs["sampler"], "step_scale", 2.0)
        return coords

    monkeypatch.setattr(sampling_api, "generate_samples", generate)
    with caplog.at_level(logging.WARNING, logger="bioemu2.inference"):
        result = emu.sample(num_samples=1, sampler=sampler, progress=False)

    np.testing.assert_allclose(result.xyz[:, 0, 0], scale / 10)
    warnings = [record.message for record in caplog.records if record.name == "bioemu2.inference"]
    assert bool(warnings) is warns
    if warns:
        assert f"step_scale={scale}" in warnings[0]
        assert mode in warnings[0]


def test_trunk_embeddings_reused_from_disk_by_new_model(
    tiny_checkpoint: Path, embeds_cache: Path
) -> None:
    cache = embeds_cache
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAA\n"})
    first = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    expected = first.embed(system, msa=msa, mode="equilibrium")
    entries = list(cache.iterdir())
    assert len(entries) == 1
    assert (entries[0] / "metadata.json").is_file()
    assert (entries[0] / "embeddings.pt").is_file()

    second = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    with patch.object(second, "_run_trunk", side_effect=AssertionError("recomputed")):
        actual = second.embed(system, msa=msa, mode="equilibrium")
    for name in TrunkEmbeddings.__dataclass_fields__:
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)


def test_trunk_cache_key_changes_with_conditioning_and_model(
    tiny_checkpoint: Path, embeds_cache: Path
) -> None:
    cache = embeds_cache
    system = System(entities=[Protein(sequence="AAAAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAAAA\n"})
    contact = Contact(token1={"asym_id": "A", "res_id": 1}, token2={"asym_id": "A", "res_id": 5})
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu", precision="32-true")
    model.embed(system, msa=msa, mode="equilibrium")
    model.embed(system, msa=MSA({"A": ">query\nAAAAA\n>other\nAAA-A\n"}), mode="equilibrium")
    model.embed(
        System(entities=[Protein(sequence="AAAAAA", asym_ids=["A"])]),
        msa=MSA({"A": ">query\nAAAAAA\n"}),
        mode="equilibrium",
    )
    model.embed(system, msa=msa, mode="structure")
    model.embed(system, msa=msa, mode="equilibrium", contacts=[contact])
    assert len(list(cache.iterdir())) == 5

    other = load_model(checkpoint_path=tiny_checkpoint, device="cpu", precision="32-true")
    assert isinstance(other, _InferenceModel)
    other.num_steps_recycle = 0
    other.embed(system, msa=msa, mode="equilibrium")
    assert len(list(cache.iterdir())) == 6
    mixed = load_model(
        checkpoint_path=tiny_checkpoint,
        device="cpu",
        precision="bf16-mixed",
    )
    mixed.embed(system, msa=msa, mode="equilibrium")
    assert len(list(cache.iterdir())) == 7


def test_trunk_cache_rejects_incomplete_entry(tiny_checkpoint: Path, embeds_cache: Path) -> None:
    cache = embeds_cache
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAA\n"})
    model.embed(system, msa=msa, mode="equilibrium")
    entry = next(cache.iterdir())
    (entry / "embeddings.pt").unlink()
    with pytest.raises(FileNotFoundError, match="embeddings.pt"):
        model.embed(system, msa=msa, mode="equilibrium")


def test_trunk_cache_recomputes_entry_without_marker(
    tiny_checkpoint: Path, embeds_cache: Path
) -> None:
    cache = embeds_cache
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    assert isinstance(model, _InferenceModel)
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAA\n"})
    model.embed(system, msa=msa, mode="equilibrium")
    entry = next(cache.iterdir())
    # An interrupted write leaves embeddings.pt without the metadata marker.
    (entry / "metadata.json").unlink()
    with patch.object(model, "_run_trunk", wraps=model._run_trunk) as compute:
        model.embed(system, msa=msa, mode="equilibrium")
    compute.assert_called_once()
    assert (entry / "metadata.json").is_file()
    assert (entry / "embeddings.pt").is_file()


def test_trunk_cache_uses_environment_directory(tiny_checkpoint: Path, embeds_cache: Path) -> None:
    cache = embeds_cache
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAA\n"})
    load_model(checkpoint_path=tiny_checkpoint, device="cpu").embed(
        system, msa=msa, mode="equilibrium"
    )
    assert len(list(cache.iterdir())) == 1


def test_trunk_cache_is_disabled_when_environment_variable_is_unset(
    tiny_checkpoint: Path,
    embeds_cache: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BIOEMU2_EMBEDS_CACHE")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    assert isinstance(model, _InferenceModel)
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAA\n"})
    with patch.object(model, "_run_trunk", wraps=model._run_trunk) as compute:
        model.embed(system, msa=msa, mode="equilibrium")
        model.embed(system, msa=msa, mode="equilibrium")
    assert compute.call_count == 2
    assert not (tmp_path / ".cache" / "bioemu2" / "trunk_embeddings").exists()
    assert not embeds_cache.exists()


def test_checkpoint_fingerprint_changes_cache_identity(
    tiny_checkpoint: Path, embeds_cache: Path
) -> None:
    cache = embeds_cache
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAA\n"})
    state = torch.load(tiny_checkpoint, weights_only=True)
    state["model_fingerprint"] = "first-release"
    torch.save(state, tiny_checkpoint)
    first = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    first.embed(system, msa=msa, mode="equilibrium")

    state["model_fingerprint"] = "second-release"
    torch.save(state, tiny_checkpoint)
    second = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    second.embed(system, msa=msa, mode="equilibrium")
    assert len(list(cache.iterdir())) == 2


def test_missing_model_fingerprint_warns_and_skips_cache(
    tiny_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = tmp_path / "embeddings"
    monkeypatch.setenv("BIOEMU2_EMBEDS_CACHE", str(cache))
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    model.embed(
        System(entities=[Protein(sequence="AAA", asym_ids=["A"])]),
        msa=MSA({"A": ">query\nAAA\n"}),
        mode="equilibrium",
    )
    assert "has no model fingerprint" in caplog.text
    assert not cache.exists()


def test_unwritable_cache_warns_before_computing(
    tiny_checkpoint: Path, embeds_cache: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    assert isinstance(model, _InferenceModel)
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAA\n"})
    with (
        patch(
            "bioemu2.embedding_cache.tempfile.TemporaryFile",
            side_effect=PermissionError(errno.EACCES, "denied"),
        ),
        patch.object(model, "_run_trunk", wraps=model._run_trunk) as compute,
    ):
        model.embed(system, msa=msa, mode="equilibrium")
    compute.assert_called_once()
    assert "not writable" in caplog.text
    assert list(embeds_cache.iterdir()) == []


@pytest.mark.parametrize(
    "failure",
    [
        OSError(errno.ENOSPC, "disk full"),
        RuntimeError("PytorchStreamWriter failed writing file data/0: file write failed"),
    ],
)
def test_write_failure_does_not_discard_embeddings(
    tiny_checkpoint: Path,
    embeds_cache: Path,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
) -> None:
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    with patch(
        "bioemu2.embedding_cache.torch.save",
        side_effect=failure,
    ):
        embeddings = model.embed(
            System(entities=[Protein(sequence="AAA", asym_ids=["A"])]),
            msa=MSA({"A": ">query\nAAA\n"}),
            mode="equilibrium",
        )
    assert isinstance(embeddings, TrunkEmbeddings)
    assert "continuing without caching" in caplog.text
    assert list(embeds_cache.iterdir()) == []


def test_save_and_load_embeddings_round_trip(tiny_checkpoint: Path, tmp_path: Path) -> None:
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAA\n"})
    embeddings = model.embed(system, msa=msa, mode="equilibrium")
    path = tmp_path / "saved" / "embeddings.pt"
    embeddings.save(path)
    assert path.is_file()
    loaded = TrunkEmbeddings.load(path, device=torch.device("cpu"))
    for name in TrunkEmbeddings.__dataclass_fields__:
        assert torch.equal(getattr(loaded, name), getattr(embeddings, name))


def test_save_embeddings_rejects_existing_file(tiny_checkpoint: Path, tmp_path: Path) -> None:
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAA\n"})
    embeddings = model.embed(system, msa=msa, mode="equilibrium")
    path = tmp_path / "embeddings.pt"
    path.write_bytes(b"keep me")
    with pytest.raises(FileExistsError, match="existing file"):
        embeddings.save(path)
    assert path.read_bytes() == b"keep me"


def test_emu_save_embeddings_rejects_non_empty_directory(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    msa = MSA({"A": ">query\nAAA\n"})
    directory = tmp_path / "saved"
    directory.mkdir()
    (directory / "existing.pt").write_bytes(b"keep me")
    emu = EmuSession(system, model=model, mode="equilibrium", msa=msa)
    with pytest.raises(FileExistsError, match="non-empty directory"):
        emu.save_embeddings(directory)
    assert (directory / "existing.pt").read_bytes() == b"keep me"
    assert not (directory / "embeddings.pt").exists()


def test_emu_session_embeddings_argument_skips_trunk_and_msa(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    msa = MSA({"A": ">query\nAAA\n"})

    directory = tmp_path / "emb"
    EmuSession(system, model=model, mode="equilibrium", msa=msa).save_embeddings(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["mode"] == "equilibrium"
    assert metadata["autocast_dtype"] == "torch.bfloat16"
    assert metadata["num_steps_recycle"] == 3
    assert set(metadata) == {
        "system",
        "contacts",
        "msa",
        "mode",
        "autocast_dtype",
        "num_steps_recycle",
    }

    # A session constructed with preloaded embeddings returns them directly,
    # without an MSA (none was supplied) or a call into the trunk.
    embeddings = TrunkEmbeddings.load(directory / "embeddings.pt", device=torch.device("cpu"))
    loaded = EmuSession(system, model=model, mode="equilibrium", embeddings=embeddings)
    with patch.object(model, "embed", wraps=model.embed) as embed:
        reused = loaded._get_embedding()
    embed.assert_not_called()
    assert reused is embeddings
    with pytest.raises(ValueError, match=r"use TrunkEmbeddings.save\(path\) instead"):
        loaded.save_embeddings(tmp_path / "reloaded")
    assert not (tmp_path / "reloaded").exists()


def test_emu_save_embeddings_records_available_model_fingerprint(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    state = torch.load(tiny_checkpoint, weights_only=True)
    state["model_fingerprint"] = "test-model"
    torch.save(state, tiny_checkpoint)
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    directory = tmp_path / "saved"
    EmuSession(
        system, model=model, mode="equilibrium", msa=MSA({"A": ">query\nAAA\n"})
    ).save_embeddings(directory)

    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["model"].startswith("test-model:")


@pytest.mark.parametrize(
    ("sample", "preset", "expected_scale"),
    [
        (predict_structure, "edm", 1.5),
        (sample_equilibrium, "edm_gd", 1.0),
        (sample_equilibrium, "edm_fkc", 2.0),
    ],
)
def test_simple_sampling_computes_msa_and_uses_task_scale(
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample,
    preset: str,
    expected_scale: float,
) -> None:
    system, _ = load_system_and_contacts(EXAMPLES_DIR / "chignolin.yaml")
    queries: list[System] = []

    def get_msa(query: System) -> MSA:
        queries.append(query)
        return MSA({"A": ">query\nGYDPETGTWG\n"})

    def generate(**kwargs):
        coords = torch.zeros(kwargs["num_samples"], kwargs["num_atoms"], 3)
        sampler = kwargs["sampler"]
        coords[:, 0, 0] = 2.0 if isinstance(sampler, FKCSampler) else sampler.step_scale
        return coords

    monkeypatch.setattr(sampling_api, "get_msa", get_msa)
    monkeypatch.setattr(sampling_api, "generate_samples", generate)
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    result = sample(
        system,
        model=model,
        sampler=preset,
        num_samples=2,
        progress=False,
    )

    assert queries == [system]
    assert result.n_frames == 2
    assert result.topology == system.topology
    np.testing.assert_allclose(result.xyz[:, 0, 0], expected_scale / 10)


@pytest.mark.parametrize("mode", ["structure", "equilibrium"])
def test_explicit_sampler_scale_is_not_overwritten(
    tiny_checkpoint: Path, monkeypatch: pytest.MonkeyPatch, mode: SamplingMode
) -> None:
    system, _ = load_system_and_contacts(EXAMPLES_DIR / "chignolin.yaml")
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    msa = MSA({"A": ">query\nGYDPETGTWG\n"})

    def generate(**kwargs):
        coords = torch.zeros(kwargs["num_samples"], kwargs["num_atoms"], 3)
        coords[:, 0, 0] = kwargs["sampler"].step_scale
        return coords

    monkeypatch.setattr(sampling_api, "generate_samples", generate)
    embeddings = model.embed(system, msa=msa, mode=mode)
    result = sample_with_embeddings(
        system,
        embeddings,
        model=model,
        sampler=EDMSampler(step_scale=0.5),
        num_samples=1,
        progress=False,
    )
    np.testing.assert_allclose(result.xyz[:, 0, 0], [0.05])


def test_simple_sampling_rejects_unknown_preset_before_msa(
    monkeypatch: pytest.MonkeyPatch, tiny_checkpoint: Path
) -> None:
    monkeypatch.setattr(sampling_api, "get_msa", lambda system: pytest.fail("queried MSA"))
    with pytest.raises(ValueError, match="Unknown sampler preset"):
        sample_equilibrium(
            System(entities=[Protein(sequence="AAA", asym_ids=["A"])]),
            model=load_model(checkpoint_path=tiny_checkpoint, device="cpu"),
            sampler="unknown",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "sampler",
    [
        EDMSampler(num_steps=2),
        GuidedFKSampler(num_steps=2, num_gd_steps=1),
        FKCSampler(num_steps=2),
    ],
)
@pytest.mark.parametrize("conditioning", ["mode", "contacts"])
def test_sampling_ignores_batch_trunk_conditioning(
    tiny_checkpoint: Path, sampler: Sampler, conditioning: str
) -> None:
    system, _ = load_system_and_contacts(EXAMPLES_DIR / "chignolin.yaml")
    msa = MSA({"A": ">query\nGYDPETGTWG\n"})
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    trunk = model.embed(system, msa=msa, mode="equilibrium")
    topology, batch = prepare_sampling_inputs(system, device=model.device)
    conditioned = batch_collate(
        [
            prepare_structure(
                system,
                msa=msa,
                mode="structure",
                contacts=[
                    Contact(
                        token1={"asym_id": "A", "res_id": 1},
                        token2={"asym_id": "A", "res_id": 5},
                    )
                ],
            )
        ]
    )
    if conditioning == "mode":
        assert "method_feature" not in batch
        changed = batch.replace(method_feature=conditioned.method_feature)
    else:
        assert "contact_conditioning" not in batch
        assert conditioned.contact_conditioning.any()
        changed = batch.replace(
            contact_conditioning=conditioned.contact_conditioning,
            contact_threshold=conditioned.contact_threshold,
        )

    samples = []
    for sampling_batch in (batch, changed):
        torch.manual_seed(42)
        samples.append(
            generate_samples(
                num_samples=3,
                batch_size=2,
                score_model=model,
                sampler=sampler,
                num_atoms=topology.n_atoms,
                system=system,
                batch=sampling_batch,
                trunk_embeddings=trunk,
                progress=False,
            )
        )
    assert samples[0].shape == (3, topology.n_atoms, 3)
    assert torch.isfinite(samples[0]).all()
    torch.testing.assert_close(samples[0], samples[1], rtol=0, atol=0)


@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_static_context_preserves_precision_without_gradients(
    tiny_checkpoint: Path, precision: str, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA GPU")
    system, _ = load_system_and_contacts(EXAMPLES_DIR / "chignolin.yaml")
    model = load_model(
        checkpoint_path=tiny_checkpoint, device=device, precision=precision, matmul_precision="high"
    )
    assert isinstance(model, _InferenceModel)
    trunk = model.embed(system, msa=MSA({"A": ">query\nGYDPETGTWG\n"}), mode="equilibrium")
    trunk = replace(
        trunk,
        single=trunk.single.requires_grad_(),
        pair=trunk.pair.requires_grad_(),
    )
    _, batch = prepare_sampling_inputs(system, device=model.device)
    with torch.enable_grad():
        context = model.compute_static_context(batch, trunk, multiplicity=2)
    with torch.no_grad(), autocast_ctx(device, get_autocast_dtype(precision)):
        expected = model.score_model.compute_static_context(batch, trunk)
    assert context.z.dtype == torch.float32
    for actual, reference in (
        (context.z, expected.z),
        (context.atom_encoder_context.q, expected.atom_encoder_context.q),
        (context.atom_encoder_context.c, expected.atom_encoder_context.c),
        (context.atom_encoder_context.p, expected.atom_encoder_context.p),
    ):
        assert not actual.requires_grad
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert isinstance(context, StaticContext)
    coords = torch.randn(2, *batch.coords.shape[1:], device=device, requires_grad=True)
    sigmas = torch.full((2,), 0.5, device=device, requires_grad=True)
    output = model.denoise(context, trunk, coords, sigmas)
    gradients = torch.autograd.grad(output.denoised_coords.square().sum(), (coords, sigmas))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_fkc_samples_structure_embeddings(tiny_checkpoint: Path) -> None:
    system, _ = load_system_and_contacts(EXAMPLES_DIR / "chignolin.yaml")
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    trunk_embeddings = model.embed(
        system,
        msa=MSA({"A": ">query\nGYDPETGTWG\n"}),
        mode="structure",
    )
    assert isinstance(trunk_embeddings, TrunkEmbeddings)
    sampler = FKCSampler(
        num_steps=3,
        potentials=[
            BondLengthPotential(sigma_tolerance=10.0, parameters={"guidance_weight": 0.05})
        ],
    )
    with (
        patch.object(
            model,
            "_run_trunk",
            side_effect=AssertionError("Trunk embeddings must be reused"),
        ),
        patch.object(
            model,
            "compute_static_context",
            wraps=model.compute_static_context,
        ) as compute_static_context,
    ):
        samples = sample_with_embeddings(
            system,
            trunk_embeddings,
            model=model,
            num_samples=3,
            batch_size=2,
            sampler=sampler,
            seed=42,
            progress=False,
        )

    assert samples.n_frames == 3
    assert samples.topology == system.topology
    assert np.isfinite(samples.xyz).all()
    assert compute_static_context.call_count == 2


def _write_x0(system, directory: Path) -> list[str]:
    """Two random frames as ``--x0-cif``/``--x0-xtc`` arguments, written like a CLI run."""
    structure = AllAtomStructure.from_system(system, mode="equilibrium")
    xyz = np.random.default_rng(0).normal(size=(2, system.topology.n_atoms, 3)).astype(np.float32)
    save_topology_cif(structure, system, xyz[0] * 10.0, directory / "topology.cif")
    mdtraj.Trajectory(xyz, system.topology).save_xtc(str(directory / "samples.xtc"))
    return ["--x0-cif", str(directory / "topology.cif"), "--x0-xtc", str(directory / "samples.xtc")]


@pytest.fixture
def x0_args(tmp_path: Path) -> list[str]:
    system, _ = load_system_and_contacts(EXAMPLES_DIR / "chignolin.yaml")
    return _write_x0(system, tmp_path)


def _mock_cli_msa(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (cli, sampling_api, propagation_api):
        monkeypatch.setattr(module, "get_msa", lambda system: MSA({"A": ">query\nGYDPETGTWG\n"}))


@pytest.mark.parametrize(
    ("command", "extra_args"),
    [
        pytest.param("sample-equilibrium", ["--num-samples", "2"], id="equilibrium"),
        pytest.param("predict-structure", ["--num-samples", "1"], id="structure"),
        pytest.param(
            "sample-equilibrium",
            ["--num-samples", "2", "--sampler", "edm_gd", "--num-steps", "3"],
            id="steered",
        ),
        pytest.param(
            "propagate",
            ["X0", "--lag-time-ns", "1"],
            id="propagate",
        ),
    ],
)
def test_cli_pipeline(
    tmp_path: Path,
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    x0_args: list[str],
    command: str,
    extra_args: list[str],
) -> None:
    """Run the CLI end to end on a tiny checkpoint without network access."""
    extra_args = [*x0_args, *extra_args[1:]] if extra_args[:1] == ["X0"] else extra_args
    _mock_cli_msa(monkeypatch)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli.main,
        [
            command,
            "--system",
            str(EXAMPLES_DIR / "chignolin.yaml"),
            "--checkpoint-path",
            str(tiny_checkpoint),
            "--out",
            str(out),
            "--physicality-filter",
            "no_filter",
            *extra_args,
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    if command == "propagate":
        xtcs, n_frames = {"samples.xtc"}, 2
    else:
        xtcs, n_frames = {"samples.xtc"}, 1 if command == "predict-structure" else 2
    extra = {"physicality.json"}
    assert {p.name for p in out.iterdir()} == xtcs | extra | {
        "topology.cif",
        "system.yaml",
        "command.txt",
    }
    trajectory = mdtraj.load_xtc(str(out / sorted(xtcs)[0]), top=str(out / "topology.cif"))
    assert trajectory.n_frames == n_frames
    cif = (out / "topology.cif").read_text()
    assert "_entity_poly." in cif and "_struct_asym." in cif
    assert trajectory.n_residues == 10
    assert np.isfinite(trajectory.xyz).all()
    if command == "propagate":
        assert trajectory.time.tolist() == [1000.0, 1000.0]
    assert json.loads((out / "physicality.json").read_text()) == [
        {"frame": i, "status": "not_checked", "reason": None} for i in range(n_frames)
    ]
    assert (
        f"mode: {'structure' if command == 'predict-structure' else 'equilibrium'}"
        in (out / "system.yaml").read_text()
    )


@pytest.mark.parametrize(
    ("command", "interruption"),
    [
        ("predict-structure", "topology"),
        ("sample-equilibrium", "samples.xtc"),
        ("predict-structure", "physicality.json"),
    ],
)
def test_cli_resumes_and_cleans_up_after_saving(
    tmp_path: Path,
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    interruption: str,
) -> None:
    monkeypatch.setattr(cli, "load_model", partial(load_model, device="cpu"))
    _mock_cli_msa(monkeypatch)
    monkeypatch.setattr(
        sampling_api,
        "check_physicality",
        lambda trajectory, mode, expected_extra_bonds: PhysicalityReport(
            passed=np.array([True] * (len(trajectory) - 1) + [False]),
            reasons=[None] * (len(trajectory) - 1) + ["clash"],
            mode=mode,
        ),
    )
    out = tmp_path / "interrupted"
    args = [
        command,
        "--system",
        str(EXAMPLES_DIR / "chignolin.yaml"),
        "--checkpoint-path",
        str(tiny_checkpoint),
        "--num-samples",
        "5",
        "--batch-size",
        "2",
        "--num-steps",
        "2",
        "--out",
        str(out),
    ]
    runner = CliRunner()
    partial_dir = out / "partial"
    save_xtc = mdtraj.Trajectory.save_xtc
    write_text = Path.write_text

    def write_xtc(self, filename):
        if Path(filename).name == interruption:
            raise OSError("interrupted")
        return save_xtc(self, filename)

    def write_report(self, data, *args, **kwargs):
        if self.name == interruption:
            raise OSError("interrupted")
        return write_text(self, data, *args, **kwargs)

    failure = (
        patch.object(sampling_api, "save_topology_cif", side_effect=OSError("interrupted"))
        if interruption == "topology"
        else patch.object(Path, "write_text", write_report)
        if interruption == "physicality.json"
        else patch.object(mdtraj.Trajectory, "save_xtc", write_xtc)
    )
    with failure, pytest.raises(OSError, match="interrupted"):
        runner.invoke(cli.main, args, catch_exceptions=False)
    saved = np.concatenate([np.load(path) for path in sorted(partial_dir.glob("*.npy"))])

    with patch.object(
        EDMSampler, "__call__", side_effect=AssertionError("Completed batches must be reused")
    ):
        resumed = runner.invoke(cli.main, args, catch_exceptions=False)
    assert resumed.exit_code == 0
    assert not partial_dir.exists()
    samples = mdtraj.load_xtc(str(out / "samples.xtc"), top=str(out / "topology.cif"))
    assert samples.n_frames == 5
    np.testing.assert_allclose(samples.xyz, saved / 10.0, rtol=0, atol=1e-3)
    assert [row["status"] for row in json.loads((out / "physicality.json").read_text())] == [
        "passed",
        "passed",
        "passed",
        "passed",
        "rejected",
    ]


@pytest.mark.parametrize("num_kept", [1, 0])
def test_cli_writes_rejected_samples(
    tmp_path: Path,
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    num_kept: int,
) -> None:
    """All samples retain their original frame indices, including rejected samples."""
    _mock_cli_msa(monkeypatch)
    monkeypatch.setattr(
        sampling_api,
        "check_physicality",
        lambda trajectory, mode, expected_extra_bonds: PhysicalityReport(
            passed=np.array([False] * (2 - num_kept) + [True] * num_kept),
            reasons=["clash"] * (2 - num_kept) + [None] * num_kept,
            mode=mode,
        ),
    )
    out = tmp_path / "out"
    args = [
        "sample-equilibrium",
        "--system",
        str(EXAMPLES_DIR / "chignolin.yaml"),
        "--checkpoint-path",
        str(tiny_checkpoint),
        "--out",
        str(out),
        "--num-samples",
        "2",
    ]

    result = CliRunner().invoke(cli.main, args, catch_exceptions=False)
    assert result.exit_code == 0
    assert f"{2 - num_kept} of 2 samples failed physicality checks." in result.output
    samples = mdtraj.load_xtc(str(out / "samples.xtc"), top=str(out / "topology.cif"))
    assert samples.n_frames == 2
    assert not (out / "samples_rejected.xtc").exists()
    assert not (out / "partial").exists()
    assert json.loads((out / "physicality.json").read_text()) == [
        {
            "frame": i,
            "status": "rejected" if i < 2 - num_kept else "passed",
            "reason": "clash" if i < 2 - num_kept else None,
        }
        for i in range(2)
    ]
    cif = mdtraj.load(str(out / "topology.cif"))
    np.testing.assert_allclose(cif.xyz[0], samples.xyz[0], atol=1e-3)


def _propagate_argv(tiny_checkpoint: Path, out: Path, x0_args: list[str]) -> list[str]:
    return (
        ["propagate", "--system", str(EXAMPLES_DIR / "chignolin.yaml")]
        + ["--checkpoint-path", str(tiny_checkpoint), "--out", str(out)]
        + [*x0_args, "--lag-time-ns", "1"]
    )


@pytest.mark.parametrize("reject_all", [False, True])
def test_propagate_reports_physicality_without_dropping_endpoints(
    tmp_path: Path,
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    x0_args: list[str],
    reject_all: bool,
) -> None:
    """All endpoints remain in order, even when all fail physicality checks."""
    _mock_cli_msa(monkeypatch)
    monkeypatch.setattr(cli, "PHYSICALITY_FRAME_CHUNK_SIZE", 1)
    checked = 0
    failure_reasons = ["bad_bonds_backbone" if reject_all else None, "within_residue_clashes"]

    def check(trajectory: mdtraj.Trajectory, mode, expected_extra_bonds) -> PhysicalityReport:
        nonlocal checked
        assert len(trajectory) == 1
        reason = failure_reasons[checked]
        checked += 1
        return PhysicalityReport(passed=np.array([reason is None]), reasons=[reason], mode=mode)

    monkeypatch.setattr(cli, "check_physicality", check)
    out = tmp_path / "out"
    args = _propagate_argv(tiny_checkpoint, out, x0_args)
    top = str(out / "topology.cif")

    result = CliRunner().invoke(cli.main, args, catch_exceptions=False)
    assert result.exit_code == 0
    assert checked == 2
    assert {p.name for p in out.iterdir() if p.suffix == ".xtc"} == {"samples.xtc"}
    endpoints = mdtraj.load_xtc(str(out / "samples.xtc"), top=top)
    assert endpoints.n_frames == 2
    assert endpoints.time.tolist() == [1000.0, 1000.0]
    assert json.loads((out / "physicality.json").read_text()) == [
        {
            "frame": 0,
            "status": "rejected" if reject_all else "passed",
            "reason": "bad_bonds_backbone" if reject_all else None,
        },
        {"frame": 1, "status": "rejected", "reason": "within_residue_clashes"},
    ]


def test_protein_free_propagation_reports_unchecked_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    system = System(entities=[DNA(sequence="ACGT", asym_ids=["D"])])
    topology = system.topology
    x0 = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)
    endpoints = mdtraj.Trajectory(np.zeros((2, topology.n_atoms, 3)), topology)
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setattr(cli, "PHYSICALITY_FRAME_CHUNK_SIZE", 1)

    with caplog.at_level(logging.INFO):
        cli._write_propagation_output(system, x0=x0, lag_time_ns=1.0, out=out, endpoints=endpoints)

    assert json.loads((out / "physicality.json").read_text()) == [
        {"frame": i, "status": "not_checked", "reason": None} for i in range(2)
    ]
    assert "2 unchecked endpoints" in caplog.text
    assert "passed physicality checks" not in caplog.text


@pytest.mark.parametrize("name", ["samples.xtc", "physicality.json", "samples_rejected.xtc"])
def test_propagate_refuses_existing_outputs(tmp_path: Path, name: str) -> None:
    out = tmp_path / "out"
    out.mkdir()
    existing = out / name
    existing.write_text("previous result")

    result = CliRunner().invoke(
        cli.main,
        [
            "propagate",
            "--system",
            "unused.yaml",
            "--x0-cif",
            "unused.cif",
            "--num-samples",
            "1",
            "--lag-time-ns",
            "1",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 2
    assert "Propagation output already exists" in result.output
    assert existing.read_text() == "previous result"
    assert not (out / "command.txt").exists()


@pytest.mark.parametrize("num_samples", [1, 3])
def test_propagate_from_cif_repeats_one_start(
    tmp_path: Path,
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    x0_args: list[str],
    num_samples: int,
) -> None:
    _mock_cli_msa(monkeypatch)
    out = tmp_path / "out"
    args = _propagate_argv(tiny_checkpoint, out, x0_args)
    args = [arg for arg in args if arg not in ("--x0-xtc", x0_args[3])]
    args += ["--physicality-filter", "no_filter", "--num-samples", str(num_samples)]
    result = CliRunner().invoke(cli.main, args, catch_exceptions=False)
    assert result.exit_code == 0
    output = mdtraj.load_xtc(str(out / "samples.xtc"), top=str(out / "topology.cif"))
    assert output.n_frames == num_samples
    assert output.time.tolist() == [1000.0] * num_samples
    assert [row["status"] for row in json.loads((out / "physicality.json").read_text())] == [
        "not_checked"
    ] * num_samples


@pytest.mark.parametrize("with_xtc", [False, True])
def test_propagate_infers_system_from_cif(
    tmp_path: Path,
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    x0_args: list[str],
    with_xtc: bool,
) -> None:
    _mock_cli_msa(monkeypatch)
    out = tmp_path / "out"
    args = _propagate_argv(tiny_checkpoint, out, x0_args)
    args = [arg for arg in args if arg not in ("--system", str(EXAMPLES_DIR / "chignolin.yaml"))]
    if not with_xtc:
        args = [arg for arg in args if arg not in ("--x0-xtc", x0_args[3])]
        args += ["--num-samples", "3"]
    result = CliRunner().invoke(
        cli.main, [*args, "--physicality-filter", "no_filter"], catch_exceptions=False
    )

    assert result.exit_code == 0
    assert System.from_yaml(out / "system.yaml") == System.from_cif(Path(x0_args[1]))
    trajectory = mdtraj.load_xtc(str(out / "samples.xtc"), top=str(out / "topology.cif"))
    assert trajectory.n_frames == (2 if with_xtc else 3)
    assert [row["frame"] for row in json.loads((out / "physicality.json").read_text())] == list(
        range(trajectory.n_frames)
    )


@pytest.mark.parametrize("explicit_system", [False, True])
@pytest.mark.parametrize("num_samples", [None, 3])
def test_python_propagate_from_cif_matches_trajectory(
    tmp_path: Path,
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_system: bool,
    num_samples: int | None,
) -> None:
    system = System(
        entities=[
            Protein(sequence="AA", asym_ids=["A"]),
            *([Ligand(smiles="CCO", asym_ids=["B"])] if explicit_system else []),
        ]
    )
    cif = Path(_write_x0(system, tmp_path)[1])
    msa = MSA({"A": ">query\nAA\n"})
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    embeddings = model.embed(system, msa=msa, mode="equilibrium")
    x0 = map_cif_atom_order(system, cif)
    expected = propagate_with_embeddings(
        system,
        embeddings,
        x0=x0 if num_samples is None else x0[[0] * num_samples],
        lag_time_ns=1.0,
        model=model,
        sampler=EDMSampler(num_steps=2),
        seed=7,
    )
    monkeypatch.setattr(propagation_api, "get_msa", lambda system: msa)
    monkeypatch.setattr(
        propagation_api, "sampler_from_preset", lambda preset, *, mode: EDMSampler(num_steps=2)
    )
    actual = propagate(
        system if explicit_system else None,
        model=model,
        x0=cif if explicit_system else str(cif),
        lag_time_ns=1.0,
        num_samples=num_samples,
        sampler="edm",
        seed=7,
    )

    assert actual.n_frames == (num_samples or 1)
    assert actual.topology == expected.topology
    np.testing.assert_allclose(actual.xyz, expected.xyz, atol=1e-5)


@pytest.mark.parametrize("num_samples", [0, -1])
def test_python_propagate_rejects_invalid_num_samples(
    tiny_checkpoint: Path, num_samples: int
) -> None:
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        propagate(model=model, x0="unused.cif", lag_time_ns=1.0, num_samples=num_samples)


def test_python_propagate_single_frame_trajectory_num_samples_matches_repetition(
    tmp_path: Path, tiny_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = System(entities=[Protein(sequence="AA", asym_ids=["A"])])
    cif = Path(_write_x0(system, tmp_path)[1])
    x0 = map_cif_atom_order(system, cif)
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    msa = MSA({"A": ">query\nAA\n"})
    monkeypatch.setattr(propagation_api, "get_msa", lambda _system: msa)
    monkeypatch.setattr(
        propagation_api, "sampler_from_preset", lambda preset, *, mode: EDMSampler(num_steps=2)
    )

    actual = propagate(system, model=model, x0=x0, lag_time_ns=1.0, num_samples=3, seed=7)
    expected = propagate(system, model=model, x0=x0[[0] * 3], lag_time_ns=1.0, seed=7)

    assert actual.n_frames == 3
    np.testing.assert_allclose(actual.xyz, expected.xyz, atol=1e-5)


def test_python_propagate_rejects_num_samples_for_multi_frame_trajectory(
    tmp_path: Path, tiny_checkpoint: Path
) -> None:
    system = System(entities=[Protein(sequence="AA", asym_ids=["A"])])
    cif = Path(_write_x0(system, tmp_path)[1])
    x0 = map_cif_atom_order(system, cif)
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    with pytest.raises(ValueError, match="num_samples requires a CIF path or a single-frame"):
        propagate(system, model=model, x0=x0[[0, 0]], lag_time_ns=1.0, num_samples=2)


def test_python_simulate_trajectory_infers_system_from_cif(
    tmp_path: Path, tiny_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = System(entities=[Protein(sequence="AA", asym_ids=["A"])])
    cif = Path(_write_x0(system, tmp_path)[1])
    monkeypatch.setattr(propagation_api, "get_msa", lambda _system: MSA({"A": ">query\nAA\n"}))
    monkeypatch.setattr(
        propagation_api,
        "sampler_from_preset",
        lambda preset, *, mode: EDMSampler(num_steps=2),
    )
    monkeypatch.setattr(propagation_api, "_report_trajectory_physicality", lambda *_: None)

    trajectories = simulate_trajectory(
        model=load_model(checkpoint_path=tiny_checkpoint, device="cpu"),
        x0=str(cif),
        lag_time_ns=1.0,
        num_trajectory_steps=2,
        display_propagation_progress=False,
    )

    assert len(trajectories) == 1
    assert trajectories[0].n_frames == 3
    np.testing.assert_allclose(trajectories[0].xyz[0], map_cif_atom_order(system, cif).xyz[0])
    np.testing.assert_allclose(trajectories[0].time, [0.0, 1000.0, 2000.0])


def test_propagate_infers_ccd_ligand_from_cif(
    tmp_path: Path,
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = System.from_description(proteins=["AA"], ligand_ccd_codes=["AP5"])
    x0_args = _write_x0(system, tmp_path)
    monkeypatch.setattr(cli, "get_msa", lambda system: MSA({"A": ">query\nAA\n"}))
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli.main,
        [
            "propagate",
            "--x0-cif",
            x0_args[1],
            "--num-samples",
            "1",
            "--lag-time-ns",
            "1",
            "--checkpoint-path",
            str(tiny_checkpoint),
            "--num-steps",
            "2",
            "--physicality-filter",
            "no_filter",
            "--out",
            str(out),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert System.from_yaml(out / "system.yaml").entities == system.entities
    trajectory = mdtraj.load_xtc(str(out / "samples.xtc"), top=str(out / "topology.cif"))
    assert trajectory.n_frames == 1
    assert trajectory.n_atoms == system.topology.n_atoms
    assert np.isfinite(trajectory.xyz).all()


@pytest.mark.parametrize("explicit_system", [False, True])
def test_python_propagate_rejects_incomplete_cif(
    tmp_path: Path,
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_system: bool,
) -> None:
    system = System(entities=[Protein(sequence="AA", asym_ids=["A"])])
    cif = Path(_write_x0(system, tmp_path)[1])
    cif_file = pdbx.CIFFile.read(cif)
    atom_site = cif_file.block["atom_site"]
    for name in atom_site:
        atom_site[name] = pdbx.CIFColumn(atom_site[name].as_array(str)[1:])
    cif_file.write(cif)
    monkeypatch.setattr(
        propagation_api,
        "get_msa",
        lambda system: pytest.fail("MSA requested for incomplete CIF"),
    )

    with pytest.raises(ValueError, match="atoms are missing"):
        propagate(
            system if explicit_system else None,
            model=load_model(checkpoint_path=tiny_checkpoint, device="cpu"),
            x0=cif,
            lag_time_ns=1.0,
        )


def test_python_propagate_requires_system_for_trajectory(tiny_checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="system is required when x0 is a trajectory"):
        propagate(
            model=load_model(checkpoint_path=tiny_checkpoint, device="cpu"),
            x0=_two_frames(),
            lag_time_ns=1.0,
        )


def test_propagate_with_embeddings_preserves_custom_sampler_scale(tiny_checkpoint: Path) -> None:
    system, _ = load_system_and_contacts(EXAMPLES_DIR / "chignolin.yaml")
    model = load_model(checkpoint_path=tiny_checkpoint, device="cpu")
    msa = MSA({"A": ">query\nGYDPETGTWG\n"})
    xyz = np.random.default_rng(0).normal(size=(1, system.topology.n_atoms, 3)).astype(np.float32)
    x0 = mdtraj.Trajectory(xyz, system.topology)

    embeddings = model.embed(system, msa=msa, mode="equilibrium")
    endpoints = [
        propagate_with_embeddings(
            system,
            embeddings,
            x0=x0,
            lag_time_ns=1.0,
            model=model,
            sampler=EDMSampler(num_steps=2, step_scale=scale),
            seed=7,
            progress=False,
        )
        for scale in (1.0, 1.5)
    ]
    assert not np.array_equal(endpoints[0].xyz, endpoints[1].xyz)


def test_propagate_rejects_incomplete_inferred_system(tmp_path: Path) -> None:
    system = System.from_description(proteins=["AA"], ligand_smiles=["CCO"])
    x0_args = _write_x0(system, tmp_path)
    args = [
        "propagate",
        "--x0-cif",
        x0_args[1],
        "--num-samples",
        "1",
        "--lag-time-ns",
        "1",
        "--out",
        str(tmp_path / "out"),
    ]
    with pytest.raises(ValueError, match="--x0-cif atoms do not match the system"):
        CliRunner().invoke(cli.main, args, catch_exceptions=False)


def test_validate_x0_from_inferred_ligand_system(tmp_path: Path) -> None:
    system, _ = load_system_and_contacts(EXAMPLES_DIR / "myoglobin_101m.yaml")
    x0_args = _write_x0(system, tmp_path)
    inferred = System.from_cif(Path(x0_args[1]))

    x0 = cli.validate_x0(mdtraj.load(x0_args[1]), inferred)

    assert inferred.entities == system.entities
    assert x0.n_frames == 1
    assert x0.topology == system.topology


@pytest.mark.parametrize("name", ["chignolin", "myoglobin_101m"])
def test_validate_x0_from_cli_outputs(tmp_path: Path, name: str) -> None:
    """samples.xtc + topology.cif from a run reload onto the system topology (ligands included)."""
    system, _ = load_system_and_contacts(EXAMPLES_DIR / f"{name}.yaml")
    _write_x0(system, tmp_path)

    x0 = cli.validate_x0(
        mdtraj.load(str(tmp_path / "samples.xtc"), top=str(tmp_path / "topology.cif")), system
    )

    assert x0.n_frames == 2 and x0.topology == system.topology
    xtc = mdtraj.load_xtc(str(tmp_path / "samples.xtc"), top=system.topology)
    np.testing.assert_array_equal(x0.xyz, xtc.xyz)


def test_cif_matches_first_xtc_frame(tmp_path: Path) -> None:
    system, _ = load_system_and_contacts(EXAMPLES_DIR / "chignolin.yaml")
    _write_x0(system, tmp_path)
    first_frame = cli.validate_x0(mdtraj.load(str(tmp_path / "topology.cif")), system)
    frames = cli.validate_x0(
        mdtraj.load(str(tmp_path / "samples.xtc"), top=str(tmp_path / "topology.cif")), system
    )
    assert first_frame.n_frames == 1
    np.testing.assert_allclose(first_frame.xyz[0], frames.xyz[0], atol=1e-3)


def test_validate_x0_rejects_mismatched_atoms(tmp_path: Path) -> None:
    system, _ = load_system_and_contacts(EXAMPLES_DIR / "chignolin.yaml")
    _write_x0(system, tmp_path)
    x0 = mdtraj.load(str(tmp_path / "samples.xtc"), top=str(tmp_path / "topology.cif"))
    short = x0.atom_slice(range(x0.n_atoms - 1))
    short.save_cif(str(tmp_path / "short.cif"))
    short.save_xtc(str(tmp_path / "short.xtc"))
    with pytest.raises(ValueError, match=r"75 atoms, system has 76"):
        cli.validate_x0(
            mdtraj.load(str(tmp_path / "short.xtc"), top=str(tmp_path / "short.cif")), system
        )
    x0.topology.atom(3).name = "XX"
    x0.save_cif(str(tmp_path / "renamed.cif"))
    with pytest.raises(ValueError, match=r"atom 3 is \('XX'"):
        cli.validate_x0(
            mdtraj.load(str(tmp_path / "samples.xtc"), top=str(tmp_path / "renamed.cif")), system
        )


def test_subcommands_expose_only_their_arguments() -> None:
    runner = CliRunner()
    sampling_help = runner.invoke(cli.main, ["sample-equilibrium", "--help"])
    propagation_help = runner.invoke(cli.main, ["propagate", "--help"])

    assert sampling_help.exit_code == propagation_help.exit_code == 0
    sampling_options = {
        line.split()[0] for line in sampling_help.output.splitlines() if line.startswith("  --")
    }
    propagation_options = {
        line.split()[0] for line in propagation_help.output.splitlines() if line.startswith("  --")
    }
    assert "--num-samples" in sampling_options
    assert "--lag-time-ns" not in sampling_options
    assert "--lag-time-ns" in propagation_options
    assert "--num-samples" in propagation_options


def test_propagate_rejects_invalid_arguments() -> None:
    runner = CliRunner()
    base = [
        "propagate",
        "--system",
        "s.yaml",
        "--out",
        "o",
        "--x0-cif",
        "t.cif",
        "--x0-xtc",
        "x.xtc",
    ]
    for bad in (["--lag-time-ns", "0"], ["--lag-time-ns", "1", "--num-ito-steps", "2"], []):
        assert runner.invoke(cli.main, [*base, *bad]).exit_code != 0
    for args in ([*base, "--num-samples", "2"], base[:-2]):
        result = runner.invoke(cli.main, [*args, "--lag-time-ns", "1"])
        assert result.exit_code == 2
        assert "Pass exactly one of --x0-xtc and --num-samples" in result.output


def test_resolve_common_resolves_system_and_checkpoint() -> None:
    system, contacts, checkpoint = cli._resolve_common(
        system=EXAMPLES_DIR / "chignolin.yaml",
        pdb_id=None,
        checkpoint=None,
        checkpoint_path=None,
        checkpoint_cache_dir=None,
    )
    assert system.entities[0].sequence == "GYDPETGTWG" and contacts is None
    assert checkpoint == "bioemu2-preview"


def test_sampling_still_requires_system_or_pdb_id(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.main, ["sample-equilibrium", "--out", str(tmp_path / "out")])
    assert result.exit_code == 2
    assert "Provide exactly one of --system or --pdb-id" in result.output


def test_potentials_yaml_replaces_default_physicality_potentials(tmp_path: Path) -> None:
    """Custom YAML configures bonded bounds and replaces, rather than extends, the defaults."""
    yaml_path = tmp_path / "pots.yaml"
    yaml_path.write_text(
        "potentials:\n  - _target_: bioemu2.denoising.steering.potentials.BondLengthPotential\n"
        "    sigma_tolerance: 0.0\n"
        "    parameters: {guidance_weight: 0.05}\n"
    )
    sampler = cli._sampler("edm_gd", yaml_path)
    assert isinstance(sampler, GuidedFKSampler)
    assert [type(p).__name__ for p in sampler.potentials] == ["BondLengthPotential"]
    potential = sampler.potentials[0]
    potential.prepare(System(entities=[Protein(sequence="A", asym_ids="A")]))
    assert isinstance(potential, BondLengthPotential)
    torch.testing.assert_close(potential.feats["bond_lower"], potential.feats["bond_upper"])

    with pytest.raises(ValueError, match="requires a steering sampler"):
        cli._sampler("edm", yaml_path)
    yaml_path.write_text("- _target_: bioemu2.denoising.steering.potentials.BondLengthPotential\n")
    with pytest.raises(ValueError, match="top-level `potentials:`"):
        cli._sampler("edm_gd", yaml_path)


@pytest.mark.parametrize(
    "argv",
    [
        ["--num-samples", "0"],
        ["--num-steps", "-1"],
        ["--batch-size", "0"],
        ["--checkpoint", "bioemu2-preview", "--checkpoint-path", "x.ckpt"],
        ["--method", "md"],
        ["--sampler", "nope"],
        ["--physicality-filter", "strict"],
        ["--precision", "16-mixed"],
    ],
)
def test_cli_rejects(argv: list[str]) -> None:
    result = CliRunner().invoke(
        cli.main,
        [
            "sample-equilibrium",
            "--system",
            str(EXAMPLES_DIR / "chignolin.yaml"),
            "--out",
            "o",
            *argv,
        ],
    )
    assert result.exit_code != 0
