import json
import shutil
import time
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import mdtraj
import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from bioemu2 import (
    MSA,
    Contact,
    EDMSampler,
    GuidedFKSampler,
    Protein,
    System,
    check_physicality,
)
from bioemu2.biomoa.models.score_model import TrunkEmbeddings as SerialTrunkEmbeddings
from bioemu2.context_parallel import api as cp
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.modules.diffusion import CPStaticContext
from bioemu2.denoising import cp_sync
from bioemu2.denoising.steering.potentials import BondLengthPotential
from bioemu2.model import _InferenceModel, load_model_for_inference
from bioemu2.sampling import generate_samples
from bioemu2.structure import AllAtomStructure, BatchAllAtomStructure
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.preparation import prepare_structure
from tests.context_parallel.helpers import spawn_cp_test, spawn_cpu_cp_test


def run_model_loading_cpu(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    manager: DistributedManager,
    *,
    directory: Path,
) -> None:
    cache = directory / f"node-{rank // 2}"

    def download(*, repo_id: str, filename: str, cache_dir: Path) -> str:
        assert manager.local_rank == 0, "A nonzero local rank reached a cold cache"
        # Delay the second node so incorrect global-rank-zero ordering exposes the race.
        if rank == 2:
            time.sleep(0.5)
        target = Path(cache_dir) / "models--microsoft--bioemu2/snapshots" / ("0" * 40) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / Path(filename).name, target)
        return str(target)

    with patch("bioemu2.checkpoints.hf_hub_download", side_effect=download):
        model = cp.load_model(
            "bioemu2-preview",
            cache_dir=cache,
            precision="32-true",
            matmul_precision=None,
        )
    assert model.device == torch.device("cpu")
    assert model.process_group is dist.group.WORLD
    parameter = next(model.score_model.parameters()).detach()
    replicas = [torch.empty_like(parameter) for _ in range(dist.get_world_size())]
    dist.all_gather(replicas, parameter)
    for replica in replicas:
        torch.testing.assert_close(parameter, replica, rtol=0, atol=0)
    for node in range(2):
        for name in ("checkpoint.ckpt", "config.yaml"):
            cached = (
                directory
                / f"node-{node}/models--microsoft--bioemu2/snapshots"
                / ("0" * 40)
                / "checkpoints/bioemu2-preview"
                / name
            )
            assert cached.read_bytes() == (directory / name).read_bytes()


def run_api_on_gpu(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    manager: DistributedManager,
    *,
    checkpoint: Path,
) -> None:
    system = System(entities=[Protein(sequence="GYDPETGTWG", asym_ids=["A"])])
    msa = MSA({"A": ">query\nGYDPETGTWG\n"})
    contacts = [Contact(token1={"asym_id": "A", "res_id": 1}, token2={"asym_id": "A", "res_id": 5})]
    model = replace(
        cp.load_model(checkpoint_path=checkpoint, precision="32-true", matmul_precision=None),
        num_steps_recycle=0,
    )
    real_batch_collate = cp.batch_collate
    received_peaks = []

    def measure_received_structure(
        structures: list[AllAtomStructure], window_size: int
    ) -> BatchAllAtomStructure:
        torch.cuda.synchronize(device)
        received_peaks.append(torch.cuda.max_memory_allocated(device))
        return real_batch_collate(structures, window_size=window_size)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    with patch.object(cp, "batch_collate", side_effect=measure_received_structure):
        trunk_embeddings = model.embed(
            system,
            msa=msa if dist.get_rank() == 0 else None,
            mode="equilibrium",
            contacts=contacts,
        )
    # Preparing and broadcasting the structure (MSA, dense contacts) must stay in host memory.
    assert received_peaks == [baseline]
    _, sampling_batch = cp._prepare_sampling_inputs(system, model)
    serial = load_model_for_inference(
        checkpoint, checkpoint.parent / "config.yaml", model.device, 0, "32-true"
    )
    structure = prepare_structure(system, msa=msa, mode="equilibrium", contacts=contacts)
    serial_batch = batch_collate([structure], window_size=sampling_batch.coords.shape[1]).to(
        model.device
    )
    assert isinstance(serial, _InferenceModel)
    with torch.no_grad():
        serial_trunk = serial._run_trunk(serial_batch)
    assert isinstance(serial_trunk, SerialTrunkEmbeddings)
    sampling_structure = AllAtomStructure.from_system(system)
    sampling_structure = sampling_structure.replace(
        coords=torch.zeros_like(sampling_structure.coords)
    )
    expected_batch = batch_collate(
        [sampling_structure], window_size=sampling_batch.coords.shape[1]
    ).to(model.device)
    # CP uses index 0 for padded atoms; atom masks keep them out of the computation.
    expected_batch.token_idx_at_atom = expected_batch.token_idx_at_atom.clamp_min(0)
    for key, expected in expected_batch.items():
        if key not in sampling_batch:
            continue
        actual = sampling_batch[key]
        if isinstance(expected, torch.Tensor):
            slices = tuple(slice(0, size) for size in expected.shape)
            torch.testing.assert_close(actual[slices], expected, msg=key)
        else:
            assert actual == expected
    changed_batch = sampling_batch.replace(
        method_feature=serial_batch.method_feature.flip(-1),
        contact_conditioning=serial_batch.contact_conditioning,
        contact_threshold=serial_batch.contact_threshold,
    )
    # Pair inputs only feed the trunk; sampling batches must not carry dense (B, N, N, *) tensors.
    for key in ("method_feature", "token_bonds", "contact_conditioning", "contact_threshold"):
        assert key not in sampling_batch, key
    with patch.object(
        model.score_model,
        "compute_static_context",
        wraps=model.score_model.compute_static_context,
    ) as compute_static_context:
        for sampler in (
            EDMSampler(num_steps=2),
            GuidedFKSampler(
                num_steps=2,
                num_gd_steps=1,
                resampling_interval=None,
                potentials=(
                    BondLengthPotential(
                        sigma_tolerance=10.0,
                        parameters={
                            "guidance_weight": 0.01,
                            "guidance_interval": 1,
                        },
                    ),
                ),
            ),
            GuidedFKSampler(
                num_steps=2,
                num_gd_steps=0,
                resampling_interval=1,
                potentials=(
                    BondLengthPotential(
                        sigma_tolerance=10.0, parameters={"resampling_weight": 0.01}
                    ),
                ),
            ),
        ):
            samples = cp.sample_with_embeddings(
                system,
                trunk_embeddings,
                model=model,
                num_samples=3,
                batch_size=2,
                sampler=sampler,
                seed=42,
                progress=False,
            )
            repeated = cp.sample_with_embeddings(
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
            assert samples.n_atoms == system.topology.n_atoms
            assert np.isfinite(samples.xyz).all()
            np.testing.assert_array_equal(samples.xyz, repeated.xyz)
            with cp._rng_seed(42, model.device), torch.no_grad():
                changed_samples = sampler(
                    batch=changed_batch,
                    score_model=model,
                    trunk_embeddings=trunk_embeddings,
                    batch_size=2,
                ).coords[:, : system.topology.n_atoms]
            changed_samples = changed_samples - changed_samples.mean(dim=1, keepdim=True)
            np.testing.assert_array_equal(samples.xyz[:2], changed_samples.cpu().numpy() / 10)
            # Match padding and batch sizes to compare identical RNG draws.
            # Override only the sampler's CP-mode check for the serial reference;
            # wrapped helpers retain real RNG synchronization and broadcasts.
            with (
                cp._rng_seed(42, model.device),
                patch(
                    "bioemu2.denoising.steering.edm_steered.cp_sync",
                    wraps=cp_sync,
                    spec_set=cp_sync,
                ) as serial_sync,
            ):
                serial_sync.is_distributed_sampling.return_value = False
                reference = generate_samples(
                    num_samples=3,
                    batch_size=2,
                    score_model=serial,
                    sampler=sampler,
                    num_atoms=system.topology.n_atoms,
                    system=system,
                    batch=sampling_batch,
                    trunk_embeddings=serial_trunk,
                    progress=False,
                )
            np.testing.assert_allclose(
                samples.xyz, reference.cpu().numpy() / 10, atol=1e-4, rtol=1e-4
            )
            replicas = [None] * dist.get_world_size()
            dist.all_gather_object(replicas, samples.xyz)
            for replica in replicas:
                np.testing.assert_allclose(samples.xyz, replica, atol=1e-6, rtol=1e-6)
        assert compute_static_context.call_count == 15
    write_context = (
        nullcontext()
        if rank == 0
        else patch.object(
            Path, "write_text", side_effect=PermissionError("Only rank zero may write outputs")
        )
    )
    with write_context:
        predicted = cp.predict_structure(
            system,
            model=model,
            msa=msa if rank == 0 else None,
            num_samples=3,
            batch_size=2,
            num_steps=2,
            seed=4,
            progress=False,
            output_dir=checkpoint.parent / "predicted",
        )
        assert predicted.n_frames == 3
        equilibrium = cp.sample_equilibrium(
            system,
            model=model,
            msa=msa if rank == 0 else None,
            num_samples=1,
            num_steps=2,
            seed=4,
            progress=False,
            output_dir=checkpoint.parent / "equilibrium",
            physicality_mode="no_filter",
        )
        assert equilibrium.n_frames == 1
    if rank == 0:
        for name, samples in (("predicted", predicted), ("equilibrium", equilibrium)):
            mode = "high_quality" if name == "predicted" else "no_filter"
            report = check_physicality(samples, mode=mode, system=system)
            output = checkpoint.parent / name
            reloaded = mdtraj.load_xtc(
                str(output / "samples.xtc"), top=str(output / "topology.cif")
            )
            np.testing.assert_allclose(reloaded.xyz, samples.xyz, rtol=0, atol=1e-3)
            assert json.loads((output / "physicality.json").read_text()) == [
                {
                    "frame": frame,
                    "status": (
                        "not_checked"
                        if mode == "no_filter"
                        else "passed"
                        if report.passed[frame]
                        else "rejected"
                    ),
                    "reason": report.reasons[frame],
                }
                for frame in range(samples.n_frames)
            ]
            assert (
                f"mode: {'structure' if name == 'predicted' else 'equilibrium'}"
                in (output / "system.yaml").read_text()
            )
    failure_context = (
        patch.object(Path, "write_text", side_effect=OSError("CP output failure"))
        if rank == 0
        else nullcontext()
    )
    with failure_context, pytest.raises(RuntimeError, match="CP output failure"):
        cp.sample_equilibrium(
            system,
            model=model,
            msa=msa if rank == 0 else None,
            num_samples=1,
            num_steps=2,
            progress=False,
            output_dir=checkpoint.parent / "output-failure",
        )
    for dtype in (None, torch.bfloat16):
        precision_model = replace(model, _autocast_dtype=dtype)
        trunk = precision_model.embed(
            system,
            msa=msa if dist.get_rank() == 0 else None,
            mode="equilibrium",
            contacts=contacts,
        )
        trunk = replace(
            trunk,
            single=trunk.single.requires_grad_(),
            pair=trunk.pair.requires_grad_(),
        )
        with torch.enable_grad():
            context = precision_model.compute_static_context(sampling_batch, trunk, 2)
        assert isinstance(context, CPStaticContext)
        assert context.z.dtype == torch.float32
        for value in (
            context.z,
            context.atom_encoder_context.q,
            context.atom_encoder_context.c,
            context.atom_encoder_context.p,
        ):
            assert not value.requires_grad
        with torch.no_grad():
            expected_z = model.score_model.structure_module.pairwise_conditioner(
                trunk.pair.float(), trunk.relative_position_encoding.float()
            )
        torch.testing.assert_close(context.z.to_local(), expected_z.to_local(), rtol=0, atol=0)
        samples = cp.sample_with_embeddings(
            system,
            trunk,
            model=precision_model,
            num_samples=2,
            sampler=EDMSampler(num_steps=2),
            seed=42,
            progress=False,
        )
        assert samples.n_frames == 2
        assert np.isfinite(samples.xyz).all()


def test_multi_node_model_loading(checkpoint: Path) -> None:
    # Four ranks on two simulated nodes of two ranks each.
    spawn_cpu_cp_test(
        partial(run_model_loading_cpu, directory=checkpoint.parent),
        local_world_size=2,
    )


@pytest.mark.gpu
@pytest.mark.requires_4gpu
def test_collective_sampling_api(checkpoint: Path) -> None:
    """Real checkpoint, CP shards and samplers; no network required."""
    spawn_cp_test(partial(run_api_on_gpu, checkpoint=checkpoint))
