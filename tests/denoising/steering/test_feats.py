import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import constants
from rdkit import Chem

from bioemu2.data_interface import Ligand, Protein, System
from bioemu2.denoising.steering import feats
from bioemu2.denoising.steering.feats import Tracker, ligand_ccd_codes_from_system


@pytest.mark.parametrize("cache_hit", [True, False], ids=["cache-hit", "cache-miss"])
def test_load_ccd_prefers_cache_and_downloads_on_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_hit: bool
):
    """Use cached CCD without Hub access and download it only on a cache miss."""
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--boltz-community--boltz-1"
    revision = "1" * 40
    cached = repo / "snapshots" / revision / "ccd.pkl"
    cached.parent.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(revision)
    payload = pickle.dumps({"LIG": Chem.MolFromSmiles("CCO")})
    if cache_hit:
        cached.write_bytes(payload)

    def download_ccd(**kwargs):
        assert not cache_hit, "Cached CCD must not contact the Hub"
        cached.write_bytes(payload)
        return str(cached)

    monkeypatch.setattr(feats, "hf_hub_download", download_ccd)
    feats._load_ccd.cache_clear()
    try:
        ccd = feats._load_ccd()
    finally:
        feats._load_ccd.cache_clear()

    assert ccd["LIG"].GetNumAtoms() == 3


def _protein_only_task() -> System:
    return System(entities=[Protein(asym_ids=["A"], sequence="GYDPETGTWG")])


def _protein_plus_ligands_task() -> System:
    """Protein (chain 0), a 2-chain CCD ligand (chains 1, 2), a SMILES ligand (chain 3)."""
    return System(
        entities=[
            Protein(asym_ids=["A"], sequence="GYDPETGTWG"),
            Ligand(asym_ids=["B", "C"], ccd_code="SAM"),
            Ligand(asym_ids=["D"], smiles="CC1=CC=CC=C1"),
        ]
    )


def test_ligand_ccd_codes_empty_for_none():
    assert ligand_ccd_codes_from_system(None) == {}


def test_ligand_ccd_codes_empty_for_protein_only():
    assert ligand_ccd_codes_from_system(_protein_only_task()) == {}


def test_ligand_ccd_codes_maps_int_asym_id_and_skips_smiles():
    # Protein occupies asym_id 0; the SAM ligand spans 1 and 2; the SMILES
    # ligand at 3 has no CCD code and must be skipped.
    assert ligand_ccd_codes_from_system(_protein_plus_ligands_task()) == {1: "SAM", 2: "SAM"}


@pytest.mark.parametrize("gpu_count", [0, 1, 2], ids=["cpu", "cuda", "two-cuda-devices"])
def test_tracker_preserves_mixed_metrics_and_sparse_rows(tmp_path: Path, gpu_count: int):
    if torch.cuda.device_count() < gpu_count:
        pytest.skip(f"Requires {gpu_count} CUDA devices")
    matrix_device = "cuda:0" if gpu_count else "cpu"
    energy_device = "cuda:1" if gpu_count == 2 else matrix_device
    matrix = (
        torch.arange(6, dtype=torch.float64, device=matrix_device).reshape(2, 3).T.requires_grad_()
    )
    energies = torch.tensor([2, -4], device=energy_device)
    tracker = Tracker(tmp_path)
    tracker.register("matrix", (3, 3, 2))
    tracker.register("energies", (3, 2))
    tracker.register("beta", (3,))
    tracker.register("sigma", (3,))
    for step in range(3):
        values = {
            "matrix": matrix + step,
            "beta": torch.tensor(step + 0.5, device=matrix_device),
            "sigma": torch.tensor(3 - step),
        }
        if step != 1:
            values["energies"] = energies + step
        tracker.log(values)

    with pytest.raises(ValueError, match="already registered"):
        tracker.register("matrix", (3, 3, 2))
    np.testing.assert_array_equal(
        np.load(tmp_path / "matrix.npy"),
        np.stack([np.arange(6).reshape(2, 3).T + step for step in range(3)]),
    )
    np.testing.assert_array_equal(np.load(tmp_path / "energies.npy"), [[2, -4], [4, -2], [0, 0]])
    np.testing.assert_array_equal(np.load(tmp_path / "beta.npy"), [0.5, 1.5, 2.5])
    np.testing.assert_array_equal(np.load(tmp_path / "sigma.npy"), [3, 2, 1])
