import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import constants
from rdkit import Chem

from bioemu2.data_interface import Ligand, Protein, System
from bioemu2.denoising.steering import feats
from bioemu2.denoising.steering.feats import Tracker
from bioemu2.denoising.steering.potentials import (
    ChiralAtomPotential,
    FeatsPotential,
    StereoBondPotential,
)
from bioemu2.structure import AllAtomStructure


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


_NAG_SMILES = "CC(=O)N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1O"
_NAG_INCHI = (
    "InChI=1S/C8H15NO6/c1-3(11)9-5-7(13)6(12)4(2-10)15-8(5)14"
    "/h4-8,10,12-14H,2H2,1H3,(H,9,11)/t4-,5-,6-,7-,8-/m1/s1"
)
_RETINAL_SMILES = "CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C=O)/C)/C"
_SULFOXIDE_SMILES = "C[S@@](=O)c1ccccc1"


@pytest.mark.parametrize(
    ("source", "cls", "index_key", "terms_per_copy"),
    [
        ({"smiles": _NAG_SMILES}, ChiralAtomPotential, "chiral_atom_index", 5),
        ({"inchi": _NAG_INCHI}, ChiralAtomPotential, "chiral_atom_index", 5),
        ({"smiles": _SULFOXIDE_SMILES}, ChiralAtomPotential, "chiral_atom_index", 0),
        ({"smiles": _RETINAL_SMILES}, StereoBondPotential, "stereo_bond_index", 4),
    ],
    ids=["smiles-chiral", "inchi-chiral", "lone-pair-chiral", "smiles-e-z"],
)
def test_smiles_and_inchi_ligands_get_stereo_terms_their_template_satisfies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: dict[str, str],
    cls: type[FeatsPotential],
    index_key: str,
    terms_per_copy: int,
):
    """Test that each copy of a SMILES/InChI ligand after a protein gets one term per
    specified stereocentre or E/Z bond, that the ligand's template satisfies them, and
    that the mirror image breaks chirality. Lone-pair stereocentres get no term.

    Template embedding is unseeded, so chiral cases use a zero buffer (a pure sign check)
    to pass for any valid conformer.
    """
    monkeypatch.setenv("BIOEMU2_TEMPLATE_CACHE", str(tmp_path))
    system = System(
        entities=[
            Protein(asym_ids=["A"], sequence="GYDPETG"),
            Ligand(asym_ids=["L", "M"], **source),
        ]
    )
    potential = cls(parameters={"buffer": 0.0 if cls is ChiralAtomPotential else 0.5236})
    potential.prepare(system)
    template_coords = AllAtomStructure.from_system(system, mode="structure").template_coords
    assert template_coords is not None
    coords = template_coords[None]

    assert potential.feats[index_key].shape[-1] == 2 * terms_per_copy
    assert potential(coords, t=0.0).item() == 0
    if cls is ChiralAtomPotential and terms_per_copy:
        assert potential(coords * torch.tensor([-1.0, 1.0, 1.0]), t=0.0).item() > 0


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
