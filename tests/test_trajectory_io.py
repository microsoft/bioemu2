import json

import mdtraj
import numpy as np
import pytest
import torch
from biotite.structure.io import load_structure, pdbx

from bioemu2 import check_physicality, save_frames
from bioemu2.data_interface import Ligand, Protein, System
from bioemu2.sampling import save_topology_cif
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.cif import save_structure_to_cif
from bioemu2.trajectory_io import map_cif_atom_order


def test_save_trajectory_writes_reloadable_xtc_and_cif(tmp_path) -> None:
    topology = mdtraj.Topology()
    for chain_id, res_seq in (("Z9", 40), ("Y8", 80)):
        chain = topology.add_chain(chain_id=chain_id)
        residue = topology.add_residue("GLY", chain, resSeq=res_seq)
        topology.add_atom("CA", mdtraj.element.carbon, residue)
    trajectory = mdtraj.Trajectory(
        xyz=np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]],
            ]
        ),
        topology=topology,
    )

    trajectory_path = save_frames(trajectory, tmp_path / "nested" / "samples")

    topology_path = trajectory_path.parent / "topology.cif"
    assert trajectory_path == tmp_path / "nested" / "samples" / "samples.xtc"
    assert topology_path.is_file()
    cif = pdbx.CIFFile.read(str(topology_path))
    atom_site = cif[next(iter(cif.keys()))]["atom_site"]
    expected_chain_ids = ["Z9", "Y8"]
    assert atom_site["label_asym_id"].as_array(str).tolist() == expected_chain_ids
    assert atom_site["auth_asym_id"].as_array(str).tolist() == expected_chain_ids
    assert atom_site["label_seq_id"].as_array(str).tolist() == ["40", "80"]
    assert atom_site["pdbx_formal_charge"].as_array(str).tolist() == ["?", "?"]
    reloaded = mdtraj.load_xtc(str(trajectory_path), top=str(topology_path))
    np.testing.assert_allclose(reloaded.xyz, trajectory.xyz, atol=1e-3)


def test_save_mdtraj_cif_preserves_known_charges_and_marks_unknown(tmp_path) -> None:
    topology = mdtraj.Topology()
    for chain_id, element, charge in (
        ("A", mdtraj.element.sodium, 1),
        ("B", mdtraj.element.chlorine, None),
    ):
        chain = topology.add_chain(chain_id=chain_id)
        residue = topology.add_residue(element.symbol.upper(), chain)
        topology.add_atom(element.symbol.upper(), element, residue, formal_charge=charge)
    trajectory = mdtraj.Trajectory(np.zeros((1, 2, 3)), topology)

    save_frames(trajectory, tmp_path)

    atom_site = pdbx.CIFFile.read(str(tmp_path / "topology.cif")).block["atom_site"]
    assert atom_site["pdbx_formal_charge"].as_array(str).tolist() == ["1", "?"]


def test_save_trajectory_rejects_empty_trajectory(tmp_path) -> None:
    trajectory = mdtraj.Trajectory(
        xyz=np.empty((0, 0, 3)),
        topology=mdtraj.Topology(),
    )
    output_dir = tmp_path / "samples"

    with pytest.raises(ValueError, match="at least one frame"):
        save_frames(trajectory, output_dir)

    assert not output_dir.exists()


def test_save_trajectory_rejects_trajectory_without_atoms(tmp_path) -> None:
    trajectory = mdtraj.Trajectory(
        xyz=np.empty((1, 0, 3)),
        topology=mdtraj.Topology(),
    )
    output_dir = tmp_path / "samples"

    with pytest.raises(ValueError, match="at least one atom"):
        save_frames(trajectory, output_dir)

    assert not output_dir.exists()


def test_save_frames_saves_unchecked_physicality_and_rejects_mismatched_report(tmp_path) -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    samples = mdtraj.Trajectory(np.zeros((2, system.topology.n_atoms, 3)), system.topology)
    report = check_physicality(samples, mode="no_filter")
    output_dir = tmp_path / "samples"
    save_frames(samples, output_dir, physicality=report)
    assert json.loads((output_dir / "physicality.json").read_text()) == [
        {"frame": frame, "status": "not_checked", "reason": None} for frame in range(2)
    ]

    invalid_dir = tmp_path / "invalid"
    with pytest.raises(ValueError, match="one result per trajectory frame"):
        save_frames(samples[:1], invalid_dir, physicality=report)
    assert not invalid_dir.exists()


def test_map_cif_atom_order_normalizes_ids_and_reorders_atoms(tmp_path) -> None:
    system = System(
        entities=[
            Protein(sequence="AC", asym_ids=["A"]),
            Protein(sequence="DE", asym_ids=["B"]),
        ]
    )
    structure = AllAtomStructure.from_system(system)
    coords = torch.arange(structure.coords.numel(), dtype=structure.coords.dtype).reshape(
        structure.coords.shape
    )
    source_path = tmp_path / "topology.cif"
    ref_path = tmp_path / "shifted.cif"
    save_structure_to_cif(
        structure.replace(coords=coords),
        source_path,
        renamed_asym_ids=system.chain_ids,
    )

    ref_structure = load_structure(str(source_path), use_author_fields=False)
    chain_b_mask = ref_structure.chain_id == "B"
    chain_b_res_ids = sorted(np.unique(ref_structure.res_id[chain_b_mask]).tolist())
    for index, res_id in enumerate(chain_b_res_ids):
        residue_mask = chain_b_mask & (ref_structure.res_id == res_id)
        ref_structure.res_id[residue_mask] = 11 + 2 * index
    ref_structure = ref_structure[np.arange(len(ref_structure) - 1, -1, -1)]
    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, ref_structure)
    cif_file.write(str(ref_path))

    trajectory = map_cif_atom_order(system, ref_path)

    assert trajectory.xyz.shape == (1, structure.num_nodes, 3)
    np.testing.assert_allclose(trajectory.xyz[0], coords.numpy() / 10.0, atol=1e-5)

    incomplete_structure = ref_structure[
        ~((ref_structure.chain_id == "B") & (ref_structure.res_id == 13))
    ]
    original_res_ids = incomplete_structure.res_id.copy()
    with pytest.raises(ValueError, match="atoms are missing"):
        map_cif_atom_order(system, incomplete_structure)
    np.testing.assert_array_equal(incomplete_structure.res_id, original_res_ids)


@pytest.mark.parametrize("ligand_definition", [{"smiles": "O"}, {"inchi": "InChI=1S/H2O/h1H2"}])
def test_map_cif_atom_order_accepts_exported_ligand_name(tmp_path, ligand_definition) -> None:
    system = System(
        entities=[
            Protein(sequence="AA", asym_ids=["A"]),
            Ligand(asym_ids=["B"], **ligand_definition),
        ]
    )
    structure = AllAtomStructure.from_system(system)
    coords = torch.arange(structure.coords.numel(), dtype=structure.coords.dtype).reshape(
        structure.coords.shape
    )
    path = tmp_path / "topology.cif"
    save_topology_cif(structure, system, coords, path)
    assert "LIG" in pdbx.CIFFile.read(str(path))["topology"]["atom_site"]["label_comp_id"].as_array(
        str
    )

    trajectory = map_cif_atom_order(system, path)
    np.testing.assert_allclose(trajectory.xyz[0], coords.numpy() / 10.0, atol=1e-5)


def test_map_cif_atom_order_rejects_relabelled_ccd_ligand(tmp_path) -> None:
    system = System(
        entities=[
            Protein(sequence="AA", asym_ids=["A"]),
            Ligand(ccd_code="ZN", asym_ids=["B"]),
        ]
    )
    structure = AllAtomStructure.from_system(system)
    path = tmp_path / "topology.cif"
    save_topology_cif(structure, system, structure.coords, path)
    ref_structure = load_structure(str(path), use_author_fields=False)
    ref_structure.res_name[ref_structure.chain_id == "B"] = "LIG"

    with pytest.raises(ValueError, match="name LIG.*full chain residue name ZN"):
        map_cif_atom_order(system, ref_structure)
