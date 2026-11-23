import mdtraj
import numpy as np
import pytest

from bioemu2.visualization import view_structure, view_structure_grid, view_trajectory


def _backbone_trajectory(offsets: list[float]) -> mdtraj.Trajectory:
    """One glycine backbone (N, CA, C), translated along x by each offset in turn.

    Superposition needs more than one atom: a single atom centres onto the origin, which
    mdtraj >= 1.11.2 rejects as an overflow in `Trajectory.superpose`.
    """
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("GLY", chain)
    topology.add_atom("N", mdtraj.element.nitrogen, residue)
    topology.add_atom("CA", mdtraj.element.carbon, residue)
    topology.add_atom("C", mdtraj.element.carbon, residue)
    base_coordinates = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    return mdtraj.Trajectory(
        xyz=np.array([base_coordinates + np.array([offset, 0.0, 0.0]) for offset in offsets]),
        topology=topology,
    )


def test_view_structure_embeds_selected_frame() -> None:
    trajectory = _backbone_trajectory([0.0, 0.1])

    html = view_structure(trajectory, frame=1)._make_html()

    assert "3Dmol.createViewer" in html
    assert "_atom_site.Cartn_x" in html
    assert html.count("ATOM") == 3


def test_view_structure_rejects_unknown_frame() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("GLY", chain)
    topology.add_atom("CA", mdtraj.element.carbon, residue)
    trajectory = mdtraj.Trajectory(xyz=np.zeros((1, 1, 3)), topology=topology)

    with pytest.raises(IndexError, match="outside a trajectory"):
        view_structure(trajectory, frame=1)


def test_view_trajectory_embeds_bounded_aligned_frames() -> None:
    trajectory = _backbone_trajectory([0.0, 1.0, 2.0, 3.0])

    html = view_trajectory(trajectory, max_frames=3)._make_html()

    assert html.count("MODEL") == 3
    assert "animate" in html


def test_view_structure_grid_embeds_aligned_frames() -> None:
    trajectory = _backbone_trajectory([float(offset) for offset in range(8)])

    html = view_structure_grid(trajectory, rows=2, columns=4)._make_html()

    assert "createViewerGrid" in html
    assert html.count("ATOM") == 24
    assert "70.000" not in html
    assert '"hetflag": false' in html
    assert html.count("addSurface") == 8
    assert '"color": "#0078d4"' in html
    assert '"opacity": 0.5' in html


def test_view_structure_grid_requires_one_frame_per_cell() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("GLY", chain)
    topology.add_atom("CA", mdtraj.element.carbon, residue)
    trajectory = mdtraj.Trajectory(xyz=np.zeros((2, 1, 3)), topology=topology)

    with pytest.raises(ValueError, match="2x4 grid requires 8 frames"):
        view_structure_grid(trajectory, rows=2, columns=4)
