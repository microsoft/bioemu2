"""Conversion and saving of trajectories and coordinates via MDTraj."""

from pathlib import Path

import mdtraj
import torch
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from bioemu2.data_interface import System
from bioemu2.physicality import PhysicalityReport
from bioemu2.structure.all_atom_structure import AllAtomStructure
from bioemu2.structure.ref_struct import (
    ensure_loaded_ref_struct,
    get_chain_residue_maps_from_atom_array,
)


def trajectory_to_coords_angstrom(
    trajectory: mdtraj.Trajectory,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert an MDTraj trajectory in nanometers to Torch coordinates in angstroms."""
    return 10 * torch.as_tensor(trajectory.xyz, device=device, dtype=dtype)


def coords_angstrom_to_trajectory(
    coords: torch.Tensor,
    topology: mdtraj.Topology,
) -> mdtraj.Trajectory:
    """Convert Torch coordinates in angstroms to an MDTraj trajectory in nanometers."""
    return mdtraj.Trajectory(
        xyz=(coords / 10.0).detach().cpu().numpy(),
        topology=topology,
    )


def save_mdtraj_cif_frame(
    trajectory: mdtraj.Trajectory,
    path: str | Path,
) -> None:
    """Write a one-frame CIF, preserving topology IDs and marking unknown charges."""
    if trajectory.n_frames != 1:
        raise ValueError("trajectory must contain exactly one frame to save a topology CIF.")
    keep_ids = all(chain.chain_id for chain in trajectory.topology.chains)
    trajectory.save_cif(path, keepIds=keep_ids)
    cif_file = pdbx.CIFFile.read(str(path))
    block = cif_file[next(iter(cif_file))]
    atom_site = block["atom_site"]
    charges = [
        "?" if atom.formal_charge is None else str(atom.formal_charge)
        for atom in trajectory.topology.atoms
    ]
    if len(charges) != len(atom_site["id"]):
        raise ValueError("Trajectory topology and saved CIF atom counts differ.")
    atom_site["pdbx_formal_charge"] = charges
    cif_file.write(str(path))


def save_frames(
    trajectory: mdtraj.Trajectory,
    output_dir: str | Path,
    *,
    physicality: PhysicalityReport | None = None,
) -> Path:
    """Save all frames and a topology CIF, optionally with a physicality report.

    Chain IDs are preserved when every chain has an ID. Pass the result of
    ``check_physicality`` as ``physicality`` to also write ``physicality.json``.
    The report must describe these frames in the same order.
    """
    if trajectory.n_frames == 0:
        raise ValueError("trajectory must contain at least one frame.")
    if trajectory.n_atoms == 0:
        raise ValueError("trajectory must contain at least one atom.")
    if physicality is not None and (
        len(physicality.passed) != trajectory.n_frames
        or len(physicality.reasons) != trajectory.n_frames
    ):
        raise ValueError("physicality report must contain one result per trajectory frame.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "samples.xtc"
    trajectory.save_xtc(trajectory_path)
    save_mdtraj_cif_frame(trajectory[0], output_dir / "topology.cif")
    if physicality is not None:
        physicality.save(output_dir / "physicality.json")
    return trajectory_path


def map_cif_atom_order(
    system: System,
    struct_or_fpath: AtomArray | Path | str,
) -> mdtraj.Trajectory:
    """Map the atom order of a structure to match the system's topology.

    This requires the chain names, residue name and order, and atom names to be
    consistent with the system's topology. All chains, residues, and atoms must
    be present. Chain order, residue indexing, and atom order need not match.

    Args:
        system: The system whose topology the structure should match.
        struct_or_fpath: The structure or path to the structure file to be mapped.

    Returns:
        An MDTraj trajectory with the atom order matching the system's topology.
    """

    ref_structure = ensure_loaded_ref_struct(struct_or_fpath).copy()
    # Input structure's residue IDs may contain author-numbering gaps despite continuous residues.
    # Compact by residue order, then reject missing target atoms via the atom-mask check below.
    for chain_id, residues in get_chain_residue_maps_from_atom_array(ref_structure).items():
        residue_id_mapping = {
            res_id: index for index, res_id in enumerate(sorted(residues), start=1)
        }
        chain_mask = ref_structure.chain_id == chain_id
        for res_id, normalized_res_id in residue_id_mapping.items():
            ref_structure.res_id[chain_mask & (ref_structure.res_id == res_id)] = normalized_res_id

    structure = AllAtomStructure.from_system(system, ref_struct_or_fpath=ref_structure)
    n_miss = (~structure.atom_mask).sum().item()
    if n_miss > 0:
        raise ValueError(f"{n_miss} atoms are missing or do not get correctly mapped onto.")
    return mdtraj.Trajectory(
        xyz=structure.coords.cpu().numpy()[None, :, :] / 10.0,
        topology=system.topology,
    )
