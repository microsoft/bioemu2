# %% [markdown]
# # Guide sampling toward a chosen protein shape
#
# BioEmu2 can guide sampling toward structures with chosen properties. This
# notebook asks for Chignolin structures whose ends are a target distance apart,
# then measures how closely the generated structures match that request.
#
# See the [steering guide](https://github.com/microsoft/bioemu2/blob/main/docs/steering.md)
# to choose a sampler and for steering methods and references.
#
# The example compares ordinary sampling with guided sampling and plots the
# distance distributions. It generates many structures so that the comparison
# is meaningful. One run took about 93 minutes on an RTX A6000 and used about
# 36 GiB of GPU memory with `batch_size` 250.
#
# For installation and hardware requirements, see the
# [notebook quickstart](https://github.com/microsoft/bioemu2/blob/main/README.md#notebook-quickstart).

# %% [markdown]
# ## Load the model and system
#
# Create an equilibrium-sampling session for Chignolin. The target asks for the
# first and last Cα atoms to be 8 Å apart.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj
import numpy as np
import torch

from bioemu2 import (
    EDMSampler,
    EmuSession,
    FKCSampler,
    Potential,
    System,
    load_model,
)
from bioemu2.paths import EXAMPLES_DIR

model = load_model("bioemu2-preview")

target_distance_angstrom = 8.0
chignolin = System.from_yaml(EXAMPLES_DIR / "systems/chignolin.yaml")
emu = EmuSession(chignolin, model=model, mode="equilibrium")


# %% [markdown]
# ## Define the measurement and the steering potential
#
# Measure each structure's Cα–Cα distance between the termini. The potential
# assigns a quadratic energy penalty to deviations from the target distance;
# steering favours structures with lower energy.

# %%
def terminal_distances_angstrom(samples: mdtraj.Trajectory) -> np.ndarray:
    ca = [atom.index for atom in samples.topology.atoms if atom.name == "CA"]
    displacement = samples.xyz[:, ca[0]] - samples.xyz[:, ca[-1]]
    return np.linalg.norm(displacement, axis=-1) * 10.0


# %%
class TerminiTargetExpPotential(Potential):
    """Energy = well * (Cα-Cα distance - target)², with distances in Å."""

    def __init__(self, target_angstrom: float) -> None:
        self.target_angstrom = target_angstrom
        self.well = 2
        self.atom_indices: tuple[int, int] | None = None

    def prepare(self, system) -> None:
        ca_indices = [atom.index for atom in system.topology.atoms if atom.name == "CA"]
        self.atom_indices = (ca_indices[0], ca_indices[-1])

    def __call__(self, coords_angstrom: torch.Tensor, *, t: float) -> torch.Tensor:
        start, end = self.atom_indices
        distance = torch.linalg.vector_norm(
            coords_angstrom[:, start] - coords_angstrom[:, end], dim=-1
        )
        return self.well * (distance - self.target_angstrom).pow(2)


# %% [markdown]
# ## Sample with and without steering
#
# First sample without steering. Reweighting those samples by the potential's
# Boltzmann factor gives the target tilted distribution without new sampling.
# Then sample with `FKCSampler`, which should reproduce that tilted distribution.
#
# Each call saves its results and completed batches in its `output_dir`. If
# sampling is interrupted, rerun the cell: completed batches are reused.

# %%
NUM_TILT_SAMPLES = 2000
TILT_BATCH_SIZE = 250
tilt_run_dir = Path("outputs/tilt_a_sample_distribution_with_steering")

chignolin_unsteered_large: mdtraj.Trajectory = emu.sample(
    num_samples=NUM_TILT_SAMPLES,
    batch_size=TILT_BATCH_SIZE,
    sampler=EDMSampler(),
    seed=42,
    output_dir=tilt_run_dir / "unsteered",
)
unsteered_d: np.ndarray = terminal_distances_angstrom(chignolin_unsteered_large)

# Tilted reference: reweight the *unsteered* samples by exp(-U(d)). No new sampling.
# `tilt_potential.well` is read back rather than duplicated, so the reference always
# matches whatever energy `TerminiTargetExpPotential` actually applies below.
tilt_potential: TerminiTargetExpPotential = TerminiTargetExpPotential(target_distance_angstrom)
tilt_weights: np.ndarray = np.exp(
    -tilt_potential.well * (unsteered_d - target_distance_angstrom) ** 2
)
tilt_weights /= tilt_weights.sum()

# The sampling batch itself is the resampling population.
# The "guidance" weight calculation requires reheat="sigma"; "gamma" is rejected.
# See docs/steering.md#configure-fkc-fkcsampler.
guidance_sampler: FKCSampler = FKCSampler(
    potentials=[tilt_potential],
    ess_threshold=0.8,
    weight_update="guidance",
    reheat="sigma",
)
chignolin_tilt_steered: mdtraj.Trajectory = emu.sample(
    num_samples=NUM_TILT_SAMPLES,
    batch_size=TILT_BATCH_SIZE,
    sampler=guidance_sampler,
    seed=42,
    output_dir=tilt_run_dir / "steered",
)
steered_d: np.ndarray = terminal_distances_angstrom(chignolin_tilt_steered)

{
    "unsteered_mean_A": unsteered_d.mean(),
    "tilted_mean_A": np.average(unsteered_d, weights=tilt_weights),
    "steered_mean_A": steered_d.mean(),
}

# %% [markdown]
# ## Compare the distributions
#
# The steered histogram should follow the reweighted reference rather than the
# unsteered distribution.

# %%
fig, ax = plt.subplots(figsize=(6, 4))
bins = np.linspace(0, max(unsteered_d.max(), steered_d.max()), 60)
ax.hist(unsteered_d, bins=bins, density=True, alpha=0.45, color="#2a78d6", label="unsteered")
ax.hist(
    unsteered_d,
    bins=bins,
    weights=tilt_weights,
    density=True,
    histtype="step",
    linewidth=2,
    color="#eb6834",
    label="tilted, reweighted by exp(-well·(d-target)²)",
)
ax.hist(
    steered_d,
    bins=bins,
    density=True,
    alpha=0.45,
    color="#1baf7a",
    label=f"FKC steered ({guidance_sampler.weight_update})",
)
ax.axvline(target_distance_angstrom, color="black", linestyle="--", linewidth=1, label="target")
ax.set_xlabel("Cα–Cα termini distance (Å)")
ax.set_ylabel("density")
ax.set_xlim(3, 10)
ax.set_title(
    f"Chignolin termini distance, tilt exp(-{tilt_potential.well:g}·(d-{target_distance_angstrom:g})²)\n"
    f"(n={NUM_TILT_SAMPLES}, weight_update={guidance_sampler.weight_update})",
    fontsize=10,
)
ax.legend()
fig.tight_layout()
fig

# %% [markdown]
# ## Save the results
#
# Save the distances, reference weights, and plot.

# %%
tilt_run_dir.mkdir(parents=True, exist_ok=True)
np.savez(tilt_run_dir / "unsteered.npz", distances=unsteered_d, tilt_weights=tilt_weights)
np.savez(tilt_run_dir / "steered.npz", distances=steered_d)
fig.savefig(tilt_run_dir / "plot.png", dpi=150)
tilt_run_dir
