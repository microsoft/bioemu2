# %% [markdown]
# # Using protein-ligand contacts to bias complex predictions towards specific binding sites
#
# A ligand may have several plausible binding sites, and some proteins have
# multiple experimentally confirmed ligand-binding sites. A structure-prediction
# model may nevertheless favor one site and sample the alternatives less often.
#
# Protein-ligand contacts can bias complex predictions towards a chosen pocket.
# This example uses BioEmu2 to compare binding-site predictions for
# [human serum albumin (HSA)](https://www.uniprot.org/uniprotkb/P02768/entry)
# and ibuprofen. [PDB 2BXG](https://www.rcsb.org/structure/2BXG) resolves ibuprofen
# at two sites, Sudlow II and FA6, giving us two experimental reference poses.
#
# From the repository root, install the example's dependencies with
# `pip install ".[cuda,viz]"`, then run the cells in order or run
# `python examples/sample_protein_ligand_constraints.py`. See
# [installation](https://github.com/microsoft/bioemu2/blob/main/README.md#installation) for supported GPUs and CUDA versions.
# The model checkpoint and experimental reference are downloaded on first use.
# Run notebook cells from the repository root or `examples`.

# %%
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import requests
import torch
from IPython.display import Markdown, display

from bioemu2 import (
    Contact,
    EDMSampler,
    EmuSession,
    Ligand,
    MSA,
    Protein,
    System,
    check_physicality,
    load_model,
    save_frames,
)
from bioemu2.data_interface import ContactToken
from bioemu2.visualization import view_structure_grid

# %% [markdown]
# ## Set up the system and pocket-specific contacts
#
# Compare predictions without contacts against predictions conditioned on
# contacts targeting Sudlow II or FA6.
# All three conditions use mature HSA and one (S)-ibuprofen ligand (`IBP`).
#
# Each contact specifies two `ContactToken` endpoints and a maximum distance in Å.
# Protein residue tokens use Cα; ligand tokens name individual atoms. For
# example, the first Sudlow II contact is HSA A410–IBP L:O2 ≤ 8.5 Å.
# Distance bounds are 2BXG chain-A Cα–ligand distances plus 1 Å, rounded up
# to 0.5 Å, with a 4 Å minimum. Contacts provide soft conditioning rather
# than hard distance limits.
#
# `load_model()` downloads the default checkpoint automatically.
# Saved predictions are reused; choose an empty output directory to rerun.

# %%
here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if (here / "examples").is_dir():
    here = here / "examples"
seed = 42
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
outputs = here / "outputs" / "ibuprofen"
protein = Protein(
    sequence=(
        "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKL"
        "CTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHP"
        "YFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQ"
        "RFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVEN"
        "DEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHE"
        "CYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHP"
        "EAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADIC"
        "TLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
    ),
    asym_ids=["A"],
)
system = System([protein, Ligand(ccd_code="IBP", asym_ids=["L"])])
# Each tuple is (HSA residue, ibuprofen atom, maximum distance in angstroms).
contact_specs = {
    "unconditioned": [],
    "site_ii": [(410, "O2", 8.5), (411, "O1", 8.0), (414, "O1", 10.0), (434, "C4", 4.5)],
    "fa6": [(351, "O2", 8.5), (482, "O1", 4.5), (350, "C4", 5.5)],
}
contacts_by_case = {
    case: [
        Contact(
            token1=ContactToken(asym_id="A", res_id=res_id),
            token2=ContactToken(asym_id="L", res_id=1, atom_name=atom_name),
            max_distance=distance,
        )
        for res_id, atom_name, distance in specs
    ]
    for case, specs in contact_specs.items()
}
conditions = {
    "unconditioned": "Default\n(no contacts)",
    "site_ii": "Target\nSudlow II",
    "fa6": "Target\nFA6",
}
run_inference = not all(
    (outputs / case / name).is_file()
    for case in conditions for name in ("samples.xtc", "topology.cif")
)
print("Sampling three conditions" if run_inference else "Reusing saved predictions")
contact_table = [
    "| Condition | Protein residue | Ligand atom | Max distance (Å) |",
    "| :-- | :-- | :-- | --: |",
]
for case, contacts in contacts_by_case.items():
    label = conditions[case].replace("\n", " ")
    for contact in contacts:
        contact_table.append(
            f"| {label} | {contact.token1.asym_id}{contact.token1.res_id} | "
            f"{contact.token2.asym_id}:{contact.token2.atom_name} | "
            f"{contact.max_distance:g} |"
        )
display(Markdown("\n".join(contact_table)))

# %% [markdown]
# ## Run structure prediction
#
# > **Demo shortcut:** supply only the HSA query sequence to skip MSA search.
# > For real applications, omit `msa` to use BioEmu2's default MSA workflow.
#
# Create a separate `EmuSession` for each condition so its embeddings include
# the chosen contacts. Generate five structures per condition using the same
# seed and plain EDM, without steering, in structure-prediction mode.
#
# For repeatable runs, keep the checkpoint, GPU, software, and sampling settings
# fixed.
# %%
if run_inference:
    model = load_model(device="cuda", precision="bf16-mixed", matmul_precision="highest")
    query_only = MSA(unpaired_a3m={"A": f">query\n{protein.sequence}\n"})
    for case, contacts in contacts_by_case.items():
        emu = EmuSession(system, model=model, mode="structure", msa=query_only, contacts=contacts)
        samples = emu.sample(
            num_samples=5, batch_size=1, seed=seed,
            sampler=EDMSampler(num_steps=200, step_scale=1.5),
        )
        save_frames(samples, outputs / case, physicality=check_physicality(samples, system=system))
        emu.cache_clear()

# %% [markdown]
# ## Compare predictions with the experimental poses
#
# Use both 2BXG chain-A ibuprofen poses for evaluation, not as model inputs.
# Align predicted HSA to Cα atoms within 10 Å of each reference ligand, mapping
# residues by their 1-based mature-HSA residue numbers.
# Calculate ligand heavy-atom RMSD by atom name, **without aligning the ligand
# separately** or correcting for symmetry.
#
# Both panels show the **same predictions** against different reference pockets.
# Lower RMSD means closer agreement with the reference pose.
# Expect the default and Sudlow II targets to match Sudlow II, and the FA6
# target to match FA6.

# %%
def load_predictions(directory: Path) -> md.Trajectory:
    return md.load_xtc(str(directory / "samples.xtc"), top=str(directory / "topology.cif"))


def pocket_ligand_rmsd(
    samples: md.Trajectory, reference: md.Trajectory, ligand_id: int,
) -> np.ndarray:
    """Align the protein pocket and return ligand heavy-atom RMSD in angstroms."""
    reference_ca = {
        a.residue.resSeq: a.index for a in reference.topology.atoms
        if a.residue.chain.chain_id == "A" and a.name == "CA"
    }
    reference_ligand = {
        a.name: a.index for a in reference.topology.atoms
        if a.residue.chain.chain_id == "A"
        and a.residue.resSeq == ligand_id and a.element.symbol != "H"
    }
    atom_names = sorted(reference_ligand)
    ligand_xyz = reference.xyz[0, [reference_ligand[name] for name in atom_names]]
    pocket_residues = [
        resid for resid, index in reference_ca.items()
        if np.linalg.norm(ligand_xyz - reference.xyz[0, index], axis=1).min() <= 1.0
    ]
    # Both topologies number mature HSA residues from 1.
    sample_ca = {
        a.residue.resSeq: a.index for a in samples.topology.atoms
        if a.residue.chain.chain_id == "A" and a.name == "CA"
    }
    sample_ligand = {
        a.name: a.index for a in samples.topology.atoms
        if a.residue.chain.chain_id == "L" and a.element.symbol != "H"
    }
    aligned = samples[:].superpose(
        reference,
        atom_indices=[sample_ca[resid] for resid in pocket_residues],
        ref_atom_indices=[reference_ca[resid] for resid in pocket_residues],
    )
    delta = aligned.xyz[:, [sample_ligand[name] for name in atom_names]] - ligand_xyz
    # MDTraj coordinates are in nanometres; report RMSD in angstroms.
    return 10 * np.sqrt(np.mean(np.sum(delta**2, axis=-1), axis=-1))


outputs.mkdir(parents=True, exist_ok=True)
reference_path = outputs / "2BXG.pdb"
if not reference_path.is_file():
    response = requests.get("https://files.rcsb.org/download/2BXG.pdb", timeout=90)
    response.raise_for_status()
    reference_path.write_bytes(response.content)
reference = md.load_pdb(str(reference_path))
rows = []
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
colors = ["#4C78A8", "#D99A50", "#5A9E7A"]
for ax, (pocket, ligand_id) in zip(axes, [("Sudlow II", 2001), ("FA6", 2002)], strict=True):
    for column, case in enumerate(conditions):
        rmsd = pocket_ligand_rmsd(load_predictions(outputs / case), reference, ligand_id)
        ax.scatter(
            column + np.linspace(-0.12, 0.12, len(rmsd)), rmsd,
            color=colors[column], s=28, alpha=0.95, edgecolors="white", linewidths=0.5,
        )
        rows.extend(
            {"case": case, "frame": frame, "reference_pocket": pocket, "ligand_rmsd_A": value}
            for frame, value in enumerate(rmsd)
        )
    ax.set(xticks=range(len(conditions)), xticklabels=list(conditions.values()))
    expected = "Default / Sudlow II target" if pocket == "Sudlow II" else "FA6 target"
    ax.set_title(
        f"Reference pocket: {pocket}\nExpected match: {expected}",
        fontsize=9, pad=8, color="#374151",
    )
    ax.tick_params(labelsize=8, length=3, colors="#5B6470")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E8ECF0", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BCC3CC")
axes[0].set_ylabel("Pocket-aligned ligand RMSD (Å)", fontsize=9, color="#374151")
fig.tight_layout(pad=1.1)
fig.savefig(outputs / "pocket_rmsd.png", dpi=180)
with (outputs / "pocket_rmsd.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
plt.close(fig)
fig

# %% [markdown]
# ## Inspect ligand placement in 3D
#
# Compare the first prediction from each condition, from left to right:
# **unconstrained**, **FA6 contacts**, and **Sudlow II contacts**.
# The views are linked: drag to rotate and scroll to zoom all three together.
# The views show HSA as a cartoon and ibuprofen as sticks.

# %%
cases_to_view = ["unconditioned", "fa6", "site_ii"]
first_predictions = md.join(
    [load_predictions(outputs / case)[0] for case in cases_to_view]
)
viewer = view_structure_grid(
    first_predictions, rows=1, columns=3, width=1200, height=420,
)
for column in range(3):
    viewer.removeAllSurfaces(viewer=(0, column))
viewer.show()

# %% [markdown]
# ## Summary
#
# In this example, protein-ligand contacts bias BioEmu2's complex predictions
# towards selected binding sites in HSA. BioEmu2 also supports other steering
# and contact-conditioning approaches that can bias predictions towards a
# specific state.
#
# This tutorial uses query-only MSAs to keep runtime short. For real applications,
# include homologous MSAs and inspect the predicted structures visually.
#
# Experimental reference: Ghuman et al. (2005), *Structural basis of the
# drug-binding specificity of human serum albumin*,
# DOI [10.1016/j.jmb.2005.07.075](https://doi.org/10.1016/j.jmb.2005.07.075).
