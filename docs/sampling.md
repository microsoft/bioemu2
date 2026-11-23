# Predict and sample structures

<!-- BEGIN GENERATED TOC -->
## Contents

- [Use the CLI](#use-the-cli)
- [Use EmuSession](#use-emusession)
- [Choose a sampling mode](#choose-a-sampling-mode)
- [Customize an EmuSession](#customize-an-emusession)
  - [Use a different model or precision](#use-a-different-model-or-precision)
  - [Condition on contacts](#condition-on-contacts)
  - [Provide an MSA](#provide-an-msa)
  - [Choose a sampler](#choose-a-sampler)
- [Check physicality](#check-physicality)
- [Propagate starting conformations](#propagate-starting-conformations)
- [Resume an interrupted sampling run](#resume-an-interrupted-sampling-run)
- [Save generated structures](#save-generated-structures)
  - [Read residue names in saved topology files](#read-residue-names-in-saved-topology-files)
<!-- END GENERATED TOC -->

## Use the CLI

Generate sixty-four independent equilibrium conformations of adenylate kinase
(PDB 1AKE):

```bash
bioemu2 sample-equilibrium \
  --pdb-id 1AKE \
  --num-samples 64 \
  --out outputs/adk-64
```

The command computes an MSA, downloads the default `bioemu2-preview` checkpoint,
and writes all sixty-four conformations to `outputs/adk-64/samples.xtc` with
their topology in `outputs/adk-64/topology.cif`. It also writes the resolved
system and command to the output directory. The first run requires network
access for the RCSB structure, MSA search, and checkpoint download.

The `predict-structure` and `sample-equilibrium` commands write
`physicality.json` with one entry per XTC frame: zero-based `frame`, `status`
(`passed`, `rejected`, or `not_checked` with `--physicality-filter no_filter`),
and the first failed check in `reason` (`null` otherwise). Rejected frames
remain in the XTC; use `frame` to associate results with frames. To exclude
failed frames, see [check physicality](#check-physicality).

`--pdb-id ID` loads a system from RCSB by calling `System.from_pdb_id()` with
its defaults: biological assembly 1, including non-water ligands. The CLI does
not expose assembly, chain, or ligand selection. For that control, or to sample
a system you defined yourself, write a [system YAML file](system.md) and pass
`--system` instead of `--pdb-id`. For example, 1AKE contains adenylate kinase
bound to an inhibitor. To sample the protein without it, select options in
[`System.from_pdb_id()`](system.md#load-a-deposited-structure-definition) and
save the system:

```python
from bioemu2 import System

system = System.from_pdb_id("1AKE", include_ligands=False)
system.to_yaml("adk-apo.yaml")
```

```bash
bioemu2 sample-equilibrium \
  --system adk-apo.yaml \
  --num-samples 64 \
  --out outputs/adk-apo
```

To resume an interrupted run, rerun the same command; see
[resume an interrupted sampling run](#resume-an-interrupted-sampling-run).

## Use EmuSession

Create an `EmuSession` to sample sixty-four independent equilibrium
conformations of adenylate kinase (PDB 1AKE):

```python
from pathlib import Path

from bioemu2 import EmuSession, System, load_model, save_frames

system = System.from_pdb_id("1AKE")
model = load_model()
emu = EmuSession(system, model=model, mode="equilibrium")
samples = emu.sample(num_samples=64)
save_frames(samples, Path("outputs/adk-python"))
```

> [!NOTE]
> `EmuSession` does not support context-parallel sampling. For systems that do
> not fit on one GPU, use the [context-parallel functional API](context-parallel-sampling.md).

A session keeps its trunk embeddings in memory, so repeated calls with
different samplers or seeds skip the trunk; see
[reuse embeddings](reuse-embeddings.md). To sample without a session, see the
[functional Python API](functional-api.md#sample-structures).

**Create a new `EmuSession` when inputs change.** An `EmuSession` caches its MSA
and trunk embeddings in memory; treat its inputs as immutable. Modifying
`EmuSession.system` in place after computing the MSA or embeddings raises an
error when you next access `EmuSession.msa` or call `EmuSession.sample()` or
`EmuSession.propagate()` to use the embeddings. Modifying contacts also raises on the next sampling or
propagation call. The model, MSA, and mode cannot be reassigned. Create a new
`EmuSession` for changed inputs; `cache_clear()` does not permit modifying
the existing session's system or contacts.

## Choose a sampling mode

BioEmu2 was trained for two different tasks, structure prediction and
equilibrium sampling:

| CLI | Session mode | Use when you want | Settings selected automatically |
| --- | --- | --- | --- |
| `bioemu2 predict-structure` | `EmuSession(..., mode="structure").sample()` | An accurate individual folded structure | Heuristic sharpening with `step_scale=1.5` |
| `bioemu2 sample-equilibrium` | `EmuSession(..., mode="equilibrium").sample()` | A range of conformations from the modeled equilibrium ensemble | `step_scale=1.0`, without sharpening |

The returned frames are independent samples, not a time series.
The [functional API](functional-api.md#sample-structures) provides corresponding
`predict_structure()` and `sample_equilibrium()` functions.

## Customize an EmuSession

`EmuSession` handles the full workflow: given a [`System`](system.md) and a model
from [`load_model()`](checkpoints.md), it computes the [MSA](msa.md), prepares
the system for the model, and selects the sampling settings for the task.
Without `output_dir`, it returns every generated frame; check physicality
separately before saving.

### Use a different model or precision

`load_model()` downloads and caches the default `bioemu2-preview` checkpoint.
Pass `device` and `precision` to choose where and how the model runs, or
`checkpoint_path` to load a local checkpoint:

```python
from bioemu2 import load_model

model = load_model(device="cuda", precision="bf16-mixed")
```

See [load a model checkpoint](checkpoints.md) for defaults, other
checkpoints, and the checkpoint cache location.

### Condition on contacts

Contacts specify pairs of residues or atoms in the system that should be close
to each other in sampled structures. They guide sampling without adding a
covalent bond. Pass contacts when creating the session; see
[define contact conditioning](system.md#define-contact-conditioning).

### Provide an MSA

MSA means multiple sequence alignment. If you omit `msa`, the session loads a
cached ColabFold alignment or queries the public ColabFold server when first
needed. See the [MSA guide](msa.md) for details. To supply a custom MSA, such as
a subset of the default alignment, pass it when creating the session:

```python
from bioemu2 import (
    EmuSession,
    System,
    get_msa,
    load_model,
)

system = System.from_pdb_id("1AKE")
model = load_model()
shallow_msa = get_msa(system)[:128]
emu = EmuSession(system, model=model, mode="equilibrium", msa=shallow_msa)
samples = emu.sample(num_samples=8)
```

### Choose a sampler

By default `EmuSession.sample()` runs the `"edm"` preset: ordinary, unsteered
EDM sampling at the step scale chosen for the task. Pass a different preset to
steer sampling towards structures with desired properties:

```python
from bioemu2 import EmuSession, System, load_model

system = System.from_pdb_id("1AKE")
emu = EmuSession(system, model=load_model(), mode="equilibrium")
samples = emu.sample(num_samples=8, sampler="edm_fk")
```

To choose a preset or configure a sampler, see
[choose a sampler](steering.md#choose-a-sampler) in the steering guide.
Executable examples are in
[`examples/improve_disulfide_physicality_with_steering.py`](../examples/improve_disulfide_physicality_with_steering.py)
and
[`examples/tilt_a_sample_distribution_with_steering.py`](../examples/tilt_a_sample_distribution_with_steering.py).

For `EDMSampler` and `GuidedFKSampler`, the session warns if
`step_scale` differs from 1.0 in equilibrium mode or 1.5 in structure mode;
it uses the supplied value unchanged. `FKCSampler` has no step scale.

## Check physicality

Without `output_dir`, `EmuSession.sample()` returns every generated frame and
leaves checking and saving to the caller. Call `check_physicality()` explicitly to obtain per-frame
results for chain breaks, clashes, and other geometry checks:

```python
from bioemu2 import EmuSession, System, check_physicality, load_model, save_frames

system = System.from_pdb_id("1AKE")
emu = EmuSession(system, model=load_model(), mode="equilibrium")
samples = emu.sample(num_samples=8, seed=42)
physicality = check_physicality(samples, system=system)
print(physicality.passed)
print(physicality.reasons)
save_frames(samples, "outputs/adk", physicality=physicality)
```

The `PhysicalityReport` retains a boolean `passed` array, the first failure
reason for each frame in input order, and a `checked` flag indicating whether
protein checks ran. Checking does not change `samples`.
`save_frames()` writes all frames plus `physicality.json` when given the
report. To save only the report, call `physicality.save("physicality.json")`.
The default check mode is `"high_quality"`. Pass `mode` to choose another:

- `PhysicalityFilterMode.NO_FILTER` or `"no_filter"`;
- `PhysicalityFilterMode.NO_VIOLATIONS_ALLOWED` or `"no_violations"`;
- `PhysicalityFilterMode.HIGH_QUALITY` or `"high_quality"`; and
- `PhysicalityFilterMode.HIGH_QUALITY_PER_RESIDUE` or `"high_quality_per_residue"`.

See [`PHYSICALITY_THRESHOLDS` and `check_allatom_physicality()`](../src/bioemu2/physicality.py)
for the checks and thresholds applied by each mode.
Filtering ignores single-residue chains and chains without a standard amino
acid; ignored chains remain in the returned frames. A standard residue
identifies a chain as protein when its topology lacks entity-type metadata.
This avoids treating standalone `UNK` ligands as protein, but means a chain
containing only modified residues cannot be identified and is not checked.
In eligible chains, modified residues participate in geometric checks, and
complete `N`–`CA`–`C`–`O` backbones participate in peptide-bond checks. See
[`remove_unphysical_frames()`](../src/bioemu2/physicality.py)
for detailed selection rules. If no eligible protein atoms remain, the report
retains the requested mode but sets `checked=False`; saved frames receive
`"not_checked"` statuses rather than being reported as passing protein checks.
No nucleic-acid or ligand physicality checks are performed. The lower-level
`remove_unphysical_frames()` still raises `ValueError` without eligible protein atoms.

To filter a saved or custom MDTraj trajectory, load it with a topology that
records its bonds and pass it to `remove_unphysical_frames()`. If that topology
contains a nonstandard bond, such as a disulfide, pass its pair of **atom
indices in the input topology** as `expected_extra_bonds`. The pair tells the
topology check the bond is intentional rather than unexpected; it does not add
a bond. Sulfur-sulfur bonds, such as disulfides, must lie within 1.80–2.25 Å;
no filtering mode tolerates a violation. Atom pairs bonded in the topology never
count as clashes. A declared bond joining a protein chain to another chain, such
as a covalent ligand, is not checked and logs a warning.

Pass `"no_filter"` or `PhysicalityFilterMode.NO_FILTER` to skip checks and
record each frame as `not_checked`; `None` and invalid strings raise `ValueError`.
Failed checks do not prevent samples from being returned or saved, even when
every frame fails. To select passing frames explicitly, use
`samples[physicality.passed]`.

## Propagate starting conformations

To generate time-lagged endpoints or a rollout from a starting structure, see
the [propagation guide](propagation.md).

## Resume an interrupted sampling run

Sampling a large ensemble can be interrupted by a pre-empted job or a process
failure. With `output_dir`, completed raw batches are stored in
`<output_dir>/batches`, so restarting the same run generates only the
unfinished batches, rather than starting over.

Pass `output_dir` to `emu.sample()` to save completed batches. If the process
stops, recreate the same session and repeat the call with the same settings
and output directory:

```python
from bioemu2 import EmuSession, System, load_model

system = System.from_pdb_id("1AKE")
emu = EmuSession(system, model=load_model(), mode="equilibrium")
samples = emu.sample(
    num_samples=4000,
    batch_size=50,
    seed=42,
    output_dir="outputs/adk-resume",
)
```

On a rerun with the same output directory, BioEmu2 logs how many completed
batches it found, skips them, and samples the rest. Batches are kept in
`<output_dir>/batches` after the final outputs are saved. The CLI behaves the
same: rerun the command with the same `--out`.

Before reusing batches, BioEmu2 checks that the system, mode, contacts, `seed`,
`num_samples`, `batch_size`, sampler type, `num_steps`, `step_scale` and checkpoint match the saved
run. A mismatch raises an error; use a new output directory or delete `<output_dir>`.
Propagation also checks the lag time and the starting frames, saved as `x0.npy`
next to `run.json`. Potentials, precision and other sampler settings are not
checked, so keep them unchanged. Use one writer per directory.

If restarting in a new process, reload the model and create a new `EmuSession`.
Embeddings are recomputed unless the [embedding disk cache](reuse-embeddings.md#cache-embeddings-to-disk-automatically)
is enabled, in which case unchanged embeddings are read from disk.

With `seed`, batch `i` uses `seed + i`, so a resumed run matches an uninterrupted one.
Without `seed`, each call draws a new seed, and a rerun without `seed` keeps
the saved batches and samples the rest with the new seed.
[Multi-GPU context-parallel sampling](context-parallel-sampling.md) does not
save batches and cannot resume.

## Save generated structures

Save an XTC trajectory with a matching CIF topology:

```python
from pathlib import Path

from bioemu2 import EmuSession, System, load_model, save_frames

system = System.from_pdb_id("1AKE")
emu = EmuSession(system, model=load_model(), mode="equilibrium")
samples = emu.sample(num_samples=8)
output_path = save_frames(
    samples,
    Path("outputs/adk"),
)
print(output_path)
```

The trajectory already carries the system topology, so no separate topology
argument is needed. `save_frames()` writes `samples.xtc` and `topology.cif`.
Without `output_dir`, the session returns all frames without writing files.
Pass `output_dir` to `emu.sample()` to also write `system.yaml` and
`physicality.json`. No frames are removed automatically.

### Read residue names in saved topology files

In a saved `topology.cif`, single-residue ligands with recognized CCD
labels retain their codes (for example, `ZN`); unknown or non-CCD labels use
`LIG`. Multi-residue non-polymer chains retain their per-residue CCD labels.
When mapping that CIF back to the same `System` with `map_cif_atom_order()`,
BioEmu2 matches `LIG` to the internal `UNK` label for SMILES/InChI ligands.
