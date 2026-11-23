# Use the functional Python API

Use the functional API to sample and propagate without an `EmuSession`.
Choose a high-level function for a complete workflow, or its low-level
equivalent for greater control over the workflow:

| Task | CLI command | High-level function | Low-level equivalent |
| --- | --- | --- | --- |
| Predict a structure | `bioemu2 predict-structure` | `predict_structure()` | `sample_with_embeddings()` with embeddings computed in `mode="structure"` and `EDMSampler(step_scale=1.5)` |
| Sample equilibrium conformations | `bioemu2 sample-equilibrium` | `sample_equilibrium()` | `sample_with_embeddings()` with embeddings computed in `mode="equilibrium"` and `EDMSampler(step_scale=1.0)` |
| Propagate starting structures | `bioemu2 propagate` | `propagate()` | `propagate_with_embeddings()` with embeddings computed in `mode="equilibrium"` and `EDMSampler(step_scale=1.0)` |

High-level functions compute embeddings on every call, unless the
[automatic disk cache](reuse-embeddings.md#cache-embeddings-to-disk-automatically)
is enabled, and accept only sampler preset names, such as `"edm"`. Low-level
functions take embeddings you compute once with `model.embed()` and a
configured sampler object, such as `EDMSampler()`. Use low-level functions for a custom MSA,
full sampler configuration, or embedding reuse across calls.

## Sample structures

Use the functional API to sample without creating an `EmuSession`:

```python
from bioemu2 import System, load_model, sample_equilibrium

system = System.from_pdb_id("1AKE")
samples = sample_equilibrium(
    system, model=load_model(), num_samples=64, output_dir="outputs/adk-python",
)
```

`predict_structure()` selects structure-prediction conditioning and
`step_scale=1.5`; `sample_equilibrium()` selects equilibrium conditioning and
`step_scale=1.0`. Both accept only sampler preset strings: `"edm"`, `"edm_gd"`,
`"edm_fk"`, and `"edm_fkc"`, not configured sampler objects; see
[choose a sampler](steering.md#choose-a-sampler) for what each preset does.
Pass `num_steps` to override the denoising step count or `potentials` to
replace a steering preset's potentials. For a custom MSA or other sampler
settings, such as step scale, use the
[low-level Python API](#customize-every-step-with-the-low-level-python-api).

With `output_dir`, these functions check physicality and save every frame,
including frames that fail the checks, to `samples.xtc`, together with
`topology.cif`, `system.yaml`, and `physicality.json`. `physicality_mode`
selects the check thresholds (default `"high_quality"`). Without `output_dir`,
they return frames without checking or saving them.

Every sampling and propagation function, including `sample_with_embeddings()`,
`propagate()`, and `simulate_trajectory()`, takes the same `output_dir`.
`simulate_trajectory()` writes one `samples_NNNNNN.xtc` and
`physicality_NNNNNN.json` per trajectory instead of `samples.xtc` and
`physicality.json`. Completed batches are kept in `<output_dir>/batches`; calling
again with the same output directory continues from them. Context-parallel
sampling keeps no batches. See
[resume an interrupted sampling run](sampling.md#resume-an-interrupted-sampling-run).

### Customize every step with the low-level Python API

To control MSA acquisition, embedding and sampler selection explicitly, call
each step yourself. This example is equivalent to the adenylate kinase
`EmuSession` example in the [README](../README.md#python-api-using-emusession):

```python
from bioemu2 import EDMSampler, System, get_msa, load_model, sample_with_embeddings

system = System.from_pdb_id("1AKE")
model = load_model()
msa = get_msa(system)
embeddings = model.embed(system, msa=msa, mode="equilibrium")
samples = sample_with_embeddings(
    system,
    embeddings,
    model=model,
    num_samples=8,
    sampler=EDMSampler(step_scale=1.0),
)
```

Call `model.embed()` once and reuse the embeddings with
`sample_with_embeddings()` or `propagate_with_embeddings()` for subsequent
runs. To save them to disk and reload
them in another process, see
[save and load specific embeddings](reuse-embeddings.md#save-and-load-specific-embeddings).

## Propagate structures

The high-level functional API offers the same input choices as the CLI: pass
a CIF path to draw multiple endpoints from one starting structure, or an
`mdtraj.Trajectory` (for example, loaded from an XTC) to draw one endpoint per
starting frame.

**Propagate from a CIF file.** Sample a starting structure and save it to a
CIF file, then pass that path to `propagate()`. Set `num_samples` to draw
several endpoints from the structure. The function infers the system from
the CIF:

```python
from pathlib import Path

from bioemu2 import System, load_model, propagate, sample_equilibrium, save_frames

system = System.from_pdb_id("1AKE")
model = load_model()
x0 = sample_equilibrium(system, model=model, num_samples=1)
start_dir = Path("outputs/adk-start")
save_frames(x0, start_dir)
x_tau = propagate(
    model=model,
    x0=start_dir / "topology.cif",
    lag_time_ns=1.0,
    num_samples=8,
)
save_frames(x_tau, Path("outputs/adk-propagated"))
```

**Propagate from several starting structures.** `sample_equilibrium()` returns
an `mdtraj.Trajectory` used as a container of independent structures. Pass it
with the matching `System` to generate one endpoint per starting frame:

```python
from pathlib import Path

from bioemu2 import System, load_model, propagate, sample_equilibrium, save_frames

system = System.from_pdb_id("1AKE")
model = load_model()
x0 = sample_equilibrium(system, model=model, num_samples=4)
x_tau = propagate(system, model=model, x0=x0, lag_time_ns=1.0)
save_frames(x_tau, Path("outputs/adk-from-frames"))
```

As `x0`, `propagate()` accepts any `mdtraj.Trajectory` whose topology matches
`system`. For a single-frame `x0`, pass `num_samples=8` to draw eight endpoints
from that frame. For a multi-frame `x0`, omit `num_samples` to draw one
endpoint per frame. When using contacts, pass the same `contacts` to both
`sample_equilibrium()` and `propagate()` so the starting structures and endpoints
use the same conditioning; see
[condition on contacts](sampling.md#condition-on-contacts).

The two propagation examples above are suboptimal: `sample_equilibrium()` and
`propagate()` each compute the trunk embeddings, so the trunk runs twice for the
same system. Instead, compute the embeddings once and reuse them for sampling
and propagation with the low-level API, as in the
[low-level functional API example](#low-level-functional-api) below.
For sampler choices, see [choose a sampler](steering.md#choose-a-sampler);
resampling samplers have
[additional constraints for propagation](propagation.md#choose-a-sampler).

### Low-level functional API

Use the low-level API whenever you propagate the same system more than once,
for example to propagate a trajectory step by step or to compute statistics at
several lag times: it computes the embeddings once instead of recomputing or
reloading them for every call. It also gives full control of MSA acquisition,
samplers, and steering potentials. To reproduce the
[EmuSession example](propagation.md#use-emusession) with this lower-level API, compute the
embeddings once, then reuse them to sample a starting structure and propagate
it:

```python
from pathlib import Path

from bioemu2 import (
    EDMSampler,
    System,
    get_msa,
    load_model,
    propagate_with_embeddings,
    sample_with_embeddings,
    save_frames,
)

system = System.from_pdb_id("1AKE")
model = load_model()
embeddings = model.embed(system, msa=get_msa(system), mode="equilibrium")
sampler = EDMSampler()
x0 = sample_with_embeddings(
    system,
    embeddings,
    model=model,
    num_samples=1,
    sampler=sampler,
)
x_tau = propagate_with_embeddings(
    system,
    embeddings,
    x0=x0,
    model=model,
    lag_time_ns=1.0,
    sampler=sampler,
)
save_frames(x_tau, Path("outputs/adk-endpoint"))
```
