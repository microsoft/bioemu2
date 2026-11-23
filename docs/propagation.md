# Propagate structures through time

<!-- BEGIN GENERATED TOC -->
## Contents

- [Use the CLI](#use-the-cli)
- [Use EmuSession](#use-emusession)
- [Use an external starting structure](#use-an-external-starting-structure)
- [Sample one endpoint per frame or several from one frame](#sample-one-endpoint-per-frame-or-several-from-one-frame)
- [Choose a sampler](#choose-a-sampler)
- [Sample endpoints at different future times](#sample-endpoints-at-different-future-times)
- [Build a time-ordered trajectory](#build-a-time-ordered-trajectory)
- [Customize each trajectory step](#customize-each-trajectory-step)
<!-- END GENERATED TOC -->

Propagation samples a structure `x_t` at a future time `t` from a starting
structure (`x0`), where `t` is the time lag (`lag_time_ns`).
Each result is an **endpoint**, not an
intermediate frame in a trajectory. To build a time-ordered trajectory, use
[`simulate_trajectory()`](#build-a-time-ordered-trajectory).

Use lags from 1 to 1000 ns (1 ns to 1 µs), the range the model mostly saw
during training. At longer lags, endpoints approach the equilibrium ensemble;
beyond a few milliseconds, use
[equilibrium sampling](sampling.md#choose-a-sampling-mode) instead.
For repeated sampling and propagation of the same system, use
`EmuSession.propagate()` to keep trunk embeddings in memory; see
[reuse embeddings](reuse-embeddings.md).

## Use the CLI

First sample a starting conformation for adenylate kinase (PDB 1AKE):

```bash
bioemu2 sample-equilibrium \
  --pdb-id 1AKE \
  --num-samples 1 \
  --out outputs/adk-start
```

Then draw eight independent endpoints from that starting structure at a 1 ns
lag:

```bash
bioemu2 propagate \
  --x0-cif outputs/adk-start/topology.cif \
  --num-samples 8 \
  --lag-time-ns 1 \
  --out outputs/adk-start-propagated
```

Pass the CIF and a matching system YAML to retain custom definitions and
contacts:

```bash
bioemu2 propagate \
  --x0-cif initial.cif \
  --system system.yaml \
  --num-samples 8 \
  --lag-time-ns 1 \
  --out outputs/from-external-cif
```

The CLI requires every starting atom to be present and in system order; it
cannot reorder atoms or select chains. Chain ID strings may differ because
chains are matched by position. For chain selection, define a system using
[`System.from_pdb_id()`](system.md#load-a-deposited-structure-definition)
and [save it to YAML](system.md#read-and-write-yaml) for `--system`. You can
instead use `--pdb-id` to load an RCSB system, or omit both system options
when the CIF has sufficient
[mmCIF metadata](system.md#metadata-used-when-importing-mmcif). See the
[system guide](system.md) for custom entities, bonds, and contacts.

To sample one endpoint for each of N starting structures, store the starting
frames in an XTC file and pass `--x0-xtc starts.xtc` instead of
`--num-samples`. Also pass `--x0-cif starts.cif` as the XTC topology; their
atom order must match.

The CLI writes `samples.xtc` with endpoints in input order, without the
starting frames; load it with the output `topology.cif`. When the starting
structure comes from `sample-equilibrium`, its `topology.cif` contains the
coordinates of frame 0 in the starting `samples.xtc`, whether or not that
frame passed physicality checks.
`physicality.json` has one entry per endpoint, in the
[same format as for sampling](sampling.md#use-the-cli). Rejected frames remain
in the XTC, so report index `i` describes frame `i`.
Rerun with the same `--out` to continue an interrupted run; see
[resume an interrupted sampling run](sampling.md#resume-an-interrupted-sampling-run).

## Use EmuSession

Sample a starting structure, then propagate it by 1 ns using the same
`EmuSession`. `EmuSession.propagate()` reuses the session's trunk embeddings:

```python
from pathlib import Path

from bioemu2 import EmuSession, System, load_model, save_frames

system = System.from_pdb_id("1AKE")
model = load_model()
emu = EmuSession(system, model=model, mode="equilibrium")
x0 = emu.sample(num_samples=1)
x_tau = emu.propagate(x0, lag_time_ns=1.0)
save_frames(x_tau, Path("outputs/adk-endpoint"))
```

`save_frames()` writes `samples.xtc` and `topology.cif` to the output
directory. To use `EmuSession.propagate()` with a structure from a CIF file,
such as one downloaded from the PDB, see
[use an external starting structure](#use-an-external-starting-structure).
To change how endpoints are sampled, see
[choose a sampler](#choose-a-sampler).

See [`examples/simulate_a_trajectory.py`](../examples/simulate_a_trajectory.py)
for a complete adenylate kinase workflow.
For more control and configurability, see the
[low-level functional API](functional-api.md#low-level-functional-api).

## Use an external starting structure

You can propagate structures starting from a `.cif` file downloaded from the
PDB or saved by BioEmu2. The file must contain coordinates for every chain,
residue, and atom in the system; BioEmu2 cannot fill in missing atoms. For
modified residues, ligands, branched entities, or explicit bonds, provide a
matching [system definition](system.md) rather than relying on inference from
the CIF.

Before passing a structure from a CIF file to `EmuSession.propagate()`, map
its coordinates to the system's atom order. The mapper can reorder atoms but
cannot fill in missing ones:

```python
from pathlib import Path

from bioemu2 import EmuSession, System, load_model, map_cif_atom_order, save_frames

system = System.from_yaml("system.yaml")
x0 = map_cif_atom_order(system, "initial.cif")
emu = EmuSession(system, model=load_model(), mode="equilibrium")
endpoints = emu.propagate(x0, lag_time_ns=1.0)
save_frames(endpoints, Path("outputs/from-external-cif"))
```

For the functional API, pass the CIF path directly to `propagate()` as in the
[functional example](functional-api.md#propagate-structures), with `system=system` if the CIF
alone cannot define the system. To propagate from mapped coordinates instead,
pass `x0` and the matching `system`. For the CLI, see [use the CLI](#use-the-cli).

## Sample one endpoint per frame or several from one frame

**One endpoint each for N starting frames:** Pass an `mdtraj.Trajectory`
containing N frames. Propagation draws one independent endpoint per frame in
input order: endpoint `i` is conditioned on starting frame `i`. The
[functional API example](functional-api.md#propagate-structures) shows this with four starting
frames. These endpoints are not consecutive points in a time-ordered trajectory.

**N endpoints from one starting frame:** With `EmuSession.propagate()`, repeat that frame:
`emu.propagate(x0[[0] * 8], lag_time_ns=1.0)` draws eight endpoints from
`x0[0]`. With functional `propagate()`, pass `num_samples=8` for a CIF path or
single-frame trajectory. With the CLI, pass `--num-samples 8` for a CIF path.

## Choose a sampler

`propagate()` accepts the `"edm"`, `"edm_gd"`, `"edm_fk"`, and `"edm_fkc"`
presets. `EmuSession.propagate()` accepts those presets and configured `EDMSampler`,
`GuidedFKSampler`, and `FKCSampler` instances. For a functional call with a
configured sampler, use the [low-level functional API](functional-api.md#low-level-functional-api).
Both `GuidedFKSampler` and `FKCSampler` can use custom steering potentials.
For what each sampler does, see
[choose a sampler](steering.md#choose-a-sampler) in the steering guide.

The `"edm_fk"` and `"edm_fkc"` presets resample particles. With resampling
enabled, starting frames must be identical within each resampling group (the
entire batch by default), because resampling can copy particles between
them. Resampling samplers cannot be used for
[time-ordered trajectories](#build-a-time-ordered-trajectory).

## Sample endpoints at different future times

To explore different timescales, sample endpoints from the **same** starting
frame at several lags. Reuse one `EmuSession` so its trunk embeddings are
computed only once:

```python
from pathlib import Path

from bioemu2 import EmuSession, System, load_model, save_frames

system = System.from_pdb_id("1AKE")
emu = EmuSession(system, model=load_model(), mode="equilibrium")
x0 = emu.sample(num_samples=1)

for lag_time_ns in (1.0, 10.0, 100.0, 1000.0):
    endpoints = emu.propagate(x0[[0] * 8], lag_time_ns=lag_time_ns)
    save_frames(endpoints, Path("outputs/propagation") / f"lag_{lag_time_ns:g}ns")
```

Each output directory contains eight independent endpoints at its named lag,
not successive time points.

## Build a time-ordered trajectory

> [!NOTE]
> Unlike elsewhere in BioEmu2, an `mdtraj.Trajectory` returned by
> `simulate_trajectory()` really is a time-ordered trajectory. Other sampling
> APIs use `mdtraj.Trajectory` as a convenient container for a bag of samples:
> their frame order does not represent elapsed time.

Use `simulate_trajectory()` for successive propagation steps, such as when
making a video. For endpoints at independent future times, use the
[multiple-lag recipe](#sample-endpoints-at-different-future-times) instead.
Samplers with particle resampling are not supported for multi-step rollouts:
copying particles between trajectories makes their starting-frame correspondence
ambiguous. Gradient-guided steering without resampling remains supported.
Each frame in `x0` starts one trajectory; repeat a frame to simulate several
trajectories from the same conformation:

```python
from bioemu2 import System, load_model, sample_equilibrium, simulate_trajectory

system = System.from_pdb_id("1AKE")
model = load_model()
x0 = sample_equilibrium(system, model=model, num_samples=1)
num_trajectories = 4
repeated_x0 = x0[[0] * num_trajectories]

trajectories = simulate_trajectory(
    system,
    model=model,
    x0=repeated_x0,
    lag_time_ns=1.0,
    num_trajectory_steps=10,
    seed=42,
)

assert len(trajectories) == num_trajectories
assert all(trajectory.n_frames == 11 for trajectory in trajectories)
```

The function computes one equilibrium embedding and reuses it for every
propagation step. Each returned trajectory includes its starting frame at time
zero, and MDTraj frame times are stored in picoseconds. Endpoints are aligned to
their own preceding frames.

After the rollout, BioEmu2 checks physicality. Without `output_dir`, it logs
how many frames pass the `high_quality` thresholds and the aggregate violation
counts. With `output_dir`, it checks with `physicality_mode` and saves the
results to one `physicality_NNNNNN.json` per trajectory, next to its
`samples_NNNNNN.xtc`. It does not remove or retry failed frames, so
every returned trajectory contains the requested number of frames.

## Customize each trajectory step

Pass a preset to `simulate_trajectory()`, or a configured sampler to
`simulate_trajectory_with_embeddings()`, when every trajectory step uses the
same settings. The presets use `step_scale=1.0`; the embedding-based operation
uses the supplied sampler unchanged.

Use `propagate_with_embeddings()` directly when sampler settings or steering
potentials must change between trajectory steps, including umbrella windows
that move over time. Compute the embedding once, then construct the sampler for
each step:

```python
from bioemu2 import (
    GuidedFKSampler,
    System,
    get_msa,
    join_trajectory_steps,
    load_model,
    propagate_with_embeddings,
    sample_equilibrium,
)
from bioemu2.denoising.steering.potentials import get_default_potentials

system = System.from_pdb_id("1AKE")
model = load_model()
msa = get_msa(system)
x0 = sample_equilibrium(system, model=model, num_samples=1)
num_trajectory_steps = 10
lag_time_ns = 1.0


def potentials_for_step(step):
    # Replace with potentials that depend on `step`, such as a moving umbrella window.
    return get_default_potentials()


trunk_embeddings = model.embed(
    system,
    msa=msa,
    mode="equilibrium",
)

step_frames = [x0]
for step in range(num_trajectory_steps):
    sampler = GuidedFKSampler(
        potentials=potentials_for_step(step),
        resampling_interval=None,
    )
    step_frames.append(
        propagate_with_embeddings(
            system,
            trunk_embeddings,
            x0=step_frames[-1],
            lag_time_ns=lag_time_ns,
            model=model,
            sampler=sampler,
            seed=42 + step,
            progress=False,
        )
    )

trajectories = join_trajectory_steps(step_frames, lag_time_ns=lag_time_ns)
```

`join_trajectory_steps()` joins frame `i` from every element of `step_frames`
and assigns frame times in picoseconds.
