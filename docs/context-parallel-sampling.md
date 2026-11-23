# Sample large structures with context parallelism

<!-- BEGIN GENERATED TOC -->
## Contents

- [Decide whether you need context parallelism](#decide-whether-you-need-context-parallelism)
- [System requirements](#system-requirements)
- [Quick start](#quick-start)
- [Launch with torchrun](#launch-with-torchrun)
- [Choose the number of GPUs](#choose-the-number-of-gpus)
- [Python API](#python-api)
  - [Supported samplers](#supported-samplers)
  - [Reuse prepared embeddings](#reuse-prepared-embeddings)
- [Current limitations](#current-limitations)
- [Environment and diagnostic logging](#environment-and-diagnostic-logging)
<!-- END GENERATED TOC -->

Context parallelism lets BioEmu2 sample systems whose pair representations are
too large to fit on one GPU. It splits those representations across several
GPUs so that each GPU needs less memory. See
[Fold-CP: A Context Parallelism Framework for Biomolecular Modeling](https://arxiv.org/abs/2603.14806)
for background on context parallelism in biomolecular models.

## Decide whether you need context parallelism

Before you turn to context parallelism, try a smaller batch size. Denoising
memory grows linearly with the batch size, so if denoising runs out of memory,
lower `--batch-size`, down to 1 if needed. Batch size does not change trunk
embedding memory, so if the trunk runs out of memory, try the single-GPU options
in [fit larger systems](speed-and-capacity.md#fit-larger-systems) or use context
parallelism.

## System requirements

- A square number of NVIDIA GPUs, at least four, such as 4, 9, or 16, which
  BioEmu2 arranges as an `n x n` grid. The GPUs split the pair representations
  of one system between them and all work on the same batch of samples, so the
  number of GPUs is not the batch size, and adding GPUs does not produce more
  samples at once.
- BioEmu2 installed with the CUDA extra. We have not tested context
  parallelism on older GPUs such as the V100.

## Quick start

Install BioEmu2 with the CUDA extra from the repository root, as described in
the [installation instructions](../README.md#installation). This quick example
runs sampling on four GPUs of one node:

```bash
python -m pip install ".[cuda]"
torchrun --standalone --nproc-per-node=4 \
    --module bioemu2.cp sample-equilibrium \
    --pdb-id 1AKE \
    --num-samples 5 --seed 42 \
    --out outputs/adk-cp
```

This example also fits on one GPU, so the command only checks your setup.

## Launch with torchrun

Context-parallel sampling runs one process per GPU, so you always start it
with `torchrun` and pass it the `bioemu2.cp` module.

The command has two parts:

```bash
torchrun <torchrun options> --module bioemu2.cp <bioemu2 cp options>
```

The context-parallel commands take the same inputs and write the same output
files as [the serial ones](sampling.md#use-the-cli), except that
`--sampler` accepts only the [supported samplers](#supported-samplers).
See the CLI help for more information.

```bash
bioemu2 cp sample-equilibrium --help
```

Set up `torchrun` as usual, except that the total number of processes across
all nodes must be a square number and every node needs access to the input
files.

On one node, run `torchrun --standalone --nproc-per-node=4 ...`. To choose
which GPUs to use, set `CUDA_VISIBLE_DEVICES`, for example
`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 ...`.

The following example runs a job on two nodes with eight GPUs each, 16 GPUs in
total. Run one `torchrun` command on each node. On the first node, run the
command below, replacing `FIRST_NODE_ADDRESS` with that node's hostname or IP
address; if port 29500 is taken, choose another free port:

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
    --master-addr=FIRST_NODE_ADDRESS --master-port=29500 \
    --module bioemu2.cp sample-equilibrium \
    --pdb-id 1AKE \
    --num-samples 5 --seed 42 \
    --out outputs/adk-cp
```

On the second node, run the same command with `--node-rank=1`.

Together the two `torchrun` calls start 16 processes, which BioEmu2 arranges
into a 4×4 grid.

> [!NOTE]
> The first process on the first node writes all output files.

## Choose the number of GPUs

Above about 2,000 tokens, the pair representation and the triangle updates on
it take up most of the memory, and both grow quadratically with the number of
tokens, so the number of GPUs you need also grows quadratically with system
size. On 192 GB B200 GPUs with an MSA depth of 4,092, the largest systems we
embedded had 3,072 tokens on one GPU, 10,240 tokens on a 2×2 grid, and 16,384 tokens on a 4×4
grid. Use the smallest grid that fits your system.

The figures below show total throughput on B200 GPUs. More GPUs let you fit
larger systems, but they do not speed up sampling in proportion to their
number.

<img src="https://github.com/microsoft/bioemu2/releases/download/readme-media/cp-throughput-b200.png" alt="Trunk embedding and denoising throughput versus tokens on 1, 4, and 16 B200 GPUs" width="600">

Dividing throughput by the number of GPUs shows what context parallelism costs.
For a given system, the fewer GPUs you use, the more efficient each one is, as
long as the system still fits. If your system fits on one GPU, running
independent jobs on separate GPUs is more efficient than context parallelism.

<img src="https://github.com/microsoft/bioemu2/releases/download/readme-media/cp-throughput-per-gpu-b200.png" alt="Per-GPU trunk embedding and denoising throughput versus tokens on 1, 4, and 16 B200 GPUs" width="600">

For batch size and precision settings, see
[improve sampling speed and scale to bigger systems](speed-and-capacity.md).

## Python API

The operations in `bioemu2.context_parallel.api` are context-parallel versions
of the functions in `bioemu2.sampling`. Refer to [sampling.md](sampling.md) for
sampling modes, contact conditioning, and physicality checks, and to
[choose a sampler](steering.md#choose-a-sampler) for sampler settings. This guide covers only what differs under context
parallelism.

`cp.predict_structure()` and `cp.sample_equilibrium()` are the high-level entry
points. They handle `output_dir`, physicality checks, and saved files as
described in the [functional Python API](functional-api.md#sample-structures),
and the context-parallel CLI calls them in the same way. Context parallelism adds
these rules:

- Every rank must call the function with identical output settings, and every
  rank returns every frame.
- With `output_dir`, only global rank 0 checks physicality and writes files.
  If its write fails, the error is broadcast and raised on every rank, so no
  rank proceeds past a failed save.
- For low-level sampling with `cp.sample_with_embeddings()`, or any call
  without `output_dir`, [check physicality](sampling.md#check-physicality) and
  call `bioemu2.save_frames()` on global rank 0 only.

The example
[`examples/sample_structures_with_context_parallelism.py`](../examples/sample_structures_with_context_parallelism.py)
uses the same `System`, `MSA`, and sampler objects as serial sampling. Its core,
without the PNG rendering, is:

```python
from pathlib import Path

import torch.distributed as dist

from bioemu2 import System, get_msa
from bioemu2.context_parallel import api as cp
from bioemu2.paths import EXAMPLES_DIR

with cp.session():
    system = System.from_yaml(EXAMPLES_DIR / "systems/myoglobin_101m.yaml")
    msa = get_msa(system) if dist.get_rank() == 0 else None
    model = cp.load_model("bioemu2-preview", precision="bf16-mixed")
    samples = cp.sample_equilibrium(
        system, model=model, msa=msa, num_samples=8,
        sampler="edm", seed=42, output_dir=Path("outputs/myoglobin-cp"),
    )
```

Every rank must enter the session and call the operations in the same order
with the same sampling settings. Call `bioemu2.get_msa` on rank 0 only and pass
`msa=None` on the other ranks; rank 0 prepares the model inputs from the MSA
and sends them to the other ranks. Model and sampling operations communicate
across all ranks, so never call them inside an `if dist.get_rank() == 0:`
block; use such a block only for work that runs on rank 0 alone, such as
checking physicality, rendering, or writing files.

The session picks each process's GPU from `LOCAL_RANK`, which `torchrun` sets.
`cp.load_model()` takes the same checkpoint arguments as `load_model()`; see
[load a model checkpoint](checkpoints.md). It returns a
`cp.ModelWithSettings` that holds the distributed model and its inference
settings.

Every process must run the same script, so launch it with `torchrun` as shown
in the [quick start](#quick-start), not with `python`. To run the repository
example on one node, install its visualization dependencies and launch it the
same way:

```bash
python -m pip install ".[cuda,viz]"
playwright install chromium
torchrun --standalone --nproc-per-node=4 \
  examples/sample_structures_with_context_parallelism.py
```

The example writes all samples, their per-frame physicality report, and a PNG
to `outputs/myoglobin-cp`. The PNG shows the eight samples in a 2×4 grid, each
aligned to the first sample, as cartoons with translucent surfaces and with
non-protein groups drawn as sticks.

### Supported samplers

The high-level `cp.predict_structure()` and `cp.sample_equilibrium()` accept
only the preset strings `"edm"`, `"edm_gd"`, and `"edm_fk"`, not configured
sampler objects. The serial `"edm_fkc"` preset is rejected. Pass `num_steps` to
override the preset's denoising step count or `potentials` to replace its
steering potentials. For a configured sampler object or a custom step scale, use
`model.embed` and `cp.sample_with_embeddings()`, which take a `Sampler` instance
directly.

Context parallelism supports `EDMSampler` and `GuidedFKSampler`, including
gradient guidance and FK resampling. With FK, resampling happens within each
batch, so a smaller `batch_size` leaves fewer samples to resample from.
`FKCSampler` and custom samplers are not supported yet.

### Reuse prepared embeddings

As for standard single-GPU sampling, you can reuse embeddings across sampling
calls with `model.embed` and `cp.sample_with_embeddings`. Choose
`batch_size` for each sampling call. The trunk outputs are reused,
while the inexpensive multiplicity-specific static shards are prepared for
that call:

```python
from bioemu2 import EDMSampler

trunk_embeddings = model.embed(
    system, msa=msa, mode="equilibrium",
)
samples = cp.sample_with_embeddings(
    system, trunk_embeddings, model=model, num_samples=8, batch_size=4,
    sampler=EDMSampler(), seed=42,
)
```

## Current limitations

Context parallelism does not yet support:

- `FKCSampler` or custom samplers; see [supported samplers](#supported-samplers).
- Propagation, including `simulate_trajectory()`.
- `EmuSession`, the `BIOEMU2_EMBEDS_CACHE` disk cache, or saving and loading
  embeddings; see
  [reuse embeddings with context parallelism](reuse-embeddings.md#reuse-embeddings-with-context-parallelism).
- [Resuming an interrupted run](sampling.md#resume-an-interrupted-sampling-run)
  from saved batches.

## Environment and diagnostic logging

The same environment settings apply to CLI and Python API runs.
BioEmu2 works out the square grid from the number of processes that `torchrun`
starts. You can also state the grid with `EMU_CP_GRID=n,n`, but it must match
that number. `EMU_CP_VERBOSITY=debug` enables detailed
logging from the distributed model implementation, without changing other
BioEmu2 loggers.

Optionally source
[`src/bioemu2/context_parallel/scripts/cp_env.sh`](../src/bioemu2/context_parallel/scripts/cp_env.sh)
before either launch. It configures allocator settings, mesh/device defaults,
and NCCL diagnostics. Set `EMU_CP_VERBOSITY` before sourcing it:

```bash
export EMU_CP_VERBOSITY=debug
source src/bioemu2/context_parallel/scripts/cp_env.sh
```
