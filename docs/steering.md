# Steer sampling

<!-- BEGIN GENERATED TOC -->
## Contents

- [Understand the target distribution and potentials](#understand-the-target-distribution-and-potentials)
  - [Potentials](#potentials)
- [Sample with steering](#sample-with-steering)
- [Choose a sampler](#choose-a-sampler)
  - [Decision guide](#decision-guide)
  - [Configure FK steering: `GuidedFKSampler`](#configure-fk-steering-guidedfksampler)
  - [Configure FKC: `FKCSampler`](#configure-fkc-fkcsampler)
- [References](#references)
<!-- END GENERATED TOC -->

Steering is a set of tools for changing the distribution a model samples at
inference time, without retraining it. You choose **potentials** that describe
the properties you want to favour. Depending on those potentials, you can improve
physicality or favour particular conformations. Steering can also be used as
part of enhanced-sampling workflows for rare events and free-energy estimation;
see Xie et al. [3].

Use the [sampling guide](sampling.md) to load a model and system. For complete
steering examples, see
[improve disulfide physicality](../examples/improve_disulfide_physicality_with_steering.py) and
[tilt a sampled distribution](../examples/tilt_a_sample_distribution_with_steering.py).

> [!NOTE]
> Steering is not the only way to influence the model. **Contact conditioning**
> supplies contacts as extra model inputs to ask for parts of the system to be
> close. This is different from applying potentials during sampling, and the two
> can be combined. See [define contact conditioning](system.md#define-contact-conditioning).

## Understand the target distribution and potentials

Let $x$ be atomic coordinates and $p_{\mathrm{base}}(x)$ be the distribution sampled by the
unsteered model for the same system and conditioning. For a potential $U(x)$,
steering aims to sample from the **tilted distribution**

$$
p_{\mathrm{target}}(x) \propto p_{\mathrm{base}}(x)\exp[-U(x)].
$$

### Potentials

A steering potential is an instance of a `Potential` subclass. It prepares
system-specific data, then returns one scalar energy $U(x)$ per sampled
structure $x$; smaller values are more favourable. For physicality steering, it
returns high energies for geometry that is wrong given the topology, such as
clashes or broken bonds, so such structures are less likely under the tilted
distribution. For enhanced sampling, a harmonic potential on specific degrees
of freedom, such as distances or angles, favours particular conformations.

The overall energy can be a sum of sub-potentials, $U(x) = \sum_i U_i(x)$, where
each sub-potential $U_i(x)$ favours a particular property of the structure $x$.
FK and FKC scale this sum differently; see the [decision guide](#decision-guide).
The potential defines the objective, not the sampler. See
[define steering potentials](potentials.md) for available potentials and how to
write your own.

Both Python steered samplers and the CLI steering options use the same
[default physicality potentials](potentials.md#improve-physicality-with-the-default-physicality-potentials)
by default. This set improves physicality, which is particularly important for
large structures. Pass an explicit `potentials=[...]` list in Python, or
`--potentials my.yaml` in the CLI, to replace it.

## Sample with steering

From the CLI, start with `--sampler edm_gd` for gradient-based steering with the
default physicality potentials:

```bash
bioemu2 sample-equilibrium \
    --pdb-id 101M \
    --out outputs/myoglobin-101m --sampler edm_gd
```

There are three steering presets on the CLI: `edm_gd` (FK guidance-only),
`edm_fk` (FK with guidance and resampling), and `edm_fkc` (FKC with guidance
and resampling). FKC SMC, without guidance, is available only in Python, as
`FKCSampler(weight_update="smc")`. See the
[decision guide](#decision-guide) for the differences. All three accept
`--potentials my.yaml` to replace the default potentials.

For a custom objective, construct the potentials, pass them to a
`GuidedFKSampler` or `FKCSampler`, then pass the sampler to
`EmuSession.sample()`. This example restrains bond lengths:

```python
from bioemu2 import (
    EmuSession,
    GuidedFKSampler,
    System,
    load_model,
)
from bioemu2.denoising.steering.potentials import BondLengthPotential
from bioemu2.paths import EXAMPLES_DIR

system = System.from_yaml(EXAMPLES_DIR / "systems/ake.yaml")
model = load_model()
emu = EmuSession(system, model=model, mode="equilibrium")
samples = emu.sample(
    num_samples=16,
    batch_size=16,
    sampler=GuidedFKSampler(
        potentials=[
            BondLengthPotential(
                sigma_tolerance=10.0,
                parameters={"guidance_weight": 0.05},
            )
        ],
        resampling_interval=None,
        num_gd_steps=20,
    ),
    seed=42,
)
```

This example uses FK guidance-only, the algorithm of `--sampler edm_gd`, with
one bond-length potential. Omit `potentials=[...]` to use the default
physicality potentials. See [choose a sampler](#choose-a-sampler) for the
algorithms and [potentials](potentials.md) for other objectives and
configuration.

For greater control, see the
[low-level functional sampling API](functional-api.md#customize-every-step-with-the-low-level-python-api).

## Choose a sampler

Each preset name selects a preconfigured sampler. Pass a preset name with
`--sampler` on the CLI, or as the `sampler` argument of `EmuSession.sample()`,
`EmuSession.propagate()`, or the high-level functions such as
`sample_equilibrium()` and `propagate()`:

| Preset | Sampler | Behaviour |
| --- | --- | --- |
| `"edm"` | `EDMSampler` | Ordinary, unsteered sampling. |
| `"edm_gd"` | `GuidedFKSampler` | Potential-gradient guidance, without particle resampling. |
| `"edm_fk"` | `GuidedFKSampler` | Potential-gradient guidance with Feynman-Kac particle resampling. |
| `"edm_fkc"` | `FKCSampler` | Feynman-Kac correction with population weighting and resampling. |

To use other settings, construct the sampler class yourself and pass it to
`EmuSession.sample()`, `EmuSession.propagate()`, `sample_with_embeddings()`, or
`propagate_with_embeddings()`; the high-level functions accept only preset
names.

BioEmu2 is a diffusion model: sampling starts from random, noisy atom
coordinates and gradually denoises them into a structure. At each step, the
model predicts cleaner coordinates and uses that prediction to continue the
denoising process. Steering modifies this process to favour structures with
lower potential energy as measured by the defined potentials.

The two available steered samplers are **`GuidedFKSampler` (FK)** and
**`FKCSampler` (FKC)**. "FK" stands for Feynman-Kac steering, as described by
Singhal et al. [1]. "FKC" stands for Feynman-Kac correctors, introduced by
Skreta et al. [2].

When sampling a batch, each candidate structure evolving through this process
is called a **particle**. Steering can act on these particles in two ways:

- **Guidance** (gradient-based steering, also called twisting): use potential
  gradients to adjust coordinates towards lower energy. This requires $U(x)$ to
  be differentiable with respect to $x$.
- **Reweighting/resampling**: assign each particle a weight, computed
  differently by the FK and FKC samplers, then draw a new population with
  replacement according to those weights. Particles with high weight tend to be
  copied and particles with low weight discarded.

### Decision guide

Use cases, with a suggested first approach:

- **Improve physicality**: start with FK guidance-only (`edm_gd`). If too many
  samples still fail [physicality checks](sampling.md#check-physicality),
  switch to FK with guidance and resampling (`edm_fk`) and check that diversity
  remains acceptable.
- **Reweight conformations towards a target distribution**, such as a harmonic
  restraint on a distance for enhanced sampling: start with FKC guidance
  (`edm_fkc`, or `FKCSampler()` in Python). Its guidance moves particles
  towards low energy, and its weights correct for it. If its backward pass
  through the denoiser does not fit in GPU memory, use a configuration without
  guidance: FK resampling-only (`GuidedFKSampler(num_gd_steps=0)`, one denoiser
  evaluation per step) or FKC SMC (`FKCSampler(weight_update="smc")`, two).
  Reducing `batch_size` to save memory also shrinks the resampling population.
  The [tilt example](../examples/tilt_a_sample_distribution_with_steering.py)
  compares FKC guidance with unsteered samples reweighted by $\exp[-U(x)]$.
- **Use a non-differentiable potential**: choose FKC SMC
  (`FKCSampler(weight_update="smc")`) or FK resampling-only
  (`GuidedFKSampler(num_gd_steps=0)`); guidance requires potential gradients.
- **Sample with [context parallelism](context-parallel-sampling.md)**: FKC is
  not supported. For physicality, use FK guidance-only (`edm_gd`) or FK with
  guidance and resampling (`edm_fk`). For reweighting, pass
  `GuidedFKSampler(num_gd_steps=0)` to `cp.sample_with_embeddings()`.
- **Build a [time-ordered trajectory](propagation.md#build-a-time-ordered-trajectory)**:
  use FK guidance-only. Multi-step rollouts do not support resampling.

Both FK and FKC approaches are summarized in the table below.
FK can be run with guidance-only, guidance and resampling, or resampling-only approaches.
FKC can be run with SMC (resampling-only) or guidance and resampling approaches.

The resampling mechanism interprets the entire batch size as a single population from which particles are resampled with replacement according to their weights.
This can correlate the particles, especially when the effective sample size is low, leading to reduced diversity in the resampled population.
To counteract this, `resampling_group_size` can be set to a value smaller than the total batch size, effectively dividing the population into independent groups for resampling.

| Approach | CLI preset | Python | Guidance | Reweighting/resampling | Cost per denoising step | Context parallelism | Tilted target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [FK guidance-only](#configure-fk-steering-guidedfksampler) | `edm_gd` | `GuidedFKSampler(resampling_interval=None)` | Gradient with respect to denoised coordinates | None | One denoiser evaluation and `num_gd_steps` potential gradients | Yes | Not targeted: no importance weights |
| [FK with guidance and resampling](#configure-fk-steering-guidedfksampler) | `edm_fk` | `GuidedFKSampler()` | As above | Every `resampling_interval` steps | As above, plus potential values on resampling steps | Yes | Weights target $\exp[-\lambda U(x)]$ but correct only part of the guidance, so not guaranteed |
| [FK resampling-only](#configure-fk-steering-guidedfksampler) | None | `GuidedFKSampler(num_gd_steps=0)` | None | Every `resampling_interval` steps | One denoiser evaluation; potential values on resampling steps | Yes | $\exp[-\lambda U(x)]$ |
| [FKC SMC](#configure-fkc-fkcsampler) | None | `FKCSampler(weight_update="smc")` | None | When the effective sample size falls below `ess_threshold` | Two denoiser evaluations; potential values | No | $\exp[-U(x)]$ |
| [FKC guidance](#configure-fkc-fkcsampler) | `edm_fkc` | `FKCSampler()` | Gradient through the denoiser with respect to noisy coordinates and noise level | When the effective sample size falls below `ess_threshold` | One denoiser evaluation with a backward pass; potential gradients | No | $\exp[-U(x)]$; weights account for the guidance |

In the last column, the tilt multiplies $p_{\mathrm{base}}(x)$ and $\lambda$ is
`fk_lambda`. FK scales each potential by its `resampling_weight`; FKC uses the
unscaled energies, so the same potentials define different targets. These
targets are reached only in the limit of exact model scores, large particle
populations, and fine discretisation; measure the result for your system. See
Singhal et al. [1] and Skreta et al. [2] for the derivations.

FK with guidance and resampling misses its target for two reasons. Its weights
correct only the guidance applied on the step before each resampling step, and
only at noise levels where noise is re-added. Gradient steps can also move
particles far from $p_{\mathrm{base}}(x)$, so a few particles carry most of the
weight and the result has high variance.

> [!IMPORTANT]
> The targets assume `step_scale=1.0`. In structure-prediction mode, such as
> `predict_structure()` or `bioemu2 predict-structure`, the `edm_gd` and
> `edm_fk` presets use `step_scale=1.5`, which sharpens the distribution. With
> `EmuSession.sample()` or `sample_with_embeddings()`, a supplied sampler's step
> scale is used unchanged.
> `FKCSampler` has no step scale and always uses the unsharpened transport.

Each sampling batch contains particles for one system and set of conditions.
Resampling operates within that batch, or within smaller groups if you set
`resampling_group_size`. Separate batches do not exchange particles.

Resampling can reduce diversity when a few particles receive most of the
weight. Compare physicality, structural accuracy, and diversity for your own
system rather than assuming one algorithm is always better.

### Configure FK steering: `GuidedFKSampler`

`GuidedFKSampler` can use guidance, reweighting/resampling, or both. Its
arguments:

- `potentials` (default: physicality potentials): a list replaces the default
  set. Each potential's `parameters` set its strength:
  - `guidance_weight` (default 0): scales its gradient during guidance; at 0
    the potential is not used for guidance.
  - `guidance_interval` (default 1): applies its gradient only on every Nth of
    the `num_gd_steps` gradient steps, starting with the first.
  - `resampling_weight` (default 1): scales its contribution to $U(x)$.

  See
  [configure strength and schedules](potentials.md#configure-strength-and-schedules).
- `num_steps` (default `200`): number of denoising steps.
- `num_gd_steps` (default `20`): gradient steps on the denoised coordinates per
  denoising step, skipped on the final step. `0` disables guidance.
- `resampling_interval` (default `3`): reweight and resample every N steps, and
  on the first and final steps. `None` disables weight computation and
  resampling.
- `resampling_group_size` (default `None`): particles per independent
  resampling group, at least 2. `None` uses the whole sampling batch.
- `fk_lambda` (default `4.0`): multiplies $U(x)$ in the importance weights.
  Without guidance (`num_gd_steps=0`), the target is
  $p_{\mathrm{base}}(x)\exp[-\lambda U(x)]$ with $\lambda$ = `fk_lambda`; with
  guidance, the weights correct only part of it, so that target is not
  guaranteed. It does not change the gradient-step size and is unused when
  `resampling_interval=None`.
- `step_scale` (default `1.0`): multiplies each denoising step; values above 1
  sharpen the distribution. Target-distribution arguments assume `1.0`.
- `noise_scale` (default `1.0`): scales the standard deviation of the noise
  re-added during denoising.

For detailed arguments and resampling behavior, see
[`GuidedFKSampler`](../src/bioemu2/denoising/samplers.py) and
[`edm_steered_sampler`](../src/bioemu2/denoising/steering/edm_steered.py).
The implementation is adapted from Boltz's [diffusion sampler][boltz-sampler].

### Configure FKC: `FKCSampler`

`FKCSampler` combines reweighting/resampling with guidance by default, or uses
reweighting/resampling alone. It has no step or noise scale, and neither mode
supports context-parallel sampling. Its arguments:

- `potentials` (default: physicality potentials): a list replaces the default
  set; an empty list disables weighting and resampling. FKC sums the returned
  energies unscaled and ignores `guidance_weight`, `guidance_interval`, and
  `resampling_weight`.
- `num_steps` (default `200`): number of denoising transitions. The last one
  ends at zero noise.
- `weight_update` (default `"guidance"`): `"guidance"` applies guidance and
  reweights/resamples, and requires `reheat="sigma"`; `"smc"` reweights/resamples
  only.
- `beta_power` (default `4.0`): exponent of $(1 - t)$ in the annealing
  schedule $\beta_t$.
  The tilt grows from zero at high noise to full strength at the clean
  endpoint; larger values delay steering until predictions are cleaner.
- `reheat` (default `"sigma"`): how particles are moved back to a higher noise
  level before each denoising proposal. `"sigma"` returns to the previous
  scheduled level, with no reheating on the first step; `"gamma"` multiplies
  the current noise level by `1 + gamma_0` and works only with
  `weight_update="smc"`.
- `gamma_0` (default `0.8`): reheating factor for `reheat="gamma"`.
- `gamma_min` (default `0.0`): `reheat="gamma"` skips reheating once the next
  noise level is at or below this value. The default reheats on every step
  except the last.
- `ess_threshold` (default `0.9`): resample when any group's normalised
  effective sample size, `1 / (N * sum(w**2))`, falls below this value.
- `resampling_interval` (default `None`): resample every N steps instead; set
  `ess_threshold=None` to use it. Set exactly one of the two.
- `resampling_group_size` (default `None`): particles per independent
  resampling group, at least 2. `None` uses the whole sampling batch.

FKC always resamples at the final step when potentials are supplied, so the
returned population has equal weights.

The FKC framework is designed around a sequence of target distributions: not
only the desired distribution of clean structures at the end, but also the
distributions that particles should follow at each noise level during
denoising. Skreta et al. [2] derive weighted dynamics intended to track this
whole sequence.

`FKCSampler` offers two ways to propose the next particle coordinates. Both
evaluate potentials on denoised coordinates.

With `weight_update="smc"`, the sampler uses the unmodified denoiser proposal,
without guidance. At each denoising step, it reheats particles, denoises them,
and updates their importance weights to account for the transition. The weight
update needs the potential at the denoised estimate of the new state, so each
step except the final zero-noise step calls the denoiser a second time. The
next step cannot reuse that call because it reheats the particles first.

With `weight_update="guidance"`, the sampler also guides the proposal towards
lower potential values. Unlike `GuidedFKSampler`, it differentiates the
potential _through the denoiser_ with respect to the noisy coordinates and
noise level. The importance weights correct for this modified proposal. It
requires a differentiable potential and denoiser.

See [`FKCSampler`](../src/bioemu2/denoising/samplers.py) and
[`edm_fkc_sampler`](../src/bioemu2/denoising/steering/edm_fkc.py) for the full
parameter definitions and update rules.

## References

1. Singhal et al. (ICML 2025),
   [*A General Framework for Inference-time Scaling and Steering of Diffusion Models*][fk-paper].
2. Skreta et al. (ICML 2025),
   [*Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts*][fkc-paper].
3. Xie et al. (2026),
   [*Enhanced Diffusion Sampling: Efficient Rare Event Sampling and Free Energy
   Calculation with Diffusion Models*][enhanced-sampling-paper], arXiv:2602.16634v2.

[fk-paper]: https://proceedings.mlr.press/v267/singhal25b.html
[fkc-paper]: https://proceedings.mlr.press/v267/skreta25a.html
[enhanced-sampling-paper]: https://arxiv.org/pdf/2602.16634v2
[boltz-sampler]: https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/src/boltz/model/modules/diffusionv2.py
