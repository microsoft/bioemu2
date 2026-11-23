# Define steering potentials

A steering potential is an instance of a `Potential` subclass. It implements
`prepare(system)` for system-specific data and
`__call__(coords_angstrom, *, t)` to return an "energy" for each structure.
Steering tries to produce samples with lower energy. Choose potentials to
describe the properties you want to favour, then pass the instances to a
sampler. See the [steering guide](steering.md) for information about the
available steered sampling algorithms.

## Choose from the available potentials

The following potentials are defined in
[`bioemu2.denoising.steering.potentials`](../src/bioemu2/denoising/steering/potentials.py).
Many are copied and adapted from Boltz's
[`potentials.py`](https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/src/boltz/model/potentials/potentials.py).

| Potential | What it penalises |
| --- | --- |
| `BondLengthPotential` | Bond lengths outside reference bounds |
| `BondAnglePotential` | Bond angles outside reference bounds |
| `InterChainVDWOverlapPotential` | Van der Waals overlaps between atoms in different chains, excluding atom pairs at most three covalent bonds apart through a declared inter-chain bond |
| `IntraChainVDWOverlapPotential` | Van der Waals overlaps within polymer chains, excluding pairs at most three covalent bonds apart |
| `PolymerLocalClashPotential` | Short-range inter-residue clashes for pairs two or three covalent bonds apart, within polymer chains or across declared bonds |
| `SymmetricChainCOMPotential` | Centres of mass of equivalent chains getting too close |
| `PoseBustersPotential` | Violations of ligand distance bounds derived from bonds, angles, and non-bonded contacts |
| `ChiralAtomPotential` | Incorrect chirality at ligand stereocentres |
| `StereoBondPotential` | Incorrect ligand bond stereochemistry |
| `PlanarBondPotential` | Non-planar ligand bonds that should be planar |

These ten potentials form the default **physicality potentials** used by
`GuidedFKSampler` and `FKCSampler`. Other potentials can describe structural
properties for different purposes, such as favouring a particular protein
conformation.

`PoseBustersPotential`, `ChiralAtomPotential`, `StereoBondPotential`, and
`PlanarBondPotential` derive their ligand constraints with RDKit from the
ligand's CCD code, SMILES, or InChI. Chiral and stereo terms apply only to
stereocentres and double bonds whose configuration the input specifies, such as
`@`/`@@` and `/`/`\` in SMILES. Stereocentres with a lone pair, such as
sulfoxides, get no chiral term.

The module also provides two more specialised potentials:
`ContactPotential` penalises atom pairs exceeding a contact distance, and
`TemplateReferencePotential` penalises deviations from a rigid-aligned reference
template.

## Improve physicality with the default physicality potentials

`GuidedFKSampler()` and `FKCSampler()` both use the default physicality potentials,
as do the CLI's `--sampler edm_gd`, `edm_fk`, and `edm_fkc`
options. Start with `edm_gd` for gradient-based steering without particle
resampling.

These potentials discourage clashes, broken bonds, and other geometry problems
during generation. They do not guarantee that every sample will pass physicality
checks or that its overall conformation is more accurate. The physicality filter
only removes invalid structures; it does not repair them. Steering is especially
useful for large structures, where few unsteered samples may pass the checks.

## Select potentials to apply

The default physicality potentials are defined in
[`get_default_potentials()`](../src/bioemu2/denoising/steering/potentials.py).
You can use a different set of potentials instead of the default physicality potentials.

In Python, construct the potentials you want and pass them to the sampler as
`potentials=[...]`. This replaces the defaults; it does not add to them. See the
[Python example](steering.md#sample-with-steering).

On the CLI, pass `--potentials my.yaml`. In that file, `_target_` names the
potential class and the remaining fields supply its constructor arguments:

```yaml
potentials:
  - _target_: bioemu2.denoising.steering.potentials.BondLengthPotential
    sigma_tolerance: 10.0
    parameters: {guidance_weight: 0.05, resampling_weight: 0.1}
```

This file also replaces the default set, so the example applies only a
bond-length restraint.

See [`potentials.py`](../src/bioemu2/denoising/steering/potentials.py) for
descriptions, settings, and parameters of the predefined potentials.

## Configure strength and schedules

For `GuidedFKSampler`, use each potential's `parameters` mapping to control its
influence, as in the example above:

| Parameter | Effect |
| --- | --- |
| `guidance_weight` | Scales the potential's coordinate gradient during guidance |
| `guidance_interval` | Applies that gradient every N inner steps |
| `resampling_weight` | Scales the potential's contribution to reweighting/resampling |

Scheduled parameter values can change as denoising progresses. Use a
`ParameterSchedule` such as
`PiecewiseStepFunction`, `ExponentialInterpolation`, or `LinearInterpolation`.
Normalised potential time runs from `t=1` at high noise to `t=0` at the clean
endpoint. See the
[`potential and schedule docstrings`](../src/bioemu2/denoising/steering/potentials.py)
for supported parameters, units, and schedule arguments.

`FKCSampler` uses the returned energies directly; it does not use
`guidance_weight`, `guidance_interval`, or `resampling_weight`. To change a
potential's strength for FKC, scale its returned energy. Potential-specific
parameters, such as a clash buffer, still determine that energy.

## Write your own potential

Define a subclass of `Potential` with `prepare(system)` to set up system-specific
data and `__call__(coords_angstrom, *, t)` to return one energy per structure.
The sampling API calls `prepare()` for you. When evaluating a potential directly
outside a sampler, call `prepare(system)` first and again when changing systems.
The [`Potential` API docstrings](../src/bioemu2/denoising/steering/potentials.py)
specify coordinate units, tensor shapes, and padding.

For guidance (gradient-based steering), the energy must be
differentiable with respect to the coordinates. Keep its calculation in PyTorch
without detaching the coordinates. Resampling-only steering needs energy values
but not gradients. See
[choose a sampler](steering.md#choose-a-sampler) for which sampler
configurations use guidance.

> [!NOTE]
> Each denoising step randomly rotates and translates the coordinates. A
> potential that changes under these transformations can give unintended
> results. Define potentials using rotation- and translation-invariant quantities,
> such as distances, rather than absolute positions. For example, a potential
> that compares coordinates with a fixed reference without first rigidly
> aligning them, or rewards elongation along a fixed global axis, changes when
> the sampler rotates or translates the structure.

For a complete custom potential, see `TerminiTargetExpPotential` in the
[chignolin example](../examples/tilt_a_sample_distribution_with_steering.py).
It caches the terminal C-alpha indices and returns a quadratic penalty on their
separation. See [sample with steering](steering.md#sample-with-steering) for how
to pass a potential to a sampler.

For contacts supplied directly as model inputs rather than as a steering
potential, use [contact conditioning](system.md#define-contact-conditioning).
