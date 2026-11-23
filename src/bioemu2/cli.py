"""Command-line interface for basic usage."""

import logging
import shlex
import sys
from collections.abc import Callable, Sequence
from copy import copy
from functools import partial
from pathlib import Path
from typing import cast, get_args

import click
import mdtraj
import torch.distributed as dist
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from bioemu2.checkpoints import DEFAULT_CHECKPOINT, SUPPORTED_CHECKPOINTS
from bioemu2.conditioning.method import SamplingMode
from bioemu2.context_parallel import api as cp_api
from bioemu2.data_interface import Contact, System, convert_boltz, load_system_and_contacts
from bioemu2.denoising.samplers import Sampler
from bioemu2.denoising.steering.potentials import Potential
from bioemu2.model import load_model
from bioemu2.msa.alignment import get_msa
from bioemu2.physicality import PhysicalityFilterMode
from bioemu2.propagation import propagate, propagate_with_embeddings
from bioemu2.sampling import (
    SamplerPreset,
    predict_structure,
    sample_equilibrium,
    sampler_from_preset,
)

logger = logging.getLogger(__name__)

SAMPLER_PRESETS = list(get_args(SamplerPreset))


def _sampler(
    preset: str, potentials: Path | None, *, mode: SamplingMode = "equilibrium"
) -> Sampler:
    """Build the sampler for ``--sampler PRESET [--potentials YAML]``.

    ``edm`` is unsteered. The other presets use the default physicality potentials;
    ``potentials`` points to a YAML with a top-level ``potentials:`` list and
    replaces that default list.
    """
    return sampler_from_preset(preset, mode=mode, potentials=_load_potentials(preset, potentials))


def _load_potentials(preset: str, path: Path | None) -> Sequence[Potential] | None:
    if path is None:
        return None
    if preset not in ("edm_gd", "edm_fk", "edm_fkc"):
        raise ValueError("--potentials requires a steering sampler (edm_gd, edm_fk, edm_fkc)")
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig) or "potentials" not in config:
        raise ValueError(f"{path} must have a top-level `potentials:` list")
    return tuple(instantiate(config.potentials))


def _common_options(function: Callable) -> Callable:
    options = [
        click.option(
            "--system",
            type=click.Path(path_type=Path, dir_okay=False),
            metavar="YAML",
            help="Path to a system YAML file describing the molecular entities and optional "
            "contacts. Packaged examples are available under bioemu2.paths.EXAMPLES_DIR. "
            "Sampling commands require --system or "
            "--pdb-id; propagate can infer the system from --x0-cif when both are omitted.",
        ),
        click.option(
            "--pdb-id",
            metavar="ID",
            help="PDB ID to download from RCSB as biological assembly 1, including all chains "
            "and ligands, for example 1UBQ or 101M. Sampling commands require --system or "
            "--pdb-id; propagate can infer the system from --x0-cif when both are omitted.",
        ),
        click.option(
            "--checkpoint",
            type=click.Choice(SUPPORTED_CHECKPOINTS),
            help="Checkpoint name in the microsoft/bioemu2 Hugging Face repository. "
            f"Defaults to {DEFAULT_CHECKPOINT} when --checkpoint-path is not given.",
        ),
        click.option(
            "--checkpoint-path",
            type=click.Path(path_type=Path, dir_okay=False),
            metavar="CKPT",
            help="Path to a local checkpoint file instead of a Hugging Face checkpoint. "
            "The matching config.yaml must be in the same directory.",
        ),
        click.option(
            "--checkpoint-cache-dir",
            type=click.Path(path_type=Path, file_okay=False),
            metavar="DIR",
            help="Directory for downloaded Hugging Face checkpoints. Defaults to the standard "
            "Hugging Face cache, which HF_HUB_CACHE or HF_HOME can relocate. Cannot be used "
            "with --checkpoint-path.",
        ),
        click.option(
            "--out",
            required=True,
            type=click.Path(path_type=Path, file_okay=False),
            metavar="DIR",
            help="Directory for generated trajectories, topology.cif, system.yaml, "
            "physicality.json, command.txt, and raw batches in batches/. It is created if it "
            "does not exist. Rerunning with the same --out continues from completed batches "
            "and overwrites the other files.",
        ),
        click.option(
            "--batch-size",
            type=click.IntRange(min=1),
            help="Number of structures denoised together. Defaults to --num-samples for "
            "sampling commands and the number of starting conformations for propagation.",
        ),
        click.option(
            "--seed",
            type=int,
            help="Seed for random number generation. By default a seed is drawn, so repeated "
            "runs generally produce different random samples. To continue a run, rerun with the "
            "same --seed or again without one.",
        ),
        click.option(
            "--num-steps",
            type=click.IntRange(min=1),
            help="Number of denoising steps. By default each sampler uses its configured "
            "value, currently 200 or 500 depending on the sampler.",
        ),
        click.option(
            "--sampler",
            type=click.Choice(SAMPLER_PRESETS),
            default="edm",
            show_default=True,
            help="Sampling algorithm. EDM is unsteered; edm_gd uses gradient guidance, "
            "edm_fk adds Feynman-Kac resampling, and edm_fkc uses the Feynman-Kac corrector.",
        ),
        click.option(
            "--potentials",
            type=click.Path(path_type=Path, dir_okay=False),
            metavar="YAML",
            help="Path to a YAML file containing a top-level 'potentials' list. This replaces "
            "the default physicality potentials and requires edm_gd, edm_fk, or edm_fkc.",
        ),
        click.option(
            "--physicality-filter",
            type=click.Choice([mode.value for mode in PhysicalityFilterMode]),
            default=PhysicalityFilterMode.HIGH_QUALITY.value,
            show_default=True,
            metavar="MODE",
            help="Physicality checks reported per frame for invalid bond lengths, clashes, "
            "chirality, or chain breaks. Sampling outputs retain all frames; use no_filter "
            "to skip checks.",
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def _speed_options(function: Callable) -> Callable:
    options = [
        click.option(
            "--precision",
            type=click.Choice(["32-true", "bf16-mixed"]),
            default="bf16-mixed",
            show_default=True,
            help="Precision for trunk embedding. Static-context computation and denoising "
            "always run in full precision; the denoiser's float32 matmul precision is set "
            "by --matmul-precision.",
        ),
        click.option(
            "--compile-denoiser",
            is_flag=True,
            help="Not recommended for light workload because compilation takes time. For large number of "
            "samples and very small system size ( < 128 token), turn it on. "
            "If use `high` or `medium` matmul precision, turn it on.",
        ),
        click.option(
            "--matmul-precision",
            type=click.Choice(["highest", "high", "medium"]),
            default="high",
            show_default=True,
            help="Float32 matmul precision during denoising. "
            "'high' uses TF32 tensor cores on supported GPUs at slightly lower precision.",
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def _resolve_common(
    *,
    system: Path | None,
    pdb_id: str | None,
    checkpoint: str | None,
    checkpoint_path: Path | None,
    checkpoint_cache_dir: Path | None,
    x0_cif: Path | None = None,
) -> tuple[System, list[Contact] | None, str | None]:
    """Resolve and validate shared CLI inputs before loading the model."""
    if (system is None and pdb_id is None and x0_cif is None) or (
        system is not None and pdb_id is not None
    ):
        raise click.UsageError("Provide exactly one of --system or --pdb-id.")
    if checkpoint is None and checkpoint_path is None:
        checkpoint = DEFAULT_CHECKPOINT
    if (checkpoint is None) == (checkpoint_path is None):
        raise click.UsageError("Provide exactly one of --checkpoint or --checkpoint-path.")
    if checkpoint_path is not None and checkpoint_cache_dir is not None:
        raise click.UsageError("--checkpoint-cache-dir cannot be used with --checkpoint-path.")
    if system is not None:
        resolved_system, contacts = load_system_and_contacts(system)
    elif pdb_id is not None:
        resolved_system, contacts = System.from_pdb_id(pdb_id), None
    else:
        assert x0_cif is not None
        resolved_system, contacts = System.from_cif(x0_cif), None
    return resolved_system, contacts, checkpoint


def validate_x0(trajectory: mdtraj.Trajectory, system: System) -> mdtraj.Trajectory:
    """Check that starting conformations have the system's atoms and topology."""
    atoms = [
        (a.name, a.element.symbol, a.residue.name, a.residue.chain.index)
        for a in trajectory.topology.atoms
    ]
    expected = [
        (a.name, a.element.symbol, a.residue.name, a.residue.chain.index)
        for a in system.topology.atoms
    ]
    if atoms != expected:
        k = next((k for k, (a, e) in enumerate(zip(atoms, expected)) if a != e), None)
        detail = (
            f"{len(atoms)} atoms, system has {len(expected)}"
            if k is None
            else f"atom {k} is {atoms[k]}, system has {expected[k]}"
        )
        raise ValueError(f"--x0-cif atoms do not match the system: {detail}")
    # Same atoms, so use the system's topology: the CIF writer adds ligand bonds that
    # System.topology lacks, and propagate() compares bonds too.
    trajectory.topology = system.topology
    return trajectory


def _write_command(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "command.txt").write_text(shlex.join(sys.argv) + "\n")


@click.group()
def main() -> None:
    """Sample structures and trajectories for a molecular system with BioEmu2."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    # huggingface_hub logs every HTTP request through httpx at INFO, including the
    # revision checks made for files that are already cached.
    logging.getLogger("httpx").setLevel(logging.WARNING)


@main.command("convert-boltz")
@click.argument("source", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option(
    "--out",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    metavar="YAML",
    help="Destination BioEmu2 system YAML. Parent directories are created; existing files "
    "are not overwritten.",
)
def convert_boltz_command(source: Path, out: Path) -> None:
    """Convert Boltz-1 or Boltz-2 input YAML to BioEmu2 system YAML.

    Both use input schema version 1. Conversion does not load a model.
    Boltz MSAs and unsupported non-contact fields are ignored with warnings.
    Boltz contact force settings are ignored with warnings; unsupported contact
    fields raise errors. See docs/boltz.md for details.
    """
    try:
        convert_boltz(source, out)
    except (OSError, TypeError, ValueError, KeyError, NotImplementedError, yaml.YAMLError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Wrote BioEmu2 system YAML to {out}")


@main.command("sample-equilibrium")
@_common_options
@_speed_options
@click.option(
    "--num-samples",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Number of structures to generate.",
)
def sample_equilibrium_command(
    num_samples: int,
    system: Path | None,
    pdb_id: str | None,
    checkpoint: str | None,
    checkpoint_path: Path | None,
    checkpoint_cache_dir: Path | None,
    out: Path,
    batch_size: int | None,
    seed: int | None,
    num_steps: int | None,
    sampler: str,
    potentials: Path | None,
    precision: str,
    physicality_filter: str,
    compile_denoiser: bool,
    matmul_precision: str | None,
) -> None:
    """Sample the equilibrium ensemble."""
    resolved_system, contacts, resolved_checkpoint = _resolve_common(
        system=system,
        pdb_id=pdb_id,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        checkpoint_cache_dir=checkpoint_cache_dir,
    )
    resolved_potentials = _load_potentials(sampler, potentials)
    _write_command(out)
    model = load_model(
        checkpoint=resolved_checkpoint,
        checkpoint_path=checkpoint_path,
        cache_dir=checkpoint_cache_dir,
        precision=precision,
        compile_denoiser=compile_denoiser,
        matmul_precision=matmul_precision,
    )
    sample_equilibrium(
        resolved_system,
        model=model,
        contacts=contacts,
        num_samples=num_samples,
        batch_size=batch_size,
        sampler=cast(SamplerPreset, sampler),
        num_steps=num_steps,
        potentials=resolved_potentials,
        seed=seed,
        output_dir=out,
        physicality_mode=physicality_filter,
    )


@main.command("predict-structure")
@_common_options
@_speed_options
@click.option(
    "--num-samples",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Number of structures to generate.",
)
def predict_structure_command(
    num_samples: int,
    system: Path | None,
    pdb_id: str | None,
    checkpoint: str | None,
    checkpoint_path: Path | None,
    checkpoint_cache_dir: Path | None,
    out: Path,
    batch_size: int | None,
    seed: int | None,
    num_steps: int | None,
    sampler: str,
    potentials: Path | None,
    precision: str,
    physicality_filter: str,
    compile_denoiser: bool,
    matmul_precision: str | None,
) -> None:
    """Predict the folded structure."""
    resolved_system, contacts, resolved_checkpoint = _resolve_common(
        system=system,
        pdb_id=pdb_id,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        checkpoint_cache_dir=checkpoint_cache_dir,
    )
    resolved_potentials = _load_potentials(sampler, potentials)
    _write_command(out)
    model = load_model(
        checkpoint=resolved_checkpoint,
        checkpoint_path=checkpoint_path,
        cache_dir=checkpoint_cache_dir,
        precision=precision,
        compile_denoiser=compile_denoiser,
        matmul_precision=matmul_precision,
    )
    predict_structure(
        resolved_system,
        model=model,
        contacts=contacts,
        num_samples=num_samples,
        batch_size=batch_size,
        sampler=cast(SamplerPreset, sampler),
        num_steps=num_steps,
        potentials=resolved_potentials,
        seed=seed,
        output_dir=out,
        physicality_mode=physicality_filter,
    )


@main.command("propagate")
@_common_options
@_speed_options
@click.option(
    "--x0-cif",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    metavar="CIF",
    help="Topology of the starting conformations; atoms must match the system.",
)
@click.option(
    "--x0-xtc",
    type=click.Path(path_type=Path, dir_okay=False),
    metavar="XTC",
    help="Sample one endpoint per input frame. Pass exactly one of --x0-xtc and --num-samples.",
)
@click.option(
    "--num-samples",
    type=click.IntRange(min=1),
    metavar="N",
    help="Number of endpoints to sample from the structure in --x0-cif. "
    "Pass exactly one of --num-samples and --x0-xtc.",
)
@click.option(
    "--lag-time-ns",
    required=True,
    type=click.FloatRange(min=0, min_open=True),
    metavar="T",
    help="Time lag in nanoseconds. We recommend 1 to 1000 (1 ns to 1 µs), the range "
    "the model mostly saw during training. Lags longer than a few milliseconds "
    "effectively disable conditioning on the starting structure.",
)
def propagate_command(
    x0_cif: Path,
    x0_xtc: Path | None,
    num_samples: int | None,
    lag_time_ns: float,
    system: Path | None,
    pdb_id: str | None,
    checkpoint: str | None,
    checkpoint_path: Path | None,
    checkpoint_cache_dir: Path | None,
    out: Path,
    batch_size: int | None,
    seed: int | None,
    num_steps: int | None,
    sampler: str,
    potentials: Path | None,
    precision: str,
    physicality_filter: str,
    compile_denoiser: bool,
    matmul_precision: str | None,
) -> None:
    """Sample endpoints at a time lag from a CIF or each frame of an XTC.

    When neither --system nor --pdb-id is provided, derive the system from
    --x0-cif; explicit system inputs remain available for custom definitions.
    Writes every endpoint in input order to samples.xtc (without starting
    frames) and per-frame physicality results to physicality.json.
    """
    if (x0_xtc is None) == (num_samples is None):
        raise click.UsageError("Pass exactly one of --x0-xtc and --num-samples.")
    resolved_system, contacts, resolved_checkpoint = _resolve_common(
        system=system,
        pdb_id=pdb_id,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        checkpoint_cache_dir=checkpoint_cache_dir,
        x0_cif=x0_cif,
    )
    resolved_sampler = _sampler(sampler, potentials)
    if num_steps is not None:
        resolved_sampler.num_steps = num_steps
    if x0_xtc is not None:
        x0 = mdtraj.load(str(x0_xtc), top=str(x0_cif))
    else:
        x0 = mdtraj.load(str(x0_cif))
        if x0.n_frames != 1:
            raise click.UsageError("--x0-cif must contain exactly one structure without --x0-xtc.")
        assert num_samples is not None
        x0 = x0[[0] * num_samples]
    x0 = validate_x0(x0, resolved_system)
    _write_command(out)
    model = load_model(
        checkpoint=resolved_checkpoint,
        checkpoint_path=checkpoint_path,
        cache_dir=checkpoint_cache_dir,
        precision=precision,
        compile_denoiser=compile_denoiser,
        matmul_precision=matmul_precision,
    )
    if num_steps is None and potentials is None:
        propagate(
            resolved_system,
            model=model,
            x0=x0,
            lag_time_ns=lag_time_ns,
            contacts=contacts,
            batch_size=batch_size,
            sampler=cast(SamplerPreset, sampler),
            seed=seed,
            output_dir=out,
            physicality_mode=physicality_filter,
        )
    else:
        msa = get_msa(resolved_system)
        trunk_embeddings = model.embed(
            resolved_system,
            msa=msa,
            mode="equilibrium",
            contacts=contacts,
        )
        propagate_with_embeddings(
            resolved_system,
            trunk_embeddings,
            x0=x0,
            lag_time_ns=lag_time_ns,
            model=model,
            batch_size=batch_size,
            sampler=resolved_sampler,
            seed=seed,
            output_dir=out,
            physicality_mode=physicality_filter,
            contacts=contacts,
        )


@main.group("cp")
def context_parallel_commands() -> None:
    """Run context-parallel (CP) sampling on a square mesh of at least 4 GPUs.

    Example: torchrun --standalone --nproc-per-node=4
    --module bioemu2.cp sample-equilibrium --system system.yaml --out output
    """


def _run_cp_sampling_command(
    *,
    mode: SamplingMode,
    num_samples: int,
    system: Path | None,
    pdb_id: str | None,
    checkpoint: str | None,
    checkpoint_path: Path | None,
    checkpoint_cache_dir: Path | None,
    out: Path,
    batch_size: int | None,
    seed: int | None,
    num_steps: int | None,
    sampler: str,
    potentials: Path | None,
    precision: str,
    matmul_precision: str | None,
    physicality_filter: str,
) -> None:
    with cp_api.session():
        resolved: list[
            tuple[System, list[Contact] | None, Sequence[Potential] | None, str | None] | None
        ] = [None]
        if dist.get_rank() == 0:
            rank0_resolved = _resolve_common(
                system=system,
                pdb_id=pdb_id,
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
                checkpoint_cache_dir=checkpoint_cache_dir,
            )
            resolved_system, contacts, resolved_checkpoint = rank0_resolved
            resolved_potentials = _load_potentials(sampler, potentials)
            resolved[0] = (resolved_system, contacts, resolved_potentials, resolved_checkpoint)
            _write_command(out)
        dist.broadcast_object_list(resolved, src=0)
        assert resolved[0] is not None
        resolved_system, contacts, resolved_potentials, resolved_checkpoint = resolved[0]
        msa = get_msa(resolved_system) if dist.get_rank() == 0 else None
        model = cp_api.load_model(
            checkpoint=resolved_checkpoint,
            checkpoint_path=checkpoint_path,
            cache_dir=checkpoint_cache_dir,
            precision=precision,
            matmul_precision=matmul_precision,
        )
        sample = cp_api.sample_equilibrium if mode == "equilibrium" else cp_api.predict_structure
        sample(
            resolved_system,
            model=model,
            msa=msa,
            contacts=contacts,
            num_samples=num_samples,
            batch_size=batch_size,
            sampler=cast(SamplerPreset, sampler),
            num_steps=num_steps,
            potentials=resolved_potentials,
            seed=seed,
            output_dir=out,
            physicality_mode=physicality_filter,
        )


_cp_commands: list[tuple[click.Command, SamplingMode]] = [
    (sample_equilibrium_command, "equilibrium"),
    (predict_structure_command, "structure"),
]
for _command, _mode in _cp_commands:
    _params = []
    for _option in _command.params:
        # The CP model loader does not support denoiser compilation.
        if _option.name == "compile_denoiser":
            continue
        _cp_option = copy(_option)
        if isinstance(_cp_option, click.Option):
            if _cp_option.name == "sampler":
                _cp_option.type = click.Choice(["edm", "edm_gd", "edm_fk"])
                _cp_option.help = (
                    "Sampling algorithm. CP supports EDM, gradient guidance, and FK resampling."
                )
            elif _cp_option.name == "potentials":
                _cp_option.help = (
                    "YAML containing a top-level 'potentials' list; replaces the default "
                    "physicality potentials and requires edm_gd or edm_fk."
                )
            elif _cp_option.name == "precision":
                _cp_option.help = (
                    "Precision for trunk embedding. Denoising runs in full precision; "
                    "its float32 matmul precision is set by --matmul-precision."
                )
            # CP saves no batches, so it neither resumes nor saves a drawn seed.
            elif _cp_option.name == "out":
                _cp_option.help = (
                    "Directory for generated trajectories, topology.cif, system.yaml, "
                    "physicality.json, and command.txt. It is created if it does not exist; "
                    "files from an earlier run are overwritten."
                )
            elif _cp_option.name == "seed":
                _cp_option.help = (
                    "Seed for random number generation. By default no seed is set, so "
                    "repeated runs generally produce different random samples."
                )
        _params.append(_cp_option)
    context_parallel_commands.add_command(
        click.Command(
            name=_command.name,
            callback=partial(_run_cp_sampling_command, mode=_mode),
            params=_params,
            help=_command.help,
        )
    )


if __name__ == "__main__":
    main()
