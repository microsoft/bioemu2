"""Local setup helpers used by the public sampling recipe."""

import hashlib
import logging
import re
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.swa_utils import AveragedModel

from bioemu2.biomoa.models.edm_schedule import EDMSchedule
from bioemu2.biomoa.models.score_model import (
    BioMoaCModel,
    DenoiseOutput,
    TrunkEmbeddings,
)
from bioemu2.biomoa.modules.diffusion import StaticContext
from bioemu2.checkpoints import (
    DEFAULT_CHECKPOINT,
    get_checkpoint_files,
    resolve_checkpoint_files,
)
from bioemu2.conditioning.method import SamplingMode
from bioemu2.context_parallel.models.score_model import CPTrunkEmbeddings
from bioemu2.context_parallel.modules.diffusion import CPStaticContext
from bioemu2.data_interface import Contact, System
from bioemu2.embedding_cache import (
    cache_directory,
    cache_writable,
    entry_metadata,
    load_embeddings,
    save_embeddings,
)
from bioemu2.msa.alignment import MSA
from bioemu2.precision import autocast_ctx, get_autocast_dtype, matmul_precision_ctx
from bioemu2.structure import BatchAllAtomStructure
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.preparation import prepare_structure

logger = logging.getLogger(__name__)


class InferenceModel[TrunkT: TrunkEmbeddings | CPTrunkEmbeddings](Protocol):
    """Model interface required for sampling structures."""

    @property
    def device(self) -> torch.device: ...

    @property
    def edm_schedule(self) -> EDMSchedule: ...

    @property
    def supports_propagation(self) -> bool: ...

    def compute_static_context(
        self,
        batch: BatchAllAtomStructure,
        trunk_embeddings: TrunkEmbeddings | CPTrunkEmbeddings,
        multiplicity: int,
    ) -> StaticContext | CPStaticContext:
        """Run the part of the structure-module processing that does not use noisy
        atom coordinates.

        ``batch`` contains structural chemistry, masks, atom-token mappings, and
        optional dynamics conditioning. Sampling mode and contact conditioning are
        already represented by ``trunk_embeddings`` and are not read from the batch.
        ``multiplicity`` is the number of structures sampled per system.
        """
        ...

    def denoise(
        self,
        context: StaticContext | CPStaticContext,
        trunk_embeddings: TrunkEmbeddings | CPTrunkEmbeddings,
        coords: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> DenoiseOutput:
        """Run the coordinate-dependent structure-module processing.

        This method is called at every diffusion timestep.

        The score-model operation supports ``B > 1`` and assumes batch-major
        coordinates: each context entry maps to ``M`` consecutive coordinate samples.
        ``M`` is inferred from the coordinate and context batch sizes.
        The low-level sampler functions currently require a single-system batch.

        Args:
            context: Coordinate-independent inputs for a batch of ``B`` systems, as
                returned by ``compute_static_context``.
            trunk_embeddings: Sequence-trunk outputs for the same ``B`` systems.
            coords: Noisy atom coordinates with shape ``(B * M, N_atom, 3)``, where
                ``M`` is the number of samples per system.
            sigmas: Noise level for each coordinate sample, with shape ``(B * M,)``.
        Returns:
            ``DenoiseOutput`` containing the unitless structure-module output in
            ``score`` and the denoised coordinates in ``denoised_coords``. Both
            tensors have the same shape as ``coords``.
        """
        ...

    def embed(
        self,
        system: System,
        *,
        msa: MSA | None,
        mode: SamplingMode,
        contacts: list[Contact] | None = None,
    ) -> TrunkT:
        """Compute the sequence-trunk embeddings for a system."""
        ...


class _InferenceModel(nn.Module):
    """Wrap ``BioMoaCModel`` as an ``InferenceModel``.

    Fix the number of trunk recycling steps and the mixed precision setting, enable
    eval mode, and freeze model parameters.
    """

    def __init__(
        self,
        score_model: BioMoaCModel,
        num_steps_recycle: int,
        autocast_dtype: torch.dtype | None,
        model_fingerprint: str,
        embedding_cache_dir: Path | None = None,
        matmul_precision: str | None = None,
    ) -> None:
        """Initialize the inference wrapper.

        Args:
            score_model: BioMoaC model whose weights have already been loaded.
            num_steps_recycle: Number of trunk recycling steps.
            autocast_dtype: Autocast dtype for trunk and static-context computation.
                Denoising is not autocast.
            model_fingerprint: Identity of the loaded checkpoint and inference config.
            embedding_cache_dir: Persistent cache directory, or None to disable caching.
        """
        super().__init__()
        self.score_model = score_model
        self.num_steps_recycle = num_steps_recycle
        self._autocast_dtype = autocast_dtype
        self.model_fingerprint = model_fingerprint
        self.embedding_cache_dir = embedding_cache_dir
        self._matmul_precision = matmul_precision

    @property
    def device(self) -> torch.device:
        return next(self.score_model.parameters()).device

    @property
    def edm_schedule(self) -> EDMSchedule:
        return self.score_model.edm_schedule

    @property
    def supports_propagation(self) -> bool:
        return self.score_model.supports_propagation

    @torch.no_grad()
    def embed(
        self,
        system: System,
        *,
        msa: MSA | None,
        mode: SamplingMode,
        contacts: list[Contact] | None = None,
    ) -> TrunkEmbeddings:
        """Compute the sequence-trunk embeddings for a system."""
        if msa is None:
            raise ValueError("msa is required for single-device embedding.")
        cache_metadata = None
        if self.embedding_cache_dir is not None:
            cache_metadata = entry_metadata(
                system,
                msa,
                mode,
                contacts,
                self.model_fingerprint,
                self._autocast_dtype,
                self.num_steps_recycle,
            )
            cached = load_embeddings(self.embedding_cache_dir, cache_metadata, self.device)
            if cached is not None:
                logger.info("Loaded trunk embeddings from disk cache.")
                return cached
            if not cache_writable(self.embedding_cache_dir):
                cache_metadata = None
        structure = prepare_structure(
            system,
            msa=msa,
            mode=mode,
            contacts=[] if contacts is None else contacts,
        )
        batch = batch_collate([structure]).to(self.device)
        embeddings = self._run_trunk(batch)
        if cache_metadata is not None:
            assert self.embedding_cache_dir is not None  # shut up mypy
            save_embeddings(self.embedding_cache_dir, cache_metadata, embeddings)
        return embeddings

    def _run_trunk(
        self,
        batch: BatchAllAtomStructure,
    ) -> TrunkEmbeddings:
        """Compute trunk embeddings using the fixed recycling and autocast settings."""

        with autocast_ctx(device_type=self.device.type, dtype=self._autocast_dtype):
            return self.score_model.compute_trunk_embeddings(batch, self.num_steps_recycle)

    @torch.no_grad()
    def compute_static_context(
        self,
        batch: BatchAllAtomStructure,
        trunk_embeddings: TrunkEmbeddings | CPTrunkEmbeddings,
        multiplicity: int,
    ) -> StaticContext:
        """Compute static structure-module context using the fixed autocast setting."""
        if not isinstance(trunk_embeddings, TrunkEmbeddings):
            raise TypeError(f"Expected type {TrunkEmbeddings}, got type {type(trunk_embeddings)}.")
        # structure model is always with fp32. no casting
        return self.score_model.compute_static_context(batch, trunk_embeddings)

    def denoise(
        self,
        context: StaticContext | CPStaticContext,
        trunk_embeddings: TrunkEmbeddings | CPTrunkEmbeddings,
        coords: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> DenoiseOutput:
        """Run the coordinate-dependent part of the structure module.

        This intentionally runs without autocasting, but at the model's float32 matmul precision.

        Args:
            context: Coordinate-independent inputs returned by
                ``compute_static_context``.
            coords: Noisy atom coordinates.
            sigmas: Noise level for each coordinate sample.

        Returns:
            ``DenoiseOutput`` containing the unitless model output and denoised
            coordinates.
        """
        if not isinstance(context, StaticContext):
            raise TypeError(f"Expected type {StaticContext}, got type {type(context)}.")
        if not isinstance(trunk_embeddings, TrunkEmbeddings):
            raise TypeError(f"Expected type {TrunkEmbeddings}, got type {type(trunk_embeddings)}.")
        with matmul_precision_ctx(self._matmul_precision):
            results = self.score_model.denoise(context, trunk_embeddings, coords, sigmas)
        return results


def _load_score_model(
    checkpoint_path: Path,
    config_path: Path,
    *,
    fingerprint_required: bool,
) -> tuple[nn.Module, str]:
    """Instantiate and load EMA weights from a BioEmu2 release checkpoint and config."""
    model_config = OmegaConf.load(config_path)
    if not isinstance(model_config, DictConfig) or "score_model" not in model_config:
        raise ValueError(f"Config {config_path} must define a score_model mapping.")
    score_model = instantiate(model_config.score_model)
    averaged_model = AveragedModel(score_model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    averaged_model.load_state_dict(checkpoint["model_ema"])
    logger.info("Loaded EMA parameters.")
    model_fingerprint = checkpoint.get("model_fingerprint")
    if not isinstance(model_fingerprint, str) or not model_fingerprint:
        if fingerprint_required:
            logger.warning(
                "Checkpoint %s has no model fingerprint; trunk embeddings will not be cached.",
                checkpoint_path,
            )
        return averaged_model.module, ""
    config_fingerprint = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return averaged_model.module, f"{model_fingerprint}:{config_fingerprint}"


def load_model_for_inference(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
    num_steps_recycle: int,
    precision: str,
    embedding_cache_dir: Path | None = None,
    compile_denoiser: bool = False,
    matmul_precision: str | None = None,
) -> InferenceModel[TrunkEmbeddings]:
    """Load an EMA score model with fixed inference settings."""
    if compile_denoiser and (device.type != "cuda"):
        raise ValueError("compile_denoiser requires a CUDA device.")
    if matmul_precision is not None and (device.type != "cuda"):
        logger.warning(
            f"matmul_precision {matmul_precision} requires a CUDA device. The setting has no effects",
        )
    autocast_dtype = get_autocast_dtype(precision)
    model, fingerprint = _load_score_model(
        checkpoint_path, config_path, fingerprint_required=embedding_cache_dir is not None
    )
    model = model.to(device)
    if not isinstance(model, BioMoaCModel):
        raise TypeError(f"Expected a BioMoaCModel, got {type(model).__name__}.")
    if compile_denoiser:
        # The denoiser layers share one code path, so one compilation serves all of them.
        for layer in model.structure_module.layers:
            layer.compile(fullgraph=True)
    # Freeze model parameters without disabling autograd for inputs. Some steering
    # algorithms differentiate denoised coordinates with respect to noisy coordinates.
    return (
        _InferenceModel(
            score_model=model,
            num_steps_recycle=num_steps_recycle,
            autocast_dtype=autocast_dtype,
            model_fingerprint=fingerprint,
            embedding_cache_dir=embedding_cache_dir if fingerprint else None,
            matmul_precision=matmul_precision,
        )
        .eval()
        .requires_grad_(False)
    )


def _check_cuda_device_supported(device: torch.device) -> None:
    """Raise if the installed PyTorch build has no kernels for a CUDA device.

    PyTorch only warns about an unsupported GPU and later fails inside the first
    kernel launch with ``no kernel image is available``. Checking up front gives an
    actionable message instead.

    Args:
        device: Device to check. Non-CUDA devices are accepted unchanged.

    Raises:
        RuntimeError: If no compiled ``sm_*`` binary or ``compute_*`` PTX in the
            PyTorch build can run on ``device``.
    """
    if device.type != "cuda":
        return
    arch_list = torch.cuda.get_arch_list()
    if not arch_list:
        return
    major, minor = torch.cuda.get_device_capability(device)
    device_cc = 10 * major + minor
    for arch in arch_list:
        match = re.fullmatch(r"(sm|compute)_(\d+)[a-z]?", arch)
        if match is None:
            continue
        kind, code_cc = match.group(1), int(match.group(2))
        # SASS binaries run on later minor revisions of the same major architecture;
        # PTX is JIT-compiled for any later architecture.
        if code_cc <= device_cc and (kind == "compute" or code_cc // 10 == major):
            return
    raise RuntimeError(
        f"{torch.cuda.get_device_name(device)} (compute capability {major}.{minor}) is not "
        f"supported by the installed PyTorch {torch.__version__}, which was built for "
        f"{', '.join(arch_list)}. For compute capability 7.0 to 7.4, reinstall with "
        '`python -m pip install ".[cuda12]" '
        "--extra-index-url https://download.pytorch.org/whl/cu126`. "
        "For a development checkout, see DEVELOPER.md."
    )


def load_model(
    checkpoint: str | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    precision: str = "bf16-mixed",
    compile_denoiser: bool = False,
    matmul_precision: str | None = "high",
) -> InferenceModel[TrunkEmbeddings]:
    """Load a BioEmu2 checkpoint from Hugging Face or from a local file.

    When neither checkpoint argument is given, load ``DEFAULT_CHECKPOINT``.
    To cache trunk embeddings across runs, set ``BIOEMU2_EMBEDS_CACHE`` to a
    directory. Embedding caching is disabled when the variable is unset.

    Args:
        checkpoint: Checkpoint directory name in the ``microsoft/bioemu2`` Hugging Face
            repository, one of ``SUPPORTED_CHECKPOINTS``.
            Defaults to ``DEFAULT_CHECKPOINT`` when ``checkpoint_path`` is not given.
        checkpoint_path: Local checkpoint file with its ``config.yaml`` in the same directory.
        cache_dir: Directory for downloaded Hugging Face checkpoints. Defaults to the
            standard Hugging Face cache, which ``HF_HUB_CACHE`` or ``HF_HOME`` can
            relocate. Not allowed with ``checkpoint_path``.
        device: Inference device. Defaults to CUDA when available, otherwise CPU.
        precision: Inference precision policy.
        compile_denoiser: Compile the denoiser layers with ``torch.compile``. The first
            denoising step compiles them, which takes a few seconds; later steps and
            batches run faster. Requires CUDA.
        matmul_precision: Float32 matmul precision during denoising, as in
            ``torch.set_float32_matmul_precision``. ``"high"`` uses TF32 tensor cores,
            which is several times faster on Ampere and newer GPUs at slightly lower
            matmul precision. ``None`` is not a precision level: it leaves PyTorch's
            global setting unchanged, which defaults to ``"highest"`` (full-precision
            float32 matmuls), so ``None`` is effectively higher precision than ``"high"``.
            Embedding and static-context computation are unaffected. Requires CUDA unless ``None``.

    Returns:
        Loaded model ready for embedding and sampling.

    Raises:
        ValueError: If both checkpoint arguments are given, ``cache_dir`` is given
            with ``checkpoint_path``, the name is not in ``SUPPORTED_CHECKPOINTS``, the
            precision is invalid, or
            ``compile_denoiser`` or ``matmul_precision`` is requested without CUDA.
        RuntimeError: If the installed PyTorch build does not support the CUDA device.
    """
    if checkpoint is not None and checkpoint_path is not None:
        raise ValueError("Pass only one of checkpoint or checkpoint_path.")
    if checkpoint_path is not None and cache_dir is not None:
        raise ValueError("cache_dir applies only to Hugging Face checkpoints, not checkpoint_path.")
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _check_cuda_device_supported(resolved_device)
    name = DEFAULT_CHECKPOINT if checkpoint is None else checkpoint
    if checkpoint_path is not None:
        checkpoint_files = resolve_checkpoint_files(checkpoint_path)
    else:
        checkpoint_files = get_checkpoint_files(name, cache_dir=cache_dir)
    return load_model_for_inference(
        checkpoint_path=checkpoint_files.checkpoint_path,
        config_path=checkpoint_files.config_path,
        device=resolved_device,
        num_steps_recycle=3,
        precision=precision,
        embedding_cache_dir=cache_directory(),
        compile_denoiser=compile_denoiser,
        matmul_precision=matmul_precision,
    )
