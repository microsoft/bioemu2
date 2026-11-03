"""Reusable high-level sampling and propagation for one molecular system."""

import json
import logging
from pathlib import Path

import mdtraj

from bioemu2.biomoa.models.score_model import _EMBEDDINGS_FILE, TrunkEmbeddings
from bioemu2.conditioning.method import SamplingMode
from bioemu2.data_interface import Contact, System
from bioemu2.denoising.samplers import EDMSampler, GuidedFKSampler, Sampler
from bioemu2.embedding_cache import contacts_digest, msa_digest, system_digest
from bioemu2.model import InferenceModel, _InferenceModel
from bioemu2.msa.alignment import MSA, get_msa
from bioemu2.propagation import propagate_with_embeddings
from bioemu2.sampling import SamplerPreset, sample_with_embeddings, sampler_from_preset

logger = logging.getLogger(__name__)


class EmuSession:
    """Sample and propagate a system while reusing its sequence-trunk embeddings.

    The model, system, MSA, task mode, and contacts are the inputs to the
    embeddings. The model, MSA, and mode are fixed for the session. An
    instance retains at most one embedding. Changing its system raises an
    error when reading the MSA, sampling, or propagating; changing contacts
    raises on the next sampling or propagation call. Cached MSAs and embeddings
    belong to the inputs used to create the session, so use a fresh instance
    when changing the system, contacts, model, or its weights. If no MSA is
    supplied, it is computed on first use. Context-parallel sampling uses
    the functional API instead.
    """

    def __init__(
        self,
        system: System,
        model: InferenceModel[TrunkEmbeddings],
        mode: SamplingMode,
        msa: MSA | None = None,
        contacts: list[Contact] | None = None,
        embeddings: TrunkEmbeddings | None = None,
    ) -> None:
        self._system = system
        self._model = model
        self._mode = mode
        self._msa = msa
        self._contacts = contacts
        self._initial_system_fingerprint = system_digest(system)
        self._initial_contacts_fingerprint = contacts_digest(contacts)
        self._embeddings: TrunkEmbeddings | None = embeddings
        self._embeddings_preloaded = embeddings is not None

    @property
    def system(self) -> System:
        return self._system

    @property
    def model(self) -> InferenceModel[TrunkEmbeddings]:
        return self._model

    @property
    def mode(self) -> SamplingMode:
        return self._mode

    @property
    def msa(self) -> MSA:
        self._checked_system_fingerprint()
        if self._msa is None:
            self._msa = get_msa(self._system)
        return self._msa

    @property
    def contacts(self) -> list[Contact] | None:
        return self._contacts

    def cache_clear(self) -> None:
        """Drop this session's in-memory embedding, without changing any disk cache."""
        self._embeddings = None
        self._embeddings_preloaded = False

    def _checked_system_fingerprint(self) -> str:
        # System is mutable for workflows such as mutation scans, but a session's
        # MSA and embeddings belong to its original inputs. Detect edits before reuse.
        fingerprint = system_digest(self._system)
        if fingerprint != self._initial_system_fingerprint:
            raise ValueError("The session's system has changed; create a new EmuSession.")
        return fingerprint

    def _checked_contacts_fingerprint(self) -> str:
        fingerprint = contacts_digest(self._contacts)
        if fingerprint != self._initial_contacts_fingerprint:
            raise ValueError("The session's contacts have changed; create a new EmuSession.")
        return fingerprint

    def _get_embedding(self) -> TrunkEmbeddings:
        # Compute the embedding once and keep it in memory for the session. A
        # value supplied through the constructor is stored here directly, so a
        # loaded embedding skips both the trunk and the MSA it would require.
        self._checked_system_fingerprint()
        self._checked_contacts_fingerprint()
        if self._embeddings is None:
            self._embeddings = self.model.embed(
                self.system, msa=self.msa, mode=self.mode, contacts=self.contacts
            )
            self._embeddings_preloaded = False
        return self._embeddings

    def save_embeddings(self, directory: Path) -> None:
        """Save this session's trunk embeddings to a directory for later reuse.

        Compute the embeddings if needed, then write ``embeddings.pt`` into
        ``directory`` alongside a ``metadata.json`` recording the input
        fingerprints (system, MSA, contacts, mode, and model when available)
        and model settings (autocast dtype and recycling steps). BioEmu2 does
        not check the metadata on load, but you can inspect it to confirm which
        inputs produced the embeddings. The directory must be empty or not yet
        exist.

        This is independent of the automatic ``BIOEMU2_EMBEDS_CACHE`` cache: you
        choose the location and manage its lifetime. Reload the embeddings with
        ``TrunkEmbeddings.load`` and pass them to a new ``EmuSession`` through its
        ``embeddings`` argument. To save embeddings supplied to the constructor,
        use ``TrunkEmbeddings.save`` instead; their original inputs are unknown.
        """
        if self._embeddings_preloaded:
            raise ValueError(
                "Cannot save preloaded embeddings with session metadata; "
                "use TrunkEmbeddings.save(path) instead."
            )
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"Refusing to save embeddings into non-empty directory {directory}; "
                "pass an empty or new directory."
            )
        if not isinstance(self.model, _InferenceModel):
            raise TypeError("Saving session embedding metadata requires a single-device model.")
        directory.mkdir(parents=True, exist_ok=True)
        embeddings = self._get_embedding()
        metadata = {
            "system": self._initial_system_fingerprint,
            "contacts": self._initial_contacts_fingerprint,
            "msa": msa_digest(self.msa),
            "mode": self.mode,
            "autocast_dtype": str(self.model._autocast_dtype),
            "num_steps_recycle": self.model.num_steps_recycle,
        }
        if self.model.model_fingerprint:
            metadata["model"] = self.model.model_fingerprint
        embeddings.save(directory / _EMBEDDINGS_FILE)
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    def sample(
        self,
        num_samples: int = 8,
        *,
        batch_size: int | None = None,
        sampler: SamplerPreset | Sampler = "edm",
        seed: int | None = None,
        progress: bool = True,
        run_dir: Path | None = None,
        partial_dir: Path | None = None,
    ) -> mdtraj.Trajectory:
        """Draw independent structures using this instance's task mode.

        ``"structure"`` selects the prediction step scale; ``"equilibrium"``
        selects the equilibrium step scale. A configured sampler is used unchanged.

        With ``run_dir`` set, a steering sampler writes per-step diagnostics to
        ``run_dir/metrics`` -- ``x_t`` and ``x0_hat`` (saved after guidance) and
        each potential's energy. Without it no tracker is created and nothing is
        recorded, which is the default.
        """
        configured_sampler = (
            sampler_from_preset(sampler, mode=self.mode) if isinstance(sampler, str) else sampler
        )
        expected_scale = 1.5 if self.mode == "structure" else 1.0
        if (
            isinstance(configured_sampler, EDMSampler | GuidedFKSampler)
            and configured_sampler.step_scale != expected_scale
        ):
            logger.warning(
                "Sampling mode %s normally uses step_scale=%s; using the supplied "
                "step_scale=%s unchanged.",
                self.mode,
                expected_scale,
                configured_sampler.step_scale,
            )
        return sample_with_embeddings(
            self.system,
            self._get_embedding(),
            model=self.model,
            num_samples=num_samples,
            batch_size=batch_size,
            sampler=configured_sampler,
            seed=seed,
            progress=progress,
            run_dir=run_dir,
            partial_dir=partial_dir,
        )

    def propagate(
        self,
        x0: mdtraj.Trajectory,
        *,
        lag_time_ns: float,
        batch_size: int | None = None,
        sampler: SamplerPreset | Sampler = "edm",
        seed: int | None = None,
        progress: bool = True,
    ) -> mdtraj.Trajectory:
        """Draw one time-lagged endpoint per starting frame, in input order.

        Propagation normally uses equilibrium embeddings; other modes log a
        warning. The checkpoint must support propagation. Use
        ``map_cif_atom_order`` to load a CIF as ``x0``.
        """
        if self.mode != "equilibrium":
            logger.warning(
                "Propagation normally uses equilibrium embeddings; using mode=%s unchanged.",
                self.mode,
            )
        if not self.model.supports_propagation:
            raise ValueError("The loaded checkpoint does not support propagation.")
        configured_sampler = (
            sampler_from_preset(sampler, mode="equilibrium")
            if isinstance(sampler, str)
            else sampler
        )
        return propagate_with_embeddings(
            self.system,
            self._get_embedding(),
            x0=x0,
            lag_time_ns=lag_time_ns,
            model=self.model,
            batch_size=batch_size,
            sampler=configured_sampler,
            seed=seed,
            progress=progress,
        )
