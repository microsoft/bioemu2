"""Spatial-tiled OuterProductMean variants with registry-based dispatch.

Two variants:
  ``vanilla``              — full einsum baseline, O(N²·C_H²) memory. Default.
  ``triton_memory_saving`` — Triton-fused kernels at inference; falls back to
                             vanilla einsum during training.


Refs: feynman#20160, feynman#20463.
"""

from enum import StrEnum

import torch
import torch.nn as nn
from torch import Tensor

from bioemu2 import HAS_TRITON

if HAS_TRITON:
    from bioemu2.biomoa.ops.triton.opm import _TritonOPMFn

_DEFAULT_TRITON_TILE_SIZE = 256
_PROJECTION_ROWS = 256


class OPMVariant(StrEnum):
    VANILLA = "vanilla"
    TRITON_MEMORY_SAVING = "triton_memory_saving"


class _OPMBase(nn.Module):
    def __init__(self, c_in: int, c_hidden: int, c_out: int, tile_size: int | None = None) -> None:
        super().__init__()
        self.c_hidden = c_hidden
        self.tile_size = _DEFAULT_TRITON_TILE_SIZE if tile_size is None else tile_size
        self.norm = nn.LayerNorm(c_in)
        self.proj_ab = nn.Linear(c_in, 2 * c_hidden, bias=False)
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_out)


class OuterProductMeanVanilla(_OPMBase):
    """Vanilla einsum baseline — O(N²·S·C_H²) time, O(N²·C_H²) memory."""

    def forward(self, m: Tensor, mask: Tensor) -> Tensor:
        """Compute outer product mean.

        Args:
            m: MSA representation ``[B, S, N, C_in]``.
            mask: Row-validity mask ``[B, S, N]`` or ``[B, N]`` (bool or 0/1
                int/float). Uniform across N within each row — a row is valid
                if any token in it is True. Masking zeroes padding positions
                after projection (bias-free linear commutes with masking so LN
                sees unmasked input, avoiding NaN on zero vectors). Normalization
                divides by the count of valid rows per batch element.

        Returns:
            Pair representation ``[B, N, N, C_out]``.
        """
        mask = mask.bool()
        m_proj = self.proj_ab(self.norm(m))
        m_proj.masked_fill_(~mask.unsqueeze(-1), 0)
        a, b = m_proj.chunk(2, dim=-1)  # [B, S, N, C_H] each

        # Force fp32 for the einsum (S-axis accumulation is bf16-unstable).
        with torch.autocast("cuda", enabled=False):
            if mask.ndim == 3:
                num_valid = mask.any(dim=-1).sum(dim=1).clamp(min=1)  # [B]
            else:
                num_valid = torch.ones(m.shape[0], device=m.device)  # [B, N] case: no S to average
            # Outer product over hidden dims — O(B·S·N²·C_H²) FLOPs, O(B·N²·C_H²) memory
            z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())  # [B, N, N, C_H, C_H]
            z = z.reshape(*z.shape[:3], -1)  # [B, N, N, C_H²]
            z = z / num_valid[:, None, None, None]  # [B, N, N, C_H²]
        return self.proj_o(z.to(m))  # [B, N, N, C_out]


class OuterProductMeanTritonMemorySaving(OuterProductMeanVanilla):
    """OPM with Triton-fused kernels at inference, vanilla einsum at training.

    Training: identical to OuterProductMeanVanilla (full einsum path).
    Inference: Triton-fused token-tiled kernel without the O(N²·C_H²) intermediate.
    Same state_dict — weights transfer freely between variants.
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        c_out: int,
        tile_size: int | None = None,
        matmul_precision: str = "ieee",
    ) -> None:
        """``matmul_precision`` selects the ``tl.dot`` input precision and
        only affects the **fp32** path (``ieee`` / ``tf32`` / ``tf32x3``).
        Under bf16/fp16 autocast, the MSA projections and the normalized outer
        products enter ``tl.dot`` in that dtype with float32 accumulation, so
        this knob is effectively a no-op for those dtypes.
        """
        super().__init__(c_in, c_hidden, c_out, tile_size=tile_size)
        if matmul_precision not in ("ieee", "tf32", "tf32x3"):
            raise ValueError(
                f"matmul_precision must be in {{'ieee','tf32','tf32x3'}}; got {matmul_precision!r}."
            )
        self._matmul_precision = matmul_precision

    def _project_masked(self, m: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return masked ``a, b`` projections, normalizing ``_PROJECTION_ROWS`` MSA rows at once.

        The float32 LayerNorm output of all rows would otherwise be the largest allocation
        of the inference forward, larger than the projections it produces.
        """
        if m.ndim != 4 or mask.ndim != 3:
            m_proj = self.proj_ab(self.norm(m))
            m_proj.masked_fill_(~mask.unsqueeze(-1), 0)
            return m_proj.chunk(2, dim=-1)

        m_proj = None
        msa_depth = m.shape[1]
        for start in range(0, msa_depth, _PROJECTION_ROWS):
            end = min(start + _PROJECTION_ROWS, msa_depth)
            rows = self.proj_ab(self.norm(m[:, start:end]))
            rows.masked_fill_(~mask[:, start:end].unsqueeze(-1), 0)
            if m_proj is None:
                m_proj = rows.new_empty(*m.shape[:-1], rows.shape[-1])
            m_proj[:, start:end] = rows
        return m_proj.chunk(2, dim=-1)

    def forward(self, m: Tensor, mask: Tensor) -> Tensor:
        """Compute outer product mean (Triton-fused at inference, vanilla at training)."""
        if not HAS_TRITON:
            raise RuntimeError("Triton outer-product mean requires Triton; install it on Linux.")
        if self.training or not m.is_cuda:
            return super().forward(m, mask)

        mask = mask.bool()
        a, b = self._project_masked(m, mask)  # [B, S, N, C_H] each

        if mask.ndim == 3:
            num_valid = mask.any(dim=-1).sum(dim=1).float().clamp(min=1)  # [B]
        else:
            num_valid = torch.ones(m.shape[0], device=m.device)  # [B, N] case: no S to average

        return _TritonOPMFn.apply(
            a,
            b,
            num_valid,
            self.proj_o.weight,
            self.proj_o.bias,
            self.tile_size,
            self._matmul_precision,
        )


REGISTRY: dict[OPMVariant, tuple[str, str]] = {
    OPMVariant.VANILLA: (__name__, "OuterProductMeanVanilla"),
    OPMVariant.TRITON_MEMORY_SAVING: (__name__, "OuterProductMeanTritonMemorySaving"),
}
