"""Triangle multiplication variant implementations.

Registry + factory pattern for selecting triangle multiplication kernels.
All variants share the same weight layout (state_dict transfers freely).

Active variants:
  FULL   — full materialization, O(N²D), fastest on GPU for small N
  CUEQ   — cuequivariance fused Triton kernel, ~2× faster
  CUEQ_TILED — cueq fused projection + tiled einsum, scales to N=8k+

Convenience wrappers expose outgoing and incoming triangle multiplication.
"""

import logging
from enum import StrEnum

import torch
import torch.nn as nn
from torch import Tensor

from bioemu2.biomoa.utils.warnings import suppress_cuequivariance_jit_deprecation

logger = logging.getLogger(__name__)

try:
    with suppress_cuequivariance_jit_deprecation():
        import cuequivariance_torch as cueq_torch
        from cuequivariance_ops.triton.utils import Precision
        from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
        from cuequivariance_ops_torch.gated_gemm_torch import (
            fused_sigmoid_gated_dual_gemm,
            fused_sigmoid_gated_dual_gemm_dual_x,
        )

    HAS_CUEQ = True
except ImportError:
    HAS_CUEQ = False

# Precision for fused_sigmoid_gated_dual_gemm matmuls in the tiled variant.
#
# cuequivariance_ops.triton.utils.Precision enum values:
#   DEFAULT (0) — auto: TF32 if allow_tf32, else TF32x3
#   TF32    (1) — single-pass TF32, 10-bit mantissa
#   TF32x3  (2) — 3-pass TF32, near-fp32 for single matmuls
#   IEEE    (3) — full IEEE-754 fp32, 23-bit mantissa
#
# The tiled variant chains 5 fused kernels (each containing matmuls) per
# forward tile, and autograd doubles that in the backward pass. TF32x3 is
# accurate for individual matmuls but error compounds through the chain:
#   DEFAULT/TF32x3 → gradient max-diff ~0.04-0.06 vs reference
#   IEEE           → gradient max-diff ~1e-5 vs reference

_DEFAULT_CUEQ_TILE_SIZE = 256


class TriMulVariant(StrEnum):
    FULL = "full"
    CUEQ = "cueq"
    CUEQ_TILED = "cueq_tiled"


class _TriMulBase(nn.Module):
    """Shared weights and helpers for all TriangleMultiplication variants."""

    def __init__(self, dim: int = 128, mode: str = "outgoing", **kwargs) -> None:
        super().__init__()
        assert mode in ("outgoing", "incoming")
        self.mode = mode
        self.dim = dim

        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)
        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

    def _project_tile(self, x_tile: Tensor, mask_tile: Tensor) -> Tensor:
        """Norm → gate → mask → return [B, tile, N, 2D]."""
        h = self.norm_in(x_tile)
        h = self.p_in(h) * self.g_in(h).sigmoid()
        h.masked_fill_(~mask_tile.bool().unsqueeze(-1), 0.0)
        return h


class TriangleMultiplicationFull(_TriMulBase):
    """Full materialization — O(N²D), fastest on GPU, torch.compile friendly."""

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x_normed = self.norm_in(x)
        x_in = x_normed
        proj = self.p_in(x_normed) * self.g_in(x_normed).sigmoid()
        proj.masked_fill_(~mask.bool().unsqueeze(-1), 0.0)
        a, b = proj.float().chunk(2, dim=-1)
        if self.mode == "outgoing":
            out = torch.einsum("bikd,bjkd->bijd", a, b)
        else:
            out = torch.einsum("bkid,bkjd->bijd", a, b)
        return self.p_out(self.norm_out(out)) * self.g_out(x_in).sigmoid()


class TriangleMultiplicationCueq(_TriMulBase):
    """cuEquivariance fused Triton kernel — ~2× faster, torch.compile friendly.

    ``matmul_precision`` (default ``"ieee"``):
      - ``"ieee"`` → strict IEEE FP32 (default).
      - ``None`` → cueq's ``Precision.NONE`` sentinel, letting cueq pick
        TF32 / TF32x3 from ``torch.backends.cuda.matmul.allow_tf32``.

    Cueq's high-level wrapper accepts only those two; anything else is
    remapped to ``None`` with a warning. Use :class:`TriangleMultiplication
    CueqTiled` for explicit TF32 / TF32x3 control.
    """

    def __init__(
        self,
        dim: int = 128,
        mode: str = "outgoing",
        matmul_precision: str | None = "ieee",
        tile_size: int | None = None,  # accepted for registry compatibility; not used.
    ) -> None:
        super().__init__(dim, mode)
        self._matmul_precision: str | None
        if matmul_precision in ("ieee", None):
            self._matmul_precision = matmul_precision
        else:
            logger.warning(
                "TriangleMultiplicationCueq: matmul_precision=%r not accepted by "
                "cueq's high-level wrapper; passing None instead so the "
                "kernel auto-picks TF32 / TF32x3 from "
                "torch.backends.cuda.matmul.allow_tf32. Use "
                "TriangleMultiplicationCueqTiled for explicit control.",
                matmul_precision,
            )
            self._matmul_precision = None

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return cueq_torch.triangle_multiplicative_update(
            x,
            direction=self.mode,
            mask=mask,
            norm_in_weight=self.norm_in.weight,
            norm_in_bias=self.norm_in.bias,
            p_in_weight=self.p_in.weight,
            g_in_weight=self.g_in.weight,
            norm_out_weight=self.norm_out.weight,
            norm_out_bias=self.norm_out.bias,
            p_out_weight=self.p_out.weight,
            g_out_weight=self.g_out.weight,
            precision=self._matmul_precision,
        )


def _to_cueqtiled_precision(matmul_precision: str) -> int:
    # cueq's NONE (allow_tf32 bridge) and DEFAULT (bare tl.dot) are intentionally
    # omitted so precision is explicit rather than inherited from global torch settings.
    mapping = {
        "ieee": Precision.IEEE.value,
        "tf32": Precision.TF32.value,
        "tf32x3": Precision.TF32x3.value,
    }
    if matmul_precision not in mapping:
        raise ValueError(
            f"TriangleMultiplicationCueqTiled matmul_precision must be one of "
            f"{sorted(mapping)}; got {matmul_precision!r}."
        )
    return mapping[matmul_precision]


class TriangleMultiplicationCueqTiled(_TriMulBase):
    """Tiled variant using cueq fused projection kernels.

    Uses cueq's fused_sigmoid_gated_dual_gemm for fast projection/gating
    (~2x faster than manual norm+linear+sigmoid) but tiles the einsum
    for O(tile × N × D) memory instead of O(N² × D).

    Best for N=3000-6000 where cueq OOMs but you want cueq-level
    projection speed. At N=3000: ~25ms projection + tiled einsum.
    """

    def __init__(
        self,
        dim: int = 128,
        mode: str = "outgoing",
        tile_size: int | None = None,
        matmul_precision: str = "ieee",
    ) -> None:
        """Tiled cueq variant.

        IEEE is the default because the tiled variant is memory-bound (tiling
        is the perf win), so full-precision matmuls have negligible throughput
        cost. Override with ``"tf32"`` / ``"tf32x3"`` if you have a benchmark
        that justifies the accuracy tradeoff.
        """
        super().__init__(dim, mode)
        self.tile_size = _DEFAULT_CUEQ_TILE_SIZE if tile_size is None else tile_size
        self._matmul_precision: int = _to_cueqtiled_precision(matmul_precision)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, N, _, D = x.shape
        tile = self.tile_size
        device = x.device
        dtype = x.dtype

        # Output in original dtype, no separate f32 buffer
        result = torch.empty(B, N, N, D, device=device, dtype=dtype)

        if self.mode == "outgoing":
            for i in range(0, N, tile):
                ie = min(i + tile, N)

                # cueq fused: norm → project → gate for i-tile
                x_i = x[:, i:ie]  # [B, T, N, D]
                normed_i = layer_norm_transpose(
                    x_i,
                    self.norm_in.weight,
                    self.norm_in.bias,
                    eps=1e-5,
                    layout="bijd->bijd",
                )  # [B, T, N, D]
                ab_i = fused_sigmoid_gated_dual_gemm(
                    normed_i,
                    self.g_in.weight,
                    self.p_in.weight,
                    mask[:, i:ie],
                    transpose_out=True,
                    precision=self._matmul_precision,
                )  # [2*D, B, T, N]
                a_i = torch.chunk(ab_i, 2, dim=0)[0]  # [D, B, T, N]
                del ab_i

                for j in range(0, N, tile):
                    je = min(j + tile, N)

                    # cueq fused: norm → project → gate for j-tile
                    x_j = x[:, j:je]  # [B, T, N, D]
                    normed_j = layer_norm_transpose(
                        x_j,
                        self.norm_in.weight,
                        self.norm_in.bias,
                        eps=1e-5,
                        layout="bijd->bijd",
                    )  # [B, T, N, D]
                    ab_j = fused_sigmoid_gated_dual_gemm(
                        normed_j,
                        self.g_in.weight,
                        self.p_in.weight,
                        mask[:, j:je],
                        transpose_out=True,
                        precision=self._matmul_precision,
                    )  # [2*D, B, T, N]
                    b_j = torch.chunk(ab_j, 2, dim=0)[1]  # [D, B, T, N]
                    del ab_j, normed_j

                    # Tiled einsum: dbik,dbjk->dbij
                    acc = torch.einsum("dbik,dbjk->dbij", a_i, b_j)  # [D, B, T, T]

                    # cueq fused output: norm → gate
                    acc_perm = acc.permute(1, 2, 3, 0).contiguous()  # [B, T, T, D]
                    out_normed = layer_norm_transpose(
                        acc_perm,
                        self.norm_out.weight,
                        self.norm_out.bias,
                        eps=1e-5,
                        layout="bijd->bijd",
                    )  # [B, T, T, D]
                    gate_input = normed_i[:, :, j:je].contiguous()  # [B, T, T, D]
                    tile_out = fused_sigmoid_gated_dual_gemm_dual_x(
                        gate_input,
                        out_normed,
                        self.g_out.weight,
                        self.p_out.weight,
                        precision=self._matmul_precision,
                    )  # [B, T, T, D]
                    result[:, i:ie, j:je] = tile_out.to(dtype)
                    del b_j, acc, acc_perm, out_normed, tile_out

                del a_i, normed_i

        else:
            # Incoming: out[i,j] = Σ_k a[k,i] * b[k,j]
            for i in range(0, N, tile):
                ie = min(i + tile, N)
                normed_i = layer_norm_transpose(
                    x[:, i:ie],  # [B, T, N, D]
                    self.norm_in.weight,
                    self.norm_in.bias,
                    eps=1e-5,
                    layout="bijd->bijd",
                )  # [B, T, N, D]

                for j in range(0, N, tile):
                    je = min(j + tile, N)
                    # Accumulate over k-tiles
                    acc = torch.zeros(D, B, ie - i, je - j, device=device, dtype=torch.float32)

                    for k in range(0, N, tile):
                        ke = min(k + tile, N)
                        x_k = x[:, k:ke]  # [B, T, N, D]
                        normed_k = layer_norm_transpose(
                            x_k,
                            self.norm_in.weight,
                            self.norm_in.bias,
                            eps=1e-5,
                            layout="bijd->bijd",
                        )  # [B, T, N, D]
                        ab_k = fused_sigmoid_gated_dual_gemm(
                            normed_k,
                            self.g_in.weight,
                            self.p_in.weight,
                            mask[:, k:ke],
                            transpose_out=True,
                            precision=self._matmul_precision,
                        )  # [2*D, B, T, N]
                        a_k, b_k = torch.chunk(ab_k, 2, dim=0)  # [D, B, T, N], [D, B, T, N]
                        # a_k[:, :, :, i:ie] and b_k[:, :, :, j:je]
                        a_ki = a_k[:, :, :, i:ie].contiguous()
                        b_kj = b_k[:, :, :, j:je].contiguous()
                        del ab_k, normed_k
                        acc = acc + torch.einsum("dbki,dbkj->dbij", a_ki, b_kj)

                    # Output gating
                    acc_perm = acc.permute(1, 2, 3, 0).contiguous()
                    out_normed = layer_norm_transpose(
                        acc_perm,
                        self.norm_out.weight,
                        self.norm_out.bias,
                        eps=1e-5,
                        layout="bijd->bijd",
                    )
                    gate_input = normed_i[:, :, j:je].contiguous()
                    tile_out = fused_sigmoid_gated_dual_gemm_dual_x(
                        gate_input,
                        out_normed,
                        self.g_out.weight,
                        self.p_out.weight,
                        precision=self._matmul_precision,
                    )
                    result[:, i:ie, j:je] = tile_out.to(dtype)

                del normed_i

        return result


# Registry mapping TriMulVariant → (module_path, class_name)
# All variants share the same state_dict layout.
REGISTRY: dict[TriMulVariant, tuple[str, str]] = {
    TriMulVariant.FULL: (__name__, "TriangleMultiplicationFull"),
    TriMulVariant.CUEQ: (__name__, "TriangleMultiplicationCueq"),
    TriMulVariant.CUEQ_TILED: (__name__, "TriangleMultiplicationCueqTiled"),
}

# Per-module tier tables: (max_N, variant_value, extra_kwargs)
A100_TIERS: list[tuple[int, str, dict]] = [
    (200, "full", {}),
    (3000, "cueq", {}),
    (5000, "cueq_tiled", {"tile_size": 512}),
    (999999, "cueq_tiled", {"tile_size": 256}),
]

B200_TIERS: list[tuple[int, str, dict]] = [
    (200, "full", {}),
    (3000, "cueq", {}),
    (5000, "cueq_tiled", {"tile_size": 512}),
    (999999, "cueq_tiled", {"tile_size": 512}),
]


class TriangleMultiplicationOutgoing(TriangleMultiplicationFull):
    """Outgoing triangle multiplication.

    Same constructor signature: ``TriangleMultiplicationOutgoing(dim=128)``.
    Same forward signature: ``forward(x, mask) -> Tensor``.
    Same weight layout — state_dict transfers freely.

    To switch variant, use ``make_tri_mul(variant, dim, "outgoing")`` instead.
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__(dim=dim, mode="outgoing")


class TriangleMultiplicationIncoming(TriangleMultiplicationFull):
    """Incoming triangle multiplication.

    Same constructor signature: ``TriangleMultiplicationIncoming(dim=128)``.
    Same forward signature: ``forward(x, mask) -> Tensor``.
    Same weight layout — state_dict transfers freely.

    To switch variant, use ``make_tri_mul(variant, dim, "incoming")`` instead.
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__(dim=dim, mode="incoming")
