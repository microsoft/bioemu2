# Vendored from NVIDIA Fold-CP (MIT): https://github.com/NVIDIA-BioNeMo/boltz-cp @ f76f37a77c854b56b6250b426af8c2d63b501f7f
# Source: src/boltz/distributed/model/modules/utils.py (single function extracted).


def validate_window_batching_parameters(
    attn_window_queries: int | None,
    attn_window_keys: int | None,
    use_window_batching: bool,
) -> None:
    """Validates parameters for window batching in attention mechanisms.

    Args:
        attn_window_queries: Size of the query window. Must be a positive even integer if provided.
        attn_window_keys: Size of the key window. Must be a positive integer if provided.
        use_window_batching: Whether window batching is enabled.

    Raises:
        ValueError: If ``attn_window_queries`` and ``attn_window_keys`` are not both None or both not None.
        ValueError: If ``use_window_batching`` is True but ``attn_window_queries`` is None.
        ValueError: If ``attn_window_queries`` is not a positive even integer.
        ValueError: If ``attn_window_keys`` is not a positive integer.
        ValueError: If ``attn_window_keys`` is not divisible by ``attn_window_queries // 2``.
    """
    if (attn_window_queries is None) != (attn_window_keys is None):
        raise ValueError(
            "attn_window_queries and attn_window_keys must be either both None or both not None"
        )

    if (attn_window_queries is None) == use_window_batching:
        raise ValueError(
            f"attn_window_queries and attn_window_keys must be None if use_window_batching is False, otherwise they must be not None, but got attn_window_queries={attn_window_queries}, attn_window_keys={attn_window_keys} and use_window_batching={use_window_batching}"
        )

    if attn_window_queries is not None:
        if not isinstance(attn_window_queries, int) or attn_window_queries <= 0:
            raise ValueError("attn_window_queries must be a positive integer")

        if attn_window_queries % 2 != 0:
            raise ValueError("attn_window_queries must be even")

    if attn_window_keys is not None:
        if not isinstance(attn_window_keys, int) or attn_window_keys <= 0:
            raise ValueError("attn_window_keys must be a positive integer")

        assert attn_window_queries is not None
        if attn_window_keys % (attn_window_queries // 2) != 0:
            raise ValueError("attn_window_keys must be divisible by attn_window_queries // 2")
