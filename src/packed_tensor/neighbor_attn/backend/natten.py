from typing import Optional, Sequence, Union

from natten.functional import na1d, na2d, na3d

from packed_tensor.packed_tensor import Indexing, PackedTensor

_NEIGHBORHOOD_ATTENTION_BY_NDIM = {1: na1d, 2: na2d, 3: na3d}


def natten_loop(
    query: PackedTensor,
    key: PackedTensor,
    value: PackedTensor,
    kernel_size: Union[int, Sequence[int]],
    dilation: Union[int, Sequence[int]] = 1,
    is_causal: bool = False,
    backend: Optional[str] = None,
) -> PackedTensor:
    """ComputeNeighborhood attention over each item of a packed tensor via NATTEN.

    ``query``, ``key`` and ``value`` must share the same packing and use the
    heads-last layout ``[*spatial, heads, dim]`` per item. NATTEN requires
    ``8 <= head_dim <= 512``. ``kernel_size`` and ``dilation`` are forwarded to
    NATTEN as either a scalar (applied to every spatial dim) or one value per
    spatial dim.
    """
    assert query.shape() == key.shape() == value.shape(), (
        "query, key and value must have identical packing"
    )

    item_shapes = query.shape()
    ndim = len(item_shapes[0]) - 2  # drop the trailing heads and dim axes
    if ndim not in _NEIGHBORHOOD_ATTENTION_BY_NDIM:
        raise ValueError(f"NATTEN supports 1-3 spatial dims, got {ndim}")
    neighborhood_attention = _NEIGHBORHOOD_ATTENTION_BY_NDIM[ndim]

    out = PackedTensor(
        indexing=Indexing(query.shape(), query.stride(), query.end_offset()),
        device=query.device,
        dtype=query.dtype,
    )

    for idx in range(len(item_shapes)):
        attended = neighborhood_attention(
            query[idx].unsqueeze(0),
            key[idx].unsqueeze(0),
            value[idx].unsqueeze(0),
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            backend=backend,
        )
        out[idx].copy_(attended.squeeze(0))

    return out
