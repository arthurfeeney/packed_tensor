import math
from typing import Sequence, Union

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from packed_tensor.packed_tensor import Indexing, PackedTensor


def natten(
    query: PackedTensor,
    key: PackedTensor,
    value: PackedTensor,
    kernel_size: Union[int, Sequence[int]],
    score_mod=None,
) -> PackedTensor:
    """Neighborhood attention over the spatial axes of a packed tensor.

    ``query``, ``key`` and ``value`` must share the same packing (identical
    shapes per item) and use the heads-last layout ``[*spatial, heads, dim]``.
    ``kernel_size`` is either a single odd window applied to every spatial
    dimension or one odd window per spatial dimension. ``score_mod`` is an
    optional FlexAttention score modifier (e.g. a relative position bias);
    masking is handled separately by the neighborhood ``mask_mod``.
    """
    assert query.shape() == key.shape() == value.shape(), (
        "query, key and value must have identical packing"
    )

    item_shapes = query.shape()
    num_heads = item_shapes[0][-2]
    head_dim = item_shapes[0][-1]
    assert all(shape[-2] == num_heads for shape in item_shapes), (
        "every item must have the same number of heads"
    )

    spatial_shapes = [tuple(shape[:-2]) for shape in item_shapes]
    ndim = len(spatial_shapes[0])
    kernel_size = _normalize_kernel_size(kernel_size, ndim)
    radii = tuple(size // 2 for size in kernel_size)
    _assert_kernel_fits(spatial_shapes, kernel_size)

    device = query.device
    token_item, token_coords, token_sizes = _build_metadata(spatial_shapes, device)
    mask_mod = _make_natten_mask_mod(token_item, token_coords, token_sizes, radii)

    total_tokens = token_item.numel()
    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device=device,
    )

    queries = _as_batched_heads(query, num_heads, head_dim)
    keys = _as_batched_heads(key, num_heads, head_dim)
    values = _as_batched_heads(value, num_heads, head_dim)
    attended = flex_attention(
        queries, keys, values, score_mod=score_mod, block_mask=block_mask
    )
    return _write_packed(attended, query, num_heads, head_dim)


def _normalize_kernel_size(kernel_size, ndim):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * ndim
    kernel_size = tuple(kernel_size)
    assert len(kernel_size) == ndim, (
        f"kernel_size has {len(kernel_size)} entries but inputs have {ndim} "
        "spatial dimensions"
    )
    # NATTEN windows are symmetric around the query, so an even window has no
    # well-defined center.
    assert all(size % 2 == 1 for size in kernel_size), "kernel sizes must be odd"
    return kernel_size


def _assert_kernel_fits(spatial_shapes, kernel_size):
    # A window larger than the item cannot slide to a valid in-bounds center.
    for dim, size in enumerate(kernel_size):
        smallest = min(spatial[dim] for spatial in spatial_shapes)
        assert size <= smallest, (
            f"kernel size {size} on dim {dim} exceeds the smallest item extent "
            f"{smallest}"
        )


def _build_metadata(spatial_shapes, device):
    """Per-token item id, spatial coordinate and item extent for every token.

    Coordinates and extents are returned as one 1-D tensor per spatial
    dimension so the ``mask_mod`` only ever does a 1-D gather, which is the
    pattern FlexAttention's masking vmap handles cleanly.
    """
    ndim = len(spatial_shapes[0])
    item_ids = []
    coords_per_dim = [[] for _ in range(ndim)]
    sizes_per_dim = [[] for _ in range(ndim)]

    for item_id, spatial_shape in enumerate(spatial_shapes):
        token_count = math.prod(spatial_shape)
        flat = torch.arange(token_count, device=device)
        unraveled = torch.unravel_index(flat, spatial_shape)
        item_ids.append(
            torch.full((token_count,), item_id, device=device, dtype=torch.int32)
        )
        for dim in range(ndim):
            coords_per_dim[dim].append(unraveled[dim].to(torch.int32))
            sizes_per_dim[dim].append(
                torch.full(
                    (token_count,),
                    spatial_shape[dim],
                    device=device,
                    dtype=torch.int32,
                )
            )

    token_item = torch.cat(item_ids)
    token_coords = [torch.cat(coords) for coords in coords_per_dim]
    token_sizes = [torch.cat(sizes) for sizes in sizes_per_dim]
    return token_item, token_coords, token_sizes


def _make_natten_mask_mod(token_item, token_coords, token_sizes, radii):
    def natten_mask_mod(batch, head, q_idx, kv_idx):
        accept = token_item[q_idx] == token_item[kv_idx]
        for dim, radius in enumerate(radii):
            coords = token_coords[dim]
            # Slide the window inward at the borders so a query near an edge
            # still sees a full-size neighborhood: the center is clamped to
            # [radius, extent - 1 - radius].
            upper_center = token_sizes[dim][q_idx] - 1 - radius
            center = torch.minimum(coords[q_idx].clamp_min(radius), upper_center)
            accept = accept & ((center - coords[kv_idx]).abs() <= radius)
        return accept

    return natten_mask_mod


def _as_batched_heads(
    packed: PackedTensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Zero-copy ``[1, heads, total_tokens, dim]`` view of the packed buffer.

    The buffer is token-major heads-last in memory, so reinterpreting it as
    ``[total_tokens, heads, dim]`` is a free view and the transpose to the
    ``[B, H, S, D]`` layout FlexAttention wants is only a stride swap.
    """
    total_tokens = packed._buffer.numel() // (num_heads * head_dim)
    tokens = packed._buffer.view(total_tokens, num_heads, head_dim)
    return tokens.transpose(0, 1).unsqueeze(0)


def _write_packed(
    attended: torch.Tensor, like: PackedTensor, num_heads: int, head_dim: int
) -> PackedTensor:
    total_tokens = attended.shape[2]
    out = PackedTensor(
        indexing=Indexing(like.shape(), like.stride(), like.end_offset()),
        device=like.device,
        dtype=like.dtype,
    )
    # Copy straight into the heads-last buffer view; copy_ handles the
    # transposed (non-contiguous) source without an extra materialization.
    out._buffer.view(total_tokens, num_heads, head_dim).copy_(
        attended.squeeze(0).transpose(0, 1)
    )
    return out
