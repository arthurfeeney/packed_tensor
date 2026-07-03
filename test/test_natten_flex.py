import math

import pytest
import torch

from packed_tensor import packed_tensor
from packed_tensor.neighbor_attn.backend import flex_attn


def _reference_neighborhood_mask(spatial_shape, kernel_size):
    """Brute-force NATTEN boolean mask for a single item, [tokens, tokens].

    Independent of the implementation's mask_mod: it enumerates coordinates and
    applies the clamped-center window directly. True means the query (row) may
    attend to the key (column).
    """
    axes = [torch.arange(size) for size in spatial_shape]
    coords = torch.stack(
        torch.meshgrid(*axes, indexing="ij"), dim=-1
    ).reshape(-1, len(spatial_shape))

    token_count = coords.shape[0]
    mask = torch.ones(token_count, token_count, dtype=torch.bool)
    for dim, (size, kernel) in enumerate(zip(spatial_shape, kernel_size)):
        radius = kernel // 2
        query_coord = coords[:, dim].view(-1, 1)
        key_coord = coords[:, dim].view(1, -1)
        center = query_coord.clamp(radius, size - 1 - radius)
        mask &= (center - key_coord).abs() <= radius
    return mask


def _reference_natten_item(query_item, key_item, value_item, kernel_size):
    """Dense per-item neighborhood attention, returns [tokens, heads, dim]."""
    spatial_shape = tuple(query_item.shape[:-2])
    num_heads, head_dim = query_item.shape[-2], query_item.shape[-1]
    token_count = math.prod(spatial_shape)

    def to_heads(tensor):
        return tensor.reshape(token_count, num_heads, head_dim).transpose(0, 1)

    mask = _reference_neighborhood_mask(spatial_shape, kernel_size)
    attended = torch.nn.functional.scaled_dot_product_attention(
        to_heads(query_item),
        to_heads(key_item),
        to_heads(value_item),
        attn_mask=mask,
    )
    return attended.transpose(0, 1).reshape(token_count, num_heads, head_dim)


@pytest.mark.parametrize(
    "spatial_shapes, kernel_size",
    [
        ([(10,), (7,)], 3),
        ([(5, 6), (4, 7)], (3, 3)),
        ([(6, 5)], (5, 3)),
        ([(3, 4, 5)], 3),
    ],
)
def test_natten_matches_dense_reference(spatial_shapes, kernel_size):
    torch.manual_seed(0)
    num_heads = 2
    head_dim = 4

    def make_items():
        return [torch.randn(*shape, num_heads, head_dim) for shape in spatial_shapes]

    query_items, key_items, value_items = make_items(), make_items(), make_items()

    query = packed_tensor.from_list(query_items)
    key = packed_tensor.from_list(key_items)
    value = packed_tensor.from_list(value_items)

    result = flex_attn.natten(query, key, value, kernel_size)

    normalized_kernel = (
        (kernel_size,) * len(spatial_shapes[0])
        if isinstance(kernel_size, int)
        else kernel_size
    )
    for idx, spatial_shape in enumerate(spatial_shapes):
        expected = _reference_natten_item(
            query_items[idx], key_items[idx], value_items[idx], normalized_kernel
        )
        got = result[idx].reshape(math.prod(spatial_shape), num_heads, head_dim)
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_natten_does_not_mix_items():
    """A query in one item must be unaffected by the contents of other items."""
    torch.manual_seed(0)
    num_heads = 2
    head_dim = 4
    kernel_size = (3, 3)
    spatial_shape = (5, 6)

    first = torch.randn(*spatial_shape, num_heads, head_dim)
    second = torch.randn(*spatial_shape, num_heads, head_dim)

    def run(items):
        query = packed_tensor.from_list(items)
        key = packed_tensor.from_list(items)
        value = packed_tensor.from_list(items)
        return flex_attn.natten(query, key, value, kernel_size)

    packed_result = run([first, second])
    isolated_result = run([first])

    token_count = math.prod(spatial_shape)
    torch.testing.assert_close(
        packed_result[0].reshape(token_count, num_heads, head_dim),
        isolated_result[0].reshape(token_count, num_heads, head_dim),
        rtol=1e-5,
        atol=1e-5,
    )
