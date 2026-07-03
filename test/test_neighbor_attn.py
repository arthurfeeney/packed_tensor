import math

import pytest
import torch

from packed_tensor import packed_tensor
from packed_tensor.neighbor_attn import Backend, neighbor_attn
from packed_tensor.neighbor_attn.backend import flex_attn


def _packed_qkv(spatial_shapes, num_heads, head_dim, seed=0):
    torch.manual_seed(seed)

    def make_items():
        return [torch.randn(*shape, num_heads, head_dim) for shape in spatial_shapes]

    return (
        packed_tensor.from_list(make_items()),
        packed_tensor.from_list(make_items()),
        packed_tensor.from_list(make_items()),
    )


def test_dispatch_flex_matches_backend():
    spatial_shapes = [(5, 6), (4, 7)]
    query, key, value = _packed_qkv(spatial_shapes, num_heads=2, head_dim=8)

    dispatched = neighbor_attn(query, key, value, kernel_size=3, backend=Backend.FLEX)
    direct = flex_attn.natten(query, key, value, kernel_size=3)

    for idx, spatial_shape in enumerate(spatial_shapes):
        token_count = math.prod(spatial_shape)
        torch.testing.assert_close(
            dispatched[idx].reshape(token_count, 2, 8),
            direct[idx].reshape(token_count, 2, 8),
        )


def test_dispatch_accepts_string_backend():
    query, key, value = _packed_qkv([(5, 6)], num_heads=2, head_dim=8)
    from_enum = neighbor_attn(query, key, value, kernel_size=3, backend=Backend.FLEX)
    from_string = neighbor_attn(query, key, value, kernel_size=3, backend="flex")
    torch.testing.assert_close(from_string[0], from_enum[0])


def test_dispatch_backends_agree():
    pytest.importorskip("natten")
    spatial_shapes = [(6, 6), (5, 4)]
    query, key, value = _packed_qkv(spatial_shapes, num_heads=2, head_dim=8)

    flex_result = neighbor_attn(query, key, value, kernel_size=3, backend=Backend.FLEX)
    natten_result = neighbor_attn(
        query, key, value, kernel_size=3, backend=Backend.NATTEN
    )

    for idx, spatial_shape in enumerate(spatial_shapes):
        token_count = math.prod(spatial_shape)
        torch.testing.assert_close(
            flex_result[idx].reshape(token_count, 2, 8),
            natten_result[idx].reshape(token_count, 2, 8),
            rtol=1e-3,
            atol=1e-3,
        )


def test_dispatch_unknown_backend_raises():
    query, key, value = _packed_qkv([(4, 4)], num_heads=2, head_dim=8)
    with pytest.raises(ValueError):
        neighbor_attn(query, key, value, kernel_size=3, backend="triton")
