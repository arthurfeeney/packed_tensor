import math

import pytest
import torch

pytest.importorskip("natten")

from packed_tensor import packed_tensor  # noqa: E402
from packed_tensor.neighbor_attn.backend import flex_attn, natten  # noqa: E402


@pytest.mark.parametrize(
    "spatial_shapes, kernel_size",
    [
        ([(10,), (7,)], 3),
        ([(5, 6), (4, 7)], (3, 3)),
        ([(4, 4, 4)], 3),
    ],
)
def test_natten_loop_matches_flex_reference(spatial_shapes, kernel_size):
    torch.manual_seed(0)
    num_heads = 2
    head_dim = 8  # NATTEN requires 8 <= head_dim <= 512

    def make_items():
        return [
            torch.randn(*shape, num_heads, head_dim) for shape in spatial_shapes
        ]

    query = packed_tensor.from_list(make_items())
    key = packed_tensor.from_list(make_items())
    value = packed_tensor.from_list(make_items())

    looped = natten.natten_loop(query, key, value, kernel_size)
    reference = flex_attn.natten(query, key, value, kernel_size)

    for idx, spatial_shape in enumerate(spatial_shapes):
        token_count = math.prod(spatial_shape)
        torch.testing.assert_close(
            looped[idx].reshape(token_count, num_heads, head_dim),
            reference[idx].reshape(token_count, num_heads, head_dim),
            rtol=1e-3,
            atol=1e-3,
        )


def test_natten_loop_does_not_mix_items():
    torch.manual_seed(0)
    num_heads, head_dim = 2, 8
    kernel_size = (3, 3)
    spatial_shape = (5, 6)

    first = torch.randn(*spatial_shape, num_heads, head_dim)
    second = torch.randn(*spatial_shape, num_heads, head_dim)

    def run(items):
        query = packed_tensor.from_list(items)
        key = packed_tensor.from_list(items)
        value = packed_tensor.from_list(items)
        return natten.natten_loop(query, key, value, kernel_size)

    packed_result = run([first, second])
    isolated_result = run([first])

    token_count = math.prod(spatial_shape)
    torch.testing.assert_close(
        packed_result[0].reshape(token_count, num_heads, head_dim),
        isolated_result[0].reshape(token_count, num_heads, head_dim),
        rtol=1e-5,
        atol=1e-5,
    )
