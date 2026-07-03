import pytest
import torch

from packed_tensor import packed_tensor

def test_packed_tensor_empty():
    empty = packed_tensor.empty(((2, 3), (4, 3)), device="cpu", dtype=torch.float32)
    assert empty.shape() == [(2, 3), (4, 3)]
    assert empty.shape(0) == (2, 3)
    assert empty.shape(1) == (4, 3)
    assert empty.stride() == [(3, 1), (3, 1)]
    assert empty.stride(0) == (3, 1)
    assert empty.stride(1) == (3, 1)
    assert empty._buffer.numel() == 18
    assert empty._buffer.dtype == torch.float32
    assert empty._buffer.device == torch.device("cpu")

@pytest.mark.parametrize(
    "shapes",
    [
        ((2, 3, 4),),
        ((2, 3, 4), (5, 6, 4)),
        ((2, 3, 4, 5), (6, 7, 8, 5)),
        ((1, 1, 1, 1, 3),),
    ],
)
def test_empty_strides_match_contiguous(shapes):
    empty = packed_tensor.empty(shapes, device="cpu", dtype=torch.float32)
    for idx, shape in enumerate(shapes):
        expected = torch.empty(shape).stride()
        assert empty.stride(idx) == expected


def test_packed_tensor_getitem():
    empty = packed_tensor.empty(((2, 3), (4, 3)), device="cpu", dtype=torch.float32)
    assert isinstance(empty[0], torch.Tensor)


def test_mixed_ndim_assert():
    shapes = ((2, 3), (4, 5, 3))
    with pytest.raises(AssertionError):
        packed_tensor.empty(shapes)

def test_mixed_last_dim_assert():
    shapes = ((2, 3), (4, 5))
    with pytest.raises(AssertionError):
        packed_tensor.empty(shapes)

def test_from_list():
    tensors = [
        torch.zeros((3, 4, 5), device="cpu", dtype=torch.float32),
        torch.ones((6, 7, 5), device="cpu", dtype=torch.float32)
    ]

    pt = packed_tensor.from_list(tensors)
    assert torch.all(pt[0] == 0)
    assert torch.all(pt[1] == 1)


def test_mm_matches_per_item_matmul():
    contraction_dim = 8
    out_dim = 16
    tensors = [
        torch.randn((3, 4, contraction_dim), device="cpu", dtype=torch.float32),
        torch.randn((6, 7, contraction_dim), device="cpu", dtype=torch.float32),
    ]
    other = torch.randn((contraction_dim, out_dim), device="cpu", dtype=torch.float32)

    pt = packed_tensor.from_list(tensors)
    result = pt.mm(other)

    for idx, tensor in enumerate(tensors):
        expected = tensor @ other
        assert result.shape(idx) == tuple(expected.shape)
        torch.testing.assert_close(result[idx], expected, rtol=1e-4, atol=1e-4)


def test_from_list_stride():
    ones = torch.ones((6, 7, 5), device="cpu", dtype=torch.float32)
    tensors = [
        torch.zeros((3, 4, 5), device="cpu", dtype=torch.float32),
        ones.permute(1, 0, 2)
    ]

    pt = packed_tensor.from_list(tensors)
    assert torch.all(pt[0] == 0)
    assert torch.all(pt[1] == 1)
    assert pt.shape(1) == (7, 6, 5)
    assert pt.stride(1) == (30, 5, 1) # converted to be contiguous

    l = pt.tolist()
    assert len(l) == 2
    assert torch.all(l[0] == tensors[0])
    assert l[0].shape == tensors[0].shape


def test_from_list_end_offsets_are_cumulative():
    # Distinct extents and fill values per item so a product-based (rather than
    # cumulative-sum) offset misroutes items or reads out of bounds.
    tensors = [
        torch.full((2, 3), 0.0, device="cpu", dtype=torch.float32),
        torch.full((4, 3), 1.0, device="cpu", dtype=torch.float32),
        torch.full((5, 3), 2.0, device="cpu", dtype=torch.float32),
    ]

    pt = packed_tensor.from_list(tensors)

    expected_end_offsets = (2 * 3, 2 * 3 + 4 * 3, 2 * 3 + 4 * 3 + 5 * 3)
    assert pt.end_offset() == expected_end_offsets
    assert pt._buffer.numel() == expected_end_offsets[-1]

    for idx, tensor in enumerate(tensors):
        assert pt.shape(idx) == tuple(tensor.shape)
        assert torch.all(pt[idx] == tensor)
