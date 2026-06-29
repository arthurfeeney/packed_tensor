import pytest
import torch

from varlen_natten import packed_tensor

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
