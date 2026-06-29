"""Tests for the generated PackedTensor initializers.

The contract of these functions is "allocate with `empty`, then fill the packed
buffer with an in-place op", so the assertions look at the flat buffer directly
rather than going through per-tensor indexing.
"""

import pytest
import torch

from packed_tensor import packed_tensor_functions
from packed_tensor.packed_tensor import empty

SHAPES = [(2, 3, 4), (5, 1, 4), (3, 3, 4)]
GENERATED_INITIALIZERS = ["zeros", "ones", "normal_", "randn", "rand"]
ELEMENTWISE = {
    "sigmoid": torch.sigmoid,
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
}


def test_all_exports_expected_names():
    assert set(packed_tensor_functions.__all__) == {
        "empty",
        *GENERATED_INITIALIZERS,
        *ELEMENTWISE,
    }


@pytest.mark.parametrize("name", GENERATED_INITIALIZERS)
def test_allocation_matches_empty(name):
    reference = empty(SHAPES, device="cpu", dtype=torch.float32)
    packed = getattr(packed_tensor_functions, name)(
        SHAPES, device="cpu", dtype=torch.float32
    )
    assert packed._buffer.shape == reference._buffer.shape
    assert packed._buffer.dtype == reference._buffer.dtype
    assert packed._buffer.device == reference._buffer.device


@pytest.mark.parametrize("name, value", [("zeros", 0.0), ("ones", 1.0)])
def test_constant_fill(name, value):
    packed = getattr(packed_tensor_functions, name)(
        SHAPES, device="cpu", dtype=torch.float32
    )
    assert torch.all(packed._buffer == value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_dtype_is_respected(dtype):
    packed = packed_tensor_functions.zeros(SHAPES, device="cpu", dtype=dtype)
    assert packed._buffer.dtype == dtype


def test_normal_forwards_mean_and_std():
    torch.manual_seed(0)
    # large element count so the sample statistics are tight enough to assert on
    packed = packed_tensor_functions.normal_(
        [(5000, 5000)], dtype=torch.float32, mean=5.0, std=2.0
    )
    assert packed._buffer.mean().item() == pytest.approx(5.0, abs=0.1)
    assert packed._buffer.std().item() == pytest.approx(2.0, abs=0.1)


def test_randn_is_standard_normal():
    torch.manual_seed(0)
    packed = packed_tensor_functions.randn([(5000, 5000)], dtype=torch.float32)
    assert packed._buffer.mean().item() == pytest.approx(0.0, abs=0.1)
    assert packed._buffer.std().item() == pytest.approx(1.0, abs=0.1)


def test_rand_within_unit_interval():
    torch.manual_seed(0)
    packed = packed_tensor_functions.rand([(5000, 5000)], dtype=torch.float32)
    assert packed._buffer.min().item() >= 0.0
    assert packed._buffer.max().item() < 1.0


@pytest.mark.parametrize("name, op", ELEMENTWISE.items())
def test_elementwise_matches_torch(name, op):
    torch.manual_seed(0)
    source = packed_tensor_functions.randn(SHAPES, dtype=torch.float32)
    result = getattr(packed_tensor_functions, name)(source)
    assert torch.equal(result._buffer, op(source._buffer))


@pytest.mark.parametrize("name", ELEMENTWISE)
def test_elementwise_is_out_of_place(name):
    torch.manual_seed(0)
    source = packed_tensor_functions.randn(SHAPES, dtype=torch.float32)
    before = source._buffer.clone()
    packed_tensor_functions.relu(source)
    assert torch.equal(source._buffer, before)
