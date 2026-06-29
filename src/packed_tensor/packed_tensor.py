from dataclasses import dataclass
from itertools import pairwise
import math
from typing import List, Optional, Tuple

import torch

def _assert_shapes(shapes) -> None:
    # all shapes must have the same number of dimensions,
    # and the last dimension must have the same size across all tensors
    assert all([len(shapes[0]) == len(shape) for shape in shapes])
    assert all([shapes[0][-1] == shape[-1] for shape in shapes])

def _assert_strides(strides) -> None:
    # strides should all have the same number of dimensions
    assert all([len(strides[0]) == len(stride) for stride in strides])
    # TODO: enforcing that strides are right-contiguous, may want to loosen
    def non_increasing(stride):
        return all(first >= second for first, second in pairwise(stride))
    assert all(non_increasing(stride) for stride in strides)

@dataclass(frozen=True)
class Indexing:
    shapes: List[Tuple[int, ...]]
    strides: List[Tuple[int, ...]]
    end_offsets: Tuple[int, ...]

    def __post_init__(self) -> None:
        _assert_shapes(self.shapes)
        _assert_strides(self.strides)

    def numel(self) -> int:
        return sum(math.prod(shape) for shape in self.shapes)

    def last_dim(self) -> int:
        return self.shapes[0][-1]


@dataclass(frozen=True)
class DeviceIndexing(Indexing):
    # Users interact with Indexing, but some operations
    # need end offsets on device, so we construct this during
    # packed tensor initialization
    end_offsets_tensor: torch.Tensor

    # delimits rows if tensors are viewed (-1, last_dim()),
    row_offsets_tensor: torch.Tensor

def _indexing_to_device_indexing(indexing: Indexing, device: torch.device):
    end_offsets_tensor=torch.tensor(indexing.end_offsets, device=device, dtype=torch.int64)
    row_offsets_tensor=(end_offsets_tensor / indexing.last_dim()).to(torch.int32)

    return DeviceIndexing(
        indexing.shapes,
        indexing.strides,
        indexing.end_offsets,
        end_offsets_tensor=end_offsets_tensor,
        row_offsets_tensor=row_offsets_tensor
    )

class PackedTensor:
    r"""
    Packed storage for multiple tensors with different shapes, e.g.

        [H1, W1, ..., C],
        [H2, W2, ..., C],
        [H3, W3, ..., C]

    All tensors must share the same rank (number of dims). They are flattened
    and concatenated into a single 1-D buffer: `packed_tensor`.
    """

    def __init__(
        self,
        indexing: Indexing,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ):
        self._indexing: DeviceIndexing = _indexing_to_device_indexing(indexing, device)
        self._buffer = torch.empty(
            self._indexing.numel(),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )

    def _get_indices(self, idx):
        if idx == 0:
            return 0, self.end_offset(0)
        return self.end_offset(idx - 1), self.end_offset(idx)

    def __getitem__(self, idx):
        start_idx, _ = self._get_indices(idx)
        return self._buffer.as_strided(
            self.shape(idx), self.stride(idx), start_idx
        )

    @property
    def device(self):
        return self._buffer.device

    @property
    def dtype(self):
        return self._buffer.dtype

    def data_ptr(self):
        return self._buffer.data_ptr()

    def stride(self, idx=None):
        if idx is None:
            return self._indexing.strides
        return self._indexing.strides[idx]

    def shape(self, idx=None):
        if idx is None:
            return self._indexing.shapes
        return self._indexing.shapes[idx]

    def end_offset(self, idx=None):
        if idx is None:
            return self._indexing.end_offsets
        return self._indexing.end_offsets[idx]

    def fill_(self, value) -> None:
        self._buffer.fill_(value)

    def copy_(self, tensor: torch.Tensor) -> None:
        self._buffer.copy_(tensor)

    def mm(self, other: torch.Tensor):
        assert other.dim() == 2
        pt = PackedTensor(
            indexing=_mm_indexing(self._indexing, other.shape[1]),
            device=self.device,
            dtype=self.dtype
        )
        pt.copy_((self._buffer.view(-1, self._indexing.last_dim()) @ other).view(-1))
        return pt


def _mm_indexing(input_indexing, out_dim: int):
    out_shapes = torch.tensor([shape[:-1] + (out_dim,) for shape in input_indexing.shapes], dtype=torch.int64, device="cpu")
    out_strides = _row_major_strides(out_shapes)
    end_offsets = torch.cumsum(out_shapes.prod(dim=1), dim=0)
    return Indexing(
        _list_of_tuple(out_shapes),
        _list_of_tuple(out_strides),
        tuple(end_offsets.tolist()),
    )

def _row_major_strides(shapes):
    # stride for dim i is the product of all trailing dim sizes, so shift the
    # sizes left by one and take a reverse cumulative product.
    strides = torch.ones_like(shapes)
    strides[:, :-1] = shapes[:, 1:]
    return torch.flip(torch.cumprod(torch.flip(strides, dims=(1,)), dim=1), dims=(1,))


def _list_of_tuple(tensor):
    return [tuple(row) for row in tensor.tolist()]


def empty(shapes, device: torch.device = None, dtype: torch.dtype = None) -> PackedTensor:
    _assert_shapes(shapes)
    shapes_tensor = torch.tensor(shapes, dtype=torch.int64, device="cpu")
    strides = _row_major_strides(shapes_tensor)
    end_offsets = torch.cumsum(shapes_tensor.prod(dim=1), dim=0)
    indexing = Indexing(
        _list_of_tuple(shapes_tensor),
        _list_of_tuple(strides),
        tuple(end_offsets.tolist()),
    )
    return PackedTensor(indexing, device, dtype)

def from_list(tensors: List[torch.Tensor]) -> PackedTensor:
    contiguous_tensors = [
        tensor.contiguous() for tensor in tensors
    ]

    shapes = [tuple(tensor.shape) for tensor in contiguous_tensors]
    strides = [tuple(tensor.stride()) for tensor in contiguous_tensors]
    end_offsets = [math.prod(shape) for shape in shapes]
    for idx in range(1, len(end_offsets)):
        end_offsets[idx] *= end_offsets[idx - 1]

    device = contiguous_tensors[0].device
    dtype = contiguous_tensors[0].dtype
    assert all(tensor.device == device for tensor in contiguous_tensors)
    assert all(tensor.dtype == dtype for tensor in contiguous_tensors)

    pt = PackedTensor(
        indexing=Indexing(
            shapes=shapes,
            strides=strides,
            end_offsets=end_offsets
        ),
        device=device,
        dtype=dtype
    )

    # Note that copy_ ignores stride. tensor is copied based
    # on memory contiguity.
    for idx, tensor in enumerate(contiguous_tensors):
        pt[idx].copy_(tensor)

    return pt
