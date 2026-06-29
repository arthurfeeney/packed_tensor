from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import torch

def _assert_shapes(shapes) -> None:
    assert all([len(shapes[0]) == len(shape) for shape in shapes])
    assert all([shapes[0][-1] == shape[-1] for shape in shapes])

@dataclass(frozen=True)
class Indexing:
    shapes: List[Tuple[int, ...]]
    strides: List[Tuple[int, ...]]
    end_offsets: Tuple[int, ...]

    def __post_init__(self) -> None:
        _assert_shapes(self.shapes)

    def numel(self) -> int:
        return sum(math.prod(shape) for shape in self.shapes)


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
        self._indexing = indexing
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
            self._indexing.shapes[idx], self._indexing.strides[idx], start_idx
        )

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

    def fill_(self, value):
        self._buffer.fill_(value)


def _row_major_strides(shapes: torch.Tensor):
    strides = torch.ones_like(shapes)
    strides[:, 1:] = shapes[:, 1:]
    strides = torch.flip(torch.cumprod(strides, dim=1), dims=(1,))
    return strides


def _list_of_tuple(tensor):
    return [tuple(row) for row in tensor.tolist()]


def empty(shapes, device: torch.device = None, dtype: torch.dtype = None):
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
