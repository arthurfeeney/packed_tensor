from enum import Enum
from typing import Optional, Sequence, Union

import torch

from packed_tensor.conv.backend import im2col, loop
from packed_tensor.packed_tensor import PackedTensor


class ConvBackend(Enum):
    LOOP = "loop"
    IM2COL = "im2col"


_DISPATCH = {
    ConvBackend.LOOP: loop.conv,
    ConvBackend.IM2COL: im2col.conv,
}


def conv2d(
    images: PackedTensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    backend: Union[ConvBackend, str] = ConvBackend.LOOP,
) -> PackedTensor:
    """2-D convolution over a packed batch of variable-resolution images.

    ``images`` holds channels-last ``[height, width, in_channels]`` items and
    ``weight`` is ``[out_channels, in_channels, kH, kW]``. See the ``conv``
    backends for layout details; ``backend`` selects the implementation.
    """
    assert weight.dim() == 4, "conv2d expects a [Cout, Cin, kH, kW] weight"
    return _dispatch(images, weight, bias, stride, padding, backend)


def conv3d(
    images: PackedTensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    backend: Union[ConvBackend, str] = ConvBackend.LOOP,
) -> PackedTensor:
    """3-D convolution over a packed batch of variable-resolution volumes.

    ``images`` holds channels-last ``[depth, height, width, in_channels]`` items
    and ``weight`` is ``[out_channels, in_channels, kD, kH, kW]``.
    """
    assert weight.dim() == 5, "conv3d expects a [Cout, Cin, kD, kH, kW] weight"
    return _dispatch(images, weight, bias, stride, padding, backend)


def _dispatch(images, weight, bias, stride, padding, backend):
    backend = ConvBackend(backend)
    return _DISPATCH[backend](images, weight, bias, stride, padding)
