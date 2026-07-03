from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F

from packed_tensor import packed_tensor
from packed_tensor.conv.common import to_ntuple
from packed_tensor.packed_tensor import PackedTensor

_CONV_BY_NDIM = {2: F.conv2d, 3: F.conv3d}


def conv(
    images: PackedTensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
) -> PackedTensor:
    """Convolution over each item of a packed tensor via ``F.conv{2,3}d``.

    Items use the channels-last layout ``[*spatial, in_channels]`` and ``weight``
    is the channels-first ``[out_channels, in_channels, *kernel]`` that the torch
    convolutions expect. Each item is convolved independently, so the output
    packing has the same rank but per-item spatial extents shrink with ``stride``
    / ``padding``.
    """
    spatial_ndim = weight.dim() - 2
    convolution = _CONV_BY_NDIM[spatial_ndim]
    stride = to_ntuple(stride, spatial_ndim)
    padding = to_ntuple(padding, spatial_ndim)

    outputs = []
    for idx in range(len(images.shape())):
        channels_first = _to_channels_first(images[idx], spatial_ndim)
        convolved = convolution(channels_first, weight, bias, stride, padding)
        outputs.append(_to_channels_last(convolved.squeeze(0), spatial_ndim))
    return packed_tensor.from_list(outputs)


def _to_channels_first(item: torch.Tensor, spatial_ndim: int) -> torch.Tensor:
    return item.permute(spatial_ndim, *range(spatial_ndim)).unsqueeze(0)


def _to_channels_last(item: torch.Tensor, spatial_ndim: int) -> torch.Tensor:
    return item.permute(*range(1, spatial_ndim + 1), 0).contiguous()
