import itertools
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F

from packed_tensor import packed_tensor
from packed_tensor.conv.common import conv_output_extent, to_ntuple
from packed_tensor.packed_tensor import PackedTensor


def conv(
    images: PackedTensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
) -> PackedTensor:
    """Convolution expressed as im2col followed by the packed matmul.

    Each item's neighborhoods are unfolded into a ``[*out_spatial, in_channels *
    prod(kernel)]`` patch tensor whose last dim matches the flattened ``weight``,
    so the whole convolution collapses into a single ``PackedTensor.mm``. This is
    the packed-native path that a fused Triton/CUDA kernel would later replace.
    """
    spatial_ndim = weight.dim() - 2
    stride = to_ntuple(stride, spatial_ndim)
    padding = to_ntuple(padding, spatial_ndim)
    kernel_size = tuple(weight.shape[2:])
    out_channels = weight.shape[0]

    # Flatten (in_channels, *kernel) so it lines up with the patch layout below.
    weight_matrix = weight.reshape(out_channels, -1).t().contiguous()

    patch_items = []
    for idx in range(len(images.shape())):
        channels_first = images[idx].permute(spatial_ndim, *range(spatial_ndim))
        patch_items.append(
            _extract_patches(channels_first, kernel_size, stride, padding)
        )

    output = packed_tensor.from_list(patch_items).mm(weight_matrix)
    if bias is not None:
        output.apply_(lambda buffer: (buffer.view(-1, out_channels) + bias).view(-1))
    return output


def _extract_patches(
    channels_first: torch.Tensor,
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
) -> torch.Tensor:
    spatial_ndim = channels_first.dim() - 1
    in_spatial = channels_first.shape[1:]
    out_spatial = tuple(
        conv_output_extent(in_spatial[dim], kernel_size[dim], stride[dim], padding[dim])
        for dim in range(spatial_ndim)
    )

    pad_amounts = []
    for dim in reversed(range(spatial_ndim)):
        pad_amounts.extend((padding[dim], padding[dim]))
    padded = F.pad(channels_first, pad_amounts)

    columns = []
    for kernel_offset in itertools.product(*(range(size) for size in kernel_size)):
        window = [slice(None)]
        for dim, offset in enumerate(kernel_offset):
            stop = offset + stride[dim] * (out_spatial[dim] - 1) + 1
            window.append(slice(offset, stop, stride[dim]))
        columns.append(padded[tuple(window)])

    stacked = torch.stack(columns, dim=1)
    patch_channels = stacked.shape[0] * stacked.shape[1]
    channels_last = stacked.permute(*range(2, 2 + spatial_ndim), 0, 1)
    return channels_last.reshape(*out_spatial, patch_channels)
