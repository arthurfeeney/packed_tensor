import pytest
import torch
import torch.nn.functional as F

from packed_tensor import packed_tensor
from packed_tensor.conv import ConvBackend, conv2d, conv3d

_DTYPES = [
    (torch.float32, 1e-4, 1e-5),
    (torch.float64, 1e-10, 1e-12),
]

# bfloat16 has an 8-bit mantissa, so compare the low-precision path against an
# fp32 reference with tolerances sized to the measured accumulation error.
_BF16_RTOL = 5e-2
_BF16_ATOL = 2e-1


def _packed_images(spatial_shapes, in_channels, dtype, seed=0):
    torch.manual_seed(seed)
    items = [
        torch.randn(*shape, in_channels, dtype=dtype) for shape in spatial_shapes
    ]
    return packed_tensor.from_list(items)


def _reference(images, weight, bias, stride, padding, convolution):
    spatial_ndim = weight.dim() - 2
    outputs = []
    for idx in range(len(images.shape())):
        channels_first = images[idx].permute(spatial_ndim, *range(spatial_ndim))
        convolved = convolution(
            channels_first.unsqueeze(0), weight, bias, stride, padding
        )
        outputs.append(
            convolved.squeeze(0).permute(*range(1, spatial_ndim + 1), 0).contiguous()
        )
    return outputs


@pytest.mark.parametrize("backend", [ConvBackend.LOOP, ConvBackend.IM2COL])
@pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
@pytest.mark.parametrize(
    "stride,padding",
    [(1, 0), (2, 1), (1, 2), ((2, 1), (1, 0))],
)
def test_conv2d_matches_torch(backend, dtype, rtol, atol, stride, padding):
    spatial_shapes = [(7, 8), (9, 6), (5, 11)]
    in_channels, out_channels, kernel = 3, 4, 3
    images = _packed_images(spatial_shapes, in_channels, dtype)
    weight = torch.randn(out_channels, in_channels, kernel, kernel, dtype=dtype)
    bias = torch.randn(out_channels, dtype=dtype)

    result = conv2d(images, weight, bias, stride, padding, backend=backend)
    expected = _reference(images, weight, bias, stride, padding, F.conv2d)

    for idx, expected_item in enumerate(expected):
        assert result.shape(idx) == tuple(expected_item.shape)
        torch.testing.assert_close(result[idx], expected_item, rtol=rtol, atol=atol)


@pytest.mark.parametrize("backend", [ConvBackend.LOOP, ConvBackend.IM2COL])
@pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1)])
def test_conv3d_matches_torch(backend, dtype, rtol, atol, stride, padding):
    spatial_shapes = [(4, 5, 6), (5, 4, 4), (6, 5, 3)]
    in_channels, out_channels, kernel = 2, 3, 3
    images = _packed_images(spatial_shapes, in_channels, dtype)
    weight = torch.randn(out_channels, in_channels, kernel, kernel, kernel, dtype=dtype)
    bias = torch.randn(out_channels, dtype=dtype)

    result = conv3d(images, weight, bias, stride, padding, backend=backend)
    expected = _reference(images, weight, bias, stride, padding, F.conv3d)

    for idx, expected_item in enumerate(expected):
        assert result.shape(idx) == tuple(expected_item.shape)
        torch.testing.assert_close(result[idx], expected_item, rtol=rtol, atol=atol)


@pytest.mark.parametrize("backend", [ConvBackend.LOOP, ConvBackend.IM2COL])
@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1), (1, 2)])
def test_conv2d_bfloat16_matches_fp32(backend, stride, padding):
    torch.manual_seed(0)
    spatial_shapes = [(7, 8), (9, 6), (5, 11)]
    in_channels, out_channels, kernel = 3, 4, 3
    items = [torch.randn(*shape, in_channels) for shape in spatial_shapes]
    weight = torch.randn(out_channels, in_channels, kernel, kernel)
    bias = torch.randn(out_channels)

    expected = _reference(
        packed_tensor.from_list(items), weight, bias, stride, padding, F.conv2d
    )
    images = packed_tensor.from_list([item.bfloat16() for item in items])
    result = conv2d(
        images, weight.bfloat16(), bias.bfloat16(), stride, padding, backend=backend
    )

    for idx, expected_item in enumerate(expected):
        assert result.shape(idx) == tuple(expected_item.shape)
        torch.testing.assert_close(
            result[idx].float(), expected_item, rtol=_BF16_RTOL, atol=_BF16_ATOL
        )


@pytest.mark.parametrize("backend", [ConvBackend.LOOP, ConvBackend.IM2COL])
@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1)])
def test_conv3d_bfloat16_matches_fp32(backend, stride, padding):
    torch.manual_seed(0)
    spatial_shapes = [(4, 5, 6), (5, 4, 4), (6, 5, 3)]
    in_channels, out_channels, kernel = 2, 3, 3
    items = [torch.randn(*shape, in_channels) for shape in spatial_shapes]
    weight = torch.randn(out_channels, in_channels, kernel, kernel, kernel)
    bias = torch.randn(out_channels)

    expected = _reference(
        packed_tensor.from_list(items), weight, bias, stride, padding, F.conv3d
    )
    images = packed_tensor.from_list([item.bfloat16() for item in items])
    result = conv3d(
        images, weight.bfloat16(), bias.bfloat16(), stride, padding, backend=backend
    )

    for idx, expected_item in enumerate(expected):
        assert result.shape(idx) == tuple(expected_item.shape)
        torch.testing.assert_close(
            result[idx].float(), expected_item, rtol=_BF16_RTOL, atol=_BF16_ATOL
        )


def test_conv2d_without_bias():
    images = _packed_images([(6, 6), (5, 7)], in_channels=3, dtype=torch.float64)
    weight = torch.randn(4, 3, 3, 3, dtype=torch.float64)

    loop_result = conv2d(images, weight, backend=ConvBackend.LOOP)
    im2col_result = conv2d(images, weight, backend=ConvBackend.IM2COL)

    for idx in range(len(images.shape())):
        torch.testing.assert_close(im2col_result[idx], loop_result[idx])


def test_dispatch_accepts_string_backend():
    images = _packed_images([(6, 6)], in_channels=3, dtype=torch.float64)
    weight = torch.randn(4, 3, 3, 3, dtype=torch.float64)
    from_enum = conv2d(images, weight, backend=ConvBackend.IM2COL)
    from_string = conv2d(images, weight, backend="im2col")
    torch.testing.assert_close(from_string[0], from_enum[0])


def test_dispatch_unknown_backend_raises():
    images = _packed_images([(6, 6)], in_channels=3, dtype=torch.float64)
    weight = torch.randn(4, 3, 3, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        conv2d(images, weight, backend="triton")
