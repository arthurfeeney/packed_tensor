import pytest
import torch

from packed_tensor import packed_tensor
from packed_tensor import packed_tensor_functions
from packed_tensor.conv import ConvBackend, conv2d


def _inductor_available() -> bool:
    try:
        torch._dynamo.reset()
        torch.compile(lambda tensor: tensor + 1, backend="inductor")(torch.zeros(1))
        return True
    except Exception:
        return False


TRACING_BACKENDS = ["eager", "aot_eager"]
_INDUCTOR_AVAILABLE = _inductor_available()

_CHANNELS = 5
torch.manual_seed(1234)
_MM_WEIGHT = torch.randn(_CHANNELS, 8)
_CONV_WEIGHT = torch.randn(4, _CHANNELS, 3, 3)

# Each op maps a PackedTensor to a PackedTensor using fixed weights so the
# compiled and eager results are directly comparable.
_OPS = {
    "relu": lambda packed: packed_tensor_functions.relu(packed),
    "gelu": lambda packed: packed_tensor_functions.gelu(packed),
    "layer_norm": lambda packed: packed_tensor_functions.layer_norm(packed),
    "softmax": lambda packed: packed_tensor_functions.softmax(packed),
    "mm": lambda packed: packed.mm(_MM_WEIGHT),
    "conv2d_loop": lambda packed: conv2d(packed, _CONV_WEIGHT, backend=ConvBackend.LOOP),
    "conv2d_im2col": lambda packed: conv2d(
        packed, _CONV_WEIGHT, backend=ConvBackend.IM2COL
    ),
    "pad_to_multiple": lambda packed: packed_tensor.pad_to_multiple(packed, 16),
}
# Pointwise and last-dim ops trace without graph breaks; the shape-dependent ops
# (mm, im2col, pad) break on data-dependent packing, which is expected.
_FULLGRAPH_OPS = ["relu", "gelu", "layer_norm", "softmax"]


def _packed(spatial_shapes, seed=0):
    torch.manual_seed(seed)
    items = [torch.randn(*shape, _CHANNELS) for shape in spatial_shapes]
    return packed_tensor.from_list(items)


@pytest.mark.parametrize("backend", TRACING_BACKENDS)
@pytest.mark.parametrize("name", list(_OPS))
def test_compiled_matches_eager(backend, name):
    op = _OPS[name]
    packed = _packed([(3, 4), (6, 7)])
    torch._dynamo.reset()
    compiled = torch.compile(op, backend=backend)

    expected = op(packed)
    result = compiled(packed)

    assert result.shape() == expected.shape()
    torch.testing.assert_close(result._buffer, expected._buffer)


@pytest.mark.parametrize("backend", TRACING_BACKENDS)
@pytest.mark.parametrize("name", _FULLGRAPH_OPS)
def test_pointwise_and_vector_ops_capture_fullgraph(backend, name):
    op = _OPS[name]
    packed = _packed([(3, 4), (6, 7)])
    torch._dynamo.reset()
    compiled = torch.compile(op, backend=backend, fullgraph=True)

    torch.testing.assert_close(compiled(packed)._buffer, op(packed)._buffer)


@pytest.mark.parametrize("backend", TRACING_BACKENDS)
def test_compiled_preserves_correctness_across_packings(backend):
    torch._dynamo.reset()
    compiled = torch.compile(lambda packed: packed_tensor_functions.layer_norm(packed), backend=backend)

    for spatial_shapes in [[(3, 4), (6, 7)], [(2, 2)], [(8, 8), (1, 9), (4, 4)]]:
        packed = _packed(spatial_shapes)
        torch.testing.assert_close(
            compiled(packed)._buffer, packed_tensor_functions.layer_norm(packed)._buffer
        )


@pytest.mark.parametrize("backend", TRACING_BACKENDS)
def test_compiled_can_build_packed_inside_region(backend):
    def build_and_norm(first, second):
        packed = packed_tensor.from_list([first, second])
        return packed_tensor_functions.layer_norm(packed).tolist()

    torch.manual_seed(0)
    first = torch.randn(3, 4, _CHANNELS)
    second = torch.randn(6, 7, _CHANNELS)
    torch._dynamo.reset()
    compiled = torch.compile(build_and_norm, backend=backend)

    results = compiled(first, second)
    expected = build_and_norm(first, second)
    for result_item, expected_item in zip(results, expected):
        torch.testing.assert_close(result_item, expected_item)


@pytest.mark.skipif(not _INDUCTOR_AVAILABLE, reason="inductor backend unavailable")
def test_inductor_matches_eager():
    packed = _packed([(3, 4), (6, 7)])
    torch._dynamo.reset()
    compiled = torch.compile(lambda packed: packed_tensor_functions.layer_norm(packed))

    torch.testing.assert_close(compiled(packed)._buffer, packed_tensor_functions.layer_norm(packed)._buffer)
