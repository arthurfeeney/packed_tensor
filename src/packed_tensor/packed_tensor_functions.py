from typing import Callable

import torch

from .packed_tensor import Indexing, PackedTensor, empty

# public initializer name -> (torch in-place buffer method, pre-bound positional args)
_INITIALIZERS = {
    "zeros": ("fill_", (0,)),
    "ones": ("fill_", (1,)),
    "normal_": ("normal_", ()),
    "randn": ("normal_", (0, 1)),
    "rand": ("uniform_", ()),
}


def _make_initializer(
    name: str, method_name: str, bound_args: tuple
) -> Callable[..., PackedTensor]:
    def initializer(shapes, device=None, dtype=None, **kwargs) -> PackedTensor:
        packed = empty(shapes, device, dtype)
        getattr(packed._buffer, method_name)(*bound_args, **kwargs)
        return packed

    initializer.__name__ = name
    initializer.__qualname__ = name
    initializer.__doc__ = (
        f"Allocate a PackedTensor for the given shapes and initialize it "
        f"in place with `{method_name}`."
    )
    return initializer


for _name, (_method_name, _bound_args) in _INITIALIZERS.items():
    globals()[_name] = _make_initializer(_name, _method_name, _bound_args)


_ELEMENTWISE_OPS = {
    "sigmoid": torch.sigmoid,
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
}


def _make_elementwise(
    name: str, op: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[[PackedTensor], PackedTensor]:
    def elementwise(packed: PackedTensor) -> PackedTensor:
        result = PackedTensor(
            indexing=Indexing(
                packed.shape(),
                packed.stride(),
                packed.end_offset()
            ),
            device=packed._buffer.device,
            dtype=packed._buffer.dtype,
        )
        result._buffer.copy_(op(packed._buffer))
        return result

    elementwise.__name__ = name
    elementwise.__qualname__ = name
    elementwise.__doc__ = (
        f"Apply `{name}` element-wise to a PackedTensor, returning a new one."
    )
    return elementwise


for _name, _op in _ELEMENTWISE_OPS.items():
    globals()[_name] = _make_elementwise(_name, _op)

__all__ = ["empty", *_INITIALIZERS, *_ELEMENTWISE_OPS]
