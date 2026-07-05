from typing import Callable

import torch

from .packed_tensor import PackedTensor, empty

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
        result = packed.apply(op)
        return result

    elementwise.__name__ = name
    elementwise.__qualname__ = name
    elementwise.__doc__ = (
        f"Apply `{name}` element-wise to a PackedTensor, returning a new one."
    )
    return elementwise


for _name, _op in _ELEMENTWISE_OPS.items():
    globals()[_name] = _make_elementwise(_name, _op)


# Ops that act over the last (feature) dimension of each vector rather than
# per element. Every entry takes the packed buffer viewed as ``[rows, last_dim]``
# and reduces/normalizes over ``dim=-1``, which is row-independent, so it applies
# to the whole packed buffer at once regardless of how items are packed.
_VECTOR_OPS = {
    "layer_norm": lambda rows, **kwargs: torch.nn.functional.layer_norm(
        rows, rows.shape[-1:], **kwargs
    ),
    "rms_norm": lambda rows, **kwargs: torch.nn.functional.rms_norm(
        rows, rows.shape[-1:], **kwargs
    ),
    "softmax": lambda rows, **kwargs: torch.softmax(rows, dim=-1, **kwargs),
    "log_softmax": lambda rows, **kwargs: torch.log_softmax(rows, dim=-1, **kwargs),
}


def _make_vector(
    name: str, op: Callable[..., torch.Tensor]
) -> Callable[..., PackedTensor]:
    def vector(packed: PackedTensor, **kwargs) -> PackedTensor:
        last_dim = packed._indexing.last_dim()

        def over_rows(buffer: torch.Tensor) -> torch.Tensor:
            return op(buffer.view(-1, last_dim), **kwargs).reshape(-1)

        return packed.apply(over_rows)

    vector.__name__ = name
    vector.__qualname__ = name
    vector.__doc__ = (
        f"Apply `{name}` over the last (feature) dimension of a PackedTensor, "
        f"returning a new one. Extra keyword arguments (e.g. `weight`, `bias`, "
        f"`eps`) are forwarded to the underlying torch op."
    )
    return vector


for _name, _op in _VECTOR_OPS.items():
    globals()[_name] = _make_vector(_name, _op)

__all__ = ["empty", *_INITIALIZERS, *_ELEMENTWISE_OPS, *_VECTOR_OPS]
