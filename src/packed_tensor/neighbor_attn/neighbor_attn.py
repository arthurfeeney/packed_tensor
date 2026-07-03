from enum import Enum
from typing import Sequence, Union

from packed_tensor.neighbor_attn.backend import flex_attn, natten
from packed_tensor.packed_tensor import PackedTensor


class Backend(Enum):
    FLEX = "flex"
    NATTEN = "natten"


_DISPATCH = {
    Backend.FLEX: flex_attn.natten,
    Backend.NATTEN: natten.natten_loop,
}


def neighbor_attn(
    query: PackedTensor,
    key: PackedTensor,
    value: PackedTensor,
    kernel_size: Union[int, Sequence[int]],
    backend: Union[Backend, str] = Backend.FLEX,
    **backend_kwargs,
) -> PackedTensor:
    """Neighborhood attention over a packed tensor, dispatched to ``backend``.

    ``backend`` accepts a ``Backend`` member or its string value (``"flex"`` /
    ``"natten"``). ``backend_kwargs`` are forwarded to the chosen backend, so
    only pass options it accepts (e.g. ``score_mod`` for flex, ``dilation`` /
    ``is_causal`` for natten).
    """
    backend = Backend(backend)
    return _DISPATCH[backend](query, key, value, kernel_size, **backend_kwargs)
