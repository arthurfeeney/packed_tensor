from typing import Sequence, Union


def to_ntuple(value: Union[int, Sequence[int]], ndim: int) -> tuple:
    if isinstance(value, int):
        return (value,) * ndim
    value = tuple(value)
    assert len(value) == ndim, (
        f"expected {ndim} values but got {len(value)}: {value}"
    )
    return value


def conv_output_extent(extent: int, kernel: int, stride: int, padding: int) -> int:
    return (extent + 2 * padding - kernel) // stride + 1
