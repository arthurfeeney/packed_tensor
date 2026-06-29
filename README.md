# Packed Tensor 

An implementation of packed multidimensional tensors. This is intended
to simplify working with (say) images of different resolutions and aspect ratios.

Packed tensors keep a collection of multiple, variable sized tensors in a single buffer
and applies single kernels to them.

This is similar to PyTorch's nested tensors, which supports one variable axis size.

**Goals**
1. support varlen neighborhood attention
2. support CNNs
3. be fully compatible with torch.compile

## Install

The repo can be setup with `uv`

```bash
uv pip install -e .
```

## Example

```python
import torch
import packed_tensor

# a batch of two 'images' with different resolutions
# There are two requirements:
# 1. the number of dimensions must match in all tensors.
# 2. the size of the last dimension must match.
tensors = [
    torch.zeros((64, 128, 3)),
    torch.ones((256, 32, 3))
]
weight = torch.ones(3, 128)

# `.from_list` can be used as a collate function
pt: PackedTensor = packed_tensor.from_list(tensors)

# apply weights 
out = packed_tensor.mm(pt, weight)
```
