# Packed Tensor 

A utility library for PyTorch that supports batches of variable-sized multidimensional tensors. 
This is intended to simplify training on (say) images of different resolutions and aspect ratios,
where padding may not always be an effective option.

A `PackedTensor` keeps a collection of multiple, variable sized tensors "packed" into a single buffer
and applies single kernels to them.

**Goals**
1. support varlen neighborhood attention
2. support CNNs and related pooling operations
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
