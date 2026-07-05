import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from packed_tensor import packed_tensor, packed_tensor_functions
from packed_tensor.conv import ConvBackend, conv2d
from packed_tensor.packed_tensor import PackedTensor


@dataclass
class VarlenViTConfig:
    in_channels: int = 3
    out_channels: int = 3
    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 3
    mlp_ratio: float = 4.0


def _linear(packed: PackedTensor, weight: torch.Tensor, bias: torch.Tensor):
    # Per-item linear layer: ``mm`` contracts the shared last dimension, then the
    # bias broadcasts over every token row of the packed buffer.
    projected = packed.mm(weight)
    out_features = weight.shape[1]
    return projected.apply_(
        lambda buffer: (buffer.view(-1, out_features) + bias).view(-1)
    )


def _sincos_2d_position_embedding(
    height: int, width: int, embed_dim: int, device, dtype
) -> torch.Tensor:
    # Parameter-free 2-D sinusoidal grid so the embedding is defined for any
    # resolution rather than a fixed learned length.
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2-D sincos"
    quarter = embed_dim // 4
    frequencies = 1.0 / (
        10000 ** (torch.arange(quarter, device=device, dtype=dtype) / quarter)
    )
    rows = torch.outer(torch.arange(height, device=device, dtype=dtype), frequencies)
    cols = torch.outer(torch.arange(width, device=device, dtype=dtype), frequencies)
    row_embedding = torch.cat([rows.sin(), rows.cos()], dim=-1)
    col_embedding = torch.cat([cols.sin(), cols.cos()], dim=-1)
    grid = torch.cat(
        [
            row_embedding[:, None, :].expand(height, width, 2 * quarter),
            col_embedding[None, :, :].expand(height, width, 2 * quarter),
        ],
        dim=-1,
    )
    return grid


class TransformerBlock(nn.Module):
    def __init__(self, config: VarlenViTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        embed_dim = config.embed_dim
        hidden_dim = int(embed_dim * config.mlp_ratio)

        self.norm1_weight = nn.Parameter(torch.ones(embed_dim))
        self.norm1_bias = nn.Parameter(torch.zeros(embed_dim))
        self.qkv_weight, self.qkv_bias = _linear_parameters(embed_dim, 3 * embed_dim)
        self.proj_weight, self.proj_bias = _linear_parameters(embed_dim, embed_dim)

        self.norm2_weight = nn.Parameter(torch.ones(embed_dim))
        self.norm2_bias = nn.Parameter(torch.zeros(embed_dim))
        self.fc1_weight, self.fc1_bias = _linear_parameters(embed_dim, hidden_dim)
        self.fc2_weight, self.fc2_bias = _linear_parameters(hidden_dim, embed_dim)

    def forward(self, packed: PackedTensor) -> PackedTensor:
        normed = packed_tensor_functions.layer_norm(
            packed, weight=self.norm1_weight, bias=self.norm1_bias
        )
        packed = packed + self._attention(normed)

        normed = packed_tensor_functions.layer_norm(
            packed, weight=self.norm2_weight, bias=self.norm2_bias
        )
        packed = packed + self._mlp(normed)
        return packed

    def _attention(self, packed: PackedTensor) -> PackedTensor:
        embed_dim = self.qkv_weight.shape[0]
        head_dim = embed_dim // self.num_heads
        qkv = _linear(packed, self.qkv_weight, self.qkv_bias)

        attended_items = []
        for idx in range(len(packed.shape())):
            spatial_shape = packed.shape(idx)[:-1]
            seq_len = math.prod(spatial_shape)
            tokens = qkv[idx].reshape(seq_len, 3 * embed_dim)
            query, key, value = (
                head.reshape(seq_len, self.num_heads, head_dim).transpose(0, 1)
                for head in tokens.chunk(3, dim=-1)
            )
            attention = F.scaled_dot_product_attention(query, key, value)
            merged = attention.transpose(0, 1).reshape(*spatial_shape, embed_dim)
            attended_items.append(merged)

        attended = packed_tensor.from_list(attended_items)
        return _linear(attended, self.proj_weight, self.proj_bias)

    def _mlp(self, packed: PackedTensor) -> PackedTensor:
        hidden = _linear(packed, self.fc1_weight, self.fc1_bias)
        hidden = packed_tensor_functions.gelu(hidden)
        return _linear(hidden, self.fc2_weight, self.fc2_bias)


class VarlenViT(nn.Module):
    def __init__(self, config: VarlenViTConfig):
        super().__init__()
        self.config = config
        self.patch_weight = nn.Parameter(
            torch.empty(
                config.embed_dim,
                config.in_channels,
                config.patch_size,
                config.patch_size,
            )
        )
        nn.init.xavier_uniform_(self.patch_weight.flatten(1).unsqueeze(0))
        self.patch_bias = nn.Parameter(torch.zeros(config.embed_dim))

        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.depth)
        )
        self.final_norm_weight = nn.Parameter(torch.ones(config.embed_dim))
        self.final_norm_bias = nn.Parameter(torch.zeros(config.embed_dim))
        # Each token predicts its whole patch, so the head widens back to a full
        # patch of output pixels that _unpatchify folds into the image.
        patch_pixels = config.patch_size * config.patch_size * config.out_channels
        self.head_weight, self.head_bias = _linear_parameters(
            config.embed_dim, patch_pixels
        )

    def forward(self, images: PackedTensor) -> PackedTensor:
        """Map a packed batch of ``[height, width, in_channels]`` images to a
        packed batch of ``[height, width, out_channels]`` images at the same
        per-item resolution."""
        self._assert_patch_aligned(images)
        patches = conv2d(
            images,
            self.patch_weight,
            self.patch_bias,
            stride=self.config.patch_size,
            backend=ConvBackend.IM2COL,
        )
        patches = self._add_position_embedding(patches)

        for block in self.blocks:
            patches = block(patches)

        patches = packed_tensor_functions.layer_norm(
            patches, weight=self.final_norm_weight, bias=self.final_norm_bias
        )
        tokens = _linear(patches, self.head_weight, self.head_bias)
        return self._unpatchify(tokens)

    def _assert_patch_aligned(self, images: PackedTensor) -> None:
        # Non-overlapping patch embedding only tiles cleanly when every spatial
        # axis is a whole number of patches; the last axis carries the channels.
        patch = self.config.patch_size
        for idx in range(len(images.shape())):
            *spatial, channels = images.shape(idx)
            assert channels == self.config.in_channels, (
                f"item {idx} has {channels} channels, "
                f"expected in_channels={self.config.in_channels}"
            )
            assert all(axis % patch == 0 for axis in spatial), (
                f"item {idx} spatial shape {tuple(spatial)} is not a multiple "
                f"of patch_size={patch}"
            )

    def _add_position_embedding(self, patches: PackedTensor) -> PackedTensor:
        embed_dim = self.config.embed_dim
        position_items = [
            _sincos_2d_position_embedding(
                *patches.shape(idx)[:-1], embed_dim, patches.device, patches.dtype
            )
            for idx in range(len(patches.shape()))
        ]
        return patches + packed_tensor.from_list(position_items)

    def _unpatchify(self, tokens: PackedTensor) -> PackedTensor:
        # Invert the patch embedding: each token's flat patch is folded back into
        # a patch_size x patch_size block, reassembling the full-resolution image.
        patch = self.config.patch_size
        out_channels = self.config.out_channels
        images = []
        for idx in range(len(tokens.shape())):
            grid_height, grid_width = tokens.shape(idx)[:-1]
            blocks = tokens[idx].reshape(
                grid_height, grid_width, patch, patch, out_channels
            )
            image = blocks.permute(0, 2, 1, 3, 4).reshape(
                grid_height * patch, grid_width * patch, out_channels
            )
            images.append(image)
        return packed_tensor.from_list(images)


def _linear_parameters(in_features: int, out_features: int):
    weight = nn.Parameter(torch.empty(in_features, out_features))
    nn.init.xavier_uniform_(weight)
    bias = nn.Parameter(torch.zeros(out_features))
    return weight, bias


def _demo() -> None:
    torch.manual_seed(0)
    config = VarlenViTConfig()
    model = VarlenViT(config).eval()

    # Different resolutions and aspect ratios in a single batch (each side a
    # multiple of the patch size).
    images = [
        torch.randn(64, 96, config.in_channels),
        torch.randn(32, 32, config.in_channels),
        torch.randn(48, 80, config.in_channels),
    ]
    packed = packed_tensor.from_list(images)

    with torch.no_grad():
        output = model(packed)

    print(f"images in batch: {len(images)}")
    print("input shapes: ", [tuple(image.shape) for image in images])
    print("output shapes:", output.shape())


def _small_model():
    torch.manual_seed(0)
    config = VarlenViTConfig(embed_dim=64, depth=2, num_heads=4)
    return VarlenViT(config).eval(), config


def _test_output_preserves_per_item_resolution() -> None:
    model, config = _small_model()
    images = [
        torch.randn(32, 48, config.in_channels),
        torch.randn(16, 16, config.in_channels),
    ]
    with torch.no_grad():
        output = model(packed_tensor.from_list(images))
    for idx, image in enumerate(images):
        height, width, _ = image.shape
        assert output.shape(idx) == (height, width, config.out_channels)


def _test_output_is_independent_of_batch_composition() -> None:
    # The whole point of varlen packing: an image's output must not depend on
    # which other images share its batch (no cross-image attention leakage).
    model, config = _small_model()
    images = [
        torch.randn(32, 48, config.in_channels),
        torch.randn(16, 16, config.in_channels),
        torch.randn(48, 32, config.in_channels),
    ]
    with torch.no_grad():
        batched = model(packed_tensor.from_list(images))
        for idx, image in enumerate(images):
            alone = model(packed_tensor.from_list([image]))
            torch.testing.assert_close(batched[idx], alone[0], rtol=1e-5, atol=1e-5)


def _test_reordering_images_reorders_output() -> None:
    model, config = _small_model()
    first = torch.randn(32, 48, config.in_channels)
    second = torch.randn(16, 16, config.in_channels)
    with torch.no_grad():
        forward = model(packed_tensor.from_list([first, second]))
        reversed_order = model(packed_tensor.from_list([second, first]))
    torch.testing.assert_close(forward[0], reversed_order[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(forward[1], reversed_order[0], rtol=1e-5, atol=1e-5)


def _test_rejects_shapes_not_multiple_of_patch() -> None:
    model, config = _small_model()
    patch = config.patch_size
    misaligned = torch.randn(patch, patch + 1, config.in_channels)
    with torch.no_grad():
        try:
            model(packed_tensor.from_list([misaligned]))
        except AssertionError:
            pass
        else:
            raise AssertionError("expected a non-multiple spatial shape to raise")


def _run_self_tests() -> None:
    _test_output_preserves_per_item_resolution()
    _test_output_is_independent_of_batch_composition()
    _test_reordering_images_reorders_output()
    _test_rejects_shapes_not_multiple_of_patch()
    print("self-tests passed")


if __name__ == "__main__":
    _demo()
    _run_self_tests()
