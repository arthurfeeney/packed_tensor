"""Visualize the attention pattern FlexAttention uses for packed NATTEN.

This builds the exact ``mask_mod`` that ``packed_tensor.attn.flex.natten`` hands
to ``create_block_mask``, then renders two views side by side:

* the dense ``[total_tokens, total_tokens]`` element mask (which query/key pairs
  are allowed), and
* the coarse ``BlockMask`` flex actually runs (full blocks, partial blocks it
  applies the mask to, and empty blocks it skips).

Example:
    uv run python vis_mask.py --shape 16 16 --shape 8 8 --kernel 3 3 --block-size 8
"""

import argparse
import math
import struct
import zlib

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask

from packed_tensor.attn.flex import natten

BACKGROUND = (250, 250, 248)
PANEL = (243, 242, 237)
BORDER = (176, 174, 168)
ALLOWED = (37, 99, 181)
PARTIAL = (170, 205, 240)
SEPARATOR = (24, 24, 24)

EMPTY_BLOCK = 0
PARTIAL_BLOCK = 1
FULL_BLOCK = 2


def build_pattern(spatial_shapes, kernel_size, block_size, device="cpu"):
    """Dense element mask, per-block class grid and item token boundaries."""
    radii = tuple(size // 2 for size in kernel_size)
    token_item, token_coords, token_sizes = natten._build_metadata(
        spatial_shapes, device
    )
    mask_mod = natten._make_natten_mask_mod(
        token_item, token_coords, token_sizes, radii
    )
    total_tokens = token_item.numel()

    queries, keys = torch.meshgrid(
        torch.arange(total_tokens, device=device),
        torch.arange(total_tokens, device=device),
        indexing="ij",
    )
    dense = mask_mod(0, 0, queries, keys).cpu().numpy()

    block_mask = create_block_mask(
        mask_mod, None, None, total_tokens, total_tokens, device=device,
        BLOCK_SIZE=block_size,
    )

    num_blocks = math.ceil(total_tokens / block_size)
    block_class = np.zeros((num_blocks, num_blocks), dtype=np.uint8)
    for query_block in range(num_blocks):
        for key_block in range(num_blocks):
            window = dense[
                query_block * block_size:(query_block + 1) * block_size,
                key_block * block_size:(key_block + 1) * block_size,
            ]
            if window.all():
                block_class[query_block, key_block] = FULL_BLOCK
            elif window.any():
                block_class[query_block, key_block] = PARTIAL_BLOCK

    token_boundaries = np.cumsum(
        [math.prod(shape) for shape in spatial_shapes]
    )[:-1]
    return dense, block_class, token_boundaries, block_mask


def _downsample_any(mask, factor):
    """Max-pool a boolean matrix so a token block maps to one lit pixel."""
    size = mask.shape[0]
    padded = math.ceil(size / factor) * factor
    canvas = np.zeros((padded, padded), dtype=bool)
    canvas[:size, :size] = mask
    pooled = canvas.reshape(
        padded // factor, factor, padded // factor, factor
    )
    return pooled.any(axis=(1, 3))


def _element_panel(dense, boundaries, max_pixels):
    total = dense.shape[0]
    factor = max(1, math.ceil(total / max_pixels))
    lit = _downsample_any(dense, factor)
    side = lit.shape[0]
    panel = np.full((side, side, 3), PANEL, dtype=np.uint8)
    rows, cols = np.nonzero(lit)
    panel[rows, cols] = ALLOWED
    for boundary in boundaries:
        line = boundary // factor
        if 0 <= line < side:
            panel[:, line] = SEPARATOR
            panel[line, :] = SEPARATOR
    return panel


def _block_panel(block_class, block_size, boundaries, cell_pixels):
    num_blocks = block_class.shape[0]
    side = num_blocks * cell_pixels
    panel = np.full((side, side, 3), PANEL, dtype=np.uint8)
    fills = {FULL_BLOCK: ALLOWED, PARTIAL_BLOCK: PARTIAL}
    for query_block in range(num_blocks):
        for key_block in range(num_blocks):
            kind = block_class[query_block, key_block]
            if kind == EMPTY_BLOCK:
                continue
            top = query_block * cell_pixels
            left = key_block * cell_pixels
            panel[top:top + cell_pixels, left:left + cell_pixels] = fills[kind]
    for boundary in boundaries:
        line = (boundary // block_size) * cell_pixels
        if 0 <= line < side:
            panel[:, line] = SEPARATOR
            panel[line, :] = SEPARATOR
    return panel


def _bordered(panel):
    panel[0, :] = BORDER
    panel[-1, :] = BORDER
    panel[:, 0] = BORDER
    panel[:, -1] = BORDER
    return panel


def compose(element_panel, block_panel, margin=20, gap=40):
    height = margin * 2 + max(element_panel.shape[0], block_panel.shape[0])
    width = (
        margin * 2 + gap + element_panel.shape[1] + block_panel.shape[1]
    )
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)

    def paste(panel, left):
        canvas[margin:margin + panel.shape[0], left:left + panel.shape[1]] = panel

    paste(element_panel, margin)
    paste(block_panel, margin + element_panel.shape[1] + gap)
    return canvas


def write_png(path, image):
    height, width, _ = image.shape
    raw = bytearray()
    for row in range(height):
        raw.append(0)
        raw.extend(image[row].tobytes())
    compressed = zlib.compress(bytes(raw), 9)

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", header))
        handle.write(chunk(b"IDAT", compressed))
        handle.write(chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape", action="append", nargs="+", type=int, metavar="SIZE", dest="shapes",
        help="axis sizes for one item; repeat per item, e.g. --shape 16 16 --shape 8 8",
    )
    parser.add_argument(
        "--kernel", nargs="+", type=int, default=[3], metavar="SIZE",
        help="window size per spatial dim, e.g. --kernel 3 3 (a single value broadcasts)",
    )
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=320,
                        help="max width of the element panel before downsampling")
    parser.add_argument("--block-cell", type=int, default=8,
                        help="pixels per block in the block-mask panel")
    parser.add_argument("--out", default="mask_pattern.png")
    args = parser.parse_args()

    spatial_shapes = [tuple(shape) for shape in (args.shapes or [[16, 16], [8, 8]])]
    ndim = len(spatial_shapes[0])
    if any(len(shape) != ndim for shape in spatial_shapes):
        parser.error("every --shape must list the same number of axes")

    kernel_size = tuple(args.kernel)
    if len(kernel_size) == 1:
        kernel_size = kernel_size * ndim
    if len(kernel_size) != ndim:
        parser.error(f"--kernel has {len(kernel_size)} dims but shapes have {ndim}")

    dense, block_class, boundaries, block_mask = build_pattern(
        spatial_shapes, kernel_size, args.block_size
    )

    total = dense.shape[0]
    print(f"shapes={spatial_shapes} kernel={kernel_size} total_tokens={total}")
    print(f"allowed pairs: {int(dense.sum())} / {total * total} "
          f"({100 * dense.mean():.1f}% dense)")
    print(f"block sparsity: {float(block_mask.sparsity()):.1f}% of blocks skipped")
    print(block_mask.to_string())

    element_panel = _bordered(_element_panel(dense, boundaries, args.max_pixels))
    block_panel = _bordered(
        _block_panel(block_class, args.block_size, boundaries, args.block_cell)
    )
    write_png(args.out, compose(element_panel, block_panel))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
