# AGENTS.md

Guidance for AI agents (and humans) working in this repository.

## What this project is

`packed_tensor` is a research codebase for **implementing and experimenting
with new variants of neighborhood attention**. The goal is primarily to implement
support for multi-dimensional packed tensors and important operations, like convolutions
and neighborhood attention. I.e., we want to support batches of images, where each image 
may have different resolutions and aspect ratios. We also want to compare multiple backend 
implementations of the same op for correctness and performance.

This is experimental research code. Prefer clarity and easy comparison between
variants over premature abstraction. It is fine to have several parallel
implementations of the "same" thing in different languages — that is the point.

## Code Conventions

1. Use descriptive variable, function, and class names. You should never use single-letter names.
2. Do not over comment. Do not write comments for things can easily be understood by reading the code.
3. Do not comment WHAT the code is doing, describe WHY the code was written as is.
4. Avoid duplicating complex blocks of code. If two blocks have a similar large code path, just make a function.
5. All code should pass `ruff check`  

## Backends

The same neighborhood-attention semantics are implemented across several backends,
which are kept **mostly separate** from one another:

1. **PyTorch + FlexAttention** — reference-style implementations using eager
   PyTorch and `torch.nn.attention.flex_attention`. These are the readable,
   correctness-defining versions.
2. **Triton** — fused Triton kernels.
3. **CUDA** — hand-written CUDA kernels exposed through a torch C++/CUDA extension.
4. **CuteDSL** — CUTLASS CuteDSL implementations.

Treat the PyTorch/FlexAttention path as a **source of truth** for semantics.
Other backends must match it numerically (within tolerance). When you add a new
variant, add or update the PyTorch reference first, then port to other backends.

### Layout

Keep each backend in its own subtree so they don't bleed into each other. 
Suggested shape (create directories as needed):

```
src/packed_tensor/natten/
  pytorch/      # eager + flex_attention impls
  triton/       # triton kernels + wrappers
  cuda/         # CUDA sources + torch extension bindings
  cutedsl/      # CuteDSL implementations
  common/       # shared, backend-agnostic helpers (shapes, masks, reference math)
tests/          # pytest correctness tests
benchmarks/     # throughput / memory benchmarks
```

Shared, backend-agnostic logic (e.g. neighborhood index math, mask construction,
the numerical reference) belongs in `common/` so every backend tests against the
same definition.

## Environment & tooling

- Use **uv** for everything. Do not invoke `pip` directly.
- Python: see `.python-version` (currently 3.10), constraint `>=3.10`.
- Build backend: `uv_build` (see `pyproject.toml`).

Common commands:

```bash
uv sync                       # create/update the environment
uv add <pkg>                  # add a runtime dependency
uv add --dev <pkg>            # add a dev/test dependency
uv run python -m ...          # run code inside the project env
uv run pytest test/           # run the test suite
uv run ruff check src/        # check that ruff format checks pass
uv run ruff format src/       # use ruff to format the source
```

When adding a dependency, update `pyproject.toml` and run `uv sync` so the
lockfile stays up to date.

## Correctness

Numerical correctness is checked with **pytest**.

- Every backend variant must have a test that compares it against the PyTorch
  reference for matching inputs.
- Parametrize over the things that actually vary the math: dtype, sequence/spatial
  shape, window/neighborhood size, causality, head count.
- Use tolerances appropriate to dtype (tight for fp32/fp64, looser for fp16/bf16).
  State the tolerance explicitly rather than relying on defaults.
- Test both the forward and, where implemented, the backward pass (gradients).
- A new backend kernel is not "done" until it passes the correctness tests against
  the reference.

Run before considering a change complete:

```bash
uv run pytest test/
```

All tests should pass before a change can be considered complete.

## Performance

Performance is tracked with **benchmarks** measuring throughput and memory use.

- Benchmarks live in `benchmarks/` and should report both latency/throughput and
  peak memory.
- Compare new variants/backends against existing ones and against sensible
  baselines (e.g. dense attention or `flex_attention`).
- Keep benchmark configs explicit and reproducible (shapes, dtype, device, warmup
  and iteration counts). Synchronize the device before timing.
- Performance claims should be backed by a benchmark run, not asserted.
- Benchmark results should be easily parseable.

## Conventions for agents

- **Don't unify backends prematurely.** Parallel implementations are expected;
  avoid refactors that couple them just to remove duplication.
- **Don't over comment.** Do not use inline comments to explain simple code.
- **Reference first.** Define new behavior in the PyTorch/FlexAttention reference,
  add a correctness test, then port to Triton/CUDA/CuteDSL.
- **Verify your work.** Run `uv run pytest` for correctness; run the relevant
  benchmark for any performance-related change. Report actual results.
- **Ask for reviews.** If you modify existing code, point it out and ask for it
  to be reviewed.
- Keep new code in the style of the surrounding backend it lives in.
- Don't commit or push unless asked.
