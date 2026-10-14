# FastNPP vs NPP — Benchmark Results

Measured on RTX PRO 6000 Blackwell (sm_120), CUDA 13.3, 1920x1080, median of 200
iterations after warmup, CUDA-event timing on a dedicated stream. Every fused
result is cross-checked against the NPP reference, so timings refer to identical
work. Reproduce with:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_BENCHMARK=ON
cmake --build build --target benchmark_fastnpp_vs_npp -j
./build/bin/Release/benchmark_fastnpp_vs_npp
```

## Results

| Operation | NPP | FastNPP | Speedup | Category |
|-----------|-----|---------|---------|----------|
| AddC_32f_C1R | 0.0058 ms | 0.0058 ms | 1.00x | element-wise parity |
| Sqrt_32f_C1R | 0.0058 ms | 0.0058 ms | 1.00x | element-wise parity |
| AddC→MulC→SubC→DivC (4 ops) | 0.0191 ms | 0.0058 ms | **3.31x** | fusion |
| FilterBoxBorder_8u_C1R 3x3 | 0.0058 ms | 0.0120 ms | 0.48x | neighbourhood (small) |
| FilterBoxBorder_8u_C1R 9x9 | 0.0468 ms | 0.0345 ms | **1.36x** | neighbourhood (large) |
| DilateBorder_8u_C1R 3x3 | 0.0058 ms | 0.0181 ms | 0.32x | neighbourhood (small) |

## Interpretation

**Element-wise fusion is the headline win.** A chain of N element-wise operations
collapses into a single FKL kernel — one global read, one write — instead of N
NPP launches. The 4-op arithmetic chain runs **3.31x** faster than four separate
NPP calls. Single element-wise ops are at parity (both saturate bandwidth).

**Large neighbourhood filters are ahead.** At 9x9 the box filter is **1.36x** faster
than NPP. The box filter uses thread fusion (multiple output pixels per thread,
stored coalesced) plus, for single-channel interior groups, sliding-window column
reuse so the overlapping vertical column sums are computed once and shared across
the group's outputs.

**Small neighbourhood filters (3x3) still trail NPP** (~0.32–0.48x). Profiled with
Nsight Compute: at 3x3 the kernel is latency-bound. Thread fusion lifted box 3x3
from 0.41x to 0.48x, but NPP's small-filter "Quad" kernels remain ahead at this
size. Box improved across the board with the optimization; morphology (Dilate)
still uses the scalar path and is the next candidate for the same treatment.

### Optimization progression (box filter, FastNPP vs NPP)

| Kernel | Naive | +interior fast-path | +thread-fusion +column-reuse |
|--------|-------|---------------------|------------------------------|
| 3x3 | 0.41x | 0.41x | **0.48x** |
| 5x5 | 0.29x | 0.29x | **0.36x** |
| 7x7 | 1.05x | 1.21x | **1.41x** |
| 9x9 | 1.00x | 1.16x | **1.36x** |

## Summary

- All operations are **bit-exact vs NPP** (verified on scalar and thread-fused paths,
  including odd image widths; mutation-tested).
- **Element-wise pipelines: materially faster** via fusion (3.3x on a 4-op chain).
- **Large filters: faster** (1.36x at 9x9) after thread-fusion + column-reuse.
- **Small filters: improved but still below NPP** (latency-bound at 3x3). Applying
  thread fusion to morphology/median is the next step.
