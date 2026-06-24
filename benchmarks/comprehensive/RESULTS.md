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
| Sqrt_32f_C1R | 0.0058 ms | 0.0058 ms | 0.99x | element-wise parity |
| AddC→MulC→SubC→DivC (4 ops) | 0.0191 ms | 0.0058 ms | **3.29x** | fusion |
| DilateBorder_8u_C1R 3x3 | 0.0060 ms | 0.0180 ms | 0.34x | neighbourhood |
| FilterBoxBorder_8u_C1R 3x3 | 0.0058 ms | 0.0140 ms | 0.41x | neighbourhood |
| Box3x3 → RShift (2 ops) | 0.0067 ms | 0.0140 ms | 0.48x | fusion (neighbourhood-bound) |

## Honest interpretation

**Where FastNPP wins — kernel fusion of element-wise ops.** A chain of N
element-wise operations collapses into a single FKL kernel, doing one global-memory
read and one write instead of N round-trips. The 4-op arithmetic chain is **3.29x**
faster than four separate NPP launches. This is FastNPP's structural advantage and
it is real and reproducible.

**Where FastNPP is at parity — single element-wise ops.** AddC, Sqrt, etc. match
NPP to within timing noise (1.00x). No regression, no win — both are memory-bound
and saturate bandwidth.

**Where FastNPP is currently slower — neighbourhood operations.** The morphology
and box/convolution/median filters are implemented as a straightforward
per-thread neighbourhood gather (each output pixel re-reads its window from global
memory). They are **bit-exact vs NPP** but **2-3x slower**, because NPP's filter
kernels use shared-memory tiling / separable passes that this first implementation
does not. The fusion of a neighbourhood op with element-wise ops still pays off
relative to doing them as separate FKL kernels, but cannot overcome the slower
neighbourhood base versus NPP's optimized filters.

## Takeaway

FastNPP's value proposition is **fusion of element-wise pipelines**, where it is
materially faster than chained NPP calls, plus **correctness parity** across every
implemented family (all ops verified bit-exact against NPP). The neighbourhood
filters are correct and usable but are an optimization target: adding shared-memory
tiling and separable-kernel passes to the FKL filter operations is the path to
matching NPP on those, and is tracked as future work.
