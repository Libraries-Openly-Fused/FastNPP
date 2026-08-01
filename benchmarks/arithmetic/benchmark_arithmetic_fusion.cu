/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// Benchmark: a chain of 4 constant arithmetic operations (AddC -> MulC -> SubC
// -> DivC) on a 32f C3 image.
//
//   NPP path     : four separate nppi*_C3R_Ctx launches over the same buffer.
//   FastNPP path : the same four operations fused into a single kernel.
//
// Both paths are timed with CUDA events after a warmup, over many iterations,
// and their outputs are cross-checked for numerical agreement so the speedup
// figure refers to identical work.

#ifdef WIN32
#include <tests/main.h>
#endif

#include <fast_npp.h>

#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace {

constexpr int kW = 1920;
constexpr int kH = 1080;
constexpr size_t kN = static_cast<size_t>(kW) * kH;
constexpr int kWarmup = 20;
constexpr int kIters = 200;

NppStreamContext makeCtx(cudaStream_t s) {
    NppStreamContext c{};
    c.hStream = s;
    cudaGetDevice(&c.nCudaDeviceId);
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, c.nCudaDeviceId);
    c.nMultiProcessorCount = p.multiProcessorCount;
    c.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
    c.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
    c.nSharedMemPerBlock = p.sharedMemPerBlock;
    c.nCudaDevAttrComputeCapabilityMajor = p.major;
    c.nCudaDevAttrComputeCapabilityMinor = p.minor;
    cudaStreamGetFlags(s, &c.nStreamFlags);
    return c;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} // namespace

int launch() {
    const float3 kAdd{3.f, 4.f, 5.f};
    const float3 kMul{1.5f, 2.f, 0.5f};
    const float3 kSub{1.f, 2.f, 3.f};
    const float3 kDiv{2.f, 4.f, 8.f};
    const Npp32f addK[3]{kAdd.x, kAdd.y, kAdd.z};
    const Npp32f mulK[3]{kMul.x, kMul.y, kMul.z};
    const Npp32f subK[3]{kSub.x, kSub.y, kSub.z};
    const Npp32f divK[3]{kDiv.x, kDiv.y, kDiv.z};

    std::vector<float3> h_src(kN), h_npp(kN), h_fnpp(kN);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(1.f, 200.f);
    for (size_t i = 0; i < kN; ++i) h_src[i] = {dist(rng), dist(rng), dist(rng)};

    const int step = kW * sizeof(float3);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    NppStreamContext ctx = makeCtx(stream);

    // ---- NPP path: 4 separate launches, in place ----
    Npp32f *d_npp;
    cudaMalloc(&d_npp, kN * sizeof(float3));
    auto nppChain = [&]() {
        nppiAddC_32f_C3R_Ctx(d_npp, step, addK, d_npp, step, {kW, kH}, ctx);
        nppiMulC_32f_C3R_Ctx(d_npp, step, mulK, d_npp, step, {kW, kH}, ctx);
        nppiSubC_32f_C3R_Ctx(d_npp, step, subK, d_npp, step, {kW, kH}, ctx);
        nppiDivC_32f_C3R_Ctx(d_npp, step, divK, d_npp, step, {kW, kH}, ctx);
    };

    // ---- FastNPP path: 4 ops fused into a single kernel ----
    fk::Ptr2D<float3> f_src(kW, kH), f_dst(kW, kH);
    fk::Stream fkStream(stream);
    auto fnppChain = [&]() {
        fk::executeOperations<fk::TransformDPP<>>(fkStream,
            fk::PerThreadRead<fk::ND::_2D, float3>::build(f_src),
            fastNPP::AddC_32f_C3R_Ctx(kAdd),
            fastNPP::MulC_32f_C3R_Ctx(kMul),
            fastNPP::SubC_32f_C3R_Ctx(kSub),
            fastNPP::DivC_32f_C3R_Ctx(kDiv),
            fk::PerThreadWrite<fk::ND::_2D, float3>::build(f_dst));
    };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < kWarmup; ++i) {
        cudaMemcpyAsync(d_npp, h_src.data(), kN * sizeof(float3), cudaMemcpyHostToDevice, stream);
        nppChain();
        cudaMemcpyAsync(f_src.ptr().data, h_src.data(), kN * sizeof(float3), cudaMemcpyHostToDevice, stream);
        fnppChain();
    }
    cudaStreamSynchronize(stream);

    // Time NPP (kernel-only: input is already resident; reset per-iter on device).
    cudaMemcpy(d_npp, h_src.data(), kN * sizeof(float3), cudaMemcpyHostToDevice);
    std::vector<float> nppTimes(kIters);
    for (int i = 0; i < kIters; ++i) {
        cudaMemcpyAsync(d_npp, h_src.data(), kN * sizeof(float3), cudaMemcpyHostToDevice, stream);
        cudaEventRecord(start, stream);
        nppChain();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&nppTimes[i], start, stop);
    }
    cudaMemcpy(h_npp.data(), d_npp, kN * sizeof(float3), cudaMemcpyDeviceToHost);

    // Time FastNPP (single fused kernel).
    cudaMemcpy(f_src.ptr().data, h_src.data(), kN * sizeof(float3), cudaMemcpyHostToDevice);
    std::vector<float> fnppTimes(kIters);
    for (int i = 0; i < kIters; ++i) {
        cudaEventRecord(start, stream);
        fnppChain();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&fnppTimes[i], start, stop);
    }
    cudaMemcpy(h_fnpp.data(), f_dst.ptr().data, kN * sizeof(float3), cudaMemcpyDeviceToHost);

    // Cross-check correctness so the timing refers to identical work.
    size_t bad = 0;
    const float* a = reinterpret_cast<const float*>(h_npp.data());
    const float* b = reinterpret_cast<const float*>(h_fnpp.data());
    for (size_t i = 0; i < kN * 3; ++i) {
        if (std::fabs(a[i] - b[i]) > 1e-3f) ++bad;
    }

    const float nppMed = median(nppTimes);
    const float fnppMed = median(fnppTimes);

    printf("=== AddC->MulC->SubC->DivC, 32f C3, %dx%d, %d iters ===\n", kW, kH, kIters);
    printf("NPP     (4 separate kernels) : median %.4f ms\n", nppMed);
    printf("FastNPP (1 fused kernel)     : median %.4f ms\n", fnppMed);
    printf("Speedup (NPP / FastNPP)      : %.2fx\n", nppMed / fnppMed);
    printf("Correctness vs NPP           : %zu / %zu mismatches\n", bad, kN * 3);

    cudaFree(d_npp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return bad == 0 ? 0 : 1;
}
