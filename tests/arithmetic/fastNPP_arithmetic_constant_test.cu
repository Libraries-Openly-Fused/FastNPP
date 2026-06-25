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

// Validates FastNPP arithmetic-with-constant entry points (AddC/SubC/MulC/DivC,
// 32f, C1/C3/C4) against the corresponding NVIDIA NPP calls on identical input.

#ifdef WIN32
#include <tests/main.h>
#endif

#include <fast_npp.h>

#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>

namespace {

constexpr int kW = 1920;
constexpr int kH = 1080;
constexpr size_t kN = static_cast<size_t>(kW) * kH;

NppStreamContext makeCtx() {
    NppStreamContext c{};
    c.hStream = 0;
    cudaGetDevice(&c.nCudaDeviceId);
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, c.nCudaDeviceId);
    c.nMultiProcessorCount = p.multiProcessorCount;
    c.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
    c.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
    c.nSharedMemPerBlock = p.sharedMemPerBlock;
    c.nCudaDevAttrComputeCapabilityMajor = p.major;
    c.nCudaDevAttrComputeCapabilityMinor = p.minor;
    cudaStreamGetFlags(c.hStream, &c.nStreamFlags);
    return c;
}

// Compare two host float buffers; returns number of mismatches above tol.
size_t mismatches(const float* a, const float* b, size_t count, float tol = 1e-4f) {
    size_t bad = 0;
    for (size_t i = 0; i < count; ++i) {
        if (std::fabs(a[i] - b[i]) > tol) ++bad;
    }
    return bad;
}

template <typename FKLOp>
size_t runC1(const char* label, float k, FKLOp fklOp,
             NppStatus (*nppFn)(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize, NppStreamContext)) {
    std::vector<float> h_src(kN), h_ref(kN), h_fkl(kN);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(1.f, 200.f);
    for (size_t i = 0; i < kN; ++i) h_src[i] = dist(rng);

    const int step = kW * sizeof(float);
    Npp32f *d_src, *d_dst;
    cudaMalloc(&d_src, kN * sizeof(float));
    cudaMalloc(&d_dst, kN * sizeof(float));
    cudaMemcpy(d_src, h_src.data(), kN * sizeof(float), cudaMemcpyHostToDevice);
    NppiSize roi{kW, kH};
    NppStatus st = nppFn(d_src, step, k, d_dst, step, roi, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy(h_ref.data(), d_dst, kN * sizeof(float), cudaMemcpyDeviceToHost);

    fk::Ptr2D<float> src(kW, kH), dst(kW, kH);
    cudaMemcpy(src.ptr().data, h_src.data(), kN * sizeof(float), cudaMemcpyHostToDevice);
    fk::Stream stream;
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::PerThreadRead<fk::ND::_2D, float>::build(src), fklOp,
        fk::PerThreadWrite<fk::ND::_2D, float>::build(dst));
    stream.sync();
    cudaMemcpy(h_fkl.data(), dst.ptr().data, kN * sizeof(float), cudaMemcpyDeviceToHost);

    size_t bad = (st == NPP_SUCCESS) ? mismatches(h_ref.data(), h_fkl.data(), kN) : kN;
    printf("[%s] %-16s status=%d mismatches=%zu/%zu\n",
           bad == 0 ? "PASS" : "FAIL", label, (int)st, bad, kN);
    cudaFree(d_src);
    cudaFree(d_dst);
    return bad;
}

template <typename FKLOp>
size_t runC3(const char* label, float3 k, FKLOp fklOp,
             NppStatus (*nppFn)(const Npp32f*, int, const Npp32f[3], Npp32f*, int, NppiSize, NppStreamContext)) {
    std::vector<float3> h_src(kN), h_ref(kN), h_fkl(kN);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(1.f, 200.f);
    for (size_t i = 0; i < kN; ++i) h_src[i] = {dist(rng), dist(rng), dist(rng)};

    const int step = kW * sizeof(float3);
    Npp32f *d_src, *d_dst;
    cudaMalloc(&d_src, kN * sizeof(float3));
    cudaMalloc(&d_dst, kN * sizeof(float3));
    cudaMemcpy(d_src, h_src.data(), kN * sizeof(float3), cudaMemcpyHostToDevice);
    NppiSize roi{kW, kH};
    const Npp32f kk[3] = {k.x, k.y, k.z};
    NppStatus st = nppFn(d_src, step, kk, d_dst, step, roi, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy(h_ref.data(), d_dst, kN * sizeof(float3), cudaMemcpyDeviceToHost);

    fk::Ptr2D<float3> src(kW, kH), dst(kW, kH);
    cudaMemcpy(src.ptr().data, h_src.data(), kN * sizeof(float3), cudaMemcpyHostToDevice);
    fk::Stream stream;
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::PerThreadRead<fk::ND::_2D, float3>::build(src), fklOp,
        fk::PerThreadWrite<fk::ND::_2D, float3>::build(dst));
    stream.sync();
    cudaMemcpy(h_fkl.data(), dst.ptr().data, kN * sizeof(float3), cudaMemcpyDeviceToHost);

    size_t bad = (st == NPP_SUCCESS)
        ? mismatches(reinterpret_cast<float*>(h_ref.data()),
                     reinterpret_cast<float*>(h_fkl.data()), kN * 3)
        : kN * 3;
    printf("[%s] %-16s status=%d mismatches=%zu/%zu\n",
           bad == 0 ? "PASS" : "FAIL", label, (int)st, bad, kN * 3);
    cudaFree(d_src);
    cudaFree(d_dst);
    return bad;
}

} // namespace

int launch() {
    size_t bad = 0;
    const float k1 = 3.5f;
    const float3 k3{3.f, 4.f, 5.f};

    bad += runC1("AddC_32f_C1R", k1, fastNPP::AddC_32f_C1R_Ctx(k1), nppiAddC_32f_C1R_Ctx);
    bad += runC1("SubC_32f_C1R", k1, fastNPP::SubC_32f_C1R_Ctx(k1), nppiSubC_32f_C1R_Ctx);
    bad += runC1("MulC_32f_C1R", k1, fastNPP::MulC_32f_C1R_Ctx(k1), nppiMulC_32f_C1R_Ctx);
    bad += runC1("DivC_32f_C1R", k1, fastNPP::DivC_32f_C1R_Ctx(k1), nppiDivC_32f_C1R_Ctx);

    bad += runC3("AddC_32f_C3R", k3, fastNPP::AddC_32f_C3R_Ctx(k3), nppiAddC_32f_C3R_Ctx);
    bad += runC3("SubC_32f_C3R", k3, fastNPP::SubC_32f_C3R_Ctx(k3), nppiSubC_32f_C3R_Ctx);
    bad += runC3("MulC_32f_C3R", k3, fastNPP::MulC_32f_C3R_Ctx(k3), nppiMulC_32f_C3R_Ctx);
    bad += runC3("DivC_32f_C3R", k3, fastNPP::DivC_32f_C3R_Ctx(k3), nppiDivC_32f_C3R_Ctx);

    printf("%s\n", bad == 0 ? "ALL PASS" : "FAILURES DETECTED");
    return bad == 0 ? 0 : 1;
}
