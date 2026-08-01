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

// Validates FastNPP two-image arithmetic (Add/Sub/Mul/Div, 32f) against NPP.

#ifdef WIN32
#include <tests/main.h>
#endif

#include <fast_npp.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>

namespace {

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

// C1: NPP computes dst = src2 OP src1.
template <typename NppFn, typename FKLChainFn>
int twoImgC1(const char* label, NppFn nppFn, FKLChainFn chainFn, bool avoidZero) {
    const int W = 512, H = 8;
    const size_t N = static_cast<size_t>(W) * H;
    std::vector<float> h1(N), h2(N), ref(N), fkl(N);
    for (size_t i = 0; i < N; ++i) {
        h1[i] = (float)(i % 200) + (avoidZero ? 1.f : -50.f);
        h2[i] = (float)((i * 7) % 300) + 1.f;
    }
    fk::Ptr2D<float> s1(W, H), s2(W, H), d(W, H);
    const int pitch = (int)s1.ptr().dims.pitch, rb = W * sizeof(float);
    Npp32f *d1, *d2, *dd;
    cudaMalloc(&d1, (size_t)pitch * H);
    cudaMalloc(&d2, (size_t)pitch * H);
    cudaMalloc(&dd, (size_t)pitch * H);
    cudaMemcpy2D(d1, pitch, h1.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d2, pitch, h2.data(), rb, rb, H, cudaMemcpyHostToDevice);
    nppFn(d1, pitch, d2, pitch, dd, pitch, {W, H}, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dd, pitch, rb, H, cudaMemcpyDeviceToHost);

    cudaMemcpy2D(s1.ptr().data, pitch, h1.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemcpy2D(s2.ptr().data, pitch, h2.data(), rb, rb, H, cudaMemcpyHostToDevice);
    fk::Stream st;
    fk::executeOperations<fk::TransformDPP<>>(st,
        chainFn(s1, s2),
        fk::PerThreadWrite<fk::ND::_2D, float>::build(d));
    st.sync();
    cudaMemcpy2D(fkl.data(), rb, d.ptr().data, pitch, rb, H, cudaMemcpyDeviceToHost);

    int bad = 0; double maxd = 0;
    for (size_t i = 0; i < N; ++i) {
        double e = std::fabs((double)ref[i] - (double)fkl[i]);
        if (e > maxd) maxd = e;
        if (e > 1e-3) ++bad;
    }
    printf("[%s] %-14s max|d|=%.2e mismatches=%d/%zu\n", bad ? "FAIL" : "PASS", label, maxd, bad, N);
    cudaFree(d1); cudaFree(d2); cudaFree(dd);
    return bad;
}

} // namespace

int launch() {
    int bad = 0;
    bad += twoImgC1("Add_32f_C1R", nppiAdd_32f_C1R_Ctx,
                    [](const fk::Ptr2D<float>& a, const fk::Ptr2D<float>& b){ return fastNPP::Add_32f_C1R_Ctx(a, b); }, false);
    bad += twoImgC1("Sub_32f_C1R", nppiSub_32f_C1R_Ctx,
                    [](const fk::Ptr2D<float>& a, const fk::Ptr2D<float>& b){ return fastNPP::Sub_32f_C1R_Ctx(a, b); }, false);
    bad += twoImgC1("Mul_32f_C1R", nppiMul_32f_C1R_Ctx,
                    [](const fk::Ptr2D<float>& a, const fk::Ptr2D<float>& b){ return fastNPP::Mul_32f_C1R_Ctx(a, b); }, false);
    bad += twoImgC1("Div_32f_C1R", nppiDiv_32f_C1R_Ctx,
                    [](const fk::Ptr2D<float>& a, const fk::Ptr2D<float>& b){ return fastNPP::Div_32f_C1R_Ctx(a, b); }, true);
    printf("%s\n", bad == 0 ? "ALL PASS" : "FAILURES DETECTED");
    return bad == 0 ? 0 : 1;
}
