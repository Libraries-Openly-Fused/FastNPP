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

// Validates FastNPP AbsDiffC, shifts (LShiftC/RShiftC) and two-image bitwise
// (And/Or/Xor) entry points against NVIDIA NPP.

#ifdef WIN32
#include <tests/main.h>
#endif

#include <fast_npp.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

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

int absDiffC() {
    const int W = 256, H = 8;
    const size_t N = (size_t)W * H;
    std::vector<Npp8u> h(N), ref(N), fkl(N);
    for (size_t i = 0; i < N; ++i) h[i] = (Npp8u)((i * 53) % 256);
    const Npp8u K = 100;
    fk::Ptr2D<uchar> s(W, H), d(W, H);
    const int pitch = (int)s.ptr().dims.pitch, rb = W;
    Npp8u *ds, *dd;
    cudaMalloc(&ds, (size_t)pitch * H); cudaMalloc(&dd, (size_t)pitch * H);
    cudaMemcpy2D(ds, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    nppiAbsDiffC_8u_C1R_Ctx(ds, pitch, dd, pitch, {W, H}, K, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dd, pitch, rb, H, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(s.ptr().data, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    fk::Stream st;
    fk::executeOperations<fk::TransformDPP<>>(st,
        fk::PerThreadRead<fk::ND::_2D, uchar>::build(s),
        fastNPP::AbsDiffC_8u_C1R_Ctx(K),
        fk::PerThreadWrite<fk::ND::_2D, uchar>::build(d));
    st.sync();
    cudaMemcpy2D(fkl.data(), rb, d.ptr().data, pitch, rb, H, cudaMemcpyDeviceToHost);
    int bad = 0; for (size_t i = 0; i < N; ++i) if (ref[i] != fkl[i]) ++bad;
    printf("[%s] AbsDiffC_8u_C1R  K=%d mismatches=%d/%zu\n", bad ? "FAIL" : "PASS", (int)K, bad, N);
    cudaFree(ds); cudaFree(dd);
    return bad;
}

template <typename NppFn, typename FKLChain>
int shiftC(const char* label, Npp32u sh, NppFn nppFn, FKLChain chain) {
    const int W = 256, H = 8;
    const size_t N = (size_t)W * H;
    std::vector<Npp8u> h(N), ref(N), fkl(N);
    for (size_t i = 0; i < N; ++i) h[i] = (Npp8u)((i * 53) % 256);
    fk::Ptr2D<uchar> s(W, H), d(W, H);
    const int pitch = (int)s.ptr().dims.pitch, rb = W;
    Npp8u *ds, *dd;
    cudaMalloc(&ds, (size_t)pitch * H); cudaMalloc(&dd, (size_t)pitch * H);
    cudaMemcpy2D(ds, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    nppFn(ds, pitch, sh, dd, pitch, {W, H}, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dd, pitch, rb, H, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(s.ptr().data, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    fk::Stream st;
    fk::executeOperations<fk::TransformDPP<>>(st,
        fk::PerThreadRead<fk::ND::_2D, uchar>::build(s), chain,
        fk::PerThreadWrite<fk::ND::_2D, uchar>::build(d));
    st.sync();
    cudaMemcpy2D(fkl.data(), rb, d.ptr().data, pitch, rb, H, cudaMemcpyDeviceToHost);
    int bad = 0; for (size_t i = 0; i < N; ++i) if (ref[i] != fkl[i]) ++bad;
    printf("[%s] %-14s sh=%u mismatches=%d/%zu\n", bad ? "FAIL" : "PASS", label, sh, bad, N);
    cudaFree(ds); cudaFree(dd);
    return bad;
}

template <typename NppFn, typename FKLChainFn>
int bwTwoImg(const char* label, NppFn nppFn, FKLChainFn chainFn) {
    const int W = 256, H = 8;
    const size_t N = (size_t)W * H;
    std::vector<Npp8u> h1(N), h2(N), ref(N), fkl(N);
    for (size_t i = 0; i < N; ++i) { h1[i] = (Npp8u)((i * 53) % 256); h2[i] = (Npp8u)((i * 97) % 256); }
    fk::Ptr2D<uchar> s1(W, H), s2(W, H), d(W, H);
    const int pitch = (int)s1.ptr().dims.pitch, rb = W;
    Npp8u *d1, *d2, *dd;
    cudaMalloc(&d1, (size_t)pitch * H); cudaMalloc(&d2, (size_t)pitch * H); cudaMalloc(&dd, (size_t)pitch * H);
    cudaMemcpy2D(d1, pitch, h1.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d2, pitch, h2.data(), rb, rb, H, cudaMemcpyHostToDevice);
    nppFn(d1, pitch, d2, pitch, dd, pitch, {W, H}, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dd, pitch, rb, H, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(s1.ptr().data, pitch, h1.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemcpy2D(s2.ptr().data, pitch, h2.data(), rb, rb, H, cudaMemcpyHostToDevice);
    fk::Stream st;
    fk::executeOperations<fk::TransformDPP<>>(st, chainFn(s1, s2),
        fk::PerThreadWrite<fk::ND::_2D, uchar>::build(d));
    st.sync();
    cudaMemcpy2D(fkl.data(), rb, d.ptr().data, pitch, rb, H, cudaMemcpyDeviceToHost);
    int bad = 0; for (size_t i = 0; i < N; ++i) if (ref[i] != fkl[i]) ++bad;
    printf("[%s] %-14s (two images) mismatches=%d/%zu\n", bad ? "FAIL" : "PASS", label, bad, N);
    cudaFree(d1); cudaFree(d2); cudaFree(dd);
    return bad;
}

} // namespace

int launch() {
    int bad = 0;
    bad += absDiffC();
    bad += shiftC("LShiftC_8u_C1R", 1, nppiLShiftC_8u_C1R_Ctx, fastNPP::LShiftC_8u_C1R_Ctx(1));
    bad += shiftC("RShiftC_8u_C1R", 2, nppiRShiftC_8u_C1R_Ctx, fastNPP::RShiftC_8u_C1R_Ctx(2));
    bad += bwTwoImg("And_8u_C1R", nppiAnd_8u_C1R_Ctx,
                    [](const fk::Ptr2D<uchar>& a, const fk::Ptr2D<uchar>& b){ return fastNPP::And_8u_C1R_Ctx(a, b); });
    bad += bwTwoImg("Or_8u_C1R", nppiOr_8u_C1R_Ctx,
                    [](const fk::Ptr2D<uchar>& a, const fk::Ptr2D<uchar>& b){ return fastNPP::Or_8u_C1R_Ctx(a, b); });
    bad += bwTwoImg("Xor_8u_C1R", nppiXor_8u_C1R_Ctx,
                    [](const fk::Ptr2D<uchar>& a, const fk::Ptr2D<uchar>& b){ return fastNPP::Xor_8u_C1R_Ctx(a, b); });
    printf("%s\n", bad == 0 ? "ALL PASS" : "FAILURES DETECTED");
    return bad == 0 ? 0 : 1;
}
