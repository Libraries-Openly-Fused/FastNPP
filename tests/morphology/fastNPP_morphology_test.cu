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

// Validates FastNPP morphology (ErodeBorder/DilateBorder) against NVIDIA NPP
// nppiErodeBorder/nppiDilateBorder with NPP_BORDER_REPLICATE.
// Uses all-active (all-ones) rectangular structuring elements so that NPP
// and the FKL rectangular-window implementation produce identical results.

#ifdef WIN32
#include <tests/main.h>
#endif

#include <fast_npp.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <random>

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

template <typename NppFn, typename FastNppFn>
int verify(const char* label, int mW, int mH, int aX, int aY,
           NppFn nppFn, FastNppFn fastNppFn) {
    const int W = 128, H = 96;
    const size_t N = (size_t)W * H;

    std::vector<Npp8u> h(N), ref(N), fkl(N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < N; ++i) h[i] = (Npp8u)dist(rng);

    // All-active rectangular mask (matches FKL rectangular morphology)
    std::vector<Npp8u> hmask(mW * mH, 1);

    // --- NPP reference ---
    fk::Ptr2D<uchar> srcNpp(W, H);
    const int pitch = (int)srcNpp.ptr().dims.pitch;
    const int rb = W;

    Npp8u *dnppSrc, *dnppDst, *dmask;
    cudaMalloc(&dnppSrc, (size_t)pitch * H);
    cudaMalloc(&dnppDst, (size_t)pitch * H);
    cudaMalloc(&dmask,   (size_t)mW * mH);
    cudaMemcpy2D(dnppSrc, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemcpy(dmask, hmask.data(), (size_t)mW * mH, cudaMemcpyHostToDevice);
    cudaMemset(dnppDst, 0, (size_t)pitch * H);

    NppiSize srcSize{ W, H };
    NppiPoint srcOffset{ 0, 0 };
    NppiSize roiSize{ W, H };
    NppiSize maskSize{ mW, mH };
    NppiPoint anchor{ aX, aY };

    nppFn(dnppSrc, pitch, srcSize, srcOffset,
          dnppDst, pitch, roiSize,
          dmask, maskSize, anchor,
          NPP_BORDER_REPLICATE, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dnppDst, pitch, rb, H, cudaMemcpyDeviceToHost);

    // --- FastNPP ---
    fk::Ptr2D<uchar> src(W, H), dst(W, H);
    cudaMemcpy2D(src.ptr().data, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemset(dst.ptr().data, 0, (size_t)pitch * H);

    fastNppFn(src, dst, mW, mH, aX, aY, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(fkl.data(), rb, dst.ptr().data, pitch, rb, H, cudaMemcpyDeviceToHost);

    int bad = 0;
    for (size_t i = 0; i < N; ++i) if (ref[i] != fkl[i]) ++bad;
    printf("[%s] %-26s mask=%dx%d anchor=(%d,%d) mismatches=%d/%zu\n",
           bad ? "FAIL" : "PASS", label, mW, mH, aX, aY, bad, N);

    cudaFree(dnppSrc); cudaFree(dnppDst); cudaFree(dmask);
    return bad;
}

} // namespace

int launch() {
    int bad = 0;
    bad += verify("ErodeBorder_8u_C1R 3x3", 3, 3, 1, 1,
        nppiErodeBorder_8u_C1R_Ctx,
        [](const fk::Ptr2D<uchar>& s, fk::Ptr2D<uchar>& d,
           int mw, int mh, int ax, int ay, NppStreamContext ctx) {
            fastNPP::ErodeBorder_8u_C1R_Ctx(s, d, mw, mh, ax, ay, ctx); });
    bad += verify("DilateBorder_8u_C1R 3x3", 3, 3, 1, 1,
        nppiDilateBorder_8u_C1R_Ctx,
        [](const fk::Ptr2D<uchar>& s, fk::Ptr2D<uchar>& d,
           int mw, int mh, int ax, int ay, NppStreamContext ctx) {
            fastNPP::DilateBorder_8u_C1R_Ctx(s, d, mw, mh, ax, ay, ctx); });
    bad += verify("ErodeBorder_8u_C1R 5x5", 5, 5, 2, 2,
        nppiErodeBorder_8u_C1R_Ctx,
        [](const fk::Ptr2D<uchar>& s, fk::Ptr2D<uchar>& d,
           int mw, int mh, int ax, int ay, NppStreamContext ctx) {
            fastNPP::ErodeBorder_8u_C1R_Ctx(s, d, mw, mh, ax, ay, ctx); });
    bad += verify("DilateBorder_8u_C1R 5x5", 5, 5, 2, 2,
        nppiDilateBorder_8u_C1R_Ctx,
        [](const fk::Ptr2D<uchar>& s, fk::Ptr2D<uchar>& d,
           int mw, int mh, int ax, int ay, NppStreamContext ctx) {
            fastNPP::DilateBorder_8u_C1R_Ctx(s, d, mw, mh, ax, ay, ctx); });
    printf("%s\n", bad == 0 ? "ALL PASS" : "FAILURES DETECTED");
    return bad == 0 ? 0 : 1;
}
