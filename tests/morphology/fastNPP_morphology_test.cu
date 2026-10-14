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

    // Allocate device memory with a pitch aligned by CUDA
    const int rb = W;
    size_t pitchBytes = 0;
    Npp8u *dSrc = nullptr, *dNppDst = nullptr, *dFastDst = nullptr, *dmask = nullptr;
    cudaMallocPitch(reinterpret_cast<void**>(&dSrc),     &pitchBytes, rb, H);
    cudaMalloc(&dNppDst,  pitchBytes * H);
    cudaMalloc(&dFastDst, pitchBytes * H);
    cudaMalloc(&dmask,    (size_t)mW * mH);

    const int pitch = static_cast<int>(pitchBytes);
    cudaMemcpy2D(dSrc, pitchBytes, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemcpy(dmask, hmask.data(), (size_t)mW * mH, cudaMemcpyHostToDevice);
    cudaMemset(dNppDst,  0, pitchBytes * H);
    cudaMemset(dFastDst, 0, pitchBytes * H);

    NppiSize srcSize{ W, H };
    NppiPoint srcOffset{ 0, 0 };
    NppiSize roiSize{ W, H };
    NppiSize maskSize{ mW, mH };
    NppiPoint anchor{ aX, aY };

    // --- NPP reference ---
    nppFn(dSrc, pitch, srcSize, srcOffset,
          dNppDst, pitch, roiSize,
          dmask, maskSize, anchor,
          NPP_BORDER_REPLICATE, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dNppDst, pitchBytes, rb, H, cudaMemcpyDeviceToHost);

    // --- FastNPP (accepts NPP parameters; converts to fk:: internally) ---
    fastNppFn(dSrc, pitch, srcSize, dFastDst, pitch, maskSize, anchor, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(fkl.data(), rb, dFastDst, pitchBytes, rb, H, cudaMemcpyDeviceToHost);

    int bad = 0;
    for (size_t i = 0; i < N; ++i) if (ref[i] != fkl[i]) ++bad;
    printf("[%s] %-26s mask=%dx%d anchor=(%d,%d) mismatches=%d/%zu\n",
           bad ? "FAIL" : "PASS", label, mW, mH, aX, aY, bad, N);

    cudaFree(dSrc); cudaFree(dNppDst); cudaFree(dFastDst); cudaFree(dmask);
    return bad;
}

} // namespace

int launch() {
    int bad = 0;
    bad += verify("ErodeBorder_8u_C1R 3x3", 3, 3, 1, 1,
        nppiErodeBorder_8u_C1R_Ctx,
        [](const Npp8u* s, Npp32s sStep, NppiSize sz,
           Npp8u* d, Npp32s dStep,
           NppiSize mSz, NppiPoint anc, NppStreamContext ctx) {
            int devID = 0; cudaGetDevice(&devID);
            fk::Ptr2D<uchar> fkSrc(reinterpret_cast<uchar*>(const_cast<Npp8u*>(s)),
                static_cast<uint>(sz.width), static_cast<uint>(sz.height),
                static_cast<uint>(sStep), fk::MemType::Device, devID);
            fk::Ptr2D<uchar> fkDst(reinterpret_cast<uchar*>(d),
                static_cast<uint>(sz.width), static_cast<uint>(sz.height),
                static_cast<uint>(dStep), fk::MemType::Device, devID);
            fastNPP::ErodeBorder_8u_C1R_Ctx(
                fk::PerThreadRead<fk::ND::_2D, uchar>::build(fkSrc),
                fk::PerThreadWrite<fk::ND::_2D, uchar>::build(fkDst),
                sz, mSz, anc, ctx); });
    bad += verify("DilateBorder_8u_C1R 3x3", 3, 3, 1, 1,
        nppiDilateBorder_8u_C1R_Ctx,
        [](const Npp8u* s, Npp32s sStep, NppiSize sz,
           Npp8u* d, Npp32s dStep,
           NppiSize mSz, NppiPoint anc, NppStreamContext ctx) {
            int devID = 0; cudaGetDevice(&devID);
            fk::Ptr2D<uchar> fkSrc(reinterpret_cast<uchar*>(const_cast<Npp8u*>(s)),
                static_cast<uint>(sz.width), static_cast<uint>(sz.height),
                static_cast<uint>(sStep), fk::MemType::Device, devID);
            fk::Ptr2D<uchar> fkDst(reinterpret_cast<uchar*>(d),
                static_cast<uint>(sz.width), static_cast<uint>(sz.height),
                static_cast<uint>(dStep), fk::MemType::Device, devID);
            fastNPP::DilateBorder_8u_C1R_Ctx(
                fk::PerThreadRead<fk::ND::_2D, uchar>::build(fkSrc),
                fk::PerThreadWrite<fk::ND::_2D, uchar>::build(fkDst),
                sz, mSz, anc, ctx); });
    bad += verify("ErodeBorder_8u_C1R 5x5", 5, 5, 2, 2,
        nppiErodeBorder_8u_C1R_Ctx,
        [](const Npp8u* s, Npp32s sStep, NppiSize sz,
           Npp8u* d, Npp32s dStep,
           NppiSize mSz, NppiPoint anc, NppStreamContext ctx) {
            int devID = 0; cudaGetDevice(&devID);
            fk::Ptr2D<uchar> fkSrc(reinterpret_cast<uchar*>(const_cast<Npp8u*>(s)),
                static_cast<uint>(sz.width), static_cast<uint>(sz.height),
                static_cast<uint>(sStep), fk::MemType::Device, devID);
            fk::Ptr2D<uchar> fkDst(reinterpret_cast<uchar*>(d),
                static_cast<uint>(sz.width), static_cast<uint>(sz.height),
                static_cast<uint>(dStep), fk::MemType::Device, devID);
            fastNPP::ErodeBorder_8u_C1R_Ctx(
                fk::PerThreadRead<fk::ND::_2D, uchar>::build(fkSrc),
                fk::PerThreadWrite<fk::ND::_2D, uchar>::build(fkDst),
                sz, mSz, anc, ctx); });
    bad += verify("DilateBorder_8u_C1R 5x5", 5, 5, 2, 2,
        nppiDilateBorder_8u_C1R_Ctx,
        [](const Npp8u* s, Npp32s sStep, NppiSize sz,
           Npp8u* d, Npp32s dStep,
           NppiSize mSz, NppiPoint anc, NppStreamContext ctx) {
            int devID = 0; cudaGetDevice(&devID);
            fk::Ptr2D<uchar> fkSrc(reinterpret_cast<uchar*>(const_cast<Npp8u*>(s)),
                static_cast<uint>(sz.width), static_cast<uint>(sz.height),
                static_cast<uint>(sStep), fk::MemType::Device, devID);
            fk::Ptr2D<uchar> fkDst(reinterpret_cast<uchar*>(d),
                static_cast<uint>(sz.width), static_cast<uint>(sz.height),
                static_cast<uint>(dStep), fk::MemType::Device, devID);
            fastNPP::DilateBorder_8u_C1R_Ctx(
                fk::PerThreadRead<fk::ND::_2D, uchar>::build(fkSrc),
                fk::PerThreadWrite<fk::ND::_2D, uchar>::build(fkDst),
                sz, mSz, anc, ctx); });
    printf("%s\n", bad == 0 ? "ALL PASS" : "FAILURES DETECTED");
    return bad == 0 ? 0 : 1;
}
