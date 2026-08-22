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

template <typename NppFn, typename FKLChainFn>
int verify(const char* label, int mW, int mH, int aX, int aY, NppFn nppFn, FKLChainFn chainFn) {
    const int W = 128, H = 96;
    const size_t N = (size_t)W * H;
    std::vector<Npp8u> h(N), ref(N), fkl(N);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> d(0, 255);
    for (size_t i = 0; i < N; ++i) h[i] = (Npp8u)d(rng);
    std::vector<Npp8u> hmask(mW * mH, 1);
    hmask[0] = 0; if (mW * mH > 4) hmask[mW * mH - 1] = 0;

    fk::Ptr2D<uchar> s(W, H), out(W, H), dmask(mW, mH);
    const int pitch = (int)s.ptr().dims.pitch, rb = W;
    const int mpitch = (int)dmask.ptr().dims.pitch;
    Npp8u *ds, *dd, *dm;
    cudaMalloc(&ds, (size_t)pitch * H); cudaMalloc(&dd, (size_t)pitch * H); cudaMalloc(&dm, mW * mH);
    cudaMemcpy2D(ds, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemcpy(dm, hmask.data(), mW * mH, cudaMemcpyHostToDevice);
    cudaMemset(dd, 0, (size_t)pitch * H);
    NppiSize ms{mW, mH}; NppiPoint an{aX, aY};
    NppiSize srcSize{W, H}; NppiPoint srcOff{0, 0};
    nppFn(ds, pitch, srcSize, srcOff, dd, pitch, {W, H}, dm, ms, an, NPP_BORDER_REPLICATE, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dd, pitch, rb, H, cudaMemcpyDeviceToHost);

    cudaMemcpy2D(s.ptr().data, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dmask.ptr().data, mpitch, hmask.data(), mW, mW, mH, cudaMemcpyHostToDevice);
    fk::Stream st;
    fk::executeOperations<fk::TransformDPP<>>(st,
        chainFn(s, dmask, mW, mH, aX, aY),
        fk::PerThreadWrite<fk::ND::_2D, uchar>::build(out));
    st.sync();
    cudaMemcpy2D(fkl.data(), rb, out.ptr().data, pitch, rb, H, cudaMemcpyDeviceToHost);

    int bad = 0; for (size_t i = 0; i < N; ++i) if (ref[i] != fkl[i]) ++bad;
    printf("[%s] %-26s mask=%dx%d anchor=(%d,%d) mismatches=%d/%zu\n",
           bad ? "FAIL" : "PASS", label, mW, mH, aX, aY, bad, N);
    cudaFree(ds); cudaFree(dd); cudaFree(dm);
    return bad;
}

} // namespace

int launch() {
    int bad = 0;
    bad += verify("ErodeBorder_8u_C1R 3x3", 3, 3, 1, 1, nppiErodeBorder_8u_C1R_Ctx,
        [](const fk::Ptr2D<uchar>& s, const fk::Ptr2D<uchar>& m, int mw, int mh, int ax, int ay){
            return fastNPP::ErodeBorder_8u_C1R_Ctx(s, m, mw, mh, ax, ay); });
    bad += verify("DilateBorder_8u_C1R 3x3", 3, 3, 1, 1, nppiDilateBorder_8u_C1R_Ctx,
        [](const fk::Ptr2D<uchar>& s, const fk::Ptr2D<uchar>& m, int mw, int mh, int ax, int ay){
            return fastNPP::DilateBorder_8u_C1R_Ctx(s, m, mw, mh, ax, ay); });
    bad += verify("ErodeBorder_8u_C1R 5x3", 5, 3, 2, 1, nppiErodeBorder_8u_C1R_Ctx,
        [](const fk::Ptr2D<uchar>& s, const fk::Ptr2D<uchar>& m, int mw, int mh, int ax, int ay){
            return fastNPP::ErodeBorder_8u_C1R_Ctx(s, m, mw, mh, ax, ay); });
    bad += verify("DilateBorder_8u_C1R 5x5", 5, 5, 2, 2, nppiDilateBorder_8u_C1R_Ctx,
        [](const fk::Ptr2D<uchar>& s, const fk::Ptr2D<uchar>& m, int mw, int mh, int ax, int ay){
            return fastNPP::DilateBorder_8u_C1R_Ctx(s, m, mw, mh, ax, ay); });
    printf("%s\n", bad == 0 ? "ALL PASS" : "FAILURES DETECTED");
    return bad == 0 ? 0 : 1;
}
