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

// Validates FastNPP linear filters (FilterBoxBorder, FilterBorder convolution)
// against NVIDIA NPP with NPP_BORDER_REPLICATE.

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

int box8u(int mW, int mH, int aX, int aY) {
    const int W = 128, H = 96;
    const size_t N = (size_t)W * H;
    std::vector<Npp8u> h(N), ref(N), fkl(N);
    std::mt19937 rng(7); std::uniform_int_distribution<int> di(0, 255);
    for (size_t i = 0; i < N; ++i) h[i] = (Npp8u)di(rng);
    fk::Ptr2D<uchar> s(W, H), out(W, H);
    const int pitch = (int)s.ptr().dims.pitch, rb = W;
    Npp8u *ds, *dd;
    cudaMalloc(&ds, (size_t)pitch * H); cudaMalloc(&dd, (size_t)pitch * H);
    cudaMemcpy2D(ds, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemset(dd, 0, (size_t)pitch * H);
    NppiSize ms{mW, mH}; NppiPoint an{aX, aY}; NppiSize ss{W, H}; NppiPoint so{0, 0};
    nppiFilterBoxBorder_8u_C1R_Ctx(ds, pitch, ss, so, dd, pitch, {W, H}, ms, an, NPP_BORDER_REPLICATE, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dd, pitch, rb, H, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(s.ptr().data, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    fk::Stream st;
    fk::executeOperations<fk::TransformDPP<>>(st,
        fastNPP::FilterBoxBorder_8u_C1R_Ctx(s, mW, mH, aX, aY),
        fk::PerThreadWrite<fk::ND::_2D, uchar>::build(out));
    st.sync();
    cudaMemcpy2D(fkl.data(), rb, out.ptr().data, pitch, rb, H, cudaMemcpyDeviceToHost);
    int bad = 0; for (size_t i = 0; i < N; ++i) if (ref[i] != fkl[i]) ++bad;
    printf("[%s] FilterBoxBorder_8u_C1R %dx%d a(%d,%d) mismatches=%d/%zu\n",
           bad ? "FAIL" : "PASS", mW, mH, aX, aY, bad, N);
    cudaFree(ds); cudaFree(dd);
    return bad;
}

int conv32f() {
    const int W = 128, H = 96, kW = 3, kH = 3, aX = 1, aY = 1;
    const size_t N = (size_t)W * H;
    std::vector<float> h(N), ref(N), fkl(N);
    std::mt19937 rng(9); std::uniform_real_distribution<float> dr(-10, 10);
    for (size_t i = 0; i < N; ++i) h[i] = dr(rng);
    std::vector<float> hk = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    fk::Ptr2D<float> s(W, H), out(W, H), ker(kW, kH);
    const int pitch = (int)s.ptr().dims.pitch, rb = W * sizeof(float), kp = (int)ker.ptr().dims.pitch;
    Npp32f *ds, *dd, *dk;
    cudaMalloc(&ds, (size_t)pitch * H); cudaMalloc(&dd, (size_t)pitch * H); cudaMalloc(&dk, kW * kH * sizeof(float));
    cudaMemcpy2D(ds, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), kW * kH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dd, 0, (size_t)pitch * H);
    NppiSize ks{kW, kH}; NppiPoint an{aX, aY}; NppiSize ss{W, H}; NppiPoint so{0, 0};
    nppiFilterBorder_32f_C1R_Ctx(ds, pitch, ss, so, dd, pitch, {W, H}, dk, ks, an, NPP_BORDER_REPLICATE, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dd, pitch, rb, H, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(s.ptr().data, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemcpy2D(ker.ptr().data, kp, hk.data(), kW * sizeof(float), kW * sizeof(float), kH, cudaMemcpyHostToDevice);
    fk::Stream st;
    fk::executeOperations<fk::TransformDPP<>>(st,
        fastNPP::FilterBorder_32f_C1R_Ctx(s, ker, kW, kH, aX, aY),
        fk::PerThreadWrite<fk::ND::_2D, float>::build(out));
    st.sync();
    cudaMemcpy2D(fkl.data(), rb, out.ptr().data, pitch, rb, H, cudaMemcpyDeviceToHost);
    int bad = 0; double maxd = 0;
    for (size_t i = 0; i < N; ++i) { double d = fabs((double)ref[i] - (double)fkl[i]); if (d > maxd) maxd = d; if (d > 1e-2) ++bad; }
    printf("[%s] FilterBorder_32f_C1R 3x3 conv mismatches=%d/%zu maxdiff=%.3e\n", bad ? "FAIL" : "PASS", bad, N, maxd);
    cudaFree(ds); cudaFree(dd); cudaFree(dk);
    return bad;
}

int medianTest(int mW, int mH, int aX, int aY) {
    const int W = 128, H = 96;
    const size_t N = (size_t)W * H;
    std::vector<Npp8u> h(N), ref(N), fkl(N);
    std::mt19937 rng(5); std::uniform_int_distribution<int> di(0, 255);
    for (size_t i = 0; i < N; ++i) h[i] = (Npp8u)di(rng);
    fk::Ptr2D<uchar> s(W, H), out(W, H);
    const int pitch = (int)s.ptr().dims.pitch, rb = W;
    Npp8u *ds, *dd;
    cudaMalloc(&ds, (size_t)pitch * H); cudaMalloc(&dd, (size_t)pitch * H);
    cudaMemcpy2D(ds, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    cudaMemset(dd, 0, (size_t)pitch * H);
    NppiSize ms{mW, mH}; NppiPoint an{aX, aY}; NppiSize ss{W, H}; NppiPoint so{0, 0};
    Npp32u bufSz = 0;
    nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx({W, H}, ms, &bufSz, NPP_BORDER_REPLICATE, makeCtx());
    Npp8u* dbuf = nullptr; if (bufSz) cudaMalloc(&dbuf, bufSz);
    nppiFilterMedianBorder_8u_C1R_Ctx(ds, pitch, ss, so, dd, pitch, {W, H}, ms, an, dbuf, NPP_BORDER_REPLICATE, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dd, pitch, rb, H, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(s.ptr().data, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    fk::Stream st;
    fk::executeOperations<fk::TransformDPP<>>(st,
        fastNPP::FilterMedianBorder_8u_C1R_Ctx(s, mW, mH, aX, aY),
        fk::PerThreadWrite<fk::ND::_2D, uchar>::build(out));
    st.sync();
    cudaMemcpy2D(fkl.data(), rb, out.ptr().data, pitch, rb, H, cudaMemcpyDeviceToHost);
    int bad = 0; for (size_t i = 0; i < N; ++i) if (ref[i] != fkl[i]) ++bad;
    printf("[%s] FilterMedianBorder_8u_C1R %dx%d a(%d,%d) mismatches=%d/%zu\n",
           bad ? "FAIL" : "PASS", mW, mH, aX, aY, bad, N);
    if (dbuf) cudaFree(dbuf);
    cudaFree(ds); cudaFree(dd);
    return bad;
}

} // namespace

int launch() {
    int bad = 0;
    bad += box8u(3, 3, 1, 1);
    bad += box8u(5, 5, 2, 2);
    bad += conv32f();
    bad += medianTest(3, 3, 1, 1);
    bad += medianTest(5, 5, 2, 2);
    printf("%s\n", bad == 0 ? "ALL PASS" : "FAILURES DETECTED");
    return bad == 0 ? 0 : 1;
}
