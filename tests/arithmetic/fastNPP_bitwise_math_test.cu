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

// Validates FastNPP bitwise (AndC/OrC/XorC/Not) and element-wise math
// (Abs/Sqr/Sqrt/Ln/Exp) entry points against NVIDIA NPP.

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

// Bitwise C1 with constant.
template <typename T, typename NppFn, typename FKLChain>
int bwConstC1(const char* label, T K, NppFn nppFn, FKLChain chain) {
    const int W = 256, H = 8;
    const size_t N = static_cast<size_t>(W) * H;
    std::vector<T> h(N), ref(N), fkl(N);
    for (size_t i = 0; i < N; ++i) h[i] = static_cast<T>((i * 53) % 256);
    fk::Ptr2D<T> s(W, H), d(W, H);
    const int pitch = (int)s.ptr().dims.pitch, rb = W * sizeof(T);
    T *ds, *dd;
    cudaMalloc(&ds, (size_t)pitch * H);
    cudaMalloc(&dd, (size_t)pitch * H);
    cudaMemcpy2D(ds, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    nppFn(ds, pitch, K, dd, pitch, {W, H}, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dd, pitch, rb, H, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(s.ptr().data, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    fk::Stream st;
    fk::executeOperations<fk::TransformDPP<>>(st,
        fk::PerThreadRead<fk::ND::_2D, T>::build(s), chain,
        fk::PerThreadWrite<fk::ND::_2D, T>::build(d));
    st.sync();
    cudaMemcpy2D(fkl.data(), rb, d.ptr().data, pitch, rb, H, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (size_t i = 0; i < N; ++i) if (ref[i] != fkl[i]) ++bad;
    printf("[%s] %-14s K=%3d mismatches=%d/%zu\n", bad ? "FAIL" : "PASS", label, (int)K, bad, N);
    cudaFree(ds); cudaFree(dd);
    return bad;
}

// Math C1 (32f), tolerance-based.
template <typename NppFn, typename FKLChain>
int mathC1(const char* label, NppFn nppFn, FKLChain chain, float lo, float hi, float tol) {
    const int W = 512, H = 8;
    const size_t N = static_cast<size_t>(W) * H;
    std::vector<float> h(N), ref(N), fkl(N);
    for (size_t i = 0; i < N; ++i) h[i] = lo + (hi - lo) * ((float)(i % 997) / 997.0f);
    fk::Ptr2D<float> s(W, H), d(W, H);
    const int pitch = (int)s.ptr().dims.pitch, rb = W * sizeof(float);
    Npp32f *ds, *dd;
    cudaMalloc(&ds, (size_t)pitch * H);
    cudaMalloc(&dd, (size_t)pitch * H);
    cudaMemcpy2D(ds, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    nppFn(ds, pitch, dd, pitch, {W, H}, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dd, pitch, rb, H, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(s.ptr().data, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    fk::Stream st;
    fk::executeOperations<fk::TransformDPP<>>(st,
        fk::PerThreadRead<fk::ND::_2D, float>::build(s), chain,
        fk::PerThreadWrite<fk::ND::_2D, float>::build(d));
    st.sync();
    cudaMemcpy2D(fkl.data(), rb, d.ptr().data, pitch, rb, H, cudaMemcpyDeviceToHost);
    int bad = 0; double maxd = 0;
    for (size_t i = 0; i < N; ++i) {
        double e = std::fabs((double)ref[i] - (double)fkl[i]);
        if (e > maxd) maxd = e;
        if (e > tol) ++bad;
    }
    printf("[%s] %-14s max|d|=%.2e mismatches=%d/%zu\n", bad ? "FAIL" : "PASS", label, maxd, bad, N);
    cudaFree(ds); cudaFree(dd);
    return bad;
}

int notC1() {
    const int W = 256, H = 8;
    const size_t N = static_cast<size_t>(W) * H;
    std::vector<Npp8u> h(N), ref(N), fkl(N);
    for (size_t i = 0; i < N; ++i) h[i] = (Npp8u)((i * 53) % 256);
    fk::Ptr2D<uchar> s(W, H), d(W, H);
    const int pitch = (int)s.ptr().dims.pitch, rb = W;
    Npp8u *ds, *dd;
    cudaMalloc(&ds, (size_t)pitch * H);
    cudaMalloc(&dd, (size_t)pitch * H);
    cudaMemcpy2D(ds, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    nppiNot_8u_C1R_Ctx(ds, pitch, dd, pitch, {W, H}, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(ref.data(), rb, dd, pitch, rb, H, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(s.ptr().data, pitch, h.data(), rb, rb, H, cudaMemcpyHostToDevice);
    fk::Stream st;
    fk::executeOperations<fk::TransformDPP<>>(st,
        fk::PerThreadRead<fk::ND::_2D, uchar>::build(s),
        fastNPP::Not_8u_C1R_Ctx(),
        fk::PerThreadWrite<fk::ND::_2D, uchar>::build(d));
    st.sync();
    cudaMemcpy2D(fkl.data(), rb, d.ptr().data, pitch, rb, H, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (size_t i = 0; i < N; ++i) if (ref[i] != fkl[i]) ++bad;
    printf("[%s] %-14s       mismatches=%d/%zu\n", bad ? "FAIL" : "PASS", "Not_8u_C1R", bad, N);
    cudaFree(ds); cudaFree(dd);
    return bad;
}

} // namespace

int launch() {
    int bad = 0;
    for (Npp8u K : {Npp8u(0x0F), Npp8u(0xAA), Npp8u(0xFF)}) {
        bad += bwConstC1<Npp8u>("AndC_8u_C1R", K, nppiAndC_8u_C1R_Ctx, fastNPP::AndC_8u_C1R_Ctx(K));
        bad += bwConstC1<Npp8u>("OrC_8u_C1R",  K, nppiOrC_8u_C1R_Ctx,  fastNPP::OrC_8u_C1R_Ctx(K));
        bad += bwConstC1<Npp8u>("XorC_8u_C1R", K, nppiXorC_8u_C1R_Ctx, fastNPP::XorC_8u_C1R_Ctx(K));
    }
    bad += notC1();

    bad += mathC1("Abs_32f_C1R",  nppiAbs_32f_C1R_Ctx,  fastNPP::Abs_32f_C1R_Ctx(),  -100.f, 100.f,  1e-6f);
    bad += mathC1("Sqr_32f_C1R",  nppiSqr_32f_C1R_Ctx,  fastNPP::Sqr_32f_C1R_Ctx(),  -50.f,  50.f,   1e-2f);
    bad += mathC1("Sqrt_32f_C1R", nppiSqrt_32f_C1R_Ctx, fastNPP::Sqrt_32f_C1R_Ctx(),  0.f,   1000.f, 1e-3f);
    bad += mathC1("Ln_32f_C1R",   nppiLn_32f_C1R_Ctx,   fastNPP::Ln_32f_C1R_Ctx(),    0.1f,  1000.f, 1e-4f);
    bad += mathC1("Exp_32f_C1R",  nppiExp_32f_C1R_Ctx,  fastNPP::Exp_32f_C1R_Ctx(),  -10.f,  10.f,   1e-2f);

    printf("%s\n", bad == 0 ? "ALL PASS" : "FAILURES DETECTED");
    return bad == 0 ? 0 : 1;
}
