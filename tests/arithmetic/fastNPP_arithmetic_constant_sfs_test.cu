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

// Validates FastNPP integer arithmetic-with-constant + scale-factor (Sfs)
// entry points against NVIDIA NPP, across scale factors and constants
// (including values that exercise saturation).

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

// Generic C1 verifier: NPP buffers use the same FKL-aligned pitch so both
// paths operate on identical memory layout.
template <typename T, typename NppFn, typename FKLChain>
int verifyC1(const char* label, T K, int sf, NppFn nppFn, FKLChain chain) {
    const int W = 256, H = 8;
    const size_t N = static_cast<size_t>(W) * H;
    std::vector<T> h_src(N), h_ref(N), h_fkl(N);
    for (size_t i = 0; i < N; ++i) h_src[i] = static_cast<T>((i * 37) % 256);

    fk::Ptr2D<T> src(W, H), dst(W, H);
    const int pitch = static_cast<int>(src.ptr().dims.pitch);
    const int rowBytes = W * sizeof(T);

    T *ds, *dd;
    cudaMalloc(&ds, static_cast<size_t>(pitch) * H);
    cudaMalloc(&dd, static_cast<size_t>(pitch) * H);
    cudaMemcpy2D(ds, pitch, h_src.data(), rowBytes, rowBytes, H, cudaMemcpyHostToDevice);
    NppiSize roi{W, H};
    NppStatus st = nppFn(ds, pitch, K, dd, pitch, roi, sf, makeCtx());
    cudaDeviceSynchronize();
    cudaMemcpy2D(h_ref.data(), rowBytes, dd, pitch, rowBytes, H, cudaMemcpyDeviceToHost);

    cudaMemcpy2D(src.ptr().data, pitch, h_src.data(), rowBytes, rowBytes, H, cudaMemcpyHostToDevice);
    fk::Stream stream;
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::PerThreadRead<fk::ND::_2D, T>::build(src),
        chain,
        fk::PerThreadWrite<fk::ND::_2D, T>::build(dst));
    stream.sync();
    cudaMemcpy2D(h_fkl.data(), rowBytes, dst.ptr().data, pitch, rowBytes, H, cudaMemcpyDeviceToHost);

    int bad = 0;
    if (st != NPP_SUCCESS) bad = (int)N;
    else for (size_t i = 0; i < N; ++i) if (h_ref[i] != h_fkl[i]) ++bad;
    printf("[%s] %-20s sf=%d K=%d  status=%d mismatches=%d/%zu\n",
           bad == 0 ? "PASS" : "FAIL", label, sf, (int)K, (int)st, bad, N);
    cudaFree(ds);
    cudaFree(dd);
    return bad;
}

} // namespace

int launch() {
    int bad = 0;
    for (int sf = 0; sf <= 4; ++sf) {
        for (Npp8u K : {Npp8u(3), Npp8u(127), Npp8u(255)}) {
            bad += verifyC1<Npp8u>("AddC_8u_C1RSfs", K, sf, nppiAddC_8u_C1RSfs_Ctx,
                                   fastNPP::AddC_8u_C1RSfs_Ctx(K, sf));
            bad += verifyC1<Npp8u>("SubC_8u_C1RSfs", K, sf, nppiSubC_8u_C1RSfs_Ctx,
                                   fastNPP::SubC_8u_C1RSfs_Ctx(K, sf));
            bad += verifyC1<Npp8u>("MulC_8u_C1RSfs", K, sf, nppiMulC_8u_C1RSfs_Ctx,
                                   fastNPP::MulC_8u_C1RSfs_Ctx(K, sf));
        }
    }
    for (int sf = 0; sf <= 6; ++sf) {
        for (Npp16u K : {Npp16u(7), Npp16u(1000), Npp16u(60000)}) {
            bad += verifyC1<Npp16u>("AddC_16u_C1RSfs", K, sf, nppiAddC_16u_C1RSfs_Ctx,
                                    fastNPP::AddC_16u_C1RSfs_Ctx(K, sf));
            bad += verifyC1<Npp16u>("MulC_16u_C1RSfs", K, sf, nppiMulC_16u_C1RSfs_Ctx,
                                    fastNPP::MulC_16u_C1RSfs_Ctx(K, sf));
        }
    }
    printf("%s\n", bad == 0 ? "ALL PASS" : "FAILURES DETECTED");
    return bad == 0 ? 0 : 1;
}
