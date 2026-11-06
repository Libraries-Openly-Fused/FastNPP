/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

           http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// fastNPP::ResizeBatch_8u32f_C3R_Advanced_Ctx against NPP on images that are not constant.
//
// NPP converts the 8 bit source to 32 bit float (nppiConvert_8u32f_C3R_Ctx) and resizes it (nppiResizeBatch_32f_C3R_Advanced_Ctx);
// the FastNPP wrapper does both in one kernel. A constant image cannot tell two sampling conventions apart, so the sources here are a
// gradient plus seeded noise (the seed is a constant: the test is reproducible). Both must agree to
// AGREEMENT_TOLERANCE * the range of the values (255). As a check of the test itself, the two sampling conventions (NPP's
// pixel centres and the x * scale of fk::Resize without the half pixel policy), computed in double precision on the host, must differ
// from each other by more than 100 times the tolerance on these images, so that a wrapper using the wrong one cannot pass.
#ifdef WIN32
#include <tests/main.h>
#endif

#include <fast_npp.h>
#include <tests/testsNppCommon.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {
constexpr int BATCH = 2;   // NPP's batch resize rejects a batch of one (NPP_BAD_ARGUMENT)
constexpr double RANGE = 255.0;
constexpr double AGREEMENT_TOLERANCE = 1e-4;   // of the range
constexpr unsigned long long SEED = 20261030ull;

NppStreamContext makeContext(const cudaStream_t stream) {
    NppStreamContext ctx{};
    gpuErrchk(cudaGetDevice(&ctx.nCudaDeviceId));
    cudaDeviceProp prop;
    gpuErrchk(cudaGetDeviceProperties(&prop, ctx.nCudaDeviceId));
    ctx.hStream = stream;
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    gpuErrchk(cudaStreamGetFlags(stream, &ctx.nStreamFlags));
    return ctx;
}

// width x height uchar3 pixels: a gradient (different for each image and channel) plus noise of +-24
std::vector<uchar3> makeImage(const int width, const int height, const int imageId, std::mt19937_64& gen) {
    std::uniform_int_distribution<int> noise(-24, 24);
    std::vector<uchar3> img(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double gx = static_cast<double>(x) / std::max(width - 1, 1), gy = static_cast<double>(y) / std::max(height - 1, 1);
            const double base[3] = {gx, gy, 0.5 * (gx + gy)};
            unsigned char c[3];
            for (int k = 0; k < 3; ++k) {
                const double g = std::fmod(base[k] + 0.17 * imageId + 0.05 * k, 1.0);
                c[k] = static_cast<unsigned char>(std::clamp(static_cast<int>(std::lround(g * 200.0 + 20.0)) + noise(gen), 0, 255));
            }
            img[static_cast<size_t>(y) * width + x] = uchar3{c[0], c[1], c[2]};
        }
    }
    return img;
}

// Bilinear resize of one channel in double precision. pixelCentre: src = (x + 0.5) * scale - 0.5, both reads clamped to the image;
// otherwise src = x * scale, the second read clamped (what fk::Resize does without the half pixel policy)
double sample(const std::vector<uchar3>& img, const int w, const int h, const int dx, const int dy, const int dw, const int dh,
              const int channel, const bool pixelCentre) {
    auto px = [&](int x, int y) {
        const uchar3& p = img[static_cast<size_t>(y) * w + x];
        return static_cast<double>(channel == 0 ? p.x : channel == 1 ? p.y : p.z);
    };
    const double fx = static_cast<double>(w) / dw, fy = static_cast<double>(h) / dh;
    const double sx = pixelCentre ? (dx + 0.5) * fx - 0.5 : dx * fx, sy = pixelCentre ? (dy + 0.5) * fy - 0.5 : dy * fy;
    const int x1 = static_cast<int>(std::floor(sx)), y1 = static_cast<int>(std::floor(sy));
    const double ax = sx - x1, ay = sy - y1;
    const int xa = std::clamp(x1, 0, w - 1), xb = std::clamp(x1 + 1, 0, w - 1), ya = std::clamp(y1, 0, h - 1), yb = std::clamp(y1 + 1, 0, h - 1);
    return px(xa, ya) * ((1 - ax) * (1 - ay)) + px(xb, ya) * (ax * (1 - ay)) + px(xa, yb) * ((1 - ax) * ay) + px(xb, yb) * (ax * ay);
}

template <int SRC_W, int SRC_H, int DST_W, int DST_H>
bool runCase(NppStreamContext& ctx, const cudaStream_t stream) {
    std::mt19937_64 gen(SEED);
    std::vector<std::vector<uchar3>> hostImages;
    std::array<fk::Ptr2D<uchar3>, BATCH> dSrc;
    std::array<fk::Ptr2D<float3>, BATCH> dSrcF32, dNpp;
    std::array<fk::Ptr2D<float>, BATCH> dPlane0, dPlane1, dPlane2;
    NppiImageDescriptor hSrc[BATCH], hSrcF32[BATCH], hDst[BATCH];
    NppiResizeBatchROI_Advanced hROI[BATCH];
    std::array<Npp32f*, BATCH> aDst[3];
    for (int i = 0; i < BATCH; ++i) {
        hostImages.push_back(makeImage(SRC_W, SRC_H, i, gen));
        dSrc[i] = fk::Ptr2D<uchar3>(SRC_W, SRC_H);
        dSrcF32[i] = fk::Ptr2D<float3>(SRC_W, SRC_H);
        dNpp[i] = fk::Ptr2D<float3>(DST_W, DST_H);
        dPlane0[i] = fk::Ptr2D<float>(DST_W, DST_H);
        dPlane1[i] = fk::Ptr2D<float>(DST_W, DST_H);
        dPlane2[i] = fk::Ptr2D<float>(DST_W, DST_H);
        gpuErrchk(cudaMemcpy2D(dSrc[i].ptr().data, dSrc[i].ptr().dims.pitch, hostImages[i].data(), SRC_W * sizeof(uchar3),
                               SRC_W * sizeof(uchar3), SRC_H, cudaMemcpyHostToDevice));
        hSrc[i] = NppiImageDescriptor{dSrc[i].ptr().data, static_cast<Npp32s>(dSrc[i].ptr().dims.pitch), NppiSize{SRC_W, SRC_H}};
        hSrcF32[i] = NppiImageDescriptor{dSrcF32[i].ptr().data, static_cast<Npp32s>(dSrcF32[i].ptr().dims.pitch), NppiSize{SRC_W, SRC_H}};
        hDst[i] = NppiImageDescriptor{dNpp[i].ptr().data, static_cast<Npp32s>(dNpp[i].ptr().dims.pitch), NppiSize{DST_W, DST_H}};
        hROI[i].oSrcRectROI = NppiRect{0, 0, SRC_W, SRC_H};
        hROI[i].oDstRectROI = NppiRect{0, 0, DST_W, DST_H};
        aDst[0][i] = dPlane0[i].ptr().data;
        aDst[1][i] = dPlane1[i].ptr().data;
        aDst[2][i] = dPlane2[i].ptr().data;
    }
    NppiImageDescriptor *dSrcF32Desc = nullptr, *dDstDesc = nullptr;
    NppiResizeBatchROI_Advanced* dROI = nullptr;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dSrcF32Desc), sizeof(hSrcF32)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dDstDesc), sizeof(hDst)));
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dROI), sizeof(hROI)));
    gpuErrchk(cudaMemcpy(dSrcF32Desc, hSrcF32, sizeof(hSrcF32), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dDstDesc, hDst, sizeof(hDst), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dROI, hROI, sizeof(hROI), cudaMemcpyHostToDevice));

    // NPP
    for (int i = 0; i < BATCH; ++i) {
        NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(reinterpret_cast<const Npp8u*>(hSrc[i].pData), hSrc[i].nStep, reinterpret_cast<Npp32f*>(hSrcF32[i].pData),
                                            hSrcF32[i].nStep, hSrcF32[i].oSize, ctx));
    }
    NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(DST_W, DST_H, dSrcF32Desc, dDstDesc, dROI, BATCH, NPPI_INTER_LINEAR, ctx));
    // FastNPP
    const auto resize = fastNPP::ResizeBatch_8u32f_C3R_Advanced_Ctx<NPPI_INTER_LINEAR, BATCH>(DST_W, DST_H, hSrc, hROI);
    const auto write = fastNPP::CopyBatch_32f_C3P3R_Ctx<BATCH>(aDst, static_cast<int>(dPlane0[0].ptr().dims.pitch), NppiSize{DST_W, DST_H});
    fastNPP::executeOperations(ctx, resize, write);
    gpuErrchk(cudaStreamSynchronize(stream));

    double maxFastVsNpp = 0, maxFastVsPixelCentre = 0, maxNppVsPixelCentre = 0, maxConventions = 0;
    for (int i = 0; i < BATCH; ++i) {
        std::vector<float3> npp(static_cast<size_t>(DST_W) * DST_H);
        std::vector<float> plane[3];
        gpuErrchk(cudaMemcpy2D(npp.data(), DST_W * sizeof(float3), dNpp[i].ptr().data, dNpp[i].ptr().dims.pitch, DST_W * sizeof(float3), DST_H,
                               cudaMemcpyDeviceToHost));
        const fk::Ptr2D<float>* planes[3] = {&dPlane0[i], &dPlane1[i], &dPlane2[i]};
        for (int c = 0; c < 3; ++c) {
            plane[c].resize(static_cast<size_t>(DST_W) * DST_H);
            gpuErrchk(cudaMemcpy2D(plane[c].data(), DST_W * sizeof(float), planes[c]->ptr().data, planes[c]->ptr().dims.pitch, DST_W * sizeof(float),
                                   DST_H, cudaMemcpyDeviceToHost));
        }
        for (int y = 0; y < DST_H; ++y) {
            for (int x = 0; x < DST_W; ++x) {
                const size_t idx = static_cast<size_t>(y) * DST_W + x;
                const float nppValue[3] = {npp[idx].x, npp[idx].y, npp[idx].z};
                for (int c = 0; c < 3; ++c) {
                    const double fast = plane[c][idx];
                    const double pc = sample(hostImages[i], SRC_W, SRC_H, x, y, DST_W, DST_H, c, true);
                    const double asym = sample(hostImages[i], SRC_W, SRC_H, x, y, DST_W, DST_H, c, false);
                    maxFastVsNpp = std::max(maxFastVsNpp, std::fabs(fast - nppValue[c]));
                    maxFastVsPixelCentre = std::max(maxFastVsPixelCentre, std::fabs(fast - pc));
                    maxNppVsPixelCentre = std::max(maxNppVsPixelCentre, std::fabs(nppValue[c] - pc));
                    maxConventions = std::max(maxConventions, std::fabs(pc - asym));
                }
            }
        }
    }
    gpuErrchk(cudaFree(dSrcF32Desc));
    gpuErrchk(cudaFree(dDstDesc));
    gpuErrchk(cudaFree(dROI));

    const double tolerance = AGREEMENT_TOLERANCE * RANGE;
    const bool agree = maxFastVsNpp <= tolerance;
    const bool discriminates = maxConventions > 100.0 * tolerance;
    std::cout << SRC_W << "x" << SRC_H << " -> " << DST_W << "x" << DST_H << ": max |FastNPP - NPP| = " << maxFastVsNpp
              << " (tolerance " << tolerance << "), |FastNPP - double reference with pixel centres| = " << maxFastVsPixelCentre
              << ", |NPP - same reference| = " << maxNppVsPixelCentre << ", the two conventions differ by up to " << maxConventions
              << " on these images" << (agree ? "" : "  ** FastNPP DOES NOT MATCH NPP **") << (discriminates ? "" : "  ** the image cannot tell the conventions apart **")
              << std::endl;
    return agree && discriminates;
}
}  // namespace

int launch() {
    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));
    NppStreamContext ctx = makeContext(stream);
    bool ok = true;
    ok &= runCase<60, 120, 64, 128>(ctx, stream);     // the crop and size of the benchmarks: a small upscale
    ok &= runCase<640, 360, 128, 72>(ctx, stream);    // a downscale by 5
    ok &= runCase<97, 53, 64, 31>(ctx, stream);       // sizes that are not multiples of anything
    gpuErrchk(cudaStreamDestroy(stream));
    if (!ok) std::cout << "FAILED" << std::endl;
    return ok ? 0 : -1;
}
