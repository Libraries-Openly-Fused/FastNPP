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

#ifndef FAST_NPP_CUH
#define FAST_NPP_CUH

#include <npp.h>
#include <nppi_geometry_transforms.h>

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/image_processing/linear_filter.h>
#include <fused_kernel/algorithms/image_processing/median_filter.h>
#include <fused_kernel/core/data/ptr_utils.h>

namespace fastNPP {

    // ===== Linear filters =====
    // FilterBoxBorder: mean filter over a mask. Matches nppiFilterBoxBorder
    // (sum-then-divide, truncated) with NPP_BORDER_REPLICATE.
#define FASTNPP_DEFINE_BOX(NPPNAME, T)                                              \
    inline auto NPPNAME(const fk::Ptr2D<T>& pSrc, int nMaskWidth, int nMaskHeight,  \
                        int nAnchorX, int nAnchorY) {                               \
        return fk::BoxFilter<fk::ND::_2D, T, fk::FilterBorder::REPLICATE>::build(    \
            pSrc, nMaskWidth, nMaskHeight, nAnchorX, nAnchorY);                     \
    }
    FASTNPP_DEFINE_BOX(FilterBoxBorder_8u_C1R_Ctx,  uchar)
    FASTNPP_DEFINE_BOX(FilterBoxBorder_8u_C3R_Ctx,  uchar3)
    FASTNPP_DEFINE_BOX(FilterBoxBorder_16u_C1R_Ctx, ushort)
    FASTNPP_DEFINE_BOX(FilterBoxBorder_32f_C1R_Ctx, float)

    // FilterBorder: general convolution with a float coefficient kernel.
    // Matches nppiFilterBorder_32f with NPP_BORDER_REPLICATE.
#define FASTNPP_DEFINE_CONV(NPPNAME, T)                                             \
    inline auto NPPNAME(const fk::Ptr2D<T>& pSrc, const fk::Ptr2D<float>& pKernel,  \
                        int nKernelWidth, int nKernelHeight, int nAnchorX, int nAnchorY) { \
        return fk::LinearFilter<fk::ND::_2D, T, fk::FilterBorder::REPLICATE,         \
                                fk::FilterRounding::NEAREST>::build(                 \
            pSrc, pKernel, nKernelWidth, nKernelHeight, nAnchorX, nAnchorY);        \
    }
    FASTNPP_DEFINE_CONV(FilterBorder_32f_C1R_Ctx, float)
    FASTNPP_DEFINE_CONV(FilterBorder_32f_C3R_Ctx, float3)

    // FilterMedianBorder: per-channel window median. Matches nppiFilterMedianBorder
    // with NPP_BORDER_REPLICATE. (Note: FKL computes the median directly, so the
    // NPP scratch-buffer argument is not needed.)
#define FASTNPP_DEFINE_MEDIAN(NPPNAME, T)                                          \
    inline auto NPPNAME(const fk::Ptr2D<T>& pSrc, int nMaskWidth, int nMaskHeight, \
                        int nAnchorX, int nAnchorY) {                              \
        return fk::MedianFilter<fk::ND::_2D, T, 49, fk::FilterBorder::REPLICATE>::build( \
            pSrc, nMaskWidth, nMaskHeight, nAnchorX, nAnchorY);                    \
    }
    FASTNPP_DEFINE_MEDIAN(FilterMedianBorder_8u_C1R_Ctx,  uchar)
    FASTNPP_DEFINE_MEDIAN(FilterMedianBorder_8u_C3R_Ctx,  uchar3)
    FASTNPP_DEFINE_MEDIAN(FilterMedianBorder_16u_C1R_Ctx, ushort)
    FASTNPP_DEFINE_MEDIAN(FilterMedianBorder_32f_C1R_Ctx, float)

    template <int INTERPOLATION_MODE, int BATCH>
    constexpr inline auto ResizeBatch_8u32f_C3R_Advanced_Ctx(const int& nMaxWidth, const int& nMaxHeight, 
                                                             const NppiImageDescriptor* const h_pBatchSrc,
                                                             const NppiResizeBatchROI_Advanced* const pBatchROI) {
        static_assert(INTERPOLATION_MODE == NPPI_INTER_LINEAR, "Interpolation mode not supported");
        // currently expecting the destination ROI's to be equal to nMaxWidth and nMaxHeight
        int currentDevice{ 0 };
        gpuErrchk(cudaGetDevice(&currentDevice));
        std::array<fk::Ptr2D<uchar3>, BATCH> srcBatch;
        for (int i = 0; i < BATCH; ++i) {
            srcBatch[i] = fk::Ptr2D<uchar3>(reinterpret_cast<uchar3*>(h_pBatchSrc[i].pData),
                                                                      h_pBatchSrc[i].oSize.width,
                                                                      h_pBatchSrc[i].oSize.height,
                                                                      h_pBatchSrc[i].nStep,
                                                                      fk::MemType::Device, currentDevice);
        }
        const fk::Size dstSize(nMaxWidth, nMaxHeight);
        return fk::PerThreadRead<fk::ND::_2D, uchar3>::build(srcBatch)
               .then(fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(dstSize));
    }

    constexpr inline auto SwapChannels_32f_C3R_Ctx(const int(&dstOrder)[3]) {
        const int3 dstOrderArray{dstOrder[0], dstOrder[1], dstOrder[2]};
        return fk::VectorReorderRT<float3>::build(dstOrderArray);
    }

    constexpr inline auto MulC_32f_C3R_Ctx(const float3& value) {
        return fk::Mul<float3>::build(value);
    }

    constexpr inline auto MulC_32f_C3R_Ctx(const float (&value)[3]) {
        return fk::Mul<float3>::build(fk::make_<float3>(value[0], value[1], value[2]));
    }

    constexpr inline auto SubC_32f_C3R_Ctx(const float3& value) {
        return fk::Sub<float3>::build(value);
    }

    constexpr inline auto SubC_32f_C3R_Ctx(const float(&value)[3]) {
        return fk::Sub<float3>::build(fk::make_<float3>(value[0], value[1], value[2]));
    }
    constexpr inline auto DivC_32f_C3R_Ctx(const float3& value) {
        return fk::Div<float3>::build(value);
    }
    constexpr inline auto DivC_32f_C3R_Ctx(const float(&value)[3]) {
        return fk::Div<float3>::build(fk::make_<float3>(value[0], value[1], value[2]));
    }
    template <size_t BATCH>
    constexpr inline auto CopyBatch_32f_C3P3R_Ctx(const std::array<Npp32f*, BATCH>  (&aDst)[3],
                                                  const int& nDstStep, const NppiSize& oSizeROI) {
        std::array<fk::SplitWriteParams<fk::ND::_2D, float3>, BATCH> params;
        for (int i = 0; i < BATCH; ++i) {
            const uint width = static_cast<uint>(oSizeROI.width);
            const uint height = static_cast<uint>(oSizeROI.height);
            const uint step = static_cast<uint>(nDstStep);
            const fk::PtrDims<fk::ND::_2D> dims{ width, height, step };
            const fk::SplitWriteParams<fk::ND::_2D, float3> param{
                {reinterpret_cast<float*>(aDst[0][i]), dims},
                {reinterpret_cast<float*>(aDst[1][i]), dims},
                {reinterpret_cast<float*>(aDst[2][i]), dims}
            };
            params[i] = param;
        }
        return fk::SplitWrite<fk::ND::_2D, float3>::build(params);
    }

    template <typename... IOps>
    void executeOperations(NppStreamContext& nppStreamCtx, const IOps&... iops) {
        fk::Stream stream(nppStreamCtx.hStream);
        fk::executeOperations<fk::TransformDPP<>>(stream, iops...);
    }

} // namespace fastNPP

#endif