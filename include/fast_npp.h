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
#include <fused_kernel/algorithms/basic_ops/bitwise.h>
#include <fused_kernel/algorithms/basic_ops/math.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/core/data/ptr_utils.h>

namespace fastNPP {

    // ===== AbsDiff with constant: |src - C| =====
    constexpr inline auto AbsDiffC_8u_C1R_Ctx(const uchar& nConstant) {
        return fk::AbsDiff<uchar>::build(nConstant);
    }
    constexpr inline auto AbsDiffC_16u_C1R_Ctx(const ushort& nConstant) {
        return fk::AbsDiff<ushort>::build(nConstant);
    }
    constexpr inline auto AbsDiffC_32f_C1R_Ctx(const float& nConstant) {
        return fk::AbsDiff<float>::build(nConstant);
    }

    // ===== Bit shifts by a constant amount =====
    // NPP takes the shift amount as Npp32u.
#define FASTNPP_DEFINE_SHIFT(NPPNAME, T, FKLOP)                            \
    constexpr inline auto NPPNAME(const Npp32u& nConstant) {               \
        return fk::FKLOP<T, Npp32u>::build(nConstant);                     \
    }
    FASTNPP_DEFINE_SHIFT(LShiftC_8u_C1R_Ctx,  uchar,  ShiftLeft)
    FASTNPP_DEFINE_SHIFT(LShiftC_16u_C1R_Ctx, ushort, ShiftLeft)
    FASTNPP_DEFINE_SHIFT(LShiftC_32s_C1R_Ctx, int,    ShiftLeft)
    FASTNPP_DEFINE_SHIFT(RShiftC_8u_C1R_Ctx,  uchar,  ShiftRight)
    FASTNPP_DEFINE_SHIFT(RShiftC_16u_C1R_Ctx, ushort, ShiftRight)
    FASTNPP_DEFINE_SHIFT(RShiftC_32s_C1R_Ctx, int,    ShiftRight)

    // ===== Two-image bitwise (And/Or/Xor) =====
#define FASTNPP_DEFINE_TWO_IMAGE_BW(NPPNAME, T, FKLOP)                          \
    inline auto NPPNAME(const fk::Ptr2D<T>& pSrc1, const fk::Ptr2D<T>& pSrc2) { \
        return fk::DualSourceRead<fk::ND::_2D, T>::build(pSrc2, pSrc1)           \
               .then(fk::FKLOP<T, T, T, fk::UnaryType>::build());               \
    }
    FASTNPP_DEFINE_TWO_IMAGE_BW(And_8u_C1R_Ctx, uchar,  BwAnd)
    FASTNPP_DEFINE_TWO_IMAGE_BW(And_8u_C3R_Ctx, uchar3, BwAnd)
    FASTNPP_DEFINE_TWO_IMAGE_BW(Or_8u_C1R_Ctx,  uchar,  BwOr)
    FASTNPP_DEFINE_TWO_IMAGE_BW(Or_8u_C3R_Ctx,  uchar3, BwOr)
    FASTNPP_DEFINE_TWO_IMAGE_BW(Xor_8u_C1R_Ctx, uchar,  BwXor)
    FASTNPP_DEFINE_TWO_IMAGE_BW(Xor_8u_C3R_Ctx, uchar3, BwXor)

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