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
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/core/data/ptr_utils.h>

namespace fastNPP {

    // ---- Arithmetic with constant, integer types with scale factor (Sfs) ----
    // NPP semantics: dst = saturate_cast<T>( round_half_even( (src OP C) * 2^-scaleFactor ) ).
    // We perform the arithmetic in float (exact for 8u/8s/16u/16s magnitudes),
    // apply the scale, then saturate-cast back to the integer type. The whole
    // chain fuses into a single kernel and composes with neighbouring ops.
    namespace detail {
        template <typename T, typename VecT, template <typename, typename, typename> class FKLOp>
        constexpr inline auto buildScaledConstChain(const VecT& c, int nScaleFactor) {
            using FloatVec = fk::VectorType_t<float, fk::cn<VecT>>;
            const float scale = 1.0f / static_cast<float>(1 << nScaleFactor);
            return fk::Cast<VecT, FloatVec>::build()
                   .then(FKLOp<FloatVec, FloatVec, FloatVec>::build(cxp::cast<FloatVec>::f(c)))
                   .then(fk::Mul<FloatVec>::build(fk::make_set<FloatVec>(scale)))
                   .then(fk::SaturateCast<FloatVec, VecT>::build());
        }
    } // namespace detail

#define FASTNPP_DEFINE_SCALED_CONST_C1(NPPNAME, DTYPE, FKLOP)                          \
    constexpr inline auto NPPNAME(const DTYPE& nConstant, int nScaleFactor) {          \
        return detail::buildScaledConstChain<DTYPE, DTYPE, fk::FKLOP>(nConstant, nScaleFactor); \
    }
#define FASTNPP_DEFINE_SCALED_CONST_C3(NPPNAME, DTYPE, VECT, FKLOP)                    \
    constexpr inline auto NPPNAME(const VECT& aConstants, int nScaleFactor) {          \
        return detail::buildScaledConstChain<VECT, VECT, fk::FKLOP>(aConstants, nScaleFactor); \
    }

    // AddC
    FASTNPP_DEFINE_SCALED_CONST_C1(AddC_8u_C1RSfs_Ctx,  uchar,  Add)
    FASTNPP_DEFINE_SCALED_CONST_C1(AddC_16u_C1RSfs_Ctx, ushort, Add)
    FASTNPP_DEFINE_SCALED_CONST_C1(AddC_16s_C1RSfs_Ctx, short,  Add)
    FASTNPP_DEFINE_SCALED_CONST_C3(AddC_8u_C3RSfs_Ctx,  uchar,  uchar3,  Add)
    FASTNPP_DEFINE_SCALED_CONST_C3(AddC_16u_C3RSfs_Ctx, ushort, ushort3, Add)
    FASTNPP_DEFINE_SCALED_CONST_C3(AddC_16s_C3RSfs_Ctx, short,  short3,  Add)
    // SubC
    FASTNPP_DEFINE_SCALED_CONST_C1(SubC_8u_C1RSfs_Ctx,  uchar,  Sub)
    FASTNPP_DEFINE_SCALED_CONST_C1(SubC_16u_C1RSfs_Ctx, ushort, Sub)
    FASTNPP_DEFINE_SCALED_CONST_C1(SubC_16s_C1RSfs_Ctx, short,  Sub)
    FASTNPP_DEFINE_SCALED_CONST_C3(SubC_8u_C3RSfs_Ctx,  uchar,  uchar3,  Sub)
    FASTNPP_DEFINE_SCALED_CONST_C3(SubC_16u_C3RSfs_Ctx, ushort, ushort3, Sub)
    FASTNPP_DEFINE_SCALED_CONST_C3(SubC_16s_C3RSfs_Ctx, short,  short3,  Sub)
    // MulC
    FASTNPP_DEFINE_SCALED_CONST_C1(MulC_8u_C1RSfs_Ctx,  uchar,  Mul)
    FASTNPP_DEFINE_SCALED_CONST_C1(MulC_16u_C1RSfs_Ctx, ushort, Mul)
    FASTNPP_DEFINE_SCALED_CONST_C1(MulC_16s_C1RSfs_Ctx, short,  Mul)
    FASTNPP_DEFINE_SCALED_CONST_C3(MulC_8u_C3RSfs_Ctx,  uchar,  uchar3,  Mul)
    FASTNPP_DEFINE_SCALED_CONST_C3(MulC_16u_C3RSfs_Ctx, ushort, ushort3, Mul)
    FASTNPP_DEFINE_SCALED_CONST_C3(MulC_16s_C3RSfs_Ctx, short,  short3,  Mul)

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