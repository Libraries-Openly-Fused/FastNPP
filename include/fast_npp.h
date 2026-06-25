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
#include <fused_kernel/core/data/ptr_utils.h>

namespace fastNPP {

    // ===== Bitwise operations =====
    // AndC / OrC / XorC with a constant, and Not (no constant). Integer types,
    // C1 / C3 / C4. Each maps the exact NPP name onto an FKL bitwise functor.
#define FASTNPP_DEFINE_BITWISE_C(NPPNAME, T, FKLOP)                        \
    constexpr inline auto NPPNAME(const T& nConstant) {                    \
        return fk::FKLOP<T>::build(nConstant);                             \
    }
#define FASTNPP_DEFINE_BITWISE_CVEC(NPPNAME, VECT, FKLOP)                  \
    constexpr inline auto NPPNAME(const VECT& aConstants) {                \
        return fk::FKLOP<VECT>::build(aConstants);                         \
    }
#define FASTNPP_DEFINE_NOT(NPPNAME, T)                                     \
    constexpr inline auto NPPNAME() { return fk::BwNot<T>::build(); }

    // AndC
    FASTNPP_DEFINE_BITWISE_C(AndC_8u_C1R_Ctx,   uchar,  BwAnd)
    FASTNPP_DEFINE_BITWISE_C(AndC_16u_C1R_Ctx,  ushort, BwAnd)
    FASTNPP_DEFINE_BITWISE_C(AndC_32s_C1R_Ctx,  int,    BwAnd)
    FASTNPP_DEFINE_BITWISE_CVEC(AndC_8u_C3R_Ctx,  uchar3,  BwAnd)
    FASTNPP_DEFINE_BITWISE_CVEC(AndC_8u_C4R_Ctx,  uchar4,  BwAnd)
    FASTNPP_DEFINE_BITWISE_CVEC(AndC_16u_C3R_Ctx, ushort3, BwAnd)
    FASTNPP_DEFINE_BITWISE_CVEC(AndC_16u_C4R_Ctx, ushort4, BwAnd)
    // OrC
    FASTNPP_DEFINE_BITWISE_C(OrC_8u_C1R_Ctx,   uchar,  BwOr)
    FASTNPP_DEFINE_BITWISE_C(OrC_16u_C1R_Ctx,  ushort, BwOr)
    FASTNPP_DEFINE_BITWISE_C(OrC_32s_C1R_Ctx,  int,    BwOr)
    FASTNPP_DEFINE_BITWISE_CVEC(OrC_8u_C3R_Ctx,  uchar3,  BwOr)
    FASTNPP_DEFINE_BITWISE_CVEC(OrC_8u_C4R_Ctx,  uchar4,  BwOr)
    FASTNPP_DEFINE_BITWISE_CVEC(OrC_16u_C3R_Ctx, ushort3, BwOr)
    FASTNPP_DEFINE_BITWISE_CVEC(OrC_16u_C4R_Ctx, ushort4, BwOr)
    // XorC
    FASTNPP_DEFINE_BITWISE_C(XorC_8u_C1R_Ctx,   uchar,  BwXor)
    FASTNPP_DEFINE_BITWISE_C(XorC_16u_C1R_Ctx,  ushort, BwXor)
    FASTNPP_DEFINE_BITWISE_C(XorC_32s_C1R_Ctx,  int,    BwXor)
    FASTNPP_DEFINE_BITWISE_CVEC(XorC_8u_C3R_Ctx,  uchar3,  BwXor)
    FASTNPP_DEFINE_BITWISE_CVEC(XorC_8u_C4R_Ctx,  uchar4,  BwXor)
    FASTNPP_DEFINE_BITWISE_CVEC(XorC_16u_C3R_Ctx, ushort3, BwXor)
    FASTNPP_DEFINE_BITWISE_CVEC(XorC_16u_C4R_Ctx, ushort4, BwXor)
    // Not
    FASTNPP_DEFINE_NOT(Not_8u_C1R_Ctx, uchar)
    FASTNPP_DEFINE_NOT(Not_8u_C3R_Ctx, uchar3)
    FASTNPP_DEFINE_NOT(Not_8u_C4R_Ctx, uchar4)

    // ===== Element-wise math (32f) =====
    // Abs / Sqr / Sqrt / Ln / Exp, C1 / C3.
#define FASTNPP_DEFINE_MATH(NPPNAME, T, FKLOP)                             \
    constexpr inline auto NPPNAME() { return fk::FKLOP<T>::build(); }

    FASTNPP_DEFINE_MATH(Abs_32f_C1R_Ctx,  float,  Abs)
    FASTNPP_DEFINE_MATH(Abs_32f_C3R_Ctx,  float3, Abs)
    FASTNPP_DEFINE_MATH(Sqr_32f_C1R_Ctx,  float,  Sqr)
    FASTNPP_DEFINE_MATH(Sqr_32f_C3R_Ctx,  float3, Sqr)
    FASTNPP_DEFINE_MATH(Sqrt_32f_C1R_Ctx, float,  Sqrt)
    FASTNPP_DEFINE_MATH(Sqrt_32f_C3R_Ctx, float3, Sqrt)
    FASTNPP_DEFINE_MATH(Ln_32f_C1R_Ctx,   float,  Ln)
    FASTNPP_DEFINE_MATH(Ln_32f_C3R_Ctx,   float3, Ln)
    FASTNPP_DEFINE_MATH(Exp_32f_C1R_Ctx,  float,  Exp)
    FASTNPP_DEFINE_MATH(Exp_32f_C3R_Ctx,  float3, Exp)

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