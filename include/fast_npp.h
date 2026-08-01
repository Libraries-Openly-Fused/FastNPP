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
#include <fused_kernel/algorithms/image_processing/morphology.h>
#include <fused_kernel/algorithms/basic_ops/bitwise.h>
#include <fused_kernel/algorithms/basic_ops/math.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/bitwise.h>
#include <fused_kernel/algorithms/basic_ops/math.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/core/data/ptr_utils.h>

namespace fastNPP {

    // ===== Morphology: Erode (min) / Dilate (max) with REPLICATE border =====
    // Implements rectangular (all-active) structuring elements over a
    // configurable mask size and anchor. Border semantics match
    // nppiErodeBorder / nppiDilateBorder with NPP_BORDER_REPLICATE when
    // a full all-ones mask is used. Each function accepts standard NPP
    // parameters and converts them to fk:: types internally.
    // Each function executes a self-contained GPU kernel (not composable
    // via executeOperations).
#define FASTNPP_DEFINE_MORPH(NPPNAME, T, NPP_T, FKL_EXEC_FN)                        \
    inline void NPPNAME(const NPP_T* pSrc, Npp32s nSrcStep, NppiSize oSrcSize,      \
                        NPP_T* pDst, Npp32s nDstStep,                                \
                        NppiSize oMaskSize, NppiPoint oAnchor,                        \
                        NppStreamContext nppStreamCtx) {                              \
        int deviceID{ 0 };                                                            \
        gpuErrchk(cudaGetDevice(&deviceID));                                          \
        fk::Ptr2D<T> fkSrc(reinterpret_cast<T*>(const_cast<NPP_T*>(pSrc)),          \
                            static_cast<uint>(oSrcSize.width),                        \
                            static_cast<uint>(oSrcSize.height),                       \
                            static_cast<uint>(nSrcStep),                              \
                            fk::MemType::Device, deviceID);                           \
        fk::Ptr2D<T> fkDst(reinterpret_cast<T*>(pDst),                               \
                            static_cast<uint>(oSrcSize.width),                        \
                            static_cast<uint>(oSrcSize.height),                       \
                            static_cast<uint>(nDstStep),                              \
                            fk::MemType::Device, deviceID);                           \
        fk::MorphologyDPPDetails<T> details{};                                        \
        details.width   = oSrcSize.width;                                             \
        details.height  = oSrcSize.height;                                            \
        details.maskW   = oMaskSize.width;                                            \
        details.maskH   = oMaskSize.height;                                           \
        details.anchorX = oAnchor.x;                                                  \
        details.anchorY = oAnchor.y;                                                  \
        fk::Stream stream(nppStreamCtx.hStream);                                      \
        fk::FKL_EXEC_FN(details,                                                      \
            fk::PerThreadRead<fk::ND::_2D, T>::build(fkSrc),                         \
            fk::PerThreadWrite<fk::ND::_2D, T>::build(fkDst),                        \
            stream);                                                                   \
    }

    FASTNPP_DEFINE_MORPH(ErodeBorder_8u_C1R_Ctx,   uchar,  Npp8u,  executeErode)
    FASTNPP_DEFINE_MORPH(ErodeBorder_8u_C3R_Ctx,   uchar3, Npp8u,  executeErode)
    FASTNPP_DEFINE_MORPH(ErodeBorder_16u_C1R_Ctx,  ushort, Npp16u, executeErode)
    FASTNPP_DEFINE_MORPH(ErodeBorder_32f_C1R_Ctx,  float,  Npp32f, executeErode)
    FASTNPP_DEFINE_MORPH(DilateBorder_8u_C1R_Ctx,  uchar,  Npp8u,  executeDilate)
    FASTNPP_DEFINE_MORPH(DilateBorder_8u_C3R_Ctx,  uchar3, Npp8u,  executeDilate)
    FASTNPP_DEFINE_MORPH(DilateBorder_16u_C1R_Ctx, ushort, Npp16u, executeDilate)
    FASTNPP_DEFINE_MORPH(DilateBorder_32f_C1R_Ctx, float,  Npp32f, executeDilate)
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
    // ===== Two-image element-wise arithmetic (32f) =====
    // NPP computes dst = pSrc2 OP pSrc1; we feed (src2, src1) into DualSourceRead
    // so the fused Unary operator reproduces NPP's operand order exactly.
    // These return a complete read->op chain; the caller appends the write.
#define FASTNPP_DEFINE_TWO_IMAGE(NPPNAME, T, FKLOP)                              \
    inline auto NPPNAME(const fk::Ptr2D<T>& pSrc1, const fk::Ptr2D<T>& pSrc2) {  \
        return fk::DualSourceRead<fk::ND::_2D, T>::build(pSrc2, pSrc1)            \
               .then(fk::FKLOP<T, T, T, fk::UnaryType>::build());                \
    }

    FASTNPP_DEFINE_TWO_IMAGE(Add_32f_C1R_Ctx,  float,  Add)
    FASTNPP_DEFINE_TWO_IMAGE(Add_32f_C3R_Ctx,  float3, Add)
    FASTNPP_DEFINE_TWO_IMAGE(Sub_32f_C1R_Ctx,  float,  Sub)
    FASTNPP_DEFINE_TWO_IMAGE(Sub_32f_C3R_Ctx,  float3, Sub)
    FASTNPP_DEFINE_TWO_IMAGE(Mul_32f_C1R_Ctx,  float,  Mul)
    FASTNPP_DEFINE_TWO_IMAGE(Mul_32f_C3R_Ctx,  float3, Mul)
    FASTNPP_DEFINE_TWO_IMAGE(Div_32f_C1R_Ctx,  float,  Div)
    FASTNPP_DEFINE_TWO_IMAGE(Div_32f_C3R_Ctx,  float3, Div)
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

    // ---- Arithmetic with constant: 32-bit float, C1 / C3 / C4 ----
    // Each maps the exact NPP entry-point name onto an FKL functor, which is
    // fully fusable with neighbouring operations in an executeOperations chain.

    // AddC
    constexpr inline auto AddC_32f_C1R_Ctx(const float& value) {
        return fk::Add<float>::build(value);
    }
    constexpr inline auto AddC_32f_C3R_Ctx(const float3& value) {
        return fk::Add<float3>::build(value);
    }
    constexpr inline auto AddC_32f_C3R_Ctx(const float (&value)[3]) {
        return fk::Add<float3>::build(fk::make_<float3>(value[0], value[1], value[2]));
    }
    constexpr inline auto AddC_32f_C4R_Ctx(const float4& value) {
        return fk::Add<float4>::build(value);
    }
    constexpr inline auto AddC_32f_C4R_Ctx(const float (&value)[4]) {
        return fk::Add<float4>::build(fk::make_<float4>(value[0], value[1], value[2], value[3]));
    }

    // MulC
    constexpr inline auto MulC_32f_C1R_Ctx(const float& value) {
        return fk::Mul<float>::build(value);
    }
    constexpr inline auto MulC_32f_C3R_Ctx(const float3& value) {
        return fk::Mul<float3>::build(value);
    }
    constexpr inline auto MulC_32f_C3R_Ctx(const float (&value)[3]) {
        return fk::Mul<float3>::build(fk::make_<float3>(value[0], value[1], value[2]));
    }
    constexpr inline auto MulC_32f_C4R_Ctx(const float4& value) {
        return fk::Mul<float4>::build(value);
    }
    constexpr inline auto MulC_32f_C4R_Ctx(const float (&value)[4]) {
        return fk::Mul<float4>::build(fk::make_<float4>(value[0], value[1], value[2], value[3]));
    }

    // SubC
    constexpr inline auto SubC_32f_C1R_Ctx(const float& value) {
        return fk::Sub<float>::build(value);
    }
    constexpr inline auto SubC_32f_C3R_Ctx(const float3& value) {
        return fk::Sub<float3>::build(value);
    }
    constexpr inline auto SubC_32f_C3R_Ctx(const float(&value)[3]) {
        return fk::Sub<float3>::build(fk::make_<float3>(value[0], value[1], value[2]));
    }
    constexpr inline auto SubC_32f_C4R_Ctx(const float4& value) {
        return fk::Sub<float4>::build(value);
    }
    constexpr inline auto SubC_32f_C4R_Ctx(const float(&value)[4]) {
        return fk::Sub<float4>::build(fk::make_<float4>(value[0], value[1], value[2], value[3]));
    }

    // DivC
    constexpr inline auto DivC_32f_C1R_Ctx(const float& value) {
        return fk::Div<float>::build(value);
    }
    constexpr inline auto DivC_32f_C3R_Ctx(const float3& value) {
        return fk::Div<float3>::build(value);
    }
    constexpr inline auto DivC_32f_C3R_Ctx(const float(&value)[3]) {
        return fk::Div<float3>::build(fk::make_<float3>(value[0], value[1], value[2]));
    }
    constexpr inline auto DivC_32f_C4R_Ctx(const float4& value) {
        return fk::Div<float4>::build(value);
    }
    constexpr inline auto DivC_32f_C4R_Ctx(const float(&value)[4]) {
        return fk::Div<float4>::build(fk::make_<float4>(value[0], value[1], value[2], value[3]));
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