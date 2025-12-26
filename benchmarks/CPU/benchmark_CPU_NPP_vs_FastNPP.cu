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

#include <tests/testsNppCommon.h>
#include <benchmarks/nppbenchmark.h>
#include <fast_npp.h>

#include "tests/main.h"

#ifdef ENABLE_BENCHMARK
constexpr char VARIABLE_DIMENSION[]{ "Batch size" };

#ifndef CUDART_MAJOR_VERSION
#error CUDART_MAJOR_VERSION Undefined!
#elif (CUDART_MAJOR_VERSION == 11)
constexpr size_t NUM_EXPERIMENTS = 8;
#elif (CUDART_MAJOR_VERSION == 12)
constexpr size_t NUM_EXPERIMENTS = 16;
#endif // CUDART_MAJOR_VERSION

constexpr size_t FIRST_VALUE = 2;
constexpr size_t INCREMENT = 10;
constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

template <typename CV_TYPE_I, typename CV_TYPE_O, int BATCH>
bool test_NPP_cpu_batchresize_x_split3D(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, fk::Stream& stream, bool enabled) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;
    if (enabled) {
        constexpr float alpha = 0.3f;
        constexpr uchar CROP_W = 60;
        constexpr uchar CROP_H = 120;
        constexpr uchar UP_W = 64;
        constexpr uchar UP_H = 128;
        try {
            // Original image from where we will crop
            constexpr uchar3 init_val{ 1, 2, 3 };
            fk::Ptr2D<uchar3> d_input(NUM_ELEMS_X, NUM_ELEMS_Y);
            fk::setTo(init_val, d_input, stream);

            // Batches of crops (fk)
            std::array<fk::Ptr2D<uchar3>, BATCH> d_cropped;
            std::array<fk::Ptr2D<float3>, BATCH> d_cropped_FP32, d_resized_npp, d_swap, d_mul, d_sub, d_div;
            std::array<fk::Ptr2D<float>, BATCH> d_channelA, d_channelB, d_channelC;
            std::array<fk::Ptr2D<float>, BATCH> h_channelA, h_channelB, h_channelC;

            // Batches of crops (npp)
            NppiImageDescriptor* hBatchSrc = nullptr, * dBatchSrc = nullptr, * hBatchSrcFP32 = nullptr, * dBatchSrcFP32 = nullptr, * hBatchDst = nullptr, * dBatchDst = nullptr;
            NppiResizeBatchROI_Advanced* dBatchROI = nullptr, * hBatchROI = nullptr;

            // Init NPP context
            NppStreamContext nppcontext = initNppStreamContext(stream);

            // Initialize function parameters
            constexpr Npp32f mulValue[3] = { alpha, alpha, alpha };
            constexpr Npp32f subValue[3] = { 1.f, 4.f, 3.2f };
            constexpr Npp32f divValue[3] = { 1.f, 4.f, 3.2f };
            constexpr NppiSize crop_size{ CROP_W, CROP_H };
            constexpr NppiSize up_size{ UP_W, UP_H };

            fk::Tensor<float> h_tensor(UP_W, UP_H, BATCH, 3, fk::MemType::HostPinned);
            fk::Tensor<float> d_tensor(UP_W, UP_H, BATCH, 3);
            std::array<Npp32f*, BATCH> d_dst_output[3];
            for (int planeId = 0; planeId < 3; ++planeId) {
                for (int batchId = 0; batchId < BATCH; ++batchId) {
                    uint plane_padding = d_tensor.ptr().dims.plane_pitch * planeId;
                    d_dst_output[planeId][batchId] =
                        reinterpret_cast<Npp32f*>(reinterpret_cast<uchar*>(fk::PtrAccessor<fk::ND::_3D>::point(fk::Point(0,0,batchId), d_tensor.ptr())) + plane_padding);
                }
            }

            // asume RGB->BGR
            const int aDstOrder[3] = { BLUE, GREEN, RED };
            gpuErrchk(cudaMallocHost(reinterpret_cast<void**>(&hBatchSrc), sizeof(NppiImageDescriptor) * BATCH));
            gpuErrchk(cudaMallocHost(reinterpret_cast<void**>(&hBatchSrcFP32), sizeof(NppiImageDescriptor) * BATCH));
            gpuErrchk(cudaMallocHost(reinterpret_cast<void**>(&hBatchDst), sizeof(NppiImageDescriptor) * BATCH));
            gpuErrchk(cudaMallocHost(reinterpret_cast<void**>(&hBatchROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));
            gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dBatchSrc), sizeof(NppiImageDescriptor) * BATCH));
            gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dBatchSrcFP32), sizeof(NppiImageDescriptor) * BATCH));
            gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dBatchDst), sizeof(NppiImageDescriptor) * BATCH));
            gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dBatchROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));

            // dest images (Rgb, 32f)
            for (int i = 0; i < BATCH; ++i) {
                // NPP variables
                d_cropped_FP32[i] = fk::Ptr2D<float3>(CROP_W, CROP_H);
                d_resized_npp[i] = fk::Ptr2D<float3>(UP_W, UP_H);
                d_swap[i] = fk::Ptr2D<float3>(UP_W, UP_H);
                d_mul[i] = fk::Ptr2D<float3>(UP_W, UP_H);
                d_sub[i] = fk::Ptr2D<float3>(UP_W, UP_H);
                d_div[i] = fk::Ptr2D<float3>(UP_W, UP_H);

                // Fill NPP Batch structs
                hBatchDst[i].pData = d_resized_npp[i].ptr().data;
                hBatchDst[i].nStep = d_resized_npp[i].ptr().dims.pitch;
                hBatchDst[i].oSize = up_size;

                const fk::Point current_start_coord(i, i);
                hBatchSrc[i].pData = reinterpret_cast<void*>(fk::PtrAccessor<fk::ND::_2D>::point(current_start_coord, d_input.ptr()));
                hBatchSrc[i].nStep = d_input.dims().pitch;
                hBatchSrc[i].oSize = crop_size;

                hBatchSrcFP32[i].pData = d_cropped_FP32[i].ptr().data;
                hBatchSrcFP32[i].nStep = d_cropped_FP32[i].dims().pitch;
                hBatchSrcFP32[i].oSize = up_size;

                hBatchROI[i].oSrcRectROI = NppiRect{ 0, 0, CROP_W, CROP_H };
                hBatchROI[i].oDstRectROI = NppiRect{ 0, 0, UP_W, UP_H };

                // Allocate pointers for split images (device)
                d_channelA[i] = fk::Ptr2D<float>(UP_W, UP_H);
                d_channelB[i] = fk::Ptr2D<float>(UP_W, UP_H);
                d_channelC[i] = fk::Ptr2D<float>(UP_W, UP_H);
                // Allocate pointers for split images (host)
                h_channelA[i] = fk::Ptr2D<float>(UP_W, UP_H, 0, fk::MemType::HostPinned);
                h_channelB[i] = fk::Ptr2D<float>(UP_W, UP_H, 0, fk::MemType::HostPinned);
                h_channelC[i] = fk::Ptr2D<float>(UP_W, UP_H, 0, fk::MemType::HostPinned);

                std::array<Npp32f*, 3> ptrs = { reinterpret_cast<Npp32f*>(d_channelA[i].ptr().data),
                                                reinterpret_cast<Npp32f*>(d_channelB[i].ptr().data),
                                                reinterpret_cast<Npp32f*>(d_channelC[i].ptr().data) };
                
            }

            gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**>(dBatchSrcFP32), hBatchSrcFP32, sizeof(NppiImageDescriptor) * BATCH,
                cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**>(dBatchSrc), hBatchSrc, sizeof(NppiImageDescriptor) * BATCH,
                cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**>(dBatchDst), hBatchDst, sizeof(NppiImageDescriptor) * BATCH,
                cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void**>(dBatchROI), hBatchROI, sizeof(NppiResizeBatchROI_Advanced) * BATCH,
                cudaMemcpyHostToDevice, stream));

            std::array<fk::RawPtr<fk::ND::_2D, uchar3>, BATCH> d_crop_fk;
            for (int i = 0; i < BATCH; i++) {
                d_crop_fk[i] = d_input.crop2D(fk::Point(i, i), fk::PtrDims<fk::ND::_2D>(CROP_W, CROP_H));
            }

            auto benchmarkTemp =
                initCPUBenchmark<ITERS, BATCH, FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>(__func__, VARIABLE_DIMENSION);
            for (int i = 0; i < ITERS; i++) {
                // NPP version
                startNPP_CPU(benchmarkTemp);
                // convert to 32f
                for (int j = 0; j < BATCH; ++j) {
                    NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(reinterpret_cast<const Npp8u*>(hBatchSrc[j].pData), hBatchSrc[j].nStep, reinterpret_cast<Npp32f*>(hBatchSrcFP32[j].pData),
                        hBatchSrcFP32[j].nStep, hBatchSrcFP32[j].oSize, nppcontext));
                }
                NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(UP_W, UP_H, dBatchSrcFP32, dBatchDst, dBatchROI, BATCH,
                    NPPI_INTER_LINEAR, nppcontext));

                for (int j = 0; j < BATCH; ++j) {
                    NPP_CHECK(nppiSwapChannels_32f_C3R_Ctx(
                        reinterpret_cast<Npp32f*>(d_resized_npp[j].ptr().data), d_resized_npp[j].ptr().dims.pitch,
                        reinterpret_cast<Npp32f*>(d_swap[j].ptr().data), d_swap[j].dims().pitch, up_size, aDstOrder, nppcontext));

                    NPP_CHECK(nppiMulC_32f_C3R_Ctx(reinterpret_cast<Npp32f*>(d_swap[j].ptr().data), d_swap[j].ptr().dims.pitch,
                        mulValue, reinterpret_cast<Npp32f*>(d_mul[j].ptr().data),
                        d_mul[j].ptr().dims.pitch, up_size, nppcontext));

                    NPP_CHECK(nppiSubC_32f_C3R_Ctx(reinterpret_cast<Npp32f*>(d_mul[j].ptr().data), d_mul[j].ptr().dims.pitch,
                        subValue, reinterpret_cast<Npp32f*>(d_sub[j].ptr().data),
                        d_sub[j].ptr().dims.pitch, up_size, nppcontext));

                    NPP_CHECK(nppiDivC_32f_C3R_Ctx(reinterpret_cast<Npp32f*>(d_sub[j].ptr().data), d_sub[j].ptr().dims.pitch,
                        divValue, reinterpret_cast<Npp32f*>(d_div[j].ptr().data),
                        d_div[j].ptr().dims.pitch, up_size, nppcontext));

                    Npp32f* const aDst_arr[3] = { reinterpret_cast<Npp32f*>(d_channelA[j].ptr().data),
                                                 reinterpret_cast<Npp32f*>(d_channelB[j].ptr().data),
                                                 reinterpret_cast<Npp32f*>(d_channelC[j].ptr().data) };

                    NPP_CHECK(nppiCopy_32f_C3P3R_Ctx(reinterpret_cast<Npp32f*>(d_div[j].ptr().data), d_div[j].ptr().dims.pitch,
                        aDst_arr, d_channelA[j].ptr().dims.pitch, up_size, nppcontext));
                }
                stopNPP_startFastNPP_CPU(benchmarkTemp, i);

                // FastNPP version
                const auto resizeIOp = 
                    fastNPP::ResizeBatch_8u32f_C3R_Advanced_Ctx<NPPI_INTER_LINEAR, BATCH>(UP_W, UP_H, hBatchSrc, hBatchROI);
                
                fastNPP::executeOperations(nppcontext, resizeIOp,
                                           fastNPP::SwapChannels_32f_C3R_Ctx(aDstOrder),
                                           fastNPP::MulC_32f_C3R_Ctx(mulValue),
                                           fastNPP::SubC_32f_C3R_Ctx(subValue),
                                           fastNPP::DivC_32f_C3R_Ctx(divValue),
                                           fastNPP::CopyBatch_32f_C3P3R_Ctx(d_dst_output, UP_W * sizeof(float), up_size));

                stopFastNPP_CPU(benchmarkTemp, i);
                if (warmup) break;
            }
            processExecution<BATCH, ITERS, batchValues.size(), batchValues>(benchmarkTemp.resF, __func__, benchmarkTemp.NPPelapsedTime, benchmarkTemp.FastNPPelapsedTime, VARIABLE_DIMENSION);
            gpuErrchk(cudaMemcpyAsync(h_tensor.ptr().data, d_tensor.ptr().data, d_tensor.sizeInBytes(), cudaMemcpyDeviceToHost, stream));
            
            // Bucle final de copia (NPP)
            for (int i = 0; i < BATCH; ++i) {
                const auto d_dims = d_channelA[i].dims();
                const auto h_dims = h_channelA[i].dims();

                gpuErrchk(cudaMemcpy2DAsync(h_channelA[i].ptr().data, h_dims.pitch, d_channelA[i].ptr().data, d_dims.pitch,
                    d_dims.width * sizeof(float), d_dims.height, cudaMemcpyDeviceToHost, stream));
                gpuErrchk(cudaMemcpy2DAsync(h_channelB[i].ptr().data, h_dims.pitch, d_channelB[i].ptr().data, d_dims.pitch,
                    d_dims.width * sizeof(float), d_dims.height, cudaMemcpyDeviceToHost, stream));
                gpuErrchk(cudaMemcpy2DAsync(h_channelC[i].ptr().data, h_dims.pitch, d_channelC[i].ptr().data, d_dims.pitch,
                    d_dims.width * sizeof(float), d_dims.height, cudaMemcpyDeviceToHost, stream));
            }

            gpuErrchk(cudaStreamSynchronize(stream));

            // free NPP Data
            gpuErrchk(cudaFree(dBatchSrc));
            gpuErrchk(cudaFree(dBatchDst));
            gpuErrchk(cudaFree(dBatchROI));
            gpuErrchk(cudaFreeHost(hBatchSrc));
            gpuErrchk(cudaFreeHost(hBatchDst));
            gpuErrchk(cudaFreeHost(hBatchROI));

            // compare data
            constexpr int COLOR_PLANE = UP_H * UP_W;
            constexpr int IMAGE_STRIDE = (COLOR_PLANE * 3);
            const float TOLERANCE = 1e-3;
            for (int j = 0; j < BATCH; ++j) {
                for (int c = 0; c < 3; ++c) {
                    for (int i = 0; i < COLOR_PLANE; ++i) {
                        const int i_tensor = i + (COLOR_PLANE * c);
                        const float result = h_tensor.ptr().data[(j * IMAGE_STRIDE) + i_tensor];
                        switch (c) {
                        case RED: {
                            float nppResult = h_channelA[j].ptr().data[i];
                            float diff = std::abs(result - nppResult);

                            passed &= diff < TOLERANCE;

                            break;
                        }
                        case GREEN: {
                            float nppResult = h_channelB[j].ptr().data[i];
                            float diff = std::abs(result - nppResult);
                            passed &= diff < TOLERANCE;
                            break;
                        }
                        case BLUE: {
                            float nppResult = h_channelC[j].ptr().data[i];
                            float diff = std::abs(result - nppResult);
                            passed &= diff < TOLERANCE;
                            break;
                        }
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            error_s << e.what();
            passed = false;
            exception = true;
            std::cerr << "Exception: " << error_s.str() << std::endl;
        }
    }

    return exception ? false : passed;
}

template <typename CV_TYPE_I, typename CV_TYPE_O, size_t... Is>
bool test_cpu_batchresize_x_split3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y,
                                    std::index_sequence<Is...> seq, fk::Stream& cv_stream, bool enabled) {
    bool passed = true;
    int dummy[] = { (passed &= test_NPP_cpu_batchresize_x_split3D<CV_TYPE_I, CV_TYPE_O, batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y,
                                                                                                cv_stream, enabled),
                    0)... };
    return passed;
}

#endif // ENABLE_BENCHMARK

int launch() {
#ifdef ENABLE_BENCHMARK
    constexpr size_t NUM_ELEMS_X = 3840;
    constexpr size_t NUM_ELEMS_Y = 2160;

    fk::Stream stream;

    std::unordered_map<std::string, bool> results;
    results["test_NPP_cpu_batchresize_x_split3D"] = true;

    const auto iSeq = std::make_index_sequence<batchValues.size()>{};

#define LAUNCH_TESTS                                                                             \
  results["test_NPP_cpu_batchresize_x_split3D"] &=                                                                             \
      test_cpu_batchresize_x_split3D<uchar3, float3>(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, stream, true);

    // Warming up for the benchmarks
    warmup = true;
    LAUNCH_TESTS
    warmup = false;

    LAUNCH_TESTS

#undef LAUNCH_TESTS

    stopCPUBenchmark();

    int returnValue = 0;
    for (const auto& [key, passed] : results) {
        if (passed) {
            std::cout << key << " passed!!" << std::endl;
        } else {
            std::cout << key << " failed!!" << std::endl;
            returnValue = -1;
        }
    }

    return returnValue;
#else
    return 0;
#endif // ENABLE_BENCHMARK
}