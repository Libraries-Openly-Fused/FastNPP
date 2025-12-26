/* Copyright 2024-2025 Oscar Amoros Huguet
   Copyright 2024 Albert Andaluz Gonzalez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

           http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tests/main.h>

#include <nppi_geometry_transforms.h>

#include <fused_kernel/core/data/size.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>
#include <fused_kernel/core/utils/parameter_pack_utils.h>
#include <fused_kernel/core/data/rect.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/fused_kernel.h>

#include <numeric>
#include <sstream>
#include <iostream>
#include <vector>
 
#include <tests/testsNppCommon.h>
#include <benchmarks/nppbenchmark.h>
#include <include/fast_npp.h>

constexpr char VARIABLE_DIMENSION[]{"Batch size"};
constexpr size_t NUM_EXPERIMENTS = 15;
constexpr size_t FIRST_VALUE = 10;
constexpr size_t INCREMENT = 10;
constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

template <int BATCH>
bool test_npp_batchresize_x_split3D(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, fk::Stream &compute_stream,
                                    bool enabled) {
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
      constexpr uchar3 init_val{1, 2, 3};
      fk::Ptr2D<uchar3> d_input(NUM_ELEMS_X, NUM_ELEMS_Y);
      fk::setTo(init_val, d_input, compute_stream);
      
      // Batches of crops (fk)
      std::array<fk::Ptr2D<uchar3>, BATCH> d_cropped;
      std::array<fk::Ptr2D<float3>, BATCH> d_cropped_FP32, d_resized_npp, d_swap, d_mul, d_sub, d_div;
      std::array<fk::Ptr2D<float>, BATCH> d_channelA, d_channelB, d_channelC;
      std::array<fk::Ptr2D<float>, BATCH> h_channelA, h_channelB, h_channelC;

      // Batches of crops (npp)
      NppiImageDescriptor *hBatchSrc = nullptr, *dBatchSrc = nullptr, *hBatchSrcFP32 = nullptr, *dBatchSrcFP32 = nullptr, *hBatchDst = nullptr, *dBatchDst = nullptr;
      NppiResizeBatchROI_Advanced *dBatchROI = nullptr, *hBatchROI = nullptr;

      // Init NPP context
      NppStreamContext nppcontext = initNppStreamContext(compute_stream);

      // Initialize function parameters
      constexpr Npp32f mulValue[3] = {alpha, alpha, alpha};
      constexpr Npp32f subValue[3] = {1.f, 4.f, 3.2f};
      constexpr Npp32f divValue[3] = {1.f, 4.f, 3.2f};
      constexpr NppiSize crop_size{CROP_W, CROP_H};
      constexpr NppiSize up_size{UP_W, UP_H};

      fk::Tensor<float> h_tensor(UP_W, UP_H, BATCH, 3, fk::MemType::HostPinned);
      fk::Tensor<float> d_tensor(UP_W, UP_H, BATCH, 3);
      std::array<Npp32f*, BATCH> d_dst_output[3];
      for (int planeId = 0; planeId < 3; ++planeId) {
          for (int batchId = 0; batchId < BATCH; ++batchId) {
              uint plane_padding = d_tensor.ptr().dims.plane_pitch * planeId;
              d_dst_output[planeId][batchId] =
                  reinterpret_cast<Npp32f*>(
                    reinterpret_cast<uchar*>(
                      fk::PtrAccessor<fk::ND::_3D>::point(fk::Point(0,0,batchId), d_tensor.ptr())
                    ) + plane_padding
                  );
          }
      }
      // crop array of images using batch resize+ ROIs
      // force fill 5
      constexpr int COLOR_PLANE = UP_H * UP_W;
      constexpr int IMAGE_STRIDE = (COLOR_PLANE * 3);
      // asume RGB->BGR
      const int aDstOrder[3] = {BLUE, GREEN, RED};
      gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&hBatchSrc), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&hBatchSrcFP32), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&hBatchDst), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&hBatchROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));
      gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&dBatchSrc), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&dBatchSrcFP32), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&dBatchDst), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&dBatchROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));

      std::vector<std::array<Npp32f *, 3>> aDst;
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

        hBatchROI[i].oSrcRectROI = NppiRect{0, 0, CROP_W, CROP_H};
        hBatchROI[i].oDstRectROI = NppiRect{0, 0, UP_W, UP_H};

        // Allocate pointers for split images (device)
        d_channelA[i] = fk::Ptr2D<float>(UP_W, UP_H);
        d_channelB[i] = fk::Ptr2D<float>(UP_W, UP_H);
        d_channelC[i] = fk::Ptr2D<float>(UP_W, UP_H);
        // Allocate pointers for split images (host)
        h_channelA[i] = fk::Ptr2D<float>(UP_W, UP_H, 0, fk::MemType::HostPinned);
        h_channelB[i] = fk::Ptr2D<float>(UP_W, UP_H, 0, fk::MemType::HostPinned);
        h_channelC[i] = fk::Ptr2D<float>(UP_W, UP_H, 0, fk::MemType::HostPinned);

        std::array<Npp32f*, 3> ptrs = {reinterpret_cast<Npp32f *>(d_channelA[i].ptr().data),
                                        reinterpret_cast<Npp32f *>(d_channelB[i].ptr().data),
                                        reinterpret_cast<Npp32f *>(d_channelC[i].ptr().data)};
        aDst.push_back(ptrs);
      }

      gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void **>(dBatchSrcFP32), hBatchSrcFP32, sizeof(NppiImageDescriptor) * BATCH,
                                cudaMemcpyHostToDevice, compute_stream));
      gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void **>(dBatchSrc), hBatchSrc, sizeof(NppiImageDescriptor) * BATCH,
                                cudaMemcpyHostToDevice, compute_stream));
      gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void **>(dBatchDst), hBatchDst, sizeof(NppiImageDescriptor) * BATCH,
                                cudaMemcpyHostToDevice, compute_stream));
      gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void **>(dBatchROI), hBatchROI, sizeof(NppiResizeBatchROI_Advanced) * BATCH,
                                cudaMemcpyHostToDevice, compute_stream));

      std::array<fk::RawPtr<fk::ND::_2D, uchar3>, BATCH> d_crop_fk;
      for (int i = 0; i < BATCH; i++) {
        d_crop_fk[i] = d_input.crop2D(fk::Point(i, i), fk::PtrDims<fk::ND::_2D>(CROP_W, CROP_H));
      }

      START_NPP_BENCHMARK

      // NPP version
      // convert to 32f
      // print2D("Values after initialization", d_input, compute_stream);

      for (int i = 0; i < BATCH; ++i) {
        NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(reinterpret_cast<const Npp8u*>(hBatchSrc[i].pData), hBatchSrc[i].nStep, reinterpret_cast<Npp32f*>(hBatchSrcFP32[i].pData),
                                            hBatchSrcFP32[i].nStep, hBatchSrcFP32[i].oSize, nppcontext));
      }

      NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(UP_W, UP_H, dBatchSrcFP32, dBatchDst, dBatchROI, BATCH,
                                                     NPPI_INTER_LINEAR, nppcontext));

      for (int i = 0; i < BATCH; ++i) {
        NPP_CHECK(nppiSwapChannels_32f_C3R_Ctx(
            reinterpret_cast<Npp32f *>(d_resized_npp[i].ptr().data), d_resized_npp[i].ptr().dims.pitch,
            reinterpret_cast<Npp32f *>(d_swap[i].ptr().data), d_swap[i].dims().pitch, up_size, aDstOrder, nppcontext));

        NPP_CHECK(nppiMulC_32f_C3R_Ctx(reinterpret_cast<Npp32f *>(d_swap[i].ptr().data), d_swap[i].ptr().dims.pitch,
                                       mulValue, reinterpret_cast<Npp32f *>(d_mul[i].ptr().data),
                                       d_mul[i].ptr().dims.pitch, up_size, nppcontext));

        NPP_CHECK(nppiSubC_32f_C3R_Ctx(reinterpret_cast<Npp32f *>(d_mul[i].ptr().data), d_mul[i].ptr().dims.pitch,
                                       subValue, reinterpret_cast<Npp32f *>(d_sub[i].ptr().data),
                                       d_sub[i].ptr().dims.pitch, up_size, nppcontext));

        NPP_CHECK(nppiDivC_32f_C3R_Ctx(reinterpret_cast<Npp32f *>(d_sub[i].ptr().data), d_sub[i].ptr().dims.pitch,
                                       divValue, reinterpret_cast<Npp32f *>(d_div[i].ptr().data),
                                       d_div[i].ptr().dims.pitch, up_size, nppcontext));

        Npp32f *const aDst_arr[3] = {reinterpret_cast<Npp32f *>(d_channelA[i].ptr().data),
                                     reinterpret_cast<Npp32f *>(d_channelB[i].ptr().data),
                                     reinterpret_cast<Npp32f *>(d_channelC[i].ptr().data)};

        NPP_CHECK(nppiCopy_32f_C3P3R_Ctx(reinterpret_cast<Npp32f *>(d_div[i].ptr().data), d_div[i].ptr().dims.pitch,
                                         aDst_arr, d_channelA[i].ptr().dims.pitch, up_size, nppcontext));
      }

      STOP_NPP_START_FK_BENCHMARK
      // do the same via fastNPP
      const auto resizeIOp = 
        fastNPP::ResizeBatch_8u32f_C3R_Advanced_Ctx<NPPI_INTER_LINEAR, BATCH>(UP_W, UP_H, hBatchSrc, hBatchROI);
                
      fastNPP::executeOperations(nppcontext, resizeIOp,
                                 fastNPP::SwapChannels_32f_C3R_Ctx(aDstOrder),
                                 fastNPP::MulC_32f_C3R_Ctx(mulValue),
                                 fastNPP::SubC_32f_C3R_Ctx(subValue),
                                 fastNPP::DivC_32f_C3R_Ctx(divValue),
                                 fastNPP::CopyBatch_32f_C3P3R_Ctx(d_dst_output, UP_W * sizeof(float), up_size));
      STOP_FK_BENCHMARK
      // copy tensor
      gpuErrchk(cudaMemcpyAsync(h_tensor.ptr().data, d_tensor.ptr().data, h_tensor.sizeInBytes(),
                                cudaMemcpyDeviceToHost, compute_stream));

      // Bucle final de copia (NPP)
      for (int i = 0; i < BATCH; ++i) {
        const auto d_dims = d_channelA[i].dims();
        const auto h_dims = h_channelA[i].dims();

        gpuErrchk(cudaMemcpy2DAsync(h_channelA[i].ptr().data, h_dims.pitch, d_channelA[i].ptr().data, d_dims.pitch,
                                    d_dims.width * sizeof(float), d_dims.height, cudaMemcpyDeviceToHost,
                                    compute_stream));
        gpuErrchk(cudaMemcpy2DAsync(h_channelB[i].ptr().data, h_dims.pitch, d_channelB[i].ptr().data, d_dims.pitch,
                                    d_dims.width * sizeof(float), d_dims.height, cudaMemcpyDeviceToHost,
                                    compute_stream));
        gpuErrchk(cudaMemcpy2DAsync(h_channelC[i].ptr().data, h_dims.pitch, d_channelC[i].ptr().data, d_dims.pitch,
                                    d_dims.width * sizeof(float), d_dims.height, cudaMemcpyDeviceToHost,
                                    compute_stream));
      }

      gpuErrchk(cudaStreamSynchronize(compute_stream));

      // free NPP Data
      gpuErrchk(cudaFree(dBatchSrc));
      gpuErrchk(cudaFree(dBatchDst));
      gpuErrchk(cudaFree(dBatchROI));
      gpuErrchk(cudaFreeHost(hBatchSrc));
      gpuErrchk(cudaFreeHost(hBatchDst));
      gpuErrchk(cudaFreeHost(hBatchROI));

      // compare data
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

      if (!passed) {
        std::cout << "BATCH size = " << BATCH << std::endl;
        std::cout << "Results fk:" << std::endl;
        for (int b = 0; b < BATCH; ++b) {

          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float value = *fk::PtrAccessor<fk::ND::_3D>::cr_point(fk::Point(x, y, b), h_tensor.ptr());
              std::cout << " " << value;
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float *const image = fk::PtrAccessor<fk::ND::_3D>::cr_point(fk::Point(x, y, b), h_tensor.ptr());
              const float pixel = *(image + h_tensor.dims().plane_pitch);
              std::cout << " " << pixel;
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float *const image = fk::PtrAccessor<fk::ND::_3D>::cr_point(fk::Point(x, y, b), h_tensor.ptr());
              const float pixel = *(image + (h_tensor.dims().plane_pitch * 2));
              std::cout << " " << pixel;
            }
            std::cout << std::endl;
          }
        }

        std::cout << "Results npp:" << std::endl;
        for (int b = 0; b < BATCH; ++b) {

          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float value = *fk::PtrAccessor<fk::ND::_2D>::cr_point(fk::Point(x, y), h_channelA[b].ptr());
              std::cout << " " << value;
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float value = *fk::PtrAccessor<fk::ND::_2D>::cr_point(fk::Point(x, y), h_channelB[b].ptr());
              std::cout << " " << value;
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float value = *fk::PtrAccessor<fk::ND::_2D>::cr_point(fk::Point(x, y), h_channelC[b].ptr());
              std::cout << " " << value;
            }
            std::cout << std::endl;
          }
        }
      }
    }

    catch (const std::exception &e) {
      error_s << e.what();
      passed = false;
      exception = true;
    }

    if (!passed) {
      if (!exception) {
        std::cout << "Test failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
      } else {
        std::cout << "Test failed!! EXCEPTION: " << error_s.str() << std::endl;
      }
    }

    return passed;
  }
  return passed;
}

template <size_t... Is>
bool launch_test_npp_batchresize_x_split3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y,
                                           std::index_sequence<Is...> seq, fk::Stream compute_stream, bool enabled) {
  bool passed = true;
  int dummy[] = {
      (passed &= test_npp_batchresize_x_split3D<batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, compute_stream, enabled),
       0)...};
  (void)dummy;

  return passed;
}

template <int BATCH>
bool test_npp_batchresize_x_split3D_No_CPU_Overhead(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, fk::Stream &compute_stream,
                                    bool enabled) {
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
      constexpr uchar3 init_val{1, 2, 3};
      fk::Ptr2D<uchar3> d_input(NUM_ELEMS_X, NUM_ELEMS_Y);
      fk::setTo(init_val, d_input, compute_stream);
      
      // Batches of crops (fk)
      std::array<fk::Ptr2D<uchar3>, BATCH> d_cropped;
      std::array<fk::Ptr2D<float3>, BATCH> d_cropped_FP32, d_resized_npp, d_swap, d_mul, d_sub, d_div;
      std::array<fk::Ptr2D<float>, BATCH> d_channelA, d_channelB, d_channelC;
      std::array<fk::Ptr2D<float>, BATCH> h_channelA, h_channelB, h_channelC;

      // Batches of crops (npp)
      NppiImageDescriptor *hBatchSrc = nullptr, *dBatchSrc = nullptr, *hBatchSrcFP32 = nullptr, *dBatchSrcFP32 = nullptr, *hBatchDst = nullptr, *dBatchDst = nullptr;
      NppiResizeBatchROI_Advanced *dBatchROI = nullptr, *hBatchROI = nullptr;

      // Init NPP context
      NppStreamContext nppcontext = initNppStreamContext(compute_stream);

      // Initialize function parameters
      constexpr Npp32f mulValue[3] = {alpha, alpha, alpha};
      constexpr Npp32f subValue[3] = {1.f, 4.f, 3.2f};
      constexpr Npp32f divValue[3] = {1.f, 4.f, 3.2f};
      constexpr NppiSize crop_size{CROP_W, CROP_H};
      constexpr NppiSize up_size{UP_W, UP_H};

      fk::Tensor<float> h_tensor(UP_W, UP_H, BATCH, 3, fk::MemType::HostPinned);
      fk::Tensor<float> d_tensor(UP_W, UP_H, BATCH, 3);
      std::array<Npp32f*, BATCH> d_dst_output[3];
      for (int planeId = 0; planeId < 3; ++planeId) {
          for (int batchId = 0; batchId < BATCH; ++batchId) {
              uint plane_padding = d_tensor.ptr().dims.plane_pitch * planeId;
              d_dst_output[planeId][batchId] =
                  reinterpret_cast<Npp32f*>(
                    reinterpret_cast<uchar*>(
                      fk::PtrAccessor<fk::ND::_3D>::point(fk::Point(0,0,batchId), d_tensor.ptr())
                    ) + plane_padding
                  );
          }
      }
      // crop array of images using batch resize+ ROIs
      // force fill 5
      constexpr int COLOR_PLANE = UP_H * UP_W;
      constexpr int IMAGE_STRIDE = (COLOR_PLANE * 3);
      // asume RGB->BGR
      const int aDstOrder[3] = {BLUE, GREEN, RED};
      gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&hBatchSrc), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&hBatchSrcFP32), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&hBatchDst), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&hBatchROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));
      gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&dBatchSrc), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&dBatchSrcFP32), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&dBatchDst), sizeof(NppiImageDescriptor) * BATCH));
      gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&dBatchROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));

      std::vector<std::array<Npp32f *, 3>> aDst;
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

        hBatchROI[i].oSrcRectROI = NppiRect{0, 0, CROP_W, CROP_H};
        hBatchROI[i].oDstRectROI = NppiRect{0, 0, UP_W, UP_H};

        // Allocate pointers for split images (device)
        d_channelA[i] = fk::Ptr2D<float>(UP_W, UP_H);
        d_channelB[i] = fk::Ptr2D<float>(UP_W, UP_H);
        d_channelC[i] = fk::Ptr2D<float>(UP_W, UP_H);
        // Allocate pointers for split images (host)
        h_channelA[i] = fk::Ptr2D<float>(UP_W, UP_H, 0, fk::MemType::HostPinned);
        h_channelB[i] = fk::Ptr2D<float>(UP_W, UP_H, 0, fk::MemType::HostPinned);
        h_channelC[i] = fk::Ptr2D<float>(UP_W, UP_H, 0, fk::MemType::HostPinned);

        std::array<Npp32f*, 3> ptrs = {reinterpret_cast<Npp32f *>(d_channelA[i].ptr().data),
                                        reinterpret_cast<Npp32f *>(d_channelB[i].ptr().data),
                                        reinterpret_cast<Npp32f *>(d_channelC[i].ptr().data)};
        aDst.push_back(ptrs);
      }

      gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void **>(dBatchSrcFP32), hBatchSrcFP32, sizeof(NppiImageDescriptor) * BATCH,
                                cudaMemcpyHostToDevice, compute_stream));
      gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void **>(dBatchSrc), hBatchSrc, sizeof(NppiImageDescriptor) * BATCH,
                                cudaMemcpyHostToDevice, compute_stream));
      gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void **>(dBatchDst), hBatchDst, sizeof(NppiImageDescriptor) * BATCH,
                                cudaMemcpyHostToDevice, compute_stream));
      gpuErrchk(cudaMemcpyAsync(reinterpret_cast<void **>(dBatchROI), hBatchROI, sizeof(NppiResizeBatchROI_Advanced) * BATCH,
                                cudaMemcpyHostToDevice, compute_stream));

      std::array<fk::RawPtr<fk::ND::_2D, uchar3>, BATCH> d_crop_fk;
      for (int i = 0; i < BATCH; i++) {
        d_crop_fk[i] = d_input.crop2D(fk::Point(i, i), fk::PtrDims<fk::ND::_2D>(CROP_W, CROP_H));
      }

      // Precompute FastNPP operations for faster execution
      const auto resizeIOp = 
        fastNPP::ResizeBatch_8u32f_C3R_Advanced_Ctx<NPPI_INTER_LINEAR, BATCH>(UP_W, UP_H, hBatchSrc, hBatchROI);
      const auto swap = fastNPP::SwapChannels_32f_C3R_Ctx(aDstOrder);
      const auto mul = fastNPP::MulC_32f_C3R_Ctx(mulValue);
      const auto sub = fastNPP::SubC_32f_C3R_Ctx(subValue);
      const auto div = fastNPP::DivC_32f_C3R_Ctx(divValue);
      const auto copy_batch = fastNPP::CopyBatch_32f_C3P3R_Ctx(d_dst_output, UP_W * sizeof(float), up_size);

      START_NPP_BENCHMARK

      // NPP version
      // convert to 32f
      // print2D("Values after initialization", d_input, compute_stream);

      for (int i = 0; i < BATCH; ++i) {
        NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(reinterpret_cast<const Npp8u*>(hBatchSrc[i].pData), hBatchSrc[i].nStep, reinterpret_cast<Npp32f*>(hBatchSrcFP32[i].pData),
                                            hBatchSrcFP32[i].nStep, hBatchSrcFP32[i].oSize, nppcontext));
      }

      NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(UP_W, UP_H, dBatchSrcFP32, dBatchDst, dBatchROI, BATCH,
                                                     NPPI_INTER_LINEAR, nppcontext));

      for (int i = 0; i < BATCH; ++i) {
        NPP_CHECK(nppiSwapChannels_32f_C3R_Ctx(
            reinterpret_cast<Npp32f *>(d_resized_npp[i].ptr().data), d_resized_npp[i].ptr().dims.pitch,
            reinterpret_cast<Npp32f *>(d_swap[i].ptr().data), d_swap[i].dims().pitch, up_size, aDstOrder, nppcontext));

        NPP_CHECK(nppiMulC_32f_C3R_Ctx(reinterpret_cast<Npp32f *>(d_swap[i].ptr().data), d_swap[i].ptr().dims.pitch,
                                       mulValue, reinterpret_cast<Npp32f *>(d_mul[i].ptr().data),
                                       d_mul[i].ptr().dims.pitch, up_size, nppcontext));

        NPP_CHECK(nppiSubC_32f_C3R_Ctx(reinterpret_cast<Npp32f *>(d_mul[i].ptr().data), d_mul[i].ptr().dims.pitch,
                                       subValue, reinterpret_cast<Npp32f *>(d_sub[i].ptr().data),
                                       d_sub[i].ptr().dims.pitch, up_size, nppcontext));

        NPP_CHECK(nppiDivC_32f_C3R_Ctx(reinterpret_cast<Npp32f *>(d_sub[i].ptr().data), d_sub[i].ptr().dims.pitch,
                                       divValue, reinterpret_cast<Npp32f *>(d_div[i].ptr().data),
                                       d_div[i].ptr().dims.pitch, up_size, nppcontext));

        Npp32f *const aDst_arr[3] = {reinterpret_cast<Npp32f *>(d_channelA[i].ptr().data),
                                     reinterpret_cast<Npp32f *>(d_channelB[i].ptr().data),
                                     reinterpret_cast<Npp32f *>(d_channelC[i].ptr().data)};

        NPP_CHECK(nppiCopy_32f_C3P3R_Ctx(reinterpret_cast<Npp32f *>(d_div[i].ptr().data), d_div[i].ptr().dims.pitch,
                                         aDst_arr, d_channelA[i].ptr().dims.pitch, up_size, nppcontext));
      }

      STOP_NPP_START_FK_BENCHMARK
      // do the same via fastNPP
      fastNPP::executeOperations(nppcontext, resizeIOp, swap, mul, sub, div, copy_batch);
      STOP_FK_BENCHMARK
      // copy tensor
      gpuErrchk(cudaMemcpyAsync(h_tensor.ptr().data, d_tensor.ptr().data, h_tensor.sizeInBytes(),
                                cudaMemcpyDeviceToHost, compute_stream));

      // Bucle final de copia (NPP)
      for (int i = 0; i < BATCH; ++i) {
        const auto d_dims = d_channelA[i].dims();
        const auto h_dims = h_channelA[i].dims();

        gpuErrchk(cudaMemcpy2DAsync(h_channelA[i].ptr().data, h_dims.pitch, d_channelA[i].ptr().data, d_dims.pitch,
                                    d_dims.width * sizeof(float), d_dims.height, cudaMemcpyDeviceToHost,
                                    compute_stream));
        gpuErrchk(cudaMemcpy2DAsync(h_channelB[i].ptr().data, h_dims.pitch, d_channelB[i].ptr().data, d_dims.pitch,
                                    d_dims.width * sizeof(float), d_dims.height, cudaMemcpyDeviceToHost,
                                    compute_stream));
        gpuErrchk(cudaMemcpy2DAsync(h_channelC[i].ptr().data, h_dims.pitch, d_channelC[i].ptr().data, d_dims.pitch,
                                    d_dims.width * sizeof(float), d_dims.height, cudaMemcpyDeviceToHost,
                                    compute_stream));
      }

      gpuErrchk(cudaStreamSynchronize(compute_stream));

      // free NPP Data
      gpuErrchk(cudaFree(dBatchSrc));
      gpuErrchk(cudaFree(dBatchDst));
      gpuErrchk(cudaFree(dBatchROI));
      gpuErrchk(cudaFreeHost(hBatchSrc));
      gpuErrchk(cudaFreeHost(hBatchDst));
      gpuErrchk(cudaFreeHost(hBatchROI));

      // compare data
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

      if (!passed) {
        std::cout << "BATCH size = " << BATCH << std::endl;
        std::cout << "Results fk:" << std::endl;
        for (int b = 0; b < BATCH; ++b) {

          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float value = *fk::PtrAccessor<fk::ND::_3D>::cr_point(fk::Point(x, y, b), h_tensor.ptr());
              std::cout << " " << value;
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float *const image = fk::PtrAccessor<fk::ND::_3D>::cr_point(fk::Point(x, y, b), h_tensor.ptr());
              const float pixel = *(image + h_tensor.dims().plane_pitch);
              std::cout << " " << pixel;
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float *const image = fk::PtrAccessor<fk::ND::_3D>::cr_point(fk::Point(x, y, b), h_tensor.ptr());
              const float pixel = *(image + (h_tensor.dims().plane_pitch * 2));
              std::cout << " " << pixel;
            }
            std::cout << std::endl;
          }
        }

        std::cout << "Results npp:" << std::endl;
        for (int b = 0; b < BATCH; ++b) {

          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float value = *fk::PtrAccessor<fk::ND::_2D>::cr_point(fk::Point(x, y), h_channelA[b].ptr());
              std::cout << " " << value;
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float value = *fk::PtrAccessor<fk::ND::_2D>::cr_point(fk::Point(x, y), h_channelB[b].ptr());
              std::cout << " " << value;
            }
            std::cout << std::endl;
          }
          std::cout << std::endl;
          for (int y = 0; y < UP_H; y++) {
            for (int x = 0; x < UP_W; x++) {
              const float value = *fk::PtrAccessor<fk::ND::_2D>::cr_point(fk::Point(x, y), h_channelC[b].ptr());
              std::cout << " " << value;
            }
            std::cout << std::endl;
          }
        }
      }
    }

    catch (const std::exception &e) {
      error_s << e.what();
      passed = false;
      exception = true;
    }

    if (!passed) {
      if (!exception) {
        std::cout << "Test failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
      } else {
        std::cout << "Test failed!! EXCEPTION: " << error_s.str() << std::endl;
      }
    }

    return passed;
  }
  return passed;
}

// test_npp_batchresize_x_split3D_No_CPU_Overhead
template <size_t... Is>
bool launch_test_npp_batchresize_x_split3D_No_CPU_Overhead(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y,
                                           std::index_sequence<Is...> seq, fk::Stream compute_stream, bool enabled) {
  bool passed = true;
  int dummy[] = {
      (passed &= test_npp_batchresize_x_split3D_No_CPU_Overhead<batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, compute_stream, enabled),
       0)...};
  (void)dummy;

  return passed;
}


int launch() {
  constexpr size_t NUM_ELEMS_X = 3840;
  constexpr size_t NUM_ELEMS_Y = 2160;

  fk::Stream stream;

  std::unordered_map<std::string, bool> results;
  results["test_npp_batchresize_x_split3D"] = true;
  results["test_npp_batchresize_x_split3D_No_CPU_Overhead"] = true;
  std::make_index_sequence<batchValues.size()> iSeq{};

  // Warmup
  warmup = true;
  launch_test_npp_batchresize_x_split3D(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, stream, true);
  launch_test_npp_batchresize_x_split3D_No_CPU_Overhead(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, stream, true);
  warmup = false;

  results["test_npp_batchresize_x_split3D"] &=
      launch_test_npp_batchresize_x_split3D(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, stream, true);
  results["test_npp_batchresize_x_split3D_No_CPU_Overhead"] &=
      launch_test_npp_batchresize_x_split3D_No_CPU_Overhead(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, stream, true);
  CLOSE_BENCHMARK
  int returnValue = 0;
  for (const auto &[key, passed] : results) {
    if (passed) {
      std::cout << key << " passed!!" << std::endl;
    } else {
      std::cout << key << " failed!!" << std::endl;
      returnValue = -1;
    }
  }
  return returnValue;
}