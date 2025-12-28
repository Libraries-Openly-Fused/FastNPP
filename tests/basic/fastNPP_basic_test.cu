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

#ifdef WIN32
#include <tests/main.h>
#endif

#include <fast_npp.h>

int launch() {
    constexpr int nMaxWidth = 1024;
    constexpr int nMaxHeight = 1024;
    constexpr int BATCH{ 50 };
    NppiImageDescriptor* h_pBatchSrc;
    NppiResizeBatchROI_Advanced* pBatchROI;
    int order[3]{ 2, 0, 1 };
    std::array<Npp32f*, BATCH> outputs[3];
    constexpr NppiSize outputSize{ 1024, 1024 };
    NppStreamContext streamContext;

    using ResizeType = decltype(fastNPP::ResizeBatch_8u32f_C3R_Advanced_Ctx<NPPI_INTER_LINEAR, BATCH>(std::declval<decltype(nMaxWidth)>(),
                                                                                                      std::declval<decltype(nMaxHeight)>(),
                                                                                                      std::declval<decltype(h_pBatchSrc)>(),
                                                                                                      std::declval<decltype(pBatchROI)>()));

    const ResizeType resize{};

    const auto swap = fastNPP::SwapChannels_32f_C3R_Ctx(order);

    const auto mul = fastNPP::MulC_32f_C3R_Ctx(float3{ 3.f, 4.f, 5.f });
    const auto sub = fastNPP::SubC_32f_C3R_Ctx(float3{ 6.f, 2.f, 1.f });
    const auto div = fastNPP::DivC_32f_C3R_Ctx(float3{ 4.f, 0.5f, 3.f });
    const auto write = fastNPP::CopyBatch_32f_C3P3R_Ctx(outputs, 1024*sizeof(float), outputSize);

    using ItCompiles = decltype(fastNPP::executeOperations(std::declval<decltype(streamContext)&>(),
                                        std::declval<decltype(resize)>(),
                                        std::declval<decltype(swap)>(),
                                        std::declval<decltype(mul)>(),
                                        std::declval<decltype(sub)>(),
                                        std::declval<decltype(div)>(),
                                        std::declval<decltype(write)>()));

    static_assert(std::is_same_v<ItCompiles, void>);

    return 0;
}