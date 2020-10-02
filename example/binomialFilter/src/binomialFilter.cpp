/* Copyright 2020 Benjamin Worpitz, Sergei Bastrakov, Jakob Krude
 *
 * This file exemplifies usage of alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */


#include "../../../test/common/include/alpaka/test/MeasureKernelRunTime.hpp"
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/alpaka.hpp>
#include <limits>>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <iostream>


//-----------------------------------------------------------------------------
//! Helper method mapping the input 2D input into 1D.
template<typename TIdx, typename TVec> ALPAKA_FN_ACC auto
mapIdx(TIdx i, TIdx j, TVec const & extent) -> const TIdx
{
    using Dim1 = alpaka::dim::DimInt<1u>;
    return alpaka::idx::mapIdx<Dim1::value>(TVec(i, j), extent)[0];
}

//-----------------------------------------------------------------------------
//! Helper method comparing two floats.
// Based on: https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
ALPAKA_FN_HOST auto almostEqual(float x, float y, int ulp = 5) -> bool
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x - y) <= std::numeric_limits<float>::epsilon() * std::fabs(x + y) * ulp
           // unless the result is subnormal
           || std::fabs(x - y) < std::numeric_limits<float>::min();
}


//#############################################################################
//! The kernel executing the parallel logic (naive version).
//! Each thread computes one result for a binomial filter.
struct Kernel
{
    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param inputBuf The buffer representing the matrix (global memory).
    //! \param resultBuf The buffer containing all computed results.
    template<typename TAcc> ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        float const * const inputBuf,
        float * const resultBuf) const -> void
    {
        auto const globalThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const extent2D = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        // Configuration check.
        using Dim = alpaka::dim::DimInt<2u>;
        static_assert(alpaka::dim::Dim<TAcc>::value == Dim::value,
            "This example is designed for 2D");

        auto i = globalThreadIdx[0];
        auto j = globalThreadIdx[1];
        // Is the current point on a "edge" of the matrix.
        if(i == 0 || i >= extent2D[0] - 1 || j == 0 || j >= extent2D[1] - 1)
        {
            resultBuf[mapIdx(i, j, extent2D)] = inputBuf[mapIdx(i, j, extent2D)];
        }else
        {
            // Compute binomialFilter
            resultBuf[mapIdx(i, j, extent2D)] =
                (0.0625f) * (// Normalisation = 1/16
                1.0f * inputBuf[mapIdx(i - 1, j - 1, extent2D)] +
                2.0f * inputBuf[mapIdx(i - 1, j + 0, extent2D)] +
                1.0f * inputBuf[mapIdx(i - 1, j + 1, extent2D)] +
                2.0f * inputBuf[mapIdx(i + 0, j - 1, extent2D)] +
                4.0f * inputBuf[mapIdx(i + 0, j + 0, extent2D)] +
                2.0f * inputBuf[mapIdx(i + 0, j + 1, extent2D)] +
                1.0f * inputBuf[mapIdx(i + 1, j - 1, extent2D)] +
                2.0f * inputBuf[mapIdx(i + 1, j + 0, extent2D)] +
                1.0f * inputBuf[mapIdx(i + 1, j + 1, extent2D)]);
        }

    }

};
//#############################################################################
//! The kernel executing the parallel logic (using shared memory).
//! Based on the naive version, except using shared memory.
struct SharedMemKernel
{
    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param inputBuf The buffer representing the matrix (global memory).
    //! \param resultBuf The buffer containing all computed results.
    template<typename TAcc> ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        float const * const inputBuf,
        float * const resultBuf) const -> void
    {
        /*
         * (n-2)^2 unique elements per block | 4n -4 overlapping elements
         * blocks are adjacent but overlapping one row/column on each side
         * the mapping of input-matrix to threads is 1:1
         *  Example:
         *   The inner cube represents one block NxM and each element one thread
         *   But for each result every of the eight neighbours is needed
         *   therefore each block has a memory size of (N+2)x(M+2)
         *   In this case the thread with the global index 7 would copy:
         *      element 1, 2, 6 and 7 into shared memory
         *
         *  1    2  3  4   5
         *    ------------
         *  6 |  7  8  9 | 10
         * 11 | 12 13 14 | 15
         * 16 | 17 18 19 | 20
         *    ------------
         * 21   22 23 24   25
         *
         */
        using Idx = alpaka::idx::Idx<TAcc>;
        using Dim = alpaka::dim::Dim<TAcc>;
        static_assert(Dim::value == 2u, "The accelerator used has to be 2 dimensional!");
        auto const gridThreadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalExtent = alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const blockThreadIdx = alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const blockExtent = alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

        Idx const globalIdxX = gridThreadIdx[0];
        Idx const globalIdxY = gridThreadIdx[1];
        Idx const globalIdx1D(mapIdx(globalIdxX, globalIdxY, globalExtent));

        Idx const blockExtentX = blockExtent[0];
        Idx const blockExtentY = blockExtent[1];
        // respectively to the memory size every thread-position is shifted
        Idx const blockThreadIdxX = blockThreadIdx[0];
        Idx const blockThreadIdxY = blockThreadIdx[1];
        // the memory size needs one more row/column on each side
        Idx const memoryExtentBlockX = blockExtentX + Idx(2);
        Idx const memoryExtentBlockY = blockExtentY + Idx(2);
        auto memoryExtentBlock = blockExtent + alpaka::vec::Vec<Dim, Idx>::all(2);

        // Get a pointer to shared memory.
        auto * const pBlockShared(alpaka::block::shared::dyn::getMem<float>(acc));
        // 1D index inside (N+2) x (M+2) "extended" block
        auto sharedMemIdx1D(mapIdx(blockThreadIdxX, blockThreadIdxY, memoryExtentBlock));

        for(Idx i(0); i + blockThreadIdxX < memoryExtentBlockX; i += blockExtentX)
        {
            for(Idx j(0); j + blockThreadIdxY < memoryExtentBlockY; j += blockExtentY)
            {
                Idx globalIdx = mapIdx(globalIdxX + i, globalIdxY + i, globalExtent);
                if(globalIdx < globalExtent.prod())
                {
                    pBlockShared[mapIdx(
                        blockThreadIdxX + i,
                        blockThreadIdxY + j,
                        memoryExtentBlock)] = inputBuf[globalIdx];
                }
            }
        }

        // Wait for all threads in this block.
        alpaka::block::sync::syncBlockThreads(acc);

        Idx const shiftedBlockThreadIdxX = blockThreadIdx[0] + 1;
        Idx const shiftedBlockThreadIdxY = blockThreadIdx[1] + 1;

        // If the global idx is on the edge of the inputMatrix: Only copy input.
        if(globalIdxX == 0 ||
           globalIdxX >= globalExtent[0] - 1 ||
           globalIdxY == 0 ||
           globalIdxY >= globalExtent[1] - 1)
        {
            resultBuf[globalIdx1D] = pBlockShared[sharedMemIdx1D];
        }else
        {
            resultBuf[globalIdx1D] = (0.0625f) * (// Normalisation = 1/16
                // Top row
                1.0f * pBlockShared[mapIdx(shiftedBlockThreadIdxX - 1, shiftedBlockThreadIdxY - 1, memoryExtentBlock)] +
                2.0f * pBlockShared[mapIdx(shiftedBlockThreadIdxX - 1, shiftedBlockThreadIdxY + 0, memoryExtentBlock)] +
                1.0f * pBlockShared[mapIdx(shiftedBlockThreadIdxX - 1, shiftedBlockThreadIdxY + 1, memoryExtentBlock)] +
                // Middle row
                2.0f * pBlockShared[mapIdx(shiftedBlockThreadIdxX + 0, shiftedBlockThreadIdxY - 1, memoryExtentBlock)] +
                4.0f * pBlockShared[mapIdx(shiftedBlockThreadIdxX + 0, shiftedBlockThreadIdxY + 0, memoryExtentBlock)] +
                2.0f * pBlockShared[mapIdx(shiftedBlockThreadIdxX + 0, shiftedBlockThreadIdxY + 1, memoryExtentBlock)] +
                // Bottom row
                1.0f * pBlockShared[mapIdx(shiftedBlockThreadIdxX + 1, shiftedBlockThreadIdxY - 1, memoryExtentBlock)] +
                2.0f * pBlockShared[mapIdx(shiftedBlockThreadIdxX + 1, shiftedBlockThreadIdxY + 0, memoryExtentBlock)] +
                1.0f * pBlockShared[mapIdx(shiftedBlockThreadIdxX + 1, shiftedBlockThreadIdxY + 1, memoryExtentBlock)]);
        }
    }
};

namespace alpaka
{
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The trait for getting the size of the block shared dynamic memory for a kernel.
            template<typename TAcc> struct BlockSharedMemDynSizeBytes<SharedMemKernel, TAcc>
            {
                //-----------------------------------------------------------------------------
                //! \return The size of the shared memory allocated for a block.
                template<typename TVec> ALPAKA_FN_HOST_ACC static auto
                getBlockSharedMemDynSizeBytes(SharedMemKernel const & sharedMemKernel,
                    TVec const & blockThreadExtent,
                    TVec const & threadElemExtent,
                    float const * const inputBuf,
                    float * const resultBuf)
                {
                    alpaka::ignore_unused(inputBuf);
                    alpaka::ignore_unused(resultBuf);

                    // Reserve the buffer for the memory of one block:
                    // On each side of the block an extra row/column is needed (therefore +2).
                    return static_cast<std::size_t>((blockThreadExtent[0] + 2) *
                                                    (blockThreadExtent[1] + 2) *
                                                    sizeof(float));
                }
            };
        }
    }
}


auto main() -> int
{
    using Dim = alpaka::dim::DimInt<2u>;
    using Idx = uint32_t;
    using Acc = alpaka::example::ExampleDefaultAcc<Dim, Idx>;
    using Vec = alpaka::vec::Vec<Dim, Idx>;
    using DevHost = alpaka::dev::DevCpu;
    auto const devAcc = alpaka::pltf::getDevByIdx<Acc>(0u);
    auto const devHost = alpaka::pltf::getDevByIdx<DevHost>(0u);

    // Problem configuration.
    const Idx extentX = 5000;
    const Idx extentY = 7000;
    const Vec extentMatrix2D(extentX, extentY);
    const Vec alpakaElementsPerThread(Vec::ones());

    // Setup.
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Idx>;
    auto workdiv = WorkDiv{
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extentMatrix2D,
            alpakaElementsPerThread,
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::EqualExtent)};

    using QueueProperty = alpaka::queue::Blocking;
    using QueueAcc = alpaka::queue::Queue<Acc, QueueProperty>;
    QueueAcc queue{devAcc};
    // Buffer configuration.
    using BufHost = alpaka::mem::buf::Buf<DevHost, float, Dim, Idx>;
    auto inputBufHost = BufHost{
        alpaka::mem::buf::alloc<float, Idx>(devHost, extentMatrix2D)};
    // two result-buffer for both methods: default/naive and with shared memory
    auto resultBufHostDefault = BufHost{
        alpaka::mem::buf::alloc<float, Idx>(devHost, extentMatrix2D)};
    auto resultBufHostShared = BufHost{
        alpaka::mem::buf::alloc<float, Idx>(devHost, extentMatrix2D)};

    float * const pInputHost = alpaka::mem::view::getPtrNative(inputBufHost);
    float * pResultHostDefault = alpaka::mem::view::getPtrNative(resultBufHostDefault);
    float * pResultHostShared = alpaka::mem::view::getPtrNative(resultBufHostDefault);

    // Generate input-matrix with random numbers.
    std::default_random_engine eng{
        static_cast<std::default_random_engine::result_type>(0)};
    // These pseudo-random numbers are implementation/platform specific.
    // The range has no special meaning.
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);
    for(Idx i(0); i < extentMatrix2D.prod(); i++)
    {
        pInputHost[i] = dist(eng);
    }

    using BufAcc =alpaka::mem::buf::Buf<Acc, float, Dim, Idx>;

    auto inputBufAcc = BufAcc{
        alpaka::mem::buf::alloc<float, Idx>(devAcc, extentMatrix2D)};

    auto resultBufAccDefault = BufAcc{
        alpaka::mem::buf::alloc<float, Idx>(devAcc, extentMatrix2D)};
    auto resultBufAccShared = BufAcc{
        alpaka::mem::buf::alloc<float, Idx>(devAcc, extentMatrix2D)};

    float const * const pInputAcc = alpaka::mem::view::getPtrNative(inputBufAcc);
    float * const pResultAccDefault = alpaka::mem::view::getPtrNative(resultBufAccDefault);
    float * const pResultAccShared = alpaka::mem::view::getPtrNative(resultBufAccDefault);

    alpaka::mem::view::copy(
        queue,
        inputBufAcc,
        inputBufHost,
        extentMatrix2D);

    // Execute.
    Kernel kernelDefault;

    SharedMemKernel kernelShared;

    auto const taskKernelDefault(
        alpaka::kernel::createTaskKernel<Acc>(
            workdiv,
            kernelDefault,
            pInputAcc,
            pResultAccDefault));

    auto const taskKernelShared(
        alpaka::kernel::createTaskKernel<Acc>(
            workdiv,
            kernelShared,
            pInputAcc,
            pResultAccShared));

    std::cout << "Execution time: "
        << "naive: "
        << alpaka::test::integ::measureTaskRunTimeMs(queue, taskKernelDefault)
        << " ms" << std::endl;
    alpaka::wait::wait(queue);
    std::cout << " shared: "
        << alpaka::test::integ::measureTaskRunTimeMs(queue, taskKernelShared)
        << " ms" << std::endl;

    alpaka::mem::view::copy(
        queue,
        resultBufHostDefault,
        resultBufAccDefault,
        extentMatrix2D);
    alpaka::mem::view::copy(
        queue,
        resultBufHostShared,
        resultBufAccShared,
        extentMatrix2D);
    alpaka::wait::wait(queue);

    for(Idx i(0); i < extentMatrix2D.prod(); i++)
    {
        if(!almostEqual(pResultHostDefault[i], pResultHostShared[i]))
        {
            return EXIT_FAILURE;
        }
    }
    std::cout << "Success" << "\n";
    return EXIT_SUCCESS;
}
