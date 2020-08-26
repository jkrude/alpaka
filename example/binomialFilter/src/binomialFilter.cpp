/* Copyright 201 Benjamin Worpitz, Sergei Bastrakov, Jakob Krude,
 *
 * This file exemplifies usage of Alpaka.
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


#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <cstdint>
#include <cstdlib>
#include <random>


//#############################################################################
//! The kernel executing the parallel logic.
//! Each thread computes one result for a binomial filter.
struct Kernel
{
    //-----------------------------------------------------------------------------
    //! The kernel entry point.
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param inputBuf The buffer representing the matrix (global memory).
    //! \param resultBuf The buffer containing all computed results.
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator ()(
        TAcc const & acc,
        float const *const inputBuf,
        float *const resultBuf) const -> void
    {
        auto const globalThreadIdx = alpaka::idx::getIdx<
            alpaka::Grid,
            alpaka::Threads>(acc);
        auto const extent2D = alpaka::workdiv::getWorkDiv<
            alpaka::Grid,
            alpaka::Threads>(acc);

        // Configuration check.
        using Dim = alpaka::dim::DimInt<2u>;
        static_assert(
            alpaka::dim::Dim<TAcc>::value ==
            Dim::value, "This example is designed for 2D");

        auto i = globalThreadIdx[0];
        auto j = globalThreadIdx[1];
        float result;
        // Is the current point on a "edge" of the matrix.
        if(i == 0 || i >= extent2D[0] - 1 || j == 0 || j >= extent2D[1] - 1)
        {
            result = inputBuf[mapIdx(i, j, extent2D)];
        }
        else
        {
            // Compute binomialFilter
            result =
                (0.0625f) * // Normalisation = 1/16
                1.0f * (inputBuf[mapIdx(i - 1, j - 1, extent2D)] +
                2.0f * inputBuf[mapIdx(i - 1, j, extent2D)] +
                1.0f * inputBuf[mapIdx(i - 1, j + 1, extent2D)] +
                2.0f * inputBuf[mapIdx(i, j - 1, extent2D)] +
                4.0f * inputBuf[mapIdx(i, j, extent2D)] +
                2.0f * inputBuf[mapIdx(i, j + 1, extent2D)] +
                1.0f * inputBuf[mapIdx(i + 1, j - 1, extent2D)] +
                2.0f * inputBuf[mapIdx(i + 1, j, extent2D)] +
                1.0f * inputBuf[mapIdx(i + 1, j + 1, extent2D)]);
        }

        resultBuf[mapIdx(i, j, extent2D)] = result;
    }


    //-----------------------------------------------------------------------------
    //! Helper method mapping the input 2D input into 1D.
    template<
        typename TIdx,
        typename TVec>
    ALPAKA_FN_ACC auto mapIdx(
        TIdx i,
        TIdx j,
        TVec const & extent) const -> TIdx
    {
        using Dim1 = alpaka::dim::DimInt<1u>;
        return alpaka::idx::mapIdx<Dim1::value>(TVec(i, j), extent)[0];
    }

};


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
    const Idx extentX = 10;
    const Idx extentY = 10;
    const Vec extentMatrix2D(extentX, extentY);
    const Vec alpakaElementsPerThread(Vec::ones());

    // Setup.
    using WorkDiv = alpaka::workdiv::WorkDivMembers<
        Dim,
        Idx>;
    auto workdiv = WorkDiv{
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extentMatrix2D,
            alpakaElementsPerThread,
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted)};

    using QueueProperty = alpaka::queue::Blocking;
    using QueueAcc = alpaka::queue::Queue<
        Acc,
        QueueProperty>;
    QueueAcc queue{devAcc};
    // Buffer configuration.
    using BufHost = alpaka::mem::buf::Buf<
        DevHost,
        float,
        Dim,
        Idx>;
    auto inputBufHost = BufHost{
        alpaka::mem::buf::alloc<
            float,
            Idx>(
                devHost,
                extentMatrix2D)};
    auto resultBufHost = BufHost{
        alpaka::mem::buf::alloc<
            float,
            Idx>(
                devHost,
                extentMatrix2D)};

    float *const pInputHost = alpaka::mem::view::getPtrNative(inputBufHost);
    float *pResultHost = alpaka::mem::view::getPtrNative(resultBufHost);

    // Generate input-matrix with random numbers.
    std::default_random_engine eng{
        static_cast<std::default_random_engine::result_type>(0)};
    // These pseudo-random numbers are implementation/platform specific!
    // The range has no special meaning.
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);
    for(Idx i(0); i < extentMatrix2D.prod(); i++)
    {
        pInputHost[i] = dist(eng);
    }

    using BufAcc =alpaka::mem::buf::Buf<
        Acc,
        float,
        Dim,
        Idx>;

    auto inputBufAcc = BufAcc{
        alpaka::mem::buf::alloc<
            float,
            Idx>(
                devAcc,
                extentMatrix2D)};

    auto resultBufAcc = BufAcc{
        alpaka::mem::buf::alloc<
            float,
            Idx>(
                devAcc,
                extentMatrix2D)};

    float const *const pInputAcc = alpaka::mem::view::getPtrNative(inputBufAcc);
    float *const pResultAcc = alpaka::mem::view::getPtrNative(resultBufAcc);

    alpaka::mem::view::copy(queue, inputBufAcc, inputBufHost, extentMatrix2D);

    // Execute: Every thread computes one result.
    Kernel kernel;
    alpaka::kernel::exec<Acc>(
        queue,
        workdiv,
        kernel,
        pInputAcc,
        pResultAcc);

    alpaka::mem::view::copy(queue, resultBufHost, resultBufAcc, extentMatrix2D);
    alpaka::wait::wait(queue);

#if !defined(ALPAKA_CI)
    for(Idx i(0); i < extentMatrix2D.prod(); i++)
    {
        std::cout << pResultHost[i] << "  ";
        if((i + 1) % (extentX) == 0)
        {
            std::cout << "\n";
        };
    }
#endif

    return EXIT_SUCCESS;
}
