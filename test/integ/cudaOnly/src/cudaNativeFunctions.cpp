/**
 * \file
 * Copyright 2016-2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE) && defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA

//-----------------------------------------------------------------------------
//! Native CUDA function.
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wmissing-prototypes"
#endif
__device__ auto userDefinedThreadFence()
-> void
{
    __threadfence();
}
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

//#############################################################################
class CudaOnlyTestKernel
{
public:
    //-----------------------------------------------------------------------------
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
        alpaka::ignore_unused(acc);

        // We should be able to call some native CUDA functions when ALPAKA_ACC_GPU_CUDA_ONLY_MODE is enabled.
        __threadfence_block();
        userDefinedThreadFence();
        __threadfence_system();

        *success = true;
    }
};


//-----------------------------------------------------------------------------
TEST_CASE("cudaOnlyModeWorking", "[cudaOnly]")
{
    using TAcc = alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<1u>, std::uint32_t>;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    CudaOnlyTestKernel kernel;

    REQUIRE(fixture(kernel));
}

#endif
