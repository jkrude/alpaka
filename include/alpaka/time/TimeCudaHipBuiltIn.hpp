/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/time/Traits.hpp>

namespace alpaka
{
    namespace time
    {
        //#############################################################################
        //! The GPU CUDA accelerator time implementation.
        class TimeCudaHipBuiltIn
        {
        public:
            using TimeBase = TimeCudaHipBuiltIn;

            //-----------------------------------------------------------------------------
            TimeCudaHipBuiltIn() = default;
            //-----------------------------------------------------------------------------
            __device__ TimeCudaHipBuiltIn(TimeCudaHipBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            __device__ TimeCudaHipBuiltIn(TimeCudaHipBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(TimeCudaHipBuiltIn const &) -> TimeCudaHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(TimeCudaHipBuiltIn &&) -> TimeCudaHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~TimeCudaHipBuiltIn() = default;
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA built-in clock operation.
            template<>
            struct Clock<
                time::TimeCudaHipBuiltIn>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto clock(
                    time::TimeCudaHipBuiltIn const &)
                -> std::uint64_t
                {
                    // This can be converted to a wall-clock time in seconds by dividing through the shader clock rate given by cudaHipDeviceProp::clockRate.
                    // This clock rate is double the main clock rate on Fermi and older cards. 
                    return
                        static_cast<std::uint64_t>(
                            clock64());
                }
            };
        }
    }
}

#endif
