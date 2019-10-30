 /* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
 
#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    #include <cuda_runtime.h>
    #if !BOOST_LANG_CUDA
        #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
    #endif
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)

    #if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
        #include <cuda_runtime_api.h>
    #else
        #if BOOST_COMP_HCC
            #include <math_functions.h>
        #else
            #include <math_functions.hpp>
        #endif
    #endif
    
    #if !BOOST_LANG_HIP
        #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
    #endif
#endif

#include <alpaka/core/Unused.hpp>

#include <type_traits>

#include <alpaka/math/asin/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in asin.
        class AsinCudaHipBuiltIn
        {
        public:
            using AsinBase = AsinCudaHipBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA asin trait specialization.
            template<
                typename TArg>
            struct Asin<
                AsinCudaHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto asin(
                    AsinCudaHipBuiltIn const & asin_ctx,
                    TArg const & arg)
                -> decltype(::asin(arg))
                {
                    alpaka::ignore_unused(asin_ctx);
                    return ::asin(arg);
                }
            };

            template<>
            struct Asin<
                AsinCudaHipBuiltIn,
                float>
            {
                __device__ static auto asin(
                    AsinCudaHipBuiltIn const & asin_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(asin_ctx);
                    return ::asinf(arg);
                }
            };
        }
    }
}

#endif
