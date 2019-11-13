/* Copyright 2019 Axel Huebl, Benjamin Worpitz
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

#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

#include <alpaka/core/Cuda.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/vec/Vec.hpp>

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! The GPU CUDA accelerator work division.
        template<
            typename TDim,
            typename TIdx>
        class WorkDivCudaHipBuiltIn
        {
        public:
            using WorkDivBase = WorkDivCudaHipBuiltIn;

            //-----------------------------------------------------------------------------
            __device__ WorkDivCudaHipBuiltIn(
                vec::Vec<TDim, TIdx> const & threadElemExtent) :
                    m_threadElemExtent(threadElemExtent)
            {}
            //-----------------------------------------------------------------------------
            __device__ WorkDivCudaHipBuiltIn(WorkDivCudaHipBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            __device__ WorkDivCudaHipBuiltIn(WorkDivCudaHipBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(WorkDivCudaHipBuiltIn const &) -> WorkDivCudaHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(WorkDivCudaHipBuiltIn &&) -> WorkDivCudaHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~WorkDivCudaHipBuiltIn() = default;

        public:
            // \TODO: Optimize! Add WorkDivCudaHipBuiltInNoElems that has no member m_threadElemExtent as well as AccGpuCudaHipRtNoElems.
            // Use it instead of AccGpuCudaHipRt if the thread element extent is one to reduce the register usage.
            vec::Vec<TDim, TIdx> const & m_threadElemExtent;
        };
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                workdiv::WorkDivCudaHipBuiltIn<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                workdiv::WorkDivCudaHipBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
    namespace workdiv
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division grid block extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivCudaHipBuiltIn<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                __device__ static auto getWorkDiv(
                    WorkDivCudaHipBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(workDiv);
                    return vec::cast<TIdx>(extent::getExtentVecEnd<TDim>(gridDim));
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator work division block thread extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivCudaHipBuiltIn<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                __device__ static auto getWorkDiv(
                    WorkDivCudaHipBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    alpaka::ignore_unused(workDiv);
                    return vec::cast<TIdx>(extent::getExtentVecEnd<TDim>(blockDim));
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator work division thread element extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivCudaHipBuiltIn<TDim, TIdx>,
                origin::Thread,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                __device__ static auto getWorkDiv(
                    WorkDivCudaHipBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    return workDiv.m_threadElemExtent;
                }
            };
        }
    }
}

#endif
