/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#pragma once

// Base classes.
#include <alpaka/core/BasicWorkDiv.hpp>         // workdiv::BasicWorkDiv
#include <alpaka/accs/omp/omp2/Idx.hpp>         // IdxOmp2
#include <alpaka/accs/omp/omp2/Atomic.hpp>      // AtomicOmp2

// User functionality.
#include <alpaka/host/Mem.hpp>                  // Copy
#include <alpaka/host/Rand.hpp>                 // rand

// Specialized traits.
#include <alpaka/traits/Acc.hpp>                // AccType
#include <alpaka/traits/Exec.hpp>               // ExecType
#include <alpaka/traits/Event.hpp>              // EventType
#include <alpaka/traits/Mem.hpp>                // SpaceType
#include <alpaka/traits/Stream.hpp>             // StreamType

// Implementation details.
#include <alpaka/accs/omp/Common.hpp>
#include <alpaka/accs/omp/omp2/Dev.hpp>         // Devices
#include <alpaka/accs/omp/omp2/Stream.hpp>      // StreamOmp2
#include <alpaka/host/mem/Space.hpp>            // SpaceHost
#include <alpaka/traits/Kernel.hpp>             // BlockSharedExternMemSizeBytes

#include <cstdint>                              // std::uint32_t
#include <vector>                               // std::vector
#include <cassert>                              // assert
#include <stdexcept>                            // std::runtime_error
#include <string>                               // std::to_string
#include <utility>                              // std::move, std::forward
#include <memory>                               // std::unique_ptr

namespace alpaka
{
    namespace accs
    {
        namespace omp
        {
            //-----------------------------------------------------------------------------
            //! The OpenMP2 accelerator.
            //-----------------------------------------------------------------------------
            namespace omp2
            {
                //-----------------------------------------------------------------------------
                //! The OpenMP2 accelerator implementation details.
                //-----------------------------------------------------------------------------
                namespace detail
                {
                    class ExecOmp2;

                    //#############################################################################
                    //! The OpenMP2 accelerator.
                    //!
                    //! This accelerator allows parallel kernel execution on the host.
                    // \TODO: Offloading?
                    //! It uses OpenMP2 to implement the parallelism.
                    //#############################################################################
                    class AccOmp2 :
                        protected alpaka::workdiv::BasicWorkDiv,
                        protected IdxOmp2,
                        protected AtomicOmp2
                    {
                    public:
                        using MemSpace = mem::SpaceHost;

                        friend class ::alpaka::accs::omp::omp2::detail::ExecOmp2;

                    private:
                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        template<
                            typename TWorkDiv>
                        ALPAKA_FCT_ACC_NO_CUDA AccOmp2(
                            TWorkDiv const & workDiv) :
                                alpaka::workdiv::BasicWorkDiv(workDiv),
                                IdxOmp2(m_v3uiGridBlockIdx),
                                AtomicOmp2(),
                                m_v3uiGridBlockIdx(Vec3<>::zeros())
                        {}

                    public:
                        //-----------------------------------------------------------------------------
                        //! Copy constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA AccOmp2(AccOmp2 const &) = delete;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                        //-----------------------------------------------------------------------------
                        //! Move constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA AccOmp2(AccOmp2 &&) = delete;
#endif
                        //-----------------------------------------------------------------------------
                        //! Copy assignment.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccOmp2 const &) -> AccOmp2 & = delete;
                        //-----------------------------------------------------------------------------
                        //! Destructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA virtual ~AccOmp2() noexcept = default;

                        //-----------------------------------------------------------------------------
                        //! \return The requested indices.
                        //-----------------------------------------------------------------------------
                        template<
                            typename TOrigin,
                            typename TUnit,
                            typename TDim = dim::Dim3>
                        ALPAKA_FCT_ACC_NO_CUDA auto getIdx() const
                        -> Vec<TDim>
                        {
                            return idx::getIdx<TOrigin, TUnit, TDim>(
                                *static_cast<IdxOmp2 const *>(this),
                                *static_cast<workdiv::BasicWorkDiv const *>(this));
                        }

                        //-----------------------------------------------------------------------------
                        //! \return The requested extents.
                        //-----------------------------------------------------------------------------
                        template<
                            typename TOrigin,
                            typename TUnit,
                            typename TDim = dim::Dim3>
                        ALPAKA_FCT_ACC_NO_CUDA auto getWorkDiv() const
                        -> Vec<TDim>
                        {
                            return workdiv::getWorkDiv<TOrigin, TUnit, TDim>(
                                *static_cast<workdiv::BasicWorkDiv const *>(this));
                        }

                        //-----------------------------------------------------------------------------
                        //! Execute the atomic operation on the given address with the given value.
                        //! \return The old value before executing the atomic operation.
                        //-----------------------------------------------------------------------------
                        template<
                            typename TOp,
                            typename T>
                        ALPAKA_FCT_ACC auto atomicOp(
                            T * const addr,
                            T const & value) const
                        -> T
                        {
                            return atomic::atomicOp<TOp, T>(
                                addr,
                                value,
                                *static_cast<AtomicOmp2 const *>(this));
                        }

                        //-----------------------------------------------------------------------------
                        //! Syncs all threads in the current block.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA auto syncBlockThreads() const
                        -> void
                        {
                            #pragma omp barrier
                        }

                        //-----------------------------------------------------------------------------
                        //! \return Allocates block shared memory.
                        //-----------------------------------------------------------------------------
                        template<
                            typename T,
                            UInt TuiNumElements>
                        ALPAKA_FCT_ACC_NO_CUDA auto allocBlockSharedMem() const
                        -> T *
                        {
                            static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                            // Assure that all threads have executed the return of the last allocBlockSharedMem function (if there was one before).
                            syncBlockThreads();

                            // Arbitrary decision: The thread with id 0 has to allocate the memory.
                            if(::omp_get_thread_num() == 0)
                            {
                                // \TODO: C++14 std::make_unique would be better.
                                m_vvuiSharedMem.emplace_back(
                                    std::unique_ptr<uint8_t[]>(
                                        reinterpret_cast<uint8_t*>(new T[TuiNumElements])));
                            }
                            syncBlockThreads();

                            return reinterpret_cast<T*>(m_vvuiSharedMem.back().get());
                        }

                        //-----------------------------------------------------------------------------
                        //! \return The pointer to the externally allocated block shared memory.
                        //-----------------------------------------------------------------------------
                        template<
                            typename T>
                        ALPAKA_FCT_ACC_NO_CUDA auto getBlockSharedExternMem() const
                        -> T *
                        {
                            return reinterpret_cast<T*>(m_vuiExternalSharedMem.get());
                        }

#ifdef ALPAKA_NVCC_FRIEND_ACCESS_BUG
                    protected:
#else
                    private:
#endif
                        // getIdx
                        Vec3<> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.

                        // allocBlockSharedMem
                        std::vector<
                            std::unique_ptr<uint8_t[]>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                        // getBlockSharedExternMem
                        std::unique_ptr<uint8_t[]> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                    };

                    //#############################################################################
                    //! The OpenMP2 accelerator executor.
                    //#############################################################################
                    class ExecOmp2 :
                        private AccOmp2
                    {
                    public:
                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        template<
                            typename TWorkDiv>
                        ALPAKA_FCT_HOST ExecOmp2(
                            TWorkDiv const & workDiv,
                            StreamOmp2 const &) :
                                AccOmp2(workDiv)
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                        }
                        //-----------------------------------------------------------------------------
                        //! Copy constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST ExecOmp2(
                            ExecOmp2 const & other) :
                                AccOmp2(static_cast<workdiv::BasicWorkDiv const &>(other))
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                        }
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                        //-----------------------------------------------------------------------------
                        //! Move constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST ExecOmp2(
                            ExecOmp2 && other) :
                                AccOmp2(static_cast<workdiv::BasicWorkDiv &&>(other))
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                        }
#endif
                        //-----------------------------------------------------------------------------
                        //! Copy assignment.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_HOST auto operator=(ExecOmp2 const &) -> ExecOmp2 & = delete;
                        //-----------------------------------------------------------------------------
                        //! Destructor.
                        //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                        ALPAKA_FCT_HOST virtual ~ExecOmp2() = default;
#else
                        ALPAKA_FCT_HOST virtual ~ExecOmp2() noexcept = default;
#endif

                        //-----------------------------------------------------------------------------
                        //! Executes the kernel functor.
                        //-----------------------------------------------------------------------------
                        template<
                            typename TKernelFunctor,
                            typename... TArgs>
                        ALPAKA_FCT_HOST auto operator()(
                            TKernelFunctor && kernelFunctor,
                            TArgs && ... args) const
                        -> void
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                            Vec3<> const v3uiGridBlockExtents(this->AccOmp2::getWorkDiv<Grid, Blocks, dim::Dim3>());
                            Vec3<> const v3uiBlockThreadExtents(this->AccOmp2::getWorkDiv<Block, Threads, dim::Dim3>());

                            auto const uiBlockSharedExternMemSizeBytes(kernel::getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccOmp2>(
                                v3uiBlockThreadExtents,
                                std::forward<TArgs>(args)...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                                << std::endl;
#endif
                            this->AccOmp2::m_vuiExternalSharedMem.reset(
                                new uint8_t[uiBlockSharedExternMemSizeBytes]);

                            // The number of threads in this block.
                            auto const uiNumThreadsInBlock(this->AccOmp2::getWorkDiv<Block, Threads, dim::Dim1>()[0]);

                            // Execute the blocks serially.
                            for(this->AccOmp2::m_v3uiGridBlockIdx[2] = 0; this->AccOmp2::m_v3uiGridBlockIdx[2]<v3uiGridBlockExtents[2]; ++this->AccOmp2::m_v3uiGridBlockIdx[2])
                            {
                                for(this->AccOmp2::m_v3uiGridBlockIdx[1] = 0; this->AccOmp2::m_v3uiGridBlockIdx[1]<v3uiGridBlockExtents[1]; ++this->AccOmp2::m_v3uiGridBlockIdx[1])
                                {
                                    for(this->AccOmp2::m_v3uiGridBlockIdx[0] = 0; this->AccOmp2::m_v3uiGridBlockIdx[0]<v3uiGridBlockExtents[0]; ++this->AccOmp2::m_v3uiGridBlockIdx[0])
                                    {
                                        // Execute the threads in parallel.

                                        // Force the environment to use the given number of threads.
                                        ::omp_set_dynamic(0);

                                        // Parallel execution of the threads in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
                                        // So we have to spawn one OS thread per thread in a block.
                                        // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
                                        // Therefore we use 'omp parallel' with the specified number of threads in a block.
                                        //
                                        // \TODO: Does this hinder executing multiple threads in parallel because their block sizes/omp thread numbers are interfering? Is this num_threads global? Is this a real use case?
                                        #pragma omp parallel num_threads(static_cast<int>(uiNumThreadsInBlock))
                                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                                            if((::omp_get_thread_num() == 0) && (this->AccOmp2::m_v3uiGridBlockIdx[2] == 0) && (this->AccOmp2::m_v3uiGridBlockIdx[1] == 0) && (this->AccOmp2::m_v3uiGridBlockIdx[0] == 0))
                                            {
                                                assert(::omp_get_num_threads()>=0);
                                                auto const uiNumThreads(static_cast<decltype(uiNumThreadsInBlock)>(::omp_get_num_threads()));
                                                std::cout << BOOST_CURRENT_FUNCTION << " omp_get_num_threads: " << uiNumThreads << std::endl;
                                                if(uiNumThreads != uiNumThreadsInBlock)
                                                {
                                                    throw std::runtime_error("The OpenMP2 runtime did not use the number of threads that had been required!");
                                                }
                                            }
#endif
                                            std::forward<TKernelFunctor>(kernelFunctor)(
                                                (*static_cast<AccOmp2 const *>(this)),
                                                std::forward<TArgs>(args)...);

                                            // Wait for all threads to finish before deleting the shared memory.
                                            this->AccOmp2::syncBlockThreads();
                                        }

                                        // After a block has been processed, the shared memory can be deleted.
                                        this->AccOmp2::m_vvuiSharedMem.clear();
                                    }
                                }
                            }
                            // After all blocks have been processed, the external shared memory can be deleted.
                            this->AccOmp2::m_vuiExternalSharedMem.reset();
                        }
                    };
                }
            }
        }
    }

    using AccOmp2 = accs::omp::omp2::detail::AccOmp2;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The OpenMP2 accelerator kernel executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::omp::omp2::detail::ExecOmp2>
            {
                using type = AccOmp2;
            };

            //#############################################################################
            //! The OpenMP2 accelerator accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = accs::omp::omp2::detail::AccOmp2;
            };

            //#############################################################################
            //! The OpenMP2 accelerator name trait specialization.
            //#############################################################################
            template<>
            struct GetAccName<
                accs::omp::omp2::detail::AccOmp2>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccOmp2";
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The OpenMP2 accelerator event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = accs::omp::omp2::detail::EventOmp2;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The OpenMP2 accelerator executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = accs::omp::omp2::detail::ExecOmp2;
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The OpenMP2 accelerator memory space trait specialization.
            //#############################################################################
            template<>
            struct SpaceType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = alpaka::mem::SpaceHost;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The OpenMP2 accelerator stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = accs::omp::omp2::detail::StreamOmp2;
            };

            //#############################################################################
            //! The OpenMP2 accelerator kernel executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                accs::omp::omp2::detail::ExecOmp2>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::omp::omp2::detail::ExecOmp2 const &)
                -> accs::omp::omp2::detail::StreamOmp2
                {
                    return accs::omp::omp2::detail::StreamOmp2(
                        accs::omp::omp2::detail::DevManOmp2::getDevByIdx(0));
                }
            };
        }
    }
}