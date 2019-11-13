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

#include <alpaka/dev/DevCudaHipRt.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/queue/QueueCudaHipRtNonBlocking.hpp>
#include <alpaka/queue/QueueCudaHipRtBlocking.hpp>
#include <alpaka/core/Cuda.hpp>

#include <stdexcept>
#include <memory>
#include <functional>

namespace alpaka
{
    namespace event
    {
        namespace cudaHip
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA RT device event implementation.
                class EventCudaHipImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventCudaHipImpl(
                        dev::DevCudaHipRt const & dev,
                        bool bBusyWait) :
                            m_dev(dev),
                            m_cudaEvent()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                m_dev.m_iDevice));
                        // Create the event on the current device with the specified flags. Valid flags include:
                        // - cudaEventDefault: Default event creation flag.
                        // - cudaEventBlockingSync : Specifies that event should use blocking synchronization.
                        //   A host thread that uses cudaEventSynchronize() to wait on an event created with this flag will block until the event actually completes.
                        // - cudaEventDisableTiming : Specifies that the created event does not need to record timing data.
                        //   Events created with this flag specified and the cudaEventBlockingSync flag not specified will provide the best performance when used with cudaStreamWaitEvent() and cudaEventQuery().
                        ALPAKA_CUDA_RT_CHECK(
                            cudaEventCreateWithFlags(
                                &m_cudaEvent,
                                (bBusyWait ? cudaEventDefault : cudaEventBlockingSync) | cudaEventDisableTiming));
                    }
                    //-----------------------------------------------------------------------------
                    EventCudaHipImpl(EventCudaHipImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    EventCudaHipImpl(EventCudaHipImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventCudaHipImpl const &) -> EventCudaHipImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(EventCudaHipImpl &&) -> EventCudaHipImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~EventCudaHipImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before cudaEventDestroy required?
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            m_dev.m_iDevice));
                        // In case event has been recorded but has not yet been completed when cudaEventDestroy() is called, the function will return immediately
                        // and the resources associated with event will be released automatically once the device has completed event.
                        // -> No need to synchronize here.
                        ALPAKA_CUDA_RT_CHECK(cudaEventDestroy(
                            m_cudaEvent));
                    }

                public:
                    dev::DevCudaHipRt const m_dev;   //!< The device this event is bound to.
                    cudaEvent_t m_cudaEvent;
                };
            }
        }

        //#############################################################################
        //! The CUDA RT device event.
        class EventCudaHipRt final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventCudaHipRt(
                dev::DevCudaHipRt const & dev,
                bool bBusyWait = true) :
                    m_spEventImpl(std::make_shared<cudaHip::detail::EventCudaHipImpl>(dev, bBusyWait))
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
            }
            //-----------------------------------------------------------------------------
            EventCudaHipRt(EventCudaHipRt const &) = default;
            //-----------------------------------------------------------------------------
            EventCudaHipRt(EventCudaHipRt &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventCudaHipRt const &) -> EventCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            auto operator=(EventCudaHipRt &&) -> EventCudaHipRt & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(EventCudaHipRt const & rhs) const
            -> bool
            {
                return (m_spEventImpl == rhs.m_spEventImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventCudaHipRt const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~EventCudaHipRt() = default;

        public:
            std::shared_ptr<cudaHip::detail::EventCudaHipImpl> m_spEventImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device event device get trait specialization.
            template<>
            struct GetDev<
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    event::EventCudaHipRt const & event)
                -> dev::DevCudaHipRt
                {
                    return event.m_spEventImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device event test trait specialization.
            template<>
            struct Test<
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto test(
                    event::EventCudaHipRt const & event)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for events on non current device.
                    cudaError_t ret = cudaSuccess;
                    ALPAKA_CUDA_RT_CHECK_IGNORE(
                        ret = cudaEventQuery(
                            event.m_spEventImpl->m_cudaEvent),
                        cudaErrorNotReady);
                    return (ret == cudaSuccess);
                }
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueCudaHipRtNonBlocking,
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaHipRtNonBlocking & queue,
                    event::EventCudaHipRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
                        event.m_spEventImpl->m_cudaEvent,
                        queue.m_spQueueImpl->m_CudaHipQueue));
                }
            };
            //#############################################################################
            //! The CUDA RT queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueCudaHipRtBlocking,
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaHipRtBlocking & queue,
                    event::EventCudaHipRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
                        event.m_spEventImpl->m_cudaEvent,
                        queue.m_spQueueImpl->m_CudaHipQueue));
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device event thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been completed.
            //! If the event is not enqueued to a queue the method returns immediately.
            template<>
            struct CurrentThreadWaitFor<
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    event::EventCudaHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for events on non current device.
                    ALPAKA_CUDA_RT_CHECK(cudaEventSynchronize(
                        event.m_spEventImpl->m_cudaEvent));
                }
            };
            //#############################################################################
            //! The CUDA RT queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCudaHipRtNonBlocking,
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCudaHipRtNonBlocking & queue,
                    event::EventCudaHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        queue.m_spQueueImpl->m_CudaHipQueue,
                        event.m_spEventImpl->m_cudaEvent,
                        0));
                }
            };
            //#############################################################################
            //! The CUDA RT queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCudaHipRtBlocking,
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCudaHipRtBlocking & queue,
                    event::EventCudaHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        queue.m_spQueueImpl->m_CudaHipQueue,
                        event.m_spEventImpl->m_cudaEvent,
                        0));
                }
            };
            //#############################################################################
            //! The CUDA RT device event wait trait specialization.
            //!
            //! Any future work submitted in any queue of this device will wait for event to complete before beginning execution.
            template<>
            struct WaiterWaitFor<
                dev::DevCudaHipRt,
                event::EventCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    dev::DevCudaHipRt & dev,
                    event::EventCudaHipRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        nullptr,
                        event.m_spEventImpl->m_cudaEvent,
                        0));
                }
            };
        }
    }
}

#endif
