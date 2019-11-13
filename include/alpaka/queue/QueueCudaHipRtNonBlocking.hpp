/* Copyright 2019 Benjamin Worpitz, René Widera
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
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/core/Cuda.hpp>

#include <stdexcept>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace alpaka
{
    namespace event
    {
        class EventCudaHipRt;
    }
}

namespace alpaka
{
    namespace queue
    {
        namespace cudaHip
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA RT non-blocking queue implementation.
                class QueueCudaHipRtNonBlockingImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST QueueCudaHipRtNonBlockingImpl(
                        dev::DevCudaHipRt const & dev) :
                            m_dev(dev),
                            m_CudaHipQueue()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                m_dev.m_iDevice));
                        // - cudaStreamDefault: Default queue creation flag.
                        // - cudaStreamNonBlocking: Specifies that work running in the created queue may run concurrently with work in queue 0 (the NULL queue),
                        //   and that the created queue should perform no implicit synchronization with queue 0.
                        // Create the queue on the current device.
                        // NOTE: cudaStreamNonBlocking is required to match the semantic implemented in the alpaka CPU queue.
                        // It would be too much work to implement implicit default queue synchronization on CPU.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaStreamCreateWithFlags(
                                &m_CudaHipQueue,
                                cudaStreamNonBlocking));
                    }
                    //-----------------------------------------------------------------------------
                    QueueCudaHipRtNonBlockingImpl(QueueCudaHipRtNonBlockingImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueCudaHipRtNonBlockingImpl(QueueCudaHipRtNonBlockingImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCudaHipRtNonBlockingImpl const &) -> QueueCudaHipRtNonBlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCudaHipRtNonBlockingImpl &&) -> QueueCudaHipRtNonBlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~QueueCudaHipRtNonBlockingImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before cudaStreamDestroy required?
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                m_dev.m_iDevice));
                        // In case the device is still doing work in the queue when cudaStreamDestroy() is called, the function will return immediately
                        // and the resources associated with queue will be released automatically once the device has completed all work in queue.
                        // -> No need to synchronize here.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaStreamDestroy(
                                m_CudaHipQueue));
                    }

                public:
                    dev::DevCudaHipRt const m_dev;   //!< The device this queue is bound to.
                    cudaStream_t m_CudaHipQueue;
                };
            }
        }

        //#############################################################################
        //! The CUDA RT non-blocking queue.
        class QueueCudaHipRtNonBlocking final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST QueueCudaHipRtNonBlocking(
                dev::DevCudaHipRt const & dev) :
                m_spQueueImpl(std::make_shared<cudaHip::detail::QueueCudaHipRtNonBlockingImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            QueueCudaHipRtNonBlocking(QueueCudaHipRtNonBlocking const &) = default;
            //-----------------------------------------------------------------------------
            QueueCudaHipRtNonBlocking(QueueCudaHipRtNonBlocking &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCudaHipRtNonBlocking const &) -> QueueCudaHipRtNonBlocking & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCudaHipRtNonBlocking &&) -> QueueCudaHipRtNonBlocking & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(QueueCudaHipRtNonBlocking const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(QueueCudaHipRtNonBlocking const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueCudaHipRtNonBlocking() = default;

        public:
            std::shared_ptr<cudaHip::detail::QueueCudaHipRtNonBlockingImpl> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT non-blocking queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueCudaHipRtNonBlocking>
            {
                using type = dev::DevCudaHipRt;
            };
            //#############################################################################
            //! The CUDA RT non-blocking queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueCudaHipRtNonBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueCudaHipRtNonBlocking const & queue)
                -> dev::DevCudaHipRt
                {
                    return queue.m_spQueueImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT non-blocking queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueCudaHipRtNonBlocking>
            {
                using type = event::EventCudaHipRt;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT sync queue enqueue trait specialization.
            template<
                typename TTask>
            struct Enqueue<
                queue::QueueCudaHipRtNonBlocking,
                TTask>
            {
                //#############################################################################
                enum class CallbackState
                {
                    enqueued,
                    notified,
                    finished,
                };

                //#############################################################################
                struct CallbackSynchronizationData : public std::enable_shared_from_this<CallbackSynchronizationData>
                {
                    std::mutex m_mutex;
                    std::condition_variable m_event;
                    CallbackState state = CallbackState::enqueued;
                };

                //-----------------------------------------------------------------------------
                static void CUDART_CB cudaHipRtCallback(cudaStream_t /*queue*/, cudaError_t /*status*/, void *arg)
                {
                    // explicitly copy the shared_ptr so that this method holds the state even when the executing thread has already finished.
                    const auto pCallbackSynchronizationData = reinterpret_cast<CallbackSynchronizationData*>(arg)->shared_from_this();

                    // Notify the executing thread.
                    {
                        std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                        pCallbackSynchronizationData->state = CallbackState::notified;
                    }
                    pCallbackSynchronizationData->m_event.notify_one();

                    // Wait for the executing thread to finish the task if it has not already finished.
                    std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                    if(pCallbackSynchronizationData->state != CallbackState::finished)
                    {
                        pCallbackSynchronizationData->m_event.wait(
                            lock,
                            [pCallbackSynchronizationData](){
                                return pCallbackSynchronizationData->state == CallbackState::finished;
                            }
                        );
                    }
                }

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaHipRtNonBlocking & queue,
                    TTask const & task)
                -> void
                {
                    auto pCallbackSynchronizationData = std::make_shared<CallbackSynchronizationData>();

                    ALPAKA_CUDA_RT_CHECK(cudaStreamAddCallback(
                        queue.m_spQueueImpl->m_CudaHipQueue,
                        cudaHipRtCallback,
                        pCallbackSynchronizationData.get(),
                        0u));

                    // We start a new std::thread which stores the task to be executed.
                    // This circumvents the limitation that it is not possible to call CUDA methods within the CUDA callback thread.
                    // The CUDA thread signals the std::thread when it is ready to execute the task.
                    // The CUDA thread is waiting for the std::thread to signal that it is finished executing the task
                    // before it executes the next task in the queue (CUDA stream).
                    std::thread t(
                        [pCallbackSynchronizationData, task](){

                            // If the callback has not yet been called, we wait for it.
                            {
                                std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                                if(pCallbackSynchronizationData->state != CallbackState::notified)
                                {
                                    pCallbackSynchronizationData->m_event.wait(
                                        lock,
                                        [pCallbackSynchronizationData](){
                                            return pCallbackSynchronizationData->state == CallbackState::notified;
                                        }
                                    );
                                }

                                task();

                                // Notify the waiting CUDA thread.
                                pCallbackSynchronizationData->state = CallbackState::finished;
                            }
                            pCallbackSynchronizationData->m_event.notify_one();
                        }
                    );

                    t.detach();
                }
            };
            //#############################################################################
            //! The CUDA RT non-blocking queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueCudaHipRtNonBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueCudaHipRtNonBlocking const & queue)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Query is allowed even for queues on non current device.
                    cudaError_t ret = cudaSuccess;
                    ALPAKA_CUDA_RT_CHECK_IGNORE(
                        ret = cudaStreamQuery(
                            queue.m_spQueueImpl->m_CudaHipQueue),
                        cudaErrorNotReady);
                    return (ret == cudaSuccess);
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT non-blocking queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueCudaHipRtNonBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueCudaHipRtNonBlocking const & queue)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for queues on non current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaStreamSynchronize(
                            queue.m_spQueueImpl->m_CudaHipQueue));
                }
            };
        }
    }
}

#endif
