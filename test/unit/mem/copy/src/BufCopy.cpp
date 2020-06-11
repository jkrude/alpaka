/* Copyright 2019 Alexander Matthes, Benjamin Worpitz
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

#include <catch2/catch.hpp>
#include <alpaka/meta/CartesianProduct.hpp>
#include <alpaka/test/dim/TestDims.hpp>
#include <alpaka/test/idx/TestIdxs.hpp>
#include <iostream>
#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/Extent.hpp>


namespace alpaka
{
    namespace test
    {
        template< >
        struct CreateExtentBufVal< 0 >
        {
            template< typename TIdx >
            static auto create( TIdx ) -> TIdx
            {
                return 13;
            }
        };

        template< >
        struct CreateExtentBufVal< 1 >
        {
            template< typename TIdx >
            static auto create( TIdx ) -> TIdx
            {
                return 7;
            }
        };

        template< >
        struct CreateExtentBufVal< 2 >
        {
            template< typename TIdx >
            static auto create( TIdx ) -> TIdx
            {
                return 5;
            }
        };
    }

}


template<
    typename Dim,
    typename Idx,
    typename Acc,
    typename Data,
    typename Vec = alpaka::vec::Vec<
        Dim,
        Idx>>
struct TestContainer
{

    using AccQueueProperty = alpaka::queue::Blocking;
    using DevQueue = alpaka::queue::Queue<
        Acc,
        AccQueueProperty
    >;
    using DevAcc = alpaka::dev::Dev< Acc >;
    using PltfAcc = alpaka::pltf::Pltf< DevAcc >;

    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf< DevHost >;

    using BufHost = alpaka::mem::buf::Buf<
        DevHost,
        Data,
        Dim,
        Idx
    >;
    using BufDevice = alpaka::mem::buf::Buf<
        DevAcc,
        Data,
        Dim,
        Idx
    >;

    using SubView = alpaka::mem::view::ViewSubView<
        DevAcc,
        Data,
        Dim,
        Idx
    >;

    DevAcc const devAcc;
    DevHost const devHost;
    DevQueue devQueue;


    // Constructor
    TestContainer( ) :
        devAcc( alpaka::pltf::getDevByIdx< PltfAcc >( 0u ) ),
        devHost( alpaka::pltf::getDevByIdx< PltfHost >( 0u ) ),
        devQueue( devAcc )
    {
    }


    auto createHostBuffer(
        Vec extents,
        bool indexed
    ) -> BufHost
    {
        BufHost bufHost(
            alpaka::mem::buf::alloc<
                Data,
                Idx
            >(
                devHost,
                extents
            )
        );
        if( indexed )
        {
            Data * const ptr = alpaka::mem::view::getPtrNative( bufHost );
            for( Idx i( 0 ); i < extents.prod( ); ++i )
            {
                ptr[i] = static_cast<Data>(i);
            }

        }
        return bufHost;
    }


    auto createDeviceBuffer( Vec extents ) -> BufDevice
    {
        BufDevice bufDevice(
            alpaka::mem::buf::alloc<
                Data,
                Idx
            >(
                devHost,
                extents
            )
        );
        return bufDevice;
    }


    auto copyToAcc(
        BufHost bufHost,
        BufDevice bufAcc,
        Vec extents
    ) -> void
    {
        alpaka::mem::view::copy(
            devQueue,
            bufAcc,
            bufHost,
            extents
        );
    }


    auto copyToHost(
        BufDevice bufAcc,
        BufHost bufHost,
        Vec extents
    ) -> void
    {
        alpaka::mem::view::copy(
            devQueue,
            bufHost,
            bufAcc,
            extents
        );
    }


    auto sliceOnDevice(
        BufDevice bufferToBeSliced,
        Vec subViewExtents,
        Vec offsets
    ) -> BufDevice
    {
        BufDevice slicedBuffer = createDeviceBuffer( subViewExtents );
        // Create a subView with an possible offset.
        SubView subView = SubView(
            bufferToBeSliced,
            subViewExtents,
            offsets
        );
        // Copy the subView into a new buffer
        alpaka::mem::view::copy(
            devQueue,
            slicedBuffer,
            subView,
            subViewExtents
        );
        return slicedBuffer;
    }


    auto compareBuffer(
        BufHost const & bufferA,
        std::vector< Data > const & vec,
        Vec extents
    ) const
    {
        // create default buffer

        Data const * const ptrA = alpaka::mem::view::getPtrNative( bufferA );
        // Test if the buffer was not modified with the copy-operations
        for( Idx i( 0 ); i < extents.prod( ); ++i )
        {
            // float and double caparison should be safe too,
            // because the data was not modified.
            REQUIRE( ptrA[i] == Approx( vec[static_cast<size_t>(i)] ) );
        }
    }
};

using CP = alpaka::meta::CartesianProduct<
    std::tuple,
    std::tuple<
        int,
        float,
        double
    >,
    alpaka::test::acc::TestAccs
>;

TEMPLATE_LIST_TEST_CASE( "bufCopyList",
    "[mem]",
    CP )
{
    using Data =std::tuple_element_t<0, TestType>;
    using Acc = std::tuple_element_t<
        1,
        TestType
    >;


    using Dim = alpaka::dim::Dim< Acc >;
    using Idx = alpaka::idx::Idx< Acc >;
    auto const extents(
        alpaka::vec::createVecFromIndexedFn<
            Dim,
            alpaka::test::CreateExtentBufVal
        >( Idx( ) )
    );
    auto const
        extentsSubView
        (
            alpaka::vec::createVecFromIndexedFn<
                Dim,
                alpaka::test::CreateExtentViewVal
            >( Idx( ) )
        );
    auto const
        offsets
        (
            extents-extentsSubView
        );
    using TestContainer =TestContainer<Dim, Idx, Acc, Data>;

    using BufHost = typename TestContainer::BufHost;
    using BufDevice = typename TestContainer::BufDevice;

    TestContainer slicingTest;

    // Setup extents, extentsSubView and offsets.

    // This is the initial buffer.
    BufHost indexedBuffer = slicingTest.createHostBuffer(
        extents,
        true
    );
    // This buffer will hold the sliced-buffer when it was copied to the host.
    BufHost resultBuffer = slicingTest.createHostBuffer(
        extentsSubView,
        false
    );

    // Copy of the indexBuffer on the deviceSide.
    BufDevice deviceBuffer = slicingTest.createDeviceBuffer( extents );

    // Start: Main-Test
    slicingTest.copyToAcc(
        indexedBuffer,
        deviceBuffer,
        extents
    );

    BufDevice slicedBuffer = slicingTest.sliceOnDevice(
        deviceBuffer,
        extentsSubView,
        offsets
    );

    slicingTest.copyToHost(
        slicedBuffer,
        resultBuffer,
        extentsSubView
    );
    
}


















