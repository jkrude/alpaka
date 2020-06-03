/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Jakob Krude
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <catch2/catch.hpp>
#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/test/acc/TestAccs.hpp>


/*
 * There are three functors used to populate the template-parameter for the main test.
 *
 * Each functor loops over a tuple of options and calls the next layer.
 *
 * The final TestSlicing functor runs the main test-code.
 *
 * TestContainer is used to simplify the main test-code by hiding the
 * details of the alpaka-architecture.
 *
 * DimHelper creates the right-sized vectors as well as the calculation for the
 * the correct result.
 *
 * Current call-chain:
 * ForEachDataType -> ForEachDim -> ForEachAcc -> TestSlicing
 */
template< typename Idx >
struct ForEachDataType;

template< typename Idx >
struct ForEachDim;

template<
    typename Idx,
    typename Data
>
struct ForEachAcc;

template<
    typename Idx,
    typename Dim,
    typename Data
>
struct TestSlicing;

template<
    typename Idx
>
struct ForEachDataType
{
    auto operator()( ) -> void
    {
        using DataTypes = std::tuple<
            int,
            float,
            double
        >;
        alpaka::meta::forEachType< DataTypes >( ForEachDim< Idx >( ) );
    }
};

template< typename Idx >
struct ForEachDim
{
    template< typename Data >
    auto operator()( ) -> void
    {
        using DimTypes = std::tuple<
            alpaka::dim::DimInt< 1u >,
            alpaka::dim::DimInt< 2u >,
            alpaka::dim::DimInt< 3u >>;
        alpaka::meta::forEachType< DimTypes >(
            ForEachAcc<
                Idx,
                Data
            >( )
        );
    }
};

template<
    typename Idx,
    typename Data
>
struct ForEachAcc
{
    template<
        typename Dim
    >
    auto operator()( ) -> void
    {
        using AccTypes = alpaka::test::acc::EnabledAccs<
            Dim,
            Idx
        >;
        alpaka::meta::forEachType< AccTypes >(
            TestSlicing<
                Idx,
                Dim,
                Data
            >( )
        );
    }
};

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
            REQUIRE( ptrA[i] == Approx( vec[i] ) );
        }
    }
};

template<
    typename Dim,
    typename Idx
>
struct DimHelper;

// 1D explicit specialisation
template<
    typename Idx
>
struct DimHelper<
    alpaka::dim::DimInt< 1u >,
    Idx
>
{
    using Vec = alpaka::vec::Vec<
        alpaka::dim::DimInt< 1u >,
        Idx
    >;


    static auto getVec(
        Idx const & elDimOne,
        Idx const & elDimTwo,
        Idx const & elDimThree
    ) -> Vec
    {
        alpaka::ignore_unused( elDimTwo );
        alpaka::ignore_unused( elDimThree );
        return Vec( elDimOne );
    }
    template< typename Data >
    static auto cmpCorrectlySlicedBuffer(
        Vec extents,
        Vec extentsSubView,
        Vec offsets
    ) -> std::vector< Data >
    {
        std::vector< Data >
            resSliced = std::vector< Data >( extentsSubView.prod( ) );
        alpaka::ignore_unused( extents );
        for( Idx i = 0; i < extentsSubView[0]; ++i )
        {
            resSliced[i] = static_cast<Data>(i + offsets[0]);
        }
        return resSliced;
    }
};


// 2D explicit specialisation
template<
    typename Idx
>
struct DimHelper<
    alpaka::dim::DimInt< 2u >,
    Idx
>
{
    using Vec = alpaka::vec::Vec<
        alpaka::dim::DimInt< 2u >,
        Idx
    >;

    static auto getVec(
        Idx const & elDimOne,
        Idx const & elDimTwo,
        Idx const & elDimThree
    ) -> Vec
    {
        alpaka::ignore_unused( elDimThree );
        return Vec(
            elDimOne,
            elDimTwo
        );
    }

    template< typename Data >
    static auto cmpCorrectlySlicedBuffer(
        Vec extents,
        Vec extentsSubView,
        Vec offsets
    ) -> std::vector< Data >
    {
        std::vector< Data >
            resSliced = std::vector< Data >( extentsSubView.prod( ) );
        for( Idx x = 0; x < extentsSubView[0]; ++x )
        {
            for( Idx y = 0; y < extentsSubView[1]; ++y )
            {
                resSliced[x * extentsSubView[1] + y] = static_cast<Data>(
                    ( x + offsets[0] ) * extents[1] + offsets[1] + y);
            }
        }
        return resSliced;
    }
};

// 3D explicit specialisation
template< typename Idx >
struct DimHelper<
    alpaka::dim::DimInt< 3u >,
    Idx
>
{
    using Vec = alpaka::vec::Vec<
        alpaka::dim::DimInt< 3u >,
        Idx
    >;

    static auto getVec(
        Idx const & elDimOne,
        Idx const & elDimTwo,
        Idx const & elDimThree
    ) -> Vec
    {
        return Vec(
            elDimOne,
            elDimTwo,
            elDimThree
        );
    }

    template< typename Data >
    static auto cmpCorrectlySlicedBuffer(
        Vec extents,
        Vec extentsSubView,
        Vec offsets
    ) -> std::vector< Data >
    {
        std::vector< Data >
            resSliced = std::vector< Data >( extentsSubView.prod( ) );
        for( Idx x = 0; x < extentsSubView[0]; ++x )
        {
            for( Idx y = 0; y < extentsSubView[1]; ++y )
            {
                for( Idx z = 0; z < extentsSubView[2]; ++z )
                {
                    resSliced[x * extentsSubView[2] * extentsSubView[1] +
                              extentsSubView[2] * y +
                              z] = static_cast<Data>(
                        ( x + offsets[0] ) * extents[2] * extents[1] +
                        ( y + offsets[1] ) * extents[2] +
                        z +
                        offsets[2]);
                }
            }
        }
        return resSliced;
    }
};

template<
    typename Idx,
    typename Dim,
    typename Data
>
struct TestSlicing
{
    template< typename Acc >
    auto operator()( ) -> void
    {
        constexpr Idx extentsDimOne = 13;
        constexpr Idx extentsDimTwo = 7;
        constexpr Idx extentsDimThree = 5;

        constexpr Idx extentsSubViewDimOne = 5;
        constexpr Idx extentsSubViewDimTwo = 3;
        constexpr Idx extentsSubViewDimThree = 3;

        constexpr Idx offsetDimOne = 3;
        constexpr Idx offsetDimTwo = 2;
        constexpr Idx offsetDimThree = 1;

        using Vec = alpaka::vec::Vec< Dim, Idx >;

        using TestContainer =TestContainer<Dim, Idx, Acc, Data>;

        using BufHost = typename TestContainer::BufHost;
        using BufDevice = typename TestContainer::BufDevice;

        TestContainer slicingTest;

        // Setup extents, extentsSubView and offsets.
        Vec extents =
            DimHelper<
                Dim,
                Idx
            >::getVec(
                extentsDimOne,
                extentsDimTwo,
                extentsDimThree
            );
        Vec extentsSubView = DimHelper< Dim, Idx >::getVec(
            extentsSubViewDimOne,
            extentsSubViewDimTwo,
            extentsSubViewDimThree
        );
        Vec offsets = DimHelper< Dim, Idx >::getVec(
            offsetDimOne,
            offsetDimTwo,
            offsetDimThree
        );
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

        // resultBuffer will be compared with the manually computed results.
        slicingTest.compareBuffer(
            resultBuffer,
            DimHelper< Dim, Idx >::template cmpCorrectlySlicedBuffer< Data >(
                extents,
                extentsSubView,
                offsets
            ),
            extentsSubView
        );
    }
};

TEST_CASE( "memBufSlicingTest",
    "[memBuf]" )
{
    /*
     * FIXME
     * Current call-chain works this way but does'nt look like it.
     * Desired (or similar):
     * ForEachDataType::call(
     *     ForEachDimension::call(
     *         ForEachAccelerator::call(
     *             TestSlicing() ) ) )
     */
    ForEachDataType< size_t >( );
}
