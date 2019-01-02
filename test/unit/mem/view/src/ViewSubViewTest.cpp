/**
 * \file
 * Copyright 2015-2019 Benjamin Worpitz
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

#include <catch2/catch.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/Extent.hpp>

#include <alpaka/core/BoostPredef.hpp>

#include <type_traits>
#include <numeric>


#if BOOST_COMP_GNUC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wcast-align" // "cast from 'std::uint8_t*' to 'Elem*' increases required alignment of target type"
#endif

namespace alpaka
{
namespace test
{
namespace mem
{
namespace view
{
    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        typename TBuf>
    auto testViewSubViewImmutable(
        alpaka::mem::view::ViewSubView<TDev, TElem, TDim, TIdx> const & view,
        TBuf & buf,
        TDev const & dev,
        alpaka::vec::Vec<TDim, TIdx> const & extentView,
        alpaka::vec::Vec<TDim, TIdx> const & offsetView)
    -> void
    {
        //-----------------------------------------------------------------------------
        alpaka::test::mem::view::testViewImmutable<
            TElem>(
                view,
                dev,
                extentView,
                offsetView);

        //-----------------------------------------------------------------------------
        // alpaka::mem::view::traits::GetPitchBytes
        // The pitch of the view has to be identical to the pitch of the underlying buffer in all dimensions.
        {
            auto const pitchBuf(alpaka::mem::view::getPitchBytesVec(buf));
            auto const pitchView(alpaka::mem::view::getPitchBytesVec(view));

            for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
            {
                REQUIRE(
                    pitchBuf[i-static_cast<TIdx>(1u)] ==
                    pitchView[i-static_cast<TIdx>(1u)]);
            }
        }

        //-----------------------------------------------------------------------------
        // alpaka::mem::view::traits::GetPtrNative
        // The native pointer has to be exactly the value we calculate here.
        {
            auto viewPtrNative(
                reinterpret_cast<std::uint8_t *>(
                    alpaka::mem::view::getPtrNative(buf)));
            auto const pitchBuf(alpaka::mem::view::getPitchBytesVec(buf));
            for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
            {
                auto const pitch = (i < static_cast<TIdx>(TDim::value)) ? pitchBuf[i] : static_cast<TIdx>(sizeof(TElem));
                viewPtrNative += offsetView[i - static_cast<TIdx>(1u)] * pitch;
            }
            REQUIRE(
                reinterpret_cast<TElem *>(viewPtrNative) ==
                alpaka::mem::view::getPtrNative(view));
        }
    }

    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        typename TBuf>
    auto testViewSubViewMutable(
        alpaka::mem::view::ViewSubView<TDev, TElem, TDim, TIdx> & view,
        TBuf & buf,
        TDev const & dev,
        alpaka::vec::Vec<TDim, TIdx> const & extentView,
        alpaka::vec::Vec<TDim, TIdx> const & offsetView)
    -> void
    {
        //-----------------------------------------------------------------------------
        testViewSubViewImmutable<
            TAcc>(
                view,
                buf,
                dev,
                extentView,
                offsetView);

        using Queue = alpaka::test::queue::DefaultQueue<TDev>;
        Queue queue(dev);
        //-----------------------------------------------------------------------------
        alpaka::test::mem::view::testViewMutable<
            TAcc>(
                queue,
                view);
    }

    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TElem>
    auto testViewSubViewNoOffset()
    -> void
    {
        using Dev = alpaka::dev::Dev<TAcc>;
        using Pltf = alpaka::pltf::Pltf<Dev>;

        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;
        using View = alpaka::mem::view::ViewSubView<Dev, TElem, Dim, Idx>;

        Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));

        auto const extentBuf(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, alpaka::test::CreateExtentBufVal>(Idx()));
        auto buf(alpaka::mem::buf::alloc<TElem, Idx>(dev, extentBuf));

        auto const extentView(extentBuf);
        auto const offsetView(alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(0)));
        View view(buf);

        alpaka::test::mem::view::testViewSubViewMutable<TAcc>(view, buf, dev, extentView, offsetView);
    }

    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TElem>
    auto testViewSubViewOffset()
    -> void
    {
        using Dev = alpaka::dev::Dev<TAcc>;
        using Pltf = alpaka::pltf::Pltf<Dev>;

        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;
        using View = alpaka::mem::view::ViewSubView<Dev, TElem, Dim, Idx>;

        Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));

        auto const extentBuf(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, alpaka::test::CreateExtentBufVal>(Idx()));
        auto buf(alpaka::mem::buf::alloc<TElem, Idx>(dev, extentBuf));

        auto const extentView(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, alpaka::test::CreateExtentViewVal>(Idx()));
        auto const offsetView(alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(1)));
        View view(buf, extentView, offsetView);

        alpaka::test::mem::view::testViewSubViewMutable<TAcc>(view, buf, dev, extentView, offsetView);
    }

    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TElem>
    auto testViewSubViewOffsetConst()
    -> void
    {
        using Dev = alpaka::dev::Dev<TAcc>;
        using Pltf = alpaka::pltf::Pltf<Dev>;

        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;
        using View = alpaka::mem::view::ViewSubView<Dev, TElem, Dim, Idx>;

        Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));

        auto const extentBuf(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, CreateExtentBufVal>(Idx()));
        auto buf(alpaka::mem::buf::alloc<TElem, Idx>(dev, extentBuf));

        auto const extentView(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, CreateExtentViewVal>(Idx()));
        auto const offsetView(alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(1)));
        View const view(buf, extentView, offsetView);

        alpaka::test::mem::view::testViewSubViewImmutable<TAcc>(view, buf, dev, extentView, offsetView);
    }
}
}
}
}
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic pop
#endif

//-----------------------------------------------------------------------------
struct TestTemplateNoOffset
{
template< typename TAcc >
void operator()()
{
    alpaka::test::mem::view::testViewSubViewNoOffset<TAcc, float>();
}
};

//-----------------------------------------------------------------------------
struct TestTemplateOffset
{
template< typename TAcc >
void operator()()
{
    alpaka::test::mem::view::testViewSubViewOffset<TAcc, float>();
}
};

//-----------------------------------------------------------------------------
struct TestTemplateConst
{
template< typename TAcc >
void operator()()
{
    alpaka::test::mem::view::testViewSubViewOffsetConst<TAcc, float>();
}
};

TEST_CASE( "viewSubViewNoOffsetTest", "[memView]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateNoOffset() );
}

TEST_CASE( "viewSubViewOffsetTest", "[memView]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateOffset() );
}

TEST_CASE( "viewSubViewOffsetConstTest", "[memView]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplateConst() );
}
