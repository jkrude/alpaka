/** Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "Defines.hpp"

//! @param NAME The Name used for the Functor, e.g. OpAbs
//! @param ARITY Enum-type can be one ... n
//! @param STD_OP Function used for the host side, e.g. std::abs
//! @param ALPAKA_OP Function used for the device side, e.g. alpaka::math::abs.
//! @param ... List of Ranges. Needs to match the arity.
#define ALPAKA_TEST_MATH_OP_FUNCTOR(NAME, ARITY, STD_OP, ALPAKA_OP, ...)\
class NAME\
{\
public:\
    /* ranges is not a constexpr, so that its accessible via for loop*/ \
    const Range ranges [ static_cast< int >( ARITY ) ] = {__VA_ARGS__};\
    static constexpr Arity arity = ARITY;\
    \
    template<\
        typename TAcc,\
        typename... TArgs,\
        typename std::enable_if< /* SFINAE: Enables if called from device. */ \
            !std::is_same<\
                TAcc,\
                std::nullptr_t>::value,\
            int>::type = 0>\
    ALPAKA_FN_ACC\
    auto operator()(\
        TAcc const & acc,\
        TArgs const & ... args ) const\
    -> decltype(\
        ALPAKA_OP(\
            acc,\
            args... ) )\
    {\
        return ALPAKA_OP(\
            acc,\
            args... );\
    }\
    \
    template<\
        typename TAcc = std::nullptr_t,\
        typename... TArgs,\
        typename std::enable_if< /* SFINAE: Enables if called from host. */ \
            std::is_same<\
                TAcc,\
                std::nullptr_t>::value,\
            int>::type = 0>\
    ALPAKA_FN_HOST\
    auto operator()(\
        TAcc const & acc,\
        TArgs const &... args ) const\
    -> decltype( STD_OP( args... ) )\
    {\
        alpaka::ignore_unused( acc );\
        return STD_OP( args... );\
    }\
    friend std::ostream & operator << (\
        std::ostream &out,\
        const NAME &op) \
    {\
        out << #NAME;\
        alpaka::ignore_unused( op ); \
        return out;\
    }\
};

ALPAKA_TEST_MATH_OP_FUNCTOR( OpAbs,
    Arity::UNARY,
    std::abs,
    alpaka::math::abs,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpAcos,
    Arity::UNARY,
    std::acos,
    alpaka::math::acos,
    Range::ONE_NEIGHBOURHOOD )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpAsin,
    Arity::UNARY,
    std::asin,
    alpaka::math::asin,
    Range::ONE_NEIGHBOURHOOD )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpAtan,
    Arity::UNARY,
    std::atan,
    alpaka::math::atan,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpCbrt,
    Arity::UNARY,
    std::cbrt,
    alpaka::math::cbrt,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpCeil,
    Arity::UNARY,
    std::ceil,
    alpaka::math::ceil,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpCos,
    Arity::UNARY,
    std::cos,
    alpaka::math::cos,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpErf,
    Arity::UNARY,
    std::erf,
    alpaka::math::erf,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpExp,
    Arity::UNARY,
    std::exp,
    alpaka::math::exp,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpFloor,
    Arity::UNARY,
    std::floor,
    alpaka::math::floor,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpLog,
    Arity::UNARY,
    std::log,
    alpaka::math::log,
    Range::POSITIVE_ONLY )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpRound,
    Arity::UNARY,
    std::round,
    alpaka::math::round,
    Range::UNRESTRICTED )
//ALPAKA_TEST_MATH_OP_FUNCTOR( OpRsqrt, Arity::UNARY, rsqr ,Range::UNRESTRICTED )
ALPAKA_TEST_MATH_OP_FUNCTOR( OpSin,
    Arity::UNARY,
    std::sin,
    alpaka::math::sin,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpSqrt,
    Arity::UNARY,
    std::sqrt,
    alpaka::math::sqrt,
    Range::POSITIVE_AND_ZERO )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpTan,
    Arity::UNARY,
    std::tan,
    alpaka::math::tan,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpTrunc,
    Arity::UNARY,
    std::trunc,
    alpaka::math::trunc,
    Range::UNRESTRICTED )

// All binary operators.
ALPAKA_TEST_MATH_OP_FUNCTOR( OpAtan2,
    Arity::BINARY,
    std::atan2,
    alpaka::math::atan2,
    Range::NOT_ZERO,
    Range::NOT_ZERO )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpFmod,
    Arity::BINARY,
    std::fmod,
    alpaka::math::fmod,
    Range::UNRESTRICTED,
    Range::NOT_ZERO )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpMax,
    Arity::BINARY,
    std::max,
    alpaka::math::max,
    Range::UNRESTRICTED,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpMin,
    Arity::BINARY,
    std::min,
    alpaka::math::min,
    Range::UNRESTRICTED,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpPow,
    Arity::BINARY,
    std::pow,
    alpaka::math::pow,
    Range::POSITIVE_AND_ZERO,
    Range::UNRESTRICTED )

ALPAKA_TEST_MATH_OP_FUNCTOR( OpRemainder,
    Arity::BINARY,
    std::remainder,
    alpaka::math::remainder,
    Range::UNRESTRICTED,
    Range::NOT_ZERO )


using BinaryFunctors = std::tuple<
    OpAtan2,
    OpFmod,
    OpMax,
    OpMin,
    OpPow,
  OpRemainder
    >;

using UnaryFunctors = std::tuple<
    OpAbs,
    OpAcos,
    OpAsin,
    OpAtan,
    OpCbrt,
    OpCeil,
    OpCos,
    OpErf,
    OpExp,
    OpFloor,
    OpLog,
    OpRound,
    OpSin,
    OpSqrt,
    OpTan,
    OpTrunc
    >;