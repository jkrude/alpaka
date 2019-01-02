#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

source ./script/travis/set.sh

#-------------------------------------------------------------------------------
# CMake
ALPAKA_CI_CMAKE_VER_SEMANTIC=( ${ALPAKA_CI_CMAKE_VER//./ } )
export ALPAKA_CI_CMAKE_VER_MAJOR="${ALPAKA_CI_CMAKE_VER_SEMANTIC[0]}"
echo ALPAKA_CI_CMAKE_VER_MAJOR: "${ALPAKA_CI_CMAKE_VER_MAJOR}"
export ALPAKA_CI_CMAKE_VER_MINOR="${ALPAKA_CI_CMAKE_VER_SEMANTIC[1]}"
echo ALPAKA_CI_CMAKE_VER_MINOR: "${ALPAKA_CI_CMAKE_VER_MINOR}"

#-------------------------------------------------------------------------------
# gcc
if [ "${CXX}" == "g++" ]
then
    ALPAKA_CI_GCC_VER_SEMANTIC=( ${ALPAKA_CI_GCC_VER//./ } )
    export ALPAKA_CI_GCC_VER_MAJOR="${ALPAKA_CI_GCC_VER_SEMANTIC[0]}"
    echo ALPAKA_CI_GCC_VER_MAJOR: "${ALPAKA_CI_GCC_VER_MAJOR}"
    if [[ -v ALPAKA_CI_GCC_VER_SEMANTIC[1] ]]
    then
        export ALPAKA_CI_GCC_VER_MINOR="${ALPAKA_CI_GCC_VER_SEMANTIC[1]}"
    else
        export ALPAKA_CI_GCC_VER_MINOR=0
    fi
    echo ALPAKA_CI_GCC_VER_MINOR: "${ALPAKA_CI_GCC_VER_MINOR}"
fi

#-------------------------------------------------------------------------------
# clang
if [ "${CXX}" == "clang++" ]
then
    ALPAKA_CI_CLANG_VER_SEMANTIC=( ${ALPAKA_CI_CLANG_VER//./ } )
    export ALPAKA_CI_CLANG_VER_MAJOR="${ALPAKA_CI_CLANG_VER_SEMANTIC[0]}"
    echo ALPAKA_CI_CLANG_VER_MAJOR: "${ALPAKA_CI_CLANG_VER_MAJOR}"
    export ALPAKA_CI_CLANG_VER_MINOR="${ALPAKA_CI_CLANG_VER_SEMANTIC[1]}"
    echo ALPAKA_CI_CLANG_VER_MINOR: "${ALPAKA_CI_CLANG_VER_MINOR}"
fi

#-------------------------------------------------------------------------------
# Boost.
export ALPAKA_CI_BOOST_BRANCH_MAJOR=${ALPAKA_CI_BOOST_BRANCH:6:1}
echo ALPAKA_CI_BOOST_BRANCH_MAJOR: "${ALPAKA_CI_BOOST_BRANCH_MAJOR}"
export ALPAKA_CI_BOOST_BRANCH_MINOR=${ALPAKA_CI_BOOST_BRANCH:8:2}
echo ALPAKA_CI_BOOST_BRANCH_MINOR: "${ALPAKA_CI_BOOST_BRANCH_MINOR}"

#-------------------------------------------------------------------------------
if [ "${ALPAKA_CI_STDLIB}" == "libc++" ]
then
    if [ "${CXX}" == "g++" ]
    then
        echo "using libc++ with g++ not yet supported."
        exit 1
    fi

    if [ "${ALPAKA_CI_DOCKER_BASE_IMAGE_NAME}" == "ubuntu:14.04" ]
    then
        echo "using libc++ with ubuntu:14.04 not supported."
        exit 1
    fi

    if (( ( ( "${ALPAKA_CI_BOOST_BRANCH_MAJOR}" == 1 ) && ( "${ALPAKA_CI_BOOST_BRANCH_MINOR}" < 65 ) ) || ( "${ALPAKA_CI_BOOST_BRANCH_MAJOR}" < 1 ) ))
    then
        echo "using libc++ with boost < 1.65 is not supported."
        exit 1
    fi
fi

#-------------------------------------------------------------------------------
# CUDA
if [ "${ALPAKA_ACC_GPU_CUDA_ENABLE}" == "ON" ] || [ "${ALPAKA_ACC_GPU_HIP_ENABLE}" == "ON" ] && [ "${ALPAKA_HIP_PLATFORM}" == "nvcc" ]
then
    if [ "${ALPAKA_CUDA_COMPILER}" == "nvcc" ]
    then
        # FIXME: BOOST_AUTO_TEST_CASE_TEMPLATE is not compilable with nvcc in Release mode.
        if [ "${CMAKE_BUILD_TYPE}" == "Release" ]
        then
            export CMAKE_BUILD_TYPE=Debug
        fi
    fi
fi

#-------------------------------------------------------------------------------
# HIP
if [ "${ALPAKA_ACC_GPU_HIP_ENABLE}" == "ON" ]
then
    # if platform is nvcc, CUDA part is already processed in this file.

    if [ "${ALPAKA_HIP_PLATFORM}" == "hcc" ]
    then
        echo "HIP(hcc) not supported yet."
        exit 1
    fi
fi
