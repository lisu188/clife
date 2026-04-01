# MIT License
#
# Copyright (c) 2019 Andrzej Lis
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.3)
project(clife)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake")

set(SOURCE_FILES
        clife.cpp
        clife.h)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -s")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.9)
    cmake_policy(SET CMP0069 NEW)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT CLIFE_IPO_SUPPORTED OUTPUT CLIFE_IPO_OUTPUT)
endif()

find_package(X11 REQUIRED)
if(NOT X11_Xext_FOUND)
    message(FATAL_ERROR "X11 Xext is required")
endif()

include_directories(${X11_INCLUDE_DIR})

add_executable(clife ${SOURCE_FILES})
target_link_libraries(clife ${X11_LIBRARIES} ${X11_Xext_LIB})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(clife PRIVATE "$<$<CONFIG:Release>:-march=native;-mtune=native>")
endif()
if(CLIFE_IPO_SUPPORTED)
    set_property(TARGET clife PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()
