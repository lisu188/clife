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

option(CLIFE_ENABLE_NATIVE "Build the release binaries with -march=native -mtune=native on GCC/Clang" ON)
option(CLIFE_ENABLE_IPO "Enable interprocedural optimization in Release when supported" OFF)
option(CLIFE_ENABLE_TESTS "Build the correctness test executable" ON)
option(CLIFE_ENABLE_STRIP "Strip Release binaries" ON)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
if(CLIFE_ENABLE_STRIP)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.9)
    cmake_policy(SET CMP0069 NEW)
endif()

find_package(Threads REQUIRED)
find_package(X11 REQUIRED)
if(NOT X11_Xext_FOUND)
    message(FATAL_ERROR "X11 Xext is required")
endif()

if(CLIFE_ENABLE_IPO AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.9)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT CLIFE_IPO_SUPPORTED OUTPUT CLIFE_IPO_OUTPUT)
endif()

add_executable(clife
        clife.cpp
        clife.h)
target_link_libraries(clife PRIVATE Threads::Threads ${X11_LIBRARIES} ${X11_Xext_LIB})
target_include_directories(clife PRIVATE ${X11_INCLUDE_DIR})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CLIFE_ENABLE_NATIVE)
        target_compile_options(clife PRIVATE "$<$<CONFIG:Release>:-march=native;-mtune=native>")
    endif()
endif()
if(CLIFE_ENABLE_IPO AND CLIFE_IPO_SUPPORTED)
    set_property(TARGET clife PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(CLIFE_ENABLE_TESTS)
    enable_testing()
    add_executable(clife_tests
            clife_tests.cpp
            clife.cpp
            clife.h)
    target_compile_definitions(clife_tests PRIVATE CLIFE_NO_MAIN)
    target_link_libraries(clife_tests PRIVATE Threads::Threads ${X11_LIBRARIES} ${X11_Xext_LIB})
    target_include_directories(clife_tests PRIVATE ${X11_INCLUDE_DIR})
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        if(CLIFE_ENABLE_NATIVE)
            target_compile_options(clife_tests PRIVATE "$<$<CONFIG:Release>:-march=native;-mtune=native>")
        endif()
    endif()
    if(CLIFE_ENABLE_IPO AND CLIFE_IPO_SUPPORTED)
        set_property(TARGET clife_tests PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
    add_test(NAME clife_tests COMMAND clife_tests)
endif()
