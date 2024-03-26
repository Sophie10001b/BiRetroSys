cmake_minimum_required(VERSION 3.21)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(PRJ BiRetroSys)
project(${PRJ})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")

message(STATUS "current dir: ${PROJECT_SOURCE_DIR}")
message(STATUS "cmake_flags: ${CMAKE_CXX_FLAGS}")

if(DEFINED USE_PARALLEL)
    unset(USE_PARALLEL CACHE)
endif()

option(USE_PARALLEL "Use OpenMP for parallel programming" OFF)

project(${PRJ})

# need to set your own pkgconfig directory
set(ENV{PKG_CONFIG_PATH} "/opt/homebrew/lib/pkgconfig")

find_package(Eigen3 REQUIRED)
find_package(RDKit REQUIRED)

# for Inference
find_package(ONNXRUNTIME REQUIRED)

# for SearchTree
find_package(PkgConfig REQUIRED)
pkg_check_modules(graphviz REQUIRED IMPORTED_TARGET libgvc libcgraph)

if(USE_PARALLEL)
    message(STATUS "Enable OpenMP Parallel")
    find_package(OpenMP REQUIRED)
else()
    message(STATUS "Disable OpenMP Parallel")
endif()

add_subdirectory(MolHandler)
add_subdirectory(Inference)
add_subdirectory(Search)
add_subdirectory(Test)

# add_executable(${PRJ} "${PROJECT_SOURCE_DIR}/Search/src/valueFunction.cpp")
# add_executable(${PRJ} "${PROJECT_SOURCE_DIR}/Test/src/search_test.cpp")
add_executable(${PRJ} "get_search.cpp")

target_link_libraries(${PRJ} PRIVATE MolHandler Inference Search Test)