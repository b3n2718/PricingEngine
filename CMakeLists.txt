# CMakeLists.txt  — für C++ IDE Support und Tests
cmake_minimum_required(VERSION 3.20)
project(mc_engine)

set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 REQUIRED)

pybind11_add_module(mc_core
    cpp/src/rng/pseudo_rng.cpp
    cpp/src/rng/sobol_rng.cpp
    cpp/src/processes/gbm.cpp
    cpp/src/processes/heston.cpp
    cpp/src/path_generator.cpp
    cpp/src/utils/cholesky.cpp
    cpp/bindings/bindings.cpp
)

target_include_directories(mc_core PRIVATE cpp/include)
target_compile_options(mc_core PRIVATE -O3 -march=native)