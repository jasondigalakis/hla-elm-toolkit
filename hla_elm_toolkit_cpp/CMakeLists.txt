cmake_minimum_required(VERSION 3.15)
project(hla_elm_toolkit_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Header-only library target (no external dependencies: everything
# needed, including the dense-matrix linear algebra, is implemented in
# include/hla_elm/*.hpp).
add_library(hla_elm INTERFACE)
target_include_directories(hla_elm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(run_demo_comparison examples/run_demo_comparison.cpp)
target_link_libraries(run_demo_comparison PRIVATE hla_elm)

enable_testing()
add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic PRIVATE hla_elm)
add_test(NAME test_basic COMMAND test_basic)
