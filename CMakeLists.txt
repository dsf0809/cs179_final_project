cmake_minimum_required(VERSION 3.10)
project(LinalgLib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build the unit tests" ON)
option(BUILD_EXAMPLES "Build example executables" ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(linalg STATIC
  src/linalg.cpp
)
target_include_directories(linalg
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(BUILD_TESTS)
  enable_testing()
  add_executable(linalg_test
    test/linalg_test.cpp
  )
  target_link_libraries(linalg_test PRIVATE linalg)
  add_test(NAME LinalgTest COMMAND linalg_test)
endif()

if(BUILD_EXAMPLES)
  add_executable(linalg_demo
    demo/main.cpp
  )
  target_link_libraries(linalg_demo PRIVATE linalg)
endif()