cmake_minimum_required(VERSION 3.24)
project(hello_avx)

set(CMAKE_CXX_STANDARD 17)

add_executable(hello_avx main.cpp)

include_directories(Common)

add_subdirectory(Ch01)
add_subdirectory(Ch02)
