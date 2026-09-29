cmake_minimum_required(VERSION 3.16)

project(mpi_conjugate_gradient VERSION 1.0.0 LANGUAGES C)

find_package(MPI REQUIRED COMPONENTS C)

add_executable(gradient src/gradient.c)

target_compile_features(gradient PRIVATE c_std_11)
target_link_libraries(gradient PRIVATE MPI::MPI_C)

if(MSVC)
    target_compile_options(gradient PRIVATE /W4)
else()
    target_compile_options(gradient PRIVATE -Wall -Wextra -Wpedantic)
endif()
