cmake_minimum_required(VERSION 3.12)
project(NJSDF_FUN)

set(CMAKE_CXX_STANDARD 11)

# Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Development)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Python_INCLUDE_DIRS})

# Add your C++ source files here
set(SOURCES NJSDF.cpp)
add_compile_options(-O3 -fopenmp -fPIC)
add_link_options(-fopenmp)

# Create shared library in output directory
add_library(NJSDF_FUN MODULE ${SOURCES})
set_target_properties(NJSDF_FUN PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Link against Python library
target_link_libraries(NJSDF_FUN PRIVATE ${Python_LIBRARIES})
