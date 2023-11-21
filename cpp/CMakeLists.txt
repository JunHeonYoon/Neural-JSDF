project(NJSDF_cpp)
cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)

LIST( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" )
MESSAGE( STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH} )

find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(RBDL REQUIRED)
find_package(Threads REQUIRED)

include_directories(SYSTEM 
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${RBDL_INCLUDE_DIR}
    include
)

set(SOURCES
    src/main.cpp
    src/NJSDF.cpp
    src/robot.cpp
    src/controller.cpp
    )

add_executable(NJSDF_cpp ${SOURCES})

# add_library(${PROJECT_NAME} src/NJSDF.cpp)

target_link_libraries(NJSDF_cpp 
    ${catkin_LIBRARIES}
    OsqpEigen::OsqpEigen
    Threads::Threads
    ${RBDL_LIBRARY}
    ${RBDL_URDFReader_LIBRARY})
