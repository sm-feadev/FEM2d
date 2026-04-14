cmake_minimum_required(VERSION 3.16)
project(mitcs6 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE) set(CMAKE_BUILD_TYPE Release) endif()
add_compile_options(-O2 -Wall -Wextra)

find_package(Eigen3 3.4 QUIET)
if(NOT TARGET Eigen3::Eigen)
  find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Dense
    HINTS /usr/include/eigen3 /usr/local/include/eigen3
          ${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen)
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 not found. Install libeigen3-dev or set EIGEN3_INCLUDE_DIR.")
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

find_library(GMSH_LIB NAMES gmsh
  HINTS /usr/lib /usr/local/lib ${CMAKE_CURRENT_SOURCE_DIR}/extern/gmsh/lib)
find_path(GMSH_INCLUDE_DIR NAMES gmsh.h
  HINTS /usr/include /usr/local/include
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/gmsh/include)
if(NOT GMSH_LIB OR NOT GMSH_INCLUDE_DIR)
  message(FATAL_ERROR "gmsh not found. Install libgmsh-dev or set GMSH_LIB/GMSH_INCLUDE_DIR.\n  SDK: https://gmsh.info/#Download")
endif()

add_executable(mitcs6 main.cpp mitcs6.cpp input_parser.cpp)
target_include_directories(mitcs6 PRIVATE ${GMSH_INCLUDE_DIR})
target_link_libraries(mitcs6 PRIVATE Eigen3::Eigen ${GMSH_LIB})
if(UNIX AND NOT APPLE)
  target_link_libraries(mitcs6 PRIVATE dl m)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plate.inp
               ${CMAKE_CURRENT_BINARY_DIR}/plate.inp COPYONLY)

message(STATUS "Build : ${CMAKE_BUILD_TYPE}")
message(STATUS "Eigen : ${EIGEN3_INCLUDE_DIR}")
message(STATUS "gmsh  : ${GMSH_LIB}")
