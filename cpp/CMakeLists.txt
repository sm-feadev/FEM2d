cmake_minimum_required(VERSION 3.16)
project(mitcs6 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Release flags ─────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O2 -Wall -Wextra)

# ── Eigen (header-only) ───────────────────────────────────────────────────────
# If installed system-wide:  apt install libeigen3-dev
# Or set EIGEN3_INCLUDE_DIR to a local copy.
find_package(Eigen3 3.4 QUIET)
if(TARGET Eigen3::Eigen)
  message(STATUS "Found Eigen3 via CMake package")
else()
  # Fallback: look for eigen3 include dir
  find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Dense
    HINTS /usr/include/eigen3 /usr/local/include/eigen3
          ${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen)
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR
      "Eigen3 not found. Install libeigen3-dev or set EIGEN3_INCLUDE_DIR.")
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
  message(STATUS "Eigen3 headers: ${EIGEN3_INCLUDE_DIR}")
endif()

# ── gmsh (shared library) ─────────────────────────────────────────────────────
# Typical system install: apt install libgmsh-dev
# Or set GMSH_LIB and GMSH_INCLUDE_DIR for a local SDK.
find_library(GMSH_LIB NAMES gmsh
  HINTS /usr/lib /usr/local/lib ${CMAKE_CURRENT_SOURCE_DIR}/extern/gmsh/lib)
find_path(GMSH_INCLUDE_DIR NAMES gmsh.h
  HINTS /usr/include /usr/local/include
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/gmsh/include)

if(NOT GMSH_LIB OR NOT GMSH_INCLUDE_DIR)
  message(FATAL_ERROR
    "gmsh not found. Install libgmsh-dev or set GMSH_LIB / GMSH_INCLUDE_DIR.\n"
    "  Download SDK: https://gmsh.info/#Download")
endif()
message(STATUS "gmsh library : ${GMSH_LIB}")
message(STATUS "gmsh headers : ${GMSH_INCLUDE_DIR}")

# ── executable ───────────────────────────────────────────────────────────────
add_executable(mitcs6 main.cpp mitcs6.cpp)
target_include_directories(mitcs6 PRIVATE ${GMSH_INCLUDE_DIR})
target_link_libraries(mitcs6 PRIVATE Eigen3::Eigen ${GMSH_LIB})

# On Linux, gmsh links against libdl and libm
if(UNIX AND NOT APPLE)
  target_link_libraries(mitcs6 PRIVATE dl m)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target: mitcs6")
