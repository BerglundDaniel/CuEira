cmake_minimum_required (VERSION 2.8)
project (CuEira)
#The version number.
set (CuEira_VERSION_MAJOR 1)
set (CuEira_VERSION_MINOR 0)

set (CMAKE_VERBOSE_MAKEFILE true)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

#Configure a header file to pass some of the CMake settings
#to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/config/CuEiraConfig.h.in"
  "${PROJECT_BINARY_DIR}/config/CuEiraConfig.h"
  )
include_directories("${PROJECT_BINARY_DIR}")

#Find required libraries
find_package( Boost 1.55 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

find_package( CUDA 5.5 REQUIRED )
include_directories( ${Cuda_INCLUDE_DIR} )
 
#Add the executable
add_executable(CuEira ${PROJECT_SOURCE_DIR}/src/Main.cpp)

#Link
TARGET_LINK_LIBRARIES( CuEira ${Boost_LIBRARIES} )






#ADD_EXECUTABLE( anyExecutable myMain.cpp )


