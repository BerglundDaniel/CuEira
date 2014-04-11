cmake_minimum_required (VERSION 2.8)
project (CuEira)

#The version number
set (CuEira_VERSION_MAJOR 1)
set (CuEira_VERSION_MINOR 0)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
include_directories("${PROJECT_BINARY_DIR}")

#Options
option(PRECISION "Set the precision for the program, either single or double" single) #FIXME
option(VERBOSE "Set CMake and compilation to verbose" on) #Change to off later

IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
  add_definitions(-DDEBUG=1)
endif (CMAKE_BUILD_TYPE MATCHES "DEBUG")

if(${VERBOSE} EQUAL ON)
  set (CMAKE_VERBOSE_MAKEFILE true)
  set (CUDA_VERBOSE_BUILD ON)
endif(${VERBOSE} EQUAL ON)

MESSAGE( STATUS "Building CuEIRA:" )

#Find required libraries
MESSAGE( STATUS "Checking packages:" )
find_package( Boost 1.55 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

find_package( CUDA REQUIRED )
include_directories( ${CUDA_INCLUDE_DIR} )

#Checks that the CUDA version is correct
if(${CUDA_VERSION_MAJOR} LESS 5)
  message( FATAL_ERROR "Requires CUDA version 5.5 or later" )
endif(${CUDA_VERSION_MAJOR} LESS 5)
if(${CUDA_VERSION_MAJOR} EQUAL 5)
  if(${CUDA_VERSION_MINOR} LESS 5)
    message( FATAL_ERROR "Requires CUDA version 5.5 or later" )
  endif(${CUDA_VERSION_MINOR} LESS 5)
endif(${CUDA_VERSION_MAJOR} EQUAL 5)

#Print some info
MESSAGE( STATUS "change a configuration variable with: cmake -D<Variable>=<Value>" )
MESSAGE( STATUS "PRECISION = ${PRECISION}" )
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
MESSAGE( STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}" )
MESSAGE( STATUS "GOOGLE_TEST_PATH = ${GOOGLE_TEST_PATH}" )
MESSAGE( STATUS "CUDA_CUBLAS_LIBRARIES = ${CUDA_CUBLAS_LIBRARIES}" )
MESSAGE( STATUS "CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}" )
MESSAGE( STATUS "BOOST_PATH = ${Boost_INCLUDE_DIR}" )

#Set preproccessor variables
add_definitions(-DCuEira_VERSION_MAJOR=${CuEira_VERSION_MAJOR})
add_definitions(-DCuEira_VERSION_MINOR=${CuEira_VERSION_MINOR})

#Compile
add_subdirectory(src)

#Tests
enable_testing()

#Check if bumblbee is used
#if(tests are enabled)

