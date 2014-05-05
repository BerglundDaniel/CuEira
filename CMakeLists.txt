cmake_minimum_required (VERSION 2.8)
project (CuEira)
enable_testing()

#The version number
set (CuEira_VERSION_MAJOR 1)
set (CuEira_VERSION_MINOR 0)

#Set various things
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(INCLUDE_DIRECTORIES
  ${PROJECT_BINARY_DIR}
)
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED OFF)
set(CUEIRA_LIB CuEira_lib)

#Options
option(VERBOSE "Set CMake and compilation to verbose" on) #Change to off later
set(PRECISION single) #single or double
set(TARGET_ARCHITECTURE fermi) #fermi or kepler
set(CPU on) #Change to of later

#Set things based on options
if(${VERBOSE})
  set (CMAKE_VERBOSE_MAKEFILE true)
  set (CUDA_VERBOSE_BUILD ON)
endif(${VERBOSE})

if(${CPU})
  set (PRECISION double)
  add_definitions(-DCPU=1)
endif(${CPU})

#Set preproccessor variables
add_definitions(-DCuEira_VERSION_MAJOR=${CuEira_VERSION_MAJOR})
add_definitions(-DCuEira_VERSION_MINOR=${CuEira_VERSION_MINOR})
#add_definitions ("-Wall") #Remove later or somethign

IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
  add_definitions(-DDEBUG=1)
endif (CMAKE_BUILD_TYPE MATCHES DEBUG)

if(${PRECISION} MATCHES single)
  add_definitions(-DPRECISION=float)
endif(${PRECISION} MATCHES single)
if(${PRECISION} MATCHES double)
  add_definitions(-DPRECISION=double)
endif(${PRECISION} MATCHES double)


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
MESSAGE( STATUS "TARGET_ARCHITECTURE = ${TARGET_ARCHITECTURE}" )
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
MESSAGE( STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}" )
MESSAGE( STATUS "CUDA_CUBLAS_LIBRARIES = ${CUDA_CUBLAS_LIBRARIES}" )
MESSAGE( STATUS "CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}" )
MESSAGE( STATUS "BOOST_PATH = ${Boost_INCLUDE_DIR}" )

#Add all sub dirs, the src sub dir will create librart ${CUEIRA_LIB} and set ${INCLUDE_DIRECTORIES}
add_subdirectory(gmock-1.7.0)
add_subdirectory(src)
include_directories(${INCLUDE_DIRECTORIES})

#Add executable and link
add_executable(CuEira ${CMAKE_CURRENT_SOURCE_DIR}/src/Main.cpp)
TARGET_LINK_LIBRARIES(CuEira ${CUEIRA_LIB} ${Boost_LIBRARIES} ${CUDA_LIBRARIES} )

#Tests
add_subdirectory(test)
include_directories(${TEST_INCLUDE_DIRECTORIES})

add_executable(CuEira_Test ${TEST_SOURCE})
target_link_libraries(CuEira_Test ${CUEIRA_LIB} gmock gtest gtest_main gmock_main ${Boost_LIBRARIES} ${CUDA_LIBRARIES})
add_test(CuEira_Test ${CMAKE_CURRENT_BINARY_DIR}/CuEira_Test)

#Check if bumblbee is used
#if(tests are enabled)

