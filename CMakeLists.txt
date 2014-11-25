cmake_minimum_required (VERSION 2.8 FATAL_ERROR)
project (CuEira)
enable_testing()

#The version number
set (CuEira_VERSION_MAJOR 0)
set (CuEira_VERSION_MINOR 2)

#Set various things
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(INCLUDE_DIRECTORIES
  ${PROJECT_BINARY_DIR}
)
set(TEST_DATA_DIR ${PROJECT_BINARY_DIR}/data)
set(CUEIRA_LIB CuEira_Core)
set(CUEIRA_GPU_LIB CuEira_GPU)
set(CUEIRA_GPU_NVCC_LIB CuEira_NVCC_GPU)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/") #Sets another directory to look for find_package files
set(Boost_USE_STATIC_LIBS off) 
set(Boost_USE_MULTITHREADED off)

#Options
option(VERBOSE "Set CMake and compilation to verbose" off)
set(PRECISION single) #single or double
set(TARGET_ARCHITECTURE kepler) #fermi or kepler #TODO probably remove this later
option(CPU "Use CPU only" off)
option(CLUSTER "Use cluster" off) #TODO not implemented
option(ENABLE_MKL "Enable MKL flag" on)
option(PROFILE "Profile" on)

#Set things based on options
if(${VERBOSE})
  set (CMAKE_VERBOSE_MAKEFILE true)
  set (CUDA_VERBOSE_BUILD ON)
endif(${VERBOSE})

if(${CPU})
  add_definitions(-DCPU=1)
endif(${CPU})
  
if(${PROFILE})
  add_definitions(-DPROFILE=1)
endif()

if(${CLUSTER})
  add_definitions(-DCLUSTER=1)
endif(${CLUSTER})

if(${TARGET_ARCHITECTURE} MATCHES fermi)
  add_definitions(-DFERMI=1)
endif()

MESSAGE( STATUS "Building CuEira:" )

#Set preproccessor variables
add_definitions(-DCuEira_VERSION_MAJOR=${CuEira_VERSION_MAJOR})
add_definitions(-DCuEira_VERSION_MINOR=${CuEira_VERSION_MINOR})
add_definitions(-DCuEira_BUILD_DIR="${TEST_DATA_DIR}")

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  MESSAGE( STATUS "Debug build." )
  add_definitions(-DDEBUG=1)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

if(${PRECISION} MATCHES single)
  add_definitions(-DPRECISION=float)
  add_definitions(-DSINGLEPRECISION=1)
endif(${PRECISION} MATCHES single)
if(${PRECISION} MATCHES double)
  add_definitions(-DPRECISION=double)
  add_definitions(-DDOUBLEPRECISION=1)
endif(${PRECISION} MATCHES double)

#Find required libraries
MESSAGE( STATUS "Checking packages:" )

find_package( Boost 1.54 COMPONENTS program_options filesystem chrono REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

#if boost 1.55 and intel, print warning TODO

#Some optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-alias")

if(${CPU})

else()
  find_package( CUDA REQUIRED )
  include_directories( ${CUDA_INCLUDE_DIRS} )

  #Checks that the CUDA version is correct
  if(${CUDA_VERSION_MAJOR} LESS 5)
    message( FATAL_ERROR "Requires CUDA version 5.5 or later" )
  endif(${CUDA_VERSION_MAJOR} LESS 5)
  if(${CUDA_VERSION_MAJOR} EQUAL 5)
    if(${CUDA_VERSION_MINOR} LESS 5)
      message( FATAL_ERROR "Requires CUDA version 5.5 or later" )
    endif(${CUDA_VERSION_MINOR} LESS 5)
  endif(${CUDA_VERSION_MAJOR} EQUAL 5)

  #Set somethings for CUDA compilation
  set(CUDA_PROPAGATE_HOST_FLAGS off) #Some c++ flags that gets passed to NVCC -Xcompiler will cause errors, for instance c++11 flag

  if(${TARGET_ARCHITECTURE} MATCHES fermi)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
  else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20) #TODO change to 30 later
  endif()

  if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g)
  endif()

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-restrict --use_fast_math)

  #Wrong place FIXME
  if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
    if(ENABLE_MKL)
      #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=sequential")
    else()
      message( FATAL_ERROR "Have to use MKL+intel" )
    endif()
  endif()

endif(${CPU})

if(${CLUSTER})
#FIND MPI TODO
endif(${CLUSTER})

#Print some info
MESSAGE( STATUS "change a configuration variable with: cmake -D<Variable>=<Value>" )
MESSAGE( STATUS "PRECISION = ${PRECISION}" )
#MESSAGE( STATUS "TARGET_ARCHITECTURE = ${TARGET_ARCHITECTURE}" )
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
MESSAGE( STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}" )
MESSAGE( STATUS "Boost_INCLUDE_DIR = ${Boost_INCLUDE_DIR}" )
MESSAGE( STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}" )
if(${CPU})
  MESSAGE( STATUS "LAPACKPP_INCLUDE_DIR = ${LAPACKPP_INCLUDE_DIR}" )
  MESSAGE( STATUS "BLAS_LIBRARIES = ${BLAS_LIBRARIES}" )
else()
  MESSAGE( STATUS "CUDA_CUBLAS_LIBRARIES = ${CUDA_CUBLAS_LIBRARIES}" )
  MESSAGE( STATUS "CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}" )
endif(${CPU})

#TODO message abou blas

if(${CLUSTER})
#TODO message mpi
endif(${CLUSTER})

#Add all sub dirs, the src sub dir will create librart ${CUEIRA_LIB} and set ${INCLUDE_DIRECTORIES}
add_subdirectory(gmock-1.7.0)
set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${gtest_SOURCE_DIR}/include
  ${gmock_SOURCE_DIR}/include
)

add_subdirectory(src)
include_directories(${INCLUDE_DIRECTORIES})

add_subdirectory(test)
include_directories(${TEST_INCLUDE_DIRECTORIES})

#Add executable and link
add_executable(CuEira ${CMAKE_CURRENT_SOURCE_DIR}/src/Main.cpp)
if(${CPU})
  TARGET_LINK_LIBRARIES(CuEira ${CUEIRA_LIB} gmock gtest gtest_main gmock_main ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${BLAS_LIBRARIES} ${MKL_LIBRARIES})
else()
  TARGET_LINK_LIBRARIES(CuEira ${CUEIRA_LIB} ${CUEIRA_GPU_LIB} ${CUEIRA_GPU_NVCC_LIB} gmock gtest gtest_main gmock_main ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${BLAS_LIBRARIES} ${MKL_LIBRARIES})
  cuda_add_cublas_to_target(CuEira)
endif(${CPU})

#Tests
add_executable(CuEira_Test ${TEST_SOURCE})

if(${CPU})
  target_link_libraries(CuEira_Test ${CUEIRA_LIB} gmock gtest gtest_main gmock_main ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${BLAS_LIBRARIES} ${MKL_LIBRARIES})
else()
  target_link_libraries(CuEira_Test ${CUEIRA_LIB} ${CUEIRA_GPU_LIB} ${CUEIRA_GPU_NVCC_LIB} gmock gtest gtest_main gmock_main ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${BLAS_LIBRARIES} ${MKL_LIBRARIES})
  cuda_add_cublas_to_target(CuEira_Test)
endif(${CPU})

add_test(CuEira_Test ${CMAKE_CURRENT_BINARY_DIR}/CuEira_Test)

#Copy the dir with the data for the tests to the build directory
file(MAKE_DIRECTORY ${TEST_DATA_DIR})
add_custom_command(TARGET CuEira_Test PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/data ${TEST_DATA_DIR})

