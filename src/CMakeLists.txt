add_subdirectory(exception)
add_subdirectory(data)
add_subdirectory(container)
add_subdirectory(configuration)
add_subdirectory(fileIO)
add_subdirectory(model)
add_subdirectory(statistics)
add_subdirectory(task)

if(NOT ${CPU})
  add_subdirectory(cuda)
endif()

if(${ENABLE_MKL})
  add_subdirectory(mkl)
endif()

set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(SOURCE
  ${SOURCE}
)

set(GPU_NVCC_SOURCE
  ${GPU_NVCC_SOURCE}
)

set(GPU_SOURCE
  ${GPU_SOURCE}
)

include_directories(${INCLUDE_DIRECTORIES})

add_library(${CUEIRA_LIB} SHARED ${SOURCE})
add_library(${CUEIRA_GPU_NVCC_LIB} SHARED ${SOURCE})
if(NOT ${CPU})
  cuda_add_library(${CUEIRA_GPU_NVCC_LIB} SHARED ${GPU_SOURCE})
endif(NOT ${CPU})
