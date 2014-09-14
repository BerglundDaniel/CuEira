add_subdirectory(transfer)
add_subdirectory(kernels)

set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

if(NOT ${CPU})
  set(GPU_SOURCE
    ${GPU_SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StreamFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/KernelWrapperFactory.cpp
  )
endif()


set(SOURCE
  ${SOURCE}
  PARENT_SCOPE
)

set(GPU_NVCC_SOURCE
  ${GPU_NVCC_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/CudaAdapter.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/KernelWrapper.cu
  PARENT_SCOPE
)
