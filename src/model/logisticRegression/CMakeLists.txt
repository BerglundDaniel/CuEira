if(${CPU})
  add_subdirectory(serial)
else()
  add_subdirectory(cuda)
endif(${CPU})

set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(GPU_SOURCE
  ${GPU_SOURCE}
  PARENT_SCOPE
)

set(GPU_NVCC_SOURCE
  ${GPU_NCC_SOURCE}
  PARENT_SCOPE
)

set(SOURCE
  ${SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/LogisticRegressionResult.cpp
  PARENT_SCOPE
)

