add_subdirectory(cpu)

if(NOT ${CPU})
  add_subdirectory(cuda)
endif()

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
  ${CMAKE_CURRENT_SOURCE_DIR}/LogisticRegression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LogisticRegressionConfiguration.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LogisticRegressionResult.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LogisticRegressionCombinedResults.cpp
  PARENT_SCOPE
)

