add_subdirectory(kernels)
add_subdirectory(logisticRegression)

set(TEST_INCLUDE_DIRECTORIES
  ${TEST_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)


set(TEST_SOURCE
  ${TEST_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/CudaAdapterTest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TransferTest.cpp
  PARENT_SCOPE
)
