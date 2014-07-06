if(${CPU})
  add_subdirectory(serial)
else()
  add_subdirectory(cuda)
endif()

set(TEST_INCLUDE_DIRECTORIES
  ${TEST_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(TEST_SOURCE
  ${TEST_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/LogisticRegressionResultTest.cpp
  PARENT_SCOPE
)

