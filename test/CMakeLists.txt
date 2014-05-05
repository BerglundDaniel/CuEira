add_subdirectory(data)

set(TEST_INCLUDE_DIRECTORIES
  ${TEST_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${gtest_SOURCE_DIR}/include
  ${gmock_SOURCE_DIR}/include
  PARENT_SCOPE
)

set(TEST_SOURCE
  ${TEST_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/ConstructorHelpers.cpp
  PARENT_SCOPE
)


