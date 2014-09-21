add_subdirectory(logisticRegression)
add_subdirectory(handlers)

set(TEST_INCLUDE_DIRECTORIES
  ${TEST_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(TEST_SOURCE
  ${TEST_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/ModelInformationTest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ModelInformationFactoryTest.cpp
  PARENT_SCOPE
)
