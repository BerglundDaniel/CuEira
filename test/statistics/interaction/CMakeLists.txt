add_subdirectory(cpu)

if(NOT ${CPU})
  add_subdirectory(cuda)
endif()

set(TEST_INCLUDE_DIRECTORIES
  ${TEST_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(TEST_SOURCE
  ${TEST_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/InteractionStatisticsTest.cpp
  PARENT_SCOPE
)