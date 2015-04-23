add_subdirectory(factories)
add_subdirectory(interaction)

set(TEST_INCLUDE_DIRECTORIES
  ${TEST_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(TEST_SOURCE
  ${TEST_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/OddsRatioStatisticsTest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/AlleleStatisticsTest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ContingencyTableTest.cpp
  PARENT_SCOPE
)
