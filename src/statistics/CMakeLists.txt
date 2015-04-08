add_subdirectory(factories)
add_subdirectory(interaction)

set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(GPU_SOURCE
  ${GPU_SOURCE}
  PARENT_SCOPE
)

set(SOURCE
  ${SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/ModelStatistics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OddsRatioStatistics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/AlleleStatistics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ContingencyTable.cpp
  PARENT_SCOPE
)

