set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(SOURCE
  ${SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/InteractionStatistics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ModelStatistics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OddsRatioStatistics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ModelStatisticsFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/AlleleStatistics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/AlleleStatisticsFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ContingencyTable.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ContingencyTableFactory.cpp
  PARENT_SCOPE
)

