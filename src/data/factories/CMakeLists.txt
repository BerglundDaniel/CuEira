set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(SOURCE
  ${SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/PersonHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PhenotypeHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/EnvironmentFactorHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DataHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CovariatesHandlerFactory.cpp
  PARENT_SCOPE
)
