set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(SOURCE
  ${SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/CpuPhenotypeHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CpuCovariatesHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CpuEnvironmentFactorHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CpuDataHandlerFactory.cpp
  PARENT_SCOPE
)
