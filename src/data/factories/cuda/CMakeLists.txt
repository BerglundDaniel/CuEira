set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(SOURCE
  ${SOURCE}
  PARENT_SCOPE
)

set(GPU_SOURCE
  ${GPU_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/CudaPhenotypeHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CudaCovariatesHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CudaEnvironmentFactorHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CudaDataHandlerFactory.cpp
  PARENT_SCOPE
)
