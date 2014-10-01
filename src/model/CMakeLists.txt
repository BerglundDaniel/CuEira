add_subdirectory(logisticRegression)
add_subdirectory(handlers)

set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(GPU_SOURCE
  ${GPU_SOURCE}
  PARENT_SCOPE
)

set(GPU_NVCC_SOURCE
  ${GPU_NCC_SOURCE}
  PARENT_SCOPE
)

set(SOURCE
  ${SOURCE}
  ${CPU_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/Model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ModelResult.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ModelConfiguration.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ModelInformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ModelInformationWithTable.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CombinedResults.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CombinedResultsLogisticRegression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ModelInformationFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CombinedResultsFactory.cpp
  PARENT_SCOPE
)
