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
  ${CMAKE_CURRENT_SOURCE_DIR}/CudaMissingDataHandler.cpp
  PARENT_SCOPE
)
