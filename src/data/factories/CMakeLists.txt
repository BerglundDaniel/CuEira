add_subdirectory(cpu)

if(NOT ${CPU})
  add_subdirectory(cuda)
endif()

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
  ${CMAKE_CURRENT_SOURCE_DIR}/PhenotypeHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PersonHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/EnvironmentFactorHandlerFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DataHandlerFactory.cpp
  PARENT_SCOPE
)
