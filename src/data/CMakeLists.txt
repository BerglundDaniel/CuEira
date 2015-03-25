add_subdirectory(enums)
add_subdirectory(factories)
add_subdirectory(cpu)
add_subdirectory(basic)

if(NOT ${CPU})
  add_subdirectory(cuda)
endif()

set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(SOURCE
  ${SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/PersonHandler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/EnvironmentFactorHandler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DataHandler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PhenotypeHandler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CovariatesMatrixHandler.cpp
  PARENT_SCOPE
)
