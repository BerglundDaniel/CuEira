add_subdirectory(cpu)
add_subdirectory(factories)

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
  ${CMAKE_CURRENT_SOURCE_DIR}/EnvironmentVector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SNPVector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/InteractionVector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PhenotypeVector.cpp
  PARENT_SCOPE
)

