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
  ${CMAKE_CURRENT_SOURCE_DIR}/ModelStatisticsFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/AlleleStatisticsFactory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ContingencyTableFactory.cpp
  PARENT_SCOPE
)

