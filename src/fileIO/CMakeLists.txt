add_subdirectory(writers)
add_subdirectory(readers)

set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(SOURCE
  ${SOURCE}
  PARENT_SCOPE
)
