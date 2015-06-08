add_subdirectory(mocks)
add_subdirectory(data)
add_subdirectory(container)
#add_subdirectory(model)
add_subdirectory(fileIO)
add_subdirectory(statistics)
add_subdirectory(task)
add_subdirectory(blas)

if(NOT ${CPU})
  add_subdirectory(cuda)
endif()

set(TEST_INCLUDE_DIRECTORIES
  ${TEST_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(TEST_SOURCE
  ${TEST_SOURCE}
  PARENT_SCOPE
)


