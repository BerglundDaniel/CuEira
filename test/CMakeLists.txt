add_subdirectory(mocks)
add_subdirectory(data)
add_subdirectory(container)
add_subdirectory(fileIO)

if(${CPU})

else()
  add_subdirectory(cuda)
endif(${CPU})

set(TEST_INCLUDE_DIRECTORIES
  ${TEST_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(TEST_SOURCE
  ${TEST_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/ConstructorHelpers.cpp
  PARENT_SCOPE
)


