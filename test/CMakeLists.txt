set(TEST_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${gtest_SOURCE_DIR}/include
)

set(TEST_HEADERS
  
)

add_subdirectory(data)

include_directories(${TEST_INCLUDE_DIRECTORIES})

add_executable(tests ${TEST_HEADERS})
target_link_libraries(tests gtest_main)
