add_subdirectory(data)

set(TEST_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${gtest_SOURCE_DIR}/include
  ${gmock_SOURCE_DIR}/include
  PARENT_SCOPE
)

set(TEST_SOURCE
  
)

include_directories(${TEST_INCLUDE_DIRECTORIES})

add_executable(CuEira_Test ${TEST_SOURCE})
target_link_libraries(CuEira_Test gmock gtest gtest_main gmock_main ${Boost_LIBRARIES} ${CUDA_LIBRARIES})
add_test(CuEira_Test ${CMAKE_CURRENT_BINARY_DIR}/CuEira_Test)
