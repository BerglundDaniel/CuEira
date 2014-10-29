add_subdirectory(data)

set(TEST_INCLUDE_DIRECTORIES
  ${TEST_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

if(NOT ${CPU})
  set(GPU_LOCAL_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/PinnedHostVectorTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PinnedHostMatrixTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DeviceVectorTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DeviceMatrixTest.cpp
  )
endif()

set(TEST_SOURCE
  ${TEST_SOURCE}
  ${GPU_LOCAL_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/RegularHostVectorTest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/RegularHostMatrixTest.cpp
  PARENT_SCOPE
)
