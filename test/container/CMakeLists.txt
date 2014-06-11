add_subdirectory(data)

set(TEST_INCLUDE_DIRECTORIES
  ${TEST_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

if(${CPU})
  set(CPU_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/LapackppHostVectorTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LapackppHostMatrixTest.cpp
  )
else()
  set(GPU_LOCAL_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/PinnedHostVectorTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PinnedHostMatrixTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DeviceVectorTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DeviceMatrixTest.cpp
  )
endif(${CPU})

set(TEST_SOURCE
  ${TEST_SOURCE}
  ${CPU_SOURCE}
  ${GPU_LOCAL_SOURCE}
  PARENT_SCOPE
)
