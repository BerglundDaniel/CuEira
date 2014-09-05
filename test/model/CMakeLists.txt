add_subdirectory(logisticRegression)

set(TEST_INCLUDE_DIRECTORIES
  ${TEST_INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

if(${CPU})
  set(CPU_TEST_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/CpuModelHandlerTest.cpp
  )
else()
  set(GPU_TEST_SOURCE
    #${CMAKE_CURRENT_SOURCE_DIR}/GpuModelHandlerTest.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/GpuModelHandlerLRIntegrationTest.cpp
  )
endif(${CPU})

set(TEST_SOURCE
  ${TEST_SOURCE}
  ${GPU_TEST_SOURCE}
  ${CPU_TEST_SOURCE}
  PARENT_SCOPE
)
