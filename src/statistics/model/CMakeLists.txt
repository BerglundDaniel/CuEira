add_subdirectory(logisticRegression)

set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

if(${CPU})
  set(CPU_SOURCE
    #${CMAKE_CURRENT_SOURCE_DIR}/CpuModelHandler.cpp
  )
else()
  set(GPU_SOURCE
    ${GPU_SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/GpuModelHandler.cpp
    PARENT_SCOPE
  )
endif()

set(SOURCE
  ${SOURCE}
  ${CPU_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/ModelHandler.cpp
  PARENT_SCOPE
)
