add_subdirectory(data)

set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

if(${CPU})
  set(CPU_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/HostVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HostMatrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LapackppHostVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LapackppHostMatrix.cpp
  )
else()
  set(GPU_SOURCE
    ${GPU_SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/HostVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HostMatrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PinnedHostVector.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/PinnedHostMatrix.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/DeviceVector.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/DeviceMatrix.cu
    PARENT_SCOPE
  )
endif(${CPU})

set(SOURCE
  ${SOURCE}
  ${CPU_SOURCE}
  PARENT_SCOPE
)

