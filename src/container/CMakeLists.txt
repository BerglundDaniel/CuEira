add_subdirectory(data)

set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

if(${CPU})
  set(CPU_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/LapackppHostVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LapackppHostMatrix.cpp
  )
else()
  set(GPU_SOURCE
    ${GPU_SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/PinnedHostVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PinnedHostMatrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DeviceVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DeviceMatrix.cpp
    PARENT_SCOPE
  )
endif(${CPU})

set(SOURCE
  ${SOURCE}
  ${CPU_SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/HostVector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HostMatrix.cpp
  PARENT_SCOPE
)

