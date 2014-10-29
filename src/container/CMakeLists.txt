add_subdirectory(data)

set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

if(NOT ${CPU})
  set(GPU_SOURCE
    ${GPU_SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/PinnedHostVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PinnedHostMatrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DeviceVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DeviceMatrix.cpp
    PARENT_SCOPE
  )
endif()

set(SOURCE
  ${SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/HostVector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HostMatrix.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/RegularHostVector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/RegularHostMatrix.cpp
  PARENT_SCOPE
)

