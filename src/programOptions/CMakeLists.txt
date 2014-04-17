include_directories("${PROJECT_BINARY_DIR}")

set(SOURCE
  ${SOURCE}
  ${CMAKE_CURRENT_SOURCE_DIR}/Configuration.cpp
  PARENT_SCOPE
)

set(HEADERS
  ${HEADERS}
  ${CMAKE_CURRENT_SOURCE_DIR}/Configuration.h
  PARENT_SCOPE
)
