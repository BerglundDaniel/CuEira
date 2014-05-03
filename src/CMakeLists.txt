add_subdirectory(data)
add_subdirectory(exception)
#add_subdirectory(container)
add_subdirectory(programOptions)
#add_subdirectory(fileIO)

set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(SOURCE
  ${SOURCE}
  #${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
)

include_directories(${INCLUDE_DIRECTORIES})

add_library(${CUEIRA_LIB} SHARED ${SOURCE})
