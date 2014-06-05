add_subdirectory(exception)
add_subdirectory(data)
add_subdirectory(container)
add_subdirectory(programOptions)
add_subdirectory(fileIO)
add_subdirectory(statistics)

if(NOT ${CPU})
  add_subdirectory(cuda)
endif(NOT ${CPU})


set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(SOURCE
  ${SOURCE}
)

set(GPU_SOURCE
  ${GPU_SOURCE}
)

include_directories(${INCLUDE_DIRECTORIES})

add_library(${CUEIRA_LIB} SHARED ${SOURCE})
if(NOT ${CPU})
  cuda_add_library(${CUEIRA_GPU_LIB} SHARED ${GPU_SOURCE})
endif(NOT ${CPU})
