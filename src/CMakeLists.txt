add_subdirectory(exception)
add_subdirectory(data)
add_subdirectory(container)
add_subdirectory(programOptions)
add_subdirectory(fileIO)

if(${CPU})
  add_subdirectory(serial)
else()
  add_subdirectory(cuda)
endif(${CPU})


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
cuda_add_library(${CUEIRA_GPU_LIB} SHARED ${GPU_SOURCE})
