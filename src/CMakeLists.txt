set(INCLUDE_DIRECTORIES
  ${PROJECT_BINARY_DIR}
)

set(SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
)

add_subdirectory(programOptions)
add_subdirectory(fileIO)

#Add the executable
add_executable(CuEira ${SOURCE} ${HEADERS})
include_directories(${INCLUDE_DIRECTORIES})

#Link
TARGET_LINK_LIBRARIES( CuEira ${Boost_LIBRARIES} ${CUDA_LIBRARIES} )
