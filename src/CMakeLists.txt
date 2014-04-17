include_directories("${PROJECT_BINARY_DIR}")

set(SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
)

add_subdirectory(programOptions)

#Add the executable
add_executable(CuEira ${SOURCE} ${HEADERS})

#Link
TARGET_LINK_LIBRARIES( CuEira ${Boost_LIBRARIES} ${CUDA_LIBRARIES} )
