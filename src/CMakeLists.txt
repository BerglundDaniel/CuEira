include_directories("${PROJECT_BINARY_DIR}")

add_subdirectory(programOptions)

#Add the executable
add_executable(CuEira ${PROJECT_SOURCE_DIR}/src/Main.cpp)

#Link
TARGET_LINK_LIBRARIES( CuEira ${Boost_LIBRARIES} ${CUDA_LIBRARIES} )
