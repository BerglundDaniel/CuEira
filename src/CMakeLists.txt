set(INCLUDE_DIRECTORIES
  ${INCLUDE_DIRECTORIES}
  ${CMAKE_CURRENT_SOURCE_DIR}
  PARENT_SCOPE
)

set(SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
)

add_subdirectory(data)
add_subdirectory(exception)
#add_subdirectory(container)
add_subdirectory(programOptions)
#add_subdirectory(fileIO)

#Add the executable
add_executable(CuEira ${SOURCE} ${HEADERS})
include_directories(${INCLUDE_DIRECTORIES})

#Link
TARGET_LINK_LIBRARIES( CuEira ${Boost_LIBRARIES} ${CUDA_LIBRARIES} )
