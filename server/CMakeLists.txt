cmake_minimum_required(VERSION 2.6)

project (server CXX)

find_package(ArrayFire REQUIRED)
find_package(Boost COMPONENTS asio filesystem thread serialization system REQUIRED)
find_package(Flann REQUIRED)
find_package(Hdf5 REQUIRED)
find_package(OpenCV REQUIRED)

set(SERVER_HEADERS server.h)

set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SERVER_CODEGEN_DIR "${PROJECT_BINARY_DIR}/codegen")

PREPEND(SERVER_HEADERS_PATHS ${PROJECT_SOURCE_DIR} ${SERVER_HEADERS})

CODEGEN_FILES(SERVER_CODEGEN_SOURCES ${SERVER_CODEGEN_DIR} ${SERVER_HEADERS})

add_custom_command(OUTPUT ${SERVER_CODEGEN_SOURCES}
  COMMAND ${NGREST_BIN_PATH}ngrestcg -i "${PROJECT_SOURCE_DIR}" -o ${SERVER_CODEGEN_DIR} -t service ${SERVER_HEADERS}
  DEPENDS ${SERVER_HEADERS_PATHS}
)

file(GLOB SERVER_SOURCES 
  ${PROJECT_SOURCE_DIR}/utils/hesaff/*.cpp
  ${PROJECT_SOURCE_DIR}/utils/hesaff/*.h
  ${PROJECT_SOURCE_DIR}/utils/mask/*.cpp
  ${PROJECT_SOURCE_DIR}/utils/mask/*.h
  ${PROJECT_SOURCE_DIR}/utils/ir/*.cpp
  ${PROJECT_SOURCE_DIR}/utils/ir/*.h
  ${PROJECT_SOURCE_DIR}/*.cpp
  ${PROJECT_SOURCE_DIR}/*.h)

list(APPEND SERVER_SOURCES ${SERVER_CODEGEN_SOURCES})

include_directories(${PROJECT_SOURCE_DIR} 
  $ENV{NGREST_EXT_INCLUDES}
  ${ArrayFire_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Flann_INCLUDE_DIRS})

add_library(server MODULE ${SERVER_SOURCES})

set_target_properties(server PROPERTIES PREFIX "")
set_target_properties(server PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SERVICES_DIR}"
)

target_link_libraries(server ngrestutils ngrestcommon ngrestjson ngrestengine 
  $ENV{NGREST_EXT_LIBS}
  ${ArrayFire_LIBRARIES}
  ${Boost_LIBRARIES}
  ${Flann_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${OpenCV_LIBS})
