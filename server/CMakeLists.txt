cmake_minimum_required(VERSION 2.6)

project (server CXX)

find_package(ArrayFire REQUIRED)
find_package(Boost COMPONENTS filesystem thread serialization system REQUIRED)
find_package(Flann REQUIRED)
find_package(GLog REQUIRED)
find_package(GTest REQUIRED)
find_package(Hdf5 REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads)

set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(PROJECT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

file(GLOB SERVER_SOURCES 
  ${PROJECT_SOURCE_DIR}/utils/hesaff/*.cpp
  ${PROJECT_SOURCE_DIR}/utils/hesaff/*.h
  ${PROJECT_SOURCE_DIR}/utils/ir/*.cpp
  ${PROJECT_SOURCE_DIR}/utils/ir/*.h
  ${PROJECT_SOURCE_DIR}/utils/taskmgr/*.cpp
  ${PROJECT_SOURCE_DIR}/utils/taskmgr/*.h
  ${PROJECT_SOURCE_DIR}/utils/*.h
  ${PROJECT_SOURCE_DIR}/*.cpp
  ${PROJECT_SOURCE_DIR}/*.h
)

file(GLOB SERVER_TESTS
  ${PROJECT_TEST_DIR}/utils/hesaff/*.cpp
  ${PROJECT_TEST_DIR}/utils/ir/*.cpp
  ${PROJECT_TEST_DIR}/utils/mask/*.cpp
)

include_directories(${PROJECT_SOURCE_DIR} 
  $ENV{NGREST_EXT_INCLUDES}
  ${ArrayFire_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Flann_INCLUDE_DIRS}
  ${GLog_INCLUDE_DIRS}
)

add_executable(server ${SERVER_SOURCES})

target_link_libraries(server
  ${ArrayFire_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${Flann_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${OpenCL_LIBRARY}
  ${OpenCV_LIBS}
)
