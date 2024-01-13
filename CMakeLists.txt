cmake_minimum_required(VERSION 3.0.2)
project(lio)
ADD_COMPILE_OPTIONS(-std=c++14 )
add_compile_definitions(SOPHUS_USE_BASIC_LOGGING)

set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")


if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  livox_ros_driver2
  geometry_msgs
  tf2
  tf2_ros
  pcl_ros
  eigen_conversions
  message_generation
)
find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED )
find_package(PCL REQUIRED)
find_package(GTSAM REQUIRED)

add_service_files(
  FILES
  SlamReLoc.srv
  SaveMap.srv
  MapConvert.srv
  SlamHold.srv
  SlamStart.srv
  SlamRelocCheck.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(

)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Sophus_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  include
)

set(COMMON_SRC src/commons.cpp)
set(LOCALIZER_SRC src/localizer/icp_localizer.cpp)
set(KDTREE_SRC include/ikd-Tree/ikd_Tree.cpp)
set(LIOBUILDER_SRC_LIST src/map_builder/imu_processor.cpp src/map_builder/fastlio_builder.cpp src/map_builder/iglio_builder.cpp)
set(VOXMAP_SRC src/voxel_map/voxel_map.cpp)
set(IESKF_SRC src/ieskf/ieskf.cpp)

set(SRC_LIST ${COMMON_SRC} ${KDTREE_SRC} ${VOXMAP_SRC} ${IESKF_SRC} ${LIOBUILDER_SRC_LIST} ${LOCALIZER_SRC})

add_executable(map_builder_node src/map_builder_node.cpp ${SRC_LIST})

target_link_libraries(map_builder_node 
  ${catkin_LIBRARIES} gtsam
)

add_executable(localizer_node src/localizer_node.cpp ${SRC_LIST})

target_link_libraries(localizer_node 
  ${catkin_LIBRARIES}
)
