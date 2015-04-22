cmake_minimum_required(VERSION 2.8.3)
project(etna_cloud)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  pcl_ros
  sensor_msgs
)

find_package(octomap REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES etna_cloud
#  CATKIN_DEPENDS pcl_ros roscpp sensor_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

include_directories(${OCTOMAP_INCLUDE_DIRS})

link_libraries(${OCTOMAP_LIBRARIES})

## Declare a cpp library
# add_library(etna_cloud
#   src/${PROJECT_NAME}/etna_cloud.cpp
# )

## Declare a cpp executable
add_executable(talker src/talker.cpp)
target_link_libraries(talker ${catkin_LIBRARIES})

add_executable(listener src/listener.cpp)
target_link_libraries(listener ${catkin_LIBRARIES})

add_executable(cloud src/cloud.cpp)
target_link_libraries(cloud ${catkin_LIBRARIES})

add_executable(find_plane src/find_plane.cpp)
target_link_libraries(find_plane ${catkin_LIBRARIES})

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(etna_cloud_node etna_cloud_generate_messages_cpp)
