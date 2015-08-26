cmake_minimum_required(VERSION 2.8.3)
project(etna_calib)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  cv_bridge
  sensor_msgs
  image_transport
)

find_package(OpenCV REQUIRED)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS},
  ${OpenCV_INCLUDE_DIRS}
)

