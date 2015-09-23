cmake_minimum_required(VERSION 2.8.3)
project(etna_scanning)

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

add_executable(pub_webcam src/pub_webcam.cpp)
target_link_libraries(pub_webcam ${catkin_LIBRARIES})
target_link_libraries(pub_webcam ${OpenCV_LIBRARIES})

add_executable(sub_webcam src/sub_webcam.cpp)
target_link_libraries(sub_webcam ${catkin_LIBRARIES})
target_link_libraries(sub_webcam ${OpenCV_LIBRARIES})

