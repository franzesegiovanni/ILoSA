cmake_minimum_required(VERSION 2.8.3)
project(ilosa)
set(CMAKE_CXX_STANDARD 11)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  tf_conversions
)

catkin_package(
    CATKIN_DEPENDS sigma
    )

catkin_python_setup()
catkin_install_python(PROGRAMS
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
