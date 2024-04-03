cmake_minimum_required(VERSION 2.8.3)
project(stairs_detection)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  image_transport
  std_msgs
  cv_bridge
  pcl_msgs
  pcl_conversions
  pcl_ros
  tf
  tf_conversions
  geometry_msgs
  message_generation
)

find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

set (CMAKE_CXX_STANDARD 11)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  stair_info.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

## The catkin_package macro generates cmake config files for your package
catkin_package(
  CATKIN_DEPENDS
  roscpp
  sensor_msgs
  image_transport
  std_msgs
  cv_bridge
  pcl_msgs
  pcl_conversions
  pcl_ros
  tf
  tf_conversions
  geometry_msgs
  message_runtime
  # INCLUDE_DIRS include  # Uncomment if the package contains header files
  # LIBRARIES stairs_detection  # Uncomment if the package creates a library
)

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(stairs src/main_stairs.cpp src/RGBD/plane.cpp src/RGBD/global_scene.cpp src/RGBD/current_scene.cpp src/stair/stair_classes.cpp src/stair/current_scene_stair.cpp src/stair/global_scene_stair.cpp)

## Add cmake target dependencies of the executable
## This ensures custom messages are built before your main executable
add_dependencies(stairs ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(stairs
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)
