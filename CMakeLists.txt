cmake_minimum_required(VERSION 3.5)
project(stairs_detection)

# Use C++11
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)
find_package(depth_image_proc)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io search visualization filters segmentation surface)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Declare a C++ executable
add_executable(${PROJECT_NAME} src/main_stairs.cpp src/RGBD/plane.cpp src/RGBD/global_scene.cpp src/RGBD/current_scene.cpp src/stair/stair_classes.cpp src/stair/current_scene_stair.cpp src/stair/global_scene_stair.cpp)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  image_transport
  std_msgs
  cv_bridge
  pcl_conversions
  tf2
  tf2_eigen
  tf2_ros
  tf2_geometry_msgs
  depth_image_proc
  OpenCV
)

# Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# Install the executable
install(TARGETS
  ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(ament_cmake rclcpp sensor_msgs image_transport std_msgs cv_bridge pcl_conversions tf2 tf2_ros tf2_geometry_msgs OpenCV)

ament_package()
