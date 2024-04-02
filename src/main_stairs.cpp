/**
* This file is part of stairs_detection.
*
* Copyright (C) 2019 Alejandro Pérez Yus <alperez at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/aperezyus/stairs_detection>
*
* stairs_detection is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* stairs_detection is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with stairs_detection. If not, see <http://www.gnu.org/licenses/>.
*/

#include <math.h>
#include <iostream>
#include <memory>
#include <functional>
#include <limits>
#include <signal.h>
#include <time.h>
#include <dirent.h> // To read directory

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tf2_eigen/tf2_eigen.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <rclcpp/rclcpp.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"


#include <pcl/PCLHeader.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>

#include <image_transport/image_transport.hpp>
#include "cv_bridge/cv_bridge.h"
#include "opencv2/imgproc/imgproc.hpp"

#include "stair/global_scene_stair.h"
#include "stair/current_scene_stair.h"
#include "stair/stair_classes.h"

static int DESCENDING_COUNT= 0;
static int ASCENDING_COUNT = 0;
static int HIT_BOTH = 0;

class MainLoop : public rclcpp::Node {
public:

    MainLoop() : Node("stairs_detection_node"), gscene() {
        color_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
        last_depth_msg_ = nullptr;
        last_depth_info_msg_ = nullptr;

        /**
         * All topics the realsense returns
        * /d435_front/color/camera_info
        * /d435_front/color/image_raw
        * /d435_front/depth/camera_info
        * /d435_front/depth/image_raw
        * /d435_front/infra1/camera_info
        * /d435_front/infra1/image_raw
        * /d435_front/infra2/camera_info
        * /d435_front/infra2/image_raw
        */
        // Subscribe to image and camera info topics with SensorDataQoS
        depth_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/d435_front/depth/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&MainLoop::depthImageCallback, this, std::placeholders::_1));

        depth_camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/d435_front/depth/camera_info", rclcpp::SensorDataQoS(),
            std::bind(&MainLoop::depthCameraInfoCallback, this, std::placeholders::_1));

        rgb_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/d435_front/color/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&MainLoop::RgbImageCallback, this, std::placeholders::_1));

        rgb_camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/d435_front/color/camera_info", rclcpp::SensorDataQoS(),
            std::bind(&MainLoop::RgbCameraInfoCallback, this, std::placeholders::_1));

        // Create a timer to periodically process data, replacing the continuous loop
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), // Adjust the rate as needed
            std::bind(&MainLoop::processData, this));

        RCLCPP_INFO(this->get_logger(), "Stair detection node created");
    }


private:

    void depthImageCallback(const sensor_msgs::msg::Image::SharedPtr depth_msg) {
        last_depth_msg_ = depth_msg;
    }

    void depthCameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr info_msg) {
        last_depth_info_msg_ = info_msg;
    }

    void RgbImageCallback(const sensor_msgs::msg::Image::SharedPtr rgb_msg) {
        last_rgb_msg_ = rgb_msg;
    }

    void RgbCameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr info_msg) {
        last_rgb_info_msg_ = info_msg;
    }

    void processData() {
        RCLCPP_INFO(this->get_logger(), "Processing Data start");

        if (last_depth_msg_ != nullptr && last_depth_info_msg_ != nullptr && last_rgb_msg_ != nullptr && last_rgb_info_msg_ != nullptr) {
            RCLCPP_INFO(this->get_logger(), "Processing Data");
            // If not currently processing, start processing the new data
            if (!is_processing_) {
                RCLCPP_INFO(this->get_logger(), "Reset for new batch of data");
                is_processing_ = true;
                gscene.reset(); // Reset gscene for the new batch of data

                //so we don't overwrite the cloud info until it's time for a new one
                sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg(new sensor_msgs::msg::PointCloud2);
                ImageToPointCloud2(last_depth_msg_, last_depth_info_msg_, last_rgb_msg_, last_rgb_info_msg_, *cloud_msg);
                pcl::fromROSMsg(*cloud_msg, *color_cloud);
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
                for (const auto& point : *color_cloud)
                {
                    if (!(point.x == 0 && point.y == 0 && point.z == 0 && point.rgb == 0))
                    {
                        cloud_filtered->push_back(point);
                    }
                }
                pcl::copyPointCloud(*cloud_filtered, *cloud);

                // Assuming 'cloud' is a pcl::PointCloud<pcl::PointXYZ>::Ptr with your data
                if (cloud_filtered && !cloud_filtered->points.empty()) {
                    pcl::io::savePCDFileASCII("output_cloud.pcd", *cloud_filtered);
                    RCLCPP_INFO(this->get_logger(), "Saved %zu points to output_cloud.pcd", cloud_filtered->points.size());
                }
            }
            bool processing_complete = false;

            if (cloud->points.size() > 0) {
                RCLCPP_INFO(this->get_logger(), "We have point cloud data");
                //gscene.reset();
                this->execute();
                // No need to call this->execute() twice unless it's intentional
            }
            else
            {
                RCLCPP_INFO(this->get_logger(), "No more data left to process");
                processing_complete = true;
            }
            

            // Reset messages to prevent reprocessing
            if (processing_complete) {
                RCLCPP_INFO(this->get_logger(), "Reset Data");
                is_processing_ = false;
                last_depth_msg_ = nullptr; // Reset messages to indicate readiness for new data
                last_depth_info_msg_ = nullptr;
            }
        }
    }

    void ImageToPointCloud2(
        const sensor_msgs::msg::Image::SharedPtr& depth_msg,
        const sensor_msgs::msg::CameraInfo::SharedPtr& depth_info_msg,
        const sensor_msgs::msg::Image::SharedPtr& rgb_msg,
        const sensor_msgs::msg::CameraInfo::SharedPtr& rgb_info_msg,
        sensor_msgs::msg::PointCloud2& cloud_msg)
    {
        RCLCPP_INFO(this->get_logger(), "Converting Data");

        // Convert ROS image messages to OpenCV images
        cv_bridge::CvImagePtr cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        cv_bridge::CvImagePtr cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::RGB8);

        // Resize RGB image to match depth image resolution
        cv::Mat resized_rgb;
        cv::resize(cv_rgb_ptr->image, resized_rgb, cv_depth_ptr->image.size());

        // Calculate constants for the projection
        float fx = 1.0f / depth_info_msg->k[0];  // Focal length in x
        float fy = 1.0f / depth_info_msg->k[4];  // Focal length in y
        float cx = depth_info_msg->k[2];         // Optical center in x
        float cy = depth_info_msg->k[5];         // Optical center in y

        // Initialize PointCloud2 message
        cloud_msg.height = depth_msg->height;
        cloud_msg.width = depth_msg->width;
        cloud_msg.is_dense = false; // There may be invalid points
        cloud_msg.is_bigendian = false;
        cloud_msg.header = depth_msg->header; // Use the header from the depth image

        // Set the fields for the PointCloud2 message
        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb"); // Updated to include RGB
        modifier.resize(depth_msg->width * depth_msg->height);

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud_msg, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud_msg, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud_msg, "b");

        const uint16_t* depth_row = reinterpret_cast<const uint16_t*>(&depth_msg->data[0]);

        for (int v = 0; v < (int)depth_msg->height; ++v, depth_row += depth_msg->step / sizeof(uint16_t))
        {
            for (int u = 0; u < (int)depth_msg->width; ++u)
            {
                uint16_t depth = depth_row[u];

                if (depth == 0) // Skip if depth value is invalid
                {
                    *iter_x = *iter_y = *iter_z = std::numeric_limits<float>::quiet_NaN();
                    *iter_r = *iter_g = *iter_b = 0;
                    continue;
                }

                float z = depth * 0.001f; // Convert depth to meters
                float x = (u - cx) * z * fx;
                float y = (v - cy) * z * fy;

                *iter_x = x;
                *iter_y = y;
                *iter_z = z;

                // Map color data
                // Accessing RGB data from resized_rgb_image
                cv::Vec3b color = resized_rgb.at<cv::Vec3b>(v, u);
                *iter_r = color[0]; // R
                *iter_g = color[1]; // G
                *iter_b = color[2]; // B

                ++iter_x; ++iter_y; ++iter_z;
                ++iter_r; ++iter_g; ++iter_b;
            }
        }
    }

    void execute() {
        RCLCPP_INFO(this->get_logger(), "Executing Algo");
        // Process cloud from current view
        CurrentSceneStair scene;
        scene.applyVoxelFilter(0.04f, cloud); // Typically 0.04m voxels works fine for this method, however, bigger number (for more efficiency) or smaller (for more accuracy) can be used
 
        // The method first attempts to find the floor automatically. The floor position allows to orient the scene to reason about planar surfaces (including stairs)
        if (!gscene.initial_floor_) {
            RCLCPP_INFO(this->get_logger(), "First time seeing scene");
            gscene.findFloor(scene.fcloud);
            gscene.computeCamera2FloorMatrix(gscene.floor_normal_);
        }
        else {
            // Compute the normals
            // scene.getNormalsNeighbors(8); // 8 Neighbours provides better accuracy, ideal for close distances (like the rosbag provided)
            scene.getNormalsNeighbors(16); // 16 works better for more irregular pointclouds, like those from far distances (e.g. Tang's dataset)
            // scene.getNormalsRadius(0.05f); // Alternatively, radius may be used instead of number of neighbors
        
            // Segment the scene in planes and clusters
            scene.regionGrowing();
            scene.extractClusters(scene.remaining_points);

            // Find and update the floor position
            gscene.findFloorFast(scene.vPlanes);
            if (gscene.new_floor_)
                gscene.computeCamera2FloorMatrix(gscene.floor_normal_);

            // Get centroids, contours and plane coefficients to floor reference
            scene.getCentroids();
            scene.getContours();
            scene.getPlaneCoeffs2Floor(gscene.c2f);
            scene.getCentroids2Floor(gscene.c2f);

            // Classify the planes in the scene
            scene.classifyPlanes();

            // Get Manhattan directions to rotate floor reference to also be aligned with vertical planes (OPTIONAL)
            gscene.getManhattanDirections(scene);

            int hit_both_cases = 0;
            // STAIR DETECTION AND MODELING
            if (scene.detectStairs()) { // First a quick check if horizontal planes may constitute staircases
                RCLCPP_INFO(this->get_logger(), "Something was detected");
                // Ascending staircase
                if (scene.getLevelsFromCandidates(scene.upstair,gscene.c2f)) { // Sort planes in levels
                    scene.upstair.modelStaircase(gscene.main_dir, gscene.has_manhattan_); // Perform the modeling
                    if (scene.upstair.validateStaircase()) { // Validate measurements
                        std::cout << "--- ASCENDING STAIRCASE ---\n" <<
                                     "- Steps: " << scene.upstair.vLevels.size()-1 << std::endl <<
                                     "- Measurements: " <<
                                     scene.upstair.step_width << "m of width, " <<
                                     scene.upstair.step_height << "m of height, " <<
                                     scene.upstair.step_length << "m of length. " << std::endl <<
                                     "- Pose (stair axis w.r.t. camera):\n" << scene.upstair.s2i.matrix() << std::endl << std::endl;

                        ASCENDING_COUNT += 1;
                        hit_both_cases += 1;
                    }

                }

                // Descending staircase
                if (scene.getLevelsFromCandidates(scene.downstair,gscene.c2f)) { // Sort planes in levels
                    scene.downstair.modelStaircase(gscene.main_dir, gscene.has_manhattan_); // Perform the modeling
                    if (scene.downstair.validateStaircase()) { // Validate measurements
                        std::cout << "--- DESCENDING STAIRCASE ---\n" <<
                                     "- Steps: " << scene.downstair.vLevels.size()-1 << std::endl <<
                                     "- Measurements: " <<
                                     scene.downstair.step_width << "m of width, " <<
                                     scene.downstair.step_height << "m of height, " <<
                                     scene.downstair.step_length << "m of length. " << std::endl <<
                                     "- Pose (stair axis w.r.t. camera):\n" << scene.downstair.s2i.matrix() << std::endl << std::endl;

                        DESCENDING_COUNT += 1;
                        hit_both_cases += 1;
                    }
                }
                if (hit_both_cases == 2)
                {
                    HIT_BOTH += 1;
                    RCLCPP_INFO(this->get_logger(), "This hit both cases");
                }
            }
            else
            {
                RCLCPP_INFO(this->get_logger(), "Nothing Detected");
                std::cout << "No staircase Detected!!!" << std::endl;
            }
        }

    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud;
    rclcpp::TimerBase::SharedPtr timer_;
    GlobalSceneStair gscene; // Global scene (i.e. functions and variables that should be kept through iterations)


    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr rgb_camera_info_sub_;
    sensor_msgs::msg::Image::SharedPtr last_depth_msg_;
    sensor_msgs::msg::CameraInfo::SharedPtr last_depth_info_msg_;
    sensor_msgs::msg::Image::SharedPtr last_rgb_msg_;
    sensor_msgs::msg::CameraInfo::SharedPtr last_rgb_info_msg_;

    // Member variable to track processing state
    bool is_processing_ = false;
};


int main(int argc, char* argv[]) {

    rclcpp::init(argc, argv);
    auto app = std::make_shared<MainLoop>();
    rclcpp::spin(app);
    rclcpp::shutdown();
    return 0;
}