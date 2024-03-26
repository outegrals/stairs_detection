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
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/PCLHeader.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>

#include "stair/visualizer_stair.h"
#include "stair/global_scene_stair.h"
#include "stair/current_scene_stair.h"
#include "stair/stair_classes.h"

static int CAPTURE_MODE = 0; // Capure mode can be 0 (reading clouds from ROS topic), 1 (reading from .pcd file), 2 (reading all *.pcd from directory)
static int DESCENDING_COUNT= 0;
static int ASCENDING_COUNT = 0;
static int HIT_BOTH = 0;

void sayHelp(){
    std::cout << "-- Arguments to pass:" << std::endl;
    std::cout << "<no args>               - If no arguments ('$ rosrun stairs_detection stairs'), by default the algorithm proceed reading point clouds from ROS topic /camera/depth_registered/points" << std::endl;
    std::cout << "pcd <path to file>      - To run a PCD example (e.g. from Tang dataset), it should be '$ rosrun stairs_detection stairs pcd /path/to.pcd'" << std::endl;
    std::cout << "dir <path to directory> - To run all PCDs in a dataset, you can point at the folder, e.g. '$ rosrun stairs_detection stairs dir /path/to/pcds/" << std::endl;
}

class MainLoop : public rclcpp::Node {
public:
    MainLoop() : Node("stairs_detection_node"), viewer(), gscene() {
        //color_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        //color_cloud_show = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        //cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        color_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        color_cloud_show.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera/depth_registered/points", 10,
            std::bind(&MainLoop::cloudCallback, this, std::placeholders::_1));
    }

    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *color_cloud);
        // Additional processing...
    }

    void startMainLoop() {
        RCLCPP_INFO(this->get_logger(), "Starting Main Loop.");

        // In ROS2, the spin function is responsible for processing callbacks. For a subscriber,
        // spinning will wait for and call the callback function whenever a new message is received.
        // rclcpp::spin(shared_from_this()) can be called if this node needs to be spun.
        // If the loop contains other periodic checks or operations outside of callbacks,
        // consider using a while loop with rclcpp::spin_some() for more controlled spinning.

        // Example with manual loop and controlled spinning
        rclcpp::Rate rate(10); // 10 Hz
        int tries = 0;
        while (rclcpp::ok()) {
            // Do any work here...

            // Spin some to handle callbacks
            rclcpp::spin_some(shared_from_this());

            rate.sleep();
            tries++;
            if (tries > 5){
                sayHelp();
                return;
            }
        }
    }


    // This functions configures and executes the loop reading from a .pcd file (capture_mode = 1)
    void startPCD(const std::string& file_path) {
        pcl::PCLPointCloud2 pcl_pc2;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        if (pcl::io::loadPCDFile(file_path, pcl_pc2) == -1) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load PCD file.");
            return;
        }

        pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);
        
        // After loading the color cloud, save a copy onto cloud
        color_cloud = temp_cloud;
        pcl::copyPointCloud(*color_cloud, *cloud);

        while (rclcpp::ok()) {
            if (cloud->points.size() > 0)
                this->execute();
            //viewer.cloud_viewer_.spinOnce(100);
            if (viewer.cloud_viewer_.wasStopped())
                break;
        }
    }

    // Helper function to read just *.pcd files in path
    bool has_suffix(const std::string& s, const std::string& suffix) {
        return (s.size() >= suffix.size()) && equal(suffix.rbegin(), suffix.rend(), s.rbegin());
    }

    void startDirectory(const std::string& directory_path) {
        DIR* dir = opendir(directory_path.c_str());
        struct dirent* entry;

        if (!dir) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open directory.");
            return;
        }
        int total_files = 0;
        while (((entry = readdir(dir)) != nullptr) && rclcpp::ok()) {
            std::string entryName = entry->d_name;
            if (has_suffix(entryName, ".pcd")) {
                std::string full_path = directory_path + "/" + entryName;
                RCLCPP_INFO(this->get_logger(), "Loading PCD: %s", full_path.c_str());

                pcl::PCLPointCloud2 pcl_pc2;
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

                if (pcl::io::loadPCDFile(full_path, pcl_pc2) == -1) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to load PCD file: %s", full_path.c_str());
                    continue;
                }
                pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);
                
                // After loading the color cloud, save a copy onto cloud
                color_cloud = temp_cloud;
                pcl::copyPointCloud(*color_cloud, *cloud);
                if (cloud->points.size() > 0)
                {
                    gscene.reset();
                    this->execute();
                    this->execute();
                }

                total_files += 1;
            }
        }
        std::cout << "Ascending Cases: " << ASCENDING_COUNT << std::endl;
        std::cout << "Descending Cases: " << DESCENDING_COUNT << std::endl;
        std::cout << "Both Cases were hit in the same PCD: " << HIT_BOTH << std::endl;
        std::cout << "Total Detected Staircases: " << ASCENDING_COUNT + DESCENDING_COUNT - HIT_BOTH << " Out of " << total_files << " Files" << std::endl;
        closedir(dir);
        // No ROS spinning required here unless you're interacting with other parts of a ROS system
    }

    void execute() {
        // Prepare viewer for this iteration
        pcl::copyPointCloud(*color_cloud,*color_cloud_show);
        viewer.cloud_viewer_.removeAllPointClouds();
        viewer.cloud_viewer_.removeAllShapes();
        viewer.createAxis();

        // Process cloud from current view
        CurrentSceneStair scene;
        scene.applyVoxelFilter(0.04f, cloud); // Typically 0.04m voxels works fine for this method, however, bigger number (for more efficiency) or smaller (for more accuracy) can be used
 
        // The method first attempts to find the floor automatically. The floor position allows to orient the scene to reason about planar surfaces (including stairs)
        if (!gscene.initial_floor_) {
            gscene.findFloor(scene.fcloud);
            gscene.computeCamera2FloorMatrix(gscene.floor_normal_);
            viewer.drawAxis(gscene.f2c);
            viewer.drawColorCloud(color_cloud_show,1);
            //viewer.cloud_viewer_.spinOnce();
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

            // Some drawing functions for the PCL to see how the method is doing untill now
            // viewer.drawNormals (scene.normals, scene.fcloud);
            // viewer.drawPlaneTypesContour(scene.vPlanes);
            // viewer.drawCloudsRandom(scene.vObstacles);
            // viewer.drawAxis(gscene.f2c);
            int hit_both_cases = 0;
            // STAIR DETECTION AND MODELING
            if (scene.detectStairs()) { // First a quick check if horizontal planes may constitute staircases
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

                        // Draw staircase
                        viewer.addStairsText(scene.upstair.i2s, gscene.f2c, scene.upstair.type);
                        viewer.drawFullAscendingStairUntil(scene.upstair,int(scene.upstair.vLevels.size()),scene.upstair.s2i);
                        viewer.drawStairAxis (scene.upstair, scene.upstair.type);
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

                        // Draw staircase
                        viewer.addStairsText(scene.downstair.i2s, gscene.f2c, scene.downstair.type);
                        viewer.drawFullDescendingStairUntil(scene.downstair,int(scene.downstair.vLevels.size()),scene.downstair.s2i);
                        viewer.drawStairAxis (scene.downstair, scene.downstair.type);
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
                std::cout << "No staircase Detected!!!" << std::endl;
            }
            // Draw color cloud and update viewer
            //viewer.drawColorCloud(color_cloud_show,1);
            //if (CAPTURE_MODE > 0)
            //    while(!viewer.cloud_viewer_.wasStopped())
            //        viewer.cloud_viewer_.spinOnce();
            //else
            //    viewer.cloud_viewer_.spinOnce();

        }

    }

private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud_show; // just for visualization of each iteration, since color_cloud keeps being updated in the callback
    ViewerStair viewer; // Visualization object
    GlobalSceneStair gscene; // Global scene (i.e. functions and variables that should be kept through iterations)

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};


void parseArguments(int argc, char ** argv, int &capture_mode){
    // Capture mode goes as follows:
    // 0 (default) - Read clouds from ROS topic '/camera/depth_registered/points/'
    // 1 - Reads pcd inserted as argument, e.g. 'pcd /path/to.pcd'
    // 2 - Reads all pcds from given directory, e.g. 'dir /path/to/pcds/
    capture_mode = 0;
    if (argc == 1) {
        capture_mode = 0; // From rosbag or live camera
    }
    else if (argc == 2) {
        if ((strcmp(argv[1], "h") == 0) or (strcmp(argv[1], "-h") == 0) or (strcmp(argv[1], "--h") == 0) or (strcmp(argv[1], "help") == 0) or (strcmp(argv[1], "-help") == 0) or (strcmp(argv[1], "--help") == 0)) {
            sayHelp();
        }
    }
    else if (argc == 3) {
        if (strcmp(argv[1], "pcd") == 0) {
            capture_mode = 1; // reads from PCD, which is the next argument

        }
        else if (strcmp(argv[1], "dir") == 0) {
            capture_mode = 2; // reads PCDs from directory
        }
    }
}


int main(int argc, char* argv[]) {

    parseArguments(argc,argv,CAPTURE_MODE);

    rclcpp::init(argc, argv);
    auto app = std::make_shared<MainLoop>();

    switch (CAPTURE_MODE) {
    case 0:
        app->startMainLoop();
        rclcpp::spin(app);
        break;
    case 1:
        app->startPCD(argv[2]); // Process a single PCD file
        break;
    case 2:
        app->startDirectory(argv[2]); // Process a directory of PCD files
        break;
    default:
        std::cerr << "Unknown mode or insufficient arguments." << std::endl;
        break;
    }

    rclcpp::shutdown();
    return 0;
}