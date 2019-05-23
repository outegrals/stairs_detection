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

//#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>
#include <signal.h>
#include <time.h>

#include <boost/filesystem.hpp>
#include <boost/bind.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <ros/topic.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <gazebo_msgs/LinkStates.h>
#include <nav_msgs/Odometry.h>
#include <ros/callback_queue.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <eigen_conversions/eigen_msg.h>

#include <pcl/PCLHeader.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/time.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "stair/visualizer_stair.h"
#include "stair/global_scene_stair.h"
#include "stair/current_scene_stair.h"
#include "stair/stair_classes.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

void sayHelp(){
    std::cout << "Arguments to pass:" << std::endl;
    std::cout << "- If no arguments ('$ rosrun stairs_detection stairs'), algorithm proceed reading from ROS topic /camera/depth_registered/points" << std::endl;
    std::cout << "- To run a PCD example (e.g. from Tang dataset), it should be '$ rosrun stairs_detection stairs pcd /path/to.pcd'" << std::endl;
}

class mainLoop {
 public:
    mainLoop() : viewer(), gscene() {
        color_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    }

    ~mainLoop() {}

    void cloudCallback(const sensor_msgs::PointCloud2 &cloud_msg) {
        pcl::fromROSMsg(cloud_msg,*color_cloud);
    }

    void startMainLoop(int argc, char* argv[]) {
        // ROS subscribing
        ros::init(argc, argv, "kinect_navigator_node");
        ros::NodeHandle nh;

        ros::Subscriber cloud_sub = nh.subscribe("/camera/depth_registered/points", 1, &mainLoop::cloudCallback, this);

        int tries = 0;
        while (cloud_sub.getNumPublishers() == 0) {
            ROS_INFO("Waiting for subscibers");
            sleep(1);
            tries++;
            if (tries > 5){
                sayHelp();
                return;
            }
        }

        ros::Rate r(100);

        capture_.reset(new ros::AsyncSpinner(0));
        capture_->start();

        while (nh.ok() && !viewer.cloud_viewer_.wasStopped()) {
            if (color_cloud->points.size() > 0) {
                pcl::copyPointCloud(*color_cloud,*cloud);
                this->execute();
            }

            r.sleep();
        }
        capture_->stop();

    }

    void startPCD(int argc, char* argv[]) {
        // ROS subscribing
        ros::init(argc, argv, "kinect_navigator_node");
        ros::NodeHandle nh;

        std::string cloud_file = argv[2];
        sensor_msgs::PointCloud2 input;
        pcl::io::loadPCDFile(cloud_file, input);
        pcl::fromROSMsg(input,*color_cloud);
        pcl::copyPointCloud(*color_cloud,*cloud);

        while (nh.ok() && !viewer.cloud_viewer_.wasStopped()) {
            if (cloud->points.size() > 0)
            this->execute();
        }

        capture_->stop();
    }

    void execute() {

        // Prepare viewer for next iteration
        viewer.cloud_viewer_.removeAllPointClouds();
        viewer.cloud_viewer_.removeAllShapes();
        viewer.createAxis();

        // Process cloud from current view
        CurrentSceneStair scene;
        scene.applyVoxelFilter(0.04f, cloud);

        if (!gscene.initial_floor_) {
            gscene.findFloor(scene.fcloud);
            gscene.computeCamera2FloorMatrix(gscene.floor_normal_);
            viewer.drawAxis(gscene.f2c);
        }
        else {
            scene.getNormalsNeighbors(16);
            //      scene.getNormalsRadius(0.05f);
            scene.regionGrowing();
            scene.extractClusters(scene.remaining_points);
            gscene.findFloorFast(scene.vPlanes);
            if (gscene.new_floor_)
                gscene.computeCamera2FloorMatrix(gscene.floor_normal_);
            scene.getCentroids();
            scene.getContours();
            scene.getPlaneCoeffs2Floor(gscene.c2f);
            scene.getCentroids2Floor(gscene.c2f);
            scene.classifyPlanes();
            gscene.getManhattanDirections(scene);

            //      viewer.drawNormals (scene.normals, scene.fcloud);
            viewer.drawPlaneTypesContour(scene.vPlanes);
            //      viewer.drawCloudsRandom(scene.vObstacles);
            viewer.drawAxis(gscene.f2c);


            if (scene.detectStairs()) {
                if (scene.getLevelsFromCandidates(scene.upstair,gscene.c2f)) {
                    scene.upstair.modelStaircase(gscene.main_dir, gscene.has_manhattan_);

                    std::cout << "ASCENDING STAIRCASE\n" <<
                                 " Steps = " << scene.upstair.vLevels.size()-1 <<
                                 ", width = " << scene.upstair.step_width <<
                                 "m, height = " << scene.upstair.step_height <<
                                 "m, length = " << scene.upstair.step_length <<
                                 "m\n Pose:\n" <<
                                 scene.upstair.i2s.matrix() << std::endl << std::endl;
                    if (scene.upstair.validateStaircase()) {
                        viewer.addStairsText(scene.upstair.i2s, gscene.f2c, scene.upstair.type);
                        viewer.drawFullAscendingStairUntil(scene.upstair,int(scene.upstair.vLevels.size()),scene.upstair.s2i);
                        viewer.drawStairAxis (scene.upstair, scene.upstair.type);
                    }
                }

                if (scene.getLevelsFromCandidates(scene.downstair,gscene.c2f)) {
                    scene.downstair.modelStaircase(gscene.main_dir, gscene.has_manhattan_);

                    std::cout << "DESCENDING STAIRCASE\n" <<
                                 " Steps = " << scene.downstair.vLevels.size()-1 <<
                                 ", width = " << scene.downstair.step_width <<
                                 "m, height = " << scene.downstair.step_height <<
                                 "m, length = " << scene.downstair.step_length <<
                                 "m\n Pose:\n" <<
                                 scene.downstair.i2s.matrix() << std::endl << std::endl;

                    if (scene.downstair.validateStaircase()) {
                        viewer.addStairsText(scene.downstair.i2s, gscene.f2c, scene.downstair.type);
                        viewer.drawFullDescendingStairUntil(scene.downstair,int(scene.downstair.vLevels.size()),scene.downstair.s2i);
                        viewer.drawStairAxis (scene.downstair, scene.downstair.type);
                    }

                }

            }

        }


        viewer.drawColorCloud(color_cloud,1);
        // viewer.cloud_viewer_.spin();
        viewer.cloud_viewer_.spinOnce();

    }

    // ROS/RGB-D
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud;
    boost::shared_ptr<ros::AsyncSpinner> capture_;

    // RGB-D
    ViewerStair viewer;
    GlobalSceneStair gscene;

};



void parseArguments(int argc, char ** argv, int &capture_mode){
    capture_mode = 0;
    if (argc == 1) {
        capture_mode = 1; // From rosbag or live camera
    }
    else if (argc == 2) {
        if ((strcmp(argv[1], "h") == 0) or (strcmp(argv[1], "-h") == 0) or (strcmp(argv[1], "--h") == 0) or (strcmp(argv[1], "help") == 0) or (strcmp(argv[1], "-help") == 0) or (strcmp(argv[1], "--help") == 0)) {
            sayHelp();
        }
    }
    else if (argc == 3) {
        if (strcmp(argv[1], "pcd") == 0) {
            capture_mode = 0; // reads from PCD, which is the next argument

        }
    }
}

int main(int argc, char* argv[]) {
    mainLoop app;
    int capture_mode;
    parseArguments(argc,argv,capture_mode);

    if (capture_mode == 1) {
        try {
            app.startMainLoop(argc, argv);
        }
        catch (const std::bad_alloc& /*e*/)  {
            cout << "Bad alloc" << endl;
        }
        catch (const std::exception& /*e*/)  {
            cout << "Exception" << endl;
        }
    }
    else {
        if (argc > 2) {
            try {
                app.startPCD(argc, argv);
            }
            catch (const std::bad_alloc& /*e*/) {
                cout << "Bad alloc" << endl;
            }
            catch (const std::exception& /*e*/) {
                cout << "Exception" << endl;
            }
        }
    }

    return 0;
}
