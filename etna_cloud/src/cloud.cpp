#include <iostream>
#include <fstream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <pcl/point_types.h>
//#include <pcl/point_cloud.h>

#include <tf/transform_listener.h>


class CloudRecordingNode {
private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_cloud;
  ros::Publisher pub_tf_cloud;

  tf::TransformListener *tf_listener;
  tf::StampedTransform transform_;

public:
  typedef pcl::PointXYZ Point;
  typedef pcl::PointCloud<Point> PointCloud;

  CloudRecordingNode() {
    sub_cloud = nh_.subscribe("/webcam/cloud", 1,  &CloudRecordingNode::cloudCallback, this);
    //  ros::Subscriber sub = nh.subscribe<PointCloud>("points2", 1, callback);

    pub_tf_cloud = nh_.advertise<PointCloud>("tf_cloud_out", 1);

    tf_listener = new tf::TransformListener();

    //Point cloud = Point(1, 2, 3);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    // Fill in the cloud data
    cloud->width  = 5;
    cloud->height = 1;
    //cloud->is_dense = false;
    cloud->points.resize (cloud->width * cloud->height);

    for (size_t i = 0; i < cloud->points.size(); ++i) {
      cloud->points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
      cloud->points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
      cloud->points[i].z = 1.0; //cloud->points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);
    }

    // cloud_out.points.push_back(point);

    for (size_t i = 0; i < cloud->points.size (); ++i)
      std::cout << "    " << cloud->points[i].x << " "
                          << cloud->points[i].y << " "
                          << cloud->points[i].z << std::endl;
    pub_tf_cloud.publish(cloud);

    ROS_INFO("Cloud node created.");

    std::ofstream file;
    // std::ios::app is the open mode "append" meaning
    // new data will be written to the end of the file.
    file.open("myfile.txt", std::ios::app);
    file << "I am here.\n";
    file.close();


//    ros::Rate loop_rate(4);
//    while (nh.ok()) {
//      msg->header.stamp = ros::Time::now ();
//      pub.publish(msg);
//      ros::spinOnce();
//      loop_rate.sleep();
//    }

  }

  ~CloudRecordingNode() {}

  void cloudCallback(const PointCloud::ConstPtr& pcl_in) {
    //tf_listener->waitForTransform("/world", (*pcl_in).header.frame_id, (*pcl_in).header.stamp, ros::Duration(5.0));
    //pcl_ros::transformPointCloud("/world", *pcl_in, pcl_out, *tf_listener);

    PointCloud pcl_out;
    pcl_ros::transformPointCloud("/world", *pcl_in, pcl_out, *tf_listener);
    pub_tf_cloud.publish(pcl_out);

    printf("Cloud: width = %d, height = %d\n", pcl_in->width, pcl_in->height);
    BOOST_FOREACH (const pcl::PointXYZ& pt, pcl_in->points)
        printf ("\t(%f, %f, %f)\n", pt.x, pt.y, pt.z);

    //ROS_INFO("TEST");
  }

};



int main (int argc, char** argv) {
  ros::init(argc, argv, "tf_cloud_node"); // voxel_filter_node

  CloudRecordingNode crn;

  ros::spin();
}
