#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static const std::string OPENCV_WINDOW = "Image window";

// class ImageSubscriber
// {
//     ros::Subscriber sub;
//
// public:
//   ImageSubscriber(ros::NodeHandle n) {
//     sub = n.subscribe("/camera/depth_registered/points", 1000, &ImageSubscriber::callBack, this);
//
//   }
//
//   void callBack(const sensor_msgs::PointCloud2Ptr& msg) {
//
//   }
// };

//int main(int argc, char** argv)
//{
//  ros::Time now;
//  // Get a synchronized capture time
//  now = ros::Time::now();
//  ROS_INFO_STREAM("Now: " << now);
//  cv::Mat frame;
//
//  sensor_msgs::Image img_msg;
//
//  //ros::Rate loop_rate(5);
//  while (nh.ok()) {
//    if(!vid_cap.read(frame))break;
//    //img_msg.header.stamp = now;
//    cv_bridge::CvImage cv_image = cv_bridge::CvImage(img_msg.header, img_msg.encoding, frame);
//    cv_image.toImageMsg(img_msg);
//    pub.publish(img_msg);
//
//    cv::imshow(OPENCV_WINDOW, frame);
//    cv::waitKey(3);
//
//    ros::spinOnce();
//    //loop_rate.sleep();
//  }
//}


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    // Convert the input file to gray
    cv::Mat gray_image;
    cv::cvtColor(cv_bridge::toCvShare(msg, "bgr8")->image, gray_image, cv::COLOR_BGR2GRAY);

    cv::imshow(OPENCV_WINDOW, gray_image);
    cv::waitKey(30);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;

  cv::namedWindow(OPENCV_WINDOW);

  cv::startWindowThread();
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("camera/image", 1, imageCallback);
  ros::spin();

  cv::destroyWindow(OPENCV_WINDOW);
}
