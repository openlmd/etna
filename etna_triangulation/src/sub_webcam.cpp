#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

static const std::string OPENCV_WINDOW = "Image window";

//class ImageSubscriber
//{

//}

//int main(int argc, char** argv)
//{
//  ros::init(argc, argv, "webcam");
//  ros::NodeHandle nh;
//  
//  image_transport::ImageTransport it(nh);
//  image_transport::Publisher pub = it.advertise("camera/image", 5);

//  cv::namedWindow(OPENCV_WINDOW);
//  
//  cv::VideoCapture vid_cap(1);
//    if(!vid_cap.isOpened()){
//        std::cout << "could not read file" << std::endl;
//        return -1;
//    }
//  
//  ros::Time now;
//  // Get a synchronized capture time
//  now = ros::Time::now();
//  // Grab a frame
//  //double frame = vid_cap.get(CV_CAP_PROP_FPS);
//  //ROS_INFO_STREAM("Frame: "<<frame<<" of "<<n_frames);
//  ROS_INFO_STREAM("Now: " << now);
//  cv::Mat frame;
//  
//  sensor_msgs::Image img_msg;
//  
//  img_msg.encoding = "bgr8";
//  img_msg.header.frame_id = "/camera0";
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
//  std::cout << "stop" << std::endl;
//  vid_cap.release();
//  std::cout << "one loop finished" << std::endl;
//  
//  cv::destroyWindow(OPENCV_WINDOW);
//}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
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
  cv::namedWindow("view");
  cv::startWindowThread();
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("camera/image", 1, imageCallback);
  ros::spin();
  cv::destroyWindow("view");
}
