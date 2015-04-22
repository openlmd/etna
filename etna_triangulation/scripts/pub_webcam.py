#!/usr/bin/env python
import cv2
import rospy
from icv.webcam import Webcam
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def camera():
    rospy.init_node('webcam')
        
    image_topic = rospy.get_param('~image', '/camera/image')
    image_pub = rospy.Publisher(image_topic, Image, queue_size=5)
    
    bridge = CvBridge()
    
    device = rospy.get_param('~device', 0)
    width = rospy.get_param('~width', 640)
    height = rospy.get_param('~height', 480)
    
    camera = Webcam(device=device)
    camera.configure(width=width, height=height)
    
    while not rospy.is_shutdown():
        try:
            frame = camera.capture()
            image_msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            image_msg.header.stamp = rospy.Time.now()
            image_pub.publish(image_msg)
        except CvBridgeError, e:
            print e

            

if __name__ == '__main__':
    camera()
    
