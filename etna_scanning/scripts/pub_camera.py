#!/usr/bin/env python
import rospy
from robscan.camera import Camera
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def camera():
    rospy.init_node('camera')

    image_topic = rospy.get_param('~image', '/camera/image')
    image_pub = rospy.Publisher(image_topic, Image, queue_size=5)

    bridge = CvBridge()

    width = rospy.get_param('width', 640)
    height = rospy.get_param('height', 480)

    camera = Camera(pixel_clock=50, exposure_time=8)
    #camera.configure(width=width, height=height)

    while not rospy.is_shutdown():
        try:
            frame = camera.capture()
            stamp = rospy.Time.now()
            if len(frame.shape) == 2:
                image_msg = bridge.cv2_to_imgmsg(frame, encoding='mono8')
            else:
                image_msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            image_msg.header.stamp = stamp
            image_pub.publish(image_msg)
        except CvBridgeError, e:
            print e


if __name__ == '__main__':
    camera()
