#!/usr/bin/env python
import os
import cv2
import rospy
import rospkg
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import tf
import std_msgs.msg
import tf.transformations as tfs
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


from robscan.profile import Profile


class Profile3D():
    def __init__(self):
        rospy.init_node('pub_profile3d', anonymous=True)

        image_topic = rospy.get_param('~image', '/camera/image')
        cloud_topic = rospy.get_param('~cloud', '/camera/cloud')
        config_file = rospy.get_param('~config', 'profile3d.yaml')

        #peaks_topic = rospy.get_param('~peaks', '/camera/peaks')
        #self.image_pub = rospy.Publisher(peaks_topic, Image, queue_size=5)

        rospy.Subscriber(image_topic, Image, self.sub_image_topic,
                         queue_size=1)

        self.sequence = 0
        self.bridge = CvBridge()
        self.pcloud = PointCloud2()

        self.cloud_pub = rospy.Publisher(cloud_topic, PointCloud2,
                                         queue_size=5)

        self.profile = Profile()
        path = rospkg.RosPack().get_path('etna_scanning')
        self.profile.load_configuration(os.path.join(path, 'config',
                                                     config_file))

        rospy.spin()

    def pub_point_cloud(self, stamp, profile3d):
        # ERROR: Calibration done in meters
        #cloud = profile3d * 0.001 # Conversion from milimeters to meters
        self.sequence = self.sequence + 1
        self.pcloud = pc2.create_cloud_xyz32(self.pcloud.header, profile3d)
        self.pcloud.header = std_msgs.msg.Header(frame_id="/camera0",
                                                 stamp=stamp,
                                                 seq=self.sequence)
        self.cloud_pub.publish(self.pcloud)

    def image_pub_peak(self, stamp, image):
        image_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        image_msg.header.stamp = stamp
        self.image_pub.publish(image_msg)

    def sub_image_topic(self, data):
        try:
            stamp = rospy.Time.now()
            image = self.bridge.imgmsg_to_cv2(data)

            rospy.loginfo(stamp)
            #stamp = data.header.stamp
            #rospy.loginfo(stamp)

            profile3d, profile2d = self.profile.points_profile(image)

            print 'Profile3d:', profile3d

            #if data.encoding == 'mono8':
            #    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if len(profile3d) > 0:
                self.pub_point_cloud(stamp, profile3d)
                #image = self.profile.draw_points(image, profile2d,
                #                                 color=(0, 0, 255),
                #                                 thickness=2)
            #self.image_pub_peak(stamp, image)
        except CvBridgeError, e:
            print e


if __name__ == '__main__':
    Profile3D()
