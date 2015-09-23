#!/usr/bin/env python
import os
import rospy
import rospkg
import rosparam
import numpy as np

import tf
import std_msgs
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField


class Cloud():
    def __init__(self):
        rospy.init_node('pub_cloud', anonymous=True)
        
        cloud_topic = rospy.get_param('~cloud', '/camera/cloud') # cameara/points
        self.cloud_pub = rospy.Publisher(cloud_topic, PointCloud2, queue_size=5)
        
        self.sequence = 0
        self.pcloud = PointCloud2()
        
        self.listener = tf.TransformListener()
        
        r = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            self.pub_point_cloud()
            r.sleep()
        
    def pub_point_cloud(self):
        stamp = rospy.Time.now()
        points3d = np.random.random((100, 3))
        points3d[:,0] = 0.01 * points3d[:,1]
        points3d[:,1] = np.linspace(-0.1, 0.1, 100)
        points3d[:,2] = 0.05 * points3d[:,2] + 0.25
        #rospy.loginfo(points3d)
        # ERROR: Calibration done in meters
        #cloud = profile3d * 0.001 # Conversion from milimeters to meters
        self.sequence = self.sequence + 1
        self.pcloud = pc2.create_cloud_xyz32(self.pcloud.header, points3d)
        self.pcloud.header = std_msgs.msg.Header(frame_id="/camera0",
                                                 stamp=stamp,
                                                 seq=self.sequence)
        self.cloud_pub.publish(self.pcloud)



if __name__ == '__main__':
    try:
        Cloud()
    except rospy.ROSInterruptException:
        pass
    
