#!/usr/bin/env python
import os
import tf
import rospy
import rospkg
import rosparam
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField


class CloudRecord():
    def __init__(self):
        rospy.init_node('sub_cloud', anonymous=True)
        
        cloud_topic = rospy.get_param('~cloud', '/camera/cloud') # cameara/points
        rospy.Subscriber(cloud_topic, PointCloud2, self.callback_point_cloud, queue_size=1)
        
        with open('test.xyz', 'w') as f: pass
        
        self.listener = tf.TransformListener()
        rospy.spin()

    def point_cloud_to_world(self, stamp, points3d):
        """Transforms the point cloud in camera coordinates to the world frame."""
        self.listener.waitForTransform("/world", "/camera0", stamp, rospy.Duration(1.0))
        (position, quaternion) =  self.listener.lookupTransform("/world", "/camera0", stamp)
        matrix = tf.transformations.quaternion_matrix(quaternion)
        matrix[:3,3] = position
        points = np.zeros((len(points3d), 3), dtype=np.float32)
        for k, point3d in enumerate(points3d):
            point = np.ones(4)
            point[:3] = point3d
            points[k] = np.dot(matrix, point)[:3]
        return points

    def callback_point_cloud(self, data):
        cloud_msg = data
        stamp = data.header.stamp
        points = pc2.read_points(cloud_msg, skip_nans=False)
        points3d = []
        for point in points:
            points3d.append(point)
        points3d = np.float32(points3d)
        rospy.loginfo(points3d)
        #TODO: Record only when the camera is moving.
        points3d = self.point_cloud_to_world(stamp, points3d)
        with open('test.xyz', 'a') as f:
            np.savetxt(f, points3d, fmt='%.6f')



if __name__ == '__main__':
    try:
        CloudRecord()
    except rospy.ROSInterruptException:
        pass
    
