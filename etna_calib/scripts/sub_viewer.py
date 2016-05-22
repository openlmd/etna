#!/usr/bin/env python
import os
import tf
import cv2
import rospy
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from robscan.profile import Profile
from icv.calibration import CameraCalibration
from icv.calibration import LaserCalibration
from icv.calibration import HandEyeCalibration

import rospkg

rospack = rospkg.RosPack()
path = rospack.get_path('etna_calib')


class ImageViewer():
    def __init__(self):
        rospy.init_node('viewer', anonymous=True)

        image_topic = rospy.get_param('~image', '/camera/image')
        # Pattern parameters
        pattern_rows = rospy.get_param('~pattern_rows', 7)
        pattern_cols = rospy.get_param('~pattern_cols', 8)
        pattern_size = rospy.get_param('~pattern_size', 10)

        config_file = rospy.get_param('~config', 'profile3d.yaml')

        rospy.Subscriber(image_topic, Image, self.callback, queue_size=10)
        rospy.on_shutdown(self.on_shutdown_hook)

        self.counter = 0
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.square_size = pattern_size
        self.grid_size = (pattern_cols-1, pattern_rows-1)

        self.laser_profile = Profile(axis=1, thr=180, method='pcog')
        self.camera_calibration = CameraCalibration(grid_size=self.grid_size, square_size=self.square_size)

        cv2.namedWindow('viewer')
        cv2.cv.SetMouseCallback('viewer', self.on_mouse, '')

        rospy.spin()

    def on_shutdown_hook(self):
        cv2.destroyWindow('viewer')
        #TODO: Fix changes in the Calibration module.
        #calibration = LaserCalibration(grid_size=self.grid_size, square_size=self.square_size, profile=self.laser_profile)
        #print os.path.join(path, 'data', 'frame*.png')
        #calibration.find_calibration_3d(os.path.join(path, 'data', 'frame*.png'))
        #calibration.save_parameters(os.path.join(path, 'config', config_file))

    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.cv.CV_EVENT_RBUTTONDOWN:
            self.counter += 1
            filename = os.path.join(path, 'data', 'frame%04i.png' %self.counter)
            cv2.imwrite(filename, self.frame)
            rospy.loginfo(filename)
            try:
                #base_link
                self.listener.waitForTransform("/base_link", "/tool0", self.stamp, rospy.Duration(1.0))
                transform = self.listener.lookupTransform("/base_link", "/tool0", self.stamp) #(trans, rot)
                filename = os.path.join(path, 'data', 'pose%04i.txt' %self.counter)
                with open(filename, 'w') as f:
                    f.write(str(transform))
                rospy.loginfo(transform)
            except:
                rospy.loginfo('The transformation is not accesible.')

    def callback(self, data):
        try:
            self.stamp = data.header.stamp
            self.frame = self.bridge.imgmsg_to_cv2(data)
             #TODO: Change to work with gray images
            if data.encoding == 'mono8':
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
            #TODO: Show chessboard and line detection.
            frame = cv2.resize(self.frame, (self.frame.shape[1]/2, self.frame.shape[0]/2))
            grid = self.camera_calibration.find_chessboard(frame)
            if grid is not None:
                self.camera_calibration.draw_chessboard(frame, grid)
            cv2.imshow("viewer", frame)
            cv2.waitKey(1)
        except CvBridgeError, e:
            print e



if __name__ == '__main__':
    ImageViewer()
