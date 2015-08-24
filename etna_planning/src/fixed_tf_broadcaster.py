#!/usr/bin/env python  
import roslib; roslib.load_manifest('abb_irb_4400')

import rospy
import tf

if __name__ == '__main__':
    rospy.init_node('laser_head')
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        br.sendTransform((0.310, 0.0, 0.090), (0.0, 0.0, 0.0, 1.0),
                         rospy.Time.now(), "head1", "tool0")
        rate.sleep()
