#!/usr/bin/env python
import roslib; roslib.load_manifest('abb_irb_4400')

import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from nav_msgs.msg import Path

import tf

from robot_path import dxf, dxf2path

FRAME = 'work_object'
FRAME = 'tool0'

def talker():
    pub = rospy.Publisher('path', Path)
    pub_pose = rospy.Publisher('pose', PoseStamped)
    pub_traj = rospy.Publisher('joint_path_command', JointTrajectory)
    rospy.init_node('path_publisher')
    
    path = Path()
    rospy.loginfo(tf.transformations.quaternion_from_euler(0, 0, 0))
    #path.header = Header(frame_id='work_object')
    path.header = Header(frame_id=FRAME)
    #path.poses = [PoseStamped(pose=Pose(Point(0, 0, 0.5), Quaternion(0, 0, 0, 1))),
    #PoseStamped(pose=Pose(Point(0, 1, 1.4), Quaternion(0, 0, 0, 1)))]
    
    traj = JointTrajectory()
    traj.points = [JointTrajectoryPoint(positions=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    rospy.loginfo(traj)
    pub_traj.publish(traj)
        
    layers = dxf.read_layers('/home/jorge/ros_workspace/robot_path/curvas_v16.dxf') 
    points1, frames1 = dxf2path.get_vectors(layers['Copa1_I'], layers['Copa1_E'])
    cut_path = dxf2path.frames2path(points1, frames1)
    
    for cut_pose in cut_path:
        (x, y, z), (q0, q1, q2, q3), proc = cut_pose
        path.poses.append(PoseStamped(pose=Pose(Point(x/1000, y/1000, z/1000), Quaternion(q0, q1, q2, q3))))
    
    pub.publish(path)
    rospy.sleep(2.0)
    
    k = 0
    N = len(cut_path)
    while not rospy.is_shutdown() and (k < N):
        (x, y, z), (q0, q1, q2, q3), proc = cut_path[k]
        rospy.loginfo("%s, %s" %(cut_path[k], rospy.get_time()))
        pose = PoseStamped(Header(frame_id=FRAME), 
                           Pose(Point(x/1000, y/1000, z/1000),
                                Quaternion(q0, q1, q2, q3)))
        pub_pose.publish(pose)
        k = k + 1
        rospy.sleep(1.0)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
        
