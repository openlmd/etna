#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def talker():
    pub_traj = rospy.Publisher('joint_path_command', JointTrajectory)
    rospy.init_node('path_publisher')
        
    traj = JointTrajectory()
    traj.header = Header(frame_id='base_link', stamp=rospy.Time.now() + rospy.Duration(1.0))
    traj.joint_names = ['joint_1', 'joint_2', 'joint_3', 
                        'joint_4', 'joint_5', 'joint_6']
    traj.points = [JointTrajectoryPoint(positions=[0, 0, 0, 0, 0, 0],
                                        velocities=[0, 0, 0, 0, 0, 0],
                                        time_from_start=rospy.Duration(1)),
                   JointTrajectoryPoint(positions=[1, 0, 0, 0, 0, 0],
                                        velocities=[0, 0, 0, 0, 0, 0])]
    rospy.loginfo(traj)
    pub_traj.publish(traj)
        
    rospy.sleep(1.0)



if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
        
