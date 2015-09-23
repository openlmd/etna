#! /usr/bin/env python
import sys
import rospy
import tf

from sensor_msgs.msg import JointState

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
from geometry_msgs.msg import PoseStamped


if __name__ == '__main__':

    rospy.init_node('test_position_ik')
    
    pub_joint_states = rospy.Publisher('joint_states', JointState)
    
    rospy.wait_for_service('compute_ik')
    get_position_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
        
    target = PoseStamped()
    target.header.frame_id = 'base_link'
    x, y, z = 1.5, 0.6, 1.5
    target.pose.position.x = x
    target.pose.position.y = y
    target.pose.position.z = z
    qx, qy, qz, qw = 0.0, -0.3, 0.0, 0.954
    target.pose.orientation.x = qx
    target.pose.orientation.y = qy
    target.pose.orientation.z = qz
    target.pose.orientation.w = qw
    
    service_request = PositionIKRequest()
    service_request.group_name = 'irb_5400'
    #service_request.robot_state = initial_state
    #service_request.ik_link_name = 'link_6'
    service_request.pose_stamped = target
    service_request.timeout.secs= 0.1
    service_request.avoid_collisions = False
    
    rospy.loginfo("Request = {0}".format(service_request))
    
    resp = get_position_ik(service_request)
    rospy.loginfo("Response = {0}".format(resp))
    
    #rospy.loginfo("Base position = [{0},{1},{2}".format(resp.solution.joint_state.position[0],resp.solution.joint_state.position[1],resp.solution.joint_state.position[2]))
    position = resp.solution.joint_state.position
    rospy.loginfo("Arm position = [{0},{1},{2},{3},{4},{5}]".format(position[0], position[1], position[2], position[3], position[4], position[5]))
    
    rospy.sleep(rospy.Duration(1.0))
    
    resp.solution.joint_state.header.stamp = rospy.Time.now()
    pub_joint_states.publish(resp.solution.joint_state)
    
    rospy.sleep(rospy.Duration(1.0))
    
