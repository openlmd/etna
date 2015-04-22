#!/usr/bin/env python
import tf
import rospy
import rospkg
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from nav_msgs.msg import Path

import robpath.polyline as dxf2path


def talker():
    rospy.init_node('path_publisher')
    
    pub_path = rospy.Publisher('path', Path)
    pub_pose = rospy.Publisher('pose', PoseStamped)
        
    path = Path()
    rospy.loginfo(tf.transformations.quaternion_from_euler(0, 0, 0))
    path.header = Header(frame_id='workobject')
    #path.header = Header(frame_id='tool0')
    #path.poses = [PoseStamped(pose=Pose(Point(0, 0, 0.5), Quaternion(0, 0, 0, 1))),
    #PoseStamped(pose=Pose(Point(0, 1, 1.4), Quaternion(0, 0, 0, 1)))]
        
    layers = dxf2path.read_layers(rospkg.RosPack().get_path('etna_planning') + '/src/robpath/models_dxf/curvas_v16.dxf') 
    points1, frames1 = dxf2path.get_vectors(layers['Copa1_I'], layers['Copa1_E'])
    cut_path = dxf2path.frames2path(points1, frames1)
    
    for cut_pose in cut_path:
        (x, y, z), (q0, q1, q2, q3), proc = cut_pose
        path.poses.append(PoseStamped(pose=Pose(Point(x/1000, y/1000, z/1000), Quaternion(q0, q1, q2, q3))))
    
    pub_path.publish(path)
    rospy.sleep(2.0)
    
    k = 0
    N = len(cut_path)
    while not rospy.is_shutdown() and (k < N):
        (x, y, z), (q0, q1, q2, q3), proc = cut_path[k]
        rospy.loginfo("%s, %s" %(cut_path[k], rospy.get_time()))
        pose = PoseStamped(Header(frame_id='workobject'), 
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
        
