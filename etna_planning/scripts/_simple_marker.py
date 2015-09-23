#!/usr/bin/env python
import roslib; roslib.load_manifest('ispace_common')
import rospy

from geometry_msgs.msg import Point, Pose2D
from visualization_msgs.msg import Marker
from visualization_msgs.msg import InteractiveMarkerControl
from interactive_markers.interactive_marker_server import *

import tf
import numpy as np


class ShapeMarker():
    def __init__(self):
        self.marker = Marker()
        self.marker.header.frame_id = "/world"
        self.marker.lifetime = rospy.Duration()
        
        self.set_scale()
        self.set_color()
        self.set_position()
        self.set_orientation()
        
    def set_color(self, color=(1.0,1.0,1.0,1.0)):
        self.marker.color.r = color[0]
        self.marker.color.g = color[1]
        self.marker.color.b = color[2]
        self.marker.color.a = color[3]
        
    def set_scale(self, scale=(1.0,1.0,1.0)):
        self.marker.scale.x = scale[0]
        self.marker.scale.y = scale[1]
        self.marker.scale.z = scale[2]
        
    def set_position(self, position=(0.0,0.0,0.0)):
        self.marker.pose.position.x = position[0]
        self.marker.pose.position.y = position[1]
        self.marker.pose.position.z = position[2]
        
    def set_orientation(self, orientation=(0.0,0.0,0.0,1.0)):
        self.marker.pose.orientation.x = orientation[0]
        self.marker.pose.orientation.y = orientation[1]
        self.marker.pose.orientation.z = orientation[2]
        self.marker.pose.orientation.w = orientation[3]
  
  
class ArrowMarker(ShapeMarker):
    def __init__(self, length):
        ShapeMarker.__init__(self)
        self.marker.type = self.marker.ARROW
        self.set_length(length)
        
    def set_length(self, length):
        self.set_scale((length, 0.1 * length, 0.1 * length))
        
class CubeMarker(ShapeMarker):
    def __init__(self):
        ShapeMarker.__init__(self)
        self.marker.type = self.marker.CUBE
        
class SphereMarker(ShapeMarker):
    def __init__(self):
        ShapeMarker.__init__(self)
        self.marker.type = self.marker.SPHERE
        
class CylinderMarker(ShapeMarker):
    def __init__(self):
        ShapeMarker.__init__(self)
        self.marker.type = self.marker.CYLINDER
        
class LinesMarker(ShapeMarker):
    def __init__(self):
        ShapeMarker.__init__(self)
        self.marker.type = self.marker.LINE_STRIP
        self.marker.points = cloud = [Point(0,0,0), Point(1,0,0), Point(1,1,0)]
        self.set_size()
        
    def set_size(self, size=0.010):
        self.set_scale(scale=(size,size,1.0))

class PointsMarker(ShapeMarker):
    def __init__(self):
        ShapeMarker.__init__(self)
        self.marker.type = self.marker.POINTS
        self.marker.points = cloud = [Point(0,0,0), Point(1,0,0), Point(1,1,0)]
        self.set_size()
        
    def set_size(self, size=0.025):
        self.set_scale(scale=(size,size,1.0))

class TextMarker(ShapeMarker):
    def __init__(self, text):
        ShapeMarker.__init__(self)
        self.marker.type = self.marker.TEXT_VIEW_FACING    
        self.set_text(text) 
        self.set_size()
        
    def set_text(self, text):
        self.marker.text = text
        
    def set_size(self, size=0.1):
        self.set_scale(scale=(1.0,1.0,size))                    
        
class MeshMarker(ShapeMarker):
    def __init__(self):
        ShapeMarker.__init__(self)
        self.marker.type = self.marker.MESH_RESOURCE
        self.marker.mesh_resource = "package://ispace_common/meshes/robot.dae"



class InteractMarker():
    def __init__(self):
        self.int_marker = InteractiveMarker()
        self.int_marker.header.frame_id = "/world"
        self.int_marker.name = "my_marker"
        self.int_marker.description = "Simple 2-DOF Control"
        self.int_marker.scale = 0.5

        self.mesh_marker = MeshMarker()
        self.mesh_marker.set_color(color=(1.0,1.0,1.0,0.5))
       
        # create a non-interactive control which contains the box
        box_control = InteractiveMarkerControl()
        box_control.always_visible = False
        box_control.markers.append(self.mesh_marker.marker)
        # add the control to the interactive marker
        self.int_marker.controls.append(box_control)
        
        # create a control which will move the box
        # this control does not contain any markers,
        # which will cause RViz to insert two arrows
        control = InteractiveMarkerControl()
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        self.int_marker.controls.append(control)
        
        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        self.int_marker.controls.append(control)

        self.set_position()
        self.set_orientation()
        
    def set_position(self, position=(0.0,0.0,0.0)):
        self.int_marker.pose.position.x = position[0]
        self.int_marker.pose.position.y = position[1]
        self.int_marker.pose.position.z = position[2]
        self.mesh_marker.set_position(position)

    def set_orientation(self, orientation=(0.0,0.0,0.0,1.0)):
        self.int_marker.pose.orientation.x = orientation[0]
        self.int_marker.pose.orientation.y = orientation[1]
        self.int_marker.pose.orientation.z = orientation[2]
        self.int_marker.pose.orientation.w = orientation[3]
        self.mesh_marker.set_orientation(orientation)


class SimpleMarker():
    def __init__(self):
        rospy.init_node("simple_marker", anonymous=True)
    
        robot_name = rospy.get_param('~robot', 'robot_1')

        # create an interactive marker server on the topic namespace simple_marker
        server = InteractiveMarkerServer("interactive_marker_%s" %robot_name)
        
        # create an interactive marker for our server
        int_marker = InteractMarker()
        
        # add the interactive marker to our collection &
        # tell the server to call processFeedback() when feedback arrives for it
        server.insert(int_marker.int_marker, self.processFeedback)

        # 'commit' changes and send to all clients
        server.applyChanges()

        self.pose = Pose2D()

        self.publisher = rospy.Publisher('/%s/goal' %robot_name, Pose2D, queue_size=1)
        rospy.Subscriber(robot_name + '/pose2d', Pose2D, self.handle_pose2d, int_marker)
            
        rospy.spin()

    def processFeedback(self, feedback):
        position, orientation = feedback.pose.position, feedback.pose.orientation
        pose = Pose2D(x=position.x, y=position.y, theta=orientation.z)
        if np.abs(self.pose.x - pose.x) > 0.1 or np.abs(self.pose.y - pose.y) > 0.1:
            self.pose = pose
            self.publisher.publish(pose)

    def handle_pose2d(self, msg, int_marker):
        rospy.loginfo(msg)
        position = (msg.x, msg.y, 0)
        orientation = tf.transformations.quaternion_from_euler(0, 0, msg.theta)
        int_marker.set_position(position)
        int_marker.set_orientation(orientation)
   


if __name__=="__main__":
    SimpleMarker()

