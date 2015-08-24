#!/usr/bin/env python
import rospy

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


class ShapeMarker():
    def __init__(self, frame_id="/world"):
        self.marker = Marker()
        self.marker.header.frame_id = frame_id
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
        ShapeMarker.__init__(self, frame_id='/world')
        self.marker.type = self.marker.ARROW
        self.set_length(length)

    def set_length(self, length):
        self.set_scale((length, 0.1 * length, 0.1 * length))

class CubeMarker(ShapeMarker):
    def __init__(self):
        ShapeMarker.__init__(self, frame_id='/world')
        self.marker.type = self.marker.CUBE

class SphereMarker(ShapeMarker):
    def __init__(self):
        ShapeMarker.__init__(self, frame_id='/world')
        self.marker.type = self.marker.SPHERE

class CylinderMarker(ShapeMarker):
    def __init__(self):
        ShapeMarker.__init__(self, frame_id='/world')
        self.marker.type = self.marker.CYLINDER

class LinesMarker(ShapeMarker):
    def __init__(self):
        ShapeMarker.__init__(self, frame_id='/world')
        self.marker.type = self.marker.LINE_STRIP
        self.marker.points = cloud = [Point(0,0,0), Point(1,0,0), Point(1,1,0)]
        self.set_size()

    def set_size(self, size=0.010):
        self.set_scale(scale=(size,size,1.0))

class PointsMarker(ShapeMarker):
    def __init__(self):
        ShapeMarker.__init__(self, frame_id='/world')
        self.marker.type = self.marker.POINTS
        self.marker.points = cloud = [Point(0,0,0), Point(1,0,0), Point(1,1,0)]
        self.set_size()

    def set_size(self, size=0.025):
        self.set_scale(scale=(size,size,1.0))

class TextMarker(ShapeMarker):
    def __init__(self, text):
        ShapeMarker.__init__(self, frame_id='/world')
        self.marker.type = self.marker.TEXT_VIEW_FACING
        self.set_text(text)
        self.set_size()

    def set_text(self, text):
        self.marker.text = text

    def set_size(self, size=0.1):
        self.set_scale(scale=(1.0,1.0,size))

class MeshMarker(ShapeMarker):
    def __init__(self, mesh_resource="package://etna_triangulation/meshes/robot.dae", frame_id='/world'):
        ShapeMarker.__init__(self, frame_id=frame_id)
        self.marker.type = self.marker.MESH_RESOURCE
        self.marker.mesh_resource = mesh_resource
        self.marker.mesh_use_embedded_materials = True

class TriangleListMarker(ShapeMarker):
    def __init__(self, frame_id='/world'):
        ShapeMarker.__init__(self, frame_id=frame_id)
        self.marker.type = self.marker.TRIANGLE_LIST
        self.marker.points = [Point(0,0,0), Point(1,0,0), Point(1,1,0),
                              Point(0,0,0), Point(1,0,0), Point(1,0,1)]
        #self.set_size()

    def set_points(self, points):
        self.marker.points = [Point(x, y, z) for x, y, z in points]

if __name__=="__main__":
    rospy.init_node("markers")

    publisher = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    #mesh_marker = MeshMarker(mesh_resource="package://etna_triangulation/meshes/test.dae")
    mesh_marker = TriangleListMarker()
    mesh_marker.set_color(color=(1.0,0.5,0.0,0.75))

    while not rospy.is_shutdown():
        publisher.publish(mesh_marker.marker)
        rospy.sleep(0.1)
