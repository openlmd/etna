#!/usr/bin/env python
import os
import sys
import rospy
import rospkg
import rosparam

from python_qt_binding import loadUi
from python_qt_binding import QtGui
from python_qt_binding import QtCore

import rviz

from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker

import tf
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField

from markers import MeshMarker, TriangleListMarker
from robpath import RobPath

import rospkg
rp = rospkg.RosPack()
path = rp.get_path('etna_planning')


class MyViz(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        ## rviz.VisualizationFrame is the main container widget of the
        ## regular RViz application. In this example, we disable everything
        ## so that the only thing visible is the 3D render window.
        self.frame = rviz.VisualizationFrame()
        self.frame.setSplashPath("")
        self.frame.initialize()

        ## The reader reads config file data into the config object.
        ## VisualizationFrame reads its data from the config object.
        reader = rviz.YamlConfigReader()
        config = rviz.Config()

#        rospack = rospkg.RosPack()
#        package_path = rospack.get_path('rviz_python_tutorial')
#        reader.readFile( config, package_path + "config.myviz" )
        reader.readFile(config, os.path.join(path, 'config', 'workcell.rviz'))
        self.frame.load(config)

        ## You can also store any other application data you like in the
        ## config object.  Here we read the window title from the map key
        ## called "Title", which has been added by hand to the config file.
        self.setWindowTitle(config.mapGetChild("Title").getValue())

        self.frame.setMenuBar(None)
        self.frame.setHideButtonVisibility(False)

        self.manager = self.frame.getManager()

        ## Since the config file is part of the source code for this
        ## example, we know that the first display in the list is the
        ## grid we want to control.  Here we just save a reference to
        ## it for later.
        self.grid_display = self.manager.getRootDisplayGroup().getDisplayAt(0)

        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(9, 0, 9, 0)
        self.setLayout(layout)

        h_layout = QtGui.QHBoxLayout()
        layout.addLayout(h_layout)

        orbit_button = QtGui.QPushButton("Orbit View")
        orbit_button.clicked.connect(self.onOrbitButtonClick)
        h_layout.addWidget(orbit_button)

        front_button = QtGui.QPushButton("Front View")
        front_button.clicked.connect(self.onFrontButtonClick)
        h_layout.addWidget(front_button)

        right_button = QtGui.QPushButton("Rigth View")
        right_button.clicked.connect(self.onRightButtonClick)
        h_layout.addWidget(right_button)

        top_button = QtGui.QPushButton("Top View")
        top_button.clicked.connect(self.onTopButtonClick)
        h_layout.addWidget(top_button)

        layout.addWidget(self.frame)

    ## switchToView() works by looping over the views saved in the
    ## ViewManager and looking for one with a matching name.
    def switchToView(self, view_name):
        view_man = self.manager.getViewManager()
        for i in range(view_man.getNumViews()):
            if view_man.getViewAt(i).getName() == view_name:
                view_man.setCurrentFrom(view_man.getViewAt(i))
                return
        print("Did not find view named %s." % view_name)

    def onOrbitButtonClick( self ):
        self.switchToView("Orbit View")

    def onFrontButtonClick( self ):
        self.switchToView("Front View")

    def onRightButtonClick( self ):
        self.switchToView("Right View")

    def onTopButtonClick( self ):
        self.switchToView("Top View")


class RobPathUI(QtGui.QMainWindow):
    def __init__(self):
        super(RobPathUI, self).__init__()
        loadUi(os.path.join(path, 'resource', 'robviz.ui'), self)

        self.boxPlot.addWidget(MyViz())

        self.btnLoad.clicked.connect(self.btnLoadClicked)
        self.btnProcessMesh.clicked.connect(self.btnProcessMeshClicked)
        self.btnSaveRapid.clicked.connect(self.btnSaveRapidClicked)
        self.btnRecord.clicked.connect(self.btnRecordClicked)

        self.sbSpeed.valueChanged.connect(self.changeSpeed)
        self.sbPower.valueChanged.connect(self.changePower)

        self.sbPositionX.valueChanged.connect(self.changePosition)
        self.sbPositionY.valueChanged.connect(self.changePosition)
        self.sbPositionZ.valueChanged.connect(self.changePosition)

        self.sbSizeX.valueChanged.connect(self.changeSize)
        self.sbSizeY.valueChanged.connect(self.changeSize)
        self.sbSizeZ.valueChanged.connect(self.changeSize)

        self.btnQuit.clicked.connect(self.btnQuitClicked)

        self.publisher = rospy.Publisher('visualization_marker', Marker, queue_size=1)

        self.recording = False
        cloud_topic = rospy.get_param('~cloud', '/camera/cloud') # cameara/points
        rospy.Subscriber(cloud_topic, PointCloud2, self.callback_point_cloud, queue_size=1)

        self.listener = tf.TransformListener()
        #rospy.spin()

        self.processing = False
        self.timer = QtCore.QTimer(self.boxPlot)
        self.timer.timeout.connect(self.updateProcess)

        self.robpath = RobPath()

    def changeSpeed(self):
        speed = self.Window.sbSpeed.value()
        self.robpath.set_speed(speed)

    def changePower(self):
        power = self.Window.sbPower.value()
        self.robpath.set_power(power)

    def updatePosition(self, position):
        x, y, z = position
        self.sbPositionX.setValue(x)
        self.sbPositionY.setValue(y)
        self.sbPositionZ.setValue(z)

    def updateSize(self, size):
        sx, sy, sz = size
        self.sbSizeX.setValue(sx)
        self.sbSizeY.setValue(sy)
        self.sbSizeZ.setValue(sz)

    def updateProcess(self):
        if self.robpath.k < len(self.robpath.levels):
            self.robpath.update_process()
            #self.plot.drawSlice(self.robpath.slices, self.robpath.path)
            #self.plot.progress.setValue(100.0 * self.robpath.k / len(self.robpath.levels))
        else:
            self.processing = False
            self.timer.stop()

    def blockSignals(self, value):
        self.sbPositionX.blockSignals(value)
        self.sbPositionY.blockSignals(value)
        self.sbPositionZ.blockSignals(value)
        self.sbSizeX.blockSignals(value)
        self.sbSizeY.blockSignals(value)
        self.sbSizeZ.blockSignals(value)

    def btnLoadClicked(self):
        self.blockSignals(True)

        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open file', './',
                                                     'Mesh Files (*.stl)')[0]
        print 'Filename:', filename
        self.robpath.load_mesh(filename)

        self.updatePosition(self.robpath.mesh.bpoint1) # Rename to position
        self.updateSize(self.robpath.mesh.bpoint2 - self.robpath.mesh.bpoint1) # Change by size

        #self.marker = MeshMarker(mesh_resource="file://"+filename, frame_id="/workobject")
        self.marker = TriangleListMarker(frame_id="/workobject")
        self.marker.set_points(0.001 * np.vstack(self.robpath.mesh.triangles))
        self.marker.set_color((0.75,0.25,0.25,0.5))
#        #rospy.loginfo()
#        self.marker.set_position((0, 0, 0))
#        self.marker.set_scale(scale=(0.001, 0.001, 0.001))
        self.publisher.publish(self.marker.marker)

        #self.Window.setWindowTitle('Mesh Viewer: %s' %filename)
        #self.robpath.load_mesh(filename)
        ## -----
        ## TODO: Change bpoints.
        #self.updatePosition(self.robpath.mesh.bpoint1) # Rename to position
        #self.updateSize(self.robpath.mesh.bpoint2 - self.robpath.mesh.bpoint1) # Change by size
        #self.Window.lblInfo.setText('Info:\n')
        ## -----
        #self.plot.drawMesh(self.robpath.mesh)

        self.blockSignals(False)

    def btnProcessMeshClicked(self):
        if self.processing:
            self.processing = False
            self.timer.stop()
        else:
            self.processing = True
            self.timer.start(100)

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
        if self.recording:
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

    def btnRecordClicked(self):
        if self.recording:
            print 'Stop record'
            self.recording = False
        else:
            print 'Recording...'
            with open('test.xyz', 'w') as f: pass
            self.recording = True

    def changePosition(self):
        x = 0.001 * self.sbPositionX.value()
        y = 0.001 * self.sbPositionY.value()
        z = 0.001 * self.sbPositionZ.value()
        #self.mesh.translate(position)
        self.marker.set_position((x, y, z))
        #TODO: Change mesh position. Include offset position.
        self.publisher.publish(self.marker.marker)

    def changeSize(self):
        sx = self.sbSizeX.value()
        sy = self.sbSizeY.value()
        sz = self.sbSizeZ.value()
        self.robpath.resize_mesh(np.float32([sx, sy, sz]))
        #TODO: Scale the marker or change the mesh loaded in the mesh.
        self.changePosition()

    def btnProcessMeshClicked(self):
        if self.processing:
            self.processing = False
            self.timer.stop()
        else:
            height = self.Window.sbHeight.value()
            width = self.Window.sbWidth.value()
            overlap = 0.01 * self.Window.sbOverlap.value()
            self.robpath.set_track(height, width, overlap)

            self.plot.drawWorkingArea()

            self.robpath.init_process()

            self.processing = True
            self.timer.start(100)

    def btnSaveRapidClicked(self):
        #filename = QtGui.QFileDialog.getOpenFileName(self.plot, 'Save file', './',
        #                                             'Rapid Modules (*.mod)')[0]
        self.robpath.save_rapid()

    def btnQuitClicked(self):
        QtCore.QCoreApplication.instance().quit()



if __name__ == '__main__':
    rospy.init_node('myviz')

    app = QtGui.QApplication(sys.argv)
    robpath = RobPathUI()
    robpath.show()
    app.exec_()
