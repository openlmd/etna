# Etna

Laser Metal Deposition (LMD) is an additive laser deposition technique focused
on direct manufacturing of metal parts, through the successive deposition of
melted tracks layer by layer, building the component directly from its CAD
model.

Unlike Selective Laser Melting (SLM), mainly used for rapid prototyping,
consisting on the deposition of powder layers and their subsequent laser melt,
it only adds the required material. Moreover, this additive manufacturing (AM)
technique works on current processing technology, known as Laser Cladding,
commonly used for repair of metal components. This makes LMD a promising
technique for retrofitting of current industrial facilities towards the digital
factory (Industrie 4.0).

## Approach

Existing dimensional monitoring approaches are not suitable for on-line path
control of LMD, because they do not resolve measurements in working coordinates
without imposing restrictions in the movements. This enables the adaptation of
laser paths to get the expected geometry. The geometrical control envisaged to
guarantee LMD parts manufacturing requires two main developments:

- To develop a 3D geometrical monitoring system, intended to identify
geometrical distortions and some superficial defects, where the deposited
tracks have to be measured on-line.
- To develop an on-line path planner and control system, intended to fix
geometrical part distortions, adapting the movement of the robot and the
parameters of the process.

A cost effective 3D robotized scanning solution can be successfully used for
on-line LMD monitoring. The solution provides measurements resolved in the
robot coordinates with no motion constraints (6-DoF). It is flexible in terms
of setup, because a novel self-calibration method is provided, which
simultaneously estimates all the calibration parameters. Moreover, this unified 
solution is suitable to automate the reparation of damaged parts.

This means a step forward on the state of the art, since to our knowledge
existing methods not solve the point cloud in working coordinates with 6 DoF  
(degrees of freedom). Besides the self-calibration method solves the tedious
task required for a flexible setup. Such capabilities are crucial for  the
on-line path control of the LMD process, and to adapt the solution to
different requirement.

## Contents

This meta-package currently contains packages that provide nodes and libraries for
on-line 3D scanning and off-line 3D path programming for ABB industrial robots. It
also includes the required utility for the scanning calibration, which was presented
 in IEEE INDIN 2015.

[INDIN 2015 presentation](./etna_scanning/doc/RobEyeEtna.pdf)

It contains four packages:
- etna_workcell: contains the working cell description files.
- etna_scanning: contains the on-line scanning tool for LMD monitoring.
- etna_calib: contains the 3D calibration tools required for scanning.
- etna_planning: contains the off-line path planning tool for LMD automation.

[![Robscan Video](./etna_scanning/media/robscan.png)](./etna_scanning/media/robscan.mp4)

The video shows the system working in a typical industrial robot, with a clear  
relaxation of requirements on the setup (without the need of complex and
accurate fixtures) while keeping a good accuracy.

## 3D scanning

The solution is based on the 3D triangulation principle, which uses a laser stripe
and a camera to determine the 3D points where the laser intersects the piece. The
arrangement of laser and camera determines the resolution and accuracy of the system.
This means, the system is able to be adapted for different part sizes with a simple
setup readjustment.

The 3D triangulation method uses point correspondences to map from 2D image points
to 3D coordinates. First, the projected laser stripe is detected using a center of
gravity peak detector to determine the points of maximum light intensity in the
image. Second, each laser peak detected is transformed to a 3D point in the camera
coordinates. Third, the 3D points are translated from camera coordinates to the
robot coordinates. Thus, since the camera is attached to the robot, the robot pose
is used for reconstructing the 3D point cloud of the piece.

To launch the on-line 3D scanning system:

```
roslaunch etna_workcell etna_workcell.launch
```

## 3D calibration

A calibration procedure is required to perform the 3D point cloud reconstruction.
The method uses an inexpensive planar checkerboard pattern. It uses the same set
of images (shown in the next figure) to perform both camera and laser calibration.
In this way, the system is calibrated in the robot coordinates, ensuring a proper
correspondence between camera and robot coordinates and avoiding complex and
accurate mechanical couplers.

![Checkerboard in different robot positions](./etna_calib/media/checkerboards.jpg)

*Automatic checkerboard detection in different robot positions for system calibration.*

The calibration procedure consists in placing fixed the checkerboard and moving 
the robot at arbitrary orientations in front of the pattern. In each position, 
an image is acquired and its corresponding robot pose is recorded. For each 
image, the checkerboard pose is estimated and the laser intersection is 
determined. Thus, the estimated point cloud is used to find the best-fit laser 
plane. Different least-squares algorithms have been used to minimize errors in 
the calibration estimation problem.

