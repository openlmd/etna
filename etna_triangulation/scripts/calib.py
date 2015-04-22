import os
import glob

from icv.profile import Profile
from icv.calibration import CameraCalibration
from icv.calibration import LaserCalibration
from icv.calibration import HandEyeCalibration

from icv.image import *
import icv.calculate as calc

np.set_printoptions(precision=4, suppress=True)


path = '../'
pattern_rows = 7
pattern_cols = 8
pattern_size = 0.010
config_file = 'profile3d.yaml'

square_size = pattern_size
grid_size = (pattern_cols-1, pattern_rows-1)

laser_profile = Profile(axis=1, thr=180, method='pcog')
camera_calibration = CameraCalibration(grid_size=grid_size, square_size=square_size)
laser_calibration = LaserCalibration(grid_size=grid_size, square_size=square_size, profile=laser_profile)
print os.path.join(path, 'data', 'frame*.png')
laser_calibration.find_calibration_3d(os.path.join(path, 'data', 'frame*.png'))
laser_calibration.save_parameters(os.path.join(path, 'config', config_file))


filenames = sorted(glob.glob('../data/pose*.txt'))
ks = [int(filename[-8:-4]) for filename in filenames]
poses_checker, poses_tool = [], []
poses_ichecker, poses_itool = [], []
for k in ks:
    print 'Frame: %i' %k
    img = read_image('../data/frame%04i.png' %k)
    grid = camera_calibration.find_chessboard(img)
    pose_checker = None
    if grid is not None:
        pose_checker = laser_calibration.get_chessboard_pose(grid)
        pose_checker = calc.pose_to_matrix((pose_checker[0], pose_checker[1]))
        img = camera_calibration.draw_chessboard(img, grid)
    show_images([img])
    with open('../data/pose%04i.txt' %k, 'r') as f:
        pose = eval(f.read())
        quatpose_tool0 = (np.array(pose[0]), np.array(pose[1]))
        pose_tool0 = calc.quatpose_to_matrix(*quatpose_tool0)
    if pose_checker is not None:
        poses_checker.append(pose_checker)
        poses_tool.append(pose_tool0)
        poses_ichecker.append(calc.matrix_invert(pose_checker))
        poses_itool.append(calc.matrix_invert(pose_tool0))
print 'Poses:', len(poses_checker), len(poses_tool), poses_checker, poses_tool

print 'Hand Eye Calibration Solution'
tlc = HandEyeCalibration()
T2C = tlc.solve(poses_tool, poses_checker)
W2K = tlc.solve(poses_itool, poses_ichecker)
print 't2c>', T2C, calc.matrix_to_rpypose(T2C)
print 'w2k>', W2K, calc.matrix_to_rpypose(W2K)
