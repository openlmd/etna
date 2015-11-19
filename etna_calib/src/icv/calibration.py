import os
import glob
import yaml
import numpy as np
import numpy.linalg as la

from image import *
from drawing import *
import fitting as fit
import calculate as calc

from robscan.profile import Profile


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

np.set_printoptions(precision=4, suppress=True)


# Homography functions

def find_homography(points, targets):
    """Finds the homography projective transformation matrix."""
    homography = cv2.findHomography(points, targets, cv2.RANSAC)[0]
    return homography


def homography_points(points2d, homography):
    """Transforms the image points to plane coordinates through homography."""
    pnts = np.float32([np.dot(homography,
                              np.float32([point[0],
                                          point[1],
                                          1])) for point in points2d])
    pnts = np.float32([pnt / pnt[2] for pnt in pnts])
    return pnts[:, :2]


class CameraCalibration():
    def __init__(self, grid_size=(7, 6), square_size=10.0):
        self.grid_size = grid_size
        self.square_size = square_size
        self.targets = self.get_pattern_points()
        self.pattern_points = np.float32([[point[0], point[1], 0]
                                          for point in self.targets])

    def get_pattern_points(self):
        points = np.indices(self.grid_size, np.float32).T.reshape(-1, 2)
        points = np.fliplr(points) * self.square_size
        #points += self.grid_orig
        return points

    def load_camera_parameters(self, filename):
        with open(filename, 'r') as f:
            data = yaml.load(f)
        self.rms = data['rms']
        self.camera_mat = np.array(data['mat'])
        self.dist_coef = np.array(data['coef'])
        return data

    def save_camera_parameters(self, filename):
        data = dict(rms=self.rms,
                    mat=self.camera_mat.tolist(),
                    coef=self.dist_coef.tolist())
        with open(filename, 'w') as f:
            f.write(yaml.dump(data))
        return data

    def find_chessboard(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        gray = cv2.resize(gray, (w / 2, h / 2))
        grid = None
        found, corners = cv2.findChessboardCorners(gray, self.grid_size)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
            grid = corners.reshape((self.grid_size[0], self.grid_size[1], 2))
            grid = grid * 2
        return grid

    def get_chessboard_pose(self, grid):
        """Gets the estimated pose for the calibration chessboard."""
        if grid is not None:
            corners = grid.reshape((-1, 2))
            return self.find_transformation(self.pattern_points, corners)
        return None

    def get_calibration(self, images):
        """Gets the camera parameters solving the calibration problem."""
        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.
        self.grids = [self.find_chessboard(img) for img in images]
        for grid in self.grids:
            if grid is not None:
                corners = grid.reshape((-1, 2))
                img_points.append(corners)
                obj_points.append(self.pattern_points)
        self.rms, self.camera_mat, self.dist_coef, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, get_size(images[0]))
        self.calculate_reprojection_errors(self.grids)
        return self.camera_mat, self.dist_coef

    def find_transformation(self, object_points, image_points):
        """Finds the rotation and translation transformation."""
        rvecs, tvecs, inliers = cv2.solvePnPRansac(object_points,
                                                   image_points,
                                                   self.camera_mat,
                                                   self.dist_coef)
        R, t = cv2.Rodrigues(rvecs)[0], tvecs.reshape(-1)
        return R, t

    def project_3d_points(self, points, pose):
        """Projects 3D points to image coordinates."""
        R, t = pose
        rvecs, tvecs = cv2.Rodrigues(R)[0], np.float32([t]).T
        imgpts, jac = cv2.projectPoints(points, rvecs, tvecs,
                                        self.camera_mat, self.dist_coef)
        return imgpts.reshape((-1, 2))

    def reprojection_error(self, points3d, imgpoints, pose):
        """Calculates the re-projection error, how exact is the estimation of
        the found parameters. This should be as close to zero as possible.
        Given the intrinsic, distortion, rotation and translation matrices,
        we first transform the object point to image point using
        cv2.projectPoints(). Then we calculate the absolute norm between the
        image points. To find the average error we calculate the arithmetical
        mean of the errors calculated."""
        imgpoints2 = self.project_3d_points(points3d, pose)
        error = cv2.norm(imgpoints, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        return error

    def chessboard_reprojection_error(self, grid):
        """Gets the reprojection error for the calibration chessboard."""
        corners = grid.reshape((-1, 2))
        points3d = self.pattern_points
        chessboard_pose = self.get_chessboard_pose(grid)
        return self.reprojection_error(points3d, corners, chessboard_pose)

    # TODO: Add rejected criterium, and data structure with data for each image
    def calculate_reprojection_errors(self, grids):
        for grid in grids:
            if grid is not None:
                print 'Error', self.chessboard_reprojection_error(grid)

    def undistort_image(self, image):
        display = cv2.undistort(image.copy(), self.camera_mat, self.dist_coef)
        return display

    def undistort_points(self, points):
        points = cv2.undistortPoints(np.float32([points]), self.camera_mat,
                                     self.dist_coef).reshape((-1, 2))
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        # Normalized coordinated are transformed to image coordinates.
        points = points * np.float32([fx, fy]) + np.float32([cx, cy])
        return points

    def draw_chessboard(self, image, corners):
        if corners is not None:
            rows, cols = corners.shape[:2]
            corners = corners.reshape((-1, 2))
            cv2.drawChessboardCorners(image, (rows, cols), corners, True)
        return image

    def draw_frame(self, img, pose, size=(30, 30, 30)):
        sx, sy, sz = size
        corners = np.float32([[0, 0, 0], [sx, 0, 0], [0, sy, 0], [0, 0, sz]])
        imgpts = self.project_3d_points(corners, pose)
        cv2.line(img, tuple(imgpts[0]), tuple(imgpts[1]), RED, 5)
        cv2.line(img, tuple(imgpts[0]), tuple(imgpts[2]), GREEN, 5)
        cv2.line(img, tuple(imgpts[0]), tuple(imgpts[3]), BLUE, 5)
        return img

    def draw_box(self, img, pose, size=(30, 30, 30), thickness=3):
        axis = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                           [0, 0, -1], [0, 1, -1], [1, 1, -1],
                           [1, 0, -1]]) * np.float32(size)
        imgpts = np.int32(self.project_3d_points(axis, pose))
        cv2.drawContours(img, [imgpts[:4]], -1, GREEN, thickness)
        for i, j in zip(range(4), range(4, 8)):
            cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), BLUE, thickness)
        cv2.drawContours(img, [imgpts[4:]], -1, RED, thickness)
        return img


# ----------------------------------------------------------------------------

class LaserCalibration(CameraCalibration):
    def __init__(self, grid_size=(7, 6), square_size=10.0, profile=Profile()):
        CameraCalibration.__init__(self, grid_size=grid_size,
                                   square_size=square_size)
        self.camera_pose = (np.eye(3), np.zeros(3))
        self.profile = profile

    def load_parameters(self, filename):
        return self.profile.load_configuration(filename)

    def save_parameters(self, filename):
        return self.profile.save_configuration(filename)

    def find_best_line2d(self, points2d):
        line = fit.LineFit()
        model, inliers = line.ransac(points2d, line, int(0.5*len(points2d)), 5)
        return model, inliers

    def find_best_plane(self, points3d):
        plane = fit.PlaneFit()
        model, inliers = plane.ransac(points3d, plane,
                                      int(0.5*len(points3d)), 20)
        return model, inliers

    def find_plane_transformation(self, plane_pose):
        """Finds the homography transformation for a plane pose."""
        points3d = calc.transform_points2d(self.targets, self.camera_pose)
        points2d = self.project_3d_points(points3d, plane_pose)
        homography = find_homography(points2d, self.targets)
        return homography

    def find_lightplane(self, points3d):
        """Calculates the transformation between the image points and the
        lightplane."""
        plane, inliers = self.find_best_plane(points3d)
        plane_pose = fit.get_plane_pose(plane)
        print 'Plane pose', plane_pose
        plane_homography = self.find_plane_transformation(plane_pose)
        print 'Homography', plane_homography
        return plane_pose, plane_homography

    def filter_chessboard_laser(self, profile3d, profile2d):
        """Filter points by reprojection error."""
        vpoints2d, vpoints3d = [], []
        for k in range(len(profile2d)):
            error = self.reprojection_error(np.float32([profile3d[k]]),
                                            np.float32([profile2d[k]]),
                                            self.camera_pose)
            if error < 1:
                vpoints2d.append(profile2d[k])
                vpoints3d.append(profile3d[k])
        profile2d = np.float32(vpoints2d)
        profile3d = np.float32(vpoints3d)
        return profile3d, profile2d

    def get_chessboard_laser(self, img, grid, chessboard_pose):
        homography = find_homography(grid.reshape((-1, 2)), self.targets)
        profile3d, profile2d = self.profile.points_profile(img, homography,
                                                           chessboard_pose)
        #profile3d, profile2d = self.filter_chessboard_laser(profile3d,
        #                                                    profile2d)
        return profile3d, profile2d

    def find_calibration_3d(self, images):
        self.get_calibration(images)
        self.images = images
        self.pattern_poses = [self.get_chessboard_pose(grid)
                              for grid in self.grids]
        profiles3d, profiles2d = [], []
        for k, img in enumerate(images):
            grid, pattern_pose = self.grids[k], self.pattern_poses[k]
            if grid is not None:
                profile3d, profile2d = self.get_chessboard_laser(img, grid,
                                                                 pattern_pose)
                if len(profile2d) > 0:
                    line, inliers = self.find_best_line2d(profile2d)
                    profiles3d.append(profile3d[inliers])
                    profiles2d.append(profile2d[inliers])
        self.profiles3d = np.vstack(profiles3d)
        self.profiles2d = np.vstack(profiles2d)
        points3d = self.profiles3d
        points2d = self.profiles2d
        plane, inliers = self.find_best_plane(points3d)
        plane_pose = fit.get_plane_pose(plane)
        print 'Plane pose', plane_pose
        print 'L', len(points3d), len(points2d)
        points3d = points3d[inliers]
        points2d = points2d[inliers]
        print 'l', len(points3d), len(points2d)
        print '> search transformation from 2d to 3d'
        self.profile.trans = fit.fit_transformation(points2d, points3d)
        self.profile.pose, self.profile.homography = self.find_lightplane(self.profiles3d)

    def show_calibration_3d(self):
        print 'Camera calibration'
        print self.camera_mat, self.dist_coef
        print 'Laser pose and transformation'
        print self.profile.pose, self.profile.homography
        mplot3d = MPlot3D(scale=0.005)
        for k, img in enumerate(self.images):
            grid, pattern_pose = self.grids[k], self.pattern_poses[k]
            if grid is not None:
                profile3d, profile2d = self.get_chessboard_laser(img, grid,
                                                                 pattern_pose)
                if len(profile2d) > 0:
                    mplot3d.draw_frame(pattern_pose)
                    mplot3d.draw_points(profile3d, color=(1, 1, 1))
                    mplot3d.draw_points(fit.apply_transformation(
                        self.profile.trans, profile2d), color=(0, 1, 1))
        plane, inliers = self.find_best_plane(self.profiles3d)
        mplot3d.draw_plane(plane, self.profiles3d[inliers])
        plane_pose = fit.get_plane_pose(plane)
        points3d = calc.transform_points2d(self.targets, plane_pose)
        mplot3d.draw_points(points3d, color=(1, 1, 1))
        mplot3d.draw_points(self.profiles3d, color=(1, 1, 0))
        mplot3d.draw_points(self.profiles3d[inliers], color=(0, 0, 1))
        # Draws camera and laser poses
        mplot3d.draw_frame(self.profile.pose, label='laser')
        mplot3d.draw_camera(self.camera_pose, color=(0.8, 0.8, 0.8))
        mplot3d.show()

    def draw_location_results(self, img, frame_rate):
        text = ''
        gray = self.profile.threshold_image(img)
        grid = self.find_chessboard(img)
        if grid is not None:
            img = draw_chessboard(img, grid)
        text += 'FPS: %.1f\n' % (frame_rate)
        profile_points = self.profile.peak_profile(gray)
        img = draw_points(img, profile_points, color=RED, thickness=2)
        img = draw_text(img, (10, 20), text)
        return img


# ----------------------------------------------------------------------------

class HandEyeCalibration():
    """Class for hand eye calibration.

    It implements the TsaiLenz method.

    References: R.Tsai, R.K.Lenz "A new Technique for Fully Autonomous
            and Efficient 3D Robotics Hand/Eye calibration", IEEE
            trans. on robotics and Automaion, Vol.5, No.3, June 1989
    """

    def __init__(self):
        pass

    def _skew(self, v):
        if len(v) == 4:
            v = v[:3]/v[3]
        skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
        return skv - skv.T

    def _quat_to_rot(self, q):
        """
        Converts a unit quaternion (3x1) to a rotation matrix (3x3).
        %
        %   R = quat2rot(q)
        %
        %   q - 3x1 unit quaternion
        %   R - 4x4 homogeneous rotation matrix (translation component is zero)
        %       q = sin(theta/2) * v
        %       teta - rotation angle
        %       v    - unit rotation axis, |v| = 1
        %
        """
        q = np.array(q).reshape(3, 1)
        p = np.dot(q.T, q).reshape(1)[0]
        if p > 1:
            print('Warning: quaternion greater than 1')
        w = np.sqrt(1 - p)
        R = np.eye(4)
        R[:3, :3] = 2*q*q.T + 2*w*self._skew(q) + np.eye(3) - 2*np.diag([p, p, p])
        return R

    def _rot_to_quat(self, R):
        """
        Converts a rotation matrix (3x3) to a unit quaternion (3x1).
        %
        %    q = rot2quat(R)
        %
        %    R - 3x3 rotation matrix, or 4x4 homogeneous matrix
        %    q - 3x1 unit quaternion
        %        q = sin(theta/2) * v
        %        teta - rotation angle
        %        v    - unit rotation axis, |v| = 1
        %
        """
        w4 = 2 * np.sqrt(1 + np.trace(R[:3, :3]))
        q = np.array([(R[2, 1] - R[1, 2]) / w4,
                      (R[0, 2] - R[2, 0]) / w4,
                      (R[1, 0] - R[0, 1]) / w4])
        return q

    def solve(self, Hgs, Hcs):  # list of poses

        # // Calculate rotational component
        M = len(Hgs)
        lhs = []
        rhs = []
        for i in range(M):
            for j in range(i+1, M):
                Hgij = np.dot(la.inv(Hgs[j]), Hgs[i])
                Pgij = 2 * self._rot_to_quat(Hgij)
                Hcij = np.dot(Hcs[j], la.inv(Hcs[i]))
                Pcij = 2 * self._rot_to_quat(Hcij)
                lhs.append(self._skew(Pgij + Pcij))
                rhs.append(Pcij - Pgij)
        lhs = np.array(lhs)
        lhs = lhs.reshape(lhs.shape[0]*3, 3)
        rhs = np.array(rhs)
        rhs = rhs.reshape(rhs.shape[0]*3)
        Pcg_, res, rank, sing = np.linalg.lstsq(lhs, rhs)
        Pcg = 2 * Pcg_ / np.sqrt(1 + np.dot(Pcg_, Pcg_))
        Rcg = self._quat_to_rot(Pcg / 2)

        # // Calculate translational component
        lhs = []
        rhs = []
        for i in range(M):
            for j in range(i+1, M):
                Hgij = np.dot(la.inv(Hgs[j]), Hgs[i])
                Hcij = np.dot(Hcs[j], la.inv(Hcs[i]))
                lhs.append(Hgij[:3, :3] - np.eye(3))
                rhs.append(np.dot(Rcg[:3, :3], Hcij[:3, 3]) - Hgij[:3, 3])
        lhs = np.array(lhs)
        lhs = lhs.reshape(lhs.shape[0]*3, 3)
        rhs = np.array(rhs)
        rhs = rhs.reshape(rhs.shape[0]*3)
        Tcg, res, rank, sing = np.linalg.lstsq(lhs, rhs)

        Hcg = np.eye(4)
        Hcg[:3, :3] = Rcg[:3, :3]
        Hcg[:3, 3] = Tcg
        return Hcg


def read_poses(filename):
    with open(filename, 'r') as f:
        pose = eval(f.read())
        tool_pose = calc.quatpose_to_matrix(*(np.array(pose[0]),
                                              np.array(pose[1])))
    return tool_pose


def read_calibration_data(dirname):
    frame_filenames = sorted(glob.glob(os.path.join(dirname, 'frame*.png')))
    pose_filenames = sorted(glob.glob(os.path.join(dirname, 'pose*.txt')))
    images = [read_image(filename) for filename in frame_filenames]
    tool_poses = [read_poses(filename) for filename in pose_filenames]
    return images, tool_poses


if __name__ == '__main__':
    from mlabplot import MPlot3D

    dirname = '../../data'
    images, tool_poses = read_calibration_data(dirname)

    laser_profile = Profile(axis=1, thr=180, method='pcog')
    laser_calibration = LaserCalibration(grid_size=(7, 6),
                                         square_size=0.010,
                                         profile=laser_profile)
    laser_calibration.find_calibration_3d(images)

    pattern_poses = laser_calibration.pattern_poses
    profiles = [laser_profile.profile_points(img)
                for img in images]
    lines = [laser_calibration.find_best_line2d(profile2d)
             for profile2d in profiles]

    for k, img in enumerate(images):
        grid = laser_calibration.grids[k]
        imgc = laser_calibration.draw_chessboard(img.copy(), grid)
        if len(profiles[k]) > 0:
            line, inliers = lines[k]
            imgc = draw_points(imgc, profiles[k], color=PURPLE, thickness=2)
            line = ((0, int(line[0] * 0 + line[1])),
                    (int(img.shape[1]), int(line[0] * img.shape[1] + line[1])))
            imgc = draw_points(imgc, profiles[k][inliers],
                               color=RED, thickness=2)
            imgc = draw_line(imgc, line, color=RED, thickness=2)
            #cv2.imwrite('board%i.png' %k, imgc)
        show_images([imgc], wait=1000)

    laser_calibration.show_calibration_3d()
    laser_calibration.save_parameters('../../config/profile3d.yaml')

    poses_checker, poses_tool = [], []
    for k in range(len(tool_poses)):
        pose_checker, pose_tool0 = None, None
        if pattern_poses[k] is not None:
            pose_checker = calc.pose_to_matrix(pattern_poses[k])
            pose_tool0 = tool_poses[k]
        poses_checker.append(pose_checker)
        poses_tool.append(pose_tool0)

    pchecker, ptool = [], []
    for k in range(len(poses_checker)):
        if poses_checker[k] is not None:
            pchecker.append(poses_checker[k])
            ptool.append(poses_tool[k])
    poses_checker, poses_tool = pchecker, ptool
    poses_ichecker = [calc.matrix_invert(pose) for pose in poses_checker]
    poses_itool = [calc.matrix_invert(pose) for pose in poses_tool]

    print 'Hand Eye Calibration Solution'
    tlc = HandEyeCalibration()
    T2C = tlc.solve(poses_tool, poses_checker)
    W2K = tlc.solve(poses_itool, poses_ichecker)
    print 'Tool2Camera:', calc.matrix_to_rpypose(T2C)
    print 'World2Checker:', calc.matrix_to_rpypose(W2K)

    mplot3d = MPlot3D(scale=0.0025)
    pp = laser_calibration.pattern_points
    world_frame = calc.rpypose_to_matrix([0, 0, 0], [0, 0, 0])
    #mplot3d.draw_frame(calc.matrix_to_pose(world_frame), label='world_frame')
    for k, tool_frame in enumerate(poses_tool):
        WC = calc.matrix_compose((tool_frame, T2C))
        #mplot3d.draw_transformation(world_frame, tool_frame)
        #mplot3d.draw_transformation(tool_frame, WC, 'tool_pose%i' % k,
        #                                            'camera_pose%i' % k)
        WK = calc.matrix_compose((WC, poses_checker[k]))
        print 'Checker %i ->' % k, WK
        print np.allclose(W2K, WK, atol=0.0001)
        mplot3d.draw_frame(calc.matrix_to_pose(WK))
        mplot3d.draw_points(calc.points_transformation(WK, pp),
                            color=(1, 1, 0))
        mplot3d.draw_frame(calc.matrix_to_pose(W2K))
        mplot3d.draw_points(calc.points_transformation(W2K, pp),
                            color=(1, 1, 1))
        img, grid = images[k], laser_calibration.grids[k]
        if grid is not None:
            chessboard_pose = pattern_poses[k]
            profile3d, profile2d = laser_calibration.get_chessboard_laser(img, grid, chessboard_pose)
            if len(profile2d) > 0:
                mplot3d.draw_points(calc.points_transformation(WC, profile3d)[10:-1:25],
                                    color=(1, 1, 1))
                # points3d = laser_profile.profile_to_points3d(profiles[k],
                #                                              laser_profile.homography,
                #                                              laser_profile.pose)
                # print points3d, laser_profile.pose, laser_profile.homography
                # mplot3d.draw_points(calc.points_transformation(WC, points3d[0:-1:50]), color=(1, 0, 0))
                points3d = fit.apply_transformation(laser_profile.trans, profiles[k])
                print points3d, laser_profile.trans
                mplot3d.draw_points(calc.points_transformation(WC, points3d[0:-1:25]), color=(0, 1, 0))
    mplot3d.show()
