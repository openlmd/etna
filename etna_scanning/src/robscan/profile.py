import cv2
import yaml
import numpy as np


import pyximport
pyximport.install()

from cpeak import cog_detector
from cpeak import peak_detector


class Profile():
    def __init__(self, axis=1, thr=180, method='max'):
        self.thr = thr
        self.axis = axis
        self.method = method
        self.trans = np.zeros((4, 3))
        self.homography = np.eye(3)
        self.pose = (np.eye(3), np.zeros(3))

    def load_configuration(self, filename):
        with open(filename, 'r') as f:
            data = yaml.load(f)
        self.thr = data['thr']
        self.axis = data['axis']
        self.method = data['method']
        self.trans = np.array(data['trans'])
        self.homography = np.array(data['homo'])
        self.pose = (np.array(data['pose']['R']),
                     np.array(data['pose']['t']))
        return data

    def save_configuration(self, filename):
        data = dict(thr=self.thr,
                    axis=self.axis,
                    method=self.method,
                    trans=self.trans.tolist(),
                    homo=self.homography.tolist(),
                    pose=dict(R=self.pose[0].tolist(),
                              t=self.pose[1].tolist()))
        with open(filename, 'w') as f:
            f.write(yaml.dump(data))
        return data

    def threshold_image(self, image, channel=2):
        if len(image.shape) > 2:
            img = image[:, :, channel].copy()
            img[image[:, :, channel] < self.thr] = 0
        else:
            img = image.copy()
            img[image < self.thr] = 0
        return img

    def peak_max_profile(self, img):
        """Returns the profile defined by maximum values."""
        if self.axis == 0:
            y = np.argmax(img, axis=self.axis)
            x = np.arange(len(y))
            v = y > 0
        else:
            x = np.argmax(img, axis=self.axis)
            y = np.arange(len(x))
            v = x > 0
        points = np.vstack((x[v], y[v])).T
        return points.astype(np.float32)

    def peak_cog_profile(self, img):
        """Returns the profile measured by the center of gravity method."""
        if self.axis == 0:
            pimg = img * np.arange(img.shape[self.axis]).reshape((-1, 1))
            y = np.sum(pimg, axis=self.axis) / (np.sum(img, axis=self.axis) + 0.00001)
            v = y > 0
        else:
            pimg = img * np.arange(img.shape[self.axis]).reshape((1, -1))
            x = np.sum(pimg, axis=self.axis) / (np.sum(img, axis=self.axis) + 0.00001)
            y = np.arange(len(x))
            v = x > 0
        points = np.vstack((x[v], y[v])).T
        return points.astype(np.float32)

    def peak_pcog_profile(self, img):
        return cog_detector(img, axis=self.axis)

    def peak_diff_profile(self, img):
        return peak_detector(img, axis=self.axis)

    def peak_profile(self, img):
        if self.method == 'max':
            return self.peak_max_profile(img)
        elif self.method == 'cog':
            return self.peak_cog_profile(img)
        elif self.method == 'pcog':
            return self.peak_pcog_profile(img)
        elif self.method == 'peak':
            return self.peak_diff_profile(img)

    def profile_points(self, image):
        blur = cv2.blur(image, (3, 3))
        gray = self.threshold_image(blur)
        profile = self.peak_profile(gray)
        return profile

    def profile_to_points3d(self, profile, homography, pose, minz=0, maxz=1000):
        """Transforms the profile image points of the laser using the
        homography and the pose of the laser plane to get the points
        in the camera frame."""
        points3d = []
        if len(profile) > 0:
            pnts = np.float32([np.dot(homography,
                                      np.float32([x, y, 1])) for x, y in profile])
            points = np.float32([pnt / pnt[2] for pnt in pnts])
            points3d = np.float32([np.dot(pose[0],
                                   np.float32([x, y, 0])) + pose[1] for x, y in points[:, :2]])
            #points3d = points3d[points3d[:, 2] > minz]
            #points3d = points3d[points3d[:, 2] < maxz]
        return points3d

    def transform_profile(self, profile):
        points3d = []
        if len(profile):
            points = np.hstack((profile, np.ones((len(profile), 1))))
            pnts3d = np.vstack([np.dot(self.trans, point) for point in points])
            points3d = pnts3d[:, :3] / pnts3d[:, 3].reshape(len(pnts3d), 1)
        return points3d

    def points_profile(self, image, homography=None, pose=None):
        """Gets the laser coordinate points in the camera frame from the peak
        profile detection in the image. Projects the laser profile on a plane.
        """
        profile = self.profile_points(image)
        if homography is None or pose is None:
            #ERROR: transformation estimation
            #points3d = self.transform_profile(profile)
            points3d = self.profile_to_points3d(profile, self.homography, self.pose)
        else:
            points3d = self.profile_to_points3d(profile, homography, pose)
        return points3d, profile

    def draw_points(self, image, points, color=(0, 0, 0), thickness=1):
        if not points.dtype == np.int32:
            points = np.round(points, 0).astype(np.int32)
        for cx, cy in points:
            cv2.circle(image, (cx, cy), thickness, color, -1)
        return image

    def profile_measurement(self, frame):
        points3d, profile = self.points_profile(frame)
        frame = self.draw_points(frame, profile,
                                 color=(0, 0, 255), thickness=2)
        if len(points3d) > 0:
            print points3d
            point3d = points3d[len(points3d)/5]
            cv2.putText(frame, '%s' % point3d, (11, 22),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255),
                        thickness=1, lineType=cv2.CV_AA)
        return frame


if __name__ == '__main__':
    import image

    img = image.read_image('../data/utest9.png')
    profile0 = Profile(axis=1, thr=180, method='pcog')
    image.show_image(profile0.profile_measurement(img))
    #cv2.imwrite('peak.png', profile0.profile_measurement(img))
    #profile0.load_configuration('triangulation0.yml')

    # Camera test
    #from webcam import Webcam
    #camera = Webcam(device=1)
    #camera.set_size((800, 600))
    #camera.set_parameters(0.30, 0.20, 0.10)
    #camera.run(callback=lambda img: profile0.profile_measurement(img))
