import cv2
import yaml
import numpy as np

from profile import Profile


class CrossDetector():
    def __init__(self):
        self.profile0 = Profile()
        self.profile0.load_configuration('../config/triangulation0.yml')
        self.profile1 = Profile()
        self.profile1.load_configuration('../config/triangulation1.yml')

    def fit_line(self, points, size):
        vx, vy, cx, cy = cv2.fitLine(points, cv2.cv.CV_DIST_HUBER, 0, 0.01, 0.01)
        m, b = vy / vx, cy - (vy / vx) * cx
        w, h = size
        p0 = np.float32([0, b])
        p1 = np.float32([-b / m, 0])
        p2 = np.float32([w, m * w + b])
        p3 = np.float32([(h - b) / m, h])
        points = []
        if p0[1] > 0 and p0[1] < h:
            points.append(p0)
        if p1[0] > 0 and p1[0] < w:
            points.append(p1)
        if p2[1] > 0 and p2[1] < h:
            points.append(p2)
        if p3[0] > 0 and p3[0] < w:
            points.append(p3)
        points = np.float32(points)
        return points

    def draw_line(self, image, line, color=(0,0,0), thickness=1):
        (x0, y0), (x1, y1) = line
        point0 = (int(round(x0)), int(round(y0)))
        point1 = (int(round(x1)), int(round(y1)))
        cv2.line(image, point0, point1, color, thickness)
        return image

    def cross_measurement(self, img, profile0, profile1):
        size = (img.shape[1], img.shape[0])
        img = self.profile0.draw_points(img, profile0, color=(0,0,255), thickness=2)
        img = self.profile1.draw_points(img, profile1, color=(0,255,255), thickness=2)
        if len(profile0) and len(profile1):
            lineh = self.fit_line(profile0, size)
            linev = self.fit_line(profile1, size)
            if len(lineh) == 2 and len(linev) == 2:
                img = self.draw_line(img, lineh, color=(0,0,255))
                img = self.draw_line(img, linev, color=(0,255,255))
        return img

    def cross_processing(self, img):
        gray = self.profile0.threshold_image(img)
        profile0 = self.profile0.peak_profile(gray)
        profile1 = self.profile1.peak_profile(gray)
        cross_img = self.cross_measurement(img, profile0, profile1)
        return cross_img


if __name__ == '__main__':
    import image

    img = image.read_image('../data/cframe0001.png')
    image.show_image(img)

    cross = CrossDetector()
    cross_img = cross.cross_processing(img)
    image.show_image(cross_img)

    #camera = Webcam(device=1)
    #camera.set_size((800, 600))
    #camera.set_parameters(0.30, 0.20, 0.10)
    #camera.run(callback=lambda img: cross.cross_processing(img))

#    calibration0 = LaserCalibration(grid_size=grid_size, square_size=square_size, axis=0)
#    calibration0.find_calibration_3d(filenames)
#    calibration0.show_calibration_3d(filenames)
#    calibration0.save_parameters('../config/triangulation0.yml')

#    calibration1 = LaserCalibration(grid_size=grid_size, square_size=square_size, axis=1)
#    calibration1.find_calibration_3d(read_images(filenames))
#    calibration1.show_calibration_3d(read_images(filenames))
#    calibration1.save_parameters('triangulation1.yml')
