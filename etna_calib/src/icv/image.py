import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def read_image(filename):
    image = cv2.imread(filename)
    return image

def read_images(pattern_name):
    filenames = sorted(glob.glob(pattern_name))
    images = [read_image(filename) for filename in filenames]
    return images

def write_image(filename, image):
    cv2.imwrite(filename, image)

def get_size(image):
    height, width = image.shape[:2]
    return width, height

def scale_image(image, scale=1.0):
    if not scale == 1.0:
        height, width = image.shape[:2]
        image = cv2.resize(image, (int(scale * width), int(scale * height)))
    return image

def rotate_image(image, angle, scale=1.0):
    image_center = (image.shape[1] / 2, image.shape[0] / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)#LINEAR)
    return result

def flip_image(image):
    return cv2.flip(image, 0)


def _imshow(image, cmap='gray'):
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap=cmap)

def show_image(image1, image2=None, cmap='gray'):
    plt.figure()
    if image2 == None:
        plt.subplot(1,1,1)
        _imshow(image1, cmap=cmap)
    else:
        plt.subplot(1,2,1)
        _imshow(image1, cmap=cmap)
        plt.subplot(1,2,2)
        _imshow(image2, cmap=cmap)
    plt.show()


def _show_image(image, wait=0, scale=1.0):
    cv2.imshow('Image', scale_image(image, scale))
    cv2.waitKey(wait)

def show_images(images, wait=500, scale=0.5):
    for image in images:
        _show_image(image, wait, scale)
    cv2.destroyAllWindows()


# Color filters

def bgr2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def rgb2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def blur(image, ksize=(3,3)):
    return cv2.GaussianBlur(image, ksize, 0)

def threshold(image, thr=39):
    return cv2.threshold(image, thr, 255, cv2.THRESH_BINARY)[1]

def normalize(image):
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    return image

# Morphological filters

def erode(image, size=3, itera=1):
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel, iterations=itera)

def dilate(image, size=3, itera=1):
    #kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel, iterations=itera)

def close(image, size=3, itera=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=itera)

def skeleton(gray, thr=127):
    img = threshold(gray, thr=thr)
    skel = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img):
        eroded = cv2.erode(img, element)
        dilated = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, dilated)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
    return skel

# Edge filters

def canny(gray):
    return cv2.Canny(gray, 80, 120)

#def sobel(image):
#    # Gradient-X, Gradient-Y
#    grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
#    grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)
#
#    abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
#    abs_grad_y = cv2.convertScaleAbs(grad_y)
#
#    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
#
#def scharr(image):
#    image = cv2.GaussianBlur(image, (3,3), 0)
#
#    grad_x = cv2.Scharr(image, cv2.CV_16S, 1, 0)
#    grad_y = cv2.Scharr(image, cv2.CV_16S, 0, 1)
#
#    abs_grad_x = cv2.convertScaleAbs(grad_x)
#    abs_grad_y = cv2.convertScaleAbs(grad_y)
#
#    return cv2.add(abs_grad_x, abs_grad_y)


#TODO: Refactor. Delete the class pattern

class Pattern():
    def __init__(self, grid_size=(5, 5), square_size=1.0, grid_orig=(0, 0)):
        self.grid_size = grid_size
        self.square_size = square_size
        self.grid_orig = grid_orig

    def get_pattern_points(self):
        points = np.zeros((np.prod(self.grid_size), 2), np.float32)
        points[:,:2] = np.indices(self.grid_size).T.reshape(-1, 2)
        points *= self.square_size
        points += self.grid_orig
        return points

    def find_chessboard(self, img):
        img = bgr2gray(img)
        rows, cols = self.grid_size
        found, corners = cv2.findChessboardCorners(img, (rows, cols))
        grid = None
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1) # termination criteria
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            grid = corners.reshape((rows, cols, 2))
        return grid

    def find_circlesboard(self, image, exp=3.0):
        img_gray = bgr2gray(image)
        img_blur = blur(img_gray)
        img_inv = 255 - (cv2.pow(img_blur.astype(np.float) / 255, exp) * 255).astype(np.uint8)
        points = cv2.findCirclesGridDefault(img_inv, self.grid_size)[1]
        return points.reshape((-1, 2))

    def draw_circlesboard(self, image, points, color=(255, 0, 0)):
        img_cir = image.copy()
        for pnt in points:
            cx, cy = np.int32(pnt)
            cv2.circle(img_cir, (cx, cy), 3, color, -1)
        return img_cir

    def draw_squaresboard(self, image, points, color=(0, 0, 255)):
        img =  image.copy()
        points = points.reshape((self.grid_size[1], self.grid_size[0], 2))
        for k in range(self.grid_size[1]): # Columns
            cv2.polylines(img, np.int32([points[k,:]]), False, (0, 0, 255), 1)
        for k in range(self.grid_size[0]): # Rows
            cv2.polylines(img, np.int32([points[:,k]]), False, (0, 0, 255), 1)
        return img


# Homography functions

def find_homography(points, targets):
    """Finds the homography projective transformation matrix."""
    homography = cv2.findHomography(points, targets, cv2.RANSAC)[0]
    return homography

def homography_points(points2d, homography):
    """Transforms the image points to get the coordinates in a homography plane."""
    pnts = np.float32([np.dot(homography, np.float32([point[0], point[1], 1])) for point in points2d])
    pnts = np.float32([pnt / pnt[2] for pnt in pnts])
    return pnts[:,:2]

def homography_image(image, homography, (width, height)):
    """Transforms the image using the projective homography transformation."""
    image_trans = cv2.warpPerspective(image, homography, (width, height))
    return image_trans


def test_homography_transformation(filename='../data/pattern1.png'):
    # Test projective transformation
    width, height = 180, 180

    # Projective transformation
    image = read_image(filename)
    pattern = Pattern((5, 5), 40, (10, 10))
    targets = pattern.get_pattern_points()
    points =  pattern.find_circlesboard(image)
    homography = find_homography(points, targets)
    print homography

    points = np.array([[294, 187]])
    print homography_points(points, homography)

    img_trans = homography_image(image, homography, (width, height))
    show_image(pattern.draw_circlesboard(image, points), img_trans)






if __name__ == "__main__":
    img1 = read_image('../data/frame0001.png')
    img2 = read_image('../data/frame0002.png')
    show_image(img1, img2, cmap='hot')

    image = rotate_image(img1, 45)
    show_image(image)

    test_homography_transformation(filename='../data/pattern1.png')
