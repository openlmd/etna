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
    if image2 is None:
        plt.subplot(1, 1, 1)
        _imshow(image1, cmap=cmap)
    else:
        plt.subplot(1, 2, 1)
        _imshow(image1, cmap=cmap)
        plt.subplot(1, 2, 2)
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


def blur(image, ksize=(3, 3)):
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


if __name__ == "__main__":
    img1 = read_image('../data/frame0001.png')
    img2 = read_image('../data/frame0002.png')
    show_image(img1, img2, cmap='hot')

    image = rotate_image(img1, 45)
    show_image(image)
