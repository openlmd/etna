import cv2
import numpy as np
from calculate import *


# Basic colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 128, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
SILVER = (192, 192, 192)
GRAY = (128, 128, 128)
MAROON = (0, 0, 128)
OLIVE = (0, 128, 128)
LIME = (0, 128, 128)
PURPLE = (128, 0, 128)
TEAL = (128, 128, 0)
NAVY = (128, 0, 0)


def draw_line(image, line, color=WHITE, thickness=1):
    (x0, y0), (x1, y1) = line
    point0 = (int(round(x0)), int(round(y0)))
    point1 = (int(round(x1)), int(round(y1)))
    cv2.line(image, point0, point1, color, thickness)
    return image

def draw_box(image, box, color=WHITE, thickness=1):
    (x, y), (w, h) = box
    cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
    return image

def draw_rectangle(image, rect, color=RED, size=None):
    if size == None:
        size = int(np.round(float(img.shape[0]) / 512))
    points = rectangle_to_polygon(rect)
    cv2.drawContours(image, [points], -1, color, size)
    return image

def draw_circle(image, circle, color=WHITE, thickness=1):
    (cx, cy), r  = circle
    cv2.circle(image, (cx, cy), r, color, thickness)
    return image

def draw_ellipse(image, rect, color=BLUE):
    size = int(np.round(float(image.shape[0]) / 512))
    cv2.ellipse(image, rect, color, size)
    return image

def draw_arrow(image, line, color=WHITE, thickness=1):
    (x0, y0), (x1, y1) = line
    # Draw arrow tail
    cv2.line(image, (x0, y0), (x1, y1), color, thickness)
    # Magnitude and angle of the arrow
    magnitude = np.sqrt((y1-y0)**2 + (x1-x0)**2)
    angle = np.arctan2(y0-y1, x0-x1)
    tipsize = 0.1 * magnitude
    # Starting point of the first line of the arrow head
    p0 = (int(x1 + tipsize * np.cos(angle + np.pi/6)),
          int(y1 + tipsize * np.sin(angle + np.pi/6)))
    cv2.line(image, p0, (x1, y1), color, thickness)
    # Starting point of the second line of the arrow head
    p0 = (int(x1 + tipsize * np.cos(angle - np.pi/6)),
          int(y1 + tipsize * np.sin(angle - np.pi/6)))
    cv2.line(image, p0, (x1, y1), color, thickness)
    return image

def draw_points(image, points, color=WHITE, thickness=1):
    if not points.dtype == np.int32:
        points = np.round(points, 0).astype(np.int32)
    if thickness == 1:
        [cv2.drawContours(image, np.int32([[point]]), -1, color, thickness) for point in points]
    else:
        for cx, cy in points:
            cv2.circle(image, (cx, cy), thickness, color, -1)
    return image

def draw_lines(image, lines, color=WHITE, thickness=1):
    for (x0, y0), (x1, y1) in lines:
        cv2.line(image, (x0, y0), (x1, y1), color, thickness)
    return image

def draw_boxes(image, boxes, color=WHITE, thickness=1):
    for box in boxes:
        draw_box(image, box, color, thickness)
    return image

# TODO: Implement this function drawing polygons. rectangle_to_polygon function
def draw_rectangles(image, rectangles, color=WHITE, thickness=1):
    for (cx, cy), (w, h), a in rectangles:
        #cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
        print 'Rectangle:', cx, cy, w, h, a
    return image

def draw_polygons(image, polygons, color=WHITE, thickness=1):
    img = image.copy()
    for points in polygons:
        points = points.reshape((-1,1,2))
        cv2.polylines(img, [points], True, color, thickness)
    return img

def draw_circles(image, circles, color=WHITE, thickness=-1):
    for (cx, cy), r in circles:
        cv2.circle(image, (cx, cy), r, color, thickness)
    return image

def draw_ellipses(image, ellipses, color=WHITE, thickness=1):
    for (x, y), (w, h), a in ellipses:
        cv2.ellipse(image, (x, y), (w, h), a, 0, 360, color, thickness)
    return image


# TODO: Refactoring

def draw_contours(image, contours, color=(255,0,0), size=None):
    img = image.copy()
    if size == None:
        size = int(np.round(float(img.shape[0]) / 512))
    if len(img.shape) < 3 and type(color) != int:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, contours, -1, color, size)
    return img

def draw_texts(image, text, point, scale=1.0, color=WHITE, thickness=2):
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    size, baseline = cv2.getTextSize(text, font, scale, thickness)
    print 'Text size:', size, baseline
    draw_box(img, ((point[0], point[1]+thickness), (size[0], -size[1]-thickness)), WHITE, -1)
    cv2.putText(img, text, tuple(point), font, scale, color, thickness)
    return img

def draw_text(image, text, pnt, color=LIME):
    img = image.copy()
    scale = float(img.shape[0]) / 512
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img, text, pnt, cv2.FONT_HERSHEY_PLAIN , scale, color,
                thickness=1, lineType=cv2.CV_AA)
    return img

def draw_text(image, (x, y), text, color=(255, 255, 255)):
    for s in text.split('\n'):
        cv2.putText(image, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0),
                    thickness = 2, lineType=cv2.CV_AA)
        cv2.putText(image, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, color,
                    lineType=cv2.CV_AA)
        x, y = x, y + 20
    return image

def draw_contours_mask(image, contours):
    mask = np.zeros(image.shape[:2], dtype='uint8')
    cv2.drawContours(mask, contours, -1, 255, -1)
    return mask

def draw_rectangle_mask(image, rect):
    points = get_polygon_from_rectangle(rect)
    return draw_contours_mask(image, [points])


def draw_frame(image, rect):
    point, size, angle = rect
    image = image.copy()
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    size = np.array(size) / 2
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), -np.sin(a)],
                    [np.sin(a), np.cos(a)]])
    pnt1 = np.array(point)
    pnt2 = pnt1 + np.dot(rot, np.array([size[0], 0]))
    pnt3 = pnt1 + np.dot(rot, np.array([0, size[1]]))
    cv2.line(image, tuple(pnt1.astype(np.int32)), tuple(pnt2.astype(np.int32)), (0, 0, 255), 2)
    cv2.line(image, tuple(pnt1.astype(np.int32)), tuple(pnt3.astype(np.int32)), (0, 255, 0), 2)
    return image

def draw_shape_model(image, model, points=[(0,0)], color=RED):
    for point in points:
        pnts = model.points + point
        image = draw_points(image, pnts, color=color)
    return image



if __name__ == "__main__":
    image = np.zeros((480, 640, 3), np.uint8)

    points = np.array([[10, 20], [150, 100]])
    line = np.float32([[0, 200], [800, 423]])

    image = draw_line(image, line, color=BLUE, thickness=1)
    image = draw_points(image, points, color=RED)
    image = draw_lines(image, [points])
    image = draw_arrow(image, points + np.array([30, 30]), color=YELLOW, thickness=1)
    image = draw_circles(image, [[points[0], 3]], color=NAVY)
    image = draw_boxes(image, [points], color=MAGENTA)
    image = draw_ellipses(image, [[points[1], points[1], 45]], color=LIME)

    points = np.array([[100,50],[200,300],[70,200],[500,100]], np.int32)
    image = draw_polygons(image, [points], color=GREEN)

    image = draw_texts(image, 'OpenCV', points[0], color=PURPLE)

    cv2.imshow('Image', image)
    cv2.waitKey()
