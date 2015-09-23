#!/usr/bin/env python
import cv2
import time


class Webcam():
    def __init__(self, device=0):
        self.cam = cv2.VideoCapture(device)
        if self.cam is None or not self.cam.isOpened():
            print 'Warning: unable to open video source: ', device
            #sys.exit()
            self.cam = cv2.VideoCapture(0)
        
        self.configure(width=1280, height=720)
        print 'Camera device:', device
        print 'Sensor size:', self.sensor_size()
        print 'Image size:', self.get_size()
        brightness, contrast, saturation = self.get_parameters()
        self.frame = 0        
        self.counter = 0        
        self.time0 = time.time()
        
    def configure(self, width=640, height=480):
        self.set_size((width, height))
        
    def get_size(self):
        width = int(self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        return width, height
        
    def set_size(self, (width, height)):
        self.cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        
    def get_parameters(self):
        brightness = self.cam.get(cv2.cv.CV_CAP_PROP_BRIGHTNESS)
        contrast = self.cam.get(cv2.cv.CV_CAP_PROP_CONTRAST)
        saturation = self.cam.get(cv2.cv.CV_CAP_PROP_SATURATION)
        print 'Brightness:', brightness
        print 'Contrast:', contrast
        print 'Saturation:', saturation
        return brightness, contrast, saturation
        
    def set_parameters(self, brightness, contrast, saturation):
        self.cam.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, brightness)
        self.cam.set(cv2.cv.CV_CAP_PROP_CONTRAST, contrast)
        self.cam.set(cv2.cv.CV_CAP_PROP_SATURATION, saturation)
        
    def sensor_size(self):
        width, height = self.get_size()
        self.set_size((10000, 10000))
        sensor_size = self.get_size()
        self.set_size((width, height))
        return sensor_size
        
    def frame_rate(self):
        frame_rate = 1. / (self.time0 - self.time1)
        return frame_rate
        
    def frame_count(self):
        return self.frame
        
    def capture(self):
        self.frame += 1
        self.time1 = self.time0
        self.time0 = time.time()
        rval, frame = self.cam.read()
        return frame
        
    def snapshoot(self, filename):
        flag, frame = self.cam.read()
        cv2.imwrite(filename, frame)
    
    def on_mouse(self, event, x, y, flags, params):
#        if event == cv2.cv.CV_EVENT_LBUTTONDOWN:
#            print 'Start Mouse Position: ' + str(x) + ', ' + str(y)
#        elif event == cv2.cv.CV_EVENT_LBUTTONUP:
#            print 'End Mouse Position: ' + str(x) + ', ' + str(y)
        if event == cv2.cv.CV_EVENT_RBUTTONDOWN:
            self.counter += 1
            filename = 'frame%04i.png' %self.counter
            cv2.imwrite(filename, self.frame)
            print filename
        
    def run(self, callback=None):
        cv2.namedWindow('Webcam', flags=cv2.CV_WINDOW_AUTOSIZE)
        cv2.cv.SetMouseCallback('Webcam', self.on_mouse, '')  
        while True:
            frame = self.capture()
            print 'FPS:', self.frame_rate()
            self.frame = frame.copy()
            if not callback == None:
                frame = callback(frame)
            cv2.imshow('Webcam', frame)
            key = cv2.waitKey(5)
            if not key == -1:
                key = key % 256
                print 'Key', key
                if key == 27:
                    break
        self.cam.release()
        cv2.destroyAllWindows()
        
              

if __name__ == '__main__':
    import sys
    import getopt
    
    options, sources = getopt.getopt(sys.argv[1:], '', ['size=',])
    
    size = (800, 600)
    for opt, arg in options:
        if opt in ('--size'):
            size = eval(arg)
    print size
    
    webcam = Webcam(device=1)
    webcam.set_size(size)
    brightness, contrast, saturation = webcam.get_parameters()
    webcam.set_parameters(0.3, 0.2, 0.1)
    width, height = webcam.get_size()
    webcam.run(callback=lambda img: cv2.resize(img, (width/2, height/2)))  
   
