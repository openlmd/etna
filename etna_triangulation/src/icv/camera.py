import cv2
import ueye
import time
import numpy as np


class Camera():
    def __init__(self, pixel_clock=50, exposure_time=5):
        self.camera = ueye.Cam()
        self.camera.SetPixelClock(pixel_clock)
        self.camera.SetExposureTime(exposure_time) # 5 ms
        print self.camera.GetPixelClockRange()
        print self.camera.GetExposureRange()
        print self.camera.GetFramesPerSecond()
        #camera.configure(binning=4)

    def configure(self, binning=2):
        """Initializes the capture device."""
        self.width = 3840 / binning
        self.height = 2748 / binning
        self.binning = binning
        if self.binning == 2:
            self.camera.SetBinning((ueye.BINNING_2X_HORIZONTAL & ueye.BINNING_MASK_HORIZONTAL) | (ueye.BINNING_2X_VERTICAL & ueye.BINNING_MASK_VERTICAL))
        elif self.binning == 4:
            self.camera.SetBinning((ueye.BINNING_4X_HORIZONTAL & ueye.BINNING_MASK_HORIZONTAL) | (ueye.BINNING_4X_VERTICAL & ueye.BINNING_MASK_VERTICAL))
        #self.camera.SetAOI(ueye.SET_IMAGE_AOI, 0, 0, 1920, 1374)
        #print self.camera.SetAOI(ueye.GET_IMAGE_AOI, 0, 0, 0, 0)

    def capture(self):
        """Captures a new image frame."""
        frame = self.camera.GrabImage()#[:self.height,:self.width]
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        return frame

    def snapshoot(self, filename):
        frame = self.capture()
        cv2.imwrite(filename, frame)

    def run(self, callback=None):
        cv2.namedWindow('Camera', flags=cv2.CV_WINDOW_AUTOSIZE)
        while True:
            t0 = time.time()
            frame = self.capture()
            if not callback == None:
                frame = callback(frame)
            cv2.imshow('Camera', frame)
            t1 = time.time()
            key = cv2.waitKey(5)
            if not key == -1:
                key = key % 256
                if key == 27:
                    break
            print 'FPS:', 1. / (t1 - t0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    camera = Camera()
    camera.run()
