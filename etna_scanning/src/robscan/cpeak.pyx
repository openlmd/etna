cimport cython

import numpy as np
cimport numpy as np


@cython.boundscheck(False)
cdef horizontal_cog_detector(np.ndarray[np.float32_t, ndim=2] img):
    cdef unsigned int rows = img.shape[0]
    cdef unsigned int cols = img.shape[1]
    cdef float pnt0, pnt1
    cdef np.ndarray[np.float32_t, ndim=2, negative_indices=False] peaks = np.zeros((cols, 2), np.float32)
    cdef unsigned int x, y
    cdef float max1, sum1, area1
    cdef float max2, sum2, area2
    cdef float peak
    cdef unsigned int count = 0
    for x in range(cols):
        max1, sum1, area1 = 0.0, 0.0001, 0.0
        max2, sum2, area2 = 0.0, 0.0, 0.0
        for y in range(rows):
            pnt1 = pnt0
            pnt0 = img[y,x]
            if pnt0 > 0:
                if not pnt1 > 0:
                    #print 'New'
                    max2, sum2, area2 = 0.0, 0.0001, 0.0
            else:
                if (pnt1 > 0):
                    #print 'End New'
                    if max2 > max1:
                        max1, sum1, area1 = max2, sum2, area2
            if pnt0 > 0 or pnt1 > 0:
                if pnt0 > max2:
                    max2 = pnt0
                sum2 = sum2 + (pnt0 + pnt1)
                area2 = area2 + ((pnt0 + pnt1) * (2 * y - 1))
        peak = area1 / (2 * sum1)
        if peak > 0:
            peaks[count] = x, peak
            count = count + 1
    return peaks[:count]

@cython.boundscheck(False)
cdef vertical_cog_detector(np.ndarray[np.float32_t, ndim=2] img):
    cdef unsigned int rows = img.shape[0]
    cdef unsigned int cols = img.shape[1]
    cdef float pnt0, pnt1
    cdef np.ndarray[np.float32_t, ndim=2, negative_indices=False] peaks = np.zeros((rows, 2), np.float32)
    cdef unsigned int x, y
    cdef float max1, sum1, area1
    cdef float max2, sum2, area2
    cdef float peak
    cdef unsigned int count = 0
    for y in range(rows):
        max1, sum1, area1 = 0.0, 0.0001, 0.0
        max2, sum2, area2 = 0.0, 0.0, 0.0
        for x in range(cols):
            pnt1 = pnt0
            pnt0 = img[y,x]
            if pnt0 > 0:
                if not pnt1 > 0:
                    #print 'New'
                    max2, sum2, area2 = 0.0, 0.0001, 0.0
            else:
                if (pnt1 > 0):
                    #print 'End New'
                    if max2 > max1:
                        max1, sum1, area1 = max2, sum2, area2
            if pnt0 > 0 or pnt1 > 0:
                if pnt0 > max2:
                    max2 = pnt0
                sum2 = sum2 + (pnt0 + pnt1)
                area2 = area2 + ((pnt0 + pnt1) * (2 * x - 1))
        peak = area1 / (2 * sum1)
        if peak > 0:
            peaks[count] = peak, y
            count = count + 1
    return peaks[:count]

def cog_detector(img, axis=0):
    img = img.astype(np.float32)
    if axis == 0:
        return horizontal_cog_detector(img)
    else:
        return vertical_cog_detector(img)


# Peak detector algorithm

# When multiple detections in a single column are given, the most significant
# among the detected is selected. The selection criteria involves the pixel
# intensity of each peak and its neighborhood, by selecting the stripe section
# with higher width x intensity product.

@cython.boundscheck(False)
cdef horizontal_peak_detector(np.ndarray[np.float32_t, ndim=2] img):
    cdef unsigned int rows = img.shape[0]
    cdef unsigned int cols = img.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2, negative_indices=False] peaks = np.zeros((cols, 2), np.float32)
    cdef unsigned int i, j
    cdef float x0, y0
    cdef float x1, y1
    cdef float diff
    cdef float peak
    cdef unsigned int count = 0
    for i in range(cols):
        y0, y1 = 0, 0
        x0, x1 = 0, 0
        for j in range(2, rows-2):
            # First derivative approximated using a second order filter
            diff = img[j+2,i] + img[j+1,i] - img[j-1,i] - img[j-2,i]
            if not y1:
                if diff > 0:
                    y0 = diff
                    x0 = j
                elif diff < 0 and y0 > 0:
                    y1 = diff
                    x1 = j
        peak = x0 - y0 * (x1 - x0) / ((y1 - y0) + 0.00001)
        if peak > 0:
            peaks[count] = i, peak
            count = count + 1
    return peaks[:count]

@cython.boundscheck(False)
cdef vertical_peak_detector(np.ndarray[np.float32_t, ndim=2] img):
    cdef unsigned int rows = img.shape[0]
    cdef unsigned int cols = img.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2, negative_indices=False] peaks = np.zeros((cols, 2), np.float32)
    cdef unsigned int i, j
    cdef float x0, y0
    cdef float x1, y1
    cdef float diff
    cdef float peak
    cdef unsigned int count = 0
    for i in range(rows):
        y0, y1 = 0, 0
        x0, x1 = 0, 0
        for j in range(2, cols-2):
            diff = img[i,j+2] + img[i,j+1] - img[i,j-1] - img[i,j-2]
            if not y1:
                if diff > 25:
                    y0 = diff
                    x0 = j
                elif diff < -25 and y0 > 0:
                    y1 = diff
                    x1 = j
        peak = x0 - y0 * (x1 - x0) / ((y1 - y0) + 0.00001)
        if peak > 0:
            peaks[count] = peak, i
            count = count + 1
    return peaks[:count]

def peak_detector(img, axis=0):
    img = img.astype(np.float32)
    if axis==0:
        return horizontal_peak_detector(img)
    else:
        return vertical_peak_detector(img)

