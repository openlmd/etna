import re
import time

import numpy as np

import calculate as calc

import platform

if platform.system() == 'Windows':
    EOL = '\n' # Windows
else:
    EOL = '\r\n' # Linux


# DXF polyline template
template = '100' + EOL
template += 'AcDbEntity' + EOL
template += '  8' + EOL
template += '(?P<layer>.+)' + EOL
template += '100' + EOL
template += 'AcDbVertex' + EOL
template += '100' + EOL
template += 'AcDb3dPolylineVertex' + EOL
template += ' 10' + EOL
template += '(?P<x>.+)' + EOL
template += ' 20' + EOL
template += '(?P<y>.+)' + EOL
template += ' 30' + EOL
template += '(?P<z>.+)' + EOL


def read_file(filename):
    f = open(filename)
    lines = f.read()
    f.close()
    pattern = re.compile(template, re.MULTILINE)
    results = pattern.findall(lines)
    layers = {}
    for layer, sx, sy, sz in results:
        if layer not in layers.keys():
            layers[layer] = []
        layers[layer].append(np.array([float(sx), float(sy), float(sz)]))
    return layers

def process_layers(layers, dist=1.0, angl=0.05):
    ang_min, ang_max = np.pi - angl, np.pi + angl
    for name, layer in layers.iteritems():
        points = []
        for k in range(len(layer)-1):
            if not len(points):
                point1 = layer[k-1]
            else:
                point1 = points[-1]
            point2, point3 = layer[k], layer[k+1]
            d12 = calc.distance(point1, point2)
            d23 = calc.distance(point2, point3)
            if d12 == 0.0:
                continue
            else:
                a123 = calc.angle(point1, point2, point3)
                if (a123 > ang_min and a123 < ang_max):
                    continue
            points.append(point2)
        layers[name] = np.array(points)
        print name, len(points)
    return layers

def read_layers(filename):
    t0 = time.time()
    layers = read_file(filename)
    t1 = time.time()
    print 'Time to read layers in file: %.2f s' %(t1 - t0)

    t0 = time.time()
    layers = process_layers(layers)
    t1 = time.time()
    print 'Time to process layers: %.2f s' %(t1 - t0)
    return layers

def get_vectors(path_int, path_ext):
    ni, ne = len(path_int), len(path_ext)
    points, vectors = [], []
    for ke in range(ni):
        min_dist = 100
        pnt1, pnt2 = path_int[ke], path_int[ke-1]
        pt2, pt3 = np.array([0, 0, 0]), np.array([0, 0, 0])
        for ki in range(ne):
            point2, point3 = path_ext[ki-1], path_ext[ki]
            dist12 = calc.distance(pnt1, point2)
            dist13 = calc.distance(pnt1, point3)
            if (dist12 + dist13) < min_dist:
                min_dist = dist12 + dist13
                pt2, pt3 = point2, point3
        vector3 = calc.normal_vector(pnt1, pt2, pt3)
        vector2 = calc.vector(pnt1, pnt2)
        vector1 = calc.cross(vector2, vector3)
        vector2 = calc.normalize(vector2)
        vector1 = calc.normalize(vector1)
        points.append(pnt1)
        vectors.append(np.array([vector1, vector2, vector3]))
    points = np.array(points)
    vectors = np.array(vectors)
    return points, vectors

def frames2path(points, frames):
    path = []
    for k, frame in enumerate(frames):
        point = points[k]
        vx, vy, vz = frame[0,:], frame[1,:], frame[2,:]
        R = np.array([[vx[0], vy[0], vz[0], 0],
                      [vx[1], vy[1], vz[1], 0],
                      [vx[2], vy[2], vz[2], 0],
                      [0, 0, 0, 1]])
        position, quaternion = calc.matrix_to_quatpose(R)
        path.append([point, quaternion, True])
    return path




if __name__ == '__main__':
    layers = read_layers('models_dxf/curvas_v16.dxf')
    print layers

    from mlabplot import MPlot3D

    mplot3d = MPlot3D()
    mplot3d.draw_layers(layers)
    mplot3d.show()

    t1 = time.time()
    copa_int, copa_ext = layers['Copa1_I'], layers['Copa1_E']
    points1, vectors1 = get_vectors(copa_int, copa_ext)
    copa_int, copa_ext = layers['Copa2_I'], layers['Copa2_E']
    points2, vectors2 = get_vectors(copa_int, copa_ext)
    t2 = time.time()
    print 'Time to calculate vectors:', t2 - t1

    t1 = time.time()
    path1 = frames2path(points1, vectors1)
    path2 = frames2path(points2, vectors2)
    path = path1
    path[-1][2] = False
    path.extend(path2)
    t2 = time.time()
    print 'Time to path calculation:', t2 - t1
    print path

    frames = np.float32([calc.quatpose_to_matrix(pose[0], pose[1])[:3,:3] for pose in path])
    points = np.float32([pose[0] for pose in path])
    poses = []
    for k in range(len(points)):
        poses.append([points[k], frames[k], path[k][2]])

    mplot3d = MPlot3D()
    mplot3d.draw_path(poses)
    mplot3d.draw_path_tools(poses)
    mplot3d.show()
