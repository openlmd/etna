import pylab
import numpy as np


def best_fitLTSQ(data):
    """Returns the best linear model for the input data in terms of least-squares."""
    G = data.copy()
    G[:,-1] = 1
    Z = data[:,-1]
    model, resid, rank, s = np.linalg.lstsq(G, Z)
    r2 = 1 - resid / (Z.size * Z.var())
    #print 'r2', r2
    return model, resid

def best_fitSVD(data):
    """Returns the best linear model for the input data using the SVD method."""
    p = (np.ones((len(data), 1)))
    AB = np.hstack([data, p])
    [u, d, v] = np.linalg.svd(AB)
    return v[-1,:] # Solution is last column of v.

def fit_line2d(points2d):
    """Best 2D line fitting using the least-squares method."""
    model, resid = best_fitLTSQ(points2d)
    return model

def fit_planeLTSQ(points3d):
    """Fits a plane to a point cloud where Z = aX + bY + c."""
    # Rearanging the equation: aX + bY -Z + c = 0
    # Gives Normal = (a, b, -1)
    model, resid = best_fitLTSQ(points3d)
    (a, b, c) = model
    #normal = (a, b, -1)
    #nn = np.linalg.norm(normal)
    #normal = normal / nn
    model = (a, b, -1, c)
    nn = np.linalg.norm(model[:3])
    model = model / nn
    return model

def fit_line3d(points3d):
    """Fit a line to a point cloud where z = mx + ny + c."""
    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = points3d.mean(axis=0)
    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(points3d - datamean)
    print 'vv', vv
    print 'Best Fit', best_fitSVD(points3d)
    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.

    # It's a straight line, so we only need 2 points.
    # And shift by the mean to get the line in the right place.
    linepts = vv[0] * np.array([[-700], [700]])
    linepts += datamean
    return linepts

def fit_plane(points3d):
    """Plane fitting."""
    # Set up constraint equations of the form  AB = 0,
    # where B is a column vector of the plane coefficients in the form:
    # a*X + b*Y + c*Z + d = 0.
    B = best_fitSVD(points3d)
    #print 'Best fit', best_fitLTSQ(points3d)
    nn = np.linalg.norm(B[0:3])
    normal = B / nn
    return normal



def cross(vector1, vector2):
    """Calculates the cross product of both vectors."""
    (u1, v1, w1), (u2, v2, w2) = vector1, vector2
    return np.array([v1 * w2 - w1 * v2, w1 * u2 - u1 * w2, u1 * v2 - v1 * u2])

def normalize(vector):
    """Scales each component of the vector to have a magnitude value of 1."""
    m = np.sqrt(np.sum(vector * vector))
    if m:
        vector = vector / m
    return vector

def get_plane_pose(plane):
    """Gets the pose of the plane."""
    [a, b, c, d] = plane
#    pnt0 = np.float32([d / -a, 0, 0])
#    pnt1 = np.float32([(c * 10 + d) / -a, 0, 10])
#    pnt2 = np.float32([(b * 10 + d) / -a, 10, 0])
#    vecx = normalize(pnt1 - pnt0)
#    vecy = normalize(pnt2 - pnt0)
#    vecz = normalize(cross(vecx, vecy))
#    vecy = normalize(cross(vecz, vecx))
    # [(b * y + c * z + d) / -a, (a * x + c * z + d) / -b, ((a * x + b * y + d) / -c)]
    points2d = np.float32([[0, 0], [1, 0], [0, 1]])
    points3d = np.float32([[x, y, (a * x + b * y + d) / -c] for x, y in points2d])
    vecx = normalize(points3d[1] - points3d[0])
    vecy = normalize(points3d[2] - points3d[0])
    vecz = normalize(cross(vecx, vecy))
    vecy = normalize(cross(vecz, vecx))
    return np.float32([vecx, vecy, vecz]).T, points3d[0]
#    return np.float32([vecx, vecy, vecz]).T, pnt0


class Fit():
    def ransac(self, data, model_class, min_samples, threshold, max_trials=10000):
        '''Fits a model to data with the RANSAC algorithm.

        :param data: numpy.ndarray
        data set to which the model is fitted, must be of shape NxD where
        N is the number of data points and D the dimensionality of the data
        :param model_class: object
        object with the following methods implemented:
        * fit(data): return the computed model
        * residuals(model, data): return residuals for each data point
        see LinearLeastSquares2D class for a sample implementation

        :param min_samples: int
        the minimum number of data points to fit a model

        :param threshold: int or float
        maximum distance for a data point to count as an inlier

        :param max_trials: int, optional
        maximum number of iterations for random sample selection, default 1000

        :returns: tuple
        best model returned by model_class.fit, best inlier indices
        '''
        best_model = None
        best_inlier_num = 0
        best_inliers = None
        data_idx = np.arange(data.shape[0])
        for _ in xrange(max_trials):
            sample = data[np.random.randint(0, data.shape[0], 2)]
            sample_model = model_class.fit(sample)
            sample_model_residua = model_class.residuals(sample_model, data)
            sample_model_inliers = data_idx[sample_model_residua<threshold]
            inlier_num = sample_model_inliers.shape[0]
            if inlier_num > best_inlier_num:
                best_inlier_num = inlier_num
                best_inliers = sample_model_inliers
        if best_inliers is not None:
            best_model = model_class.fit(data[best_inliers])
        return best_model, best_inliers

class LineFit(Fit):
    def function(self, x, model):
        m, b = model
        y = m * x + b
        return y

    def fit(self, data):
        model = fit_line2d(data)
        return model

    def residuals(self, model, data):
        m, b = model
        d = (m * data[:,0] + b - data[:,1]) / (m + 0.00001)
        return np.abs(d)

class PlaneFit(Fit):
    def function(self, x, y, model):
        a, b, c, d = model
        z = a * x + b * y + d / -c
        return z

    def fit(self, data):
        model = fit_planeLTSQ(data)
        return model

    def residuals(self, model, data):
        a, b, c, d = model
        D = (a * data[:,0] + b * data[:,1] + c * data[:,2] + d) / np.sqrt(a**2 + b**2 + c**2)
        return np.abs(D)




def generate_test_line_data(point0, point1, samples=200):
    # Generate some data that lies along a line
    data = []
    for k in range(len(point0)):
        data.append(np.linspace(point0[k], point1[k], samples))
    data = np.vstack(data).T
    data += np.random.normal(size=data.shape)
    # Generate some faulty data
    if len(point0) == 2:
        data[:30] += np.array([0, 50]) * np.random.random(size=(30,2))
    return data

def generate_test_plane_data(point0, point1, samples=20):
    x = np.linspace(point0[0], point1[0], samples)
    y = np.linspace(point0[1], point1[1], samples)
    z = np.linspace(point0[2], point1[2], samples)
    xx, yy = np.meshgrid(x, y)
    xx, zz = np.meshgrid(x, z)
    points3d = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    points3d += np.random.normal(size=points3d.shape) * 0.4
    points3d[:2] += np.array([0, 0, 10]) * np.random.random(size=(2,3))
    return points3d


def test_fit_line():
    points3d = generate_test_line_data((3200, 15, -5), (2300, 1.2, 3))
    model = fit_line3d(points3d)
    print 'Model line 3D:', model
    mplot3d = MPlot3D()
    mplot3d.draw_line(model, points3d)
    mplot3d.show()

def test_fit_plane(filename='../data/downsampled.xyz'):
    points3d = np.loadtxt(filename)

    normal = fit_plane(points3d)
    plane_pose = get_plane_pose(normal)
    print 'Normal:', normal
    print 'Plane pose:', plane_pose

    plane = PlaneFit()
    models = plane.fit(points3d)
    modelr, inliers = plane.ransac(points3d, plane, 3, 5)
    print 'Model plane:', models, modelr, inliers

    mplot3d = MPlot3D(scale=0.001)
    #mplot3d.draw_plane(normal, points3d)
    #mplot3d.draw_plane(modelr, points3d[inliers], color=(0,1,0))
    mplot3d.draw_points(points3d)
    mplot3d.draw_points(points3d[inliers], color=(0, 1, 0))
    mplot3d.draw_frame(plane_pose)
    mplot3d.show()


if __name__ == "__main__":
    #test_fit_line()
    #test_fit_plane()

    from mlabplot import MPlot3D

    WHITE = (1, 1, 1)
    RED = (1, 0, 0)
    BLUE = (0, 0, 1)

    import calculate as calc

    filename = '../../../etna_cloud/data/downsampled.xyz'
    filename = '../../../etna_cloud/data/test4_downsampled.xyz'
    #filename = '../data/test.xyz'
    cloud = np.loadtxt(filename)

    mplot3d = MPlot3D()
    mplot3d.draw_cloud(cloud)
    mplot3d.show()

    plane = PlaneFit()
    models = plane.fit(cloud)
    #modelr, inliers = plane.ransac(cloud, plane, 50, 0.0025)
    plane_model, inliers = plane.ransac(cloud, plane, int(0.5*len(cloud)), 0.0025)
    outliers = [k for k in range(len(cloud)) if k not in inliers]
    plane_pose = get_plane_pose(plane_model)
    print models, plane_model, plane_pose

    #pose = (pose[0], np.zeros(3))
    mplane = calc.pose_to_matrix(plane_pose)
    implane = calc.matrix_invert(mplane)
    tcloud = calc.points_transformation(implane, cloud)

    mplot3d = MPlot3D(scale=0.0025)
    #mplot3d.draw_frame((np.eye(3), np.zeros(3)))
    mplot3d.draw_cloud(cloud)
    #mplot3d.draw_points(cloud, color=WHITE)
    mplot3d.draw_points(tcloud[inliers], color=RED)
    mplot3d.draw_points(tcloud[outliers], color=BLUE)
    mplot3d.show()

    print np.std(tcloud[inliers][:, 2])
    #test()

    points3d = tcloud[outliers]
    point_min = np.min(points3d, axis=0)
    points_0 = (np.round(points3d - point_min, 3) / 0.0001).astype(np.int32)
    points_0[:, 0] = points_0[:, 0] / 10
    points_0[:, 1] = points_0[:, 1] / 10
    #points_1 = (np.round(points3d_1 - point_min, 3) * 1000).astype(np.int32)
    x_max = np.max(points_0[:, 0])
    y_max = np.max(points_0[:, 1])
    print 'Points', x_max, y_max

    zmap = np.zeros((int(x_max+1), int(y_max+1)))
    zmap[points_0[:, 0], points_0[:, 1]] = points_0[:, 2]


    from pylab import *

    figure()
    imshow(zmap, cmap='gray')
    colorbar()
    show()
