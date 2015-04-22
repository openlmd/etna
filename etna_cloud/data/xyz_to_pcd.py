import sys
import numpy as np


if len(sys.argv) < 2:
    print 'Usage: python xyz_to_pcd.py cloud.xyz'
    sys.exit()
    
filename = sys.argv[1]
points3d = np.loadtxt(filename)

nfilename = '%s.pcd' %filename[:-4]
with open(nfilename, 'w') as f:
    f.write('# .PCD v.7 - Point Cloud Data file format\n')
    f.write('VERSION .7\n')
    f.write('FIELDS x y z\n')
    f.write('SIZE 4 4 4\n')
    f.write('TYPE F F F\n')    
    f.write('COUNT 1 1 1\n')
    f.write('WIDTH %i\n' %len(points3d))
    f.write('HEIGHT 1\n')
    f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
    f.write('POINTS %i\n' %len(points3d))
    f.write('DATA ascii\n')
    np.savetxt(f, points3d, fmt='%.6f')
    
