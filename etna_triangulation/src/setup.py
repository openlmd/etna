from distutils.core import setup, Extension
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("cmatching", ["cmatching.pyx"])]
)

# python setup.py build_ext --inplace