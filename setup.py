#import setuptools

# import numpy
from numpy.distutils.core import setup
from numpy.distutils.core import Extension

cmdclass = {}
ext_modules = []

### COMPILE FROM C ###

# ext_modules += [Extension("cython_/cython_functions", sources=["cython_/cython_functions.c"],
#                              include_dirs=[numpy.get_include()])]
#

### COMPILE FROM CYTHON ###

from Cython.Distutils import build_ext

# ext_modules += [Extension("cython_.cython_functions", ["cython_/cython_functions.c"],
#                             include_dirs=[numpy.get_include()])]

cmdclass.update({'build_ext': build_ext})

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

print(use_cython)

if use_cython:
    ext_modules += [
        Extension("duly.cython_functions", ["cython_/cython_functions.pyx"]),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("duly.cython_functions", ["cython_/cython_functions.c"]),
    ]


setup(name='duly', packages=['duly'],
      install_requires=['numpy', 'scipy', 'scikit-learn'],
      cmdclass=cmdclass,
      ext_modules=ext_modules)