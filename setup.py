import os
import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize


import setuptools

import numpy
from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


cmdclass = {}
ext_modules = []

### COMPILE FROM C ###

ext_modules += [Extension("cython_/cython_functions", sources=["cython_/cython_functions.c"],
                             include_dirs=[numpy.get_include()])]


setup(name='duly', packages=['duly'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'Cython', 'pytest'],
      cmdclass=cmdclass,
      ext_modules=ext_modules)

### COMPILE FROM CYTHON ### NOT WORKING


# exts = [Extension(name='duly.cython_functions',
#                   sources=["cython_/cython_functions.pyx", "cython_/cython_functions.c"],
#                   include_dirs=[numpy.get_include()])]

# setup(name='duly', packages=['duly'],
#       install_requires=['numpy', 'scipy', 'scikit-learn'],
#       cmdclass=cmdclass,
#       ext_modules=cythonize(exts))
