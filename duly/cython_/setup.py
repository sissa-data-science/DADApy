from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("cython_functions.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
)

# setup(ext_modules=cythonize('gradient_computation.pyx', annotate = True), include_dirs=[numpy.get_include()])
# setup(ext_modules=cythonize('compute_clusters.pyx', annotate = True), include_dirs=[numpy.get_include()])
# setup(ext_modules=cythonize(['compute_clusters.pyx','gradient_computation.pyx'], annotate = True), include_dirs=[numpy.get_include()])
