from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    name="debug_cython_grads_2",
    #name="debug_cython_grads",
    #ext_modules=cythonize("*.c", compiler_directives={'language_level' : "3"}),
    #ext_modules=cythonize("debug_cython_grads.c", compiler_directives={'language_level' : "3"}),
    ext_modules=cythonize("debug_cython_grads_2.pyx", compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()]
)
