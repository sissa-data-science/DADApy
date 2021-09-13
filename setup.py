from setuptools import setup, Extension

cmdclass = {}
ext_modules = []


### COMPILE FROM C ###


class get_numpy_include(object):
    """Defer numpy.get_include() until after numpy is installed.
    From: https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    """

    def __str__(self):
        import numpy

        return numpy.get_include()


# ext_modules += [Extension("duly.cython_.cython_functions", sources=["duly/cython_/cython_functions.c"],
#                           include_dirs=[get_numpy_include()])]

ext_modules += [
    Extension(
        "duly.cython_.cython_clustering",
        sources=["duly/cython_/cython_clustering.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "duly.cython_.cython_grads",
        sources=["duly/cython_/cython_grads.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "duly.cython_.cython_maximum_likelihood_opt",
        sources=["duly/cython_/cython_maximum_likelihood_opt.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "duly.cython_.cython_periodic_dist",
        sources=["duly/cython_/cython_periodic_dist.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "duly.cython_.cython_density",
        sources=["duly/cython_/cython_density.c"],
        include_dirs=[get_numpy_include()],
    )
]

setup(
    name="duly",
    packages=["duly", "duly.utils_"],
    install_requires=["numpy", "scipy", "scikit-learn", "Cython", "pytest"],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)

### COMPILE FROM CYTHON ### NOT WORKING


# exts = [Extension(name='duly.cython_functions',
#                   sources=["cython_/cython_functions.pyx", "cython_/cython_functions.c"],
#                   include_dirs=[numpy.get_include()])]

# setup(name='duly', packages=['duly'],
#       install_requires=['numpy', 'scipy', 'scikit-learn'],
#       cmdclass=cmdclass,
#       ext_modules=cythonize(exts))
