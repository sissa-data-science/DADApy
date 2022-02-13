from setuptools import setup, Extension

cmdclass = {}

class get_numpy_include(object):
    """Defer numpy.get_include() until after numpy is installed.
    From: https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    """

    def __str__(self):
        import numpy

        return numpy.get_include()

ext_modules = []

ext_modules += [
    Extension(
        "dadapy.cython_.cython_clustering",
        sources=["dadapy/cython_/cython_clustering.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "dadapy.cython_.cython_grads",
        sources=["dadapy/cython_/cython_grads.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "dadapy.cython_.cython_maximum_likelihood_opt",
        sources=["dadapy/cython_/cython_maximum_likelihood_opt.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "dadapy.cython_.cython_periodic_dist",
        sources=["dadapy/cython_/cython_periodic_dist.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "dadapy.cython_.cython_density",
        sources=["dadapy/cython_/cython_density.c"],
        include_dirs=[get_numpy_include()],
    )
]


setup(
    name="dadapy",
    url="https://dadapy.readthedocs.io/",
    description="A Python package for Distance-based Analysis of DAta-manifolds.",
    long_description="A Python package for Distance-based Analysis of DAta-manifolds.",
    packages=["dadapy", "dadapy.utils_"],
    install_requires=["numpy", "scipy", "scikit-learn", "matplotlib"],
    extras_require={"dev": ["tox", "black", "isort", "pytest"]},
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)

### COMPILE FROM CYTHON ### NOT WORKING
#
# from Cython.Build import cythonize
# ext_modules = cythonize("dadapy/cython_/*.pyx")
#
# setup(
#     name="dadapy",
#     url="https://dadapy.readthedocs.io/",
#     description="A Python package for Distance-based Analysis of DAta-manifolds.",
#     long_description="A Python package for Distance-based Analysis of DAta-manifolds.",
#     packages=["dadapy", "dadapy.utils_"],
#     install_requires=["numpy", "scipy", "scikit-learn", "matplotlib"],
#     extras_require={"dev": ["tox", "black", "isort", "pytest"]},
#     cmdclass=cmdclass,
#     ext_modules=ext_modules,
# )