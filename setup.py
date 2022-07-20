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
        "dadapy._cython.cython_clustering",
        sources=["dadapy/_cython/cython_clustering.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "dadapy._cython.cython_maximum_likelihood_opt",
        sources=["dadapy/_cython/cython_maximum_likelihood_opt.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "dadapy._cython.cython_maximum_likelihood_opt_full",
        sources=["dadapy/_cython/cython_maximum_likelihood_opt_full.c"],
        include_dirs=[get_numpy_include()],
    )
]


ext_modules += [
    Extension(
        "dadapy._cython.cython_density",
        sources=["dadapy/_cython/cython_density.c"],
        include_dirs=[get_numpy_include()],
    )
]


setup(
    name="dadapy",
    version="0.1.1",
    url="https://dadapy.readthedocs.io/",
    description="A Python package for Distance-based Analysis of DAta-manifolds.",
    long_description="A Python package for Distance-based Analysis of DAta-manifolds.",
    packages=["dadapy", "dadapy._utils"],
    install_requires=["numpy", "scipy", "scikit-learn", "matplotlib"],
    extras_require={"dev": ["tox", "black", "isort", "pytest"]},
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
