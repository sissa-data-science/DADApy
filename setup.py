import os

from setuptools import Extension, setup


class get_numpy_include(object):
    """Defer numpy.get_include() until after numpy is installed.
    From: https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    """

    def __str__(self):
        import numpy

        return numpy.get_include()


NUMPY_MACROS = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    ("NPY_TARGET_VERSION", "NPY_1_22_API_VERSION"),
]


def cython_extension(name):
    return Extension(
        f"dadapy._cython.{name}",
        sources=[f"dadapy/_cython/{name}.c"],
        include_dirs=[get_numpy_include()],
        define_macros=NUMPY_MACROS,
    )


serial_modules = [
    "cython_clustering",
    "cython_clustering_v2",
    "cython_maximum_likelihood_opt",
    "cython_maximum_likelihood_opt_full",
    "cython_density",
    "cython_overlap",
    "cython_grads",
]

parallel_modules = [
    "cython_distances",
    "cython_differentiable_imbalance",
]

ext_modules = [cython_extension(name) for name in serial_modules]
exts_parallel = [cython_extension(name) for name in parallel_modules]

# Check if the '-fopenmp' flag is supported
openmp_supported = os.system("gcc -fopenmp -E - < /dev/null > /dev/null 2>&1") == 0

if openmp_supported:
    # If '-fopenmp' is supported, add the extra compile and link arguments
    # Installing cython_distances using OpenMP
    for ext_parallel in exts_parallel:
        ext_parallel.extra_compile_args.append("-fopenmp")
        ext_parallel.extra_link_args.append("-fopenmp")

# If OpenMP is not available, the C extension to compute distances in discrete spaces will not run in parallel.

ext_modules += exts_parallel

setup(
    packages=["dadapy", "dadapy._utils"],
    ext_modules=ext_modules,
    include_package_data=True,
    package_data={"dadapy": ["_utils/discrete_volumes/*.dat"]},
)
