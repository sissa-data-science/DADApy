import os

from setuptools import Extension, setup


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
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

ext_modules += [
    Extension(
        "dadapy._cython.cython_clustering_v2",
        sources=["dadapy/_cython/cython_clustering_v2.c"],
        include_dirs=[get_numpy_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

ext_modules += [
    Extension(
        "dadapy._cython.cython_maximum_likelihood_opt",
        sources=["dadapy/_cython/cython_maximum_likelihood_opt.c"],
        include_dirs=[get_numpy_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

ext_modules += [
    Extension(
        "dadapy._cython.cython_maximum_likelihood_opt_full",
        sources=["dadapy/_cython/cython_maximum_likelihood_opt_full.c"],
        include_dirs=[get_numpy_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]


ext_modules += [
    Extension(
        "dadapy._cython.cython_density",
        sources=["dadapy/_cython/cython_density.c"],
        include_dirs=[get_numpy_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

ext_modules += [
    Extension(
        "dadapy._cython.cython_overlap",
        sources=["dadapy/_cython/cython_overlap.c"],
        include_dirs=[get_numpy_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

ext_modules += [
    Extension(
        "dadapy._cython.cython_grads",
        sources=["dadapy/_cython/cython_grads.c"],
        include_dirs=[get_numpy_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

exts_parallel = [
    Extension(
        "dadapy._cython.cython_distances",
        sources=["dadapy/_cython/cython_distances.c"],
        include_dirs=[get_numpy_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "dadapy._cython.cython_differentiable_imbalance",
        sources=["dadapy/_cython/cython_differentiable_imbalance.c"],
        include_dirs=[get_numpy_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

# Check if the '-fopenmp' flag is supported
command = "gcc -fopenmp -E - < /dev/null > /dev/null 2>&1"

parallel_ext_names = {
    "dadapy._cython.cython_density",
    "dadapy._cython.cython_grads",
    "dadapy._cython.cython_distances",
    "dadapy._cython.cython_differentiable_imbalance",
}

if os.system(command) == 0:
    # If '-fopenmp' is supported, add the extra compile and link arguments.
    for ext in ext_modules + exts_parallel:
        if ext.name in parallel_ext_names:
            ext.extra_compile_args = list(ext.extra_compile_args or [])
            ext.extra_link_args = list(ext.extra_link_args or [])
            ext.extra_compile_args.append("-fopenmp")
            ext.extra_link_args.append("-fopenmp")

# If OpenMP is not available, the C extension to compute distances in discrete spaces will not run in parallel.

ext_modules += exts_parallel

setup(
    packages=["dadapy", "dadapy._utils"],
    ext_modules=ext_modules,
    include_package_data=True,
    package_data={"dadapy": ["_utils/discrete_volumes/*.dat"]},
)
