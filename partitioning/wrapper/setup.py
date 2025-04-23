from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

metis_include_dir = "/usr/local/include"
metis_lib_dir = "/usr/local/lib"

extensions = [
    Extension(
        "partitioner",
        ["partition.pyx"],
        include_dirs=[".", metis_include_dir, np.get_include()],
        library_dirs=[metis_lib_dir],
        libraries=["metis"],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    name="partitioner",
    ext_modules=cythonize(extensions),
)
