from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

metis_include_dir = "/usr/local/include"
metis_lib_dir = "/usr/local/lib"

# Define the extension module
extensions = [
    Extension(
        "partitioner",  # Name of the extension
        ["partition.pyx"],  # Your Cython source file
        include_dirs=[np.get_include(), metis_include_dir],
        library_dirs=[metis_lib_dir],
        libraries=["metis"],  # Link with libmetis
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

# Setup configuration
setup(
    name="partitioner",
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
)