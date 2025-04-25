from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

metis_include_dir = "/usr/local/include"
metis_lib_dir = "/usr/local/lib"

extensions = [
    Extension(
        "metis_wrapper.partition",
        ["metis_wrapper/partition.pyx"],
        include_dirs=[np.get_include(), "/usr/local/include"],
        library_dirs=["/usr/local/lib"],
        libraries=["metis"],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
]

setup(
    name="metis_wrapper",
    packages=["metis_wrapper"],
    ext_modules=cythonize(
        extensions,
        language_level='3',
        include_path=["metis_wrapper"]
    ),
)
