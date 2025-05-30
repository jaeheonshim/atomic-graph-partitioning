from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import os

conda_prefix = os.environ.get("CONDA_PREFIX", "")

extensions = [
    Extension(
        "metis_cython.partition",
        ["src/metis_cython/partition.pyx"],
        include_dirs=[
            np.get_include(),
            os.path.join(conda_prefix, "include"),
        ],
        library_dirs=[
            os.path.join(conda_prefix, "lib"),
        ],
        libraries=["metis"],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
]

setup(
    name="metis_wrapper",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(
        extensions,
        language_level='3',
        include_path=["src/metis_cython"]
    ),
)
