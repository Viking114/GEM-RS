from Cython.Build import cythonize
from distutils.core import setup
import numpy as np

setup(
    ext_modules=cythonize("glove_core.pyx"),
    include_dirs=[np.get_include()],
    include_package_data=True,
)