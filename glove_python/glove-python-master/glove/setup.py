from Cython.Build import cythonize
from distutils.core import setup
import numpy as np

setup(
    ext_modules=cythonize("corpus_cython.pyx"),
)