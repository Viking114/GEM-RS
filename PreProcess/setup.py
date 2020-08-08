import glob
import os
import platform
import subprocess
import sys

from setuptools import Command, Extension, setup, find_packages
from setuptools.command.test import test as TestCommand

def define_extensions():

    compile_args = ['-fopenmp',
                    '-ffast-math']

    # There are problems with illegal ASM instructions
    # when using the Anaconda distribution (at least on OSX).
    # This could be because Anaconda uses its own assembler?
    # To work around this we do not add -march=native if we
    # know we're dealing with Anaconda
    if 'anaconda' not in sys.version.lower():
        compile_args.append('-march=native')

    glove_corpus = "corpus_cython.pyx"

    return [Extension("glove.corpus_cython", [glove_corpus],
                      language='C++',
                      libraries=[],
                      extra_link_args=compile_args,
                      extra_compile_args=compile_args)]

class Cythonize(Command):
    """
    Compile the extension .pyx files.
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        import Cython
        from Cython.Build import cythonize

        cythonize(define_extensions(cythonize=True))


class Clean(Command):
    """
    Clean build files.
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        pth = os.path.dirname(os.path.abspath(__file__))

        subprocess.call(['rm', '-rf', os.path.join(pth, 'build')])
        subprocess.call(['rm', '-rf', os.path.join(pth, '*.egg-info')])
        subprocess.call(['find', pth, '-name', '*.pyc', '-type', 'f', '-delete'])
        subprocess.call(['rm', os.path.join(pth, 'glove', 'corpus_cython.so')])
        subprocess.call(['rm', os.path.join(pth, 'glove', 'glove_cython.so')])


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['tests/']

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

setup(
    long_description='',
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    tests_require=['pytest'],
    cmdclass={'test': PyTest, 'cythonize': Cythonize, 'clean': Clean},
    ext_modules=define_extensions()
)
