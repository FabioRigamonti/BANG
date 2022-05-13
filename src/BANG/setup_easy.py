import numpy
from setuptools import setup, find_packages
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import os

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())
        
ext_modules=[
             Extension("utils_easy",
                       sources=["utils_easy.pyx"],
                       libraries=["m"], # Unix-like specific
                       #extra_compile_args=["-O3","-ffast-math",'-fopenmp'],
                       extra_compile_args=["-O3",'-fopenmp'],
                       include_dirs=[numpy.get_include()],
                       extra_link_args=['-fopenmp']
                       )
             ]

setup(
    name="utils_easy",
    ext_modules=cythonize(ext_modules, language_level = "3"),
)
