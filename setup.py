from setuptools import find_packages, setup

import numpy as np
from Cython.Build import cythonize


with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="BANG-Fabio-Rigamonti",
    version="0.0.7",
    packages=['src/BANG'],
    author="Fabio Rigamonti",
    description="BANG: BAyesian modelliNg of Galaxies",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/FabioRigamonti/BANG",
    ext_modules=cythonize(["src/BANG/utils_easy.pyx"]),
    include_dirs=np.get_include(),
    #install_requires=[
    #    'astropy>=5.0.4',
    #    'corner>=2.2.1',
    #    'cpnest>=0.11.3',
    #    'Cython==0.29.27',
    #    'matplotlib>=3.5.1',
    #    'numba==0.52',
    #    'numpy>=1.21.5',
    #    'PyYAML>=6.0',
    #    'scipy>=1.7.3',
    #    'setuptools>=42'
    #]
)