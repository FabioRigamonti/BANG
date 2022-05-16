#from setuptools import find_packages, setup
#import setuptools   
#import numpy as np
#from Cython.Build import cythonize
#
#
#with open("README.md", 'r') as f:
#    long_description = f.read()
#
#setup(
#    name="BANG-Fabio-Rigamonti",
#    version="0.0.9",
#    #packages=['src/BANG'],
#    author="Fabio Rigamonti",
#    description="BANG: BAyesian modelliNg of Galaxies",
#    long_description=long_description,
#    long_description_content_type='text/markdown',
#    url="https://github.com/FabioRigamonti/BANG",
#    package_dir={"": "src"},
#    packages=setuptools.find_packages(where="src"),
#    ext_modules=cythonize(["src/BANG/utils_easy.pyx"]),
#    include_dirs=np.get_include(),
#)

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BANGal",
    version="0.0.0",
    author="Fabio Rigamonti",
    author_email="frigamonti@uninsubria.it",
    description="BANG: BAyesian modelliNg of Galaxies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FabioRigamonti/BANG",
    project_urls={
        "Bug Tracker": "https://github.com/FabioRigamonti/BANG/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    #ext_modules=cythonize(["src/BANG/utils_easy.pyx"]),
    #include_dirs=np.get_include(),
    python_requires=">=3.8",
    test_suite='tests',
)