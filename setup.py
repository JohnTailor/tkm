#Author: Johannes Schneider, University of Liechtenstein
#Paper: Topic modeling based on Keywords and Context, Please cite: https://arxiv.org/abs/1710.02650 (under submission as of 10/2017)

#Install compiler aka.ms/vcpython #C:\Users\jschneid\AppData\Local\Programs\Common\Microsoft\Visual C++ for Python\9.0\ #VS90COMNTOOLS
#Go to folder with lda.pyx ,eg.
#Compile: python setup.py build_ext --inplace
#or use IDE

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension("_topicAssign",["_topicAssign.pyx"],include_dirs = [np.get_include()])
]


setup(
    name = "MainStuff",
    ext_modules = cythonize(extensions),
)
