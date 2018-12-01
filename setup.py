# Author: Johannes Schneider, University of Liechtenstein
# Paper: Topic modeling based on Keywords and Context
# Please cite: https://arxiv.org/abs/1710.02650 (accepted at SDM 2018)

# Install compiler aka:
# .ms/vcpython #C:\Users\jschneid\AppData\Local\Programs\Common\Microsoft\Visual C++ for Python\9.0\ #VS90COMNTOOLS
# Go to folder with lda.pyx, etc.
# Compile: python setup.py build_ext --inplace
# (or use an IDE)

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

install_requires = [
    "Cython",
    "numpy",
    "scipy",
    "stemming"

]

extensions = [
    Extension("_topicAssign", ["_topicAssign.pyx"],
              include_dirs=[np.get_include()])
]

setup_args = {
    "name": "TopicKeywordModel",
    "ext_modules": cythonize(extensions),
    "python_requires": ">= 3",
    "author": "Johannes Schneider",
    "install_requires": install_requires
}


if __name__ == "__main__":
    setup(**setup_args)
