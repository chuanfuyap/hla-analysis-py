"""
Install instructions for HaPy
"""
import os
from setuptools import find_packages, setup

# Haplotype-Analysis
# install locally with 'pip install .'


def read(rel_path: str) -> str:
    "reads a txt file"
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    "read a text file and extract version"
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(name="hapy",
      version=get_version("hapy/__init__.py"),
      description='Tools to haplotype analysis for genomics studies.',
      author='Chuan Fu Yap',
      author_email='yapchuanfu@gmail.com',
      license='GNU GPLv3',
      #packages=find_namespace_packages(include=['hapy.data', "hapy.stats","hapy"]),
      packages=find_packages(),
      python_requires='>3.3',
      install_requires=['pandas', 'numpy', 'scipy', 'statsmodels'])
