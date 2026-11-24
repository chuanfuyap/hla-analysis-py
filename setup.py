"""
Install instructions for hapy
"""
import os
from setuptools import find_packages, setup

# Haplotype-Analysis
# install locally with 'pip install .'


def read(rel_path: str) -> str:
    "reads a txt file"
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp: #pylint: disable=W1514
        return fp.read()

def get_version(rel_path: str) -> str:
    "read a text file and extract version"
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(name="hla-hapy",
      version=get_version("hapy/__init__.py"),
      description='Tools to HLA analysis in genomics studies.',
      author='Chuan Fu Yap',
      author_email='yapchuanfu@gmail.com',
      license='GNU GPLv3',
      packages=find_packages(),
      python_requires='==3.11.*',
      entry_points={"console_scripts": [
          "hapy-download-reference=hapy.data.hla_download:main",
      ]},
      install_requires=[
        'pandas>=1.0.0,<2.0.0',
        'numpy>=1.20.0,<2.0.0',
        'scipy>=1.7.0,<1.10.0',
        'statsmodels>=0.13.0,<0.14.0',
        'lifelines>=0.27.0,<0.30.0',
        'scikit-learn>=1.0.0,<1.10.0',
        'pgenlib>=0.90,<1',
        'matplotlib>=3.6,<4'
    ],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/chuanfuyap/haplo-analysis-py",
        project_urls={
            "Issues": "https://github.com/chuanfuyap/haplo-analysis-py/issues",
        },
        license_files=["LICENSE"],
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.11",
        ],
)