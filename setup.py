from setuptools import setup, find_namespace_packages
### Haplotype-Analysis 
### install locally with 'pip install .'
setup(name="hapy",
        version="0.1dev1",
        description='Tools to haplotype analysis for genomics studies.',
        author='Chuan Fu Yap',
        author_email='yapchuanfu@gmail.com',
        license='GNU GPLv3',
        packages=find_namespace_packages(include=['hapy.data', "hapy.stats"]),
        install_requires=['pandas', 'numpy', 'scipy', 'statsmodels'])