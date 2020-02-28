from setuptools import setup, find_packages

setup(
    name='asymmetric_kde',
    version='1.2',
    description='Asymmetric kernel density estimation',
    author='Till Hoffmann',
    author_email='tah13@imperial.ac.uk',
    url='https://github.com/tillahoffmann/asymmetric_kde',
    requires=['numpy', 'scipy'],
    packages=find_packages(),
)
