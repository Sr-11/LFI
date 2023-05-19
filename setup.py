# from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='lfi',
    version='0.0.1',
    description='Kernel-Based Tests for Likelihood-Free Hypothesis Testing',
    author='Patrik Robert Gerber, Tianze Jiang, Yury Polyanskiy, Rui Sun',
    url='https://github.com/Sr-11/LFI',
    license='MIT',
    packages=find_packages()
    install_requires=[
        'python>=3.6.0',
        'numpy>=1.22.3',
        'torch>=1.13.1',
        'importlib>=1.0.4',
        'tqdm>=4.64.1',
        'scipy>=1.7.3',
        'pandas>=1.5.2',
        'pyroc>=0.1.1',
        'matplotlib>=3.6.2',
        'IPython>=8.8.0'
    ],
)