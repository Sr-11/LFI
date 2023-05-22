# from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='lfi',
    version='0.0.1',
    description='Kernel-Based Tests for Likelihood-Free Hypothesis Testing',
    author='Removed for submission',
    url='Removed for submission',
    license='Removed for submission',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'torch',
        'importlib',
        'tqdm',
        'scipy',
        'pandas',
        'pyroc',
        'matplotlib',
        'IPython',
        'requests',
    ]
)