"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the version string from the VERSION file
with open(path.join(here, 'VERSION'), 'r') as f:
    version = f.readline().strip()

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='redec_keras',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,

    description='DEC Reclustering for keras',
    long_description=long_description,

    # The project's main homepage.
    # url='https://github.com/zooniverse/hco-experiments/tree/master/swap',

    # Author details
    author='Michael Laraia',
    author_email='larai002@umn.edu',

    # Choose your license
    license='MIT',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'matplotlib',
        'numpy',
        'sklearn',
        'scikit-image',
        'keras',
        'pandas',
        'progressbar2',
        'dec-keras',
        'tqdm',
    ],
)
