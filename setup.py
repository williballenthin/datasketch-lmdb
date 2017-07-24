#!/usr/bin/env python
from setuptools import setup, find_packages

# For Testing:
#
# python3.4 setup.py register -r https://testpypi.python.org/pypi
# python3.4 setup.py bdist_wheel upload -r https://testpypi.python.org/pypi
# python3.4 -m pip install -i https://testpypi.python.org/pypi
#
# For Realz:
#
# python3.4 setup.py register
# python3.4 setup.py bdist_wheel upload
# python3.4 -m pip install

setup(
    name='datasketch-lmdb',
    version='0.1',
    description='Extension to datasketch MinHash LSH that persists to lmdb database',
    author='Willi Ballenthin',
    author_email='willi.ballenthin@gmail.com',
    url='https://github.com/williballenthin/datasketch-lmdb',
    license='Apache License 2.0',
    install_requires=[
        'lmdb',
        'msgpack-python',
    ],
    packages=find_packages(exclude=['*.tests', '*.tests.*']),
    entry_points={
        "console_scripts": [
        ]
      },

    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
    ],
)
