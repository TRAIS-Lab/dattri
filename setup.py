#!/usr/bin/env python

from setuptools import setup

long_description = '''
DAttri
'''

exclude_patterns = ["*__pycache__*", "*ipynb_checkpoints*"]
VERSION = open("version.txt", 'r').read().strip()

def setup_package():
    metadata = dict(
        name='dattri',
        version=VERSION,
        description='dattri',
        long_description=long_description,
        long_description_content_type="text/markdown",
        author='TARIS-lab',
        url='https://github.com/TRAIS-Lab/dattri',
        packages=['dattri'],
        install_requires=[],
        include_package_data=True,
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: Implementation :: CPython'],
        platforms=['mac', 'linux', 'windows']
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
