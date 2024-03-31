"""Setup."""
from pathlib import Path

from setuptools import find_packages, setup

long_description = """
DAttri
"""

exclude_patterns = ["*__pycache__*", "*ipynb_checkpoints*"]
with Path("version.txt").open() as version_file:
    VERSION = version_file.read().strip()


def setup_package():
    """Set up the package."""
    metadata = {
        "name": "dattri",
        "version": VERSION,
        "description": "dattri",
        "long_description": long_description,
        "long_description_content_type": "text/markdown",
        "author": "TARIS-Lab",
        "url": "https://github.com/TRAIS-Lab/dattri",
        "packages": find_packages(),
        "install_requires": [],
        "include_package_data": True,
        "classifiers": [
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: Implementation :: CPython"],
        "platforms": ["mac", "linux", "windows"],
    }

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
