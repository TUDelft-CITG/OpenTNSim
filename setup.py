#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for OpenTNSim.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import sys

from pkg_resources import require, VersionConflict
from setuptools import setup, find_packages

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)

requires = [
    "pandas>=0.24.0",
    "openclsim>=0.16",
    "numpy",
    "simpy",
    "networkx>=3",
    "geopandas",
    "shapely>=2",
    "scipy",
    "click",
    "matplotlib",
    # read shapefiles
    "pyproj",
    # unit conversions
    "pint",
    "plotly",
    "simplekml",
    "nose",
    # downloading
    "requests",
    "requests-cache",
    # web mode
    "Flask>=1.0",
    "Flask-cors",
    "sphinx_rtd_theme",
    "Dill",
    # deprecate old functions
    "Deprecated",
    "tqdm",
]

setup_requirements = [
    "pytest-runner",
]

tests_require = [
    "geopandas",
    "ipyleaflet",
    "pytest<7",
    "pytest-cov",
    "pytest-timeout",
    "pytest-datadir",
    "cython",
    "nbmake",
    # extra dpendencies used by nontebooks
    "pyyaml",
    "openpyxl",
    # geo info
    "fiona",
    "geopandas",
    "momepy",
    "folium",
    "colorcet",
    "notebook==6.4",
    "nbconvert==6.4",
    "jupyter",
    "jupyter-book==0.13.1",
    "ipywidgets==7.7",
    "jsonschema==3.0",
    "jupyterlab_widgets==3",
]

with open("README.md", "r") as des:
    long_description = des.read()

setup(
    author="Mark van Koningsveld",
    author_email="m.vankoningsveld@tudelft.nl",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    description="The OpenTNSim package aims to facilitate the analysis of network performance for different network configurations, fleet compositions and traffic rules.",
    entry_points={
        "console_scripts": [
            "opentnsim=opentnsim.cli:cli",
        ],
    },
    install_requires=requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="OpenTNSim",
    name="opentnsim",
    packages=find_packages(include=["opentnsim"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=tests_require,
    extras_require={"testing": tests_require},
    url="https://github.com/TUDelft-CITG/OpenTNSim",
    version="1.3.4",
    zip_safe=False,
)
