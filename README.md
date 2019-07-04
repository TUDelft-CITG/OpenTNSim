[ ![Documentation](https://img.shields.io/badge/sphinx-documentation-informational.svg)](https://oedm.vanoord.com/proxy/circleci_no_redirect/github/TUDelft-CITG/Transport-Network-Analysis/master/latest/727b95b70301407d3c0af44e1af2039fd9486f6f/tmp/artifacts/docs/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](https://github.com/TUDelft-CITG/Transport-Network-Analysis/blob/master/LICENSE.txt)

[![CircleCI](https://circleci.com/gh/TUDelft-CITG/Transport-Network-Analysis.svg?style=svg&circle-token=59b1f167ed771129459d86e822fd2faaae8f4a34)](https://circleci.com/gh/TUDelft-CITG/Transport-Network-Analysis)


## Transport Network Analysis

* Documentation can be found: [here](https://oedm.vanoord.com/proxy/circleci_no_redirect/github/TUDelft-CITG/Transport-Network-Analysis/master/latest/727b95b70301407d3c0af44e1af2039fd9486f6f/tmp/artifacts/docs/index.html)

## Description

The Transport Network Analysis package aims to facilitate basic nautical traffic simulations. For routing issues this package primarily makes use of the python package networkx. It furthermore contains a number of generically formulated vessel classes. These classes can easily be used and expanded to enable investigation of: travel times, rough cost estimates, fuel use, emissions, congestions, etc. To simulate how vessels move over the network and sometimes interact with eachother and with the infrastructure the SimPy package is used. A number of basic visualisation routines is included to enable easy inspection of the simulation results.

## Installation

Installation using *pip install transport_network_analysis* is not yet available. Running following three lines in your command prompt will allow you installing the package as well:

``` bash
# Download the package
git clone https://github.com/TUDelft-CITG/Transport-Network-Analysis

# Go to the correct folder
cd Transport-Network-Analysis

# Install package
pip install -U pip
pip install -U setuptools
pip install sphinx
pip install -e .
```
