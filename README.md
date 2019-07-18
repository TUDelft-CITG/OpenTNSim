[![Documentation](https://img.shields.io/badge/sphinx-documentation-informational.svg)](https://opentnsim.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](https://github.com/TUDelft-CITG/Transport-Network-Analysis/blob/master/LICENSE.txt)

[![CircleCI](https://circleci.com/gh/TUDelft-CITG/OpenTNSim.svg?style=svg&circle-token=59b1f167ed771129459d86e822fd2faaae8f4a34)](https://circleci.com/gh/TUDelft-CITG/OpenTNSim)


## Transport Network Analysis

* Documentation can be found: [here](https://opentnsim.readthedocs.io/)

## Description

The Transport Network Analysis package aims to facilitate basic nautical traffic simulations. For routing issues this package primarily makes use of the python package networkx. It furthermore contains a number of generically formulated vessel classes. These classes can easily be used and expanded to enable investigation of: travel times, rough cost estimates, fuel use, emissions, congestions, etc. To simulate how vessels move over the network and sometimes interact with eachother and with the infrastructure the SimPy package is used. A number of basic visualisation routines is included to enable easy inspection of the simulation results.

## Installation

``` bash
# Download the package
pip install opentnsim
```
