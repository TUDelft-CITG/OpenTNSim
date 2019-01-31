[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CircleCI](https://circleci.com/gh/TUDelft-CITG/Transport-Network-Analysis.svg?style=svg&circle-token=59b1f167ed771129459d86e822fd2faaae8f4a34)](https://circleci.com/gh/TUDelft-CITG/Transport-Network-Analysis)
[ ![Coverage](https://oedm.vanoord.com/proxy/circleci_no_redirect/github/TUDelft-CITG/Transport-Network-Analysis/master/latest/727b95b70301407d3c0af44e1af2039fd9486f6f/tmp/artifacts/coverage.svg)](https://oedm.vanoord.com/proxy/circleci_no_redirect/github/TUDelft-CITG/Transport-Network-Analysis/master/latest/727b95b70301407d3c0af44e1af2039fd9486f6f/tmp/artifacts/index.html) 
[ ![Documentation](https://img.shields.io/badge/sphinx-documentation-brightgreen.svg)](https://oedm.vanoord.com/proxy/circleci_no_redirect/github/TUDelft-CITG/Transport-Network-Analysis/master/latest/727b95b70301407d3c0af44e1af2039fd9486f6f/tmp/artifacts/docs/index.html)


Transport Network Analysis
==========================

Transport network analysis package 

Description
===========

The Transport Network Analysis package aims to facilitate basic nautical traffic simulations. For routing issues this package primarily makes use of the python package networkx. It furthermore contains a number of generically formulated vessel classes. These classes can easily be used and expanded to enable investigation of: travel times, rough cost estimates, fuel use, emissions, congestions, etc. To simulate how vessels move over the network and sometimes interact with eachother and with the infrastructure the SimPy package is used. A number of basic visualisation routines is included to enable easy inspection of the simulation results.

Installations
=============

    git clone https://github.com/TUDelft-CITG/Transport-Network-Analysis

    cd Transport-Network-Analysis

    pip install -U pip

    pip install -U setuptools

    pip install sphinx

    pip install -e .


Note
====

This project has been set up using PyScaffold 3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

