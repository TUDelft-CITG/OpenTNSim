**CircleCI:** [![CircleCI][circleci_image]][circleci_link]

[circleci_link]: https://circleci.com/gh/TUDelft-CITG/Transport-Network-Analysis
[circleci_image]: https://circleci.com/gh/TUDelft-CITG/Transport-Network-Analysis.svg?style=svg

==========================
transport-network-analysis
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

