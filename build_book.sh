#!/bin/sh
# Copy the examples from the notebooks
cp notebooks/*.ipynb book/examples/
# copy the data from the notebooks folder
mkdir -p book/examples/data/
cp notebooks/data/* book/examples/data/
# copy the figures from the notebooks folder
mkdir -p book/examples/figures/
cp notebooks/figures/* book/examples/figures/
# Copy the api docs from the docs folder
cp docs/opentnsim.rst book/docs
# Copy the Authors list from the root folder
cp AUTHORS.rst book/docs/
# Copy logos
cp -r docs/_static book/docs


# Build the Jupyter Book
pip install jupyter-book==1.0.4
jupyter-book build book
