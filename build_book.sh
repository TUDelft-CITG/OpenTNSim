#!/bin/sh
# Copy the examples from the notebooks
cp notebooks/*.ipynb book/examples/
# copy the data from the notebooks folder
mkdir -p book/examples/data/
cp notebooks/data/* book/examples/data/
# copy the figures from the notebooks folder
mkdir -p book/examples/figures/
cp notebooks/figures/* book/examples/figures/
# Copy logos
mkdir -p book/docs
cp -r docs/_static book/docs


# Build the Jupyter Book
cd book && NODE_OPTIONS="--dns-result-order=ipv4first" HOST=127.0.0.1 jupyter-book build --html --keep-host --ci
