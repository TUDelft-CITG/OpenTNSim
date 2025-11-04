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

# Remove old Node.js and install Node.js 20.x from NodeSource repository
apt-get update
apt-get install -y curl
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g npm@latest

# Build the Jupyter Book
jupyter-book build book
