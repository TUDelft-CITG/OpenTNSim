# Start with pyramid app image
FROM continuumio/miniconda3

# Install conda stuff first
RUN conda install pyproj

WORKDIR /Transport-Network-Analysis
ADD . /Transport-Network-Analysis

# Then install rest via pip
RUN python setup.py develop
RUN pip install coverage coverage-badge