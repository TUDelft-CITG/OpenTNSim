# Start with pyramid app image
FROM continuumio/miniconda3

# Install conda stuff first
# install gdal library
RUN conda install nomkl pyproj gdal

WORKDIR /OpenTNSim
ADD . /OpenTNSim

# Install the application
RUN pip install -e .
# and the testing dependencies
RUN pip install -e .[testing]
