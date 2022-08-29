# Start with pyramid app image
FROM continuumio/miniconda3

# Install conda stuff first
# install gdal library
RUN conda install nomkl pyproj
RUN conda install -c conda-forge gdal nomkl

WORKDIR /OpenTNSim
ADD . /OpenTNSim
RUN pip install --upgrade pip
# Install the application
RUN pip install -e .
# and the testing dependencies
RUN pip install -e .[testing]
