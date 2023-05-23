# Start with pyramid app image
FROM continuumio/miniconda3
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN apt install -y build-essential python3-dev

# Install conda stuff first
# install gdal library
RUN conda install -c conda-forge mamba nomkl
RUN mamba install -c conda-forge pyproj
RUN mamba install -c conda-forge gdal

WORKDIR /OpenTNSim
ENV PROJ_DATA=/opt/conda/share/proj
ADD . /OpenTNSim
RUN pip install --upgrade pip
# Install the application
RUN pip install -e .
# and the testing dependencies
RUN pip install -e .[testing]
