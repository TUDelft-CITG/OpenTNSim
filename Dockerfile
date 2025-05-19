# Start with pyramid app image
FROM continuumio/miniconda3
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN apt install -y build-essential python3-dev ffmpeg

# Install conda stuff first
# install gdal library
# RUN conda install cctbx202211::libsqlite
RUN conda install gdal

WORKDIR /OpenTNSim
ENV PROJ_DATA=/opt/conda/share/proj
ADD . /OpenTNSim
RUN pip install --upgrade pip
# Install the application
RUN pip install -e .
# and the testing dependencies
RUN pip install -e .[testing]
