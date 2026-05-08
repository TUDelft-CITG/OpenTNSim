# Start with pyramid app image
# FROM continuumio/miniconda3
# ENV DEBIAN_FRONTEND noninteractive
# RUN apt update
# RUN apt install -y build-essential python3-dev ffmpeg

# # Install conda stuff first
# # install gdal library
# RUN conda install gdal

# WORKDIR /OpenTNSim
# ENV PROJ_DATA=/opt/conda/share/proj
# ADD . /OpenTNSim
# RUN pip install --upgrade pip
# # Install the application
# RUN pip install -e .

# # and the testing dependencies
# RUN pip install -e .[testing,zsf]

FROM continuumio/miniconda3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# install conda dependencies
RUN conda install -y gdal setuptools

WORKDIR /OpenTNSim
ENV PROJ_DATA=/opt/conda/share/proj

COPY . /OpenTNSim

RUN pip install --upgrade pip setuptools wheel

# install package
RUN pip install -e .

# install extras
RUN pip install -e .[testing,zsf]