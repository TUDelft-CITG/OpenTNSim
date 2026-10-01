FROM continuumio/miniconda3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# conda deps
RUN conda install -y gdal setuptools

WORKDIR /OpenTNSim
ENV PROJ_DATA=/opt/conda/share/proj

COPY . /OpenTNSim

RUN python -m pip install --upgrade pip "setuptools<81" wheel

RUN python -m pip install coverage coverage-badge

RUN python -m pip install -e .
RUN python -m pip install -e ".[testing,zsf]"