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

# ALWAYS bind pip to the active python
RUN python -m pip install --upgrade pip setuptools wheel

# install coverage tooling
RUN python -m pip install coverage coverage-badge

# install package (IMPORTANT FIX)
RUN python -m pip install -e .
RUN python -m pip install -e ".[testing,zsf]"