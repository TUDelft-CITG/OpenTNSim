# Start with pyramid app image
FROM continuumio/miniconda3
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN apt install -y build-essential python3-dev ffmpeg

# Install conda stuff first
# install gdal library and nodejs 22 LTS (required for Jupyter Book 2)
RUN conda install -y gdal "nodejs>=22,<23"

WORKDIR /OpenTNSim
ENV PROJ_DATA=/opt/conda/share/proj
ADD . /OpenTNSim
RUN pip install --upgrade pip
# Install the application
RUN pip install -e .

# and the testing dependencies
RUN pip install -e .[testing]
