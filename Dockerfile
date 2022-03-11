# Start with pyramid app image
FROM continuumio/miniconda3

# Install conda stuff first
RUN conda install nomkl pyproj

WORKDIR /OpenTNSim
ADD . /OpenTNSim

# Install the application
RUN pip install -e .