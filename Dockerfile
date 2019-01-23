# Start with pyramid app image
FROM continuumio/anaconda3

WORKDIR Transport-Network-Analysis
ADD . /Transport-Network-Analysis

# Install everything
RUN pip install -r ./requirements.txt --quiet
RUN conda install pyproj
RUN pip install -e .

EXPOSE 8888

# Build with: docker build -t transport_network_analysis .