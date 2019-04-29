# Importing libraries
# import pytest library for testing
import pytest

# import pandas for data handling
import pandas as pd

# import numpy for math
import numpy as np

# import simpy for simulation environment
import simpy

# import networkx for graphs
import networkx as nx

# import spatial libraries
import pyproj
import shapely.geometry

# import date and time handling
import datetime

# import random for random selection
import random

# tranport network analysis package
import transport_network_analysis.core as core
import transport_network_analysis.model as model


"""
Testing the VesselGenerator class and Simulation class of model.py
"""

@pytest.fixture()
def vessel_database():
    return pd.read_csv("tests/vessels/vessels.csv")

@pytest.fixture()
def vessel_type():
    vessel = type('Vessel', 
              (core.Identifiable, core.Log, core.Movable, core.HasContainer,
               core.VesselProperties, core.HasResource, core.Routeable), 
              {})
    
    return vessel

@pytest.fixture()
def graph():
    FG = nx.Graph()

    node_1 = {"Name": "Node 1", "Geometry": shapely.geometry.Point(4.49540, 51.905505)}
    node_2 = {"Name": "Node 2", "Geometry": shapely.geometry.Point(4.48935, 51.907995)}
    node_3 = {"Name": "Node 3", "Geometry": shapely.geometry.Point(4.48330, 51.910485)}
    node_4 = {"Name": "Node 4", "Geometry": shapely.geometry.Point(4.48440, 51.904135)}
    nodes = [node_1, node_2, node_3, node_4]

    for node in nodes:
        FG.add_node(node["Name"], 
                    name = node["Name"],
                    geometry = node["Geometry"], 
                    position = (node["Geometry"].x, node["Geometry"].y))

    edges = [[node_1, node_2], [node_2, node_3], [node_2, node_4], [node_3, node_4]]
    for edge in edges:
        if edge != [node_2, node_4]:
            FG.add_edge(edge[0]["Name"], edge[1]["Name"], Width = 25, Height = 25, Depth = 25)
        else:
            FG.add_edge(edge[0]["Name"], edge[1]["Name"], Width = 6.5, Height = 25, Depth = 25)
    return FG


"""
Actual testing starts below
"""

