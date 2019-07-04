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
import opentnsim.core as core
import opentnsim.model as model


"""
Testing the VesselGenerator class and Simulation class of model.py
"""

@pytest.fixture()
def vessel_database():
    return pd.read_csv("tests/vessels/vessels.csv")

@pytest.fixture()
def vessel_type():
    vessel = type('Vessel', 
              (core.Identifiable, core.Movable, core.HasContainer,
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

def test_route_selection_small(vessel_database, vessel_type, graph):

    # Generate a vessel with a width property
    env = simpy.Environment()
    env.FG = graph

    # Make a test vessel with a width < 6.5
    random.seed(4)
    vessel_info = vessel_database.sample(n = 1, random_state = int(1000 * random.random()))

    # Create a path based on the vessel width
    vessel_width = vessel_info["width"].values[0]
    assert vessel_width < 6.5

    edges = []
    nodes = []
    for edge in graph.edges(data = True):
        if edge[2]["Width"] > vessel_width:
            edges.append(edge)
            
            nodes.append(graph.nodes[edge[0]])
            nodes.append(graph.nodes[edge[1]])

    subGraph = graph.__class__()

    for node in nodes:
        subGraph.add_node(node["name"],
                name = node["name"],
                geometry = node["geometry"], 
                position = (node["geometry"].x, node["geometry"].y))

    for edge in edges:
        subGraph.add_edge(edge[0], edge[1], attr_dict = edge[2])

    path = nx.dijkstra_path(subGraph, subGraph.nodes["Node 1"]["name"], subGraph.nodes["Node 4"]["name"])

    assert path == ["Node 1", "Node 2", "Node 4"]


def test_route_selection_large(vessel_database, vessel_type, graph):

    # Generate a vessel with a width property
    env = simpy.Environment()
    env.FG = graph

    # Make a test vessel with a width < 6.5
    random.seed(3)
    vessel_info = vessel_database.sample(n = 1, random_state = int(1000 * random.random()))

    # Create a path based on the vessel width
    vessel_width = vessel_info["width"].values[0]
    assert vessel_width > 6.5

    edges = []
    nodes = []
    for edge in graph.edges(data = True):
        if edge[2]["Width"] > vessel_width:
            edges.append(edge)
            
            nodes.append(graph.nodes[edge[0]])
            nodes.append(graph.nodes[edge[1]])

    subGraph = graph.__class__()

    for node in nodes:
        subGraph.add_node(node["name"],
                name = node["name"],
                geometry = node["geometry"], 
                position = (node["geometry"].x, node["geometry"].y))

    for edge in edges:
        subGraph.add_edge(edge[0], edge[1], attr_dict = edge[2])

    path = nx.dijkstra_path(subGraph, subGraph.nodes["Node 1"]["name"], subGraph.nodes["Node 4"]["name"])

    assert path == ["Node 1", "Node 2", "Node 3", "Node 4"]

def test_route_selection_self(vessel_database, vessel_type, graph):

    # Create a vessel generator
    generator = model.VesselGenerator(vessel_type, vessel_database)

    # Create a simulation object
    simulation_start = datetime.datetime(2019, 1, 1)
    sim = model.Simulation(simulation_start, graph)
    sim.add_vessels(origin = list(graph)[0], destination = list(graph)[-1], vessel_generator = generator)

    # Run the simulation
    sim.run(duration = 24 * 60 * 60)

    for vessel in sim.environment.vessels:
        if vessel.width < 6.5:
            assert vessel.route == ["Node 1", "Node 2", "Node 4"]
        else:
            assert vessel.route == ["Node 1", "Node 2", "Node 3", "Node 4"]