# Importing libraries
# import pytest library for testing
import pytest

# import pandas for data handling
import pandas as pd

# import simpy for simulation environment
import simpy

# import networkx for graphs
import networkx as nx

# import spatial libraries
import pyproj
import shapely.geometry

# import random for random selection
import random

# tranport network analysis package
import transport_network_analysis.core as core
import transport_network_analysis.model as model


"""
Testing the VesselGenerator class of model.py
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
    nodes = [node_1, node_2, node_3]

    for node in nodes:
        FG.add_node(node["Name"], 
                    geometry = node["Geometry"], 
                    Position = (node["Geometry"].x, node["Geometry"].y))

    edges = [[node_1, node_2], [node_2, node_3]]
    for edge in edges:
        FG.add_edge(edge[0]["Name"], edge[1]["Name"], weight = 1)

    return FG


"""
Actual testing starts below
"""

def test_make_vessel(vessel_database, vessel_type, graph):

    # Generate a vessel
    path = nx.dijkstra_path(graph, list(graph)[0], list(graph)[-1])
    env = simpy.Environment()
    env.FG = graph

    generator = model.VesselGenerator(vessel_type, vessel_database)
    generated_vessel = generator.generate(env, "Test vessel", path)

    # Make the test vessel
    random.seed(4)
    vessel_info = vessel_database.sample(n = 1, random_state = int(1000 * random.random()))
    vessel_data = {}

    vessel_data["env"] = env
    vessel_data["name"] = "Test vessel"
    vessel_data["route"] = path
    vessel_data["geometry"] = nx.get_node_attributes(graph, "geometry")[path[0]]

    for key in vessel_info:
        if key == "vessel_id":
            vessel_data["id"] = vessel_info[key].values[0]
        else:
            vessel_data[key] = vessel_info[key].values[0]

    test_vessel = vessel_type(**vessel_data)

    for key in generated_vessel.__dict__:
        if key != "container" or key != "resource":
            generated_vessel.__dict__[key] == test_vessel.__dict__[key]