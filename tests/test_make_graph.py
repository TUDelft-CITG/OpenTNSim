# Importing libraries
# import pytest library for testing
import pytest

# tranport network analysis package
import opentnsim.core as core
import opentnsim.graph_module as graph_module

# import spatial libraries
import pyproj
import shapely.geometry

# import graph package and simulation package
import networkx
import simpy


# Creating the test objects
# Make the graph
@pytest.fixture()
def graph():
    graph = graph_module.Graph()
    graph.graph = graph.graph.to_directed()

    node_1 = {"Name": "Node 1", "Geometry": shapely.geometry.Point(4.4954, 51.905505)}
    node_2 = {"Name": "Node 2", "Geometry": shapely.geometry.Point(4.4833, 51.910485)}

    nodes = [node_1, node_2]

    for node in nodes:
        graph.graph.add_node(
            node["Name"],
            geometry=node["Geometry"],
            Position=(node["Geometry"].x, node["Geometry"].y),
        )

    edge = [node_1, node_2]
    graph.graph.add_edge(edge[0]["Name"], edge[1]["Name"])

    return graph.graph


# Actual testing starts here
def test_type(graph):
    # Test if graph is initialized correctly as a directed graph
    assert type(graph) == networkx.classes.digraph.DiGraph

    # Test if location of the nodes is correct
    node_1 = list(graph.nodes(data=True))[0]
    node_2 = list(graph.nodes(data=True))[1]

    assert shapely.geometry.Point(4.4954, 51.905505) == node_1[1]["geometry"]
    assert shapely.geometry.Point(4.4833, 51.910485) == node_2[1]["geometry"]

    # Test is there is indeed only one edge
    edge = list(graph.edges(data=False))

    assert 1 == len(edge)

    # Test if the edge is indeed from node_1 to node_2
    node_1 = "Node 1"
    node_2 = "Node 2"

    assert node_1 == edge[0][0]
    assert node_2 == edge[0][1]
