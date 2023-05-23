import networkx as nx
import shapely

import pytest

import opentnsim.utils

@pytest.fixture
def valid_graph():
    graph = nx.Graph()
    graph.add_edge('a', 'b')
    graph.edges['a', 'b']['geometry'] = shapely.LineString([[0, 0], [0, 1]])
    return graph

@pytest.fixture
def empty_graph():
    graph = nx.Graph()
    return graph


@pytest.fixture
def no_geometry_graph():
    graph = nx.Graph()
    return graph


@pytest.fixture
def invalid_node_graph():
    graph = nx.Graph()
    graph.add_node(0)
    return graph

@pytest.fixture
def invalid_edge_graph():
    graph = nx.Graph()
    graph.add_edge(0, 1)
    return graph



def test_network_check_valid(valid_graph, recwarn):
    """check if a valid network adheres to the convention"""
    assert len(recwarn) == 0, "expected 0 warnings before test started"
    opentnsim.utils.network_check(valid_graph)
    assert len(recwarn) == 0, "expected 0 warnings"

def test_network_check_empty(empty_graph):
    """check if a valid network adheres to the convention"""
    with pytest.warns(opentnsim.utils.NetworkWarning):
        opentnsim.utils.network_check(empty_graph)


def test_network_check_no_geometry(no_geometry_graph):
    """check if a valid network adheres to the convention"""
    with pytest.warns(opentnsim.utils.NetworkWarning):
        opentnsim.utils.network_check(no_geometry_graph)

def test_network_check_invalid_nodes(invalid_node_graph):
    """check if a valid network adheres to the convention"""
    with pytest.warns(opentnsim.utils.NetworkWarning):
         opentnsim.utils.network_check(invalid_node_graph)

def test_network_check_invalid_edges(invalid_edge_graph):
    """check if a valid network adheres to the convention"""
    with pytest.warns(opentnsim.utils.NetworkWarning):
        opentnsim.utils.network_check(invalid_edge_graph)
