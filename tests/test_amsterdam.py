import networkx as nx
import pytest
import shapely
import simpy

import opentnsim


@pytest.fixture
def env():
    env = simpy.Environment(initial_time=0)
    print(f"made environment {env}")
    return env


@pytest.fixture
def simple_graph(env):
    graph = nx.Graph()
    graph.add_edge("a", "b")
    point_a = shapely.Point([0, 0])
    point_b = shapely.Point([0, 1])
    graph.edges["a", "b"]["geometry"] = shapely.LineString([point_a, point_b])
    node_a = graph.nodes["a"]
    node_b = graph.nodes["b"]
    node_a["geometry"] = point_a
    node_b["geometry"] = point_b
    return graph


def add_resources(env, graph):
    edge = graph.edges[("a", "b")]
    edge["Resources"] = simpy.Resource(env, capacity=1)


@pytest.fixture
def vessel_cls():
    Vessel = type(
        "Vessel",
        (
            opentnsim.core.Identifiable,
            opentnsim.core.Movable,
            opentnsim.core.Routable,
            opentnsim.core.HasResource,
            opentnsim.core.ExtraMetadata,
        ),
        {},
    )
    return Vessel


@pytest.fixture
def vessels(env, vessel_cls, simple_graph):
    v = 1
    geometry_a = simple_graph.nodes["a"]["geometry"]
    geometry_b = simple_graph.nodes["b"]["geometry"]
    route_a = ["a", "b"]
    route_b = ["b", "a"]

    add_resources(env, simple_graph)
    env.graph = simple_graph

    # instantiate vessel using the vessel_cls
    vessel_a = vessel_cls(env=env, name="vessel_a", v=v, geometry=geometry_a, node="a", route=route_a, graph=simple_graph)
    vessel_b = vessel_cls(env=env, name="vessel_b", v=v, geometry=geometry_b, node="b", route=route_b, graph=simple_graph)
    vessels = [vessel_a, vessel_b]
    return vessels


def test_network_check_valid(simple_graph):
    """check if we two ships with an edge with 1 capacity wait for eachother"""
    assert ("a", "b") in simple_graph.edges
    edge = simple_graph.edges[("a", "b")]

    add_resources(env, simple_graph)
    assert "Resources" in edge


def test_vessels(vessels):
    vessel_a = vessels[0]
    vessel_b = vessels[1]
    assert hasattr(vessel_a, "id")


def test_simulate(vessels, env):
    # specify the process that needs to be executed
    vessel_a = vessels[0]
    vessel_b = vessels[1]

    env.process(vessel_a.move())
    env.process(vessel_b.move())

    env.run()

    print(f"env of main {env}")

    last_t_a = vessel_a.logbook[-1]["Timestamp"]
    last_t_b = vessel_b.logbook[-1]["Timestamp"]

    assert last_t_b > last_t_a, "we expect b to finish sailing after a"
