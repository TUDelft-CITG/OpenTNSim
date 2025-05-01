# Importing libraries
import datetime
import time

# Used for making the graph to visualize our problem
import networkx as nx

# package(s) for data handling

# OpenTNSim
import opentnsim

# spatial libraries
import shapely.geometry

import simpy
import pytest


@pytest.fixture
def graph():
    # 1. create graph
    FG = nx.DiGraph()

    node_0 = {"n": "0", "geometry": shapely.geometry.Point(0, 0)}
    node_1 = {"n": "1", "geometry": shapely.geometry.Point(0.1, 0)}

    nodes = [node_0, node_1]
    for node in nodes:
        FG.add_node(node["n"], geometry=node["geometry"])

    edge_info = {"GeneralDepth": [6.0]}
    # share edge info accross all edges
    FG.add_edge("0", "1", Info=edge_info)
    FG.add_edge("1", "0", Info=edge_info)
    return FG


@pytest.fixture
def vessel(graph, env):
    Vessel = type(
        "Vessel",
        (opentnsim.core.Identifiable, opentnsim.core.Movable, opentnsim.core.Routeable),
        {},
    )

    # Create a dict with all important settings
    data_vessel = {
        "env": env,
        "name": "Vessel",
        "route": ["0", "1"],
        "geometry": graph.nodes["0"]["geometry"],
        "v": 4,
    }  # constant speed of the vessel

    vessel = Vessel(**data_vessel)
    return vessel


@pytest.fixture
def env(graph):
    # 4. Run simulation
    # Start simpy environment
    simulation_start = datetime.datetime.now()
    env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))
    env.epoch = time.mktime(simulation_start.timetuple())

    env.graph = graph
    return env


def test_basic_energy(vessel, env, graph):
    # Start the simulation
    env.process(vessel.move())
    env.run()

    print(vessel.logbook)
