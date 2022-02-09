# Importing libraries

# Used for mathematical functions
import math

# package(s) related to time, space and id
import datetime, time
import platform

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy
import pandas as pd

# spatial libraries
import pyproj
import shapely.geometry
from simplekml import Kml, Style

# package(s) for data handling
import numpy as np
import matplotlib.pyplot as plt

# OpenTNSim
import opentnsim

# Used for making the graph to visualize our problem
import networkx as nx

import pytest

# Creating the test objects


def test_correction_factors():
    corf = opentnsim.energy.correction_factors()
    assert corf.shape[0] > 10, "Correction factors are not available"


@pytest.fixture
def nodes():
    Node = type("Site", (opentnsim.core.Identifiable, opentnsim.core.Locatable), {})

    data_node_1 = {"name": "Node 1", "geometry": shapely.geometry.Point(0, 0)}
    data_node_2 = {"name": "Node 2", "geometry": shapely.geometry.Point(0.1, 0)}
    data_node_3 = {"name": "Node 3", "geometry": shapely.geometry.Point(0.2, 0)}
    data_node_4 = {"name": "Node 4", "geometry": shapely.geometry.Point(0.3, 0)}

    node_1 = Node(**data_node_1)
    node_2 = Node(**data_node_2)
    node_3 = Node(**data_node_3)
    node_4 = Node(**data_node_4)

    nodes = [node_1, node_2, node_3, node_4]
    return nodes


@pytest.fixture
def graph(nodes):
    """create a graph with 3 edges"""
    FG = nx.DiGraph()

    positions = {}
    for node in nodes:
        positions[node.name] = (node.geometry.x, node.geometry.y)
        FG.add_node(node.name, geometry=node.geometry)

    path = [
        [nodes[0], nodes[1]],  # From node 1 to node 2
        [nodes[1], nodes[2]],  # From node 2 to node 3
        [nodes[2], nodes[3]],  # From node 3 to node 4
        [nodes[3], nodes[2]],  # From node 4 to node 3
        [nodes[2], nodes[1]],  # From node 3 to node 2
        [nodes[1], nodes[0]],  # From node 2 to node 1
    ]

    for edge in path:
        # add depth to Info.GeneralDepth
        FG.add_edge(edge[0].name, edge[1].name, weight=1, Info={"GeneralDepth": 6})

    return FG


@pytest.fixture
def vessel():
    # Make your preferred class out of available mix-ins.
    TransportResource = type(
        "Vessel",
        (
            opentnsim.core.Identifiable,
            opentnsim.core.Movable,
            opentnsim.core.Routeable,
        ),
        {},
    )

    # Create a dict with all important settings
    data_vessel = {
        "env": None,
        "name": None,
        "route": None,
        "geometry": None,
        "v": 4,
    }  # constant speed of the vessel

    vessel = TransportResource(**data_vessel)
    return vessel


@pytest.fixture
def energy_vessel():
    # Make your preferred class out of available mix-ins.
    TransportResource = type(
        "Vessel",
        (
            opentnsim.core.Identifiable,
            opentnsim.core.Movable,
            opentnsim.core.Routeable,
            opentnsim.core.VesselProperties,  # needed to add vessel properties
            opentnsim.core.HasContainer,  # needed to calculate filling degree for draught
            opentnsim.core.ConsumesEnergy,
            opentnsim.core.ExtraMetadata,
        ),
        {},
    )  # needed to calculate resistances

    # Create a dict with all important settings
    data_vessel = {
        "env": None,
        "name": None,
        "route": None,
        "geometry": None,
        "v": None,  # m/s
        "type": "Dortmund-Eems (L <= 74 m)",  # <-- note that inputs from hereon are due to the added mixins!
        "B": 11.4,
        "L": 110,
        "capacity": 2500,  # maximum designed payload
        "level": 2500,  # actual payload
        "H_e": None,
        "H_f": None,
        "T": 3.5,
        "P_installed": 1750.0,
        "P_tot_given": 396,  # kW
        "L_w": 3.0,
        "C_b": 0.85,
        "c_year": 1990,
        "current_year": None,
    }

    vessel = TransportResource(**data_vessel)
    return vessel


# Actual testing starts here
def test_basic_simulation(graph, vessel, nodes):
    """test a basic simulation on the graph"""
    FG = graph

    node_1 = nodes[0]
    node_4 = nodes[3]

    path = nx.dijkstra_path(FG, node_1.name, node_4.name)

    # 4. Run simulation
    # Start simpy environment
    simulation_start = datetime.datetime.now()
    env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))
    env.epoch = time.mktime(simulation_start.timetuple())

    # Add graph to environment
    env.FG = FG

    # Add environment and path to the vessel
    vessel.env = env  # the created environment
    vessel.name = "Vessel No.1"
    vessel.route = path  # the route (the sequence of nodes, as stored as the second column in the path)
    vessel.geometry = env.FG.nodes[path[0]][
        "geometry"
    ]  # a shapely.geometry.Point(lon,lat) (here taken as the starting node of the vessel)

    # Start the simulation
    env.process(vessel.move())
    env.run()

    # 5. Test output
    def calculate_distance(orig, dest):
        wgs84 = pyproj.Geod(ellps="WGS84")

        distance = wgs84.inv(orig[0], orig[1], dest[0], dest[1])[2]

        return distance

    vessel_path_x = []
    vessel_path_t = []

    for t in range(0, len(vessel.log["Message"])):
        vessel_path_x0 = (vessel.log["Geometry"][0].x, vessel.log["Geometry"][0].y)
        vessel_path_t0 = vessel.log["Timestamp"][0].timestamp() / 60
        vessel_path_x.append(
            calculate_distance(
                vessel_path_x0,
                (vessel.log["Geometry"][t].x, vessel.log["Geometry"][t].y),
            )
        )
        vessel_path_t.append(
            vessel.log["Timestamp"][t].timestamp() / 60 - vessel_path_t0
        )

    # Test if max distance / speed roughly is equal to simulation time (NB: should be exactly the same (bug))
    np.testing.assert_almost_equal(
        (vessel_path_x[-1] / vessel.v) / 60,
        vessel_path_t[-1],
        decimal=7,
        err_msg="",
        verbose=True,
    )


def test_fixed_power_varying_depth(graph, energy_vessel, nodes):
    """test a basic simulation on the graph"""

    # get the graph from the fixtures
    FG = graph
    # here we'll use a vessel with energy mixed in
    vessel = energy_vessel

    # set the middle edge to a different waterdepth
    middle_edges = [(nodes[1].name, nodes[2].name), (nodes[2].name, nodes[1].name)]
    for e in middle_edges:
        edge = FG.edges[e]
        edge["Info"]["GeneralDepth"] = 4

    path = nx.dijkstra_path(FG, nodes[0].name, nodes[3].name)

    # 4. Run simulation
    # Start simpy environment
    simulation_start = datetime.datetime(2000, 1, 1)
    env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))
    env.epoch = time.mktime(simulation_start.timetuple())

    # Add graph to environment
    env.FG = FG

    # Add environment and path to the vessel
    vessel.env = env  # the created environment
    vessel.name = "Vessel No.1"
    vessel.route = path  # the route (the sequence of nodes, as stored as the second column in the path)
    vessel.geometry = env.FG.nodes[path[0]][
        "geometry"
    ]  # a shapely.geometry.Point(lon,lat) (here taken as the starting node of the vessel)

    # Start the simulation
    env.process(vessel.move())
    env.run()

    df = pd.DataFrame.from_dict(vessel.log)
    dt_1 = df.loc[2]["Timestamp"] - df.loc[0]["Timestamp"]
    dt_2 = df.loc[4]["Timestamp"] - df.loc[2]["Timestamp"]
    assert dt_2 > dt_1, f"second edge {dt_2} should take longer than first edge {dt_1}"


def test_power2v(graph, energy_vessel, nodes):
    """test a basic simulation on the graph"""

    # get the graph from the fixtures
    FG = graph
    # here we'll use a vessel with energy mixed in
    vessel = energy_vessel

    # set the middle edge to a different waterdepth
    middle_e = (nodes[1].name, nodes[2].name)
    edge = FG.edges[middle_e]
    edge["Info"]["GeneralDepth"] = 2.5

    # Add environment and path to the vessel
    simulation_start = datetime.datetime(2000, 1, 1)
    env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))
    env.epoch = time.mktime(simulation_start.timetuple())

    v = opentnsim.energy.power2v(vessel, edge)
    assert v < 5, "power should be low"
