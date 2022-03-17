"""Tests for sailing strategies"""

import datetime
import time

import networkx as nx
import shapely.geometry
import simpy
import numpy as np

import opentnsim.core
import opentnsim.strategy

import pytest


@pytest.fixture
def graph():
    """Example graph with depths of 6 except for the middle edge which has a depth of 2.5"""
    FG = nx.DiGraph()
    nodes = []
    path = []

    Node = type('Site', (opentnsim.core.Identifiable, opentnsim.core.Locatable), {})

    data_node_1 = {"name": "Node 1",
                   "geometry": shapely.geometry.Point(0, 0)}
    data_node_2 = {"name": "Node 2",
                   "geometry": shapely.geometry.Point(0.8983, 0)}  # 0.8983 degree =100km
    data_node_3 = {"name": "Node 3",
                   "geometry": shapely.geometry.Point(1.7966, 0)}   # 1.7966 degree =200km
    data_node_4 = {"name": "Node 4",
                   "geometry": shapely.geometry.Point(2.6949, 0)}    # 2.6949 degree =300km

    node_1 = Node(**data_node_1)
    node_2 = Node(**data_node_2)
    node_3 = Node(**data_node_3)
    node_4 = Node(**data_node_4)

    nodes = [node_1, node_2, node_3, node_4]

    positions = {}
    for node in nodes:
        positions[node.name] = (node.geometry.x, node.geometry.y)
        FG.add_node(node.name, geometry = node.geometry)

    path = [[node_1, node_2], # From node 1 to node 2
            [node_2, node_3], # From node 2 to node 3
            [node_3, node_4], # From node 3 to node 4
            [node_4, node_3], # From node 4 to node 3
            [node_3, node_2], # From node 3 to node 2
            [node_2, node_1]] # From node 2 to node 1


    for edge in path:
        # For the energy consumption calculation we add info to the graph. We need depth info for resistance.
        # NB: the CalculateEnergy routine expects the graph to have "Info" that contains "GeneralDepth"
        #     this may not be very generic!
        FG.add_edge(edge[0].name, edge[1].name, weight = 1, Info = {"GeneralDepth": 6})

    middle_edges = [
        (node_2.name, node_3.name),
        (node_3.name, node_2.name)
    ]
    for e in middle_edges:
        edge = FG.edges[e]
        edge['Info']['GeneralDepth'] = 2.5
    return FG

@pytest.fixture
def path(graph):
    """full path from Node 1 to Node 4"""
    FG = graph
    path = nx.dijkstra_path(FG, 'Node 1', 'Node 4')
    return path

@pytest.fixture
def path_bottleneck(graph):
    """path from Node 2 to Node 3, the shallow part"""
    FG = graph
    path_bottleneck = nx.dijkstra_path(FG, 'Node 2', 'Node 3')
    return path_bottleneck


@pytest.fixture
def env(graph):
    """create a simpy environment with a graph attached"""

    FG = graph
    # Start simpy environment
    simulation_start = datetime.datetime(2000, 1, 1)
    env = simpy.Environment(initial_time = time.mktime(simulation_start.timetuple()))
    env.epoch = time.mktime(simulation_start.timetuple())

    # Add graph to environment
    env.FG = FG
    return env

@pytest.fixture
def T_strategy():
    return 2.1


@pytest.fixture
def vessel(env, path, T_strategy):
    """create a fresh instance of vessel"""

    data_vessel = {
        "env": None,
        "name": None,
        "route": None,
        "geometry": None,
        "v": 1,  # m/s
        "type": 'Dortmund-Eems (L <= 74 m)', # <-- note that inputs from hereon are due to the added mixins!
        "B": 11.4,
        "L": 110,
        "capacity": 2500,   # maximum designed payload
        "level":2500,     # actual payload
        "H_e": None,
        "H_f": None,
        "T": None,
        "safety_margin": 0.3, # for tanker vessel with rocky bed the safety margin is recommended as 0.3 m
        "h_squat":None,
        "P_installed": 1750.0,
        "P_tot_given": None, # kW
        "L_w": 3.0 ,
        "C_B": 0.85,
        "C_year": 1990,
        "current_year": None,
        "bulbous_bow": None
    }
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

    #    print('start the vessel')
    vessel = TransportResource(**data_vessel)
    vessel.name = 'Vessel No.1'
    # add environment and path to the vessel
    vessel.env = env                                        #the created environment
    vessel.route = path                                     #the route (the sequence of nodes, as stored as the second column in the path)

    # position vessel at the start of the path
    vessel.geometry = env.FG.nodes[path[0]]['geometry']     #a shapely.geometry.Point(lon,lat) (here taken as the starting node of the vessel)

    # new properties added
    vessel._T =  T_strategy # start with the minimum operational draft (and fill up)
    return vessel

@pytest.fixture
def vessel_type():
    return "Tanker"


def test_sailing_strategy(graph, path, vessel, T_strategy):
    """Test to see if we can call formulate_sailing_strategies"""
    # a deep ship with 40cm under keel clearance
    # Do this computation for this T_strategy (independent of the vessel.T)
    v_T2v, v_P2v, v_max_final = opentnsim.strategy.get_v_max_for_bottleneck(FG=graph, path=path, vessel=vessel, T_strategy=T_strategy)
    # these are the results previously. Use this as a regression test to make sure results stay the same.
    np.testing.assert_almost_equal(v_T2v, 4.405382019875273)
    np.testing.assert_almost_equal(v_P2v, 3.9146400450887224)


def test_payload_strategy(vessel, T_strategy, vessel_type):
    """Test to see if we can call formulate_sailing_strategies"""
    payload = opentnsim.strategy.T2Payload(vessel=vessel, T_strategy=T_strategy, vessel_type=vessel_type)

    # these are the results previously. Use this as a regression test to make sure results stay the same.
    np.testing.assert_almost_equal(payload, +1112.7660789593735)
