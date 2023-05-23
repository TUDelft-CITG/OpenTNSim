"""Here we test the estimation of fuel use with and without current influence for single and round trips in section 1 and 3."""


# Importing libraries

# package(s) related to time, space and id
import datetime, time

import pathlib

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy
import pandas as pd

# spatial libraries
import shapely.geometry

# package(s) for data handling
import numpy as np

# OpenTNSim
import opentnsim

# Used for making the graph to visualize our problem
import networkx as nx

import pytest

import utils


# if you wnat to make a new expected dataframe, update the expected numbers like this
# df.to_csv("test_ManJiang_et_al_2022_current_influence_limiting_depth_single_and_round_trips_expected.csv", index=False)
@pytest.fixture
def expected_df():
    path = pathlib.Path(__file__)
    return utils.get_expected_df(path)


# Creating the test objects


# Actual testing starts here
# - tests 3 fixed velocities to return the right P_tot
# - tests 3 fixed power to return indeed the same P_tot
# - tests 3 fixed power to return indeed the same v
# todo: current tests do work with vessel.h_squat=True ... issues still for False
def test_simulation(expected_df):
    # specify a number of coordinate along your route (coords are: lon, lat)
    coords = [[0, 0], [0.8983, 0], [1.7966, 0], [2.6949, 0]]

    # for each edge (between above coordinates) specify the depth (m)
    depths = [6, 2.5, 6]

    # check of nr of coords and nr of depths align
    assert len(coords) == len(depths) + 1, "nr of depths does not correspond to nr of coords"

    # create a graph based on coords and depths
    graph = nx.DiGraph()
    nodes = []
    path = []

    # add nodes
    Node = type("Site", (opentnsim.core.Identifiable, opentnsim.core.Locatable), {})

    for index, coord in enumerate(coords):
        data_node = {
            "name": "Node " + str(index),
            "geometry": shapely.geometry.Point(coord[0], coord[1]),
        }
        nodes.append(Node(**data_node))

    positions = {}
    for node in nodes:
        positions[node.name] = (node.geometry.x, node.geometry.y)
        graph.add_node(node.name, geometry=node.geometry)

    # add edges
    path = [[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)]

    for index, edge in enumerate(path):
        # For the energy consumption calculation we add info to the graph. We need depth info for resistance.
        # NB: the CalculateEnergy routine expects the graph to have "Info" that contains "GeneralDepth"
        #     this may not be very generic!
        graph.add_edge(edge[0].name, edge[1].name, weight=1, Info={"GeneralDepth": depths[index]})

    # toggle to undirected and back to directed to make sure all edges are two way traffic
    graph = graph.to_undirected()
    graph = graph.to_directed()

    # Make your preferred class out of available mix-ins.
    TransportResource = type(
        "Vessel",
        (
            opentnsim.core.Identifiable,
            opentnsim.core.Movable,
            opentnsim.core.Routeable,
            opentnsim.core.VesselProperties,
            opentnsim.energy.ConsumesEnergy,
            opentnsim.core.ExtraMetadata,
        ),
        {},
    )

    # Create a dict with all important settings
    data_vessel = {
        "env": None,
        "name": None,
        "route": None,
        "geometry": None,
        "v": None,  # m/s
        "type": None,
        "B": 11.4,
        "L": 110,
        "H_e": None,
        "H_f": None,
        "T": 2.05,
        "safety_margin": 0.2,  # for tanker vessel with rocky bed the safety margin is recommended as 0.3 m
        "h_squat": True,  # if consider the ship squatting while moving, set to True, otherwise set to False
        "P_installed": 1750.0,  # kW
        "P_tot_given": None,  # kW
        "bulbous_bow": False,  # if a vessel has no bulbous_bow, set to False; otherwise set to True.
        "P_hotel_perc": 0.05,
        "P_hotel": None,  # None: calculate P_hotel from percentage
        "x": 2,
        "L_w": 3.0,
        "C_B": 0.85,
        "C_year": 1990,
    }

    vessel = TransportResource(**data_vessel)

    path = nx.dijkstra_path(graph, nodes[0].name, nodes[3].name)

    # Actual testing starts here
    def run_simulation(V_s, P_tot_given):
        # Start simpy environment
        simulation_start = datetime.datetime.now()
        env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))
        env.epoch = time.mktime(simulation_start.timetuple())

        # Add graph to environment
        env.graph = graph

        # Add environment and path to the vessel
        # create a fresh instance of vessel
        vessel = TransportResource(**data_vessel)
        vessel.env = env  # the created environment
        vessel.name = "Vessel No.1"
        vessel.route = path  # the route (the sequence of nodes, as stored as the second column in the path)
        vessel.geometry = env.graph.nodes[path[0]][
            "geometry"
        ]  # a shapely.geometry.Point(lon,lat) (here taken as the starting node of the vessel)
        vessel.v = V_s
        vessel.P_tot_given = P_tot_given

        # Start the simulation
        env.process(vessel.move())
        env.run()

        return vessel

    # prepare input data to loop through
    input_data = {"V_s": [3.5], "P_tot_given": [None]}

    # loop through the various input data
    for index, value in enumerate(input_data["V_s"]):
        # Run a basic simulation with V_s and P_tot_given combi
        vessel = run_simulation(input_data["V_s"][index], input_data["P_tot_given"][index])

        # create an EnergyCalculation object and perform energy consumption calculation
        energycalculation = opentnsim.energy.EnergyCalculation(graph, vessel)
        energycalculation.calculate_energy_consumption()

    # create dataframe from energy calculation computation
    df = pd.DataFrame.from_dict(energycalculation.energy_use)

    # add current influence
    # with current speed =0.5m/s
    U_c = 0.5
    # delta_t will be longer when upstream, shorter when downstream
    delta_t_up = df["distance"] / (df["distance"] / df["delta_t"] - U_c)
    delta_t_down = df["distance"] / (df["distance"] / df["delta_t"] + U_c)
    # total emission&fuel consumption will be larger when upstream(because of longer delta_t), smaller when downstream(because of shorter delta_t)
    df["total_fuel_consumption_kg"] = df["total_diesel_consumption_C_year_ICE_mass"] / 1000  # kg without current
    df["total_fuel_consumption_up_kg"] = df["total_diesel_consumption_C_year_ICE_mass"] / 1000 * (delta_t_up / df["delta_t"])  # kg
    df["total_fuel_consumption_down_kg"] = (
        df["total_diesel_consumption_C_year_ICE_mass"] / 1000 * (delta_t_down / df["delta_t"])
    )  # kg
    df["total_fuel_consumption_round_no_current_kg"] = df["total_diesel_consumption_C_year_ICE_mass"] / 1000 * 2  # kg
    df["total_fuel_consumption_round_current_kg"] = df["total_fuel_consumption_up_kg"] + df["total_fuel_consumption_down_kg"]  # kg

    # test the estimation of fuel consumption with and without current influence for section 1

    # if you wnat to make a new expected dataframe, update the expected numbers like this
    # df.to_csv("test_ManJiang_et_al_2022_current_influence_limiting_depth_single_and_round_trips_expected.csv", index=False)

    columns_to_test = [column for column in df.columns if "fuel" in column]
    pd.testing.assert_frame_equal(expected_df[columns_to_test], df[columns_to_test], check_exact=False)
