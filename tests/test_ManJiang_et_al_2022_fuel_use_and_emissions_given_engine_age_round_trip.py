"""Here we test the code for estimating fuel consumption and emission rates of CO2, PM10 and NOx for the round trip given engine age."""


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


@pytest.fixture
def expected_df():
    path = pathlib.Path(__file__)
    return utils.get_expected_df(path)

# Creating the test objects

# todo: current tests do work with vessel.h_squat=True ... issues still for False
def test_simulation(expected_df):
    # specify a number of coordinate along your route (coords are: lon, lat)
    coords = [[0, 0], [0.8983, 0], [1.7966, 0], [2.6949, 0]]

    # for each edge (between above coordinates) specify the depth (m)
    depths = [6, 6, 6]

    # check of nr of coords and nr of depths align
    assert (
        len(coords) == len(depths) + 1
    ), "nr of depths does not correspond to nr of coords"

    # create a graph based on coords and depths
    FG = nx.DiGraph()
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
        FG.add_node(node.name, geometry=node.geometry)

    # add edges
    path = [[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)]

    for index, edge in enumerate(path):
        # For the energy consumption calculation we add info to the graph. We need depth info for resistance.
        # NB: the CalculateEnergy routine expects the graph to have "Info" that contains "GeneralDepth"
        #     this may not be very generic!
        FG.add_edge(
            edge[0].name, edge[1].name, weight=1, Info={"GeneralDepth": depths[index]}
        )

    # toggle to undirected and back to directed to make sure all edges are two way traffic
    FG = FG.to_undirected()
    FG = FG.to_directed()

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
        "v": 3.5,  # m/s
        "type": None,
        "B": 11.4,
        "L": 110,
        "H_e": None,
        "H_f": None,
        "T": 3.5,
        "safety_margin": 0.3,  # for tanker vessel with rocky bed the safety margin is recommended as 0.3 m
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

    path = nx.dijkstra_path(FG, nodes[0].name, nodes[3].name)

    # Actual testing starts here
    def run_simulation(C_year):

        # Start simpy environment
        simulation_start = datetime.datetime.now()
        env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))
        env.epoch = time.mktime(simulation_start.timetuple())

        # Add graph to environment
        env.FG = FG

        # Add environment and path to the vessel
        # create a fresh instance of vessel
        vessel = TransportResource(**data_vessel)
        vessel.env = env  # the created environment
        vessel.name = "Vessel No.1"
        vessel.route = path  # the route (the sequence of nodes, as stored as the second column in the path)
        vessel.geometry = env.FG.nodes[path[0]][
            "geometry"
        ]  # a shapely.geometry.Point(lon,lat) (here taken as the starting node of the vessel)
        vessel.C_year = C_year

        # Start the simulation
        env.process(vessel.move())
        env.run()

        return vessel

    # prepare input data to loop through
    input_data = {"C_year": [1970, 1980, 1990, 2000, 2010, 2020]}

    plot_data = {}

    # loop through the various input data
    for index, value in enumerate(input_data["C_year"]):
        # Run a basic simulation with V_s and P_tot_given combi
        vessel = run_simulation(C_year=input_data["C_year"][index])

        # create an EnergyCalculation object and perform energy consumption calculation
        energycalculation = opentnsim.energy.EnergyCalculation(FG, vessel)
        energycalculation.calculate_energy_consumption()

        # create dataframe from energy calculation computation
        df = pd.DataFrame.from_dict(energycalculation.energy_use)

        # add/modify some comlums to suit our plotting needs
        df["total_fuel_consumption_600km"] = (
            df["total_diesel_consumption_C_year_ICE_mass"] * 6 / 1000
        )  # kg, a round trip is 600km in total
        df["total_emission_CO2_600km"] = df["total_emission_CO2"] * 6 / 1000  # kg
        df["total_emission_PM10_600km"] = df["total_emission_PM10"] * 6 / 1000  # kg
        df["total_emission_NOX_600km"] = df["total_emission_NOX"] * 6 / 1000  # kg

        # Note that we make a dict to collect all plot data.
        # We use labels like ['c_year = 1970'] organise the data in the dict

        label = "C_year = " + str(input_data["C_year"][index])
        plot_data[label + ", total_fuel_consumption_600km"] = list(
            df.total_fuel_consumption_600km[[0]]
        )
        plot_data[label + ", total_emission_CO2_600km"] = list(
            df.total_emission_CO2_600km[[0]]
        )
        plot_data[label + ", total_emission_PM10_600km"] = list(
            df.total_emission_PM10_600km[[0]]
        )
        plot_data[label + ", total_emission_NOX_600km"] = list(
            df.total_emission_NOX_600km[[0]]
        )

        # Note that we make a dict to collect all plot data.
        # We use labels like ['c_year = 1970'] organise the data in the dict

        label = "C_year = " + str(input_data["C_year"][index])
        plot_data[label + ", total_fuel_consumption_600km"] = list(
            df.total_fuel_consumption_600km[[0]]
        )
        plot_data[label + ", total_emission_CO2_600km"] = list(
            df.total_emission_CO2_600km[[0]]
        )
        plot_data[label + ", total_emission_PM10_600km"] = list(
            df.total_emission_PM10_600km[[0]]
        )
        plot_data[label + ", total_emission_NOX_600km"] = list(
            df.total_emission_NOX_600km[[0]]
        )
    plot_df = pd.DataFrame(data=plot_data)
    
    
    # utils.create_expected_df(path=pathlib.Path(__file__), df=plot_df)
    columns_to_test = [
        column
        for column in plot_df.columns
    ]
    pd.testing.assert_frame_equal(
        expected_df[columns_to_test], plot_df[columns_to_test], check_exact=False
    )

