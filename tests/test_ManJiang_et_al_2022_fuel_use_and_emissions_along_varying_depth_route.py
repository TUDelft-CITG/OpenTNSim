"""Here we test the code for estimating fuel consumption and emission rates of CO2, PM10 and NOx for the three waterway sections along the route."""
# Importing libraries

# package(s) related to time, space and id
import datetime, time

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

# Creating the test objects

# Actual testing starts here
# - tests 3 fixed velocities to return the right P_tot
# - tests 3 fixed power to return indeed the same P_tot
# - tests 3 fixed power to return indeed the same v
# todo: current tests do work with vessel.h_squat=True ... issues still for False
def test_simulation():
    # specify a number of coordinate along your route (coords are: lon, lat)
    coords = [[0, 0], [0.8983, 0], [1.7966, 0], [2.6949, 0]]

    # for each edge (between above coordinates) specify the depth (m)
    depths = [6, 4.5, 6]

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
        "v": None,  # m/s
        "type": None,
        "B": 11.4,
        "L": 110,
        "H_e": None,
        "H_f": None,
        "T": 3.5,  # <=== here we should enter the value from the T strategy notebook
        "safety_margin": 0.2,  # for tanker vessel with sandy bed the safety margin is recommended as 0.2 m
        "h_squat": True,  # if consider the ship squatting while moving, set to True, otherwise set to False
        "P_installed": 1750.0,
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
    def run_simulation(V_s, P_tot_given):
        # Start simpy environment
        simulation_start = datetime.datetime.now()
        env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))
        env.epoch = time.mktime(simulation_start.timetuple())

        # Add graph to environment
        env.FG = FG

        # Add environment and path to the vessel
        # create a fresh instance of vessel
        vessel = TransportResource(**data_vessel)
        vessel.name = "Vessel No.1"
        vessel.env = env  # the created environment
        vessel.route = path  # the route (the sequence of nodes, as stored as the second column in the path)
        vessel.geometry = env.FG.nodes[path[0]][
            "geometry"
        ]  # a shapely.geometry.Point(lon,lat) (here taken as the starting node of the vessel)
        vessel.v = V_s
        vessel.P_tot_given = P_tot_given

        # Start the simulation
        env.process(vessel.move())
        env.run()

        return vessel

    # prepare input data to loop through
    input_data = {
        "V_s": [3.0, 3.5, 4.0, None, None, None],
        "P_tot_given": [None, None, None, 333, 473, 707],
    }

    # create empty plot data
    plot_data = {}

    # loop through the various input data
    for index, value in enumerate(input_data["V_s"]):
        # Run a basic simulation with V_s and P_tot_given combi
        vessel = run_simulation(
            input_data["V_s"][index], input_data["P_tot_given"][index]
        )

        # create an EnergyCalculation object and perform energy consumption calculation
        energycalculation = opentnsim.energy.EnergyCalculation(FG, vessel)
        energycalculation.calculate_energy_consumption()

        # create dataframe from energy calculation computation
        df = pd.DataFrame.from_dict(energycalculation.energy_use)

        # add/modify some comlums to suit our plotting needs
        df["fuel_kg_per_km"] = (df["total_fuel_consumption"] / 1000) / (
            df["distance"] / 1000
        )
        df["CO2_g_per_km"] = (df["total_emission_CO2"]) / (df["distance"] / 1000)
        df["PM10_g_per_km"] = (df["total_emission_PM10"]) / (df["distance"] / 1000)
        df["NOx_g_per_km"] = (df["total_emission_NOX"]) / (df["distance"] / 1000)

        label = (
            "V_s = "
            + str(input_data["V_s"][index])
            + " P_tot_given = "
            + str(input_data["P_tot_given"][index])
        )

        # Note that we make a dict to collect all plot data.
        # We use labels like ['V_s = None P_tot_given = 274 fuel_kg_km'] to organise the data in the dict
        # The [0, 0, 1, 1, 2, 2] below creates a list per section
        plot_data[label + " fuel_kg_per_km"] = list(
            df.fuel_kg_per_km[[0, 0, 1, 1, 2, 2]]
        )
        plot_data[label + " CO2_g_per_km"] = list(df.CO2_g_per_km[[0, 0, 1, 1, 2, 2]])
        plot_data[label + " PM10_g_per_km"] = list(df.PM10_g_per_km[[0, 0, 1, 1, 2, 2]])
        plot_data[label + " NOx_g_per_km"] = list(df.NOx_g_per_km[[0, 0, 1, 1, 2, 2]])

    # todo: this test should be modified to test the fuel use and emission (looking at test name)
    # test the estimation of fuel consumption and emission rates of CO2, PM10 and NOx in section 1

    # test the estimation of fuel consumption and emission rates of CO2, PM10 and NOx in section 1
    np.testing.assert_almost_equal(
        7.837832301403712,
        plot_data["V_s = 3.0 P_tot_given = None fuel_kg_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        24867.304301726326,
        plot_data["V_s = 3.0 P_tot_given = None CO2_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        14.98497866583688,
        plot_data["V_s = 3.0 P_tot_given = None PM10_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        346.6988207378088,
        plot_data["V_s = 3.0 P_tot_given = None NOx_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        9.257709061364135,
        plot_data["V_s = 3.5 P_tot_given = None fuel_kg_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        29372.1860219644,
        plot_data["V_s = 3.5 P_tot_given = None CO2_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        16.561925094606398,
        plot_data["V_s = 3.5 P_tot_given = None PM10_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        398.49667501122605,
        plot_data["V_s = 3.5 P_tot_given = None NOx_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        11.533658858078743,
        plot_data["V_s = 4.0 P_tot_given = None fuel_kg_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        36593.15401335892,
        plot_data["V_s = 4.0 P_tot_given = None CO2_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        20.18525814009707,
        plot_data["V_s = 4.0 P_tot_given = None PM10_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        505.09194954938715,
        plot_data["V_s = 4.0 P_tot_given = None NOx_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        7.839086092433789,
        plot_data["V_s = None P_tot_given = 333 fuel_kg_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        24871.282238721753,
        plot_data["V_s = None P_tot_given = 333 CO2_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        14.985716116624024,
        plot_data["V_s = None P_tot_given = 333 PM10_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        346.7367894223322,
        plot_data["V_s = None P_tot_given = 333 NOx_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        9.26206168494091,
        plot_data["V_s = None P_tot_given = 473 fuel_kg_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        29385.995709494342,
        plot_data["V_s = None P_tot_given = 473 CO2_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        16.568206928090493,
        plot_data["V_s = None P_tot_given = 473 PM10_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        398.6816996885613,
        plot_data["V_s = None P_tot_given = 473 NOx_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        11.538078892105544,
        plot_data["V_s = None P_tot_given = 707 fuel_kg_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        36607.17757586213,
        plot_data["V_s = None P_tot_given = 707 CO2_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        20.192914358397385,
        plot_data["V_s = None P_tot_given = 707 PM10_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        505.309812232846,
        plot_data["V_s = None P_tot_given = 707 NOx_g_per_km"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )

    # test the estimation of fuel consumption and emission rates of CO2, PM10 and NOx in section 2

    np.testing.assert_almost_equal(
        8.21275465803749,
        plot_data["V_s = 3.0 P_tot_given = None fuel_kg_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        26056.830687773498,
        plot_data["V_s = 3.0 P_tot_given = None CO2_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        15.4470714311579,
        plot_data["V_s = 3.0 P_tot_given = None PM10_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        360.5979582619547,
        plot_data["V_s = 3.0 P_tot_given = None NOx_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        10.278855744922756,
        plot_data["V_s = 3.5 P_tot_given = None fuel_kg_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        32612.005954345837,
        plot_data["V_s = 3.5 P_tot_given = None CO2_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        18.171507176091854,
        plot_data["V_s = 3.5 P_tot_given = None PM10_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        442.3896778403027,
        plot_data["V_s = 3.5 P_tot_given = None NOx_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        14.66168926584354,
        plot_data["V_s = 4.0 P_tot_given = None fuel_kg_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        46517.5413979945,
        plot_data["V_s = 4.0 P_tot_given = None CO2_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        25.8841261315404,
        plot_data["V_s = 4.0 P_tot_given = None PM10_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        651.0213521686129,
        plot_data["V_s = 4.0 P_tot_given = None NOx_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        8.022460036867594,
        plot_data["V_s = None P_tot_given = 333 fuel_kg_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        25453.077753334455,
        plot_data["V_s = None P_tot_given = 333 CO2_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        15.336265886936538,
        plot_data["V_s = None P_tot_given = 333 PM10_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        354.8477466856465,
        plot_data["V_s = None P_tot_given = 333 NOx_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        9.622663173608945,
        plot_data["V_s = None P_tot_given = 473 fuel_kg_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        30530.085887177473,
        plot_data["V_s = None P_tot_given = 473 CO2_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        17.213259852951897,
        plot_data["V_s = None P_tot_given = 473 PM10_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        414.2036449253217,
        plot_data["V_s = None P_tot_given = 473 NOx_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        12.205883840879995,
        plot_data["V_s = None P_tot_given = 707 fuel_kg_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        38725.940549701074,
        plot_data["V_s = None P_tot_given = 707 CO2_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        21.361646882726653,
        plot_data["V_s = None P_tot_given = 707 PM10_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        534.5563113793555,
        plot_data["V_s = None P_tot_given = 707 NOx_g_per_km"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
