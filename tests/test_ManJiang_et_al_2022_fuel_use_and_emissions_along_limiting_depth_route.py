'''Here we test the code for estimating fuel consumption and emission rates of CO2, PM10 and NOx for the three waterway sections along the route.'''


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

    Node = type("Site", (opentnsim.core.Identifiable, opentnsim.core.Locatable), {})

    data_node_1 = {"name": "Node 1", "geometry": shapely.geometry.Point(0, 0)}
    data_node_2 = {"name": "Node 2", "geometry": shapely.geometry.Point(0.8983, 0)}
    data_node_3 = {"name": "Node 3", "geometry": shapely.geometry.Point(1.7966, 0)}
    data_node_4 = {"name": "Node 4", "geometry": shapely.geometry.Point(2.6949, 0)}

    node_1 = Node(**data_node_1)
    node_2 = Node(**data_node_2)
    node_3 = Node(**data_node_3)
    node_4 = Node(**data_node_4)

    nodes = [node_1, node_2, node_3, node_4]

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
    middle_edges = [
        (node_2.name, node_3.name),
        (node_3.name, node_2.name)
    ]

    for e in middle_edges:
        edge = FG.edges[e]
        edge['Info']['GeneralDepth'] = 2.5
        
    # Make your preferred class out of available mix-ins.
    TransportResource = type(
        "Vessel",
        (
            opentnsim.core.Identifiable,
            opentnsim.core.Movable,
            opentnsim.core.Routeable,
            opentnsim.core.VesselProperties,
            opentnsim.core.ConsumesEnergy,
            opentnsim.core.ExtraMetadata,
        ),
        {},
    )

    # Create a dict with all important settings
    data_vessel = {"env": None,
                   "name": None,
                   "route": None,
                   "geometry": None,
                   "v": None,  # m/s
                   "type": None,
                   "B": 11.4,
                   "L": 110,
                   "H_e": None,
                   "H_f": None,
                   "T": 2,
                   "safety_margin": 0.3,  # for tanker vessel with rocky bed the safety margin is recommended as 0.3 m
                   "h_squat": True,  # if consider the ship squatting while moving, set to True, otherwise set to False
                   "P_installed": 1750.0, # kW
                   "P_tot_given": None,  # kW
                   "bulbous_bow": False,  # if a vessel has no bulbous_bow, set to False; otherwise set to True.
                   "P_hotel_perc": 0.05,
                   "P_hotel": None,  # None: calculate P_hotel from percentage
                   "L_w": 3.0,
                   "C_B": 0.85,
                   "C_year": 1990,
                   }

    vessel = TransportResource(**data_vessel)

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
        vessel.name = 'Vessel No.1'
        vessel.env = env  # the created environment
        vessel.route = path  # the route (the sequence of nodes, as stored as the second column in the path)
        vessel.geometry = env.FG.nodes[path[0]]['geometry']  # a shapely.geometry.Point(lon,lat) (here taken as the starting node of the vessel)
        vessel.v = V_s
        vessel.P_tot_given = P_tot_given

        # Start the simulation
        env.process(vessel.move())
        env.run()

        return vessel

    # prepare input data to loop through
    input_data = {'V_s': [3.0, 3.5, None, None], 'P_tot_given': [None, None, 212, 289]}

    # create empty plot data
    plot_data = {}

    # loop through the various input data
    for index, value in enumerate(input_data['V_s']):
        # Run a basic simulation with V_s and P_tot_given combi
        vessel = run_simulation(input_data['V_s'][index], input_data['P_tot_given'][index])

        # create an EnergyCalculation object and perform energy consumption calculation
        energycalculation = opentnsim.energy.EnergyCalculation(FG, vessel)
        energycalculation.calculate_energy_consumption()

        # create dataframe from energy calculation computation
        df = pd.DataFrame.from_dict(energycalculation.energy_use)

        # add/modify some comlums to suit our plotting needs
        df['fuel_kg_per_km'] = (df['total_fuel_consumption'] / 1000) / (df['distance'] / 1000)
        df['CO2_g_per_km'] = (df['total_emission_CO2']) / (df['distance'] / 1000)
        df['PM10_g_per_km'] = (df['total_emission_PM10']) / (df['distance'] / 1000)
        df['NOx_g_per_km'] = (df['total_emission_NOX']) / (df['distance'] / 1000)

        label = 'V_s = ' + str(input_data['V_s'][index]) + ' P_tot_given = ' + str(input_data['P_tot_given'][index])

        # Note that we make a dict to collect all plot data.
        # We use labels like ['V_s = None P_tot_given = 274 fuel_kg_km'] to organise the data in the dict
        # The [0, 0, 1, 1, 2, 2] below creates a list per section
        plot_data[label + ' fuel_kg_per_km']   = list(df.fuel_kg_per_km[[0, 0, 1, 1, 2, 2]])
        plot_data[label + ' CO2_g_per_km'] = list(df.CO2_g_per_km[[0, 0, 1, 1, 2, 2]])
        plot_data[label + ' PM10_g_per_km'] = list(df.PM10_g_per_km[[0, 0, 1, 1, 2, 2]])
        plot_data[label + ' NOx_g_per_km'] = list(df.NOx_g_per_km[[0, 0, 1, 1, 2, 2]])

    # todo: this test should be modified to test the fuel use and emission (looking at test name)
    # test the estimation of fuel consumption and emission rates of CO2, PM10 and NOx in section 1

# test the estimation of fuel consumption and emission rates of CO2, PM10 and NOx in section 1
    np.testing.assert_almost_equal(5.1776, plot_data['V_s = 3.0 P_tot_given = None fuel_kg_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(5.9119, plot_data['V_s = 3.5 P_tot_given = None fuel_kg_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)
  
    np.testing.assert_almost_equal(16427.1882, plot_data['V_s = 3.0 P_tot_given = None CO2_g_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(18757.0696, plot_data['V_s = 3.5 P_tot_given = None CO2_g_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)

    np.testing.assert_almost_equal(11.7777, plot_data['V_s = 3.0 P_tot_given = None PM10_g_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(11.7543, plot_data['V_s = 3.5 P_tot_given = None PM10_g_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)

    np.testing.assert_almost_equal(251.6599, plot_data['V_s = 3.0 P_tot_given = None NOx_g_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(266.2670, plot_data['V_s = 3.5 P_tot_given = None NOx_g_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)

    np.testing.assert_almost_equal(5.1747, plot_data['V_s = None P_tot_given = 212 fuel_kg_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(5.9101, plot_data['V_s = None P_tot_given = 289 fuel_kg_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)
   
    np.testing.assert_almost_equal(16418.0953, plot_data['V_s = None P_tot_given = 212 CO2_g_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(18751.2460, plot_data['V_s = None P_tot_given = 289 CO2_g_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)

    np.testing.assert_almost_equal(11.7786, plot_data['V_s = None P_tot_given = 212 PM10_g_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(11.7525, plot_data['V_s = None P_tot_given = 289 PM10_g_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)

    np.testing.assert_almost_equal(251.6164, plot_data['V_s = None P_tot_given = 212 NOx_g_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(266.2045, plot_data['V_s = None P_tot_given = 289 NOx_g_per_km'][0], decimal=2, err_msg='not almost equal', verbose=True) 
    
    # test the estimation of fuel consumption and emission rates of CO2, PM10 and NOx in section 2

    np.testing.assert_almost_equal(6.8960, plot_data['V_s = 3.0 P_tot_given = None fuel_kg_per_km'][1], decimal=0, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(12.7998, plot_data['V_s = 3.5 P_tot_given = None fuel_kg_per_km'][1], decimal=0, err_msg='not almost equal', verbose=True)
    
    np.testing.assert_almost_equal(21879.3371, plot_data['V_s = 3.0 P_tot_given = None CO2_g_per_km'][1], decimal=0, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(40610.4602, plot_data['V_s = 3.5 P_tot_given = None CO2_g_per_km'][1], decimal=0, err_msg='not almost equal', verbose=True)

    np.testing.assert_almost_equal(13.7115, plot_data['V_s = 3.0 P_tot_given = None PM10_g_per_km'][1], decimal=0, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(22.4056, plot_data['V_s = 3.5 P_tot_given = None PM10_g_per_km'][1], decimal=0, err_msg='not almost equal', verbose=True)

    np.testing.assert_almost_equal(310.5964, plot_data['V_s = 3.0 P_tot_given = None NOx_g_per_km'][1], decimal=0, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(559.1833, plot_data['V_s = 3.5 P_tot_given = None NOx_g_per_km'][1], decimal=0, err_msg='not almost equal', verbose=True)

    np.testing.assert_almost_equal(5.7584, plot_data['V_s = None P_tot_given = 212 fuel_kg_per_km'][1], decimal=2, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(6.8940, plot_data['V_s = None P_tot_given = 289 fuel_kg_per_km'][1], decimal=2, err_msg='not almost equal', verbose=True)
  
    np.testing.assert_almost_equal(18270.0350, plot_data['V_s = None P_tot_given = 212 CO2_g_per_km'][1], decimal=2, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(21872.9348, plot_data['V_s = None P_tot_given = 289 CO2_g_per_km'][1], decimal=2, err_msg='not almost equal', verbose=True)

    np.testing.assert_almost_equal(13.1072, plot_data['V_s = None P_tot_given = 212 PM10_g_per_km'][1], decimal=2, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(13.7091, plot_data['V_s = None P_tot_given = 289 PM10_g_per_km'][1], decimal=2, err_msg='not almost equal', verbose=True)

    np.testing.assert_almost_equal(279.9984, plot_data['V_s = None P_tot_given = 212 NOx_g_per_km'][1], decimal=2, err_msg='not almost equal', verbose=True)
    np.testing.assert_almost_equal(310.5219, plot_data['V_s = None P_tot_given = 289 NOx_g_per_km'][1], decimal=2, err_msg='not almost equal', verbose=True)
    

    
