"""Here we test the code for estimating renewable energy sources use along the validation route (Amsterdam-Ludwigshafen) with varying water depth."""
# Importing libraries
# package(s) related to time, space and id
import logging
import datetime, time
import pathlib
import platform
import itertools
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

# Used for mathematical functions
import math             
import tqdm
# Used for making the graph to visualize our problem
import networkx as nx   

import plotly.express as px
from plotly.subplots import make_subplots

import pytest
import utils


@pytest.fixture
def expected_df():
    path = pathlib.Path(__file__)
    return utils.get_expected_df(path)

# Creating the test objects

# Actual testing starts here

def test_simulation(expected_df):
    # specify a number of coordinate along your route (coords are: lon, lat)
    coords = [
        [0,0],
        [0.646776,0],
        [4.087265,0], 
        [4.536415,0],
        [5.3898,0]
    ] 

    # for each edge (between above coordinates) specify the depth (m)
    depths = [6, 4.5, 3.2, 4.5]

    # check of nr of coords and nr of depths align
    assert len(coords) == len(depths) + 1, 'nr of depths does not correspond to nr of coords'
    # create a graph based on coords and depths
    FG = nx.DiGraph()
    nodes = []
    path = []

    # add nodes
    Node = type('Site', (opentnsim.core.Identifiable, opentnsim.core.Locatable), {})

    for index, coord in enumerate(coords):
        data_node = {"name": "Node " + str(index), "geometry": shapely.geometry.Point(coord[0], coord[1])}
        nodes.append(Node(**data_node))

    positions = {}
    for node in nodes:
        positions[node.name] = (node.geometry.x, node.geometry.y)
        FG.add_node(node.name, geometry = node.geometry)

    # add edges
    path = [[nodes[i], nodes[i+1]] for i in range(len(nodes)-1)]

    for index, edge in enumerate(path):
        # For the energy consumption calculation we add info to the graph. We need depth info for resistance.
        # NB: the CalculateEnergy routine expects the graph to have "Info" that contains "GeneralDepth" 
        #     this may not be very generic!
        FG.add_edge(edge[0].name, edge[1].name, weight = 1, Info = {"GeneralDepth": depths[index]})

    # toggle to undirected and back to directed to make sure all edges are two way traffic
    FG = FG.to_undirected() 
    FG = FG.to_directed() 
    # Make your preferred class out of available mix-ins.
    TransportResource = type(
        "Vessel",
        (
            opentnsim.core.Identifiable,
            opentnsim.core.Movable,
            opentnsim.core.VesselProperties,  # needed to add vessel properties
            opentnsim.energy.ConsumesEnergy,
            opentnsim.core.ExtraMetadata,
        ),
        {},
    )  # needed to calculate resistances
    # Create a dict with all important settings

    data_vessel = {
        "env": None,
        "name": 'Vessel',
        "route": None,
        "geometry": None,
        "v": 1,  # m/s
        "type": None,
        "B": 11.4,
        "L": 135,
        "H_e": None, 
        "H_f": None, 
        "T": 2.6,
        "safety_margin": 0.3, # for tanker vessel with rocky bed the safety margin is recommended as 0.3 m
        "h_squat": True, # if consider the ship squatting while moving, set to True, otherwise set to False. Note that here we have disabled h_squat calculation since we regard the water depth h_0 is already reduced by squat effect. This applies to figures 3, 5, 7, 8 and 9.
        "payload":None,
        "vessel_type":"Tanker", #vessel types: "Container","Dry_SH","Dry_DH","Barge","Tanker". ("Dry_SH" means dry bulk single hull, "Dry_DH" means dry bulk double hull)    
        "P_installed": 2000, # kW  
        "P_tot_given": None, # kW
        "bulbous_bow": False, # if a vessel has no bulbous_bow, set to False; otherwise set to True.
        "P_hotel_perc": 0,
        "P_hotel": None, # None: calculate P_hotel from percentage
        "x": 2,# number of propellers
        "L_w": 3.0 ,
        "C_B":0.9, 
        "C_year": 2000,
    }             

    path = nx.dijkstra_path(FG, nodes[0].name, nodes[4].name)
    
    def run_simulation(V_s):
    
        # Start simpy environment
        simulation_start = datetime.datetime.now()
        env = simpy.Environment(initial_time = time.mktime(simulation_start.timetuple()))
        env.epoch = time.mktime(simulation_start.timetuple())

        # Add graph to environment
        env.FG = FG

        # Add environment and path to the vessel
        # create a fresh instance of vessel
        vessel = TransportResource(**data_vessel)
        vessel.env = env                                        #the created environment
        vessel.name = 'Vessel No.1'                     
        vessel.route = path                                     #the route (the sequence of nodes, as stored as the second column in the path)
        vessel.geometry = env.FG.nodes[path[0]]['geometry']     #a shapely.geometry.Point(lon,lat) (here taken as the starting node of the vessel)
        vessel.v = V_s
        # vessel.P_tot_given = P_tot_given

        # Start the simulation
        env.process(vessel.move())
        env.run()

        return vessel
    input_data = {'V_s': [3.33]} # 605km/50hr=12.1 km/h
    # create empty plot data
    plot_data = {}

    # loop through the various input data
    for index, value in enumerate(input_data['V_s']):

        # Run a basic simulation with V_s and P_tot_given combi
        vessel = run_simulation(input_data['V_s'][index])

        # create an EnergyCalculation object and perform energy consumption calculation
        energycalculation = opentnsim.energy.EnergyCalculation(FG, vessel)
        energycalculation.calculate_energy_consumption()

        # create dataframe from energy calculation computation
        df = pd.DataFrame.from_dict(energycalculation.energy_use)

        # add/modify some comlums to suit our plotting needs


        label = 'V_s = ' + str(input_data['V_s'][index]) 

        plot_data[label + ' total_LH2_consumption_PEMFC_mass']   = list(df.total_LH2_consumption_PEMFC_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_LH2_consumption_SOFC_mass']   = list(df.total_LH2_consumption_SOFC_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_LH2_consumption_PEMFC_vol']   = list(df.total_LH2_consumption_SOFC_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_LH2_consumption_SOFC_vol']   = list(df.total_LH2_consumption_SOFC_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eLNG_consumption_PEMFC_mass']   = list(df.total_eLNG_consumption_PEMFC_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eLNG_consumption_SOFC_mass']   = list(df.total_eLNG_consumption_SOFC_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eLNG_consumption_PEMFC_vol']   = list(df.total_eLNG_consumption_PEMFC_vol[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eLNG_consumption_SOFC_vol']   = list(df.total_eLNG_consumption_SOFC_vol[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eLNG_consumption_ICE_mass']   = list(df.total_eLNG_consumption_ICE_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eLNG_consumption_ICE_vol']   = list(df.total_eLNG_consumption_ICE_vol[[0, 0, 1, 1, 2, 2, 3, 3]])  
        plot_data[label + ' total_eMethanol_consumption_PEMFC_mass']   = list(df.total_eMethanol_consumption_PEMFC_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eMethanol_consumption_SOFC_mass']   = list(df.total_eMethanol_consumption_SOFC_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eMethanol_consumption_PEMFC_vol']   = list(df.total_eMethanol_consumption_PEMFC_vol[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eMethanol_consumption_SOFC_vol']   = list(df.total_eMethanol_consumption_SOFC_vol[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eMethanol_consumption_ICE_mass']   = list(df.total_eMethanol_consumption_ICE_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eMethanol_consumption_ICE_vol']   = list(df.total_eMethanol_consumption_ICE_vol[[0, 0, 1, 1, 2, 2, 3, 3]])  

        plot_data[label + ' total_eNH3_consumption_PEMFC_mass']   = list(df.total_eNH3_consumption_PEMFC_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eNH3_consumption_SOFC_mass']   = list(df.total_eNH3_consumption_SOFC_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eNH3_consumption_PEMFC_vol']   = list(df.total_eNH3_consumption_PEMFC_vol[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eNH3_consumption_SOFC_vol']   = list(df.total_eNH3_consumption_SOFC_vol[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eNH3_consumption_ICE_mass']   = list(df.total_eNH3_consumption_ICE_mass[[0, 0, 1, 1, 2, 2, 3, 3]])
        plot_data[label + ' total_eNH3_consumption_ICE_vol']   = list(df.total_eNH3_consumption_ICE_vol[[0, 0, 1, 1, 2, 2, 3, 3]])  

        plot_data[label + ' total_Li_NMC_Battery_mass']   = list(df.total_Li_NMC_Battery_mass[[0, 0, 1, 1, 2, 2, 3, 3]])  
        plot_data[label + ' total_Li_NMC_Battery_vol']   = list(df.total_Li_NMC_Battery_vol[[0, 0, 1, 1, 2, 2, 3, 3]])  
        plot_data[label + ' total_Battery2000kWh_consumption_num']   = list(df.total_Battery2000kWh_consumption_num[[0, 0, 1, 1, 2, 2, 3, 3]])  

    
    plot_df = pd.DataFrame(data=plot_data)
    
    
    # utils.create_expected_df(path=pathlib.Path(__file__), df=plot_df)
    columns_to_test = [
        column
        for column in plot_df.columns
    ]
    pd.testing.assert_frame_equal(
        expected_df[columns_to_test], plot_df[columns_to_test], check_exact=False
    )