#!/usr/bin/env python3

import copy
import datetime
import pickle
import itertools
import time
import functools

import simpy
import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
import shapely.geometry
import matplotlib.pyplot as plt

import opentnsim.core

def run_simulation(geometry, route, graph, engine_order=0.8):
    Vessel = type(
        'Vessel', 
        (
            opentnsim.core.Identifiable, 
            opentnsim.core.Movable, 
            opentnsim.core.Routeable, 
            opentnsim.core.VesselProperties,
            opentnsim.energy.ConsumesEnergy,
            opentnsim.core.ExtraMetadata
        ), 
        {}
    )

    max_v = 5.0
    P_installed = 1750
    
    data_vessel = {
        "env": None,
        "name": "NausBot",
        "route": None,
        "geometry": geometry,
        "type": "Va",
        "B": 11.4,
        "L": 110,
        'P_installed': P_installed, 
        'L_w': 3, 
        "T": 3.5,
        'C_year': 1997,
        "v": engine_order * max_v
    
    } 
    vessel = Vessel(**data_vessel)
    
        

    # start simpy environment (specify the start time and add the graph to the environment)
    simulation_start = datetime.datetime.now()
    env = simpy.Environment(initial_time = time.mktime(simulation_start.timetuple()))
    env.FG = graph

    # add environment to the vessel, and specify the vessels route and current location (beginning of the path)
    vessel.env = env
    vessel.route = route
    
    # specify the process that needs to be executed
    env.process(vessel.move())

    # start the simulation
    env.run()

    #Determine trip time
    start_time = simulation_start.timestamp()
        

    end = vessel.log["Timestamp"][-1]

    end_time = end.timestamp()

    energycalculation = opentnsim.energy.EnergyCalculation(graph, vessel)
    energycalculation.calculate_energy_consumption()

    # create dataframe from energy calculation computation
    energy_df = pd.DataFrame.from_dict(energycalculation.energy_use)

    return end_time-start_time, end, energy_df, vessel




def generate_route_alternatives(graph):
    """generate a list of dictionaries, where each dictionary is a route alternatives with name, waypoints and a route."""
    #Conditional toevoegen die rekening houdt met de visited nodes?
    route_alternatives = [
        {
            "name": "direct",
            "waypoints": ['A', 'F', 'H', 'L']
        },
        {
            "name": "redirect",
            "waypoints": ['A', 'F', 'L', 'F']
        },
    ]


    for alternative in route_alternatives:
        route = []    
        waypoints = alternative["waypoints"]
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            path = nx.shortest_path(graph, a, b)
            route.extend(path)

        route = [
            k 
            for (k, g)
            in itertools.groupby(route)
        ]


        alternative["route"] = route
        alternative["og_route"] = route
    return route_alternatives


def generate_engine_alternatives():
    """generate a list of dictionaries, where each dictionary is an engine order setting."""
    engine_orders = [0.5, 0.6, 0.7, 0.8, 0.9]
    engine_alternatives = []
    for engine_order in engine_orders:
        alternative = {"engine_order": engine_order} 
        engine_alternatives.append(alternative)
    return engine_alternatives

def generate_all_alternatives(graph):
    """generate a pandas dataframe with alternatives for green routing and green steaming"""
    alternatives = []
    engine_alternatives = generate_engine_alternatives()
    for engine_alternative in engine_alternatives:
        for route_alternative in generate_route_alternatives(graph):
            alternative = copy.deepcopy(route_alternative)
            alternative.update(engine_alternative)
            alternatives.append(alternative)
            
        
    alternatives_df = pd.DataFrame(alternatives)
    # not the direct route
    alternatives_df['green_routing'] = alternatives_df['name'] == 'redirect'
    # not the maximum power
    highest_engine_order = engine_alternatives[-1]["engine_order"]
    alternatives_df['green_steaming'] = alternatives_df['engine_order'] < highest_engine_order
    return alternatives_df


def add_kpi(alternatives_df, berth_available, graph, geometry, visited_nodes, waypoints=None):

    result = alternatives_df.copy()
    result['remaining_route'] = None
    result['remaining_route'] = result['remaining_route'].astype(object)
    for idx, row in alternatives_df.iterrows():
        print(type(row['route'][0]))
        remaining_route = copy.copy(row['route'])
        print('Removing', visited_nodes, 'from', remaining_route)
        for node in visited_nodes:
            print('Node', node, 'Route', remaining_route[0])
            if remaining_route[0] == node:
                removed_node = remaining_route.pop(0)
        print('Remaining route', remaining_route)
        #current_node = row['route'][0]
        #geometry = graph.nodes[current_node]['geometry']
        engine_order = row['engine_order']
        duration, eta, energy_df, vessel = run_simulation(geometry=geometry, route=remaining_route, graph=graph, engine_order=engine_order)

        energy_sum = energy_df.sum(axis=0, numeric_only=True) 
        result.at[idx, 'duration'] = duration
        result.at[idx, 'total_energy'] = energy_sum['total_energy']
        result.at[idx, 'total_emission_CO2'] = energy_sum['total_emission_CO2']
        result.at[idx, 'total_emission_PM10'] = energy_sum['total_emission_PM10']
        result.at[idx, 'total_emission_NOX'] = energy_sum['total_emission_NOX']
        result.at[idx, 'total_diesel_consumption_ICE_vol'] = energy_sum['total_diesel_consumption_ICE_vol']
        result.at[idx, 'vessel'] = vessel
        result.at[idx, 'remaining_route'] = remaining_route
    # TODO: replace with a timeout on_pass_edge 
    if not berth_available:    
        # add waiting time to direct routes
        result.loc[result['name'] == 'direct', 'duration']  += 10

    return result

    