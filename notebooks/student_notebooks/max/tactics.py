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

# Maximum CO2 emission for normalization
# Maximum PM10 emission for normalization

MAX_CO2 = 3119.526428
MAX_PM10 = 2.056981 

# We can't compute the true hotel power emission
# So we'll estimate it
# We'll use the 0.6 engine order emission / duration * 2.44
WAIT_EMISSION_PM10_PER_SECOND = 0.817459 / 23.846344 * 2.44

def compute_passages(vessel):
    """compute the passages of vessel on some edges of interest"""
    log_df = pd.DataFrame(vessel.log)
    
    c_d = 'Sailing from node C to node D sub edge 0 stop'
    e_d = 'Sailing from node E to node D sub edge 0 stop'
    c_d_idx = log_df['Message'] == c_d
    e_d_idx = log_df['Message'] == e_d
    c_d_df = log_df[c_d_idx]
    e_d_df = log_df[e_d_idx]
    
    # time of arrival of first passage from c to d
    if c_d_df.shape[0] >= 1:
        c_d_t = c_d_df.iloc[0]['Timestamp'] - vessel.env.simulation_start
        c_d_t = c_d_t.total_seconds()
    else:
        c_d_t = None

    # time of arrival of first passage from e to d
    if e_d_df.shape[0] >= 1:
        e_d_t = e_d_df.iloc[0]['Timestamp'] - vessel.env.simulation_start
        e_d_t = e_d_t.total_seconds()
    else:
        e_d_t = None
    return {
        "c_d_t": c_d_t,
        "e_d_t": e_d_t
    }
        

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
    env.simulation_start = simulation_start

    # add environment to the vessel, and specify the vessels route and current location (beginning of the path)
    vessel.env = env
    vessel.route = route
    
    # specify the process that needs to be executed
    env.process(vessel.move())

    # start the simulation
    env.run()

    #Determine trip time
    start_time = simulation_start.timestamp()
        
    passages = compute_passages(vessel)
    end = vessel.log["Timestamp"][-1]

    end_time = end.timestamp()

    energycalculation = opentnsim.energy.EnergyCalculation(graph, vessel)
    energycalculation.calculate_energy_consumption()

    # create dataframe from energy calculation computation
    energy_df = pd.DataFrame.from_dict(energycalculation.energy_use)

    results =  {
        "duration": end_time-start_time, 
        "eta": end, 
        "energy_df": energy_df, 
        "vessel": vessel
    }
    results.update(passages)
    return results




def generate_route_alternatives(graph):
    """generate a list of dictionaries, where each dictionary is a route alternatives with name, waypoints and a route."""
    #Conditional toevoegen die rekening houdt met de visited nodes?
    route_alternatives = [
        {
            "name": "direct",
            "waypoints": ['A', 'E', 'H', 'L']
        },
        {
            "name": "redirect",
            "waypoints": ['A', 'F','I', 'L', 'J', 'F']
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
    engine_orders = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
        results = run_simulation(geometry=geometry, route=remaining_route, graph=graph, engine_order=engine_order)
        duration = results["duration"]
        eta = results["eta"]
        energy_df = results["energy_df"]
        vessel = results["vessel"]
        c_d_t = results["c_d_t"]
        e_d_t = results["e_d_t"]

        energy_sum = energy_df.sum(axis=0, numeric_only=True) 
        
        result.at[idx, 'duration'] = duration
        result.at[idx, 'total_energy'] = energy_sum['total_energy']
        result.at[idx, 'total_emission_CO2'] = energy_sum['total_emission_CO2']
        result.at[idx, 'total_emission_PM10'] = energy_sum['total_emission_PM10']
        result.at[idx, 'total_emission_NOX'] = energy_sum['total_emission_NOX']
        result.at[idx, 'total_diesel_consumption_ICE_vol'] = energy_sum['total_diesel_consumption_ICE_vol']
        result.at[idx, 'vessel'] = vessel
        result.at[idx, 'remaining_route'] = remaining_route
        result.at[idx, "c_d_t"] = c_d_t
        result.at[idx, "e_d_t"] = e_d_t
    # TODO: replace with a timeout on_pass_edge 
    if not berth_available:    
        # add waiting time to direct routes dependent on engine order
        direct_idx = result["name"] == "direct"
        redirect_idx = result["name"] == "redirect"
        wait_time = 20 - result.loc[direct_idx, 'c_d_t']
        result.loc[direct_idx, "wait_time"] = wait_time
        result.loc[redirect_idx, "wait_time"] = 0
    else:
        result['wait_time'] = 0
    result['wait_emission_pm10'] = result['wait_time'] * WAIT_EMISSION_PM10_PER_SECOND
    # compute the sorting variables
    result["strategy_duration"] = result["duration"] + result["wait_time"]
    result["strategy_pm10"] = result['total_emission_PM10'] + result['wait_emission_pm10']
    result["strategy_fuel"] = result["total_diesel_consumption_ICE_vol"]
    # evenly weigh co2 and pm10 emissions (normalize and then mean)
    result["strategy_mca"] = ((result["total_emission_CO2"] / MAX_CO2) + (result["strategy_pm10"] / MAX_PM10)) / 2
    
        
        
    
    return result
    

    