# ## Installed packages
# package(s) related to time, space and id
import datetime, time
import os
import io
import functools
import logging
import pickle
import random
import math

# package(s) related to the simulation
import enum
import simpy
import scipy as sc
import math
import networkx as nx  
import numpy as np
import pandas as pd
import re
import yaml as yaml
import time
import bisect
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pytest

from dataclasses import dataclass
from enum import Enum
from scipy import interpolate
from scipy.signal import correlate
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# OpenTNSim
from opentnsim import core
from opentnsim import plot
from opentnsim import model

# spatial libraries 
import shapely.geometry
import shapely.wkt
import pyproj
import shapely.geometry
import folium
import datetime
import time as timepy

# package(s) for data handling
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# define the coorinate systemb
geod = pyproj.Geod(ellps="WGS84")

location_vessel_database = "Vessels/richtlijnen-vaarwegen-2017.csv"

Node = type('Site', (core.Identifiable, core.Log, core.Locatable, core.HasResource), {})
nodes = []
path = []
coords = []

coords.append([0,0])#node_2
lon,lat,_ = pyproj.Geod(ellps='WGS84').fwd(0,0,0,0.001)
coords.append([lon,lat])#node_0
lon,lat,_ = pyproj.Geod(ellps='WGS84').fwd(0,0,270,0.001)
coords.append([lon,lat])#node_1
lon,lat,_ = pyproj.Geod(ellps='WGS84').fwd(0,0,90,1000)
coords.append([lon,lat])#node_3
lon,lat,_ = pyproj.Geod(ellps='WGS84').fwd(lon,lat,90,25000)
coords.append([lon,lat])#node_4
lon,lat,_ = pyproj.Geod(ellps='WGS84').fwd(lon,lat,90,1000)
coords.append([lon,lat])#node_5
lon,lat,_ = pyproj.Geod(ellps='WGS84').fwd(lon,lat,90,1000)
coords.append([lon,lat])#node_6
lon,lat,_ = pyproj.Geod(ellps='WGS84').fwd(lon,lat,90,5*300)
coords.append([lon,lat])#node_7

node_names = ['Node 2','Node 0','Node 1','Node 3','Node 4','Node 5','Node 6','Node 7']
for d in range(len(coords)):
    data_node = {"env": [],
                 "name": node_names[d],
                 "geometry": shapely.geometry.Point(coords[d][0], coords[d][1])}
    node = Node(**data_node)
    nodes.append(node)
    
path.append([nodes[0],nodes[1]]) 
path.append([nodes[1],nodes[0]])
path.append([nodes[0],nodes[2]]) 
path.append([nodes[2],nodes[0]])
path.append([nodes[0],nodes[3]]) 
path.append([nodes[3],nodes[0]])
path.append([nodes[3],nodes[4]]) 
path.append([nodes[4],nodes[3]])
path.append([nodes[4],nodes[5]]) 
path.append([nodes[5],nodes[4]])
path.append([nodes[5],nodes[6]]) 
path.append([nodes[6],nodes[5]])
path.append([nodes[6],nodes[7]]) 
path.append([nodes[7],nodes[6]])

FG = nx.DiGraph()

positions = {}
for node in nodes:
    positions[node.name] = (node.geometry.x, node.geometry.y)
    FG.add_node(node.name, geometry = node.geometry)

for edge in path:
    FG.add_edge(edge[0].name, edge[1].name, weight = 1, Info = {})
    
class window_method(Enum):
    critical_cross_current = 'Critical cross-current'
    point_based = 'Point-based'

class vessel_characteristics(Enum):
    min_ge_Length = ['minLength','>=']
    min_gt_Length = ['minLength','>']
    max_le_Length = ['maxLength','<=']
    max_lt_Length = ['maxLength','<']
    min_ge_Draught = ['minDraught','>=']
    min_gt_Draught = ['minDraught','>']
    max_le_Draught = ['maxDraught','<=']
    max_lt_Draught = ['maxDraught','<']
    min_ge_Beam = ['minBeam','>=']
    min_gt_Beam = ['minBeam','>']
    max_le_Beam = ['maxBeam','<=']
    max_lt_Beam = ['maxBeam','<']
    min_ge_UKC = ['minUKC','>=']
    min_gt_UKC = ['minUKC','>']
    max_le_UKC = ['maxUKC','<=']
    max_lt_UKC = ['maxUKC','<']
    Type = ['Type','==']

class vessel_direction(Enum):
    inbound = 'inbound'
    outbound = 'outbound'

class vessel_type(Enum):
    GeneralCargo = 'GeneralCargo'
    LiquidBulk = 'LiquidBulk'
    Container = 'Container'
    DryBulk = 'DryBulk'
    MultiPurpose = 'MultiPurpose'
    Reefer = 'Reefer'
    RoRo = 'RoRo'
    Barge = 'Barge'

class accessibility(Enum):
    non_accessible = 0
    accessible = -1
    slack_water = 'min'

class tidal_period(Enum):
    Flood = 'Flood'
    Ebb = 'Ebb'

class current_velocity_type(Enum):
    CurrentVelocity = 'Current velocity'
    LongitudinalCurrent = 'Longitudinal current'
    CrossCurrent = 'Cross-current'

@dataclass
class vessel_specifications:
    vessel_characteristics: dict #{item of vessel_characteristics class: user-defined value,...}
    vessel_method: str #string containing the operators between the vessel characteristics (symbolized by x): e.g. '(x and x) or x'
    vessel_direction: str #item of vessel_direction class

    def characteristic_dicts(self):
        characteristic_dicts = {}
        for characteristic in self.vessel_characteristics:
            characteristic_dict = {characteristic.value[0]: [characteristic.value[1],self.vessel_characteristics[characteristic]]}
            characteristic_dicts = characteristic_dicts | characteristic_dict
        return characteristic_dicts

@dataclass
class window_specifications:
    window_method: str #item of window_method class
    current_velocity_values: dict #{tidal_period.Flood.value: user-defined value or item from accessibility class,...}
    current_velocity_ranges: dict = dict #if window_method is point-based: {tidal_period.Ebb.value: user-defined value,...}

@dataclass
class vtw_window_specifications:
    ukc_s: dict #{tidal_period.Flood.value: user-defined value or item from accessibility class,...}
    ukc_p: dict #{tidal_period.Flood.value: user-defined value or item from accessibility class,...}
    fwa: dict #{tidal_period.Flood.value: user-defined value or item from accessibility class,...}

@dataclass
class vertical_tidal_window_input:
    vessel_specifications: vessel_specifications #class
    window_specifications: window_specifications #class     

@dataclass
class horizontal_tidal_window_input:
    vessel_specifications: vessel_specifications #class
    window_specifications: window_specifications #class     
    condition: dict #{'Origin':node, 'Destination': node}
    data: list #Calculated input: [node,]
    
runs = [['range','range','If'],
        ['range','range','IIf'],
        ['range','range','IIIf'],
        ['range','range','Ie'],
        ['range','range','IIe'],
        ['range','range','IIIe'],        
        ['range','accessible','If'],
        ['range','accessible','IIf'],
        ['range','accessible','IIIf'],
        ['range','accessible','Ie'],
        ['range','accessible','IIe'],
        ['range','accessible','IIIe'],        
        ['range','non-accessible','If'],
        ['range','non-accessible','IIf'],
        ['range','non-accessible','IIIf'],
        ['range','non-accessible','Ie'],
        ['range','non-accessible','IIe'],
        ['range','non-accessible','IIIe'],        
        ['range','slack','If'],
        ['range','slack','IIf'],
        ['range','slack','IIIf'],
        ['range','slack','Ie'],
        ['range','slack','IIe'],
        ['range','slack','IIIe'],   
        ['point','slack','If'],
        ['point','slack','IIf'],
        ['point','slack','IIIf'],
        ['point','slack','Ie'],
        ['point','slack','IIe'],
        ['point','slack','IIIe'],  
        ['point','accessible','If'],
        ['point','accessible','IIf'],
        ['point','accessible','IIIf'],
        ['point','accessible','Ie'],
        ['point','accessible','IIe'],
        ['point','accessible','IIIe'],       
        ['point','non-accessible','If'],
        ['point','non-accessible','IIf'],
        ['point','non-accessible','IIIf'],
        ['point','non-accessible','Ie'],
        ['point','non-accessible','IIe'],
        ['point','non-accessible','IIIe'],      
        ['accessible','non-accessible','If'],
        ['accessible','non-accessible','IIf'],
        ['accessible','non-accessible','IIIf'],
        ['accessible','non-accessible','Ie'],
        ['accessible','non-accessible','IIe'],
        ['accessible','non-accessible','IIIe'],      
        ['accessible','accessible','If'],
        ['accessible','accessible','IIf'],
        ['accessible','accessible','IIIf'],
        ['accessible','accessible','Ie'],
        ['accessible','accessible','IIe'],
        ['accessible','accessible','IIIe'],      
        ['accessible','slack','If'],
        ['accessible','slack','IIf'],
        ['accessible','slack','IIIf'],
        ['accessible','slack','Ie'],
        ['accessible','slack','IIe'],
        ['accessible','slack','IIIe'],        
        ['non-accessible','slack','If'],
        ['non-accessible','slack','IIf'],
        ['non-accessible','slack','IIIf'],
        ['non-accessible','slack','Ie'],
        ['non-accessible','slack','IIe'],
        ['non-accessible','slack','IIIe'],        
        ['non-accessible','non-accessible','If'],
        ['non-accessible','non-accessible','IIf'],
        ['non-accessible','non-accessible','IIIf'],
        ['non-accessible','non-accessible','Ie'],
        ['non-accessible','non-accessible','IIe'],
        ['non-accessible','non-accessible','IIIe'],
        ['slack','slack','If'],
        ['slack','slack','IIf'],
        ['slack','slack','IIIf'],
        ['slack','slack','Ie'],
        ['slack','slack','IIe'],
        ['slack','slack','IIIe']]

#@pytest.fixture
def test(flood_restriction,ebb_restriction,initial_tidal_period):
    if initial_tidal_period == 'If':
        delay = (24/120)*12.5*60*60
    elif initial_tidal_period == 'IIf':
        delay = (1/4)*12.5*60*60
    elif initial_tidal_period == 'IIIf':
        delay = (1/12)*12.5*60*60
    elif initial_tidal_period == 'Ie':
        delay = (84/120)*12.5*60*60
    elif initial_tidal_period == 'IIe':
        delay = (3/4)*12.5*60*60
    elif initial_tidal_period == 'IIIe':
        delay = (7/12)*12.5*60*60
        
    if flood_restriction == 'range':
        flood_value = 0.5
        flood_range = 0.25
    elif flood_restriction == 'point':  
        flood_value = 0.5
        flood_range = 0
    elif flood_restriction == 'accessible':  
        flood_value = -1
        flood_range = 0
    elif flood_restriction == 'non-accessible':  
        flood_value = 0
        flood_range = 0
    elif flood_restriction == 'slack':  
        flood_value = 'min'
        flood_range = 0.25
    if ebb_restriction == 'range':
        ebb_value = 0.5
        ebb_range = 0.25
    elif ebb_restriction == 'point':  
        ebb_value = 0.5
        ebb_range = 0
    elif ebb_restriction == 'accessible':  
        ebb_value = -1
        ebb_range = 0
    elif ebb_restriction == 'non-accessible':  
        ebb_value = 0
        ebb_range = 0
    elif ebb_restriction == 'slack':  
        ebb_value = 'min'
        ebb_range = 0.25

    for edge in enumerate(FG.edges):
        if 'Terminal' in FG.edges[edge[1]]:
            FG.edges[edge[1][1],edge[1][0]]['Terminal'] = FG.edges[edge[1]]['Terminal']

    def current_direction_calculator(delay):
        current_directions = []
        tidal_period = 'Flood'
        tidal_period_count = 0
        for t in range(len(time)):
            if t == 0:
                if tidal_period == 'Flood':
                    current_direction = 90/360*2*np.pi
                elif tidal_period == 'Ebb':
                    current_direction = 270/360*2*np.pi
                current_directions.append(current_direction)
                continue

            if tidal_period_count == 0 and (time[t])%(0.25*12.5*60*60+delay+0.00001) < (time[t-1])%(0.25*12.5*60*60+delay+0.00001):
                if tidal_period == 'Flood':
                    tidal_period = 'Ebb'
                elif tidal_period == 'Ebb':
                    tidal_period = 'Flood'
                tidal_period_count += 1

            elif tidal_period_count >= 1 and (time[t]-0.25*12.5*60*60-delay)%(0.5*12.5*60*60) < (time[t-1]-0.25*12.5*60*60-delay)%(0.5*12.5*60*60):
                if tidal_period == 'Flood':
                    tidal_period = 'Ebb'
                elif tidal_period == 'Ebb':
                    tidal_period = 'Flood'
                tidal_period_count += 1

            if tidal_period == 'Flood':
                current_direction = 90/360*2*np.pi
            elif tidal_period == 'Ebb':
                current_direction = 270/360*2*np.pi    
            current_directions.append(current_direction)
        return current_directions

    simulation_start = datetime.datetime(2022,1,1)
    duration = 5*12.5*60*60
    time = np.arange(0,duration,600)
    MBLs = [50,50,50,15,15,50,50,50]
    widths = [1000,1000,1000,1000,1000,1000,1000,1000]
    delay = delay

    water_level_node_0 = water_level_node_1 = water_level_node_2 = np.sin(time/(12.5*60*60)*2*np.pi+delay/(12.5*60*60)*2*np.pi)
    water_level_node_3 = np.sin(time/(12.5*60*60)*2*np.pi-(1000/10)/(12.5*60*60)*2*np.pi+delay/(12.5*60*60)*2*np.pi)
    water_level_node_4 = np.sin(time/(12.5*60*60)*2*np.pi-(26000/10)/(12.5*60*60)*2*np.pi+delay/(12.5*60*60)*2*np.pi)
    water_level_node_5 = np.sin(time/(12.5*60*60)*2*np.pi-(27000/10)/(12.5*60*60)*2*np.pi+delay/(12.5*60*60)*2*np.pi)
    water_level_node_6 = np.sin(time/(12.5*60*60)*2*np.pi-(28000/10)/(12.5*60*60)*2*np.pi+delay/(12.5*60*60)*2*np.pi)
    water_level_node_7 = np.sin(time/(12.5*60*60)*2*np.pi-(29500/10)/(12.5*60*60)*2*np.pi+delay/(12.5*60*60)*2*np.pi)
    water_levels = [water_level_node_0,water_level_node_1,water_level_node_2,water_level_node_3,water_level_node_4,water_level_node_5,water_level_node_6,water_level_node_7]

    current_velocity_node_0 = current_velocity_node_1 = current_velocity_node_2 = abs(np.cos(time/(12.5*60*60)*2*np.pi+delay/(12.5*60*60)*2*np.pi))
    current_velocity_node_3 = abs(np.cos(time/(12.5*60*60)*2*np.pi-(1000/10)/(12.5*60*60)*2*np.pi+delay/(12.5*60*60)*2*np.pi))
    current_velocity_node_4 = abs(np.cos(time/(12.5*60*60)*2*np.pi-(26000/10)/(12.5*60*60)*2*np.pi+delay/(12.5*60*60)*2*np.pi))
    current_velocity_node_5 = abs(np.cos(time/(12.5*60*60)*2*np.pi-(27000/10)/(12.5*60*60)*2*np.pi+delay/(12.5*60*60)*2*np.pi))
    current_velocity_node_6 = abs(np.cos(time/(12.5*60*60)*2*np.pi-(28000/10)/(12.5*60*60)*2*np.pi+delay/(12.5*60*60)*2*np.pi))
    current_velocity_node_7 = abs(np.cos(time/(12.5*60*60)*2*np.pi-(29500/10)/(12.5*60*60)*2*np.pi+delay/(12.5*60*60)*2*np.pi))
    current_velocities = [current_velocity_node_0,current_velocity_node_1,current_velocity_node_2,current_velocity_node_3,current_velocity_node_4,current_velocity_node_5,current_velocity_node_6,current_velocity_node_7]

    current_direction_node_0 = current_direction_node_1 = current_direction_node_2 = current_direction_calculator(0-delay)
    current_direction_node_3 = current_direction_calculator(1000/10)
    current_direction_node_4 = current_direction_calculator(26000/10)
    current_direction_node_5 = current_direction_calculator(27000/10)
    current_direction_node_6 = current_direction_calculator(28000/10)
    current_direction_node_7 = current_direction_calculator(29500/10)
    current_directions = [current_direction_node_0,current_direction_node_1,current_direction_node_2,current_direction_node_3,current_direction_node_4,current_direction_node_5,current_direction_node_6,current_direction_node_7]

    #define variables
    depth = [[],[]]
    width = [[],[]]
    MBL = [[],[]]
    water_level= [[],[]]
    current_velocity = [[],[]]
    current_direction = [[],[]]

    # depth according to MBL values, and waterway navigational width obtained from measuring on HavenKaart
    MBL[1] = MBLs
    depth[1] = MBL[1]
    width[1] = widths

    # load water level, velocity magnitude and direction time series to each node
    for nodes in enumerate(FG.nodes):
        MBL[0].append(FG.nodes[nodes[1]]['geometry'])
        width[0].append(FG.nodes[nodes[1]]['geometry'])
        depth[0].append((FG.nodes[nodes[1]]['geometry']))
        water_level[0].append((FG.nodes[nodes[1]]['geometry']))
        water_level[1].append([[],[]])
        current_velocity[0].append((FG.nodes[nodes[1]]['geometry']))
        current_velocity[1].append([[],[]])
        current_direction[0].append((FG.nodes[nodes[1]]['geometry']))
        current_direction[1].append([[],[]])

    for col in enumerate(water_levels): #load water level
        water_level[1][col[0]][0]=[x+simulation_start.timestamp() for x in time]
        water_level[1][col[0]][1]=list(col[1])

    for col in enumerate(current_velocities): #load velocity magnitude
        current_velocity[1][col[0]][0]=[x+simulation_start.timestamp() for x in time]
        current_velocity[1][col[0]][1]=list(col[1])

    for col in enumerate(current_directions): #load velocity direction
        current_direction[1][col[0]][0]=[x+simulation_start.timestamp() for x in time]
        current_direction[1][col[0]][1]=list(col[1])

    core.NetworkProperties.append_data_to_nodes(FG,width,depth,MBL,water_level,current_velocity,current_direction)
    knots = 0.51444444444444

    for node in FG.nodes:
        vertical_tidal_window_inputs = []

        vessel_specification = vessel_specifications({vessel_characteristics.min_ge_Draught: 0},
                                                      'x',vessel_direction.inbound.value)

        window_specification = vtw_window_specifications({'ukc_s': 0.0},
                                                         {'ukc_p': 0.075},
                                                         {'fwa': 0.025})

        vertical_tidal_window_inputs.append(vertical_tidal_window_input(vessel_specifications = vessel_specification,
                                                                        window_specifications = window_specification))

        vessel_specification = vessel_specifications({vessel_characteristics.min_ge_Draught: 0},
                                                      'x',vessel_direction.outbound.value)

        window_specification = vtw_window_specifications({'ukc_s': 0.0},
                                                         {'ukc_p': 0.075},
                                                         {'fwa': 0.025})

        vertical_tidal_window_inputs.append(vertical_tidal_window_input(vessel_specifications = vessel_specification,
                                                                        window_specifications = window_specification))

        core.NetworkProperties.append_vertical_tidal_restriction_to_network(FG,node,vertical_tidal_window_inputs)


    horizontal_tidal_window_inputs = []

    #Inbound_Vessels_Condition1
    vessel_specification = vessel_specifications({vessel_characteristics.min_ge_Draught: 0},
                                                  'x',vessel_direction.inbound.value)

    window_specification = window_specifications(window_method.point_based.value,
                                                 {tidal_period.Flood.value: flood_value,tidal_period.Ebb.value: ebb_value},
                                                 {tidal_period.Flood.value: flood_range,tidal_period.Ebb.value: ebb_range})

    horizontal_tidal_window_inputs.append(horizontal_tidal_window_input(vessel_specifications = vessel_specification,
                                                                        window_specifications = window_specification,
                                                                        condition = {'Origin': 'Node 3', 'Destination': 'Node 5'},
                                                                        data = ['Node 4', current_velocity_type.CurrentVelocity.value]))

    #Outbound_Vessels_Condition1
    vessel_specification = vessel_specifications({vessel_characteristics.min_ge_Draught: 0},
                                                  'x',vessel_direction.outbound.value)

    window_specification = window_specifications(window_method.point_based.value,
                                                 {tidal_period.Flood.value: accessibility.accessible.value,tidal_period.Ebb.value: ebb_value},
                                                 {tidal_period.Flood.value: flood_range,tidal_period.Ebb.value: ebb_range})

    horizontal_tidal_window_inputs.append(horizontal_tidal_window_input(vessel_specifications = vessel_specification,
                                                                        window_specifications = window_specification,
                                                                        condition = {'Origin': 'Node 5', 'Destination': 'Node 3'},
                                                                        data = ['Node 4', current_velocity_type.CurrentVelocity.value]))

    core.NetworkProperties.append_horizontal_tidal_restriction_to_network(FG,'Node 4',horizontal_tidal_window_inputs)

    class Environment:
        def __init__(self,start_time,network):
            self.now = start_time
            self.FG = network

    class Vessel:
        def __init__(self,start_time,T,ukc,typ,v,L,B,mccur,mwt,bound,network,start_node,end_node):
            self.name = 'Tanker'
            self.env = Environment(start_time,network)
            self.T_f = T
            self.L = L
            self.v = v
            self.B = B
            self.ukc = ukc
            self.metadata = {}
            self.metadata['ukc'] = ukc
            self.metadata['max_cross_current'] = mccur
            self.metadata['max_waiting_time'] = mwt
            self.bound = bound
            self.type = typ
            self.route = nx.dijkstra_path(self.env.FG, start_node, end_node)

    vessel = Vessel(start_time = simulation_start.timestamp(),
                        T = 10,
                        L = 180,
                        v = 4.5,
                        B = 27,
                        ukc = 0,
                        mccur = 0,
                        mwt = 48*60*60,
                        bound = 'inbound',
                        typ = 'Handysize',
                        network = FG,
                        start_node = 'Node 1',
                        end_node = 'Node 7')

    times_tidal_window_simulated = core.VesselTrafficService.provide_sail_in_times_tidal_window(vessel,
                                                                     route =vessel.route,
                                                                     plot=False)

    #times = [t[0] for t in times_horizontal_tidal_window if t[0] >= simulation_start.timestamp() and t[0] <= simulation_start.timestamp()+vessel.metadata['max_waiting_time']]

    def root_calculation(critcur):   
        def intp(critcur):
            intp = sc.interpolate.CubicSpline(FG.nodes['Node 4']['Info']['Horizontal tidal restriction']['Data']['Node 3'][0],
                                              [y-critcur for y in FG.nodes['Node 4']['Info']['Horizontal tidal restriction']['Data']['Node 3'][1]])
            return intp.roots()

        roots = [root for root in intp(critcur) if root >= simulation_start.timestamp() and root <= simulation_start.timestamp()+vessel.metadata['max_waiting_time']]

        return roots

    #@pytest.fixture
    def tidal_window_calculator(flood,ebb,typ):    
        roots_up = root_calculation(0.5*1.25)
        roots_normal = root_calculation(0.5)
        roots_down = root_calculation(0.5*0.75)
        roots_min = root_calculation(0.25)
        roots_tide = [t[0] for t in FG.nodes['Node 4']['Info']['Tidal periods'] if t[0] < simulation_start.timestamp()+vessel.metadata['max_waiting_time']]
        root_sim = simulation_start.timestamp()
        times_tidal_window_calculated = []
        if flood == 'range':
            if ebb == 'range':
                if typ == 'If':
                    for root in enumerate(roots_up):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIf':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_up):
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIf':
                    for root in enumerate(roots_up):
                        if root[0] == 0:
                            times_tidal_window_calculated.append([root_sim, 'Stop'])
                            continue
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                if typ == 'Ie':
                    for root in enumerate(roots_up):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIe':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_up):
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIe':
                    for root in enumerate(roots_up):
                        if root[0] == 0:
                            times_tidal_window_calculated.append([root_sim, 'Stop'])
                            continue
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
            elif ebb == 'accessible':
                if typ == 'If':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0] == 0:
                            continue
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_up):
                        if root[0] == 0:
                            continue
                        if (root[0])%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if root[0] == 0:
                            continue
                        if (root[0])%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIf':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0] == 0:
                            continue
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_up):
                        if (root[0]-3)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0]-3)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIf':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_up):
                        if root[0] == 0:
                            continue
                        if (root[0])%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0])%4 == 0 or root[0] == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'Ie':
                    for root in enumerate(roots_tide):
                        if root[0] == 0:
                            continue
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_up):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIe':
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_up):
                        if (root[0]-1)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0]-1)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIe':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_up):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
            elif ebb == 'non-accessible':
                if typ == 'If':
                    for root in enumerate(roots_up):
                        if root[0] == 0:
                            continue
                        if (root[0])%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if root[0] == 0:
                            continue
                        if (root[0])%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIf':
                    for root in enumerate(roots_up):
                        if (root[0]-3)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0]-3)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIf':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_up):
                        if root[0] == 0:
                            continue
                        if (root[0])%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0])%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'Ie':
                    for root in enumerate(roots_up):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIe':
                    for root in enumerate(roots_up):
                        if (root[0]-1)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0]-1)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIe':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_up):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
            elif ebb == 'slack':
                if typ == 'If':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_min):
                        if (root[0]%4 == 0 or (root[0]-1)%4 == 0) and root[0] != 1:
                            continue
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_up):
                        if root[0] == 0:
                            continue
                        if (root[0])%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if root[0] == 0:
                            continue
                        if (root[0])%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIf':
                    for root in enumerate(roots_min):
                        if root[0]%4 == 0 or (root[0]-1)%4 == 0:
                            continue
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_up):
                        if (root[0]-3)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0]-3)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIf':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_min):
                        if (root[0])%4 == 0 or (root[0]-1)%4 == 0:
                            continue
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_up):
                        if (root[0])%4 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0])%4 == 0 or root[0] == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'Ie':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_min):
                        if root[0] == 0:
                            continue
                        if (root[0]-2)%4 == 0 or (root[0]-3)%4 == 0:
                            continue
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_up):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIe':
                    for root in enumerate(roots_min):
                        if root[0] == 0 or root[0] == 1:
                            continue
                        if (root[0]-2)%4 == 0 or (root[0]-3)%4 == 0:
                            continue
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_up):
                        if (root[0]-1)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0]-1)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIe':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_min):
                        if (root[0]-2)%4 == 0 or (root[0]-3)%4 == 0:
                            continue
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_up):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_down):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
        elif flood == 'point':
            if ebb == 'accessible':
                if typ == 'If':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_normal):    
                        if root[0]%4 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
                elif typ == 'IIf':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        else:
                            if root[0] == 1:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_normal):    
                        if (root[0]-3)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
                elif typ == 'IIIf':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_normal):    
                        if root[0]%4 == 0:
                            if root[0] == 0:
                                times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
                elif typ == 'Ie':
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_normal):    
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
                elif typ == 'IIe':
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_normal):    
                        if (root[0]-1)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
                elif typ == 'IIIe':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        else:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_normal):    
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
            elif ebb == 'non-accessible':
                if typ == 'If':
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_normal):    
                        if root[0]%4 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
                elif typ == 'IIf':
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_normal):    
                        if (root[0]-3)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
                elif typ == 'IIIf':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_normal):    
                        if root[0]%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
                elif typ == 'Ie':
                    for root in enumerate(roots_tide):
                        if (root[0]-1)%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_normal):    
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
                elif typ == 'IIe':
                    for root in enumerate(roots_tide):
                        if (root[0]-1)%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_normal):    
                        if (root[0]-1)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
                elif typ == 'IIIe':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_tide):
                        if (root[0]-1)%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    for root in enumerate(roots_normal):    
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
            elif ebb == 'slack':
                if typ == 'If':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_min):
                        if root[0]%4 == 1 or root[0]%4 == 3:
                            if root[0] == 1:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        elif (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_normal):    
                        if root[0]%4 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()  
                elif typ == 'IIf':
                    for root in enumerate(roots_min):
                        if root[0]%4 == 1 or root[0]%4 == 3:
                            if root[0] == 1:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        elif (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_normal):    
                        if (root[0]-3)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()        
                elif typ == 'IIIf':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_min):
                        if root[0]%4 == 1 or root[0]%4 == 3:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        elif (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_normal):           
                        if root[0]%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()  
                elif typ == 'Ie':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_min):
                        if root[0] == 0:
                            continue
                        if root[0]%4 == 1 or root[0]%4 == 3:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        elif (root[0])%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_normal):    
                        if (root[0]-2)%4 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()  
                elif typ == 'IIe':
                    for root in enumerate(roots_min):
                        if root[0]%4 == 1 or root[0]%4 == 3:
                            if root[0] == 1:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        elif root[0]%4 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_normal):    
                        if (root[0]-1)%4 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
                elif typ =='IIIe':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_min):
                        if root[0]%4 == 1 or root[0]%4 == 3:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        elif root[0]%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_normal):           
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort() 
        elif flood == 'accessible':
            if ebb == 'accessible':
                if typ == 'If':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])  
                    times_tidal_window_calculated.sort()
                elif typ == 'IIf':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop']) 
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIf':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])  
                    times_tidal_window_calculated.sort()
                elif typ == 'Ie':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])  
                    times_tidal_window_calculated.sort()
                elif typ == 'IIe':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop']) 
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIe':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])  
                    times_tidal_window_calculated.sort()
            elif ebb == 'non-accessible':
                if typ == 'If':
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIf':
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIf':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'Ie':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        if root[0]%2 == 1:
                            if root[0] == 1:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIe':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        if root[0]%2 == 1:
                            if root[0] == 1:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIe':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_tide):
                        if root[0]%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                        if root[0]%2 == 1:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
            elif ebb == 'slack':
                if typ == 'If':
                    for root in enumerate(roots_min):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_tide):
                        if (root[0]-2)%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIf':
                    for root in enumerate(roots_min):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_tide):
                        if (root[0]-2)%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIf':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_min):
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_tide):
                        if (root[0]-2)%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'Ie':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_min):
                        if root[0]%4 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_tide):
                        if (root[0]-1)%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIe':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_min):
                        if root[0]%4 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_tide):
                        if (root[0]-1)%2 == 0:
                            if root[0] == 1:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIe':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_min):
                        if root[0]%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    for root in enumerate(roots_tide):
                        if (root[0]-1)%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                    times_tidal_window_calculated.sort()
        elif flood == 'non-accessible':
            if ebb == 'non-accessible':
                if typ == 'If':
                    pass
                elif typ == 'IIf':
                    pass
                elif typ == 'IIIf':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                elif typ == 'Ie':
                    pass
                elif typ == 'IIe':
                    pass
                elif typ == 'IIIe':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
            elif ebb == 'slack':
                if typ == 'If':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_min):
                        if (root[0]-3)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIf':
                    for root in enumerate(roots_min):
                        if (root[0]-3)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIf':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_min):
                        if (root[0]-3)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if (root[0]-2)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'Ie':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_min):
                        if (root[0]-1)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if (root[0]-4)%4 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIe':
                    for root in enumerate(roots_min):
                        if (root[0]-1)%4 == 0:
                            if root[0] == 1:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if (root[0]-4)%4 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIe':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_min):
                        if (root[0]-1)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if (root[0]-4)%4 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
        elif ebb == 'slack':
            if ebb == 'slack':
                if typ == 'If':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_min):
                        if (root[0]-1)%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if (root[0]-2)%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIf':
                    for root in enumerate(roots_min):
                        if (root[0]-1)%2 == 0:
                            if root[0] == 1:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if (root[0]-2)%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIf':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_min):
                        if (root[0]-1)%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if (root[0]-2)%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'Ie':
                    times_tidal_window_calculated.append([root_sim, 'Stop'])
                    for root in enumerate(roots_min):
                        if (root[0]-1)%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if (root[0]-2)%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIe':
                    for root in enumerate(roots_min):
                        if (root[0]-1)%2 == 0:
                            if root[0] == 1:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if (root[0]-2)%2 == 0:
                            if root[0] == 0:
                                continue
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
                elif typ == 'IIIe':
                    times_tidal_window_calculated.append([root_sim, 'Start'])
                    for root in enumerate(roots_min):
                        if (root[0]-1)%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Start'])
                        if (root[0]-2)%2 == 0:
                            times_tidal_window_calculated.append([root[1]-26000/4.5, 'Stop'])
                    times_tidal_window_calculated.sort()
        return times_tidal_window_calculated

    times_tidal_window_calculated = tidal_window_calculator(flood_restriction,ebb_restriction,initial_tidal_period)
    times_tidal_window_simulated = [time for time in times_tidal_window_simulated if time[0] >= simulation_start.timestamp() and time[0] <= simulation_start.timestamp()+vessel.metadata['max_waiting_time']]

    for time in enumerate(times_tidal_window_calculated):
        np.testing.assert_almost_equal(time[1][0], times_tidal_window_simulated[time[0]][0], decimal=-1, err_msg='', verbose=True)
        np.testing.assert_equal(time[1][1], times_tidal_window_simulated[time[0]][1], err_msg='', verbose=True)
    return

for run in runs:
    print(run)
    test(run[0],run[1],run[2])

