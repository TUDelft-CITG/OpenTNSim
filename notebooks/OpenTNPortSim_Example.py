#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import simpy
import scipy as sc
import math
import networkx as nx  
import numpy as np
import pandas as pd
import re
import yaml as yaml
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

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

# package(s) for data handling
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# define the coorinate system
geod = pyproj.Geod(ellps="WGS84")

location_vessel_database = "Vessels/richtlijnen-vaarwegen-2017.csv"


# In[2]:


Node = type('Site', (core.Identifiable, core.Log, core.Locatable, core.HasResource), {})
nodes = []
path = []

coords = []
coords.append([0.00,0.00])

for i in range(4):
    coords.append([geod.fwd(coords[i][0],coords[i][1],90,5000)[0],geod.fwd(coords[i][0],coords[i][1],90,5000)[1]])
    
coords.append([geod.fwd(coords[2][0],coords[2][1],180,2000)[0],geod.fwd(coords[2][0],coords[2][1],180,2000)[1]]) #5
coords.append([geod.fwd(coords[5][0],coords[5][1],180,2000)[0],geod.fwd(coords[5][0],coords[5][1],180,2000)[1]]) #6

coords.append([geod.fwd(coords[4][0],coords[4][1],180,2000)[0],geod.fwd(coords[4][0],coords[4][1],180,2000)[1]]) #7
coords.append([geod.fwd(coords[7][0],coords[7][1],180,2000)[0],geod.fwd(coords[7][0],coords[7][1],180,2000)[1]]) #8

coords.append([geod.fwd(coords[5][0],coords[5][1],270,3000)[0],geod.fwd(coords[5][0],coords[5][1],270,3000)[1]]) #9
coords.append([geod.fwd(coords[6][0],coords[6][1],270,3000)[0],geod.fwd(coords[6][0],coords[6][1],270,3000)[1]]) #10

coords.append([geod.fwd(coords[7][0],coords[7][1],270,3000)[0],geod.fwd(coords[7][0],coords[7][1],270,3000)[1]]) #11
coords.append([geod.fwd(coords[8][0],coords[8][1],270,3000)[0],geod.fwd(coords[8][0],coords[8][1],270,3000)[1]]) #12

coords.append([geod.fwd(coords[6][0],coords[6][1],180,2000)[0],geod.fwd(coords[6][0],coords[6][1],180,2000)[1]]) #13
coords.append([geod.fwd(coords[8][0],coords[8][1],180,2000)[0],geod.fwd(coords[8][0],coords[8][1],180,2000)[1]]) #14
coords.append([geod.fwd(coords[13][0],coords[13][1],90,5000)[0],geod.fwd(coords[13][0],coords[13][1],90,5000)[1]]) #15
coords.append([geod.fwd(coords[14][0],coords[14][1],90,5000)[0],geod.fwd(coords[14][0],coords[14][1],90,5000)[1]]) #16
coords.append([geod.fwd(coords[4][0],coords[4][1],90,5000)[0],geod.fwd(coords[4][0],coords[4][1],90,5000)[1]]) #17

coords.append([geod.fwd(coords[1][0],coords[1][1],0,2000)[0],geod.fwd(coords[1][0],coords[1][1],0,2000)[1]]) #18

for d in range(len(coords)):
    data_node = {"env": [],
                 "name": "Node " + str(d+1),
                 "geometry": shapely.geometry.Point(coords[d][0], coords[d][1])}
    node = Node(**data_node)
    nodes.append(node)

for i in range(4):
    path.append([nodes[i],nodes[i+1]]) 
    path.append([nodes[i+1],nodes[i]])

path.append([nodes[2],nodes[5]])     
path.append([nodes[5],nodes[2]]) 
path.append([nodes[5],nodes[6]])     
path.append([nodes[6],nodes[5]]) 
path.append([nodes[4],nodes[7]])     
path.append([nodes[7],nodes[4]]) 
path.append([nodes[7],nodes[8]])     
path.append([nodes[8],nodes[7]]) 
path.append([nodes[5],nodes[9]])     
path.append([nodes[9],nodes[5]]) 
path.append([nodes[6],nodes[10]])     
path.append([nodes[10],nodes[6]]) 
path.append([nodes[7],nodes[11]])     
path.append([nodes[11],nodes[7]]) 
path.append([nodes[8],nodes[12]])     
path.append([nodes[12],nodes[8]]) 
path.append([nodes[6],nodes[13]])     
path.append([nodes[13],nodes[6]]) 
path.append([nodes[8],nodes[14]])     
path.append([nodes[14],nodes[8]]) 
path.append([nodes[14],nodes[15]])     
path.append([nodes[15],nodes[14]]) 
path.append([nodes[13],nodes[15]])     
path.append([nodes[15],nodes[13]]) 
path.append([nodes[14],nodes[16]])     
path.append([nodes[16],nodes[14]]) 
path.append([nodes[4],nodes[17]])     
path.append([nodes[17],nodes[4]]) 
path.append([nodes[17],nodes[4]]) 
path.append([nodes[17],nodes[4]]) 
path.append([nodes[1],nodes[18]]) 
path.append([nodes[18],nodes[1]]) 
    
FG = nx.DiGraph()

positions = {}
for node in nodes:
    positions[node.name] = (node.geometry.x, node.geometry.y)
    FG.add_node(node.name, geometry = node.geometry)

for edge in path:
    FG.add_edge(edge[0].name, edge[1].name, weight = 1)

fig, ax = plt.subplots(figsize=(10, 10))
nx.draw(FG, positions)
plt.axis('equal')
plt.show()


# In[3]:


simulation_start = datetime.datetime.now()
sim = model.Simulation(simulation_start,FG)
env = sim.environment
duration = 400000


# In[4]:


env.FG = FG

#vessel_generator_1 = core.VesselGenerators.sea_going_vessels_generator()


#vessel_generator_2 = core.VesselGenerators.barges_generator()


#vessel_generator_3 = core.VesselGenerators.barges_generator()


#turning_basin_1 = core.TurningBasins.is_turning_basin()

origin_1 = core.IsOrigin(env = env, name = 'Origin 1')

anchorage_1 = core.IsAnchorage(env = env, name = 'Anchorage 1', node = 'Node 19', typ = 'sea_going_vessels')

terminal_1 = core.IsTerminal(env = env, name = 'Container Terminal 1',length = 1200, node_start = 'Node 6', node_end = 'Node 10')
terminal_2 = core.IsTerminal(env = env, name = 'Container Terminal 2',length = 1200, node_start = 'Node 7', node_end = 'Node 11')
terminal_3 = core.IsTerminal(env = env, name = 'Container Terminal 3',length = 1200, node_start = 'Node 8', node_end = 'Node 12')
terminal_4 = core.IsTerminal(env = env, name = 'Container Terminal 4',length = 1200, node_start = 'Node 9', node_end = 'Node 13')


# In[5]:


FG.nodes["Node 1"]["Origin"] = [origin_1]

FG.nodes["Node 19"]["Anchorage"] = [anchorage_1]

#FG.nodes["Node 3"]["Turning Basin"] = [turning_basin_1]

#FG.nodes["Node 5"]["Turning Basin"] = [turning_basin_1]

#FG.nodes["Node 6"]["Turning Basin"] = [turning_basin_1]

#FG.nodes["Node 7"]["Turning Basin"] = [turning_basin_1]

#FG.nodes["Node 8"]["Turning Basin"] = [turning_basin_1]

#FG.nodes["Node 9"]["Turning Basin"] = [turning_basin_1]

FG.nodes["Node 1"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 1"]["Junction"].name = ['waterway_access']
FG.nodes["Node 1"]["Junction"].type = ['two-way_traffic']
FG.nodes["Node 19"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 19"]["Junction"].name = ['anchorage_access']
FG.nodes["Node 19"]["Junction"].type = ['two-way_traffic']
FG.nodes["Node 18"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 18"]["Junction"].name = ['waterway_access']
FG.nodes["Node 18"]["Junction"].type = ['two-way_traffic']
FG.nodes["Node 17"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 17"]["Junction"].name = ['waterway_access']
FG.nodes["Node 17"]["Junction"].type = ['two-way_traffic']
FG.nodes["Node 10"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 10"]["Junction"].name = ['harbour_basin_access']
FG.nodes["Node 10"]["Junction"].type = ['one-way_traffic']
FG.nodes["Node 11"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 11"]["Junction"].name = ['harbour_basin_access']
FG.nodes["Node 11"]["Junction"].type = ['one-way_traffic']
FG.nodes["Node 12"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 12"]["Junction"].name = ['harbour_basin_access']
FG.nodes["Node 12"]["Junction"].type = ['one-way_traffic']
FG.nodes["Node 13"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 13"]["Junction"].name = ['harbour_basin_access']
FG.nodes["Node 13"]["Junction"].type = ['one-way_traffic']

FG.nodes["Node 2"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 2"]["Junction"].name = ['waterway_access','waterway_access','anchorage_access']
FG.nodes["Node 2"]["Junction"].type = ['two-way_traffic','two-way_traffic','two-way_traffic']
FG.nodes["Node 6"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 6"]["Junction"].name = ['waterway_access','waterway_access','harbour_basin_access']
FG.nodes["Node 6"]["Junction"].type = ['two-way_traffic','two-way_traffic','one-way_traffic']
FG.nodes["Node 7"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 7"]["Junction"].name = ['waterway_access','harbour_basin_access','waterway_access']
FG.nodes["Node 7"]["Junction"].type = ['two-way_traffic','one-way_traffic','two-way_traffic']
FG.nodes["Node 8"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 8"]["Junction"].name = ['waterway_access','waterway_access','harbour_basin_access']
FG.nodes["Node 8"]["Junction"].type = ['two-way_traffic','two-way_traffic','one-way_traffic']
FG.nodes["Node 9"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 9"]["Junction"].name = ['waterway_access','harbour_basin_access','waterway_access']
FG.nodes["Node 9"]["Junction"].type = ['two-way_traffic','one-way_traffic','two-way_traffic']
FG.nodes["Node 3"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 3"]["Junction"].name = ['waterway_access','waterway_access','waterway_access']
FG.nodes["Node 3"]["Junction"].type = ['two-way_traffic','two-way_traffic','two-way_traffic']
FG.nodes["Node 5"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 5"]["Junction"].name = ['waterway_access','waterway_access','waterway_access']
FG.nodes["Node 5"]["Junction"].type = ['two-way_traffic','two-way_traffic','two-way_traffic']
FG.nodes["Node 15"]["Junction"] = core.IsJunction(env = [], name = [], sections = [], typ = [])
FG.nodes["Node 15"]["Junction"].name = ['waterway_access','waterway_access','waterway_access']
FG.nodes["Node 15"]["Junction"].type = ['two-way_traffic','two-way_traffic','two-way_traffic']

junction_nodes = []
for node in list(FG.nodes):
    if 'Junction' in FG.nodes[node]:
        junction_nodes.append(node)
        
for node1 in junction_nodes:
    names = []
    sections = []
    types = []
    for node2 in junction_nodes:
        if node1 == node2:
            continue
            
        route = nx.dijkstra_path(FG, node1, node2)
        section = True
        for node in route[1:-1]:
            if 'Junction' in FG.nodes[node]:
                section = False
                break

        if section:
            sections.append([node1,node2])
            names.append(FG.nodes[node1]["Junction"].name[len(sections)-1])
            types.append(FG.nodes[node1]["Junction"].type[len(sections)-1])
    
    FG.nodes[node1]["Junction"] = [core.IsJunction(env = env, name = names, sections = sections, typ = types)]
            
FG.edges['Node 6','Node 10']["Terminal"] = [terminal_1]
FG.edges['Node 7','Node 11']["Terminal"] = [terminal_2]
FG.edges['Node 8','Node 12']["Terminal"] = [terminal_3]
FG.edges['Node 9','Node 13']["Terminal"] = [terminal_4]

for edge in enumerate(FG.edges):
    if 'Terminal' in FG.edges[edge[1]]:
        FG.edges[edge[1][1],edge[1][0]]['Terminal'] = FG.edges[edge[1]]['Terminal']


# In[6]:


df = pd.DataFrame()
df[0] = ['Container Vessel','Dry Bulk Vessel','Tanker']
df[1] = [300,300,300] #[366,427,330]
df[2] = [49,55,55]
df[3] = [28,28,28]
df[4] = 0.5*df[3]
df[5] = [66,66,66]
df[6] = df[5]-(df[3]-df[4])
df[7] = [120,120,120]
df[8] = [36*60,72*60,40*60]
df.columns = ['type','L','B','T_f','T_e','H_e','H_f','t_b','t_l']
df


# In[7]:


vessel_db = pd.read_csv(location_vessel_database)
vessel_db.columns = ['vessel_id','type','B','L','H_e','H_f','T_e','T_f','C','P']
vessel_db['t_b'] = 10
vessel_db['t_l'] = [vessel_db['C'][0]/vessel_db['C'][5]*360,vessel_db['C'][1]/vessel_db['C'][5]*360,vessel_db['C'][2]/vessel_db['C'][5]*360,vessel_db['C'][3]/vessel_db['C'][5]*360,vessel_db['C'][4]/vessel_db['C'][5]*360,360]
vessel_db


# In[8]:


Vessel = type('Vessel', 
              (core.Identifiable, core.Movable, core.Routeable, core.VesselProperties, core.ExtraMetadata), {})

generator_sea = model.VesselGenerator(Vessel,df)
generator_inland = model.VesselGenerator(Vessel,vessel_db)


# In[9]:


origin = 'Node 1'
destination = 'Node 11'
sim.add_vessels(vessel_generator = generator_sea, simulation_start = simulation_start, origin = origin, destination = destination, arrival_distribution = (3600/5000), arrival_process = 'Uniform')

#origin = 'Node 17'
#destination = 'Node 11'
#sim.add_vessels(vessel_generator = generator_inland, simulation_start = simulation_start, origin = origin, destination = destination, arrival_distribution = (3600/10000), arrival_process = 'Uniform')

#origin = 'Node 18'
#destination = 'Node 11'
#sim.add_vessels(vessel_generator = generator_inland, simulation_start = simulation_start, origin = origin, destination = destination, arrival_distribution = (3600/10000), arrival_process = 'Uniform')


# In[10]:


depth = [[],[]]
width = [[],[]]
water_level = [[],[]]
time = np.arange(0,duration,60)
phase_lag = [0,300,600,950,1300,725,850,1440,1580,725,850,1440,1580,1090,1830,1460,2430,3030,300]
depth[1] = [32,32,28,20,20,28,28,20,20,32,32,24,24,7,7,7,7,7,32]
width[1] = [100,100,100,100,100,500,500,500,500,500,500,500,500,50,50,50,50,100,100]
for nodes in enumerate(FG.nodes):
    depth[0].append(FG.nodes[nodes[1]]['geometry'])
    width[0].append(FG.nodes[nodes[1]]['geometry'])
    water_level[0].append((FG.nodes[nodes[1]]['geometry']))
    water_level[1].append([[],[]])
    for t in range(len(time)):
        water_level[1][nodes[0]][0].append(time[t]+simulation_start.timestamp())
        water_level[1][nodes[0]][1].append(2.5*np.sin(2*math.pi*(time[t]+simulation_start.timestamp())/(45000)-2*phase_lag[nodes[0]]/45000*math.pi))

core.NetworkProperties.append_data_to_nodes(FG,width,depth,water_level)
core.NetworkProperties.append_info_to_edges(FG)


# In[11]:


sim.run(duration = duration)


# In[12]:


vessels = sim.environment.vessels
env = sim.environment


# In[13]:


df = pd.DataFrame.from_dict(vessels[0].log)
df


# In[14]:


edge_count = np.zeros(len(FG.edges))
for v in vessels:
    df = pd.DataFrame.from_dict(v.log)
    for message in df['Message']:
        if 'Sailing' in message and 'stop' in message:
            r = re.search('Sailing from node (.+?) to node (.+?) stop', message)
            if r:
                node1 = r.group(1)
                node2 = r.group(2)
            for e in enumerate(FG.edges):
                if (node1,node2) == e[1]:
                    edge_count[e[0]] += 1

edge_count_final = np.zeros(len(FG.edges))
for e in enumerate(FG.edges):
    for e2 in enumerate(FG.edges):
        if [e[1][0],e[1][1]] == [e2[1][1],e2[1][0]]:
            edge_count_final[e[0]] = edge_count[e[0]]+edge_count[e2[0]]
            edge_count_final[e2[0]] = edge_count_final[e[0]] 
            break


# In[15]:


colormap = cm.get_cmap('RdYlBu_r', 256)
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')
ax = fig.add_axes([0, 0.4, 1, 0.3]);
nx.draw(FG, positions, node_size = 10, node_color ='k', with_labels = True, horizontalalignment = 'right', verticalalignment = 'bottom', edge_color = edge_count_final, edge_cmap = colormap, arrows = False, width= 4)
plt.axis('equal')
cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap), ax=ax, ticks=[0, 1])
cbar.ax.set_yticklabels(['low','high'])  # vertically oriented colorbar
plt.title('Traffic Intensity of Port X',fontsize = 14, fontweight='bold')
plt.show()


# In[16]:


edge_count = []
for edge in enumerate(FG.edges):
    edge_count.append(FG.edges[edge[1]]['Info']['Depth'][0])

colormap = cm.get_cmap('Blues', 256)
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')
ax = fig.add_axes([0, 0.4, 1, 0.3]);
nx.draw(FG, positions, node_size = 10, node_color ='k', with_labels = True, horizontalalignment = 'right', verticalalignment = 'bottom', edge_color = edge_count, edge_cmap = colormap, edge_vmin = 0, arrows = False, width= 4)
plt.axis('equal')
cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap), ax=ax, ticks=[0, 1])
cbar.ax.set_yticklabels(['shallow','deep'])  # vertically oriented colorbar
plt.title('Bathymetry of Port X',fontsize = 14, fontweight='bold')
plt.show()


# In[17]:


def calculate_distance(orig, dest):
    wgs84 = pyproj.Geod(ellps='WGS84')
    
    distance = wgs84.inv(orig[0], orig[1], 
                         dest[0], dest[1])[2]
    
    return distance

vessel_path_x = []
vessel_path_t = []

list_of_nodes = list(vessels[0].env.FG.nodes)

for node in list_of_nodes:
    if 'Origin' in vessels[0].env.FG.nodes[node].keys():
        origin = node
        
    if 'Anchorage' in vessels[0].env.FG.nodes[node].keys():
        list_of_nodes.remove(node)

    if 'Junction' in vessels[0].env.FG.nodes[node].keys() and vessels[0].env.FG.nodes[node]['Junction'][0].name == ['waterway_access','waterway_access','anchorage_access']:
        virtual_anchorage = node
        
for v in range(0,len(vessels)):
    vessel_path_xt = []
    vessel_path_tt = []
    distance = 0
    direction = 0
    vessel_path_t0 = simulation_start.timestamp()
    vessel_path_xt.append(distance)
    vessel_path_tt.append(vessels[v].log["Timestamp"][0].timestamp()-vessel_path_t0)
    for t in range(1,len(vessels[v].log["Message"])):  
        if 'Deberthing stop' in vessels[v].log["Message"][t]:
            direction = 1
        for node1 in list_of_nodes: 
            for node2 in list_of_nodes:
                if (vessels[v].log["Message"][t] == 'Sailing from node ' + node1 + ' to node ' + node2 + ' start' or 
                    vessels[v].log["Message"][t] == 'Sailing from node ' + node1 + ' to node ' + node2 + ' stop'):
                    if node1 == origin and node2 == virtual_anchorage:
                        distance_to_anchorage = calculate_distance((vessels[v].env.FG.nodes[node1]['geometry'].x,vessels[v].env.FG.nodes[node1]['geometry'].y),(vessels[v].env.FG.nodes[node2]['geometry'].x,vessels[v].env.FG.nodes[node2]['geometry'].y))
                    
                    if direction == 0:
                        distance += calculate_distance((vessels[v].log["Geometry"][t-1].x,vessels[v].log['Geometry'][t-1].y),(vessels[v].log["Geometry"][t].x,vessels[v].log['Geometry'][t].y))
                    elif direction == 1:
                        distance -= calculate_distance((vessels[v].log["Geometry"][t-1].x,vessels[v].log['Geometry'][t-1].y),(vessels[v].log["Geometry"][t].x,vessels[v].log['Geometry'][t].y))
                    vessel_path_xt.append(distance)
                    vessel_path_tt.append(vessels[v].log["Timestamp"][t].timestamp()-vessel_path_t0)
                    break
                    
    vessel_path_x.append(vessel_path_xt)
    vessel_path_t.append(vessel_path_tt)
    


# In[18]:


terminal = vessels[0].env.FG.edges['Node 7','Node 11']['Terminal'][0]
time_available_quay_length = []
available_quay_length = []
quay_level = terminal.length.capacity
time_available_quay_length.append(0)
available_quay_length.append(quay_level)
for t in range(len(terminal.log["Message"])):
    time_available_quay_length.append(terminal.log["Timestamp"][t].timestamp()-simulation_start.timestamp())
    available_quay_length.append(quay_level)
    time_available_quay_length.append(terminal.log["Timestamp"][t].timestamp()-simulation_start.timestamp())
    available_quay_length.append(terminal.log["Value"][t])
    quay_level = terminal.log["Value"][t]
    
anchorage = vessels[0].env.FG.nodes['Node 19']['Anchorage'][0]
time_anchorage_occupation = []
anchorage_occupation = []
anchorage_capacity = 0
time_anchorage_occupation.append(0)
anchorage_occupation.append(anchorage_capacity)
for t in range(len(anchorage.log["Message"])):
    time_anchorage_occupation.append(anchorage.log["Timestamp"][t].timestamp()-simulation_start.timestamp())
    anchorage_occupation.append(anchorage_capacity)
    time_anchorage_occupation.append(anchorage.log["Timestamp"][t].timestamp()-simulation_start.timestamp())
    anchorage_occupation.append(anchorage.log["Value"][t])
    anchorage_capacity = anchorage.log["Value"][t]


# In[19]:


start = 0
end = 100000
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(16, 9),gridspec_kw={'width_ratios': [3.5, 1, 1]})
ax1.axvline(distance_to_anchorage,color = 'k',linestyle = '--')
ax1.axvline(14000,color = 'k',linestyle = '--')
ax1.text(distance_to_anchorage,1.02*end,'Anchorage',horizontalalignment  = 'center')
ax1.text(14000,1.02*end,'Terminal',horizontalalignment  = 'center')
for v in reversed(range(0,len(vessels))):
    ax1.plot(vessel_path_x[v],vessel_path_t[v])
ax1.set_title("Tracking diagram for vessels calling at port", fontweight='bold', pad = 42)
ax1.set_xlabel('Distance [m]')
ax1.set_xlim([0,14000+terminal.length.capacity])
ax1.set_ylabel('Time [s]')
ax1.set_ylim([start,end]);

ax2.plot(available_quay_length,time_available_quay_length)
ax2.axvline(vessels[0].L,color = 'k', linestyle = '--')
ax2.set_title("Available quay length \n over time", fontweight='bold', pad = 32)
ax2.text(vessels[0].L,1.01*end,'Required quay \n length',horizontalalignment = 'center')
ax2.set_xlim([0,1.1*max(available_quay_length)])
ax2.yaxis.set_visible(False)
ax2.set_ylim([start,end]);

ax3.plot([eta+28 for eta in water_level[1][0][1]],[t-simulation_start.timestamp() for t in water_level[1][0][0]])
ax3.axvline(28.5,color = 'k', linestyle = '--')
ax3.set_title("Available water depth \n over time", fontweight='bold', pad = 32)
ax3.text(28.5,1.01*end,'Required water \n depth',horizontalalignment = 'center')
ax3.yaxis.set_visible(False)
ax3.set_ylim([start,end]);


# In[20]:


df = pd.DataFrame.from_dict(terminal.log)
df


# In[21]:


anchorage = vessels[0].env.FG.nodes['Node 19']['Anchorage'][0]
df = pd.DataFrame.from_dict(anchorage.log)
df


# In[22]:


fig,ax = plt.subplots(figsize=(16, 9))
plt.plot(time_anchorage_occupation,anchorage_occupation)
plt.xlabel('Time [s]')
plt.xlim([start,end])
plt.ylabel('Number of vessels')
plt.ylim([0,math.ceil(1.1*np.max(anchorage_occupation))])
plt.title("Anchorage occupancy", fontweight='bold', pad = 12)
plt.show()


# In[ ]:


start = 0
end = 90000
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(16, 3),gridspec_kw={'width_ratios': [1, 1, 1, 1]})
ax3.plot(time_anchorage_occupation,anchorage_occupation)
ax3.set_xlabel('Time [s]')
ax3.set_xlim([start,end])
ax3.set_ylabel('Number of vessels')
ax3.set_ylim([0,math.ceil(1.1*np.max(anchorage_occupation))])
ax3.set_title("Anchorage occupancy", fontweight='bold', pad = 12)
plt.show()


# In[ ]:


edge = ['Node 7','Node 11']
vessel = vessels[0]
index = 1
if 'Junction' in vessel.env.FG.nodes[edge[1]].keys() and 'one-way_traffic' in vessel.env.FG.nodes[edge[0]]['Junction'][0].type[index]:
    print('hi')


# In[ ]:


b = [[0,0],[0,3000]]


# In[ ]:


[[1,300],[0,2700]]


# In[ ]:


[[1,300],[1,300],[1,300],[1,300],[0,1800]]


# In[ ]:


a = [[1,0],[1,300],[0,300],[0,600],[1,600],[1,900],[1,1200],[0,1200],[0,3000]]


# In[ ]:


def calculate_quay_length_level(terminal):
    qL = terminal.available_quay_lengths
    l = [0]
    for i in range(len(qL)):
        if i == 0 or qL[i][1] == qL[i-1][1] or qL[i][0] == 1:
            if i == len(qL)-1:
                new_level = l[-1]
            continue

        l.append(qL[i][1]-qL[i-1][1])
        new_level = l[-1]
    return new_level

def request_terminal_access(vessel,terminal):
    qL = terminal.available_quay_lengths
    L = vessel.L
    
    def pick_minimum_length():
        qL = terminal.available_quay_lengths
        move_to_anchorage = False
        l = [0]
        index_i = 0
        for i in range(len(qL)):
            if i == 0 or qL[i][1] == qL[i-1][1] or qL[i][0] == 1:
                if i == len(qL)-1 and not index_i:
                    move_to_anchorage = True
                continue

            l.append(qL[i][1]-qL[i-1][1])
            l.sort()
            
            print(i,l)
            for j in range(len(l)): 
                if L <= l[j]:
                    index_i = i
                    break

                elif j == len(l)-1 and not index_i:
                    move_to_anchorage = True
                    
        return index_i,move_to_anchorage
                    
    index_i,move_to_anchorage = pick_minimum_length()
    print(index_i,move_to_anchorage)
    
    def adjust_available_quay_lengths(index_i):
        i = index_i
        if qL[i-1][0] == 0:
            qL[i-1][0] = 1

        if qL[i][0] == 0 and qL[i][1] == qL[i-1][1]+L:
            qL[i][0] = 1
        else:
            qL.insert(i,[1,L+qL[i-1][1]])
            qL.insert(i+1,[0,L+qL[i-1][1]])

        vessel.quay_position = 0.5*L+qL[i-1][1]
        new_level = calculate_quay_length_level(terminal)
        #terminal.get(old_level-new_level)
        return

    if not move_to_anchorage: 
        adjust_available_quay_lengths(index_i)

    if move_to_anchorage:
        #adjust_available_quay_lengths(index_i)
        pass
        #move_to_anchorage()

    terminal.available_quay_lengths = qL
    return


def release_request_terminal_access(vessel,terminal):
    qL = terminal.available_quay_lengths
    position = 450
    old_level = calculate_quay_length_level(terminal)
    for i in range(len(qL)):
        if i == 0:
            continue
        if qL[i-1][1] < position and qL[i][1] > position:
            break

    if i == 1:
        qL[i-1][0] = 0
        qL[i][0] = 0

    elif i == len(qL)-1:
        qL[i-1][0] = 0
        qL[i][0] = 0

    else:
        qL[i-1][0] = 0
        qL[i][0] = 0 

    to_remove = []    
    for i in enumerate(qL):
        for j in enumerate(qL):
            if i[0] != j[0] and i[1][0] == 0 and j[1][0] == 0 and i[1][1] == j[1][1]:
                to_remove.append(i[0])

    for i in list(reversed(to_remove)):
        qL.pop(i)
    
    new_level = calculate_quay_length_level(terminal)
    #terminal.put(new_level-old_level)
    terminal.available_quay_lengths = qL
    return


# In[ ]:


#terminal.available_quay_lengths = [[0,0],[0,1200]]
vessel = vessels[0]
terminal = vessels[0].env.FG.edges['Node 7','Node 11']['Terminal'][0]
request_terminal_access(vessel,terminal)
#release_request_terminal_access(vessel,terminal)
terminal.available_quay_lengths


# In[ ]:


x = []
y = []
for i in qL:
    x.append(i[1])
    y.append(i[0])
plt.plot(x,y)


# In[ ]:


for i in range(len(qL)):
    if i == 0 or qL[i][1] == qL[i-1][1] or qL[i][0] == 1:
        if i == len(qL)-1 and not index_i:
            move_to_anchorage = True
        continue

    l.append(qL[i][1]-qL[i-1][1])
    l.sort()


# In[ ]:


qL


# In[ ]:


qL = qL#[[0,0],[0,1200]]

L = 300
l = []
index_i = 0
[qL,position] = request_terminal_access(l,L,qL,index_i = 0)
qL = release_request_terminal_access(qL,position)


# In[ ]:


if 'Terminal' in vessels[0].env.FG.edges['Node 7','Node 11'].keys():
    print('hi')


# In[ ]:





# In[ ]:




