"""Main module."""

# package(s) related to time, space and id
import json
import logging
import uuid

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy
import random
import networkx as nx
import numpy as np
import math
import bisect
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import time as timepy

# OpenTNSim
from opentnsim import energy_consumption_module
from opentnsim import lock_module
from operator import itemgetter
from sklearn.decomposition import PCA

# spatial libraries
import pyproj
import shapely.geometry

# additional packages
import datetime

logger = logging.getLogger(__name__)


class SimpyObject:
    """General object which can be extended by any class requiring a simpy environment
    env: a simpy Environment
    """

    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env

class HasResource(SimpyObject):
    """Something that has a resource limitation, a resource request must be granted before the object can be used.

    nr_resources: nr of requests that can be handled simultaneously"""

    def __init__(self, nr_resources=1, priority=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.resource = (
            simpy.PriorityResource(self.env, capacity=nr_resources)
            if priority
            else simpy.Resource(self.env, capacity=nr_resources)
        )

class Identifiable:
    """Mixin class: Something that has a name and id

    name: a name
    id: a unique id generated with uuid"""

    def __init__(self, name, id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.name = name
        # generate some id, in this case based on m
        self.id = id if id else str(uuid.uuid1())
        
class HasType:
    """Mixin class: Something that has a name and id

    name: a name
    id: a unique id generated with uuid"""

    def __init__(self, type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.type = type

class Locatable:
    """Mixin class: Something with a geometry (geojson format)

    geometry: can be a point as well as a polygon"""

    def __init__(self, geometry, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.geometry = geometry
        self.node = None

class Neighbours:
    """Can be added to a locatable object (list)
    travel_to: list of locatables to which can be travelled"""

    def ___init(self, travel_to, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.neighbours = travel_to

class HasLength(SimpyObject): #used by IsLock and IsLineUpArea to regulate number of vessels in each lock cycle and calculate repsective position in lock chamber/line-up area
    """Mixin class: Something with a storage capacity
    capacity: amount the container can hold
    level: amount the container holds initially
    total_requested: a counter that helps to prevent over requesting"""

    def __init__(self, length, remaining_length=0, total_requested=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.length = simpy.Container(self.env, capacity = length, init=remaining_length)
        self.pos_length = simpy.Container(self.env, capacity = length, init=remaining_length)

class HasContainer(SimpyObject):
    """Mixin class: Something with a storage capacity
    capacity: amount the container can hold
    level: amount the container holds initially
    container: a simpy object that can hold stuff
    total_requested: a counter that helps to prevent over requesting"""

    def __init__(self, capacity, level=0, total_requested=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.container = simpy.Container(self.env, capacity, init=level)
        self.total_requested = total_requested

    @property
    def is_loaded(self):
        return True if self.container.level > 0 else False

    @property
    def filling_degree(self):
        return self.container.level / self.container.capacity

class Log(SimpyObject):
    """Mixin class: Something that has logging capability

    log: log message [format: 'start activity' or 'stop activity']
    t: timestamp
    value: a value can be logged as well
    geometry: value from locatable (lat, lon)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.log = {"Message": [], "Timestamp": [], "Value": [], "Geometry": []}

    def log_entry(self, log, t, value, geometry_log):
        """Log"""
        self.log["Message"].append(log)
        self.log["Timestamp"].append(datetime.datetime.fromtimestamp(t))
        self.log["Value"].append(value)
        self.log["Geometry"].append(geometry_log)

    def get_log_as_json(self):
        json = []
        for msg, t, value, geometry_log in zip(
            self.log["Message"],
            self.log["Timestamp"],
            self.log["Value"],
            self.log["Geometry"],
        ):
            json.append(
                dict(message=msg, time=t, value=value, geometry_log=geometry_log)
            )
        return json

class ExtraMetadata:
    """store all leftover keyword arguments as metadata property (use as last mixin)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        # store all other properties as metadata
        self.metadata = kwargs

class VesselProperties:
    """Mixin class: Something that has vessel properties
    This mixin is updated to better accommodate the ConsumesEnergy mixin

    type: can contain info on vessel type (avv class, cemt_class or other)
    B: vessel width
    L: vessel length
    H_e: vessel height unloaded
    H_f: vessel height loaded
    T_e: draught unloaded
    T_f: draught loaded

    Add information on possible restrictions to the vessels, i.e. height, width, etc.
    """

    def __init__(
            self,
            type,
            B,
            L,
            H_e,
            H_f,
            T_e,
            T_f,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        """Initialization"""
        self.type = type

        self.B = B
        self.L = L

        self.H_e = H_e
        self.H_f = H_e

        self.T_e = T_e
        self.T_f = T_f

    @property
    def H(self):
        """ Calculate current height based on filling degree """

        return (
                self.filling_degree * (self.H_f - self.H_e)
                + self.H_e
        )

    @property
    def T(self):
        """ Calculate current draught based on filling degree

        Here we should implement the rules from Van Dorsser et al
        https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships
        """

        return (
                self.filling_degree * (self.T_f - self.T_e)
                + self.T_e
        )

    def get_route(
            self,
            origin,
            destination,
            graph=None,
            minWidth=None,
            minHeight=None,
            minDepth=None,
            randomSeed=4,
    ):
        """ Calculate a path based on vessel restrictions """

        graph = graph if graph else self.env.FG
        minWidth = minWidth if minWidth else 1.1 * self.B
        minHeight = minWidth if minHeight else 1.1 * self.H
        minDepth = minWidth if minDepth else 1.1 * self.T

        # Check if information on restrictions is added to the edges
        random.seed(randomSeed)
        edge = random.choice(list(graph.edges(data=True)))
        edge_attrs = list(edge[2].keys())

        # IMPROVE THIS TO CHECK ALL EDGES AND COMBINATIONS OF RESTRICTIONS

        if all(item in edge_attrs for item in ["Width", "Height", "Depth"]):
            edges = []
            nodes = []

            for edge in graph.edges(data=True):
                if (
                        edge[2]["Width"] >= minWidth
                        and edge[2]["Height"] >= minHeight
                        and edge[2]["Depth"] >= minDepth
                ):
                    edges.append(edge)

                    nodes.append(graph.nodes[edge[0]])
                    nodes.append(graph.nodes[edge[1]])

            subGraph = graph.__class__()

            for node in nodes:
                subGraph.add_node(
                    node["name"],
                    name=node["name"],
                    geometry=node["geometry"],
                    position=(node["geometry"].x, node["geometry"].y),
                )

            for edge in edges:
                subGraph.add_edge(edge[0], edge[1], attr_dict=edge[2])

            try:
                return nx.dijkstra_path(subGraph, origin, destination)
                # return nx.bidirectional_dijkstra(subGraph, origin, destination)
            except:
                raise ValueError(
                    "No path was found with the given boundary conditions."
                )

        # If not, return shortest path
        else:
            return nx.dijkstra_path(graph, origin, destination)

class IsJunction(Identifiable,HasType, Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
        self,
        sections,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        "Initialization"

        self.sections = sections
        self.section = []

        for edge in enumerate(self.sections):
            if self.type[edge[0]] == 'one-way_traffic':
                direction = 0
                if self.env.FG.nodes[edge[1][0]]['geometry'].x != self.env.FG.nodes[edge[1][1]]['geometry'].x:
                    if self.env.FG.nodes[edge[1][0]]['geometry'].x < self.env.FG.nodes[edge[1][1]]['geometry'].x:
                        direction = 1
                elif self.env.FG.nodes[edge[1][0]]['geometry'].y < self.env.FG.nodes[edge[1][1]]['geometry'].y:
                    direction = 1

                if direction:
                    if 'access1' not in dir(self):
                        self.access1 = []
                        self.access2 = []

                    self.access1.append({edge[1][0]: simpy.PriorityResource(self.env, capacity=1),})
                    self.access2.append({edge[1][1]: simpy.PriorityResource(self.env, capacity=1),})

            self.section.append({edge[1][1]: simpy.PriorityResource(self.env, capacity=10000),})

class IsTerminal(HasType, HasLength, Identifiable, Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""
    def __init__(
        self,
        node_start,
        node_end,
        length,
        type,
        jetty_locations = [],
        jetty_lengths=[],
        *args,
        **kwargs
    ):

        "Initialization"
        self.type = type
        self.output = {}
        super().__init__(type=type,length=length, remaining_length=length, *args, **kwargs)

        if self.type == 'quay':
            self.terminal = {
                node_start: simpy.PriorityResource(self.env, capacity=100),
            }

            self.available_quay_lengths = [[0,0],[0,length]]

        elif self.type == 'jetty':
            self.terminal = []
            self.jetties_occupied = 0
            self.jetty_locations = jetty_locations
            self.jetty_lengths = jetty_lengths
            for jetty in range(len(jetty_locations)):
                self.terminal.append({
                    node_start: simpy.PriorityResource(self.env, capacity=1),
                })


class IsOrigin(Identifiable,Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
class IsAnchorage(Identifiable,HasType,Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
        self,
        node,
        max_capacity,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.max_capacity = max_capacity
        self.output = {}
        self.anchorage_area = {
            node: simpy.PriorityResource(self.env, capacity=max_capacity),
        }
        
class NetworkProperties:
    """Class: a collection of functions that append properties to the network"""

    def __init__(self,
                 network,
                 W,
                 D,
                 eta,
                 *args,
                 **kwargs
    ):
        super().__init__(*args, **kwargs)

    def append_data_to_nodes(network,W,D,MBL,eta,vmag,vdir):
        """ Function: appends geometric and hydrodynamic data to the nodes

            Input:
                - network: a graph constructed with the DiGraph class of the networkx package
                - W: a list with [[],[]]-format containing the geometry of the individual nodes imported as points constructed with
                     the geometry class of the shapely package in the first column and the actual waterway widths in [m] as floats
                     at the nodes in the second column. The indexes should be matching, containing information about the same node.
                - D: a list with [[],[]]-format containing the geometry of the nodes imported as points constructed with
                     the geometry class of the shapely package in the first column and the actual waterway depths in [m] as floats
                     at the nodes in the second column. The indexes should be matching, containing information about the same node.
                - MBL: a list with [[],[]]-format containing the geometry of the individual nodes imported as points constructed with
                       the geometry class of the shapely package in the first column and the actual maintained water depths of
                       the waterays in [m] as floats in the second column. The indexes should be matching, containing information about
                       the same node.
                - eta: a list with [[],[]]-format containing the geometry of the individual nodes imported as points constructed with
                       the geometry class of the shapely package in the first column and a list with [[],[]]-format of time
                       series of the actual water levels in [m] in the second column. The indexes should be matching, containing information
                       about the same node. The list in the second column should consist of the timestamps of the data in the first column
                       and the values of the property as floats in the second column. The time series should be in chronological order.
                - vmag: a list with [[],[]]-format containing the geometry of the individual nodes imported as points constructed with
                        the geometry class of the shapely package in the first column and a list with [[],[]]-format of time
                        series of the actual current velocity magnitudes in [m/s] in the second column. The indexes should be matching,
                        containing information about the same node. The list in the second column should consist of the timestamps of the
                        data in the first column and the values of the property as floats in the second column. The time series should be
                        in chronological order.
                - vdir: a list with [[],[]]-format containing the geometry of the individual nodes imported as points constructed with
                        the geometry class of the shapely package in the first column and a list with [[],[]]-format of time
                        series of the actual current directions in [rad] in the second column. The indexes should be matching, containing
                        information about the same node. The list in the second column should consist of the timestamps of the data in the
                        first column and the values of the property as floats in the second column. The time series should be in chronological
                        order. Additionally, the reference system of the current direction should be in [0,2pi]-form with the northern direction
                        corresponding to 0 and 2pi radians.
        """

        #Some functions used in the append_data_to_node_function
        def fixed2bearing(east_velocity, north_velocity, principal_direction):
            """ Function: calculates the velocity components in a reference frame parallel to the principal direction of the
                          velocity cluster. It returns the decomposed current velocity magnitudes in a reference frame relative
                          to the given principal direction.

                Input:
                    - east_velocity: velocity in Eastern direction in m/s (float)
                    - north_velocity: velocity in Northern direction in m/s (float)
                    - principal_direction: principal direction in degrees in fixed reference frame North-East (float)
            """

            bearing_rad = np.radians(principal_direction)
            x_velocity = np.cos(bearing_rad) * east_velocity + np.sin(bearing_rad) * north_velocity
            y_velocity = -1 * np.sin(bearing_rad) * east_velocity + np.cos(bearing_rad) * north_velocity

            return x_velocity, y_velocity

        def fixed2principal_components(east_velocity_list, north_velocity_list):
            """ Function: calculates the principal components from a cluster of velocities. It returns principal direction
                          in degrees (float)

                Input:
                    - east_velocity_list: list of velocities in Eastern direction (list)
                    - north_velocity_list: list of velocities in Northern direction (list)
            """

            pca = PCA(n_components=2)
            X = np.column_stack((east_velocity_list, north_velocity_list))
            X_pca = pca.fit(X)
            y_pca = pca.components_[:, 1][0]
            x_pca = pca.components_[:, 0][0]
            theta = np.arctan2(y_pca, x_pca)
            alpha = np.degrees(theta)

            # Correction for positive alpha in coordinate system with positive x-direction upestuary
            if alpha >= 0:
                alpha = alpha - 180  # in degrees
            return alpha

        def H99(t,wlev,node):
            """ Function: calculates the water level which is exceeded 99% of the tides for a given node in the network.

                Input:
                    - t: a list containing the chronological sequence of timestamps for the specific node, derived from the
                         list of the time series containing the water level at the specific node
                    - wlev: a list containing the chronological sequence of water levels for the specific node, derived from the
                            list of the time series containing the water level at the specific node
                    - node: the name string of the node in the given network
            """

            # Deriving some properties
            mean_wlev = np.mean(wlev)
            deltat = t[1] - t[0]

            # Interpolation of the variation of the water level and calculation of the zero crossing times
            intp = sc.interpolate.CubicSpline(t, [w - mean_wlev for w in wlev])
            roots = []
            for root in intp.roots():
                if root < network.nodes[node]['Info']['Water level'][0][0] or root > network.nodes[node]['Info']['Water level'][0][-1]:
                    continue
                roots.append(root)

            # Calculation of the maximum water level for each tidal cycle (when water level is greater than mean water level)
            max_index = []
            max_t = []
            max_wlev = []
            prev_root = t[0]
            for root in enumerate(roots):
                if wlev[0] <= mean_wlev:
                    if root[0] % 2 == 0:
                        prev_root = root[1]
                        continue
                else:
                    if root[0] % 2 == 1:
                        prev_root = root[1]
                        continue

                index_range_min = bisect.bisect_right(t, prev_root)
                if t[index_range_min] - prev_root < deltat / 2: index_range_min = index_range_min - 1
                index_range_max = bisect.bisect_right(t, root[1])
                if index_range_max == len(t) or t[index_range_max] - root[1] < deltat / 2: index_range_max = index_range_max - 1
                max_index.append(np.argmax(wlev[index_range_min:index_range_max]) + index_range_min)
                max_t.append(t[max_index[-1]])
                max_wlev.append(wlev[max_index[-1]])

            # Sorting lists from highest to lowest maximum water level
            zipped_lists = zip(max_wlev, max_index, max_t)
            sorted_pairs = sorted(zipped_lists)
            tuples = zip(*sorted_pairs)
            mwlev, mindex, mt = [list(tuple) for tuple in tuples]

            # Returning the water level in the list closest to the 99% value
            return mwlev[round(len(max_wlev) * 0.01)]

        def append_info_to_edges(network):
            """ Function: appends nodal data to the edges of the network for visualisation purposes only

                Input:
                    - network: a graph constructed with the DiGraph class of the networkx package

            """

            # Loops over the edges of the network
            for edge in enumerate(network.edges):
                # Adds parameters to the dictionary
                network.edges[edge[1]]['Info']['Width'] = []
                network.edges[edge[1]]['Info']['Depth'] = []
                network.edges[edge[1]]['Info']['MBL'] = []

                #Appends data to the edges
                network.edges[edge[1]]['Info']['Width'].append(np.min([network.nodes[edge[1][0]]['Info']['Width'][0], network.nodes[edge[1][1]]['Info']['Width'][0]]))

                #If there is a terminal in the edge, the greatest value for the MBL or depth of the two nodes creating the edge are used
                if 'Terminal' in network.edges[edge[1]]:
                    network.edges[edge[1]]['Info']['Depth'].append(np.max([network.nodes[edge[1][0]]['Info']['Depth'][0], network.nodes[edge[1][1]]['Info']['Depth'][0]]))
                    network.edges[edge[1]]['Info']['MBL'].append(np.max([network.nodes[edge[1][0]]['Info']['MBL'][0], network.nodes[edge[1][1]]['Info']['MBL'][0]]))
                else:
                    network.edges[edge[1]]['Info']['Depth'].append(np.min([network.nodes[edge[1][0]]['Info']['Depth'][0], network.nodes[edge[1][1]]['Info']['Depth'][0]]))
                    network.edges[edge[1]]['Info']['MBL'].append(np.min([network.nodes[edge[1][0]]['Info']['MBL'][0], network.nodes[edge[1][1]]['Info']['MBL'][0]]))

        def calculate_and_append_current_components_to_nodes(network):
            """ Function: calculates the longitudinal and lateral (cross-) current velocity with respect to the sailing path of
                          the vessel (alignment of the edge) for each node (there can be multiple time series for each node as
                          multiple edges can be connected at this node) and appends this information to the nodes of the network.

                Input:
                    - network: a graph constructed with the DiGraph class of the networkx package
            """

            #Loops over the nodes of the network and finds the edges attached to this node by a second loop over the nodes
            for node1 in network.nodes:
                nodes = []
                for node2 in network.nodes:
                    if (node1, node2) in network.edges:
                        nodes.append(node2)

                #If there is only one edge connected to the node (boundary nodes): longitudinal and cross-currents are assumed 0
                if len(nodes) == 1:
                    network.nodes[node1]['Info']['Cross-current'][nodes[0]] = [[], []]
                    network.nodes[node1]['Info']['Longitudinal current'][nodes[0]] = [[], []]
                    time_current_velocity = network.nodes[node1]['Info']['Current velocity'][0]
                    for t in range(len(time_current_velocity)):
                        network.nodes[node1]['Info']['Cross-current'][nodes[0]][0].append(time_current_velocity[t])
                        network.nodes[node1]['Info']['Cross-current'][nodes[0]][1].append(0)
                        network.nodes[node1]['Info']['Longitudinal current'][nodes[0]][0].append(time_current_velocity[t])
                        network.nodes[node1]['Info']['Longitudinal current'][nodes[0]][1].append(0)
                    continue

                # Else if there is multiple edges connected to the node: loop over the connected edges
                for node2 in nodes:
                    network.nodes[node1]['Info']['Cross-current'][node2] = [[], []]
                    network.nodes[node1]['Info']['Longitudinal current'][node2] = [[], []]
                    time_current = network.nodes[node1]['Info']['Current velocity'][0]
                    current_velocity = network.nodes[node1]['Info']['Current velocity'][1]
                    current_direction = network.nodes[node1]['Info']['Current direction'][1]

                    #Calculation of the orientation of the edge
                    origin_lat = network.nodes[node2]['geometry'].x
                    origin_lon = network.nodes[node2]['geometry'].y
                    node_lat = network.nodes[node1]['geometry'].x
                    node_lon = network.nodes[node1]['geometry'].y
                    course, _, _ = pyproj.Geod(ellps="WGS84").inv(origin_lat, origin_lon, node_lat, node_lon)
                    if course < 0:
                        course = 360 + course

                    #Calculation of the current velocity components
                    for t in range(len(time_current)):
                        network.nodes[node1]['Info']['Cross-current'][node2][0].append(time_current[t])
                        network.nodes[node1]['Info']['Cross-current'][node2][1].append(abs(current_velocity[t] * np.sin((current_direction[t] - course) / 180 * math.pi)))
                        network.nodes[node1]['Info']['Longitudinal current'][node2][0].append(time_current[t])
                        network.nodes[node1]['Info']['Longitudinal current'][node2][1].append(abs(current_velocity[t] * np.cos((current_direction[t] - course) / 180 * math.pi)))

        #Continuation of the append_data_to_nodes function by looping over all the nodes
        for node in enumerate(network.nodes):
            #Creating a dictionary attached to the nodes of the network
            network.nodes[node[1]]['Info'] = {'Tidal periods': [],'Horizontal tidal restriction': {},'Vertical tidal restriction': {}, 'Width': [], 'Depth': [], 'MBL': [], 'H_99%': [], 'Water level': [[],[]],  'Current velocity': [[],[]], 'Current direction': [[],[]], 'Cross-current': {}, 'Longitudinal current': {}}

            #Appending the specific data to the network if the geometry of the node of the data is the same as the geometry of the node of the network for the static data
            if (MBL[0][node[0]].x,MBL[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                network.nodes[node[1]]['Info']['MBL'].append(MBL[1][node[0]])
            if (W[0][node[0]].x,W[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                network.nodes[node[1]]['Info']['Width'].append(W[1][node[0]])
            if (D[0][node[0]].x,D[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                network.nodes[node[1]]['Info']['Depth'].append(D[1][node[0]])

            # Appending the data to the specific lists by looping over the times in the time series for the dynamic data
            if (eta[0][node[0]].x,eta[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                for t in range(len(eta[1][node[0]][0])):
                    network.nodes[node[1]]['Info']['Water level'][0].append(eta[1][node[0]][0][t])
                    network.nodes[node[1]]['Info']['Water level'][1].append(eta[1][node[0]][1][t])
            if (vmag[0][node[0]].x,vmag[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                for t in range(len(vmag[1][node[0]][0])):
                    network.nodes[node[1]]['Info']['Current velocity'][0].append(vmag[1][node[0]][0][t])
                    network.nodes[node[1]]['Info']['Current velocity'][1].append(vmag[1][node[0]][1][t])
            if (vdir[0][node[0]].x,vdir[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                for t in range(len(vdir[1][node[0]][0])):
                    network.nodes[node[1]]['Info']['Current direction'][0].append(vdir[1][node[0]][0][t])
                    network.nodes[node[1]]['Info']['Current direction'][1].append(vdir[1][node[0]][1][t])

            # Calculation of the water level which is exceeded 99% of the tides
            network.nodes[node[1]]['Info']['H_99%'].append(H99(network.nodes[node[1]]['Info']['Water level'][0],network.nodes[node[1]]['Info']['Water level'][1],node[1]))

            # Calculation of the eastern and northern current velocities
            east_vel = []
            north_vel = []
            t_vel = network.nodes[node[1]]['Info']['Current direction'][0]
            cur_mag = network.nodes[node[1]]['Info']['Current velocity'][1]
            cur_dir = network.nodes[node[1]]['Info']['Current direction'][1]
            for i in range(len(t_vel)):
                east_vel.append(cur_mag[i] * np.sin(cur_dir[i] / 180 * np.pi))
                north_vel.append(cur_mag[i] * np.cos(cur_dir[i] / 180 * np.pi))

            # Calculation of the principal current velocity
            prim = fixed2principal_components(east_vel,north_vel)
            vel = [fixed2bearing(x, y, prim) for x, y in zip(east_vel,north_vel)]
            vel_prim = [x[0] for x in vel]

            # Calculation of the start and stop times of the individual tidal cycles
            interp = sc.interpolate.CubicSpline(t_vel, vel_prim)
            roots = interp.roots()
            times_tidal_periods = []
            prev_root = 0
            for root in interp.roots():
                if root > t_vel[0] and root < t_vel[-1] and vel_prim[bisect.bisect_right(t_vel, root)] > 0 and root - prev_root > 0.25 * 12.5 * 60 * 60:
                    times_tidal_periods.append([root, 'Flood Start'])
                    prev_root = root
                elif root > t_vel[0] and root < t_vel[-1] and vel_prim[bisect.bisect_right(t_vel, root)] < 0 and root - prev_root > 0.25 * 12.5 * 60 * 60:
                    times_tidal_periods.append([root, 'Ebb Start'])
                    prev_root = root
            network.nodes[node[1]]['Info']['Tidal periods'] = times_tidal_periods

        # Appending longitudinal and cross-current velocity components to the nodes of the network
        calculate_and_append_current_components_to_nodes(network)

        # Appending static data to the edges
        append_info_to_edges(network)

    def append_vertical_tidal_restriction_to_network(network,node,vertical_tidal_window_input):
        """ Function: appends vertical tidal restrictions to the node of the network

            Input:
                - network: a graph constructed with the DiGraph class of the networkx package
                - node: the name string of the node in the given network
                - vertical_tidal_window_input: assembly of specific information that defines the restriction (see specific input classes in the notebook)
        """

        #Specifies two parameters in the dictionary with a corresponding data structure of lists
        network.nodes[node]['Info']['Vertical tidal restriction']['Type'] = [[], [], []]
        network.nodes[node]['Info']['Vertical tidal restriction']['Specification'] = [[], [], [], [], []]

        #Loops over the number of types of restrictions that may hold for different classes of vessels
        for input_data in vertical_tidal_window_input:
            # Unpacks the data for flood and ebb and appends it to a list
            ukc_p = []
            ukc_s = []
            fwa = []
            for info in input_data.window_specifications.ukc_p:
                ukc_p.append(input_data.window_specifications.ukc_p[info])
            for info in input_data.window_specifications.ukc_s:
                ukc_s.append(input_data.window_specifications.ukc_s[info])
            for info in input_data.window_specifications.fwa:
                fwa.append(input_data.window_specifications.fwa[info])

            # Appends the specific data regarding the type of the restriction to data structure
            network.nodes[node]['Info']['Vertical tidal restriction']['Type'][0].append(ukc_s)
            network.nodes[node]['Info']['Vertical tidal restriction']['Type'][1].append(ukc_p)
            network.nodes[node]['Info']['Vertical tidal restriction']['Type'][2].append(fwa)

            # Unpacks the data for the different vessel criteria and appends it to a list
            vessel_characteristics_type = []
            vessel_characteristics_spec = []
            vessel_characteristics_value = []
            vessel_characteristics = input_data.vessel_specifications.characteristic_dicts()
            for info in vessel_characteristics:
                vessel_characteristics_type.append(info)
                vessel_characteristics_spec.append(vessel_characteristics[info][0])
                vessel_characteristics_value.append(vessel_characteristics[info][1])

            # Unravels the boolean operators between the restrictions and appends it to a list
            vessel_method_list = []
            sign_list = input_data.vessel_specifications.vessel_method.split()
            for sign in sign_list:
                if sign[0] != '(' and sign[-1] != ')' and sign != 'x':
                    vessel_method_list.append(sign)

            # Appends the specific data regarding the properties for which the restriction holds to data structure
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][0].append(vessel_characteristics_type)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][1].append(vessel_characteristics_value)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][2].append(input_data.vessel_specifications.vessel_direction)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][3].append(vessel_characteristics_spec)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][4].append(vessel_method_list)

    def append_horizontal_tidal_restriction_to_network(network,node,horizontal_tidal_window_input):
        """ Function: appends horizontal tidal restrictions to the node of the network

            Input:
                - network: a graph constructed with the DiGraph class of the networkx package
                - node: the name string of the node in the given network
                - horizontal_tidal_window_input: assembly of specific information that defines the restriction (see specific input classes in the notebook)
        """

        # Specifies two parameters in the dictionary with a corresponding data structure of lists
        network.nodes[node]['Info']['Horizontal tidal restriction']['Type'] = [[], [], []]
        network.nodes[node]['Info']['Horizontal tidal restriction']['Data'] = {}
        network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'] = [[], [], [], [], [], []]

        # Loops over the number of types of restrictions that may hold for different classes of vessels
        for input_data in enumerate(horizontal_tidal_window_input):
            # Unpacks the data for flood and ebb and appends it to a list
            current_velocity_value = []
            for info in input_data[1].window_specifications.current_velocity_values:
                current_velocity_value.append(input_data[1].window_specifications.current_velocity_values[info])

            # Dependent on the type of restriction, additional information should be unpacked and appended to a list
            current_velocity_range = []
            # - if no necessary:
            if input_data[1].window_specifications.current_velocity_ranges == dict:
                current_velocity_range = []
            # - elif necessary:
            else:
                for info in input_data[1].window_specifications.current_velocity_ranges:
                    current_velocity_range.append(input_data[1].window_specifications.current_velocity_ranges[info])

            # Appends the specific data regarding the type of the restriction to data structure
            network.nodes[node]['Info']['Horizontal tidal restriction']['Type'][0].append(input_data[1].window_specifications.window_method)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Type'][1].append(current_velocity_value)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Type'][2].append(current_velocity_range)

            # Unpacks the data for the different vessel criteria and appends it to a list
            vessel_characteristics_type = []
            vessel_characteristics_spec = []
            vessel_characteristics_value = []
            vessel_characteristics = input_data[1].vessel_specifications.characteristic_dicts()
            for info in vessel_characteristics:
                vessel_characteristics_type.append(info)
                vessel_characteristics_spec.append(vessel_characteristics[info][0])
                vessel_characteristics_value.append(vessel_characteristics[info][1])

            # Unravels the boolean operators between the restrictions and appends it to a list
            vessel_method_list = []
            sign_list = input_data[1].vessel_specifications.vessel_method.split()
            for sign in sign_list:
                if sign[0] != '(' and sign[-1] != ')' and sign != 'x':
                    vessel_method_list.append(sign)

            # Appends the specific data regarding the properties for which the restriction holds to data structure
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][0].append(vessel_characteristics_type)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][1].append(vessel_characteristics_value)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][2].append(input_data[1].vessel_specifications.vessel_direction)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][3].append(vessel_characteristics_spec)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][4].append(vessel_method_list)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][5].append([input_data[1].condition['Origin'], input_data[1].condition['Destination']])

            # Unpacks the specific current velocity data used for the restriction and appends it to the data structure
            # - if pre-calculated data from the model should be used:
            if type(input_data[1].data[1]) == str:
                # - if raw current velocity data should be used:
                if input_data[1].data[1] == 'Current velocity':
                    for n in network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][5][input_data[0]]:
                        network.nodes[node]['Info']['Horizontal tidal restriction']['Data'][n] = network.nodes[input_data[1].data[0]]['Info'][input_data[1].data[1]]
                # - elif longitudinal or cross-current velocity should be used:
                else:
                    network.nodes[node]['Info']['Horizontal tidal restriction']['Data'] = network.nodes[input_data[1].data[0]]['Info'][input_data[1].data[1]]
            # - elif manual data should be used:
            else:
                for n in nodes:
                    network.nodes[node]['Info']['Horizontal tidal restriction']['Data'][n][0] = input_data[1].data[node][0]
                    network.nodes[node]['Info']['Horizontal tidal restriction']['Data'][n][1] = input_data[1].data[node][1]

class VesselTrafficService:
    """Class: a collection of functions that processes requests of vessels regarding the nautical processes on ow to enter the port safely"""

    def provide_ukc_clearance(vessel,node):
        """ Function: calculates the sail-in-times for a specific vessel with certain properties and a pre-determined route and provides this information to the vessel

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - vertical_tidal_window:
                - horizontal_tidal_window:
                - route:
                - out:
                - plot:
                - sailing_time_correction:
                - visualization_calculation:

        """
        interp_wdep, ukc_s, ukc_p, fwa = provide_sail_in_times_tidal_window(vessel, [node], ukc_calc=True)
        net_ukc = interp_wdep(vessel.env.now) - (vessel.T_f + vessel.metadata['ukc'])

        return net_ukc

    def provide_sail_in_times_tidal_window(vessel,route,out=False,plot=False,sailing_time_correction=True,visualization_calculation=False,ukc_calc=False):
        """ Function: calculates the sail-in-times for a specific vessel with certain properties and a pre-determined route and provides this information to the vessel

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                - out: a bool that indicates whether the vessels sails outbound or not (inbound)
                - plot: provide a visualization of the calculation for each vessel
                - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)
                - visualization_calculation: a bool that indicates whether the calculation should be made for a single node or not (for a route with multiple nodes)

        """

        # Functions used to calculate the sail-in-times for a specific vessel
        def tidal_window_restriction_determinator(vessel, types, specifications):
            """ Function: determines which tidal window restriction applies to the vessel at the specific node

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - types: the type of the restriction
                    - specifications: the specific data regarding the properties for which the restriction holds

            """

            # Predefined bool
            boolean = True

            # Determining if and which restriction applies for the vessel by looping over the restriction class
            for restriction_class in enumerate(specifications[0]):
                # - if restriction does not apply to vessel because it is for vessels sailing in the opposite direction: continue loop
                if vessel.bound != specifications[2][restriction_class[0]]:
                    continue
                # - else: looping over the restriction criteria
                for restriction_type in enumerate(restriction_class[1]):
                    # - if previous condition is not met and there are no more restriction criteria and the previous condition has an 'AND' boolean statement: continue loop
                    if not boolean and restriction_type[0] == len(restriction_class[1]) - 1 and specifications[-1][restriction_class[0]][restriction_type[0] - 1] == 'and':
                        continue
                    # - if previous condition is not met and there are more restriction criteria and the next condition has an 'AND' boolean statement: continue loop
                    if not boolean and restriction_type[0] != len(restriction_class[1]) - 1 and specifications[-1][restriction_class[0]][restriction_type[0]] == 'and':
                        continue
                    # - if previous condition is not met and the next condition has an 'OR' boolean statement: continue loop with predefined boolean
                    if not boolean and restriction_type[0] != len(restriction_class[1]) - 1 and specifications[-1][restriction_class[0]][restriction_type[0]] == 'or':
                        boolean = True
                        continue

                    # Extracting the correct vessel property for the restriction type
                    if restriction_type[1].find('Length') != -1: value = getattr(vessel, 'L')
                    if restriction_type[1].find('Draught') != -1: value = getattr(vessel, 'T_f')
                    if restriction_type[1].find('Width') != -1: value = getattr(vessel, 'B')
                    if restriction_type[1].find('UKC') != -1: _, _, value = []
                    if restriction_type[1].find('Type') != -1: value = getattr(vessel, 'type')

                    # Determine if the value for the property satisfies the condition of the restriction type
                    df = pd.DataFrame({'Value': [value],'Restriction': [specifications[1][restriction_class[0]][restriction_type[0]]]})
                    boolean = df.eval('Value' + specifications[3][restriction_class[0]][restriction_type[0]] + 'Restriction')[0]

                    # - if condition is not met: continue loop
                    if not boolean and restriction_type[0] != len(restriction_class[1]) - 1:
                        continue

                    # - if one of the conditions is met and the restriction contains an 'OR' boolean statement:
                    if boolean and restriction_type[0] != len(restriction_class[1]) - 1 and specifications[-1][restriction_class[0]][restriction_type[0]] == 'or':
                        no_tidal_window = False
                        break
                    # - elif all the conditions are met and the restriction contains solely 'AND' boolean statements:
                    elif boolean and restriction_type[0] == len(restriction_class[1]) - 1:
                        no_tidal_window = False
                        break

                # - if condition is met: break the loop
                if boolean == True:
                    break

                # - else: restart the loop with predefined bool
                else:
                    boolean = True

            return restriction_class[0], no_tidal_window

        def times_vertical_tidal_window(vessel,route,axis=[],plot=False,sailing_time_correction=True,ukc_calc=False):
            """ Function: calculates the windows available to sail-in and -out of the port given the vertical tidal restrictions according to the tidal window policy.

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                    - plot: provide a visualization of the calculation for each vessel
                    - axis: axes class from the matplotlib package
                    - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

            """

            # Function used in the determination of the calculation of the sail-in-times given the policy for determining the vertical tidal windows
            def minimum_available_water_depth_along_route(vessel, route, axis=[], plot=False, sailing_time_correction=True, ukc_calc=False):
                """ Function: calculates the minimum available water depth (predicted/modelled/measured water level minus the local maintained bed level) along the route over time,
                              subtracted with the difference between the gross ukc and net ukc (hence: subtracted with additional safety margins consisting of vessel-related factors
                              and water level factors). The bottom-related factors are already accounted for in the use of the MBL instead of the actual depth.

                    Input:
                        - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                        - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                        - plot: provide a visualization of the calculation for each vessel
                        - axis: axes class from the matplotlib package
                        - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

                """

                # Continuation of the calculation of the minimum available water depth along the route of the vessel by predefining some parameters
                network = vessel.env.FG
                distance_to_next_node = 0
                ukc = []
                wdep_nodes = []
                t_min_wdep = network.nodes[route[0]]['Info']['Water level'][0]
                min_wdep = []

                # Start of calculation by looping over the nodes of the route
                for nodes in enumerate(route):
                    # - if a correction for the sailing time should be applied: the total distance should be keep track of
                    if sailing_time_correction and not ukc_calc:
                        if not nodes[1] == route[0]:
                            distance_to_next_node += pyproj.Geod(ellps='WGS84').inv(network.nodes[route[nodes[0] - 1]]['geometry'].x,
                                                                                    network.nodes[route[nodes[0] - 1]]['geometry'].y,
                                                                                    network.nodes[route[nodes[0]]]['geometry'].x,
                                                                                    network.nodes[route[nodes[0]]]['geometry'].y)[2]

                    # Sailing time is calculated by the distance of the route up to the specific node, divided by the sailing speed (if no correction for the sailing time should be applied: distance is always 0)
                    sailing_time_to_next_node = distance_to_next_node / vessel.v

                    # Importing some node specific data on the time series of the water levels, depth, and ukc policy (corrected for the sailing time if correction was applied)
                    t_wlev = [t - sailing_time_to_next_node for t in network.nodes[route[nodes[0]]]['Info']['Water level'][0]]
                    wlev = network.nodes[route[nodes[0]]]['Info']['Water level'][1]
                    depth = network.nodes[route[nodes[0]]]['Info']['MBL'][0]
                    types = network.nodes[nodes[1]]['Info']['Vertical tidal restriction']['Type']
                    specifications = network.nodes[nodes[1]]['Info']['Vertical tidal restriction']['Specification']

                    # Determine which restriction applies to vessel
                    restriction_index, _ = tidal_window_restriction_determinator(vessel, types, specifications)

                    # Calculate ukc policy based on the applied restriction
                    ukc_s = types[0][restriction_index][0]
                    ukc_p = types[1][restriction_index][0] * vessel.T_f
                    fwa = types[2][restriction_index][0] * vessel.T_f

                    # Interpolation of the available water depth over time and append data to list of the available water depths for the whole route
                    interp_wdep = sc.interpolate.CubicSpline(t_wlev, [y + depth - ukc_s - ukc_p - fwa for y in wlev])
                    if ukc_calc:
                        return interp_wdep, ukc_s, ukc_p, fwa
                    wdep = interp_wdep(t_min_wdep)
                    wdep_nodes.append(wdep)

                    # Add to axes of the plot (if plot is requested): the available water depths at all the nodes of the route
                    if plot:
                        axes.plot(t_min_wdep, wdep, color='lightskyblue', alpha=0.4)

                # Pick the minimum of the water depths for each time and each node
                min_wdep = [min(idx) for idx in zip(*wdep_nodes)]

                return t_min_wdep, min_wdep

            #Continuation of the calculation of the windows given the vertical tidal restrictions by setting some parameters
            if not ukc_calc:
                times_vertical_tidal_window = []
                water_depth_required = vessel.T_f + vessel.metadata['ukc'] #vessel draught + additional vessel-related factors (if applicable)

            #Calculation of the minimum available water depth along the route of the vessel
            if not ukc_calc:
                [new_t, min_wdep, _, _, _] = minimum_available_water_depth_along_route(vessel, route, axis)
            else:
                interp_wdep, ukc_s, ukc_p, fwa = minimum_available_water_depth_along_route(vessel, route, ukc_calc=True)
                return interp_wdep, ukc_s, ukc_p, fwa

            #Interpolation of the net ukc
            root_interp_water_level_at_edge = sc.interpolate.CubicSpline(new_t,[x-water_depth_required for x in min_wdep])

            #If there is not enough available water depth over time: no tidal window
            if np.max([x - water_depth_required for x in min_wdep]) < 0:
                times_vertical_tidal_window.append([vessel.waiting_time_start, 'Start']) #tidal restriction starts at t_start
                times_vertical_tidal_window.append([np.max(new_t), 'Stop']) #tidal restriction ends at t_end
            #Else if there is enough available water depth at certain moments: calculate tidal windows by looping over the roots of the interpolation
            else:
                for root in root_interp_water_level_at_edge.roots():
                    #-if root falls within time series of the data on the net ukc, and the net ukc a moment later than the root is higher than the required water depth:
                    if root > new_t[0] and root < new_t[-1] and min_wdep[bisect.bisect_right(new_t,root)] > water_depth_required:
                        # -if there are no values in the list of vertical tidal windows yet or there are values and the last value indicates that the tidal restriction has started and not ended and the net ukc a moment ealier than the root is less than the required water depth
                        if (times_vertical_tidal_window == [] or (times_vertical_tidal_window != [] and times_vertical_tidal_window[-1][1] != 'Stop')) and min_wdep[bisect.bisect_right(new_t,root)-1] < water_depth_required:
                            times_vertical_tidal_window.append([root, 'Stop']) #tidal restriction ends at t=root
                    # -if root falls within time series of the data on the net ukc, and the net ukc a moment later than the root is less than the required water depth:
                    elif root > new_t[0] and root < new_t[-1] and min_wdep[bisect.bisect_right(new_t,root)] < water_depth_required:
                        # -if there are no values in the list of vertical tidal windows yet or there are values and the last value indicates that the tidal restriction has ended and not ended and the net ukc a moment ealier than the root is higher than the required water depth
                        if (times_vertical_tidal_window == [] or (times_vertical_tidal_window != [] and times_vertical_tidal_window[-1][1] != 'Start')) and min_wdep[bisect.bisect_right(new_t,root)-1] > water_depth_required:
                            times_vertical_tidal_window.append([root, 'Start']) #tidal restriction starts at t=root

                #If the sail-in or -out-times given the vertical tidal restrictions are not empty: set the initial value at t=0
                if times_vertical_tidal_window != []:
                    #-if the first value in the list indicates that the tidal restriction starts: append that the tidal restriction ends at t=0
                    if times_vertical_tidal_window[0][1] == 'Start' and times_vertical_tidal_window[0][0] > vessel.waiting_time_start:
                            times_vertical_tidal_window.insert(0,[vessel.waiting_time_start, 'Stop'])
                    # -if the first value in the list indicates that the tidal restriction stop: append that the tidal restriction starts at t=0
                    if times_vertical_tidal_window[0][1] == 'Stop' and times_vertical_tidal_window[0][0] > vessel.waiting_time_start:
                            times_vertical_tidal_window.insert(0,[vessel.waiting_time_start, 'Start'])

            #If there are still no sail-in or -out-times (meaning the vessel can enter regardless the vertical tidal restriction): set start- and end-times
            if times_vertical_tidal_window == []:
                times_vertical_tidal_window.append([vessel.waiting_time_start, 'Stop']) #tidal restriction stops at t_start
                times_vertical_tidal_window.append([np.max(new_t), 'Start']) #tidal restriction starts at t_end

            #If plot is requested: plot the minimum available water depth, the required water depth of the vessel, the resulting vertical tidal windows, and add some lay-out
            if plot:
                axis.plot(new_t,min_wdep,color='deepskyblue')
                axis.plot([x[0] for x in times_vertical_tidal_window], (vessel.metadata['ukc']+vessel.T_f) * np.ones(len(times_vertical_tidal_window)), color='deepskyblue',marker='o',linestyle='None')
                axis.text(vessel.env.now+vessel.metadata['max_waiting_time'], 1.01*(vessel.metadata['ukc']+vessel.T_f), 'Required water depth', color='deepskyblue',horizontalalignment='center')
                axis.axhline((vessel.metadata['ukc']+vessel.T_f), color='deepskyblue', linestyle='--')
                axis.set_ylim([0,vessel.T_f+5])

            #Return the sail-in or -out-times given the vertical tidal restrictions over the route of the vessel
            return times_vertical_tidal_window

        def times_horizontal_tidal_window(vessel,route,axis,plot,sailing_time_correction,visualization_calculation):
            """ Function: calculates the windows available to sail-in and -out of the port given the horizontal tidal restrictions according to the tidal window policy.

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                    - plot: provide a visualization of the calculation for each vessel
                    - axis: axes class from the matplotlib package
                    - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

            """

            def tidal_windows_tightest_current_restriction_along_route(vessel, route, plot=False, axis=[], sailing_time_correction=True):
                """ Function: calculates the normative current restrictions along the route over time and calculates the resulting horizontal tidal windows from these locations.

                    Input:
                        - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                        - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                        - plot: provide a visualization of the calculation for each vessel
                        - axis: axes class from the matplotlib package
                        - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

                """

                # Predefining some parameters
                network = vessel.env.FG
                distance_to_next_node = 0
                mccur_nodes = []
                max_ccur = []
                list_of_nodes = []
                critcur_nodes = []
                t_max_ccur = network.nodes[route[0]]['Info']['Water level'][0]

                # Start of calculation by looping over the nodes of the route
                for nodes in enumerate(route):
                    # - if a correction for the sailing time should be applied: the total distance should be keep track of
                    if sailing_time_correction:
                        if not nodes[1] == route[0]:
                            distance_to_next_node += pyproj.Geod(ellps='WGS84').inv(network.nodes[route[nodes[0] - 1]]['geometry'].x,
                                                                                    network.nodes[route[nodes[0] - 1]]['geometry'].y,
                                                                                    network.nodes[route[nodes[0]]]['geometry'].x,
                                                                                    network.nodes[route[nodes[0]]]['geometry'].y)[2]

                    # Sailing time is calculated by the distance of the route up to the specific node, divided by the sailing speed (if no correction for the sailing time should be applied: distance is always 0)
                    sailing_time_to_next_node = distance_to_next_node / vessel.v

                    # If there is no horizontal tidal restriction at the specific node in the route: continue loop
                    if network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction'] == {}:
                        continue

                    # Else: importing some node specific data on the type and specifications of the horizontal tidal restriction and predefining some bools
                    no_tidal_window = True
                    boolean = True
                    types = network.nodes[nodes[1]]['Info']['Horizontal tidal restriction']['Type']
                    specifications = network.nodes[nodes[1]]['Info']['Horizontal tidal restriction']['Specification']

                    # Determine which restriction applies to vessel
                    restriction_index, no_tidal_window = tidal_window_restriction_determinator(vessel, types,specifications)

                    # If no horizontal tidal window applies to vessel at the specific node: continue loop over nodes of the route of the vessel
                    if no_tidal_window:
                        continue

                    # Predefining some parameters
                    ccur = []
                    crit_ccur = []
                    t_ccur = []
                    horizontal_tidal_window = []
                    crit_vel_horizontal_tidal_window = []
                    crit_ccur_flood_old = types[1][restriction_index][0]
                    crit_ccur_ebb_old = types[1][restriction_index][1]

                    ###!!!ATTENTION REQUIRED!!!###
                    restriction_on_route = False
                    if visualization_calculation:
                        restriction_on_route = True
                        if vessel.bound == 'inbound':
                            n1 = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][0]
                            n2 = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][1]
                        elif vessel.bound == 'outbound':
                            n2 = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][0]
                            n1 = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][1]
                    else:
                        for n1 in vessel.route[:nodes[0]]:
                            if n1 == network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction'][
                                'Specification'][5][restriction_index][0]:
                                for n2 in vessel.route[nodes[0]:]:
                                    if n2 == network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction'][
                                        'Specification'][5][restriction_index][1]:
                                        restriction_on_route = True
                                        break
                                break

                    # Else if there applies a horizontal tidal window and it has the type 'critical cross-current': continue tidal window calculation
                    if types[0][restriction_index] == 'Critical cross-current':
                        if crit_ccur_flood_old != -1:
                            crit_ccur_flood = crit_ccur_flood_old - vessel.metadata['max_cross_current']
                        else:
                            crit_ccur_flood = 10
                        if crit_ccur_ebb_old != -1:
                            crit_ccur_ebb = crit_ccur_ebb_old - vessel.metadata['max_cross_current']
                        else:
                            crit_ccur_ebb = 10

                        if nodes[1] != route[0] or visualization_calculation:
                            t_ccur = [t - sailing_time_to_next_node for t in network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Data'][n1][0]]
                            cur = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Data'][n1][1]
                            interp_ccur = sc.interpolate.CubicSpline(t_ccur, cur)
                            times_tidal_periods = [z[0] - sailing_time_to_next_node for z in network.nodes[route[nodes[0]]]['Info']['Tidal periods']]
                            tidal_periods = [z[1] for z in network.nodes[route[nodes[0]]]['Info']['Tidal periods']]
                            t_ccur_tidal_periods = [tidal_periods[bisect.bisect_right(times_tidal_periods,y + sailing_time_to_next_node)] if y + sailing_time_to_next_node <= times_tidal_periods[-1] else tidal_periods[-1] for y in t_ccur]
                            ccur.append([y - crit_ccur_flood if t_ccur_tidal_periods[x] == 'Ebb Start' else y - crit_ccur_ebb for x, y in enumerate(interp_ccur(t_max_ccur))])
                            crit_ccur.append([crit_ccur_flood if t_ccur_tidal_periods[x] == 'Ebb Start' else crit_ccur_ebb for x, y in enumerate(network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Data'][n1][1])])
                        if nodes[1] != route[-1] or visualization_calculation:
                            t_ccur = [t - sailing_time_to_next_node for t in network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Data'][n2][0]]
                            cur = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Data'][n2][1]
                            interp_ccur = sc.interpolate.CubicSpline(t_ccur, cur)
                            times_tidal_periods = [z[0] - sailing_time_to_next_node for z in network.nodes[route[nodes[0]]]['Info']['Tidal periods']]
                            tidal_periods = [z[1] for z in network.nodes[route[nodes[0]]]['Info']['Tidal periods']]
                            t_ccur_tidal_periods = [tidal_periods[bisect.bisect_right(times_tidal_periods,y + sailing_time_to_next_node)] if y + sailing_time_to_next_node <= times_tidal_periods[-1] else tidal_periods[-1] for y in t_ccur]
                            ccur.append([y - crit_ccur_flood if t_ccur_tidal_periods[x] == 'Ebb Start' else y - crit_ccur_ebb for x, y in enumerate(interp_ccur(t_max_ccur))])
                            crit_ccur.append([crit_ccur_flood if t_ccur_tidal_periods[x] == 'Ebb Start' else crit_ccur_ebb for x, y in enumerate(network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Data'][n2][1])])

                        mccur = [max(idx) for idx in zip(*ccur)]
                        critcur = [crit_ccur[np.argmax(val)][idx] for idx, val in enumerate(zip(*ccur))]
                        critcur_nodes.append(critcur)
                        mccur_nodes.append(mccur)

                        if plot:
                            axis.plot(t_max_ccur, [y + critcur[x] for x, y in enumerate(mccur)], color='lightcoral', alpha=0.4)
                            axis.plot(t_max_ccur, [y if y != 10 else None for y in critcur], color='lightcoral', linestyle='--', alpha=0.4)

                    elif types[0][restriction_index] == 'Point-based':
                        if type(crit_ccur_flood_old) != str:
                            crit_ccur_flood = crit_ccur_flood_old - vessel.metadata['max_cross_current']
                        if type(crit_ccur_ebb_old) != str:
                            crit_ccur_ebb = crit_ccur_ebb_old - vessel.metadata['max_cross_current']

                        if nodes[1] != route[0] or visualization_calculation:
                            t_ccur = [t - sailing_time_to_next_node for t in network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Data'][n1][0]]
                            times_tidal_periods = [z[0] - sailing_time_to_next_node for z in network.nodes[route[nodes[0]]]['Info']['Tidal periods']]
                            tidal_periods = [z[1] for z in network.nodes[route[nodes[0]]]['Info']['Tidal periods']]
                            t_ccur_tidal_periods = [tidal_periods[bisect.bisect_right(times_tidal_periods,y + sailing_time_to_next_node)] if y + sailing_time_to_next_node <= times_tidal_periods[-1] else tidal_periods[-1] for y in t_ccur]
                            ccur.append(network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Data'][n1][1])
                        if nodes[1] != route[-1] or visualization_calculation:
                            t_ccur = [t - sailing_time_to_next_node for t in network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Data'][n2][0]]
                            times_tidal_periods = [z[0] - sailing_time_to_next_node for z in network.nodes[route[nodes[0]]]['Info']['Tidal periods']]
                            tidal_periods = [z[1] for z in network.nodes[route[nodes[0]]]['Info']['Tidal periods']]
                            t_ccur_tidal_periods = [tidal_periods[bisect.bisect_right(times_tidal_periods,y + sailing_time_to_next_node)] if y + sailing_time_to_next_node <= times_tidal_periods[-1] else tidal_periods[-1] for y in t_ccur]
                            ccur.append(network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Data'][n2][1])

                        mccur = [max(idx) for idx in zip(*ccur)]
                        interp_ccur = sc.interpolate.CubicSpline(t_ccur, mccur)
                        mccur = interp_ccur(t_max_ccur)
                        previous_time = t_ccur[0]
                        p_flood = types[2][restriction_index][0]
                        p_ebb = types[2][restriction_index][1]
                        critcur = -10 * np.ones(len(mccur))
                        critcur_nodes.append(critcur)
                        mccur_nodes.append(critcur)

                        if type(crit_ccur_flood_old) != str:
                            interp_ccur_t1 = sc.interpolate.CubicSpline(t_ccur,[ccur - crit_ccur_flood * (1 + p_flood) for ccur in mccur])
                            interp_ccur_t2 = sc.interpolate.CubicSpline(t_ccur,[ccur - crit_ccur_flood * (1 - p_flood) for ccur in mccur])
                            crit_ccur_t1 = crit_ccur_flood * (1 + p_flood)
                            crit_ccur_t2 = crit_ccur_flood * (1 - p_flood)
                            roots1 = interp_ccur_t1.roots()
                            roots2 = interp_ccur_t2.roots()
                            t_ccur_tidal_periods1 = [tidal_periods[bisect.bisect_right(times_tidal_periods,y + sailing_time_to_next_node)] if y + sailing_time_to_next_node <= times_tidal_periods[-1] else tidal_periods[-1] for y in roots1]
                            t_ccur_tidal_periods2 = [tidal_periods[bisect.bisect_right(times_tidal_periods,y + sailing_time_to_next_node)] if y + sailing_time_to_next_node <= times_tidal_periods[-1] else tidal_periods[-1] for y in roots2]

                        if type(crit_ccur_ebb_old) != str:
                            interp_ccur_t3 = sc.interpolate.CubicSpline(t_ccur,[ccur - crit_ccur_ebb * (1 + p_ebb) for ccur in mccur])
                            interp_ccur_t4 = sc.interpolate.CubicSpline(t_ccur,[ccur - crit_ccur_ebb * (1 - p_ebb) for ccur in mccur])
                            crit_ccur_t3 = crit_ccur_ebb * (1 + p_ebb)
                            crit_ccur_t4 = crit_ccur_ebb * (1 - p_ebb)
                            roots3 = interp_ccur_t3.roots()
                            roots4 = interp_ccur_t4.roots()
                            t_ccur_tidal_periods3 = [tidal_periods[bisect.bisect_right(times_tidal_periods,y + sailing_time_to_next_node)] if y + sailing_time_to_next_node <= times_tidal_periods[-1] else tidal_periods[-1] for y in roots3]
                            t_ccur_tidal_periods4 = [tidal_periods[bisect.bisect_right(times_tidal_periods,y + sailing_time_to_next_node)] if y + sailing_time_to_next_node <= times_tidal_periods[-1] else tidal_periods[-1] for y in roots4]

                        previous_time = t_ccur[0]
                        for period in enumerate(times_tidal_periods):
                            if tidal_periods[period[0]] == 'Ebb Start':
                                next_previous_time_tidal_periods = bisect.bisect_right(times_tidal_periods,period[1]) - 2
                                if next_previous_time_tidal_periods < 0: next_previous_time_tidal_periods = 0
                                index_previous_time = bisect.bisect_right(t_ccur,previous_time - sailing_time_to_next_node) - 1
                                index_now = bisect.bisect_right(t_ccur, period[1] - sailing_time_to_next_node)
                                if crit_ccur_flood_old == -1:
                                    if crit_ccur_ebb_old != 'min':
                                        horizontal_tidal_window.append([times_tidal_periods[next_previous_time_tidal_periods], 'Stop'])
                                        crit_vel_horizontal_tidal_window.append(0)
                                    horizontal_tidal_window.append([period[1], 'Start'])
                                    crit_vel_horizontal_tidal_window.append(0)
                                elif crit_ccur_flood_old == 'min':
                                    if mccur[index_previous_time:index_now] != []:
                                        interp = sc.interpolate.CubicSpline(t_ccur[index_previous_time:index_now],[y - p_flood for y in mccur[index_previous_time:index_now]])
                                        times = [x for x in interp.roots() if (x >= t_ccur[index_previous_time] and x <= t_ccur[index_now])]
                                        if len(times) < 2: times = [t_ccur[index_previous_time], t_ccur[index_now]]
                                        horizontal_tidal_window.append([next(item for item in times if item is not None) + sailing_time_to_next_node,'Start'])
                                        crit_vel_horizontal_tidal_window.append(p_flood)
                                        horizontal_tidal_window.append([next(item for item in reversed(times) if item is not None) + sailing_time_to_next_node,'Stop'])
                                        crit_vel_horizontal_tidal_window.append(p_flood)
                                else:
                                    if mccur[index_previous_time:index_now] != []:
                                        if np.max(mccur[index_previous_time:index_now]) < crit_ccur_t1:
                                            horizontal_tidal_window.append([t_ccur[np.argmax(mccur[index_previous_time:index_now]) + index_previous_time] + sailing_time_to_next_node,'Stop'])
                                            crit_vel_horizontal_tidal_window.append(np.max(mccur[index_previous_time:index_now]))
                                        index_previous_time2 = np.argmax(mccur[index_previous_time:index_now]) + index_previous_time
                                        if np.min(mccur[index_previous_time2:index_now]) > crit_ccur_t2 and p_flood != 0 and crit_ccur_ebb_old != 'min':
                                            horizontal_tidal_window.append([t_ccur[np.argmin(mccur[index_previous_time2:index_now]) + index_previous_time2] + sailing_time_to_next_node,'Start'])
                                            crit_vel_horizontal_tidal_window.append(np.min(mccur[index_previous_time2:index_now]))
                                        elif p_flood == 0 and crit_ccur_ebb_old != 'min':
                                            horizontal_tidal_window.append([period[1], 'Start'])
                                            crit_vel_horizontal_tidal_window.append(0)
                                        elif crit_ccur_ebb_old == 'min':
                                            interp = sc.interpolate.CubicSpline(t_ccur[index_previous_time:index_now],[y - p_ebb for y in mccur[index_previous_time:index_now]])
                                            times = [x for x in interp.roots() if (x >= t_ccur[index_previous_time] and x <= t_ccur[index_now])]
                                            if len(times) < 2: times = [t_ccur[index_previous_time], t_ccur[index_now]]
                                            horizontal_tidal_window.append([next(item for item in times if item is not None) + sailing_time_to_next_node,'Start'])
                                            crit_vel_horizontal_tidal_window.append(p_ebb)

                            if tidal_periods[period[0]] == 'Flood Start':
                                next_previous_time_tidal_periods = bisect.bisect_right(times_tidal_periods,period[1]) - 2
                                if next_previous_time_tidal_periods < 0: next_previous_time_tidal_periods = 0
                                index_previous_time = bisect.bisect_right(t_ccur,previous_time - sailing_time_to_next_node) - 1
                                index_now = bisect.bisect_right(t_ccur, period[1] - sailing_time_to_next_node)
                                if crit_ccur_ebb_old == -1:
                                    if crit_ccur_flood_old != 'min':
                                        horizontal_tidal_window.append([times_tidal_periods[next_previous_time_tidal_periods], 'Stop'])
                                        crit_vel_horizontal_tidal_window.append(0)
                                    horizontal_tidal_window.append([period[1], 'Start'])
                                    crit_vel_horizontal_tidal_window.append(0)
                                elif crit_ccur_ebb_old == 'min':
                                    if mccur[index_previous_time:index_now] != []:
                                        interp = sc.interpolate.CubicSpline(t_ccur[index_previous_time:index_now],[y - p_ebb for y in mccur[index_previous_time:index_now]])
                                        times = [x for x in interp.roots() if (x >= t_ccur[index_previous_time] and x <= t_ccur[index_now])]
                                        if len(times) < 2: times = [t_ccur[index_previous_time], t_ccur[index_now]]
                                        horizontal_tidal_window.append([next(item for item in times if item is not None) + sailing_time_to_next_node,'Start'])
                                        crit_vel_horizontal_tidal_window.append(p_ebb)
                                        horizontal_tidal_window.append([next(item for item in reversed(times) if item is not None) + sailing_time_to_next_node,'Stop'])
                                        crit_vel_horizontal_tidal_window.append(p_ebb)
                                else:
                                    if mccur[index_previous_time:index_now] != []:
                                        if np.max(mccur[index_previous_time:index_now]) < crit_ccur_t3:
                                            horizontal_tidal_window.append([t_ccur[np.argmax(mccur[index_previous_time:index_now]) + index_previous_time] + sailing_time_to_next_node,'Stop'])
                                            crit_vel_horizontal_tidal_window.append(np.max(mccur[index_previous_time:index_now]))
                                        index_previous_time2 = np.argmax(mccur[index_previous_time:index_now]) + index_previous_time
                                        if np.min(mccur[index_previous_time2:index_now]) > crit_ccur_t4 and p_ebb != 0 and crit_ccur_flood_old != 'min':
                                            horizontal_tidal_window.append([t_ccur[np.argmin(mccur[index_previous_time2:index_now]) + index_previous_time2] + sailing_time_to_next_node,'Start'])
                                            crit_vel_horizontal_tidal_window.append(np.min(mccur[index_previous_time2:index_now]))
                                        elif p_ebb == 0 and crit_ccur_flood_old != 'min':
                                            horizontal_tidal_window.append([period[1], 'Start'])
                                            crit_vel_horizontal_tidal_window.append(0)
                                        elif crit_ccur_flood_old == 'min':
                                            interp = sc.interpolate.CubicSpline(t_ccur[index_previous_time:index_now],[y - p_flood for y in mccur[index_previous_time:index_now]])
                                            times = [x for x in interp.roots() if (x >= t_ccur[index_previous_time] and x <= t_ccur[index_now])]
                                            if len(times) < 2: times = [t_ccur[index_previous_time], t_ccur[index_now]]
                                            horizontal_tidal_window.append([next(item for item in times if item is not None) + sailing_time_to_next_node, 'Start'])
                                            crit_vel_horizontal_tidal_window.append(p_flood)

                            previous_time = period[1]

                        if type(crit_ccur_flood_old) != str:
                            for root in enumerate(roots1):
                                index = bisect.bisect_right(t_ccur, root[1])
                                if index >= len(t_ccur): index -= 1
                                if mccur[index] < crit_ccur_t1 and t_ccur_tidal_periods1[root[0]] == 'Ebb Start':
                                    if crit_ccur_flood_old != -1:
                                        horizontal_tidal_window.append([root[1] + sailing_time_to_next_node, 'Stop'])
                                        crit_vel_horizontal_tidal_window.append(crit_ccur_t1)

                        if type(crit_ccur_flood_old) != str:
                            for root in enumerate(roots2):
                                index = bisect.bisect_right(t_ccur, root[1])
                                if index >= len(t_ccur): index -= 1
                                if mccur[index] < crit_ccur_t2 and t_ccur_tidal_periods2[root[0]] == 'Ebb Start':
                                    if crit_ccur_flood_old != -1:
                                        horizontal_tidal_window.append([root[1] + sailing_time_to_next_node, 'Start'])
                                        crit_vel_horizontal_tidal_window.append(crit_ccur_t2)

                        if type(crit_ccur_ebb_old) != str:
                            for root in enumerate(roots3):
                                index = bisect.bisect_right(t_ccur, root[1])
                                if index >= len(t_ccur): index -= 1
                                if mccur[index] < crit_ccur_t3 and t_ccur_tidal_periods3[root[0]] == 'Flood Start':
                                    if crit_ccur_ebb_old != -1:
                                        horizontal_tidal_window.append([root[1] + sailing_time_to_next_node, 'Stop'])
                                        crit_vel_horizontal_tidal_window.append(crit_ccur_t3)

                        if type(crit_ccur_ebb_old) != str:
                            for root in enumerate(roots4):
                                index = bisect.bisect_right(t_ccur, root[1])
                                if index >= len(t_ccur): index -= 1
                                if mccur[index] < crit_ccur_t4 and t_ccur_tidal_periods4[root[0]] == 'Flood Start':
                                    if crit_ccur_ebb_old != -1:
                                        horizontal_tidal_window.append([root[1] + sailing_time_to_next_node, 'Start'])
                                        crit_vel_horizontal_tidal_window.append(crit_ccur_t4)

                        zipped_lists = zip(horizontal_tidal_window, crit_vel_horizontal_tidal_window)
                        sorted_pairs = sorted(zipped_lists)
                        tuples = zip(*sorted_pairs)
                        horizontal_tidal_window, crit_vel_horizontal_tidal_window = [list(tuple) for tuple in tuples]
                        indexes_to_be_removed = []
                        for time in range(len(horizontal_tidal_window)):
                            if time == 0:
                                continue
                            elif horizontal_tidal_window[time][1] == 'Stop' and horizontal_tidal_window[time - 1][1] == 'Stop':
                                indexes_to_be_removed.append(time - 1)
                            elif horizontal_tidal_window[time][1] == 'Start' and horizontal_tidal_window[time - 1][1] == 'Start':
                                indexes_to_be_removed.append(time)

                        for remove_index in list(reversed(indexes_to_be_removed)):
                            horizontal_tidal_window.pop(remove_index)
                            crit_vel_horizontal_tidal_window.pop(remove_index)

                        horizontal_tidal_windows.extend(horizontal_tidal_window)

                        if plot:
                            axis.plot(t_max_ccur, [y for x, y in enumerate(mccur)], color='rosybrown')
                            axis.plot([x[0] for x in horizontal_tidal_window], crit_vel_horizontal_tidal_window,color='sienna', marker='o', linestyle='None')

                max_ccur = [max(idx) for idx in zip(*mccur_nodes)]
                max_critcur = [critcur_nodes[np.argmax(val)][idx] for idx, val in enumerate(zip(*mccur_nodes))]
                if plot and max_critcur != []:
                    axis.plot(t_max_ccur, [y if y != 10 else None for y in max_critcur], color='lightcoral', alpha=0.4,linestyle='--')

                return t_max_ccur, max_ccur, max_critcur, horizontal_tidal_windows

            #Continuation of the calculation
            network = vessel.env.FG
            critical_cross_current_velocity = vessel.metadata['max_cross_current']
            max_cur = []
            for node in route:
                max_cur.append(np.max(network.nodes[node]['Info']['Current velocity'][1]))
            [new_t, max_ccur, max_crit_ccur,add_horizontal_tidal_windows] = tidal_windows_tightest_current_restriction_along_route(vessel,route,axis,plot,sailing_time_correction,visualization_calculation=visualization_calculation)
            if max_ccur != []:
                interp_max_cross_current = sc.interpolate.CubicSpline(new_t, [x for x in max_ccur])
                times_horizontal_tidal_window = []
                crit_ccurs_horizontal_tidal_window = []
                list_of_list_indexes = []
                list_indexes = [0, 1]
                for root in interp_max_cross_current.roots():
                    if root > new_t[0] and root < new_t[-1] and max_ccur[bisect.bisect_right(new_t,root)] < 0:
                        if times_horizontal_tidal_window == [] or (times_horizontal_tidal_window != [] and times_horizontal_tidal_window[-1][1] != 'Stop'):
                            times_horizontal_tidal_window.append([root, 'Stop'])
                            crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[bisect.bisect_right(new_t,root)])
                            list_of_list_indexes.append(0)
                    elif root > new_t[0] and root < new_t[-1] and max_ccur[bisect.bisect_right(new_t,root)] > 0:
                        if times_horizontal_tidal_window == [] or (times_horizontal_tidal_window != [] and times_horizontal_tidal_window[-1][1] != 'Start'):
                            times_horizontal_tidal_window.append([root, 'Start'])
                            crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[bisect.bisect_right(new_t,root)])
                            list_of_list_indexes.append(0)

                if times_horizontal_tidal_window == []:
                    times_horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                    crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[bisect.bisect_right(new_t, vessel.waiting_time_start)])
                    list_of_list_indexes.append(0)
                    times_horizontal_tidal_window.append([np.max(new_t), 'Start'])
                    crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[-1])
                    list_of_list_indexes.append(0)

                if times_horizontal_tidal_window[-1][1] == 'Start' and times_horizontal_tidal_window[-1][0] < np.max(new_t):
                    times_horizontal_tidal_window.append([np.max(new_t), 'Stop'])
                    crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[-1])
                    list_of_list_indexes.append(0)

                if add_horizontal_tidal_windows != []:
                    for time in add_horizontal_tidal_windows:
                        times_horizontal_tidal_window.append(time)
                        crit_ccurs_horizontal_tidal_window.append(0)
                        list_of_list_indexes.append(1)

                list_of_list_indexes = [x for _, x in sorted(zip(times_horizontal_tidal_window, list_of_list_indexes))]
                crit_ccurs_horizontal_tidal_window = [x for _, x in sorted(zip(times_horizontal_tidal_window, crit_ccurs_horizontal_tidal_window))]
                times_horizontal_tidal_window.sort()

                indexes_to_be_removed = []
                for list_index in list_indexes:
                    for time1 in range(len(times_horizontal_tidal_window)):
                        if times_horizontal_tidal_window[time1][1] == 'Start' and list_of_list_indexes[time1] == list_index:
                            for time2 in range(len(times_horizontal_tidal_window)):
                                if time2 > time1 and times_horizontal_tidal_window[time2][1] == 'Stop' and list_of_list_indexes[time2] == list_index:
                                    indexes = np.arange(time1 + 1, time2, 1)
                                    for index in indexes:
                                        indexes_to_be_removed.append(index)
                                    break

                indexes_to_be_removed.sort()
                indexes_to_be_removed = list(dict.fromkeys(indexes_to_be_removed))

                for remove_index in list(reversed(indexes_to_be_removed)):
                    times_horizontal_tidal_window.pop(remove_index)
                    list_of_list_indexes.pop(remove_index)
                    crit_ccurs_horizontal_tidal_window.pop(remove_index)

                indexes_to_be_removed = []
                for time in range(len(times_horizontal_tidal_window)):
                    if time == 0:
                        continue
                    elif times_horizontal_tidal_window[time][1] == 'Stop' and times_horizontal_tidal_window[time - 1][1] == 'Stop':
                        indexes_to_be_removed.append(time - 1)
                    elif times_horizontal_tidal_window[time][1] == 'Start' and times_horizontal_tidal_window[time - 1][1] == 'Start':
                        indexes_to_be_removed.append(time)

                for remove_index in list(reversed(indexes_to_be_removed)):
                    times_horizontal_tidal_window.pop(remove_index)
                    list_of_list_indexes.pop(remove_index)
                    crit_ccurs_horizontal_tidal_window.pop(remove_index)

                if times_horizontal_tidal_window[0][1] == 'Start' and times_horizontal_tidal_window[0][0] > vessel.waiting_time_start:
                    times_horizontal_tidal_window.insert(0,[vessel.waiting_time_start, 'Stop'])
                    crit_ccurs_horizontal_tidal_window.insert(0,max_crit_ccur[bisect.bisect_right(new_t, vessel.waiting_time_start)])
                elif times_horizontal_tidal_window[0][1] == 'Stop' and times_horizontal_tidal_window[0][0] > vessel.waiting_time_start:
                    times_horizontal_tidal_window.insert(0,[vessel.waiting_time_start, 'Start'])
                    crit_ccurs_horizontal_tidal_window.insert(0,max_crit_ccur[bisect.bisect_right(new_t, vessel.waiting_time_start)])

                if plot:
                    axis.plot(new_t,[y if y != 10 else None for y in max_crit_ccur], color='indianred', linestyle='--')
                    axis.plot([t for t in new_t], [y + max_crit_ccur[i] for i,y in enumerate(max_ccur)],color='indianred')
                    axis.set_ylabel('Cross-current velocity [m/s]', color='indianred')
                    y_loc = [y if y != 10 else None for y in max_crit_ccur][bisect.bisect_right(new_t, vessel.env.now + vessel.metadata['max_waiting_time'])]
                    if y_loc == None: y_loc = next(item for item in [y if y != 10 else None for y in max_crit_ccur][bisect.bisect_right(new_t, vessel.env.now + vessel.metadata['max_waiting_time']):] if item is not None)
                    if y_loc >= 0:
                        axis.plot([x[0] for x in times_horizontal_tidal_window],[y for y in crit_ccurs_horizontal_tidal_window], color='indianred', marker='o',linestyle='None')
                        axis.text(vessel.env.now + vessel.metadata['max_waiting_time'],y_loc, 'Critical cross-current', color='indianred',horizontalalignment='center')
                    axis.set_ylim([0,np.max(max_cur)])
            else:
                times_horizontal_tidal_window = []

            return times_horizontal_tidal_window

        def times_tidal_window(vessel,route,axes=[[],[]],plot=False,sailing_time_correction=True,ukc_calc=False):
            """ Function: calculates the windows available to sail-in and -out of the port by combining the tidal windows of the horizontal and vertical tidal restrictions given the tidal window polciy

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                    - plot: provide a visualization of the calculation for each vessel
                    - axes: list of two axes classes from the matplotlib package
                    - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

            """
            if ukc_calc:
                interp_wdep, ukc_s, ukc_p, fwa = times_vertical_tidal_window(vessel, route, ukc_calc=ukc_calc)
                return interp_wdep, ukc_s, ukc_p, fwa

            list_of_times_vertical_tidal_window = []
            list_of_times_horizontal_tidal_window = []

            list_of_times_vertical_tidal_window = times_vertical_tidal_window(vessel,route,axes[0],plot,sailing_time_correction)
            list_of_times_horizontal_tidal_window = times_horizontal_tidal_window(vessel,route,axes[1],plot,sailing_time_correction)

            list_indexes = [0,1]
            times_tidal_window = []
            list_of_list_indexes = []

            for time in list_of_times_vertical_tidal_window:
                times_tidal_window.append(time)
                list_of_list_indexes.append(0)
            for time in list_of_times_horizontal_tidal_window:
                times_tidal_window.append(time)
                list_of_list_indexes.append(1)

            list_of_list_indexes = [x for _, x in sorted(zip(times_tidal_window, list_of_list_indexes))]
            times_tidal_window.sort()

            indexes_to_be_removed = []
            for list_index in list_indexes:
                for time1 in range(len(times_tidal_window)):
                    if times_tidal_window[time1][1] == 'Start' and list_of_list_indexes[time1] == list_index:
                        for time2 in range(len(times_tidal_window)):
                            if time2 > time1 and times_tidal_window[time2][1] == 'Stop' and list_of_list_indexes[time2] == list_index:
                                indexes = np.arange(time1 + 1, time2, 1)
                                for index in indexes:
                                    indexes_to_be_removed.append(index)
                                break

            indexes_to_be_removed.sort()
            indexes_to_be_removed = list(dict.fromkeys(indexes_to_be_removed))

            for remove_index in list(reversed(indexes_to_be_removed)):
                times_tidal_window.pop(remove_index)
                list_of_list_indexes.pop(remove_index)

            indexes_to_be_removed = []
            for time in range(len(times_tidal_window)):
                if time == 0:
                    continue
                elif times_tidal_window[time][1] == 'Stop' and times_tidal_window[time - 1][1] == 'Stop':
                    indexes_to_be_removed.append(time - 1)
                elif times_tidal_window[time][1] == 'Start' and times_tidal_window[time - 1][1] == 'Start':
                    indexes_to_be_removed.append(time)

            for remove_index in list(reversed(indexes_to_be_removed)):
                times_tidal_window.pop(remove_index)
                list_of_list_indexes.pop(remove_index)

            return times_tidal_window

        # Continuation of the calculation of the available sail-in-times by setting the starting time and some lists
        if not ukc_calc:
            vessel.waiting_time_start = vessel.env.now
            axes = [[],[]]
        else:
            interp_wdep, ukc_s, ukc_p, fwa = times_tidal_window(vessel, route, ukc_calc=ukc_calc)
            return interp_wdep, ukc_s, ukc_p, fwa

        # If plot requested: create an empty figure
        if plot:
            fig, ax1 = plt.subplots(figsize=[10, 10])
            ax2 = ax1.twinx()
            axes = [ax1,ax2]

        # Running the above functions to determine the available-sail-in-times
        available_sail_in_times = times_tidal_window(vessel,route,axes,plot,sailing_time_correction)

        # If plot requested: add the following to the plot
        if plot:
            # Predefining some lists
            line = []
            linelist = []

            # Appending the tidal windows to the list by looping over the resulting available sail-in-times
            for t2 in range(len(available_sail_in_times)):
                if available_sail_in_times[t2][1] == 'Start' and t2 != 0:
                    linelist.append([available_sail_in_times[t2][0], available_sail_in_times[t2 - 1][0]])

            # Visualization of the list of tidal windows
            for lines in linelist:
                line, = ax1.plot([lines[1], lines[0]],[0.05,0.05],color='darkslateblue', marker='o')

            # Lay-out of the plot
            plt.title('Tidal window calculation for a '+str(vessel.type) + '-class vessel, sailing ' + str(vessel.bound))
            plt.xlim([vessel.env.now, vessel.env.now + 2 * vessel.metadata['max_waiting_time']])
            plt.legend([line], ['Tidal window'])
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('Experienced water depth [m]', color='deepskyblue')
            ax2.spines['left'].set_color('deepskyblue')
            ax2.spines['right'].set_color('indianred')
            plt.show()

        return available_sail_in_times

class Routeable:
    """Mixin class: Something with a route (networkx format)

    route: a networkx path"""

    def __init__(self, route, complete_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.route = route
        self.complete_path = complete_path

class IsLockWaitingArea(HasResource, Identifiable, Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
        self,
        node, #a string which indicates the location of the start of the waiting area
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)
        """Initialization"""

        waiting_area_resources = 100
        self.waiting_area = {node: simpy.PriorityResource(self.env, capacity=waiting_area_resources),}

class IsLockLineUpArea(HasResource, HasLength, Identifiable, Log):
    """Mixin class: Something has line-up area object properties as part of the lock complex [in SI-units]:
            creates a line-up area with the following resources:
                - enter_line_up_area: resource used when entering the line-up area (assures one-by-one entry of the line-up area by vessels)
                - line_up_area: resource with unlimited capacity used to formally request access to the line-up area
                - converting_while_in_line_up_area: resource used when requesting for an empty conversion of the lock chamber
                - pass_line_up_area: resource used to pass the second encountered line-up area"""

    def __init__(
        self,
        node, #a string which indicates the location of the start of the line-up area
        lineup_length, #a float which contains the length of the line-up area
        *args,
        **kwargs):

        super().__init__(length = lineup_length, remaining_length = lineup_length, *args, **kwargs)

        """Initialization"""
        # Lay-Out
        self.enter_line_up_area = {node: simpy.PriorityResource(self.env, capacity=1),} #used to regulate one by one entering of line-up area, so capacity must be 1
        self.line_up_area = {node: simpy.PriorityResource(self.env, capacity=100),} #line-up area itself, infinite capacity, as this is regulated by the HasLength, so capacity = inf
        self.converting_while_in_line_up_area = {node: simpy.PriorityResource(self.env, capacity=1),} #used to minimize the number of empty convertion requests by one by multiple waiting vessels, so capacity must be 1
        self.pass_line_up_area = {node: simpy.PriorityResource(self.env, capacity=1),} #used to prevent vessel from entering the lock before all previously locked vessels have passed the line-up area one by one, so capacity must be 1

class IsTurningBasin(HasResource, Identifiable, Log):
    def __init__(
        self,
        node, #a string which indicates the location of the start of the waiting area
        length,
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)
        """Initialization"""
        self.length = length
        self.turning_basin = {node: simpy.PriorityResource(self.env, capacity=1),}

class IsLock(HasResource, HasLength, Identifiable, Log):
    """Mixin class: Something which has lock chamber object properties as part of a lock complex [in SI-units] """

    def __init__(
        self,
        node_1, #a string which indicates the location of the first pair of lock doors
        node_2, #a string which indicates the center of the lock chamber
        node_3, #a string which indicates the location of the second pair of lock doors
        lock_length, #a float which contains the length of the lock chamber
        lock_width, #a float which contains the width of the lock chamber
        lock_depth, #a float which contains the depth of the lock chamber
        doors_open, #a float which contains the time it takes to open the doors
        doors_close, #a float which contains the time it takes to close the doors
        wlev_dif, #a float or list of floats which resembles the water level difference over the lock
        disch_coeff, #a float which contains the discharge coefficient of filling system
        opening_area, #a float which contains the cross-sectional area of filling system
        opening_depth, #a float which contains the depth at which filling system is located
        simulation_start, #a datetime which contains the simulation start time
        operating_time,
        grav_acc = 9.81, #a float which contains the gravitational acceleration
        *args,
        **kwargs):

        """Initialization"""
        # Properties
        self.lock_length = lock_length
        self.lock_width = lock_width
        self.lock_depth = lock_depth
        self.wlev_dif = wlev_dif
        self.disch_coeff = disch_coeff
        self.grav_acc = grav_acc
        self.opening_area = opening_area
        self.opening_depth = opening_depth
        self.operating_time = operating_time
        self.simulation_start = simulation_start.timestamp()

        super().__init__(length = lock_length, remaining_length = lock_length, *args, **kwargs)

        self.doors_1 = {node_1: simpy.PriorityResource(self.env, capacity = 1),} #Only one ship can pass at a time: capacity = 1, request can have priority
        self.doors_2 = {node_3: simpy.PriorityResource(self.env, capacity = 1),} #Only one ship can pass at a time: capacity = 1, request can have priority

        # Operating
        self.doors_open = doors_open
        self.doors_close = doors_close

        # Water level
        assert node_1 != node_3

        self.node_1 = node_1
        self.node_3 = node_3
        self.water_level = random.choice([node_1, node_3])

    def operation_time(self, environment):
        """ Function which calculates the operation time:
                based on the constant or nearest in the signal of the water level difference

            Input:
                - environment: see init function"""

        if type(self.wlev_dif) == list: #picks the wlev_dif from measurement signal closest to the discrete time
            operating_time = (2*self.lock_width*self.lock_length*abs(self.wlev_dif[1][np.abs(self.wlev_dif[0]-(environment.now-self.simulation_start)).argmin()]))/(self.disch_coeff*self.opening_area*math.sqrt(2*self.grav_acc*self.opening_depth))

        elif type(self.wlev_dif) == float or type(self.wlev_dif) == int: #constant water level difference
            operating_time = (2*self.lock_width*self.lock_length*abs(self.wlev_dif))/(self.disch_coeff*self.opening_area*math.sqrt(2*self.grav_acc*self.opening_depth))

        return operating_time

    def convert_chamber(self, environment, new_level, number_of_vessels):
        """ Function which converts the lock chamber and logs this event.

            Input:
                - environment: see init function
                - new_level: a string which represents the node and indicates the side at which the lock is currently levelled
                - number_of_vessels: the total number of vessels which are levelled simultaneously"""

        # Close the doors
        self.log_entry("Lock doors closing start", environment.now, number_of_vessels, self.water_level)
        yield environment.timeout(self.doors_close)
        self.log_entry("Lock doors closing stop", environment.now, number_of_vessels, self.water_level)

        # Convert the chamber
        self.log_entry("Lock chamber converting start", environment.now, number_of_vessels, self.water_level)

        # Water level will shift
        yield environment.timeout(self.operation_time(environment))
        self.change_water_level(new_level)
        self.log_entry("Lock chamber converting stop", environment.now, number_of_vessels, self.water_level)
        # Open the doors
        self.log_entry("Lock doors opening start", environment.now, number_of_vessels, self.water_level)
        yield environment.timeout(self.doors_open)
        self.log_entry("Lock doors opening stop", environment.now, number_of_vessels, self.water_level)

    def change_water_level(self, side):
        """ Function which changes the water level in the lock chamber and priorities in queue """

        self.water_level = side

        for request in self.resource.queue:
            request.priority = -1 if request.priority == 0 else 0

            if request.priority == -1:
                self.resource.queue.insert(
                    0, self.resource.queue.pop(self.resource.queue.index(request))
                )
            else:
                self.resource.queue.insert(
                    -1, self.resource.queue.pop(self.resource.queue.index(request))
                )

class PassSection():
    def release_access_previous_section(vessel, origin):
        for n in reversed(vessel.route[:vessel.route.index(origin)]):
            if 'Junction' in vessel.env.FG.nodes[n]:
                junction = vessel.env.FG.nodes[n]['Junction'][0]
                for section in enumerate(junction.section):
                    if origin not in list(section[1].keys()):
                        continue
                    section[1][origin].release(vessel.request_access_section)

                    if junction.type[section[0]] == 'one-way_traffic':
                        if 'access1' not in dir(junction):
                            junction = vessel.env.FG.nodes[origin]['Junction'][0]
                            for section in enumerate(junction.section):
                                if n not in list(section[1].keys()):
                                    continue
                                junction.access2[0][n].release(vessel.request_access_entrance_section) #section[0]
                                junction.access1[0][origin].release(vessel.request_access_exit_section) #section[0]
                        else:
                            junction.access1[0][n].release(vessel.request_access_entrance_section) #section[0]
                            junction.access2[0][origin].release(vessel.request_access_exit_section) #section[0]
                        break
                break
        return

    def request_access_next_section(vessel, origin, destination):
        for n in vessel.route[vessel.route.index(destination):]:
            if 'Junction' in vessel.env.FG.nodes[n]:
                junction = vessel.env.FG.nodes[origin]['Junction'][0]
                for section in enumerate(junction.section):
                    if n not in list(section[1].keys()):
                        continue
                    vessel.stopping_distance = 15 * vessel.L
                    vessel.stopping_time = vessel.stopping_distance / vessel.v
                    if section[1][n].users != [] and (section[1][n].users[-1].ta + vessel.stopping_time) > vessel.env.now:
                        vessel.request_access_section = section[1][n].request()
                        section[1][n].users[-1].id = vessel.id
                        section[1][n].users[-1].ta = (section[1][n].users[-2].ta + vessel.stopping_time)
                        yield vessel.env.timeout((section[1][n].users[-2].ta + vessel.stopping_time) - vessel.env.now)
                    else:
                        vessel.request_access_section = section[1][n].request()
                        section[1][n].users[-1].ta = vessel.env.now
                        section[1][n].users[-1].id = vessel.id

                    if junction.type[section[0]] == 'one-way_traffic':
                        if 'access1' not in dir(junction):
                            junction = vessel.env.FG.nodes[n]['Junction'][0]
                            for section in enumerate(junction.section):
                                if origin not in list(section[1].keys()):
                                    continue

                                vessel.request_access_entrance_section = junction.access2[0][origin].request() #section[0]
                                junction.access2[0][origin].users[-1].id = vessel.id #section[0]
                                vessel.request_access_exit_section = junction.access1[0][n].request() #section[0]
                                junction.access1[0][n].users[-1].id = vessel.id #section[0]

                        else:
                            vessel.request_access_entrance_section = junction.access1[0][origin].request() #section[0]
                            junction.access1[0][origin].users[-1].id = vessel.id #section[0]
                            vessel.request_access_exit_section = junction.access2[0][n].request() #section[0]
                            junction.access2[0][n].users[-1].id = vessel.id #section[0]
                        break
                break
        return

class PassTerminal:
    def waiting_time_for_tidal_window(vessel,route,max_waiting_time = True,delay=0,out=False,plot=False):
        if 'sail_in_times' not in dir(vessel):
            vessel.sail_in_times = VesselTrafficService.provide_sail_in_times_tidal_window(vessel,route=route,out=out,plot=plot)
        if out:
            sail_in_times = vessel.sail_in_times
            vessel.sail_in_times = vessel.sail_out_times

        waiting_time = 0
        current_time = vessel.env.now+delay
        for t in range(len(vessel.sail_in_times)):
            if vessel.sail_in_times[t][1] == 'Start':
                if t == len(vessel.sail_in_times)-1:
                    waiting_time = vessel.sail_in_times[t][0] - current_time
                    break
                else:
                    continue
            if current_time >= vessel.sail_in_times[t][0]:
                waiting_time = 0
                if t == len(vessel.sail_in_times)-1 or current_time < vessel.sail_in_times[t+1][0]:
                    break
            elif current_time <= vessel.sail_in_times[t][0]:
                if current_time < vessel.sail_in_times[t-1][0]:
                    waiting_time = 0
                else:
                    waiting_time = vessel.sail_in_times[t][0] - current_time
                break
            elif t == len(vessel.sail_in_times) - 1:
                waiting_time = vessel.sail_in_times[t][0] - current_time
            else:
                continue
        if out:
            vessel.sail_in_times = sail_in_times
        if not out:
            network = vessel.env.FG
            distance_to_node = 0
            route.reverse()
            for n in range(len(route)):
                if n == 0:
                    continue
                elif n == len(route)-1:
                    break
                distance_to_node += pyproj.Geod(ellps='WGS84').inv(network.nodes[route[n - 1]]['geometry'].x,
                                                                   network.nodes[route[n - 1]]['geometry'].y,
                                                                   network.nodes[route[n]]['geometry'].x,
                                                                   network.nodes[route[n]]['geometry'].y)[2]

            sailing_time = distance_to_node / vessel.v
            delay = sailing_time + 2 * vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60 + waiting_time
            current_time = vessel.env.now + delay
            if 'sail_out_times' not in dir(vessel):
                vessel.bound = 'outbound'
                vessel.sail_out_times = VesselTrafficService.provide_sail_in_times_tidal_window(vessel, route, out=True, plot=plot)
                vessel.bound = 'inbound'
            for t in range(len(vessel.sail_out_times)):
                if vessel.sail_out_times[t][1] == 'Start':
                    continue
                if current_time >= vessel.sail_out_times[t][0]:
                    if t == len(vessel.sail_out_times)-1 or current_time < vessel.sail_out_times[t+1][0]:
                        break
                elif current_time <= vessel.sail_out_times[t][0]:
                    if vessel.sail_out_times[t][0]-current_time >= vessel.metadata['max_waiting_time']:
                        waiting_time = vessel.metadata['max_waiting_time']
                    else:
                        break
                elif t == len(vessel.sail_out_times)-1:
                    waiting_time = vessel.metadata['max_waiting_time']
                else:
                    continue
            route.reverse()
        return waiting_time

    def move_to_anchorage(vessel,node):
        network = vessel.env.FG
        vessel.waiting_in_anchorage = True
        nodes_of_anchorages = []
        capacity_of_anchorages = []
        users_of_anchorages = []
        sailing_distances_from_anchorages = []
        for node_anchorage in network.nodes:
            if 'Anchorage' in network.nodes[node_anchorage]:
                nodes_of_anchorages.append(node_anchorage)
                capacity_of_anchorages.append(vessel.env.FG.nodes[node_anchorage]['Anchorage'][0].anchorage_area[node_anchorage].capacity)
                users_of_anchorages.append(len(vessel.env.FG.nodes[node_anchorage]['Anchorage'][0].anchorage_area[node_anchorage].users))
                route_from_anchorage = nx.dijkstra_path(vessel.env.FG, node_anchorage, vessel.route[-1])
                sailing_distance_from_anchorage = 0
                for route_node in enumerate(route_from_anchorage):
                    if route_node[0] == 0:
                        continue
                    _, _, distance = vessel.wgs84.inv(vessel.env.FG.nodes[route_from_anchorage[route_node[0]-1]]['geometry'].x,
                                                      vessel.env.FG.nodes[route_from_anchorage[route_node[0]-1]]['geometry'].y,
                                                      vessel.env.FG.nodes[route_from_anchorage[route_node[0]]]['geometry'].x,
                                                      vessel.env.FG.nodes[route_from_anchorage[route_node[0]]]['geometry'].y)
                    sailing_distance_from_anchorage += distance
                sailing_distances_from_anchorages.append(sailing_distance_from_anchorage)

        sorted_nodes_anchorages = [nodes for (distances,nodes) in sorted(zip(sailing_distances_from_anchorages, nodes_of_anchorages))]
        sorted_users_of_anchorages = [nodes for (distances,nodes) in sorted(zip(sailing_distances_from_anchorages, users_of_anchorages))]
        sorted_capacity_of_anchorages = [nodes for (distances,nodes) in sorted(zip(sailing_distances_from_anchorages, capacity_of_anchorages))]

        node_anchorage = sorted_nodes_anchorages[np.argmin(sailing_distances_from_anchorages)]
        for node_anchorage_area in enumerate(sorted_nodes_anchorages):
            if sorted_users_of_anchorages[node_anchorage_area[0]] < sorted_capacity_of_anchorages[node_anchorage_area[0]]:
                node_anchorage = node_anchorage_area[1]
                break

        if node_anchorage != node_anchorage_area[1]:
           vessel.return_to_sea = True
           vessel.waiting_time = vessel.metadata['max_waiting_time']
        anchorage = network.nodes[node_anchorage]['Anchorage'][0]
        vessel.route_after_anchorage = []
        vessel.true_origin = vessel.route[0]
        current_time = vessel.env.now
        vessel.route_after_anchorage = nx.dijkstra_path(vessel.env.FG, node_anchorage, vessel.route[-1])
        vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route[-1], vessel.route[0])
        yield from Movable.pass_edge(vessel,vessel.route[node], vessel.route[node+1])
        vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[node+1], node_anchorage)
        vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[node]]
        vessel.env.process(vessel.move())
        return

    def pass_anchorage(vessel, node):
        network = vessel.env.FG
        anchorage = network.nodes[node]['Anchorage'][0]
        yield from Movable.pass_edge(vessel, vessel.route[vessel.route.index(node) - 1],
                                     vessel.route[vessel.route.index(node)])

        vessel.anchorage_access = anchorage.anchorage_area[node].request()
        yield vessel.anchorage_access

        anchorage.log_entry("Vessel arrival", vessel.env.now, len(anchorage.anchorage_area[node].users),nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )

        current_time = vessel.env.now
        ukc = VesselTrafficService.provide_ukc_clearance(vessel,node)
        vessel.log_entry("Waiting in anchorage start", vessel.env.now, ukc,nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
        edge = vessel.route_after_anchorage[-2], vessel.route_after_anchorage[-1]
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        sailing_distance = 0
        for nodes in enumerate(vessel.route_after_anchorage[:-1]):
            _, _, distance = vessel.wgs84.inv(vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0]]]['geometry'].x,
                                              vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0]]]['geometry'].y,
                                              vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0] + 1]]['geometry'].x,
                                              vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0] + 1]]['geometry'].y)
            sailing_distance += distance

        if vessel.return_to_sea == False and vessel.waiting_for_availability_terminal:
            yield vessel.waiting_time_in_anchorage | vessel.env.timeout(vessel.metadata['max_waiting_time'])
            new_current_time = vessel.env.now
            if new_current_time - current_time >= vessel.metadata['max_waiting_time']:
                vessel.return_to_sea = True
                vessel.waiting_time = 0
                if terminal.type == 'quay': terminal = terminal.terminal[edge[0]]
                elif terminal.type == 'jetty': terminal = terminal.terminal[vessel.index_jetty_position][edge[0]]
                terminal.release(vessel.waiting_time_in_anchorage)
            else:
                if terminal.type == 'quay':
                    vessel.index_quay_position, _ = PassTerminal.pick_minimum_length(vessel, terminal)
                    PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)
                    terminal = terminal.terminal[edge[0]]
                elif terminal.type == 'jetty':
                    terminal = terminal.terminal[vessel.index_jetty_position][edge[0]]
                    vessel.access_terminal = terminal.request(priority=-1)
                    terminal.release(vessel.waiting_time_in_anchorage)
                    yield vessel.access_terminal

                vessel.sail_in_times = VesselTrafficService.provide_sail_in_times_tidal_window(vessel, vessel.route_after_anchorage,out=False)
                for t in range(len(vessel.sail_in_times)):
                    if vessel.sail_in_times[t][1] == 'Start':
                        if t == len(vessel.sail_in_times) - 1:
                            waiting_time = vessel.sail_in_times[t][0] - current_time
                            if waiting_time >= vessel.metadata['max_waiting_time']:
                                vessel.return_to_sea = True
                                vessel.waiting_time = 0
                                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                    terminal.release(vessel.access_terminal)
                            else:
                                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                    terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.sail_in_times[t][0] - new_current_time
                                    vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                    waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel, True,horizontal_tidal_window=True,max_waiting_time=True,out=True,route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=True)
                                    terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                    terminal.users[-1].type = vessel.type
                                    yield vessel.env.timeout(vessel.sail_in_times[t][0] - new_current_time)
                            break
                        else:
                            continue

                    if new_current_time >= vessel.sail_in_times[t][0]:
                        if t == len(vessel.sail_in_times) - 1 or new_current_time < vessel.sail_in_times[t + 1][0]:
                            if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
                                vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel, True,horizontal_tidal_window=True,max_waiting_time=True,out=True,route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=True)
                                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                terminal.users[-1].type = vessel.type
                            break

                    elif new_current_time <= vessel.sail_in_times[t][0]:
                        if new_current_time < vessel.sail_in_times[t - 1][0]:
                            if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
                                vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel, True,horizontal_tidal_window=True,max_waiting_time=True,out=True,route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=True)
                                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                terminal.users[-1].type = vessel.type
                        else:
                            waiting_time = vessel.sail_in_times[t][0] - current_time
                            if waiting_time >= vessel.metadata['max_waiting_time']:
                                vessel.return_to_sea = True
                                vessel.waiting_time = 0
                                terminal.release(vessel.access_terminal)
                            else:
                                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                    terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.sail_in_times[t][0] - new_current_time
                                    vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                    waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel, True,horizontal_tidal_window=True,max_waiting_time=True,out=True,route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=True)
                                    terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                    terminal.users[-1].type = vessel.type
                                    yield vessel.env.timeout(vessel.sail_in_times[t][0] - new_current_time)
                        break

                    elif t == len(vessel.sail_in_times) - 1:
                        waiting_time = vessel.sail_in_times[t][0] - current_time
                        if waiting_time >= vessel.metadata['max_waiting_time']:
                            vessel.return_to_sea = True
                            vessel.waiting_time = 0
                            terminal.release(vessel.access_terminal)
                        else:
                            if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.sail_in_times[t][0] - new_current_time
                                vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel, True,horizontal_tidal_window=True,max_waiting_time=True,out=True,route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=True)
                                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                terminal.users[-1].type = vessel.type
                                yield vessel.env.timeout(vessel.sail_in_times[t][0] - new_current_time)
                    else:
                        continue

        elif vessel.return_to_sea == False and vessel.waiting_time:
            if terminal.type == 'jetty':
                terminal = terminal.terminal[vessel.index_jetty_position][edge[0]]
                terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.waiting_time
                vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel, True,horizontal_tidal_window=True,max_waiting_time=True, out=True,route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=True)
                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                terminal.users[-1].type = vessel.type
            elif terminal.type == 'quay':
                PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)
            new_current_time = vessel.env.now
            if new_current_time - current_time >= vessel.metadata['max_waiting_time']:
                vessel.return_to_sea = True
                vessel.waiting_time = 0
            yield vessel.env.timeout(vessel.waiting_time)

        if vessel.return_to_sea == False:
            ukc = VesselTrafficService.provide_ukc_clearance(vessel,node)
            vessel.log_entry("Waiting in anchorage stop", vessel.env.now, ukc, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
            vessel.route = vessel.route_after_anchorage
            PassSection.release_access_previous_section(vessel, vessel.route[0])
            yield from PassSection.request_access_next_section(vessel, vessel.route[0], vessel.route[1])
            vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
            vessel.env.process(vessel.move())
            anchorage.anchorage_area[node].release(vessel.anchorage_access)
            anchorage.log_entry("Vessel departure", vessel.env.now, len(anchorage.anchorage_area[node].users),nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
        else:
            if 'waiting_time_in_anchorage' in dir(vessel):
                vessel.waiting_time_in_anchorage.cancel()
            yield vessel.env.timeout(vessel.waiting_time)
            ukc = VesselTrafficService.provide_ukc_clearance(vessel,node)
            vessel.log_entry("Waiting in anchorage stop", vessel.env.now, ukc, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[vessel.route.index(node)], vessel.true_origin)
            PassSection.release_access_previous_section(vessel, vessel.route[0])
            yield from PassSection.request_access_next_section(vessel, vessel.route[0], vessel.route[1])
            vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
            vessel.env.process(vessel.move())
            anchorage.anchorage_area[node].release(vessel.anchorage_access)
            anchorage.log_entry("Vessel departure", vessel.env.now, len(anchorage.anchorage_area[node].users),nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]],)

    def pick_minimum_length(vessel, terminal):
        available_quay_lengths = [0]
        aql = terminal.available_quay_lengths
        index_quay_position = 0
        move_to_anchorage = False
        for index in range(len(aql)):
            if index == 0 or aql[index][1] == aql[index - 1][1] or aql[index][0] == 1:
                if index == len(aql) - 1 and not index_quay_position:
                    move_to_anchorage = True
                continue

            available_quay_lengths.append(aql[index][1] - aql[index - 1][1])

            for jndex in range(len(available_quay_lengths)):
                if vessel.L <= available_quay_lengths[jndex]:
                    index_quay_position = index
                    break

                elif jndex == len(available_quay_lengths) - 1 and not index_quay_position:
                    move_to_anchorage = True

            if index_quay_position != 0:
                break

        return index_quay_position, move_to_anchorage

    def calculate_quay_length_level(terminal):
        aql = terminal.available_quay_lengths
        available_quay_lengths = [0]
        for i in range(len(aql)):
            if i == 0 or aql[i][1] == aql[i - 1][1] or aql[i][0] == 1:
                if i == len(aql) - 1:
                    new_level = available_quay_lengths[-1]
                continue

            available_quay_lengths.append(aql[i][1] - aql[i - 1][1])
            new_level = np.max(available_quay_lengths)
        return new_level

    def adjust_available_quay_lengths(vessel, terminal, index_quay_position):
        aql = terminal.available_quay_lengths
        old_level = PassTerminal.calculate_quay_length_level(terminal)
        if aql[index_quay_position - 1][0] == 0:
            aql[index_quay_position - 1][0] = 1

        if aql[index_quay_position][0] == 0 and aql[index_quay_position][1] == aql[index_quay_position - 1][1] + vessel.L:
            aql[index_quay_position][0] = 1
        else:
            aql.insert(index_quay_position, [1, vessel.L + aql[index_quay_position - 1][1]])
            aql.insert(index_quay_position + 1, [0, vessel.L + aql[index_quay_position - 1][1]])

        terminal.available_quay_lengths = aql
        vessel.quay_position = 0.5 * vessel.L + aql[index_quay_position - 1][1]
        new_level = PassTerminal.calculate_quay_length_level(terminal)
        if old_level != new_level and vessel.waiting_in_anchorage != True:
            terminal.length.get(old_level - new_level)
        elif vessel.waiting_in_anchorage == True:
            new_level = old_level-vessel.L-new_level
            if new_level < 0:
                terminal.length.put(-new_level)
            elif new_level > 0:
                terminal.length.get(new_level)
        return

    def request_terminal_access(vessel, edge, node):
        node = vessel.route.index(node)
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        vessel.bound = 'inbound' ##to_be_removed
        vessel.move_to_anchorage = False
        vessel.waiting_in_anchorage = False
        vessel.waiting_for_availability_terminal = False

        def checks_waiting_time_due_to_tidal_window(vessel, route, node, maximum_waiting_time = False):
            vessel.waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel,vertical_tidal_window=True,route=route,horizontal_tidal_window = True,max_waiting_time = True,delay=0,out=False,plot=False)
            if vessel.waiting_time >= vessel.metadata['max_waiting_time'] and maximum_waiting_time:
                vessel.return_to_sea = True
                vessel.waiting_time = 0
            else:
                vessel.return_to_sea = False

        checks_waiting_time_due_to_tidal_window(vessel, route = vessel.route, node = node, maximum_waiting_time=True)
        available_turning_basin = False
        for basin in vessel.route:
            if 'Turning Basin' in vessel.env.FG.nodes[basin].keys():
                turning_basin = vessel.env.FG.nodes[basin]['Turning Basin'][0]
                if turning_basin.length >= vessel.L:
                    available_turning_basin = True
                    break

        if available_turning_basin == False:
            vessel.return_to_sea = True
            vessel.waiting_time = 0

        if not vessel.return_to_sea:
            if terminal.type == 'jetty':
                minimum_waiting_time = []
                vessels_waiting = []

                for jetty in enumerate(terminal.terminal):
                    if vessel.L > terminal.jetty_lengths[jetty[0]]:
                        vessel.move_to_anchorage = True
                        continue
                    if jetty[1][edge[0]].users == [] and jetty[1][edge[0]].queue == []:
                        vessel.index_jetty_position = jetty[0]
                        vessel.move_to_anchorage = False
                        break
                    else:
                        if jetty[1][edge[0]].queue == []:
                            minimum_waiting_time.append(jetty[1][edge[0]].users[-1].etd)
                        else:
                            minimum_waiting_time.append(0)
                        vessels_waiting.append(len(jetty[1][edge[0]].queue))
                        if jetty[0] != len(terminal.terminal)-1:
                            continue
                    vessel.move_to_anchorage = True

            elif terminal.type == 'quay':
                aql = terminal.available_quay_lengths
                if terminal.length.get_queue == []:
                    vessel.index_quay_position,vessel.move_to_anchorage = PassTerminal.pick_minimum_length(vessel, terminal)
                else:
                    vessel.move_to_anchorage = True

            if vessel.waiting_time and not vessel.move_to_anchorage:
                yield from PassTerminal.move_to_anchorage(vessel, node)

            elif vessel.move_to_anchorage:
                vessel.waiting_for_availability_terminal = True

                if terminal.type == 'quay':
                    vessel.waiting_time_in_anchorage = terminal.length.get(vessel.L)

                elif terminal.type == 'jetty':
                    vessel.index_jetty_position = []
                    indices_empty_queue_for_jetty = [waiting_vessels[0] for waiting_vessels in enumerate(vessels_waiting) if waiting_vessels[1] == 0]
                    if indices_empty_queue_for_jetty != []:
                        min_minimum_waiting_time = minimum_waiting_time[indices_empty_queue_for_jetty[0]]  # vessel.env.now+vessel.metadata['max_waiting_time']
                        for index in indices_empty_queue_for_jetty:
                            if minimum_waiting_time[index] <= min_minimum_waiting_time and vessel.L <= terminal.jetty_lengths[index]:
                                min_minimum_waiting_time = minimum_waiting_time[index]
                                vessel.index_jetty_position = index
                    if vessel.index_jetty_position == []:
                        indexes = []
                        for length in enumerate(terminal.jetty_lengths):
                            if vessel.L <= length[1]:
                                indexes.append(length[0])
                        if indexes != []:
                            vessel.index_jetty_position = np.min([y[0] for y in enumerate(vessels_waiting) if y[0] in indexes])

                    if vessel.index_jetty_position != []:
                        vessel.waiting_time_in_anchorage = terminal.terminal[vessel.index_jetty_position][edge[0]].request()
                    else:
                        vessel.return_to_sea = True
                        vessel.waiting_time = 0

                yield from PassTerminal.move_to_anchorage(vessel, node)

            else:
                if terminal.type == 'quay':
                    PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)

            if vessel.return_to_sea == False:
                if terminal.type == 'quay':
                    terminal = terminal.terminal[edge[0]]
                elif terminal.type == 'jetty':
                    terminal = terminal.terminal[vessel.index_jetty_position][edge[0]]

                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty' and vessel.waiting_for_availability_terminal == True:
                    pass
                else:
                    vessel.access_terminal = terminal.request()
                    yield vessel.access_terminal

                    if vessel.env.FG.edges[edge]["Terminal"][0].type == 'quay':
                        vessel.env.FG.edges[edge]["Terminal"][0].available_quay_lengths = aql

                    if vessel.move_to_anchorage:
                        vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route[-1], vessel.true_origin)
                    elif vessel.waiting_time:
                        vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route_after_anchorage[-1], vessel.true_origin)
                    else:
                        vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route[-1], vessel.route[0])

                    route = vessel.route_after_terminal
                    route.reverse()
                    sailing_distance = 0
                    for nodes in enumerate(route[:-1]):
                        _, _, distance = vessel.wgs84.inv(vessel.env.FG.nodes[route[nodes[0]]]['geometry'].x,
                                                          vessel.env.FG.nodes[route[nodes[0]]]['geometry'].y,
                                                          vessel.env.FG.nodes[route[nodes[0] + 1]]['geometry'].x,
                                                          vessel.env.FG.nodes[route[nodes[0] + 1]]['geometry'].y)
                        sailing_distance += distance
                    terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
                    vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                    waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel,True,horizontal_tidal_window=True,max_waiting_time=True,out=True,route = vessel.route_after_terminal, delay=vessel_etd-vessel.env.now,plot=False)
                    terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                    terminal.users[-1].type = vessel.type
        else:
            yield from PassTerminal.move_to_anchorage(vessel, node)

    def pass_terminal(vessel,edge):
        yield from Movable.pass_edge(vessel, edge[0], edge[1])
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        index = vessel.route[vessel.route.index(edge[1]) - 1]

        if terminal.type == 'quay':
            terminal.pos_length.get(vessel.L)
            terminal.log_entry("Arrival of vessel", vessel.env.now, terminal.length.capacity-terminal.pos_length.level,nx.get_node_attributes(vessel.env.FG, "geometry")[index], )
        elif terminal.type == 'jetty':
            terminal.jetties_occupied += 1
            terminal.log_entry("Arrival of vessel", vessel.env.now, terminal.jetties_occupied,nx.get_node_attributes(vessel.env.FG, "geometry")[index], )

        # Berthing
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Berthing start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_b']*60)
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Berthing stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        for section in enumerate(vessel.env.FG.nodes[edge[1]]['Junction'][0].section):
            if list(section[1].keys())[0] == edge[1]:
                break

        if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
            PassSection.release_access_previous_section(vessel, edge[1])

        # Unloading
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Unloading start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_l']*60/2)
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Unloading stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Loading
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Loading start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_l']*60/2)
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Loading stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # New Route
        vessel.bound = 'outbound'
        if 'true_origin' in dir(vessel):
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[vessel.route.index(edge[1])], vessel.true_origin)
        else:
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[vessel.route.index(edge[1])], vessel.route[0])

        # Waiting for tidal window
        for nodes in enumerate(vessel.route):
            required_water_depth = vessel.T_f + vessel.metadata['ukc']
            minimum_water_depth = np.min(vessel.env.FG.nodes[vessel.route[nodes[0]]]['Info']['Water level'][1])
            if required_water_depth > minimum_water_depth:
                break

        vessel.waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel,vertical_tidal_window=True,horizontal_tidal_window=True,max_waiting_time = False,route=vessel.route,out=True,plot=False)
        vessel.bound = 'outbound' ##to_be_removed later
        if vessel.waiting_time:
            ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
            vessel.log_entry("Waiting for tidal window start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
            yield vessel.env.timeout(np.max([0,vessel.waiting_time-vessel.metadata['t_b']*60]))
            ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
            vessel.log_entry("Waiting for tidal window stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Deberthing
        for section in enumerate(vessel.env.FG.nodes[edge[1]]['Junction'][0].section):
            if list(section[1].keys())[0] == edge[1]:
                break

        if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
            yield from PassSection.request_access_next_section(vessel, edge[1], edge[0])

        if 'Turning Basin' in vessel.env.FG.nodes[edge[0]].keys():
            turning_basin = vessel.env.FG.nodes[edge[0]]['Turning Basin'][0]
            if turning_basin.length >= vessel.L:
                vessel.request_access_turning_basin = turning_basin.turning_basin[edge[0]].request()
                yield vessel.request_access_turning_basin

        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Deberthing start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_b']*60)
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Deberthing stop", vessel.env.now, ukc, shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        yield from Movable.pass_edge(vessel, edge[1], edge[0])
        PassSection.release_access_previous_section(vessel, edge[0])
        vessel.route = vessel.route[1:]

        if terminal.type == 'quay':
            old_level = PassTerminal.calculate_quay_length_level(terminal)

            def readjust_available_quay_lengths(terminal,position):
                aql = terminal.available_quay_lengths
                for i in range(len(aql)):
                    if i == 0:
                        continue
                    if aql[i - 1][1] < position and aql[i][1] > position:
                        break

                if i == 1:
                    aql[i - 1][0] = 0
                    aql[i][0] = 0

                elif i == len(aql) - 1:
                    aql[i - 1][0] = 0
                    aql[i][0] = 0

                else:
                    aql[i - 1][0] = 0
                    aql[i][0] = 0

                to_remove = []
                for i in enumerate(aql):
                    for j in enumerate(aql):
                        if i[0] != j[0] and i[1][0] == 0 and j[1][0] == 0 and i[1][1] == j[1][1]:
                            to_remove.append(i[0])

                for i in list(reversed(to_remove)):
                    aql.pop(i)

                return aql

            terminal.available_quay_lengths = readjust_available_quay_lengths(terminal,vessel.quay_position)
            new_level = PassTerminal.calculate_quay_length_level(terminal)
            if old_level != new_level:
                terminal.length.put(new_level - old_level)
            terminal.pos_length.put(vessel.L)
            terminal.log_entry("Departure of vessel", vessel.env.now, terminal.length.capacity-terminal.pos_length.level,nx.get_node_attributes(vessel.env.FG, "geometry")[index], )
            terminal.terminal[edge[0]].release(vessel.access_terminal)

        elif terminal.type == 'jetty':
            terminal.jetties_occupied -= 1
            terminal.log_entry("Departure of vessel", vessel.env.now, terminal.jetties_occupied,nx.get_node_attributes(vessel.env.FG, "geometry")[index],)
            terminal.terminal[vessel.index_jetty_position][edge[0]].release(vessel.access_terminal)

        vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
        vessel.env.process(vessel.move())
        vessel.leaving_port = True

class Output():
    def general_output(sim):
        sim.output['Avg_turnaround_time'] = []
        sim.output['Avg_sailing_time'] = []
        sim.output['Avg_waiting_time'] = []
        sim.output['Avg_service_time'] = []
        sim.output['Total_throughput'] = []

    def vessel_dependent_output(vessel):
        vessel.output['route'] = []
        vessel.output['speed'] = []
        vessel.output['sailing_time'] = []
        vessel.output['freeboard'] = []
        vessel.output['throughput'] = []
        vessel.output['n_encouters'] = []
        vessel.output['n_overtaking'] = []
        vessel.output['n_overtaken'] = []
        vessel.output['terminals_called'] = []
        vessel.output['anchorages_used'] = []
        vessel.output['waiting_time'] = []
        vessel.output['service_time'] = []
        vessel.output['total_sailing_time'] = []
        vessel.output['total_waiting_time'] = []
        vessel.output['total_service_time'] = []
        vessel.output['turnaround_time'] = []

    def node_dependent_output(network):
        for node in network.nodes:
            if 'Anchorage' in network.nodes[node].keys():
                network.nodes[node]['Anchorage'][0].output['Anchorage_time'] = []
                network.nodes[node]['Anchorage'][0].output['Anchorage_availability'] = []
                network.nodes[node]['Anchorage'][0].output['Anchorage_waiting_time'] = []
                network.nodes[node]['Anchorage'][0].output['Anchorage_occupancy'] = []

            if 'Turning basin' in network.nodes[node].keys():
                network.nodes[node]['Turning basin'][0].output['Turning_basin_time'] = []
                network.nodes[node]['Turning basin'][0].output['Turning_basin_availability'] = []
                network.nodes[node]['Turning basin'][0].output['Turning_basin_waiting_time'] = []
                network.nodes[node]['Turning basin'][0].output['Turning_basin_occupancy'] = []

    def edge_dependent_output(network):
        for edge in network.edges:
            network.edges[edge]['Output'] = {'v_traffic': [],
                                             'h_req_traffic': [],
                                             'throughput_traffic': [],
                                             'n_encounters': [],
                                             'n_overtakes': [],
                                             't_passages': [],
                                             'n_passages': []}

            if 'Terminal' in network.edges[edge].keys():
                if network.edges[edge]['Terminal'][0].type == 'jetty':
                    units = len(network.edges[edge]['Terminal'][0].jetty_lengths)
                else:
                    units = 1
                network.edges[edge]['Terminal'][0].output['Berth_vessels_served'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Berth_arrival_times'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Berth_departure_times'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Berth_net_throughput'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Berth_waiting_time'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Berth_service_time'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Berth_productivity'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Mean_berth_productivity'] = []
                network.edges[edge]['Terminal'][0].output['Berth_interarrival_times'] = []
                network.edges[edge]['Terminal'][0].output['Berth_interdeparture_times'] = []
                network.edges[edge]['Terminal'][0].output['Mean_berth_occupancy'] = []
                network.edges[edge]['Terminal'][0].output['Terminal_availability'] = []
                network.edges[edge]['Terminal'][0].output['Terminal_waiting_time'] = []
                network.edges[edge]['Terminal'][0].output['Terminal_service_time'] = []
                network.edges[edge]['Terminal'][0].output['Terminal_occupancy'] = []
                network.edges[edge]['Terminal'][0].output['Terminal_vessels_served'] = []
                network.edges[edge]['Terminal'][0].output['Terminal_net_throughput'] = []

    def real_time_terminal_output(terminal,berth,vessel):
        if network.edges[edge]['Terminal'][0].type == 'jetty':
            terminal_type = 'Berth'
        elif network.edges[edge]['Terminal'][0].type == 'terminal':
            terminal_type = 'Terminal'
        for berth in terminal.berths:
            terminal.output[terminal_type+'_vessels_served'][berth].extend(vessel)
            terminal.output[terminal_type+'_vessels_lengths'][berth].extend(vessel.L)
            terminal.output[terminal_type+'_arrival_times'][berth].extend(vessel.arrival_time)
            terminal.output[terminal_type+'_departure_times'][berth].extend(vessel.departure_time)
            terminal.output[terminal_type+'_net_throughput'][berth].extend(vessel.throughput_departure - vessel.throughput_arrival)
            terminal.output[terminal_type+'_waiting_time'][berth].extend(vessel.waiting_time)
            terminal.output[terminal_type+'_service_time'][berth].extend(vessel.service_time)
            terminal.output[terminal_type+'_turnaround_time'][berth].extend(vessel.service_time + vessel.waiting_time)
            terminal.output[terminal_type+'_productivity'][berth].extend(vessel.service_time / (vessel.waiting_time + vessel.service_time))

    def post_process_terminal_output(terminal, berth, simulation_duration):
        if terminal.type == 'jetty':
            for berth in terminal.berths:
                terminal.output['Mean_berth_productivity'][berth] = np.sum(terminal.output['Berth_sevice_time']) / (np.sum(terminal.output['Berth_waiting_time']) + np.sum(terminal.output['Berth_service_time']))
                terminal.output['Berth_interarrival_times'][berth]= [terminal.output['Berth_arrival_times'][berth][t]-terminal.output['Berth_arrival_times'][berth][t-1] for t in range(len(terminal.output['Berth_arrival_times'][berth])) if t>0]
                terminal.output['Berth_interdeparture_times'][berth] = [terminal.output['Berth_departure_times'][berth][t] - terminal.output['Berth_departure_times'][berth][t - 1] for t in range(len(terminal.output['Berth_departure_times'][berth])) if t > 0]
                terminal.output['Berth_occupancy'][berth] = np.sum([terminal.output['Berth_departure_times'][berth][t]- terminal.output['Berth_arrival_times'][berth][t] for t in range(len(terminal.output['Berth_vessels_served'][berth]))])/simulation_duration
                terminal.output['Berth_availability'][berth] = 1-terminal.output['Mean_berth_occupancy'][berth]
                terminal.output['Effective_berth_occupancy'][berth] = np.sum([terminal.output['Berth_vessels_lengths'][berth][t]*terminal.output['Berth_productivity'][berth][t]*terminal.output['Berth_turnaround_time'][berth][t] for t in range(len(terminal.output['Terminal_vessels_served'][berth]))]) / (terminal.length*simulation_duration)
                terminal.output['Potential_berth_availability'][berth] = 1-terminal.output['Mean_effective_berth_occupancy'][berth]
                terminal.output['Mean_berth_waiting_time'][berth] = np.mean(terminal.output['Berth_waiting_time'][berth])
                terminal.output['Mean_berth_service_time'][berth] = np.mean(terminal.output['Berth_service_time'][berth])
                terminal.output['Mean_berth_turnaround_time'][berth] = np.mean(terminal.output['Berth_turnaround_time'][berth])
                terminal.output['Berth_total_net_throughput'][berth] = np.sum(terminal.output['Berth_net_throughput'][berth])
            terminal.output['Terminal_occupancy'] = np.mean(terminal.output['Mean_berth_occupancy'])
            terminal.output['Terminal_availability'] = 1-np.mean(terminal.output['Mean_berth_occupancy'])
            terminal.output['Effective_terminal_occupancy'] = np.mean(terminal.output['Mean_effective_berth_occupancy'])
            terminal.output['Potential_terminal_availability'] = 1-np.mean(terminal.output['Mean_effective_berth_occupancy'])
            terminal.output['Terminal_waiting_time'] = np.mean(terminal.output['Berth_waiting_time'])
            terminal.output['Terminal_service_time'] = np.mean(terminal.output['Berth_service_time'])
            terminal.output['Terminal_turnaround_time'] = np.mean(terminal.output['Berth_turnaround_time'])
            terminal.output['Terminal_total_net_throughput'] = np.sum(terminal.output['Berth_net_throughput'])

        elif network.edges[edge]['Terminal'][0].type == 'terminal':
            terminal.output['Mean_terminal_productivity'] = np.sum(terminal.output['Berth_sevice_time'][berth]) / (np.sum(terminal.output['Berth_waiting_time'][berth]) + np.sum(terminal.output['Berth_sevice_time'][berth]))
            terminal.output['Terminal_interarrival_times'] = [terminal.output['Berth_arrival_times'][berth][t] - terminal.output['Berth_arrival_times'][berth][t - 1] for t in range(len(terminal.output['Berth_arrival_times'][berth])) if t > 0]
            terminal.output['Terminal_interdeparture_times'] = [terminal.output['Berth_departure_times'][berth][t] - terminal.output['Berth_departure_times'][berth][t - 1] for t in range(len(terminal.output['Berth_departure_times'][berth])) if t > 0]
            terminal.output['Quay_occupancy'] = [terminal.output['Terminal_vessels_lengths'][berth][t]*terminal.output['Terminal_turnaround_time'][berth][t] for t in range(len(terminal.output['Terminal_vessels_served'][berth]))] / (terminal.length*simulation_duration)
            terminal.output['Effective_quay_occupancy'] = np.sum([terminal.output['Terminal_vessels_lengths'][berth][t]*terminal.output['Terminal_productivity'][berth][t]*terminal.output['Terminal_turnaround_time'][berth][t] for t in range(len(terminal.output['Terminal_vessels_served'][berth]))]) / (terminal.length*simulation_duration)
            terminal.output['Terminal_occupancy'] = [] #information about the number of cranes per vessels required (not yet included in the model)
            terminal.output['Terminal_availability'] = [] #idem
            terminal.output['Effective_terminal_occupancy'] = [] #idem
            terminal.output['Potential_terminal_availability'] = [] #idem
            terminal.output['Terminal_waiting_time'] = np.mean(terminal.output['Terminal_waiting_time'])
            terminal.output['Terminal_service_time'] = np.mean(terminal.output['Terminal_service_time'])
            terminal.output['Terminal_turnaround_time'] = np.mean(terminal.output['Terminal_turnaround_time'])
            terminal.output['Terminal_total_net_throughput'] = np.sum(terminal.output['Terminal_net_throughput'])

class Movable(Locatable, Routeable, Log):
    """Mixin class: Something can move
    Used for object that can move with a fixed speed
    geometry: point used to track its current location
    v: speed"""

    def __init__(self, v=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.v = v
        self.wgs84 = pyproj.Geod(ellps="WGS84")

    def move(self):
        """determine distance between origin and destination, and
        yield the time it takes to travel it
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].
        """
        self.distance = 0

        # Check if vessel is at correct location - if not, move to location
        if self.geometry != nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]:
            orig = self.geometry
            dest = nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]
            self.distance += self.wgs84.inv(shapely.geometry.asShape(orig).x,
                                            shapely.geometry.asShape(orig).y,
                                            shapely.geometry.asShape(dest).x,
                                            shapely.geometry.asShape(dest).y,)[2]

            yield self.env.timeout(self.distance / self.current_speed)
            self.log_entry("Sailing to start", self.env.now, self.distance, dest)

        # Move over the path and log every step
        for node in enumerate(self.route):
            self.node = node[1]

            if node[0] + 2 <= len(self.route):
                origin = self.route[node[0]]
                destination = self.route[node[0] + 1]

            # Leave and access waterway section
            if 'Junction' in self.env.FG.nodes[origin].keys():
                if 'Anchorage' not in self.env.FG.nodes[origin].keys():
                    PassSection.release_access_previous_section(self, origin)
                    yield from PassSection.request_access_next_section(self, origin, destination)

            if 'Turning Basin' in self.env.FG.nodes[origin].keys():
                turning_basin = self.env.FG.nodes[origin]['Turning Basin'][0]
                ukc = VesselTrafficService.provide_ukc_clearance(vessel, origin)
                if self.bound == 'inbound' and turning_basin.length >= self.L:
                    self.log_entry("Vessel Turning Start", self.env.now, ukc, self.env.FG.nodes[origin]['geometry'])
                    turning_basin.log_entry("Vessel Turning Start", self.env.now, 0, self.env.FG.nodes[origin]['geometry'] )
                    yield self.env.timeout(10*60)
                    ukc = VesselTrafficService.provide_ukc_clearance(vessel, origin)
                    turning_basin.log_entry("Vessel Turning Stop", self.env.now, 10*60, self.env.FG.nodes[origin]['geometry'] )
                    self.log_entry("Vessel Turning Stop", self.env.now, ukc, self.env.FG.nodes[origin]['geometry'])
                    self.bound = 'outbound'
                else:
                    self.log_entry("Passing Turning Basin", self.env.now, ukc, self.env.FG.nodes[origin]['geometry'])
                    turning_basin.log_entry("Vessel Passing", self.env.now, 0,self.env.FG.nodes[origin]['geometry'])
                turning_basin.turning_basin[origin].release(self.request_access_turning_basin)

            if 'Turning Basin' in self.env.FG.nodes[destination].keys():
                turning_basin = self.env.FG.nodes[destination]['Turning Basin'][0]
                if turning_basin.length >= self.L:
                    self.request_access_turning_basin = turning_basin.turning_basin[destination].request()
                    yield self.request_access_turning_basin

            #PassLock
            if "Waiting area" in self.env.FG.nodes[destination].keys():         #if waiting area is located at next node
                yield from lock_module.PassLock.approach_waiting_area(self, destination)

            if "Waiting area" in self.env.FG.nodes[origin].keys():              #if vessel is in waiting area
                yield from lock_module.PassLock.leave_waiting_area(self, origin)

            if "Line-up area" in self.env.FG.nodes[destination].keys(): #if vessel is approaching the line-up area
                lock_module.PassLock.approach_lineup_area(self, destination)

            if "Line-up area" in self.env.FG.nodes[origin].keys(): #if vessel is located in the line-up
                lineup_areas = self.env.FG.nodes[origin]["Line-up area"]
                for lineup_area in lineup_areas:
                    if lineup_area.name != self.lock_name: #picks the assigned parallel lock chain
                        continue

                    index_node_lineup_area = self.route.index(origin)
                    for node_lock in self.route[index_node_lineup_area:]:
                        if 'Lock' in self.env.FG.nodes[node_lock].keys():
                            yield from lock_module.PassLock.leave_lineup_area(self,origin)
                            break

                        elif 'Waiting area' in self.env.FG.nodes[node_lock].keys(): #if vessel is leaving the lock complex
                            yield from lock_module.PassLock.leave_opposite_lineup_area(self,origin)
                            break

            if "Lock" in self.env.FG.nodes[origin].keys(): #if vessel in lock
                yield from lock_module.PassLock.leave_lock(self,origin)
                self.v = 4*self.v

            # Anchorage
            if 'Anchorage' in self.env.FG.nodes[destination].keys() and self.route[-1] == destination:
                yield from PassTerminal.pass_anchorage(self, destination)
                break

            # Request for a terminal
            if "Origin" in self.env.FG.nodes[origin] and 'leaving_port' not in dir(self):
                self.bound = 'inbound' ##to be removed later
                yield from PassTerminal.request_terminal_access(self, [self.route[-2], self.route[-1]], origin)
                if self.waiting_in_anchorage:
                    break

            # Terminal
            if 'Terminal' in self.env.FG.edges[origin, destination].keys() and self.route[-1] == destination:
                yield from PassTerminal.pass_terminal(self, [origin, destination])
                break

            else:
                yield from self.pass_edge(origin, destination)

        logger.debug("  distance: " + "%4.2f" % self.distance + " m")
        logger.debug("  sailing:  " + "%4.2f" % self.current_speed + " m/s")
        logger.debug("  duration: " + "%4.2f" % ((self.distance / self.current_speed) / 3600) + " hrs")

    def pass_edge(self, origin, destination):
        edge = self.env.FG.edges[origin, destination]
        orig = nx.get_node_attributes(self.env.FG, "geometry")[origin]
        dest = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        if "Lock" in self.env.FG.nodes[origin].keys():
            orig = shapely.geometry.Point(self.lock_pos_lat,self.lock_pos_lon)

        if "Lock" in self.env.FG.nodes[destination].keys():
            dest = shapely.geometry.Point(self.lock_pos_lat,self.lock_pos_lon)

        if "Line-up area" in self.env.FG.nodes[origin].keys():
            orig = shapely.geometry.Point(self.lineup_pos_lat,self.lineup_pos_lon)

        if "Line-up area" in self.env.FG.nodes[destination].keys():
            dest = shapely.geometry.Point(self.lineup_pos_lat,self.lineup_pos_lon)

        if 'Terminal' in edge.keys():
            terminal = edge['Terminal'][0]

            if self.route[-1] == destination:
                self.wgs84 = pyproj.Geod(ellps="WGS84")
                [origin_lat,
                 origin_lon,
                 destination_lat,
                 destination_lon] = [self.env.FG.nodes[origin]['geometry'].x,
                                     self.env.FG.nodes[origin]['geometry'].y,
                                     self.env.FG.nodes[destination]['geometry'].x,
                                     self.env.FG.nodes[destination]['geometry'].y]
                fwd_azimuth, _, _ = self.wgs84.inv(origin_lat, origin_lon,destination_lat, destination_lon)

                if terminal.type == 'quay':
                    position = self.quay_position

                elif terminal.type == 'jetty':
                    position = terminal.jetty_locations[self.index_jetty_position]

                [self.terminal_pos_lat, self.terminal_pos_lon,_] = self.wgs84.fwd(self.env.FG.nodes[origin]['geometry'].x,self.env.FG.nodes[origin]['geometry'].y,fwd_azimuth, position)
                dest = shapely.geometry.Point(self.terminal_pos_lat,self.terminal_pos_lon)

            else:
                orig = shapely.geometry.Point(self.terminal_pos_lat, self.terminal_pos_lon)

        if 'geometry' in edge:
            edge_route = np.array(edge['geometry'])

            # check if edge is in the sailing direction, otherwise flip it
            distance_from_start = self.wgs84.inv(orig.x,orig.y,edge_route[0][0],edge_route[0][1],)[2]
            distance_from_stop = self.wgs84.inv(orig.x,orig.y,edge_route[-1][0],edge_route[-1][1],)[2]
            if distance_from_start>distance_from_stop:
                # when the distance from the starting point is greater than from the end point
                edge_route = np.flipud(np.array(edge['geometry']))

            for index, pt in enumerate(edge_route[:-1]):
                sub_orig = shapely.geometry.Point(edge_route[index][0], edge_route[index][1])
                sub_dest = shapely.geometry.Point(edge_route[index+1][0], edge_route[index+1][1])

                distance = self.wgs84.inv(shapely.geometry.asShape(sub_orig).x,
                                          shapely.geometry.asShape(sub_orig).y,
                                          shapely.geometry.asShape(sub_dest).x,
                                          shapely.geometry.asShape(sub_dest).y,)[2]
                self.distance += distance
                ukc = VesselTrafficService.provide_ukc_clearance(vessel, origin)
                self.log_entry("Sailing from node {} to node {} sub edge {} start".format(origin, destination, index), self.env.now, ukc, sub_orig,)
                yield self.env.timeout(distance / self.current_speed)
                ukc = VesselTrafficService.provide_ukc_clearance(vessel, destination)
                self.log_entry("Sailing from node {} to node {} sub edge {} stop".format(origin, destination, index), self.env.now, ukc, sub_dest,)
            self.geometry = dest

        else:
            distance = self.wgs84.inv(shapely.geometry.asShape(orig).x,
                                      shapely.geometry.asShape(orig).y,
                                      shapely.geometry.asShape(dest).x,
                                      shapely.geometry.asShape(dest).y,)[2]

            self.distance += distance
            arrival = self.env.now

            # Act based on resources
            if "Resources" in edge.keys():
                ukc = VesselTrafficService.provide_ukc_clearance(vessel, origin)
                with self.env.FG.edges[origin, destination]["Resources"].request() as request:
                    yield request

                    if arrival != self.env.now:
                        self.log_entry("Waiting to pass edge {} - {} start".format(origin, destination), arrival, ukc, orig,)
                        ukc = VesselTrafficService.provide_ukc_clearance(vessel, origin)
                        self.log_entry("Waiting to pass edge {} - {} stop".format(origin, destination), self.env.now, ukc, orig,)

                    self.log_entry("Sailing from node {} to node {} start".format(origin, destination), self.env.now, 0, orig,)
                    yield self.env.timeout(distance / self.current_speed)
                    ukc = VesselTrafficService.provide_ukc_clearance(vessel, destination)
                    self.log_entry("Sailing from node {} to node {} stop".format(origin, destination), self.env.now, ukc, dest,)

            else:
                ukc = VesselTrafficService.provide_ukc_clearance(vessel, origin)
                self.log_entry("Sailing from node {} to node {} start".format(origin, destination), self.env.now, ukc, orig,)
                yield self.env.timeout(distance / self.current_speed)
                ukc = VesselTrafficService.provide_ukc_clearance(vessel, destination)
                self.log_entry("Sailing from node {} to node {} stop".format(origin, destination), self.env.now, ukc, dest,)

    @property
    def current_speed(self):
        return self.v

class ContainerDependentMovable(Movable, HasContainer):
    """ContainerDependentMovable class
    Used for objects that move with a speed dependent on the container level
    compute_v: a function, given the fraction the container is filled (in [0,1]), returns the current speed"""

    def __init__(self, compute_v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.compute_v = compute_v
        self.wgs84 = pyproj.Geod(ellps="WGS84")

    @property
    def current_speed(self):
        return self.compute_v(self.container.level / self.container.capacity)

