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
import xarray as xr
import numpy as np
import math
import bisect
import scipy as sc
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
import time as timepy
import hatyan
import sys
import os

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

        # Loop over the sections which are connected to the junction
        for edge in enumerate(self.sections):
            # If section is type 'one-way-traffic'
            if self.type[edge[0]] == 'one-way_traffic':
                #Set default direction parameter
                direction = 0
                #If longitude of first node of the edge is smaller than the second node and strictly not equal to each other: set direction = 1
                if self.env.FG.nodes[edge[1][0]]['geometry'].x != self.env.FG.nodes[edge[1][1]]['geometry'].x:
                    if self.env.FG.nodes[edge[1][0]]['geometry'].x < self.env.FG.nodes[edge[1][1]]['geometry'].x:
                        direction = 1
                # Else is longitudes are equal: if latitude of first node of the edge is smaller than the second node: set direction = 1
                elif self.env.FG.nodes[edge[1][0]]['geometry'].y < self.env.FG.nodes[edge[1][1]]['geometry'].y:
                    direction = 1

                # If direction: append two access resources to that node
                if direction:
                    if 'access1' not in dir(self):
                        self.access1 = []
                        self.access2 = []

                    self.access1.append({edge[1][0]: simpy.PriorityResource(self.env, capacity=1),})
                    self.access2.append({edge[1][1]: simpy.PriorityResource(self.env, capacity=1),})

            # Append a resource to the section with 'infinite' capacity
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

        def astronomical_tide(signal_time,signal_values):
            old_stdout = sys.stdout  # backup current stdout
            sys.stdout = open(os.devnull, "w")
            signal_datetime = [datetime.datetime.fromtimestamp(y) for y in signal_time]
            const_list = hatyan.get_const_list_hatyan('tidalcycle')
            ts_meas = pd.DataFrame({'values': signal_values}, index=signal_datetime)
            ts_meas = hatyan.crop_timeseries(ts=ts_meas, times_ext=signal_datetime);
            comp_frommeas, comp_allyears = hatyan.get_components_from_ts(ts=ts_meas, const_list=const_list, nodalfactors=True, return_allyears=True, fu_alltimes=True, analysis_peryear=True)
            ts_prediction = hatyan.prediction(comp=comp_frommeas, nodalfactors=True, xfac=True, fu_alltimes=True, times_ext=signal_datetime, timestep_min=10)
            sys.stdout = old_stdout  # reset old stdout
            return [[index.timestamp()-3600 for index in ts_prediction.index],[value for value in ts_prediction['values']]]

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
                if index_range_min == index_range_max: continue
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

        def tidal_periods(t_astro_tide_wlev, astro_tide_wlev):
            """ Function: calculates the water level which is exceeded 99% of the tides for a given node in the network.

                Input:
                    - t_astro_wlev: a list containing the chronological sequence of timestamps for the specific node, derived from the
                                    list of the time series containing the water level at the specific node
                    - astro_wlev: a list containing the chronological sequence of water levels for the specific node, derived from the
                                  list of the time series containing the water level at the specific node
            """

            # Deriving some properties
            mean_astro_wlev = np.mean(astro_tide_wlev)

            # Interpolation of the variation of the water level and calculation of the zero crossing times
            cor_astro_tide = [astro_wlev - mean_astro_wlev for astro_wlev in astro_tide_wlev]
            intp = sc.interpolate.CubicSpline(t_astro_tide_wlev, cor_astro_tide)
            times_tidal_periods = []
            index_prev_root = 0
            for root in intp.roots():
                index_current_root = bisect.bisect_right(t_astro_tide_wlev, root)
                if root >= t_astro_tide_wlev[0] and root <= t_astro_tide_wlev[-1] and np.max(cor_astro_tide[index_prev_root:index_current_root]) > 0:
                    index = cor_astro_tide[index_prev_root:index_current_root].index(np.max(cor_astro_tide[index_prev_root:index_current_root]))
                    time_start_next_tidal_period = t_astro_tide_wlev[index + index_prev_root]
                    times_tidal_periods.append([time_start_next_tidal_period, 'Ebb Start'])
                    index_prev_root = index_current_root
                elif root >= t_astro_tide_wlev[0] and root <= t_astro_tide_wlev[-1] and np.min(cor_astro_tide[index_prev_root:index_current_root]) < 0:
                    index = cor_astro_tide[index_prev_root:index_current_root].index(np.min(cor_astro_tide[index_prev_root:index_current_root]))
                    time_start_next_tidal_period = t_astro_tide_wlev[index + index_prev_root]
                    times_tidal_periods.append([time_start_next_tidal_period, 'Flood Start'])
                    index_prev_root = index_current_root

            return times_tidal_periods

        def append_info_to_edges(network):
            """ Function: appends nodal data to the edges of the network for visualisation purposes only

                Input:
                    - network: a graph constructed with the DiGraph class of the networkx package

            """

            #Function that is used in the calculation of info that needs to be appended to the edges
            def lag_finder(time_signal, signal1, signal2):
                nsamples = len(signal2)
                b, a = sc.signal.butter(2, 0.1)
                signal2 = sc.signal.filtfilt(b, a, signal2)
                # regularize datasets by subtracting mean and dividing by s.d.
                signal2 -= np.mean(signal2);
                signal2 /= np.std(signal2)
                signal1 -= np.mean(signal1);
                signal1 /= np.std(signal1)
                # Find cross-correlation
                xcorr = sc.signal.correlate(signal2, signal1)

                # delta time array to match xcorr
                dt = np.arange(1 - nsamples, nsamples)
                time_step = time_signal[1] - time_signal[0]
                delay = dt[xcorr.argmax()] * time_step
                return delay

            # Loops over the edges of the network
            for edge in enumerate(network.edges):
                # Adds parameters to the dictionary
                network.edges[edge[1]]['Info']['Width'] = []
                network.edges[edge[1]]['Info']['Depth'] = []
                network.edges[edge[1]]['Info']['MBL'] = []
                network.edges[edge[1]]['Info']['Tidal phase lag'] = []

                #Appends data to the edges
                network.edges[edge[1]]['Info']['Width'] = np.min([network.nodes[edge[1][0]]['Info']['Width'], network.nodes[edge[1][1]]['Info']['Width']])
                network.edges[edge[1]]['Info']['Tidal phase lag'] = lag_finder(network.nodes[edge[1][0]]['Info']['Astronomical tide'][0],
                                                                               network.nodes[edge[1][0]]['Info']['Astronomical tide'][1],
                                                                               network.nodes[edge[1][1]]['Info']['Astronomical tide'][1])

                #If there is a terminal in the edge, the greatest value for the MBL or depth of the two nodes creating the edge are used
                if 'Terminal' in network.edges[edge[1]]:
                    network.edges[edge[1]]['Info']['Depth'] = np.max([network.nodes[edge[1][0]]['Info']['Depth'], network.nodes[edge[1][1]]['Info']['Depth']])
                    network.edges[edge[1]]['Info']['MBL'] = np.max([network.nodes[edge[1][0]]['Info']['MBL'], network.nodes[edge[1][1]]['Info']['MBL']])
                else:
                    network.edges[edge[1]]['Info']['Depth'] = np.min([network.nodes[edge[1][0]]['Info']['Depth'], network.nodes[edge[1][1]]['Info']['Depth']])
                    network.edges[edge[1]]['Info']['MBL'] = np.min([network.nodes[edge[1][0]]['Info']['MBL'], network.nodes[edge[1][1]]['Info']['MBL']])

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
            network.nodes[node[1]]['Info'] = {'Tidal periods': [],'Horizontal tidal restriction': {},'Vertical tidal restriction': {}, 'Width': [], 'Depth': [], 'MBL': [], 'Astronomical tide': [], 'H_99%': [], 'Water level': [[],[]],  'Current velocity': [[],[]], 'Current direction': [[],[]], 'Cross-current': {}, 'Longitudinal current': {}}

            #Appending the specific data to the network if the geometry of the node of the data is the same as the geometry of the node of the network for the static data
            if (MBL[0][node[0]].x,MBL[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                network.nodes[node[1]]['Info']['MBL'] = MBL[1][node[0]]
            if (W[0][node[0]].x,W[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                network.nodes[node[1]]['Info']['Width'] = W[1][node[0]]
            if (D[0][node[0]].x,D[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                network.nodes[node[1]]['Info']['Depth'] = D[1][node[0]]

            # Appending the data to the specific lists by looping over the times in the time series for the dynamic data
            if (eta[0][node[0]].x,eta[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                network.nodes[node[1]]['Info']['Water level'][0] = eta[1][node[0]][0]
                network.nodes[node[1]]['Info']['Water level'][1] = eta[1][node[0]][1]
            if (vmag[0][node[0]].x,vmag[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                network.nodes[node[1]]['Info']['Current velocity'][0] = vmag[1][node[0]][0]
                network.nodes[node[1]]['Info']['Current velocity'][1] = vmag[1][node[0]][1]
            if (vdir[0][node[0]].x,vdir[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                network.nodes[node[1]]['Info']['Current direction'][0] = vdir[1][node[0]][0]
                network.nodes[node[1]]['Info']['Current direction'][1] = vdir[1][node[0]][1]

            # Calculation of the water level which is exceeded 99% of the tides
            network.nodes[node[1]]['Info']['Astronomical tide'] = astronomical_tide(network.nodes[node[1]]['Info']['Water level'][0],network.nodes[node[1]]['Info']['Water level'][1])
            network.nodes[node[1]]['Info']['H_99%'] = H99(network.nodes[node[1]]['Info']['Astronomical tide'][0],network.nodes[node[1]]['Info']['Astronomical tide'][1],node[1])
            network.nodes[node[1]]['Info']['Tidal periods'] = tidal_periods(network.nodes[node[1]]['Info']['Astronomical tide'][0],
                                                                            network.nodes[node[1]]['Info']['Astronomical tide'][1])

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
        network.nodes[node]['Info']['Vertical tidal restriction']['Specification'] = [[], [], [], [], [], []]

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
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][3].append(vessel_characteristics_spec)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][4].append(vessel_method_list)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][5].append([])

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

    @staticmethod
    def provide_sailing_direction(vessel,edge):
        initial_bound = vessel.bound
        network = vessel.env.FG
        phase_lag = network.edges[edge]['Info']['Tidal phase lag']
        if phase_lag < 0: bound = 'outbound'
        elif phase_lag > 0: bound = 'inbound'
        else: bound = initial_bound
        return bound

    def provide_ukc_clearance(vessel,node,delay=0,components_calc = False):
        """ Function: calculates the sail-in-times for a specific vssel with certain properties and a pre-determined route and provides this information to the vessel

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node:
                - components_calc:

        """

        interp_wdep, ukc_s, ukc_p, fwa = VesselTrafficService.provide_sail_in_times_tidal_window(vessel, [node], ukc_calc=True)
        net_ukc = interp_wdep(vessel.env.now+delay) - (vessel.T_f + vessel.metadata['ukc'])

        if components_calc == False:
            return net_ukc
        else:
            if net_ukc < ukc_s+ukc_p+fwa:
                return ukc_s+ukc_p+fwa
            else:
                return net_ukc

    def provide_sail_in_times_tidal_window(vessel,route,plot=False,sailing_time_correction=True,visualization_calculation=False,ukc_calc=False):
        """ Function: calculates the sail-in-times for a specific vessel with certain properties and a pre-determined route and provides this information to the vessel

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                - plot: provide a visualization of the calculation for each vessel
                - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)
                - visualization_calculation: a bool that indicates whether the calculation should be made for a single node or not (for a route with multiple nodes)

        """

        # Functions used to calculate the sail-in-times for a specific vessel
        def tidal_window_restriction_determinator(vessel, route, types, specifications, node, sailing_time_to_next_node):
            """ Function: determines which tidal window restriction applies to the vessel at the specific node

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route:
                    - types: the type of the restriction
                    - specifications: the specific data regarding the properties for which the restriction holds
                    - node: a string that defines the node of the tidal window restriction

            """

            # Predefined bool
            boolean = True
            no_tidal_window = True
            # Determining if and which restriction applies for the vessel by looping over the restriction class
            for restriction_class in enumerate(specifications[0]):
                # - if restriction does not apply to vessel because it is for vessels sailing in the opposite direction: continue loop
                if vessel.bound != specifications[2][restriction_class[0]]:
                    continue
                if specifications[5][restriction_class[0]] != [] and (specifications[5][restriction_class[0]][0] not in route or specifications[5][restriction_class[0]][1] not in route):
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
                    if restriction_type[1].find('Beam') != -1: value = getattr(vessel, 'B')
                    if restriction_type[1].find('UKC') != -1: value = VesselTrafficService.provide_ukc_clearance(vessel,node,sailing_time_to_next_node,components_calc = True)
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
                    depth = network.nodes[route[nodes[0]]]['Info']['MBL']
                    types = network.nodes[nodes[1]]['Info']['Vertical tidal restriction']['Type']
                    specifications = network.nodes[nodes[1]]['Info']['Vertical tidal restriction']['Specification']

                    # Determine which restriction applies to vessel
                    restriction_index, _ = tidal_window_restriction_determinator(vessel, route, types, specifications, nodes[1], sailing_time_to_next_node)

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
                        axis.plot(t_min_wdep, wdep, color='lightskyblue', alpha=0.4)

                # Pick the minimum of the water depths for each time and each node
                min_wdep = [min(idx) for idx in zip(*wdep_nodes)]

                return t_min_wdep, min_wdep

            #Continuation of the calculation of the windows given the vertical tidal restrictions by setting some parameters
            if not ukc_calc:
                times_vertical_tidal_window = []
                water_depth_required = vessel.T_f + vessel.metadata['ukc'] #vessel draught + additional vessel-related factors (if applicable)

            #Calculation of the minimum available water depth along the route of the vessel
            if not ukc_calc:
                new_t, min_wdep = minimum_available_water_depth_along_route(vessel, route, axis, plot, sailing_time_correction, ukc_calc)
            else:
                interp_wdep, ukc_s, ukc_p, fwa = minimum_available_water_depth_along_route(vessel, route, ukc_calc=True)
                return interp_wdep, ukc_s, ukc_p, fwa

            #Interpolation of the net ukc
            root_interp_water_level_at_edge = sc.interpolate.CubicSpline(new_t,[x-water_depth_required for x in min_wdep])

            #If there is not enough available water depth over time: no tidal window
            if np.max([x - water_depth_required for x in min_wdep]) < 0:
                times_vertical_tidal_window.append([vessel.waiting_time_start, 'Start']) #tidal restriction starts at t_start
                times_vertical_tidal_window.append([vessel.env.now+vessel.metadata['max_waiting_time'], 'Stop']) #tidal restriction ends at t_end
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
                times_vertical_tidal_window.append([vessel.env.now+vessel.metadata['max_waiting_time'], 'Start']) #tidal restriction starts at t_end

            #If plot is requested: plot the minimum available water depth, the required water depth of the vessel, the resulting vertical tidal windows, and add some lay-out
            if plot:
                axis.plot(new_t,min_wdep,color='deepskyblue')
                axis.plot([x[0] for x in times_vertical_tidal_window], (vessel.metadata['ukc']+vessel.T_f) * np.ones(len(times_vertical_tidal_window)), color='deepskyblue',marker='o',linestyle='None')
                axis.text(vessel.env.now+vessel.metadata['max_waiting_time'], 1.01*(vessel.metadata['ukc']+vessel.T_f), 'Required water depth', color='deepskyblue',horizontalalignment='center')
                axis.axhline((vessel.metadata['ukc']+vessel.T_f), color='deepskyblue', linestyle='--')
                axis.set_ylim([0,vessel.T_f+5])

            #Return the sail-in or -out-times given the vertical tidal restrictions over the route of the vessel
            return times_vertical_tidal_window

        def times_horizontal_tidal_window(vessel,route,axis=[],plot=plot,sailing_time_correction=True,visualization_calculation=False):
            """ Function: calculates the windows available to sail-in and -out of the port given the horizontal tidal restrictions according to the tidal window policy.

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                    - plot: provide a visualization of the calculation for each vessel
                    - axis: axes class from the matplotlib package
                    - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

            """

            def tidal_windows_tightest_current_restriction_along_route(vessel, route, axis=[], plot=False, sailing_time_correction=True, visualization_calculation=False):
                """ Function: calculates the normative current restrictions along the route over time and calculates the resulting horizontal tidal windows from these locations.

                    Input:
                        - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                        - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                        - plot: provide a visualization of the calculation for each vessel
                        - axis: axes class from the matplotlib package
                        - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

                """

                # Functions used for the calculation of the normative restriction along the route and the subsequent horizontal tidal windows over time
                def calculate_and_store_normative_current_velocity_data(node, data_nodes, sailing_time_to_next_node, t_ccur_raw, ccur, critcur_nodes, mccur_nodes, method, crit_ccur=[], crit_ccur_flood=0, crit_ccur_ebb=0):
                    """ Function: imports the current velocity data time series, corrects the time for the sailing time of the vessel, and subtracts the velocity data with the critical
                                  cross-current in order to find the times when the velocity exceeds the limit (by interpolation). The correct critical cross-current limit is picked by
                                  determining to what tidal period part of the time series belongs to (by determining what is the next tidal period). It appends the corrected data and
                                  the corresponding limits to separate lists.

                        Input:
                            - node: the name string of the node in the given network
                            - data_nodes: a list containing the two name string of the node in the given network the data is located. These locations depend on the direction of the vessel around the node of the restriction.
                            - sailing_time_to_next_node: time in s that the vessel needs to reach the node at which the restiction holds
                            - t_ccur_raw: a list of the raw, uncorrected timestamps of the time series for the current velocity
                            - ccur: a list to which the corrected current data will be appended
                            - critcur_nodes: a list to which the list of critical cross-currents corresponding to the maximum corrected current data will be appended to (for each node at which a horizontal tidal restriction is installed separately)
                            - mccur_nodes: a list to which the list of maximum corrected current data will be appended to (for each node at which a horizontal tidal restriction is installed separately)
                            - method: type of horizontal tidal restriction (method used)
                            - crit_ccur: a list to which the critical current velocity limit corresponding to the current data will be appended
                            - crit_ccur_flood: critical cross-current limit applied during flood in m/s
                            - crit_ccur_ebb: critical cross-current limit applied during ebb in m/s

                    """

                    # Setting some default bools
                    critical_cross_current_method_in_restrictions = False
                    point_based_method_in_restrictions = False

                    # Looping over the two nodes, one prior and one after the node at which the horizontal tidal restriction is installed, determining the direction of the vessel
                    for rep in range(2):
                        # Import current data and correct time series with sailing time, and import tidal periods
                        t_ccur = [t - sailing_time_to_next_node for t in network.nodes[route[node]]['Info']['Horizontal tidal restriction']['Data'][data_nodes[rep]][0]]
                        cur = network.nodes[route[node]]['Info']['Horizontal tidal restriction']['Data'][data_nodes[rep]][1]
                        times_tidal_periods = [z[0] - sailing_time_to_next_node for z in network.nodes[route[node]]['Info']['Tidal periods']]
                        tidal_periods = [z[1] for z in network.nodes[route[node]]['Info']['Tidal periods']]

                        # Different procedure for the methods:
                        if method == 'Critical cross-current':
                            critical_cross_current_method_in_restrictions = True
                            # Determine the next tidal period of the time series
                            t_ccur_tidal_periods = [tidal_periods[bisect.bisect_right(times_tidal_periods,y)] if y < times_tidal_periods[-1] else tidal_periods[-1] for y in t_ccur]
                            # Interpolate new time series
                            interp_ccur = sc.interpolate.CubicSpline(t_ccur, cur)
                            # If the next tidal period is ebb, then it means that the current period is flood. Hence, append interpolated current data subtracted with the critical cross-current, and the critical cross-current itself to the predefined lists
                            ccur.append([y - crit_ccur_flood if t_ccur_tidal_periods[x] == 'Ebb Start' else y - crit_ccur_ebb for x, y in enumerate(interp_ccur(t_ccur))])
                            crit_ccur.append([crit_ccur_flood if t_ccur_tidal_periods[x] == 'Ebb Start' else crit_ccur_ebb for x, y in enumerate(network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Data'][data_nodes[rep]][1])])

                        elif method == 'Point-based':
                            point_based_method_in_restrictions = True
                            # Just append the raw current data to the predefined list
                            ccur.append(network.nodes[route[node]]['Info']['Horizontal tidal restriction']['Data'][data_nodes[rep]][1])

                    # Take the maximum of both lists with the exceedance of the cross-current (positive values)
                    mccur = [max(idx) for idx in zip(*ccur)]

                    if method == 'Critical cross-current':
                        # Take the cross-currents of both lists which belong to the maximum cross-current limit exceeding velocities
                        critcur = [crit_ccur[np.argmax(val)][idx] for idx, val in enumerate(zip(*ccur))]
                        critcur_nodes.append(critcur)
                        mccur_nodes.append(mccur)

                    if method == 'Point-based':
                        # Interpolate new time series
                        interp_ccur = sc.interpolate.CubicSpline(t_ccur, mccur)
                        mccur = interp_ccur(t_ccur)
                        critcur = 10*np.ones(len(mccur))
                        critcur_nodes.append(critcur)
                        mccur_nodes.append(-10*np.ones(len(mccur)))

                    #Returns the determined tidal periods which may be used in a later stage of the calculation
                    return critical_cross_current_method_in_restrictions, point_based_method_in_restrictions, t_ccur, mccur, critcur, times_tidal_periods, tidal_periods

                def calculate_and_append_individual_horizontal_tidal_windows_critical_cross_current_method(new_t,max_ccur,max_crit_ccur):
                    """ Function: calculates and appends the horizontal tidal windows (available sail-in and -out times given the horizontal tidal restrictions) and corresponding critical cross-current
                                  velocities to the appropriate lists for the 'critical cross-curent'-method

                        Input:
                            - new_t: time of the time series for the current velocity corrected for the sailing time
                            - max_ccur: a list of the governing current velocities of the time series of the governing cross-currents in m/s
                            - max_crit_ccur: a list containing the critical-cross currents corrspeonding to the governing current velocities of the time series of the governing cross-currents in m/s

                    """

                    times_horizontal_tidal_window = []
                    crit_ccurs_horizontal_tidal_window = []

                    # Interpolate the data of the exceedance of the critical cross-current velocity
                    interp_max_cross_current = sc.interpolate.CubicSpline(new_t, [x for x in max_ccur])
                    # Loop over the roots (crossings where the cross-current velocity exceeds the cross-current velocity limit)
                    for root in interp_max_cross_current.roots():
                        # If the root falls within the time series and the next cross-current does not exceed the limit and the list of tidal restrictions is still empty or there are some restrictions identified but the previous restriction is not a stopping criteria
                        if root > new_t[0] and root < new_t[-1] and max_ccur[bisect.bisect_right(new_t, root)] < 0:
                            if times_horizontal_tidal_window == [] or (times_horizontal_tidal_window != [] and times_horizontal_tidal_window[-1][1] != 'Stop'):
                                # Append a stopping time of the restriction and the corresponding critical cross-current
                                times_horizontal_tidal_window.append([root, 'Stop'])
                                crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[bisect.bisect_right(new_t, root)])
                        # If the root falls within the time series and the next cross-current exceeds the limit as well and the list of tidal restrictions is still empty or there are some restrictions identified but the previous restriction is not a starting criteria
                        elif root > new_t[0] and root < new_t[-1] and max_ccur[bisect.bisect_right(new_t, root)] > 0:
                            if times_horizontal_tidal_window == [] or (times_horizontal_tidal_window != [] and times_horizontal_tidal_window[-1][1] != 'Start'):
                                # Append a starting time of the restriction and the corresponding critical cross-current
                                times_horizontal_tidal_window.append([root, 'Start'])
                                crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[bisect.bisect_right(new_t, root)])

                    # If the list of tidal windows are still empty: there is no horizontal tidal restriction for the particular vessel, so restriction stops at t_start and ends at t_end of the simulation
                    if times_horizontal_tidal_window == []:
                        times_horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                        crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[bisect.bisect_right(new_t, vessel.waiting_time_start)])
                        times_horizontal_tidal_window.append([np.max(new_t), 'Start'])
                        crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[-1])

                    # If the last restriction contains a starting time and is smaller than the end of the time series: add a stopping time restriction at the t_end of the simulation
                    if times_horizontal_tidal_window[-1][1] == 'Start' and times_horizontal_tidal_window[-1][0] < np.max(new_t):
                        times_horizontal_tidal_window.append([np.max(new_t), 'Stop'])
                        crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[-1])

                    return times_horizontal_tidal_window, crit_ccurs_horizontal_tidal_window

                def finding_current_velocity_bounds_point_based_method(t_ccur, mccur, p_range, crit_ccur_tidal_period, times_tidal_periods, tidal_periods, sailing_time_to_next_node):
                    """ Function: defines the bounds around the point-based current velocity and finds the times at which the current velocity crosses these bounds by solving the roots of the interpolation.
                                  Furthermore, the tidal periods for each time the current velocity intersects with the bounds around the point-based current velocity.

                        Input:
                            - t_ccur: time of the time series for the current velocity corrected for the sailing time
                            - mccur: a list of times of the governing cross-currents in m/s
                            - p_range: the range in % around the point-based current velocity
                            - crit_ccur_tidal_period: critical cross-current limit applied during the tidal period in m/s
                            - times_tidal_periods: the starting times of the tidal periods
                            - tidal_periods: the definition of the tidal periods (Ebb or Flood Start)
                            - sailing_time_to_next_node: time in s that the vessel needs to reach the node at which the restiction holds

                    """

                    # Interpolate the interpolated cross-current velocity, corrected for the sailing time, subtracted with the critical cross-current for the upper and lower bound separately and calculate roots, and corresponding next tidal periods
                    crit_ccur_upper_bound = crit_ccur_tidal_period * (1 + p_range)
                    crit_ccur_lower_bound = crit_ccur_tidal_period * (1 - p_range)
                    interp_ccur_upper_bound = sc.interpolate.CubicSpline(t_ccur, [ccur - crit_ccur_upper_bound for ccur in mccur])
                    interp_ccur_lower_bound = sc.interpolate.CubicSpline(t_ccur, [ccur - crit_ccur_lower_bound for ccur in mccur])
                    roots_upper_bound = interp_ccur_upper_bound.roots()
                    roots_lower_bound = interp_ccur_lower_bound.roots()
                    t_ccur_tidal_periods_upper_bound = [tidal_periods[bisect.bisect_right(times_tidal_periods,y)] if y <= times_tidal_periods[-1] else tidal_periods[-1] for y in roots_upper_bound]
                    t_ccur_tidal_periods_lower_bound = [tidal_periods[bisect.bisect_right(times_tidal_periods,y)] if y <= times_tidal_periods[-1] else tidal_periods[-1] for y in roots_lower_bound]
                    return crit_ccur_upper_bound, crit_ccur_lower_bound, roots_upper_bound ,roots_lower_bound, t_ccur_tidal_periods_upper_bound, t_ccur_tidal_periods_lower_bound

                def calculate_and_append_individual_horizontal_tidal_windows_point_based_method(previous_time, next_time, next_tidal_period, sailing_time_to_next_node, crit_ccur_bound_tidal_period, roots_bounds_tidal_period, t_ccur_tidal_periods_bounds_period, crit_ccur_tidal_periods, p_tidal_periods, t_ccur, mccur, horizontal_tidal_window, crit_vel_horizontal_tidal_window):
                    """ Function: calculates and appends the horizontal tidal windows (available sail-in and -out times given the horizontal tidal restrictions) and corresponding critical cross-current
                                  velocities to the appropriate lists for the 'point-based'-method

                        Input:
                            - previous_time: stopping (starting) time of the previous (current) tidal window
                            - next_time:
                            - next_tidal_period: the string containing information about the next tidal period (Flood/Ebb Start)
                            - sailing_time_to_next_node: time in s that the vessel needs to reach the node at which the restiction holds
                            - crit_ccur_bound_tidal_period: a list containing the upper and lower bounds around the critical cross-currents in m/s respectively
                            - roots_bounds_tidal_period: a list containing the lists with the times that the current velocity intersects with the upper and lower bound around the point-based current velocity respectively
                            - t_ccur_tidal_periods_bounds_period: a list containing the lists with the tidal periods for each time the current velocity intersects with the upper and lower bound around the point-based current velocity respectively
                            - crit_ccur_tidal_periods: a list containing the critical cross-currents in m/s during flood and ebb respectively
                            - p_tidal_periods: a list containing the ranges around the critical cross-currents in [-] during flood and ebb respectively
                            - t_ccur: a list of times of the times series of the governing cross-currents in s
                            - mccur: a list of the current velocities of the time series of the governing cross-currents in m/s
                            - horizontal_tidal_window: a list to which the horizontal tidal windows should be appended
                            - crit_vel_horizontal_tidal_window: a list to which the critical cross-currents corresponding to the horizontal tidal windows should be appended

                    """

                    # Determine the time index for the boundary of the next tidal period and if index is smaller than 0, then set index to 0
                    next_previous_time_tidal_periods = times_tidal_periods.index(next_time)-1
                    if next_previous_time_tidal_periods < 0: next_previous_time_tidal_periods = 0
                    # Calculate the time index of the starting point and ending point of the current tidal period (corrected for the sailing time)
                    start_time = previous_time
                    stop_time = next_time
                    index_previous_time = bisect.bisect_right(t_ccur, start_time)-1
                    index_now = bisect.bisect_right(t_ccur, vessel.waiting_time_start)
                    index_next = bisect.bisect_right(t_ccur, stop_time)
                    # If the flood tide is freely accessible (critical cross-current limit is -1)
                    tide_index = 0
                    next_tide_index = 1
                    if next_tidal_period == 'Flood Start': tide_index = 1
                    next_tide_index = next_tide_index - tide_index
                    if crit_ccur_tidal_periods[tide_index] == -1:
                        # If the accessibility of the ebb tide is not limited to the slack tides: add stop of tidal restriction at the start of flood tide and append 0 m/s as critical cross-current to list
                        if crit_ccur_tidal_periods[next_tide_index] != 'min' and index_previous_time != 0:
                            horizontal_tidal_window.append([times_tidal_periods[next_previous_time_tidal_periods], 'Stop'])
                            crit_vel_horizontal_tidal_window.append(0)
                        # And add the start of the tidal restriction at the end of flood tide and append 0 m/s as critical cross-current to list
                        horizontal_tidal_window.append([next_time, 'Start'])
                        crit_vel_horizontal_tidal_window.append(0)

                    # Else if the accessibility during the ebb tide is limited to the slack waters and there is current data within this tidal period
                    elif crit_ccur_tidal_periods[tide_index] == 'min':
                        if mccur[index_previous_time:index_next] != []:
                            # Interpolation of the maximum current velocities during this specific tidal period subtracted with the critical point-based cross-current and calculate the times at which the cross-current equals this critical limit
                            interp = sc.interpolate.CubicSpline(t_ccur[index_previous_time:index_next],[y - p_tidal_periods[tide_index] for y in mccur[index_previous_time:index_next]])
                            times = [x for x in interp.roots() if (x >= previous_time and x <= next_time)]
                            # If there are no crossings (minimum number of crossings should be at least two)
                            if len(times) < 1: times = [t_ccur[index_previous_time], t_ccur[index_next]]
                            # Take the first and last value of the crossings, and append to tidal windows and critical cross-currents: restriction starts at time of first crossing and ends at time of last crossing
                            time = next(item for item in times if item is not None)
                            if time < t_ccur[np.argmax(mccur[index_previous_time:index_next]) + index_previous_time]:
                                horizontal_tidal_window.append([next(item for item in times if item is not None),'Start'])
                                crit_vel_horizontal_tidal_window.append(p_tidal_periods[tide_index])
                                horizontal_tidal_window.append([next(item for item in reversed(times) if item is not None),'Stop'])
                                crit_vel_horizontal_tidal_window.append(p_tidal_periods[tide_index])
                            else:
                                horizontal_tidal_window.append([next(item for item in times if item is not None),'Stop'])
                                crit_vel_horizontal_tidal_window.append(p_tidal_periods[tide_index])
                            if index_previous_time == 0: #next_previous_time_tidal_periods
                                if mccur[index_now] <= p_tidal_periods[tide_index]:
                                    horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                                    crit_vel_horizontal_tidal_window.append(0)
                                else:
                                    horizontal_tidal_window.append([vessel.waiting_time_start,'Start'])
                                    crit_vel_horizontal_tidal_window.append(0)

                    # Else (flood tide is restricted and accessilibity of ebb tide is not limited to the slack waters) if there is current data within this tidal period
                    elif mccur[index_previous_time:index_next] != []:
                        # If the maximum cross-current in the tidal period does not exceed the upper bound limit: append to list that restriction stops at the maximum cross-current, with the corresponding maximum cross-current
                        if np.max(mccur[index_previous_time:index_next]) < crit_ccur_bound_tidal_period[0]:
                            if p_tidal_periods[tide_index] != 0:
                                if crit_ccur_tidal_periods[next_tide_index] != 0:
                                    horizontal_tidal_window.append([t_ccur[np.argmax(mccur[index_previous_time:index_next]) + index_previous_time],'Stop'])
                                    crit_vel_horizontal_tidal_window.append(np.max(mccur[index_previous_time:index_next]))
                                else:
                                    horizontal_tidal_window.append([t_ccur[np.argmax(mccur[index_previous_time:index_next]) + index_previous_time],'Start'])
                                    crit_vel_horizontal_tidal_window.append(np.max(mccur[index_previous_time:index_next]))
                            else:
                                if index_previous_time == 0: #next_previous_time_tidal_periods
                                    if mccur[index_now] <= p_tidal_periods[tide_index]:
                                        horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                                        crit_vel_horizontal_tidal_window.append(0)
                                    else:
                                        horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                                        crit_vel_horizontal_tidal_window.append(0)
                                    #horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                                    #crit_vel_horizontal_tidal_window.append(0)
                                else:
                                    if mccur[index_now] <= p_tidal_periods[tide_index]:
                                        horizontal_tidal_window.append([t_ccur[index_previous_time], 'Stop'])
                                        crit_vel_horizontal_tidal_window.append(0)
                                    else:
                                        horizontal_tidal_window.append([t_ccur[index_previous_time], 'Start'])
                                        crit_vel_horizontal_tidal_window.append(0)
                                    #horizontal_tidal_window.append([t_ccur[index_previous_time], 'Stop'])
                                    #crit_vel_horizontal_tidal_window.append(0)

                        # Limit the analysis to the tidal period from this time to the end of the period
                        index_previous_time2 = np.argmax(mccur[index_previous_time:index_next]) + index_previous_time
                        # If the maximum cross-current in the remainder of the tidal period does not exceed the lower bound limit and flood is accessible (crit-current is not 0 m/s and the ebb accessibility is not limited to the slack waters (critical cross-current is not 'min'): append to list that restriction starts at the minimum cross-current, with the corresponding minimum cross-current
                        if np.min(mccur[index_previous_time2:index_next]) > crit_ccur_bound_tidal_period[1] and crit_ccur_tidal_periods[tide_index] != 0 and p_tidal_periods[tide_index] != 0 and crit_ccur_tidal_periods[1] != 'min':
                            horizontal_tidal_window.append([t_ccur[np.argmin(mccur[index_previous_time2:index_next]) + index_previous_time2],'Start'])
                            crit_vel_horizontal_tidal_window.append(np.min(mccur[index_previous_time2:index_next]))
                        # Else if the flood tide is not accessible and the accessibility of the ebb tide is not limited to the slack waters: only append the start time of the tidal window and the corresponding cross-current (= 0 m/s)
                        elif p_tidal_periods[tide_index] == 0 and crit_ccur_tidal_periods[next_tide_index] != 'min':
                            horizontal_tidal_window.append([next_time, 'Start'])
                            crit_vel_horizontal_tidal_window.append(0)
                            if index_previous_time == 0 and p_tidal_periods[next_tide_index] != 0 and crit_ccur_tidal_periods[next_tide_index] != 0: #next_previous_time_tidal_periods
                                if mccur[index_now] <= p_tidal_periods[next_tide_index]:
                                    horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                                    crit_vel_horizontal_tidal_window.append(0)
                                else:
                                    horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                                    crit_vel_horizontal_tidal_window.append(0)
                            if next_previous_time_tidal_periods >= len(times_tidal_periods)-2:
                                horizontal_tidal_window.append([next_time, 'Stop'])
                                crit_vel_horizontal_tidal_window.append(0)

                        # Else if the accessibility of the ebb tide is limited to the slack waters: only append the start time of the tidal window and the corresponding cross-current (= critical cross-current during ebb)
                        elif crit_ccur_tidal_periods[next_tide_index] == 'min':
                            # Interpolation of the maximum current velocities during this specific tidal period subtracted with the critical point-based cross-current for ebb tide and calculate the times at which the cross-current equals this critical limit
                            interp = sc.interpolate.CubicSpline(t_ccur[index_previous_time:index_next],[y - p_tidal_periods[next_tide_index] for y in mccur[index_previous_time:index_next]])
                            times = [x for x in interp.roots() if (x >= previous_time and x <= next_time)]
                            # If there are no crossings (minimum number of crossings should be at least two)
                            if len(times) < 1: times = [t_ccur[index_previous_time], t_ccur[index_next]]
                            # Take the first and last value of the crossings, and append to tidal windows and critical cross-currents: restriction starts at time of first crossing
                            time = next(item for item in times if item is not None)
                            if time < t_ccur[np.argmax(mccur[index_previous_time:index_next])+index_previous_time]:
                                horizontal_tidal_window.append([time,'Start'])
                                crit_vel_horizontal_tidal_window.append(p_tidal_periods[1])
                            if index_previous_time == 0: #next_previous_time_tidal_periods
                                if (mccur[index_previous_time] <= crit_ccur_bound_tidal_period[1] and p_tidal_periods[tide_index] != 0) or (mccur[index_previous_time] > crit_ccur_bound_tidal_period[1] and p_tidal_periods[tide_index] == 0):
                                    horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                                    crit_vel_horizontal_tidal_window.append(0)
                                else:
                                    horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                                    crit_vel_horizontal_tidal_window.append(0)

                        if crit_ccur_tidal_periods[tide_index] == 0 and index_previous_time == 0:
                            horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                            crit_vel_horizontal_tidal_window.append(0)

                        elif crit_ccur_tidal_periods[next_tide_index] == 0 and index_previous_time == 0:
                            if mccur[index_now] < crit_ccur_bound_tidal_period[1] and p_tidal_periods[tide_index] != 0:
                                horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                                crit_vel_horizontal_tidal_window.append(0)
                            elif mccur[index_now] > crit_ccur_bound_tidal_period[0] and p_tidal_periods[tide_index] != 0:
                                horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                                crit_vel_horizontal_tidal_window.append(0)
                            elif mccur[index_now] > crit_ccur_bound_tidal_period[0] and p_tidal_periods[tide_index] == 0:
                                horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                                crit_vel_horizontal_tidal_window.append(0)

                    # If none of the above applies and hence the regular cross-current velocity bounds (resulting from finding_current_velocity_bounds_point_based_method) apply:
                    if type(crit_ccur_tidal_periods[tide_index]) != str and crit_ccur_tidal_periods[tide_index] != -1 and crit_ccur_tidal_periods[tide_index] != 0:
                        for bound_index in [0,1]:
                            bool_restriction = 'Stop'
                            if bound_index == 1: bool_restriction = 'Start'
                            for root in enumerate(roots_bounds_tidal_period[bound_index]):
                                if root[1] >= start_time and root[1] <= stop_time:
                                    # calculate the time index of the root
                                    index = bisect.bisect_right(t_ccur, root[1])
                                    # if the index is greater than the length of the time series: index = -1
                                    if index >= len(t_ccur): index -= 1
                                    # if the critical cross-current at the index is lower than the upper bound of the critical cross-current and is located in the flood tidal period and the accessibility during flood tide is not limited to the slack waters: append root (corrected for the sailing time) and the corresponding cross-current to the stopping times of the tidal restriction
                                    if mccur[index] < crit_ccur_bound_tidal_period[bound_index] and t_ccur_tidal_periods_bounds_period[bound_index][root[0]] == next_tidal_period:
                                        if crit_ccur_tidal_periods[tide_index] != -1:
                                            horizontal_tidal_window.append([root[1], bool_restriction])
                                            crit_vel_horizontal_tidal_window.append(crit_ccur_bound_tidal_period[bound_index])

                    #Data appended to the lists, nothing has to be returned
                    return

                # Predefining some parameters
                network = vessel.env.FG
                distance_to_next_node = 0
                mccur_nodes = []
                t_ccur = network.nodes[route[0]]['Info']['Current velocity'][0]
                max_ccur = []
                list_of_nodes = []
                critcur_nodes = []
                times_horizontal_tidal_windows = []
                crit_ccurs_horizontal_tidal_windows = []
                list_of_list_indexes = []
                critical_cross_current_method_in_restrictions = False
                point_based_method_in_restrictions = False
                t_ccur_raw = network.nodes[route[0]]['Info']['Current velocity'][0]

                # Start of calculation by looping over the nodes of the route
                for nodes in enumerate(route):
                    # - if a correction for the sailing time should be applied: the total distance should be keep track of
                    if nodes[0] == 0:
                        if 'bound' not in dir(vessel): vessel.bound = 'inbound'  # to be removed later: should be part of the vessel generator in which the user should define the initial direction of the vessel
                    else:
                        vessel.bound = VesselTrafficService.provide_sailing_direction(vessel, [route[route.index(nodes[1])-1], nodes[1]])

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
                    types = network.nodes[nodes[1]]['Info']['Horizontal tidal restriction']['Type']
                    specifications = network.nodes[nodes[1]]['Info']['Horizontal tidal restriction']['Specification']

                    # Determine which restriction applies to vessel
                    restriction_index, no_tidal_window = tidal_window_restriction_determinator(vessel, route, types, specifications, nodes[1], sailing_time_to_next_node)

                    # If no horizontal tidal window applies to vessel at the specific node: continue loop over nodes of the route of the vessel
                    if no_tidal_window:
                        continue

                    # Else if there applies a horizontal tidal window: continue tidal window calculation by predefining some parameters and importing critical cross-currents
                    ccur = []
                    crit_ccur = []
                    times_horizontal_tidal_window = []
                    crit_vel_horizontal_tidal_window = []
                    crit_ccur_flood_old = types[1][restriction_index][0]
                    crit_ccur_ebb_old = types[1][restriction_index][1]

                    # Determination of the direction of the vessel and the nodes that the vessel passes by in order, in order to extract the correct data
                    restriction_on_route = False
                    # -if the calculation is requested to be performed for visualization purposes (no route, but for single node), then use the given direction of the specified vessel
                    if visualization_calculation:
                        restriction_on_route = True
                        if vessel.bound == 'inbound':
                            n1 = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][0]
                            n2 = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][1]
                        elif vessel.bound == 'outbound':
                            n2 = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][0]
                            n1 = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][1]
                    # -else: select the nodes by the modelled direction of the vessel by nested looping over the nodes in the route
                    else:
                        for n1 in route[:nodes[0]]:
                            if n1 == network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][0]:
                                for n2 in route[nodes[0]:]:
                                    if n2 == network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][1]:
                                        restriction_on_route = True
                                        break
                                break

                    #Determining critical cross-current limits based on critical cross-current limit input (potentially integers [-1,0] or strings) by subtracting it with vessel-specific cross-current (if applicable)
                    # if passing the node during flood period is not restricted to the slack water (critical cross-current = type<str> = 'min') or freely or never accessible, then import cricital cross-current for flood and subtract it with vessel-specific cross-current (if applicable)
                    if crit_ccur_flood_old != -1 and crit_ccur_flood_old != 0 and type(crit_ccur_flood_old) != str:
                        crit_ccur_flood = crit_ccur_flood_old - vessel.metadata['max_cross_current']
                    # else if flood is freely accessible: set critical cross-current for flood at the very high level of 10 m/s
                    elif crit_ccur_flood_old == -1:
                        crit_ccur_flood = -1
                    elif crit_ccur_flood_old == 0:
                        crit_ccur_flood = 0
                    # if passing the node during ebb period is not restricted to the slack water (critical cross-current = type<str> = 'min') or freely or never accessible, then import cricital cross-current for ebb and subtract it with vessel-specific cross-current (if applicable)
                    if crit_ccur_ebb_old != -1 and crit_ccur_ebb_old != 0 and type(crit_ccur_ebb_old) != str:
                        crit_ccur_ebb = crit_ccur_ebb_old - vessel.metadata['max_cross_current']
                    # else if flood is freely accessible: set critical cross-current for ebb at the very high level of 10 m/s
                    elif crit_ccur_ebb_old == -1:
                        crit_ccur_ebb = -1
                    elif crit_ccur_ebb_old == 0:
                        crit_ccur_ebb = 0

                    #Import and store data
                    if types[0][restriction_index] == 'Critical cross-current':
                        critical_cross_current_method_in_restrictions,_,t_ccur,mccur,critcur,_,_ = calculate_and_store_normative_current_velocity_data(nodes[0],[n1,n2],sailing_time_to_next_node,t_ccur_raw,ccur,critcur_nodes,mccur_nodes,'Critical cross-current',crit_ccur,crit_ccur_flood,crit_ccur_ebb)
                        # If a plot is requested, then plot the cross-current velocities and the corresponding critical cross-currents
                        if plot:
                            axis.plot(t_ccur, [y + critcur[x] for x, y in enumerate(mccur)], color='lightcoral', alpha=0.4)
                            axis.plot(t_ccur, [y if y != 10 else None for y in critcur], color='lightcoral', linestyle='--', alpha=0.4)

                    elif types[0][restriction_index] == 'Point-based':
                        _,point_based_method_in_restrictions,t_ccur,mccur,critcur,times_tidal_periods,tidal_periods = calculate_and_store_normative_current_velocity_data(nodes[0],[n1,n2],sailing_time_to_next_node,t_ccur_raw,ccur,critcur_nodes,mccur_nodes,'Point-based')
                        # Setting some default parameters and lists and importing some information about the range outside the point of entry of vessels
                        p_flood = types[2][restriction_index][0]
                        p_ebb = types[2][restriction_index][1]

                        # if passing the node during flood period is not restricted to the slack water (critical cross-current = type<str> = 'min')
                        if type(crit_ccur_flood_old) != str:
                            # Interpolate the interpolated cross-current velocity, corrected for the sailing time, subtracted with the critical cross-current for the upper and lower bound separately and calculate roots, and corresponding next tidal periods
                            [crit_ccur_upper_bound_flood,
                             crit_ccur_lower_bound_flood,
                             roots_upper_bound_flood,
                             roots_lower_bound_flood,
                             t_ccur_tidal_periods_upper_bound_flood,
                             t_ccur_tidal_periods_lower_bound_flood] = finding_current_velocity_bounds_point_based_method(t_ccur, mccur, p_flood, crit_ccur_flood, times_tidal_periods, tidal_periods, sailing_time_to_next_node)
                        else:
                            [crit_ccur_upper_bound_flood,
                             crit_ccur_lower_bound_flood,
                             roots_upper_bound_flood,
                             roots_lower_bound_flood,
                             t_ccur_tidal_periods_upper_bound_flood,
                             t_ccur_tidal_periods_lower_bound_flood] = [crit_ccur_flood_old,crit_ccur_flood_old,[],[],[],[]]
                            crit_ccur_flood = crit_ccur_upper_bound_flood
                        # if passing the node during ebb period is not restricted to the slack water (critical cross-current = type<str> = 'min')
                        if type(crit_ccur_ebb_old) != str:
                            # Interpolate the interpolated cross-current velocity, corrected for the sailing time, subtracted with the critical cross-current for the upper and lower bound separately and calculate roots, and corresponding next tidal periods
                            [crit_ccur_upper_bound_ebb,
                             crit_ccur_lower_bound_ebb,
                             roots_upper_bound_ebb,
                             roots_lower_bound_ebb,
                             t_ccur_tidal_periods_upper_bound_ebb,
                             t_ccur_tidal_periods_lower_bound_ebb] = finding_current_velocity_bounds_point_based_method(t_ccur, mccur, p_ebb, crit_ccur_ebb, times_tidal_periods, tidal_periods,sailing_time_to_next_node)
                        else:
                            [crit_ccur_upper_bound_ebb,
                             crit_ccur_lower_bound_ebb,
                             roots_upper_bound_ebb,
                             roots_lower_bound_ebb,
                             t_ccur_tidal_periods_upper_bound_ebb,
                             t_ccur_tidal_periods_lower_bound_ebb] = [crit_ccur_ebb_old,crit_ccur_ebb_old,[],[],[],[]]
                            crit_ccur_ebb = crit_ccur_upper_bound_ebb

                        #Setting a a default starting time for the tidal period and looping over each tidal period to determine whether there are special cases (special restrictions or currents within tidal periods that do not reach the lower and/or upper bound)
                        previous_time = t_ccur[0]
                        for period in enumerate(times_tidal_periods):
                            #If next tidal period is ebb, hence current tidal period is flood:
                            if tidal_periods[period[0]] == 'Ebb Start':
                                calculate_and_append_individual_horizontal_tidal_windows_point_based_method(previous_time,
                                                                                                            period[1],
                                                                                                            tidal_periods[period[0]],
                                                                                                            sailing_time_to_next_node,
                                                                                                            [crit_ccur_upper_bound_flood,crit_ccur_lower_bound_flood],
                                                                                                            [roots_upper_bound_flood,roots_lower_bound_flood],
                                                                                                            [t_ccur_tidal_periods_upper_bound_flood,t_ccur_tidal_periods_lower_bound_flood],
                                                                                                            [crit_ccur_flood,crit_ccur_ebb],
                                                                                                            [p_flood,p_ebb],
                                                                                                            t_ccur,
                                                                                                            mccur,
                                                                                                            times_horizontal_tidal_window,
                                                                                                            crit_vel_horizontal_tidal_window)

                            # Else if next tidal period is flood, hence current tidal period is ebb:
                            elif tidal_periods[period[0]] == 'Flood Start':
                                calculate_and_append_individual_horizontal_tidal_windows_point_based_method(previous_time,
                                                                                                            period[1],
                                                                                                            tidal_periods[period[0]],
                                                                                                            sailing_time_to_next_node,
                                                                                                            [crit_ccur_upper_bound_ebb,crit_ccur_lower_bound_ebb],
                                                                                                            [roots_upper_bound_ebb,roots_lower_bound_ebb],
                                                                                                            [t_ccur_tidal_periods_upper_bound_ebb,t_ccur_tidal_periods_lower_bound_ebb],
                                                                                                            [crit_ccur_flood,crit_ccur_ebb],
                                                                                                            [p_flood,p_ebb],
                                                                                                            t_ccur,
                                                                                                            mccur,
                                                                                                            times_horizontal_tidal_window,
                                                                                                            crit_vel_horizontal_tidal_window)

                            # Setting new starting time for the tidal period equalling the end time of the previous tidal period
                            previous_time = period[1]

                        # Combine the lists for the restrictions and corresponding critical cross-currents and sort on the starting and stopping times of the restrictions
                        zipped_lists = zip(times_horizontal_tidal_window, crit_vel_horizontal_tidal_window)
                        sorted_pairs = sorted(zipped_lists)
                        tuples = zip(*sorted_pairs)
                        times_horizontal_tidal_window, crit_vel_horizontal_tidal_window = [list(tuple) for tuple in tuples]

                        # Set a default list of indexes in the restrictions list that should be removed based on the following conditions by looping over all the times
                        indexes_to_be_removed = []
                        for time in range(len(times_horizontal_tidal_window)):
                            # Skip the first index
                            if time == 0:
                                continue
                            # If there are two stopping restrictions in sequence in the list, add the index of the previous stopping time of the restriction to the list of indexes to be removed
                            elif times_horizontal_tidal_window[time][1] == 'Stop' and times_horizontal_tidal_window[time - 1][1] == 'Stop':
                                indexes_to_be_removed.append(time - 1)
                            # If there are two starting restrictions in sequence in the list, add the index of the next starting time of the restriction to the list of indexes to be removed
                            elif times_horizontal_tidal_window[time][1] == 'Start' and times_horizontal_tidal_window[time - 1][1] == 'Start':
                                indexes_to_be_removed.append(time)

                        # Remove the indexes in the list of the restrictions and corresponding cross-currents
                        for remove_index in list(reversed(indexes_to_be_removed)):
                            times_horizontal_tidal_window.pop(remove_index)
                            crit_vel_horizontal_tidal_window.pop(remove_index)

                        # Add the horizontal tidal window to the list of all horizontal tidal windows along the route
                        times_horizontal_tidal_windows.extend(times_horizontal_tidal_window)
                        crit_ccurs_horizontal_tidal_windows.extend(crit_vel_horizontal_tidal_window)
                        list_of_list_indexes.extend(np.ones(len(times_horizontal_tidal_window)))

                        # If a plot is requested: plot the maximum cross-currents and the critical point-based cross-currents
                        if plot:
                            axis.plot(t_ccur, [y for x, y in enumerate(mccur)], color='rosybrown')
                            axis.plot([x[0] for x in times_horizontal_tidal_window], crit_vel_horizontal_tidal_window,color='sienna', marker='o', linestyle='None')

                # Pick the maximum cross-current and cross-current limits of all the tidal restrictions
                max_ccur = [max(idx) for idx in zip(*mccur_nodes)]
                max_critcur = [critcur_nodes[np.argmax(val)][idx] for idx, val in enumerate(zip(*mccur_nodes))]

                if critical_cross_current_method_in_restrictions:
                    [times_horizontal_tidal_window,
                     crit_ccurs_horizontal_tidal_window] = calculate_and_append_individual_horizontal_tidal_windows_critical_cross_current_method(t_ccur,max_ccur,max_critcur)
                    times_horizontal_tidal_windows.extend(times_horizontal_tidal_window)
                    crit_ccurs_horizontal_tidal_windows.extend(crit_ccurs_horizontal_tidal_window)
                    list_of_list_indexes.extend(np.zeros(len(times_horizontal_tidal_window)))

                # If a plot is requested and the list of maximum cross-current limits is not empty: plot these maximum cross-currents except for when the tidal period is fully accessible (cross-current limit was set to 10 m/s)
                if plot and max_critcur != []:
                    axis.plot(t_ccur, [y if y != 10 else None for y in max_critcur], color='lightcoral', alpha=0.4,linestyle='--')

                # Return the modified time series for the cross-currents, critical cross-currents and if applicable the calculated horizontal tidal windows
                return t_ccur, max_ccur, max_critcur, times_horizontal_tidal_windows, crit_ccurs_horizontal_tidal_windows, list_of_list_indexes

            #Continuation of the calculation by defining the network
            network = vessel.env.FG

            # Calculate the maximum current-velocity at each node of the route in order to set the y-lim of the plot properly
            if plot:
                max_cur = []
                for node in route:
                    max_cur.append(np.max(network.nodes[node]['Info']['Current velocity'][1]))

            # Set some default parameters
            times_horizontal_tidal_window = []
            crit_ccurs_horizontal_tidal_window = []
            list_of_list_indexes = []
            list_indexes = [0, 1]

            # Determine the data of the governing horizontal tidal restriction (critical cross-current method) and/or the governing horizontal tidal restriction (point-based method)
            [t_ccur, max_ccur, max_crit_ccur, times_horizontal_tidal_window, crit_ccurs_horizontal_tidal_window, list_of_list_indexes] = tidal_windows_tightest_current_restriction_along_route(vessel, route, axis, plot, sailing_time_correction, visualization_calculation)

            # If the list of tidal windows are still empty: there is no horizontal tidal restriction for the particular vessel, so restriction stops at t_start and ends at t_end of the simulation
            if times_horizontal_tidal_window == []:
                times_horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                crit_ccurs_horizontal_tidal_window.append(0)
                list_of_list_indexes.append(0)  # integer = 0 meaning that the restriction is of type critical cross-current
                times_horizontal_tidal_window.append([np.max(t_ccur), 'Start'])
                crit_ccurs_horizontal_tidal_window.append(0)
                list_of_list_indexes.append(0)  # integer = 0 meaning that the restriction is of type critical cross-current

            # Combine the list of the horizontal tidal windows and the restriction types and critical cross-currents respectively and sort it chronologically
            list_of_list_indexes = [x for _, x in sorted(zip(times_horizontal_tidal_window, list_of_list_indexes))]
            crit_ccurs_horizontal_tidal_window = [x for _, x in sorted(zip(times_horizontal_tidal_window, crit_ccurs_horizontal_tidal_window))]
            times_horizontal_tidal_window.sort()

            # Set a default list of indexes in the restrictions list that should be removed based on the following conditions by nested looping over the restriction types and all the starting and stopping times of the restriction
            indexes_to_be_removed = []
            for list_index in list_indexes:
                for time1 in range(len(times_horizontal_tidal_window)):
                    # If the time is a starting time of a restriction and is the same as the selected restriction type: loop over the times again and find the next stopping time of the restriction with the same restriction type
                    if times_horizontal_tidal_window[time1][1] == 'Start' and list_of_list_indexes[time1] == list_index:
                        for time2 in range(len(times_horizontal_tidal_window)):
                            if time2 > time1 and times_horizontal_tidal_window[time2][1] == 'Stop' and list_of_list_indexes[time2] == list_index:
                                # select all the integers between the starting and stopping time of the restriction and add them to the indexes to be removed
                                indexes = np.arange(time1 + 1, time2, 1)
                                for index in indexes:
                                    indexes_to_be_removed.append(index)
                                break

            # Sort the indexes to be removed and remove the duplicates
            indexes_to_be_removed.sort()
            indexes_to_be_removed = list(dict.fromkeys(indexes_to_be_removed))

            # Remove the times of the restriction, the corresponding type of restriction and critical cross-current of the restriction by looping over the indexes to be removed
            for remove_index in list(reversed(indexes_to_be_removed)):
                times_horizontal_tidal_window.pop(remove_index)
                list_of_list_indexes.pop(remove_index)
                crit_ccurs_horizontal_tidal_window.pop(remove_index)

            # Set a new default list of indexes in the restrictions list that should be removed based on the following conditions by looping over all the times
            indexes_to_be_removed = []
            for time in range(len(times_horizontal_tidal_window)):
                # Skip the first index
                if time == 0:
                    continue
                # If there are two stopping restrictions in sequence in the list, add the index of the previous stopping time of the restriction to the list of indexes to be removed
                elif times_horizontal_tidal_window[time][1] == 'Stop' and times_horizontal_tidal_window[time - 1][1] == 'Stop':
                    indexes_to_be_removed.append(time - 1)
                # If there are two starting restrictions in sequence in the list, add the index of the next starting time of the restriction to the list of indexes to be removed
                elif times_horizontal_tidal_window[time][1] == 'Start' and times_horizontal_tidal_window[time - 1][1] == 'Start':
                    indexes_to_be_removed.append(time)

            # Again remove the times of the restriction, the corresponding type of restriction and critical cross-current of the restriction by looping over the indexes to be removed
            for remove_index in list(reversed(indexes_to_be_removed)):
                times_horizontal_tidal_window.pop(remove_index)
                list_of_list_indexes.pop(remove_index)
                crit_ccurs_horizontal_tidal_window.pop(remove_index)

            # If the first restriction in the list is a starting time and is greater than the starting time of the waiting time of the vessel: add a stopping condition at the starting time of the vessel and add the corresponding critical cross-current
            if times_horizontal_tidal_window[0][1] == 'Start' and times_horizontal_tidal_window[0][0] > vessel.waiting_time_start:
                times_horizontal_tidal_window.insert(0,[vessel.waiting_time_start, 'Stop'])
                crit_ccurs_horizontal_tidal_window.insert(0,max_crit_ccur[bisect.bisect_right(t_ccur, vessel.waiting_time_start)])
            # If the first restriction in the list is a stopping time and is greater than the starting time of the waiting time of the vessel: add a starting condition at the starting time of the vessel and add the corresponding critical cross-current
            elif times_horizontal_tidal_window[0][1] == 'Stop' and times_horizontal_tidal_window[0][0] > vessel.waiting_time_start:
                times_horizontal_tidal_window.insert(0,[vessel.waiting_time_start, 'Start'])
                crit_ccurs_horizontal_tidal_window.insert(0,max_crit_ccur[bisect.bisect_right(t_ccur, vessel.waiting_time_start)])

            # If a plot is requested: plot the governing maximum cross-current limits and cross-current with a corresponding label and add a lay-out
            if plot:
                axis.plot(t_ccur,[y if y != 10 else None for y in max_crit_ccur], color='indianred', linestyle='--')
                axis.plot([t for t in t_ccur], [y + max_crit_ccur[i] for i,y in enumerate(max_ccur)],color='indianred')
                axis.set_ylabel('Cross-current velocity [m/s]', color='indianred')
                axis.set_ylim([0, 1.05*np.max(max_cur)])
                y_loc = [y if y != 10 else None for y in max_crit_ccur][bisect.bisect_right(t_ccur, vessel.env.now + 0.5*vessel.metadata['max_waiting_time'])]
                # If the location of the label for the critical cross-current is not visible in the plot (exceeds the y-lim): calculate a new location for the label when the critical cross-current is visible
                if True in [y < 1.05*np.max(max_cur) for y in max_crit_ccur[bisect.bisect_right(t_ccur, vessel.env.now + 0.5*vessel.metadata['max_waiting_time']):bisect.bisect_right(t_ccur, vessel.env.now + 1.5*vessel.metadata['max_waiting_time'])]] and y_loc == None:
                    y_loc = next(item for item in [y if y < 1.05*np.max(max_cur) else None for y in max_crit_ccur][bisect.bisect_right(t_ccur, vessel.env.now + vessel.metadata['max_waiting_time']):] if item is not None)
                if y_loc != None and y_loc < 1.05*np.max(max_cur): #if a new value is found: plot the critical cross-currents limits corresponding to the horizontal tidal windows
                    axis.plot([x[0] for x in times_horizontal_tidal_window],[y for y in crit_ccurs_horizontal_tidal_window], color='indianred', marker='o',linestyle='None')
                    axis.text(vessel.env.now + vessel.metadata['max_waiting_time'],y_loc, 'Critical cross-current', color='indianred',horizontalalignment='center')

            # Return the horizontal tidal windows
            return times_horizontal_tidal_window

        def times_tidal_window(vessel,route,axes=[[],[]],plot=False,sailing_time_correction=True,visualization_calculation=False,ukc_calc=False):
            """ Function: calculates the windows available to sail-in and -out of the port by combining the tidal windows of the horizontal and vertical tidal restrictions given the tidal window polciy

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                    - plot: provide a visualization of the calculation for each vessel
                    - axes: list of two axes classes from the matplotlib package
                    - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

            """

            # If just a calculation of the net_ukc is required:
            if ukc_calc:
                interp_wdep, ukc_s, ukc_p, fwa = times_vertical_tidal_window(vessel, route, ukc_calc=ukc_calc)
                return interp_wdep, ukc_s, ukc_p, fwa

            # Else: calculate the tidal window restriction for the vertical and horizontal tide respectively
            list_of_times_vertical_tidal_window = times_vertical_tidal_window(vessel,route,axes[0],plot,sailing_time_correction,ukc_calc)
            list_of_times_horizontal_tidal_window = times_horizontal_tidal_window(vessel,route,axes[1],plot,sailing_time_correction,visualization_calculation)

            # Set some default parameters
            list_indexes = [0,1]
            times_tidal_window = []
            list_of_list_indexes = []

            #Append data to the lists and append a corresponding indentification index (0=vertical, 1=horizontal)
            for time in list_of_times_vertical_tidal_window:
                times_tidal_window.append(time)
                list_of_list_indexes.append(0)
            for time in list_of_times_horizontal_tidal_window:
                times_tidal_window.append(time)
                list_of_list_indexes.append(1)

            #Sort lists on time
            list_of_list_indexes = [x for _, x in sorted(zip(times_tidal_window, list_of_list_indexes))]
            times_tidal_window.sort()

            #Remove values that fall within restrictive periods of both tidal policies.
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

            #Sort the indexes that should be removed and remove duplicates
            indexes_to_be_removed.sort()
            indexes_to_be_removed = list(dict.fromkeys(indexes_to_be_removed))

            #Remove the values on the indexes that should be removed
            for remove_index in list(reversed(indexes_to_be_removed)):
                times_tidal_window.pop(remove_index)
                list_of_list_indexes.pop(remove_index)

            # Set a new default list of indexes in the restrictions list that should be removed based on the following conditions by looping over all the times
            indexes_to_be_removed = []
            for time in range(len(times_tidal_window)):
                if time == 0:
                    continue
                # If there are two stopping restrictions in sequence in the list, add the index of the previous stopping time of the restriction to the list of indexes to be removed
                elif times_tidal_window[time][1] == 'Stop' and times_tidal_window[time - 1][1] == 'Stop':
                    indexes_to_be_removed.append(time - 1)
                # If the first restriction in the list is a stopping time and is greater than the starting time of the waiting time of the vessel: add a starting condition at the starting time of the vessel and add the corresponding critical cross-current
                elif times_tidal_window[time][1] == 'Start' and times_tidal_window[time - 1][1] == 'Start':
                    indexes_to_be_removed.append(time)

            # Again remove the values on the indexes that should be removed
            for remove_index in list(reversed(indexes_to_be_removed)):
                times_tidal_window.pop(remove_index)
                list_of_list_indexes.pop(remove_index)

            # Return the final tidal windows
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
        available_sail_in_times = times_tidal_window(vessel,route,axes,plot,sailing_time_correction,visualization_calculation,ukc_calc)

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
            plt.title('Tidal window calculation for a '+str(vessel.name) + '-class ' +str(vessel.type) + ', sailing ' + str(vessel.bound))
            plt.xlim([vessel.env.now-0.5*vessel.metadata['max_waiting_time'], vessel.env.now + 1.5*vessel.metadata['max_waiting_time']])
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
    """Mixin class: Something which has a turning basin object properties as part of a lock complex [in SI-units] """

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
        opening_time, #a float which contains the time for the filling system to gradually open
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
        self.opening_time = opening_time
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
        """ Function which calculates the operation time: based on the constant or nearest in the signal of the water level difference

            Input:
                - environment: see init function"""
        h_dif = 0
        if type(self.wlev_dif) == list: #picks the wlev_dif from measurement signal closest to the discrete time
            h_dif = abs(self.wlev_dif[1][np.abs(self.wlev_dif[0]-(environment.now-self.simulation_start)).argmin()])
            
        elif type(self.wlev_dif) == float or type(self.wlev_dif) == int: #constant water level difference
            h_dif = abs(self.wlev_dif)

        operating_time = (self.opening_time / 2) + (2 * self.lock_width * self.lock_length * h_dif) / (self.disch_coeff * self.opening_area * math.sqrt(2 * self.grav_acc * h_dif))

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

class PassSection:
    """Mixin class: Collection of functions that release and request sections. Important to obey the traffic regulations (safety distance and one-way-traffic) """

    def release_access_previous_section(vessel, origin):
        """ Function: when a vessel sails out of a section, it releases the request of the previous section

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - origin: string that contains the node that the vessel is currently on

        """

        # Reversely loop over the nodes of the route of the vessel (from the node that it is currently on backwards)
        for n in reversed(vessel.route[:vessel.route.index(origin)]):
            # If a junction is encountered on that node: extract information of the previous junction
            if 'Junction' in vessel.env.FG.nodes[n]:
                junction = vessel.env.FG.nodes[n]['Junction'][0]
                # Loop over the section of that junction
                for section in enumerate(junction.section):
                    # Pick the correct section by checking which section contains the current node of the vessel:
                    if origin not in list(section[1].keys()):
                        continue
                    # Release the request of that section made previously by the vessel
                    section[1][origin].release(vessel.request_access_section)

                    # If the section is of type 'one-way traffic':
                    if junction.type[section[0]] == 'one-way_traffic':
                        # If the entrance/exit resources are not in the previous junction: find the current junction and release the specific requests at that junction
                        if 'access1' not in dir(junction):
                            junction = vessel.env.FG.nodes[origin]['Junction'][0]
                            for section in enumerate(junction.section):
                                if n not in list(section[1].keys()):
                                    continue
                                junction.access2[0][n].release(vessel.request_access_entrance_section) #section[0]
                                junction.access1[0][origin].release(vessel.request_access_exit_section) #section[0]
                        # Else: release the specific requests of the entrance/exit resources at the previous junction
                        else:
                            junction.access1[0][n].release(vessel.request_access_entrance_section) #section[0]
                            junction.access2[0][origin].release(vessel.request_access_exit_section) #section[0]
                        break
                break
        return

    def request_access_next_section(vessel, origin, destination):
        """ Function: when a vessel sails out of a section, it releases the request of the previous section

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - origin: string that contains the node of the route that the vessel is currently on
                - destination: string that contains the node of the route that the vessel is heading to

        """

        # Loop over the nodes of the route of the vessel (from the node that it is heading to onwards)
        for n in vessel.route[vessel.route.index(destination):]:
            # If a junction is encountered on that node: extract information of the current junction
            if 'Junction' in vessel.env.FG.nodes[n]:
                junction = vessel.env.FG.nodes[origin]['Junction'][0]
                # Loop over the section of that junction
                for section in enumerate(junction.section):
                    # Pick the correct section by checking which section contains the current node of the vessel:
                    if n not in list(section[1].keys()):
                        continue
                    # Setting the stopping distance and stopping time
                    vessel.stopping_distance = 15 * vessel.L
                    vessel.stopping_time = vessel.stopping_distance / vessel.v
                    # If there is already a vessel present in the section and the time the current vessel arrives within the safety distance: request access and yield timeout until safety distance is complied to
                    if section[1][n].users != [] and (section[1][n].users[-1].ta + vessel.stopping_time) > vessel.env.now:
                        vessel.request_access_section = section[1][n].request()
                        section[1][n].users[-1].id = vessel.id
                        section[1][n].users[-1].ta = (section[1][n].users[-2].ta + vessel.stopping_time)
                        yield vessel.env.timeout((section[1][n].users[-2].ta + vessel.stopping_time) - vessel.env.now)
                    # Else if there are no other vessels present in the section: request access
                    else:
                        vessel.request_access_section = section[1][n].request()
                        section[1][n].users[-1].ta = vessel.env.now
                        section[1][n].users[-1].id = vessel.id

                    # If the section is of type 'one-way traffic':
                    if junction.type[section[0]] == 'one-way_traffic':
                        # If the entrance/exit resources are not in the current junction: find the next junction and request the specific requests at that junction
                        if 'access1' not in dir(junction):
                            junction = vessel.env.FG.nodes[n]['Junction'][0]
                            for section in enumerate(junction.section):
                                if origin not in list(section[1].keys()):
                                    continue

                                vessel.request_access_entrance_section = junction.access2[0][origin].request() #section[0]
                                junction.access2[0][origin].users[-1].id = vessel.id #section[0]
                                vessel.request_access_exit_section = junction.access1[0][n].request() #section[0]
                                junction.access1[0][n].users[-1].id = vessel.id #section[0]

                        # Else: request the specific requests for the entrance/exit resources at the current junction
                        else:
                            vessel.request_access_entrance_section = junction.access1[0][origin].request() #section[0]
                            junction.access1[0][origin].users[-1].id = vessel.id #section[0]
                            vessel.request_access_exit_section = junction.access2[0][n].request() #section[0]
                            junction.access2[0][n].users[-1].id = vessel.id #section[0]
                        break
                break
        return

class PassTerminal:
    """Mixin class: Collection of interacting functions that handle the vessels that call at a terminal and take the correct measures"""

    def waiting_time_for_tidal_window(vessel,route,delay=0,plot=False):
        """ Function: calulates the time that a vessel has to wait depending on the available tidal windows

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - route: a list of strings that resemble the route of the vessel (can be different than the vessel.route)
                - delay: a delay that can be included to calculate a future situation
                - plot: bool that specifies if a plot is requested or not

        """

        # If the sail-in times of the vessel is not calculated before: request VesselTrafficService what are the available sail-in times given the tidal window policy
        if 'sail_in_times' not in dir(vessel):
            vessel.sail_in_times = VesselTrafficService.provide_sail_in_times_tidal_window(vessel,route=route,plot=plot)
        # If a vessel is bound for offshore (meaning that sail-out times have already been calculated): store its sail-in times in a parameter and use the sail-out times as input (sail-in times)
        if vessel.bound == 'outbound':
            sail_in_times = vessel.sail_in_times
            vessel.sail_in_times = vessel.sail_out_times

        # Set default parameters
        waiting_time = 0
        current_time = vessel.env.now+delay

        # Loop over the available sail-in (or sail-out) times:
        for t in range(len(vessel.sail_in_times)):
            # If the next sail-in time contains a starting condition for a restriction: if it is the last time, then let the vessel wait wait for this time, else continue the loop
            if vessel.sail_in_times[t][1] == 'Start':
                if t == len(vessel.sail_in_times)-1:
                    waiting_time = vessel.sail_in_times[t][0] - current_time
                    break
                else:
                    continue
            # If the current time of the vessel is greater or equal to the next sail-in time containing a stopping condition for a restriction and is smaller than the next starting time of a restriction: waiting time = 0
            if current_time >= vessel.sail_in_times[t][0]:
                waiting_time = 0
                if t == len(vessel.sail_in_times)-1 or current_time < vessel.sail_in_times[t+1][0]:
                    break
            # Else if the current time of the vessel is smaller or equal to the next sail-in time containing a stopping condition for a restriction
            elif current_time <= vessel.sail_in_times[t][0]:
                # And is smaller than the previous starting time of a restriction: waiting time = 0
                if current_time < vessel.sail_in_times[t-1][0]:
                    waiting_time = 0
                # Waiting time = next stopping time - current time
                else:
                    waiting_time = vessel.sail_in_times[t][0] - current_time
                break
            # Else if it is the last time, then let the vessel wait wait for this time
            elif t == len(vessel.sail_in_times) - 1:
                waiting_time = vessel.sail_in_times[t][0] - current_time
            # Else if none of the above conditions hold: continue the loop
            else:
                continue

        # If vessel is bound for offshore: reset the sail-in times to their original
        if vessel.bound == 'outbound':
            vessel.sail_in_times = sail_in_times

        # Else if the vessel is bound for the terminal: request the sail-out times by reversing the route
        elif vessel.bound == 'inbound':
            network = vessel.env.FG
            distance_to_node = 0
            route.reverse()
            # Calculate the total time that a vessel will spend in the port before returning: sailing time of the vessel + 2 times (de)berthing time + (un)loading time
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
            # Calculate delay and include in current time
            delay = sailing_time + 2 * vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60 + waiting_time
            current_time = vessel.env.now + delay
            # Request the sail-out times by temporarily setting the vessel bound as outbound
            vessel.bound = 'outbound'
            vessel.sail_out_times = VesselTrafficService.provide_sail_in_times_tidal_window(vessel,route=route,plot=plot)
            vessel.bound = 'inbound'

            # Loop over the provided sail-out times: if there is no suitable tidal window or if the waiting time is too large: set waiting time to maximum waiting time of vessel (it will return without entering the port)
            for t in range(len(vessel.sail_out_times)):
                # If the next sail-in time contains a starting condition for a restriction: continue the loop
                if vessel.sail_out_times[t][1] == 'Start':
                    continue
                # If the current time of the vessel is greater or equal to the next sail-in time containing a stopping condition for a restriction and is smaller than the next starting time of a restriction: break loop
                if current_time >= vessel.sail_out_times[t][0]:
                    if t == len(vessel.sail_out_times)-1 or current_time < vessel.sail_out_times[t+1][0]:
                        break
                # Else if the current time of the vessel is smaller or equal to the next sail-in time containing a stopping condition for a restriction
                elif current_time <= vessel.sail_out_times[t][0]:
                    # Determine if the waiting time is allowed or not: if not revise waiting time to maximum waiting time, else break the loop
                    if vessel.sail_out_times[t][0]-current_time >= vessel.metadata['max_waiting_time']:
                        waiting_time = vessel.metadata['max_waiting_time']
                    else:
                        break
                # Else if it is the last time, then revise waiting time to maximum waiting time
                elif t == len(vessel.sail_out_times)-1:
                    waiting_time = vessel.metadata['max_waiting_time']
                # Else if none of the conditions holds: continue the loop
                else:
                    continue
            # Reset the route
            route.reverse()

        # Return the vessel waiting time. If the vessel is not able to return within the maximum waiting time at the terminal: the waiting time is unacceptable and the vessel will return
        return waiting_time

    def move_to_anchorage(vessel,node):
        """ Function: moves a vessel to the anchorage area instead of continuing its route to the terminal if a vessel is required to wait in the anchorage area

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node: a string that contains the node of the route that the vessel is currently on

        """

        # Set some default parameters:
        network = vessel.env.FG
        vessel.waiting_in_anchorage = True
        nodes_of_anchorages = []
        capacity_of_anchorages = []
        users_of_anchorages = []
        sailing_distances_from_anchorages = []

        # Loop over the nodes of the network and identify all the anchorage areas:
        for node_anchorage in network.nodes:
            if 'Anchorage' in network.nodes[node_anchorage]:
                #Extract information over the individual anchorage areas: capacity, users, and the sailing distance to the anchorage area from the designated terminal the vessel is planning to call
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

        # Sort the lists based on the sailing distance to the anchorage area from the designated terminal the vessel is planning to call
        sorted_nodes_anchorages = [nodes for (distances,nodes) in sorted(zip(sailing_distances_from_anchorages, nodes_of_anchorages))]
        sorted_users_of_anchorages = [nodes for (distances,nodes) in sorted(zip(sailing_distances_from_anchorages, users_of_anchorages))]
        sorted_capacity_of_anchorages = [nodes for (distances,nodes) in sorted(zip(sailing_distances_from_anchorages, capacity_of_anchorages))]

        # Take the anchorage area that is closest to the designated terminal the vessel is planning to call if there is sufficient capacity:
        node_anchorage = sorted_nodes_anchorages[np.argmin(sailing_distances_from_anchorages)]
        for node_anchorage_area in enumerate(sorted_nodes_anchorages):
            if sorted_users_of_anchorages[node_anchorage_area[0]] < sorted_capacity_of_anchorages[node_anchorage_area[0]]:
                node_anchorage = node_anchorage_area[1]
                break

        # If there is not an available anchorage area: leave the port after entering the anchorage area
        if node_anchorage != node_anchorage_area[1]:
           vessel.return_to_sea = True
           vessel.waiting_time = vessel.metadata['max_waiting_time']
        # Set the route that the vessel will take after calling at the terminal (back to the origin) and after waiting in the anchorage area
        anchorage = network.nodes[node_anchorage]['Anchorage'][0]
        vessel.route_after_anchorage = []
        vessel.true_origin = vessel.route[0]
        current_time = vessel.env.now
        vessel.route_after_anchorage = nx.dijkstra_path(vessel.env.FG, node_anchorage, vessel.route[-1])
        vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route[-1], vessel.route[0])
        # Start the moving process by setting the route of the vessel from current node to chosen anchorage area
        yield from Movable.pass_edge(vessel,vessel.route[node], vessel.route[node+1])
        vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[node+1], node_anchorage)
        vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[node]]
        vessel.env.process(vessel.move())
        return

    def pass_anchorage(vessel, node):
        """ Function: function that handles a vessel waiting in an anchorage area

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node: string that contains the node of the route that the vessel is moving to (the anchorage area)

        """

        # Set default parameter and extract information of anchorage area
        network = vessel.env.FG
        anchorage = network.nodes[node]['Anchorage'][0]

        # Moves the vessel to the node of the anchorage area
        #yield from Movable.pass_edge(vessel, vessel.route[vessel.route.index(node) - 1],vessel.route[vessel.route.index(node)])

        # Request access to the anchorage area and log this to the anchorage area log and vessel log (including the calculated value for the net ukc)
        vessel.anchorage_access = anchorage.anchorage_area[node].request()
        yield vessel.anchorage_access
        anchorage.log_entry("Vessel arrival", vessel.env.now, len(anchorage.anchorage_area[node].users),nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
        current_time = vessel.env.now
        ukc = VesselTrafficService.provide_ukc_clearance(vessel,node)
        vessel.log_entry("Waiting in anchorage start", vessel.env.now, ukc,nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )

        # Determine the sailing distance to the destined terminal (important for later in the calculation)
        edge = vessel.route_after_anchorage[-2], vessel.route_after_anchorage[-1]
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        sailing_distance = 0
        for nodes in enumerate(vessel.route_after_anchorage[:-1]):
            _, _, distance = vessel.wgs84.inv(vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0]]]['geometry'].x,
                                              vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0]]]['geometry'].y,
                                              vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0] + 1]]['geometry'].x,
                                              vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0] + 1]]['geometry'].y)
            sailing_distance += distance

        # If the vessel is allowed in (so has not to return to sea) and should wait in the anchorage area for an available berth
        if vessel.return_to_sea == False and vessel.waiting_for_availability_terminal:
            # Vessel waits for the available berth, or if it takes longer than the permitted maximum waiting time: the vessel will wait until that specific waiting time
            yield vessel.waiting_time_in_anchorage | vessel.env.timeout(vessel.metadata['max_waiting_time'])
            new_current_time = vessel.env.now
            # If waiting time is greater or equal than the maximum waiting time: vessel returns to sea and releases its request for the berth at the terminal of call
            if new_current_time - current_time >= vessel.metadata['max_waiting_time']:
                vessel.return_to_sea = True
                vessel.waiting_time = 0
                if terminal.type == 'quay': terminal = terminal.terminal[edge[0]]
                elif terminal.type == 'jetty': terminal = terminal.terminal[vessel.index_jetty_position][edge[0]]
                terminal.release(vessel.waiting_time_in_anchorage)
            # Else:
            else:
                # If terminal of call is of type 'quay': determine the quay position, request the quay length and adjust the available quay length
                if terminal.type == 'quay':
                    vessel.index_quay_position, _ = PassTerminal.request_quay_position(vessel, terminal)
                    PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)
                    terminal = terminal.terminal[edge[0]]
                # Else if terminal of call is of type 'jetty': request terminal again (with priority) before releasing the initial request (based on which the waiting time was calculated)
                elif terminal.type == 'jetty':
                    terminal = terminal.terminal[vessel.index_jetty_position][edge[0]]
                    vessel.access_terminal = terminal.request(priority=-1)
                    terminal.release(vessel.waiting_time_in_anchorage)
                    yield vessel.access_terminal

                # Determine the new sail-in times of the vessel (since the vessel had to wait in the anchorage area, the available sail-in times are now different)
                vessel.sail_in_times = VesselTrafficService.provide_sail_in_times_tidal_window(vessel, vessel.route_after_anchorage)
                # Loop over the sail-in times to determine if the vessel should wait or not, or even has to return to sea (due to non-allowable waiting time by exceeding the set maximum)
                for t in range(len(vessel.sail_in_times)):
                    # If the next sail-in time contains a starting condition for a restriction: if it is the last time, then let the vessel wait wait for this time (if permitted, else return to sea), else continue the loop
                    if vessel.sail_in_times[t][1] == 'Start':
                        if t == len(vessel.sail_in_times) - 1:
                            waiting_time = vessel.sail_in_times[t][0] - current_time
                            if waiting_time >= vessel.metadata['max_waiting_time']:
                                vessel.return_to_sea = True
                                vessel.waiting_time = 0
                                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                    terminal.release(vessel.access_terminal)
                            else:
                                # Yield waiting time and if terminal of call is of type 'jetty': calculate the estimated time of arrival and departure (important for the queuing procedure)
                                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                    terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.sail_in_times[t][0] - new_current_time
                                    vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                    vessel.bound = 'outbound'
                                    waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel,route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                                    vessel.bound = 'inbound'
                                    terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                    terminal.users[-1].type = vessel.type
                                yield vessel.env.timeout(vessel.sail_in_times[t][0] - new_current_time)
                            break
                        else:
                            continue
                    # If the current time of the vessel is greater or equal to the next sail-in time containing a stopping condition for a restriction and is smaller than the next starting time of a restriction
                    if new_current_time >= vessel.sail_in_times[t][0]:
                        if t == len(vessel.sail_in_times) - 1 or new_current_time < vessel.sail_in_times[t + 1][0]:
                            # No waiting time and if terminal of call is of type 'jetty': calculate the estimated time of arrival and departure (important for the queuing procedure)
                            if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
                                vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                vessel.bound = 'outbound'
                                waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                                vessel.bound = 'inbound'
                                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                terminal.users[-1].type = vessel.type
                            break
                    # Else if the current time of the vessel is smaller or equal to the next sail-in time containing a stopping condition for a restriction
                    elif new_current_time <= vessel.sail_in_times[t][0]:
                        # If the current time of the vessel is smaller than the next starting time of a restriction:
                        if new_current_time < vessel.sail_in_times[t - 1][0]:
                            # No waiting time and if terminal of call is of type 'jetty': calculate the estimated time of arrival and departure (important for the queuing procedure)
                            if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
                                vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                vessel.bound = 'outbound'
                                waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                                vessel.bound = 'inbound'
                                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                terminal.users[-1].type = vessel.type
                        # Else: there is waiting time: determine if allowed or not (exceeds the maximum permitted or not)
                        else:
                            waiting_time = vessel.sail_in_times[t][0] - current_time
                            if waiting_time >= vessel.metadata['max_waiting_time']:
                                vessel.return_to_sea = True
                                vessel.waiting_time = 0
                                terminal.release(vessel.access_terminal)
                            else:
                                # Yield waiting time and if terminal of call is of type 'jetty': calculate the estimated time of arrival and departure (important for the queuing procedure)
                                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                    terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.sail_in_times[t][0] - new_current_time
                                    vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                    vessel.bound = 'outbound'
                                    waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                                    vessel.bound = 'inbound'
                                    terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                    terminal.users[-1].type = vessel.type
                                yield vessel.env.timeout(vessel.sail_in_times[t][0] - new_current_time)
                        break
                    # Else if it is the last time, then there is waiting time, so determine if allowed or not (exceeds the maximum permitted or not)
                    elif t == len(vessel.sail_in_times) - 1:
                        waiting_time = vessel.sail_in_times[t][0] - current_time
                        if waiting_time >= vessel.metadata['max_waiting_time']:
                            vessel.return_to_sea = True
                            vessel.waiting_time = 0
                            terminal.release(vessel.access_terminal)
                        else:
                            # Yield waiting time and if terminal of call is of type 'jetty': calculate the estimated time of arrival and departure (important for the queuing procedure)
                            if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.sail_in_times[t][0] - new_current_time
                                vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                vessel.bound = 'outbound'
                                waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                                vessel.bound = 'inbound'
                                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                terminal.users[-1].type = vessel.type
                            yield vessel.env.timeout(vessel.sail_in_times[t][0] - new_current_time)
                    # Else if none of the conditions holds: continue the loop
                    else:
                        continue

        # Else if the vessel (so has not to return to sea) and should wait in the anchorage area for an available tidal window:
        elif vessel.return_to_sea == False and vessel.waiting_time:
            # If terminal of call is of type 'jetty': calculate the estimated time of arrival and departure (important for the queuing procedure) including the waiting time for the tidal window
            if terminal.type == 'jetty':
                terminal = terminal.terminal[vessel.index_jetty_position][edge[0]]
                terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.waiting_time
                vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                vessel.bound = 'outbound'
                waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                vessel.bound = 'inbound'
                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                terminal.users[-1].type = vessel.type
            # Else if terminal of call is of type 'quay': vessel will pick predetermined quay position, get this length, so adjust available quay lengths
            elif terminal.type == 'quay':
                PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)
            # Determine the new time: if the maximum allowabel waiting time will be or has been exceeded: return to sea, otherwise yield waiting time
            new_current_time = vessel.env.now + vessel.waiting_time
            if new_current_time - current_time >= vessel.metadata['max_waiting_time']:
                vessel.return_to_sea = True
                vessel.waiting_time = 0
            yield vessel.env.timeout(vessel.waiting_time)

        # If vessel does not has to return to sea: log this in the anchorage log and vessel log, set the route from anchorage to terminal, release the access to the section to the anchorage area, and request access to the first section on its route and initate the move
        if vessel.return_to_sea == False:
            ukc = VesselTrafficService.provide_ukc_clearance(vessel,node)
            vessel.log_entry("Waiting in anchorage stop", vessel.env.now, ukc, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
            vessel.route = vessel.route_after_anchorage
        # Else if the vessel has to return to sea: log this as well in the log files of the anchorage and vessel, set route back to origin, release the access to the section to the anchorage area, and request access to the first section on its route and initate the move
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

    def request_quay_position(vessel, terminal):
        """ Function that claims a length along the quay equal to the length of the vessel itself and calculates the relative position of the vessel along the quay. If there are multiple
            relative positions possible, the vessel claims the first position. If there is no suitable position availalble (vessel does not fit), then it returns the action
            of moving to the anchorage area.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class

        """

        # Set some default parameters
        available_quay_lengths = [0]
        aql = terminal.available_quay_lengths #the current configuration of vessels located at the quay
        index_quay_position = 0
        move_to_anchorage = False

        # Loop over the locations of the current configuration of vessels located at the quay
        for index in range(len(aql)):
            # If the index of the locaton is 0, or if the previous location is the same as the current location (and hence the index of the location is not 0) or if the location is not available (value = 1):
            if index == 0 or aql[index][1] == aql[index - 1][1] or aql[index][0] == 1:
                # Continue, else if its the last index and there is not yet a suitable index found for an available location: return that vessel has to move to anchorage
                if index == len(aql) - 1 and not index_quay_position:
                    move_to_anchorage = True
                continue

            # If there is an available location: append indexes to list
            available_quay_lengths.append(aql[index][1] - aql[index - 1][1])

            # Loop over the list:
            for jndex in range(len(available_quay_lengths)):
                # If there is the available location is suitable (available length of that location is greater than the vessel length): return index and break loop
                if vessel.L <= available_quay_lengths[jndex]:
                    index_quay_position = index
                    break

                # Else: if there were not available locations found: return that vessel has to move to anchorage
                elif jndex == len(available_quay_lengths) - 1 and not index_quay_position:
                    move_to_anchorage = True

            # The index can only still be default if the vessel has to move to the anchorage area: so break the loop then
            if index_quay_position != 0:
                break

        return index_quay_position, move_to_anchorage

    def calculate_quay_length_level(terminal):
        """ Function that keeps track of the maximum length that is available at the quay

            Input:
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class

        """

        # Set default parameters
        aql = terminal.available_quay_lengths
        available_quay_lengths = [0]

        # Loop over the position indexes
        for index in range(len(aql)):
            # If the index of the locaton is 0, or if the previous location is the same as the current location (and hence the index of the location is not 0) or if the location is not available (value = 1):
            if index == 0 or aql[index][1] == aql[index - 1][1] or aql[index][0] == 1:
                # Continue, else if its the last index and there is not yet a suitable index found for an available location: return that available length is the last one in the list (=0)
                if index == len(aql) - 1:
                    new_level = available_quay_lengths[-1]
                continue

            # If there is an available location: append length to list and return the maximum of the list
            available_quay_lengths.append(aql[index][1] - aql[index - 1][1])
            new_level = np.max(available_quay_lengths)
        return new_level

    def adjust_available_quay_lengths(vessel, terminal, index_quay_position):
        """ Function that adjusts the available quay lenghts and positions given a honored request of a vessel at a given position

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class
                - index_quay_position: quay position index at which the vessel is located at the quay with respect to the other vessels

        """

        # Import the locations of the current configuration of vessels located at the quay
        aql = terminal.available_quay_lengths

        # Determine the current maximum available length of the terminal
        old_level = PassTerminal.calculate_quay_length_level(terminal)

        # If the value of the position index before the honered quay position (start of the available position) is still available (=0), change it to 1
        if aql[index_quay_position - 1][0] == 0:
            aql[index_quay_position - 1][0] = 1

        # If the value of the honered quay position (end of the available position) is still available (=0) and the end of this position equals the start of the position added with the vessel length, change it to 1
        if aql[index_quay_position][0] == 0 and aql[index_quay_position][1] == aql[index_quay_position - 1][1] + vessel.L:
            aql[index_quay_position][0] = 1

        # Else insert a new stopping location in the locations of the current configuration of vessels located at the quay by twice adding the vessel length to the start position of the location, once with a occupied value (=1), followed by a available value (=0)
        else:
            aql.insert(index_quay_position, [1, vessel.L + aql[index_quay_position - 1][1]])
            aql.insert(index_quay_position + 1, [0, vessel.L + aql[index_quay_position - 1][1]])

        # Replace the list of the locations of the current configuration of vessels located at the quay of the terminal
        terminal.available_quay_lengths = aql
        # Calculate the quay position and append to the vessel (mid-length of the vessel + starting length of the position)
        vessel.quay_position = 0.5 * vessel.L + aql[index_quay_position - 1][1]
        # Determine the new current maximum available length of the terminal
        new_level = PassTerminal.calculate_quay_length_level(terminal)
        # If the old level does not equal (is greater than) the new level and the vessel does not have to wait in the anchorage first: then claim the difference between these lengths
        if old_level != new_level and vessel.waiting_in_anchorage != True:
            terminal.length.get(old_level - new_level)
        # Else if the vessel has to wait in the anchorage first: calculate the difference between the lengths corrected by the vessel length to be claimed by the vessel (account for this vessel, so that it has priority over new vessels)
        elif vessel.waiting_in_anchorage == True:
            new_level = old_level-vessel.L-new_level
            # If this difference is negative: give absolute length back to terminal
            if new_level < 0:
                terminal.length.put(-new_level)
            # Else if this difference is positive: claim this length of the terminal
            elif new_level > 0:
                terminal.length.get(new_level)
        return

    def request_terminal_access(vessel, edge, node):
        """ Function: function that handles the request of a vessel to access the terminal of call: it lets the vessel move to the correct terminal (quay position and jetty) or moves it to the
            anchorage area to wait on either the terminal (quay or jetty) availability or tidal window

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - edge: list of two strings that form the edge at which the terminal is located
                - node: string that contains the node of the route that the vessel is currently on (either the origin or anchorage area)

        """

        # Set some default parameters
        node = vessel.route.index(node)
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        vessel.move_to_anchorage = False
        vessel.waiting_in_anchorage = False
        vessel.waiting_for_availability_terminal = False

        # Function that is used in the request procedure
        def checks_waiting_time_due_to_tidal_window(vessel, route, node, maximum_waiting_time = False):
            """ Function: function that checks if the vessel arrives beyond a tidal window and calculates the waiting time if that is the case (or return no waiting time otherwise)

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route: list of strings that resemble the route of the vessel (can be different than vessel.route)
                    - node: string that contains the node of the route that the vessel is currently on (origin)
                    - maximum_waiting_time: bool that specifies if there is a maximum to the waiting time of the vessel

            """

            # Calculate the waiting time using the waiting_time_for_tidal_window-function
            vessel.waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel,route=route,delay=0,plot=False)
            # If there is a maximum waiting time and this is exceeded: vessel will have to return to sea with zero waiting time
            if vessel.waiting_time >= vessel.metadata['max_waiting_time'] and maximum_waiting_time:
                vessel.return_to_sea = True
                vessel.waiting_time = 0
            # Else: vessel does not have to return to sea
            else:
                vessel.return_to_sea = False

        # Calculate the waiting time and whether this waiting time is acceptable
        checks_waiting_time_due_to_tidal_window(vessel, route = vessel.route, node = node, maximum_waiting_time=True)

        # Set default parameter
        available_turning_basin = False

        # Loop over the nodes of the route and determine whether there is turning basin that suits the vessel (dependent on a vessel length restriction)
        for basin in vessel.route:
            if 'Turning Basin' in vessel.env.FG.nodes[basin].keys():
                turning_basin = vessel.env.FG.nodes[basin]['Turning Basin'][0]
                if turning_basin.length >= vessel.L:
                    available_turning_basin = True
                    break

        # If there is no available turning basin: vessel will have to return to sea without waiting in the anchorage area
        if available_turning_basin == False:
            vessel.return_to_sea = True
            vessel.waiting_time = 0

        # If vessel does not have to return to sea:
        if not vessel.return_to_sea:
            # If the terminal is of type 'jetty:
            if terminal.type == 'jetty':
                # Set empty default lists
                minimum_waiting_time = []
                vessels_waiting = []

                # Loop over the jetties of the terminal
                for jetty in enumerate(terminal.terminal):
                    # If the length of the vessel is greater than the maximum allowable length of the jetty: vessel has to move to the anchorage area (for now), but continue loop
                    if vessel.L > terminal.jetty_lengths[jetty[0]]:
                        vessel.move_to_anchorage = True
                        continue
                    # Else if the length of the vessel is less or equal than the maximum allowable length of the jetty and there are no vessels at this jetty or waiting for this jetty: vessel does not have to move to the anchorage area, break loop
                    if jetty[1][edge[0]].users == [] and jetty[1][edge[0]].queue == []:
                        vessel.index_jetty_position = jetty[0]
                        vessel.move_to_anchorage = False
                        break
                    # Else if the length of the vessel is less or equal than the maximum allowable length of the jetty but there are vessel at this jetty already or waiting for this jetty:
                    else:
                        # If the queue is still empty: calculate the estimated time of departure of the currently (un)loading vessel and append to list
                        if jetty[1][edge[0]].queue == []:
                            minimum_waiting_time.append(jetty[1][edge[0]].users[-1].etd)
                        # Else append zero to the list
                        else:
                            minimum_waiting_time.append(0)
                        # Append the number of vessels waiting in the queue for the specific jetty to the list
                        vessels_waiting.append(len(jetty[1][edge[0]].queue))

                        # Continue loop if the jetty is not the last jetty in the list
                        if jetty[0] != len(terminal.terminal)-1:
                            continue

                    # If loop is not broken, then vessel has to move to the anchorage area
                    vessel.move_to_anchorage = True

            # Else if the terminal is of type 'quay:
            elif terminal.type == 'quay':
                # Import the locations of the current configuration of vessels located at the quay
                aql = terminal.available_quay_lengths
                # If the queue of vessels waiting for an available quay length is still empty: request quay position
                if terminal.length.get_queue == []:
                    vessel.index_quay_position,vessel.move_to_anchorage = PassTerminal.request_quay_position(vessel, terminal)
                # Else if this queue is not empty: vessel has to move to anchorage area (according to FCFS-policy)
                else:
                    vessel.move_to_anchorage = True

            # If the vessel has some waiting time due to the fact that it has arrived beyond a tidal window (move to anchorage still set to False)
            if vessel.waiting_time and not vessel.move_to_anchorage:
                yield from PassTerminal.move_to_anchorage(vessel, node)

            # Else if the vessel has to wait because there is no available spot at the terminal (move to anchorage was set to True)
            elif vessel.move_to_anchorage:
                # Set bool that says that vessel has to wait in the anchorage area because its waiting for an available spot at the terminal
                vessel.waiting_for_availability_terminal = True

                # If the terminal is of type 'quay:
                if terminal.type == 'quay':
                    # Make a request by getting the length of the terminal equal to the length of the vessel (which was not available), which functions as a yield timeout event equal to the waiting time for availability terminal
                    vessel.waiting_time_in_anchorage = terminal.length.get(vessel.L)

                # Else if the terminal is of type 'jetty:
                elif terminal.type == 'jetty':
                    # Set defual position index of the jetty
                    vessel.index_jetty_position = []

                    # Determine which indexes have an empty queue (jetties which have no vessels waiting)
                    indices_empty_queue_for_jetty = [waiting_vessels[0] for waiting_vessels in enumerate(vessels_waiting) if waiting_vessels[1] == 0]
                    # If this list of indexes is not empty:
                    if indices_empty_queue_for_jetty != []:
                        # Set default minimum waiting time (equal to the first jetty without queue)
                        min_minimum_waiting_time = minimum_waiting_time[indices_empty_queue_for_jetty[0]]
                        # Loop over the jetty indexes which do not have a queue:
                        for index in indices_empty_queue_for_jetty:
                            # If the minimum waiting time of the jetty is equal or less than the minimum waiting time for a jetty that was found up to now and the length of the vessel fits the jetty length: overwrite minimum waiting time with the minimum waiting time of that jetty and (temporarily) set the index of this jetty as the jetty index the vessel will be waiting for
                            if minimum_waiting_time[index] <= min_minimum_waiting_time and vessel.L <= terminal.jetty_lengths[index]:
                                min_minimum_waiting_time = minimum_waiting_time[index]
                                vessel.index_jetty_position = index
                    # Else if the list was empty and there is still no chosen jetty index: pick the jetty with the least number of vessels waiting in the queue
                    if vessel.index_jetty_position == []:
                        # Set defaul indexes list:
                        indexes = []
                        # Loop over the lenghts of the jetties:
                        for length in enumerate(terminal.jetty_lengths):
                            #If the length of the vessel fits in the jetty: append index to list
                            if vessel.L <= length[1]:
                                indexes.append(length[0])
                        # If the list of indexes are not empty: pick the jetty with the least number of vessels in the queue waiting for this particular jetty
                        if indexes != []:
                            vessel.index_jetty_position = np.min([y[0] for y in enumerate(vessels_waiting) if y[0] in indexes])

                    # If the jetty has been chosen: request that jetty which will function as the timeout event equalling the time that the jetty will be available
                    if vessel.index_jetty_position != []:
                        vessel.waiting_time_in_anchorage = terminal.terminal[vessel.index_jetty_position][edge[0]].request()
                    # Else if there is no suitable jetty: vessel will return to sea without waiting in the anchorage area
                    else:
                        vessel.return_to_sea = True
                        vessel.waiting_time = 0

                # Move vessel to the anchorage area
                yield from PassTerminal.move_to_anchorage(vessel, node)

            # Else if the vessel does not have to wait for either an available terminal or tidal window
            else:
                if terminal.type == 'quay':
                    PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)

            # If the vessel does not have to return to sea
            if vessel.return_to_sea == False:
                # Import information about the terminal, based on the terminal type (and if type 'jetty': based on the picked jetty position).
                if terminal.type == 'quay':
                    terminal = terminal.terminal[edge[0]]
                elif terminal.type == 'jetty':
                    terminal = terminal.terminal[vessel.index_jetty_position][edge[0]]

                # If the terminal type is 'jetty' and the vessel has to wait for an available jetty: pass; else: continue code
                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty' and vessel.waiting_for_availability_terminal == True:
                    pass
                else:
                    # Request access to the terminal
                    vessel.access_terminal = terminal.request()
                    yield vessel.access_terminal

                    # If terminal type is 'quay': revise the locations of the current configuration of vessels located at the quay with the new configuration
                    if vessel.env.FG.edges[edge]["Terminal"][0].type == 'quay':
                        vessel.env.FG.edges[edge]["Terminal"][0].available_quay_lengths = aql

                    # Set route after the terminal
                    if vessel.waiting_time:
                        vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route_after_anchorage[-1], vessel.true_origin)
                    else:
                        vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route[-1], vessel.route[0])

                    # If terminal is of type 'jetty': calculate the estimated time of departure consisting of the estimated time of arrival (sailing distance + berthing time + current time) + waiting time + (un)loading time + berthing time again
                    if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                        # Take the route when the vessel is leaving the terminal and reverse this route:
                        route = vessel.route_after_terminal
                        route.reverse()
                        sailing_distance = 0
                        # Loop over the route to calculate the total sailing distance and time
                        for nodes in enumerate(route[:-1]):
                            _, _, distance = vessel.wgs84.inv(vessel.env.FG.nodes[route[nodes[0]]]['geometry'].x,
                                                              vessel.env.FG.nodes[route[nodes[0]]]['geometry'].y,
                                                              vessel.env.FG.nodes[route[nodes[0] + 1]]['geometry'].x,
                                                              vessel.env.FG.nodes[route[nodes[0] + 1]]['geometry'].y)
                            sailing_distance += distance
                        terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
                        vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                        waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel,route = vessel.route_after_terminal, delay=vessel_etd-vessel.env.now,plot=False)
                        terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                        terminal.users[-1].type = vessel.type

        # Else if vessel has to return to sea: move vessel to anchorage first
        else:
            yield from PassTerminal.move_to_anchorage(vessel, node)

    def pass_terminal(vessel,edge):
        """ Function: function that handles the vessel at the terminal

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - edge: list of two strings that form the edge at which the terminal is located

        """

        # Import information about the terminal and the corresponding index of the start of the edge at which the terminal is located
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        index = vessel.route[vessel.route.index(edge[1]) - 1]

        # Calculate the location of the assigned position along the edge at which the terminal is located and finish the pass_edge move
        vessel.terminal_accessed = True
        vessel.wgs84 = pyproj.Geod(ellps="WGS84")
        [origin_lat,
         origin_lon,
         destination_lat,
         destination_lon] = [vessel.env.FG.nodes[edge[0]]['geometry'].x,
                             vessel.env.FG.nodes[edge[0]]['geometry'].y,
                             vessel.env.FG.nodes[edge[1]]['geometry'].x,
                             vessel.env.FG.nodes[edge[1]]['geometry'].y]
        fwd_azimuth, _, _ = vessel.wgs84.inv(origin_lat, origin_lon, destination_lat, destination_lon)

        if terminal.type == 'quay':
            position = vessel.quay_position

        elif terminal.type == 'jetty':
            position = terminal.jetty_locations[vessel.index_jetty_position]

        [vessel.terminal_pos_lat, vessel.terminal_pos_lon, _] = vessel.wgs84.fwd(vessel.env.FG.nodes[edge[0]]['geometry'].x,
                                                                                 vessel.env.FG.nodes[edge[0]]['geometry'].y,
                                                                                 fwd_azimuth, position)

        orig = nx.get_node_attributes(vessel.env.FG, "geometry")[edge[0]]
        dest = shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon)
        distance = vessel.wgs84.inv(shapely.geometry.asShape(orig).x,
                                    shapely.geometry.asShape(orig).y,
                                    shapely.geometry.asShape(dest).x,
                                    shapely.geometry.asShape(dest).y, )[2]

        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[0])
        vessel.log_entry("Sailing from node {} to node {} start".format(edge[0], edge[1]), vessel.env.now, ukc, orig, )
        yield vessel.env.timeout(distance / vessel.current_speed)
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Sailing from node {} to node {} stop".format(edge[0], edge[1]), vessel.env.now, ukc, dest, )

        # If the terminal is of type 'quay': log in logfile of terminal keeping track of the available length (by getting the so-called position length)
        if terminal.type == 'quay':
            terminal.pos_length.get(vessel.L)
            terminal.log_entry("Arrival of vessel", vessel.env.now, terminal.length.capacity-terminal.pos_length.level,nx.get_node_attributes(vessel.env.FG, "geometry")[index], )
        # Else if terminal is of type 'jetty': log in logfile of terminal calculating keeping track of number of jetties to be occupied
        elif terminal.type == 'jetty':
            terminal.jetties_occupied += 1
            terminal.log_entry("Arrival of vessel", vessel.env.now, terminal.jetties_occupied,nx.get_node_attributes(vessel.env.FG, "geometry")[index], )

        # Berthing: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to berth
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Berthing start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_b']*60)
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Berthing stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # If terminal is part of a junction: release request of this section (vessel is berthed and not in channel/basin)
        if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
            PassSection.release_access_previous_section(vessel, edge[1])

        # Unloading: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to unload
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Unloading start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_l']*60/2)
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Unloading stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Loading: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to load
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Loading start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_l']*60/2)
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Loading stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Determine the new route of the vessel (depending on whether the vessel came from the anchorage area or sailed to the terminal directly) and changing the direction of the vessel
        vessel.bound = 'outbound'  # to be removed later
        if 'true_origin' in dir(vessel):
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[vessel.route.index(edge[1])], vessel.true_origin)
        else:
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[vessel.route.index(edge[1])], vessel.route[0])
        if edge == [vessel.route[1],vessel.route[0]]: vessel.bound = 'outbound'

        # Calculate if the vessel has to wait due to be ready for departure beyond an available tidal window
        vessel.waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel,route=vessel.route,delay=0,plot=False)
        # If there is waiting time: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the waiting time
        if vessel.waiting_time:
            ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
            vessel.log_entry("Waiting for tidal window start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
            yield vessel.env.timeout(np.max([0,vessel.waiting_time-vessel.metadata['t_b']*60]))
            ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
            vessel.log_entry("Waiting for tidal window stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Deberthing: if the terminal is part of an section, request access to this section first
        if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
            yield from PassSection.request_access_next_section(vessel, edge[1], edge[0])

        # Deberthing: if the terminal is attached to a turning basin: check whether the vessel can turn here (if the length of the vessel allows to turn the vessel in this basin)
        if 'Turning Basin' in vessel.env.FG.nodes[edge[0]].keys():
            turning_basin = vessel.env.FG.nodes[edge[0]]['Turning Basin'][0]
            if turning_basin.length >= vessel.L:
                vessel.request_access_turning_basin = turning_basin.turning_basin[edge[0]].request()
                yield vessel.request_access_turning_basin

        # Deberthing: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to deberth
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Deberthing start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_b']*60)
        ukc = VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Deberthing stop", vessel.env.now, ukc, shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Move vessel to start node of the terminal, release request of this section, and change vessel route by removing the first node of the route (as vessel will already be located in the second node of the route after the move event)
        PassSection.release_access_previous_section(vessel, edge[0])

        # If the terminal is of type 'quay'
        if terminal.type == 'quay':
            # Determine the old maximum available length of the quay
            old_level = PassTerminal.calculate_quay_length_level(terminal)

            # Function that is used to readjust the available quay lengths:
            def readjust_available_quay_lengths(terminal,position):
                """ Function that readjusts the available quay lenghts and positions given a release of a request of a vessel at a given position

                    Input:
                        - terminal: the terminal of call of the vessel, created with the IsTerminal-class
                        - position: quay position index at which the vessel is located at the quay with respect to the other vessels

                """

                # Import the locations of the current configuration of vessels located at the quay
                aql = terminal.available_quay_lengths
                # Loop over the position indexes
                for index in range(len(aql)):
                    # Skip the first position index
                    if index == 0:
                        continue
                    # If the position of the vessel falls within the position bounds in the current configuration: break loop (save index)
                    if aql[index - 1][1] < position and aql[index][1] > position:
                        break

                # Set both values of these position bounds to zero (available again)
                aql[index - 1][0] = 0
                aql[index][0] = 0

                # Set a default list of redundant indexes to be removed
                to_remove = []
                # Nested loop over the position indexes
                for index in enumerate(aql):
                    for jndex in enumerate(aql):
                        # If the two indexes are not equal and the value at position index 1 and index 2 are both zero (available) and the locations of the two indexes are equal: remove the first positional index
                        if index[0] != jndex[0] and index[1][0] == 0 and jndex[1][0] == 0 and index[1][1] == jndex[1][1]:
                            to_remove.append(index[0])

                # If there are indexes to be removed, loop over these indexes and remove them
                for index in list(reversed(to_remove)):
                    aql.pop(index)

                # Return the locations of the new configuration of vessels located at the quay
                return aql

            # Readjust the available quay lengths as the vessel is leaving the terminal
            terminal.available_quay_lengths = readjust_available_quay_lengths(terminal,vessel.quay_position)
            # Calculate the new maximum available quay length
            new_level = PassTerminal.calculate_quay_length_level(terminal)
            # If this length does not equal the current maximum available quay length (is smaller), then put this length back to the quay
            if old_level != new_level:
                terminal.length.put(new_level - old_level)
            # Give vessel length back to keep track of the total claimed vessel length and log this value and the departure event in the logfile of the terminal, and release the request of the vessel to access the terminal
            terminal.pos_length.put(vessel.L)
            terminal.log_entry("Departure of vessel", vessel.env.now, terminal.length.capacity-terminal.pos_length.level,nx.get_node_attributes(vessel.env.FG, "geometry")[index], )
            terminal.terminal[edge[0]].release(vessel.access_terminal)

        # Else if the terminal is of type 'jetty': adjust number of vessels occupying a jetty, log the departure of this vessel and this number, and release the request of the vessel for the specific jetty
        elif terminal.type == 'jetty':
            terminal.jetties_occupied -= 1
            terminal.log_entry("Departure of vessel", vessel.env.now, terminal.jetties_occupied,nx.get_node_attributes(vessel.env.FG, "geometry")[index],)
            terminal.terminal[vessel.index_jetty_position][edge[0]].release(vessel.access_terminal)

        # Initiate move of vessel back to sea, setting a bool of leaving port to true
        vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
        vessel.env.process(vessel.move())
        vessel.leaving_port = True

class Output():
    """ mixing class: collection of unfinished functions that should store the output to the vessels and the network in the future development of OpenTNSim"""

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

    def __init__(self, v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.v = v
        self.on_pass_node = []
        self.on_look_ahead_to_node = []
        self.on_pass_edge = []
        self.wgs84 = pyproj.Geod(ellps="WGS84")
        self.metadata['v0'] = self.v

    def initial_timeout(self):
        yield self.env.timeout(self.metadata['start_time'])
        self.metadata['start_time'] = 0
        self.metadata['v0'] = self.v

    def move(self):
        """determine distance between origin and destination, and
        yield the time it takes to travel it
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].
        """
        self.distance = 0
        if self.metadata['start_time'] != 0: yield from Movable.initial_timeout(self)

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

            try:
                yield from self.look_ahead_to_node(destination)
            except simpy.exceptions.Interrupt as e:
                break

            try:
                yield from self.pass_node(origin)
            except simpy.exceptions.Interrupt as e:
                break

            try:
                yield from self.pass_edge(origin, destination)
            except simpy.exceptions.Interrupt as e:
                break

            if node[0] + 2 == len(self.route):
                break

        logger.debug("  distance: " + "%4.2f" % self.distance + " m")
        logger.debug("  sailing:  " + "%4.2f" % self.current_speed + " m/s")
        logger.debug("  duration: " + "%4.2f" % ((self.distance / self.current_speed) / 3600) + " hrs")

    def look_ahead_to_node(self,destination):
        for gen in self.on_look_ahead_to_node:
            try:
                yield from gen(destination)
            except simpy.exceptions.Interrupt as e:
                logger.debug("Re-routing", exc_info=True)
                raise simpy.exceptions.Interrupt('Re-routing')
                break

    def pass_node(self,origin):
        for gen in self.on_pass_node:
            try:
                yield from gen(origin)
            except simpy.exceptions.Interrupt as e:
                logger.debug("Re-routing", exc_info=True)
                raise simpy.exceptions.Interrupt('Re-routing')
                break

    def pass_edge(self, origin, destination):
        edge = self.env.FG.edges[origin, destination]
        orig = nx.get_node_attributes(self.env.FG, "geometry")[origin]
        dest = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        for gen in self.on_pass_edge:
            try:
                yield from gen(origin,destination)
            except simpy.exceptions.Interrupt as e:
                logger.debug("Re-routing", exc_info=True)
                raise simpy.exceptions.Interrupt('Re-routing')
                break

        if "Lock" in self.env.FG.nodes[origin].keys():
            orig = shapely.geometry.Point(self.lock_pos_lat,self.lock_pos_lon)

        if "Lock" in self.env.FG.nodes[destination].keys():
            dest = shapely.geometry.Point(self.lock_pos_lat,self.lock_pos_lon)

        if "Line-up area" in self.env.FG.nodes[origin].keys():
            orig = shapely.geometry.Point(self.lineup_pos_lat,self.lineup_pos_lon)

        if "Line-up area" in self.env.FG.nodes[destination].keys():
            dest = shapely.geometry.Point(self.lineup_pos_lat,self.lineup_pos_lon)

        if "Terminal" in self.env.FG.edges[origin,destination].keys():
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
                ukc = VesselTrafficService.provide_ukc_clearance(self, origin)
                self.log_entry("Sailing from node {} to node {} sub edge {} start".format(origin, destination, index), self.env.now, ukc, sub_orig,)
                yield self.env.timeout(distance / self.current_speed)
                ukc = VesselTrafficService.provide_ukc_clearance(self, destination)
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
                ukc = VesselTrafficService.provide_ukc_clearance(self, origin)
                with self.env.FG.edges[origin, destination]["Resources"].request() as request:
                    yield request

                    if arrival != self.env.now:
                        self.log_entry("Waiting to pass edge {} - {} start".format(origin, destination), arrival, ukc, orig,)
                        ukc = VesselTrafficService.provide_ukc_clearance(self, origin)
                        self.log_entry("Waiting to pass edge {} - {} stop".format(origin, destination), self.env.now, ukc, orig,)

                    self.log_entry("Sailing from node {} to node {} start".format(origin, destination), self.env.now, 0, orig,)
                    yield self.env.timeout(distance / self.current_speed)
                    ukc = VesselTrafficService.provide_ukc_clearance(self, destination)
                    self.log_entry("Sailing from node {} to node {} stop".format(origin, destination), self.env.now, ukc, dest,)

            else:
                if 'Info' in self.env.FG.nodes[origin]:
                    ukc = VesselTrafficService.provide_ukc_clearance(self, origin)
                else:
                    ukc = []
                self.log_entry("Sailing from node {} to node {} start".format(origin, destination), self.env.now, self.current_speed, orig,)
                yield self.env.timeout(distance / self.current_speed)
                if 'Info' in self.env.FG.nodes[destination]:
                    ukc = VesselTrafficService.provide_ukc_clearance(self, destination)
                else:
                    ukc = []
                self.log_entry("Sailing from node {} to node {} stop".format(origin, destination), self.env.now, self.current_speed, dest,)
            self.geometry = dest

    @property
    def current_speed(self):
        return self.v

class HasTerminal(Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_edge.append(self.pass_terminal)

    def pass_terminal(self,origin,destination):
        # Terminal
        if 'Terminal' in self.env.FG.edges[origin, destination].keys() and self.route[-1] == destination:
            yield from PassTerminal.pass_terminal(self, [origin, destination])
            raise simpy.exceptions.Interrupt('New route determined')

class HasOrigin(Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.request_terminal_access)

    def request_terminal_access(self, origin):
        # Request for a terminal
        if "Origin" in self.env.FG.nodes[origin] and 'leaving_port' not in dir(self):
            self.bound = 'inbound'  ##to be removed later
            self.terminal_accessed = False
            yield from PassTerminal.request_terminal_access(self, [self.route[-2], self.route[-1]], origin)
            if self.waiting_in_anchorage:
                raise simpy.exceptions.Interrupt('New route determined')

class HasAnchorage(Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_look_ahead_to_node.append(self.pass_anchorage_area)

    def pass_anchorage_area(self,destination):
        # Anchorage
        if 'Anchorage' in self.env.FG.nodes[destination].keys() and self.route[-1] == destination:
            yield from PassTerminal.pass_anchorage(self, destination)
            raise simpy.exceptions.Interrupt('New route determined')

class HasLock(Movable):
    def __init__(self, lock_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock_name = lock_name
        self.on_pass_node.append(self.leave_lock_chamber)

    def leave_lock_chamber(self,origin):
        if "Lock" in self.env.FG.nodes[origin].keys():  # if vessel in lock
            yield from lock_module.PassLock.leave_lock(self, origin)
            #self.v = 4 * self.v
            #self.v = self.metadata['v0']
            self.v = self.metadata['speed_reduction'][1] * self.metadata['v0']

class HasWaitingArea(Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.leave_waiting_area)
        #self.on_pass_node.append(self.speed_reduction)
        self.on_look_ahead_to_node.append(self.approach_waiting_area)

    def approach_waiting_area(self,destination):
        if "Waiting area" in self.env.FG.nodes[destination].keys():  # if waiting area is located at next node
            yield from lock_module.PassLock.approach_waiting_area(self, destination)

    def leave_waiting_area(self, origin):
        if "Waiting area" in self.env.FG.nodes[origin].keys():  # if vessel is in waiting area
            yield from lock_module.PassLock.leave_waiting_area(self, origin)

'''    def speed_reduction(self, origin):
        if "Speed reduction" in self.env.FG.nodes[origin].keys():  # if vessel is in waiting area
            yield from lock_module.PassLock.test(self, origin)
            #self.v = 100
            #yield from lock_module.PassLock.leave_waiting_area(self, origin)'''

class HasSpeedReduction(Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.speed_reduction)
    
    def speed_reduction(self, origin):
        if "Speed reduction" in self.env.FG.nodes[origin].keys():
            yield from lock_module.PassLock.speed_reduction(self, origin)

class HasLineUpArea(Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.leave_lineup_area)
        self.on_look_ahead_to_node.append(self.approach_lineup_area)

    def approach_lineup_area(self,destination):
        if "Line-up area" in self.env.FG.nodes[destination].keys():  # if vessel is approaching the line-up area
            yield from lock_module.PassLock.approach_lineup_area(self, destination)

    def leave_lineup_area(self,origin):
        if "Line-up area" in self.env.FG.nodes[origin].keys():  # if vessel is located in the line-up
            lineup_areas = self.env.FG.nodes[origin]["Line-up area"]
            for lineup_area in lineup_areas:
                if lineup_area.name != self.lock_name:  # picks the assigned parallel lock chain
                    continue

                index_node_lineup_area = self.route.index(origin)
                for node_lock in self.route[index_node_lineup_area:]:
                    if 'Lock' in self.env.FG.nodes[node_lock].keys():
                        yield from lock_module.PassLock.leave_lineup_area(self, origin)
                        break

                    elif 'Waiting area' in self.env.FG.nodes[node_lock].keys():  # if vessel is leaving the lock complex
                        yield from lock_module.PassLock.leave_opposite_lineup_area(self, origin)
                        break

class HasSection(Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_edge.append(self.request_access_new_section)

    def request_access_new_section(self,origin,destination):
        # Leave and access waterway section
        if 'Junction' in self.env.FG.nodes[origin].keys():
            if 'Anchorage' not in self.env.FG.nodes[origin].keys():
                PassSection.release_access_previous_section(self, origin)
                yield from PassSection.request_access_next_section(self, origin, destination)

class HasTurningBasin(Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.enter_turning_basin)
        self.on_look_ahead_to_node.append(self.request_turning_basin)

    def enter_turning_basin(self, origin):
        if 'Turning Basin' in self.env.FG.nodes[origin].keys():
            turning_basin = self.env.FG.nodes[origin]['Turning Basin'][0]
            ukc = VesselTrafficService.provide_ukc_clearance(self, origin)
            if self.bound == 'outbound' and turning_basin.length >= self.L:
                self.log_entry("Vessel Turning Start", self.env.now, ukc, self.env.FG.nodes[origin]['geometry'])
                turning_basin.log_entry("Vessel Turning Start", self.env.now, 0, self.env.FG.nodes[origin]['geometry'])
                yield self.env.timeout(10 * 60)
                ukc = VesselTrafficService.provide_ukc_clearance(self, origin)
                turning_basin.log_entry("Vessel Turning Stop", self.env.now, 10 * 60,
                                        self.env.FG.nodes[origin]['geometry'])
                self.log_entry("Vessel Turning Stop", self.env.now, ukc, self.env.FG.nodes[origin]['geometry'])
            else:
                self.log_entry("Passing Turning Basin", self.env.now, ukc, self.env.FG.nodes[origin]['geometry'])
                turning_basin.log_entry("Vessel Passing", self.env.now, 0, self.env.FG.nodes[origin]['geometry'])
            turning_basin.turning_basin[origin].release(self.request_access_turning_basin)

    def request_turning_basin(self, destination):
        if 'Turning Basin' in self.env.FG.nodes[destination].keys():
            turning_basin = self.env.FG.nodes[destination]['Turning Basin'][0]
            if turning_basin.length >= self.L:
                self.request_access_turning_basin = turning_basin.turning_basin[destination].request()
                yield self.request_access_turning_basin

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

