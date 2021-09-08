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
import scipy as sc

# OpenTNSim
from opentnsim import energy_consumption_module
from opentnsim import lock_module

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

    def __init__(self, typ, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.type = typ

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
        self.ukc = 0.5 #0.1*self.T_f

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
        jetty_locations = [],
        *args,
        **kwargs
    ):

        "Initialization"
        super().__init__(length=length, remaining_length=length, *args, **kwargs)

        if self.type == 'quay':
            self.terminal = {
                node_start: simpy.PriorityResource(self.env, capacity=100),
            }

            self.available_quay_lengths = [[0,0],[0,length]]

        elif self.type == 'jetty':
            self.terminal = []
            self.jetties_occupied = 0
            self.jetty_locations = jetty_locations
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
        self.anchorage_area = {
            node: simpy.PriorityResource(self.env, capacity=max_capacity),
        }
        
class NetworkProperties:
    def __init__(self,
                 network,
                 W,
                 D,
                 eta,
                 *args,
                 **kwargs
    ):
        super().__init__(*args, **kwargs)
        
    def append_data_to_nodes(network,W,D,eta,vmag,vdir):
        for node in enumerate(network.nodes):
            network.nodes[node[1]]['Info'] = {'Width': [], 'Depth': [], 'Water level': [[],[]], 'Current velocity': [[],[]], 'Current direction': [[],[]]}
            if (W[0][node[0]].x,W[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                network.nodes[node[1]]['Info']['Width'].append(W[1][node[0]])
            if (D[0][node[0]].x,D[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                network.nodes[node[1]]['Info']['Depth'].append(D[1][node[0]])
            if (eta[0][node[0]].x,eta[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                for t in range(len(eta[1][node[0]][0])):
                    network.nodes[node[1]]['Info']['Water level'][0].append(eta[1][node[0]][0][t])
                    network.nodes[node[1]]['Info']['Water level'][1].append(eta[1][node[0]][1][t])
            if (vmag[0][node[0]].x,vmag[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                for t in range(len(eta[1][node[0]][0])):
                    network.nodes[node[1]]['Info']['Current velocity'][0].append(vmag[1][node[0]][0][t])
                    network.nodes[node[1]]['Info']['Current velocity'][1].append(vmag[1][node[0]][1][t])
            if (vdir[0][node[0]].x,vdir[0][node[0]].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                for t in range(len(vdir[1][node[0]][0])):
                    network.nodes[node[1]]['Info']['Current direction'][0].append(vdir[1][node[0]][0][t])
                    network.nodes[node[1]]['Info']['Current direction'][1].append(vdir[1][node[0]][1][t])

    def append_info_to_edges(network):        
        for edge in enumerate(network.edges):
            network.edges[edge[1]]['Info'] = {'Width': [], 'Depth': [], 'Water level': [[],[[],[]]], 'Current velocity': [[],[[],[]]], 'Current direction': [[],[[],[]]]}
            network.edges[edge[1]]['Info']['Width'].append(np.min([network.nodes[edge[1][0]]['Info']['Width'][0],network.nodes[edge[1][1]]['Info']['Width'][0]]))
            if 'Terminal' in network.edges[edge[1]]:
                network.edges[edge[1]]['Info']['Depth'].append(np.max([network.nodes[edge[1][0]]['Info']['Depth'][0],network.nodes[edge[1][1]]['Info']['Depth'][0]]))
            else:
                network.edges[edge[1]]['Info']['Depth'].append(np.min([network.nodes[edge[1][0]]['Info']['Depth'][0],network.nodes[edge[1][1]]['Info']['Depth'][0]]))
            for t in range(len(network.nodes[edge[1][0]]['Info']['Water level'][0])):
                network.edges[edge[1]]['Info']['Water level'][0].append(network.nodes[edge[1][0]]['Info']['Water level'][0][t])
                network.edges[edge[1]]['Info']['Water level'][1][0].append(network.nodes[edge[1][0]]['Info']['Water level'][1][t])
                network.edges[edge[1]]['Info']['Water level'][1][1].append(network.nodes[edge[1][1]]['Info']['Water level'][1][t])
            for t in range(len(network.nodes[edge[1][0]]['Info']['Current velocity'][0])):
                network.edges[edge[1]]['Info']['Current velocity'][0].append(network.nodes[edge[1][0]]['Info']['Current velocity'][0][t])
                network.edges[edge[1]]['Info']['Current velocity'][1][0].append(network.nodes[edge[1][0]]['Info']['Current velocity'][1][t])
                network.edges[edge[1]]['Info']['Current velocity'][1][1].append(network.nodes[edge[1][1]]['Info']['Current velocity'][1][t])

    def calculate_available_sail_in_times(vessel,vertical_tidal_window,horizontal_tidal_window):
        available_sail_in_times = []
        vessel.waiting_time_start = vessel.env.now
        vessel.max_waiting_time = vessel.waiting_time_start + 24 * 60 * 60

        def times_vertical_tidal_window(vessel):
            times_vertical_tidal_window = []
            water_depth_required = vessel.T_f + vessel.ukc
            [new_t, min_wdep] = NetworkProperties.minimum_water_per_edge_as_experienced_by_vessel(vessel)
            root_interp_water_level_at_edge = sc.interpolate.CubicSpline(new_t,min_wdep)
            for root in root_interp_water_level_at_edge.roots():
                if root > np.min(new_t) and root < np.max(new_t) and min_wdep[[i for i, x in enumerate(list(new_t > root)) if x][0]] > water_depth_required:
                    times_vertical_tidal_window.append([root, 'Stop'])
                elif root > np.min(new_t) and root < np.max(new_t) and min_wdep[[i for i, x in enumerate(list(new_t > root)) if x][0]] < water_depth_required:
                    times_vertical_tidal_window.append([root, 'Start'])
            return times_vertical_tidal_window

        def times_horizontal_tidal_window(vessel):
            network = vessel.env.FG
            current_time = vessel.env.now
            route = vessel.route
            waiting_time = 0
            time_edge_is_navigable = 0
            max_cross_current_velocity = 2
            times_horizontal_tidal_windows = []

            for nodes in enumerate(route):
                if nodes[0] == 0:
                    continue
                elif nodes[1] == route[-1]:
                    continue
                [cross_current,root_interp_cross_current_orig,new_t, cor_cross_current] = NetworkProperties.cross_current_calculator(vessel,max_cross_current_velocity,nodes[0])

                if max_cross_current_velocity < cross_current:
                    times_edge_is_navigable = root_interp_cross_current_orig.roots()
                    times_horizontal_tidal_window = []
                    for root in root_interp_cross_current_orig.roots():
                        if root > np.min(new_t) and root < np.max(new_t) and cor_cross_current[[i for i, x in enumerate(list(new_t > root)) if x][0]] < 0:
                            times_horizontal_tidal_window.append([root, 'Stop'])
                        elif root > np.min(new_t) and root < np.max(new_t) and cor_cross_current[[i for i, x in enumerate(list(new_t > root)) if x][0]] > 0:
                            times_horizontal_tidal_window.append([root, 'Start'])

                    times_horizontal_tidal_windows.append(times_horizontal_tidal_window)

            return times_horizontal_tidal_windows

        def times_tidal_window(vessel,vertical_tidal_window,horizontal_tidal_window):
            list_of_times_vertical_tidal_window = []
            list_of_times_horizontal_tidal_windows = []

            if vertical_tidal_window:
                list_of_times_vertical_tidal_window = times_vertical_tidal_window(vessel)
            if horizontal_tidal_window:
                list_of_times_horizontal_tidal_windows = times_horizontal_tidal_window(vessel)
            list_indexes = list(np.arange(0, len(list_of_times_horizontal_tidal_windows) + 1))
            times_tidal_window = []
            list_of_list_indexes = []

            for time in list_of_times_vertical_tidal_window:
                times_tidal_window.append(time)
                list_of_list_indexes.append(0)
            for index in range(len(list_indexes)-1):
                for time in list_of_times_horizontal_tidal_windows[index]:
                    times_tidal_window.append(time)
                    list_of_list_indexes.append(index+1)
                for time in list_of_times_horizontal_tidal_windows[index]:
                    times_tidal_window.append(time)
                    list_of_list_indexes.append(index+1)

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

            for list_index in list_indexes:
                for time1 in range(len(times_tidal_window)):
                    if times_tidal_window[time1][1] == 'Stop' and list_of_list_indexes[time1] == list_index:
                        for time2 in range(len(times_tidal_window)):
                            if time2 < time1 and times_tidal_window[time2][1] == 'Start' and list_of_list_indexes[time2] == list_index:
                                break
                        if time2 > time1:
                            indexes = np.arange(0, time1, 1)
                            for index in indexes:
                                indexes_to_be_removed.append(index)

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

        def max_waiting_time_in_anchorage(vessel, times_tidal_window):
            for time in range(len(times_tidal_window)):
                if times_tidal_window[time][0] > vessel.max_waiting_time and times_tidal_window[time][1] == 'Stop':
                    break
            if not times_tidal_window or times_tidal_window[time - 1][0] > vessel.max_waiting_time:
                max_waiting_time_in_anchorage = vessel.max_waiting_time
            else:
                max_waiting_time_in_anchorage = times_tidal_window[time - 1][0]
            return max_waiting_time_in_anchorage

        times_tidal_window = times_tidal_window(vessel,vertical_tidal_window,horizontal_tidal_window)
        for time in range(len(times_tidal_window)):
            if times_tidal_window[time][0] < max_waiting_time_in_anchorage(vessel, times_tidal_window):
                available_sail_in_times.append(times_tidal_window[time])
            else:
                vessel.return_to_sea = True

        available_sail_in_times.insert(0, [vessel.waiting_time_start, 'Start'])
        available_sail_in_times.append([max_waiting_time_in_anchorage(vessel, times_tidal_window), 'Stop'])
        return available_sail_in_times

    def minimum_water_per_edge_as_experienced_by_vessel(vessel):
        network = vessel.env.FG
        route = vessel.route
        min_wdep = [[] for _ in range(len(route) - 1)]
        new_t = [[] for _ in range(len(route) - 1)]
        distance_to_next_node = 0
        for nodes in enumerate(route):
            if nodes[1] == route[0]:
                wlev_node = network.edges[route[0], route[1]]['Info']['Water level'][1][0]
                depth = network.edges[route[0], route[1]]['Info']['Depth'][0]
                max_wdep = np.max(wlev_node) + depth
                continue

            distance_to_next_node += pyproj.Geod(ellps='WGS84').inv(network.nodes[route[nodes[0] - 1]]['geometry'].x,
                                                                    network.nodes[route[nodes[0] - 1]]['geometry'].y,
                                                                    network.nodes[route[nodes[0]]]['geometry'].x,
                                                                    network.nodes[route[nodes[0]]]['geometry'].y)[2]

            t_wlev = network.edges[route[nodes[0] - 1], route[nodes[0]]]['Info']['Water level'][0]

            sailing_time_to_next_node = distance_to_next_node / vessel.v
            wlev_node1 = network.edges[route[0], route[1]]['Info']['Water level'][1][0]
            depth1 = network.edges[route[0], route[1]]['Info']['Depth'][0]
            wlev_node2 = network.edges[route[nodes[0] - 1], route[nodes[0]]]['Info']['Water level'][1][1]
            depth2 = network.edges[route[nodes[0] - 1], route[nodes[0]]]['Info']['Depth'][0]

            if np.max([np.max(wlev_node1) + depth1, np.max(wlev_node2) + depth2]) > max_wdep:
                max_wdep = np.max([np.max(wlev_node1) + depth1, np.max(wlev_node2) + depth2])

            interp_wdep_node1 = sc.interpolate.CubicSpline(t_wlev, [y + depth1 for y in wlev_node1])
            eta_next_node = [t - sailing_time_to_next_node for t in t_wlev]
            interp_wdep_node2 = sc.interpolate.CubicSpline(eta_next_node, [y + depth2 for y in wlev_node2])

            if eta_next_node[-1] - vessel.env.now >= 3000:
                new_t[nodes[0] - 1] = np.arange(vessel.env.now, eta_next_node[-1], 300)
            elif eta_next_node[-1] > vessel.env.now:
                new_t[nodes[0] - 1] = np.linspace(vessel.env.now, eta_next_node[-1], 10)
            elif eta_next_node[-1] < vessel.env.now:
                new_t[nodes[0] - 1] = np.linspace(eta_next_node[-1], vessel.env.now, 10)
            else:
                new_t[nodes[0] - 1] = np.arange(eta_next_node[-1], vessel.env.now, 300)

            for t in new_t[nodes[0] - 1]:
                min_wdep[nodes[0] - 1].append(np.min([interp_wdep_node1(t), interp_wdep_node2(t)]))

        minimum_water_depth = []
        time_minimum_water_depth = new_t[nodes[0] - 1]

        for t in range(len(time_minimum_water_depth)):
            min_water_depth = max_wdep
            for edge in range(len(new_t)):
                if min_wdep[edge][t] < min_water_depth:
                    min_water_depth = min_wdep[edge][t]
            minimum_water_depth.append(min_water_depth)

        return time_minimum_water_depth, minimum_water_depth

    def cross_current_calculator(vessel,max_cross_current_velocity,node_index):
        network = vessel.env.FG
        node = vessel.route[node_index]
        time_current_velocity_node = network.nodes[vessel.route[node_index]]['Info']['Current velocity'][0]
        current_velocity_node = network.nodes[vessel.route[node_index]]['Info']['Current velocity'][1]
        time_current_direction_node = network.nodes[vessel.route[node_index]]['Info']['Current direction'][0]
        current_direction_node = network.nodes[vessel.route[node_index]]['Info']['Current direction'][1]

        origin_lat = vessel.env.FG.nodes[vessel.route[node_index - 1]]['geometry'].x
        origin_lon = vessel.env.FG.nodes[vessel.route[node_index - 1]]['geometry'].y
        node_lat = vessel.env.FG.nodes[vessel.route[node_index]]['geometry'].x
        node_lon = vessel.env.FG.nodes[vessel.route[node_index]]['geometry'].y
        destination_lat = vessel.env.FG.nodes[vessel.route[node_index + 1]]['geometry'].x
        destination_lon = vessel.env.FG.nodes[vessel.route[node_index + 1]]['geometry'].y

        course, _, _ = pyproj.Geod(ellps="WGS84").inv(origin_lat, origin_lon, node_lat, node_lon)
        if course < 0:
            course = 360 + course
        heading, _, _ = pyproj.Geod(ellps="WGS84").inv(node_lat, node_lon, destination_lat, destination_lon)
        if heading < 0:
            heading = 360 + heading

        distance_to_node = 0
        for n in range(len(vessel.route)):
            if n == 0:
                continue
            elif vessel.route[n] == node:
                break
            distance_to_node += pyproj.Geod(ellps='WGS84').inv(network.nodes[vessel.route[n - 1]]['geometry'].x,
                                                               network.nodes[vessel.route[n - 1]]['geometry'].y,
                                                               network.nodes[vessel.route[n]]['geometry'].x,
                                                               network.nodes[vessel.route[n]]['geometry'].y)[2]

        sailing_time = distance_to_node / vessel.v
        interp_current_velocity_orig = sc.interpolate.CubicSpline(time_current_velocity_node, current_velocity_node)
        interp_current_direction_orig = sc.interpolate.interp1d(time_current_direction_node, current_direction_node)
        corrected_cross_current = [y - max_cross_current_velocity for y in abs(interp_current_velocity_orig(time_current_velocity_node) * np.sin((interp_current_direction_orig(time_current_velocity_node) - course) / 180 * math.pi) -
                                                                               interp_current_velocity_orig(time_current_velocity_node) * np.sin((interp_current_direction_orig(time_current_velocity_node) - heading) / 180 * math.pi))]

        time_corrected_cross_current = [t - sailing_time for t in time_current_velocity_node]
        interpolated_corrected_cross_current_signal = sc.interpolate.CubicSpline(time_corrected_cross_current,corrected_cross_current)

        new_t = [t - sailing_time for t in time_current_velocity_node]

        if vessel.env.now+sailing_time < time_current_velocity_node[-1]:
            actual_cross_current = abs(interp_current_velocity_orig(vessel.env.now+sailing_time) * np.sin((interp_current_direction_orig(vessel.env.now+sailing_time) - course) / 180 * math.pi) -
                                       interp_current_velocity_orig(vessel.env.now+sailing_time) * np.sin((interp_current_direction_orig(vessel.env.now+sailing_time) - heading) / 180 * math.pi))
        else:
            actual_cross_current = 0

        return [actual_cross_current, interpolated_corrected_cross_current_signal, new_t, corrected_cross_current]

    def waiting_time_for_tidal_window(vessel,vertical_tidal_window,horizontal_tidal_window = True):
        waiting_time_vertical_tidal_window = 0
        waiting_time_horizontal_tidal_window = 0

        def waiting_time_for_vertical_tidal_window(vessel):
            network = vessel.env.FG
            current_time = vessel.env.now
            route = vessel.route
            waiting_time = 0
            time_edge_is_navigable = 0
            time_minimum_water_depth, minimum_water_depth = NetworkProperties.minimum_water_per_edge_as_experienced_by_vessel(vessel)
            water_depth_required = vessel.T_f + vessel.ukc
            interp_water_level_at_edge = sc.interpolate.CubicSpline(time_minimum_water_depth, minimum_water_depth)
            corrected_water_level = [y - water_depth_required for y in minimum_water_depth]
            root_interp_water_level_at_edge = sc.interpolate.CubicSpline(time_minimum_water_depth,corrected_water_level)
            water_depth_at_edge = interp_water_level_at_edge(current_time)
            if water_depth_required > water_depth_at_edge:
                times_edge_is_navigable = root_interp_water_level_at_edge.roots()
                for t2 in times_edge_is_navigable:
                    if t2 >= current_time:
                        time_edge_is_navigable = t2 - current_time
                        break

            if time_edge_is_navigable > waiting_time:
                waiting_time = time_edge_is_navigable

            return waiting_time

        def waiting_time_for_horizontal_tidal_window(vessel):
            network = vessel.env.FG
            current_time = vessel.env.now
            route = vessel.route
            waiting_time = 0
            time_edge_is_navigable = 0
            max_cross_current_velocity = 2

            for nodes in enumerate(route):
                if nodes[0] == 0:
                    continue
                elif nodes[1] == route[-1]:
                    continue

                [cross_current,root_interp_cross_current_orig,_,_] = NetworkProperties.cross_current_calculator(vessel,max_cross_current_velocity,nodes[0])

                if max_cross_current_velocity < cross_current:
                    times_edge_is_navigable = root_interp_cross_current_orig.roots()
                    for t2 in times_edge_is_navigable:
                        if t2 >= current_time:
                            time_edge_is_navigable = t2 - current_time
                            break

                if time_edge_is_navigable > waiting_time:
                    waiting_time = time_edge_is_navigable

            return waiting_time

        if vertical_tidal_window:
            waiting_time_vertical_tidal_window = waiting_time_for_vertical_tidal_window(vessel)
        #if horizontal_tidal_window:
        #    waiting_time_horizontal_tidal_window = waiting_time_for_horizontal_tidal_window(vessel)

        waiting_time = np.max([waiting_time_vertical_tidal_window,waiting_time_horizontal_tidal_window])

        return waiting_time

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
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        """Initialization"""

        waiting_area_resources = 100
        self.waiting_area = {
            node: simpy.PriorityResource(self.env, capacity=waiting_area_resources),
        }

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
        **kwargs
    ):
        super().__init__(length = lineup_length, remaining_length = lineup_length, *args, **kwargs)
        
        """Initialization"""
        # Lay-Out
        self.enter_line_up_area = { #used to regulate one by one entering of line-up area, so capacity must be 1
            node: simpy.PriorityResource(self.env, capacity=1),
        }
        
        self.line_up_area = { #line-up area itself, infinite capacity, as this is regulated by the HasLength, so capacity = inf
            node: simpy.PriorityResource(self.env, capacity=100),
        }

        self.converting_while_in_line_up_area = { #used to minimize the number of empty convertion requests by one by multiple waiting vessels, so capacity must be 1
            node: simpy.PriorityResource(self.env, capacity=1),
        }

        self.pass_line_up_area = { #used to prevent vessel from entering the lock before all previously locked vessels have passed the line-up area one by one, so capacity must be 1
            node: simpy.PriorityResource(self.env, capacity=1),
        }

class IsTurningBasin(HasResource, Identifiable, Log):
    def __init__(
        self,
        node, #a string which indicates the location of the start of the waiting area
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        """Initialization"""

        self.turning_basin = {
            node: simpy.PriorityResource(self.env, capacity=1),
        }

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
        **kwargs
    ):

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
        
        self.doors_1 = {
            node_1: simpy.PriorityResource(self.env, capacity = 1), #Only one ship can pass at a time: capacity = 1, request can have priority
        }
        self.doors_2 = {
            node_3: simpy.PriorityResource(self.env, capacity = 1), #Only one ship can pass at a time: capacity = 1, request can have priority
        }

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
        self.log_entry(
            "Lock chamber converting start", environment.now, number_of_vessels, self.water_level
        )

        # Water level will shift
        yield environment.timeout(self.operation_time(environment))
        self.change_water_level(new_level)
        self.log_entry(
            "Lock chamber converting stop", environment.now, number_of_vessels, self.water_level
        )
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
        for node in reversed(vessel.route[:vessel.route.index(origin)]):
            if 'Junction' in vessel.env.FG.nodes[node]:
                junction = vessel.env.FG.nodes[node]['Junction'][0]
                for section in enumerate(junction.section):
                    if origin not in list(section[1].keys()):
                        continue

                    section[1][origin].release(vessel.request_access_section)

                    if junction.type[section[0]] == 'one-way_traffic':
                        if 'access1' not in dir(junction):
                            junction = vessel.env.FG.nodes[origin]['Junction'][0]
                            for section in enumerate(junction.section):
                                if node not in list(section[1].keys()):
                                    continue
                                junction.access2[section[0]][node].release(vessel.request_access_entrance_section)
                                junction.access1[section[0]][origin].release(vessel.request_access_exit_section)
                        else:
                            junction.access1[section[0]][node].release(vessel.request_access_entrance_section)
                            junction.access2[section[0]][origin].release(vessel.request_access_exit_section)
                        break
                break
        return

    def request_access_next_section(vessel, origin, destination):
        for node in vessel.route[vessel.route.index(destination):]:
            if 'Junction' in vessel.env.FG.nodes[node]:
                junction = vessel.env.FG.nodes[origin]['Junction'][0]
                for section in enumerate(junction.section):
                    if node not in list(section[1].keys()):
                        continue
                    vessel.stopping_distance = 15 * vessel.L
                    vessel.stopping_time = vessel.stopping_distance / vessel.v
                    if section[1][node].users != [] and (section[1][node].users[-1].ta + vessel.stopping_time) > vessel.env.now:
                        vessel.request_access_section = section[1][node].request()
                        section[1][node].users[-1].id = vessel.id
                        section[1][node].users[-1].ta = (section[1][node].users[-2].ta + vessel.stopping_time)
                        yield vessel.env.timeout((section[1][node].users[-2].ta + vessel.stopping_time) - vessel.env.now)
                    else:
                        vessel.request_access_section = section[1][node].request()
                        section[1][node].users[-1].ta = vessel.env.now
                        section[1][node].users[-1].id = vessel.id

                    if junction.type[section[0]] == 'one-way_traffic':
                        if 'access1' not in dir(junction):
                            junction = vessel.env.FG.nodes[destination]['Junction'][0]
                            for section in enumerate(junction.section):
                                if origin not in list(section[1].keys()):
                                    continue

                                vessel.request_access_entrance_section = junction.access2[section[0]][origin].request()
                                junction.access2[section[0]][origin].users[-1].id = vessel.id
                                #yield vessel.request_access_entrance_section
                                vessel.request_access_exit_section = junction.access1[section[0]][node].request()
                                #yield vessel.request_access_exit_section
                                junction.access1[section[0]][node].users[-1].id = vessel.id

                        else:
                            vessel.request_access_entrance_section = junction.access1[section[0]][origin].request()
                            junction.access1[section[0]][origin].users[-1].id = vessel.id
                            #yield vessel.request_access_entrance_section
                            vessel.request_access_exit_section = junction.access2[section[0]][node].request()
                            #yield vessel.request_access_exit_section
                            junction.access2[section[0]][node].users[-1].id = vessel.id
                        break
                break
        return

class PassTerminal():
    def move_to_anchorage(vessel,node):
        network = vessel.env.FG
        vessel.waiting_in_anchorage = True
        vessel.return_to_sea = False
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

        for node_anchorage_area in enumerate(sorted_nodes_anchorages):
            if sorted_users_of_anchorages[node_anchorage_area[0]] < sorted_capacity_of_anchorages[node_anchorage_area[0]]:
                node_anchorage = node_anchorage_area[1]
                break

        if node_anchorage != node_anchorage_area[1]:
            print(node_anchorage,node_anchorage_area[1])
            vessel.return_to_sea = True

        anchorage = network.nodes[node_anchorage]['Anchorage'][0]
        vessel.route_after_anchorage = []
        if not vessel.return_to_sea:
            vessel.true_origin = vessel.route[0]
            current_time = vessel.env.now
            vessel.route_after_anchorage = nx.dijkstra_path(vessel.env.FG, node_anchorage, vessel.route[-1])
            yield from Movable.pass_edge(vessel,vessel.route[node], vessel.route[node+1])
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[node+1], node_anchorage)
            vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[node]]
            vessel.env.process(vessel.move())
        else:
            yield from Movable.pass_edge(vessel, vessel.route[node], vessel.route[node + 1])
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[node + 1], vessel.route[node])
            vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[node]]
            vessel.env.process(vessel.move())

        return

    def pass_anchorage(vessel,node):
        network = vessel.env.FG
        anchorage = network.nodes[node]['Anchorage'][0]
        yield from Movable.pass_edge(vessel, vessel.route[vessel.route.index(node)-1], vessel.route[vessel.route.index(node)])

        vessel.anchorage_access = anchorage.anchorage_area[node].request()
        yield vessel.anchorage_access

        anchorage.log_entry("Vessel arrival", vessel.env.now, len(anchorage.anchorage_area[node].users),
                            nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )

        current_time = vessel.env.now
        vessel.log_entry("Waiting in anchorage start", vessel.env.now, 0,
                         nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
        edge = vessel.route_after_anchorage[-2],vessel.route_after_anchorage[-1]
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]

        if vessel.waiting_for_availability_terminal:
            yield vessel.waiting_time_in_anchorage
            new_current_time = vessel.env.now

            if terminal.type == 'quay':
                vessel.index_quay_position,_ = PassTerminal.pick_minimum_length(vessel, terminal)
                PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)
                terminal = terminal.terminal[edge[0]]
            elif terminal.type == 'jetty':
                terminal = terminal.terminal[vessel.index_jetty_position][edge[0]]
                vessel.access_terminal = terminal.request(priority = -1)
                terminal.release(vessel.waiting_time_in_anchorage)
                yield vessel.access_terminal

            vessel.route = vessel.route_after_anchorage
            vessel.available_sail_in_times = NetworkProperties.calculate_available_sail_in_times(vessel,True,False)

            for t in range(len(vessel.available_sail_in_times)):
                if vessel.available_sail_in_times[t][1] == 'Start':
                    continue

                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                    sailing_distance = 0
                    for nodes in enumerate(vessel.route[:-2]):
                        _, _, distance = vessel.wgs84.inv(vessel.env.FG.nodes[vessel.route[nodes[0]]]['geometry'].x,
                                                          vessel.env.FG.nodes[vessel.route[nodes[0]]]['geometry'].y,
                                                          vessel.env.FG.nodes[vessel.route[nodes[0] + 1]]['geometry'].x,
                                                          vessel.env.FG.nodes[vessel.route[nodes[0] + 1]]['geometry'].y)
                        sailing_distance += distance

                if new_current_time >= vessel.available_sail_in_times[t-1][0] and new_current_time <= vessel.available_sail_in_times[t][0]:
                    if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                        terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
                        terminal.users[-1].etd = terminal.users[-1].eta + 30 * 60 + 4 * 3600 + 4 * 3600 + 10 * 60
                    break
                elif new_current_time < vessel.available_sail_in_times[t-1][0]:
                    if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                        terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.available_sail_in_times[t-1][0]-new_current_time
                        terminal.users[-1].etd = terminal.users[-1].eta + 30 * 60 + 4 * 3600 + 4 * 3600 + 10 * 60
                    yield vessel.env.timeout(vessel.available_sail_in_times[t-1][0]-new_current_time)
                elif new_current_time > vessel.available_sail_in_times[t][0]:
                    continue

        elif vessel.waiting_time:
            yield vessel.env.timeout(vessel.waiting_time)

        vessel.log_entry("Waiting in anchorage stop", vessel.env.now, vessel.env.now-current_time,
                         nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
        vessel.route = vessel.route_after_anchorage
        PassSection.release_access_previous_section(vessel, vessel.route[0])
        yield from PassSection.request_access_next_section(vessel, vessel.route[0], vessel.route[1])
        vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
        vessel.env.process(vessel.move())

        anchorage.anchorage_area[node].release(vessel.anchorage_access)

        anchorage.log_entry("Vessel departure", vessel.env.now, len(anchorage.anchorage_area[node].users),
                            nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )

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
            available_quay_lengths.sort()

            for jndex in range(len(available_quay_lengths)):
                if vessel.L <= available_quay_lengths[jndex]:
                    index_quay_position = index
                    break

                elif jndex == len(available_quay_lengths) - 1 and not index_quay_position:
                    move_to_anchorage = True

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
        return

    def request_terminal_access(vessel, edge, node):
        node = vessel.route.index(node)
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        vessel.move_to_anchorage = False
        vessel.waiting_in_anchorage = False
        vessel.waiting_for_availability_terminal = False

        def checks_waiting_time_due_to_tidal_window(vessel, node):
            for nodes in enumerate(vessel.route[node:]):
                if nodes[1] == vessel.route[node]:
                    continue

                required_water_depth = vessel.T_f + vessel.ukc
                minimum_water_depth = (np.min(vessel.env.FG.edges[vessel.route[node + nodes[0] - 1],
                                                                  vessel.route[node + nodes[0]]]['Info']['Water level'][1]) +
                                       vessel.env.FG.edges[vessel.route[node + nodes[0] - 1],
                                                           vessel.route[node + nodes[0]]]['Info']['Depth'][0])
                vessel.waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel,required_water_depth > minimum_water_depth)
                if required_water_depth > minimum_water_depth:
                    break

        if terminal.type == 'jetty':
            minimum_waiting_time = []
            vessels_waiting = []
            for jetty in enumerate(terminal.terminal):
                if jetty[1][edge[0]].users == [] and jetty[1][edge[0]].queue == []:
                    vessel.index_jetty_position = jetty[0]
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

        if vessel.move_to_anchorage:
            vessel.waiting_for_availability_terminal = True

            if terminal.type == 'quay':
                vessel.waiting_time_in_anchorage = terminal.length.get(vessel.L)

            elif terminal.type == 'jetty':
                checks_waiting_time_due_to_tidal_window(vessel, node)
                indices_empty_queue_for_jetty = [waiting_vessels[0] for waiting_vessels in enumerate(vessels_waiting) if waiting_vessels[1] == 0]
                minimum_waiting_time = [np.max([x,vessel.waiting_time+vessel.env.now]) for x in minimum_waiting_time]
                min_minimum_waiting_time = -1
                for index in indices_empty_queue_for_jetty:
                    if minimum_waiting_time[index] < 0 and minimum_waiting_time[index] < min_minimum_waiting_time:
                        min_minimum_waiting_time = minimum_waiting_time[index]
                        vessel.index_jetty_position = index
                if min_minimum_waiting_time <= 0:
                    vessel.index_jetty_position = np.argmin(vessels_waiting)
                vessel.waiting_time_in_anchorage = terminal.terminal[vessel.index_jetty_position][edge[0]].request()

            yield from PassTerminal.move_to_anchorage(vessel, node)

        else:
            if terminal.type == 'quay':
                PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)
            checks_waiting_time_due_to_tidal_window(vessel, node)
            if vessel.waiting_time:
                yield from PassTerminal.move_to_anchorage(vessel, node)

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

            sailing_distance = 0
            for nodes in enumerate(vessel.route[:-2]):
                _, _, distance = vessel.wgs84.inv(vessel.env.FG.nodes[vessel.route[nodes[0]]]['geometry'].x,
                                                  vessel.env.FG.nodes[vessel.route[nodes[0]]]['geometry'].y,
                                                  vessel.env.FG.nodes[vessel.route[nodes[0] + 1]]['geometry'].x,
                                                  vessel.env.FG.nodes[vessel.route[nodes[0] + 1]]['geometry'].y)
                sailing_distance += distance

            terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
            terminal.users[-1].etd = terminal.users[-1].eta + 30 * 60 + 4 * 3600 + 4 * 3600 + 10 * 60

    def pass_terminal(vessel,edge):
        yield from Movable.pass_edge(vessel, edge[0], edge[1])
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        index = vessel.route[vessel.route.index(edge[1]) - 1]

        if terminal.type == 'quay':
            terminal.pos_length.get(vessel.L)
            terminal.log_entry("Arrival of vessel", vessel.env.now, terminal.length.capacity-terminal.pos_length.level,
                               nx.get_node_attributes(vessel.env.FG, "geometry")[index], )
        elif terminal.type == 'jetty':
            terminal.jetties_occupied += 1
            terminal.log_entry("Arrival of vessel", vessel.env.now, terminal.jetties_occupied,
                               nx.get_node_attributes(vessel.env.FG, "geometry")[index], )

        # Berthing
        vessel.log_entry("Berthing start", vessel.env.now, 0,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(30 * 60)
        vessel.log_entry("Berthing stop", vessel.env.now, 30 * 60,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        for section in enumerate(vessel.env.FG.nodes[edge[0]]['Junction'][0].section):
            if list(section[1].keys())[0] == edge[1]:
                break

        if 'Junction' in vessel.env.FG.nodes[edge[0]].keys():
            PassSection.release_access_previous_section(vessel, edge[1])

        # Unloading
        vessel.log_entry("Unloading start", vessel.env.now, 0,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(4 * 3600)
        vessel.log_entry("Unloading stop", vessel.env.now, 4 * 3600,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Loading
        vessel.log_entry("Loading start", vessel.env.now, 0,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(4 * 3600)
        vessel.log_entry("Loading stop", vessel.env.now, 4 * 3600,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # New Route
        vessel.dir = 0
        if 'true_origin' in dir(vessel):
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[vessel.route.index(edge[1])], vessel.true_origin)
        else:
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[vessel.route.index(edge[1])], vessel.route[0])

        # Waiting for tidal window
        for nodes in enumerate(vessel.route):
            if nodes[1] == vessel.route[0]:
                continue

            required_water_depth = vessel.T_f
            minimum_water_depth = (np.min(vessel.env.FG.edges[vessel.route[nodes[0] - 1],
                                                              vessel.route[nodes[0]]]['Info']['Water level'][1]) +
                                          vessel.env.FG.edges[vessel.route[nodes[0] - 1],
                                                              vessel.route[nodes[0]]]['Info']['Depth'][0])

            vessel.waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel,required_water_depth > minimum_water_depth)
            if vessel.waiting_time:
                vessel.log_entry("Waiting for tidal window start", vessel.env.now, 0,
                                 shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
                yield vessel.env.timeout(vessel.waiting_time)

                vessel.log_entry("Waiting for tidal window stop", vessel.env.now, vessel.waiting_time,
                                 shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
                break

        # Deberthing
        for section in enumerate(vessel.env.FG.nodes[edge[0]]['Junction'][0].section):
            if list(section[1].keys())[0] == edge[1]:
                break

        if 'Junction' in vessel.env.FG.nodes[edge[0]].keys():
            yield from PassSection.request_access_next_section(vessel, edge[1], edge[0])

        if 'Turning Basin' in vessel.env.FG.nodes[edge[0]].keys():
            turning_basin = vessel.env.FG.nodes[edge[0]]['Turning Basin'][0]
            vessel.request_access_turning_basin = turning_basin.turning_basin[edge[0]].request()
            yield vessel.request_access_turning_basin

        vessel.log_entry("Deberthing start", vessel.env.now, 0,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(10 * 60)
        vessel.log_entry("Deberthing stop", vessel.env.now, 10 * 60,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

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

            terminal.log_entry("Departure of vessel", vessel.env.now, terminal.length.capacity-terminal.pos_length.level,
                               nx.get_node_attributes(vessel.env.FG, "geometry")[index], )
            terminal.terminal[edge[0]].release(vessel.access_terminal)

        elif terminal.type == 'jetty':
            terminal.jetties_occupied -= 1

            terminal.log_entry("Departure of vessel", vessel.env.now, terminal.jetties_occupied,
                               nx.get_node_attributes(vessel.env.FG, "geometry")[index], )
            terminal.terminal[vessel.index_jetty_position][edge[0]].release(vessel.access_terminal)

        vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
        vessel.env.process(vessel.move())
        vessel.leaving_port = True

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
        if (
            self.geometry
            != nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]
        ):
            orig = self.geometry
            dest = nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]

            print("Origin", orig)
            print("Destination", dest)

            self.distance += self.wgs84.inv(
                shapely.geometry.asShape(orig).x,
                shapely.geometry.asShape(orig).y,
                shapely.geometry.asShape(dest).x,
                shapely.geometry.asShape(dest).y,
            )[2]

            yield self.env.timeout(self.distance / self.current_speed)
            self.log_entry("Sailing to start", self.env.now, self.distance, dest)


        # Move over the path and log every step
        for node in enumerate(self.route):
            self.node = node[1]

            if node[0] + 2 <= len(self.route):
                origin = self.route[node[0]]
                destination = self.route[node[0] + 1]

            #Leave and access waterway section
            if 'Junction' in self.env.FG.nodes[origin].keys():
                if 'Anchorage' not in self.env.FG.nodes[origin].keys():
                    PassSection.release_access_previous_section(self,origin)
                    yield from PassSection.request_access_next_section(self, origin, destination)

            if 'Turning Basin' in self.env.FG.nodes[origin].keys():
                turning_basin = self.env.FG.nodes[origin]['Turning Basin'][0]
                if self.dir == 0:
                    self.log_entry("Vessel Turning Start", self.env.now, 0, self.env.FG.nodes[origin]['geometry'])
                    turning_basin.log_entry("Vessel Turning Start", self.env.now, 0, self.env.FG.nodes[origin]['geometry'] )
                    yield self.env.timeout(10*60)
                    turning_basin.log_entry("Vessel Turning Stop", self.env.now, 10*60, self.env.FG.nodes[origin]['geometry'] )
                    self.log_entry("Vessel Turning Stop", self.env.now, 10*60, self.env.FG.nodes[origin]['geometry'])
                    self.dir = -1
                else:
                    self.log_entry("Passing Turning Basin", self.env.now, 0, self.env.FG.nodes[origin]['geometry'])
                    turning_basin.log_entry("Vessel Passing", self.env.now, 0,self.env.FG.nodes[origin]['geometry'])
                turning_basin.turning_basin[origin].release(self.request_access_turning_basin)

            if 'Turning Basin' in self.env.FG.nodes[destination].keys():
                turning_basin = self.env.FG.nodes[destination]['Turning Basin'][0]
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
                yield from self.pass_edge(origin, destination)
                self.v = 4*self.v

            # Request for a terminal
            if "Origin" in self.env.FG.nodes[origin] and 'leaving_port' not in dir(self):
                self.dir = -1
                yield from PassTerminal.request_terminal_access(self, [self.route[-2], self.route[-1]], origin)
                if self.waiting_in_anchorage:

                    break

            # Anchorage
            if 'Anchorage' in self.env.FG.nodes[destination].keys() and self.route[-1] == destination:
                yield from PassTerminal.pass_anchorage(self, destination)
                break

            # Terminal
            if 'Terminal' in self.env.FG.edges[origin, destination].keys() and self.route[-1] == destination:
                yield from PassTerminal.pass_terminal(self, [origin, destination])
                break

            else:
                # print('I am going to go to the next node {}'.format(destination))
                yield from self.pass_edge(origin, destination)

            if node[0] + 2 == len(self.route):
                break

        # self.geometry = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        logger.debug("  distance: " + "%4.2f" % self.distance + " m")
        logger.debug("  sailing:  " + "%4.2f" % self.current_speed + " m/s")
        logger.debug(
            "  duration: "
            + "%4.2f" % ((self.distance / self.current_speed) / 3600)
            + " hrs"
        )

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
                fwd_azimuth, _, _ = self.wgs84.inv(origin_lat, origin_lon,
                                                          destination_lat, destination_lon)

                if terminal.type == 'quay':
                    position = self.quay_position

                elif terminal.type == 'jetty':
                    position = terminal.jetty_locations[self.index_jetty_position]

                [self.terminal_pos_lat, self.terminal_pos_lon,_] = self.wgs84.fwd(self.env.FG.nodes[origin]['geometry'].x,
                                                                           self.env.FG.nodes[origin]['geometry'].y,
                                                                           fwd_azimuth, position)
                dest = shapely.geometry.Point(self.terminal_pos_lat,self.terminal_pos_lon)

            else:
                orig = shapely.geometry.Point(self.terminal_pos_lat, self.terminal_pos_lon)

        if 'geometry' in edge:
            edge_route = np.array(edge['geometry'])

            # check if edge is in the sailing direction, otherwise flip it
            distance_from_start = self.wgs84.inv(
                    orig.x,
                    orig.y,
                    edge_route[0][0],
                    edge_route[0][1],
                )[2]
            distance_from_stop = self.wgs84.inv(
                    orig.x,
                    orig.y,
                    edge_route[-1][0],
                    edge_route[-1][1],
                )[2]
            if distance_from_start>distance_from_stop:
                # when the distance from the starting point is greater than from the end point
                edge_route = np.flipud(np.array(edge['geometry']))

            for index, pt in enumerate(edge_route[:-1]):
                sub_orig = shapely.geometry.Point(edge_route[index][0], edge_route[index][1])
                sub_dest = shapely.geometry.Point(edge_route[index+1][0], edge_route[index+1][1])

                distance = self.wgs84.inv(
                    shapely.geometry.asShape(sub_orig).x,
                    shapely.geometry.asShape(sub_orig).y,
                    shapely.geometry.asShape(sub_dest).x,
                    shapely.geometry.asShape(sub_dest).y,
                )[2]
                self.distance += distance
                self.log_entry("Sailing from node {} to node {} sub edge {} start".format(origin, destination, index), self.env.now, 0, sub_orig,)
                yield self.env.timeout(distance / self.current_speed)
                self.log_entry("Sailing from node {} to node {} sub edge {} stop".format(origin, destination, index), self.env.now, 0, sub_dest,)
            self.geometry = dest
            # print('   My new origin is {}'.format(destination))
        else:
            distance = self.wgs84.inv(
                shapely.geometry.asShape(orig).x,
                shapely.geometry.asShape(orig).y,
                shapely.geometry.asShape(dest).x,
                shapely.geometry.asShape(dest).y,
            )[2]

            self.distance += distance
            arrival = self.env.now

            # Act based on resources
            if "Resources" in edge.keys():
                with self.env.FG.edges[origin, destination]["Resources"].request() as request:
                    yield request

                    if arrival != self.env.now:
                        self.log_entry("Waiting to pass edge {} - {} start".format(origin, destination), arrival, 0, orig,)
                        self.log_entry("Waiting to pass edge {} - {} stop".format(origin, destination), self.env.now, 0, orig,)

                    self.log_entry("Sailing from node {} to node {} start".format(origin, destination), self.env.now, 0, orig,)
                    yield self.env.timeout(distance / self.current_speed)
                    self.log_entry("Sailing from node {} to node {} stop".format(origin, destination), self.env.now, 0, dest,)

            else:
                self.log_entry("Sailing from node {} to node {} start".format(origin, destination), self.env.now, 0, orig,)
                yield self.env.timeout(distance / self.current_speed)
                self.log_entry("Sailing from node {} to node {} stop".format(origin, destination), self.env.now, 0, dest,)

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
