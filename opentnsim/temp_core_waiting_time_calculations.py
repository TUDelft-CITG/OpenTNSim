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
import matplotlib.pyplot as plt

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
                for t in range(len(vmag[1][node[0]][0])):
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

    def calculate_available_sail_in_times(vessel,vertical_tidal_window,horizontal_tidal_window,max_waiting_time,route,delay=0,out=False,plot=False):
        available_sail_in_times = []
        vessel.waiting_time_start = vessel.env.now+delay
        vessel.max_waiting_time = vessel.waiting_time_start + 2*vessel.metadata['max_waiting_time']
        ax1 = []
        ax2 = []
        if plot:
            #pass
            fig, ax1 = plt.subplots(figsize=[10, 10])
            ax2 = ax1.twinx()

        def times_vertical_tidal_window(vessel,route,pl,max_waiting_time,out=out,plot=plot,delay=0):
            times_vertical_tidal_window = []
            water_depth_required = vessel.T_f + vessel.metadata['ukc']
            [new_t, min_wdep,sailing_time] = NetworkProperties.minimum_water_per_edge_as_experienced_by_vessel(vessel,route,out=out,delay=delay)
            root_interp_water_level_at_edge = sc.interpolate.CubicSpline(new_t,[x-water_depth_required for x in min_wdep])
            if np.max([x - water_depth_required for x in min_wdep]) < 0:
                times_vertical_tidal_window.append([vessel.waiting_time_start-sailing_time, 'Start'])
                times_vertical_tidal_window.append([vessel.max_waiting_time, 'Stop'])
            else:
                for root in root_interp_water_level_at_edge.roots():
                    if root > np.min(new_t) and root < np.max(new_t) and min_wdep[[i for i, x in enumerate(list(new_t > root)) if x][0]] > water_depth_required:
                        times_vertical_tidal_window.append([root-sailing_time, 'Stop'])
                    elif root > np.min(new_t) and root < np.max(new_t) and min_wdep[[i for i, x in enumerate(list(new_t > root)) if x][0]] < water_depth_required:
                        times_vertical_tidal_window.append([root-sailing_time, 'Start'])

                if times_vertical_tidal_window != []:
                    if times_vertical_tidal_window[0][1] == 'Start' and times_vertical_tidal_window[0][0] > vessel.waiting_time_start-sailing_time:
                        times_vertical_tidal_window.insert(0,[vessel.waiting_time_start-sailing_time, 'Stop'])
                    elif times_vertical_tidal_window[0][1] == 'Stop' and times_vertical_tidal_window[0][0] > vessel.waiting_time_start-sailing_time:
                        times_vertical_tidal_window.insert(0,[vessel.waiting_time_start-sailing_time, 'Start'])

                    if times_vertical_tidal_window[-1][1] == 'Start' and times_vertical_tidal_window[-1][0] < max_waiting_time:
                        times_vertical_tidal_window.append([max_waiting_time, 'Stop'])
            dot1 = []
            if plot:
                #pass
                ax1.plot([t-sailing_time for t in new_t],min_wdep,color='deepskyblue')
                ax1.plot([x[0] for x in times_vertical_tidal_window], (vessel.metadata['ukc']+vessel.T_f) * np.ones(len(times_vertical_tidal_window)), color='dodgerblue',marker='o',linestyle='None')
                ax1.text(np.mean(new_t), 1.001*(vessel.metadata['ukc']+vessel.T_f), 'Required water depth', color='dodgerblue',horizontalalignment='center')
                ax1.axhline((vessel.metadata['ukc']+vessel.T_f), color='dodgerblue', linestyle='--')
                linelist = []
                for t2 in range(len(times_vertical_tidal_window)):
                    if times_vertical_tidal_window[t2][1] == 'Start' and t2 != 0:
                        linelist.append([times_vertical_tidal_window[t2][0], times_vertical_tidal_window[t2 - 1][0]])
                for lines in linelist:
                    dot1, = ax2.plot([lines[1], lines[0]], [0,0], color='dodgerblue',marker='o')
            return times_vertical_tidal_window, dot1

        def times_horizontal_tidal_window(vessel,route,pl,max_waiting_time,plot=plot,delay=0):
            network = vessel.env.FG
            current_time = vessel.env.now+delay
            waiting_time = 0
            time_edge_is_navigable = 0
            max_cross_current_velocity = vessel.metadata['max_cross_current']
            times_horizontal_tidal_windows = []

            for nodes in enumerate(route):
                if nodes[0] == 0:
                    continue
                elif nodes[1] == route[-1]:
                    continue

                if nodes[1] != 'Node 15':
                    continue

                [cross_current, root_interp_cross_current_orig, new_t, cor_cross_current, _,sailing_time] = NetworkProperties.cross_current_calculator(vessel, max_cross_current_velocity,nodes[0], route=route, delay=delay)
                if np.min(cor_cross_current) < 0:
                    times_edge_is_navigable = root_interp_cross_current_orig.roots()
                    times_horizontal_tidal_window = []
                    if plot:
                        ax2.plot([t - sailing_time for t in new_t], [y + 0.25 for y in cor_cross_current],color='lightcoral')
                        ax2.set_ylabel('Cross-current velocity [m/s]', color='lightcoral')
                    for root in root_interp_cross_current_orig.roots():
                        if root > np.min(new_t) and root < np.max(new_t) and cor_cross_current[[i for i, x in enumerate(list(new_t > root)) if x][0]] < 0:
                            times_horizontal_tidal_window.append([root - sailing_time, 'Stop'])
                        elif root > np.min(new_t) and root < np.max(new_t) and cor_cross_current[[i for i, x in enumerate(list(new_t > root)) if x][0]] > 0:
                            times_horizontal_tidal_window.append([root - sailing_time, 'Start'])

                    if times_horizontal_tidal_window == []:
                        times_horizontal_tidal_window.append([vessel.waiting_time_start - sailing_time, 'Stop'])
                        times_horizontal_tidal_window.append([max_waiting_time, 'Start'])

                    if times_horizontal_tidal_window[0][1] == 'Start' and times_horizontal_tidal_window[0][0] > vessel.waiting_time_start - sailing_time:
                        times_horizontal_tidal_window.append([vessel.waiting_time_start - sailing_time, 'Stop'])
                    if times_horizontal_tidal_window[-1][1] == 'Start' and times_horizontal_tidal_window[-1][0] < max_waiting_time:
                        times_horizontal_tidal_window.append([max_waiting_time, 'Stop'])

                    times_horizontal_tidal_windows.append(times_horizontal_tidal_window)
            dot2 = []
            if plot:
                #pass
                ax2.plot([t - sailing_time for t in new_t], [y + 0.25 for y in cor_cross_current],color='lightcoral')
                ax2.set_ylabel('Cross-current velocity [m/s]', color='lightcoral')
                ax2.plot([x[0] for x in times_horizontal_tidal_window],0.25* np.ones(len(times_horizontal_tidal_window)), color='indianred',marker='o',linestyle='None')
                ax2.axhline(0.25, color='indianred', linestyle='--')
                ax2.text(np.mean(new_t), 1.05 * 0.25, 'Critical cross_current', color='indianred', horizontalalignment='center')
                linelist = []
                for t1 in times_horizontal_tidal_windows:
                    for t2 in range(len(t1)):
                        if t1[t2][1] == 'Start'and t2 != 0:
                            linelist.append([t1[t2][0],t1[t2-1][0]])
                    for lines in linelist:
                        dot2, = ax2.plot([lines[1],lines[0]],[-0.05,-0.05],color='indianred',marker='o')
            return times_horizontal_tidal_windows, dot2

        def times_tidal_window(vessel,vertical_tidal_window,horizontal_tidal_window,route,pl1,pl2,max_waiting_time,plot=plot,delay=0):
            list_of_times_vertical_tidal_window = []
            list_of_times_horizontal_tidal_windows = []
            dot1 = []
            dot2 = []
            if vertical_tidal_window:
                list_of_times_vertical_tidal_window,dot1 = times_vertical_tidal_window(vessel,route,max_waiting_time=max_waiting_time,pl=pl1,plot=plot,delay=delay)
            if horizontal_tidal_window and not list_of_times_vertical_tidal_window == [[vessel.waiting_time_start, 'Start'],[vessel.max_waiting_time, 'Stop']]:
                list_of_times_horizontal_tidal_windows,dot2 = times_horizontal_tidal_window(vessel,route,max_waiting_time=max_waiting_time,pl=pl2,plot=plot,delay=delay)

            if vertical_tidal_window and list_of_times_vertical_tidal_window:
                tmin = np.min([x[0] for x in list_of_times_vertical_tidal_window])
            else:
                tmin = 0

            list_indexes = list(np.arange(0, len(list_of_times_horizontal_tidal_windows) + 1))
            times_tidal_window = []
            list_of_list_indexes = []

            for time in list_of_times_vertical_tidal_window:
                times_tidal_window.append(time)
                list_of_list_indexes.append(0)
            for index in range(len(list_indexes)-1):
                for time in list_of_times_horizontal_tidal_windows[index]:
                    if time[0] >= tmin:
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

            #for list_index in list_indexes:
            #    for time1 in range(len(times_tidal_window)):
            #        if times_tidal_window[time1][1] == 'Stop' and list_of_list_indexes[time1] == list_index:
            #            for time2 in range(len(times_tidal_window)):
            #                if time2 < time1 and times_tidal_window[time2][1] == 'Start' and list_of_list_indexes[time2] == list_index:
            #                    break
            #            if time2 > time1:
            #                indexes = np.arange(0, time1, 1)
            #                for index in indexes:
            #                    indexes_to_be_removed.append(index)

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
            return times_tidal_window,tmin,dot1,dot2

        available_sail_in_times,tmin,dot1,dot2 = times_tidal_window(vessel,vertical_tidal_window,horizontal_tidal_window,route,max_waiting_time=vessel.max_waiting_time,pl1=ax1,pl2=ax2,plot=plot,delay=delay)
        dot3 = []
        if plot:
            #pass
            linelist = []
            for t2 in range(len(available_sail_in_times)):
                if available_sail_in_times[t2][1] == 'Start' and t2 != 0:
                    linelist.append([available_sail_in_times[t2][0], available_sail_in_times[t2 - 1][0]])
            for lines in linelist:
                dot3, = ax2.plot([lines[1], lines[0]],[-0.09,-0.09],color='darkslateblue', marker='o')
            plt.ylim([-0.1,1])
            plt.title('Tidal window calculation'+str(vessel.type))
            plt.legend([dot1, dot2, dot3], ['Vertical tidal window', 'Horizontal tidal window', 'Resulting tidal window'])
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('Experienced water depth [m]', color='deepskyblue')
            ax2.spines['left'].set_color('deepskyblue')
            ax2.spines['right'].set_color('lightcoral')
            plt.show()
        #print(available_sail_in_times,vessel.env.now)
        return available_sail_in_times

    def minimum_water_per_edge_as_experienced_by_vessel(vessel,route,delay=0,out=False):
        network = vessel.env.FG
        current_time = vessel.env.now+delay
        min_wdep = [[] for _ in range(len(route) - 1)]
        new_t = [[] for _ in range(len(route) - 1)]
        distance_to_next_node = 0
        i = -1
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

            sailing_time_to_next_node = distance_to_next_node / vessel.v

            if nodes[1] != 'Node 7' and nodes[1] != 'Node 17':
                continue

            i += 1
            if i == 0:
                sailing_time_to_start_tidal_window = sailing_time_to_next_node
            t_wlev_old = network.edges[route[nodes[0] - 1], route[nodes[0]]]['Info']['Water level'][0]
            wlev_node1_old = network.edges[route[0], route[1]]['Info']['Water level'][1][0]
            wlev_node2_old = network.edges[route[nodes[0] - 1], route[nodes[0]]]['Info']['Water level'][1][1]
            it_wlev = [t[0] for t in enumerate(t_wlev_old) if t[1] >= current_time - vessel.metadata['max_waiting_time']-(sailing_time_to_next_node+sailing_time_to_start_tidal_window) and t[1] <= current_time + 3 * vessel.metadata['max_waiting_time']-(sailing_time_to_next_node+sailing_time_to_start_tidal_window)]
            t_wlev = [t for t in t_wlev_old if t >= current_time-vessel.metadata['max_waiting_time']-(sailing_time_to_next_node+sailing_time_to_start_tidal_window) and t <= current_time+3*vessel.metadata['max_waiting_time']-(sailing_time_to_next_node+sailing_time_to_start_tidal_window)]
            wlev_node1 = [wlev[1] for wlev in enumerate(wlev_node1_old) if wlev[0] in it_wlev]
            wlev_node2 = [wlev[1] for wlev in enumerate(wlev_node2_old) if wlev[0] in it_wlev]
            depth1 = network.edges[route[0], route[1]]['Info']['Depth'][0]
            depth2 = network.edges[route[nodes[0] - 1], route[nodes[0]]]['Info']['Depth'][0]
            if np.max([np.max(wlev_node1) + depth1, np.max(wlev_node2) + depth2]) > max_wdep:
                max_wdep = np.max([np.max(wlev_node1) + depth1, np.max(wlev_node2) + depth2])

            interp_wdep_node1 = sc.interpolate.CubicSpline(t_wlev, [y + depth1 for y in wlev_node1])
            eta_next_node = [t for t in t_wlev]
            interp_wdep_node2 = sc.interpolate.CubicSpline(eta_next_node, [y + depth2 for y in wlev_node2])

            if current_time+vessel.metadata['max_waiting_time'] - current_time >= 3000:
                new_t[i] = np.arange(current_time, current_time+2*vessel.metadata['max_waiting_time']+sailing_time_to_start_tidal_window, 300)
            else:
                new_t[i] = np.linspace(current_time, current_time+2*vessel.metadata['max_waiting_time']+sailing_time_to_start_tidal_window, 10)

            for t in new_t[i]:
                 min_wdep[i].append(np.min([interp_wdep_node1(t), interp_wdep_node2(t)]))

        minimum_water_depth = []
        time_minimum_water_depth = [t for t in new_t[i]]
        for t in range(len(time_minimum_water_depth)):
            min_water_depth = max_wdep
            for edge in range(i+1):
                if min_wdep[edge][t] < min_water_depth:
                    min_water_depth = min_wdep[edge][t]
            minimum_water_depth.append(min_water_depth)
        return time_minimum_water_depth, minimum_water_depth, sailing_time_to_start_tidal_window

    def cross_current_calculator(vessel,max_cross_current_velocity,node_index,route,delay=0):
        network = vessel.env.FG
        node = route[node_index]
        current_time = vessel.env.now+delay

        distance_to_node = 0
        for n in range(len(route)):
            if n == 0:
                continue
            elif route[n] == node:
                break
            distance_to_node += pyproj.Geod(ellps='WGS84').inv(network.nodes[route[n - 1]]['geometry'].x,
                                                               network.nodes[route[n - 1]]['geometry'].y,
                                                               network.nodes[route[n]]['geometry'].x,
                                                               network.nodes[route[n]]['geometry'].y)[2]

        sailing_time = distance_to_node / vessel.v

        time_current_velocity_node_old = network.nodes[route[node_index]]['Info']['Current velocity'][0]
        current_velocity_node_old = network.nodes[route[node_index]]['Info']['Current velocity'][1]
        time_current_direction_node_old = network.nodes[route[node_index]]['Info']['Current direction'][0]
        current_direction_node_old = network.nodes[route[node_index]]['Info']['Current direction'][1]
        it_vel = [t[0] for t in enumerate(time_current_velocity_node_old) if t[1] >= current_time - vessel.metadata['max_waiting_time'] and t[1] <= current_time + 3 *vessel.metadata['max_waiting_time']]
        it_dir = [t[0] for t in enumerate(time_current_direction_node_old) if t[1] >= current_time - vessel.metadata['max_waiting_time'] and t[1] <= current_time + 3 *vessel.metadata['max_waiting_time']]
        time_current_velocity_node = [t for t in time_current_velocity_node_old if t >= current_time - vessel.metadata['max_waiting_time'] and t <= current_time + 3 *vessel.metadata['max_waiting_time']]
        time_current_direction_node = [t for t in time_current_direction_node_old if t >= current_time - vessel.metadata['max_waiting_time'] and t <= current_time + 3 *vessel.metadata['max_waiting_time']]
        current_velocity_node = [vel[1] for vel in enumerate(current_velocity_node_old) if vel[0] in it_vel]
        current_direction_node = [dir[1] for dir in enumerate(current_direction_node_old) if dir[0] in it_dir]

        origin_lat = vessel.env.FG.nodes[route[node_index - 1]]['geometry'].x
        origin_lon = vessel.env.FG.nodes[route[node_index - 1]]['geometry'].y
        node_lat = vessel.env.FG.nodes[route[node_index]]['geometry'].x
        node_lon = vessel.env.FG.nodes[route[node_index]]['geometry'].y
        destination_lat = vessel.env.FG.nodes[route[node_index + 1]]['geometry'].x
        destination_lon = vessel.env.FG.nodes[route[node_index + 1]]['geometry'].y

        course, _, _ = pyproj.Geod(ellps="WGS84").inv(origin_lat, origin_lon, node_lat, node_lon)
        if course < 0:
            course = 180 + course
        heading, _, _ = pyproj.Geod(ellps="WGS84").inv(node_lat, node_lon, destination_lat, destination_lon)
        if heading < 0:
            heading = 180 + heading

        interp_current_velocity_orig = sc.interpolate.CubicSpline(time_current_velocity_node, current_velocity_node)
        interp_current_direction_orig = sc.interpolate.interp1d(time_current_direction_node, current_direction_node)
        cross_current = []

        if current_time + vessel.metadata['max_waiting_time'] - current_time >= 3000:
            new_t = np.arange(current_time, current_time + 2*vessel.metadata['max_waiting_time']+sailing_time, 300)
        else:
            new_t = np.linspace(current_time, current_time + 2*vessel.metadata['max_waiting_time']+sailing_time, 10)

        for t in new_t:
            cross_current.append(np.max([abs(interp_current_velocity_orig(t) * np.sin((interp_current_direction_orig(t) - course) / 180 * math.pi)),
                                         abs(interp_current_velocity_orig(t) * np.sin((interp_current_direction_orig(t) - heading) / 180 * math.pi))]))
        corrected_cross_current = [y - max_cross_current_velocity for y in cross_current]
        interpolated_corrected_cross_current_signal = sc.interpolate.CubicSpline(new_t,corrected_cross_current)

        if current_time+sailing_time < time_current_velocity_node[-1]:
            actual_cross_current = np.max([abs(interp_current_velocity_orig(current_time) * np.sin((interp_current_direction_orig(current_time) - course) / 180 * math.pi)),
                                           abs(interp_current_velocity_orig(current_time) * np.sin((interp_current_direction_orig(current_time) - heading) / 180 * math.pi))])
        else:
            actual_cross_current = 0
        return [actual_cross_current, interpolated_corrected_cross_current_signal, new_t, corrected_cross_current,cross_current, sailing_time]

    def waiting_time_for_tidal_window(vessel,vertical_tidal_window,route,horizontal_tidal_window = True,max_waiting_time = True,delay=0,out=False,plot=False):
        sail_in_times = NetworkProperties.calculate_available_sail_in_times(vessel, vertical_tidal_window, horizontal_tidal_window,max_waiting_time,route=route,delay=delay,out=out,plot=plot)
        waiting_time = 0
        current_time = vessel.env.now+delay
        for t in range(len(sail_in_times)):
            if sail_in_times[t][1] == 'Start':
                if t == len(sail_in_times)-1:
                    waiting_time = sail_in_times[t][0] - current_time
                    #waiting_time = vessel.metadata['max_waiting_time']
                    break
                else:
                    continue
            if current_time >= sail_in_times[t][0]:
                waiting_time = 0
                if t == len(sail_in_times)-1 or current_time < sail_in_times[t+1][0]:
                    break
            elif current_time <= sail_in_times[t][0]:
                if current_time < sail_in_times[t-1][0]:
                    waiting_time = 0
                else:
                    waiting_time = sail_in_times[t][0] - current_time
                break
            elif t == len(sail_in_times) - 1:
                waiting_time = sail_in_times[t][0] - current_time
                #waiting_time = vessel.metadata['max_waiting_time']
            else:
                continue
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
            sail_out_times = NetworkProperties.calculate_available_sail_in_times(vessel, vertical_tidal_window,horizontal_tidal_window, max_waiting_time, route=route, delay=delay,plot=plot)
            for t in range(len(sail_out_times)):
                if sail_out_times[t][1] == 'Start':
                    continue
                if current_time >= sail_out_times[t][0]:
                    if t == len(sail_out_times)-1 or current_time < sail_out_times[t+1][0]:
                        break
                elif current_time <= sail_out_times[t][0]:
                    if sail_out_times[t][0]-current_time >= vessel.metadata['max_waiting_time']:
                        waiting_time = vessel.metadata['max_waiting_time']
                    else:
                        break
                elif t == len(sail_out_times)-1:
                    waiting_time = vessel.metadata['max_waiting_time']
                else:
                    continue
            route.reverse()
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
                                junction.access2[0][node].release(vessel.request_access_entrance_section) #section[0]
                                junction.access1[0][origin].release(vessel.request_access_exit_section) #section[0]
                        else:
                            junction.access1[0][node].release(vessel.request_access_entrance_section) #section[0]
                            junction.access2[0][origin].release(vessel.request_access_exit_section) #section[0]
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
                            junction = vessel.env.FG.nodes[node]['Junction'][0]
                            for section in enumerate(junction.section):
                                if origin not in list(section[1].keys()):
                                    continue

                                vessel.request_access_entrance_section = junction.access2[0][origin].request() #section[0]
                                junction.access2[0][origin].users[-1].id = vessel.id #section[0]
                                #yield vessel.request_access_entrance_section
                                vessel.request_access_exit_section = junction.access1[0][node].request() #section[0]
                                #yield vessel.request_access_exit_section
                                junction.access1[0][node].users[-1].id = vessel.id #section[0]

                        else:
                            vessel.request_access_entrance_section = junction.access1[0][origin].request() #section[0]
                            junction.access1[0][origin].users[-1].id = vessel.id #section[0]
                            #yield vessel.request_access_entrance_section
                            vessel.request_access_exit_section = junction.access2[0][node].request() #section[0]
                            #yield vessel.request_access_exit_section
                            junction.access2[0][node].users[-1].id = vessel.id #section[0]
                        break
                break
        return

class PassTerminal():
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

        #if node_anchorage != node_anchorage_area[1]:
        #    vessel.return_to_sea = True
        #    vessel.waiting_time = vessel.metadata['max_waiting_time']

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

        anchorage.log_entry("Vessel arrival", vessel.env.now, len(anchorage.anchorage_area[node].users),
                            nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )

        current_time = vessel.env.now
        vessel.log_entry("Waiting in anchorage start", vessel.env.now, 0,
                         nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
        edge = vessel.route_after_anchorage[-2], vessel.route_after_anchorage[-1]
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]

        # def checks_waiting_time_due_to_tidal_window(vessel, route, maximum_waiting_time = False, node=0):
        #     for nodes in enumerate(vessel.route[node:]):
        #         if nodes[1] == vessel.route[node]:
        #             continue
        #
        #         required_water_depth = vessel.T_f + vessel.metadata['ukc']
        #         minimum_water_depth = (np.min(vessel.env.FG.edges[vessel.route[node + nodes[0] - 1],
        #                                                           vessel.route[node + nodes[0]]]['Info']['Water level'][1]) +
        #                                vessel.env.FG.edges[vessel.route[node + nodes[0] - 1],
        #                                                    vessel.route[node + nodes[0]]]['Info']['Depth'][0])
        #         if required_water_depth > minimum_water_depth:
        #             break
        #
        #     vessel.waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel,True,route = route,plot=True)
        #     if vessel.waiting_time >= vessel.metadata['max_waiting_time'] and maximum_waiting_time:
        #        vessel.return_to_sea = True
        #        vessel.waiting_time = 0
        #     else:
        #        vessel.return_to_sea = False

        #if vessel.return_to_sea == False:
        #   checks_waiting_time_due_to_tidal_window(vessel,route=vessel.route_after_anchorage)
        if vessel.return_to_sea == False and vessel.waiting_for_availability_terminal:
            yield vessel.waiting_time_in_anchorage #| vessel.env.timeout(vessel.metadata['max_waiting_time']-(vessel.env.now-current_time))
            new_current_time = vessel.env.now

            #if new_current_time - current_time >= vessel.metadata['max_waiting_time']:
            #    vessel.return_to_sea = True
            #    vessel.waiting_time = 0
            #else:
            if terminal.type == 'quay':
                vessel.index_quay_position, _ = PassTerminal.pick_minimum_length(vessel, terminal)
                PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)
                terminal = terminal.terminal[edge[0]]
            elif terminal.type == 'jetty':
                terminal = terminal.terminal[vessel.index_jetty_position][edge[0]]
                vessel.access_terminal = terminal.request(priority=-1)
                terminal.release(vessel.waiting_time_in_anchorage)
                yield vessel.access_terminal

            vessel.available_sail_in_times = NetworkProperties.calculate_available_sail_in_times(vessel, True, True,True,route=vessel.route_after_anchorage,out=False)
            for t in range(len(vessel.available_sail_in_times)):
                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                    sailing_distance = 0
                    for nodes in enumerate(vessel.route_after_anchorage[:-1]):
                        _, _, distance = vessel.wgs84.inv(
                            vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0]]]['geometry'].x,
                            vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0]]]['geometry'].y,
                            vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0] + 1]]['geometry'].x,
                            vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0] + 1]]['geometry'].y)
                        sailing_distance += distance

                if vessel.available_sail_in_times[t][1] == 'Start':
                    if t == len(vessel.available_sail_in_times) - 1:
                        waiting_time = vessel.available_sail_in_times[t][0] - current_time
                        #if waiting_time >= vessel.metadata['max_waiting_time']:
                        #    vessel.return_to_sea = True
                        #    vessel.waiting_time = 0
                        #    terminal.release(vessel.access_terminal)
                        #else:
                        terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.available_sail_in_times[t][0] - new_current_time
                        vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                        terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60+ NetworkProperties.waiting_time_for_tidal_window(vessel,True,horizontal_tidal_window=True,max_waiting_time=True,out=True,route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                        yield vessel.env.timeout(vessel.available_sail_in_times[t][0] - new_current_time)
                        break
                    else:
                        continue

                elif new_current_time <= vessel.available_sail_in_times[t][0]:
                    if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                        if new_current_time < vessel.available_sail_in_times[t - 1][0]:
                            terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
                            vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b']*60 + vessel.metadata['t_l']*60
                            terminal.users[-1].etd = vessel_etd+vessel.metadata['t_b'] * 60+NetworkProperties.waiting_time_for_tidal_window(vessel, True, horizontal_tidal_window=True,max_waiting_time=True, out=True,route=vessel.route_after_terminal, delay=vessel_etd-vessel.env.now,plot=False)
                        else:
                            waiting_time = vessel.available_sail_in_times[t][0] - current_time
                            # if waiting_time >= vessel.metadata['max_waiting_time']:
                            #    vessel.return_to_sea = True
                            #    vessel.waiting_time = 0
                            #    terminal.release(vessel.access_terminal)
                            # else:
                            terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.available_sail_in_times[t][0] - new_current_time
                            vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                            terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60+NetworkProperties.waiting_time_for_tidal_window(vessel, True, horizontal_tidal_window=True, max_waiting_time=True, out=True,route=vessel.route_after_terminal, delay=vessel_etd - vessel.env.now, plot=False)
                            yield vessel.env.timeout(vessel.available_sail_in_times[t][0] - new_current_time)
                        break

                elif new_current_time >= vessel.available_sail_in_times[t][0]:
                    if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                        if t == len(vessel.available_sail_in_times) - 1 or new_current_time < vessel.available_sail_in_times[t + 1][0]:
                            terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
                            vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b']*60 + vessel.metadata['t_l']*60
                            terminal.users[-1].etd = vessel_etd+vessel.metadata['t_b'] * 60+NetworkProperties.waiting_time_for_tidal_window(vessel, True, horizontal_tidal_window=True,max_waiting_time=True, out=True,route=vessel.route_after_terminal, delay=vessel_etd-vessel.env.now,plot=False)
                            break

                # elif vessel.available_sail_in_times[t-1][0] - new_current_time > vessel.metadata['max_waiting_time']:
                #     if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                #         waiting_time = vessel.available_sail_in_times[t][0] - new_current_time
                #         terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.available_sail_in_times[t][0] - new_current_time
                #         vessel_etd = terminal.users[-1].eta + 2 * vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                #         terminal.users[-1].etd = vessel_etd + NetworkProperties.waiting_time_for_tidal_window(vessel,True,horizontal_tidal_window=True,max_waiting_time=True,out=True,route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                #         yield vessel.env.timeout(vessel.available_sail_in_times[t][0] - new_current_time)
                #         break

                elif t == len(vessel.available_sail_in_times) - 1:
                    waiting_time = vessel.available_sail_in_times[t][0] - current_time
                    # if waiting_time >= vessel.metadata['max_waiting_time']:
                    #    vessel.return_to_sea = True
                    #    vessel.waiting_time = 0
                    #    terminal.release(vessel.access_terminal)
                    # else:
                    terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.available_sail_in_times[t][0] - new_current_time
                    vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                    terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + NetworkProperties.waiting_time_for_tidal_window(vessel, True,horizontal_tidal_window=True,max_waiting_time=True,out=True,route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                    yield vessel.env.timeout(vessel.available_sail_in_times[t][0] - new_current_time)
                    break
                else:
                    continue

        elif vessel.return_to_sea == False and vessel.waiting_time:
            #new_current_time = vessel.env.now
            #if new_current_time - current_time >= vessel.metadata['max_waiting_time']:
            #    vessel.return_to_sea = True
            #    vessel.waiting_time = 0
            yield vessel.env.timeout(vessel.waiting_time)

        if vessel.return_to_sea == False:
            vessel.log_entry("Waiting in anchorage stop", vessel.env.now, vessel.env.now - current_time,
                             nx.get_node_attributes(vessel.env.FG, "geometry")[
                                 vessel.route[vessel.route.index(node)]], )
            vessel.route = vessel.route_after_anchorage
            PassSection.release_access_previous_section(vessel, vessel.route[0])
            yield from PassSection.request_access_next_section(vessel, vessel.route[0], vessel.route[1])
            vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
            vessel.env.process(vessel.move())

            anchorage.anchorage_area[node].release(vessel.anchorage_access)

            anchorage.log_entry("Vessel departure", vessel.env.now, len(anchorage.anchorage_area[node].users),
                                nx.get_node_attributes(vessel.env.FG, "geometry")[
                                    vessel.route[vessel.route.index(node)]], )
        else:
            if 'waiting_time_in_anchorage' in dir(vessel):
                vessel.waiting_time_in_anchorage.cancel()
            yield vessel.env.timeout(vessel.waiting_time)
            vessel.log_entry("Waiting in anchorage stop", vessel.env.now, vessel.env.now - current_time,
                             nx.get_node_attributes(vessel.env.FG, "geometry")[
                                 vessel.route[vessel.route.index(node)]], )
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[vessel.route.index(node)], vessel.true_origin)
            PassSection.release_access_previous_section(vessel, vessel.route[0])
            yield from PassSection.request_access_next_section(vessel, vessel.route[0], vessel.route[1])
            vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
            vessel.env.process(vessel.move())

            anchorage.anchorage_area[node].release(vessel.anchorage_access)
            anchorage.log_entry("Vessel departure", vessel.env.now, len(anchorage.anchorage_area[node].users),
                                nx.get_node_attributes(vessel.env.FG, "geometry")[
                                    vessel.route[vessel.route.index(node)]], )

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
        vessel.return_to_sea = False

        def checks_waiting_time_due_to_tidal_window(vessel, route, node, maximum_waiting_time = False):
            for nodes in enumerate(vessel.route[node:]):
                if nodes[1] == vessel.route[node]:
                    continue

                required_water_depth = vessel.T_f + vessel.metadata['ukc']
                minimum_water_depth = (np.min(vessel.env.FG.edges[vessel.route[node + nodes[0] - 1],
                                                                  vessel.route[node + nodes[0]]]['Info']['Water level'][1]) +
                                       vessel.env.FG.edges[vessel.route[node + nodes[0] - 1],
                                                           vessel.route[node + nodes[0]]]['Info']['Depth'][0])
                if required_water_depth > minimum_water_depth:
                    break

            vessel.waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel,required_water_depth > minimum_water_depth,route=route,plot=False)
            #if vessel.waiting_time >= vessel.metadata['max_waiting_time'] and maximum_waiting_time:
            #    vessel.return_to_sea = True
            #    vessel.waiting_time = 0
            #else:
            #    vessel.return_to_sea = False

        checks_waiting_time_due_to_tidal_window(vessel, route = vessel.route, node = node, maximum_waiting_time=True)
        if not vessel.return_to_sea:
            if terminal.type == 'jetty':
                minimum_waiting_time = []
                vessels_waiting = []
                for jetty in enumerate(terminal.terminal):
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
                    #checks_waiting_time_due_to_tidal_window(vessel, route = vessel.route_after_anchorage,node = node)
                    vessel.waiting_time_in_anchorage = terminal.length.get(vessel.L)

                elif terminal.type == 'jetty':
                    vessel.index_jetty_position = []
                    indices_empty_queue_for_jetty = [waiting_vessels[0] for waiting_vessels in enumerate(vessels_waiting) if waiting_vessels[1] == 0]
                    if indices_empty_queue_for_jetty != []:
                        min_minimum_waiting_time = minimum_waiting_time[indices_empty_queue_for_jetty[0]]  # vessel.env.now+vessel.metadata['max_waiting_time']
                        for index in indices_empty_queue_for_jetty:
                            if minimum_waiting_time[index] <= min_minimum_waiting_time:
                                min_minimum_waiting_time = minimum_waiting_time[index]
                                vessel.index_jetty_position = index
                    else:
                        vessel.index_jetty_position = np.argmin(vessels_waiting)
                    vessel.waiting_time_in_anchorage = terminal.terminal[vessel.index_jetty_position][edge[0]].request()

                yield from PassTerminal.move_to_anchorage(vessel, node)

            else:
                if terminal.type == 'quay':
                    PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)

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
                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + NetworkProperties.waiting_time_for_tidal_window(vessel, True,horizontal_tidal_window=True,max_waiting_time=True, out=True,route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)

        else:
            yield from PassTerminal.move_to_anchorage(vessel, node)

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
        yield vessel.env.timeout(vessel.metadata['t_b']*60)
        vessel.log_entry("Berthing stop", vessel.env.now, vessel.metadata['t_b']*60,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        for section in enumerate(vessel.env.FG.nodes[edge[1]]['Junction'][0].section):
            if list(section[1].keys())[0] == edge[1]:
                break

        if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
            PassSection.release_access_previous_section(vessel, edge[1])

        # Unloading
        vessel.log_entry("Unloading start", vessel.env.now, 0,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_l']*60/2)
        vessel.log_entry("Unloading stop", vessel.env.now, vessel.metadata['t_l']*60/2,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Loading
        vessel.log_entry("Loading start", vessel.env.now, 0,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_l']*60/2)
        vessel.log_entry("Loading stop", vessel.env.now, vessel.metadata['t_l']*60/2,
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

            required_water_depth = vessel.T_f + vessel.metadata['ukc']
            minimum_water_depth = (np.min(vessel.env.FG.edges[vessel.route[nodes[0] - 1],
                                                              vessel.route[nodes[0]]]['Info']['Water level'][1]) +
                                          vessel.env.FG.edges[vessel.route[nodes[0] - 1],
                                                              vessel.route[nodes[0]]]['Info']['Depth'][0])
            if required_water_depth > minimum_water_depth:
                break

        vessel.waiting_time = NetworkProperties.waiting_time_for_tidal_window(vessel,required_water_depth > minimum_water_depth,horizontal_tidal_window=True,max_waiting_time = False,route=vessel.route,out=True,plot=False)
        if vessel.waiting_time:
            vessel.log_entry("Waiting for tidal window start", vessel.env.now, 0,
                             shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
            yield vessel.env.timeout(vessel.waiting_time)

            vessel.log_entry("Waiting for tidal window stop", vessel.env.now, vessel.waiting_time,
                             shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Deberthing
        for section in enumerate(vessel.env.FG.nodes[edge[1]]['Junction'][0].section):
            if list(section[1].keys())[0] == edge[1]:
                break

        if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
            yield from PassSection.request_access_next_section(vessel, edge[1], edge[0])

        if 'Turning Basin' in vessel.env.FG.nodes[edge[0]].keys():
            turning_basin = vessel.env.FG.nodes[edge[0]]['Turning Basin'][0]
            vessel.request_access_turning_basin = turning_basin.turning_basin[edge[0]].request()
            yield vessel.request_access_turning_basin

        vessel.log_entry("Deberthing start", vessel.env.now, 0,
                         shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_b']*60)
        vessel.log_entry("Deberthing stop", vessel.env.now, vessel.metadata['t_b']*60,
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

