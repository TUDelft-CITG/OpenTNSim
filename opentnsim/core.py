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
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
class IsTerminal(HasLength, Identifiable, Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
        self,
        node_start,
        node_end,
        length,
        *args,
        **kwargs
    ):
        super().__init__(length = length, remaining_length = length, *args, **kwargs)
        "Initialization"
        
        self.terminal = {
            node_start: simpy.PriorityResource(self.env, capacity = 100), #Only one ship can pass at a time: capacity = 1, request can have priority
        }

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
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
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
        
    def append_data_to_nodes(network,W,D,eta):
        assert len(network.nodes) == len(W[0]) == len(D[0]) == len(eta[0])
        
        for node in enumerate(network.nodes):
            network.nodes[node[1]]['Info'] = {'Width': [], 'Depth': [], 'Water level': [[],[]]}
            for n in range(len(W[0])):
                if (W[0][n].x,W[0][n].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                    network.nodes[node[1]]['Info']['Width'].append(W[1][n])
                if (D[0][n].x,D[0][n].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                    network.nodes[node[1]]['Info']['Depth'].append(D[1][n])
                if (eta[0][n].x,eta[0][n].y) == (network.nodes[node[1]]['geometry'].x,network.nodes[node[1]]['geometry'].y):
                    for t in range(len(eta[1][n][0])):
                        network.nodes[node[1]]['Info']['Water level'][0].append(eta[1][n][0][t])
                        network.nodes[node[1]]['Info']['Water level'][1].append(eta[1][n][1][t])
                                                                                
    def append_info_to_edges(network):        
        for edge in enumerate(network.edges):
            network.edges[edge[1]]['Info'] = {'Width': [], 'Depth': [], 'Water level': [[],[[],[]]]}
            network.edges[edge[1]]['Info']['Width'].append(np.min([network.nodes[edge[1][0]]['Info']['Width'][0],network.nodes[edge[1][1]]['Info']['Width'][0]]))
            if 'Terminal' in network.edges[edge[1]]:
                network.edges[edge[1]]['Info']['Depth'].append(np.max([network.nodes[edge[1][0]]['Info']['Depth'][0],network.nodes[edge[1][1]]['Info']['Depth'][0]]))
            else:
                network.edges[edge[1]]['Info']['Depth'].append(np.min([network.nodes[edge[1][0]]['Info']['Depth'][0],network.nodes[edge[1][1]]['Info']['Depth'][0]]))
            for t in range(len(network.nodes[edge[1][0]]['Info']['Water level'][0])):
                network.edges[edge[1]]['Info']['Water level'][0].append(network.nodes[edge[1][0]]['Info']['Water level'][0][t])
                network.edges[edge[1]]['Info']['Water level'][1][0].append(network.nodes[edge[1][0]]['Info']['Water level'][1][t])
                network.edges[edge[1]]['Info']['Water level'][1][1].append(network.nodes[edge[1][1]]['Info']['Water level'][1][t])
    
    def waiting_time_for_tidal_window(vessel,simulation_start):
        network = vessel.env.FG
        current_time = vessel.env.now
        route = vessel.route
        ukc = 0.5
        waiting_time = 0
        
        def minimum_water_per_edge_as_experienced_by_vessel(vessel):
            network = vessel.env.FG
            route = vessel.route
            min_wlev = [[] for _ in range(len(route)-1)]
            new_t = [[] for _ in range(len(route)-1)]
            distance_to_next_node = 0
            for nodes in enumerate(route):
                if nodes[1] == route[0]:
                    continue
        
                distance_to_next_node += pyproj.Geod(ellps='WGS84').inv(network.nodes[route[nodes[0]-1]]['geometry'].x,
                                                                        network.nodes[route[nodes[0]-1]]['geometry'].y,
                                                                        network.nodes[route[nodes[0]]]['geometry'].x,
                                                                        network.nodes[route[nodes[0]]]['geometry'].y)[2]
        
                t_wlev = network.edges[route[nodes[0]-1],route[nodes[0]]]['Info']['Water level'][0]
                sailing_time_to_next_node = distance_to_next_node/vessel.v
                wlev_node1 = network.edges[route[0],route[1]]['Info']['Water level'][1][0]
                wlev_node2 = network.edges[route[nodes[0]-1],route[nodes[0]]]['Info']['Water level'][1][1]
                interp_wlev_node1 = sc.interpolate.CubicSpline(t_wlev,wlev_node1)
                eta_next_node = [t-sailing_time_to_next_node for t in t_wlev]
                interp_wlev_node2 = sc.interpolate.CubicSpline(eta_next_node,wlev_node2)
                new_t[nodes[0]-1] = np.arange(0,eta_next_node[-1],100)

                for t in new_t[nodes[0]-1]:
                    min_wlev[nodes[0]-1].append(np.min([interp_wlev_node1(t),interp_wlev_node2(t)]))
                    
            return new_t,min_wlev
    
        new_t,min_wlev = minimum_water_per_edge_as_experienced_by_vessel(vessel)
        
        for nodes in enumerate(route):
            if route[0] == nodes[1]:
                continue
            
            time_edge_is_navigable = 0
            water_level_required = vessel.T_f + ukc
            depth_of_edge = network.edges[route[nodes[0]-1],route[nodes[0]]]['Info']['Depth'][0]
            time_after_simulation_start = current_time-simulation_start
            interp_water_level_at_edge = sc.interpolate.CubicSpline(new_t[nodes[0]-1],min_wlev[nodes[0]-1])
            root_interp_water_level_at_edge = sc.interpolate.CubicSpline(new_t[nodes[0]-1],[y-(water_level_required-depth_of_edge) for y in min_wlev[nodes[0]-1]])
            water_level_at_edge = interp_water_level_at_edge(time_after_simulation_start)
    
            if water_level_required > depth_of_edge+water_level_at_edge:
                times_edge_is_navigable = root_interp_water_level_at_edge.roots()
                for t in times_edge_is_navigable:
                    if t >= time_after_simulation_start:
                        time_edge_is_navigable = t-time_after_simulation_start
                        break
                        
            if time_edge_is_navigable > waiting_time:
                waiting_time = time_edge_is_navigable
                
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

class PassTerminal():
    @staticmethod
    def request_terminal_access(vessel, edge, node):
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        #if terminal.length.capacity > vessel.L or 'Anchorage' in vessel.env.FG.nodes[node].keys():

        def request_quay_length():
            get_quay_length = terminal.length.get(vessel.L)
            print(terminal.length.capacity,terminal.length.level)
            return get_quay_length

        yield request_quay_length()
        vessel.access_terminal = terminal.terminal[edge[0]].request()
        yield vessel.access_terminal
        sailing_distance = 0
        for nodes in enumerate(vessel.route[:-2]):
            _,_,distance = vessel.wgs84.inv(vessel.env.FG.nodes[vessel.route[nodes[0]]]['geometry'].x, 
                                            vessel.env.FG.nodes[vessel.route[nodes[0]]]['geometry'].y, 
                                            vessel.env.FG.nodes[vessel.route[nodes[0]+1]]['geometry'].x, 
                                            vessel.env.FG.nodes[vessel.route[nodes[0]+1]]['geometry'].y)
            sailing_distance += distance
        terminal.terminal[edge[0]].users[-1].eta = vessel.env.now + sailing_distance/vessel.v
        terminal.terminal[edge[0]].users[-1].etd = terminal.terminal[edge[0]].users[-1].eta + 30*60 + 4*3600 + 4*3600 + 10*60

        #else:
        #    vessel.get_quay_length = terminal.length.get(vessel.L)
        #    yield vessel.get_quay_length
        #    vessel.waiting_for_terminal = 'True'
    
    def berthing(vessel,node):
        vessel.log_entry("Berthing start", vessel.env.now, 0, 
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)-1]],)
        yield vessel.env.timeout(30*60)
        vessel.log_entry("Berthing stop", vessel.env.now, 30*60, 
                         nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)-1]],)
        
    def unloading(vessel,node):
        vessel.log_entry("Unloading start", vessel.env.now, 0, 
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)-1]],)
        yield vessel.env.timeout(4*3600)
        vessel.log_entry("Unloading stop", vessel.env.now, 4*3600, 
                         nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)-1]],)
    
    def loading(vessel,node):
        vessel.log_entry("Loading start", vessel.env.now, 0, 
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)-1]],)
        yield vessel.env.timeout(4*3600)
        vessel.log_entry("Loading stop", vessel.env.now, 4*3600, 
                         nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)-1]],)
    
    def deberthing(vessel,node):
        vessel.log_entry("Deberthing start", vessel.env.now, 0, 
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)-1]],)
        yield vessel.env.timeout(10*60)
        vessel.log_entry("Deberthing stop", vessel.env.now, 10*60, 
                         nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)-1]],)
        
        
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

    def move(self, simulation_start):
        """determine distance between origin and destination, and
        yield the time it takes to travel it
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].
        """
        self.distance = 0
        self.simulation_start = simulation_start.timestamp()

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
            
            #Request for a terminal
            if "Origin" in self.env.FG.nodes[origin] and 'leaving_port' not in dir(self):

                terminal = self.env.FG.edges['Node 2','Node 3']["Terminal"][0]
                self.log_entry("Waiting to access terminal start", self.env.now, 0,0)
                yield from PassTerminal.request_terminal_access(self, [self.route[-2], self.route[-1]], origin)
                self.log_entry("Waiting to access terminal stop", self.env.now, 0, 0)

            #TidalWindow
            if "Junction" in self.env.FG.nodes[destination] and 'leaving_port' not in dir(self) and 'accessing_terminal' not in dir(self):
                junction_type = self.env.FG.nodes[destination]['Junction'][0].type
        
                if junction_type == 'anchorage_access':        
                    for nodes in enumerate(self.route[node[0]:]):
                        if nodes[1] == self.route[node[0]]:
                            continue 
        
                        required_water_depth = self.T_f
                        minimum_water_depth = np.min((self.env.FG.edges[self.route[node[0]+nodes[0]-1],
                                                                                                self.route[node[0]+nodes[0]]]['Info']['Water level'][1])+
                                                                              self.env.FG.edges[self.route[node[0]+nodes[0]-1],
                                                                                                self.route[node[0]+nodes[0]]]['Info']['Depth'][0])
                        
                        if required_water_depth > minimum_water_depth:
                            waiting_time = NetworkProperties.waiting_time_for_tidal_window(self,self.simulation_start)
                            if waiting_time:
                                network = self.env.FG
                                yield from self.pass_edge(origin, destination)
                                for n in network.nodes:
                                    if 'Anchorage' in network.nodes[n]:    
                                        break
        
                                self.route_after_anchorage = []
                                self.true_origin = self.route[0]
                                for n2 in self.route[(nodes[0]-1):]:
                                    self.route_after_anchorage.append(n2)
                                self.route_after_anchorage.insert(0,n)
                                self.waiting_time_in_anchorage = waiting_time
                                self.route = nx.dijkstra_path(self.env.FG, self.route[self.route.index(destination)],n)
                                self.geometry = nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]
                                self.env.process(self.move(simulation_start))   
                                break
                        elif 'waiting_for_terminal' in dir(self):   
                            self.route_after_anchorage = []
                            self.true_origin = self.route[0]
                            for n2 in self.route[(nodes[0]-1):]:
                                self.route_after_anchorage.append(n2)
                            self.route_after_anchorage.insert(0,n)
                            self.waiting_time_in_anchorage = waiting_time
                            self.route = nx.dijkstra_path(self.env.FG, self.route[self.route.index(destination)],n)
                            self.geometry = nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]
                            self.env.process(self.move(simulation_start))   
                        
                    if 'waiting_time_in_anchorage' in dir(self):
                        break
                
            if 'Anchorage' in self.env.FG.nodes[destination].keys():
                yield from self.pass_edge(origin, destination)
                self.log_entry("Waiting in anchorage start", self.env.now, 0, nx.get_node_attributes(self.env.FG, "geometry")[node[1]],)
                yield self.env.timeout(self.waiting_time_in_anchorage)
                PassTerminal.request_terminal_access(self,[self.route[-2],self.route[-1]],origin)
                self.log_entry("Waiting in anchorage stop", self.env.now, self.waiting_time_in_anchorage, nx.get_node_attributes(self.env.FG, "geometry")[node[1]],)
                self.route = self.route_after_anchorage
                self.geometry = nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]
                self.env.process(self.move(simulation_start))
                self.accessing_terminal = 'True' #obsolete
                break
                        
            #Terminal
            if 'Terminal' in self.env.FG.edges[origin,destination].keys():
                yield from PassTerminal.berthing(self,destination)
                yield from PassTerminal.unloading(self,destination)
                yield from PassTerminal.loading(self,destination)
                yield from PassTerminal.deberthing(self,destination)
                terminal = self.env.FG.edges[origin,destination]["Terminal"][0]
                terminal.length.put(self.L)
                terminal.terminal[origin].release(self.access_terminal)
                if 'true_origin' in dir(self):
                    self.route = nx.dijkstra_path(self.env.FG, self.route[self.route.index(origin)], self.true_origin)
                else:
                    self.route = nx.dijkstra_path(self.env.FG, self.route[self.route.index(origin)], self.route[0])
                self.geometry = nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]
                self.env.process(self.move(simulation_start))
                self.leaving_port = 'True'
                break
            
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
