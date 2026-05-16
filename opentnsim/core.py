"""Main module."""

# package(s) related to time, space and id
import datetime
import logging
import random
import uuid
import warnings
from typing import Union

import deprecated
import networkx as nx
import numpy as np

# spatial libraries
import pyproj
import shapely
import shapely.geometry
import shapely.ops

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy

# Use OpenCLSim objects for core objects
from openclsim.core import Identifiable, Locatable, SimpyObject, Log

import opentnsim.energy
import opentnsim.graph as graph_module


# additional packages
import json
import pandas as pd
import pytz


logger = logging.getLogger(__name__)


Geometry = shapely.Geometry


class HasResource(SimpyObject):
    """Something that has a resource limitation, a resource request must be granted before the object can be used.

    nr_resources: nr of requests that can be handled simultaneously"""

    def __init__(self, capacity = 1, parallel_resources = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        if not parallel_resources:
            self.resource = simpy.PriorityResource(self.env, capacity=capacity)

        else:
            self.resource = {}
            for resource_name, capacity in parallel_resources.items():
                self.resource[resource_name] = simpy.PriorityResource(self.env, capacity=capacity)


class Neighbours:
    """Can be added to a locatable object (list)

    - travel_to: list of locatables to which can be travelled
    """

    def __init__(self, travel_to, *args, **kwargs):
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

    @property
    def max_load(self):
        """return the maximum cargo to load"""
        # independent of trip
        return self.container.capacity - self.container.level


class HasLoad:
    """Mixin class with load dependent height (H) and draught (T). The filling
    degree (filling_degree: fraction) will interpolate between empty and full
    height and draught."""

    def __init__(self, H_e, H_f, T_e, T_f, filling_degree=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.H_e = H_e
        self.H_f = H_f
        self.T_e = T_e
        self.T_f = T_f
        self.filling_degree = filling_degree

    @property
    def T(self):
        # base draught on filling degree
        T = self.filling_degree * (self.T_f - self.T_e) + self.T_e
        return T

    @property
    def H(self):
        """Calculate current height based on filling degree"""

        return self.filling_degree * (self.H_f - self.H_e) + self.H_e


class Routable(SimpyObject):
    """Mixin class: Something with a route (networkx format)

    - route: list of node-IDs
    - position_on_route: index of position
    """

    def __init__(self, origin, destination, next_destinations=[], *args, **kwargs):
        """Initialization"""
        env = kwargs.get("env")
        super().__init__(*args, **kwargs)
        self.origin = origin
        self.destination = destination
        self.next_destinations = next_destinations
        self.route = nx.dijkstra_path(env.FG, origin, self.destination)
        self.position_on_route = 0

    @property
    def graph(self):
        """
        Return the graph of the underlying environment.

        If it's multigraph cast to corresponding type
        If you want the multidigraph use the HasMultiGraph mixin

        """
        graph = None
        if hasattr(self.env, "graph"):
            graph = self.env.graph
        elif hasattr(self.env, "FG"):
            graph = self.env.FG
        else:
            raise ValueError("Routable expects .graph to be present on env")

        if isinstance(graph, nx.MultiDiGraph):
            return nx.DiGraph(graph)
        elif isinstance(graph, nx.MultiGraph):
            return nx.Graph(graph)
        return graph



class WithCurrent:

    def get_edge_current(self, edge):

        if hasattr(self.env, "get_current"):
            u = edge.get("u", None)
            v = edge.get("v", None)
            if u is not None and v is not None:
                return float(self.env.get_current(u, v, self.env.now))

        return float(edge.get("current_ms", 0.0))


    def compute_speeds_on_edge(self, edge):
        """
          v_w = speed through water
          v_c = speed with current
          v_g = speed over ground
        """
        v_g = self.v
        v_c = self.get_edge_current(edge)
        v_w = v_g - v_c

        self.v_w = v_w
        self.v_c = v_c
        self.v_g = v_g

        return v_w, v_c, v_g



    

class Movable(WithCurrent, Locatable, Routable, Log):
    """Mixin class: Something can move.

    Used for object that can move with a fixed speed

    - geometry: point used to track its current location
    - v: speed
    - on_pass_edge_functions can contain a list of generators in the form of on_pass_edge(source: Point, destination: Point) -> yield event
    - on_pass_node_functions can contain a list of generators in the form of on_pass_node(source: Point) -> yield event
    """

    def __init__(self, env, origin, destination, *args, **kwargs):
        geometry = env.FG.nodes[origin]['geometry']
        super().__init__(origin=origin, destination=destination, geometry=geometry, env=env, *args, **kwargs)
        """Initialization"""
        self.on_pass_node_functions = []
        self.on_pass_edge_functions = []
        self.on_complete_pass_edge_functions = []
        self.on_look_ahead_to_node_functions = []
        self.on_pass_node = self.on_pass_node_functions
        self.on_pass_edge = self.on_pass_edge_functions
        self.on_complete_pass_edge = self.on_complete_pass_edge_functions
        self.on_look_ahead_to_node = self.on_look_ahead_to_node_functions

        self.wgs84 = pyproj.Geod(ellps="WGS84")
        
        self.v_c = 0.0   # current speed (m/s)
        self.v_g = 0.0     # speed over ground (m/s)

    def move(self):
        """determine distance between origin and destination, and
        yield the time it takes to travel it
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].
        """

        # time-out if arrival time lies in future
        if hasattr(self, "metadata") and "arrival_time" in self.metadata \
            and hasattr(self.env, "simulation_start"):
                delay = (self.metadata['arrival_time'] - self.env.simulation_start).total_seconds()
                if delay > 0:
                    yield self.env.timeout(delay)
                self.arrival_time = self.env.now
                self.metadata['arrival_time'] = self.env.simulation_start
                
        # default distance to next node
        self.distance = 0

        for idx,method in enumerate(self.on_pass_node_functions):
            if method.__str__().split('method ')[1].split(' of <')[0] == 'HasPortAccess.request_terminal_access':
                self.on_pass_node.insert(0, self.on_pass_node.pop(idx))
                break

        self.update_route_status_report()

        # Check if vessel is at correct location - if not, move to location
        vessel_origin_location = nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]
        if self.geometry != vessel_origin_location:
            start_location = self.geometry
            logger.debug("Origin: {orig}")
            logger.debug("Destination: {dest}")

            self.distance += self.wgs84.inv(start_location.x,
                                            start_location.y,
                                            vessel_origin_location.x,
                                            vessel_origin_location.y
                                            )[2]

            v_w = getattr(self, "v", 0.0)
            v_g = max(v_w, 1e-6)
            self.v_w = v_w
            self.v_c = 0.0
            self.v_g = v_g


            yield self.env.timeout(self.distance / v_g)
            self.log_entry("Sailing to start", self.env.now, self.output.copy(), vessel_origin_location)

        # Move over the path and log every step
        for index, edge in enumerate(zip(self.route[:-1], self.route[1:])):
            self.current_node, self.next_node = edge # origin and destination
            start_location = nx.get_node_attributes(self.env.FG, "geometry")[self.current_node]
            end_location = nx.get_node_attributes(self.env.FG, "geometry")[self.next_node]

            # It is important for the locking module that the message of sailing should be before passing the first node in preparation of the actual sailing
            k = sorted(self.multidigraph[self.current_node][self.next_node],key=lambda x: self.multidigraph[self.current_node][self.next_node][x]['geometry'].length)[0]
            status_report = self.update_sailing_status_report(self.current_node, self.next_node, (self.current_node, self.next_node, k))
            self.log_entry("Sailing from node {} to node {} start".format(self.current_node, self.next_node),
                           self.env.now, status_report, start_location)

            yield from self.pass_node(self.current_node)

            # update to current position
            self.geometry = nx.get_node_attributes(self.graph, "geometry")[self.current_node]
            self.position_on_route = index

            # are we already at destination?
            if self.next_node == self.current_node:
                break

            yield from self.pass_edge(self.current_node, self.next_node, end_location)
            yield from self.complete_pass_edge(self.next_node)

            # we arrived at destination
            # update to new position
            self.geometry = nx.get_node_attributes(self.graph, "geometry")[self.next_node]
            self.current_node = self.next_node
            self.position_on_route = index + 1

            yield from self.look_ahead_to_node(self.next_node)

        logger.debug("  distance: " + "%4.2f" % self.distance + " m")
        if self.current_speed is not None:
            v_g = getattr(self, "v_g", self.current_speed)
            logger.debug("  sailing:  " + "%4.2f" % self.current_speed + " m/s")
            logger.debug("  v_ground: " + "%4.2f" % v_g + " m/s")
            logger.debug("  duration: " + "%4.2f" % ((self.distance / v_g) / 3600) + " hrs")
        else:
            logger.debug("  current_speed:  not set")
        self.update_route_status_report(True)

    def pass_node(self, node):
        # call all on_pass_node_functions
        for on_pass_node_function in self.on_pass_node_functions:
            yield from on_pass_node_function(node)

##################################################################
    def pass_edge(self, origin, destination, end_location):
        edge = self.graph.edges[origin, destination]
        orig = nx.get_node_attributes(self.graph, "geometry")[origin]
        k = sorted(self.multidigraph[origin][destination], key=lambda x: self.multidigraph[origin][destination][x]['geometry'].length)[0]
        
        md = self.multidigraph.edges[origin, destination, k]
        
        if 'length_m' in md and md['length_m'] is not None:
            self.distance = md['length_m']
        
        elif 'geometry' in md and md['geometry'] is not None:
            d = 0.0
            coords = list(md['geometry'].coords)
            for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
                d += self.wgs84.inv(x1, y1, x2, y2)[2]
            self.distance = d

        elif 'length' in md and md['length'] is not None:
            self.distance = md['length']

        else:
            raise ValueError("No distance found for this edge")

####################################################################
        next_node = None
        if self.route[-1] != destination:
            next_node = self.route[self.route.index(destination)+1]

        for on_pass_edge_function in self.on_pass_edge_functions:
            yield from on_pass_edge_function(origin, destination)

        # remember when we arrived at the edge
        arrival = self.env.now

        # This is the case if we are sailing on power
        value = 0
        if getattr(self, "P_tot_given", None) is not None:
            edge = self.graph.edges[origin, destination]
            depth = self.graph.get_edge_data(origin, destination)["Info"]["GeneralDepth"]



            # You can input more power than is realistic
            # There are two mechanisms that reduce the power given:
            # 1. The grounding speed:
            (
                upperbound,
                selected,
                results_df,
            ) = opentnsim.strategy.get_upperbound_for_power2v(self, width=150, depth=depth, margin=0)


            # Here the upperbound is used to estimate the actual velocity
            self.v = self.power2v(self, edge, upperbound)
            power_used = self.P_tot_given
            
            # store upperbound velocity
            # TODO: remove these three fields after debugging
            self.selected = selected
            self.results_df = results_df
            self.upperbound = upperbound
            # use upperbound power (used to compute the sailing speed)
            value = power_used

        # Maximum speed restriction may be limiting the on power speed
        if 'vessel_traffic_service' in dir(self.env):
            e2 = dict(edge)
            e2["u"] = origin
            e2["v"] = destination
            self.v = self.env.vessel_traffic_service.provide_speed(self, e2)


        # Wait for edge resources to become available
        if "Resources" in edge.keys():
            with self.graph.edges[origin, destination]["Resources"].request() as request:
                yield request
                # we had to wait, log it
                if arrival != self.env.now:
                    self.log_entry(
                        "Waiting to pass edge {} - {} start".format(origin, destination),
                        arrival,
                        value,
                        orig,
                    )
                    self.log_entry(
                        "Waiting to pass edge {} - {} stop".format(origin, destination),
                        self.env.now,
                        value,
                        orig,
                    )

        # default velocity based on current speed.
        e = dict(edge)
        e["u"] = origin
        e["v"] = destination
        v_w, v_c, v_g = self.compute_speeds_on_edge(e)
        #v_w, v_c, v_g = self.compute_speeds_on_edge(edge)
        timeout = self.distance / v_g
        yield self.env.timeout(timeout)
        if next_node:
            status_report = self.update_sailing_status_report(self.next_node,next_node,(self.current_node, self.next_node, k))
        else:
            status_report = self.update_sailing_status_report(self.next_node,self.next_node,(self.current_node, self.next_node, k))
        self.log_entry("Sailing from node {} to node {} stop".format(self.current_node, self.next_node), self.env.now, status_report, end_location)
        self.geometry = end_location

    def complete_pass_edge(self,destination):
        for gen in self.on_complete_pass_edge_functions:
            try:
                yield from gen(destination)
            except simpy.exceptions.Interrupt as e:
                logger.debug("Completed", exc_info=True)
                raise simpy.exceptions.Interrupt('Completed')

    def look_ahead_to_node(self,destination):
        for gen in self.on_look_ahead_to_node_functions:
            try:
                yield from gen(destination)
            except simpy.exceptions.Interrupt as e:
                logger.debug("Re-routing", exc_info=True)
                raise simpy.exceptions.Interrupt('Re-routing')

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


class HasCapacity(SimpyObject):

    def __init__(self, length, remaining_length=0, total_requested=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.length = simpy.Container(self.env, capacity=length, init=remaining_length)
        self.pos_length = simpy.Container(self.env, capacity=length, init=remaining_length)


class ExtraMetadata:
    """store all leftover keyword arguments as metadata property (use as last mixin)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        # store all other properties as metadata
        self.metadata = kwargs


class HasType:
    """Mixin class: Something that has a type (such as a vessel may be a container vessel, tanker, etc.

    type: string"""

    def __init__(self, type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.type = type
