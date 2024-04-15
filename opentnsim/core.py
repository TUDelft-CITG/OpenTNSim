"""Main module."""

# package(s) related to time, space and id
import json
import logging
import uuid

# package(s) related to the simulation
import simpy
import networkx as nx
import numpy as np
import pandas as pd
import pytz

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
        self.env = env
        super().__init__(*args, **kwargs)

class IsDetectorNode:

    def __init__(
        self,
        infrastructure,
        *args,
        **kwargs
    ):

        self.infrastructure = infrastructure
        super().__init__(*args, **kwargs)

class HasLength(SimpyObject): #used by IsLock and IsLineUpArea to regulate number of vessels in each lock cycle and calculate repsective position in lock chamber/line-up area
    """Mixin class: Something with a storage capacity
    capacity: amount the container can hold
    level: amount the container holds initially
    total_requested: a counter that helps to prevent over requesting"""

    def __init__(self, length, remaining_length=0, total_requested=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.length = simpy.Container(self.env, capacity = length, init=remaining_length)

class HasCapacity(SimpyObject):

    def __init__(self, length, remaining_length=0, total_requested=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.length = simpy.Container(self.env, capacity=length, init=remaining_length)
        self.pos_length = simpy.Container(self.env, capacity=length, init=remaining_length)

class HasResource(SimpyObject):
    """Something that has a resource limitation, a resource request must be granted before the object can be used.

    nr_resources: nr of requests that can be handled simultaneously"""

    def __init__(self, capacity = 1, independent_resources = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        if not independent_resources:
            self.resource = simpy.PriorityResource(self.env, capacity=capacity)

        else:
            self.resource = {}
            for resource_name,capacity in independent_resources.items():
                self.resource[resource_name] = simpy.PriorityResource(self.env, capacity=capacity)

class HasType:
    """Mixin class: Something that has a name and id

    name: a name
    id: a unique id generated with uuid"""

    def __init__(self, type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.type = type

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

class Locatable:
    """Mixin class: Something with a geometry (geojson format)

    geometry: can be a point as well as a polygon"""

    def __init__(self, geometry, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.geometry = geometry
        self.node = None

class Log:
    """Mixin class: Something that has logging capability

    log: log message [format: 'start activity' or 'stop activity']
    t: timestamp
    value: a value can be logged as well
    geometry: value from locatable (lat, lon)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.log = {"Time": [], "Location": [], "Action": [], "Status": []}

    def log_entry(self, time, location, action, status_report):
        """Log"""
        self.log["Time"].append(pd.Timestamp(datetime.datetime.fromtimestamp(time)).to_datetime64())
        self.log["Location"].append(location)
        self.log["Action"].append(action)
        self.log["Status"].append(status_report)

    def get_log_as_json(self):
        json = []
        for time, location, action, status in zip(self.log["Time"],self.log["Location"],self.log["Action"],self.log["Status"]):
            json.append(dict(time=time, location=location, action=action, status=status))
        return json

class Neighbours:
    """Can be added to a locatable object (list)
    travel_to: list of locatables to which can be travelled"""

    def ___init(self, travel_to, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.neighbours = travel_to

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

class Routeable:
    """Mixin class: Something with a route (networkx format)

    route: a networkx path"""

    def __init__(self, route, complete_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.route = route
        self.complete_path = complete_path

class Movable(Locatable, Routeable, Log):
    """Mixin class: Something can move
    Used for object that can move with a fixed speed
    geometry: point used to track its current location"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.on_pass_node = []
        self.on_look_ahead_to_node = []
        self.on_pass_edge = []
        self.on_complete_pass_edge = []
        self.wgs84 = pyproj.Geod(ellps="WGS84")

    def move(self):
        """determine distance between origin and destination, and
        yield the time it takes to travel it
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].
        """
        for idx,method in enumerate(self.on_pass_node):
            if method.__str__().split('method ')[1].split(' of <')[0] == 'HasPortAccess.request_terminal_access':
                self.on_pass_node.insert(0, self.on_pass_node.pop(idx))
                break

        yield self.env.timeout((self.metadata['arrival_time'] - self.env.simulation_start).total_seconds())
        self.arrival_time = self.env.now
        self.metadata['arrival_time'] = self.env.simulation_start #resets delay

        self.distance = 0
        self.update_route_status_report()
        # Check if vessel is at correct location - if not, move to location
        if self.geometry != nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]:
            start_location = self.geometry
            end_location = nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]
            self.distance += self.wgs84.inv(start_location.x,start_location.y,end_location.x,end_location.y)[2]
            yield self.env.timeout(self.distance / self.v)
            self.log_entry(self.env.now,end_location,"Sailing to start",self.output.copy())

        # Move over the path and log every step
        for node in enumerate(self.route):
            self.node = node[1]
            if node[0] + 2 <= len(self.route):
                self.current_node = self.route[node[0]]
                self.next_node = self.route[node[0] + 1]
                origin = self.current_node
                destination = self.next_node
                k = sorted(self.env.FG[origin][destination],key=lambda x: self.env.FG[origin][destination][x]['geometry'].length)[0]
                status_report = self.update_sailing_status_report(origin, destination,(origin, destination, k))
                start_location = nx.get_node_attributes(self.env.FG, "geometry")[self.current_node]
                end_location = nx.get_node_attributes(self.env.FG, "geometry")[self.next_node]
                self.log_entry(self.env.now, start_location,"Sailing from node {} to node {} start".format(origin, destination),status_report)

                try:
                    yield from self.pass_node(self.current_node)
                except simpy.exceptions.Interrupt as e:
                    break

                try:
                    yield from self.pass_edge(self.current_node, self.next_node, start_location, end_location)
                except simpy.exceptions.Interrupt as e:
                    break

                try:
                    yield from self.complete_pass_edge(self.next_node)
                except simpy.exceptions.Interrupt as e:
                    break

                try:
                    yield from self.look_ahead_to_node(self.next_node)
                except simpy.exceptions.Interrupt as e:
                    break

                current_speed = self.env.vessel_traffic_service.provide_speed(self, (self.current_node, self.next_node))
                logger.debug("  distance: " + "%4.2f" % self.distance + " m")
                logger.debug("  sailing:  " + "%4.2f" % current_speed + " m/s")
                logger.debug("  duration: " + "%4.2f" % ((self.distance / current_speed) / 3600) + " hrs")

            if node[0] + 2 == len(self.route):
                break

        self.update_route_status_report(True)

    def look_ahead_to_node(self,destination):
        for gen in self.on_look_ahead_to_node:
            try:
                yield from gen(destination)
            except simpy.exceptions.Interrupt as e:
                logger.debug("Re-routing", exc_info=True)
                raise simpy.exceptions.Interrupt('Re-routing')

    def pass_node(self,origin):
        for gen in self.on_pass_node:
            try:
                yield from gen(origin)
            except simpy.exceptions.Interrupt as e:
                logger.debug("Re-routing", exc_info=True)
                raise simpy.exceptions.Interrupt('Re-routing')

    def pass_edge(self, origin, destination, start_location, end_location):
        k = sorted(self.env.FG[origin][destination], key=lambda x: self.env.FG[origin][destination][x]['geometry'].length)[0]
        self.distance = self.env.FG.edges[origin, destination, k]['length']

        next_node = None
        if self.route[-1] != destination:
            next_node = self.route[self.route.index(destination)+1]

        for gen in self.on_pass_edge:
            try:
                yield from gen(origin,destination)
            except simpy.exceptions.Interrupt as e:
                logger.debug("Re-routing", exc_info=True)
                raise simpy.exceptions.Interrupt('Re-routing')

        sailing = True
        while sailing:
            try:
                timeout = self.env.vessel_traffic_service.provide_sailing_time(self,[origin,destination],distance=self.distance)['Time'].sum()
                yield self.env.timeout(timeout)
            except simpy.exceptions.Interrupt as e:
                pass
            else:
                break

        if next_node:
            status_report = self.update_sailing_status_report(self.next_node,next_node,(self.current_node, self.next_node, k))
        else:
            status_report = self.update_sailing_status_report(self.next_node,self.next_node,(self.current_node, self.next_node, k))

        self.log_entry(self.env.now, end_location, "Sailing from node {} to node {} stop".format(self.current_node, self.next_node), status_report)
        self.geometry = end_location

    def complete_pass_edge(self,destination):
        for gen in self.on_complete_pass_edge:
            try:
                yield from gen(destination)
            except simpy.exceptions.Interrupt as e:
                logger.debug("Completed", exc_info=True)
                raise simpy.exceptions.Interrupt('Completed')