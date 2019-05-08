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

# spatial libraries
import pyproj
import shapely.geometry

# additional packages
import datetime, time

logger = logging.getLogger(__name__)

class SimpyObject:
    """General object which can be extended by any class requiring a simpy environment

    env: a simpy Environment
    """
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env


class HasResource(SimpyObject):
    """HasProcessingLimit class

    Adds a limited Simpy resource which should be requested before the object is used for processing."""

    def __init__(self, nr_resources=1, priority = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""

        self.resource = simpy.PriorityResource(self.env, capacity=nr_resources) if priority else simpy.Resource(self.env, capacity=nr_resources)


class Identifiable:
    """Something that has a name and id

    name: a name
    id: a unique id generated with uuid"""

    def __init__(self, name, id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.name = name
        # generate some id, in this case based on m
        self.id = id if id else str(uuid.uuid1())


class Locatable:
    """Something with a geometry (geojson format)

    geometry: can be a point as well as a polygon"""

    def __init__(self, geometry, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.geometry = geometry
        self.node = None


class HasContainer(SimpyObject):
    """Container class

    capacity: amount the container can hold
    level: amount the container holds initially
    container: a simpy object that can hold stuff"""

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
    """Log class

    log: log message [format: 'start activity' or 'stop activity']
    t: timestamp
    value: a value can be logged as well
    geometry: value from locatable (lat, lon)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.log = {"Message": [],
                    "Timestamp": [],
                    "Value": [],
                    "Geometry": []}

    def log_entry(self, log, t, value, geometry_log):
        """Log"""
        self.log["Message"].append(log)
        self.log["Timestamp"].append(datetime.datetime.fromtimestamp(t))
        self.log["Value"].append(value)
        self.log["Geometry"].append(geometry_log)

    def get_log_as_json(self):
        json = []
        for msg, t, value, geometry_log in zip(self.log["Message"], self.log["Timestamp"], self.log["Value"], self.log["Geometry"]):
            json.append(dict(message=msg, time=t, value=value, geometry_log=geometry_log))
        return json


class Routeable:
    """Something with a route (networkx format)
    route: a networkx path"""

    def __init__(self, route, complete_path = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.route = route
        self.complete_path = complete_path
        
class Person:
    """A person in the public transport network
    with a route, possible transfers and route 
    information."""  
    
    def __init__(self, route_info, transfers, transferstations, duration, lines, class_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.route_info = route_info
        self.transfers = transfers
        self.transferstations = transferstations
        self.duration = duration
        self.lines = lines
        self.class_id = class_id
    

class Movable(Locatable, Routeable, Log):
    """Movable class

    Used for object that can move with a fixed speed
    geometry: point used to track its current location
    v: speed"""

    def __init__(self, v=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.v = v
        self.wgs84 = pyproj.Geod(ellps='WGS84')

    def move(self):
        """determine distance between origin and destination, and
        yield the time it takes to travel it
        
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].
        """
        # Move over the path and log every step
        for node in enumerate(self.route):
            self.node = node[1]

            origin = self.route[node[0]]
            destination = self.route[node[0] + 1]
            edge = self.env.FG.edges[origin, destination]
            
            yield from self.pass_edge(origin, destination)

            if node[0] + 2 == len(self.route):
                break
    
    def pass_edge(self, origin, destination):
        edge = self.env.FG.edges[origin, destination]
        orig = nx.get_node_attributes(self.env.FG, "geometry")[origin]
        dest = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        # Check for (un)load
        try:
            node_type = nx.get_node_attributes(self.env.FG, "object_type")[origin]
            to_load = []

            if isinstance(node_type, Station) and isinstance(self, Mover):
                if len(node_type.units) > 0:
                    for unit in node_type.units:
                        if unit.lines[0] == self.name:
                            if unit.transfers > 0 and unit.transferstations[0] in self.route[self.route.index(origin):]:
                                to_load.append(unit)

                            elif unit.route[-1] in self.route[self.route.index(origin):]: 
                                to_load.append(unit)
                
                for unit in to_load:
                    node_type.units.remove(unit)
                
                if len(to_load) > 0:
                    self.load(to_load)

        except:           
            pass

        self.log_entry("Driving from node {} to node {} start".format(origin, destination), self.env.now, 0, orig)
        yield self.env.timeout(edge["duration"] * 60)
        self.log_entry("Driving from node {} to node {} stop".format(origin, destination), self.env.now, 0, dest)
        self.geometry = dest
        
        try:
            node_type = nx.get_node_attributes(self.env.FG, "object_type")[destination]

            if isinstance(node_type, Station) and isinstance(self, Mover):
                self.unload()
                yield self.env.timeout(15)

        
        except:
            pass

        
class Mover():
    """ 
    Mover class 

    Used to move objects from one location to another
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""

        self.units = []
    
    def load(self, units):
        """ Load self """
        
        self.log_entry("Loading start", self.env.now, 0, self.geometry)

        for unit in units:
            self.units.append(unit)
            unit.log_entry("Waiting for {} stop".format(unit.lines[0]), self.env.now, 0, self.geometry)
            unit.log_entry("In {} start".format(unit.lines[0]), self.env.now, 0, self.geometry)
        
        self.log_entry("Loading stop", self.env.now, 30, self.geometry)

    
    def unload(self):
        """ Unload self """

        self.log_entry("Unloading start", self.env.now, 0, self.geometry)
        
        to_remove = []
        to_transfer = []
        
        for unit in self.units:
            if nx.get_node_attributes(self.env.FG, "geometry")[unit.route[-1]] == self.geometry:
                unit.log_entry("In {} stop".format(unit.lines[0]), self.env.now, 0, self.geometry)
                to_remove.append(unit)
            
            elif unit.transfers > 0:
                if nx.get_node_attributes(self.env.FG, "geometry")[unit.transferstations[0]] == self.geometry:
                    unit.log_entry("In {} stop".format(unit.lines[0]), self.env.now, 0, self.geometry)
                    unit.log_entry("Transfer to {}".format(unit.lines[1]), self.env.now, 0, self.geometry)
                    to_transfer.append(unit)
                
        for unit in to_remove:
            self.units.remove(unit)
            
        for unit in to_transfer:
            # Set unit to the transfernode
            transfernode = unit.transferstations[0]
            self.env.FG.nodes[transfernode]["object_type"].units.append(unit)
                            
            # Update remaining route
            unit.transfers -= 1
            unit.transferstations.pop(0)
            unit.lines.pop(0)
            
            # Remove from transport
            self.units.remove(unit)            
            
        self.env.timeout(30)
        self.log_entry("Unloading stop", self.env.now, 30, self.geometry)

class Station(HasContainer):
    """ Station class """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""

        self.units = []