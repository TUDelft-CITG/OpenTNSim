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


class VesselProperties:
    """Mixin class: Something that has vessel properties
    vessel_type: can contain info on vessel type (avv class, cemt_class or other)
    width: vessel width
    length: vessel length
    height_empty: vessel height unloaded
    height_full: vessel height loaded
    draught_empty: draught unloaded
    draught_full: draught loaded
    Add information on possible restrictions to the vessels, i.e. height, width, etc.
    """

    def __init__(
        self,
        vessel_type,
        width,
        length,
        height_empty,
        height_full,
        draught_empty,
        draught_full,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        """Initialization"""
        self.vessel_type = vessel_type

        self.width = width
        self.length = length

        self.height_empty = height_empty
        self.height_full = height_full

        self.draught_empty = draught_empty
        self.draught_full = draught_full

    @property
    def current_height(self):
        """ Calculate current height based on filling degree """

        return (
            self.filling_degree * (self.height_full - self.height_empty)
            + self.height_empty
        )

    @property
    def current_draught(self):
        """ Calculate current draught based on filling degree """

        return (
            self.filling_degree * (self.draught_full - self.draught_empty)
            + self.draught_empty
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
        minWidth = minWidth if minWidth else 1.1 * self.width
        minHeight = minWidth if minHeight else 1.1 * self.current_height
        minDepth = minWidth if minDepth else 1.1 * self.current_draught

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
            except:
                raise ValueError(
                    "No path was found with the given boundary conditions."
                )

        # If not, return shortest path
        else:
            return nx.dijkstra_path(graph, origin, destination)


class HasEnergy:
    """Mixin class: Something that has energy usage.
    installed_power: installed engine power [kW]
    resistance: Rtot unloaded [N]
    resistance_empty: Rtot loaded [N]
    emissionfactor: emission factor [-]
    """

    def __init__(self, installed_power, resistance, resistance_empty, emissionfactor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """Initialization"""
        self.installed_power = installed_power
        self.resistance = resistance
        self.resistance_empty = resistance_empty
        self.emissionfactor = emissionfactor


class Routeable:
    """Mixin class: Something with a route (networkx format)
    route: a networkx path"""

    def __init__(self, route, complete_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.route = route
        self.complete_path = complete_path

class IsLockWaitingArea(HasResource, Identifiable, Log):
    """Mixin class: Something has lock object properties
    properties in meters
    operation in seconds
    """

    def __init__(
        self,
        node,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        """Initialization"""
        
        waiting_area_resources = 100
        self.waiting_area = {
            node: simpy.Resource(self.env, capacity=waiting_area_resources),
        }
        
        departure_resources = 2
        self.departure = {
            node: simpy.PriorityResource(self.env, capacity=departure_resources),
        }
        
class IsLockLineUpArea(HasResource, Identifiable, Log):
    """Mixin class: Something has lock object properties
    properties in meters
    operation in seconds
    """

    def __init__(
        self,
        node,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        """Initialization"""
 
        self.lock_queue_length = 0
        
        # Lay-Out
        self.line_up_area = {
            node: simpy.Resource(self.env, capacity=1),
        }
        
        departure_resources = 1
        self.departure = {
            node: simpy.Resource(self.env, capacity=departure_resources),
        }

class IsLock(HasResource, Identifiable, Log):
    """Mixin class: Something has lock object properties
    properties in meters
    operation in seconds
    """

    def __init__(
        self,
        node_1,
        node_2,
        node_3,
        lock_length,
        lock_width,
        lock_depth,
        doors_open,
        doors_close,
        operating_time,
        waiting_area=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        """Initialization"""
        
        # Properties
        self.lock_length = lock_length
        self.lock_width = lock_width
        self.lock_depth = lock_depth

        # Operating
        self.doors_open = doors_open
        self.doors_close = doors_close
        self.operating_time = operating_time

        # Water level
        assert node_1 != node_3

        self.node_1 = node_1
        self.node_3 = node_3
        self.water_level = random.choice([node_1, node_3])

    def convert_chamber(self, environment, new_level):
        """ Convert the water level """
        # Close the doors
        self.log_entry("Lock doors closing start", environment.now, self.water_level, 0)
        yield environment.timeout(self.doors_close)
        self.log_entry("Lock doors closing stop", environment.now, self.water_level, 0)

        # Convert the chamber
        self.log_entry(
            "Lock chamber converting start", environment.now, self.water_level, 0
        )
        
        # Water level will shift
        self.change_water_level(new_level)
        
        yield environment.timeout(self.operating_time)
        self.log_entry(
            "Lock chamber converting stop", environment.now, self.water_level, 0
        )

        # Open the doors
        self.log_entry("Lock doors opening start", environment.now, self.water_level, 0)
        yield environment.timeout(self.doors_open)
        self.log_entry("Lock doors opening stop", environment.now, self.water_level, 0)

    def change_water_level(self, side):
        """ Change water level and priorities in queue """

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


class Movable(Locatable, Routeable, Log):
    """Mixin class: Something can move
    Used for object that can move with a fixed speed
    geometry: point used to track its current location
    v: speed"""

    def __init__(self, v=1, *args, **kwargs):
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
        speed = self.v
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
                
            
            if "Waiting area" in self.env.FG.nodes[destination].keys():
                locks = self.env.FG.nodes[destination]["Waiting area"]
                for lock in locks:
                    lock 
                    loc = self.route.index(destination)
                    for r in self.route[loc:]:
                        if 'Line-up area' in self.env.FG.nodes[r].keys():                  
                            wait_for_waiting_area = self.env.now
                            access_waiting_area = lock.waiting_area[destination].request()
                            yield access_waiting_area 
                    
                            if wait_for_waiting_area != self.env.now:
                                waiting = self.env.now - wait_for_waiting_area
                                self.log_entry("Waiting to enter waiting area start", wait_for_waiting_area, 0, nx.get_node_attributes(self.env.FG, "geometry")[origin],)
                                self.log_entry("Waiting to enter waiting area stop", self.env.now, waiting, nx.get_node_attributes(self.env.FG, "geometry")[origin],)
                        
            if "Waiting area" in self.env.FG.nodes[origin].keys():   
                locks = self.env.FG.nodes[origin]["Waiting area"]
                for lock in locks:
                    if 'departure_area' in locals():
                        lock.departure[origin].release(departure_area)
                    else:
                        loc = self.route.index(origin)
                        for r in self.route[loc:]:
                            if 'Line-up area' in self.env.FG.nodes[r].keys():
                                locks2 = self.env.FG.nodes[r]["Line-up area"]
                                break
                        
                        for r2 in self.route[loc:]:
                            if 'Lock' in self.env.FG.nodes[r2].keys():
                                locks3 = self.env.FG.nodes[r2]["Lock"]
                                break
                        
                        self.lock_name = []
                        for lock3 in locks3:
                            if lock3.water_level == self.route[self.route.index(r2)-1]:
                                for lock2 in locks2:
                                    if lock2.name == lock3.name:
                                        if lock2.lock_queue_length == 0:
                                            self.lock_name = lock3.name
                                    break
                        
                        lock_queue_length = [];
                        if self.lock_name == []:
                            for lock2 in locks2:
                                lock_queue_length.append(lock2.lock_queue_length)
       
                            self.lock_name = locks2[lock_queue_length.index(min(lock_queue_length))].name
                            
                        for lock2 in locks2:
                            if lock2.name == self.lock_name:
                                lock2.lock_queue_length += 1
                        
                        for lock2 in locks2:
                            if lock2.name == self.lock_name:                         
                                self.v = 0.5*speed
                                break
                        
                        wait_for_lineup_area = self.env.now
                        lock.waiting_area[origin].release(access_waiting_area)
                        access_lineup_area = lock2.line_up_area[r].request()
                        yield access_lineup_area
                        access_departure_waiting_area = lock.departure[origin].request()
                        yield access_departure_waiting_area
            
                        if wait_for_lineup_area != self.env.now:
                            self.v = 0.25*speed
                            waiting = self.env.now - wait_for_lineup_area
                            self.log_entry("Waiting in waiting area start", wait_for_lineup_area, 0, nx.get_node_attributes(self.env.FG, "geometry")[origin])
                            self.log_entry("Waiting in waiting area stop", self.env.now, waiting, nx.get_node_attributes(self.env.FG, "geometry")[origin])  
            
                        lock.departure[origin].release(access_departure_waiting_area)
            
            if "Line-up area" in self.env.FG.nodes[origin].keys():   
                locks = self.env.FG.nodes[origin]["Line-up area"]
                for lock in locks:
                    if lock.name == self.lock_name:
                        if 'departure_lock' in locals():
                            lock.departure[origin].release(departure_lock)
                        else:
                            loc = self.route.index(origin)
                            for r in self.route[loc:]:
                                if 'Lock' in self.env.FG.nodes[r].keys():
                                    locks = self.env.FG.nodes[r]["Lock"]
                                    for lock2 in locks:
                                        if lock2.name == self.lock_name:
                                            self.v = 0.25*speed
                                            wait_for_lock_entry = self.env.now
                                            if lock2.resource.users != []:
                                                yield self.env.timeout(lock2.doors_close)     
                                                yield self.env.timeout(lock2.operating_time) 
                                             
                                            access_lock = lock2.resource.request(priority=-1 if self.route[self.route.index(r)-1] == lock2.water_level else 0)
                                            yield access_lock
                                            access_departure_lineup_area = lock.departure[origin].request()
                                            yield access_departure_lineup_area
                                            
                                            if self.route[self.route.index(r)-1] != lock2.water_level:
                                                yield from lock2.convert_chamber(self.env, self.route[self.route.index(r)-1])
                                            
                                            for r2 in self.route[(loc+1):]:
                                                if 'Line-up area' in self.env.FG.nodes[r2].keys():
                                                    locks = self.env.FG.nodes[r2]["Line-up area"]
                                                    for lock3 in locks:
                                                        if lock3.name == self.lock_name:
                                                            departure_lock = lock3.departure[r2].request()
                                                            yield departure_lock
                                                    
                                            for r3 in self.route[(loc+1):]:
                                                if 'Waiting area' in self.env.FG.nodes[r3].keys():
                                                    locks = self.env.FG.nodes[r3]["Waiting area"]
                                                    for lock4 in locks:
                                                        departure_area = lock4.departure[r3].request(priority=-1)
                                                        yield departure_area
                                            
                                            if wait_for_lock_entry != self.env.now:
                                                waiting = self.env.now - wait_for_lock_entry
                                                self.log_entry("Waiting in line-up area start", wait_for_lock_entry, 0, nx.get_node_attributes(self.env.FG, "geometry")[origin])
                                                self.log_entry("Waiting in line-up area stop", self.env.now, waiting, nx.get_node_attributes(self.env.FG, "geometry")[origin])  
                                            
                                            lock.line_up_area[origin].release(access_lineup_area)
                                            lock.departure[origin].release(access_departure_lineup_area) 

                                            for r4 in self.route[:(loc-1)]:
                                                if 'Line-up area' in self.env.FG.nodes[r4].keys():
                                                    locks = self.env.FG.nodes[r4]["Line-up area"]
                                                    for lock4 in locks:
                                                        if lock4.name == self.lock_name:
                                                            lock4.lock_queue_length -= 1                                             
                                
            if "Lock" in self.env.FG.nodes[origin].keys():
                locks = self.env.FG.nodes[origin]["Lock"] 
                for lock in locks:
                    if lock.name == self.lock_name:
                        self.log_entry("Passing lock start", self.env.now, 0, nx.get_node_attributes(self.env.FG, "geometry")[origin])
                        yield from lock.convert_chamber(self.env, destination)
            
                        lock.resource.release(access_lock)
                        passage_time = lock.doors_close + lock.operating_time + lock.doors_open
                        self.log_entry("Passing lock stop", self.env.now, passage_time, nx.get_node_attributes(self.env.FG, "geometry")[origin],)
                        yield from self.pass_edge(origin, destination)
                        self.v = speed
                    
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