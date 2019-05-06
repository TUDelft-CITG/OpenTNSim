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


class Neighbours:
    """Can be added to a locatable object (list)
    
    travel_to: list of locatables to which can be travelled"""

    def ___init(self, travel_to, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.neighbours = travel_to


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

class VesselProperties:
    """
    Add information on possible restrictions to the vessels.
    Height, width, etc.
    """

    def __init__(self, vessel_type, installed_power,
                 width, length,
                 height_empty, height_full, 
                 draught_empty, draught_full, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        """Initialization"""
        self.vessel_type = vessel_type
        self.installed_power = installed_power

        self.width = width
        self.length = length

        self.height_empty = height_empty
        self.height_full = height_full

        self.draught_empty = draught_empty
        self.draught_full = draught_full

    @property
    def current_height(self):
        """ Calculate current height based on filling degree """

        return self.filling_degree * (self.height_full - self.height_empty) + self.height_empty

    @property
    def current_draught(self):
        """ Calculate current draught based on filling degree """

        return self.filling_degree * (self.draught_full - self.draught_empty) + self.draught_empty
    
    def get_route(self, origin, destination,
                  graph = None,
                  minWidth = None, 
                  minHeight = None, 
                  minDepth = None,
                  randomSeed = 4):
        """ Calculate a path based on vessel restrictions """

        graph = graph if graph else self.env.FG
        minWidth = minWidth if minWidth else 1.1 * self.width
        minHeight = minWidth if minHeight else 1.1 * self.current_height
        minDepth = minWidth if minDepth else 1.1 * self.current_draught

        # Check if information on restrictions is added to the edges
        random.seed(randomSeed)
        edge = random.choice(list(graph.edges(data = True)))
        edge_attrs = list(edge[2].keys())
        

        # IMPROVE THIS TO CHECK ALL EDGES AND COMBINATIONS OF RESTRICTIONS

        if all(item in edge_attrs for item in ["Width", "Height", "Depth"]):
            edges = []
            nodes = []

            for edge in graph.edges(data = True):
                if edge[2]["Width"] >= minWidth and edge[2]["Height"] >= minHeight and edge[2]["Depth"] >= minDepth:
                    edges.append(edge)
                    
                    nodes.append(graph.nodes[edge[0]])
                    nodes.append(graph.nodes[edge[1]])

            subGraph = graph.__class__()

            for node in nodes:
                subGraph.add_node(node["name"],
                        name = node["name"],
                        geometry = node["geometry"], 
                        position = (node["geometry"].x, node["geometry"].y))

            for edge in edges:
                subGraph.add_edge(edge[0], edge[1], attr_dict = edge[2])

            try:
                return nx.dijkstra_path(subGraph, origin, destination)
            except:
                raise ValueError("No path was found with the given boundary conditions.")

        # If not, return shortest path
        else:
            return nx.dijkstra_path(graph, origin, destination)
            

class HasEnergy:
    """
    Add information on energy use and effects on energy use.
    """

    def __init__(self, emissionfactor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        """Initialization"""
        self.emissionfactor = emissionfactor
        self.energy_use = {'total': 0,'stationary': 0}
        self.co2_footprint = {'total_footprint': 0,'stationary': 0}
        self.mki_footprint = {'total_footprint': 0,'stationary': 0}
    
    @property
    def power(self):
        return 2 * (self.current_speed * self.resistance * 10**-3)  #kW
    
    def calculate_energy_consumption(self):
        """Calculation of energy consumption based on total time in system and properties"""

        stationary_phase_indicator = ['Doors closing stop', 'Converting chamber stop', 'Doors opening stop', 'aiting to pass lock stop']
        
        times = self.log['Timestamp']
        messages = self.log['Message']
        
        for i in range(len(times) - 1):
            delta_t = times[i+1] - times[i]
            
            if messages[i + 1] in stationary_phase_indicator:
                energy_delta =  self.power *  delta_t / 3600 # KJ/3600

                self.energy_use['total_energy'] += energy_delta * 0.15
                self.energy_use['stationary'] += energy_delta * 0.15
            
            else:
                self.energy_use['total_energy'] += self.power * delta_t / 3600


class Routeable:
    """Something with a route (networkx format)
    route: a networkx path"""

    def __init__(self, route, transfers = None, route_info = None, transferstations = None, duration = None, lines = None, complete_path = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.route = route
        self.complete_path = complete_path
        self.route_info = route_info
        self.transfers = transfers
        self.transferstations = transferstations
        self.duration = duration
        self.lines = lines

class IsLock(HasResource, Identifiable, Log):
    """
    Create a lock object

    properties in meters
    operation in seconds
    """

    def __init__(self, node_1, node_2, lock_length, lock_width, lock_depth,
                 doors_open, doors_close, operating_time, waiting_area = True, *args, **kwargs):
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
        assert node_1 != node_2

        self.node_1 = node_1
        self.node_2 = node_2
        self.water_level = random.choice([node_1, node_2])

        # Lay-Out
        self.line_up_area = {node_1: simpy.Resource(self.env, capacity = 1), node_2: simpy.Resource(self.env, capacity = 1)}
        
        waiting_area_resources = 1 if waiting_area else 100
        self.waiting_area = {node_1: simpy.Resource(self.env, capacity = waiting_area_resources), 
                             node_2: simpy.Resource(self.env, capacity = waiting_area_resources)} 
    
    def convert_chamber(self, environment, new_level):
        """ Convert the water level """

        # Close the doors
        self.log_entry("Lock doors closing start", environment.now, self.water_level, 0)
        yield environment.timeout(self.doors_close)
        self.log_entry("Lock doors closing stop", environment.now, self.water_level, 0)

        # Convert the chamber
        self.log_entry("Lock chamber converting start", environment.now, self.water_level, 0)

        # Water level will shift
        self.change_water_level(new_level)

        yield environment.timeout(self.operating_time)
        self.log_entry("Lock chamber converting stop", environment.now, self.water_level, 0)
        
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
                self.resource.queue.insert(0, self.resource.queue.pop(self.resource.queue.index(request)))
            else:
                self.resource.queue.insert(-1, self.resource.queue.pop(self.resource.queue.index(request)))

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
        self.distance = 0

        # Check if vessel is at correct location - if note, move to location
        if self.geometry != nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]:
            orig = self.geometry
            dest = nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]

            self.distance += self.wgs84.inv(shapely.geometry.asShape(orig).x, shapely.geometry.asShape(orig).y, 
                                            shapely.geometry.asShape(dest).x, shapely.geometry.asShape(dest).y)[2]
    
            yield self.env.timeout(self.distance / self.current_speed)
            self.log_entry("Driving to start", self.env.now, self.distance, dest)

        # Move over the path and log every step
        for node in enumerate(self.route):
            self.node = node[1]

            origin = self.route[node[0]]
            destination = self.route[node[0] + 1]
            edge = self.env.FG.edges[origin, destination]

            if "Lock" in edge.keys():
                queue_length = []
                lock_ids = []

                for lock in edge["Lock"]:
                    queue = 0
                    queue += len(lock.resource.users)
                    queue += len(lock.line_up_area[self.node].users)
                    queue += len(lock.waiting_area[self.node].users) + len(lock.waiting_area[self.node].queue)

                    queue_length.append(queue)
                    lock_ids.append(lock.id)
                
                lock_id = lock_ids[queue_length.index(min(queue_length))]
                yield from self.pass_lock(origin, destination, lock_id)
            
            else:
                yield from self.pass_edge(origin, destination)

            if node[0] + 2 == len(self.route):
                break

        self.geometry = nx.get_node_attributes(self.env.FG, "geometry")[destination]
        logger.debug('  distance: ' + '%4.2f' % self.distance + ' m')
        logger.debug('  driving:  ' + '%4.2f' % self.current_speed + ' m/s')
        logger.debug('  duration: ' + '%4.2f' % ((self.distance / self.current_speed) / 3600) + ' hrs')
    
    def pass_edge(self, origin, destination):
        edge = self.env.FG.edges[origin, destination]
        orig = nx.get_node_attributes(self.env.FG, "geometry")[origin]
        dest = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        distance = self.wgs84.inv(shapely.geometry.asShape(orig).x, shapely.geometry.asShape(orig).y, 
                                  shapely.geometry.asShape(dest).x, shapely.geometry.asShape(dest).y)[2]

        self.distance += distance
        arrival = self.env.now
                        
        # Check for (un)load
        try:
            node_type = nx.get_node_attributes(self.env.FG, "object_type")[origin]
            to_load = []

            if isinstance(node_type, Station) and isinstance(self, Mover):
                if len(node_type.units) > 0:
                    for unit in node_type.units:
                        if unit.route[-1] in nx.dijkstra_path(self.env.FG, origin, self.route[-1]):
                            to_load.append(unit)
                            node_type.units.remove(unit)
                
                if len(to_load) > 0:
                    self.load(to_load)

        except:
            pass
        
        # Act based on resources
        if "Resources" in edge.keys():
            with self.env.FG.edges[origin, destination]["Resources"].request() as request:
                yield request

                if arrival != self.env.now:
                    self.log_entry("Waiting to pass edge {} - {} start".format(origin, destination), arrival, 0, orig)
                    self.log_entry("Waiting to pass edge {} - {} stop".format(origin, destination), self.env.now, 0, orig)  

                self.log_entry("Driving from {} to {} start".format(origin, destination), self.env.now, 0, orig)
                yield self.env.timeout(edge["duration"] * 60)
                self.log_entry("Driving from {} to {} stop".format(origin, destination), self.env.now, 0, dest)
        
        else:
            self.log_entry("Driving from {} to {} start".format(origin, destination), self.env.now, 0, orig)
            self.log_entry("Passengers: {}".format(len(self.units)), self.env.now, 0, self.geometry)
            yield self.env.timeout(edge["duration"] * 60)
            self.log_entry("Driving from {} to {} stop".format(origin, destination), self.env.now, 0, dest)
            self.geometry = dest
        
        try:
            node_type = nx.get_node_attributes(self.env.FG, "object_type")[destination]

            if isinstance(node_type, Station) and isinstance(self, Mover):
                self.unload()
                yield self.env.timeout(15)
        
        except:
            pass
        

    def pass_lock(self, origin, destination, lock_id):
        """Pass the lock"""

        locks = self.env.FG.edges[origin, destination]["Lock"]
        for lock in locks:
            if lock.id == lock_id:
                break
        
        # Request access to waiting area
        wait_for_waiting_area = self.env.now

        access_waiting_area = lock.waiting_area[origin].request()
        yield access_waiting_area

        if wait_for_waiting_area != self.env.now:
            waiting = self.env.now - wait_for_waiting_area
            self.log_entry("Waiting to enter waiting area start", wait_for_waiting_area, 0, self.geometry)
            self.log_entry("Waiting to enter waiting area stop", self.env.now, waiting, self.geometry)

        # Request access to line-up area
        wait_for_lineup_area = self.env.now

        access_line_up_area = lock.line_up_area[origin].request()
        yield access_line_up_area
        lock.waiting_area[origin].release(access_waiting_area)

        if wait_for_lineup_area != self.env.now:
            waiting = self.env.now - wait_for_lineup_area
            self.log_entry("Waiting in waiting area start", wait_for_lineup_area, 0, self.geometry)
            self.log_entry("Waiting in waiting area stop", self.env.now, waiting, self.geometry)
        
        # Request access to lock
        wait_for_lock_entry = self.env.now
        access_lock = lock.resource.request(priority = -1 if origin == lock.water_level else 0)
        yield access_lock

        # Shift water level
        if origin != lock.water_level: yield from lock.convert_chamber(self.env, origin)

        lock.line_up_area[origin].release(access_line_up_area)

        if wait_for_lock_entry != self.env.now:
            waiting = self.env.now - wait_for_lock_entry
            self.log_entry("Waiting in line-up area start", wait_for_lock_entry, 0, self.geometry)
            self.log_entry("Waiting in line-up area stop", self.env.now, waiting, self.geometry)
        
        # Vessel inside the lock
        self.log_entry("Passing lock start", self.env.now, 0, self.geometry)
        
        # Shift the water level
        yield from lock.convert_chamber(self.env, destination)
        
        # Vessel outside the lock
        lock.resource.release(access_lock)
        passage_time = (lock.doors_close + lock.operating_time + lock.doors_open)
        self.log_entry("Passing lock stop", self.env.now, passage_time, nx.get_node_attributes(self.env.FG, "geometry")[destination])

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
        self.wgs84 = pyproj.Geod(ellps='WGS84')

    @property
    def current_speed(self):
        return self.compute_v(self.container.level / self.container.capacity)

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
#             if unit.lines[0] == self.name:
#                 self.log_entry(unit.lines, self.env.now, 0, self.geometry)
#                 unit.lines.pop(0)
#                 self.log_entry(unit.lines, self.env.now, 0, self.geometry)
                
            self.units.append(unit)
            unit.log_entry("Waiting for metro stop", self.env.now, 0, self.geometry)
            unit.log_entry("In metro start", self.env.now, 0, self.geometry)
        
        self.log_entry("Loading stop", self.env.now, 0, self.geometry)
    
    def unload(self):
        """ Unload self """

        self.log_entry("Unloading start", self.env.now, 0, self.geometry)
        for unit in self.units:            
            
            if unit.transfers > 0 and nx.get_node_attributes(self.env.FG, "geometry")[unit.transferstations[0]] == self.geometry:
                unit.log_entry("In metro stop", self.env.now, 0, self.geometry)
                unit.log_entry("Start transfer", self.env.now, 0, self.geometry)
                
                # Set unit to the transfernode
                transfernode = unit.transferstations[0]
                yield self.env.timeout(2 * 60)
                self.env.FG.nodes[transfernode]["object_type"].units.append(unit)
                
                # Update remaining route
                unit.transferstations.pop(0)
                unit.transfers -= 1
                
                # Remove unit from transport
                unit.log_entry("Stop transfer", self.env.now, 0, self.geometry)
                self.units.remove(unit)

            elif nx.get_node_attributes(self.env.FG, "geometry")[unit.route[-1]] == self.geometry:
                unit.log_entry("In metro stop", self.env.now, 0, self.geometry)
                self.units.remove(unit)
        
        self.log_entry("Unloading stop", self.env.now, 0, self.geometry)

class Station(HasContainer):
    """ Station class """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""

        self.units = []