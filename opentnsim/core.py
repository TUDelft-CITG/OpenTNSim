"""Main module."""

# package(s) related to time, space and id
import json
import logging
import uuid
import pathlib
import datetime
import time

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy
import random
import networkx as nx
import numpy as np
import math
import pandas as pd

# spatial libraries
import pyproj
import shapely.geometry

# additional packages

import opentnsim.energy
import opentnsim.graph_module

logger = logging.getLogger(__name__)


class SimpyObject:
    """General object which can be extended by any class requiring a simpy environment

    - env: a simpy Environment
    """

    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env


class HasResource(SimpyObject):
    """Something that has a resource limitation, a resource request must be granted before the object can be used.

    - nr_resources: nr of requests that can be handled simultaneously
    """

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

    - name: a name
    - id: a unique id generated with uuid
    """

    def __init__(self, name, id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.name = name
        # generate some id, in this case based on m
        self.id = id if id else str(uuid.uuid1())


class Locatable:
    """Mixin class: Something with a geometry (geojson format)

    - geometry: can be a point as well as a polygon
    """

    def __init__(self, geometry, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.geometry = geometry
        self.node = None


class Neighbours:
    """Can be added to a locatable object (list)

    - travel_to: list of locatables to which can be travelled
    """

    def ___init(self, travel_to, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.neighbours = travel_to


class HasLength(SimpyObject):
    """Mixin class: Something with a storage capacity

    capacity: amount the container can hold
    level: amount the container holds initially
    total_requested: a counter that helps to prevent over requesting
    """

    def __init__(self, length, remaining_length=0, total_requested=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.length = simpy.Container(self.env, capacity=length, init=remaining_length)
        self.pos_length = simpy.Container(
            self.env, capacity=length, init=remaining_length
        )


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
    This mixin is updated to better accommodate the ConsumesEnergy mixin

    - type: can contain info on vessel type (avv class, cemt_class or other)
    - B: vessel width
    - L: vessel length
    - h_min: vessel minimum water depth, can also be extracted from the network edges if they have the property ['Info']['GeneralDepth']
    - T: actual draught
    - C_B: block coefficient ('fullness') [-]
    - safety_margin : the water area above the waterway bed reserved to prevent ship grounding due to ship squatting during sailing, the value of safety margin depends on waterway bed material and ship types. For tanker vessel with rocky bed the safety margin is recommended as 0.3 m based on Van Dorsser et al. The value setting for safety margin depends on the risk attitude of the ship captain and shipping companies.
    - h_squat: the water depth considering ship squatting while the ship moving (if set to False, h_squat is disabled)
    - payload: cargo load [ton], the actual draught can be determined by knowing payload based on van Dorsser et al's method.(https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)
    - vessel_type: vessel type can be selected from "Container","Dry_SH","Dry_DH","Barge","Tanker". ("Dry_SH" means dry bulk single hull, "Dry_DH" means dry bulk double hull), based on van Dorsser et al's paper.(https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)
    Alternatively you can specify draught based on filling degree
    - H_e: vessel height unloaded
    - H_f: vessel height loaded
    - T_e: draught unloaded
    - T_f: draught loaded

    """

    # TODO: add blockage factor S to vessel properties

    def __init__(
        self,
        type,
        B,
        L,
        h_min=None,
        T=None,
        C_B=None,
        H_e=None,
        H_f=None,
        T_e=None,
        T_f=None,
        safety_margin=None,
        h_squat=None,
        payload=None,
        vessel_type=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        """Initialization
        """
        self.type = type
        self.B = B
        self.L = L
        # hidden because these can also computed on the fly
        self._T = T
        self._h_min = h_min
        self.C_B = C_B
        # alternative  options
        self.H_e = H_e
        self.H_f = H_f
        self.T_e = T_e
        self.T_f = T_f
        self.safety_margin = safety_margin
        self.h_squat = h_squat
        self.payload = payload
        self.vessel_type = vessel_type

    @property
    def T(self):
        """Compute the actual draught

        There are 3 ways to get actual draught
        - by directly providing actual draught values in the notebook
        - Or by providing ship draughts in fully loaded state and empty state, the actual draught will be computed based on filling degree


        """
        if self._T is not None:
            # if we were passed a T value, use tha one
            T = self._T
        elif self.T_f is not None and self.T_e is not None:
            # base draught on filling degree
            T = self.filling_degree * (self.T_f - self.T_e) + self.T_e
        # elif self.payload is not None and self.vessel_type is not None:
        # else:
        #     T = opentnsim.strategy.Payload2T(self, Payload_strategy = self.payload, vessel_type = self.vessel_type, bounds=(0, 40))  # this need to be tested
        # todo: for later possibly include payload2T

        return T

    @property
    def h_min(self):
        if self._h_min is not None:
            h_min = self._h_min
        else:
            h_min = opentnsim.graph_module.get_minimum_depth(
                graph=self.env.FG, route=self.route
            )

        return h_min

    def calculate_max_sinkage(self, v, h_0):
        """Calculate the maximum sinkage of a moving ship

        the calculation equation is described in Barrass, B. & Derrett, R.'s book (2006), Ship Stability for Masters and Mates, chapter 42. https://doi.org/10.1016/B978-0-08-097093-6.00042-6

        some explanation for the variables in the equation:
        - h_0: water depth
        - v: ship velocity relative to the water
        - 150: Here we use the standard width 150 m as the waterway width

        """

        max_sinkage = (
            (self.C_B * ((self.B * self._T) / (150 * h_0)) ** 0.81)
            * ((v * 1.94) ** 2.08)
            / 20
        )

        return max_sinkage

    def calculate_h_squat(self, v, h_0):

        if self.h_squat:
            h_squat = h_0 - self.calculate_max_sinkage(v, h_0)

        else:
            h_squat = h_0

        return h_squat

    @property
    def H(self):
        """Calculate current height based on filling degree"""

        return self.filling_degree * (self.H_f - self.H_e) + self.H_e

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
        """Calculate a path based on vessel restrictions"""

        graph = graph if graph else self.env.FG
        minWidth = minWidth if minWidth else 1.1 * self.B
        minHeight = minHeight if minHeight else 1.1 * self.H
        minDepth = minDepth if minDepth else 1.1 * self.T

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


class Routeable:
    """Mixin class: Something with a route (networkx format)

    - route: a networkx path
    """

    def __init__(self, route, complete_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.route = route
        self.complete_path = complete_path


class IsLockWaitingArea(HasResource, Identifiable, Log):
    """Mixin class: Something has lock object properties

    - properties in meters
    - operation in seconds
    """

    def __init__(self, node, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization
        """

        waiting_area_resources = 100
        self.waiting_area = {
            node: simpy.PriorityResource(self.env, capacity=waiting_area_resources),
        }

        # departure_resources = 4
        # self.departure = {
        #    node: simpy.PriorityResource(self.env, capacity=departure_resources),
        # }


class IsLockLineUpArea(HasResource, HasLength, Identifiable, Log):
    """Mixin class: Something has lock object properties
    - properties in meters
    - operation in seconds
    """

    def __init__(self, node, lineup_length, *args, **kwargs):
        super().__init__(
            length=lineup_length, remaining_length=lineup_length, *args, **kwargs
        )
        """Initialization"""

        self.lock_queue_length = 0

        # Lay-Out
        self.enter_line_up_area = {
            node: simpy.PriorityResource(self.env, capacity=1),
        }

        self.line_up_area = {
            node: simpy.PriorityResource(self.env, capacity=100),
        }

        self.converting_while_in_line_up_area = {
            node: simpy.PriorityResource(self.env, capacity=1),
        }

        self.pass_line_up_area = {
            node: simpy.PriorityResource(self.env, capacity=1),
        }


class HasLockDoors(SimpyObject):
    def __init__(self, node_1, node_3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization
        """

        self.doors_1 = {
            node_1: simpy.PriorityResource(self.env, capacity=1),
        }
        self.doors_2 = {
            node_3: simpy.PriorityResource(self.env, capacity=1),
        }


class IsLock(HasResource, HasLength, HasLockDoors, Identifiable, Log):
    """Mixin class: Something has lock object properties
    - properties in meters
    - operation in seconds
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
        wlev_dif,
        disch_coeff,
        grav_acc,
        opening_area,
        opening_depth,
        simulation_start,
        operating_time,
        *args,
        **kwargs,
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
        self.simulation_start = simulation_start.timestamp()
        self.operating_time = operating_time

        # Operating
        self.doors_open = doors_open
        self.doors_close = doors_close

        # Water level
        assert node_1 != node_3

        self.node_1 = node_1
        self.node_3 = node_3
        self.water_level = random.choice([node_1, node_3])

        super().__init__(
            length=lock_length,
            remaining_length=lock_length,
            node_1=node_1,
            node_3=node_3,
            *args,
            **kwargs,
        )

    def operation_time(self, environment):
        if type(self.wlev_dif) == list:
            operating_time = (
                2
                * self.lock_width
                * self.lock_length
                * abs(
                    self.wlev_dif[1][
                        np.abs(
                            self.wlev_dif[0] - (environment.now - self.simulation_start)
                        ).argmin()
                    ]
                )
            ) / (
                self.disch_coeff
                * self.opening_area
                * math.sqrt(2 * self.grav_acc * self.opening_depth)
            )

        elif type(self.wlev_dif) == float or type(self.wlev_dif) == int:
            operating_time = (
                2 * self.lock_width * self.lock_length * abs(self.wlev_dif)
            ) / (
                self.disch_coeff
                * self.opening_area
                * math.sqrt(2 * self.grav_acc * self.opening_depth)
            )
        assert not isinstance(
            operating_time, complex
        ), f"operating_time number should not be complex: {operating_time}"

        return operating_time

    def convert_chamber(self, environment, new_level, number_of_vessels):
        """Convert the water level"""

        # Close the doors
        self.log_entry(
            "Lock doors closing start",
            environment.now,
            number_of_vessels,
            self.water_level,
        )
        yield environment.timeout(self.doors_close)
        self.log_entry(
            "Lock doors closing stop",
            environment.now,
            number_of_vessels,
            self.water_level,
        )

        # Convert the chamber
        self.log_entry(
            "Lock chamber converting start",
            environment.now,
            number_of_vessels,
            self.water_level,
        )

        # Water level will shift
        self.change_water_level(new_level)
        yield environment.timeout(self.operation_time(environment))
        self.log_entry(
            "Lock chamber converting stop",
            environment.now,
            number_of_vessels,
            self.water_level,
        )
        # Open the doors
        self.log_entry(
            "Lock doors opening start",
            environment.now,
            number_of_vessels,
            self.water_level,
        )
        yield environment.timeout(self.doors_open)
        self.log_entry(
            "Lock doors opening stop",
            environment.now,
            number_of_vessels,
            self.water_level,
        )

    def change_water_level(self, side):
        """Change water level and priorities in queue"""

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

    - geometry: point used to track its current location
    - v: speed
    """

    def __init__(self, v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.v = v
        self.edge_functions = []
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

            logger.debug("Origin: {orig}")
            logger.debug("Destination: {dest}")

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
                    loc = self.route.index(destination)
                    for r in self.route[loc:]:
                        if "Line-up area" in self.env.FG.nodes[r].keys():
                            wait_for_waiting_area = self.env.now
                            access_waiting_area = lock.waiting_area[
                                destination
                            ].request()
                            yield access_waiting_area

                            if wait_for_waiting_area != self.env.now:
                                waiting = self.env.now - wait_for_waiting_area
                                self.log_entry(
                                    "Waiting to enter waiting area start",
                                    wait_for_waiting_area,
                                    0,
                                    nx.get_node_attributes(self.env.FG, "geometry")[
                                        origin
                                    ],
                                )
                                self.log_entry(
                                    "Waiting to enter waiting area stop",
                                    self.env.now,
                                    waiting,
                                    nx.get_node_attributes(self.env.FG, "geometry")[
                                        origin
                                    ],
                                )

            if "Waiting area" in self.env.FG.nodes[origin].keys():
                locks = self.env.FG.nodes[origin]["Waiting area"]
                for lock in locks:
                    loc = self.route.index(origin)
                    for r in self.route[loc:]:
                        if "Line-up area" in self.env.FG.nodes[r].keys():
                            locks2 = self.env.FG.nodes[r]["Line-up area"]
                            for r2 in self.route[loc:]:
                                if "Lock" in self.env.FG.nodes[r2].keys():
                                    locks3 = self.env.FG.nodes[r2]["Lock"]
                                    break

                            self.lock_name = []
                            for lock3 in locks3:
                                if (
                                    lock3.water_level
                                    == self.route[self.route.index(r2) - 1]
                                ):
                                    for lock2 in locks2:
                                        if lock2.name == lock3.name:
                                            if lock2.lock_queue_length == 0:
                                                self.lock_name = lock3.name
                                        break

                            lock_queue_length = []
                            if self.lock_name == []:
                                for lock2 in locks2:
                                    lock_queue_length.append(lock2.lock_queue_length)

                                self.lock_name = locks2[
                                    lock_queue_length.index(min(lock_queue_length))
                                ].name

                            for lock2 in locks2:
                                if lock2.name == self.lock_name:
                                    lock2.lock_queue_length += 1

                            for lock2 in locks2:
                                if lock2.name == self.lock_name:
                                    self.v = 0.5 * speed
                                    break

                            wait_for_lineup_area = self.env.now
                            lock.waiting_area[origin].release(access_waiting_area)

                            if self.route[self.route.index(r2) - 1] == lock3.node_1:
                                if (
                                    lock3.doors_2[lock3.node_3].users != []
                                    and lock3.doors_2[lock3.node_3].users[0].priority
                                    == -1
                                ):
                                    if self.L < lock2.length.level + lock3.length.level:
                                        access_lineup_length = lock2.length.get(self.L)
                                    elif self.L < lock2.length.level:
                                        if lock2.length.level == lock2.length.capacity:
                                            access_lineup_length = lock2.length.get(
                                                self.L
                                            )
                                        elif (
                                            lock2.line_up_area[r].users != []
                                            and lock3.length.level
                                            < lock2.line_up_area[r].users[0].length
                                        ):
                                            access_lineup_length = lock2.length.get(
                                                self.L
                                            )
                                        else:
                                            if lock2.length.get_queue == []:
                                                access_lineup_length = lock2.length.get(
                                                    lock2.length.capacity
                                                )
                                                lock2.length.get_queue[
                                                    -1
                                                ].length = self.L
                                                yield access_lineup_length
                                                correct_lineup_length = (
                                                    lock2.length.put(
                                                        lock2.length.capacity - self.L
                                                    )
                                                )
                                            else:
                                                total_length_waiting_vessels = 0
                                                for q in reversed(
                                                    range(len(lock2.length.get_queue))
                                                ):
                                                    if (
                                                        lock2.length.get_queue[q].amount
                                                        == lock2.length.capacity
                                                    ):
                                                        break
                                                for q2 in range(
                                                    q, len(lock2.length.get_queue)
                                                ):
                                                    total_length_waiting_vessels += (
                                                        lock2.length.get_queue[
                                                            q2
                                                        ].length
                                                    )

                                                if (
                                                    self.L
                                                    > lock2.length.capacity
                                                    - total_length_waiting_vessels
                                                ):
                                                    access_lineup_length = (
                                                        lock2.length.get(
                                                            lock2.length.capacity
                                                        )
                                                    )
                                                    lock2.length.get_queue[
                                                        -1
                                                    ].length = self.L
                                                    yield access_lineup_length
                                                    correct_lineup_length = (
                                                        lock2.length.put(
                                                            lock2.length.capacity
                                                            - self.L
                                                        )
                                                    )
                                                else:
                                                    access_lineup_length = (
                                                        lock2.length.get(self.L)
                                                    )
                                                    lock2.length.get_queue[
                                                        -1
                                                    ].length = self.L
                                                    yield access_lineup_length
                                    else:
                                        if lock2.length.get_queue == []:
                                            access_lineup_length = lock2.length.get(
                                                lock2.length.capacity
                                            )
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(
                                                lock2.length.capacity - self.L
                                            )
                                        else:
                                            total_length_waiting_vessels = 0
                                            for q in reversed(
                                                range(len(lock2.length.get_queue))
                                            ):
                                                if (
                                                    lock2.length.get_queue[q].amount
                                                    == lock2.length.capacity
                                                ):
                                                    break
                                            for q2 in range(
                                                q, len(lock2.length.get_queue)
                                            ):
                                                total_length_waiting_vessels += (
                                                    lock2.length.get_queue[q2].length
                                                )

                                            if (
                                                self.L
                                                > lock2.length.capacity
                                                - total_length_waiting_vessels
                                            ):
                                                access_lineup_length = lock2.length.get(
                                                    lock2.length.capacity
                                                )
                                                lock2.length.get_queue[
                                                    -1
                                                ].length = self.L
                                                yield access_lineup_length
                                                correct_lineup_length = (
                                                    lock2.length.put(
                                                        lock2.length.capacity - self.L
                                                    )
                                                )
                                            else:
                                                access_lineup_length = lock2.length.get(
                                                    self.L
                                                )
                                                lock2.length.get_queue[
                                                    -1
                                                ].length = self.L
                                                yield access_lineup_length

                                else:
                                    if lock2.length.level == lock2.length.capacity:
                                        access_lineup_length = lock2.length.get(self.L)
                                    elif (
                                        lock2.line_up_area[r].users != []
                                        and self.L
                                        < lock2.line_up_area[r].users[-1].lineup_dist
                                        - 0.5 * lock2.line_up_area[r].users[-1].length
                                    ):
                                        access_lineup_length = lock2.length.get(self.L)
                                    else:
                                        if lock2.length.get_queue == []:
                                            access_lineup_length = lock2.length.get(
                                                lock2.length.capacity
                                            )
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(
                                                lock2.length.capacity - self.L
                                            )
                                        else:
                                            total_length_waiting_vessels = 0
                                            for q in reversed(
                                                range(len(lock2.length.get_queue))
                                            ):
                                                if (
                                                    lock2.length.get_queue[q].amount
                                                    == lock2.length.capacity
                                                ):
                                                    break
                                            for q2 in range(
                                                q, len(lock2.length.get_queue)
                                            ):
                                                total_length_waiting_vessels += (
                                                    lock2.length.get_queue[q2].length
                                                )

                                            if (
                                                self.L
                                                > lock2.length.capacity
                                                - total_length_waiting_vessels
                                            ):
                                                access_lineup_length = lock2.length.get(
                                                    lock2.length.capacity
                                                )
                                                lock2.length.get_queue[
                                                    -1
                                                ].length = self.L
                                                yield access_lineup_length
                                                correct_lineup_length = (
                                                    lock2.length.put(
                                                        lock2.length.capacity - self.L
                                                    )
                                                )
                                            else:
                                                access_lineup_length = lock2.length.get(
                                                    self.L
                                                )
                                                lock2.length.get_queue[
                                                    -1
                                                ].length = self.L
                                                yield access_lineup_length

                            elif self.route[self.route.index(r2) - 1] == lock3.node_3:
                                if (
                                    lock3.doors_1[lock3.node_1].users != []
                                    and lock3.doors_1[lock3.node_1].users[0].priority
                                    == -1
                                ):
                                    if self.L < lock2.length.level + lock3.length.level:
                                        access_lineup_length = lock2.length.get(self.L)
                                    elif self.L < lock2.length.level:
                                        if lock2.length.level == lock2.length.capacity:
                                            access_lineup_length = lock2.length.get(
                                                self.L
                                            )
                                        elif (
                                            lock2.line_up_area[r].users != []
                                            and lock3.length.level
                                            < lock2.line_up_area[r].users[0].length
                                        ):
                                            access_lineup_length = lock2.length.get(
                                                self.L
                                            )
                                        else:
                                            if lock2.length.get_queue == []:
                                                access_lineup_length = lock2.length.get(
                                                    lock2.length.capacity
                                                )
                                                yield access_lineup_length
                                                correct_lineup_length = (
                                                    lock2.length.put(
                                                        lock2.length.capacity - self.L
                                                    )
                                                )
                                                yield correct_lineup_length
                                            else:
                                                total_length_waiting_vessels = 0
                                                for q in reversed(
                                                    range(len(lock2.length.get_queue))
                                                ):
                                                    if (
                                                        lock2.length.get_queue[q].amount
                                                        == lock2.length.capacity
                                                    ):
                                                        break
                                                for q2 in range(
                                                    q, len(lock2.length.get_queue)
                                                ):
                                                    total_length_waiting_vessels += (
                                                        lock2.length.get_queue[
                                                            q2
                                                        ].length
                                                    )

                                                if (
                                                    self.L
                                                    > lock2.length.capacity
                                                    - total_length_waiting_vessels
                                                ):
                                                    access_lineup_length = (
                                                        lock2.length.get(
                                                            lock2.length.capacity
                                                        )
                                                    )
                                                    lock2.length.get_queue[
                                                        -1
                                                    ].length = self.L
                                                    yield access_lineup_length
                                                    correct_lineup_length = (
                                                        lock2.length.put(
                                                            lock2.length.capacity
                                                            - self.L
                                                        )
                                                    )
                                                else:
                                                    access_lineup_length = (
                                                        lock2.length.get(self.L)
                                                    )
                                                    lock2.length.get_queue[
                                                        -1
                                                    ].length = self.L
                                                    yield access_lineup_length
                                    else:
                                        if lock2.length.get_queue == []:
                                            access_lineup_length = lock2.length.get(
                                                lock2.length.capacity
                                            )
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(
                                                lock2.length.capacity - self.L
                                            )
                                        else:
                                            total_length_waiting_vessels = 0
                                            for q in reversed(
                                                range(len(lock2.length.get_queue))
                                            ):
                                                if (
                                                    lock2.length.get_queue[q].amount
                                                    == lock2.length.capacity
                                                ):
                                                    break
                                            for q2 in range(
                                                q, len(lock2.length.get_queue)
                                            ):
                                                total_length_waiting_vessels += (
                                                    lock2.length.get_queue[q2].length
                                                )

                                            if (
                                                self.L
                                                > lock2.length.capacity
                                                - total_length_waiting_vessels
                                            ):
                                                access_lineup_length = lock2.length.get(
                                                    lock2.length.capacity
                                                )
                                                lock2.length.get_queue[
                                                    -1
                                                ].length = self.L
                                                yield access_lineup_length
                                                correct_lineup_length = (
                                                    lock2.length.put(
                                                        lock2.length.capacity - self.L
                                                    )
                                                )
                                            else:
                                                access_lineup_length = lock2.length.get(
                                                    self.L
                                                )
                                                lock2.length.get_queue[
                                                    -1
                                                ].length = self.L
                                                yield access_lineup_length
                                else:
                                    if lock2.length.level == lock2.length.capacity:
                                        access_lineup_length = lock2.length.get(self.L)
                                    elif (
                                        lock2.line_up_area[r].users != []
                                        and self.L
                                        < lock2.line_up_area[r].users[-1].lineup_dist
                                        - 0.5 * lock2.line_up_area[r].users[-1].length
                                    ):
                                        access_lineup_length = lock2.length.get(self.L)
                                    else:
                                        if lock2.length.get_queue == []:
                                            access_lineup_length = lock2.length.get(
                                                lock2.length.capacity
                                            )
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(
                                                lock2.length.capacity - self.L
                                            )
                                        else:
                                            total_length_waiting_vessels = 0
                                            for q in reversed(
                                                range(len(lock2.length.get_queue))
                                            ):
                                                if (
                                                    lock2.length.get_queue[q].amount
                                                    == lock2.length.capacity
                                                ):
                                                    break
                                            for q2 in range(
                                                q, len(lock2.length.get_queue)
                                            ):
                                                total_length_waiting_vessels += (
                                                    lock2.length.get_queue[q2].length
                                                )

                                            if (
                                                self.L
                                                > lock2.length.capacity
                                                - total_length_waiting_vessels
                                            ):
                                                access_lineup_length = lock2.length.get(
                                                    lock2.length.capacity
                                                )
                                                lock2.length.get_queue[
                                                    -1
                                                ].length = self.L
                                                yield access_lineup_length
                                                correct_lineup_length = (
                                                    lock2.length.put(
                                                        lock2.length.capacity - self.L
                                                    )
                                                )
                                            else:
                                                access_lineup_length = lock2.length.get(
                                                    self.L
                                                )
                                                lock2.length.get_queue[
                                                    -1
                                                ].length = self.L
                                                yield access_lineup_length

                            if len(lock2.line_up_area[r].users) != 0:
                                self.lineup_dist = (
                                    lock2.line_up_area[r].users[-1].lineup_dist
                                    - 0.5 * lock2.line_up_area[r].users[-1].length
                                    - 0.5 * self.L
                                )
                            else:
                                self.lineup_dist = lock2.length.capacity - 0.5 * self.L

                            self.wgs84 = pyproj.Geod(ellps="WGS84")
                            [
                                lineup_area_start_lat,
                                lineup_area_start_lon,
                                lineup_area_stop_lat,
                                lineup_area_stop_lon,
                            ] = [
                                self.env.FG.nodes[self.route[self.route.index(r)]][
                                    "geometry"
                                ].x,
                                self.env.FG.nodes[self.route[self.route.index(r)]][
                                    "geometry"
                                ].y,
                                self.env.FG.nodes[self.route[self.route.index(r) + 1]][
                                    "geometry"
                                ].x,
                                self.env.FG.nodes[self.route[self.route.index(r) + 1]][
                                    "geometry"
                                ].y,
                            ]
                            fwd_azimuth, _, _ = self.wgs84.inv(
                                lineup_area_start_lat,
                                lineup_area_start_lon,
                                lineup_area_stop_lat,
                                lineup_area_stop_lon,
                            )
                            [
                                self.lineup_pos_lat,
                                self.lineup_pos_lon,
                                _,
                            ] = self.wgs84.fwd(
                                self.env.FG.nodes[self.route[self.route.index(r)]][
                                    "geometry"
                                ].x,
                                self.env.FG.nodes[self.route[self.route.index(r)]][
                                    "geometry"
                                ].y,
                                fwd_azimuth,
                                self.lineup_dist,
                            )

                            access_lineup_area = lock2.line_up_area[r].request()
                            lock2.line_up_area[r].users[-1].length = self.L
                            lock2.line_up_area[r].users[-1].id = self.id
                            lock2.line_up_area[r].users[
                                -1
                            ].lineup_pos_lat = self.lineup_pos_lat
                            lock2.line_up_area[r].users[
                                -1
                            ].lineup_pos_lon = self.lineup_pos_lon
                            lock2.line_up_area[r].users[
                                -1
                            ].lineup_dist = self.lineup_dist
                            lock2.line_up_area[r].users[-1].n = len(
                                lock2.line_up_area[r].users
                            )
                            lock2.line_up_area[r].users[-1].v = 0.25 * speed
                            lock2.line_up_area[r].users[-1].wait_for_next_cycle = False
                            yield access_lineup_area

                            enter_lineup_length = lock2.enter_line_up_area[r].request()
                            yield enter_lineup_length
                            lock2.enter_line_up_area[r].users[0].id = self.id

                            if wait_for_lineup_area != self.env.now:
                                self.v = 0.25 * speed
                                waiting = self.env.now - wait_for_lineup_area
                                self.log_entry(
                                    "Waiting in waiting area start",
                                    wait_for_lineup_area,
                                    0,
                                    nx.get_node_attributes(self.env.FG, "geometry")[
                                        origin
                                    ],
                                )
                                self.log_entry(
                                    "Waiting in waiting area stop",
                                    self.env.now,
                                    waiting,
                                    nx.get_node_attributes(self.env.FG, "geometry")[
                                        origin
                                    ],
                                )
                            break

            if "Line-up area" in self.env.FG.nodes[destination].keys():
                locks = self.env.FG.nodes[destination]["Line-up area"]
                for lock in locks:
                    if lock.name == self.lock_name:
                        loc = self.route.index(destination)
                        orig = shapely.geometry.Point(
                            self.lineup_pos_lat, self.lineup_pos_lon
                        )
                        for r in self.route[loc:]:
                            if "Lock" in self.env.FG.nodes[r].keys():
                                locks = self.env.FG.nodes[r]["Lock"]
                                for lock2 in locks:
                                    for q in range(
                                        len(lock.line_up_area[destination].users)
                                    ):
                                        if (
                                            lock.line_up_area[destination].users[q].id
                                            == self.id
                                        ):
                                            if (
                                                self.route[self.route.index(r) - 1]
                                                == lock2.node_1
                                            ):
                                                if (
                                                    lock2.doors_2[lock2.node_3].users
                                                    != []
                                                    and lock2.doors_2[lock2.node_3]
                                                    .users[0]
                                                    .priority
                                                    == -1
                                                ):
                                                    if q <= 1 and lock.line_up_area[
                                                        destination
                                                    ].users[q].n != lock.line_up_area[
                                                        destination
                                                    ].users[
                                                        q
                                                    ].n - len(
                                                        lock2.resource.users
                                                    ):
                                                        self.lineup_dist = (
                                                            lock.length.capacity
                                                            - 0.5 * self.L
                                                        )
                                            elif (
                                                self.route[self.route.index(r) - 1]
                                                == lock2.node_3
                                            ):
                                                if (
                                                    lock2.doors_1[lock2.node_1].users
                                                    != []
                                                    and lock2.doors_1[lock2.node_1]
                                                    .users[0]
                                                    .priority
                                                    == -1
                                                ):
                                                    if q <= 1 and lock.line_up_area[
                                                        destination
                                                    ].users[q].n != lock.line_up_area[
                                                        destination
                                                    ].users[
                                                        q
                                                    ].n - len(
                                                        lock2.resource.users
                                                    ):
                                                        self.lineup_dist = (
                                                            lock.length.capacity
                                                            - 0.5 * self.L
                                                        )
                                            [
                                                self.lineup_pos_lat,
                                                self.lineup_pos_lon,
                                                _,
                                            ] = self.wgs84.fwd(
                                                self.env.FG.nodes[
                                                    self.route[
                                                        self.route.index(destination)
                                                    ]
                                                ]["geometry"].x,
                                                self.env.FG.nodes[
                                                    self.route[
                                                        self.route.index(destination)
                                                    ]
                                                ]["geometry"].y,
                                                fwd_azimuth,
                                                self.lineup_dist,
                                            )
                                            lock.line_up_area[destination].users[
                                                q
                                            ].lineup_pos_lat = self.lineup_pos_lat
                                            lock.line_up_area[destination].users[
                                                q
                                            ].lineup_pos_lon = self.lineup_pos_lon
                                            lock.line_up_area[destination].users[
                                                q
                                            ].lineup_dist = self.lineup_dist
                                            break

            if "Line-up area" in self.env.FG.nodes[origin].keys():
                locks = self.env.FG.nodes[origin]["Line-up area"]
                for lock in locks:
                    if lock.name == self.lock_name:
                        loc = self.route.index(origin)
                        orig = shapely.geometry.Point(
                            self.lineup_pos_lat, self.lineup_pos_lon
                        )
                        for r in self.route[loc:]:
                            if "Lock" in self.env.FG.nodes[r].keys():
                                locks = self.env.FG.nodes[r]["Lock"]
                                lock.enter_line_up_area[origin].release(
                                    enter_lineup_length
                                )
                                for q in range(len(lock.line_up_area[origin].users)):
                                    if lock.line_up_area[origin].users[q].id == self.id:
                                        if q > 0:
                                            _, _, distance = self.wgs84.inv(
                                                orig.x,
                                                orig.y,
                                                lock.line_up_area[origin]
                                                .users[0]
                                                .lineup_pos_lat,
                                                lock.line_up_area[origin]
                                                .users[0]
                                                .lineup_pos_lon,
                                            )
                                            yield self.env.timeout(distance / self.v)
                                            break

                                for lock2 in locks:
                                    if lock2.name == self.lock_name:
                                        self.v = 0.25 * speed
                                        wait_for_lock_entry = self.env.now

                                        for r2 in self.route[(loc + 1) :]:
                                            if (
                                                "Line-up area"
                                                in self.env.FG.nodes[r2].keys()
                                            ):
                                                locks = self.env.FG.nodes[r2][
                                                    "Line-up area"
                                                ]
                                                for lock3 in locks:
                                                    if lock3.name == self.lock_name:
                                                        break
                                                break

                                        if (
                                            self.route[self.route.index(r) - 1]
                                            == lock2.node_1
                                        ):
                                            if (
                                                len(lock2.doors_2[lock2.node_3].users)
                                                != 0
                                            ):
                                                if (
                                                    lock2.doors_2[lock2.node_3]
                                                    .users[0]
                                                    .priority
                                                    == -1
                                                ):
                                                    if (
                                                        self.L
                                                        > (
                                                            lock2.resource.users[
                                                                -1
                                                            ].lock_dist
                                                            - 0.5
                                                            * lock2.resource.users[
                                                                -1
                                                            ].length
                                                        )
                                                        or lock2.resource.users[
                                                            -1
                                                        ].converting
                                                        == True
                                                    ):
                                                        access_lock_door2 = (
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].request(priority=-1)
                                                        )
                                                        yield access_lock_door2
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].release(access_lock_door2)

                                                        wait_for_next_cycle = (
                                                            lock3.pass_line_up_area[
                                                                r2
                                                            ].request()
                                                        )
                                                        yield wait_for_next_cycle
                                                        lock3.pass_line_up_area[
                                                            r2
                                                        ].release(wait_for_next_cycle)

                                                    if (
                                                        lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].users
                                                        != []
                                                    ):
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].request(
                                                            priority=-1
                                                        )
                                                        yield waiting_during_converting
                                                        lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].release(
                                                            waiting_during_converting
                                                        )

                                                    elif (
                                                        len(
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].users
                                                        )
                                                        == 0
                                                        or (
                                                            len(
                                                                lock2.doors_1[
                                                                    lock2.node_1
                                                                ].users
                                                            )
                                                            != 0
                                                            and lock2.doors_1[
                                                                lock2.node_1
                                                            ]
                                                            .users[0]
                                                            .priority
                                                            != -1
                                                        )
                                                    ) and self.route[
                                                        self.route.index(r) - 1
                                                    ] != lock2.water_level:
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].request()
                                                        yield waiting_during_converting
                                                        yield from lock2.convert_chamber(
                                                            self.env,
                                                            self.route[
                                                                self.route.index(r) - 1
                                                            ],
                                                            0,
                                                        )
                                                        lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].release(
                                                            waiting_during_converting
                                                        )

                                                    access_lock_door1 = lock2.doors_1[
                                                        lock2.node_1
                                                    ].request()
                                                    yield access_lock_door1

                                                    if (
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].users
                                                        != []
                                                        and lock2.doors_2[lock2.node_3]
                                                        .users[0]
                                                        .priority
                                                        == -1
                                                    ):
                                                        access_lock_door2 = (
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].request(priority=-1)
                                                        )
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].release(
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].users[0]
                                                        )
                                                        yield access_lock_door2
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].users[0].id = self.id
                                                    else:
                                                        access_lock_door2 = (
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].request(priority=-1)
                                                        )
                                                        yield access_lock_door2
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].users[0].id = self.id

                                                else:
                                                    if (
                                                        lock3.converting_while_in_line_up_area[
                                                            r2
                                                        ].users
                                                        != []
                                                    ):
                                                        waiting_during_converting = lock3.converting_while_in_line_up_area[
                                                            r2
                                                        ].request()
                                                        yield waiting_during_converting
                                                        lock3.converting_while_in_line_up_area[
                                                            r2
                                                        ].release(
                                                            waiting_during_converting
                                                        )

                                                    access_lock_door1 = lock2.doors_1[
                                                        lock2.node_1
                                                    ].request()
                                                    yield access_lock_door1

                                                    if (
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].users
                                                        != []
                                                        and lock2.doors_2[lock2.node_3]
                                                        .users[0]
                                                        .priority
                                                        == -1
                                                    ):
                                                        access_lock_door2 = (
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].request(priority=-1)
                                                        )
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].release(
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].users[0]
                                                        )
                                                        yield access_lock_door2
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].users[0].id = self.id
                                                    else:
                                                        access_lock_door2 = (
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].request(priority=-1)
                                                        )
                                                        yield access_lock_door2
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].users[0].id = self.id
                                            else:
                                                if (
                                                    lock2.doors_2[lock2.node_3].users
                                                    != []
                                                    and lock2.doors_2[lock2.node_3]
                                                    .users[0]
                                                    .priority
                                                    == -1
                                                ):
                                                    access_lock_door1 = lock2.doors_1[
                                                        lock2.node_1
                                                    ].request()
                                                    yield access_lock_door1
                                                    access_lock_door2 = lock2.doors_2[
                                                        lock2.node_3
                                                    ].request(priority=-1)
                                                    lock2.doors_2[lock2.node_3].release(
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].users[0]
                                                    )
                                                    yield access_lock_door2
                                                    lock2.doors_2[lock2.node_3].users[
                                                        0
                                                    ].id = self.id

                                                elif (
                                                    lock2.doors_2[lock2.node_3].users
                                                    != []
                                                    and lock2.doors_2[lock2.node_3]
                                                    .users[0]
                                                    .priority
                                                    == 0
                                                ):
                                                    access_lock_door1 = lock2.doors_1[
                                                        lock2.node_1
                                                    ].request()
                                                    yield access_lock_door1
                                                    access_lock_door2 = lock2.doors_2[
                                                        lock2.node_3
                                                    ].request(priority=-1)
                                                    yield access_lock_door2
                                                    lock2.doors_2[lock2.node_3].users[
                                                        0
                                                    ].id = self.id

                                                else:
                                                    if (
                                                        lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].users
                                                        != []
                                                    ):
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].request(
                                                            priority=-1
                                                        )
                                                        yield waiting_during_converting
                                                        lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].release(
                                                            waiting_during_converting
                                                        )
                                                        access_lock_door1 = (
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].request()
                                                        )
                                                        yield access_lock_door1

                                                    elif (
                                                        len(
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].users
                                                        )
                                                        == 0
                                                        or (
                                                            len(
                                                                lock2.doors_1[
                                                                    lock2.node_1
                                                                ].users
                                                            )
                                                            != 0
                                                            and lock2.doors_1[
                                                                lock2.node_1
                                                            ]
                                                            .users[0]
                                                            .priority
                                                            != -1
                                                        )
                                                    ) and self.route[
                                                        self.route.index(r) - 1
                                                    ] != lock2.water_level:
                                                        access_lock_door1 = (
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].request()
                                                        )
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].request()
                                                        yield waiting_during_converting
                                                        yield from lock2.convert_chamber(
                                                            self.env,
                                                            self.route[
                                                                self.route.index(r) - 1
                                                            ],
                                                            0,
                                                        )
                                                        lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].release(
                                                            waiting_during_converting
                                                        )

                                                    elif (
                                                        len(
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].users
                                                        )
                                                        != 0
                                                        and lock2.doors_1[lock2.node_1]
                                                        .users[0]
                                                        .priority
                                                        == -1
                                                    ):
                                                        access_lock_door1 = (
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].request()
                                                        )
                                                        yield access_lock_door1

                                                    else:
                                                        access_lock_door1 = (
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].request()
                                                        )

                                                    if (
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].users
                                                        != []
                                                        and lock2.doors_2[lock2.node_3]
                                                        .users[0]
                                                        .priority
                                                        == -1
                                                    ):
                                                        access_lock_door2 = (
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].request(priority=-1)
                                                        )
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].release(
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].users[0]
                                                        )
                                                        yield access_lock_door2
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].users[0].id = self.id
                                                    else:
                                                        access_lock_door2 = (
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].request(priority=-1)
                                                        )
                                                        yield access_lock_door2
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].users[0].id = self.id

                                        elif (
                                            self.route[self.route.index(r) - 1]
                                            == lock2.node_3
                                        ):
                                            if (
                                                len(lock2.doors_1[lock2.node_1].users)
                                                != 0
                                            ):
                                                if (
                                                    lock2.doors_1[lock2.node_1]
                                                    .users[0]
                                                    .priority
                                                    == -1
                                                ):
                                                    if (
                                                        self.L
                                                        > (
                                                            lock2.resource.users[
                                                                -1
                                                            ].lock_dist
                                                            - 0.5
                                                            * lock2.resource.users[
                                                                -1
                                                            ].length
                                                        )
                                                        or lock2.resource.users[
                                                            -1
                                                        ].converting
                                                        == True
                                                    ):
                                                        access_lock_door1 = (
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].request(priority=-1)
                                                        )
                                                        yield access_lock_door1
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].release(access_lock_door1)

                                                        wait_for_next_cycle = (
                                                            lock3.pass_line_up_area[
                                                                r2
                                                            ].request()
                                                        )
                                                        yield wait_for_next_cycle
                                                        lock3.pass_line_up_area[
                                                            r2
                                                        ].release(wait_for_next_cycle)

                                                    if (
                                                        lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].users
                                                        != []
                                                    ):
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].request(
                                                            priority=-1
                                                        )
                                                        yield waiting_during_converting
                                                        lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].release(
                                                            waiting_during_converting
                                                        )

                                                    elif (
                                                        len(
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].users
                                                        )
                                                        == 0
                                                        or (
                                                            len(
                                                                lock2.doors_2[
                                                                    lock2.node_3
                                                                ].users
                                                            )
                                                            != 0
                                                            and lock2.doors_2[
                                                                lock2.node_3
                                                            ]
                                                            .users[0]
                                                            .priority
                                                            != -1
                                                        )
                                                    ) and self.route[
                                                        self.route.index(r) - 1
                                                    ] != lock2.water_level:
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].request()
                                                        yield waiting_during_converting
                                                        yield from lock2.convert_chamber(
                                                            self.env,
                                                            self.route[
                                                                self.route.index(r) - 1
                                                            ],
                                                            0,
                                                        )
                                                        lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].release(
                                                            waiting_during_converting
                                                        )

                                                    access_lock_door2 = lock2.doors_2[
                                                        lock2.node_3
                                                    ].request()
                                                    yield access_lock_door2

                                                    if (
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].users
                                                        != []
                                                        and lock2.doors_1[lock2.node_1]
                                                        .users[0]
                                                        .priority
                                                        == -1
                                                    ):
                                                        access_lock_door1 = (
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].request(priority=-1)
                                                        )
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].release(
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].users[0]
                                                        )
                                                        yield access_lock_door1
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].users[0].id = self.id
                                                    else:
                                                        access_lock_door1 = (
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].request(priority=-1)
                                                        )
                                                        yield access_lock_door1
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].users[0].id = self.id

                                                else:
                                                    if (
                                                        lock3.converting_while_in_line_up_area[
                                                            r2
                                                        ].users
                                                        != []
                                                    ):
                                                        waiting_during_converting = lock3.converting_while_in_line_up_area[
                                                            r2
                                                        ].request()
                                                        yield waiting_during_converting
                                                        lock3.converting_while_in_line_up_area[
                                                            r2
                                                        ].release(
                                                            waiting_during_converting
                                                        )

                                                    access_lock_door2 = lock2.doors_2[
                                                        lock2.node_3
                                                    ].request()
                                                    yield access_lock_door2

                                                    if (
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].users
                                                        != []
                                                        and lock2.doors_1[lock2.node_1]
                                                        .users[0]
                                                        .priority
                                                        == -1
                                                    ):
                                                        access_lock_door1 = (
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].request(priority=-1)
                                                        )
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].release(
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].users[0]
                                                        )
                                                        yield access_lock_door1
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].users[0].id = self.id
                                                    else:
                                                        access_lock_door1 = (
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].request(priority=-1)
                                                        )
                                                        yield access_lock_door1
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].users[0].id = self.id
                                            else:
                                                if (
                                                    lock2.doors_1[lock2.node_1].users
                                                    != []
                                                    and lock2.doors_1[lock2.node_1]
                                                    .users[0]
                                                    .priority
                                                    == -1
                                                ):
                                                    access_lock_door2 = lock2.doors_2[
                                                        lock2.node_3
                                                    ].request()
                                                    yield access_lock_door2
                                                    access_lock_door1 = lock2.doors_1[
                                                        lock2.node_1
                                                    ].request(priority=-1)
                                                    lock2.doors_1[lock2.node_1].release(
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].users[0]
                                                    )
                                                    yield access_lock_door1
                                                    lock2.doors_1[lock2.node_1].users[
                                                        0
                                                    ].id = self.id

                                                elif (
                                                    lock2.doors_1[lock2.node_1].users
                                                    != []
                                                    and lock2.doors_1[lock2.node_1]
                                                    .users[0]
                                                    .priority
                                                    == 0
                                                ):
                                                    access_lock_door2 = lock2.doors_2[
                                                        lock2.node_3
                                                    ].request()
                                                    yield access_lock_door2
                                                    access_lock_door1 = lock2.doors_1[
                                                        lock2.node_1
                                                    ].request(priority=-1)
                                                    yield access_lock_door1
                                                    lock2.doors_1[lock2.node_1].users[
                                                        0
                                                    ].id = self.id

                                                else:
                                                    if (
                                                        lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].users
                                                        != []
                                                    ):
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].request(
                                                            priority=-1
                                                        )
                                                        yield waiting_during_converting
                                                        lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].release(
                                                            waiting_during_converting
                                                        )
                                                        access_lock_door2 = (
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].request()
                                                        )
                                                        yield access_lock_door2

                                                    elif (
                                                        len(
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].users
                                                        )
                                                        == 0
                                                        or (
                                                            len(
                                                                lock2.doors_2[
                                                                    lock2.node_3
                                                                ].users
                                                            )
                                                            != 0
                                                            and lock2.doors_2[
                                                                lock2.node_3
                                                            ]
                                                            .users[0]
                                                            .priority
                                                            != -1
                                                        )
                                                    ) and self.route[
                                                        self.route.index(r) - 1
                                                    ] != lock2.water_level:
                                                        access_lock_door2 = (
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].request()
                                                        )
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].request()
                                                        yield waiting_during_converting
                                                        yield from lock2.convert_chamber(
                                                            self.env,
                                                            self.route[
                                                                self.route.index(r) - 1
                                                            ],
                                                            0,
                                                        )
                                                        lock.converting_while_in_line_up_area[
                                                            origin
                                                        ].release(
                                                            waiting_during_converting
                                                        )

                                                    elif (
                                                        len(
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].users
                                                        )
                                                        != 0
                                                        and lock2.doors_2[lock2.node_3]
                                                        .users[0]
                                                        .priority
                                                        == -1
                                                    ):
                                                        access_lock_door2 = (
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].request()
                                                        )
                                                        yield access_lock_door2

                                                    else:
                                                        access_lock_door2 = (
                                                            lock2.doors_2[
                                                                lock2.node_3
                                                            ].request()
                                                        )

                                                    if (
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].users
                                                        != []
                                                        and lock2.doors_1[lock2.node_1]
                                                        .users[0]
                                                        .priority
                                                        == -1
                                                    ):
                                                        access_lock_door1 = (
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].request(priority=-1)
                                                        )
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].release(
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].users[0]
                                                        )
                                                        yield access_lock_door1
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].users[0].id = self.id
                                                    else:
                                                        access_lock_door1 = (
                                                            lock2.doors_1[
                                                                lock2.node_1
                                                            ].request(priority=-1)
                                                        )
                                                        yield access_lock_door1
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].users[0].id = self.id

                                        access_lock_length = lock2.length.get(self.L)
                                        access_lock = lock2.resource.request()

                                        access_lock_pos_length = lock2.pos_length.get(
                                            self.L
                                        )
                                        self.lock_dist = (
                                            lock2.pos_length.level + 0.5 * self.L
                                        )
                                        yield access_lock_pos_length

                                        lock2.resource.users[-1].id = self.id
                                        lock2.resource.users[-1].length = self.L
                                        lock2.resource.users[
                                            -1
                                        ].lock_dist = self.lock_dist
                                        lock2.resource.users[-1].converting = False
                                        if (
                                            self.route[self.route.index(r) - 1]
                                            == lock2.node_1
                                        ):
                                            lock2.resource.users[-1].dir = 1.0
                                        else:
                                            lock2.resource.users[-1].dir = 2.0

                                        if wait_for_lock_entry != self.env.now:
                                            waiting = self.env.now - wait_for_lock_entry
                                            self.log_entry(
                                                "Waiting in line-up area start",
                                                wait_for_lock_entry,
                                                0,
                                                orig,
                                            )
                                            self.log_entry(
                                                "Waiting in line-up area stop",
                                                self.env.now,
                                                waiting,
                                                orig,
                                            )

                                        self.wgs84 = pyproj.Geod(ellps="WGS84")
                                        [
                                            doors_origin_lat,
                                            doors_origin_lon,
                                            doors_destination_lat,
                                            doors_destination_lon,
                                        ] = [
                                            self.env.FG.nodes[
                                                self.route[self.route.index(r) - 1]
                                            ]["geometry"].x,
                                            self.env.FG.nodes[
                                                self.route[self.route.index(r) - 1]
                                            ]["geometry"].y,
                                            self.env.FG.nodes[
                                                self.route[self.route.index(r) + 1]
                                            ]["geometry"].x,
                                            self.env.FG.nodes[
                                                self.route[self.route.index(r) + 1]
                                            ]["geometry"].y,
                                        ]
                                        fwd_azimuth, _, distance = self.wgs84.inv(
                                            doors_origin_lat,
                                            doors_origin_lon,
                                            doors_destination_lat,
                                            doors_destination_lon,
                                        )
                                        [
                                            self.lock_pos_lat,
                                            self.lock_pos_lon,
                                            _,
                                        ] = self.wgs84.fwd(
                                            self.env.FG.nodes[
                                                self.route[self.route.index(r) - 1]
                                            ]["geometry"].x,
                                            self.env.FG.nodes[
                                                self.route[self.route.index(r) - 1]
                                            ]["geometry"].y,
                                            fwd_azimuth,
                                            self.lock_dist,
                                        )

                                        for r4 in reversed(self.route[: (loc - 1)]):
                                            if (
                                                "Line-up area"
                                                in self.env.FG.nodes[r4].keys()
                                            ):
                                                locks = self.env.FG.nodes[r4][
                                                    "Line-up area"
                                                ]
                                                for lock4 in locks:
                                                    if lock4.name == self.lock_name:
                                                        lock4.lock_queue_length -= 1
                                break

                            elif "Waiting area" in self.env.FG.nodes[r].keys():
                                for r2 in reversed(self.route[: (loc - 1)]):
                                    if "Lock" in self.env.FG.nodes[r2].keys():
                                        locks = self.env.FG.nodes[r2]["Lock"]
                                        for lock2 in locks:
                                            if lock2.name == self.lock_name:
                                                if (
                                                    self.route[self.route.index(r2) + 1]
                                                    == lock2.node_3
                                                    and len(
                                                        lock2.doors_2[
                                                            lock2.node_3
                                                        ].users
                                                    )
                                                    != 0
                                                    and lock2.doors_2[lock2.node_3]
                                                    .users[0]
                                                    .id
                                                    == self.id
                                                ):
                                                    lock2.doors_2[lock2.node_3].release(
                                                        access_lock_door2
                                                    )
                                                elif (
                                                    self.route[self.route.index(r2) + 1]
                                                    == lock2.node_1
                                                    and len(
                                                        lock2.doors_1[
                                                            lock2.node_1
                                                        ].users
                                                    )
                                                    != 0
                                                    and lock2.doors_1[lock2.node_1]
                                                    .users[0]
                                                    .id
                                                    == self.id
                                                ):
                                                    lock2.doors_1[lock2.node_1].release(
                                                        access_lock_door1
                                                    )

                                                lock.pass_line_up_area[origin].release(
                                                    departure_lock
                                                )
                                                lock2.resource.release(access_lock)
                                                departure_lock_length = (
                                                    lock2.length.put(self.L)
                                                )
                                                departure_lock_pos_length = (
                                                    lock2.pos_length.put(self.L)
                                                )
                                                yield departure_lock_length
                                                yield departure_lock_pos_length
                                        break

            if "Line-up area" in self.env.FG.nodes[self.route[node[0] - 1]].keys():
                locks = self.env.FG.nodes[self.route[node[0] - 1]]["Line-up area"]
                for lock in locks:
                    if lock.name == self.lock_name:
                        loc = self.route.index(origin)
                        for r in self.route[loc:]:
                            if "Lock" in self.env.FG.nodes[r].keys():
                                locks = self.env.FG.nodes[r]["Lock"]
                                lock.line_up_area[self.route[node[0] - 1]].release(
                                    access_lineup_area
                                )
                                departure_lineup_length = lock.length.put(self.L)
                                yield departure_lineup_length

            if "Lock" in self.env.FG.nodes[origin].keys():
                locks = self.env.FG.nodes[origin]["Lock"]
                for lock in locks:
                    if lock.name == self.lock_name:
                        if self.route[self.route.index(origin) - 1] == lock.node_1:
                            lock.doors_1[lock.node_1].release(access_lock_door1)
                        elif self.route[self.route.index(origin) - 1] == lock.node_3:
                            lock.doors_2[lock.node_3].release(access_lock_door2)
                        orig = shapely.geometry.Point(
                            self.lock_pos_lat, self.lock_pos_lon
                        )
                        loc = self.route.index(origin)
                        for r2 in reversed(self.route[loc:]):
                            if "Line-up area" in self.env.FG.nodes[r2].keys():
                                locks = self.env.FG.nodes[r2]["Line-up area"]
                                for lock3 in locks:
                                    if lock3.name == self.lock_name:
                                        departure_lock = lock3.pass_line_up_area[
                                            r2
                                        ].request(priority=-1)
                                        break
                                break

                        for r in reversed(self.route[: (loc - 1)]):
                            if "Line-up area" in self.env.FG.nodes[r].keys():
                                locks = self.env.FG.nodes[r]["Line-up area"]
                                for lock2 in locks:
                                    if lock2.name == self.lock_name:
                                        for q2 in range(0, len(lock.resource.users)):
                                            if lock.resource.users[q2].id == self.id:
                                                break

                                        start_time_in_lock = self.env.now
                                        self.log_entry(
                                            "Passing lock start", self.env.now, 0, orig
                                        )

                                        if (
                                            len(lock2.line_up_area[r].users) != 0
                                            and lock2.line_up_area[r].users[0].length
                                            < lock.length.level
                                        ):
                                            if (
                                                self.route[self.route.index(origin) - 1]
                                                == lock.node_1
                                            ):
                                                access_line_up_area = (
                                                    lock2.enter_line_up_area[
                                                        r
                                                    ].request()
                                                )
                                                yield access_line_up_area
                                                lock2.enter_line_up_area[r].release(
                                                    access_line_up_area
                                                )
                                                access_lock_door1 = lock.doors_1[
                                                    lock.node_1
                                                ].request()
                                                yield access_lock_door1
                                                lock.doors_1[lock.node_1].release(
                                                    access_lock_door1
                                                )

                                            elif (
                                                self.route[self.route.index(origin) - 1]
                                                == lock.node_3
                                            ):
                                                access_line_up_area = (
                                                    lock2.enter_line_up_area[
                                                        r
                                                    ].request()
                                                )
                                                yield access_line_up_area
                                                lock2.enter_line_up_area[r].release(
                                                    access_line_up_area
                                                )
                                                access_lock_door2 = lock.doors_2[
                                                    lock.node_3
                                                ].request()
                                                yield access_lock_door2
                                                lock.doors_2[lock.node_3].release(
                                                    access_lock_door2
                                                )

                                        if lock.resource.users[0].id == self.id:
                                            lock.resource.users[0].converting = True
                                            number_of_vessels = len(lock.resource.users)
                                            yield from lock.convert_chamber(
                                                self.env, destination, number_of_vessels
                                            )
                                        else:
                                            for u in range(len(lock.resource.users)):
                                                if lock.resource.users[u].id == self.id:
                                                    lock.resource.users[
                                                        u
                                                    ].converting = True
                                                    yield self.env.timeout(
                                                        lock.doors_close
                                                        + lock.operation_time(self.env)
                                                        + lock.doors_open
                                                    )
                                                    break

                        yield departure_lock

                        self.log_entry(
                            "Passing lock stop",
                            self.env.now,
                            self.env.now - start_time_in_lock,
                            orig,
                        )
                        [self.lineup_pos_lat, self.lineup_pos_lon] = [
                            self.env.FG.nodes[self.route[self.route.index(r2)]][
                                "geometry"
                            ].x,
                            self.env.FG.nodes[self.route[self.route.index(r2)]][
                                "geometry"
                            ].y,
                        ]
                        yield from self.pass_edge(origin, destination)
                        self.v = speed

            else:
                # print('I am going to go to the next node {}'.format(destination))
                yield from self.pass_edge(origin, destination)

            if node[0] + 2 == len(self.route):
                break

        # self.geometry = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        logger.debug("  distance: " + "%4.2f" % self.distance + " m")
        if self.current_speed is not None:
            logger.debug("  sailing:  " + "%4.2f" % self.current_speed + " m/s")
            logger.debug(
                "  duration: "
                + "%4.2f" % ((self.distance / self.current_speed) / 3600)
                + " hrs"
            )
        else:
            logger.debug("  current_speed:  not set")

    def pass_edge(self, origin, destination):
        edge = self.env.FG.edges[origin, destination]
        orig = nx.get_node_attributes(self.env.FG, "geometry")[origin]
        dest = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        for edge_function in self.edge_functions:
            edge_function(orig, dest, edge)

        if "Lock" in self.env.FG.nodes[origin].keys():
            orig = shapely.geometry.Point(self.lock_pos_lat, self.lock_pos_lon)

        if "Lock" in self.env.FG.nodes[destination].keys():
            dest = shapely.geometry.Point(self.lock_pos_lat, self.lock_pos_lon)

        if "Line-up area" in self.env.FG.nodes[origin].keys():
            orig = shapely.geometry.Point(self.lineup_pos_lat, self.lineup_pos_lon)

        if "Line-up area" in self.env.FG.nodes[destination].keys():
            dest = shapely.geometry.Point(self.lineup_pos_lat, self.lineup_pos_lon)

        if "geometry" in edge:
            edge_route = np.array(edge["geometry"].coords)

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
            if distance_from_start > distance_from_stop:
                # when the distance from the starting point is greater than from the end point
                edge_route = np.flipud(np.array(edge["geometry"].coords))

            for index, pt in enumerate(edge_route[:-1]):
                sub_orig = shapely.geometry.Point(
                    edge_route[index][0], edge_route[index][1]
                )
                sub_dest = shapely.geometry.Point(
                    edge_route[index + 1][0], edge_route[index + 1][1]
                )

                distance = self.wgs84.inv(
                    shapely.geometry.asShape(sub_orig).x,
                    shapely.geometry.asShape(sub_orig).y,
                    shapely.geometry.asShape(sub_dest).x,
                    shapely.geometry.asShape(sub_dest).y,
                )[2]
                self.distance += distance
                self.log_entry(
                    "Sailing from node {} to node {} sub edge {} start".format(
                        origin, destination, index
                    ),
                    self.env.now,
                    0,
                    sub_orig,
                )
                yield self.env.timeout(distance / self.current_speed)
                self.log_entry(
                    "Sailing from node {} to node {} sub edge {} stop".format(
                        origin, destination, index
                    ),
                    self.env.now,
                    0,
                    sub_dest,
                )
            self.geometry = dest
            # print('   My new origin is {}'.format(destination))
        else:
            distance = self.wgs84.inv(
                shapely.geometry.shape(orig).x,
                shapely.geometry.shape(orig).y,
                shapely.geometry.shape(dest).x,
                shapely.geometry.shape(dest).y,
            )[2]

            self.distance += distance

            value = 0

            # remember when we arrived at the edge
            arrival = self.env.now

            v = self.current_speed

            # This is the case if we are sailing on power
            if getattr(self, "P_tot_given", None) is not None:
                edge = self.env.FG.edges[origin, destination]
                depth = self.env.FG.get_edge_data(origin, destination)["Info"][
                    "GeneralDepth"
                ]

                # estimate 'grounding speed' as a useful upperbound
                (
                    upperbound,
                    selected,
                    results_df,
                ) = opentnsim.strategy.get_upperbound_for_power2v(
                    self, width=150, depth=depth, margin=0
                )
                v = self.power2v(self, edge, upperbound)
                # use computed power
                value = self.P_given

            # determine time to pass edge
            timeout = distance / v

            # Wait for edge resources to become available
            if "Resources" in edge.keys():
                with self.env.FG.edges[origin, destination][
                    "Resources"
                ].request() as request:
                    yield request
                    # we had to wait, log it
                    if arrival != self.env.now:
                        self.log_entry(
                            "Waiting to pass edge {} - {} start".format(
                                origin, destination
                            ),
                            arrival,
                            value,
                            orig,
                        )
                        self.log_entry(
                            "Waiting to pass edge {} - {} stop".format(
                                origin, destination
                            ),
                            self.env.now,
                            value,
                            orig,
                        )

            # default velocity based on current speed.
            self.log_entry(
                "Sailing from node {} to node {} start".format(origin, destination),
                self.env.now,
                value,
                orig,
            )
            yield self.env.timeout(timeout)
            self.log_entry(
                "Sailing from node {} to node {} stop".format(origin, destination),
                self.env.now,
                value,
                dest,
            )

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


class ExtraMetadata:
    """store all leftover keyword arguments as metadata property (use as last mixin)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        # store all other properties as metadata
        self.metadata = kwargs
