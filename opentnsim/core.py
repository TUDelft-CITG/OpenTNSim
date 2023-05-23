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
import opentnsim.graph_module

# additional packages


logger = logging.getLogger(__name__)


Geometry = shapely.Geometry


class HasResource(SimpyObject):
    """Something that has a resource limitation, a resource request must be granted before the object can be used.

    - nr_resources: nr of requests that can be handled simultaneously
    """

    def __init__(self, nr_resources=1, priority=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.resource = (
            simpy.PriorityResource(self.env, capacity=nr_resources) if priority else simpy.Resource(self.env, capacity=nr_resources)
        )


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
        self.pos_length = simpy.Container(self.env, capacity=length, init=remaining_length)


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


class VesselProperties:
    """Mixin class: Something that has vessel properties
    This mixin is updated to better accommodate the ConsumesEnergy mixin

    - type: can contain info on vessel type (avv class, cemt_class or other)
    - B: vessel width
    - L: vessel length
    - h_min: vessel minimum water depth, can also be extracted from the network edges if they have the property
      ['Info']['GeneralDepth']
    - T: actual draught
    - safety_margin : the water area above the waterway bed reserved to prevent ship grounding due to ship squatting during sailing,
      the value of safety margin depends on waterway bed material and ship types. For tanker vessel with rocky bed the safety
      margin is recommended as 0.3 m based on Van Dorsser et al. The value setting for safety margin depends on the risk attitude
      of the ship captain and shipping companies.
    - h_squat: the water depth considering ship squatting while the ship moving (if set to False, h_squat is disabled)
    - payload: cargo load [ton], the actual draught can be determined by knowing payload based on van Dorsser et al's method.
      (https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)
    - vessel_type: vessel type can be selected from "Container","Dry_SH","Dry_DH","Barge","Tanker".
      ("Dry_SH" means dry bulk single hull, "Dry_DH" means dry bulk double hull),
      based on van Dorsser et al's paper.
      (https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)
    Alternatively you can specify draught based on filling degree
    - H_e: vessel height unloaded
    - H_f: vessel height loaded
    - T_e: draught unloaded
    - T_f: draught loaded
    - renewable_fuel_mass: renewable fuel mass on board [kg]
    - renewable_fuel_volume: renewable fuel volume on board [m3]
    - renewable_fuel_required_space: renewable fuel required storage space (consider packaging factor) on board  [m3]
    """

    # TODO: add blockage factor S to vessel properties

    def __init__(
        self,
        type,
        B,
        L,
        h_min=None,
        T=None,
        safety_margin=None,
        h_squat=None,
        payload=None,
        vessel_type=None,
        renewable_fuel_mass=None,
        renewable_fuel_volume=None,
        renewable_fuel_required_space=None,
        *args,
        **kwargs
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
        # alternative  options
        self.safety_margin = safety_margin
        self.h_squat = h_squat
        self.payload = payload
        self.vessel_type = vessel_type
        self.renewable_fuel_mass = renewable_fuel_mass
        self.renewable_fuel_volume = renewable_fuel_volume
        self.renewable_fuel_required_space = renewable_fuel_required_space

    @property
    def T(self):
        """Compute the actual draught.
        This will default to using the draught passed by the constructor. If it is None it will try to find one in the super class.
        """
        if self._T is not None:
            # if we were passed a T value, use tha one
            T = self._T
        elif self.T_f is not None and self.T_e is not None:
            # base draught on filling degree
            T = self.filling_degree * (self.T_f - self.T_e) + self.T_e
        elif self.payload is not None and self.vessel_type is not None:
            T = opentnsim.strategy.Payload2T(
                self,
                Payload_strategy=self.payload,
                vessel_type=self.vessel_type,
                bounds=(0, 40),
            )  # this need to be tested
        # todo: for later possibly include Payload2T

        return T

    @property
    def h_min(self):
        if self._h_min is not None:
            h_min = self._h_min
        else:
            h_min = opentnsim.graph_module.get_minimum_depth(graph=self.graph, route=self.route)

        return h_min

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

        graph = graph if graph else self.graph
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
                if edge[2]["Width"] >= minWidth and edge[2]["Height"] >= minHeight and edge[2]["Depth"] >= minDepth:
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
            except nx.NetworkXNoPath:
                raise ValueError("No path was found with the given boundary conditions.")

        # If not, return shortest path
        else:
            return nx.dijkstra_path(graph, origin, destination)


class Routable(SimpyObject):
    """Mixin class: Something with a route (networkx format)

    - route: list of node-IDs
    - position_on_route: index of position
    """

    def __init__(self, route, complete_path=None, *args, **kwargs):
        """Initialization"""
        super().__init__(*args, **kwargs)
        env = kwargs.get("env")
        # if env is given and env is not None
        if env is not None:
            has_fg = hasattr(env, "FG")
            has_graph = hasattr(env, "graph")
            if has_fg and not has_graph:
                warnings.warn(".FG attribute has been renamed to .graph, please update your code", DeprecationWarning)
            assert (
                has_fg or has_graph
            ), "Routable expects `.graph` (a networkx graph) to be present as an attribute on the environment"
        super().__init__(*args, **kwargs)
        self.route = route
        # start at start of route
        self.position_on_route = 0
        self.complete_path = complete_path

    @property
    def graph(self):
        if hasattr(self.env, "graph"):
            return self.env.graph
        elif hasattr(self.env, "FG"):
            return self.env.FG
        else:
            raise ValueError("Routable expects .graph to be present on env")


@deprecated.deprecated(reason="Use Routable instead of Routeable")
class Routeable(Routable):
    """Old name for Mixin class: renamed to Routable."""


class Movable(Locatable, Routable, Log):
    """Mixin class: Something can move.

    Used for object that can move with a fixed speed

    - geometry: point used to track its current location
    - v: speed
    - on_pass_edge_functions can contain a list of generators in the form of on_pass_edge(source: Point, destination: Point) -> yield event
    """

    def __init__(self, v: float, *args, **kwargs):
        """Initialization"""
        self.v = v
        super().__init__(*args, **kwargs)
        self.on_pass_edge_functions = []
        self.wgs84 = pyproj.Geod(ellps="WGS84")

    def move(self, destination: Union[Locatable, Geometry, str] = None, engine_order: float = 1.0, duration: float = None):
        """determine distance between origin and destination, and
        yield the time it takes to travel it
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].
        """

        # simplify destination to node or geometry
        if isinstance(destination, Locatable):
            destination = destination.geometry

        self.distance = 0
        speed = self.v

        # Check if vessel is at correct location - if not, move to location
        first_n = self.route[0]
        first_node = self.graph.nodes[first_n]
        first_geometry = first_node["geometry"]

        if self.geometry != first_geometry:
            orig = self.geometry
            dest = first_geometry

            logger.debug("Origin: {orig}")
            logger.debug("Destination: {dest}")

            self.distance += self.wgs84.inv(
                shapely.geometry.shape(orig).x,
                shapely.geometry.shape(orig).y,
                shapely.geometry.shape(dest).x,
                shapely.geometry.shape(dest).y,
            )[2]

            yield self.env.timeout(self.distance / self.current_speed)
            self.log_entry("Sailing to start", self.env.now, self.distance, dest)

        # Move over the path and log every step
        for i, edge in enumerate(zip(self.route[:-1], self.route[1:])):
            # name it a, b here, to avoid confusion with destination argument
            a, b = edge

            # update to current position
            self.geometry = nx.get_node_attributes(self.graph, "geometry")[a]
            self.node = a
            self.position_on_route = i

            # are we already at destination?
            if destination is not None:
                # for geometry we need to use the shapely equivalent
                if isinstance(destination, Geometry) and destination.equals(self.geometry):
                    break
                # or the node equivalence
                if destination == self.node:
                    break

            yield from self.pass_edge(a, b)

            # we arrived at destination
            # update to new position
            self.geometry = nx.get_node_attributes(self.graph, "geometry")[b]
            self.node = b
            self.position_on_route = i + 1

        logger.debug("  distance: " + "%4.2f" % self.distance + " m")
        if self.current_speed is not None:
            logger.debug("  sailing:  " + "%4.2f" % self.current_speed + " m/s")
            logger.debug("  duration: " + "%4.2f" % ((self.distance / self.current_speed) / 3600) + " hrs")
        else:
            logger.debug("  current_speed:  not set")

    def pass_edge(self, origin, destination):
        edge = self.graph.edges[origin, destination]
        orig = nx.get_node_attributes(self.graph, "geometry")[origin]
        dest = nx.get_node_attributes(self.graph, "geometry")[destination]

        for on_pass_edge_function in self.on_pass_edge_functions:
            yield from on_pass_edge_function(origin, destination)

        # TODO: there is an issue here. If geometry is available, resources and power are ignored.
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
                sub_orig = shapely.geometry.Point(edge_route[index][0], edge_route[index][1])
                sub_dest = shapely.geometry.Point(edge_route[index + 1][0], edge_route[index + 1][1])

                distance = self.wgs84.inv(
                    shapely.geometry.shape(sub_orig).x,
                    shapely.geometry.shape(sub_orig).y,
                    shapely.geometry.shape(sub_dest).x,
                    shapely.geometry.shape(sub_dest).y,
                )[2]
                self.distance += distance
                self.log_entry(
                    "Sailing from node {} to node {} sub edge {} start".format(origin, destination, index),
                    self.env.now,
                    0,
                    sub_orig,
                )
                yield self.env.timeout(distance / self.current_speed)
                self.log_entry(
                    "Sailing from node {} to node {} sub edge {} stop".format(origin, destination, index),
                    self.env.now,
                    0,
                    sub_dest,
                )
            self.geometry = dest
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
                edge = self.graph.edges[origin, destination]
                depth = self.graph.get_edge_data(origin, destination)["Info"]["GeneralDepth"]

                # estimate 'grounding speed' as a useful upperbound
                (
                    upperbound,
                    selected,
                    results_df,
                ) = opentnsim.strategy.get_upperbound_for_power2v(self, width=150, depth=depth, margin=0)
                v = self.power2v(self, edge, upperbound)
                # use computed power
                value = self.P_given

            # determine time to pass edge
            timeout = distance / v

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
        self.geometry = dest

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
