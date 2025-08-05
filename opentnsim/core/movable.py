"""
Mixin classes for movable objects.

The following classes are provided:
- Movable
- ContainerDependentMovable
"""
# package(s) for documentation, debugging, saving and loading
import logging
import warnings
import deprecated
from typing import Union

# math packages
import numpy as np

# spatial libraries
import pyproj
import shapely
import shapely.geometry
from shapely import Geometry
import networkx as nx
import simpy

# use OpenCLSim objects for core objects (identifiable is imported for later use)
import opentnsim.strategy
from openclsim.core import SimpyObject, Locatable, Log
from opentnsim.core.container import HasContainer
from opentnsim.energy.mixins import ConsumesEnergy

# get logger
logger = logging.getLogger(__name__)


class Routable(SimpyObject):
    """Mixin class: Something with a route (networkx format)

    Parameters
    ----------
    route: list
        list of node-IDs
    complete_path: list, optional
        ???
    args, kwargs:
        passed to SimpyObject. Must at least contain parameter env: simpy.Environment.

    Attributes
    -----------
    route: list
        list of node-IDs
    complete_path: list, optional
        ???
    position_on_route: int
        index of position on the route
    """

    def __init__(self, route, complete_path=None, *args, **kwargs):
        """Initialization"""
        # check env input
        env = kwargs.get("env")
        # if env is given and env is not None
        if env is not None:
            has_fg = hasattr(env, "FG")
            has_graph = hasattr(env, "graph")
            if has_fg and not has_graph:
                warnings.warn(".FG attribute has been renamed to .graph, please update your code", DeprecationWarning)
                env.graph = env.FG
            assert (
                has_fg or has_graph
            ), "Routable expects `.graph` (a networkx graph) to be present as an attribute on the environment"

        # initialization
        super().__init__(*args, **kwargs)
        self.route = route
        # start at start of route
        self.position_on_route = 0
        self.complete_path = complete_path

        self.check_attributes()

    def check_attributes(self):
        """Check if all required attributes are set."""
        # check if route is set
        if self.route is None or not isinstance(self.route, list):
            raise ValueError("Routable requires a route (list of node IDs) to be set")
        # check if env is set
        if not hasattr(self, "env") or not isinstance(self.env, simpy.Environment):
            raise ValueError("Routable requires an environment (simpy.Environment) to be set")
        # check if route is on graph
        if not all(node in self.graph.nodes for node in self.route):
            raise ValueError("Routable route must be on the graph")

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
        else:
            raise ValueError("Routable expects .graph to be present on env")

        if isinstance(graph, nx.MultiDiGraph):
            return nx.DiGraph(graph)
        elif isinstance(graph, nx.MultiGraph):
            return nx.Graph(graph)
        return graph


@deprecated.deprecated(reason="Use Routable instead of Routeable")
class Routeable(Routable):
    """Old name for Mixin class: renamed to Routable."""


class Movable(Locatable, Routable, Log):
    """Mixin class: Something can move.

    Used for object that can move with a fixed speed

    Parameters
    ----------
    v: float
        speed of the object (in m/s)
    geometry: shapely.geometry.Point
        passed to Locatable. point used to track its current location
    node: str, optional
        passed to Locatable,
    route: list, optional
        passed to Routable,
    complete_path: list, optional
        passed to Routable,

    Attributes
    ----------
    v: float
        speed of the object (in m/s)
    on_pass_edge_functions: list
        list of functions to call when passing an edge
    on_pass_node_functions: list
        list of functions to call when passing a node
    wsg84: pyproj.Geod
        used for distance computation

    """

    def __init__(self, v: float, *args, **kwargs):
        """Initialization"""
        super().__init__(*args, **kwargs)
        self.v = v
        self.distance = 0
        self.on_pass_node_functions = []
        self.on_pass_edge_functions = []
        self.wgs84 = pyproj.Geod(ellps="WGS84")

        self._check_attributes()

        # resource memory for passing nodes and edges
        self.req = None
        self.resource = None

    def _check_attributes(self):
        """Check if all required attributes are set."""
        # each node on route should have a geometry
        geoms = nx.get_node_attributes(self.graph, "geometry")
        if not all(node in geoms for node in self.route):
            raise ValueError(
                "Nodes on route must have a geometry attribute. Missing geometries for nodes: {}".format(
                    [node for node in self.route if node not in geoms]
                )
            )

        # warning if current not in edge info
        # TODO

        # Error when P_tot_given is set, but no GeneralDepth in edge info
        # TODO

    # TODO: Move was eerst een functie met 'destination' als argument, maar dat is nu niet meer het geval. Willen we dat dit weg is?
    def move(self):
        """determine distance between origin and destination, and
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].
        Yields
        ------
        time it takes to travel the distance to the destination.

        """

        # Check if vessel is at correct location - if not, move to location
        yield from self._move_to_start()

        # Move over the path and log every step
        for index, edge in enumerate(zip(self.route[:-1], self.route[1:])):
            # update current position
            self.current_node, self.next_node = edge  # origin and destination
            self.geometry = nx.get_node_attributes(self.graph, "geometry")[self.current_node]
            self.position_on_route = index

            yield from self.pass_node(self.current_node)

            # are we already at destination?
            if self.next_node == self.current_node:
                warnings.warn(
                    "Route passes node {} twice consecutively..".format(self.current_node),
                    UserWarning,
                )
                continue

            yield from self.pass_edge(self.current_node, self.next_node)

            # we arrived at destination
            # update to new position
            self.geometry = nx.get_node_attributes(self.graph, "geometry")[self.next_node]
            self.current_node = self.next_node
            self.position_on_route = index + 1

        # arrived at end of route. release resource if needed
        if self.req is not None:
            self._release_resource()

        logger.debug("  distance: " + "%4.2f" % self.distance + " m")
        if self.current_speed is not None:
            logger.debug("  sailing:  " + "%4.2f" % self.current_speed + " m/s")
            logger.debug("  duration: " + "%4.2f" % ((self.distance / self.current_speed) / 3600) + " hrs")
        else:
            logger.debug("  current_speed:  not set")

    def _move_to_start(self):
        """Move to the start of the route.

        Yields
        ------
        The time it takes to move to the start of the route.
        """
        # Check if vessel is at correct location - if not, move to location
        vessel_origin_location = nx.get_node_attributes(self.env.graph, "geometry")[self.route[0]]
        if self.geometry != vessel_origin_location:
            self.log_entry_v0("Sailing to start start", self.env.now, self.distance, self.geometry)
            start_location = self.geometry
            logger.debug("Origin: {orig}")
            logger.debug("Destination: {dest}")

            self.distance += self.wgs84.inv(start_location.x, start_location.y, vessel_origin_location.x, vessel_origin_location.y)[
                2
            ]

            yield self.env.timeout(self.distance / self.current_speed)
            self.geometry = vessel_origin_location
            self.log_entry_v0("Sailing to start stop", self.env.now, self.distance, self.geometry)

    def pass_node(self, node):
        """pass a node and call all on_pass_node_functions

        Parameters
        ----------
        node: str
            the node to pass

        Yields
        ------
        The time it takes to pass the node.
        """

        # request resource if needed
        if "Resources" in self.graph.nodes[node].keys() and self.req is None:
            arrival = self.env.now  # remember when we arrived at the node
            # request
            yield from self._request_resource(self.graph.nodes[node]["Resources"])

            # we had to wait, log it
            if arrival != self.env.now:
                self.log_entry_v0(
                    "Waiting to pass node {} start".format(node),
                    arrival,
                    self.distance,
                    self.graph.nodes[node]["geometry"],
                )
                self.log_entry_v0(
                    "Waiting to pass node {} stop".format(node),
                    self.env.now,
                    self.distance,
                    self.graph.nodes[node]["geometry"],
                )

        # call all on_pass_node_functions
        for on_pass_node_function in self.on_pass_node_functions:
            yield from on_pass_node_function(node)

        # release resource if needed
        if self.req is not None:
            # only release if resource is not needed in the next edge
            if self.next_edge is None:
                self._release_resource()
            if "Resources" not in self.graph.edges[self.next_edge]:
                self._release_resource()
            elif self.graph.edges[self.next_edge]["Resources"] != self.resource:
                self._release_resource()
            else:
                pass

    def _release_resource(self):
        """Release the resource if it is not needed in the next edge."""
        self.resource.release(self.req)
        self.req = None
        self.resource = None

    def _request_resource(self, resource):
        self.resource = resource
        self.req = self.resource.request()
        yield self.req

    @property
    def next_edge(self):
        """Return the next edge on the route.

        Returns
        -------
        tuple
            (origin, destination) of the next edge on the route.
        """
        if self.position_on_route < len(self.route) - 1:
            return self.route[self.position_on_route], self.route[self.position_on_route + 1]
        else:
            return None

    def pass_edge(self, origin, destination):
        """pass an edge and call all on_pass_edge_functions.

        Parameters
        ----------
        origin: str
            the origin node of the edge
        destination: str
            the destination node of the edge

        Yields
        ------
        The time it takes to pass the edge.
        """
        edge = self.graph.edges[origin, destination]
        orig = nx.get_node_attributes(self.graph, "geometry")[origin]
        dest = nx.get_node_attributes(self.graph, "geometry")[destination]

        distance = self.wgs84.inv(
            shapely.geometry.shape(orig).x,
            shapely.geometry.shape(orig).y,
            shapely.geometry.shape(dest).x,
            shapely.geometry.shape(dest).y,
        )[2]

        # calculate velocity based on depth and power, if possible.
        self.v = self._compute_velocity_on_edge(origin, destination)

        # Check if the edge has current info
        # NB: positive current is directed from origin to destination
        current = self._get_current(origin, destination)

        # Wait for edge resources to become available
        # TODO: Misschien moeten we Resources ook onder Info hangen?
        if "Resources" in edge.keys() and self.req is None:
            arrival = self.env.now  # remember when we arrived at the edge
            yield from self._request_resource(self.graph.edges[origin, destination]["Resources"])
            # we had to wait, log it
            if arrival != self.env.now:
                self.log_entry_v0(
                    "Waiting to pass edge {} - {} start".format(origin, destination),
                    arrival,
                    self.distance,
                    orig,
                )
                self.log_entry_v0(
                    "Waiting to pass edge {} - {} stop".format(origin, destination),
                    self.env.now,
                    self.distance,
                    orig,
                )

        self.log_entry_v0(
            "Sailing from node {} to node {} start".format(self.current_node, self.next_node),
            self.env.now,
            self.distance,
            orig,
        )

        # on pass edge functions
        for on_pass_edge_function in self.on_pass_edge_functions:
            yield from on_pass_edge_function(origin, destination)

        # default velocity based on current speed.
        timeout = distance / (self.current_speed + current)
        yield self.env.timeout(timeout)
        self.distance += distance

        self.log_entry_v0(
            "Sailing from node {} to node {} stop".format(self.current_node, self.next_node),
            self.env.now,
            self.distance,
            dest,
        )
        self.geometry = dest

        # release resource if needed
        if "Resources" in edge.keys():
            # only release if resource is not needed in the next node
            if "Resources" not in self.graph.nodes[destination].keys():
                self._release_resource()
            elif self.resource != self.graph.nodes[destination]["Resources"]:
                self._release_resource()
                self.resource = None
            else:
                pass

    def _get_current(self, origin, destination):
        """Get the current on the edge

        Parameters
        ----------
        origin: str
            the origin node of the edge
        destination: str
            the destination node of the edge

        Returns
        -------
        float
            the current on the edge (in m/s)
        """
        if "Info" not in self.graph.edges[origin, destination].keys():
            # no info on the current, return 0
            return 0.0
        elif "Current" not in self.graph.edges[origin, destination]["Info"].keys():
            # no info on current, return 0
            return 0.0
        elif not isinstance(self.graph, nx.DiGraph):
            raise TypeError(
                "Current is only available on a DiGraph. Use a Digraph to use current in your calculations.",
                UserWarning,
            )
            return 0.0
        current = self.graph.edges[origin, destination]["Info"]["Current"]

        if (self.current_speed + current) <= 0:
            raise ValueError(
                f"Current {current} m/s is larger than current speed {self.current_speed} m/s. "
                "This will result in a negative speed, which is not allowed.",
                UserWarning,
            )
        return current

    @property
    def current_speed(self):
        """return the current speed of the vessel"""
        return self.v

    def _compute_velocity_on_edge(self, origin, destination):
        """compute the velocity on an edge, based on the energy module and the depth.

        parameters
        ----------
        origin: str
            the origin node of the edge
        destination: str
            the destination node of the edge
        """
        if isinstance(self, ConsumesEnergy) and self.P_tot_given is not None:
            edge = self.graph.edges[origin, destination]
            try:
                depth = self.graph.get_edge_data(origin, destination)["Info"]["GeneralDepth"]
            except KeyError:
                raise ValueError(
                    f"Edge {origin} - {destination} has no GeneralDepth in Info. " f"\n Add info or remove ConsumesEnergy mixin"
                )
            # You can input more power than is realistic
            # There are two mechanisms that reduce the power given:
            # 1. The grounding speed:
            # TODO: Als we dit laten staan, moeten we get_upperbound_for_power2v ook checken en testen.
            # TODO get_upperbound_for_power2v heeft een width hardcoded op 150. Is dat handig?
            (
                upperbound,
                selected,
                results_df,
            ) = opentnsim.strategy.get_upperbound_for_power2v(self, width=150, depth=depth, margin=0)

            # Here the upperbound is used to estimate the actual velocity
            power_used = min(self.P_tot_given, upperbound)
            return self.power2v(self, edge, power_used)
        else:
            # if no ConsumesEnergy mixin, use the default speed
            return self.v


class ContainerDependentMovable(Movable, HasContainer):
    """Mixin class: ContainerDependentMovable class
    Used for objects that move with a speed dependent on the container level

    Parameters
    ----------
    compute_v: function
        a function, given the fraction the container is filled (in [0,1]), returns the current speed
    v: float
        passed to Movable, speed of the object (in m/s)
    geometry: shapely.geometry.Point
        passed to Movable. point used to track its current location
    node: str, optional
        passed to Movable,
    route: list, optional
        passed to Movable,
    complete_path: list, optional
        passed to Movable,
    Capacity: float
        passed to HasContainer, the capacity of the container, which may either be continuous (like water) or discrete (like apples)
    level: int, default=0
        passed to HasContainer, level of the container at the beginning of the simulation
    total_requested: int, default=0
        passed to HasContainer, total amount that has been requested at the beginning of the simulation

    Attributes
    ----------
    compute_v: function
        a function, given the fraction the container is filled (in [0,1]), returns the current speed
    current_speed: float
        the current speed of the vessel (in m/s), based on the filling degree of the container
    """

    def __init__(self, compute_v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.compute_v = compute_v
        self.wgs84 = pyproj.Geod(ellps="WGS84")

    @property
    def current_speed(self):
        """return the current speed of the vessel, based on the filling degree of the container"""
        return self.compute_v(self.filling_degree)
