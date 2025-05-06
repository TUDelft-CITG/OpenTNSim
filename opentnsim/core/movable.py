"""
Mixin classes for movable objects.

The following classes are provided:
- Movable
- ContainerDependentMovable
"""
# packkage(s) for documentation, debugging, saving and loading
import logging
from typing import Union

# math packages
import numpy as np

# spatial libraries
import pyproj
import shapely
import shapely.geometry
from shapely import Geometry
import networkx as nx

# Use OpenCLSim objects for core objects (identifiable is imported for later use.)
from openclsim.core import Locatable, Log
from opentnsim.core.routable import Routable
from opentnsim.core.container import HasContainer

# get logger
logger = logging.getLogger(__name__)



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
        self.v = v
        super().__init__(*args, **kwargs)
        self.on_pass_edge_functions = []
        self.on_pass_node_functions = []
        self.wgs84 = pyproj.Geod(ellps="WGS84")

    def move(self, destination: Union[Locatable, Geometry, str] = None):
        """determine distance between origin and destination, and
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].

        Parameters
        ----------
        destination: str, Locatable or Geometry, optional
            the destination to move to. If None, move to the end of the route.

        Yields
        ------
        time it takes to travel the distance to the destination.

        """

        # simplify destination to node or geometry
        if isinstance(destination, Locatable):
            destination = destination.geometry

        self.distance = 0

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

            node = a

            yield from self.pass_node(node)

            # we are now at the node
            self.node = node

            # update to current position
            self.geometry = nx.get_node_attributes(self.graph, "geometry")[a]
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

        # call all on_pass_node_functions
        for on_pass_node_function in self.on_pass_node_functions:
            yield from on_pass_node_function(node)

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
                # Here the upperbound is used to estimate the actual velocity
                power_used = min(self.P_tot_given, upperbound)
                self.v = self.power2v(self, edge, power_used)
                # store upperbound velocity as hidden variables (for inspection of solver)
                self._selected = selected
                self._results_df = results_df
                self._upperbound = upperbound
                # use upperbound power (used to compute the sailing speed)
                value = power_used

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
        """return the current speed of the vessel"""
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