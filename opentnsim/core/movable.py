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
        super().__init__(*args, **kwargs)
        env = kwargs.get("env")
        # if env is given and env is not None
        # TODO Niet zeker of dit nu werkt. Test toevoegen.
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
        """
        Return the graph of the underlying environment.

        If it's multigraph cast to corresponding type
        If you want the multidigraph use the HasMultiGraph mixin

        """
        graph = None
        if hasattr(self.env, "graph"):
            graph = self.env.graph
        elif hasattr(self.env, "FG"):
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
        self.on_pass_node_functions = []
        self.on_pass_edge_functions = []
        self.on_complete_pass_edge_functions = []
        self.on_look_ahead_to_node_functions = []
        self.wgs84 = pyproj.Geod(ellps="WGS84")

    # TODO: Move was eerst een functie met 'destination' als argument, maar dat is nu niet meer het geval. Willen we dat dit weg is?
    def move(self):
        """determine distance between origin and destination, and
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].
        Yields
        ------
        time it takes to travel the distance to the destination.

        """

        # default distance to next node
        self.distance = 0

        # Check if vessel is at correct location - if not, move to location
        yield from self._move_to_start()

        # Move over the path and log every step
        for index, edge in enumerate(zip(self.route[:-1], self.route[1:])):
            self.current_node, self.next_node = edge  # origin and destination
            start_location = nx.get_node_attributes(self.env.graph, "geometry")[self.current_node]
            end_location = nx.get_node_attributes(self.env.graph, "geometry")[self.next_node]

            # It is important for the locking module that the message of sailing should be before passing the first node in preparation of the actual sailing
            # TODO: Hier loggen we de status, weer met gebruik van de HasOutput mixin.
            # TODO: overweging als we dit wel zo laten: willen we de update_status_report functies als (evt standaard) self.pass_edge_functies hebben?

            # TODO: Sailing start en stop moeten allebij in pass edge. Bijv als een edge een resource heeft gaat anders het loggen van de wait time mis.
            # TODO: als je pass_node al wilt loggen, dan moet het een berichtje zijn als Passing node {} start/stop
            # self.log_entry(
            #     "Sailing from node {} to node {} start".format(self.current_node, self.next_node),
            #     self.env.now,
            #     0,
            #     start_location,
            # )

            yield from self.pass_node(self.current_node)

            # update to current position
            # TODO waarom wordt self.current node al eerder geupdate, en self.geometry pas hier?
            self.geometry = nx.get_node_attributes(self.graph, "geometry")[self.current_node]
            self.position_on_route = index

            # are we already at destination?
            # TODO: Dit lijkt mij een gekke regel. Dit zou betekenen dat we twee keer op dezelfde node komen, en dat we dan stoppen. Dat lijkt me dan een fout in de routeberekening, maar ik zou dan niet stoppen. We willen toch nog steeds naar de laatste node op de route?
            if self.next_node == self.current_node:
                break

            # TODO als we end_location steeds geburiken, zou ik er een attribute van maken.
            yield from self.pass_edge(self.current_node, self.next_node)
            yield from self.complete_pass_edge(self.next_node)

            # we arrived at destination
            # update to new position
            self.geometry = nx.get_node_attributes(self.graph, "geometry")[self.next_node]
            self.current_node = self.next_node
            self.position_on_route = index + 1

            yield from self.look_ahead_to_node(self.next_node)

        logger.debug("  distance: " + "%4.2f" % self.distance + " m")
        if self.current_speed is not None:
            logger.debug("  sailing:  " + "%4.2f" % self.current_speed + " m/s")
            logger.debug("  duration: " + "%4.2f" % ((self.distance / self.current_speed) / 3600) + " hrs")
        else:
            logger.debug("  current_speed:  not set")

    def _move_to_start(self):
        """Move to the start of the route.
        TODO: write test!
        TODO: DE self.output.copy is nieuw ten opzichte van de main branch. Daarvoor moet het al een self.HasOutput object zijn, dus lijkt me niet handig dat dit in Movable zit. Verder nadenken over wat we dan graag in de
        output willen hebben. Was in main: self.log_entry("Sailing to start", self.env.now, self.distance, dest)

        """
        # Check if vessel is at correct location - if not, move to location
        vessel_origin_location = nx.get_node_attributes(self.env.graph, "geometry")[self.route[0]]
        if self.geometry != vessel_origin_location:
            start_location = self.geometry
            logger.debug("Origin: {orig}")
            logger.debug("Destination: {dest}")

            self.distance += self.wgs84.inv(start_location.x, start_location.y, vessel_origin_location.x, vessel_origin_location.y)[
                2
            ]

            yield self.env.timeout(self.distance / self.current_speed)
            self.log_entry_v0("Sailing to start", self.env.now, self.output.copy(), vessel_origin_location)

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

        distance = self.wgs84.inv(
            shapely.geometry.shape(orig).x,
            shapely.geometry.shape(orig).y,
            shapely.geometry.shape(dest).x,
            shapely.geometry.shape(dest).y,
        )[2]

        self.distance += distance

        value = 0  # remember when we arrived at the edge
        arrival = self.env.now

        # This is the case if we are sailing on power
        value = 0
        # TODO: Laten we dit ook gewoon een on_pass_edge functie maken die bij de ConsumesEnergy mixin zit?
        if getattr(self, "P_tot_given", None) is not None:
            edge = self.graph.edges[origin, destination]
            # TODO: willen we ervan uitgaan dat de edge altijd een 'Info' heeft met GeneralDepth?
            depth = self.graph.get_edge_data(origin, destination)["Info"]["GeneralDepth"]

            # You can input more power than is realistic
            # There are two mechanisms that reduce the power given:
            # 1. The grounding speed:
            # TODO: Als we dit laten staan, moeten we get_upperbound_for_power2v ook checken en testen.
            (
                upperbound,
                selected,
                results_df,
            ) = opentnsim.strategy.get_upperbound_for_power2v(self, width=150, depth=depth, margin=0)

            # Here the upperbound is used to estimate the actual velocity
            power_used = min(self.P_tot_given, upperbound)
            self.v = self.power2v(self, edge, power_used)
            # store upperbound velocity
            # TODO: remove these three fields after debugging
            self.selected = selected
            self.results_df = results_df
            self.upperbound = upperbound
            # use upperbound power (used to compute the sailing speed)
            value = power_used

        # Wait for edge resources to become available
        # TODO: Op zich mooi, maar willen we dit niet ook gewoon een functie in on_pass_edge_functies maken?
        # TODO: wat is orig? orig en dest zijn de geometries van de start en stop van de trip (short for origin destination)
        # TODO: write test! Nu werkte het wachten niet!
        if "Resources" in edge.keys():
            with self.graph.edges[origin, destination]["Resources"].request() as request:
                yield request

                # we had to wait, log it
                if arrival != self.env.now:
                    self.log_entry_v0(
                        "Waiting to pass edge {} - {} start".format(origin, destination),
                        arrival,
                        value,
                        orig,
                    )
                    self.log_entry_v0(
                        "Waiting to pass edge {} - {} stop".format(origin, destination),
                        self.env.now,
                        value,
                        orig,
                    )

                self.log_entry_v0(
                    "Sailing from node {} to node {} start".format(self.current_node, self.next_node),
                    self.env.now,
                    0,
                    orig,
                )

                # default velocity based on current speed.
                timeout = distance / self.current_speed
                yield self.env.timeout(timeout)

                self.log_entry_v0(
                    "Sailing from node {} to node {} stop".format(self.current_node, self.next_node),
                    self.env.now,
                    0,
                    dest,
                )
                self.geometry = dest


        else:
            self.log_entry_v0(
                "Sailing from node {} to node {} start".format(self.current_node, self.next_node),
                self.env.now,
                0,
                orig,
            )

            # default velocity based on current speed.
            timeout = distance / self.current_speed
            yield self.env.timeout(timeout)

            self.log_entry_v0(
                "Sailing from node {} to node {} stop".format(self.current_node, self.next_node),
                self.env.now,
                0,
                dest,
            )
            self.geometry = dest

    def complete_pass_edge(self, destination):
        # TODO: Waarom een try/except. Als het niet lukt, dan lijkt het me dat de functie gewoon niet goed is gedefinieerd. Als de simulatie blijft draaien krijg je misschien verkeerde output?
        for gen in self.on_complete_pass_edge_functions:
            try:
                yield from gen(destination)
            except simpy.exceptions.Interrupt as e:
                logger.debug("Completed", exc_info=True)
                raise simpy.exceptions.Interrupt("Completed")

    def look_ahead_to_node(self, destination):
        # TODO: Waarom een try/except. Als het niet lukt, dan lijkt het me dat de functie gewoon niet goed is gedefinieerd. Als de simulatie blijft draaien krijg je misschien verkeerde output?
        for gen in self.on_look_ahead_to_node_functions:
            try:
                yield from gen(destination)
            except simpy.exceptions.Interrupt as e:
                logger.debug("Re-routing", exc_info=True)
                raise simpy.exceptions.Interrupt("Re-routing")

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
