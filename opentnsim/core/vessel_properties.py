"""
Mixin classes related to vessel properties.

The following classes are provided:
- HasLength
- HasLoad
- Vesselproperties

"""
# packkage(s) for documentation, debugging, saving and loading
import logging

# math packages
import random

# spatial libraries
import networkx as nx

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy

# Use OpenCLSim objects for core objects (identifiable is imported for later use.)
from openclsim.core import SimpyObject
import opentnsim.graph_module

# get logger
logger = logging.getLogger(__name__)


class HasLength(SimpyObject):
    """Mixin class: Something with a length. The length is modelled as a storage capacity

    Parameters
    -----------
    length: float
        length that can be requested
    remaining_length: float, default=0
        length that is still available at the beginning of the simulation.
    args, kwargs:
        passed to SimpyObject. Must at least contain parameter env: simpy.Environment.

    Attributes
    -----------
    length: simpy.Container
        the container that is used to limit the length that can be requested.
    pos_length: simpy.Container
        the container that is used to limit the length that can be requested.
    """

    def __init__(self, length: float, remaining_length: float = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.length = simpy.Container(self.env, capacity=length, init=remaining_length)
        self.pos_length = simpy.Container(self.env, capacity=length, init=remaining_length)



class HasLoad:
    """Mixin class with load dependent height (H) and draught (T). The filling
    degree (filling_degree: fraction) will interpolate between empty and full
    height and draught.

    Parameters
    ----------
    H_e: float
        height of the vessel when empty
    H_f: float
        height of the vessel when full
    T_e: float
        draught of the vessel when empty
    T_f: float
        draught of the vessel when full
    filling_degree: float, default=0
        fraction of the vessel that is filled upon creation

    """

    def __init__(self, H_e, H_f, T_e, T_f, filling_degree=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.H_e = H_e
        self.H_f = H_f
        self.T_e = T_e
        self.T_f = T_f
        self.filling_degree = filling_degree

    @property
    def T(self):
        """Calculate current draught based on filling degree"""
        T = self.filling_degree * (self.T_f - self.T_e) + self.T_e
        return T

    @property
    def H(self):
        """Calculate current height based on filling degree"""
        return self.filling_degree * (self.H_f - self.H_e) + self.H_e


class VesselProperties:
    """Mixin class: Something that has vessel properties
    This mixin is updated to better accommodate the ConsumesEnergy mixin

    Parameters
    ----------
    type: str
        can contain info on vessel type (avv class, cemt_class or other)
    B: float
        vessel width
    L: float
        vessel length
    h_min: float, optional
        vessel minimum water depth.
        The wather-depth can also be extracted from the network edges if they have the property ['Info']['GeneralDepth']
    T: float, optional
        actual draught.
        If not given, it will be calculated based on the filling degree (use mixin HasLoad) or based on the payload and vessel type
    safety_margin: float, optional
      the water area above the waterway bed reserved to prevent ship grounding due to ship squatting during sailing
      the value of safety margin depends on waterway bed material and ship types. For tanker vessel with rocky bed the safety
      margin is recommended as 0.3 m based on Van Dorsser et al. The value setting for safety margin depends on the risk attitude
      of the ship captain and shipping companies.
    h_squat: float, optional
        the water depth considering ship squatting while the ship moving (if set to False, h_squat is disabled)
    payload: float, optional
        cargo load [ton], the actual draught can be determined by knowing payload based on van Dorsser et al's method.
        (https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)
    vessel_type: str, optional
        vessel_type: vessel type can be selected from "Container","Dry_SH","Dry_DH","Barge","Tanker".
        ("Dry_SH" means dry bulk single hull, "Dry_DH" means dry bulk double hull),
        based on van Dorsser et al's paper.
        (https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)
    renewable_fuel_mass: float, optional
        renewable fuel mass on board [kg]
    renewable_fuel_volume: float, optional
        renewable fuel volume on board [m3]
    renewable_fuel_required_space: float, optional
        renewable fuel required storage space (consider packaging factor) on board  [m3]
    """

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
        """get the minimum water depth. if not given, h_min is the minimal water depth of the graph.."""
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
        """Calculate a path based on vessel restrictions
        restrictions are only applied if the graph has the attributes ['Width', 'Height', 'Depth']

        Parameters
        ----------
        origin: str
            ID of the starting node
        destination: str
            ID of the destination node
        graph: networkx.Graph, default = self.graph
            The graph to use for the pathfinding
        minWidth: float, default = 1.1 * self.B
            Minimum width of the path
        minHeight: float, default = 1.1 * self.H
            Minimum height of the path
        minDepth: float, default = 1.1 * self.T
            Minimum depth of the path
        randomSeed: int, default = 4
            Seed for the random number generator

        """

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