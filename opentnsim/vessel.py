# package(s) related to the simulation
import random
import networkx as nx
import numpy as np
import math
import bisect
import pandas as pd

from opentnsim import core, output, graph


class ExtraMetadata:
    """store all leftover keyword arguments as metadata property (use as last mixin)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        # store all other properties as metadata
        self.metadata = kwargs


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

    def __init__(
        self,
        type,
        B,
        L,
        v = 4,
        bound = 'inbound',
        h_min=None,
        T=None,
        H=None,
        H_e=None,
        H_f=None,
        T_e=None,
        T_f=None,
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
        # optional parameters
        self.H_e = H_e
        self.H_f = H_f
        self.T_e = T_e
        self.T_f = T_f
        # hidden because these can also computed on the fly
        self._T = T
        self._H = H
        self._h_min = h_min
        # alternative options for port accessibility
        self.v = v
        self.bound = bound
        # alternative options for energy consumption
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
            # if we were passed a T value, use that one
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
    def H(self):
        """Compute the actual draught.
        This will default to using the draught passed by the constructor. If it is None it will try to find one in the super class.
        """
        if self._H is not None:
            # if we were passed a T value, use that one
            H = self._H
        elif self.H_f is not None and self.H_e is not None:
            # base draught on filling degree
            H = self.filling_degree * (self.H_f - self.H_e) + self.H_e

        return H

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

class IsVessel(core.Identifiable,
               core.Movable,
               VesselProperties,
               ExtraMetadata,
               graph.HasMultiDiGraph,
               output.HasOutput):

    def __init__(self, *args, **kwargs):
        self.overruled_speed = pd.DataFrame(data=[], columns=['Speed'],index=pd.MultiIndex.from_arrays([[], [], []], names=('node_start', 'node_stop', 'k')))
        super().__init__(*args, **kwargs)

