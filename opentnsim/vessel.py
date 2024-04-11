# package(s) related to the simulation
import random
import networkx as nx
import numpy as np
import math
import bisect

class ExtraMetadata:
    """store all leftover keyword arguments as metadata property (use as last mixin)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        # store all other properties as metadata
        self.metadata = kwargs

class VesselProperties:
    """Mixin class: Something that has vessel properties
    This mixin is updated to better accommodate the ConsumesEnergy mixin

    type: can contain info on vessel type (avv class, cemt_class or other)
    B: vessel width
    L: vessel length
    H_e: vessel height unloaded
    H_f: vessel height loaded
    T_e: draught unloaded
    T_f: draught loaded

    Add information on possible restrictions to the vessels, i.e. height, width, etc.
    """

    def __init__(
            self,
            type,
            B,
            L,
            T,
            H,
            origin,
            destination,
            v,
            next_destination = '',
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        """Initialization"""
        self.type = type
        self.B = B
        self.L = L
        self._T = T
        self._H = H
        self.v = v
        self.origin = origin
        self.destination = list(destination)
        self.next_destination = list(next_destination)
        self.bound = 'inbound'

    @property
    def H(self):
        H = self._H
        return H

    @property
    def T(self):
        T = self._T
        return T

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
        minWidth = minWidth if minWidth else 1.1 * self.B
        minHeight = minWidth if minHeight else 1.1 * self.H
        minDepth = minWidth if minDepth else 1.1 * self.T

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