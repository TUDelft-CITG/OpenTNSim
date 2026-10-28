# -*- coding: utf-8 -*-

"""Graph module."""
# packkage(s) for documentation, debugging, saving and loading
import logging
import os
import pickle

# package(s) for data handling
import networkx as nx
import functools
import yaml
import io
import warnings
import math

# spatial libraries
import pyproj
import requests
import requests_cache
from shapely.geometry import Point, LineString, shape

# package(s) related to the simulation
import simpy

# OpenTNSim
import opentnsim
from opentnsim.core import Identifiable, Locatable, SimpyObject
from opentnsim.graph.utils import validate_geometry, check_node
from opentnsim.graph.calculations import calculate_length_m, transform_geometry
from opentnsim.graph.visualizations import plot_graph

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Determine the wgs84 geoid
wgs84 = pyproj.Geod(ellps="WGS84")

# inject caching
requests_cache.install_cache("fis_cache")

fis_urls = {
    "0.2": "https://zenodo.org/record/4578289/files/network_digital_twin_v0.2.pickle",
    "0.3": "https://zenodo.org/record/6673604/files/network_digital_twin_v0.3.pickle",
}
euris_urls = {
    "0.1": "https://zenodo.org/records/17298014/files/export-graph-v0.1.0.pickle"
}
networks = {
    "fis": fis_urls,
    "euris": euris_urls,
}


class OnNode(SimpyObject):
    def __init__(self, node, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node = node
        if node not in self.env.graph.nodes:
            raise ValueError(f"Node {node} does not exist in the graph.")


class OnEdge(SimpyObject):
    def __init__(self, edge, *args, **kwargs):
        self.edge = edge
        super().__init__(*args, **kwargs)

        if edge not in self.env.graph.edges:
            raise ValueError(f"Node {edge} does not exist in the graph.")


class HasCRS(Locatable):
    def __init__(self, *args, epsg=4326, **kwargs):
        self.epsg = epsg
        self.crs = pyproj.CRS.from_epsg(epsg) #checks epsg
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'geometry'):
            raise ValueError(f"Object does not have a geometry, but does have a coordinate reference system (crs).")

        if self.geometry is not None:
            validate_geometry(self.geometry, self.crs)


class Node(Identifiable, HasCRS):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.geometry, Point):
            raise ValueError(
                "Node geometry should be a shapely.geometry.Point."
            )
        if self.geometry.is_empty:
            raise ValueError(
                "Node geometry should not be empty."
            )


class Edge(Identifiable, HasCRS):

    def __init__(
        self,
        node_0: Node,
        node_1: Node,
        geometry=None,
        length_m=None,
        *args,
        **kwargs,
    ):
        kwargs.setdefault("name", (node_0.name, node_1.name))
        print(kwargs)
        super().__init__(geometry = geometry, *args, **kwargs)

        self.node_0 = node_0
        self.node_1 = node_1

        node_0_geometry = node_0.geometry
        node_1_geometry = node_1.geometry
        if self.geometry is None:
            if node_0.epsg != self.epsg:
                node_0_geometry = transform_geometry(node_0_geometry, node_0.epsg, self.epsg)
            if node_1.epsg != self.epsg:
                node_1_geometry = transform_geometry(node_1_geometry, node_1.epsg, self.epsg)
            self.geometry = LineString([
                node_0_geometry.coords[0],
                node_1_geometry.coords[0],
            ])

        if not isinstance(self.geometry, LineString):
            raise ValueError(
                "Edge geometry should be a LineString."
            )
        if self.geometry.is_empty:
            raise ValueError(
                "Edge geometry should not be empty."
            )

        if length_m is None:
            length_m = calculate_length_m(self.geometry, self.crs)

        if not math.isfinite(length_m) or length_m < 0:
            raise ValueError("length_m must be a finite, non-negative value.")

        self.length_m = length_m

class GraphMixin:
    """Shared functionality for directed and undirected spatial graphs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def add_node(self, node: Node | None = None, **attr) -> None:
        """Add a Node object or construct one from keyword attributes."""

        if node is not None:
            check_node(node)
            attributes = {**vars(node), **attr, "node": node}
        else:
            try:
                node = Node(**attr)
            except TypeError as exc:
                raise TypeError(
                    "When adding a node by attributes, provide the "
                    "required Node constructor arguments, including name "
                    "and geometry."
                ) from exc

            attributes = {**vars(node), "node": node}

        if node.name in self:
            warnings.warn(
                f"Node name {node.name!r} already exists. Skipping node.",
                UserWarning,
                stacklevel=2,
            )
            return

        if node.epsg != 4326:
            node.geometry = transform_geometry(node.geometry, node.epsg, 4326)
            node.epsg = 4326
            node.crs = pyproj.CRS.from_epsg(node.epsg)
        attributes["geometry"] = node.geometry

        super().add_node(node.name, **attributes)

    def add_edge(self, edge: Edge | tuple[str, str], **attr) -> None:
        """Add an Edge object or construct one from two node names."""


        if isinstance(edge, tuple):
            if len(edge) != 2:
                raise ValueError(
                    "A tuple edge must contain exactly two node names."
                )

            node_0, node_1 = edge

            if node_0 not in self or node_1 not in self:
                raise ValueError(
                    f"Both nodes {node_0!r} and {node_1!r} "
                    "must exist in the graph."
                )

            edge = Edge(
                self.nodes[node_0]["node"],
                self.nodes[node_1]["node"],
            )

        elif not isinstance(edge, Edge):
            if not isinstance(edge, Identifiable):
                raise TypeError("edge must be an instance of Identifiable.")
            elif not isinstance(edge, HasCRS):
                raise TypeError("edge must have a coordinate reference system (HasCRS).")
            elif not hasattr(edge, 'node_0'):
                raise ValueError("edge must have an attributes 'node_0' of a Node class type.")
            elif hasattr(edge, 'node_0'):
                node = edge.node_0
                try:
                    check_node(node)
                except ValueError as e:
                    raise ValueError(f"'node_0' is not of Node class type: {e}.")
            elif not hasattr(edge, 'node_1'):
                raise ValueError("edge must have an attributes 'node_1' of a Node class type.")
            elif hasattr(edge, 'node_1'):
                node = edge.node_1
                try:
                    check_node(node)
                except ValueError as e:
                    raise ValueError(f"'node_1' is not of Node class type: {e}.")

        u = edge.node_0.name
        v = edge.node_1.name

        if u not in self or v not in self:
            raise ValueError(
                f"Both nodes {u!r} and {v!r} must exist in the graph."
            )

        if not self.is_multigraph() and self.has_edge(u, v):
            if self.is_directed():
                message = (
                    f"A directed edge from {u!r} to {v!r} already exists. "
                    "Skipping this edge."
                )
            else:
                message = (
                    f"An edge between {u!r} and {v!r} already exists. "
                    "Skipping this edge."
                )

            warnings.warn(message, UserWarning, stacklevel=2)
            return

        if edge.epsg != 4326:
            edge.geometry = transform_geometry(edge.geometry, edge.epsg, 4326)
        attributes = {
            **vars(edge),
            **attr,
            "geometry": edge.geometry,
        }

        super().add_edge(u, v, **attributes)


    def from_shape(
        self,
        file_location,
        shapefile,
        simplify=True,
        strict=True,
    ):
        """Load a graph from a shapefile using the legacy GDAL workflow."""

        from osgeo import ogr

        self.graph = opentnsim.core.utils.read_shp(
            os.path.join(file_location, shapefile),
            simplify=simplify,
            strict=strict,
        )
        self.graph_info = opentnsim.core.utils.info(self.graph)

        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataset = driver.Open(
            os.path.join(file_location, shapefile)
        )

        if dataset is None:
            raise ValueError(
                f"Could not open shapefile {shapefile!r}."
            )

        self.SpatialRef = dataset.GetLayer().GetSpatialRef()

    def transform_projection(self, to_EPSG):
        """Create a legacy GDAL coordinate transformation."""

        from osgeo import osr

        out_spatial_ref = osr.SpatialReference()
        out_spatial_ref.ImportFromEPSG(to_EPSG)

        return osr.CoordinateTransformation(
            self.SpatialRef,
            out_spatial_ref,
        )

    def change_projection(self, transform, point):
        """Transform a point using a legacy GDAL transformation."""

        from osgeo import ogr

        point = ogr.CreateGeometryFromWkt(str(point))
        point.Transform(transform)

        return point.GetX(), point.GetY()

    def create_graph_new_projection(self, to_EPSG=4326):
        """Rebuild the legacy self.graph using a new projection."""

        new_graph = nx.Graph()
        transform = self.transform_projection(to_EPSG)

        nodes_dict = {}

        for old_node, data in self.graph.nodes(data=True):
            coordinates = self.change_projection(
                transform,
                Point(old_node[0], old_node[1]),
            )

            # Preserve the existing coordinate ordering and naming behavior.
            name = "({:f}, {:f})".format(
                coordinates[1],
                coordinates[0],
            )
            geometry = Point(coordinates[1], coordinates[0])

            nodes_dict[old_node] = name

            new_graph.add_node(
                name,
                name=name,
                Position=(coordinates[1], coordinates[0]),
                geometry=geometry,
                Old=data,
            )

        for u, v, data in self.graph.edges(data=True):
            new_graph.add_edge(
                nodes_dict[u],
                nodes_dict[v],
                Info=data,
            )

        new_graph = new_graph.to_directed()
        self.graph = new_graph
        self.graph_info = opentnsim.utils.info(new_graph)

    def add_resources(self, edges, resources, environment):
        """Attach SimPy resources to edges in the legacy self.graph."""

        if len(edges) != len(resources):
            raise ValueError(
                "edges and resources must have the same length."
            )

        for edge, capacity in zip(edges, resources):
            self.graph.edges[edge]["Resources"] = simpy.Resource(
                environment,
                capacity=capacity,
            )

    def plot(self):
        """Plot this graph."""

        return plot_graph(self, static=True)


class Graph(GraphMixin, nx.Graph):
    """Undirected graph with CRS-aware nodes and edges."""
    pass


class DiGraph(GraphMixin, nx.DiGraph):
    """Directed graph with CRS-aware nodes and edges."""
    pass


class MultiGraph(GraphMixin, nx.MultiGraph):
    """Undirected graph that supports parallel edges."""
    pass


class MultiDiGraph(GraphMixin, nx.MultiDiGraph):
    """Directed graph that supports parallel edges."""
    pass


class FIS:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @functools.lru_cache
    def load_fis_network(url):
        """load the topological fairway information system network (vaarweginformatie.nl)"""

        # get the data from the url
        resp = requests.get(url)
        # convert to file object
        stream = io.StringIO(resp.text)

        # This will take a minute or two
        # Here we convert the network to a networkx object
        G = yaml.load(stream, Loader=yaml.Loader)

        # some brief info
        n_bytes = len(resp.content)
        msg = """Loaded network from {url} file size {mb:.2f}MB. Network has {n_nodes} nodes and {n_edges} edges."""
        summary = msg.format(url=url, mb=n_bytes / 1000**2, n_edges=len(G.edges), n_nodes=len(G.nodes))
        logger.info(summary)

        # The topological network contains information about the original geometry.
        # Let's convert those into python shapely objects for easier use later
        for n in G.nodes:
            G.nodes[n]["geometry"] = Point(G.nodes[n]["X"], G.nodes[n]["Y"])
        for e in G.edges:
            edge = G.edges[e]
            edge["geometry"] = shapely.wkt.loads(edge["Wkt"])

        return G

    @staticmethod
    def import_FIS(url):

        fname = "fis_cache\\{}.pkl".format("FIS")
        if os.path.exists(fname):
            print("I am loading cached network")
            with open(fname, "rb") as pkl_file:
                graph = pickle.load(pkl_file)
                pkl_file.close()

        else:
            print("I am getting new network")
            graph = FIS.load_fis_network(url)

            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, "wb") as pkl_file:
                pickle.dump(graph, pkl_file)
                pkl_file.close()

        return graph
    

    def load_network(network="fis", version="0.3"):
        """load the pickle version of the fairway information system network

        Parameters
        ----------
        network : str
            The network to load. Choose "fis" or "euris". Default is "fis".
        version : str
            The version of the network to load. Choose 0.2 or 0.3. Default is "0.3".
        """
        urls = networks[network]
        url = urls[version]
        resp = requests.get(url)
        # convert the response to a file
        f = io.BytesIO(resp.content)

        # read the graph
        graph = pickle.load(f)

        # convert the edges and nodes geometry to shapely objects
        for e in graph.edges:
            edge = graph.edges[e]
            edge["geometry"] = shape(edge["geometry"])
        for n in graph.nodes:
            node = graph.nodes[n]
            node["geometry"] = shape(node["geometry"])

        return graph
