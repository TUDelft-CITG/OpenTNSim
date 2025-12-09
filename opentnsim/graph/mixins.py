# -*- coding: utf-8 -*-

"""Graph module."""
import functools
# packkage(s) for documentation, debugging, saving and loading
import logging
import os
import pickle
import uuid
from itertools import cycle

# matplotlib
import matplotlib.pyplot as plt
# package(s) for data handling
import networkx as nx
import yaml

# spatial libraries
import pyproj
import requests
import shapely.geometry
# package(s) related to the simulation
import simpy
from opentnsim.core import Identifiable, Locatable
# OpenTNSim
from opentnsim.graph import utils
from shapely.geometry import LineString
from shapely.ops import transform

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Determine the wgs84 geoid
wgs84 = pyproj.Geod(ellps="WGS84")


class Node(Identifiable, Locatable):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DiGraph:

    def __init__(self, edges, weights=[1], geometries=[None], edges_info={}, crs="EPSG:4326", bidirectional=True, *args, **kwargs):
        """edges: a list of tuples of two Node-objects"""

        super().__init__(*args, **kwargs)
        self.graph = nx.DiGraph()
        CRS = pyproj.CRS(crs)
        wgs84 = pyproj.CRS("EPSG:4326")
        CRS_to_wgs84 = pyproj.Transformer.from_crs(CRS, wgs84, always_xy=True).transform
        for index, ((node_I, node_II), weight, geometry, edge_info) in enumerate(
            zip(edges, cycle(weights), cycle(geometries), cycle([edges_info]))
        ):
            if node_I.name not in self.graph.nodes:
                node_I.geometry = transform(CRS_to_wgs84, node_I.geometry)
                self.graph.add_node(node_I.name, geometry=node_I.geometry)
            if node_II.name not in self.graph.nodes:
                node_II.geometry = transform(CRS_to_wgs84, node_II.geometry)
                self.graph.add_node(node_II.name, geometry=node_II.geometry)
            if not geometry:
                geometry = LineString([node_I.geometry, node_II.geometry])
            geod = pyproj.Geod(ellps="WGS84")
            length = geod.geometry_length(geometry)
            Info = {}
            for key, value in edge_info.items():
                Info[key] = value[index]
            self.graph.add_edge(
                node_I.name,
                node_II.name,
                weight=weight,
                geometry=geometry,
                length=length,
                Info=Info,
            )
            if bidirectional:
                self.graph.add_edge(
                    node_II.name,
                    node_I.name,
                    weight=weight,
                    geometry=geometry.reverse(),
                    length=length,
                    Info=Info,
                )


class Graph:
    """General networkx object

    Initialize a nx.Graph() element

    Attributes
    ----------
    graph : networkx.Graph
        The graph object
    graph_info : dict
        The graph information
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = nx.Graph()
        self.graph_info = utils.info(self.graph)

    def from_shape(self, file_location, shapefile, simplify=True, strict=True):
        """Generate nx.Graph() from shapefile
        Make sure to install the required package gdal.

        run pip show gdal to check if gdal is installed.

        Parameters
        ----------
        file_location: Path
            location on server of the shapefile
        shapefile: str
            name of the shapefile (including .shp)
        simplify: bool
            if True, the graph is simplified
        strict: bool
            if True, the graph is strict
        """
        from osgeo import ogr, osr

        # Create graph
        self.graph = utils.read_shp(os.path.join(file_location, shapefile), simplify=simplify, strict=strict)
        self.graph_info = utils.info(self.graph)

        # Get spatial reference
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataset = driver.Open(os.path.join(file_location, shapefile))
        self.SpatialRef = dataset.GetLayer().GetSpatialRef()

    def change_projection(self, transform, point):
        """Transform one point on the graph

        Make sure to install the required package gdal (for osgeo).
        run pip show gdal to check if gdal is installed.

        Parameters
        ----------
        transform:
        """
        from osgeo import ogr, osr

        point = ogr.CreateGeometryFromWkt(str(point))

        point.Transform(transform)
        point.ExportToWkt()

        return point.GetX(), point.GetY()

    def create_graph_new_projection(self, to_EPSG=4326):
        """redefine self.graph with the new projection

        Make sure to install the required package gdal (for osgeo).
        run pip show gdal to check if gdal is installed.

        Parameters
        ----------
        to_EPSG: int
            The EPSG code to transform the graph to
        """
        new_graph = nx.Graph()
        transform = utils.transform_projection(from_spatialref=self.SpatialRef, to_EPSG=to_EPSG)

        # Required to prevent loop-in-loop
        nodes_dict = {}

        # Add original nodes and edges to new graph
        for i, node in enumerate(self.graph.nodes(data=True)):
            # TODO: depending on the coordinate transformation x, y might refer to x,y or latitude, longitude.
            # Shapely assumes always x/lon, y/lat
            coordinates = self.change_projection(
                transform,
                shapely.geometry.Point(list(self.graph.nodes)[i][0], list(self.graph.nodes)[i][1]),
            )
            name = "({:f}, {:f})".format(coordinates[1], coordinates[0])
            geometry = shapely.geometry.Point(coordinates[1], coordinates[0])

            nodes_dict[list(self.graph.nodes)[i]] = name
            new_graph.add_node(name, name=name, Position=(coordinates[1], coordinates[0]), geometry=geometry, Old=node[1])

        for edge in self.graph.edges(data=True):
            node_1 = nodes_dict[edge[0]]
            node_2 = nodes_dict[edge[1]]

            new_graph.add_edge(node_1, node_2, Info=edge[2])

        new_graph = new_graph.to_directed()

        if utils.info(new_graph) != self.graph_info:
            self.graph = new_graph
            self.graph_info = utils.info(new_graph)
        else:
            print("Conversion did not create an exact similar graph")

            print("")
            print("Original graph")
            print(self.graph_info)

            print("")
            print("New graph")
            print(utils.info(new_graph))

            self.graph = new_graph
            self.graph_info = utils.info(new_graph)

    def add_resources(self, edges, resources, environment):
        """Add resources to the edges of the graph

        Parameters
        ----------
        edges: list
            List of edges to which the resources should be added
        resources: list
            List of resources to be added to the edges. Should be same length as edges
        environment: simpy.Environment
            The simpy environment to which the resources should be added
        """
        for i, edge in enumerate(edges):
            self.graph.edges[edge]["Resources"] = simpy.Resource(environment, capacity=resources[i])

    def plot(
        self,
        size=[10, 10],
        with_labels=False,
        node_size=0.5,
        font_size=2,
        width=0.2,
        arrowsize=3,
    ):
        """Plot the graph
        Parameters
        ----------
        size: list
            The size of the figure
        with_labels: bool
            If True, the labels of the nodes are shown
        node_size: int
            The size of the nodes, default is 0.5
        font_size: int
            The size of the font, default is 2
        width: int
            The width of the edges, default is 0.2
        arrowsize: int
            The size of the arrows, default is 3
        """
        plt.figure(figsize=size)

        # If graph has positional attributes
        try:
            nx.draw(
                self.graph,
                nx.get_node_attributes(self.graph, "Position"),
                with_labels=with_labels,
                node_size=node_size,
                font_size=font_size,
                width=width,
                arrowsize=arrowsize,
            )
        # If graph does not have any positional information
        except:
            nx.draw(self.graph)

        plt.show()


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
            G.nodes[n]["geometry"] = shapely.geometry.Point(G.nodes[n]["X"], G.nodes[n]["Y"])
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


class HasMultiDiGraph:
    """This locking module uses a MultiDiGraph to represent the network. This converts other graphs to a MultiDiGraph."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def multidigraph(self):
        # create a multidigraph copy of graph if it was not done before
        if hasattr(self, "env"):
            graph_class = self.env
        else:
            graph_class = self
        if not hasattr(graph_class, "_multidigraph"):
            graph_class._multidigraph = self.copy()
        return graph_class._multidigraph

    def copy(self):
        if hasattr(self,"env"):
            graph_class = self.env
        else:
            graph_class = self
        multidigraph = graph_class.graph
        if not isinstance(graph_class.graph, nx.MultiDiGraph):
            multidigraph = nx.MultiDiGraph(multidigraph)
        return multidigraph
