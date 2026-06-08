# -*- coding: utf-8 -*-

"""Graph module."""
# packkage(s) for documentation, debugging, saving and loading
import logging
import os
import pickle
from itertools import cycle

# package(s) for data handling
import networkx as nx
import functools
import yaml
import io

# spatial libraries
import pyproj
import requests
import requests_cache
import shapely.geometry
from shapely.geometry import LineString
from shapely.ops import transform

# package(s) related to the simulation
import simpy

# OpenTNSim
import opentnsim
from opentnsim.core import Identifiable, Locatable, SimpyObject
from opentnsim.graph import utils
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
        self.graph = opentnsim.core.utils.read_shp(os.path.join(file_location, shapefile), simplify=simplify, strict=strict)
        self.graph_info = opentnsim.core.utils.info(self.graph)

        # Get spatial reference
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataset = driver.Open(os.path.join(file_location, shapefile))
        self.SpatialRef = dataset.GetLayer().GetSpatialRef()


    def transform_projection(self, to_EPSG):
        """create a transformation object to transform the graph to a new projection
        Make sure to install the required package gdal.

        run pip show gdal to check if gdal is installed.
        Parameters
        ----------
        to_EPSG: int
            The EPSG code to transform the graph to
        """

        from osgeo import ogr, osr

        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(to_EPSG)

        # Transform the coordinates
        transform = osr.CoordinateTransformation(self.SpatialRef, outSpatialRef)

        return transform


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
        transform = self.transform_projection(to_EPSG)

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

        if opentnsim.utils.info(new_graph) != self.graph_info:
            self.graph = new_graph
            self.graph_info = opentnsim.utils.info(new_graph)
        else:
            print("Conversion did not create an exact similar graph")

            print("")
            print("Original graph")
            print(self.graph_info)

            print("")
            print("New graph")
            print(opentnsim.utils.info(new_graph))

            self.graph = new_graph
            self.graph_info = opentnsim.utils.info(new_graph)


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

    def plot(self):
        fig = plot_graph(self, static = True)
        return fig


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
            edge["geometry"] = shapely.geometry.shape(edge["geometry"])
        for n in graph.nodes:
            node = graph.nodes[n]
            node["geometry"] = shapely.geometry.shape(node["geometry"])

        return graph
