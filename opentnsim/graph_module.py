# -*- coding: utf-8 -*-

"""Graph module."""
# packkage(s) for documentation, debugging, saving and loading
import logging
import os

# package(s) for data handling
import networkx as nx
import numpy as np

# spatial libraries
import pyproj
import shapely.geometry

# matplotlib
import matplotlib.pyplot as plt

# package(s) related to the simulation
import simpy

# OpenTNSim
import opentnsim.utils

logger = logging.getLogger(__name__)

# Determine the wgs84 geoid
wgs84 = pyproj.Geod(ellps="WGS84")


def geom_to_edges(geom, properties):
    """Generate edges from a geometry, yielding an edge id and edge properties. The edge_id consists of a tuple of coordinates"""
    if not geom.geom_type in ["LineString", "MultiLineString"]:
        msg = "Only ['LineString', 'MultiLineString'] are supported, got {}".format(geom.geom_type)
        raise ValueError(msg)
    if geom.geom_type == "MultiLineString":
        for geom in geom.geoms:
            yield from geom_to_edges(geom, properties)
    elif geom.geom_type == "LineString":
        edge_properties = properties.copy()
        edge_source_coord = geom.coords[0]
        edge_target_coord = geom.coords[-1]
        edge_properties["Wkt"] = shapely.wkt.dumps(geom)
        edge_properties["Wkb"] = shapely.wkb.dumps(geom)
        edge_properties["Json"] = shapely.geometry.mapping(geom)
        edge_properties["e"] = [edge_source_coord, edge_target_coord]
        edge_id = (edge_source_coord, edge_target_coord)
        yield edge_id, edge_properties


def geom_to_node(geom: shapely.geometry.Point, properties: dict):
    if not geom.geom_type == "Point":
        msg = "Only 'Point' is supported, got {}".format(geom.geom_type)
        raise ValueError(msg)
    node_properties = properties.copy()
    node_properties["Wkt"] = shapely.wkt.dumps(geom)
    node_properties["Wkb"] = shapely.wkb.dumps(geom)
    node_properties["Json"] = shapely.geometry.mapping(geom)
    node_properties["n"] = geom.coords[0]
    node_id = geom.coords[0]
    return node_id, node_properties


def gdf_to_nx(gdf):
    """Convert a geopandas dataframe to a networkx DiGraph"""
    FG = nx.DiGraph()
    for _, feature in gdf.iterrows():
        geom = feature.geometry
        if geom is None:
            raise nx.NetworkXError("Bad data: feature missing geometry")
        properties = feature.drop(labels=["geometry"])
        # in case we have single points in the geometry, add them as nodes
        if geom.geom_type == "Point":
            node_idx = geom.coords[0]
            FG.add_node(node_idx, **properties)
            continue
        if geom.geom_type in ["LineString", "MultiLineString"]:
            for edge_id, edge_properties in geom_to_edges(geom, properties):
                node_source, node_target = edge_properties["e"]
                source_geom = shapely.geometry.Point(*node_source)
                node_id, node_properties = geom_to_node(source_geom, {})
                FG.add_node(edge_id[0], **node_properties)
                target_geom = shapely.geometry.Point(*node_target)
                node_id, node_properties = geom_to_node(source_geom, {})
                FG.add_node(edge_id[1], **node_properties)
                FG.add_edge(edge_id[0], edge_id[1], **edge_properties)
    return FG


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
        self.graph_info = opentnsim.utils.info(self.graph)

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
        self.graph = opentnsim.utils.read_shp(os.path.join(file_location, shapefile), simplify=simplify, strict=strict)
        self.graph_info = opentnsim.utils.info(self.graph)

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


def get_minimum_depth(graph, route):
    """return the minimum depth on the route based on the GeneralDepth in the Info dictionary

    Parameters
    ----------
    graph: networkx.Graph
        The graph object. Edges in the graph should have a property called Info (dict), with key GeneralDepth
    route: list
        The route to check the depth for. The route is a list of node ids.

    Returns
    -------
    float
        The minimum depth on the route
    """
    # loop over the route
    depths = []
    # loop over all node pairs (e: edge numbers)
    for e in zip(route[:-1], route[1:]):
        # get the properties
        edge = graph.get_edge_data(e[0], e[1])
        # lookup the depth
        depth = edge["Info"]["GeneralDepth"]
        # remember
        depths.append(depth)
        # find the minimum
    h_min = np.min(depths)
    return h_min


def compute_distance(edge, orig, dest):
    """compute distance from origin to destination.
    The distance is computed based on the edge geometry.
    If the edge has no geometry, returns the distance 'as the crow flies'.

    Parameters
    ----------
    edge: dict
        The edge to compute the distance for.
    orig: shapely.geometry.Point
        The origin point
    dest: shapely.geometry.Point
        The destination point

    """
    if "geometry" not in edge:
        distance = wgs84.inv(
            shapely.geometry.shape(orig).x,
            shapely.geometry.shape(orig).y,
            shapely.geometry.shape(dest).x,
            shapely.geometry.shape(dest).y,
        )[2]
        return distance

    edge_route = np.array(edge["geometry"].coords)

    # check if edge is in the sailing direction, otherwise flip it
    distance_from_start = wgs84.inv(
        orig.x,
        orig.y,
        edge_route[0][0],
        edge_route[0][1],
    )[2]
    distance_from_stop = wgs84.inv(
        orig.x,
        orig.y,
        edge_route[-1][0],
        edge_route[-1][1],
    )[2]
    if distance_from_start > distance_from_stop:
        # when the distance from the starting point is greater than from the end point
        edge_route = np.flipud(np.array(edge["geometry"].coords))

    distance = 0
    for index, pt in enumerate(edge_route[:-1]):
        sub_orig = shapely.geometry.Point(edge_route[index][0], edge_route[index][1])
        sub_dest = shapely.geometry.Point(edge_route[index + 1][0], edge_route[index + 1][1])

        distance += wgs84.inv(
            shapely.geometry.asShape(sub_orig).x,
            shapely.geometry.asShape(sub_orig).y,
            shapely.geometry.asShape(sub_dest).x,
            shapely.geometry.asShape(sub_dest).y,
        )[2]
    return distance
