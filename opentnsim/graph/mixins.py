# -*- coding: utf-8 -*-

"""Graph module."""
# packkage(s) for documentation, debugging, saving and loading
import logging
import os
import functools

# matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot

# package(s) for data handling
import networkx as nx
import numpy as np
import os
import pickle
import requests
import uuid
import yaml
from itertools import cycle

from plotly.offline import init_notebook_mode, iplot


# spatial libraries
import pyproj
import shapely.geometry
from shapely.geometry import Point, LineString
from shapely.ops import transform

# package(s) related to the simulation
import simpy

# OpenTNSim
from opentnsim.graph import utils
from opentnsim.graph import mixins as graph_module
from opentnsim.core import Identifiable, Locatable

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Determine the wgs84 geoid
wgs84 = pyproj.Geod(ellps="WGS84")


def get_geometry_of_edge(graph, edge):
    """get the geometry of the edge in WGS84

    Parameters
    ----------
    graph: networkx.Graph
        The graph object.
    edge : tuple
        The edge to get the length of. is a tuple of two node-names.

    Returns
    -------
    float
        The length of the edge in meters.
    """

    edge_info = graph.edges[edge]
    if "geometry" not in edge_info:
        orig = nx.get_node_attributes(graph, "geometry")[edge[0]]
        dest = nx.get_node_attributes(graph, "geometry")[edge[1]]
        geometry = LineString([orig,dest])
        graph.edges[edge]["geometry"] = geometry
    else:
        geometry = graph.edges[edge]["geometry"]

    coordinates_x = geometry.coords.xy[0]
    coordinates_y = geometry.coords.xy[1]
    min_coordinates_x = np.min(coordinates_x)
    max_coordinates_x = np.max(coordinates_x)
    min_coordinates_y = np.min(coordinates_y)
    max_coordinates_y = np.max(coordinates_y)
    if not isinstance(edge_info["geometry"], shapely.geometry.LineString):
        raise ValueError(f"Edge geometry in edge {edge}: attribute must be a shapely LineString.")
    if min_coordinates_x < -180. or max_coordinates_x > 180. or min_coordinates_y < -90. or max_coordinates_y > 90.:
        raise ValueError(f"Edge geometry in edge {edge}: attribute is not defined in WGS84.")

    return geometry


def determine_length_of_edge_geometry(graph, edge, current_crs="EPSG:4326", crs_meter="EPSG:4087"):
    wgs84 = pyproj.CRS(current_crs)
    wgs84_m = pyproj.CRS(crs_meter)
    wgs84_to_wgs84_m = pyproj.transformer.Transformer.from_crs(wgs84, wgs84_m, always_xy=True).transform
    geometry = get_geometry_of_edge(graph, edge)
    geometry_m = transform(wgs84_to_wgs84_m, geometry)
    length_m = geometry_m.length
    return length_m


def get_length_of_edge(graph, edge, current_crs="EPSG:4326", crs_meter="EPSG:4087"):
    """get the length of an edge in meters

    Parameters
    ----------
    graph: networkx.Graph
        The graph object.
    edge : tuple
        The edge to get the length of. is a tuple of two node-names.

    Returns
    -------
    float
        The length of the edge in meters.
    """

    edge_info = graph.edges[edge]
    if "length_m" in edge_info:
        pass
    else:
        length_m = determine_length_of_edge_geometry(graph, edge, current_crs, crs_meter)
        graph.edges[edge]["length_m"] = length_m

    return edge_info["length_m"]


def find_closest_node(G, point):
    """find the closest node on the graph from a given point"""

    distance = np.full((len(G.nodes)), fill_value=np.nan)
    for ii, n in enumerate(G.nodes):
        distance[ii] = point.distance(G.nodes[n]["geometry"])
    name_node = list(G.nodes)[np.argmin(distance)]
    distance_node = np.min(distance)

    return name_node, distance_node


def calculate_distance(geom_start, geom_stop):
    """method to calculate the distance (as the bird flies) in meters between two geometries

    Parameters
    ----------
    geom_start : shapely.geometry.Point
        Starting point geometry. must contain x and y attributes.
    geom_stop : shapely.geometry.Point
        Stopping point geometry. must contain x and y attributes.

    Returns
    -------
    float
        Distance in meters between the two geometries.
    """

    wgs84 = pyproj.Geod(ellps="WGS84")

    # distance between two points
    return float(wgs84.inv(geom_start.x, geom_start.y, geom_stop.x, geom_stop.y)[2])


def calculate_distance_along_path(graph, path):
    """method to calculate the greater circle distance along path in meters from WGS84 coordinates

    Parameters
    ----------
    graph : networkx.Graph
        The graph object.
    path : list
        List of nodes that together form a path.

    Returns
    -------
    float
        Path length in meters.
    """

    path_length = 0

    for node in enumerate(path[:-1]):
        orig = nx.get_node_attributes(graph, "geometry")[path[node[0]]]
        dest = nx.get_node_attributes(graph, "geometry")[path[node[0]+1]]
        path_length += calculate_distance(orig, dest)

        if node[0] + 2 == len(path):
                    break

    return path_length

def calculate_depth(geom_start, geom_stop, graph):
    """method to calculate the depth of the waterway in meters between two geometries.

    Parameters
    ----------
    geom_start : shapely.geometry.Point
        Starting point geometry. Must represent a node in graph graph.
    geom_stop : shapely.geometry.Point
        Stopping point geometry. must represent a node in graph graph.
    graph : networkx.Graph
        The graph containing vaarweginformatie.nl data, with nodes and edges.
        Must contain 'Info' attribute on edges with 'GeneralDepth'.
        Must contain an edge between geom_start and geom_stop.

    Returns
    -------
    float
        The depth of the waterway between the two geometries in meters.

    Raises
    ------
    ValueError
        If geom_start or geom_stop are not nodes in the graph graph.
        If there is no edge between the two nodes in the graph graph.
        If the depth data is not available for the edge between the two nodes.
    """

    depth = 0

    # The node on the graph of vaarweginformatie.nl closest to geom_start and geom_stop

    node_start = find_closest_node(graph, geom_start)[0]
    node_stop = find_closest_node(graph, geom_stop)[0]

    # Read from the graph data from vaarweginformatie.nl the General depth of each edge
    # TODO: check it this needs to be made more general, now relies on ['Info'] to be present
    if node_start == node_stop:
        return np.nan  # if the start and stop nodes are the same, return 0 depth

    try:
        if "Info" in graph.get_edge_data(node_start, node_stop).keys():
            depth = graph.get_edge_data(node_start, node_stop)["Info"]["GeneralDepth"]

        elif "GeneralDepth" in graph.get_edge_data(node_start, node_stop).keys():
            depth = graph.get_edge_data(node_start, node_stop)["GeneralDepth"]
        else:
            return np.nan  # if no depth data is available, return NaN
    except:
        depth = np.nan  # When there is no data of the depth available of this edge, it gives a message

    h_0 = depth

    # depth of waterway between two points
    return h_0


def geom_to_edges(geom, properties):
    """Generate edges from a geometry, yielding an edge id and edge properties. The edge_id consists of a tuple of coordinates"""
    if geom.geom_type not in ["LineString", "MultiLineString"]:
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
    graph = nx.DiGraph()
    for _, feature in gdf.iterrows():
        geom = feature.geometry
        if geom is None:
            raise nx.NetworkXError("Bad data: feature missing geometry")
        properties = feature.drop(labels=["geometry"])
        # in case we have single points in the geometry, add them as nodes
        if geom.geom_type == "Point":
            node_idx = geom.coords[0]
            graph.add_node(node_idx, **properties)
            continue
        if geom.geom_type in ["LineString", "MultiLineString"]:
            for edge_id, edge_properties in geom_to_edges(geom, properties):
                node_source, node_target = edge_properties["e"]
                source_geom = shapely.geometry.Point(*node_source)
                _, node_properties = geom_to_node(source_geom, {})
                graph.add_node(edge_id[0], **node_properties)
                _, node_properties = geom_to_node(source_geom, {})
                graph.add_node(edge_id[1], **node_properties)
                graph.add_edge(edge_id[0], edge_id[1], **edge_properties)
    return graph


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
        if hasattr(self,"env"):
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


def plot_graph(graph, static: bool = False):
    """method to plot a graph

    Parameters
    ----------
    graph : networkx.Graph
        A graph object.
    static : bool, optional
        If True, returns a static Plotly figure object.
        If False, displays the figure

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Object that contains a graph figure.
    """

    # Labels
    labels = {node: node for node in graph.nodes()}
    edge_labels = {(u, v): f"{d['weight']} km" for u, v, d in graph.edges(data=True)}

    # positions
    positions = {node: (graph.nodes[node]["geometry"].x, graph.nodes[node]["geometry"].y) for node in graph.nodes}

    # Edge labels in meters
    edge_labels = {}
    for u, v in graph.edges():
        origin = graph.nodes[u]['geometry']
        destination = graph.nodes[v]['geometry']
        distance_m = graph_module.calculate_distance(origin, destination)
        edge_labels[(u, v)] = f"{int(distance_m)} m"

    # Edge traces and arrow annotations
    edge_traces = []
    arrow_annotations = []
    for u, v in graph.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(width=2, color='red'),
            mode='lines',
            hoverinfo='none'
        ))
        arrow_annotations.append(go.layout.Annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,  # Closed arrowhead
            arrowsize=2,
            arrowwidth=1,
            arrowcolor='red'
        ))

    # Node trace
    node_trace = go.Scatter(
        x=[positions[node][0] for node in graph.nodes()],
        y=[positions[node][1] for node in graph.nodes()],
        mode='markers+text',
        marker=dict(color='darkblue', size=20),
        text=[labels[node] for node in graph.nodes()],
        textposition='middle center',
        textfont=dict(color='white', size=15),
        hoverinfo='text'
    )

    # Edge label annotations
    edge_label_annotations = []
    for (u, v), label in edge_labels.items():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        x_mid = (x0 + x1) / 2
        y_mid = (y0 + y1) / 2
        edge_label_annotations.append(go.layout.Annotation(
            x=x_mid, y=y_mid,
            text=label,
            showarrow=False,
            font=dict(color='black', size=15)
        ))

    # Combine annotations
    annotations = arrow_annotations + edge_label_annotations

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Directed Geographic Network Graph (WGS84 Projection) with Edge Lengths in Meters",
        xaxis=dict(title="Longitude", showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title="Latitude", showgrid=False, zeroline=False, showticklabels=False),
        annotations=annotations,
        showlegend=False,
        plot_bgcolor='white',
    )

    if static is False:
        # Initialize notebook mode for Plotly
        init_notebook_mode(connected=True)
        # Display the figure in a Jupyter notebook
        iplot(fig)
    else:
        return fig
