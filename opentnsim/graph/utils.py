"""Utility functions for OpenTNSim graphs."""

# packkage(s) for documentation, debugging, saving and loading
import warnings

# numerical libraries
import numpy as np

# spatial libraries
import pandas as pd
import math
import networkx as nx
import shapely
import pyproj
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import split, snap
from shapely.strtree import STRtree
from scipy.spatial import cKDTree

class NetworkWarning(Warning):
    pass


def expand_path_edges(G, node_path):
    edge_paths = [[]]
    is_multidigraph = G.is_multigraph()

    for u, v in zip(node_path[:-1], node_path[1:]):
        edges = G.get_edge_data(u, v)
        new_paths = []

        if is_multidigraph:
            for k in edges:
                for path in edge_paths:
                    new_paths.append(path + [(u, v, k)])
        else:
            for path in edge_paths:
                new_paths.append(path + [(u, v)])

        edge_paths = new_paths

    return edge_paths


def node_path_to_edge_path(G, node_path, weight="weight"):
    """
    Convert a node path to an edge path, selecting the lowest weight edge
    for MultiDiGraph. Works for both DiGraph and MultiDiGraph.

    Parameters
    ----------
    G : networkx.Graph or networkx.MultiDiGraph
        The graph.
    node_path : list
        Sequence of nodes from a path.
    weight : str, optional
        Edge attribute used as weight. Default is 'weight'.

    Returns
    -------
    edge_path : list
        For DiGraph: [(u,v), ...]
        For MultiDiGraph: [(u,v,k), ...]
    """
    edge_path = []
    is_multidigraph = G.is_multigraph()

    for u, v in zip(node_path[:-1], node_path[1:]):
        edges = G.get_edge_data(u, v)

        if edges is None:
            raise ValueError(f"No edge between {u} and {v} in graph")

        if is_multidigraph:
            best_k = None
            best_weight = float("inf")

            for k, data in edges.items():
                w = data.get(weight)
                if w is None:
                    if best_k is None:
                        best_k = k
                else:
                    if w < best_weight:
                        best_weight = w
                        best_k = k

            edge_path.append((u, v, best_k))
        else:
            edge_path.append((u, v))

    return edge_path


def build_graph_spatial_index(graph):
    """
    Build spatial indexes for nodes and edges and store them in graph.graph.
    """
    # Avoid rebuilding
    if "node_lookup" in graph.graph and len(graph.graph['node_lookup']) == len(graph.nodes):
        return

    node_geoms = []
    node_lookup = []
    for n, data in graph.nodes(data=True):
        geom = data.get("geometry")
        if geom is not None and not geom.is_empty:
            node_geoms.append((geom.x, geom.y))
            node_lookup.append(n)

    node_coords = np.array(node_geoms)
    node_tree = cKDTree(node_coords)

    graph.graph["node_kdtree"] = node_tree
    graph.graph["node_lookup"] = node_lookup

    # ---- Edges ----
    edge_geoms = []
    edge_lookup = {}
    for u, v, data in graph.edges(data=True):
        geom = data.get("geometry")
        if geom is not None and not geom.is_empty:
            edge_geoms.append(geom)
            edge_lookup[geom] = (u, v)

    edge_tree = STRtree(edge_geoms)
    graph.graph["edge_spatial_tree"] = edge_tree
    graph.graph["edge_lookup"] = edge_lookup


def check_graph_is_multidigraph_type(graph):
    is_multidigraph = False
    if isinstance(graph, nx.MultiGraph) or isinstance(graph, nx.MultiDiGraph):
        is_multidigraph = True
    return is_multidigraph


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
    is_multidigraph = check_graph_is_multidigraph_type(graph)
    if not is_multidigraph:
        edge_info = graph.edges[edge[:2]]
    else:
        edge_info = graph.edges[edge]

    if "geometry" not in edge_info:
        orig = nx.get_node_attributes(graph, "geometry")[edge[0]]
        dest = nx.get_node_attributes(graph, "geometry")[edge[1]]
        geometry = LineString([orig, dest])
        edge_info["geometry"] = geometry
    else:
        geometry = edge_info["geometry"]

    coordinates_x = geometry.coords.xy[0]
    coordinates_y = geometry.coords.xy[1]
    min_coordinates_x = np.min(coordinates_x)
    max_coordinates_x = np.max(coordinates_x)
    min_coordinates_y = np.min(coordinates_y)
    max_coordinates_y = np.max(coordinates_y)
    if not isinstance(edge_info["geometry"], shapely.geometry.LineString):
        raise ValueError(f"Edge geometry in edge {edge}: attribute must be a shapely LineString.")
    if min_coordinates_x < -180.0 or max_coordinates_x > 180.0 or min_coordinates_y < -90.0 or max_coordinates_y > 90.0:
        raise ValueError(f"Edge geometry in edge {edge}: attribute is not defined in WGS84.")

    return geometry


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
    from opentnsim.graph.calculations import calculate_length_of_edge
    edge_info = graph.edges[edge]
    if "length_m" in edge_info:
        pass
    else:
        length_m = calculate_length_of_edge(graph, edge, current_crs, crs_meter)
        graph.edges[edge]["length_m"] = length_m

    return edge_info["length_m"]


def get_longest_common_subroute(route1, route2):
    n, m = len(route1), len(route2)
    dp = [[[] for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if route1[i] == route2[j]:
                dp[i + 1][j + 1] = dp[i][j] + [route1[i]]
            else:
                dp[i + 1][j + 1] = max(
                    dp[i][j + 1],
                    dp[i + 1][j],
                    key=len
                )
    return dp[n][m]


def find_closest_node(graph, point):
    """
    Find the closest node to a Shapely Point using KDTree.
    """
    build_graph_spatial_index(graph)

    tree = graph.graph["node_kdtree"]
    lookup = graph.graph["node_lookup"]

    _, idx = tree.query([point.x, point.y])
    closest_node = lookup[idx]

    return closest_node


def find_closest_edge(graph, point):
    """
    Find the closest edge to a Shapely Point using STRtree.
    """
    build_graph_spatial_index(graph)

    tree = graph.graph["edge_spatial_tree"]
    lookup = graph.graph["edge_lookup"]

    idx = tree.nearest(point)
    geom = tree.geometries[idx]
    edge = lookup[geom]

    return edge


def network_check(graph):
    """Assertions about the graphs used in OpenTNSim"""
    # TODO Determine where we should save this function. This function is not called by any mixins.
    node_type = (str, shapely.Point)
    ok = True

    if not isinstance(graph, nx.Graph):
        warnings.warn("graph should be of type nx.Graph", NetworkWarning)
        ok = False
    if len(graph.nodes) < 2:
        warnings.warn("there should be at least 2 nodes in the graph", NetworkWarning)
        ok = False
    if len(graph.edges) < 1:
        warnings.warn("there should be at least 1 edge in the graph", NetworkWarning)
        ok = False
    if not all(isinstance(n, node_type) for n in graph.nodes.keys()):
        warnings.warn("all keys should be str or Point", NetworkWarning)
        ok = False
    # check all edges
    for e in graph.edges.keys():
        if len(e) != 2:
            warnings.warn(f"edge keys should be tuples of length 2, {e} was not", NetworkWarning)
            ok = False
            # stop checking
            break

        source, _ = e
        if not isinstance(source, node_type):
            warnings.warn(f"edges should be of a tuple of Points or str, {e} was not ", NetworkWarning)
            ok = False
            # stop checking
            break
    for e, edge in graph.edges.items():
        if "geometry" not in edge:
            warnings.warn(f"edges should have of geometry attribute, {e} did not ", NetworkWarning)
            ok = False
        elif not isinstance(edge["geometry"], (str, shapely.Geometry)):
            warnings.warn(f"edges geometry should be of type string or Geometry, {edge['geometry']} was not.", NetworkWarning)
            ok = False
    return ok


# Ignore functions copied from networkx
# // START-NOSCAN
def read_shp(path, simplify=True, geom_attrs=True, strict=True):
    """Generates a networkx.DiGraph from shapefiles.

       read_shp used to be part of NetworkX.
       See https://networkx.org/documentation/latest/auto_examples/index.html#geospatial.

    Point geometries are
    translated into nodes, lines into edges. Coordinate tuples are used as
    keys. Attributes are preserved, line geometries are simplified into start
    and end coordinates. Accepts a single shapefile or directory of many
    shapefiles.

    "The Esri Shapefile or simply a shapefile is a popular geospatial vector
    data format for geographic information systems software [1]_."

    Parameters
    ----------
    path : file or string
       File, directory, or filename to read.

    simplify:  bool
        If True, simplify line geometries to start and end coordinates.
        If False, and line feature geometry has multiple segments, the
        non-geometric attributes for that feature will be repeated for each
        edge comprising that feature.

    geom_attrs: bool
        If True, include the Wkb, Wkt and Json geometry attributes with
        each edge.

        NOTE:  if these attributes are available, write_shp will use them
        to write the geometry.  If nodes store the underlying coordinates for
        the edge geometry as well (as they do when they are read via
        this method) and they change, your geomety will be out of sync.

    strict: bool
        If True, raise NetworkXError when feature geometry is missing or
        GeometryType is not supported.
        If False, silently ignore missing or unsupported geometry in features.

    Returns
    -------
    G : NetworkX graph

    Raises
    ------
    ImportError
       If ogr module is not available.

    RuntimeError
       If file cannot be open or read.

    NetworkXError
       If strict=True and feature is missing geometry or GeometryType is
       not supported.

    Examples
    --------
    >>> G = nx.read_shp("test.shp")  # doctest: +SKIP

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Shapefile
    """
    try:
        from osgeo import ogr
    except ImportError as err:
        raise ImportError("read_shp requires OGR: https://www.gdal.org/") from err

    if not isinstance(path, str):
        return

    net = nx.DiGraph()
    shp = ogr.Open(path)
    if shp is None:
        raise RuntimeError(f"Unable to open {path}")
    for lyr in shp:
        fields = [x.GetName() for x in lyr.schema]
        for f in lyr:
            g = f.geometry()
            if g is None:
                if strict:
                    raise nx.NetworkXError("Bad data: feature missing geometry")
                else:
                    continue
            flddata = [f.GetField(f.GetFieldIndex(x)) for x in fields]
            attributes = dict(zip(fields, flddata))
            attributes["ShpName"] = lyr.GetName()
            # Note:  Using layer level geometry type
            if g.GetGeometryType() == ogr.wkbPoint:
                net.add_node((g.GetPoint_2D(0)), **attributes)
            elif g.GetGeometryType() in (ogr.wkbLineString, ogr.wkbMultiLineString):
                for edge in edges_from_line(g, attributes, simplify, geom_attrs):
                    e1, e2, attr = edge
                    net.add_edge(e1, e2)
                    net[e1][e2].update(attr)
            else:
                if strict:
                    raise nx.NetworkXError(f"GeometryType {g.GetGeometryType()} not supported")

    return net


def edges_from_line(geom, attrs, simplify=True, geom_attrs=True):
    """
    Generate edges for each line in geom
    Written as a helper for read_shp

    Parameters
    ----------

    geom:  ogr line geometry
        To be converted into an edge or edges

    attrs:  dict
        Attributes to be associated with all geoms

    simplify:  bool
        If True, simplify the line as in read_shp

    geom_attrs:  bool
        If True, add geom attributes to edge as in read_shp


    Returns
    -------
     edges:  generator of edges
        each edge is a tuple of form
        (node1_coord, node2_coord, attribute_dict)
        suitable for expanding into a networkx Graph add_edge call

    .. deprecated:: 2.6
    """
    msg = (
        "edges_from_line is deprecated and will be removed in 3.0."
        "See https://networkx.org/documentation/latest/auto_examples/index.html#geospatial."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    try:
        from osgeo import ogr
    except ImportError as err:
        raise ImportError("edges_from_line requires OGR: " "https://www.gdal.org/") from err

    if geom.GetGeometryType() == ogr.wkbLineString:
        if simplify:
            edge_attrs = attrs.copy()
            last = geom.GetPointCount() - 1
            if geom_attrs:
                edge_attrs["Wkb"] = geom.ExportToWkb()
                edge_attrs["Wkt"] = geom.ExportToWkt()
                edge_attrs["Json"] = geom.ExportToJson()
            yield (geom.GetPoint_2D(0), geom.GetPoint_2D(last), edge_attrs)
        else:
            for i in range(0, geom.GetPointCount() - 1):
                pt1 = geom.GetPoint_2D(i)
                pt2 = geom.GetPoint_2D(i + 1)
                edge_attrs = attrs.copy()
                if geom_attrs:
                    segment = ogr.Geometry(ogr.wkbLineString)
                    segment.AddPoint_2D(pt1[0], pt1[1])
                    segment.AddPoint_2D(pt2[0], pt2[1])
                    edge_attrs["Wkb"] = segment.ExportToWkb()
                    edge_attrs["Wkt"] = segment.ExportToWkt()
                    edge_attrs["Json"] = segment.ExportToJson()
                    del segment
                yield (pt1, pt2, edge_attrs)

    elif geom.GetGeometryType() == ogr.wkbMultiLineString:
        for i in range(geom.GetGeometryCount()):
            geom_i = geom.GetGeometryRef(i)
            yield from edges_from_line(geom_i, attrs, simplify, geom_attrs)


def write_shp(G, outdir):
    """Writes a networkx.DiGraph to two shapefiles, edges and nodes.

       write_shp used to be part of networx.
       See https://networkx.org/documentation/latest/auto_examples/index.html#geospatial.

    Nodes and edges are expected to have a Well Known Binary (Wkb) or
    Well Known Text (Wkt) key in order to generate geometries. Also
    acceptable are nodes with a numeric tuple key (x,y).

    "The Esri Shapefile or simply a shapefile is a popular geospatial vector
    data format for geographic information systems software [1]_."

    Parameters
    ----------
    G : NetworkX graph
        Directed graph
    outdir : directory path
       Output directory for the two shapefiles.

    Returns
    -------
    None

    Examples
    --------
    nx.write_shp(digraph, '/shapefiles') # doctest +SKIP

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Shapefile
    """
    try:
        from osgeo import ogr
    except ImportError as err:
        raise ImportError("write_shp requires OGR: https://www.gdal.org/") from err
    # easier to debug in python if ogr throws exceptions
    ogr.UseExceptions()

    def netgeometry(key, data):
        if "Wkb" in data:
            geom = ogr.CreateGeometryFromWkb(data["Wkb"])
        elif "Wkt" in data:
            geom = ogr.CreateGeometryFromWkt(data["Wkt"])
        elif type(key[0]).__name__ == "tuple":  # edge keys are packed tuples
            geom = ogr.Geometry(ogr.wkbLineString)
            _from, _to = key[0], key[1]
            try:
                geom.SetPoint(0, *_from)
                geom.SetPoint(1, *_to)
            except TypeError:
                # assume user used tuple of int and choked ogr
                _ffrom = [float(x) for x in _from]
                _fto = [float(x) for x in _to]
                geom.SetPoint(0, *_ffrom)
                geom.SetPoint(1, *_fto)
        else:
            geom = ogr.Geometry(ogr.wkbPoint)
            try:
                geom.SetPoint(0, *key)
            except TypeError:
                # assume user used tuple of int and choked ogr
                fkey = [float(x) for x in key]
                geom.SetPoint(0, *fkey)

        return geom

    # Create_feature with new optional attributes arg (should be dict type)
    def create_feature(geometry, lyr, attributes=None):
        feature = ogr.Feature(lyr.GetLayerDefn())
        feature.SetGeometry(g)
        if attributes is not None:
            # Loop through attributes, assigning data to each field
            for field, data in attributes.items():
                feature.SetField(field, data)
        lyr.CreateFeature(feature)
        feature.Destroy()

    # Conversion dict between python and ogr types
    OGRTypes = {int: ogr.OFTInteger, str: ogr.OFTString, float: ogr.OFTReal}

    # Check/add fields from attribute data to Shapefile layers
    def add_fields_to_layer(key, value, fields, layer):
        # Field not in previous edges so add to dict
        if type(value) in OGRTypes:
            fields[key] = OGRTypes[type(value)]
        else:
            # Data type not supported, default to string (char 80)
            fields[key] = ogr.OFTString
        # Create the new field
        newfield = ogr.FieldDefn(key, fields[key])
        layer.CreateField(newfield)

    drv = ogr.GetDriverByName("ESRI Shapefile")
    shpdir = drv.CreateDataSource(outdir)
    # delete pre-existing output first otherwise ogr chokes
    try:
        shpdir.DeleteLayer("nodes")
    except:
        pass
    nodes = shpdir.CreateLayer("nodes", None, ogr.wkbPoint)

    # Storage for node field names and their data types
    node_fields = {}

    def create_attributes(data, fields, layer):
        attributes = {}  # storage for attribute data (indexed by field names)
        for key, value in data.items():
            # Reject spatial data not required for attribute table
            if key != "Json" and key != "Wkt" and key != "Wkb" and key != "ShpName":
                # Check/add field and data type to fields dict
                if key not in fields:
                    add_fields_to_layer(key, value, fields, layer)
                # Store the data from new field to dict for CreateLayer()
                attributes[key] = value
        return attributes, layer

    for n in G:
        data = G.nodes[n]
        g = netgeometry(n, data)
        attributes, nodes = create_attributes(data, node_fields, nodes)
        create_feature(g, nodes, attributes)

    try:
        shpdir.DeleteLayer("edges")
    except:
        pass
    edges = shpdir.CreateLayer("edges", None, ogr.wkbLineString)

    # New edge attribute write support merged into edge loop
    edge_fields = {}  # storage for field names and their data types

    for edge in G.edges(data=True):
        data = G.get_edge_data(*edge)
        g = netgeometry(edge, data)
        attributes, edges = create_attributes(edge[2], edge_fields, edges)
        create_feature(g, edges, attributes)

    nodes, edges = None, None


def info(G, n=None):
    """Return a summary of information for the graph G or a single node n.

    The summary includes the number of nodes and edges, or neighbours for a single
    node.

    Parameters
    ----------
    G : Networkx graph
       A graph
    n : node (any hashable)
       A node in the graph G

    Returns
    -------
    info : str
        A string containing the short summary

    Raises
    ------
    NetworkXError
        If n is not in the graph G

    .. deprecated:: 2.7
       ``info`` is deprecated and will be removed in NetworkX 3.0.
    """
    if n is None:
        return str(G)
    if n not in G:
        raise nx.NetworkXError(f"node {n} not in graph")
    info = ""  # append this all to a string
    info += f"Node {n} has the following properties:\n"
    info += f"Degree: {G.degree(n)}\n"
    info += "Neighbors: "
    info += " ".join(str(nbr) for nbr in G.neighbors(n))
    return info


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
                node_source, _ = edge_properties["e"]
                source_geom = shapely.geometry.Point(*node_source)
                _, node_properties = geom_to_node(source_geom, {})
                graph.add_node(edge_id[0], **node_properties)
                _, node_properties = geom_to_node(source_geom, {})
                graph.add_node(edge_id[1], **node_properties)
                graph.add_edge(edge_id[0], edge_id[1], **edge_properties)
    return graph


def get_trajectory(graph, node_1, node_2):
    from opentnsim.graph.calculations import reverse_geometry
    geometry = None
    route = nx.dijkstra_path(graph, node_1, node_2)
    is_multidigraph = check_graph_is_multidigraph_type(graph)
    edge_length_m = 0
    for edge in zip(route[:-1], route[1:]):
        edge = get_edge(graph, edge, is_multidigraph)
        edge_geometry = graph.edges[edge]['geometry']
        edge_length_m += graph.edges[edge]['length_m']
        aligned = check_if_geometry_is_aligned_with_edge(graph, edge)
        if not aligned:
            edge_geometry = reverse_geometry(edge_geometry)

        if geometry:
            geometry = shapely.ops.linemerge(MultiLineString([geometry, edge_geometry]))
        else:
            geometry = edge_geometry

    return geometry, edge_length_m


def get_trajectory_between_locations(graph, point_1, point_2, tolerance = 0.0001):
    edge_1 = find_closest_edge(graph, point_1)
    edge_2 = find_closest_edge(graph, point_2)

    route = get_largest_route_between_edges(graph, edge_1, edge_2)
    geometry = get_trajectory(graph, route[0], route[-1])[0]

    lines_1 = split(snap(geometry, point_1, tolerance), point_1).geoms
    distance = math.inf
    splitting_index = 0
    for index, line_1 in enumerate(lines_1):
        distance_to_line = line_1.distance(point_2)
        if distance_to_line < distance:
            splitting_index = index
            distance = distance_to_line

    geometry = lines_1[splitting_index]
    geometry = split(snap(geometry, point_2, tolerance), point_2).geoms[splitting_index]
    return geometry


def get_closest_location_on_edge_to_point(graph, edge, point):
    edge_geometry = graph.edges[edge]["geometry"]
    point_on_edge = edge_geometry.interpolate(edge_geometry.project(point))
    return point_on_edge


def check_if_geometry_is_aligned_with_edge(graph, edge):
    node_start = edge[0]
    node_stop = edge[1]
    edge_geometry = get_geometry_of_edge(graph, edge)
    first_point = Point(edge_geometry.coords[0])
    distance_to_edge_nodes = {}
    for node in [node_start, node_stop]:
        node_geometry = graph.nodes[node]["geometry"]
        distance_to_edge_nodes[node] = first_point.distance(node_geometry)
    closest_node = min(distance_to_edge_nodes, key=distance_to_edge_nodes.get)
    aligned = closest_node == node_start
    return aligned


def get_edge_at_distance_from_node(graph, node_1, node_2, distance):
    route = nx.dijkstra_path(graph, node_1, node_2)
    total_length = 0
    edge = None
    is_multidigraph = check_graph_is_multidigraph_type(graph)
    for edge in zip(route[:-1], route[1:]):
        edge = get_edge(graph, edge, is_multidigraph)
        edge_length = graph.edges[edge]['length_m']
        total_length += edge_length
        if total_length < distance:
            continue
        break
    return edge


def get_edge(graph, edge, is_multidigraph=False):
    edge = edge[:2]
    if is_multidigraph:
        k = sorted(graph[edge[0]][edge[1]], key=lambda x: get_length_of_edge(graph, (edge[0], edge[1], x)))[0]
        edge = (edge[0], edge[1], k)
    return edge


def _get_edges_from_geometry(graph, geometry, crs_m, m=False):
    if m:
        from opentnsim.graph.calculations import transform_geometry
        geometry = transform_geometry(geometry, epsg_in = crs_m, epsg_out = "EPSG:4326")
    edges = []
    for edge in graph.edges:
        if geometry.intersects(graph.edges[edge]["geometry"]):
            edges.append(edge)
    return edges


def get_edges(graph, route):
    edges = []
    is_multidigraph = check_graph_is_multidigraph_type(graph)
    for _, edge in enumerate(zip(route[:-1], route[1:])):
        edge = get_edge(graph, edge, is_multidigraph)
        edges.append(edge)
    return edges


def create_transformer(crs_in = "EPSG:4326", crs_out = "EPSG:4087"):
    transformer = pyproj.Transformer.from_crs(crs_in, crs_out, always_xy=True).transform
    return transformer


def get_edges_at_a_distance(graph, start_node, end_node, threshold):
    """
    Returns all edges (u, v) where the shortest-path distance
    from end_node to u and v lie on opposite sides of threshold.
    """

    # Copy graph to prevent pollution of graph
    graph_copy = graph.copy()

    # Block the input edge so we only move forward
    if graph_copy.has_edge(start_node, end_node):
        graph_copy.remove_edge(start_node, end_node)
    if graph_copy.has_edge(end_node, start_node):
        graph_copy.remove_edge(end_node, start_node)

    # Compute shortest-path distances from end_node
    distances = nx.single_source_dijkstra_path_length(graph_copy, source=end_node, weight='length_m')
    crossing_edges = []
    for edge in graph_copy.edges:
        u, v = edge[:2]
        if u not in distances or v not in distances:
            continue
        du = distances[u]
        dv = distances[v]

        # Check if threshold lies strictly between them
        if (du < threshold and dv > threshold) or (dv < threshold and du > threshold):
            crossing_edges.append(edge)

    return crossing_edges


def _get_all_simple_edge_paths(G, source, target, cutoff=None):
    def dfs(current, target, visited, path, depth):
        if cutoff is not None and depth > cutoff:
            return

        if current == target:
            yield list(path)
            return

        neighbors = G.successors(current) if G.is_directed() else G.neighbors(current)

        for neighbor in neighbors:
            if neighbor in visited:
                continue

            if G.is_multigraph():
                edge_iter = G[current][neighbor].items()
            else:
                edge_iter = [(None, G[current][neighbor])]

            for key, data in edge_iter:
                if G.is_multigraph():
                    path.append((current, neighbor, key))
                else:
                    path.append((current, neighbor))

                visited.add(neighbor)

                yield from dfs(neighbor, target, visited, path, depth + 1)

                visited.remove(neighbor)
                path.pop()

    yield from dfs(source, target, {source}, [], 0)


def get_sailing_distance(graph, edge_route):
    """
    Calculates sailing distance of a route

    Parameters
    ----------
    vessel :
        a type including the following parent-classes: Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    edge : tuple
        tuple resembles an edge with: a start_node [u] as str, end_node (v) as str

    Returns
    -------
    sailing_distance_over_route : float
        sailing distance along the route in [m]
    """

    # calculate sailing distance along route
    sailing_distance = 0
    sailing_distance_df = pd.DataFrame(columns=['node_start','node_stop','distance'])
    for edge in edge_route:
        edge_distance = graph.edges[edge]['length_m']
        sailing_distance += edge_distance
        sailing_distance_df.loc[len(sailing_distance_df),:] = [edge[0],edge[1],edge_distance]

    return sailing_distance, sailing_distance_df


def get_edge_speed(vessel, graph, edge):
    edge_info = graph.edges[edge]
    sailing_speed = vessel.v
    if 'restricted_speed' in edge_info.keys():
        restricted_speed = edge_info['restricted_speed']
        if sailing_speed > restricted_speed:
            sailing_speed = restricted_speed
    if 'overruled_speed' in edge_info.keys():
        sailing_speed = edge_info['overruled_speed']
    return sailing_speed


def get_sailing_speed(vessel, graph, edge_route):
    """
    Provides the speed along a vessel's route

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    route : list of str
        str resemble node names that have to be in the graph

    Returns
    vessel_speed_over_route : pd.DataFrame
        vessel speed per edge of the route
    -------
    """
    # construct dataframe of speed information per edge
    vessel_sailing_speed_df = pd.DataFrame(columns=['node_start','node_stop','speed'])
    total_sailing_distance = 0.
    total_sailing_time = 0.
    average_sailing_speed = 0.
    for edge in edge_route:
        edge_info = graph.edges[edge]
        sailing_speed = get_edge_speed(vessel, graph, edge)

        sailing_distance = edge_info['length_m']
        total_sailing_distance += sailing_distance
        total_sailing_time += sailing_distance/sailing_speed

        vessel_sailing_speed_df.loc[len(vessel_sailing_speed_df),:] = [edge[0],edge[1],sailing_speed]
    if total_sailing_time:
        average_sailing_speed = total_sailing_distance/total_sailing_time
    return average_sailing_speed, vessel_sailing_speed_df


def get_sailing_time(vessel, edge_route):
    """
    Calculates sailing time of vessel

    Parameters
    ----------
    vessel :
        a type including the following parent-classes: Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    route : list of str
        str resemble node names that have to be in the graph
    edges : list of tuples
        tuples resemble edges with: a start_node [u] as str, end_node (v) as str, and identifier (k) as int

    Returns
    -------
    sailing_time_over_route : pd.DataFrame
        dataframe with edges as (multi)index and the following column-information: Speed, Distance, Time

    """
    graph = vessel.env.graph
    _, sailing_distance_df = get_sailing_distance(graph, edge_route)
    _, vessel_sailing_speed_df = get_sailing_speed(vessel, graph, edge_route)
    sailing_time_df = pd.merge(sailing_distance_df,vessel_sailing_speed_df)
    sailing_time_df['time'] = sailing_time_df['distance'] / sailing_time_df['speed']
    sailing_time = sailing_time_df['time'].sum()
    return sailing_time, sailing_time_df


def get_heading(vessel, graph, edge):
    is_multidigraph = check_graph_is_multidigraph_type(graph)
    edge = get_edge(graph, edge, is_multidigraph)
    edge_geometry = vessel.env.graph.edges[edge]["geometry"]
    heading = np.degrees(math.atan2(edge_geometry.coords[0][0] - edge_geometry.coords[-1][0],
                                    edge_geometry.coords[0][1] - edge_geometry.coords[-1][1]))
    return heading


def get_sailing_information_on_edge_to_distance_on_another_edge(
        vessel, edge_route, distance_sailed_on_first_edge=0., distance_to_be_sailed_on_last_edge=0.
):
    """
    Calculates the distance from a location along an edge A to another location along an edge B

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    route : list of str
        str resemble node names that have to be in the graph
    distance_sailed_on_first_edge : float
        distance that is already covered on the edge at which the vessel is currently sailing
    distance_to_be_sailed_on_last_edge : float
        distance on the last edge that the vessel has to sail to reach its location of interest
    edges : list of tuples
        tuples resemble edges with: a start_node [u] as str, end_node (v) as str, and identifier (k) as int

    Returns
    -------
    sailing_information_df : pd.DataFrame
        dataframe with edges as (multi)index and the following column-information: Speed, Distance, Time

    """

    # obtain dataframe with information of sailing speed, distance and time along route
    _, sailing_information_df = get_sailing_time(vessel=vessel, edge_route=edge_route)

    # determine indexes of first and last edges
    if not sailing_information_df.empty:
        index_first_edge = pd.Index([sailing_information_df.iloc[0].name])
        index_last_edge = pd.Index([sailing_information_df.iloc[-1].name])

        # determine distance that must still be sailed on the current edge of the vessel
        distance_to_sail_on_first_edge = (sailing_information_df.loc[index_first_edge, 'distance']-distance_sailed_on_first_edge)

        # adjust information of the sailing distance and sailing time on the first and last edges
        sailing_information_df.loc[index_first_edge, 'time'] = sailing_information_df.loc[index_first_edge, 'time']*(distance_to_sail_on_first_edge/sailing_information_df.loc[index_first_edge, 'distance'])
        sailing_information_df.loc[index_first_edge, 'distance'] = distance_to_sail_on_first_edge
        sailing_information_df.loc[index_last_edge, 'time'] = sailing_information_df.loc[index_last_edge, 'time']*(distance_to_be_sailed_on_last_edge/sailing_information_df.loc[index_last_edge, 'distance'])
        sailing_information_df.loc[index_last_edge, 'distance'] = distance_to_be_sailed_on_last_edge

    return sailing_information_df


def get_closest_edge_to_geometry(graph, geometry):
    distance_to_edge = {}
    geometry = geometry
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in distance")
        for edge in graph.edges(data=True):
            edge_name = (edge[0], edge[1])
            edge_info = edge[2]
            edge_geometry = edge_info["geometry"]
            distance_to_edge[edge_name] = geometry.distance(edge_geometry)

    closest_edge = list(distance_to_edge.keys())[np.argmin(list(distance_to_edge.values()))]
    return closest_edge


def get_closest_node_to_geometry(graph, geometry):
    closest_edge = get_closest_edge_to_geometry(graph, geometry)
    distance_to_node = {}
    geometry = geometry
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in distance")
        for node in closest_edge:
            node_geometry = graph.nodes[node]["geometry"]
            distance_to_node[node] = geometry.distance(node_geometry)

    closest_node = list(distance_to_node.keys())[np.argmin(list(distance_to_node.values()))]
    return closest_node


def get_closest_location_on_edge_to_point(graph, edge, point):
    edge_geometry = graph.edges[edge]["geometry"]
    point_on_edge = edge_geometry.interpolate(edge_geometry.project(point))
    return point_on_edge


def find_nodes_in_a_polygon(graph, polygon):
    nodes = []
    for node in graph.nodes(data=True):
        node_info = node[1]
        node_geometry = node_info["geometry"]
        if polygon.intersects(node_geometry):
            nodes.append(node[0])
    return nodes


def find_edges_in_a_polygon(graph, polygon):
    edges = []
    for edge in graph.edges(data=True):
        start_node = edge[0]
        end_node = edge[1]
        edge_info = edge[2]
        edge_geometry = edge_info["geometry"]
        if polygon.intersects(edge_geometry):
            edges.append((start_node, end_node))
    return edges


def find_edges_based_on_shared_node(graph, node):
    edges = []
    for edge in graph.edges:
        if node in edge:
            edges.append(edge)
    return edges


def remove_node_from_network(graph, node):
    graph.remove_node(node)
    return graph


def find_closest_node_of_edge_to_target_edge(graph, edge, target_edge):
    if edge == target_edge or edge == (target_edge[1], target_edge[0]):
        warnings.warn(f"Edges are the same, start_node of target_edge is returned.")
        return target_edge[0]

    routes = {}
    for target_node in target_edge:
        for node in edge:
            length_route = len(nx.dijkstra_path(graph, node, target_node))
            if node not in routes.keys() or length_route < routes[node]:
                routes[node] = len(nx.dijkstra_path(graph, node, target_node))
    closest_node_to_lock = min(routes, key=routes.get)
    return closest_node_to_lock


def find_closest_node_of_target_edge_to_geometry(graph, geometry, target_edge):
    distances_to_node = {}
    for node in target_edge:
        node_geometry = graph.nodes[node]["geometry"]
        distances_to_node[node] = node_geometry.distance(geometry)
    closest_node_to_geometry = min(distances_to_node, key=distances_to_node.get)
    return closest_node_to_geometry


def align_network_geometries_with_edge_directions(graph):
    from opentnsim.graph.calculations import reverse_geometry
    for edge in graph.edges:
        start_node = edge[0]
        end_node = edge[1]
        start_node_geometry = graph.nodes[start_node]["geometry"]
        end_node_geometry = graph.nodes[end_node]["geometry"]
        edge_geometry = graph.edges[edge]["geometry"]
        edge_geometry_coordinates_x, edge_geometry_coordinates_y = edge_geometry.coords.xy
        starting_point_edge_geometry = Point(edge_geometry_coordinates_x[0], edge_geometry_coordinates_y[0])
        distance_starting_point_edge_geometry_with_start_node_geometry = starting_point_edge_geometry.distance(
            start_node_geometry)
        distance_starting_point_edge_geometry_with_end_node_geometry = starting_point_edge_geometry.distance(
            end_node_geometry)
        if distance_starting_point_edge_geometry_with_start_node_geometry > distance_starting_point_edge_geometry_with_end_node_geometry:
            reversed_edge_geometry = reverse_geometry(edge_geometry)
            graph.edges[edge]["geometry"] = reversed_edge_geometry
            graph.edges[edge]["Wkt"] = str(reversed_edge_geometry)
        graph.edges[edge]["weight"] = 1
    return graph


def compare_two_edge_info(graph, edge_A, edge_B):
    edge_info_A = graph.edges[edge_A]
    edge_info_B = graph.edges[edge_B]
    shared_items = {k: edge_info_A[k] for k in edge_info_A if
                    k in edge_info_B and str(edge_info_A[k]) == str(edge_info_B[k])}
    missing_items_A = edge_info_A.keys() - shared_items.keys()
    missing_items_B = edge_info_B.keys() - shared_items.keys()
    return shared_items, missing_items_A, missing_items_B


def get_largest_route_between_edges(graph, edge1, edge2):
    if set(edge1) == set(edge2):
        return list(edge1)

    sources = list(edge1)
    targets = list(edge2)

    lengths = nx.multi_source_dijkstra_path_length(graph, sources, weight=None)
    paths = nx.multi_source_dijkstra_path(graph, sources, weight=None)

    best_target = None
    best_length = -1

    for t in targets:
        if t in lengths and lengths[t] > best_length:
            best_length = lengths[t]
            best_target = t

    if best_target is None:
        raise nx.NetworkXNoPath

    return paths[best_target]
