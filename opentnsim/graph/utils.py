"""Utility functions for OpenTNSim graphs."""

# packkage(s) for documentation, debugging, saving and loading
import warnings

# numerical libraries
import numpy as np

# spatial libraries
import networkx as nx
import shapely
import pyproj
from shapely.geometry import LineString
from shapely.ops import transform


class NetworkWarning(Warning):
    pass


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
        geometry = LineString([orig, dest])
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
    if min_coordinates_x < -180.0 or max_coordinates_x > 180.0 or min_coordinates_y < -90.0 or max_coordinates_y > 90.0:
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


def transform_projection(from_spatialref, to_EPSG):
    """create a transformation object to transform the graph to a new projection
    Make sure to install the required package gdal.

    run pip show gdal to check if gdal is installed.
    Parameters
    ----------
    to_EPSG: int
        The EPSG code to transform the graph to
    """

    from osgeo import ogr, osr

    to_spatialref = osr.SpatialReference()
    to_spatialref.ImportFromEPSG(to_EPSG)

    # Transform the coordinates
    transform = osr.CoordinateTransformation(from_spatialref, to_spatialref)
    return transform


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
