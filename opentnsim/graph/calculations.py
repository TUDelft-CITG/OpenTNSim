import pyproj
import networkx as nx
from shapely import reverse
from shapely.ops import transform

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


def transform_geometry(geometry, epsg_in="EPSG:4326", epsg_out='EPSG:4087'):
    crs_in = pyproj.CRS(epsg_in)
    crs_out = pyproj.CRS(epsg_out)
    crs_in_to_crs_out = pyproj.transformer.Transformer.from_crs(crs_in, crs_out, always_xy=True).transform
    geometry_transformed = transform(crs_in_to_crs_out, geometry)
    return geometry_transformed


def calculate_location_over_edges(graph, edge, interpolation_length, crs_m = 'EPSG:4087'):
    geometry = graph.edges[edge]["geometry"]
    if geometry is None or geometry.is_empty:
        return None
    geometry_m = transform_geometry(geometry, epsg_out = crs_m)
    interpolation_point_m = geometry_m.line_interpolate_point(interpolation_length)
    interpolation_point = transform_geometry(interpolation_point_m, epsg_in = crs_m, epsg_out = "EPSG:4326")
    return interpolation_point


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


def reverse_geometry(geometry):
    reversed_geometry = reverse(geometry)
    return reversed_geometry


def calculate_distance_over_network_to_location(graph, node_1, node_2, location,tolerance=0.0001):
    geod = pyproj.Geod(ellps="WGS84")
    geometry = get_trajectory(node_1,node_2)
    geometries = shapely.ops.split(shapely.ops.snap(geometry, location, tolerance=tolerance), location).geoms
    distance_sailed = 0
    distance_to_go = 0
    if len(geometries) < 2:
        if graph.nodes[node_1]['geometry'] == location:
            distance_to_go = geod.geometry_length(geometries[0])
        elif graph.nodes[node_2]['geometry'] == location:
            distance_sailed = geod.geometry_length(geometries[0])
        elif graph.nodes[node_1]['geometry'].distance(location) > graph.nodes[node_2]['geometry'].distance(location):
            distance_sailed = geod.geometry_length(geometries[0])
        else:
            distance_to_go = geod.geometry_length(geometries[0])
    else:
        distance_sailed = geod.geometry_length(geometries[0])
        distance_to_go = geod.geometry_length(geometries[1])
    return distance_sailed,distance_to_go


def calculate_distance_from_location_over_edge(graph,edge,location,tolerance=0.0001):
    geod = pyproj.Geod(ellps="WGS84")
    geometry = graph.edges[edge]['geometry']
    distance_sailed = 0
    distance_to_go = 0
    if geometry.coords[0] == location.coords[0]:
        distance_to_go = graph.edges[(edge[0],edge[1],edge[2])]['length']
    elif geometry.coords[-1] == location.coords[0]:
        distance_sailed = graph.edges[(edge[0],edge[1],edge[2])]['length']
    else:
        lines = shapely.ops.split(shapely.ops.snap(geometry, location, tolerance), location).geoms
        for index, line in enumerate(lines):
            distance = 0
            for point_I, point_II in zip(line.coords[:-1], line.coords[1:]):
                sub_edge_geometry = LineString([Point(point_I), Point(point_II)])
                distance += geod.geometry_length(sub_edge_geometry)
            if not index:
                distance_sailed = distance
            else:
                distance_to_go = distance
    return distance_sailed, distance_to_go


def calculate_length_of_edge(graph, edge, current_crs="EPSG:4326", crs_meter="EPSG:4087"):
    wgs84 = pyproj.CRS(current_crs)
    wgs84_m = pyproj.CRS(crs_meter)
    wgs84_to_wgs84_m = pyproj.transformer.Transformer.from_crs(wgs84, wgs84_m, always_xy=True).transform
    geometry = graph.edges[edge]["geometry"]
    geometry_m = transform(wgs84_to_wgs84_m, geometry)
    length_m = geometry_m.length
    return length_m