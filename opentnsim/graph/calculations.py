import math
import numpy as np
import networkx as nx
import pyproj
from scipy.ndimage import rotate
from scipy.spatial import ConvexHull
from shapely import reverse
from shapely.geometry import Point, Polygon
from shapely.ops import transform, linemerge, split, snap
from opentnsim.graph.utils import (find_edges_based_on_shared_node, compare_two_edge_info, remove_node_from_network,
                                   create_transformer, get_trajectory, find_closest_node, find_closest_edge,
                                   get_largest_route_between_edges)
import warnings


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
    # The node on the graph of vaarweginformatie.nl closest to geom_start and geom_stop

    edge = find_closest_edge(graph, geom_start)
    node_start, node_stop = edge[:2]

    # Read from the graph data from vaarweginformatie.nl the General depth of each edge
    try:
        if "GeneralDepth" in graph.get_edge_data(node_start, node_stop).keys():
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


def transform_geometry(geometry, epsg_in = "EPSG:4326", epsg_out = 'EPSG:4087', transformer = None):
    if transformer is None:
        transformer = create_transformer(epsg_in, epsg_out)
    geometry_transformed = transform(transformer, geometry)
    return geometry_transformed


def transform_route_geometry(env, node_start, node_stop, crs_in = "EPSG:4326", crs_out = "EPSG:4087"):
    route_geometry = get_trajectory(env.graph, node_start, node_stop)
    route_geometry_transformed = transform_geometry(route_geometry, crs_in, crs_out)
    return route_geometry_transformed


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


def calculate_distance_over_network_to_location(graph, node_1, node_2, location, tolerance=0.0001):
    geod = pyproj.Geod(ellps="WGS84")
    geometry = get_trajectory(graph, node_1,node_2)[0]
    geometries = split(snap(geometry, location, tolerance=tolerance), location).geoms
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


def calculate_distance_between_locations_along_edges(graph, location_1, location_2):
    edge_1 = find_closest_edge(graph, location_1)
    edge_2 = find_closest_edge(graph, location_2)
    route = get_largest_route_between_edges(graph, edge_1, edge_2)
    geometry, length_m = get_trajectory(graph, route[0], route[-1])
    geometry_length = geometry.length

    distance_1 = geometry.project(location_1)
    distance_2 = geometry.project(location_2)

    fraction_1 = distance_1 / geometry_length
    fraction_2 = distance_2 / geometry_length

    distance_m_1 = fraction_1 * length_m
    distance_m_2 = fraction_2 * length_m

    distance_m = round(np.abs(distance_m_1 - distance_m_2),2)
    return distance_m


def calculate_distance_from_location_over_edge(graph,edge,location,tolerance=0.0001):
    geod = pyproj.Geod(ellps="WGS84")
    geometry = graph.edges[edge]['geometry']
    distance_sailed = 0
    distance_to_go = 0
    if geometry.coords[0] == location.coords[0]:
        distance_to_go = graph.edges[(edge[0],edge[1],edge[2])]['length_m']
    elif geometry.coords[-1] == location.coords[0]:
        distance_sailed = graph.edges[(edge[0],edge[1],edge[2])]['length_m']
    else:
        lines = split(snap(geometry, location, tolerance), location).geoms
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


def calculate_distance_along_geometry_to_nodes_of_edge(graph, start_node, end_node):
    route = nx.dijkstra_path(graph,start_node,end_node)
    length = 0
    for edge in zip(route[:-1],route[1:]):
        length += graph.edges[edge]['length_m']
    return length


def calculate_length_of_splitted_edge_geometries(graph, edge, edge_geometries):
    edge_geometry_lengths = [edge_geometry.length for edge_geometry in edge_geometries]
    sum_edge_length = np.sum(edge_geometry_lengths)
    fractive_edge_parts_lengths = edge_geometry_lengths / sum_edge_length
    edge_parts_lenghts_m = fractive_edge_parts_lengths * graph.edges[edge]["length_m"]
    return edge_parts_lenghts_m


def merge_edges(graph, edge_A, edge_B):
    start_junction_id = list(set(edge_A) - set(edge_B))[0]
    end_junction_id = list(set(edge_B) - set(edge_A))[0]

    edge_info_A = graph.edges[edge_A]
    edge_info_B = graph.edges[edge_B]
    edge_geometry_A = graph.edges[edge_A]["geometry"]
    edge_geometry_B = graph.edges[edge_B]["geometry"]
    new_edge_geometry = linemerge([edge_geometry_A, edge_geometry_B])

    shared_items, missing_items_A, missing_items_B = compare_two_edge_info(graph, edge_A, edge_B)
    new_edge_data = shared_items

    new_edge_data['StartJunctionId'] = start_junction_id
    new_edge_data['EndJunctionId'] = end_junction_id
    new_edge_data['GeoType'] = np.nan
    new_edge_data['Wkt'] = str(new_edge_geometry)
    new_edge_data['geometry'] = new_edge_geometry
    new_edge_data['length'] = edge_info_A['length'] + edge_info_B['length']
    new_edge_data['length_m'] = edge_info_A['length_m'] + edge_info_B['length_m']

    if (start_junction_id, end_junction_id) in graph.edges:
        warnings.warn(f"Edge ({start_junction_id},{end_junction_id}) is already part of the network, merging aborted.")
        return graph

    graph.add_edge(start_junction_id, end_junction_id, **new_edge_data)
    return graph


def merge_two_consecutive_edges_based_on_shared_node(graph, node):
    if node not in graph.nodes:
        warnings.warn(f"Node ({node}) does not exist in graph, merging aborted.")
        return graph

    edges = find_edges_based_on_shared_node(graph, node)
    number_of_edges = len(edges)
    number_of_allowed_edges = 2
    if graph.is_directed():
        number_of_allowed_edges = 4
    if number_of_edges != number_of_allowed_edges:
        if number_of_edges > number_of_allowed_edges:
            warnings.warn(f"Node ({node}) has multiple ({number_of_edges}) edges, merging aborted.")
        else:
            warnings.warn(f"Node ({node}) has only one edge, merging aborted.")
        return graph

    if graph.is_directed():
        edge_AA, edge_AB = edges[0],edges[2]
        graph = merge_edges(graph, edge_AA, edge_AB)
        edge_BA, edge_BB = (edge_AB[1],edge_AB[0]), (edge_AA[1],edge_AA[0])
        graph = merge_edges(graph, edge_BA, edge_BB)
    else:
        edge_A, edge_B = edges
        graph = merge_edges(graph, edge_A, edge_B)

    graph = remove_node_from_network(graph, node)
    return graph


def calculate_bounding_rectangle(geometry):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    Parameters
    ----------
    geometry : shapely.geometry.Poylgon
        polygon of the object

    Returns
    -------
    rval : shapely.geometry.Poylgon
        polygon of the bounding rectangle
    """

    geometry_coordinates = geometry.exterior.coords
    points = np.array(geometry_coordinates)

    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    bounding_rectangle = Polygon([Point(y, x) for x, y in rval])
    return bounding_rectangle


def calculate_object_dimensions_and_alignment(geometry):
    exterior_coords = geometry.exterior.coords
    side_lengths = []
    for i in range(len(exterior_coords) - 1):
        p1 = exterior_coords[i]
        p2 = exterior_coords[i + 1]
        length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        side_lengths.append(length)
    length = np.max(side_lengths)
    width = np.min(side_lengths)

    length_index = int(np.argmax(side_lengths))
    p1 = exterior_coords[length_index]
    p2 = exterior_coords[length_index + 1]

    # Calculate angle in radians
    angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    # Convert to degrees
    angle_deg = math.degrees(angle_rad)
    return length, width, angle_deg


def flip_coordinates(geometry):
    flipped_geometry = transform(lambda x, y: (y, x), geometry)
    return flipped_geometry


def reverse_geometry(geometry):
    reversed_geometry = reverse(geometry)
    return reversed_geometry


def split_edge_based_on_point_along_edge(graph, edge, point):
    edge_geometry = graph.edges[edge]["geometry"]
    split_point = point.buffer(1e-9)
    if not edge_geometry.intersects(split_point):
        warnings.warn(f"Point is not located along the edge, returned the edge and None.")
        return edge_geometry, None
    splitted_edge_geometries = split(edge_geometry, split_point).geoms
    first_edge_geometry = splitted_edge_geometries[0]
    second_edge_geometry = splitted_edge_geometries[2]
    return first_edge_geometry, second_edge_geometry


def split_edge_based_on_geometry_along_edge(graph, edge, geometry):
    edge_geometry = graph.edges[edge]["geometry"]
    if not edge_geometry.intersects(geometry):
        warnings.warn(f"Geometry is not located along the edge, returned the edge, geometry, and None.")
        return edge_geometry, geometry, None
    splitted_edge_geometries = split(edge_geometry, geometry).geoms
    first_edge_geometry = splitted_edge_geometries[0]
    edge_in_geometry = splitted_edge_geometries[1]
    second_edge_geometry = splitted_edge_geometries[2]
    return first_edge_geometry, edge_in_geometry, second_edge_geometry


def calculate_the_distances_from_doors_to_edge_nodes(graph, lock_edge, lock_edge_geometries, lock_edge_lengths):
    distance_from_start_node_to_lock_doors_A = None
    distance_from_end_node_to_lock_doors_B = None
    for geometry_index, geometry in enumerate(lock_edge_geometries):
        distances_to_node = {}
        for lock_node in lock_edge:
            distances_to_node[lock_node] = geometry.distance(graph.nodes[lock_node]["geometry"])
        closest_node = min(distances_to_node, key=distances_to_node.get)
        if distances_to_node[closest_node] > 0.0001:
            geometry_index_name = 'First'
            if geometry_index:
                geometry_index_name = 'Second'
            warnings.warn(f"{geometry_index_name} lock edge geometry does not touch any node")
        lock_node_index = list(distances_to_node.keys()).index(closest_node)
        if not lock_node_index:
            distance_from_start_node_to_lock_doors_A = lock_edge_lengths[geometry_index]
        else:
            distance_from_end_node_to_lock_doors_B = lock_edge_lengths[geometry_index]
    if distance_from_start_node_to_lock_doors_A is None:
        warnings.warn(f"Distance_from_start_node_to_lock_doors_A not found")
    if distance_from_end_node_to_lock_doors_B is None:
        warnings.warn(f"Distance_from_end_node_to_lock_doors_B not found")
    return distance_from_start_node_to_lock_doors_A, distance_from_end_node_to_lock_doors_B

