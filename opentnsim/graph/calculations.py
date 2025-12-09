
# calculation libraries
import numpy as np
# import spatial libraries
import pyproj
import shapely
import networkx as nx


wgs84 = pyproj.Geod(ellps="WGS84")

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