import pyproj
import networkx as nx

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