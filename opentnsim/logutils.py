"""
Core utiltities related to logging.
"""

# %% IMPORT DEPENDENCIES
# generic
import logging

import pandas as pd

from shapely import Point

# internal
from opentnsim.energy import find_closest_node
from opentnsim.graph import edge_key, edge_length, wgs84

logger = logging.getLogger(__name__)


def calculate_distance(start, stop):
    """Distance in m between two points on the WGS84 ellipsoid."""
    return wgs84.inv(start.x, start.y, stop.x, stop.y)[2]


def sailed_length(graph, u, v):
    """Length in m of the sailed edge u -> v; None if the edge or its length is missing."""
    if not graph.has_edge(u, v):
        return None
    key = (u, v, edge_key(graph, u, v)) if graph.is_multigraph() else (u, v)
    return edge_length(graph.edges[key])


def logbook2eventtable(objs, graph=None):
    """
    Transform object logbooks into a 'minimum event table'.

    Implements the basic 'event table' concept as proposed by Van der Werff:

    Van der Werff, S.E., F. Baart and M. van Koningsveld (2025). “Merging Multiple
    System Perspectives: The Key to Effective Inland Shipping Emission-Reduction
    Policy Design.” Journal of Marine Science and engineering 13(4), 716.
    https://doi.org/10.3390/jmse13040716

    Van der Werff, S.E., S. Eppenga, A. van der Hout, F. Baart and M. van
    Koningsveld (2025). “Multi-perspective nautical safety risk assessment of
    allisions with offshore wind parks.” Applied Ocean Research 158(2025),104564.
    https://doi.org/10.1016/j.apor.2025.104564

    For waterborne traffic over a network, a unique event is defined by:
     - a unique vessel,
     - a specific section of the waterway,
     - a specific time.

    Parameters
    ----------
    objs: list
        List of OpenTNSim simulation objects with log information.
    graph: networkx graph, optional
        Gives the distance along the sailed edge and the edge nodes of each event.

    Returns
    -------
    eventtable: pandas.DataFrame
        DataFrame with all events from obj.logbook attributes in objs.
    """
    # check if all objects have a logbook with expected structure
    for obj in objs:
        if (
            not hasattr(obj, "logbook")
            or not hasattr(obj, "id")
            or not hasattr(obj, "name")
        ):
            raise ValueError(
                f"Object {obj} does not have a logbook or id/name attributes."
            )

    # construct all logged events
    events = []
    for obj in objs:
        df = pd.DataFrame.from_dict(obj.logbook)
        df.sort_values(by="Timestamp", inplace=True)

        for i in range(0, len(df)):
            start_row = df.iloc[i]
            if start_row["Message"].endswith(" start"):
                activity = start_row["Message"].replace(" start", "")
            else:
                continue  # skip non-start messages

            later = df["Timestamp"] > start_row["Timestamp"]
            stops = df[(df["Message"] == activity + " stop") & later]
            if stops.empty:
                logger.warning(f"No stop message for '{activity}' of {obj.name}, event skipped")
                continue
            stop_row = stops.iloc[0]

            start_time = start_row["Timestamp"]
            stop_time = stop_row["Timestamp"]
            start_location = start_row["Geometry"]
            stop_location = stop_row["Geometry"]

            duration_seconds = (stop_time - start_time).total_seconds()
            if isinstance(start_location, Point):
                distance_meters = calculate_distance(start_location, stop_location)
            else:
                distance_meters = None

            # with a graph: the edge nodes of the event and the length of the sailed edge
            node_u = node_v = length = None
            if graph is not None and activity.startswith("Sailing from node "):
                node_u, node_v = activity.replace("Sailing from node ", "").split(" to node ")
                u, v = (node_u, node_v) if graph.has_edge(node_u, node_v) else (node_v, node_u)
                length = sailed_length(graph, u, v)
            elif graph is not None and isinstance(start_location, Point):
                node_u = find_closest_node(graph, start_location)[0]
                node_v = find_closest_node(graph, stop_location)[0]
                length = sailed_length(graph, node_u, node_v)
            if length is not None:
                distance_meters = length

            events.append(
                {
                    "object id": obj.id,
                    "object name": obj.name,
                    "activity name": activity,
                    "start location": start_location,
                    "stop location": stop_location,
                    "start node": node_u,
                    "stop node": node_v,
                    "start time": start_time,
                    "stop time": stop_time,
                    "distance (m)": distance_meters,
                    "duration (s)": duration_seconds,
                }
            )

    # Final DataFrame
    eventtable = pd.DataFrame(events)

    return eventtable
