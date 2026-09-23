"""
Core utiltities related to logging.

PATCHED (see CHANGES.md):
 - P5: unmatched "start" messages no longer crash on .iloc[0]; they are logged and skipped
 - P1: sailing events now carry "start node" and "stop node" columns so that the
   energy step can look up the edge current per event
 - edge-length based distances (length_m -> length -> great-circle fallback) retained
"""

# %% IMPORT DEPENDENCIES
# generic
import logging

import pandas as pd

from shapely import Point

# internal
import opentnsim.mixins as mixins

logger = logging.getLogger(__name__)


def logbook2eventtable(objs, graph=None):

    """
    Transform object logbooks into a 'minimum event table'.

    Implements the basic 'event table' concept as proposed by Van der Werff:

    Van der Werff, S.E., F. Baart and M. van Koningsveld (2025). "Merging Multiple
    System Perspectives: The Key to Effective Inland Shipping Emission-Reduction
    Policy Design." Journal of Marine Science and engineering 13(4), 716.
    https://doi.org/10.3390/jmse13040716

    Van der Werff, S.E., S. Eppenga, A. van der Hout, F. Baart and M. van
    Koningsveld (2025). "Multi-perspective nautical safety risk assessment of
    allisions with offshore wind parks." Applied Ocean Research 158(2025),104564.
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
        When given, sailing distances are taken from the edge ("length_m", then
        "length", then great-circle fallback) and the event table carries the
        edge nodes ("start node", "stop node").

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
    n_unmatched = 0
    for obj in objs:
        df = pd.DataFrame.from_dict(obj.logbook)
        df.sort_values(by="Timestamp", inplace=True)

        for i in range(0, len(df)):
            start_row = df.iloc[i]
            if start_row["Message"].endswith(" start"):
                activity = start_row["Message"].replace(" start", "")
            else:
                continue  # skip non-start messages

            # PATCH P5: guard against unmatched start messages instead of crashing
            stops = df[
                (df["Message"] == activity + " stop")
                & (df["Timestamp"] > start_row["Timestamp"])
            ]
            if stops.empty:
                n_unmatched += 1
                logger.warning(
                    "Unmatched start message '%s' for object %s at %s; event skipped.",
                    start_row["Message"], obj.id, start_row["Timestamp"],
                )
                continue
            stop_row = stops.iloc[0]

            start_time = start_row["Timestamp"]
            stop_time = stop_row["Timestamp"]
            start_location = start_row["Geometry"]
            stop_location = stop_row["Geometry"]

            duration_seconds = (stop_time - start_time).total_seconds()

            distance_meters = None
            # PATCH P1: remember the edge nodes so the energy step can look up currents
            node_u = None
            node_v = None

            msg = str(start_row["Message"])

            if graph is not None and msg.startswith("Sailing from node ") and " to node " in msg:
                try:
                    u = msg.split("Sailing from node ")[1].split(" to node ")[0]
                    v = msg.split(" to node ")[1].rsplit(" start", 1)[0]
                    node_u, node_v = u, v

                    try:
                        if graph.is_multigraph():
                            k = sorted(
                                graph[u][v],
                                key=lambda kk: graph[u][v][kk].get("geometry", None).length
                                if graph[u][v][kk].get("geometry", None) is not None else 1e18
                            )[0]
                            e = graph.edges[u, v, k]
                        else:
                            e = graph.edges[u, v]
                    except Exception:
                        if graph.is_multigraph():
                            k = sorted(
                                graph[v][u],
                                key=lambda kk: graph[v][u][kk].get("geometry", None).length
                                if graph[v][u][kk].get("geometry", None) is not None else 1e18
                            )[0]
                            e = graph.edges[v, u, k]
                        else:
                            e = graph.edges[v, u]

                    if "length_m" in e and e["length_m"] is not None:
                        distance_meters = float(e["length_m"])
                    elif "length" in e and e["length"] is not None:
                        distance_meters = float(e["length"])
                    else:
                        distance_meters = mixins.calculate_distance(start_location, stop_location)

                except Exception:
                    distance_meters = mixins.calculate_distance(start_location, stop_location)

            elif graph is not None and isinstance(start_location, Point) and isinstance(stop_location, Point):
                u = mixins.find_closest_node(graph, start_location)[0]
                v = mixins.find_closest_node(graph, stop_location)[0]
                node_u, node_v = u, v

                try:
                    if graph.is_multigraph():
                        k = sorted(
                            graph[u][v],
                            key=lambda kk: graph[u][v][kk].get("geometry", None).length
                            if graph[u][v][kk].get("geometry", None) is not None else 1e18
                        )[0]
                        e = graph.edges[u, v, k]
                    else:
                        e = graph.edges[u, v]

                    if "length_m" in e and e["length_m"] is not None:
                        distance_meters = float(e["length_m"])
                    elif "length" in e and e["length"] is not None:
                        distance_meters = float(e["length"])
                    else:
                        distance_meters = mixins.calculate_distance(start_location, stop_location)

                except Exception:
                    distance_meters = mixins.calculate_distance(start_location, stop_location)

            elif isinstance(start_location, Point):
                distance_meters = mixins.calculate_distance(start_location, stop_location)

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

    if n_unmatched:
        logger.warning("logbook2eventtable: %d unmatched start messages were skipped.", n_unmatched)

    # Final DataFrame
    eventtable = pd.DataFrame(events)

    return eventtable
