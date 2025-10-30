"""
Core utiltities related to logging.
"""

# %% IMPORT DEPENDENCIES
# generic
import pandas as pd

from shapely import Point

# internal
import opentnsim.graph.mixins as mixins


# # %% CONVERT LOG TO EVENT TABLE
# def logbook2eventtable(objs):
#     """
#     Transform object logbooks into a 'minimum event table'.

#     Implements the basic 'event table' concept as proposed by Van der Werff:

#     Van der Werff, S.E., F. Baart and M. van Koningsveld (2025). “Merging Multiple
#     System Perspectives: The Key to Effective Inland Shipping Emission-Reduction
#     Policy Design.” Journal of Marine Science and engineering 13(4), 716.
#     https://doi.org/10.3390/jmse13040716

#     Van der Werff, S.E., S. Eppenga, A. van der Hout, F. Baart and M. van
#     Koningsveld (2025). “Multi-perspective nautical safety risk assessment of
#     allisions with offshore wind parks.” Applied Ocean Research 158(2025),104564.
#     https://doi.org/10.1016/j.apor.2025.104564

#     For waterborne traffic over a network, a unique event is defined by:
#      - a unique vessel,
#      - a specific section of the waterway,
#      - a specific time.

#     Parameters
#     ----------
#     objs: list
#         List of OpenTNSim simulation objects with log information.

#     Returns
#     -------
#     eventtable: pandas.DataFrame
#         DataFrame with all events from obj.logbook attributes in objs.
#     """
#     # check if all objects have a logbook with expected structure
#     for obj in objs:
#         if (
#             not hasattr(obj, "logbook")
#             or not hasattr(obj, "id")
#             or not hasattr(obj, "name")
#         ):
#             raise ValueError(
#                 f"Object {obj} does not have a logbook or id/name attributes."
#             )

#     # construct all logged events
#     events = []
#     for obj in objs:
#         df = pd.DataFrame.from_dict(obj.logbook)

#         unique_activities = df["Message"].str.replace(" start", "").str.replace(" stop", "").unique()
#         for activity in unique_activities:
#             start_time = df[df.Message == activity + " start"]["Timestamp"].values[0]
#             stop_time = df[df.Message == activity + " stop"]["Timestamp"].values[0]

#             start_location = df[df.Message == activity + " start"]["Geometry"].values[0]
#             stop_location = df[df.Message == activity + " stop"]["Geometry"].values[0]

#             duration_seconds = (stop_time - start_time) / pd.Timedelta(seconds=1)

#             if isinstance(start_location, Point):
#                 distance_meters = opentnsim.graph.calculate_distance(start_location, stop_location)
#             else:
#                 distance_meters = None

#             events.append(
#                 {
#                     "object id": obj.id,
#                     "object name": obj.name,
#                     "activity name": activity,
#                     "start location": start_location,
#                     "stop location": stop_location,
#                     "start time": start_time,
#                     "stop time": stop_time,
#                     "distance (m)": distance_meters,
#                     "duration (s)": duration_seconds,
#                 }
#             )

#     # Final DataFrame
#     eventtable = pd.DataFrame(events)

#     return eventtable

def logbook2eventtable(objs):
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

            stop_row = df[(df["Message"] == activity + " stop") & (df["Timestamp"] > df.iloc[i]["Timestamp"])].iloc[0]

            start_time = start_row["Timestamp"]
            stop_time = stop_row["Timestamp"]
            start_location = start_row["Geometry"]
            stop_location = stop_row["Geometry"]

            duration_seconds = (stop_time - start_time).total_seconds()
            if isinstance(start_location, Point):
                distance_meters = mixins.calculate_distance(start_location, stop_location)
            else:
                distance_meters = None

            events.append(
                {
                    "object id": obj.id,
                    "object name": obj.name,
                    "activity name": activity,
                    "start location": start_location,
                    "stop location": stop_location,
                    "start time": start_time,
                    "stop time": stop_time,
                    "distance (m)": distance_meters,
                    "duration (s)": duration_seconds,
                }
            )

    # Final DataFrame
    eventtable = pd.DataFrame(events)

    return eventtable
