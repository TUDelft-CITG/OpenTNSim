"""
Core utiltities related to logging.
"""

# %% IMPORT DEPENDENCIES
# generic
import pandas as pd

# internal
import opentnsim.graph


# %% CONVERT LOG TO EVENT TABLE
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
        for i in range(0, len(df), 2):
            activity = df.iloc[i]["Message"].replace(" start", "")

            start_time = df.iloc[i]["Timestamp"]
            stop_time = df.iloc[i + 1]["Timestamp"]
            start_location = df.iloc[i]["Geometry"]
            stop_location = df.iloc[i + 1]["Geometry"]

            duration_seconds = (stop_time - start_time).total_seconds()
            distance_meters = opentnsim.graph.calculate_distance(
                start_location, stop_location
            )

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
