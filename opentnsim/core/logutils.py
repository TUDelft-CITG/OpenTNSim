"""
Core utiltities related to logging.
"""

# %% IMPORT DEPENDENCIES
# generic
import pandas as pd
import numpy as np

from shapely import Point

# internal
import opentnsim.graph.mixins as mixins
from opentnsim.graph.calculations import calculate_distance_between_locations_along_edges, calculate_distance_along_geometry_to_nodes_of_edge
from opentnsim.graph.utils import find_closest_node

# # %% CONVERT LOG TO EVENT TABLE
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
    graph = None
    obj_ids = []
    for obj in objs:
        if (
            not hasattr(obj, "logbook")
            or not hasattr(obj, "id")
            or not hasattr(obj, "name")
            or not hasattr(obj, "env")
        ):
            raise ValueError(
                f"Object {obj} does not have a logbook or id/name attributes."
            )
        obj_ids.append(obj.id)
        graph = obj.env.graph

    if graph is None:
        return pd.DataFrame()

    # construct all logged events
    events = []
    eventtable = pd.DataFrame(columns=["object id", "object name", "activity name", "start location", "stop location",
                                       "start time", "stop time", "distance (m)", "duration (s)"])
    for obj in objs:
        df = pd.DataFrame.from_dict(obj.logbook)
        if df.empty:
            continue

        for i in range(0, len(df)):
            start_row = df.iloc[i]
            if start_row["Message"].endswith(" start"):
                activity = start_row["Message"].replace(" start", "")
            else:
                continue
            try:
                stop_row = df[(df["Message"] == activity + " stop") & (df["Timestamp"] > df.iloc[i]["Timestamp"])].iloc[0]
            except:
                continue

            start_time = start_row["Timestamp"]
            stop_time = stop_row["Timestamp"]
            start_location = start_row["Geometry"]
            stop_location = stop_row["Geometry"]
            duration_seconds = (stop_time - start_time).total_seconds()
            if isinstance(start_location, Point):
                start_node = find_closest_node(graph, start_location)
                stop_node  = find_closest_node(graph, stop_location)
            
                if start_node == stop_node:
                    distance_meters = 0.0
                else:
                    distance_meters = calculate_distance_along_geometry_to_nodes_of_edge(graph, start_node, stop_node)
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

    # To dataFrame
    if len(events):
        eventtable = pd.DataFrame(events)

    df = eventtable.copy()

    # Identifying subprocesses
    df['is_subprocess'] = False
    for i, main_row in df.iterrows():
        # Check other rows for subprocesses inside this main row
        mask = (
                (df['object id'] == main_row['object id']) &
                (df['start time'] >= main_row['start time']) &
                (df['stop time'] <= main_row['stop time']) &
                (df.index > i)
        )
        df.loc[mask, 'is_subprocess'] = True

    # Identifying main process has subprocesses
    df['has_subprocesses'] = False
    for i, main_row in df[df['is_subprocess'] == False].iterrows():
        mask = (
                (df['object id'] == main_row['object id']) &
                (df['start time'] >= main_row['start time']) &
                (df['stop time'] <= main_row['stop time']) &
                (df.index != i) &
                (df['is_subprocess'])
        )
        if mask.any():
            df.loc[i, 'has_subprocesses'] = True

    # Initialize columns
    df['main activity name'] = None
    df['subactivity name'] = None

    # Subprocess rows
    df.loc[df['is_subprocess'] == False, 'main activity name'] = df.loc[df['is_subprocess'] == False, 'activity name']
    df.loc[df['is_subprocess'], 'subactivity name'] = df['activity name']
    df['main activity name'] = df['main activity name'].ffill()
    
    # Container for gap segments
    gap_segments = []

    # Loop over main processes with subprocesses
    for idx, main_row in df[(df['is_subprocess'] == False) & (df['has_subprocesses'] == True)].iterrows():

        subs = df[
            (df['object id'] == main_row['object id']) &
            (df['start time'] >= main_row['start time']) &
            (df['stop time'] <= main_row['stop time']) &
            (df['is_subprocess'])
            ].sort_values('start time')

        current_start = main_row['start time']
        current_start_loc = main_row['start location']
        gap_count = 1


        for _, sub in subs.iterrows():
            if sub['start time'] == current_start and sub['stop time'] == main_row['stop time']:
                # Copy the main row and adjust columns for the gap
                gap_row = main_row.copy()

                gap_row['activity name'] = f"{main_row['activity name']} ({gap_count})"
                gap_row['subactivity name'] = sub['subactivity name']

                gap_row['start time'] = sub['start time']
                gap_row['stop time'] = sub['stop time']

                gap_row['start location'] = sub['start location']
                gap_row['stop location'] = sub['stop location']
                gap_row['distance (m)'] = calculate_distance_between_locations_along_edges(
                    graph,
                    gap_row['start location'],
                    gap_row['stop location']
                )
                gap_row['duration (s)'] = (sub['stop time'] - sub['start time']).total_seconds()

                gap_row['is_subprocess'] = False
                gap_row['has_subprocesses'] = False

                gap_segments.append(gap_row)

                current_start = max(current_start, sub['stop time'])
                current_start_loc = sub['stop location']
                break

            if sub['start time'] == current_start and sub['stop time'] < main_row['stop time']:
                # Copy the main row and adjust columns for the gap
                gap_row = main_row.copy()

                gap_row['activity name'] = f"{main_row['activity name']} ({gap_count})"
                gap_row['subactivity name'] = sub['subactivity name']

                gap_row['start time'] = sub['start time']
                gap_row['stop time'] = sub['stop time']

                gap_row['start location'] = sub['start location']
                gap_row['stop location'] = sub['stop location']
                gap_row['distance (m)'] = calculate_distance_between_locations_along_edges(
                    graph,
                    gap_row['start location'],
                    gap_row['stop location']
                )
                gap_row['duration (s)'] = (sub['stop time'] - sub['start time']).total_seconds()

                gap_row['is_subprocess'] = False
                gap_row['has_subprocesses'] = False

                gap_segments.append(gap_row)

                gap_count += 1

            # Move start pointer forward
            current_start = max(current_start, sub['stop time'])
            current_start_loc = sub['stop location']
            if sub['start time'] > current_start and sub['stop time'] <= main_row['stop time']:
                # Copy the main row and adjust columns for the gap
                gap_row = main_row.copy()

                gap_row['activity name'] = f"{main_row['activity name']} ({gap_count})"
                gap_row['subactivity name'] = sub['subactivity name']

                gap_row['start time'] = current_start
                gap_row['stop time'] = sub['start time']

                gap_row['start location'] = current_start_loc
                gap_row['stop location'] = sub['start location']
                gap_row['distance (m)'] = calculate_distance_between_locations_along_edges(
                    graph,
                    gap_row['start location'],
                    gap_row['stop location']
                )
                gap_row['duration (s)'] = (sub['start time'] - current_start).total_seconds()

                gap_row['is_subprocess'] = False
                gap_row['has_subprocesses'] = False

                gap_segments.append(gap_row)

                gap_count += 1

            # Move start pointer forward
            current_start = np.max([current_start, sub['stop time']])
            current_start_loc = sub['stop location']

        # Gap after last subprocess
        if current_start < main_row['stop time'] and not current_start == main_row['start time']:
            gap_row = main_row.copy()

            gap_row['activity name'] = f"{main_row['activity name']} ({gap_count})"
            gap_row['subactivity name'] = "" #sub['subactivity name']

            gap_row['start time'] = current_start
            gap_row['stop time'] = main_row['stop time']

            gap_row['start location'] = current_start_loc
            gap_row['stop location'] = main_row['stop location']

            gap_row['distance (m)'] = calculate_distance_between_locations_along_edges(
                graph,
                gap_row['start location'],
                gap_row['stop location']
            )

            gap_row['duration (s)'] = (main_row['stop time'] - current_start).total_seconds()

            gap_row['is_subprocess'] = False
            gap_row['has_subprocesses'] = False

            gap_segments.append(gap_row)
        

    # Convert gap segments to DataFrame
    gaps_df = pd.DataFrame(gap_segments)

    # Keep original subprocesses and any main processes without gaps
    main_without_subs = df[(df['is_subprocess'] == False) & (df['has_subprocesses'] == False)]
    sub_df = df[df['is_subprocess'] == True]

    # Combine all
    combined_df = pd.concat([main_without_subs, gaps_df], ignore_index=True)
    combined_df = combined_df.sort_values(['object id', 'start time']).reset_index(drop=True)
    combined_df['main activity name'] = combined_df['main activity name'].ffill()

    combined_df = combined_df.sort_values(
        by=['object id', 'start time'],
        key=lambda col: col.map({v: i for i, v in enumerate(obj_ids)}) if col.name == 'object id' else col
    )
    combined_df = combined_df.reset_index(drop=True)
    combined_df.loc[combined_df['subactivity name'].isna(), 'subactivity name'] = ''
    eventtable = combined_df[['object id', 'object name', 'main activity name', 'subactivity name',
                              'start location', 'stop location', 'start time', 'stop time',
                              'distance (m)', 'duration (s)']]

    return eventtable
