# -*- coding: utf-8 -*-
"""
Core utiltities related to logging.
"""

# %% IMPORT DEPENDENCIES
# generic
import pandas as pd
from opentnsim.graph.utils import get_trajectory_between_locations, find_closest_node, find_closest_edge
from shapely.ops import split, linemerge, snap
from shapely.geometry import MultiPoint, MultiLineString, LineString, Point
import geopandas as gpd
import networkx as nx

# internal
import opentnsim.graph.mixins as mixins
from opentnsim.graph.calculations import calculate_distance_between_locations_along_edges

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
        df.sort_values(by="Timestamp", inplace=True)
        if df.empty:
            continue

        for i in range(0, len(df)):
            start_row = df.iloc[i]
            if start_row["Message"].endswith(" start"):
                activity = start_row["Message"].replace(" start", "")
            else:
                continue  # skip non-start messages

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
                distance_meters = calculate_distance_between_locations_along_edges(graph, start_location, stop_location)
            else:
                distance_meters = None

            events.append(
                {
                    "object id": obj.id,
                    "object name": obj.name,
                    "activity name": activity,
                    "start location": start_location,
                    "stop location": stop_location,
                    "start time": start_time.round('s'),
                    "stop time": stop_time.round('s'),
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
                (df.index != i)
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

            if sub['start time'] > current_start:
                # Copy the main row and adjust columns for the gap
                gap_row = main_row.copy()

                gap_row['activity name'] = f"{main_row['activity name']} ({gap_count})"

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
            current_start = max(current_start, sub['stop time'])
            current_start_loc = sub['stop location']

        # Gap after last subprocess
        if current_start < main_row['stop time']:
            gap_row = main_row.copy()

            gap_row['activity name'] = f"{main_row['activity name']} ({gap_count})"

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
    combined_df = pd.concat([sub_df, main_without_subs, gaps_df], ignore_index=True)
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


def eventtable2gdf(df_eventtable, objs):
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
        return df_eventtable

    def get_unique_geometries(row):
        start = row['start location']
        stop = row['stop location']
        if start == stop:
            return Point(start)
        else:
            return LineString([start, stop])

    def determine_if_geometry_on_node(row, graph):
        geometry = row['geometry']
        start_location = row['start location']
        stop_location = row['stop location']
        node_start = find_closest_node(graph, start_location)
        node_stop = find_closest_node(graph, stop_location)
        start_location_on_node = False
        stop_location_on_node = False
        if graph.nodes[node_start]['geometry'] == start_location:
            start_location_on_node = True
        if graph.nodes[node_stop]['geometry'] == stop_location:
            stop_location_on_node = True
        return start_location_on_node, stop_location_on_node

    def split_edge_in_graph(G, u, v, segments):
        """
        Replace edge (u, v) with multiple segment edges.

        segments: list of LineStrings (ordered!)
        """
        G_new = G.copy()

        # Get original edge data
        edge_data = G.edges[u, v].copy()

        # Remove original edge
        G_new.remove_edge(u, v)

        # Helper: create node id from coordinate
        def node_id(coord):
            return tuple(coord)  # or str(coord) if needed

        # Build nodes + edges
        prev_node = u

        for i, seg in enumerate(segments):
            start = seg.coords[0]
            end = seg.coords[-1]

            # Add start node if needed
            if prev_node == u:
                current_start = u
            else:
                current_start = node_id(start)
                G_new.add_node(current_start, x=start[0], y=start[1], geometry=Point(start))

            # Add end node
            if i == len(segments) - 1:
                current_end = v
            else:
                current_end = node_id(end)
                G_new.add_node(current_end, x=end[0], y=end[1], geometry=Point(end))

            # Add new edge
            G_new.add_edge(current_start, current_end, weight=1, geometry=seg)

            prev_node = current_end

        return G_new

    def copy_graph_and_remove_information(graph, keep_edge_columns=['weigth', 'geometry'],
                                          keep_node_columns=['geometry']):
        graph_new = graph.copy()
        for n, data in graph_new.nodes(data=True):
            keys = list(data.keys())
            for k in keys:
                if k not in keep_node_columns:
                    del data[k]
        for u, v, data in graph_new.edges(data=True):
            keys = list(data.keys())
            for k in keys:
                if k not in keep_node_columns:
                    del data[k]
        return graph_new

    def weighted_avg(group, col, weight_col):
        weight_sum = group[weight_col].sum()
        if weight_sum == 0.:
            return 0.
        return (group[col] * group[weight_col]).sum() / weight_sum

    def update_locations(row):
        geom = row['geometry']
        if isinstance(geom, LineString):
            row['start location'] = Point(geom.coords[0])
            row['stop location'] = Point(geom.coords[-1])
        elif isinstance(geom, Point):
            row['start location'] = geom
            row['stop location'] = geom
        return row

    def expand_row_by_edges(row, graph):
        """
        Expands a row into multiple rows if it has multiple edges.
        Distance and duration are split proportionally.
        """
        edges = row['edges']
        if not edges or len(edges) <= 1:
            return [row]

        # Get edge lengths
        edge_lengths = []
        edge_geoms = []
        for edge in edges:
            u, v = edge
            geom = graph.edges[edge]['geometry']
            if isinstance(geom, LineString):
                length = geom.length
            else:
                # fallback: compute from coords
                length = LineString([u, v]).length
            edge_lengths.append(length)
            edge_geoms.append(geom if isinstance(geom, LineString) else LineString([u, v]))

        total_length = sum(edge_lengths)

        # Split distance and duration proportionally
        new_rows = []
        for length, geom in zip(edge_lengths, edge_geoms):
            new_row = row.copy()
            fraction = length / total_length if total_length > 0 else 0
            new_row['distance (m)'] = row['distance (m)'] * fraction
            new_row['duration (s)'] = row['duration (s)'] * fraction
            new_row['geometry'] = geom
            # Update start_node and stop_node for this segment
            new_row['start_node'] = geom.coords[0]
            new_row['stop_node'] = geom.coords[-1]
            new_rows.append(new_row)

        return new_rows

    graph_new = copy_graph_and_remove_information(graph)
    graph_new = graph_new.to_undirected()
    df_eventtable['geometry'] = df_eventtable.apply(get_unique_geometries, axis=1)
    df = pd.DataFrame(df_eventtable.apply(determine_if_geometry_on_node, graph=graph_new, axis=1), columns=['result'])
    df_eventtable[['start_location_on_node', 'stop_location_on_node']] = pd.DataFrame(df['result'].tolist(),
                                                                                      index=df.index)
    unique_locations = pd.unique(pd.concat([df_eventtable[~df_eventtable.start_location_on_node]['start location'],
                                            df_eventtable[~df_eventtable.start_location_on_node]['stop location']]))
    edge_splits = {}
    for unique_location in unique_locations:
        edge = find_closest_edge(graph, unique_location)
        if edge not in edge_splits.keys():
            edge_splits[edge] = []
        edge_splits[edge].append(unique_location)

    for edge, points in edge_splits.items():
        edge_geometry = graph.edges[edge]['geometry']
        splitter = snap(MultiPoint(points), edge_geometry, tolerance=1e-6)
        segments = split(edge_geometry, splitter)
        graph_new = split_edge_in_graph(graph_new, edge[0], edge[1], segments.geoms)

    df_eventtable['start_node'] = df_eventtable['start location'].apply(lambda x: find_closest_node(graph_new, x))
    df_eventtable['stop_node'] = df_eventtable['stop location'].apply(lambda x: find_closest_node(graph_new, x))
    df_eventtable['route'] = df_eventtable.apply(lambda x: nx.dijkstra_path(graph_new, x.start_node, x.stop_node),
                                                 axis=1)
    df_eventtable['edges'] = df_eventtable['route'].apply(lambda r: list(zip(r[:-1], r[1:])))

    expanded_rows = []

    for idx, row in df_eventtable.iterrows():
        expanded_rows.extend(expand_row_by_edges(row, graph_new))

    df_expanded = pd.DataFrame(expanded_rows)
    df_expanded = df_expanded.reset_index(drop=True)
    df_expanded = df_expanded.apply(update_locations, axis=1)
    df_expanded['geometry'] = df_expanded.apply(get_unique_geometries, axis=1)
    df_expanded['start_node'] = df_expanded['start location'].apply(lambda x: find_closest_node(graph_new, x))
    df_expanded['stop_node'] = df_expanded['stop location'].apply(lambda x: find_closest_node(graph_new, x))
    df_expanded['network_geometry'] = df_expanded.apply(
        lambda x: graph_new.edges[(x.start_node, x.stop_node)]['geometry'] if x.start_node != x.stop_node else
        graph_new.nodes[x.start_node]['geometry'], axis=1)

    first_cols = ['network_geometry', 'waterway width (m)']
    sum_cols = ['distance (m)', 'duration (s)', 'total_energy (kWh)',
                'diesel_consumption (g)', 'CO2_emission_total (g)',
                'PM10_emission_total (g)', 'NOX_emission_total (g)']
    mean_cols = ['engine age (year)', 'P_tot (kW)', 'P_given (kW)', 'P_installed (kW)']
    weighted_cols_t = ['diesel_consumption_s (g/s)', 'CO2_emission_per_s (g/s)',
                       'PM10_emission_per_s (g/s)', 'NOX_emission_per_s (g/s)']
    weighted_cols_m = ['diesel_consumption_m (g/m)', 'CO2_emission_per_m (g/m)',
                       'PM10_emission_per_m (g/m)', 'NOX_emission_per_m (g/m)']

    # Keep only columns that exist
    first_cols = [c for c in first_cols if c in df_expanded.columns]
    sum_cols = [c for c in sum_cols if c in df_expanded.columns]
    mean_cols = [c for c in mean_cols if c in df_expanded.columns]
    weighted_cols_t = [c for c in weighted_cols_t if c in df_expanded.columns]
    weighted_cols_m = [c for c in weighted_cols_m if c in df_expanded.columns]

    # Build aggregation dictionary
    agg_dict = {col: 'sum' for col in sum_cols}
    agg_dict.update({col: 'first' for col in first_cols})
    agg_dict.update({col: 'mean' for col in mean_cols})
    if 'object id' in df_expanded.columns:
        agg_dict['object id'] = lambda x: list(pd.unique(x))

    # Group by geometry key
    df_expanded['geom_key'] = df_expanded['network_geometry'].apply(lambda g: g.wkt)
    grouped = df_expanded.groupby('geom_key')
    df_edges = grouped.agg(agg_dict).reset_index()

    # Weighted averages
    for col in weighted_cols_t:
        weight_col = 'duration (s)'
        if weight_col in df_expanded.columns:
            df_edges[col] = grouped.apply(lambda g: weighted_avg(g, col, weight_col),
                                          include_groups=False).values

    for col in weighted_cols_m:
        weight_col = 'distance (m)'
        if weight_col in df_expanded.columns:
            df_edges[col] = grouped.apply(lambda g: weighted_avg(g, col, weight_col),
                                          include_groups=False).values

    # Renaming
    renaming_columns = {
        'distance (m)': 'total_distance_sailed (m)',
        'duration (s)': 'total_residence_time (s)',
        'P_tot (kW)': 'average_P_tot (kW)',
        'P_given (kW)': 'average_P_given (kW)',
        'P_installed (kW)': 'average_P_installed (kW)',
        'engine age (year)': 'average_engine_age (year)',
        'total_energy (kWh)': 'total_energy_consumed (kWh)',
        'diesel_consumption (g)': 'total_diesel_consumed (g)',
        'CO2_emission_total (g)': 'total_CO2_emitted (g)',
        'PM10_emission_total (g)': 'total_PM10_emitted (g)',
        'NOX_emission_total (g)': 'total_NOX_emitted (g)',
        'diesel_consumption_s (g/s)': 'average_diesel_consumption_s (g/s)',
        'CO2_emission_per_s (g/s)': 'average_CO2_emission_per_s (g/s)',
        'PM10_emission_per_s (g/s)': 'average_PM10_emission_per_s (g/s)',
        'NOX_emission_per_s (g/s)': 'average_NOX_emission_per_s (g/s)',
        'diesel_consumption_m (g/m)': 'average_diesel_consumption_m (g/m)',
        'CO2_emission_per_m (g/m)': 'average_CO2_emission_per_m (g/m)',
        'PM10_emission_per_m (g/m)': 'average_PM10_emission_per_m (g/m)',
        'NOX_emission_per_m (g/m)': 'average_NOX_emission_per_m (g/m)',
        'network_geometry': 'geometry',
        'object id': 'vessel_ids',
    }

    # Filter renaming to only existing columns
    renaming_columns = {k: v for k, v in renaming_columns.items() if k in df_edges.columns}

    df_edges = df_edges.rename(columns=renaming_columns)
    df_edges = df_edges[list(renaming_columns.values())]

    gdf_edges = gpd.GeoDataFrame(df_edges, geometry='geometry', crs='EPSG:4326')
    return gdf_edges
