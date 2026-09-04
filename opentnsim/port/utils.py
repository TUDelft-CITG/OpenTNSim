import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import shapely
import inspect
import textwrap
import inspect
import re
import ast
from opentnsim.graph.utils import get_sailing_time, get_trajectory, node_path_to_edge_path
from opentnsim.graph.calculations import transform_geometry
from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from opentnsim.port.mixins.rules import RuleEngine, TrafficRules
pd.options.mode.chained_assignment = None


def get_vessel_from_id(env, vessel_ids):
    vessels = []
    for vessel_id in vessel_ids:
        for vessel in env.vessels:
            if vessel.id == vessel_id:
                vessels.append(vessel)
    return vessels


def create_logbook_with_directed_distances(vessel, route, epsg_out='EPSG:4087'):
    first_index = 0
    df = pd.DataFrame(vessel.logbook)
    corrected_df = pd.DataFrame()
    for _, route_sailed in enumerate(vessel.routes_sailed):
        selected_route = [node for node in route_sailed if node in route]
        route_geometry, _ = get_trajectory(vessel.env.graph,route[0],route[-1])
        route_geometry_m = transform_geometry(route_geometry, epsg_out = epsg_out)
        mask = df.index > first_index
        mask2 = df[mask].Message.apply(lambda x: bool(re.search(rf'\b{selected_route[-1]}\b', x)) and 'stop' in x)
        last_index_df = df[mask][mask2]
        if last_index_df.empty:
            continue
        last_index = last_index_df.iloc[0].name
        df_route = df[(df.index >= first_index) & (df.index <= last_index)]
        gdf_route = gpd.GeoDataFrame(df_route,geometry='Geometry',crs="EPSG:4326")
        gdf_route = gdf_route.to_crs(epsg_out)
        df_route['Value'] = gdf_route['Geometry'].apply(lambda x: route_geometry_m.project(x))
        first_index = last_index + 1
        corrected_df = pd.concat([corrected_df, df_route])
    corrected_df = corrected_df.ffill()
    return corrected_df


def update_terminal_planning(vessel, delay=0.):
    #TODO: this probably does not work any longer
    vessel.terminal.replan_vessels_terminal_berths(vessel,delay)
    for queued_vessel_id, _ in vessel.terminal.queue.iterrows():
        queued_vessel = get_vessel_from_id(vessel.env, [queued_vessel_id])[0]
        if hasattr(queued_vessel,'waiting_event') and queued_vessel.waiting_event.is_alive:
            queued_vessel.waiting_event.interrupt()


def check_if_vessel_is_tide_bound(df_tidal_availability):
    tide_bound = False
    if False in df_tidal_availability.values:
        tide_bound = True
    return tide_bound


def determine_nearest_anchorage_area(vessel, node, route = None):
    # Loop over the nodes of the network and identify all the anchorage areas:
    sailing_time_to_anchorages = []
    capacity_of_anchorages = []
    anchorage_area_found = False
    while not anchorage_area_found:
        anchorage_area_names = []
        for anchorage_area_name, anchorage_area in vessel.port.anchorage_areas.items():
            if route is not None and anchorage_area.node not in route:
                continue

            # Determine if the anchorage area can be reached
            route_to_anchorage = nx.dijkstra_path(vessel.env.graph, node, anchorage_area.node)
            edge_route_to_anchorage = node_path_to_edge_path(vessel.env.graph, route_to_anchorage)
            sailing_time_to_anchorage, _ = get_sailing_time(vessel, edge_route_to_anchorage)
            sailing_time_to_anchorages.append(sailing_time_to_anchorage)
            anchorage_capacity = anchorage_area.resource.capacity 
            anchorage_intensity = anchorage_area.resource.count
            anchorage_level = anchorage_capacity - anchorage_intensity
            capacity_of_anchorages.append(anchorage_level > 0)
            anchorage_area_names.append(anchorage_area_name)
        anchorage_selection_df = pd.DataFrame({'Sailing time': sailing_time_to_anchorages,
                                               'Capacity': capacity_of_anchorages},
                                               index = anchorage_area_names)
        if anchorage_selection_df.empty:
            route = None
        else:
            anchorage_area_found = True
    suitable_anchorage_areas = anchorage_selection_df[anchorage_selection_df.Capacity]
    suitable_anchorage_areas = suitable_anchorage_areas.sort_values('Sailing time')
    anchorage_area = vessel.port.anchorage_areas[suitable_anchorage_areas.iloc[0].name]
    return anchorage_area


def determine_if_vessel_needs_to_sail_to_the_anchorage_area(env, vessel, origin, waiting_time):
    sail_to_anchorage_area = False
    nearest_anchorage_area = determine_nearest_anchorage_area(vessel, origin)
    route_to_anchorage_area = nx.dijkstra_path(env.graph, origin, nearest_anchorage_area.node)
    route_after_anchorage_area = nx.dijkstra_path(env.graph, nearest_anchorage_area.node, vessel.route[-1])
    edge_route_to_terminal = node_path_to_edge_path(vessel.env.graph, vessel.route)
    edge_route_to_anchorage = node_path_to_edge_path(vessel.env.graph, route_to_anchorage_area)
    edge_route_after_anchorage = node_path_to_edge_path(vessel.env.graph, route_after_anchorage_area)
    sailing_time_to_terminal, _ = get_sailing_time(vessel, edge_route_to_terminal)
    sailing_time_to_anchorage_area, _ = get_sailing_time(vessel, edge_route_to_anchorage)
    new_sailing_time_to_terminal, _ = get_sailing_time(vessel, edge_route_after_anchorage)
    delay_of_sailing_to_terminal = new_sailing_time_to_terminal - sailing_time_to_terminal + sailing_time_to_anchorage_area
    if delay_of_sailing_to_terminal <= waiting_time and delay_of_sailing_to_terminal > 0.:
        sail_to_anchorage_area = True
    return sail_to_anchorage_area


def get_berth(vessel, berths):
    berth = None
    if not hasattr(vessel,'berth'):
        return berth
    for b in berths:
        if b.name == vessel.berth:
            berth = b
            break
    return berth


def determine_new_route_for_vessel(vessel):
    new_route = None
    origin = vessel.route[-1]
    if vessel.next_destination is not None:
        destination = vessel.next_destination
        new_route = nx.dijkstra_path(vessel.env.graph,origin,destination)
    elif len(vessel.next_terminals):
        vessel.next_terminals = vessel.next_terminals[1:]
    return new_route


# def determine_vessel_waiting_events(port_accessed, vessel, port_availability_df, conflict_df, delay = 0.):
#     df = port_availability_df.copy()
#     df['Combined'] = df.all(axis=1)

#     with pd.option_context("future.no_silent_downcasting", True):
#         df = df.ffill().bfill()
#     df = df[df != df.shift()].dropna(how='all')
#     with pd.option_context("future.no_silent_downcasting", True):
#         df = df.ffill()
    
#     df = df.join(conflict_df[['edges', 'conflict_type', 'vessels_in_conflict']],how='left')

#     def get_waiting_time_reason(lst):
#         if not lst:
#             return ""
#         elif len(lst) == 1:
#             return lst[0]
#         else:
#             return ", ".join(lst[:-1]) + " and " + lst[-1]

#     current_time = datetime.datetime.fromtimestamp(vessel.env.now) + pd.Timedelta(seconds = delay)
#     previous_events = df[df.index <= current_time]
#     future_events = df[df.index > current_time]
#     last_previous_event_index = previous_events.index.max()
#     if pd.isna(last_previous_event_index):
#         previous_event = df.iloc[0:0]
#     else:
#         previous_event = df.loc[[last_previous_event_index]]
#     df = pd.concat([previous_event, future_events])
#     idx = df.index.to_list()
#     idx[0] = current_time
#     df.index = idx

#     cols_to_check = df.columns.drop('Combined')
#     df['Reason'] = df[cols_to_check].apply(
#         lambda row: get_waiting_time_reason(list(row[row.eq(False)].index)),
#         axis=1)

#     port_available_df = df[df['Combined'] == True]
#     waiting_events_dict = {}
#     conflict_edges_dict = {}
#     conflicts_dict = {}
#     vessels_in_conflict_dict = {}
#     if port_available_df.empty:
#         return waiting_events_dict, conflict_edges_dict, conflicts_dict, vessels_in_conflict_dict
#     waiting_time_end = port_available_df.iloc[0].name
#     waiting_events = df.loc[:waiting_time_end]
#     waiting_reasons = waiting_events['Reason'][:-1]
#     conflict_edges = waiting_events['edges'][:-1]
#     conflicts = waiting_events['conflict_type'][:-1]
#     vessels_in_conflict = waiting_events['vessels_in_conflict'][:-1]
#     waiting_times = (waiting_events.index.to_series().shift(-1) - waiting_events.index).apply(lambda x: x.total_seconds())
#     for index, (waiting_reason, waiting_time, edge, conflict, vessels) in enumerate(zip(waiting_reasons, waiting_times, conflict_edges, conflicts, vessels_in_conflict)):
#         index += 1
#         waiting_events_dict[waiting_reason + f' ({index})'] = waiting_time
#         conflict_edges_dict[waiting_reason + f' ({index})'] = edge
#         conflicts_dict[waiting_reason + f' ({index})'] = conflict
#         vessels_in_conflict_dict[waiting_reason + f' ({index})'] = vessels
#     vessel.port_accessed = port_accessed
#     return waiting_events_dict, conflict_edges_dict, conflicts_dict, vessels_in_conflict_dict

def determine_vessel_waiting_events(
    port_accessed,
    vessel,
    port_availability_df,
    conflict_df,
    delay=0.0,
):
    """
    Determine the waiting events for a vessel.

    The conflict_df is expected to contain conflict information with
    potentially multiple rows per conflict block. Conflict information
    is propagated over the corresponding waiting period.

    Returns
    -------
    waiting_events_dict
    conflict_edges_dict
    conflicts_dict
    vessels_in_conflict_dict
    """

    # 1. Prepare availability dataframe
    df = port_availability_df.copy()

    # Overall availability
    df["Combined"] = df.all(axis=1)

    # Remove consecutive duplicate availability states
    with pd.option_context("future.no_silent_downcasting", True):
        df = df.ffill().bfill()

    df = df[df.ne(df.shift()).any(axis=1)]

    with pd.option_context("future.no_silent_downcasting", True):
        df = df.ffill()

    # 2. Add conflict information
    conflict_cols = [
        "edges",
        "conflict_type",
        "vessels_in_conflict",
        "rules",
        "downtime",
        "_block",
    ]

    available_conflict_cols = [
        col for col in conflict_cols
        if col in conflict_df.columns
    ]

    conflict_info = conflict_df[available_conflict_cols].copy()

    # Make sure both indexes are datetime
    conflict_info.index = pd.to_datetime(conflict_info.index)
    df.index = pd.to_datetime(df.index)

    # Exact join first
    df = df.join(
        conflict_info,
        how="left",
    )

    # 3. Propagate conflict information through the waiting period
    conflict_cols_in_df = [
        col for col in available_conflict_cols
        if col in df.columns
    ]

    if conflict_cols_in_df:
        with pd.option_context("future.no_silent_downcasting", True):
            df[conflict_cols_in_df] = df[conflict_cols_in_df].ffill()

    # 4. Determine current simulation time
    current_time = (
        datetime.datetime.fromtimestamp(vessel.env.now)
        + pd.Timedelta(seconds=delay)
    )

    previous_events = df[df.index <= current_time]
    future_events = df[df.index > current_time]

    last_previous_event_index = previous_events.index.max()

    if pd.isna(last_previous_event_index):
        previous_event = df.iloc[0:0]
    else:
        previous_event = df.loc[[last_previous_event_index]]

    df = pd.concat([previous_event, future_events])

    # Replace first timestamp with the actual current simulation time
    if len(df) > 0:
        idx = df.index.to_list()
        idx[0] = current_time
        df.index = idx

    # 5. Determine why the vessel is waiting
    def get_waiting_time_reason(row):
        reasons = []

        for column in port_availability_df.columns:
            if column in row.index and row[column] is False:
                reasons.append(column)

        if not reasons:
            return ""

        if len(reasons) == 1:
            return reasons[0]

        return ", ".join(reasons[:-1]) + " and " + reasons[-1]

    cols_to_check = [
        col
        for col in port_availability_df.columns
        if col != "Combined"
    ]

    df["Reason"] = df[cols_to_check].apply(
        get_waiting_time_reason,
        axis=1,
    )

    # 6. Find first time at which everything becomes available
    port_available_df = df[df["Combined"] == True]

    waiting_events_dict = {}
    conflict_edges_dict = {}
    conflicts_dict = {}
    vessels_in_conflict_dict = {}
    rules = {}
    downtimes = {}

    if port_available_df.empty:
        vessel.port_accessed = port_accessed

        return (
            waiting_events_dict,
            conflict_edges_dict,
            conflicts_dict,
            vessels_in_conflict_dict,
            rules,
            downtimes,
        )

    waiting_time_end = port_available_df.iloc[0].name

    waiting_events = df.loc[:waiting_time_end]

    # 7. Calculate duration of every waiting interval
    waiting_times = (
        waiting_events.index.to_series().shift(-1)
        - waiting_events.index
    ).apply(
        lambda x: x.total_seconds()
        if pd.notna(x)
        else None
    )

    # 8. Build dictionaries
    event_number = 0
    for i in range(len(waiting_events) - 1):

        row = waiting_events.iloc[i]

        waiting_reason = row["Reason"]

        # Ignore rows where there is no actual waiting
        if not waiting_reason:
            continue

        waiting_time = waiting_times.iloc[i]

        if waiting_time is None:
            continue

        event_number += 1

        key = f"{waiting_reason} ({event_number})"

        # Conflict information
        edge = row.get("edges", None)
        conflict = row.get("conflict_type", None)
        vessels = row.get("vessels_in_conflict", None)
        rule = row.get("rules", None)
        downtime = row.get("downtime", None)

        # Convert NaN to None
        if isinstance(edge, float) and pd.isna(edge):
            edge = None

        if isinstance(conflict, float) and pd.isna(conflict):
            conflict = None

        if isinstance(vessels, float) and pd.isna(vessels):
            vessels = None

        # Store
        waiting_events_dict[key] = waiting_time
        conflict_edges_dict[key] = edge
        conflicts_dict[key] = conflict
        vessels_in_conflict_dict[key] = vessels
        rules[key] = rule
        downtimes[key] = downtime

    # 9. Restore vessel state
    vessel.port_accessed = port_accessed
    
    return (
        waiting_events_dict,
        conflict_edges_dict,
        conflicts_dict,
        vessels_in_conflict_dict,
        rules,
        downtimes
    )

def determine_vessel_priority(vessel, tide_bound = False, leaving_port = False):
    priority = 0
    if tide_bound:
        priority += 1
    if leaving_port:
        priority += 1
    vessel.priority = priority
    return priority


def get_accessibility_info(vessel, origin, berth = None, leaving_port = False):
    df_tidal_availability_per_waterway = get_tidal_availability_info(vessel)
    df_terminal_availability = get_terminal_availability_info(vessel, origin, berth, leaving_port)
    df_waterway_availability_per_waterway, conflicts_dfs = get_waterway_availability_info(vessel, origin)
    waterways = find_waterways_to_be_passed(vessel)

    #Combine the dataframes
    current_time = datetime.datetime.fromtimestamp(vessel.env.now) - pd.Timedelta(hours=24)
    dfs = []
    for waterway_name in waterways.keys():
        df_tidal_availability = df_tidal_availability_per_waterway[waterway_name]
        df_waterway_availability = df_waterway_availability_per_waterway[waterway_name]
        df_waterway_availability = df_waterway_availability.rename('Traffic')
        port_availability_df = pd.concat(
            [
                df_tidal_availability,
                df_terminal_availability,
                df_waterway_availability,
            ]
        )

        port_availability_df = (
            port_availability_df[port_availability_df.index >= current_time]
            .sort_index()
            .ffill()
            .bfill()
        )

        port_availability_df.columns = pd.MultiIndex.from_product(
            [[waterway_name], port_availability_df.columns]
        )

        dfs.append(port_availability_df)

    port_availability_per_waterway = pd.concat(dfs, axis=1)
    return port_availability_per_waterway, conflicts_dfs


def find_waterways_to_be_passed(vessel):
    passing_waterways = {}

    for node in vessel.route:
        waterway = None
        if "Waterway" in vessel.env.graph.nodes[node]:
            waterway = vessel.env.graph.nodes[node]["Waterway"]
        else:
            continue

        if vessel.position_on_route:
            previous_index = vessel.position_on_route - 1
            previous_node = vessel.route[previous_index]
            if "Waterway" in vessel.env.graph.nodes[previous_node]:
                previous_waterway = vessel.env.graph.nodes[previous_node]["Waterway"]
                if previous_waterway.name == waterway.name:
                    continue

        if waterway.name not in passing_waterways.keys():
            passing_waterways[waterway.name] = waterway

    return passing_waterways


def get_waterway_availability_info(vessel, origin):
    passing_waterways = find_waterways_to_be_passed(vessel)
    df_waterways_availability = pd.DataFrame()
    availability_dfs = []
    conflicts_dfs = []
    for waterway in passing_waterways.values():
        waterway_route = get_oriented_waterway_route(waterway, vessel)
        index_waterway_route_start = vessel.route.index(waterway_route[0])+1
        route_to_waterway_start = vessel.route[:index_waterway_route_start]
        edge_route_to_waterway_start = list(zip(route_to_waterway_start[:-1],route_to_waterway_start[1:]))
        sailing_time_to_waterway, _ = get_sailing_time(vessel, edge_route_to_waterway_start)
        availability_df = waterway.check_waterway_availability_info(vessel, origin, sailing_time_to_waterway)
        availability_df = availability_df.rename(columns={'Traffic': waterway.name})
        conflicts_df = availability_df.copy()
        conflicts_df = conflicts_df.drop(columns = waterway.name)
        availability_df = availability_df[[waterway.name]]
        availability_dfs.append(availability_df)
        conflicts_dfs.append(conflicts_df)
    
    df_waterways_availability = pd.concat(availability_dfs)
    df_waterways_availability = df_waterways_availability.sort_index()
    df_waterways_availability = df_waterways_availability.ffill().bfill()
    if df_waterways_availability.empty:
        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        df_waterways_availability.loc[current_time,'Traffic'] = True
    return df_waterways_availability, conflicts_dfs


def get_terminal_availability_info(vessel, origin, berth = None, leaving_port = False):
    df_terminal_availability = pd.DataFrame()
    if not leaving_port and berth is not None:
        df_terminal_availability = vessel.terminal.provide_terminal_availability_info(vessel, origin, berth)
    return df_terminal_availability


def check_if_route_contains_restrictions(vessel):
    contains_restriction = False
    for node_start, node_end in zip(vessel.route[:-1],vessel.route[1:]):
        edge = (node_start, node_end)
        if 'Depth_restriction' in vessel.env.graph.nodes[node_start].keys():
            contains_restriction = True
            break
        elif 'Depth_restriction' in vessel.env.graph.nodes[node_end].keys():
            contains_restriction = True
            break
        elif 'Depth_restriction' in vessel.env.graph.edges[edge].keys():
            contains_restriction = True
            break
    return contains_restriction


def get_tidal_availability_info(vessel):
    from opentnsim.port.calculations import calculate_tidal_windows

    has_tidal_window_policy = check_if_route_contains_restrictions(vessel)
    route = vessel.route
    time_start = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now)) - np.timedelta64(12,'h')

    edge_route = node_path_to_edge_path(vessel.env.graph, route)
    sailing_time, _ = get_sailing_time(vessel, edge_route)
    sailing_time = max(pd.Timedelta(seconds=sailing_time), pd.Timedelta(hours=96))

    time_end = np.datetime64(
        datetime.datetime.fromtimestamp(vessel.env.now) + sailing_time
    )

    tidal_window_results = pd.DataFrame(columns=["Accessibility"])
    if has_tidal_window_policy:
        tidal_window_results = calculate_tidal_windows(vessel, route, time_start, time_end)
        vessel.tidal_window_calculations[vessel.trip_index] = tidal_window_results

    frames = []

    for waterway_name, sub_trip in tidal_window_results.iterrows():
        df_tidal_availability = (
            sub_trip["tidal_accessibility"][["Accessibility"]]
            .eq("Accessible")
            .rename(columns={"Accessibility": "Tide"})
        )

        # convert to MultiIndex: (waterway_name, Tide)
        df_tidal_availability.columns = pd.MultiIndex.from_product(
            [[waterway_name], df_tidal_availability.columns]
        )

        frames.append(df_tidal_availability)

    if frames:
        df_tidal_availability_waterways = pd.concat(frames, axis=1)
    else:
        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        df_tidal_availability_waterways = pd.DataFrame(
            {("single_waterway", "Tide"): [True]},
            index=[current_time],
        )

    df_tidal_availability_waterways = (
        df_tidal_availability_waterways
        .sort_index(axis=1)
        .ffill()
        .bfill()
    )

    return df_tidal_availability_waterways


def provide_trajectory(env, node_1, node_2):
    nodes = nx.dijkstra_path(env.graph, node_1, node_2)
    for loc, edge in enumerate(zip(nodes[:-1], nodes[1:])):
        geom = env.graph.edges[edge[0], edge[1]]["geometry"]
        if loc:
            multi_line = shapely.geometry.MultiLineString([final_geometry, geom])
            final_geometry = shapely.ops.linemerge(multi_line)
        else:
            final_geometry = geom
    return final_geometry


def provide_waiting_time_for_inbound_tidal_window(vessel, route, time_start=None, time_stop=None, delay=0):
    """Function: calculates the time that a vessel has to wait depending on the available tidal windows

    Input:
        - vessel: an identity which is Identifiable, Movable, and Routeable, and has VesselProperties
        - route: a list of strings that resemble the route of the vessel (can be different than the vessel.route)
        - delay: a delay that can be included to calculate a future situation

    """
    from opentnsim.port.calculations import calculate_tidal_windows

    # Create sub-routes based on anchorage areas on the route
    if not time_start:
        time_start = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now)).to_datetime64()
    if not time_stop:
        time_stop = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now + pd.Timedelta(days=2).total_seconds())).to_datetime64()

    _, tidal_windows = calculate_tidal_windows(vessel, route, time_start, time_stop, delay)

    waiting_time = pd.Timedelta('NaT')
    for window in tidal_windows:
        if time_start > window[1]:
            continue
        if time_start >= window[0]:
            waiting_time = pd.Timedelta(0, "s")
        else:
            waiting_time = window[0] - time_start
        break

    waiting_time = waiting_time.total_seconds()
    return waiting_time

def provide_waiting_time_for_outbound_tidal_window(vessel, route, delay=0):
    vessel.bound = "outbound"
    vessel._T -= vessel.metadata["(un)loading"][0]
    waiting_time = provide_waiting_time_for_inbound_tidal_window(vessel, route=route, delay=delay)
    vessel._T += vessel.metadata["(un)loading"][0]
    vessel.bound = "inbound"
    return waiting_time


def provide_nearest_anchorage_area(vessel, node):
    from opentnsim.port.calculations import calculate_ukc_clearance
    hydromanager = HydrodynamicDataManager()
    hydrodynamic_information = hydromanager.hydrodynamic_data
    nodes_of_anchorages = []
    capacity_of_anchorages = []
    users_of_anchorages = []
    sailing_times_to_anchorages = []
    # Loop over the nodes of the network and identify all the anchorage areas:
    for node_anchorage in vessel.multidigraph.nodes:
        if "Anchorage Area" in vessel.multidigraph.nodes[node_anchorage]:
            # Determine if the anchorage area can be reached
            anchorage_reachable = True
            route_to_anchorage = nx.dijkstra_path(vessel.multidigraph, node, node_anchorage)
            for node_on_route in route_to_anchorage:
                station_index = list(hydrodynamic_information["STATION"]).index(node_on_route)
                min_water_level = np.min(hydrodynamic_information["Water level"][station_index].values)
                _, _, _, required_water_depth, _, MBL = calculate_ukc_clearance(vessel, node)
                if min_water_level + MBL < required_water_depth:
                    anchorage_reachable = False
                    break

            if not anchorage_reachable:
                continue

            # Extract information over the individual anchorage areas:
            # capacity, users, and the sailing distance to the anchorage area
            # from the designated terminal the vessel is planning to call
            nodes_of_anchorages.append(node_anchorage)
            capacity_of_anchorages.append(vessel.multidigraph.nodes[node_anchorage]["Anchorage"][0].resource.capacity)
            users_of_anchorages.append(len(vessel.multidigraph.nodes[node_anchorage]["Anchorage"][0].resource.users))
            route_from_anchorage = nx.dijkstra_path(vessel.multidigraph, node_anchorage, vessel.route[-1])
            edge_route_from_anchorage = node_path_to_edge_path(vessel.env.graph, route_from_anchorage)
            sailing_time_to_anchorage = get_sailing_time(vessel, edge_route_from_anchorage)["Time"].sum()
            sailing_times_to_anchorages.append(sailing_time_to_anchorage)

    # Sort the lists based on the sailing distance to the anchorage area from the designated terminal
    #  the vessel is planning to call
    sorted_nodes_anchorages = [nodes for (distances, nodes) in sorted(zip(sailing_times_to_anchorages, nodes_of_anchorages))]
    sorted_users_of_anchorages = [nodes for (distances, nodes) in sorted(zip(sailing_times_to_anchorages, users_of_anchorages))]
    sorted_capacity_of_anchorages = [
        nodes for (distances, nodes) in sorted(zip(sailing_times_to_anchorages, capacity_of_anchorages))
    ]

    # Take the anchorage area that is closest to the designated terminal the vessel is planning to call if there
    # is sufficient capacity:
    node_anchorage = 0
    for anchorage_index, node_anchorage in enumerate(sorted_nodes_anchorages):
        if sorted_users_of_anchorages[anchorage_index] < sorted_capacity_of_anchorages[anchorage_index]:
            # node anchorage is found
            break

    return node_anchorage


def add_ukc_policy_to_edge(graph, edge, rules, distance_of_series = None):

    u, v = edge[:2]

    edge_route = [edge]
    if not graph.has_edge(u, v):
        route = nx.dijkstra_path(graph, u, v)
        edge_route = list(zip(route[:-1],route[1:]))
    
    if distance_of_series is None:
        distance_of_series = graph.edges[edge]['length_m']/2
    
    distance_of_series_rev = graph.edges[edge]['length_m'] - distance_of_series
    for edge in edge_route:
        if "Depth_restriction" not in graph.edges[edge]:
            graph.edges[edge]["Depth_restriction"] = RuleEngine(default = lambda v: 0.0)
            graph.edges[edge]["Depth_restriction_at_distance"] = lambda e: distance_of_series if e == edge else distance_of_series_rev
    
        for rule in rules:
            graph.edges[edge]["Depth_restriction"].add_rule(
                condition=rule.condition,
                policy=rule.policy,
                name=rule.name,
            )


def add_traffic_encountering_restriction_to_edge(
        graph, 
        edge, 
        rules, 
        name = "Traffic_encountering_restriction",
        default_value = TrafficRules.allowed,
    ):

    u, v = edge[:2]

    edge_route = [edge]
    if not graph.has_edge(u, v):
        route = nx.dijkstra_path(graph, u, v)
        edge_route = list(zip(route[:-1],route[1:]))

    for edge in edge_route:
        if name not in graph.edges[edge]:
            graph.edges[edge][name] = RuleEngine(default = default_value)
        
        for rule in rules:
            graph.edges[edge][name].add_rule(
                condition=rule.condition,
                policy=rule.policy,
                name=rule.name,
            )

def add_traffic_overtaking_restriction_to_edge(graph, edge, rules, name = "Traffic_overtaking_restriction"):
    add_traffic_encountering_restriction_to_edge(graph, edge, rules, name = name)

def add_traffic_reservation_to_edge(graph, edge, rules, name = "Traffic_reservation"):
    add_traffic_encountering_restriction_to_edge(graph, edge, rules, name = name)

def add_traffic_encountering_exception_to_edge(graph, edge, rules, name = "Traffic_encountering_exception"):
    add_traffic_encountering_restriction_to_edge(graph, edge, rules, name = name, default_value = TrafficRules.prohibited)

def add_traffic_overtaking_exception_to_edge(graph, edge, rules, name = "Traffic_overtaking_restriction"):
    add_traffic_encountering_exception_to_edge(graph, edge, rules, name = name)


def render_rule(node, indent=0):

    pad = "  " * indent

    if isinstance(node, AnyOf):
        lines = [" "]

        lines.append(
            render_rule(node.expr, indent)
        )

        return "\n".join(lines)

    if isinstance(node, And):

        lines = [f"{pad}ALL of:"]

        for item in node.items:
            lines.append(
                render_rule(item, indent + 1)
            )

        return "\n".join(lines)

    if isinstance(node, Or):

        lines = [f"{pad}ANY of:"]

        for item in node.items:
            lines.append(
                render_rule(item, indent + 1)
            )

        return "\n".join(lines)

    if isinstance(node, Compare):

        return (
            f"{pad}"
            f"{node.left} {node.op} {node.right}"
        )

    if isinstance(node, Call):

        return (
            f"{pad}"
            f"{node.name}("
            f"{', '.join(node.args)})"
        )

    return f"{pad}{node}"


def parse_rule(fn):
    from opentnsim.port.mixins.rules import RuleParser
    expr = extract_return_expr(fn)
    parser = RuleParser(namespace=fn.__globals__)
    return parser.visit(expr)


def extract_return_expr(fn):

    # get raw source
    source = textwrap.dedent(inspect.getsource(fn))

    # parse independently
    tree = ast.parse(source)

    for node in ast.walk(tree):

        # CASE 1: return in def
        if isinstance(node, ast.Return):
            return node.value

        # CASE 2: lambda body
        if isinstance(node, ast.Lambda):
            return node.body

    raise ValueError(
        f"Cannot extract expression from {fn}. "
        f"AST root types: {[type(n) for n in tree.body]}"
    )

def get_vessel_direction_with_waterway(routeA, routeB):
    shared = set(routeA) & set(routeB)

    A_shared = [n for n in routeA if n in shared]
    B_shared = [n for n in routeB if n in shared]

    direction = 0 if A_shared == B_shared else 1
    return direction

def get_oriented_waterway_route(waterway, vessel):
    waterway_route = waterway.route
    direction = get_vessel_direction_with_waterway(waterway_route, vessel.route)
    if direction:
        waterway_route = waterway.route_reversed
    return waterway_route