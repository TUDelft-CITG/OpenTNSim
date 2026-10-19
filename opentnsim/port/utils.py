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
import heapq
from collections import defaultdict
from opentnsim.graph.utils import get_sailing_time, get_trajectory, node_path_to_edge_path
from opentnsim.graph.calculations import transform_geometry
from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from opentnsim.port.mixins.rules import RuleEngine, TrafficRules
pd.options.mode.chained_assignment = None


def get_vessel_from_id(env, vessel_ids):
    vessels = []
    for vessel_id in vessel_ids:
        for vessel in env.vessels.values():
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


def determine_vessel_waiting_events(
    port,
    port_availability_df,
    delay = 0.,
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

    # 1) Prepare availability dataframe
    df = port_availability_df.copy()

    # Overall availability
    df["Combined"] = df.all(axis=1)

    # Remove consecutive duplicate availability states
    with pd.option_context("future.no_silent_downcasting", True):
        df = df.ffill().bfill()

    df = df[df.ne(df.shift()).any(axis=1)]

    with pd.option_context("future.no_silent_downcasting", True):
        df = df.ffill()

    # 2) Remove True periods shorter than 1 minute
    combined = df["Combined"].astype("boolean")
    groups = combined.ne(combined.shift()).cumsum()
    run_start = df.index.to_series().groupby(groups).transform("first")
    next_run_start = run_start.groupby(groups).transform("first").shift(-1)
    next_run_value = combined.groupby(groups).transform("first").shift(-1)
    remove = (
        combined.eq(True)
        & next_run_value.eq(False)
        & next_run_start.notna()
        & (next_run_start - run_start < pd.Timedelta(minutes=1))
    )
    df = df.loc[~remove]

    # 3) Determine why the vessel is waiting
    def get_waiting_time_reason(row, cols_to_check):
        reasons = []

        for column in cols_to_check:
            if column in row.index and row[column] is not True:
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
        lambda row: get_waiting_time_reason(row, cols_to_check),
        axis=1,
    )

    # 4) Find first time at which everything becomes available
    current_time = pd.Timestamp(datetime.datetime.fromtimestamp(port.env.now))
    current_time += pd.Timedelta(seconds = delay)
    first_pos = df.index.searchsorted(current_time, side="right") - 1
    df = df.iloc[max(first_pos, 0):]  

    port_available_df = df[df["Combined"] == True]
    waiting_events_dict = {}
    if port_available_df.empty:
        # trip not possible
        waiting_events_dict['Not possible'] = None
        return waiting_events_dict
    
    first_entry_time = port_available_df.iloc[0].name
    waiting_events = df.loc[:first_entry_time]
    waiting_events.index = [current_time, *waiting_events.index[1:]]

    # 5) Calculate duration of every waiting interval
    waiting_times = (
        waiting_events.index.to_series().shift(-1)
        - waiting_events.index
    ).apply(
        lambda x: x.total_seconds()
        if pd.notna(x)
        else None
    )

    # 6) Build dictionaries
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

        # Store
        waiting_events_dict[key] = waiting_time

    return waiting_events_dict


def get_accessibility_info(vessel, origin, waterways, terminal, berth, delay = pd.Timedelta(seconds = 0)):
    df_tidal_availability_per_waterway = get_tidal_availability_info(vessel, delay)
    df_terminal_availability = get_terminal_availability_info(vessel, origin, terminal, berth, delay)
    df_waterway_availability_per_waterway, conflicts_dfs = get_waterway_availability_info(vessel, origin, waterways, delay)
    
    #Combine the dataframes
    current_time = datetime.datetime.fromtimestamp(vessel.env.now) - pd.Timedelta(hours=24)
    dfs = []
    for _, waterway_info in waterways.iterrows():
        waterway_name = waterway_info.waterway_name
        df_tidal_availability = df_tidal_availability_per_waterway[waterway_name]
        df_waterway_availability = df_waterway_availability_per_waterway[waterway_name]

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
    passing_waterways = pd.DataFrame(
        columns = ['waterway_name', 'waterway', 'start_node', 'stop_node','time_sailing_start', 'time_sailing_stop'],
        )

    current_time = datetime.datetime.fromtimestamp(vessel.env.now)
    time_sailing_start = current_time
    route_start_index = vessel.position_on_route
    for index, node in enumerate(vessel.route):
        next_node = node
        previous_node = node
        if index:
            previous_index = index - 1
            previous_node = vessel.route[previous_index]

        previous_waterways = []
        if "Waterway" in vessel.env.graph.nodes[previous_node]:
            previous_waterways = vessel.env.graph.nodes[previous_node]["Waterway"]

        next_waterways = []
        if "Waterway" in vessel.env.graph.nodes[next_node]:
            next_waterways = vessel.env.graph.nodes[next_node]["Waterway"]

        common_waterways = list(set(previous_waterways) & set(next_waterways))
        for waterway in common_waterways:
            if waterway.name in passing_waterways.waterway_name.values:
                continue
            waterway_route = get_oriented_waterway_route(waterway, vessel)
            waterway_start_node, waterway_stop_node = waterway_route[0], waterway_route[-1]
            waterway_route_start_index = vessel.route.index(waterway_start_node)
            route_to_waterway = vessel.route[route_start_index:(waterway_route_start_index+1)]
            edge_route = list(zip(route_to_waterway[:-1],route_to_waterway[1:]))
            sailing_time_to_waterway, _ = get_sailing_time(vessel, edge_route)
            last_message = pd.DataFrame(vessel.logbook).iloc[-1] if len(vessel.logbook) > 0 else None
            if last_message is not None and 'Sailing' in last_message.Message:
                start_time_sailing_on_current_node = last_message.Timestamp
                sailing_time_on_current_edge = current_time - start_time_sailing_on_current_node
                sailing_time_to_waterway -= sailing_time_on_current_edge.total_seconds()

            if passing_waterways.empty and waterway_start_node != vessel.route[0]:
                index = 0
                start_node = vessel.route[0]
                stop_node = waterway_start_node
                time_sailing_stop = time_sailing_start + pd.Timedelta(seconds = sailing_time_to_waterway)
                passing_waterways.loc[index, :] = [
                    'trip_start', None, start_node, stop_node, time_sailing_start, time_sailing_stop]
            
            index = len(passing_waterways)
            time_sailing_start += pd.Timedelta(seconds = sailing_time_to_waterway)
            edge_route_waterway = list(zip(waterway_route[:-1],waterway_route[1:]))
            sailing_time_on_waterway, _ = get_sailing_time(vessel, edge_route_waterway)
            time_sailing_stop = time_sailing_start + pd.Timedelta(seconds = sailing_time_on_waterway)
            passing_waterways.loc[index, :] = [
                waterway.name, waterway, waterway_start_node, waterway_stop_node, time_sailing_start, time_sailing_stop]
            route_start_index = vessel.route.index(waterway_stop_node)
            time_sailing_start = time_sailing_stop

    if passing_waterways.empty:
        index = 0
        start_node, stop_node = vessel.route[0], vessel.route[-1]
        time_sailing_start = current_time
        sailing_time_on_waterway, _ = get_sailing_time(vessel, vessel.edge_route)
        time_sailing_stop = time_sailing_start + pd.Timedelta(seconds = sailing_time_on_waterway)
        passing_waterways.loc[index, :] = [
            'trip_start', None, start_node, stop_node, time_sailing_start, time_sailing_stop]
    elif waterway_stop_node != vessel.route[-1]:
        index = len(passing_waterways)
        start_node, stop_node = waterway_stop_node, vessel.route[-1]
        route_over_waterway = vessel.route[route_start_index:]
        edge_route_over_waterway = list(zip(route_over_waterway[:-1],route_over_waterway[1:]))
        sailing_time_on_waterway, _ = get_sailing_time(vessel, edge_route_over_waterway)
        time_sailing_stop = time_sailing_start + pd.Timedelta(seconds = sailing_time_on_waterway)
        passing_waterways.loc[index, :] = [
            'trip_end', None, start_node, stop_node, time_sailing_start, time_sailing_stop]

    passing_waterways['time_sailing_start'] = pd.to_datetime(passing_waterways['time_sailing_start'])
    passing_waterways['time_sailing_stop'] = pd.to_datetime(passing_waterways['time_sailing_stop'])
    passing_waterways['sailing_time_to_waterway'] = passing_waterways['time_sailing_start'] - passing_waterways.loc[0, 'time_sailing_start']
    return passing_waterways


def get_waterway_availability_info(
    vessel,
    origin,
    waterways,
    delay=pd.Timedelta(seconds=0),
):
    availability_dfs = []
    conflicts_dfs = []

    for _, waterway_info in waterways.iterrows():
        waterway = waterway_info.waterway
        waterway_name = waterway_info.waterway_name
        sailing_time_to_waterway = (
            waterway_info.sailing_time_to_waterway
        )

        current_time = datetime.datetime.fromtimestamp(
            vessel.env.now
        )

        if waterway is not None:
            availability_df, conflict_df = (
                waterway.check_waterway_availability_info(
                    vessel,
                    origin,
                    sailing_time_to_waterway.total_seconds()
                    + delay.total_seconds(),
                )
            )
            availability_df.columns = pd.MultiIndex.from_tuples(
                [(waterway_name, "Traffic")],
                names=["waterway_name", ""],
            )

        else:
            availability_df = pd.DataFrame(
                { (waterway_name, "Traffic"): [True] },
                index=pd.DatetimeIndex(
                    [current_time], name="time"
                ),
            )
            availability_df.columns = pd.MultiIndex.from_tuples(
                availability_df.columns,
                names=["waterway_name", ""],
            )

            conflict_df = pd.DataFrame(columns=[
                "time_start",
                "time_stop",
                "edge",
                "type",
                "vessel",
                "rule"
                ], 
                index=pd.DatetimeIndex(
                    [], name="time"
                )
            )

        availability_dfs.append(availability_df)
        conflicts_dfs.append(conflict_df)

    df_waterways_availability = pd.concat(availability_dfs, axis=1)
    df_waterways_availability = (df_waterways_availability.sort_index().ffill().bfill())
    return df_waterways_availability, conflicts_dfs


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


def get_tidal_availability_info(vessel, delay = pd.Timedelta(seconds=0)):
    tidal_window_results = vessel.tidal_window_calculations[vessel.trip_id]
    if not tidal_window_results.empty:
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
            current_time = datetime.datetime.fromtimestamp(vessel.env.now) + delay
            df_tidal_availability_waterways = pd.DataFrame(
                {"Tide": [True]},
                index=[current_time],
            )

        df_tidal_availability_waterways = (
            df_tidal_availability_waterways
            .sort_index(axis=1)
            .ffill()
            .bfill()
        )

    else:
        df_tidal_availability_waterways = tidal_window_results['tidal_accessibility']['Accessibility']  == 'Accessible'
        df_tidal_availability_waterways = df_tidal_availability_waterways.to_frame(name="Tide")

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

def add_ukc_policy_to_node(graph, node, rules):

    if "Depth_restriction" not in graph.nodes[node]:
        graph.nodes[node]["Depth_restriction"] = RuleEngine(default = lambda v: 0.0)
        graph.nodes[node]["Depth_restriction_at_distance"] = 0.

    for rule in rules:
        graph.nodes[node]["Depth_restriction"].add_rule(
            condition=rule.condition,
            policy=rule.policy,
            name=rule.name,
        )


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


def get_waterway(vessel, edge = None):
    waterway = None
    if edge:
        node_A = edge[0]
        node_info_A = vessel.env.graph.nodes[node_A]
        waterways_A = []
        if 'Waterway' not in node_info_A:
            waterways_A = node_info_A['Waterway']

        node_B = edge[1]
        node_info_B = vessel.env.graph.nodes[node_B]
        waterways_B = []
        if 'Waterway' not in node_info_B:
            waterways_B = node_info_B['Waterway']

        common_waterways = list(set(waterways_A) & set(waterways_B))
        if len(common_waterways):
            waterway = common_waterways[0]
    return waterway


def classify_tidal_access(tidal_windows):
    """Classify tidal accessibility based on the total number of windows.

    Rules:
        0 windows: Unable to enter the port.
        1–2 windows: Not tide-bound.
        More than 2 windows: Tide-bound.

    Args:
        tidal_windows: An iterable of lists of tidal windows,
            potentially from multiple rows for the same waterway.

    Returns:
        The tidal accessibility classification.
    """
    n_windows = sum(len(windows) for windows in tidal_windows)

    if n_windows == 0:
        return "unable_to_enter"
    if n_windows <= 2:
        return "not_tide_bound"
    return "tide_bound"


def check_berth_can_host_vessel(berth, vessel):
    #TODO: we should build in a better check
    berth_can_host_vessel = True
    if berth is not None:
        berth_can_host_vessel = berth.berth_length >= vessel.L
    return berth_can_host_vessel


def check_vessel_is_tide_bound(vessel, route = None, start_time = pd.Timestamp('NaT')):
    from opentnsim.port.calculations import calculate_minimum_waiting_time_for_tidal_window,  calculate_tidal_windows
    if route is None:
        route = vessel.route
    if pd.isna(start_time):
        start_time = datetime.datetime.fromtimestamp(vessel.env.now)
    time_start = np.datetime64(start_time - np.timedelta64(12,'h'))
    edge_route = node_path_to_edge_path(vessel.env.graph, route)
    sailing_time, _ = get_sailing_time(vessel, edge_route)
    sailing_time = max(pd.Timedelta(seconds=sailing_time), pd.Timedelta(hours=96))
    time_end = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now) + sailing_time)
    has_tidal_window_policy = check_if_route_contains_restrictions(vessel)
    tidal_window_results = pd.DataFrame(columns=["Accessibility"])
    vessel_can_reach_berth = True
    is_tide_bound = False
    tidal_waiting_time = pd.Timedelta(seconds = 0.)
    if not has_tidal_window_policy:
        return vessel_can_reach_berth, is_tide_bound, tidal_waiting_time, tidal_window_results
    
    tidal_window_results = calculate_tidal_windows(vessel, route, time_start, time_end)
    tidal_windows = (
        tidal_window_results
        .groupby("waterway")["tidal_windows"]
        .apply(classify_tidal_access)
    )

    vessel_can_reach_berth = ~tidal_windows.eq("unable_to_enter").any()
    is_tide_bound = tidal_windows.eq("tide_bound").any()
    tidal_waiting_time = pd.Timedelta(seconds = 0)
    if is_tide_bound:
        tidal_waiting_time = (
            tidal_window_results.groupby("waterway")["tidal_windows"]
            .apply(lambda s: calculate_minimum_waiting_time_for_tidal_window(
                vessel, sum(s, []), time_start=start_time
            ))
            .max()
        )

    return vessel_can_reach_berth, is_tide_bound, tidal_waiting_time, tidal_window_results


def sort_port_trips(port, terminals):
    """
    Sort port.vessel_trips using:
    1. Accepted rows first, preserving their original relative order.
    2. Planned rows sorted by terminal dependencies.
    3. Earliest port arrival time among eligible planned operations.

    Dependencies take precedence over arrival time for planned rows.
    Cycles are broken deterministically.

    Returns
    -------
    sorted_trips : pd.DataFrame
        The complete sorted trip planning.

    changed_trips : pd.DataFrame
        Planned rows from the first position where the overall row
        order differs from the original order, excluding accepted rows.
        Empty if unchanged.
    """

    arrival_col = "time_arrival_at_port"
    vessel_col = "vessel_id"
    operation_col = "operation_id"
    dependency_col = "operation_waiting_for"
    status_col = "status"
    original_col = "_original_position"
    normalized_status_col = "_normalized_status"

    trips = port.vessel_trips.copy()

    if trips.empty:
        return trips.copy(), trips.copy()

    # Track original row positions independently of index labels.
    trips[original_col] = range(len(trips))

    trips[arrival_col] = pd.to_datetime(
        trips[arrival_col], errors="coerce"
    )

    # Normalize status values.
    trips[normalized_status_col] = (
        trips[status_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Separate accepted, planned, and other rows.
    # Accepted rows retain their original relative order.
    accepted_trips = trips[
        trips[normalized_status_col] == "accepted"
    ].copy()

    planned_trips = trips[
        trips[normalized_status_col] == "planned"
    ].copy()

    other_trips = trips[
        ~trips[normalized_status_col].isin(
            ["accepted", "planned"]
        )
    ].copy()

    # Earliest arrival time for each vessel.
    arrival = trips.groupby(vessel_col)[arrival_col].min()

    # Collect berth planning from all terminals.
    planning_frames = []
    planned_vessels = set(
        planned_trips[vessel_col].dropna()
    )

    for _, terminal in terminals.items():
        berth_planning = terminal.berth_planning

        if berth_planning is not None and not berth_planning.empty:
            required_cols = [
                operation_col,
                vessel_col,
                dependency_col,
            ]

            if all(
                col in berth_planning.columns
                for col in required_cols
            ):
                planning = berth_planning.loc[
                    berth_planning[vessel_col].isin(
                        planned_vessels
                    ),
                    required_cols,
                ].copy()

                if not planning.empty:
                    planning_frames.append(planning)

    # Sort planned rows by arrival if no terminal planning exists.
    if not planning_frames:
        sorted_planned = planned_trips.sort_values(
            [arrival_col, original_col],
            kind="stable",
            na_position="last",
        ).copy()

    else:
        operations = pd.concat(
            planning_frames,
            ignore_index=True,
        ).drop_duplicates(
            subset=operation_col,
            keep="first",
        )

        ops = operations.set_index(
            operation_col
        ).to_dict("index")

        operation_ids = set(ops)

        predecessors = defaultdict(set)
        successors = defaultdict(set)

        # Build the dependency graph for planned operations.
        for operation_id, operation in ops.items():
            dependency = operation.get(dependency_col)

            if (
                pd.notna(dependency)
                and dependency in operation_ids
                and dependency != operation_id
            ):
                predecessors[operation_id].add(dependency)
                successors[dependency].add(operation_id)

        def priority(operation_id):
            vessel_id = ops[operation_id][vessel_col]
            time = arrival.get(vessel_id, pd.NaT)

            if pd.isna(time):
                time = pd.Timestamp.max

            return (
                time,
                str(operation_id),
                type(operation_id).__name__,
                repr(operation_id),
            )

        remaining = set(operation_ids)
        ready = []
        ordered_operations = []

        # Start with operations that have no unresolved dependencies.
        for operation_id in operation_ids:
            if not predecessors[operation_id]:
                heapq.heappush(
                    ready,
                    (priority(operation_id), operation_id),
                )

        while remaining:
            while ready and ready[0][1] not in remaining:
                heapq.heappop(ready)

            if ready:
                _, operation_id = heapq.heappop(ready)
            else:
                # Cycle: choose a deterministic fallback.
                operation_id = min(
                    remaining,
                    key=priority,
                )

            remaining.remove(operation_id)
            ordered_operations.append(operation_id)

            for next_operation in successors[operation_id]:
                predecessors[next_operation].discard(
                    operation_id
                )

                if (
                    next_operation in remaining
                    and not predecessors[next_operation]
                ):
                    heapq.heappush(
                        ready,
                        (
                            priority(next_operation),
                            next_operation,
                        ),
                    )

        # Convert operation order into vessel order.
        vessel_order = list(dict.fromkeys(
            ops[operation_id][vessel_col]
            for operation_id in ordered_operations
        ))

        rank = {
            vessel_id: index
            for index, vessel_id in enumerate(vessel_order)
        }

        planned_trips["_rank"] = (
            planned_trips[vessel_col]
            .map(rank)
            .fillna(len(rank))
        )

        sorted_planned = (
            planned_trips.sort_values(
                ["_rank", arrival_col, original_col],
                kind="stable",
                na_position="last",
            )
            .drop(columns=["_rank"])
        )

    # Accepted rows first, then other statuses, then planned rows.
    sorted_trips = pd.concat(
        [
            accepted_trips,
            other_trips,
            sorted_planned,
        ],
        ignore_index=True,
    )

    # Find the first position where the overall row order differs.
    original_positions = sorted_trips[
        original_col
    ].to_numpy()

    changed_indices = [
        i
        for i, (actual, expected) in enumerate(
            zip(
                original_positions,
                range(len(sorted_trips)),
            )
        )
        if actual != expected
    ]

    if changed_indices:
        first_changed = changed_indices[0]

        # Keep only planned rows from the first changed position onward.
        changed_trips = sorted_trips.iloc[
            first_changed:
        ].copy()

        changed_trips = changed_trips[
            changed_trips[normalized_status_col] == "planned"
        ].copy()

    else:
        changed_trips = sorted_trips.iloc[0:0].copy()

    # Remove temporary columns from both outputs.
    temporary_cols = [
        original_col,
        normalized_status_col,
    ]

    sorted_trips = sorted_trips.drop(
        columns=temporary_cols
    )

    changed_trips = (
        changed_trips
        .drop(columns=temporary_cols)
        .reset_index(drop=True)
    )

    return sorted_trips.reset_index(drop=True), changed_trips


def find_berth_based_on_name(terminal, berth_name):
    selected_berth = None
    for berth in terminal.berths.items:
        if berth.name == berth_name:
            selected_berth = berth
            break
    return selected_berth


def get_berth_availability_without_vessel(vessel, berth, berth_availability = None):
    from opentnsim.port.mixins.berth import IsQuay, IsJetty

    terminal = berth.terminal
    if berth_availability is None:
        terminal_berth_availability = berth.terminal.berth_availability.copy()
        berth_availability = terminal_berth_availability[berth.name]
    
    # corrct the berth availability for the vessel (if vessel was already assigned, then remove this vessel from the availability)
    if vessel.berth_operation_id in terminal.berth_planning.operation_id.values and vessel.berth == berth:
        index = terminal.berth_planning[terminal.berth_planning.operation_id == vessel.berth_operation_id].iloc[-1].name
        eta_berth = (terminal.berth_planning.loc[index, 'ETA_berth']).round('s')
        etd_berth = (terminal.berth_planning.loc[index, 'ETD_berth']).round('s')
        mask = (berth_availability.index >= eta_berth)&(berth_availability.index < etd_berth)
        if isinstance(berth, IsQuay):
            berth_availability.loc[mask] += vessel.L
        elif isinstance(berth, IsJetty):
            berth_availability.loc[mask] += 1

    return berth_availability


def get_berth_availability_info(vessel, origin, berth, delay = pd.Timedelta(seconds = 0), bool = True):
    from opentnsim.port.calculations import calculate_sailing_time_to_berth
    from opentnsim.port.mixins.berth import IsQuay

    # get availability
    berth_availability = berth.terminal.berth_availability[berth.name]

    # correct the berth availability for the vessel (if vessel was already assigned, then remove this vessel from the availability)
    berth_availability = get_berth_availability_without_vessel(vessel, berth, berth_availability)

    # calculate sailing time to berth
    sailing_time_to_berth = pd.Timedelta(seconds = 0)
    if origin != berth.node:
        sailing_time_to_berth = calculate_sailing_time_to_berth(vessel, origin, berth)
    berth_availability.index -= sailing_time_to_berth + delay

    # get future availability of fit berths
    if bool:
        if isinstance(berth, IsQuay):
            berth_availability = (berth_availability >= vessel.L)
        else:
            berth_availability = (berth_availability > 0)
    return berth_availability


def get_terminal_berths_availability(vessel, origin, terminal, delay = pd.Timedelta(seconds = 0), bool = True):
    berths = terminal.berths.items
    
    terminal_berth_availability = pd.DataFrame()
    for berth in berths:
        berth_availability = get_berth_availability_info(vessel, origin, berth, delay, bool)
        terminal_berth_availability[berth.name] = berth_availability
    
    terminal_berth_availability = (terminal_berth_availability
        .sort_index()
        .ffill()
        .bfill()
    )
    return terminal_berth_availability


def get_terminal_availability_info(vessel, origin, terminal, berth = None, delay = pd.Timedelta(seconds = 0)):
    terminal_availability = pd.DataFrame()
    if berth is not None:
        terminal_availability = terminal.provide_terminal_availability_info(vessel, origin, berth, delay)
    return terminal_availability