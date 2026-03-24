import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import pyproj
import shapely
import re
from shapely.ops import transform
from pyproj import Transformer
from opentnsim.graph.utils import get_sailing_time, get_trajectory
from opentnsim.graph.calculations import transform_geometry
from opentnsim.port.calculations import calculate_tidal_windows
from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from IPython.display import display
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
    for index, route_sailed in enumerate(vessel.routes_sailed):
        selected_route = [node for node in route_sailed if node in route]
        route_geometry = get_trajectory(vessel.env.graph,route[0],route[-1])
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
        maximum_sailed_distance = df_route.Value.max()
        first_index = last_index + 1
        corrected_df = pd.concat([corrected_df, df_route])
    corrected_df = corrected_df.ffill()
    return corrected_df


def update_terminal_planning(vessel, delay=0.):
    vessel.terminal.replan_vessels_terminal_berths(vessel,delay)
    for queued_vessel_id,vessel_info in vessel.terminal.queue.iterrows():
        queued_vessel = get_vessel_from_id(vessel.env, [queued_vessel_id])[0]
        if hasattr(queued_vessel,'waiting_event') and queued_vessel.waiting_event.is_alive:
            queued_vessel.waiting_event.interrupt()


def check_if_vessel_is_tide_bound(df_tidal_availability):
    tide_bound = False
    if False in df_tidal_availability.values:
        tide_bound = True
    return tide_bound


def determine_nearest_anchorage_area(vessel, node):
    # Loop over the nodes of the network and identify all the anchorage areas:
    sailing_time_to_anchorages = []
    capacity_of_anchorages = []
    for anchorage_area in vessel.terminal.port.anchorage_areas:
        # Determine if the anchorage area can be reached
        route_to_anchorage = nx.dijkstra_path(vessel.env.graph, node, anchorage_area.node)
        sailing_time_to_anchorage, _ = get_sailing_time(vessel, route_to_anchorage)
        sailing_time_to_anchorages.append(sailing_time_to_anchorage)
        capacity_of_anchorages.append(anchorage_area.resource.capacity > 0)

    anchorage_selection_df = pd.DataFrame({'Sailing time': sailing_time_to_anchorages,
                                           'Capacity': capacity_of_anchorages})

    suitable_anchorage_areas = anchorage_selection_df[anchorage_selection_df.Capacity]
    suitable_anchorage_areas = suitable_anchorage_areas.sort_values('Sailing time')
    anchorage_area = vessel.terminal.port.anchorage_areas[suitable_anchorage_areas.iloc[0].name]
    return anchorage_area


def determine_if_vessel_needs_to_sail_to_the_anchorage_area(env, vessel, origin, waiting_time):
    sail_to_anchorage_area = False
    nearest_anchorage_area = determine_nearest_anchorage_area(vessel, origin)
    route_to_anchorage_area = nx.dijkstra_path(env.graph, origin, nearest_anchorage_area.node)
    route_after_anchorage_area = nx.dijkstra_path(env.graph, nearest_anchorage_area.node, vessel.route[-1])
    sailing_time_to_terminal, _ = get_sailing_time(vessel, vessel.route)
    sailing_time_to_anchorage_area, _ = get_sailing_time(vessel, route_to_anchorage_area)
    new_sailing_time_to_terminal, _ = get_sailing_time(vessel, route_after_anchorage_area)
    delay_of_sailing_to_terminal = new_sailing_time_to_terminal - sailing_time_to_terminal + sailing_time_to_anchorage_area
    if delay_of_sailing_to_terminal <= waiting_time:
        sail_to_anchorage_area = True
    return sail_to_anchorage_area


def determine_new_route_for_vessel(vessel):
    new_route = None
    origin = vessel.route[-1]
    if vessel.next_destination is not None:
        destination = vessel.next_destination
        new_route = nx.dijkstra_path(vessel.env.graph,origin,destination)
    elif len(vessel.next_terminals):
        next_terminal = vessel.next_terminals[-1]
        vessel.next_terminals = vessel.next_terminals[1:]
        berth = vessel.request_terminal_access(vessel, origin)
    return new_route


def determine_vessel_waiting_events(port_accessed, vessel, port_availability_df):
    port_availability_df['Combined'] = port_availability_df.all(axis=1)
    with pd.option_context("future.no_silent_downcasting", True):
        port_availability_df = port_availability_df.ffill()
        port_availability_df = port_availability_df.bfill()
    port_availability_df = port_availability_df[port_availability_df != port_availability_df.shift()].dropna(how='all')
    with pd.option_context("future.no_silent_downcasting", True):
        port_availability_df = port_availability_df.ffill()

    def get_waiting_time_reason(lst):
        if not lst:
            return ""
        elif len(lst) == 1:
            return lst[0]
        else:
            return ", ".join(lst[:-1]) + " and " + lst[-1]

    current_time = datetime.datetime.fromtimestamp(vessel.env.now)
    previous_events = port_availability_df[port_availability_df.index <= current_time]
    future_events = port_availability_df[port_availability_df.index > current_time]
    last_previous_event_index = previous_events.index.max()
    if pd.isna(last_previous_event_index):
        previous_event = port_availability_df.iloc[0:0]
    else:
        previous_event = port_availability_df.loc[[last_previous_event_index]]
    port_availability_df = pd.concat([previous_event, future_events])
    port_availability_df.index.values[0] = current_time

    cols_to_check = port_availability_df.columns.drop('Combined')
    port_availability_df['Reason'] = port_availability_df[cols_to_check].apply(
        lambda row: get_waiting_time_reason(list(row[row.eq(False)].index)),
        axis=1)

    port_available_df = port_availability_df[port_availability_df['Combined'] == True]
    if port_available_df.empty:
        return None
    waiting_time_end = port_available_df.iloc[0].name
    waiting_events = port_availability_df.loc[:waiting_time_end]
    waiting_reasons = waiting_events['Reason'][:-1]
    waiting_times = (waiting_events.index.to_series().shift(-1) - waiting_events.index).apply(lambda x: x.total_seconds())
    waiting_events = {}
    for index, (waiting_reason, waiting_time) in enumerate(zip(waiting_reasons,waiting_times)):
        index += 1
        waiting_events[waiting_reason + f' ({index})'] = waiting_time
    vessel.port_accessed = port_accessed
    return waiting_events


def determine_vessel_priority(vessel, tide_bound = False, leaving_port = False):
    priority = 0
    if tide_bound:
        priority += 1
    if leaving_port:
        priority += 1
    vessel.priority = priority
    return priority


def get_accessibility_info(vessel, origin, berth = None, leaving_port = False):
    df_tidal_availability = get_tidal_availability_info(vessel)
    tide_bound = check_if_vessel_is_tide_bound(df_tidal_availability)
    priority = determine_vessel_priority(vessel, tide_bound, leaving_port)
    df_terminal_availability = get_terminal_availability_info(vessel, origin, berth, leaving_port)
    df_waterways_availability = get_waterway_availability_info(vessel, origin, priority)

    #Combine the dataframes
    port_availability_df = pd.concat([df_tidal_availability,df_terminal_availability,df_waterways_availability],axis=1)
    port_availability_df = port_availability_df.sort_index()
    with pd.option_context("future.no_silent_downcasting", True):
        port_availability_df = port_availability_df.ffill()
    port_availability_df = port_availability_df[port_availability_df != port_availability_df.shift()].dropna(how='all')
    with pd.option_context("future.no_silent_downcasting", True):
        port_availability_df = port_availability_df.ffill()
    with pd.option_context("future.no_silent_downcasting", True):
        port_availability_df = port_availability_df.bfill()
    return port_availability_df, priority


def find_waterways_to_be_passed(vessel):
    passing_waterways = {}
    for node in vessel.route:
        waterway = None
        if "Waterway" in vessel.env.graph.nodes[node]:
            waterway = vessel.env.graph.nodes[node]["Waterway"]

        if waterway and waterway.name not in passing_waterways.keys():
            passing_waterways[waterway.name] = waterway
    return passing_waterways


def get_waterway_availability_info(vessel, origin, priority):
    passing_waterways = find_waterways_to_be_passed(vessel)
    df_waterways_availability = pd.DataFrame()
    for waterway in passing_waterways.values():
        availability_df = waterway.get_waterway_availability_info(vessel, origin, priority)
        df_waterways_availability = pd.concat([df_waterways_availability,availability_df])
    if df_waterways_availability.empty:
        df_waterways_availability.loc[vessel.env.simulation_start,'Traffic'] = True
    return df_waterways_availability


def get_terminal_availability_info(vessel, origin, berth = None, leaving_port = False):
    df_terminal_availability = pd.DataFrame()
    if not leaving_port:
        df_terminal_availability = vessel.terminal.provide_terminal_availability_info(vessel, origin, berth)
    return df_terminal_availability


def check_if_route_contains_restrictions(vessel):
    contains_restriction = False
    for node in vessel.route:
        if 'Vertical tidal restriction' in vessel.env.graph.nodes[node].keys():
            contains_restriction = True
            break
    return contains_restriction


def get_tidal_availability_info(vessel):
    has_tidal_window_policy = check_if_route_contains_restrictions(vessel)
    route = vessel.route
    time_start = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now))
    sailing_time, _ = get_sailing_time(vessel, route)
    sailing_time = np.max([pd.Timedelta(seconds=sailing_time), pd.Timedelta(hours=48)])
    time_end = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now) + sailing_time)
    df_tidal_availability = pd.DataFrame(columns=['Accessibility'])
    if vessel.trip_index in vessel.tidal_window_calculations.keys() and len(vessel.tidal_window_calculations[vessel.trip_index]):
        df_tidal_availability = vessel.tidal_window_calculations[vessel.trip_index]['tidal_accessibility']
    elif has_tidal_window_policy:
        tidal_window_results = calculate_tidal_windows(vessel, route, time_start, time_end)
        df_tidal_availability = tidal_window_results['tidal_accessibility']
        vessel.tidal_window_calculations[vessel.trip_index] = tidal_window_results
    df_tidal_availability['Tide'] = df_tidal_availability['Accessibility'] == 'Accessible'
    return df_tidal_availability[['Tide']]


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
                _, _, _, required_water_depth, _, MBL = provide_ukc_clearance(vessel, node)
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
            sailing_time_to_anchorage = get_sailing_time(vessel, route_from_anchorage)[
                "Time"
            ].sum()
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





