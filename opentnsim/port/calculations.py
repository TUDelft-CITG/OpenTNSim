import bisect
import math
import matplotlib.dates as mdates
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
from pyproj.transformer import Transformer
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import find_peaks
from shapely.ops import linemerge, transform
from shapely.geometry import LineString, MultiLineString
import xarray as xr

from opentnsim.graph.utils import get_sailing_time, get_longest_common_subroute, node_path_to_edge_path
from opentnsim.graph.calculations import transform_geometry, transform_route_geometry
from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from opentnsim.environment.utils import get_water_depth, get_governing_current_velocity


def calculate_total_waiting_time(waiting_events):
    total_waiting_time = 0.
    if waiting_events is not None and len(waiting_events):
        total_waiting_time = sum(waiting_events.values())
    return total_waiting_time


def calculate_inerpolated_water_levels_over_network(graph, hydrodynamic_data, method = 'nearest'):
    geod = pyproj.Geod(ellps="WGS84")
    distances_to_measurement_stations = pd.DataFrame(columns=(hydrodynamic_data.STATION.values))
    stations = hydrodynamic_data.STATION.values
    for node in graph.nodes:
        node_info = graph.nodes[node]
        geometry = node_info['geometry']
        longitude, latitude = geometry.coords.xy
        distances_to_station_m = []
        for station_meas, (_, geometry_meas) in zip(stations, hydrodynamic_data.attrs.items()):
            longitude_meas, latitude_meas = geometry_meas.coords.xy
            _, _, distance_to_station_m = geod.inv(longitude, latitude, longitude_meas, latitude_meas)
            distances_to_station_m.append(distance_to_station_m[0])
        distances_to_measurement_stations.loc[node, :] = distances_to_station_m

    distances_to_measurement_stations['distance_min'] = distances_to_measurement_stations.min(axis=1)
    interpolatable_nodes = distances_to_measurement_stations[distances_to_measurement_stations['distance_min'] < 50000]

    if method == 'nearest':
        node_arrays = {}
        nearest_df = interpolatable_nodes.drop(columns="distance_min").idxmin(axis=1)
        nearest_station = nearest_df.to_dict()
        for node, station in nearest_station.items():
            node_arrays[node] = hydrodynamic_data['Water level'].sel({'STATION': station})
        hydrodynamic_data_nodes = xr.concat(list(node_arrays.values()), dim='STATION')
        hydrodynamic_data_nodes = hydrodynamic_data_nodes.transpose('TIME', 'STATION')


    elif method == 'weighted':
        dist = interpolatable_nodes[stations]
        power = 2

        weights = 1 / dist ** power
        weights_df = weights.div(weights.sum(axis=1), axis=0)

        weights_xr = xr.DataArray(
            weights_df.values,
            dims=("Node", "STATION"),
            coords={"Node": weights.index, "STATION": weights.columns}
        )

        hydrodynamic_data_nodes = xr.dot(hydrodynamic_data['Water level'], weights_xr, dims="STATION")
        hydrodynamic_data_nodes = hydrodynamic_data_nodes.rename({'Node': 'STATION'})
    hydrodynamic_data = xr.Dataset()
    hydrodynamic_data['Water level'] = hydrodynamic_data_nodes
    return hydrodynamic_data

def calculate_depth_values_over_route(env, node_start, node_stop, offset = 500):
    hydromanager = HydrodynamicDataManager()
    hydrodynamic_data = hydromanager.hydrodynamic_data
    water_depth = hydrodynamic_data['Water level'] + hydrodynamic_data['Nautical depth']
    route = nx.dijkstra_path(env.graph, node_start, node_stop)
    transformed_geometry = transform_route_geometry(env, node_start, node_stop)
    node_distances = {}
    node_water_depths = {}
    node_times = {}
    for index,node in enumerate(route):
        offset_applied = 0.
        if not index:
            offset_applied = offset
        elif index == len(route)-1:
            offset_applied = - offset
        transformed_node = transform_geometry(env.graph.nodes[node]['geometry'])
        distance_to_node = transformed_geometry.project(transformed_node)
        node_water_depths[node] = water_depth.sel({'STATION': node}).values
        node_distances[node] = np.ones(len(node_water_depths[node])) * distance_to_node + offset_applied + 0.001
        node_times[node] = water_depth.TIME.values

        infrastructure = None
        if 'Anchorage' in env.graph.nodes[node].keys():
            infrastructure = env.graph.nodes[node]['Anchorage']
        elif 'Berth' in env.graph.nodes[node].keys():
            infrastructure = env.graph.nodes[node]['Berth'][0]

        if infrastructure is None:
            continue

        if not index:
            boundary_offsets = np.array([-offset, offset]) - 0.001
        else:
            boundary_offsets = np.array([-offset, offset]) + 0.001

        for boundary,boundary_offset in enumerate(boundary_offsets):
            if isinstance(node, str):
                boundary = str(boundary)
            node_water_depths[node + boundary] = hydrodynamic_data['Water level'].sel({'STATION': node}).values + infrastructure.depth
            node_distances[node + boundary] = np.ones(len(node_water_depths[node + boundary])) * distance_to_node + boundary_offset
            node_times[node + boundary] = water_depth.TIME.values

    return node_distances, node_times, node_water_depths


def calculate_interpolated_depth_values(env, node_start, node_stop, ddistance=1000, offset=500):
    node_distances, node_times, node_water_depths = calculate_depth_values_over_route(env, node_start, node_stop, offset)
    node_distances = np.concatenate(list(node_distances.values()))
    node_times = np.concatenate(list(node_times.values()))
    node_water_depths = np.concatenate(list(node_water_depths.values()))

    node_times, time_idx = np.unique(node_times, return_inverse=True)
    node_times_num = mdates.date2num(node_times)

    interpolated_distance = np.linspace(node_distances.min(), node_distances.max(), ddistance)  # horizontal resolution
    interpolated_depth = np.full((len(node_times), len(interpolated_distance)), np.nan)

    for i, y_val in enumerate(node_times):
        mask = time_idx == i
        node_distances_idx = node_distances[mask]
        node_water_depths_idx = node_water_depths[mask]

        if len(node_distances_idx) < 2:
            continue

        idx = np.argsort(node_distances_idx)
        f = interp1d(node_distances_idx[idx], node_water_depths_idx[idx], kind='linear',
                     bounds_error=False, fill_value=np.nan)

        interpolated_depth[i, :] = f(interpolated_distance)

    return interpolated_distance, node_times_num, interpolated_depth


def calculate_vertical_tidal_windows(vessel, route, time_start, time_end, delay=0):
    """Function: calculates the windows available to sail-in and -out of the port given the
      vertical tidal restrictions according to the tidal window policy.

    Input:
        - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
        - route: a list of strings of node names that resemble the route that the vessel is planning
          to sail (can be different than vessel.route)
        - sailing_time_correction: a bool that indicates whether the calculation should correct for
          sailing_speed (dynamic calculation) or not (static calculation)

    """
    vertical_tidal_accessibility = pd.DataFrame(columns=["Limit", "Accessibility"])
    net_ukcs = calculate_minimum_available_water_depth_along_route(vessel, route, time_start, time_end, delay)
    time_net_ukc = net_ukcs.index.to_numpy()
    net_ukc = net_ukcs["min_net_ukc"].to_numpy()

    # Determine zero crossings
    zero_crossings = np.nonzero(np.diff(np.sign(net_ukc)))[0]
    for root in [time_net_ukc[i] for i in zero_crossings]:
        # -if the net ukc a moment later than the root is higher than the required water depth:
        if net_ukc[bisect.bisect_right(time_net_ukc, root)] > 0:
            vertical_tidal_accessibility.loc[root, :] = [0, "Accessible"]
        elif net_ukc[bisect.bisect_right(time_net_ukc, root)] < 0:
            vertical_tidal_accessibility.loc[root, :] = [0, "Inaccessible"]

    # Default values
    if vertical_tidal_accessibility.empty:
        if not len(net_ukc) or np.max(net_ukc) < 0.0:
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [0, "Inaccessible"]
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [0,"Inaccessible",]
        else:
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [0, "Accessible"]
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [0, "Accessible"]
    else:
        if vertical_tidal_accessibility.iloc[0].Accessibility == "Inaccessible":
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [0,"Accessible",]
        else:
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [0,"Inaccessible",]
        if vertical_tidal_accessibility.iloc[-1].Accessibility == "Inaccessible":
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [0,"Accessible",]
        else:
                vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [0,"Inaccessible",]

    # Return the sail-in or -out-times given the vertical tidal restrictions over the route of the vessel
    vertical_tidal_accessibility = vertical_tidal_accessibility.sort_index()
    vertical_tidal_accessibility["Condition"] = "Water level"
    vertical_tidal_windows = [[window_start[0], window_end[0]] for window_start, window_end in zip(vertical_tidal_accessibility.iloc[:-1].iterrows(), vertical_tidal_accessibility.iloc[1:].iterrows()) if window_start[1].Accessibility == "Accessible"]
    return vertical_tidal_accessibility, vertical_tidal_windows, net_ukcs


# Functions used to calculate the sail-in-times for a specific vessel
def determine_tidal_window_restriction(vessel, route, specifications, node, delay=0):
    """Function: determines which tidal window restriction applies to the vessel at the specific node

    Input:
        - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
        - route: a list of strings of node names that resemble the route that the vessel is planning
        to sail (can be different than vessel.route)
        - specifications: the specific data regarding the properties for which the restriction holds
        - node: a string that defines the node of the tidal window restriction

    """

    # Predefined bool
    boolean = True
    no_tidal_window = True
    # Determining if and which restriction applies for the vessel by looping over the restriction class
    for restriction_class in enumerate(specifications[0]):
        # - if restriction does not apply to vessel because it is for vessels sailing in the opposite
        # direction: continue loop
        if vessel.bound != specifications[2][restriction_class[0]]:
            continue
        if specifications[5][restriction_class[0]] != [] and (
            (specifications[5][restriction_class[0]][0] not in route)
            or (specifications[5][restriction_class[0]][1] not in route)
        ):
            continue
        # - else: looping over the restriction criteria
        for restriction_type in enumerate(restriction_class[1]):
            # - if previous condition is not met and there are no more restriction criteria and the previous
            #  condition has an 'AND' boolean statement: continue loop

            if (
                not boolean
                and restriction_type[0] == len(restriction_class[1]) - 1
                and specifications[4][restriction_class[0]][restriction_type[0] - 1] == "and"
            ):
                continue
            # - if previous condition is not met and there are more restriction criteria and the next condition
            # has an 'AND' boolean statement: continue loop
            if (
                not boolean
                and restriction_type[0] != len(restriction_class[1]) - 1
                and specifications[4][restriction_class[0]][restriction_type[0]] == "and"
            ):
                continue
            # - if previous condition is not met and the next condition has an 'OR' boolean statement:
            # continue loop with predefined boolean
            if (
                not boolean
                and restriction_type[0] != len(restriction_class[1]) - 1
                and specifications[4][restriction_class[0]][restriction_type[0]] == "or"
            ):
                boolean = True
                continue

            # Extracting the correct vessel property for the restriction type
            if restriction_type[1].find("Length") != -1:
                value = getattr(vessel, "L")
            if restriction_type[1].find("Draught") != -1:
                value = getattr(vessel, "T")
            if restriction_type[1].find("Beam") != -1:
                value = getattr(vessel, "B")
            if restriction_type[1].find("UKC") != -1:
                value, _, _, _, _, _ = calculate_ukc_clearance(vessel, node, delay)
            if restriction_type[1].find("Type") != -1:
                value = getattr(vessel, "type")
            # Determine if the value for the property satisfies the condition of the restriction type
            df = pd.DataFrame({"Value": [value], "Restriction": [specifications[1][restriction_class[0]][restriction_type[0]]]})
            boolean = df.eval("Value" + specifications[3][restriction_class[0]][restriction_type[0]] + "Restriction")[0]

            # - if condition is not met: continue loop
            if not boolean and restriction_type[0] != len(restriction_class[1]) - 1:
                continue

            # - if one of the conditions is met and the restriction contains an 'OR' boolean statement:
            if (
                boolean
                and restriction_type[0] != len(restriction_class[1]) - 1
                and specifications[-1][restriction_class[0]][restriction_type[0]] == "or"
            ):
                no_tidal_window = False
                break
            # - elif all the conditions are met and the restriction contains solely 'AND' boolean statements:
            elif boolean and restriction_type[0] == len(restriction_class[1]) - 1:
                no_tidal_window = False
                break

        # - if condition is met: break the loop
        if boolean is True:
            break

        # - else: restart the loop with predefined bool
        else:
            boolean = True

    return restriction_class[0], no_tidal_window


def calculate_horizontal_tidal_window(vessel,
                                      time_start_index,
                                      time_end_index,
                                      critical_limits=[],
                                      cross_current_limit_dataframe=pd.DataFrame(),
                                      flood=True,
                                      ebb=True,
                                      decreasing=False):
    hydromanager = HydrodynamicDataManager()
    hydrodynamic_data = hydromanager.hydrodynamic_data
    station = hydrodynamic_data.STATION.values
    delta_time = (hydrodynamic_data.TIME.values[1] - hydrodynamic_data.TIME.values[0])
    time_start_index = np.max([0,time_start_index - int(np.timedelta64(12, "h") / delta_time)])
    currents_time = hydrodynamic_data.TIME.values[time_start_index:time_end_index]
    currents_data, _ = get_governing_current_velocity(vessel, station, time_start_index, time_end_index)
    index_prev_root = 0
    roots = CubicSpline(currents_time, currents_data).roots()
    roots = [root for root in roots if root >= currents_time[0].astype(float) and root <= currents_time[-1].astype(float)]
    times_horizontal_tidal_period = []
    for root in roots:
        root = pd.Timestamp(root).to_datetime64()
        index_current_root = bisect.bisect_right(currents_time, root) - 2
        if index_current_root == -1:
            index_current_root = index_current_root + 1
        if len(currents_data[index_prev_root:index_current_root]) == 0:
            continue
        cvel_diff_cross = currents_data[index_current_root + 1] - currents_data[index_current_root - 1]
        if cvel_diff_cross < 0:
            times_horizontal_tidal_period.append([root, "Ebb Start"])
            index_prev_root = index_current_root
        elif cvel_diff_cross > 0:
            times_horizontal_tidal_period.append([root, "Flood Start"])
            index_prev_root = index_current_root

    tidal_periods = [condition for condition in times_horizontal_tidal_period if condition[0] <= currents_time[-1]]
    currents_time = np.append(currents_time, np.array([tide[0] for tide in tidal_periods if tide[0] not in currents_time], dtype="datetime64[ns]"))
    currents_data = [abs(value) for value in currents_data]
    currents_data = np.append(currents_data, -999 * np.ones(len(tidal_periods)))
    currents_time, currents_data = [np.array(data) for data in zip(*sorted(zip(currents_time, currents_data)))]
    # Find the intersection points with critical current velocity
    current_intersections = []
    if isinstance(cross_current_limit_dataframe, pd.DataFrame) and not cross_current_limit_dataframe.empty:
        critical_limit = np.interp(
            currents_time.astype("float"),
            cross_current_limit_dataframe.index.to_numpy().astype("float"),
            cross_current_limit_dataframe.Limit.to_numpy(),
        )
        idx = np.argwhere(np.diff(np.sign(critical_limit - currents_data))).flatten()
        roots = currents_time[idx]
        current_intersections.extend(roots.astype(dtype="datetime64[ns]"))
        critical_current_velocity = np.interp(
            np.array(current_intersections).astype("float"), currents_time.astype("float"), critical_limit
        )
        horizontal_tidal_accessibility = pd.DataFrame(
            data=critical_current_velocity, columns=["Limit"], index=current_intersections
        )
    elif isinstance(critical_limits, list):
        critical_current_velocity = []
        for critical_limit in critical_limits:
            idx = np.argwhere(
                np.diff(np.sign([current_velocity - critical_limit for current_velocity in currents_data]))
            ).flatten()
            roots = currents_time[idx]
            current_intersections.extend(roots.astype(dtype="datetime64[ns]"))
            critical_current_velocity.extend(np.ones(len(roots)) * critical_limit)
        horizontal_tidal_accessibility = pd.DataFrame(
            data=critical_current_velocity, columns=["Limit"], index=current_intersections
        )
    horizontal_tidal_accessibility = horizontal_tidal_accessibility.sort_index()

    # Determine the tidal period of the found interpolation points
    horizontal_tidal_accessibility["Period"] = ""
    horizontal_tidal_accessibility["Period_nr"] = -999
    for period_nr, (tidal_period_start, tidal_period_end) in enumerate(zip(tidal_periods[:-1], tidal_periods[1:])):
        tidal_period = tidal_period_start[1].split(" ")[0]
        if tidal_period == "Rising":
            tidal_period = "Flood"
        if tidal_period == "Falling":
            tidal_period = "Ebb"
        tidal_period_start = tidal_period_start[0]
        tidal_period_end = tidal_period_end[0]
        horizontal_tidal_accessibility.loc[
            horizontal_tidal_accessibility[
                (horizontal_tidal_accessibility.index >= tidal_period_start)
                & (horizontal_tidal_accessibility.index <= tidal_period_end)
            ].index,
            "Period",
        ] = tidal_period
        horizontal_tidal_accessibility.loc[
            horizontal_tidal_accessibility[
                (horizontal_tidal_accessibility.index >= tidal_period_start)
                & (horizontal_tidal_accessibility.index <= tidal_period_end)
            ].index,
            "Period_nr",
        ] = period_nr
    horizontal_tidal_accessibility["Period_nr"] = horizontal_tidal_accessibility["Period_nr"].astype(int)

    # Filter the found interpolation points: remove from errors
    # (multiple interpolated numbers and flood/ebb values if not required)
    if decreasing:
        selected_horizontal_tidal_accessibility = pd.DataFrame(columns=horizontal_tidal_accessibility.columns)
        if flood:
            selected_horizontal_tidal_accessibility = pd.concat(
                [
                    selected_horizontal_tidal_accessibility,
                    horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period == "Flood"],
                ]
            )
        if ebb:
            selected_horizontal_tidal_accessibility = pd.concat(
                [
                    selected_horizontal_tidal_accessibility,
                    horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period == "Ebb"],
                ]
            )
        horizontal_tidal_accessibility = selected_horizontal_tidal_accessibility.sort_index()
        horizontal_tidal_accessibility = horizontal_tidal_accessibility.loc[
            horizontal_tidal_accessibility.index.drop_duplicates(keep=False)
        ]

    # Correct the found interpolation points
    if decreasing:
        tide_number = horizontal_tidal_accessibility.iloc[0]["Period_nr"]
        number_of_tidal_periods = horizontal_tidal_accessibility.iloc[-1]["Period_nr"]
        end_time_windows = []
        for period_nr in [
            idx for idx, count in horizontal_tidal_accessibility.value_counts("Period_nr").items() if count == 2
        ]:
            sub_df = horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period_nr == period_nr]
            end_time_windows.append(tidal_periods[period_nr + 1][0] - sub_df.iloc[-1].name)
        mean_end_time_window = np.mean(end_time_windows)
        missing_tides = set(list(np.arange(tide_number, number_of_tidal_periods, 2))) - set(
            list(dict.fromkeys(horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period == "Flood"].Period_nr))
        )
        for tide_index in missing_tides:
            starting_time = np.datetime64(tidal_periods[tide_index][0])
            closing_time = np.datetime64(tidal_periods[tide_index + 1][0] - mean_end_time_window)
            next_index = bisect.bisect_right(currents_time, closing_time)
            previous_index = next_index - 1
            current_velocity = np.interp(
                closing_time,
                [currents_time[previous_index], currents_time[next_index]],
                [currents_data[previous_index], currents_data[next_index]],
            )
            horizontal_tidal_accessibility.loc[starting_time, :] = [0, "Flood", tide_index]
            horizontal_tidal_accessibility.loc[closing_time, :] = [current_velocity, "Flood", tide_index]

        for period_nr, count in [
            (idx, count) for idx, count in horizontal_tidal_accessibility.value_counts("Period_nr").items() if count != 2
        ]:
            if count == 1:
                for iloc, (loc, info) in enumerate(
                    horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period_nr == period_nr].iterrows()
                ):
                    if not iloc % 2 and info.Limit != np.max(critical_limits):
                        starting_time = np.datetime64(tidal_periods[info.Period_nr][0])
                        horizontal_tidal_accessibility.loc[starting_time] = info
                        horizontal_tidal_accessibility.loc[starting_time, "Limit"] = np.max(critical_limits)
                        break
            else:
                horizontal_tidal_accessibility = horizontal_tidal_accessibility.drop(
                    horizontal_tidal_accessibility[
                        (horizontal_tidal_accessibility.Period_nr == period_nr)
                        & (horizontal_tidal_accessibility.Limit == np.max(critical_limits))
                    ].index[:-1]
                )
                horizontal_tidal_accessibility = horizontal_tidal_accessibility.drop(
                    horizontal_tidal_accessibility[
                        (horizontal_tidal_accessibility.Period_nr == period_nr)
                        & (horizontal_tidal_accessibility.Limit == np.min(critical_limits))
                    ].index[:-1]
                )
                if len(horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period_nr == period_nr]) < 2:
                    for iloc, (loc, info) in enumerate(
                        horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period_nr == period_nr].iterrows()
                    ):
                        if not iloc % 2 and info.Limit != np.max(critical_limits):
                            starting_time = np.datetime64(tidal_periods[info.Period_nr][0])
                            horizontal_tidal_accessibility.loc[starting_time] = info
                            horizontal_tidal_accessibility.loc[starting_time, "Limit"] = np.max(critical_limits)
                            break

    else:
        for period_nr in [
            idx for idx, count in horizontal_tidal_accessibility.value_counts("Period_nr").items() if count % 2
        ]:
            for loc, info in horizontal_tidal_accessibility[
                horizontal_tidal_accessibility.Period_nr == period_nr
            ].iterrows():
                if (
                    currents_data[list(currents_time).index(loc) - 1] < info.Limit
                    and currents_data[list(currents_time).index(loc) + 1] < info.Limit
                ):
                    horizontal_tidal_accessibility[loc + np.timedelta64(1, "ns")] = info
                    break

    horizontal_tidal_accessibility = horizontal_tidal_accessibility.sort_index()
    horizontal_tidal_accessibility = horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period_nr != -999]
    # Add accessibility information to intersection points
    if not horizontal_tidal_accessibility.empty:
        if not decreasing:
            horizontal_tidal_accessibility.loc[list(horizontal_tidal_accessibility.index)[1::2], "Accessibility"] = (
                "Accessible"
            )
            horizontal_tidal_accessibility.loc[list(horizontal_tidal_accessibility.index)[::2], "Accessibility"] = (
                "Inaccessible"
            )
            limiting_currents = np.interp(
                horizontal_tidal_accessibility.index.to_numpy().astype(float),
                cross_current_limit_dataframe.index.to_numpy().astype(float),
                cross_current_limit_dataframe.Limit.to_numpy(),
            )
            horizontal_tidal_accessibility["Accessibility"] = [
                "Accessible" if interpcur < limit else accessibility
                for accessibility, interpcur, limit in zip(
                    horizontal_tidal_accessibility.Accessibility.to_numpy(),
                    horizontal_tidal_accessibility.Limit.to_numpy(),
                    limiting_currents,
                )
            ]
            horizontal_tidal_accessibility["Limit"] = [
                0 if interpcur < limit else limit
                for interpcur, limit in zip(horizontal_tidal_accessibility.Limit.to_numpy(), limiting_currents)
            ]
            horizontal_tidal_accessibility = horizontal_tidal_accessibility[horizontal_tidal_accessibility.Limit != -1.0]
        else:
            horizontal_tidal_accessibility.loc[list(horizontal_tidal_accessibility.index)[1::2], "Accessibility"] = (
                "Inaccessible"
            )
            horizontal_tidal_accessibility.loc[list(horizontal_tidal_accessibility.index)[::2], "Accessibility"] = (
                "Accessible"
            )
        horizontal_tidal_accessibility["Condition"] = "Current velocity"
        horizontal_tidal_accessibility = horizontal_tidal_accessibility[["Limit", "Condition", "Accessibility"]]

    return horizontal_tidal_accessibility, station

def calculate_horizontal_tidal_windows(vessel, route, time_start, time_end, delay=0):
    # Start calculation
    horizontal_tidal_restriction_nodes = []
    horizontal_tidal_restriction_stations = []
    window_specifications = []
    horizontal_tidal_accessibility = pd.DataFrame(columns=["Limit", "Condition", "Accessibility"])
    horizontal_tidal_window = False
    for route_index, node_name in enumerate(route):
        if "Horizontal tidal restriction" in vessel.multidigraph.nodes[node_name].keys():
            horizontal_tidal_window = True
            edge_route = node_path_to_edge_path(vessel.env.graph, route[: (route_index + 1)])
            sailing_time_to_next_node, _ = get_sailing_time(vessel, edge_route)
            specifications = vessel.multidigraph.nodes[node_name]["Horizontal tidal restriction"]["Specification"]
            restriction_index, no_tidal_window = determine_tidal_window_restriction(
                vessel, route, specifications, node_name, delay=delay
            )
            if no_tidal_window:
                continue
            hydrodynamic_data = vessel.multidigraph.nodes[node_name]["Horizontal tidal restriction"]["Data"][restriction_index]
            cross_current_limit = vessel.multidigraph.nodes[node_name]["Horizontal tidal restriction"]["Limit"][restriction_index]
            window_specifications = vessel.multidigraph.nodes[node_name]["Horizontal tidal restriction"]["Type"][restriction_index]
            time_start_index = np.max([0, np.absolute(hydrodynamic_data.TIME.values - (time_start + np.timedelta64(int(delay), "s"))).argmin() - 2,])
            time_end_index = np.absolute(hydrodynamic_data.TIME.values - (time_end + np.timedelta64(int(delay), "s"))).argmin()
            if window_specifications.window_method == "Maximum":
                next_horizontal_tidal_accessibility, station = calculate_horizontal_tidal_window(
                    vessel,
                    time_start_index,
                    time_end_index,
                    hydrodynamic_data,
                    cross_current_limit_dataframe=cross_current_limit,
                )
            if window_specifications.window_method == "Point-based":
                if isinstance(window_specifications.current_velocity_values["Flood"], list) and not isinstance(
                    window_specifications.current_velocity_values["Ebb"], list
                ):
                    next_horizontal_tidal_accessibility, station = calculate_horizontal_tidal_window(
                        vessel,
                        time_start_index,
                        time_end_index,
                        hydrodynamic_data,
                        critical_limits=cross_current_limit,
                        ebb=False,
                        decreasing=True,
                    )
                elif isinstance(window_specifications.current_velocity_values["Ebb"], list) and not isinstance(
                    window_specifications.current_velocity_values["Flood"], list
                ):
                    next_horizontal_tidal_accessibility, station = calculate_horizontal_tidal_window(
                        vessel,
                        time_start_index,
                        time_end_index,
                        hydrodynamic_data,
                        critical_limits=cross_current_limit,
                        flood=False,
                        decreasing=True,
                    )
                else:
                    next_horizontal_tidal_accessibility, station = calculate_horizontal_tidal_window(
                        vessel,
                        time_start_index,
                        time_end_index,
                        hydrodynamic_data,
                        critical_limits=cross_current_limit,
                        decreasing=True,
                    )

            horizontal_tidal_restriction_nodes.append(node_name)
            horizontal_tidal_restriction_stations.append(station)
            next_horizontal_tidal_accessibility_time_correction = np.timedelta64(
                int(sailing_time_to_next_node), "s"
            )
            next_horizontal_tidal_accessibility.index -= next_horizontal_tidal_accessibility_time_correction
            if horizontal_tidal_accessibility.empty:
                horizontal_tidal_accessibility = next_horizontal_tidal_accessibility
            else:
                horizontal_tidal_accessibility = calculate_combined_tidal_windows(
                    horizontal_tidal_accessibility, next_horizontal_tidal_accessibility
                )
    if horizontal_tidal_accessibility.empty or not horizontal_tidal_window:
        horizontal_tidal_accessibility = pd.DataFrame(columns=["Limit", "Condition", "Accessibility"])
        horizontal_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [
            0,
            "Current velocity",
            "Accessible",
        ]
        horizontal_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [
            0,
            "Current velocity",
            "Accessible",
        ]
    else:
        if horizontal_tidal_accessibility.iloc[0].Accessibility == "Inaccessible":
            horizontal_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [
                0,
                "Current velocity",
                "Accessible",
            ]
        else:
            horizontal_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [
                0,
                "Current velocity",
                "Inaccessible",
            ]
        if horizontal_tidal_accessibility.iloc[-1].Accessibility == "Inaccessible":
            horizontal_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [
                0,
                "Current velocity",
                "Accessible",
            ]
        else:
            horizontal_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [
                0,
                "Current velocity",
                "Inaccessible",
            ]

    horizontal_tidal_accessibility = horizontal_tidal_accessibility.sort_index()
    horizontal_tidal_windows = [
        [window_start[0], window_end[0]]
        for window_start, window_end in zip(
            horizontal_tidal_accessibility.iloc[:-1].iterrows(), horizontal_tidal_accessibility.iloc[1:].iterrows()
        )
        if window_start[1].Accessibility == "Accessible"
    ]

    return (
        horizontal_tidal_accessibility,
        horizontal_tidal_windows,
        horizontal_tidal_restriction_nodes,
        horizontal_tidal_restriction_stations,
        window_specifications,
    )

def calculate_combined_tidal_windows(tidal_window_1, tidal_window_2):
    tidal_accessibility = pd.concat([tidal_window_1,tidal_window_2],axis=1)
    with pd.option_context("future.no_silent_downcasting", True):
        tidal_accessibility = tidal_accessibility.bfill().infer_objects(copy=False)
    tidal_accessibility = tidal_accessibility.sort_index()
    tidal_accessibility_limit = [limit_1 if not math.isnan(limit_1) else limit_2 for limit_1, limit_2 in tidal_accessibility.Limit.to_numpy()]
    tidal_accessibility_condition = [condition_1 if isinstance(condition_1, str) else condition_2 for condition_1, condition_2 in tidal_accessibility.Condition.to_numpy()]
    tidal_accessibility_accessibility = ["Accessible" if accessibility_1 == accessibility_2 and accessibility_1 == "Accessible" else "Inaccessible" for accessibility_1, accessibility_2 in tidal_accessibility.Accessibility.to_numpy()]
    tidal_accessibility = tidal_accessibility.drop(["Limit", "Condition", "Accessibility"], axis=1)
    tidal_accessibility["Limit"] = tidal_accessibility_limit
    tidal_accessibility["Condition"] = tidal_accessibility_condition
    tidal_accessibility["Accessibility"] = tidal_accessibility_accessibility
    accessible_indexes = [idx for idx, accessibility in enumerate((tidal_accessibility.Accessibility == "Accessible").to_numpy()) if accessibility]
    inaccessible_indexes = [idx for idx, inaccessibility in enumerate((tidal_accessibility.Accessibility == "Inaccessible").to_numpy()) if inaccessibility]
    tidal_window_indexes = np.sort(np.append(accessible_indexes, inaccessible_indexes))
    tidal_accessibility = tidal_accessibility.iloc[tidal_window_indexes]
    return tidal_accessibility

def calculate_minimum_available_water_depth_along_route(vessel, route, time_start, time_end, delay=0):
    """Function: calculates the minimum available water depth (predicted/modelled/measured water level
              minus the local maintained bed level) along the route over time,
              subtracted with the difference between the gross ukc and net ukc
              (hence: subtracted with additional safety margins consisting of vessel-related factors
              and water level factors). The bottom-related factors are already accounted for in the
              use of the Nautical depth instead of the actual depth.

    Input:
        - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
        - route: a list of strings of node names that resemble the route that the vessel is planning
        to sail (can be different than vessel.route)
        - delay:
    """
    hydromanager = HydrodynamicDataManager()
    hydrodynamic_data = hydromanager.hydrodynamic_data
    time_start_index = np.max([0, np.absolute(hydrodynamic_data.TIME.values - (time_start + np.timedelta64(int(delay), "s"))).argmin() - 2,])
    time_end_index = np.absolute(hydrodynamic_data.TIME.values - (time_end + np.timedelta64(int(delay), "s"))).argmin()
    net_ukc = pd.DataFrame()
    times = hydrodynamic_data["TIME"].values[time_start_index:time_end_index]
    t_step = times[1] - times[0]
    t_boundaries = []
    # Start of calculation by looping over the nodes of the route
    for route_index, node_name in enumerate(route):
        node_index = list(hydrodynamic_data["STATION"].values).index(node_name)
        edge_route = node_path_to_edge_path(vessel.env.graph, route[: (route_index + 1)])
        sailing_time_to_next_node, _ = get_sailing_time(vessel, edge_route)
        time_correction_index = int(np.round(sailing_time_to_next_node / (t_step / np.timedelta64(1, "s"))))
        time_end_index_node = np.min([len(hydrodynamic_data["Water level"][node_index])-1,
                                      time_end_index + time_correction_index])
        times = hydrodynamic_data["TIME"].values[time_start_index:time_end_index_node]
        water_level = hydrodynamic_data["Water level"][node_index].values[time_start_index:time_end_index_node]
        _, _, _, required_water_depth, _, _ = calculate_ukc_clearance(vessel, node_name, delay)
        MBL = hydrodynamic_data["Nautical depth"][node_index].values[time_start_index:time_end_index_node]
        water_depth = water_level + MBL
        net_ukc_node = pd.DataFrame([available_water_depth - required_water_depth for available_water_depth in water_depth],columns=[node_name],index=times)
        net_ukc = pd.concat([net_ukc,net_ukc_node],axis=1)
        t_boundaries.append(time_correction_index)

    from IPython.display import display
    min_net_ukc = net_ukc.min(axis=1).min()
    net_ukc_corrected = net_ukc.copy()
    window = False
    column_index = 0
    window_stop = 0
    for column_index,(boundary_start,boundary_stop) in enumerate(zip(t_boundaries[:-1],t_boundaries[1:])):
        window_start = boundary_start
        window_stop = int(np.ceil(np.mean([boundary_start,boundary_stop])))
        window = window_stop - window_start
        net_UKC_node_start = net_ukc.iloc[:, column_index]

        if window:
            net_UKC_node_start = net_ukc.iloc[:, column_index].rolling(window=window, center=False).min().shift(-window_start-window)
        window_start = int(np.floor(np.mean([boundary_start,boundary_stop])))
        window_stop = boundary_stop
        window = window_stop - window_start

        net_UKC_node_stop = net_ukc.iloc[:, column_index]
        if window:
            net_UKC_node_stop = net_ukc.iloc[:, column_index].rolling(window=window,center=False).min().shift(-window_start)
        net_ukc_node = pd.concat([net_UKC_node_start, net_UKC_node_stop], axis=1)
        net_ukc_node_min = net_ukc_node.min(axis=1)
        net_ukc_corrected[route[column_index]] = net_ukc_node_min

    if window:
        net_ukc_corrected.iloc[:,column_index+1] = net_ukc.iloc[:, -1].rolling(window=window,center=False).min().shift(-window_stop)
    net_ukc = net_ukc_corrected.ffill().fillna(min_net_ukc)
    net_ukc["min_net_ukc"] = net_ukc.min(axis=1)
    return net_ukc


def calculate_ukc_clearance(vessel, node, delay=0):
    """Function: calculates the sail-in-times for a specific vssel with certain properties
    and a pre-determined route and provides this information to the vessel

    Input:
        - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
        - node:
        - components_calc:

    """
    MBL, _, available_water_depth = get_water_depth(vessel, node, delay)
    ukc_s, ukc_p, ukc_r, fwa = np.zeros(4)
    if "Vertical tidal restriction" in vessel.multidigraph.nodes[node].keys():
        ukcs_s, ukcs_p, ukcs_r, fwas = vessel.multidigraph.nodes[node]["Vertical tidal restriction"]["Type"]
        specifications = vessel.multidigraph.nodes[node]["Vertical tidal restriction"]["Specification"]

        # Determine which restriction applies to vessel
        restriction_index, _ = determine_tidal_window_restriction(vessel, [node], specifications, node, delay=delay)

        # Calculate ukc policy based on the applied restriction
        ukc_s = ukcs_s[restriction_index]
        ukc_p = ukcs_p[restriction_index] * vessel.T
        ukc_r = ukcs_r[restriction_index][0] * (vessel.T - ukcs_r[restriction_index][1])
        fwa = fwas[restriction_index] * vessel.T
    extra_ukc = 0.
    if 'metadata' in dir(vessel) and "ukc" in vessel.metadata.keys():
        extra_ukc = vessel.metadata["ukc"]
    ship_related_factors = {"ukc_s": ukc_s, "ukc_p": ukc_p, "ukc_r": ukc_r, "fwa": fwa, "extra_ukc": extra_ukc}
    required_water_depth = vessel.T + sum(ship_related_factors.values())
    net_ukc = available_water_depth - required_water_depth
    gross_ukc = available_water_depth - vessel.T
    return net_ukc, gross_ukc, available_water_depth, required_water_depth, ship_related_factors, MBL


def calculate_tidal_windows(vessel, route, time_start, time_end, delay=0):
    hydromanager = HydrodynamicDataManager()
    hydrodynamic_data = hydromanager.hydrodynamic_data
    time_start_index = np.max([0, np.absolute(hydrodynamic_data.TIME.values - (time_start + np.timedelta64(int(delay), "s"))).argmin() - 2, ])
    time_end_index = np.absolute(hydrodynamic_data.TIME.values - (time_end + np.timedelta64(int(delay), "s"))).argmin()

    vertical_tidal_accessibility, \
    vertical_tidal_windows, \
    net_ukcs = calculate_vertical_tidal_windows(vessel, route, time_start, time_end, delay)

    horizontal_tidal_accessibility,\
    horizontal_tidal_windows,\
    horizontal_tidal_restriction_nodes,\
    horizontal_tidal_restriction_stations,\
    window_specifications = calculate_horizontal_tidal_windows(vessel, route, time_start, time_end, delay)

    tidal_accessibility = calculate_combined_tidal_windows(vertical_tidal_accessibility, horizontal_tidal_accessibility)
    tidal_windows = [[window_start[0], window_end[0]] for window_start, window_end in zip(tidal_accessibility.iloc[:-1].iterrows(), tidal_accessibility.iloc[1:].iterrows()) if window_start[1].Accessibility == "Accessible"]

    tidal_window_results = {'time_start_index':time_start_index,
                            'time_end_index':time_end_index,
                            'route':route,
                            'bound':vessel.bound,
                            'draught':vessel.T,
                            'vertical_tidal_accessibility':vertical_tidal_accessibility,
                            'vertical_tidal_windows':vertical_tidal_windows,
                            'net_ukcs':net_ukcs,
                            'horizontal_tidal_accessibility':horizontal_tidal_accessibility,
                            'horizontal_tidal_windows':horizontal_tidal_windows,
                            'horizontal_tidal_restriction_nodes':horizontal_tidal_restriction_nodes,
                            'horizontal_tidal_restriction_stations':horizontal_tidal_restriction_stations,
                            'window_specifications':window_specifications,
                            'tidal_accessibility':tidal_accessibility,
                            'tidal_windows':tidal_windows}
    return tidal_window_results


def calculate_ukc_per_tidal_period(vessel, trip_index=0, duration=pd.Timedelta(days=2)):
    net_ukc = vessel.tidal_window_calculations[trip_index]['net_ukcs']['min_net_ukc']
    tidal_signal = net_ukc.values
    time = net_ukc.index

    # Detect peaks (high tides)
    peaks, _ = find_peaks(tidal_signal)

    # Detect troughs (low tides)
    troughs, _ = find_peaks(-tidal_signal)

    # Calculate tidal periods (high-to-high)
    time_stop = net_ukc.index[0] + duration
    net_ukc[net_ukc.index <= time_stop]
    tidal_periods = pd.DataFrame(columns=['Time_start', 'Time_end', 'Duration', 'Peak', 'Through'])
    for index, through_index in enumerate(troughs):
        if not index:
            time_start = time[through_index]
            continue
        time_end = time[through_index]
        tidal_periods.loc[index - 1, 'Time_start'] = time_start
        tidal_periods.loc[index - 1, 'Time_end'] = time_end
        time_start = time_end

    for index, period in tidal_periods.iterrows():
        net_ukc_period = net_ukc[(net_ukc.index > period.Time_start) & (net_ukc.index < period.Time_end)]
        tidal_periods.loc[index, 'Duration'] = period.Time_end - period.Time_start
        tidal_periods.loc[index, 'Peak'] = net_ukc_period.max()
        tidal_periods.loc[index, 'Through'] = net_ukc_period.min()
    return tidal_periods


def calculate_accessibility(vessel, trip_index=0, duration=pd.Timedelta(days=2)):
    # tidal calculation results
    tidal_accessibility_df = vessel.tidal_window_calculations[trip_index]['tidal_accessibility']
    net_ukc = vessel.tidal_window_calculations[trip_index]['net_ukcs']['min_net_ukc']
    tidal_accessibility_df = tidal_accessibility_df[tidal_accessibility_df.index >= net_ukc.index[0]]

    # set end time
    end_time = tidal_accessibility_df.index[0] + duration

    # reset index
    df = tidal_accessibility_df.copy()
    df.index.name = "Time_start"
    df = df.reset_index()
    df["Time_start"] = pd.to_datetime(df["Time_start"])

    # only considering duration
    df = df[df["Time_start"] <= end_time]

    # add window end time and window_id
    df['Time_end'] = df['Time_start'].shift(-1)
    df.loc[df.index[-1], 'Time_end'] = end_time

    # determine windows
    df['Change'] = df['Accessibility'].ne(df['Accessibility'].shift())
    df['Window_id'] = df['Change'].cumsum()

    window_summary = df.groupby('Window_id').agg(
        Time_start=('Time_start', 'min'),
        Time_end=('Time_end', 'max'),
        Accessibility=('Accessibility', 'first'),
        Condition=('Condition', lambda x: x.value_counts(normalize=True).to_dict())
    ).reset_index()

    # add duration
    window_summary['Duration'] = window_summary['Time_end'] - window_summary['Time_start']

    # calculate duration
    window_summary['Duration'] = window_summary['Duration'].dt.total_seconds()
    windows_summary_accessible = window_summary[window_summary['Accessibility'] == 'Accessible']

    # determine shortest, longest, and average window
    df = windows_summary_accessible.copy()
    shortest_window_id = df.loc[df['Duration'].idxmin(), 'Window_id']
    shortest_window_duration = df['Duration'].min()
    longest_window_id = df.loc[df['Duration'].idxmax(), 'Window_id']
    longest_window_duration = df['Duration'].max()
    mean_window_duration = df['Duration'].mean()

    # determine accessibility percentage
    total_seconds = duration.total_seconds()
    accessible_seconds = df.loc[df['Accessibility'] == 'Accessible', 'Duration'].sum()
    accessibility_percentage_time = (accessible_seconds / total_seconds) * 100

    # determine accessibility per tide
    tidal_periods = calculate_ukc_per_tidal_period(vessel, trip_index, duration)
    tidal_periods["key"] = 1
    df["key"] = 1
    merged = tidal_periods.merge(df, on="key", suffixes=("_window", "_tide")).drop("key", axis=1)

    # Overlap condition
    overlap = ((merged["Time_start_window"] <= merged["Time_end_tide"]) &
               (merged["Time_end_window"] >= merged["Time_start_tide"]))
    overlapping_windows = merged[overlap].copy()
    accessibility_percentage_tide = len(tidal_periods)/len(overlapping_windows)*100

    # determine normative conditions
    condition_totals = {}
    for _, row in df.iterrows():
        duration_sec = row['Duration']
        for cond, pct in row['Condition'].items():
            condition_totals[cond] = condition_totals.get(cond, 0) + duration_sec * (pct / 100)

    total_seconds = sum(condition_totals.values())
    condition_percent_total = {k: v / total_seconds * 100 for k, v in condition_totals.items()}

    # store results of analysis
    results = pd.Series({
        'number_of_tidal_windows': len(df),
        'shortest_tidal_window_id': shortest_window_id,
        'shortest_tidal_window_duration (s)': pd.Timedelta(seconds=shortest_window_duration),
        'longest_tidal_window_id': longest_window_id,
        'longest_tidal_window_duration (s)': pd.Timedelta(seconds=longest_window_duration),
        'mean_tidal_window_duration (s)': pd.Timedelta(seconds=mean_window_duration),
        'accessibility_percentage (time %)': accessibility_percentage_time,
        'accessibility_percentage (tide %)': accessibility_percentage_tide,
        'causes_tidal_window (cause %)': condition_percent_total
    })

    return results


def calculate_berth_planning_information(berth, time_start = None, time_stop = None):
    berth_planning_df = berth.historic_berth_planning.copy()
    if time_start is None:
        time_start = berth_planning_df.index[0]
    if time_stop is None:
        time_stop = berth_planning_df.index[-1]
    berth_planning_df = berth_planning_df[(berth_planning_df.index >= time_start) &
                                          (berth_planning_df.index <= time_stop)]
    df = berth_planning_df.copy()

    results = []
    for timestamp in df.index:
        row = df.loc[timestamp].dropna()
        row = row.sort_index()

        vessel_segments = []
        current_vessel = None
        start_segment = None
        prev_segment = None
        for segment, vessel in row.items():
            if vessel != current_vessel:
                if current_vessel is not None:
                    vessel_segments.append((current_vessel, start_segment, prev_segment))
                current_vessel = vessel
                start_segment = segment
            prev_segment = segment

        if current_vessel is not None:
            vessel_segments.append((current_vessel, start_segment, prev_segment))

        # save results with timestamp
        for vessel, start_seg, end_seg in vessel_segments:
            results.append((vessel, timestamp, start_seg, end_seg))

    occupied_df = pd.DataFrame(results, columns=['vessel_id', 'timestamp', 'start_berth', 'end_berth'])
    occupied_df.sort_values(['vessel_id', 'timestamp'], inplace=True)
    occupied_df.reset_index(drop=True, inplace=True)
    occupied_summary = occupied_df.groupby(['vessel_id', 'start_berth', 'end_berth']).agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max')
    ).reset_index()
    occupied_summary['duration'] = occupied_summary['end_time'] - occupied_summary['start_time']
    return occupied_summary


def calculate_berth_performance(berth, time_start = None, time_stop = None):
    from opentnsim.port.mixins.berth import IsQuay, IsJetty
    berth_planning_df = berth.historic_berth_planning.copy()
    if time_start is None:
        time_start = berth_planning_df.index[0]
    if time_stop is None:
        time_stop = berth_planning_df.index[-1]
    berth_planning_df = berth_planning_df[(berth_planning_df.index >= time_start) &
                                          (berth_planning_df.index <= time_stop)]
    duration = berth_planning_df.index[-1] - berth_planning_df.index[0]
    occupied_df = calculate_berth_planning_information(berth, time_start, time_stop)
    occupied_df = occupied_df.sort_values('start_time')

    total_vessels_handled = len(occupied_df)
    shortest_occupation = occupied_df.duration.min()
    shortest_occupation_id = occupied_df.duration.idxmin()
    mean_occupation = occupied_df.duration.mean()
    longest_occupation = occupied_df.duration.max()
    longest_occupation_id = occupied_df.duration.idxmax()
    total_occupied_duration = occupied_df.duration.sum()
    berth_occupancy = np.nan
    vessels_at_berth = {}
    for vessel_id in occupied_df.vessel_id:
        for vessel in berth.env.vessels:
            if vessel.id == vessel_id:
                vessels_at_berth[vessel.id] = vessel

    routes_to_terminal = []
    for port_entry_node in berth.terminal.port.port_entry_nodes:
        routes_to_terminal.append(nx.dijkstra_path(berth.env.graph,port_entry_node,berth.node))

    routes = []
    berth_waiting_time_causes = {}
    total_waiting_time_at_berth = pd.Timedelta(seconds=0)
    for _,vessel_berth_info in occupied_df.iterrows():
        berthing_start_time = vessel_berth_info.start_time
        vessel = vessels_at_berth[vessel_berth_info.vessel_id]
        vessel_df = pd.DataFrame(vessel.logbook)
        vessel_df = vessel_df[vessel_df.Timestamp <= berthing_start_time]
        df_sailing = vessel_df[vessel_df["Message"].str.contains("Sailing", case=False, na=False)].copy()
        sailing_message = r"Sailing from node (.*?) to node (.*?) (start|stop)"
        df_sailing[["node_start", "node_stop", "event"]] = (df_sailing["Message"].str.extract(sailing_message))
        edges = df_sailing[df_sailing["event"] == "start"][["node_start", "node_stop"]]
        route = [edges.iloc[0]["node_start"]] + edges["node_stop"].tolist()

        final_route_to_terminal = []
        for route_to_terminal in routes_to_terminal:
            final_route = get_longest_common_subroute(route,route_to_terminal)
            if len(final_route) > len(final_route_to_terminal):
                final_route_to_terminal = final_route

        edges = list(zip(final_route_to_terminal[:-1], final_route_to_terminal[1:]))
        df_to_terminal = df_sailing[df_sailing.apply(lambda x: (x["node_start"], x["node_stop"]) in edges, axis=1)]
        df_to_terminal = df_to_terminal[df_to_terminal["event"].isin(["start", "stop"])]
        first_index_sailing = df_to_terminal.index[0]
        last_index_sailing = df_to_terminal.index[-1]
        df_sailing = vessel_df[(vessel_df.index >= first_index_sailing)& (vessel_df.index <= last_index_sailing)]
        df_sailing = df_sailing[df_sailing["Message"].str.contains("Waiting", case=False, na=False)]
        vessel_waiting_causes = {}
        vessel_total_waiting_time= pd.Timedelta(seconds=0)
        if not df_sailing.empty:
            waiting_message = r"Waiting for (.*?) (start|stop)"
            df_sailing[["waiting_type", "event"]] = df_sailing["Message"].str.extract(waiting_message)
            waiting_df = df_sailing[df_sailing["waiting_type"].notna()].copy()
            waiting_durations = (waiting_df.pivot(index="waiting_type", columns="event", values="Timestamp"))
            waiting_durations["duration"] = (waiting_durations["stop"] - waiting_durations["start"])
            for waiting_cause, waiting_event in waiting_durations.iterrows():
                if waiting_cause not in vessel_waiting_causes.keys():
                    vessel_waiting_causes[waiting_cause] = pd.Timedelta(seconds=0)
                vessel_waiting_causes[waiting_cause] += waiting_event.duration
                vessel_total_waiting_time += waiting_event.duration

        last_index_before_sailing = first_index_sailing - 1
        df_before_sailing = vessel_df[(vessel_df.index <= last_index_before_sailing)]
        if not df_before_sailing.empty:
            waiting_message = r"Waiting for (.*?) (start|stop)"
            df_before_sailing[["waiting_type", "event"]] = df_before_sailing["Message"].str.extract(waiting_message)
            waiting_df = df_before_sailing[df_before_sailing["waiting_type"].notna()].copy()
            consecutive_indices = [i for i in range(last_index_before_sailing, min(waiting_df.index) - 1, -1) if
                                   all(j in waiting_df.index for j in range(i, last_index_before_sailing + 1))]
            waiting_df = waiting_df[waiting_df.index.isin(consecutive_indices)]
            waiting_df = waiting_df.rename(columns={"Timestamp":"start"})
            waiting_df["next_timestamp"] = waiting_df["start"].shift(-1)
            waiting_df["next_event"] = waiting_df["event"].shift(-1)
            waiting_df["next_type"] = waiting_df["waiting_type"].shift(-1)
            waiting_events = waiting_df[
                (waiting_df["event"] == "start") &
                (waiting_df["next_event"] == "stop") &
                (waiting_df["waiting_type"] == waiting_df["next_type"])
                ].copy()
            waiting_events["stop"] = waiting_events["next_timestamp"]
            waiting_events["duration"] = waiting_events["stop"] - waiting_events["start"]
            waiting_durations = waiting_events[["waiting_type", "start", "stop", "duration"]]
            waiting_durations["duration"] = (waiting_durations["stop"] - waiting_durations["start"])
            waiting_durations = waiting_durations.groupby("waiting_type")["duration"].sum()

            for waiting_cause, waiting_duration in waiting_durations.items():
                if waiting_cause not in vessel_waiting_causes.keys():
                    vessel_waiting_causes[waiting_cause] = pd.Timedelta(seconds=0)
                vessel_waiting_causes[waiting_cause] += waiting_duration
                vessel_total_waiting_time += waiting_duration

        for waiting_cause, waiting_time in vessel_waiting_causes.items():
            if waiting_cause not in berth_waiting_time_causes.keys():
                berth_waiting_time_causes[waiting_cause] = pd.Timedelta(seconds=0)
            berth_waiting_time_causes[waiting_cause] += waiting_time
        total_waiting_time_at_berth += vessel_total_waiting_time

    average_vessel_length = np.mean([vessel.L for vessel in vessels_at_berth.values()])
    average_waiting_time = total_waiting_time_at_berth/len(vessels_at_berth)
    waiting_cause_reasons = {waiting_cause: np.round((waiting_time / total_waiting_time_at_berth)*100,1)
                             for waiting_cause, waiting_time in berth_waiting_time_causes.items()}
    waiting_time_rate = total_waiting_time_at_berth

    if not len(vessels_at_berth):
        berth_occupancy = 0.
    elif isinstance(berth, IsQuay):
        number_of_berths = berth.berth_length/(average_vessel_length)
        berth_occupancy = total_occupied_duration/(number_of_berths*duration)*100
    elif isinstance(berth, IsJetty):
        berth_occupancy = total_occupied_duration/duration*100

    results = pd.Series({'Total vessels handled':total_vessels_handled,
                         'Average vessel length': average_vessel_length,
                         'Shortest service time':shortest_occupation,
                         'Vessel_id shortest service time': shortest_occupation_id,
                         'Average service time': mean_occupation,
                         'Longest service time': longest_occupation,
                         'Vessel_id longest service time': longest_occupation_id,
                         'Total service time':total_occupied_duration,
                         'Berth occupancy (%)':berth_occupancy,
                         'Average waiting time':average_waiting_time,
                         'Total waiting time':total_waiting_time_at_berth,
                         'Waiting time rate':total_waiting_time_at_berth/total_occupied_duration,
                         'Causes waiting time': waiting_cause_reasons})
    return results