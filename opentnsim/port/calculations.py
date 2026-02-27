import bisect
import math
import matplotlib.dates as mdates
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
from scipy.interpolate import interp1d
from shapely.ops import linemerge, transform
from shapely.geometry import LineString, MultiLineString
from pyproj.transformer import Transformer

import xarray as xr


def provide_trajectory(graph, node_1, node_2):
    nodes = nx.dijkstra_path(graph, node_1, node_2)
    final_geometry = LineString()
    multigraph = False
    if isinstance(graph, nx.MultiDiGraph):
        multigraph = True
    for loc, edge in enumerate(zip(nodes[:-1], nodes[1:])):
        if multigraph:
            k = sorted(multidigraph[edge[0]][edge[1]], key=lambda x: multidigraph[edge[0]][edge[1]][x]["geometry"].length)[0]
            geom = multidigraph.edges[edge[0], edge[1], k]["geometry"]
        else:
            geom = graph.edges[edge[0], edge[1]]["geometry"]

        if not loc:
            final_geometry = geom
            continue

        multi_line = MultiLineString([final_geometry, geom])
        final_geometry = linemerge(multi_line)

    return final_geometry


def transform_geometry(geometry, crs_in = "EPSG:4326", crs_out = "EPSG:3857"):
    proj_in = pyproj.CRS(crs_in)
    proj_out = pyproj.CRS(crs_out)
    transformer = Transformer.from_crs(proj_in, proj_out, always_xy=True).transform
    geometry_transformed = transform(transformer,geometry)
    return geometry_transformed


def transform_route_geometry(env, node_start, node_stop, crs_in = "EPSG:4326", crs_out = "EPSG:3857"):
    route_geometry = provide_trajectory(env.graph, node_start, node_stop)
    route_geometry_transformed = transform_geometry(route_geometry, crs_in, crs_out)
    return route_geometry_transformed


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
    hydrodynamic_data = env.vessel_traffic_service.hydrodynamic_information
    water_depth = hydrodynamic_data['Water level'] + hydrodynamic_data['MBL']
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
            node_water_depths[node + str(boundary)] = hydrodynamic_data['Water level'].sel({'STATION': node}).values + infrastructure.depth
            node_distances[node + str(boundary)] = np.ones(len(node_water_depths[node + str(boundary)])) * distance_to_node + boundary_offset
            node_times[node + str(boundary)] = water_depth.TIME.values

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


def calculate_horizontal_tidal_windows(vessel, route, time_start, time_end, delay=0):
    def calculate_horizontal_tidal_window(
        vessel,
        time_start_index,
        time_end_index,
        hydrodynamic_data,
        critical_limits=[],
        cross_current_limit_dataframe=pd.DataFrame(),
        flood=True,
        ebb=True,
        decreasing=False,
    ):
        station = hydrodynamic_data.STATION.values
        time_start_index = np.max(
            [
                0,
                time_start_index
                - int(np.timedelta64(12, "h") / (hydrodynamic_data.TIME.values[1] - hydrodynamic_data.TIME.values[0])),
            ]
        )
        currents_time = hydrodynamic_data.TIME.values[time_start_index:time_end_index]
        currents_data, _ = vessel.env.vessel_traffic_service.provide_governing_current_velocity(vessel, station, time_start_index, time_end_index)
        index_prev_root = 0
        roots = sc.interpolate.CubicSpline(currents_time, currents_data).roots()

        roots_cv = [
            root for root in roots if root >= currents_time[0].astype(float) and root <= currents_time[-1].astype(float)
        ]
        times_horizontal_tidal_period = []
        for root in roots_cv:
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
        currents_time = np.append(
            currents_time, np.array([tide[0] for tide in tidal_periods if tide[0] not in currents_time], dtype="datetime64[ns]")
        )
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

    # Start calculation
    horizontal_tidal_restriction_nodes = []
    horizontal_tidal_restriction_stations = []
    window_specifications = []
    horizontal_tidal_accessibility = pd.DataFrame(columns=["Limit", "Condition", "Accessibility"])
    horizontal_tidal_window = False
    for route_index, node_name in enumerate(route):
        if "Horizontal tidal restriction" in vessel.multidigraph.nodes[node_name].keys():
            horizontal_tidal_window = True
            sailing_time_to_next_node = vessel.env.vessel_traffic_service.provide_sailing_time(vessel, route[: (route_index + 1)])
            specifications = vessel.multidigraph.nodes[node_name]["Horizontal tidal restriction"]["Specification"]
            restriction_index, no_tidal_window = determine_tidal_window_restriction(
                vessel, route, specifications, node_name, delay=delay
            )
            if no_tidal_window:
                continue
            hydrodynamic_data = vessel.multidigraph.nodes[node_name]["Horizontal tidal restriction"]["Data"][
                restriction_index
            ]
            cross_current_limit = vessel.multidigraph.nodes[node_name]["Horizontal tidal restriction"]["Limit"][
                restriction_index
            ]
            window_specifications = vessel.multidigraph.nodes[node_name]["Horizontal tidal restriction"]["Type"][
                restriction_index
            ]
            time_start_index = np.max(
                [
                    0,
                    np.absolute(
                        vessel.env.vessel_traffic_service.hydrodynamic_information.TIME.values - (time_start + np.timedelta64(int(delay), "s"))
                    ).argmin()
                    - 2,
                ]
            )
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
                int(sailing_time_to_next_node["Time"].sum()), "s"
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
              use of the MBL instead of the actual depth.

    Input:
        - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
        - route: a list of strings of node names that resemble the route that the vessel is planning
        to sail (can be different than vessel.route)
        - delay:
    """
    hydrodynamic_information = vessel.env.vessel_traffic_service.hydrodynamic_information
    time_start_index = np.max([0, np.absolute(hydrodynamic_information.TIME.values - (time_start + np.timedelta64(int(delay), "s"))).argmin() - 2,])
    time_end_index = np.absolute(hydrodynamic_information.TIME.values - (time_end + np.timedelta64(int(delay), "s"))).argmin()
    net_ukc = pd.DataFrame()
    times = hydrodynamic_information["TIME"].values[time_start_index:time_end_index]
    t_step = times[1] - times[0]
    t_boundaries = []
    # Start of calculation by looping over the nodes of the route
    for route_index, node_name in enumerate(route):
        node_index = list(hydrodynamic_information["STATION"].values).index(node_name)
        sailing_time_to_next_node = vessel.env.vessel_traffic_service.provide_sailing_time(vessel, route[: (route_index + 1)])
        time_correction_index = int(np.round(sailing_time_to_next_node["Time"].sum() / (t_step / np.timedelta64(1, "s"))))
        time_end_index = np.min([len(hydrodynamic_information["Water level"][node_index])-1,time_end_index + time_correction_index])
        times = hydrodynamic_information["TIME"].values[time_start_index:time_end_index]
        water_level = hydrodynamic_information["Water level"][node_index].values[time_start_index:time_end_index]
        _, _, _, required_water_depth, _, _ = calculate_ukc_clearance(vessel, node_name, delay)
        MBL = hydrodynamic_information["MBL"][node_index].values[time_start_index:time_end_index]
        water_depth = water_level + MBL
        net_ukc_node = pd.DataFrame([available_water_depth - required_water_depth for available_water_depth in water_depth],columns=[node_name],index=times)
        net_ukc = pd.concat([net_ukc,net_ukc_node],axis=1)
        t_boundaries.append(time_correction_index)

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
    # ignore water_level
    MBL, _, available_water_depth = vessel.env.vessel_traffic_service.provide_water_depth(vessel, node, delay)

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
    time_start_index = np.max([0, np.absolute(vessel.env.vessel_traffic_service.hydrodynamic_information.TIME.values - (time_start + np.timedelta64(int(delay), "s"))).argmin() - 2, ])
    time_end_index = np.absolute(vessel.env.vessel_traffic_service.hydrodynamic_information.TIME.values - (time_end + np.timedelta64(int(delay), "s"))).argmin()

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