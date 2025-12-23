from opentnsim.port.utils import transform_geometry, transform_route_geometry

import networkx as nx
import numpy as np
import matplotlib.dates as mdates
from scipy.interpolate import interp1d

def calculate_total_waiting_time(waiting_events):
    total_waiting_time = 0.
    if waiting_events is not None and len(waiting_events):
        total_waiting_time = sum(waiting_events.values())
    return total_waiting_time


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


def calculate_interpolated_depth_values(env, node_start, node_stop, offset=500):
    node_distances, node_times, node_water_depths = calculate_depth_values_over_route(env, node_start, node_stop, offset)

    node_distances = np.concatenate(list(node_distances.values()))
    node_times = np.concatenate(list(node_times.values()))
    node_water_depths = np.concatenate(list(node_water_depths.values()))

    node_times, time_idx = np.unique(node_times, return_inverse=True)
    node_times_num = mdates.date2num(node_times)

    interpolated_distance = np.linspace(node_distances.min(), node_distances.max(), 200)  # horizontal resolution
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

