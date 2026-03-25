import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from opentnsim.graph.utils import get_trajectory
from opentnsim.graph.calculations import transform_geometry
from operator import itemgetter

def create_time_distance_plot(lock_chamber, xlimmin=None, xlimmax=None, ylimmin=None, ylimmax=None, method='Matplotlib'):
    """Create a time-distance plot of vessels passing a lock complex

    Parameters
    ----------
    vessels: list of vessel type objects
        the vessels that have been simulated (a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput)
    xlimmin : float
        minimum x coordinate as distance front the lock complex (should be negative) [m]
    xlimmax : float
        maximum x coordinate as distance front the lock complex (should be positive) [m]
    ylimmin : pd.Timestamp
        minimum time (should be equal or greater that the simulation start time)
    ylimmax : pd.Timestamp
        maximum time (should be equal or smaller that the simulation stop time)

    Returns
    -------
    nothing, but creates a plot

    """
    lock_complex = lock_chamber.lock_complex
    vessel_ids = lock_complex.vessel_planning[lock_complex.vessel_planning.lock_chamber == lock_chamber.name].id
    try:
        vessels = np.array([itemgetter(*vessel_ids)(lock_complex.env.vessels)]).flatten()
    except:
        vessels = []

    # create lock edge geometry in [m]
    route_between_nodes_of_registration = []
    nodes = [lock_complex.registration_nodes[0], lock_chamber.edge[0], lock_complex.registration_nodes[-1]]
    for i in range(len(nodes) - 1):
        segment = nx.dijkstra_path(lock_chamber.env.graph, nodes[i], nodes[i + 1])
        if i > 0:
            segment = segment[1:]
        route_between_nodes_of_registration.extend(segment)

    lock_edge_geometry = get_trajectory(lock_complex.env.graph, route_between_nodes_of_registration[0],
                                        route_between_nodes_of_registration[-1])
    lock_edge_geometry_m = transform_geometry(lock_edge_geometry, epsg_out=lock_chamber.crs_m)

    # plot the lock geometry over time
    location_lock_gate_A_m = transform_geometry(lock_chamber.gate_A.geometry, epsg_out=lock_chamber.crs_m)
    location_lock_gate_B_m = transform_geometry(lock_chamber.gate_B.geometry, epsg_out=lock_chamber.crs_m)
    x_lock_gateA = (lock_edge_geometry_m.line_locate_point(location_lock_gate_A_m))
    x_lock_gateB = (lock_edge_geometry_m.line_locate_point(location_lock_gate_B_m))
    x_correction_indirection = x_lock_gateA + lock_chamber.lock_length / 2
    x_correction_outdirection = x_lock_gateB - lock_chamber.lock_length / 2

    # determine the accepted messages for plotting
    accepted_messages = []
    for node_start, node_end in zip(route_between_nodes_of_registration[:-1], route_between_nodes_of_registration[1:]):
        accepted_messages.extend([f"Sailing from node {node_start} to node {node_end} start",
                                  f"Sailing from node {node_end} to node {node_start} start",
                                  f"Sailing from node {node_start} to node {node_end} stop",
                                  f"Sailing from node {node_end} to node {node_start} stop"])

    accepted_messages.extend(["Waiting for other vessel in lock operation start",
                              "Waiting for other vessel in lock operation stop",
                              "Waiting for lock operation start",
                              "Waiting for lock operation stop",
                              "Sailing to first lock gate start",
                              "Sailing to first lock gate stop",
                              "Sailing to position in lock start",
                              "Sailing to position in lock stop",
                              "Levelling start",
                              "Levelling stop",
                              "Sailing to second lock gate start",
                              "Sailing to second lock gate stop",
                              "Sailing to lock complex exit start",
                              "Sailing to lock complex exit stop"])

    # loop over vessels to extract time and distance from lock passage messages and store them in a list
    all_times = []
    all_distances = []
    traces = []
    for vessel in vessels:
        times = []
        distances = []
        vessel_df = pd.DataFrame(vessel.logbook)
        vessel_df["Geometry"] = vessel_df["Geometry"].apply(lambda x: transform_geometry(x, epsg_out=lock_chamber.crs_m))
        x_correction = 0.0
        for index, message_info in vessel_df.iterrows():
            time = message_info.Timestamp
            distance = lock_edge_geometry_m.line_locate_point(message_info.Geometry)
            route = vessel.route
            if lock_chamber.start_node not in route or lock_chamber.end_node not in route:
                continue

            if message_info.Message in accepted_messages:
                if message_info.Message == f"Sailing from node {lock_chamber.start_node} to node {lock_chamber.end_node} start":
                    x_correction = x_correction_indirection
                elif message_info.Message == f"Sailing from node {lock_chamber.end_node} to node {lock_chamber.start_node} start":
                    x_correction = x_correction_outdirection
                times.append(time)
                distances.append(distance)
        distances = np.array(distances) - x_correction
        all_times.append(times)
        all_distances.append(distances)

        # Add vessel trace with vessel.name in legend
        if method == 'Plotly':
            traces.append(go.Scatter(x=distances, y=times, mode='lines', name=vessel.name))

    if method == 'Matplotlib':
        fig, ax = plt.subplots()
        for distances, times in zip(all_distances, all_times):
            ax.plot(distances, times)
    elif method == 'Plotly':
        fig = go.Figure(data=traces)

    # Determine y-axis limits
    all_y_values = [t for sublist in all_times for t in sublist]
    if all_y_values:
        if ylimmin is None:
            ylimmin = min(all_y_values)
        if ylimmax is None:
            ylimmax = max(all_y_values)

    # Determine x-axis limits
    sailing_distance_to_crossing_point = lock_chamber.sailing_distance_to_crossing_point + lock_chamber.lock_length / 2
    if xlimmin is None:
        xlimmin = -2 * sailing_distance_to_crossing_point
    if xlimmax is None:
        xlimmax = 2 * sailing_distance_to_crossing_point

    if method == 'Matplotlib':
        lock_extend_x = np.array(
            [x_lock_gateA, x_lock_gateA, x_lock_gateB, x_lock_gateB]) - x_correction_indirection
        ax.fill(lock_extend_x, [ylimmin, ylimmax, ylimmax, ylimmin], color="lightgrey", zorder=0)
    elif method == 'Plotly':
        fig.add_shape(type="rect",
                      x0=x_lock_gateA - x_correction_indirection, x1=x_lock_gateB - x_correction_indirection,
                      y0=ylimmin, y1=ylimmax,
                      fillcolor="lightgrey", opacity=0.5,
                      layer="below", line_width=0,
                      name="Lock Geometry")

    # plot the lock phases
    lock_df = pd.DataFrame(lock_chamber.logbook)
    for index, message_info in lock_df.iterrows():
        message_found = False
        if message_info.Message == "Lock gate opening stop" and index != 0:
            time_start = lock_df.loc[index - 1, "Timestamp"]
            time_stop = message_info.Timestamp
            color = "darkgrey"
            name = "Lock gate opening"
            message_found = True
        if message_info.Message == "Lock gate closing stop" and index != 0:
            time_start = lock_df.loc[index - 1, "Timestamp"]
            time_stop = message_info.Timestamp
            color = "darkgrey"
            name = "Lock gate closing"
            message_found = True
        if message_info.Message == "Lock chamber converting stop" and index != 0:
            time_start = lock_df.loc[index - 1, "Timestamp"]
            time_stop = message_info.Timestamp
            color = "grey"
            name = "Lock chamber converting"
            message_found = True

        if method == 'Matplotlib' and message_found:
            ax.fill(lock_extend_x, [time_start, time_stop, time_stop, time_start], color=color, zorder=0)
        elif method == 'Plotly' and message_found:
            fig.add_shape(type="rect",
                          x0=x_lock_gateA - x_correction_indirection, x1=x_lock_gateB - x_correction_indirection,
                          y0=time_start, y1=time_stop,
                          fillcolor=color, opacity=0.5,
                          layer="below", line_width=0,
                          name=name)

    # plot the approach points
    sailing_distance_to_crossing_point = lock_chamber.sailing_distance_to_crossing_point + lock_chamber.lock_length / 2
    xlabel = "Distance from Lock Complex [m]"
    ylabel = "Timestamp"
    title = "Time-Distance Plot of Vessel Movements"
    if method == 'Matplotlib':
        ax.axvline(-sailing_distance_to_crossing_point, color="lightgrey", zorder=0)
        ax.axvline(sailing_distance_to_crossing_point, color="lightgrey", zorder=0)
        ax.set_xlim([xlimmin, xlimmax])
        ax.set_ylim([ylimmin, ylimmax])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    elif method == 'Plotly':
        fig.add_vline(x=-sailing_distance_to_crossing_point, line=dict(color="lightgrey"))
        fig.add_vline(x=sailing_distance_to_crossing_point, line=dict(color="lightgrey"))
        fig.update_layout(title=title,
                          xaxis_title=xlabel,
                          yaxis_title=ylabel,
                          xaxis_range=[xlimmin, xlimmax],
                          yaxis_range=[ylimmin, ylimmax],
                          showlegend=True)

    return fig