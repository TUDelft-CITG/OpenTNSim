import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from opentnsim.graph.utils import get_trajectory
from opentnsim.graph.calculations import transform_geometry
from opentnsim.graph.visualizations import (visualize_node_in_folium_plot, visualize_edge_in_folium_plot,
                                            visualize_geometry_point_in_folium_plot,
                                            visualize_geometry_polygon_in_folium_plot)
from opentnsim.lock.utils import _get_vessels_that_passed_the_lock_chamber
import folium
from IPython.display import display, HTML

def add_locking_phases_to_plot(lock_chamber, fig, extend, time_axis = 'x', method='Matplotlib'):
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
        if message_info.Message == "Lock levelling stop" and index != 0:
            time_start = lock_df.loc[index - 1, "Timestamp"]
            time_stop = message_info.Timestamp
            color = "grey"
            name = "Lock levelling"
            message_found = True

        if method == 'Matplotlib' and message_found:
            extend = [extend[0],extend[0],extend[-1],extend[-1]]
            if time_axis == 'y':
                fig.fill(extend, [time_start, time_stop, time_stop, time_start], color=color, zorder=-1)
            else:
                fig.fill([time_start, time_stop, time_stop, time_start], extend, color=color, zorder=-1)
        elif method == 'Plotly' and message_found:
            x_data = [time_start,time_stop]
            y_data = extend
            if time_axis == 'y':
                x_data = extend
                y_data = [time_start,time_stop]
            fig.add_shape(type="rect",
                          x0=x_data[0],
                          x1=x_data[-1],
                          y0=y_data[0],
                          y1=y_data[-1],
                          fillcolor=color, opacity=0.5,
                          layer="below", line_width=0,
                          name=name)


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
    vessels = _get_vessels_that_passed_the_lock_chamber(lock_chamber)

    # create lock edge geometry in [m]
    route_between_nodes_of_registration = nx.dijkstra_path(lock_complex.env.graph, lock_complex.registration_nodes[0],
                                                           lock_complex.registration_nodes[1])
    lock_edge_geometry = get_trajectory(lock_complex.env.graph, route_between_nodes_of_registration[0],
                                        route_between_nodes_of_registration[-1])[0]
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
                              "Waiting for lock gate closing start",
                              "Waiting for lock gate closing stop",
                              "Waiting for other vessels in lock start",
                              "Waiting for other vessels in lock stop",
                              "Waiting for lock levelling start",
                              "Waiting for lock levelling stop",
                              "Waiting for lock gate opening start",
                              "Waiting for lock gate opening stop",
                              "Waiting for other vessels to leave lock start",
                              "Waiting for other vessels to leave lock stop",
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

    lock_extend_x = np.array([x_lock_gateA, x_lock_gateA, x_lock_gateB, x_lock_gateB]) - x_correction_indirection
    if method == 'Matplotlib':
        ax.fill(lock_extend_x, [ylimmin, ylimmax, ylimmax, ylimmin], color="lightgrey", zorder=-2)
    elif method == 'Plotly':
        fig.add_shape(type="rect",
                      x0=lock_extend_x[0], x1=lock_extend_x[-1],
                      y0=ylimmin, y1=ylimmax,
                      fillcolor="lightgrey", opacity=0.5,
                      layer="below", line_width=0,
                      name="Lock Geometry")

    # plot the lock phases
    if method == 'Matplotlib':
        add_locking_phases_to_plot(lock_chamber, ax,lock_extend_x,time_axis='y',method=method)
    elif method == 'Plotly':
        add_locking_phases_to_plot(lock_chamber, fig, lock_extend_x, time_axis='y', method=method)

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

def plot_saltwater_intrusion(lock_chamber, ZSF_results):
    fig, axes = plt.subplots(2, 1, figsize=[11,6])

    stacked_salinity_lock = pd.DataFrame({
        "time": pd.concat([ZSF_results["time_start"], ZSF_results["time_stop"]], ignore_index=True),
        "salinity": pd.concat([ZSF_results["salinity_lock_start"], ZSF_results["salinity_lock_stop"]], ignore_index=True)
    }).sort_values("time").reset_index(drop=True)

    ax = axes[0]
    ax.plot(ZSF_results.time_start.values,ZSF_results['salinity_sea'].values, color='lightblue', label='Sea')
    ax.plot(ZSF_results.time_start.values,ZSF_results['salinity_lake'].values, color='C0', label='Lake')
    ax.plot(stacked_salinity_lock.time.values,stacked_salinity_lock.salinity.values,color='k',label='Lock')
    xlim = [stacked_salinity_lock.time.values[1],stacked_salinity_lock.time.values[-2]]
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ylim = ax.get_ylim()
    add_locking_phases_to_plot(lock_chamber, ax, ylim, time_axis='x', method='Matplotlib')
    ax.set_ylabel('Salt\nconcentration\n'+r'[kgm$^{-3}$]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc='upper right',frameon=False, bbox_to_anchor = [1.11,1.0])

    ax = axes[1]
    ax.plot(ZSF_results.time_stop.values,ZSF_results['mass_transport_lake'].cumsum().apply(lambda x: x * -1),color='gold')
    ylim = ax.get_ylim()
    add_locking_phases_to_plot(lock_chamber, ax, ylim, time_axis='x', method='Matplotlib')
    ax.set_ylabel('Salt mass\n[kg]')
    ax.set_xlabel('Time')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y\n%H:%M"))
    plt.xticks(rotation=45);


def spatially_visualize_lock_complex(lock_complex):
    m = folium.Map(tiles="cartodbpositron")

    graph = lock_complex.env.graph
    # Plotting the nearby graph
    for registration_node_1, registration_node_2 in zip(lock_complex.registration_nodes[:-1],
                                                        lock_complex.registration_nodes[1:]):
        route = nx.dijkstra_path(graph, registration_node_1, registration_node_2)
        edges = []
        for edge in zip(route[:-1], route[1:]):
            edges.append(edge)

        for edge in zip(route[:-1], route[1:]):
            visualize_edge_in_folium_plot(m, graph, edge)
            visualize_node_in_folium_plot(m, graph, edge[0], color='darkviolet', size=10)
            visualize_node_in_folium_plot(m, graph, edge[1], color='darkviolet', size=10)

    for name, lock_chamber in lock_complex.lock_chambers.items():
        if lock_chamber.geometry is not None:
            visualize_geometry_polygon_in_folium_plot(m, lock_chamber.geometry)

        visualize_geometry_point_in_folium_plot(m, lock_chamber.gate_A.geometry, color='green', size=10,
                                                label = name + ' - Gate A')
        visualize_geometry_point_in_folium_plot(m, lock_chamber.gate_B.geometry, color='red', size=10 ,
                                                label = name + ' - Gate B')

    for name, waiting_area in lock_complex.waiting_areas.items():
        visualize_geometry_point_in_folium_plot(m, waiting_area.geometry, color='black', size=10, label = name)

    m.fit_bounds(m.get_bounds())

    return m


def show_results(summary: pd.Series):
    def dict_table(d):
        rows = "".join(
            f"""
            <tr>
                <td style="text-align:left;padding-right:20px;">{k}</td>
                <td style="text-align:right;width:80px;">{v:.2f}</td>
            </tr>
            """
            for k, v in d.items()
        )
        return f"<table style='border-collapse:collapse;width:100%'>{rows}</table>"

    rows = ""
    for k, v in summary.items():
        if isinstance(v, dict):
            value = dict_table(v)
        else:
            value = f"<span style='float:right'>{v}</span>"

        rows += f"""
        <tr>
            <th style="text-align:left;padding-right:30px;vertical-align:middle">{k}</th>
            <td style="width:300px">{value}</td>
        </tr>
        """

    html = f"""
    <table style="border-collapse:collapse;width:500px">
    {rows}
    </table>
    """

    display(HTML(html))