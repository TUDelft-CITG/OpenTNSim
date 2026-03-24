import copy
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import pytz
import folium
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull

from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from opentnsim.environment.utils import get_governing_current_velocity
from opentnsim.graph.utils import get_sailing_time
from opentnsim.port.calculations import calculate_interpolated_depth_values
from opentnsim.port.utils import create_logbook_with_directed_distances


def lighten_color(color, alpha):
    """
    Lighten a Matplotlib color by blending it with white.
    alpha: 0 (no change) to 1 (white)
    """
    # Convert color to RGB
    rgb = np.array(mcolors.to_rgb(color))
    r, g, b = rgb
    r_new, g_new, b_new = (1 - (1 - r) * alpha, 1 - (1 - g) * alpha, 1 - (1 - b) * alpha,)
    return r_new, g_new, b_new


def merge_figures(fig1, fig2):
    new_fig, new_ax = plt.subplots()

    for fig in (fig1, fig2):
        for ax in fig.axes:
            for line in ax.get_lines():
                new_ax.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    linewidth=line.get_linewidth(),
                    label=line.get_label())

    new_ax.legend()
    return new_fig, new_ax


def plot_vessels_over_route(env, node_start, node_stop, vessels, ddistance=1000,
                            xmin=None, xmax = None, ymin = None, ymax = None, zmin=0, zmax = 15, dz = 1, levels = []):
    interpolated_distance, node_times_num, interpolated_depth = calculate_interpolated_depth_values(env, node_start, node_stop, ddistance)
    route = nx.dijkstra_path(env.graph,node_start,node_stop)
    fig, ax = plt.subplots()
    plt.close()
    if vessels is None:
        vessels = env.vessels

    for idx, vessel in enumerate(vessels):
        fig_vessel = vessel.plot_time_distance_diagram(route)
        if not idx:
            ymin = fig_vessel.axes[0].get_ylim()[0]
        ymax = fig_vessel.axes[0].get_ylim()[-1]
        fig, ax = merge_figures(fig,fig_vessel)
        ax = fig.axes[0]
        ylims = [ymin,ymax]
        plt.close()

    handles, labels = ax.get_legend_handles_labels()

    pcm = ax.pcolormesh(
        interpolated_distance,
        node_times_num,
        interpolated_depth,
        shading='nearest',  # no vertical interpolation
        cmap='Blues',
        norm=mpl.colors.Normalize(zmin,zmax),
        zorder=-2,
        alpha=0.5
    )

    if not levels:
        levels = np.arange(zmin, zmax + dz, dz)

    cs = ax.contour(
        interpolated_distance,
        node_times_num,
        interpolated_depth,
        levels = levels,
        colors='k',
        linewidths=0.5,
        zorder=-1
    )

    ax.clabel(cs, inline=True, fontsize=8,zorder=0)

    fig.colorbar(pcm, label='Available water depth [m]')
    plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.legend(handles, labels)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.27, 0.925),
        frameon = False,
        borderaxespad=0)

    if xmin is None:
        xmin = np.min(interpolated_distance)
    if xmax is None:
        xmax = np.max(interpolated_distance)
    if ymin is None:
        ymin = np.min(ylims)
    if ymax is None:
        ymax = np.max(ylims)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.close()
    return fig


def plot_time_distance_diagram(vessel, route):
    df = create_logbook_with_directed_distances(vessel, route)
    fig, ax  = plt.subplots()
    ax.plot(df.Value, df.Timestamp, label=vessel.name, linewidth=2, zorder=1)
    ax.set_ylim(df.Timestamp.min(),df.Timestamp.max()+pd.Timedelta(hours=1))
    plt.close()
    return fig


def plot_berths(berths, m = None, longitude = 0, latitude = 0, zoom_start = 5):
    if not m:
        m = folium.Map(location=[latitude, longitude], zoom_start=zoom_start, tiles="cartodbpositron")
    style = {'fillColor': '#000000', 'color': '#000000', 'opacity': 0}
    tooltip_berths = folium.GeoJsonTooltip(fields=["Port", "Company", "Berth"],
                                           aliases=["City:", "Port/Company:", "Berth:"],
                                           localize=True,
                                           sticky=False,
                                           labels=True,
                                           style="""
                                        background-color: #F0EFEF;
                                        border: 2px solid black;
                                        border-radius: 3px;
                                        box-shadow: 3px;
                                    """,
                                           max_width=800, )
    popup_berths = folium.GeoJsonPopup(
        fields=["Port", "Company", "Berth", "Berth_type", "Cargo_type", "Maximum_vessel_length",
                "Maximum_vessel_draught"],
        aliases=["City:", "Port/Company:", "Berth:", "Berth type:", "Cargo type:", "Maximum vessel length:",
                 "Maximum vessel draught"],
        localize=True,
        labels=True,
        style="""
                                        background-color: #F0EFEF;
                                        border: 2px solid black;
                                        border-radius: 3px;
                                        box-shadow: 3px;
                                    """,
        max_width=800, )
    folium.GeoJson(berths.set_crs('EPSG:4326'), style_function=lambda x: style, tooltip=tooltip_berths,
                   popup=popup_berths).add_to(m)


def plot_anchorage_areas(anchorage_areas, m = None, longitude = 0, latitude = 0, zoom_start = 5):
    if not m:
        m = folium.Map(location=[latitude, longitude], zoom_start=zoom_start, tiles="cartodbpositron")
    style_anchorage = {'fillColor': '#4CFF00', 'color': '#4CFF00', 'fillOpacity': 0, 'dashArray': '4'}
    tooltip_anchorage = folium.GeoJsonTooltip(fields=["Name", "Port"],
                                              aliases=["Name:", "Port:"],
                                              localize=True,
                                              sticky=False,
                                              labels=True,
                                              style="""
                                        background-color: #F0EFEF;
                                        border: 2px solid black;
                                        border-radius: 3px;
                                        box-shadow: 3px;
                                    """,
                                              max_width=800, )
    popup_anchorage = folium.GeoJsonPopup(fields=["Name", "Port", "Detail"],
                                          aliases=["Name:", "Port:", "Detail:"],
                                          localize=True,
                                          labels=True,
                                          style="""
                                        background-color: #F0EFEF;
                                        border: 2px solid black;
                                        border-radius: 3px;
                                        box-shadow: 3px;
                                    """,
                                          max_width=800, )
    folium.GeoJson(anchorage_areas.set_crs('EPSG:4326'), style_function=lambda x: style_anchorage,
                   tooltip=tooltip_anchorage, popup=popup_anchorage).add_to(m)


def plot_turning_basins(turning_basins, m = None, longitude = 0, latitude = 0, zoom_start = 5):
    if not m:
        m = folium.Map(location=[latitude, longitude], zoom_start=zoom_start, tiles="cartodbpositron")

    style_turning_basin = {'fillColor': '#FF0000', 'color': '#FF0000', 'fillOpacity':0,'dashArray':'4'}
    tooltip_turning_basin = folium.GeoJsonTooltip(fields=["Name", "Port"],
                                    aliases=["Name:","Port:"],
                                    localize=True,
                                    sticky=False,
                                    labels=True,
                                    style="""
                                        background-color: #F0EFEF;
                                        border: 2px solid black;
                                        border-radius: 3px;
                                        box-shadow: 3px;
                                    """,
                                    max_width=800,)
    popup_turning_basin = folium.GeoJsonPopup(fields=["Name", "Port", "Diameter"],
                                aliases=["Name:","Port:","Diameter:"],
                                localize=True,
                                labels=True,
                                style="""
                                        background-color: #F0EFEF;
                                        border: 2px solid black;
                                        border-radius: 3px;
                                        box-shadow: 3px;
                                    """,
                                max_width=800,)

    folium.GeoJson(turning_basins.set_crs('EPSG:4326'),style_function=lambda x:style_turning_basin,
                   tooltip=tooltip_turning_basin,popup=popup_turning_basin).add_to(m)
    
    
def create_plot_vertical_tidal_window(vessel, trip_index = 0, plot = False):
    tidal_window_calculation_results = vessel.tidal_window_calculations[trip_index]
    time_start_index = tidal_window_calculation_results['time_start_index']
    time_end_index = tidal_window_calculation_results['time_end_index']
    route = tidal_window_calculation_results['route']
    bound = tidal_window_calculation_results['bound']
    draught = tidal_window_calculation_results['draught']
    vertical_tidal_windows = tidal_window_calculation_results['vertical_tidal_windows']
    net_ukcs = tidal_window_calculation_results['net_ukcs']
    hydromanager = HydrodynamicDataManager()
    hydrodynamic_information = hydromanager.hydrodynamic_data

    fig, ax = plt.subplots(figsize=[16 * 2 / 3, 6])
    ax.set_facecolor('none')

    # Plot net UKC
    (net_UKC,) = ax.plot(net_ukcs["min_net_ukc"], color="C0", linewidth=2, zorder=2)
    minimum_required_net_ukc = ax.axhline(0, color="C0", linestyle="--", linewidth=2)

    for node in route:
        ax.plot(net_ukcs[node], color='grey', zorder=1)

    if not net_ukcs["min_net_ukc"].empty:
        ax.set_ylim(np.min([np.floor(np.min(net_ukcs["min_net_ukc"].to_numpy())), -1.0]),
                         np.max([np.ceil(np.max(net_ukcs["min_net_ukc"])), 1.0]),)

    vertical_tidal_window_polygons = []
    for window in vertical_tidal_windows:
        vertical_tidal_window_polygon = Polygon(
            [Point((window[0] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[0]),
             Point((window[0] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[1]),
             Point((window[1] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[1]),
             Point((window[1] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[0]), ]
        )
        vertical_tidal_window_polygons.append(vertical_tidal_window_polygon)

    # Plot vertical tidal windows
    vertical_tidal_window = None
    for polygon in vertical_tidal_window_polygons:
        polygon_x = []
        for timestamp in polygon.exterior.xy[0]:
            polygon_x.append(pd.Timestamp(datetime.datetime.fromtimestamp(timestamp, tz=pytz.utc)))
        polygon_y = list(polygon.exterior.xy[1])
        color = lighten_color("C0", alpha=0.25)
        (vertical_tidal_window,) = ax.fill(polygon_x, polygon_y,
                                           facecolor=color, edgecolor="none", zorder=-1)

    # Figure bounds
    ax.set_xlim(hydrodynamic_information.TIME.values[time_start_index],
                hydrodynamic_information.TIME.values[time_end_index - 36], )

    # Figure ticks
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    handles = [net_UKC, minimum_required_net_ukc, vertical_tidal_window]
    labels = ["Net UKC", "Minimum required net UKC", "Vertical tidal windows"]
    legend_handles = []
    legend_labels = []
    for handle, label in zip(handles, labels):
        if handle is not None:
            legend_handles.append(handle)
            legend_labels.append(label)

    ax.set_xlabel("Start time of trip")
    ax.set_ylabel("Minimum net UKC experienced over entire vessel route [m]")

    if plot:
        ax.legend(legend_handles, legend_labels, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.set_title(
            f"Vertical tidal windows of {vessel.type}-class vessel '{vessel.name}' with "
            f"a draught of {np.round(draught, 2)}m and\na length of {np.round(vessel.L)}m sailing {bound} from"
            f" node '{route[0]}' to node '{route[-1]}'."
        )
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    else:
        plt.close(fig)
    return fig, legend_handles, legend_labels


def create_plot_horizontal_tidal_window(vessel, trip_index = 0, plot = False):
    tidal_window_calculation_results = vessel.tidal_window_calculations[trip_index]
    time_start_index = tidal_window_calculation_results['time_start_index']
    time_end_index = tidal_window_calculation_results['time_end_index']
    route = tidal_window_calculation_results['route']
    bound = tidal_window_calculation_results['bound']
    draught = tidal_window_calculation_results['draught']
    horizontal_tidal_windows = tidal_window_calculation_results['horizontal_tidal_windows']
    horizontal_tidal_restriction_nodes = tidal_window_calculation_results['horizontal_tidal_restriction_nodes']
    horizontal_tidal_restriction_stations = tidal_window_calculation_results['horizontal_tidal_restriction_stations']
    hydromanager = HydrodynamicDataManager()
    hydrodynamic_information = hydromanager.hydrodynamic_data

    # Create figure
    fig, ax = plt.subplots(figsize=[16 * 2 / 3, 6])
    ax.set_facecolor('none')

    # Plot vertical tidal windows
    horizontal_tidal_window = None
    for window in horizontal_tidal_windows:
        (horizontal_tidal_window,) = ax.fill(
            [window[0], window[0], window[1], window[1]],
            [-1.5, 1.5, 1.5, -1.5],
            facecolor="firebrick",
            alpha=0.25,
            edgecolor="none",
        )

    # Plot governing current velocity
    current_velocity = None
    for node, station in zip(horizontal_tidal_restriction_nodes, horizontal_tidal_restriction_stations):
        governing_current_velocity, _ = get_governing_current_velocity(vessel,station,time_start_index,time_end_index)
        sailing_time, _ = get_sailing_time(vessel, route[: (route.index(node) + 1)]) + delay
        horizontal_tidal_accessibility_time_correction = np.timedelta64(int(sailing_time), "s")
        horizontal_tidal_accessibility_time = hydrodynamic_information.TIME.values[time_start_index:time_end_index]
        horizontal_tidal_accessibility_time -= horizontal_tidal_accessibility_time_correction
        (current_velocity,) = ax.plot(horizontal_tidal_accessibility_time, governing_current_velocity,
                                      color="firebrick", linewidth=3,)
    ax.axhline(0, color="k", linewidth=.5)

    # Calculate vertical and horizontal tidal windows
    horizontal_tidal_window_polygons = []
    for window in horizontal_tidal_windows:
        horizontal_tidal_window_polygon = Polygon(
            [Point((window[0] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[0]),
             Point((window[0] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[1]),
             Point((window[1] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[1]),
             Point((window[1] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[0]), ]
        )
        horizontal_tidal_window_polygons.append(horizontal_tidal_window_polygon)

    # Plot horizontal tidal windows
    for polygon in horizontal_tidal_window_polygons:
        polygon_x = []
        for timestamp in polygon.exterior.xy[0]:
            polygon_x.append(pd.Timestamp(datetime.datetime.fromtimestamp(timestamp, tz=pytz.utc)))
        polygon_y = list(polygon.exterior.xy[1])
        color = lighten_color("firebrick", alpha=0.25)
        (horizontal_tidal_window,) = ax.fill(polygon_x, polygon_y,
                                             facecolor=color, edgecolor="none",
                                             zorder=-1)

    # Figure bounds
    ax.set_xlim(hydrodynamic_information.TIME.values[time_start_index],
                hydrodynamic_information.TIME.values[time_end_index - 36])
    ax.set_ylim(-1.5, 1.5)

    # Figure ticks
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    # Figure axes
    ax.set_xlabel("Date")
    ax.set_ylabel("Current velocity [m/s]")

    # Legend
    handles = [current_velocity, horizontal_tidal_window]
    labels = ["Current velocity", "Horizontal tidal windows"]
    legend_handles = []
    legend_labels = []
    for handle, label in zip(handles, labels):
        if handle is not None:
            legend_handles.append(handle)
            legend_labels.append(label)
    #
    # Figure bounds
    ax.set_xlim(hydrodynamic_information.TIME.values[time_start_index],
                hydrodynamic_information.TIME.values[time_end_index - 36], )

    if plot:
        ax.legend(legend_handles, legend_labels, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.set_title(
            f"Horizontal tidal windows of {vessel.type}-class vessel '{vessel.name}' with "
            f"a draught of {np.round(draught, 2)}m and\na length of {np.round(vessel.L)}m sailing {bound} from"
            f" node '{route[0]}' to node '{route[-1]}'."
        )
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    else:
        plt.close(fig)
    return fig, legend_handles, legend_labels


def plot_tidal_windows(vessel, trip_index = 0, plot_all = False):
    tidal_window_calculation_results = vessel.tidal_window_calculations[trip_index]
    route = tidal_window_calculation_results['route']
    bound = tidal_window_calculation_results['bound']
    draught = tidal_window_calculation_results['draught']
    tidal_windows = tidal_window_calculation_results['tidal_windows']

    # Create figure
    fig_final, ax_left = plt.subplots(figsize=[16 * 2 / 3, 6])
    plt.close()
    ax_left.set_facecolor('none')
    ax_right = ax_left.twinx()
    ax_left.set_facecolor('none')

    # Plot governing current velocity
    fig_left, ax_left_handles, ax_left_labels = vessel.create_plot_vertical_tidal_window(trip_index, plot_all)
    fig_right, ax_right_handles, ax_right_labels = vessel.create_plot_horizontal_tidal_window(trip_index, plot_all)

    for fig_target, ax_target in zip([fig_left, fig_right], [ax_left, ax_right]):
        for ax_old in fig_target.get_axes():
            for line in ax_old.get_lines():
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if len(xdata) == 2:
                    ax_target.axhline(
                        ydata[0],
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        marker=line.get_marker(),
                        alpha=line.get_alpha(),
                        label=line.get_label(),
                        zorder=line.get_zorder()
                    )

                else:
                    ax_target.plot(
                        xdata,
                        ydata,
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        marker=line.get_marker(),
                        alpha=line.get_alpha(),
                        label=line.get_label(),
                        zorder=line.get_zorder()
                    )

            # Copy limits
            ax_target.set_xlim(ax_old.get_xlim())
            ax_target.set_ylim(ax_old.get_ylim())

            # Copy labels and title
            ax_target.set_xlabel(ax_old.get_xlabel(), rotation=ax_old.xaxis.label.get_rotation())
            ax_target.set_ylabel(ax_old.get_ylabel(), rotation=ax_old.yaxis.label.get_rotation())

            # Copy tick positions and rotation
            ax_target.set_xticks(ax_old.get_xticks())
            ax_target.set_yticks(ax_old.get_yticks())
            for old_tick, new_tick in zip(ax_old.get_xticklabels(), ax_target.get_xticklabels()):
                new_tick.set_rotation(old_tick.get_rotation())
                new_tick.set_ha(old_tick.get_ha())
                new_tick.set_va(old_tick.get_va())

            # Copy x-axis formatter/locator (important for dates)
            ax_target.xaxis.set_major_formatter(ax_old.xaxis.get_major_formatter())
            ax_target.xaxis.set_major_locator(ax_old.xaxis.get_major_locator())
            ax_target.xaxis.set_minor_formatter(ax_old.xaxis.get_minor_formatter())
            ax_target.xaxis.set_minor_locator(ax_old.xaxis.get_minor_locator())

    ax_right.yaxis.set_label_position("right")

    # Calculate tidal windows
    tidal_window_polygons = []
    for window in tidal_windows:
        tidal_window_polygon = Polygon(
                [Point((window[0] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[0]),
                 Point((window[0] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[1]),
                 Point((window[1] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[1]),
                 Point((window[1] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[0]),]
            )
        tidal_window_polygons.append(tidal_window_polygon)

    # Plot tidal windows
    tidal_window = None
    window_y = [ax_left.get_ylim()[0], ax_left.get_ylim()[1], ax_left.get_ylim()[1], ax_left.get_ylim()[0]]
    color = lighten_color("limegreen", alpha=0.4)
    for window in tidal_windows:
        window_x = [window[0], window[0], window[1], window[1]]
        (tidal_window,) = ax_left.fill(window_x, window_y,
                                       facecolor=color, edgecolor="none", zorder=-5,)

    # Figure axes
    no_window_x = [0, 0, 0, 0]
    no_window_y = [ax_left.get_ylim()[0], ax_left.get_ylim()[1], ax_left.get_ylim()[1], ax_left.get_ylim()[0]]
    (no_tidal_window,) = ax_left.fill(no_window_x, no_window_y,
                                      facecolor='white', edgecolor="none", zorder=-1,)




    handles = np.append(ax_left_handles, ax_right_handles)
    labels = np.append(ax_left_labels, ax_right_labels)
    legend_handles = []
    legend_labels = []
    for handle, label in zip(handles,labels):
        if isinstance(handle,mpatches.Polygon):
            continue
        legend_handles.append(handle)
        legend_labels.append(label)

    handles = []
    labels = []
    for handle, label in zip([tidal_window, no_tidal_window],["Tidal windows", "No tidal window"]):
        if handle is not None:
            handles.append(handle)
            labels.append(label)

    legend_handles = np.append(legend_handles, handles)
    legend_labels = np.append(legend_labels, labels)

    ax_left.legend(legend_handles, legend_labels, frameon=False, loc="upper left", bbox_to_anchor=(1.05, 1.0),)
    ax_left.set_title(
        f"Accessibility of {vessel.type}-class vessel '{vessel.name}' with "
        f"a draught of {np.round(draught, 2)}m and\na length of {np.round(vessel.L)}m sailing {bound} from"
        f" node '{route[0]}' to node '{route[-1]}'."
    )
    return fig_final


def plot_berth_planning(berth):
    from opentnsim.port.mixins.berth import IsQuay, IsJetty

    fig, ax = plt.subplots()
    historic_quay_planning_plot = berth.historic_berth_planning.stack().reset_index()
    historic_quay_planning_plot.columns = ['timestamp', 'column', 'id']
    historic_quay_planning_plot = historic_quay_planning_plot[historic_quay_planning_plot['id'].notna()]
    historic_quay_planning_plot_id_mapping = {id_: list(zip(sub_df['timestamp'], sub_df['column'])) for id_, sub_df
                                              in historic_quay_planning_plot.groupby('id', sort=False)}
    for vessel_id in historic_quay_planning_plot_id_mapping.keys():
        vessel_occupancy = historic_quay_planning_plot_id_mapping[vessel_id]
        timestamps, quay_position = zip(*vessel_occupancy)
        quay_position = list(quay_position)
        quay_position = np.array(quay_position, dtype=float)
        timestamps = list(timestamps)
        timestamps = mdates.date2num(timestamps)
        quay_position_over_time = np.column_stack((quay_position, timestamps))
        try:
            quay_position_over_time_polygons = ConvexHull(quay_position_over_time)
            ax.fill(quay_position_over_time[quay_position_over_time_polygons.vertices, 0],
                    quay_position_over_time[quay_position_over_time_polygons.vertices, 1], )
        except:
            continue

    plt.gca().yaxis_date()
    plt.xlim(0,berth.berth_length)
    plt.ylabel("Time")
    if isinstance(berth, IsQuay):
        plt.xlabel("Quay length [m]")
    elif isinstance(berth, IsJetty):
        plt.xlabel("Berth length [m]")
    plt.close()
    return fig