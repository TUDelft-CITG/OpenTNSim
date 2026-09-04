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
from opentnsim.environment.utils import get_governing_current_velocity, get_nearest_time_index
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
    

def plot_vertical_tidal_window( 
        vessel,
        vertical_tidal_windows, 
        net_ukcs,
        route, 
        time_start, 
        time_end, 
        draught, 
        bound,
        plot = True):
    
    hydromanager = HydrodynamicDataManager()
    hydrodynamic_information = hydromanager.hydrodynamic_data

    times = hydrodynamic_information.TIME.values
    time_start_index = time_start
    if not isinstance(time_start, int):
        time_start_index = get_nearest_time_index(times, time_start)
    time_end_index = time_end
    if not isinstance(time_end, int):
        time_end_index = get_nearest_time_index(times, time_end)

    fig, ax = plt.subplots(figsize=[16 * 2 / 3, 6])
    ax.set_facecolor('none')

    # Plot net UKC
    (net_UKC,) = ax.plot(net_ukcs["min_net_ukc"], color="C0", linewidth=2, zorder=2)
    minimum_required_net_ukc = ax.axhline(0, color="C0", linestyle="--", linewidth=2)

    for node in route:
        if node not in net_ukcs.columns:
            continue
        ax.plot(net_ukcs[node], color='grey', zorder=1)

    if not net_ukcs["min_net_ukc"].empty:
        ax.set_ylim(np.min([np.floor(np.min(net_ukcs["min_net_ukc"].to_numpy())), -1.0]),
                         np.max([np.ceil(np.max(net_ukcs["min_net_ukc"])), 1.0]),)

    vertical_tidal_window_polygons = []
    for window in vertical_tidal_windows:
        vertical_tidal_window_polygon = Polygon(
            [Point((window[0] - pd.Timestamp("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[0]),
             Point((window[0] - pd.Timestamp("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[1]),
             Point((window[1] - pd.Timestamp("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[1]),
             Point((window[1] - pd.Timestamp("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[0]), ]
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
    labels = ["Margin to minimum required water depth", "Minimum required water depth margin", "Vertical tidal windows"]
    legend_handles = []
    legend_labels = []
    for handle, label in zip(handles, labels):
        if handle is not None:
            legend_handles.append(handle)
            legend_labels.append(label)

    ax.set_xlabel("Start time of trip")
    ax.set_ylabel("Margin to minimum required water depth [m]")

    if plot:
        ax.legend(legend_handles, legend_labels, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        # ax.set_title(
        #     f"Vertical tidal windows of {vessel.type}-class vessel '{vessel.name}' with "
        #     f"a draught of {np.round(draught, 2)}m and\na length of {np.round(vessel.L)}m sailing {bound} from"
        #     f" node '{route[0]}' to node '{route[-1]}'."
        # )
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    else:
        plt.close(fig)
    return fig, legend_handles, legend_labels
    
    
def create_plot_vertical_tidal_window(vessel, trip_index = 0, plot = False):
    try:
        tidal_window_calculation_results = vessel.tidal_window_calculations[trip_index]
    except:
        return None, None, None
    
    for waterway, results in tidal_window_calculation_results.iterrows():
        time_start_index = results['time_start_index']
        time_end_index = results['time_end_index']
        route = results['route']
        bound = results['bound']
        draught = results['draught']
        vertical_tidal_windows = results['vertical_tidal_windows']
        net_ukcs = results['net_ukcs']
        fig, legend_handles, legend_labels = plot_vertical_tidal_window(
            vessel,
            vertical_tidal_windows,
            net_ukcs, 
            route, 
            int(time_start_index), 
            int(time_end_index), 
            draught, 
            bound,
            plot)
    return fig, legend_handles, legend_labels


def create_plot_horizontal_tidal_window(vessel, trip_index = 0, delay = 0.,plot = False):
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
        try:
            (horizontal_tidal_window,) = ax.fill(
                [window[0], window[0], window[1], window[1]],
                [-1.5, 1.5, 1.5, -1.5],
                facecolor="firebrick",
                alpha=0.25,
                edgecolor="none",
            )
        except:
            pass

    # Plot governing current velocity
    current_velocity = None
    for node, station in zip(horizontal_tidal_restriction_nodes, horizontal_tidal_restriction_stations):
        try:
            governing_current_velocity, _ = get_governing_current_velocity(vessel,station,time_start_index,time_end_index)
            sailing_time, _ = get_sailing_time(vessel, route[: (route.index(node) + 1)]) + delay
            horizontal_tidal_accessibility_time_correction = np.timedelta64(int(sailing_time), "s")
            horizontal_tidal_accessibility_time = hydrodynamic_information.TIME.values[time_start_index:time_end_index]
            horizontal_tidal_accessibility_time -= horizontal_tidal_accessibility_time_correction
            (current_velocity,) = ax.plot(horizontal_tidal_accessibility_time, governing_current_velocity,
                                        color="firebrick", linewidth=3,)
        except:
            pass
    ax.axhline(0, color="k", linewidth=.5)

    # Calculate vertical and horizontal tidal windows
    horizontal_tidal_window_polygons = []
    for window in horizontal_tidal_windows:
        try:
            horizontal_tidal_window_polygon = Polygon(
                [Point((window[0] - pd.Timestamp("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[0]),
                Point((window[0] - pd.Timestamp("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[1]),
                Point((window[1] - pd.Timestamp("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[1]),
                Point((window[1] - pd.Timestamp("1970-01-01")) / np.timedelta64(1, "s"), ax.get_ylim()[0]), ]
            )
            horizontal_tidal_window_polygons.append(horizontal_tidal_window_polygon)
        except:
            pass

    # Plot horizontal tidal windows
    for polygon in horizontal_tidal_window_polygons:
        try:
            polygon_x = []
            for timestamp in polygon.exterior.xy[0]:
                polygon_x.append(pd.Timestamp(datetime.datetime.fromtimestamp(timestamp, tz=pytz.utc)))
            polygon_y = list(polygon.exterior.xy[1])
            color = lighten_color("firebrick", alpha=0.25)
            (horizontal_tidal_window,) = ax.fill(polygon_x, polygon_y,
                                                facecolor=color, edgecolor="none",
                                                zorder=-1)
        except:
            pass

    # Figure bounds
    try:
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
    except:
        pass

    # Legend
    handles = [current_velocity, horizontal_tidal_window]
    labels = ["Current velocity", "Horizontal tidal windows"]
    legend_handles = []
    legend_labels = []
    for handle, label in zip(handles, labels):
        if handle is not None:
            legend_handles.append(handle)
            legend_labels.append(label)
            
    # Figure bounds
    try:
        ax.set_xlim(hydrodynamic_information.TIME.values[time_start_index],
                    hydrodynamic_information.TIME.values[time_end_index - 36], )
    except:
        pass

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


def plot_tidal_windows(vessel, canal = None, trip_index = 0, plot_all = False):
    if canal is not None:
        tidal_window_calculation_results = vessel.tidal_window_calculations[trip_index].loc[canal]
    else:
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
                [Point((window[0] - pd.Timestamp("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[0]),
                 Point((window[0] - pd.Timestamp("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[1]),
                 Point((window[1] - pd.Timestamp("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[1]),
                 Point((window[1] - pd.Timestamp("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[0]),]
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



import plotly.graph_objects as go
import numpy as np

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import colors as mcolors
from collections import defaultdict
from itertools import combinations

def get_vessels_by_name(env, vessel_mmsis):
    return [env.vessels[name] for name in vessel_mmsis if name in env.vessels]

def add_point_tidal_window_to_plotly(
    fig,
    waterway,
    hydrodynamic_data,
    determinantes,
    draught=10.5,
    draught_range=None,
    offset=200,
    start_distance=0,
    hydro_property="water depth",
    time_start=None,
    time_end=None,
    colorscale=None,
    showscale=True,
):
    """
    Add tidal window bands to an existing Plotly figure.

    Parameters
    ----------
    fig : go.Figure
    waterway : Waterway
    hydrodynamic_data : xr.Dataset
    determinantes : dict
        {(node_start, node_stop): distance_along_edge}
    """
    if draught_range is None:
        draught_range = [draught]

    if time_start is None:
        time_start = pd.Timestamp(hydrodynamic_data.TIME.min().values)

    if time_end is None:
        time_end = pd.Timestamp(hydrodynamic_data.TIME.max().values)

    edge_route = list(
        zip(
            waterway.edge_distances["node_start"],
            waterway.edge_distances["node_stop"],
        )
    )

    # --------------------------------------------------------------
    # Determine common color normalization
    # --------------------------------------------------------------

    if hydro_property == "water depth":

        global_min = np.inf
        global_max = -np.inf

        for edge in determinantes:

            if edge not in edge_route:
                continue

            ds = hydrodynamic_data.sel(
                TIME=slice(time_start, time_end),
                STATION=edge,
            )

            wd = (
                ds["Water level"]
                - ds["Nautical depth"]
            ).values

            global_min = min(global_min, np.nanmin(wd))
            global_max = max(global_max, np.nanmax(wd))

        zmin = global_min
        zmax = global_max
        colorscale = colorscale or "Blues"

    elif hydro_property == "Margen neto de seguridad mínimo bajo la quilla [m]":

        zmin = -0.5
        zmax = 0.5
        colorscale = colorscale or "RdBu"

    elif hydro_property == "Margen bruto de seguridad mínimo bajo la quilla [m]":

        zmin = -1
        zmax = 1
        colorscale = colorscale or "RdBu"

    else:

        zmin = None
        zmax = None
        colorscale = colorscale or "Blues"

    xticks = []
    xticklabels = []

    # --------------------------------------------------------------
    # Draw each determining point
    # --------------------------------------------------------------
    trace_groups = []

    for current_draught in draught_range:
    
        current_group = []
    
        for edge, distance in determinantes.items():

            if edge not in edge_route:
                continue
    
            distance_info = waterway.edge_distances[
                (waterway.edge_distances["node_start"] == edge[0])
                & (waterway.edge_distances["node_stop"] == edge[1])
            ].iloc[0]
    
            xmid = (
                distance_info["distance_start"]
                + distance
                + start_distance
            )
    
            x0 = xmid - offset
            x1 = xmid + offset
    
            xticks.append(xmid)
            xticklabels.append(edge)
    
            required_depth = estimate_required_water_depth(
                waterway.env,
                edge,
                current_draught,
            )
    
            ds = hydrodynamic_data.sel(
                TIME=slice(time_start, time_end),
                STATION=edge,
            )
    
            wl = ds["Water level"].values
            wd = wl - ds["Nautical depth"].values
    
            if hydro_property == "Margen neto de seguridad mínimo bajo la quilla [m]":
                values = wd - required_depth
    
            elif hydro_property == "Margen bruto de seguridad mínimo bajo la quilla [m]":
                values = wd - current_draught
    
            else:
                values = wd
    
            times = pd.to_datetime(ds.TIME.values)
    
            # Two identical columns to create a vertical strip
            z = np.column_stack([values, values])
    
            fig.add_trace(
                go.Heatmap(
                    x=[x0, x1],
                    y=times,
                    z=z,
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    visible=(current_draught == draught),
                    hovertemplate=(
                        "Distance: %{x:.0f} m<br>"
                        "Time: %{y}<br>"
                        "Value: %{z:.2f}<extra></extra>"
                    ),
                    showscale=(
                        True 
                        if (current_draught == draught_range[0]
                            and len(current_group) == 0
                            and showscale)
                        else False
                    ),
                    colorbar=dict(
                        title='',
                        thickness=8,
                        len=0.75,
                        y=0.5,
                        yanchor="middle",
                        x=1.45,          # just left of the legend
                        xanchor="left",
                    ),
                )
            )
            current_group.append(len(fig.data)-1)
            
            # Only show one colorbar
            showscale = False
        trace_groups.append(current_group)

    fig.add_annotation(
        x=1.45,
        y=0.5,
        xref="paper",
        yref="paper",
        text=hydro_property,
        textangle=-90,
        showarrow=False,
        xanchor="right",
        yanchor="middle",
        font=dict(size=14),
    )

    return {
        "fig": fig,
        "xticks": xticks,
        "xticklabels": xticklabels,
        "draught_groups": trace_groups,
        "all_heatmap_traces": [
            idx 
            for group in trace_groups
            for idx in group
        ],
    }

# ADDED algorithm
from opentnsim.port.utils import get_vessel_direction_with_waterway
from opentnsim.port.calculations import estimate_required_water_depth
import matplotlib.colors as mcolors
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from itertools import combinations
from matplotlib.lines import Line2D
import textwrap


def get_vessel_logbook_through_waterway(vessel, waterway, extra_nodes = None):
    vessel_df = pd.DataFrame(vessel.logbook)
    if vessel_df.empty:
        return vessel_df
    vessel_df[["node_start", "node_stop"]] = (
        vessel_df["Message"]
        .str.extract(r"from node (.*?) to node (.*)")
        .apply(lambda col: col.str.replace(r"\s+(start|stop)$", "", regex=True))
    )

    n = len(vessel_df)
    keep = pd.Series(False, index=vessel_df.index)

    waterway_route = copy(waterway.route)
    if extra_nodes is not None:
        waterway_route.extend(extra_nodes)
    
    is_waiting = vessel_df["Message"].str.contains("Waiting for", na=False)
    valid_sailing = (
        vessel_df["node_start"].isin(waterway_route)
        & vessel_df["node_stop"].isin(waterway_route)
    )
    keep = keep | valid_sailing
    first_valid_sailing_idx = vessel_df.index[valid_sailing].min()

    pre_sailing = vessel_df.index < first_valid_sailing_idx
    waiting_pre = vessel_df[pre_sailing & is_waiting].copy()

    gap = waiting_pre.index.to_series().diff().fillna(1) != 1
    group = gap.cumsum()
    last_group = group.max()
    last_block_idx = waiting_pre.index[group == last_group]
    try:
        if last_block_idx[-1]+1 == first_valid_sailing_idx:
            keep.loc[last_block_idx] = True
    except:
        pass

    vessel_df_waterway = vessel_df[keep]
    if vessel_df_waterway.empty:
        return vessel_df_waterway
    vessel_df_waterway = vessel_df_waterway.sort_values('Timestamp')
    vessel_df_waterway['Value'] -= vessel_df_waterway['Value'].iloc[0]
    direction = get_vessel_direction_with_waterway(waterway.route, vessel.route)
    if direction:
        vessel_df_waterway['Value'] -= vessel_df_waterway['Value'].iloc[-1]
        vessel_df_waterway['Value'] = vessel_df_waterway['Value']*-1
    return vessel_df_waterway

def add_point_tidal_window_to_plot(
    ax,
    waterway,
    determinantes,
    hydrodynamic_data,
    time_start=None,
    time_end=None,
    ylim=None,
    draught=10.5,
    offset=200,
    start_distance = 0,
    hydro_property="water depth",
    zorder = 1,
):

    if time_start is not None and time_end is not None:
        ylim = (time_start, time_end)

    if ylim is None:
        ylim = ax.get_ylim()
    else:
        ax.set_ylim(ylim)
        ylim = ax.get_ylim()

    t0, t1 = mdates.num2date(ylim)
    t0 = pd.Timestamp(t0).tz_localize(None)
    t1 = pd.Timestamp(t1).tz_localize(None)

    edge_route = list(
        zip(
            waterway.edge_distances["node_start"],
            waterway.edge_distances["node_stop"],
        )
    )

    global_min = np.inf
    global_max = -np.inf

    if hydro_property == "water depth":

        for edge, distance in determinantes.items():

            if edge not in edge_route:
                continue

            hydrodynamic_data_sel = hydrodynamic_data.sel(
                TIME=slice(t0, t1),
                STATION=edge
            )

            wl = (
                hydrodynamic_data_sel["Water level"]
                - hydrodynamic_data_sel["Nautical depth"]
            ).values

            global_min = min(global_min, np.nanmin(wl))
            global_max = max(global_max, np.nanmax(wl))
        
        shared_norm = mcolors.Normalize(
            vmin=global_min,
            vmax=global_max
        )

    elif hydro_property == "Margen neto de seguridad mínimo bajo la quilla [m]":

        shared_norm = mcolors.Normalize(-0.5, 0.5)

    elif hydro_property == "Margen bruto de seguridad mínimo bajo la quilla [m]":

        shared_norm = mcolors.Normalize(-1, 1)

    else:
        shared_norm = None

    pcm = None

    xticks = []
    xticklabels = []
    for edge, distance in determinantes.items():
        if edge not in edge_route:
            continue

        distance_info_edge = waterway.edge_distances[
            (waterway.edge_distances["node_start"] == edge[0]) &
            (waterway.edge_distances["node_stop"] == edge[1])
        ].iloc[0]

        distance_determinante = (
            distance_info_edge["distance_start"] + distance + start_distance
        )

        required_depth = estimate_required_water_depth(
            waterway.env,
            edge,
            draught
        )

        x0 = distance_determinante - offset
        x01 = distance_determinante
        x1 = distance_determinante + offset
        xticks.append(x01)
        xticklabels.append(edge)
        
        hydrodynamic_data_sel = hydrodynamic_data.sel(
            TIME=slice(t0, t1),
            STATION=edge
        )

        wl = hydrodynamic_data_sel["Water level"].values
        wd = wl - hydrodynamic_data_sel["Nautical depth"].values

        cmap = "Blues"
        if hydro_property == "Margen neto de seguridad mínimo bajo la quilla [m]":
            wl = wd - required_depth
            cmap = "RdBu"

        elif hydro_property == "Margen bruto de seguridad mínimo bajo la quilla [m]":
            wl = wd - draught
            cmap = "RdBu"

        elif hydro_property == "Profundidad [m]":
            wl = wd

        y = pd.to_datetime(hydrodynamic_data_sel.TIME.values)

        X, Y = np.meshgrid([x0, x1], y)
        C = np.tile(wl[:, None], (1, 2))
        pcm = ax.pcolormesh(
            X,
            Y,
            C,
            shading="auto",
            cmap=cmap,
            norm=shared_norm,
            zorder = zorder,
        )

    return pcm, xticks, xticklabels


def plot_time_distance_diagram(waterway,  time_start, time_end, draught, hydro_property, color_function = None):
    if time_start is not None and time_end is not None:
        ylim = (time_start, time_end)
        
    fig, ax = plt.subplots(figsize=[16,3])    
    

    for _, vessel in waterway.env.vessels.items():
        vessel_df_waterway = get_vessel_logbook_through_waterway(vessel, waterway)
        color = None
        if color_function is not None:
            color = color_function(vessel)
        
        if not vessel_df_waterway.empty:
            if color is not None:
                ax.plot(vessel_df_waterway.Value, vessel_df_waterway.Timestamp, color=color)
            else:
                ax.plot(vessel_df_waterway.Value, vessel_df_waterway.Timestamp)
    for _, node_info in waterway.node_distances.iterrows():
        ax.axvline(node_info.distance, color = 'k', linestyle = '--', linewidth = 1)

    ax.set_xticks(waterway.node_distances.distance.to_list());
    ax.set_xticklabels(waterway.node_distances.node, rotation=45, ha='right');
    ax.set_ylim(ylim)
    xlim = ax.get_xlim()
    offset = (xlim[-1] - xlim[0])/500
    pcm, _, _ = add_point_tidal_window_to_plot(ax, waterway, time_start, time_end, ylim, draught, offset, hydro_property)
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.colorbar(pcm, ax=ax, label=hydro_property)
    return fig

def compute_waterway_offsets(waterways):
    offsets = {}
    current_offset = 0

    for w in waterways:
        max_dist = w.node_distances.distance.max()
        offsets[id(w)] = current_offset
        current_offset += max_dist

    return offsets

def plot_time_distance_diagram_multi(
    waterways,
    hydrodynamic_data,
    determinantes,
    time_start,
    time_end,
    draught,
    hydro_property,
    start_distance,
    offset,
    vessels = None,
    vessels_to_show_rules = None,
    show_rules = False,
    xlim = None,
    extra_nodes = None,
    color_function = None,
    figsize = (12,4)
):

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[100, 1, 10, 5],  # plot | colorbar | legend
        wspace=0.05
    )
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = ax1.twiny()
    ylim = (time_start, time_end)
    
    cax = fig.add_subplot(gs[0, 1])
    spacer = fig.add_subplot(gs[0, 2])  # invisible gap
    lax = fig.add_subplot(gs[0, 3])

    offsets = compute_waterway_offsets(waterways)
    
    pcm = None

    xticks = []
    xticklabels = []
    xticks2 = []
    xticklabels2 = []
    for waterway in waterways:

        x_shift = offsets[id(waterway)] + start_distance

        if vessels is None:
            vessels = list(waterway.env.vessels.values())

        for vessel in vessels:
            color = None
            if color_function is not None:
                color = color_function(vessel)
            vessel_df = get_vessel_logbook_through_waterway(vessel, waterway, extra_nodes)
            if not vessel_df.empty:
                if color is not None:
                    ax1.plot(
                        vessel_df.Value + x_shift,
                        vessel_df.Timestamp,
                        color=color,
                        zorder = 2,
                    )
                else:
                    ax1.plot(
                        vessel_df.Value + x_shift,
                        vessel_df.Timestamp,
                        zorder = 2,
                    )

        for _, node_info in waterway.node_distances.iterrows():
            label = node_info['node'].replace('_','')
            if 'KM' not in label:
                label += f' KM{int((node_info.distance + x_shift)/1000)}.0'
            elif '.' not in label:
                label += '.0'
            if label not in xticklabels:
                xticks.append(node_info.distance + x_shift)
                xticklabels.append(label)
            ax1.axvline(
                node_info.distance + x_shift,
                color="k",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                zorder = 0,
            )

        
        pcm, xtick, xticklabel = add_point_tidal_window_to_plot(
            ax=ax1,
            waterway=waterway,
            hydrodynamic_data=hydrodynamic_data,
            determinantes=determinantes,
            time_start=time_start,
            time_end=time_end,
            ylim=ylim,
            draught=draught,
            offset=offset,
            start_distance = x_shift,
            hydro_property=hydro_property,
            zorder = 1,
        )
        
        xticks2.extend(xtick)
        xticklabels2.extend(xticklabel)

    ax1.set_ylim(ylim)
    ax1.yaxis_date()
    ylim = ax1.get_ylim()
    locator = mdates.HourLocator(byhour=range(0, 24, 3), interval=1)

    handle = None
    if show_rules:
        edge_df = waterway.edge_distances
        vessels_show_rules = get_vessels_by_name(waterway.env, vessels_to_show_rules)
        vessel_pairs = list(combinations(vessels_show_rules, 2))
        restriction_per_edge = {}
    
        for vessel_pair in vessel_pairs:
            for edge in zip(edge_df["node_start"], edge_df["node_stop"]):
    
                restriction, _, _ = waterway.check_for_encountering_conflicts(
                    edge,
                    vessel_pair
                )
    
                # Keep edge restricted if any vessel pair has a conflict
                if restriction:
                    restriction_per_edge[edge] = True

        for _, edge_info in edge_df.iterrows():
            edge = (edge_info.node_start,edge_info.node_stop)
            if edge not in restriction_per_edge.keys():
                continue
            if not restriction_per_edge[edge]:
                continue
            x0 = edge_info.distance_start + x_shift
            x1 = edge_info.distance_stop + x_shift
            handle = ax1.fill([x0, x0, x1, x1],[ylim[0],ylim[1],ylim[1],ylim[0]], zorder = -1, color='none', edgecolor="indianred", hatch="//",label='Zonas de prohibición de\ncruces o adelantamientos')
        
    def custom_fmt(x, pos):
        dt = mdates.num2date(x)
    
        # show date only at midnight
        if dt.hour == 0 and dt.minute == 0:
            return dt.strftime('%d-%b %H:%M')
        else:
            return dt.strftime('%H:%M')
    
    ax1.yaxis.set_major_locator(locator)
    ax1.yaxis.set_major_formatter(FuncFormatter(custom_fmt))

    if pcm is not None:
        wrapped = textwrap.fill(hydro_property, width=30)
        fig.colorbar(pcm, cax=cax, label= "Margen neto de profundidad [m]")

    ax1.set_xticks(xticks);
    #xticklabels = [textwrap.fill(label, 18) for label in xticklabels]
    ax1.set_xticklabels(xticklabels,ha='center', rotation = 90)
    ax2.set_xticks(xticks2)
    xticklabels2 = [
        'Determinante\nCanal Intermedio\nKM83 (10.6 m)',
        'Determinante\nPaso Banco Chico\nKM58.4 (10.8 m)',
        'Determinante\nCanal Punta Indio\nKM155 (10.5 m)']
    #xticklabels2 = [textwrap.fill(label, 18) for label in xticklabels2]
    ax2.set_xticklabels(xticklabels2, rotation = 90, ha='left')
    ax1.set_xlabel('Distancia [KM]')
    ax1.set_ylabel('Fecha y hora')
    if xlim is not None:
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)

    legend_items = {
    'Buque Clase A': '#00AFDD',
    'Buque Clase B': '#E6362A',
    'Buque Clase C': '#0F68AE',
    'Buque con reserva del canal': 'k'
    }
    
    handles = [
        Line2D([0], [0], color=color, lw=2, label=label)
        for label, color in legend_items.items()
    ]

    if handle is not None:
        handles.append(handle[0])
    
    lax.legend(handles=handles, loc='center left', frameon = False)
    lax.axis('off')
    spacer.axis('off')
    fig.subplots_adjust(bottom=0.35)
    return fig, ax1


def plot_time_distance_diagram_multi_plotly(
    waterways,
    draught_values = np.arange(9.5, 11.6, 0.1),
    time_start=None,
    time_end=None,
    hydrodynamic_data = None,
    determinantes = None,
    hydro_property = None,
    vessels=None,
    start_distance=0,
    xlim=None,
    extra_nodes=None,
    height=1000,
    width=900,
    show_rules = False,
    color_function = None,
    vessels_to_show_rules = [],
):

    fig = go.Figure()

    offsets = compute_waterway_offsets(waterways)
    xticks = []
    xticklabels = []
    xmin = None
    xmax = None
    shown_vessels = set()
    trace_to_vessel = []
    vessel_times = {}
    vessel_trace_indices = defaultdict(list)
    background_trace_indices = []
    heatmap_draught_groups = {
        i: []
        for i in range(len(draught_values))
    }
    initial_time = None
    for waterway in waterways:

        x_shift = offsets[id(waterway)] + start_distance

        if hydrodynamic_data is not None:
            heatmap_info = add_point_tidal_window_to_plotly(
                fig=fig,
                waterway=waterway,
                hydrodynamic_data=hydrodynamic_data,
                determinantes=determinantes,
                time_start=time_start,
                time_end=time_end,
                start_distance=x_shift,      # include the offset
                draught=10.5,
                draught_range=draught_values,
                offset=200,
                hydro_property=hydro_property,
                showscale=True,
            )

            for i, group in enumerate(heatmap_info["draught_groups"]):
                heatmap_draught_groups[i].extend(group)
                        
            background_trace_indices.extend(
                heatmap_info["all_heatmap_traces"]
            )

        current_vessels = (
            list(waterway.env.vessels.values())
            if vessels is None else vessels
        )

        for vessel in current_vessels:
            color = None
            if color_function is not None:
                color = color_function(vessel)

            vessel_df = get_vessel_logbook_through_waterway(
                vessel,
                waterway,
                extra_nodes,
            )

            if vessel_df.empty:
                continue

            customdata = np.column_stack([
                np.full(len(vessel_df), getattr(vessel, "id", "")),
                np.full(len(vessel_df), getattr(vessel, "name", "").split('(')[0]),
                np.full(len(vessel_df), vessel.type),
                np.full(len(vessel_df), vessel.classification),
                np.full(len(vessel_df), getattr(vessel, "L", np.nan)),
                np.full(len(vessel_df), getattr(vessel, "B", np.nan)),
                np.full(len(vessel_df), getattr(vessel, "T", np.nan)),
            ])

            vessel_name = vessel.name
            start_time = vessel_df.Timestamp.min()
            end_time = vessel_df.Timestamp.max()
            if initial_time is None or start_time < initial_time:
                initial_time = start_time
            
            if vessel_name not in vessel_times:
                vessel_times[vessel_name] = {
                    "start": start_time,
                    "end": end_time,
                }
            else:
                vessel_times[vessel_name]["start"] = min(
                    vessel_times[vessel_name]["start"],
                    start_time,
                )
                vessel_times[vessel_name]["end"] = max(
                    vessel_times[vessel_name]["end"],
                    end_time,
                )
            if color is not None:
                fig.add_trace(
                    go.Scattergl(
                        x=vessel_df.Value + x_shift,
                        y=vessel_df.Timestamp,
                        mode="lines",
                        line=dict(color=color, width=2),
                        name=vessel.name,
                        legendgroup=vessel.id,
                        showlegend=vessel.id not in shown_vessels,
                        customdata=customdata,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "MMSI: %{customdata[1]}<br>"
                            "Type: %{customdata[2]}<br>"
                            "Class: %{customdata[3]}<br>"
                            "Length: %{customdata[4]} m<br>"
                            "Beam: %{customdata[5]} m<br>"
                            "Draught: %{customdata[6]} m<br>"
                            "Distance: %{x:.0f} m<br>"
                            "Time: %{y|%d-%b-%Y %H:%M:%S}"
                            "<extra></extra>"
                        ),
                    )
                )
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=vessel_df.Value + x_shift,
                        y=vessel_df.Timestamp,
                        mode="lines",
                        line=dict(width=2),
                        name=vessel.name,
                        legendgroup=vessel.id,
                        showlegend=vessel.id not in shown_vessels,
                        customdata=customdata,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "MMSI: %{customdata[1]}<br>"
                            "Type: %{customdata[2]}<br>"
                            "Class: %{customdata[3]}<br>"
                            "Length: %{customdata[4]} m<br>"
                            "Beam: %{customdata[5]} m<br>"
                            "Draught: %{customdata[6]} m<br>"
                            "Distance: %{x:.0f} m<br>"
                            "Time: %{y|%d-%b-%Y %H:%M:%S}"
                            "<extra></extra>"
                        ),
                    )
                )
            trace_index = len(fig.data) - 1
            vessel_trace_indices[vessel.id].append(trace_index)
            trace_to_vessel.append(vessel.id)
            shown_vessels.add(vessel.id)

        # Node lines
        for _, node_info in waterway.node_distances.iterrows():

            x = node_info.distance + x_shift
            if xmin is None:
                xmin = x
            elif x < xmin:
                xmin = x

            if xmax is None:
                xmax = x
            elif x > xmax:
                xmax = x
                
            label = node_info["node"].replace("_", "")

            if "KM" not in label:
                label += f" KM{int(x/1000)}.0"
            elif "." not in label:
                label += ".0"

            if label not in xticklabels:
                xticks.append(x)
                xticklabels.append(label)

            fig.add_vline(
                x=x,
                line_dash="dash",
                line_color="black",
                line_width=1,
                opacity=0.4,
            )

    fig.update_xaxes(
        tickvals=xticks,
        ticktext=xticklabels,
        tickangle=90,
        title="Distance [m]",
    )

    if time_start is not None and time_end is not None:
        fig.update_yaxes(
            range=[time_start, time_end],
            title="Date and time",
        )
    else:
        yaxis_kwargs = dict(
            title="Date and time",
            autorange=True,
        )
        
        # If requested, use a fixed time window
        if time_start is not None and time_end is not None:
            yaxis_kwargs["range"] = [time_start, time_end]
            yaxis_kwargs["autorange"] = False
        
        fig.update_yaxes(**yaxis_kwargs)
            

    fig.update_layout(
        template="simple_white",
    
        width=width,
        height=height,
    
        hovermode="closest",
    
        dragmode="pan",          # drag to move through time
        uirevision=True,         # preserve zoom when updating
    
        margin=dict(
            l=80,
            r=30,
            t=30,
            b=180,
        ),
    
        legend_title="Vessels",
    )

    # Allow vertical zoom/pan
    if time_start is not None:
        initial_time = time_start
    if time_end is not None:
        initial_end = time_end
    else:
        initial_end = initial_time + pd.Timedelta(days = 2)

    for waterway in waterways:

        x_shift = offsets[id(waterway)] + start_distance

        if show_rules:
            edge_df = waterway.edge_distances

            vessels_show_rules = get_vessels_by_name(waterway.env, vessels_to_show_rules)
            vessel_pairs = list(combinations(vessels_show_rules, 2))
            restriction_per_edge = {}
        
            for vessel_pair in vessel_pairs:
                for edge in zip(edge_df["node_start"], edge_df["node_stop"]):
        
                    restriction, _, _ = waterway.check_for_encountering_conflicts(
                        edge,
                        vessel_pair
                    )
        
                    # Keep edge restricted if any vessel pair has a conflict
                    if restriction:
                        restriction_per_edge[edge] = True

            for edge, restricted in restriction_per_edge.items():
        
                if not restricted:
                    continue
        
                edge_info = edge_df[
                    (edge_df.node_start == edge[0]) &
                    (edge_df.node_stop == edge[1])
                ].iloc[0]
        
                x0 = edge_info.distance_start + x_shift
                x1 = edge_info.distance_stop + x_shift
                
                # vertical rectangle spanning full time axis
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    fillcolor="rgba(205,92,92,0.25)",
                    line=dict(
                        color="indianred",
                        width=1,
                    ),
                    layer="below",
                )
        
    # Limit horizontal navigation
    fig.update_xaxes(
        range=[xmin, xmax],
        minallowed=xmin,
        maxallowed=xmax,
        fixedrange=False,
    )
    
    fig.update_yaxes(
        range=[initial_time, initial_end],
        fixedrange=False,
        autorange=False,
    )

    # Final layout
    fig.update_layout(
        dragmode="pan",
        uirevision=True,
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        )
    )

    unique_times = sorted(
        set(
            [v["start"] for v in vessel_times.values()] +
            [v["end"] for v in vessel_times.values()]
        )
    )
    
        
    def get_visibility(min_time, max_time):
    
        visible = [True] * len(fig.data)   # everything visible initially
    
        # Vessel traces
        for vessel, times in vessel_times.items():
    
            show = (
                times["start"] <= max_time
                and times["end"] >= min_time
            )
    
            for idx in vessel_trace_indices[vessel]:
                visible[idx] = show
    
        # Heatmaps (and any other background traces) remain visible
        for idx in background_trace_indices:
            visible[idx] = (
                idx in heatmap_draught_groups[5]
            )
            
        return visible
        
    
    # --- Minimum time slider ---
    steps_min = []
    for t in unique_times:
        steps_min.append(
            dict(
                method="update",
                args=[
                    {
                        "visible": get_visibility(
                            t,
                            unique_times[-1],
                        )
                    }
                ],
                label=t.strftime("%d-%b %H:%M"),
            )
        )
    
    # --- Maximum time slider ---
    steps_max = []
    
    for t in unique_times:
    
        steps_max.append(
            dict(
                method="update",
                args=[
                    {
                        "visible": get_visibility(
                            unique_times[0],
                            t,
                        )
                    }
                ],
                label=t.strftime("%d-%b %H:%M"),
            )
        )

    draught_steps = []

    for i, draught in enumerate(draught_values):
    
        visible = [True] * len(fig.data)
    
        # hide all heatmaps
        for idx in background_trace_indices:
            visible[idx] = False
    
        # activate selected draught
        for idx in heatmap_draught_groups[i]:
            visible[idx] = True
    
        draught_steps.append(
            dict(
                method="update",
                args=[
                    {
                        "visible": visible
                    }
                ],
                label=f"{draught:.1f} m",
            )
        )
    
    fig.update_layout(
        sliders=[
            dict(
                active=5,
                currentvalue=dict(
                    prefix="Draught: ",
                    #suffix=" m",
                    font=dict(size=12),
                ),
                pad=dict(t=30),
                x=0,
                y=1.2,
                len=0.4,
                steps=draught_steps,
            ),
        ]
    )

    fig.show(
        config={
            "displayModeBar": True,
            "scrollZoom": False,
            "modeBarButtonsToAdd": [
                "zoom2d",
                "pan2d",
            ],
        }
    )
    
    return fig