import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import folium

from opentnsim.port.calculations import calculate_interpolated_depth_values
from opentnsim.port.utils import create_logbook_with_directed_distances

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

    fig, ax = plt.subplots()
    plt.close()
    if vessels is None:
        vessels = env.vessels

    for idx, vessel in enumerate(vessels):
        fig_vessel = vessel.plot_time_distance_diagram()
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


def plot_time_distance_diagram(vessel):
    df = create_logbook_with_directed_distances(vessel)
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