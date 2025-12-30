import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

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
