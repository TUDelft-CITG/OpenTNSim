import simpy
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import pytz
from shapely.geometry import Point, Polygon, MultiPolygon
from matplotlib import dates
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import copy

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


def copy_patch(patch, ax_target, zorder = None):
    """Create a new patch in ax_target with the same properties as patch."""
    cls = type(patch)  # Rectangle, Polygon, etc.
    if not zorder:
        zorder = patch.get_zorder()
    # Extract common properties
    kwargs = {
        'facecolor': patch.get_facecolor(),
        'edgecolor': patch.get_edgecolor(),
        'linewidth': patch.get_linewidth(),
        'linestyle': patch.get_linestyle(),
        'alpha': patch.get_alpha(),
        'label': patch.get_label(),
        'zorder': zorder
    }

    # Handle specific patch types
    if isinstance(patch, mpatches.Polygon):
        new_patch = mpatches.Polygon(patch.get_xy(), **kwargs)
    elif isinstance(patch, mpatches.Rectangle):
        new_patch = mpatches.Rectangle(
            (patch.get_x(), patch.get_y()),
            patch.get_width(),
            patch.get_height(),
            **kwargs
        )
    else:
        # fallback: try to copy as-is
        new_patch = cls(**kwargs)

    ax_target.add_patch(new_patch)


def check_if_route_contains_restrictions(self):
    contains_restriction = False
    for node in self.route:
        if 'Vertical tidal restriction' in self.env.graph.nodes[node].keys():
            contains_restriction = True
            break
    return contains_restriction


def find_route_with_restrictions(self):
    routes_with_restrictions = []
    route_with_restriction = []
    for node_start, node_stop in zip(self.route[:-1], self.route[1:]):
        if 'Vertical tidal restriction' in self.env.graph.nodes[node_start].keys():
            route_with_restriction.append(node_start)
        if 'Vertical tidal restriction' not in self.env.graph.nodes[node_stop].keys():  # create new route
            routes_with_restrictions.append(route_with_restriction)
            route_with_restriction = []
        elif node_stop == self.route[-1]:  # if last node
            route_with_restriction.append(node_stop)
            routes_with_restrictions.append(route_with_restriction)
    return routes_with_restrictions


class HasDraughtRestrictions:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.on_pass_node_functions.append(self.wait_for_tidal_window)
        self.bound = 'inbound'
        self.tidal_window_calculations = []

        # def request_tidal_window(self, origin):
    #     infrastructure_suitable_to_wait_for_tidal_window = ['Waiting Area', 'Terminal']
    #     suitable_location_to_wait_for_tidal_window = False
    #     for infrastructure in infrastructure_suitable_to_wait_for_tidal_window:
    #         if infrastructure in self.env.graph.nodes[origin].keys():
    #             suitable_location_to_wait_for_tidal_window = True
    #             break
    #
    #     if not suitable_location_to_wait_for_tidal_window:
    #         return self.tidal_waiting_time
    #
    #     contains_restriction = check_if_route_contains_restrictions(self)
    #     if not contains_restriction:
    #         return self.tidal_waiting_time
    #
    #     routes_with_restrictions = find_route_with_restrictions(self)
    #     self.tidal_waiting_time = 0.
    #
    #     if origin != routes_with_restrictions[0][0]:
    #         return self.tidal_waiting_time
    #
    #     for route in routes_with_restrictions:
    #         self.tidal_waiting_time = self.env.self_traffic_service.provide_waiting_time_for_inbound_tidal_window(self=self, route=route, delay=0, plot=True)
    #
    #     if math.isnan(self.tidal_waiting_time):
    #         raise simpy.exceptions.Interrupt('Port not accessible for self.')
    #
    #     return self.tidal_waiting_time


    def wait_for_tidal_window(self, origin, waiting_time = None):
        if waiting_time is not None:
            self.tidal_waiting_time = waiting_time
        self.log_entry_v0("Waiting for tidal window start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        yield self.env.timeout(self.tidal_waiting_time)
        self.log_entry_v0("Waiting for tidal window stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        self.tidal_waiting_time = 0.


    def create_plot_vertical_tidal_window(self, calculation_index = 0, plot = False):
        tidal_window_calculation_results = self.tidal_window_calculations[calculation_index]
        time_start_index = tidal_window_calculation_results['time_start_index']
        time_end_index = tidal_window_calculation_results['time_end_index']
        route = tidal_window_calculation_results['route']
        bound = tidal_window_calculation_results['bound']
        draught = tidal_window_calculation_results['draught']
        vertical_tidal_windows = tidal_window_calculation_results['vertical_tidal_windows']
        net_ukcs = tidal_window_calculation_results['net_ukcs']
        vessel_traffic_service = self.env.vessel_traffic_service
        hydrodynamic_information = vessel_traffic_service.hydrodynamic_information

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
        ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d %H:%M"))

        handles = [net_UKC, minimum_required_net_ukc, vertical_tidal_window]
        labels = ["Net UKC", "Minimum required net UKC", "Vertical tidal windows"]
        legend_handles = []
        legend_labels = []
        for handle, legend in zip(handles, labels):
            legend_handles.append(handle)
            legend_labels.append(legend)

        ax.set_xlabel("Start time of trip")
        ax.set_ylabel("Minimum net UKC experienced over entire self route [m]")

        if plot:
            ax.legend(legend_handles, legend_labels, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
            ax.set_title(
                f"Vertical tidal windows of {self.type}-class self '{self.name}' with "
                f"a draught of {np.round(draught, 2)}m and\na length of {np.round(self.L)}m sailing {bound} from"
                f" node '{route[0]}' to node '{route[-1]}'."
            )
            fig.tight_layout()
            plt.show()
        else:
            plt.close(fig)
        return fig, legend_handles, legend_labels


    def create_plot_horizontal_tidal_window(self, calculation_index = 0, plot = False):
        tidal_window_calculation_results = self.tidal_window_calculations[calculation_index]
        time_start_index = tidal_window_calculation_results['time_start_index']
        time_end_index = tidal_window_calculation_results['time_end_index']
        route = tidal_window_calculation_results['route']
        horizontal_tidal_windows = tidal_window_calculation_results['horizontal_tidal_windows']
        horizontal_tidal_restriction_nodes = tidal_window_calculation_results['horizontal_tidal_restriction_nodes']
        horizontal_tidal_restriction_stations = tidal_window_calculation_results['horizontal_tidal_restriction_stations']
        vessel_traffic_service = self.env.vessel_traffic_service
        hydrodynamic_information = vessel_traffic_service.hydrodynamic_information

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

            governing_current_velocity, _ = vessel_traffic_service.provide_governing_current_velocity(vessel,
                                                                                                      station,
                                                                                                      time_start_index,
                                                                                                      time_end_index)
            sailing_time = self.provide_sailing_time(vessel, route[: (route.index(node) + 1)])["Time"].sum() + delay
            horizontal_tidal_accessibility_time_correction = np.timedelta64(int(sailing_time), "s")
            horizontal_tidal_accessibility_time = hydrodynamic_information.TIME.values[time_start_index:time_end_index]
            horizontal_tidal_accessibility_time -= horizontal_tidal_accessibility_time_correction
            (current_velocity,) = ax.plot(horizontal_tidal_accessibility_time, governing_current_velocity,
                                          color="firebrick", linewidth=3,)
        ax.axhline(0, color="k", linewidth=2)

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
        ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d %H:%M"))

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
                f"Horizontal tidal windows of {self.type}-class self '{self.name}' with "
                f"a draught of {np.round(self.T, 2)}m and\na length of {np.round(self.L)}m sailing {self.bound} from"
                f" node '{route[0]}' to node '{route[-1]}'."
            )
            fig.tight_layout()
            plt.show()
        else:
            plt.close(fig)
        return fig, legend_handles, legend_labels


    def plot_tidal_window(self, calculation_index = 0):
        tidal_window_calculation_results = self.tidal_window_calculations[calculation_index]
        route = tidal_window_calculation_results['route']
        tidal_windows = tidal_window_calculation_results['tidal_windows']

        # Create figure
        fig, ax_left = plt.subplots(figsize=[16 * 2 / 3, 6])
        ax_left.set_facecolor('none')
        ax_right = ax_left.twinx()
        ax_left.set_facecolor('none')

        # Plot governing current velocity
        fig_left, ax_left_handles, ax_left_labels = self.create_plot_vertical_tidal_window(calculation_index)
        fig_right, ax_right_handles, ax_right_labels = self.create_plot_horizontal_tidal_window(calculation_index)

        for fig, ax_target in zip([fig_left, fig_right], [ax_left, ax_right]):
            for ax_old in fig.get_axes():
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
                ax_target.set_title(ax_old.get_title())

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
        window_y = [ax_left.get_ylim()[0], ax_left.get_ylim()[1], ax_left.get_ylim()[1], ax_left.get_ylim()[0]]
        color = lighten_color("limegreen", alpha=0.4)
        for window in tidal_windows:
            window_x = [window[0], window[0], window[1], window[1]]
            (tidal_window,) = ax_left.fill(window_x, window_y,
                                           facecolor=color, edgecolor="none", zorder=-5,)
        if not tidal_windows:
            window_x = [0, 0, 0, 0]
            tidal_window = ax_left.fill(window_x, window_y,
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

        handles = [tidal_window, no_tidal_window]
        labels = ["Tidal window", "No tidal window"]
        legend_handles = np.append(legend_handles, handles)
        legend_labels = np.append(legend_labels, labels)

        ax_left.legend(legend_handles, legend_labels, frameon=False, loc="upper left", bbox_to_anchor=(1.05, 1.0),)
        ax_left.set_title(
            f"Accessibility of {self.type}-class self '{self.name}' with "
            f"a draught of {np.round(self.T, 2)}m and\na length of {np.round(self.L)}m sailing {self.bound} from"
            f" node '{route[0]}' to node '{route[-1]}'."
        )
        return fig