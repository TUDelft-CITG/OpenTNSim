from opentnsim.port.visualizations import (create_plot_vertical_tidal_window,
                                           create_plot_horizontal_tidal_window,
                                           plot_tidal_windows)
from opentnsim.core import VesselProperties

def find_route_with_restrictions(self):
    routes_with_restrictions = []
    route_with_restriction = []
    for node_start, node_stop in zip(self.route[:-1], self.route[1:]):
        edge = (node_start, node_stop)
        if 'Depth_restriction' in self.env.graph.nodes[node_start].keys():
            route_with_restriction.append(node_start)
        elif 'Depth_restriction' in self.env.graph.edges[edge].keys():
            route_with_restriction.append(node_start)

        if 'Current_restriction' not in self.env.graph.nodes[node_stop].keys():  # create new route
            routes_with_restrictions.append(route_with_restriction)
            route_with_restriction = []
        elif 'Current_restriction' not in self.env.graph.edges[edge].keys():  # create new route
            routes_with_restrictions.append(route_with_restriction)
            route_with_restriction = []
        if node_stop == self.route[-1]:  # if last node
            route_with_restriction.append(node_stop)
            routes_with_restrictions.append(route_with_restriction)
    return routes_with_restrictions


class HasDraughtRestrictions(VesselProperties):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tidal_window_calculations = {}

    def create_plot_vertical_tidal_window(self, canal = None, trip_index = 0, plot = False):
        fig, legend_handles, legend_labels = create_plot_vertical_tidal_window(self, canal, trip_index, plot)
        if fig is None:
            return None
        return fig, legend_handles, legend_labels


    def create_plot_horizontal_tidal_window(self, canal = None, trip_index = 0, plot = False):
        fig, legend_handles, legend_labels = create_plot_horizontal_tidal_window(self, canal, trip_index, plot)
        return fig, legend_handles, legend_labels


    def plot_tidal_windows(self, canal = None, trip_index = 0, plot_all = False):
        fig_final = plot_tidal_windows(self, canal, trip_index, plot_all)
        return fig_final