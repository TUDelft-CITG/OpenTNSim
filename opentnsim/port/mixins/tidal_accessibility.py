from opentnsim.port.visualizations import (create_plot_vertical_tidal_window,
                                           create_plot_horizontal_tidal_window,
                                           plot_tidal_windows)


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
        self.bound = 'inbound'
        self.tidal_window_calculations = {}


    def create_plot_vertical_tidal_window(self, trip_index = 0, plot = False):
        fig, legend_handles, legend_labels = create_plot_vertical_tidal_window(self, trip_index, plot)
        return fig, legend_handles, legend_labels


    def create_plot_horizontal_tidal_window(self, trip_index = 0, plot = False):
        fig, legend_handles, legend_labels = create_plot_horizontal_tidal_window(self, trip_index, plot)
        return fig, legend_handles, legend_labels


    def plot_tidal_windows(self, trip_index = 0, plot_all = False):
        fig_final = plot_tidal_windows(self, trip_index, plot_all)
        return fig_final