from opentnsim.port.visualizations import (create_plot_vertical_tidal_window,
                                           create_plot_horizontal_tidal_window,
                                           plot_tidal_windows)
from opentnsim.core import VesselProperties


class HasDraughtRestrictions(VesselProperties):
    """Mixin class: Something has draught restrictions
        
    Used for object that has tidal windows in a port

    Attributes
    ----------
    tidal_window_calculations: dict [str, pandas.DataFrame]
        dictionary that stores all the calculated tidal windows per trip_id

    Methods
    -------
    create_plot_vertical_tidal_window:
        creating a time-series plot of the resulting vertical (water level) tidal windows around the time of sailing
    create_plot_horizontal_tidal_window: 
        creating a time-series plot of the resulting horizontal (current velocity) tidal windows around the time of sailing
    plot_tidal_windows: 
        creating a time-series plot of the resulting tidal windows around the time of sailing
    """
    def __init__(self, *args, **kwargs):
        """Initialization"""
        super().__init__(*args, **kwargs)
        self.tidal_window_calculations = {}

    def create_plot_vertical_tidal_window(self, name_waterway_of_interest = None, trip_index = 0, plot = False):
        fig, legend_handles, legend_labels = create_plot_vertical_tidal_window(self, name_waterway_of_interest, trip_index, plot)
        if fig is None:
            return None
        return fig, legend_handles, legend_labels

    def create_plot_horizontal_tidal_window(self, name_waterway_of_interest = None, trip_index = 0, plot = False):
        fig, legend_handles, legend_labels = create_plot_horizontal_tidal_window(self, name_waterway_of_interest, trip_index, plot)
        return fig, legend_handles, legend_labels

    def plot_tidal_windows(self, name_waterway_of_interest = None, trip_index = 0, plot_all = False):
        fig_final = plot_tidal_windows(self, name_waterway_of_interest, trip_index, plot_all)
        return fig_final
    