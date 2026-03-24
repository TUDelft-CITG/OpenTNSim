# config.py
import numpy as np
import xarray as xr
import warnings
from opentnsim.core import SimpyObject


def check_hydrodynamic_data_coordinates(hydrodynamic_data):
    accepted_coordinates = ['STATION','TIME','LAYER']
    for coordinate in list(hydrodynamic_data.coords):
        if coordinate not in accepted_coordinates:
            ValueError(f"Data coordinate {coordinate} is not supported, only {accepted_coordinates} are supported")
    if 'STATION' not in list(hydrodynamic_data.coords):
        ValueError(f"Missing ''STATION'' coordinate in data.")
    if 'TIME' not in list(hydrodynamic_data.coords):
        ValueError(f"Missing ''TIME'' coordinate in data.")


def check_hydrodynamic_data_variables(hydrodynamic_data):
    accepted_data_variables = ['Water level', 'Current velocity', 'Current direction', 'Nautical depth', 'Salinity', 'Temperature']
    for data_variable in list(hydrodynamic_data.data_vars):
        if data_variable not in accepted_data_variables:
            warnings.warn(f"Data column {data_variable} is not used in the simulation, only {accepted_data_variables} are supported")


def check_hydrodynamic_data_temporal_coverage(env, hydrodynamic_data):
    time = hydrodynamic_data.TIME.values
    time_min, time_max = time.min(),time.max()
    if time_min > np.datetime64(env.simulation_start):
        ValueError(f"There is no available data starting at the simulation start time ''{env.simulation_start}''.")
    elif time_max < np.datetime64(env.simulation_stop):
        ValueError(f"There is no available data until the simulation stop time ''{env.simulation_stop}''.")


def check_hydrodynamic_data_spatial_coverage(graph, hydrodynamic_data):
    stations = hydrodynamic_data.STATION.values
    for node in graph.nodes:
        if node not in stations:
            ValueError(f"There is no data for node ''{node}''.")


def transpose_data_in_accepted_order(hydrodynamic_data):
    if 'LAYER' not in list(hydrodynamic_data.coords):
        hydrodynamic_data = hydrodynamic_data.transpose('STATION','TIME')
    else:
        hydrodynamic_data = hydrodynamic_data.transpose('STATION', 'TIME', 'LAYER')
    return hydrodynamic_data


def sort_data(graph, hydrodynamic_data):
    nodes = graph.nodes
    times = sorted(hydrodynamic_data.TIME.values)
    hydrodynamic_data = hydrodynamic_data.reindex(STATION=nodes)
    hydrodynamic_data = hydrodynamic_data.reindex(TIME=times)
    if 'LAYER' in list(hydrodynamic_data.coords):
        layers = sorted(hydrodynamic_data.LAYER.values)
        hydrodynamic_data = hydrodynamic_data.reindex(LAYER=layers)
    return hydrodynamic_data


class HydrodynamicDataManager:
    """
    Singleton class to manage hydrodynamic data.
    This class ensures that hydrodynamic data is loaded only once and can be accessed globally.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HydrodynamicDataManager, cls).__new__(cls)
            cls._instance.hydrodynamic_data = None
            cls._instance.hydrodynamic_times = None

        return cls._instance

    def _get_hydrodynamic_data_value(self, time, node, hydrodynamic_property):
        """Gets the value of a hydrodynamic property at a certain time and node

        Parameters
        ----------
        time : np.datetime64
            the time
        node : str
            the node name in the graph
        hydrodynamic_property : str
            the hydrodynamic property: "Water level", "Current velocity", "Salinity" (if included in the hydrodynamic data)

        Returns
        -------
        value : float
            the value of a hydrodynamic property at the specified time and node
        """
        if self.hydrodynamic_data is None:
            return None

        # determine the time_index and station_inex
        time_index = self._get_time_index_of_hydrodynamic_data(time)
        station_index = self._get_station_index_of_hydrodynamic_data(node)

        # determine the property
        if isinstance(self.hydrodynamic_data, xr.Dataset):
            value = self.hydrodynamic_data[hydrodynamic_property][station_index][time_index].values.copy()
        else:
            value = self.hydrodynamic_data[hydrodynamic_property][station_index][time_index].copy()

        return value

    def _get_time_index_of_hydrodynamic_data(self, time):
        """Gets the time index in the hydrodynamic data closest to a time

        Parameters
        ----------
        env : Simpy.Environment
            the simulation environment (to access the hydrodynamic data).
            the time

        Returns
        -------
        time_index : int
            the time index of the hydrodynamic data closest to the time
        """
        if self.hydrodynamic_data is None:
            return None

        # determine the time_index
        if isinstance(self.hydrodynamic_data, xr.Dataset):
            time_index = np.absolute(self.hydrodynamic_data["TIME"].values - time).argmin()
        else:
            time_index = np.absolute(self.hydrodynamic_data["TIME"] - time).argmin()

        return time_index

    def _get_station_index_of_hydrodynamic_data(self, node):
        """Gets the node's station index in the hydrodynamic data

        Parameters
        ----------
        node : str
            the node name in the graph

        Returns
        -------
        station_index : str
            the time index of the hydrodynamic data closest to the time
        """
        if self.hydrodynamic_data is None:
            return None

        if isinstance(self.hydrodynamic_data, xr.Dataset):
            station_index = np.where(np.array(list((self.hydrodynamic_data["STATION"].values))) == node)[0][0]
        else:
            station_index = np.where(np.array(list((self.hydrodynamic_data["STATION"]))) == node)[0]

        return station_index

    def _get_hydrodynamic_data_series(self, time, node, hydrodynamic_property):
        """Gets the time series of a hydrodynamic property at a certain node from a certain time onwards

        Parameters
        ----------
        time : np.datetime64
            the time
        node : str
            the node name in the graph
        hydrodynamic_property : str
            the hydrodynamic property: "Water level", "Current velocity", "Salinity" (if included in the hydrodynamic data)

        Returns
        -------
        series : float
            the time series of a hydrodynamic property at the specified node from the specified time onwards
        """
        if self.hydrodynamic_data is None:
            return np.array([])

        # determine the time_index and station_index
        time_index = self._get_time_index_of_hydrodynamic_data(time)
        station_index = self._get_station_index_of_hydrodynamic_data(node)

        # determine the property
        if isinstance(self.hydrodynamic_data, xr.Dataset):
            series = self.hydrodynamic_data[hydrodynamic_property][station_index][time_index:].values.copy()
        else:
            series = self.hydrodynamic_data[hydrodynamic_property][station_index][time_index:].copy()

        return series


class HydrodynamicData(SimpyObject):

    def __init__(self, hydrodynamic_data: xr.Dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.env,'graph'):
            raise NotImplementedError("A graph needs to be added to the environment as env.graph = graph")
        check_hydrodynamic_data_coordinates(hydrodynamic_data)
        check_hydrodynamic_data_variables(hydrodynamic_data)
        check_hydrodynamic_data_temporal_coverage(self.env, hydrodynamic_data)
        check_hydrodynamic_data_spatial_coverage(self.env.graph, hydrodynamic_data)
        hydrodynamic_data = transpose_data_in_accepted_order(hydrodynamic_data)
        hydrodynamic_data = sort_data(self.env.graph, hydrodynamic_data)
        self.env.hydrodynamics = True
        hydro_manager = HydrodynamicDataManager()
        hydro_manager.hydrodynamic_data = hydrodynamic_data
