# config.py
import numpy as np

import xarray as xr

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

    def _get_hydrodynamic_data_value(self, hydrodynamic_information_path, time, node, hydrodynamic_property):
        """Gets the value of a hydrodynamic property at a certain time and node

        Parameters
        ----------
        hydrodynamic_information_path : path
            the path to hydrodynamic data (if None, no hydrodynamic data is included in the simulation)
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
        value = np.nan
        if hydrodynamic_information_path is None:
            return value

        # determine the time_index and station_inex
        time_index = self._get_time_index_of_hydrodynamic_data(hydrodynamic_information_path, time)
        station_index = self._get_station_index_of_hydrodynamic_data(hydrodynamic_information_path, node)

        # determine the property
        if isinstance(self.hydrodynamic_data, xr.Dataset):
            value = self.hydrodynamic_data[hydrodynamic_property][station_index][time_index].values.copy()
        else:
            value = self.hydrodynamic_data[hydrodynamic_property][station_index][time_index].copy()

        return value

    def _get_time_index_of_hydrodynamic_data(self, hydrodynamic_information_path, time):
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
        time_index = 0
        if hydrodynamic_information_path is None:
            return time_index

        # determine the time_index
        if isinstance(self.hydrodynamic_data, xr.Dataset):
            time_index = np.absolute(self.hydrodynamic_times - time).argmin().values
        else:
            time_index = np.absolute(self.hydrodynamic_times - time).argmin()

        return time_index

    def _get_station_index_of_hydrodynamic_data(self, hydrodynamic_information_path, node):
        """Gets the node's station index in the hydrodynamic data

        Parameters
        ----------
        hydrodynamic_information_path : path
            the path to hydrodynamic data (if None, no hydrodynamic data is included in the simulation)
        node : str
            the node name in the graph

        Returns
        -------
        station_index : str
            the time index of the hydrodynamic data closest to the time
        """

        station_index = 0
        if hydrodynamic_information_path is None:
            return station_index

        if isinstance(self.hydrodynamic_data, xr.Dataset):
            station_index = np.where(np.array(list((self.hydrodynamic_data["STATION"].values))) == node)[0][0]
        else:
            station_index = np.where(np.array(list((self.hydrodynamic_data["STATION"]))) == node)[0]

        return station_index

    def _get_hydrodynamic_data_series(self, hydrodynamic_information_path, time, node, hydrodynamic_property):
        """Gets the time series of a hydrodynamic property at a certain node from a certain time onwards

        Parameters
        ----------
        hydrodynamic_information_path : path
            the path to hydrodynamic data (if None, no hydrodynamic data is included in the simulation)
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
        series = np.array([np.nan])
        if hydrodynamic_information_path is None:
            return series

        # determine the time_index and station_inex
        time_index = self._get_time_index_of_hydrodynamic_data(hydrodynamic_information_path, time)
        station_index = self._get_station_index_of_hydrodynamic_data(hydrodynamic_information_path, node)

        # determine the property
        if isinstance(self.hydrodynamic_data, xr.Dataset):
            series = self.hydrodynamic_data[hydrodynamic_property][station_index][time_index:].values.copy()
        else:
            series = self.hydrodynamic_data[hydrodynamic_property][station_index][time_index:].copy()

        return series
