# config.py
import numpy as np
import xarray as xr
from opentnsim.core import SimpyObject

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
        """Gets the interpolated value of a hydrodynamic property at a certain time and node

        Parameters
        ----------
        time : np.datetime64
            the time
        node : str
            the node name in the graph
        hydrodynamic_property : str
            the hydrodynamic property: "Water level", "Current velocity", "Salinity"

        Returns
        -------
        value : float
            the interpolated value of a hydrodynamic property at the specified time and node
        """
        value = np.nan
        if self.hydrodynamic_data is None:
            return value

        # convert time to float for interpolation
        time_float = np.datetime64(time, 's').astype(float)

        # get station index
        station_index = self._get_station_index_of_hydrodynamic_data(node)

        # get time series
        H_time = self.hydrodynamic_data['TIME'].values.astype('datetime64[s]').astype(float)

        # get data series at that station
        if isinstance(self.hydrodynamic_data, xr.Dataset):
            data_series = self.hydrodynamic_data[hydrodynamic_property][station_index, :].values
        else:
            data_series = self.hydrodynamic_data[hydrodynamic_property][station_index, :]

        # interpolate in time
        value = np.interp(time_float, H_time, data_series)

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
        series = np.array([])
        if self.hydrodynamic_data is None:
            return series

        # determine the time_index and station_index
        time_index = self._get_time_index_of_hydrodynamic_data(time)
        station_index = self._get_station_index_of_hydrodynamic_data(node)

        # determine the property
        if isinstance(self.hydrodynamic_data, xr.Dataset):
            series = self.hydrodynamic_data[hydrodynamic_property][station_index][time_index:].values.copy()
        else:
            series = self.hydrodynamic_data[hydrodynamic_property][station_index][time_index:].copy()

        return series


    def _get_interpolated_hydrodynamic_series(
            self,
            target_times,
            node,
            hydrodynamic_property,
    ):
        """
        Interpolates a hydrodynamic time series onto a new set of timestamps.

        Parameters
        ----------
        target_times : np.ndarray of np.datetime64
            The timestamps to interpolate to
        node : str
            Node name in the graph
        hydrodynamic_property : str
            Property name (e.g. "Water level", "Current velocity", "Salinity")
        allow_extrapolation : bool, optional
            If False, raises an error when target_times fall outside original range

        Returns
        -------
        interpolated_series : np.ndarray
            Interpolated values at target_times
        """
        if not isinstance(target_times, np.ndarray):
            if not isinstance(target_times, list):
                target_times = [target_times]
            target_times = np.array(target_times)

        if self.hydrodynamic_data is None:
            return np.array([])

        # Get station index
        station_index = self._get_station_index_of_hydrodynamic_data(node)

        # Extract original time + data
        if hasattr(self.hydrodynamic_data, "coords"):  # xarray
            original_times = self.hydrodynamic_data.coords["TIME"].values
            series = self.hydrodynamic_data[hydrodynamic_property][station_index].values
        else:
            original_times = self.hydrodynamic_data["TIME"]
            series = self.hydrodynamic_data[hydrodynamic_property][station_index]

        # Convert datetime64 → float (ns since epoch) for interpolation
        original_times_num = original_times.astype("datetime64[ns]").astype(np.int64)
        target_times_num = target_times.astype("datetime64[ns]").astype(np.int64)

        # Bounds checking
        t_min, t_max = original_times_num.min(), original_times_num.max()
        if target_times_num.min() < t_min or target_times_num.max() > t_max:
            raise ValueError("Extrapolating outside original time range.")

        # Interpolation
        interpolated_series = np.interp(
            target_times_num,
            original_times_num,
            series
        )

        return interpolated_series

from opentnsim.environment.utils import (
    check_hydrodynamic_data_coordinates,
    check_hydrodynamic_data_variables,
    check_hydrodynamic_data_temporal_coverage,
    check_hydrodynamic_data_spatial_coverage,
    transpose_data_in_accepted_order,
    sort_data
)


class HydrodynamicData(SimpyObject):

    def __init__(self, hydrodynamic_data: xr.Dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.env,'graph'):
            raise NotImplementedError("A graph needs to be added to the environment as env.graph = graph")
        check_hydrodynamic_data_coordinates(hydrodynamic_data)
        check_hydrodynamic_data_variables(hydrodynamic_data)
        check_hydrodynamic_data_temporal_coverage(self.env, hydrodynamic_data)
        # check_hydrodynamic_data_spatial_coverage(self.env.graph, hydrodynamic_data)
        hydrodynamic_data = transpose_data_in_accepted_order(hydrodynamic_data)
        # hydrodynamic_data = sort_data(self.env.graph, hydrodynamic_data)
        self.env.hydrodynamics = True
        hydro_manager = HydrodynamicDataManager()
        hydro_manager.hydrodynamic_data = hydrodynamic_data
