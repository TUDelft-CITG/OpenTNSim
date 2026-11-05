# config.py
import numpy as np
import xarray as xr
from opentnsim.core import SimpyObject
import datetime

class HydrodynamicDataManager:
    """
    Singleton class to manage hydrodynamic data.
    This class ensures that hydrodynamic data is loaded only once
    and can be accessed globally.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.hydrodynamic_data = None
            cls._instance.hydrodynamic_times = None
            cls._instance.hydrodynamic_times_ns = None

        # Ensure caches exist, including for older instances
        if not hasattr(cls._instance, "station_cache"):
            cls._instance.station_cache = {}

        if not hasattr(cls._instance, "station_data_cache"):
            cls._instance.station_data_cache = {}

        return cls._instance

    def get_station_data(self, station, parameter):
        """
        Get a hydrodynamic time series for a station.

        Parameters
        ----------
        station : str
            Station name.
        parameter : str
            Hydrodynamic property, e.g. "Water level",
            "Nautical depth", "Current velocity", or "Salinity".

        Returns
        -------
        np.ndarray
            Time series for the requested property.
        """

        if self.hydrodynamic_data is None:
            return np.array([])

        if parameter in self.cached_parameters:
            if station not in self.station_data_cache:
                raise KeyError(
                    f"No hydrodynamic data cached for station '{station}'."
                )

            return self.station_data_cache[station][parameter]

        # Fallback for properties not included in the cache
        station_index = self._get_station_index_of_hydrodynamic_data(
            station
        )

        if station_index is None:
            raise KeyError(
                f"Station '{station}' not found in hydrodynamic data."
            )

        if isinstance(self.hydrodynamic_data, xr.Dataset):
            return self.hydrodynamic_data[parameter.title()][
                station_index
            ].values

        return self.hydrodynamic_data[parameter.title()][station_index]

    def get_water_level(self, station):
        """Get the water level time series for a station."""
        return self.get_station_data(station, "Water level")

    def get_mbl(self, station):
        """Get the nautical depth (MBL) time series for a station."""
        return self.get_station_data(station, "Nautical depth")

    def get_water_depth(self, station, start_index=None, end_index=None):
        """
        Get water depth (water level minus nautical depth).

        Optionally restrict the result to a time interval using indices.
        """

        water_level = self.get_water_level(station)
        mbl = self.get_mbl(station)

        if start_index is not None or end_index is not None:
            water_level = water_level[start_index:end_index]
            mbl = mbl[start_index:end_index]

        return water_level - mbl

    def get_station_time_slice(
        self,
        station,
        parameter,
        start_index,
        end_index,
    ):
        """Get a time slice of a station's hydrodynamic data."""

        return self.get_station_data(
            station, parameter
        )[start_index:end_index]

    def get_time_array(self):
        """Get timestamps as integer nanoseconds since Unix epoch."""

        if self.hydrodynamic_data is None:
            return np.array([], dtype=np.int64)

        if self.hydrodynamic_times is None:
            if isinstance(self.hydrodynamic_data, xr.Dataset):
                times = self.hydrodynamic_data["TIME"].values
            else:
                times = self.hydrodynamic_data["TIME"]

            self.hydrodynamic_times = np.asarray(
                times
            ).astype("datetime64[ns]")

        if self.hydrodynamic_times_ns is None:
            self.hydrodynamic_times_ns = (
                self.hydrodynamic_times.astype(np.int64)
            )

        return self.hydrodynamic_times_ns

    def get_time_slice(self, start_index, end_index):
        """Get timestamps between two indices, in datetime64[ns]."""

        return self.get_time_array()[
            start_index:end_index
        ].astype("datetime64[ns]")

    def _get_station_index_of_hydrodynamic_data(self, node):
        """Get the station index of a node in the hydrodynamic data."""

        if self.hydrodynamic_data is None:
            return None

        if isinstance(self.hydrodynamic_data, xr.Dataset):
            stations = self.hydrodynamic_data["STATION"].values
        else:
            stations = self.hydrodynamic_data["STATION"]

        indices = np.flatnonzero(np.asarray(stations) == node)

        if len(indices) == 0:
            return None

        return int(indices[0])    

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

    def get_water_depth_at_simulation_time(
        self,
        node,
        simulation_time,
    ):
        """
        Get MBL, water level and available water depth at the
        nearest hydrodynamic timestamp to a simulation time.

        Parameters
        ----------
        node : str
            Graph node identifying the hydrodynamic station.
        simulation_time : float
            Simulation time in seconds since the Unix epoch.

        Returns
        -------
        tuple
            MBL, water level, available water depth.
        """

        station = self.station_cache.get(node)

        if station is None:
            raise ValueError(f"No station found for node: {node}")

        water_level_array = self.station_data_cache[station]['Water level']
        mbl_array = self.station_data_cache[station]['Nautical depth']

        # Convert simulation time to integer nanoseconds
        current_time_ns = np.datetime64(
            datetime.datetime.fromtimestamp(simulation_time),
            "ns",
        ).astype(np.int64)

        # Get cached integer timestamps
        time_array = self.get_time_array()

        if len(time_array) == 0:
            raise ValueError("No hydrodynamic timestamps available.")

        # Find nearest timestamp
        time_index = np.searchsorted(time_array, current_time_ns)

        if time_index == len(time_array):
            time_index -= 1

        elif time_index > 0:
            if (
                current_time_ns - time_array[time_index - 1]
                < time_array[time_index] - current_time_ns
            ):
                time_index -= 1

        water_level = water_level_array[time_index]
        MBL = mbl_array[time_index]
        
        return MBL, water_level, water_level - MBL

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
        time = np.datetime64(time)
        if isinstance(self.hydrodynamic_data, xr.Dataset):
            time_index = np.absolute(self.hydrodynamic_data["TIME"].values - time).argmin()
        else:
            time_index = np.absolute(self.hydrodynamic_data["TIME"] - time).argmin()

        return time_index

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
    precompute_station_mapping,
    transpose_data_in_accepted_order,
)

class HydrodynamicData(SimpyObject):

    def __init__(self, hydrodynamic_data: xr.Dataset, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if not hasattr(self.env, 'graph'):
            raise NotImplementedError(
                "A graph needs to be added to the environment"
            )

        check_hydrodynamic_data_coordinates(hydrodynamic_data)
        check_hydrodynamic_data_variables(hydrodynamic_data)
        check_hydrodynamic_data_temporal_coverage(
            self.env, hydrodynamic_data
        )

        hydrodynamic_data = transpose_data_in_accepted_order(
            hydrodynamic_data
        )

        self.env.hydrodynamics = True

        hydro_manager = HydrodynamicDataManager()

        hydro_manager.hydrodynamic_data = hydrodynamic_data

        # Precompute time array
        hydro_manager.time_array = (
            hydrodynamic_data["TIME"].values
            .astype("datetime64[ns]")
            .astype(np.int64)
        )

        hydro_manager.station_data_cache = {}

        accepted_vars = [
            "Water level",
            "Current velocity",
            "Current direction",
            "Nautical depth",
            "Salinity",
            "Temperature",
        ]

        stations = hydrodynamic_data["STATION"].values

        available_vars = {
            var: hydrodynamic_data[var].values
            if var in hydrodynamic_data.data_vars
            else None
            for var in accepted_vars
        }

        hydro_manager.station_data_cache = {
            station: {
                var: values[i] if values is not None else None
                for var, values in available_vars.items()
            }
            for i, station in enumerate(stations)
        }

        hydro_manager.cached_parameters = list(available_vars.keys())

        # Precompute node-to-station mapping
        precompute_station_mapping(
            hydro_manager,
            self.env.graph
        )
        
