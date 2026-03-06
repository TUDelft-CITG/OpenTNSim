import pandas as pd
import numpy as np
import datetime
import xarray as xr
from pyproj import Transformer
from shapely.geometry import Point
import warnings

from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from opentnsim.graph.utils import get_length_of_edge, get_closest_node_to_geometry

def get_water_depth(vessel, node, delay=0):
    hydromanager = HydrodynamicDataManager()
    node_index = list(hydromanager.hydrodynamic_data["STATION"][:]).index(node)
    current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now + delay)).to_datetime64()
    time_index = np.absolute(hydromanager.hydrodynamic_data['TIME'].values - current_time).argmin()
    water_level = hydromanager.hydrodynamic_data["Water level"][node_index, time_index].values
    MBL = hydromanager.hydrodynamic_data["Nautical depth"][node_index, time_index].values
    available_water_depth = water_level - MBL
    return MBL, water_level, available_water_depth

def depth_averaged_current_velocity(interpolation_depth, times, relative_layer_height, current_velocity, station_index):
    layer_boundaries = []
    average_current_velocity = []
    number_of_layers = len(relative_layer_height)
    water_level = hydromanager.hydrodynamic_data["Water level"][:, station_index].data
    MBL = hydromanager.hydrodynamic_data["Nautical depth"][:, station_index].data
    water_depth = water_level - MBL
    relative_water_depth = np.outer(water_depth, relative_layer_height)
    cumulative_water_depth = np.cumsum(relative_water_depth, axis=1)

    for ti in range(len(times)):
        layer_boundaries.append(np.interp(interpolation_depth, cumulative_water_depth[ti], np.arange(0, number_of_layers, 1)))
    layer_boundary = np.floor(layer_boundaries)
    relative_boundary_layer_thickness = layer_boundaries - layer_boundary

    for ti in range(len(times)):
        if int(layer_boundary[ti]) + 2 < len(relative_layer_height):
            rel_layer_heights = relative_layer_height[0 : int(layer_boundary[ti]) + 2].copy()
            rel_layer_heights[-1] = rel_layer_heights[-1] * relative_boundary_layer_thickness[ti]
            average_current_velocity.append(
                np.average(current_velocity[ti][0 : int(layer_boundary[ti]) + 2], weights=rel_layer_heights)
            )
        elif int(layer_boundary[ti]) == 0:
            average_current_velocity = current_velocity[ti]
        else:
            average_current_velocity.append(np.average(current_velocity[ti], weights=relative_layer_height))

    return average_current_velocity

def get_governing_current_velocity(vessel, node, time_start_index, time_end_index):
    hydromanager = HydrodynamicDataManager()
    hydrodynamic_data = hydromanager.hydrodynamic_data
    station_index = list(hydrodynamic_data["STATION"][:]).index(node)
    times = hydrodynamic_data['TIME'].values[time_start_index:time_end_index]
    relative_layer_height = hydrodynamic_data["LAYER"][:].data
    current_velocity = hydrodynamic_data["Current velocity"][time_start_index:time_end_index, station_index].data

    if "LAYER" in list(hydromanager.hydrodynamic_data["Current velocity"].dimensions):
        if vessel._T <= 5:
            current_velocity = depth_averaged_current_velocity(5, times, relative_layer_height, current_velocity, station_index)
        elif vessel._T <= 15:
            current_velocity = depth_averaged_current_velocity(
                vessel._T, times, relative_layer_height, current_velocity, station_index
            )
        else:
            current_velocity = [np.average(current_velocity[t], weights=relative_layer_height) for t in range(len(times))]

    if len(current_velocity) > 2:
        current_governing_current_velocity = current_velocity[2]
    else:
        current_governing_current_velocity = current_velocity[-1]

    return current_velocity, current_governing_current_velocity


def create_default_hydrodynamic_dataset(env,properties = ['Water level', 'Nautical depth']):
    default_values = {'Water level': 999., 'Nautical depth': -999.}
    stations = list(env.graph.nodes)
    times = pd.date_range(env.simulation_start,env.simulation_stop, freq=pd.Timedelta(minutes=5))
    hydrodynamic_data = xr.Dataset()
    for property_ in properties:
        default_value = default_values[property_]
        property_data = np.ones([len(stations),len(times)])*default_value
        property_da = xr.DataArray(property_data,coords={'STATION':stations, 'TIME':times})
        hydrodynamic_data[property_] = property_da
    return hydrodynamic_data


def add_specific_environmental_data_on_node(hydrodynamic_dataset, node, environmental_property, times, values):
    hydro_times = hydrodynamic_dataset.TIME.values
    if not isinstance(times[0],np.datetime64):
        times = [np.datetime64(time) for time in times]

    times_align = (times[0] <= hydro_times[0]) & (times[-1] >= hydro_times[-1])
    if not times_align:
        raise ValueError("Times do not cover the spectrum of the dataset")

    times_in_common, values_in_times, _ = np.intersect1d(np.array(times), np.array(hydro_times), return_indices=True)
    values = values[values_in_times]
    dataarray = xr.DataArray([values], coords={"STATION":[node], "TIME":times_in_common})
    dataarray = dataarray.interp(TIME=hydrodynamic_dataset.TIME)
    dataarray = dataarray.squeeze("STATION")

    hydrodynamic_dataset[environmental_property].loc[dict(STATION=node, TIME=dataarray.TIME)] = dataarray.values
    return hydrodynamic_dataset


def add_specific_environmental_data(hydrodynamic_dataset, new_dataset):
    for environmental_property in list(new_dataset.data_vars):
        if environmental_property not in hydrodynamic_dataset.data_vars:
            continue

        data_interp = new_dataset[environmental_property].interp(TIME=hydrodynamic_dataset.TIME)

        # check if interpolation produced valid values
        if data_interp.isnull().all():
            warnings.warn(f"No matching TIME overlap for '{environmental_property}'. "
                          "Original hydrodynamic values retained.")
            continue

        hydro_var = hydrodynamic_dataset[environmental_property]

        hydrodynamic_dataset[environmental_property].loc[
            dict(STATION=data_interp.STATION, TIME=data_interp.TIME)
        ] = np.where(
            np.isnan(data_interp.values),
            hydro_var.loc[dict(STATION=data_interp.STATION, TIME=data_interp.TIME)].values,
            data_interp.values
        )

    return hydrodynamic_dataset


def interpolate_data_on_route(hydrodynamic_data, route, graph, variables=None):
    if variables is None:
        variables = list(hydrodynamic_data.data_vars)

    node_start = route[0]
    node_end = route[-1]
    total_distance_along_route = 0
    distances_along_route = [total_distance_along_route]
    for edge in zip(route[:-1], route[1:]):
        total_distance_along_route += get_length_of_edge(graph, edge)
        distances_along_route.append(total_distance_along_route)

    distances = np.array(distances_along_route)
    fraction_distances = distances / distances[-1]
    fraction_distances_da = xr.DataArray(fraction_distances, dims="route")

    for environmental_property in variables:
        data_start = hydrodynamic_data[environmental_property].sel({'STATION': node_start})
        data_end = hydrodynamic_data[environmental_property].sel({'STATION': node_end})
        interpolated_data = data_start + (data_end - data_start) * fraction_distances_da
        for i, node in enumerate(route):
            hydrodynamic_data[environmental_property].loc[dict(STATION=node)] = interpolated_data.isel(route=i)

    return hydrodynamic_data


def add_lonlat_to_xr_dataset(ds, x="X", y="Y", epsg_in=None):
    if "STATION" not in list(ds.coords):
        raise ValueError('The dataset does not have a "STATION"-coordinate')
    epgs_in_ds = ("EPSG" in list(ds.coords))
    if epsg_in is None and not epgs_in_ds:
        raise ValueError('The dataset does not have a "EPSG"-coordinate, while no epsg_in-parameter has been given to the function')

    lon = np.empty(ds.sizes["STATION"])
    lat = np.empty(ds.sizes["STATION"])

    for i in range(ds.sizes["STATION"]):
        if epgs_in_ds:
            epsg_code = ds["EPSG"].isel(STATION=i).item()
            if not isinstance(epsg_code, str):
                epsg_in = f"EPSG:{epsg_code}"
            elif "EPSG" not in epsg_code:
                epsg_in = "EPSG:"+epsg_code
            else:
                epsg_in = epsg_code

        transformer = Transformer.from_crs(epsg_in, "EPSG:4326", always_xy=True)
        x_val = ds[x].isel(STATION=i).item()
        y_val = ds[y].isel(STATION=i).item()
        lon[i], lat[i] = transformer.transform(x_val, y_val)

    ds = ds.assign_coords(LON=("STATION", lon),LAT=("STATION", lat))
    return ds


def add_closest_node_to_xr_dataset(ds, graph, lon="LON", lat="LAT"):
    closest_node_per_location = []
    for i in range(ds.sizes["STATION"]):
        lon_val = ds[lon].isel(STATION=i).item()
        lat_val = ds[lat].isel(STATION=i).item()
        point = Point(lon_val,lat_val)
        closest_node_per_location.append(get_closest_node_to_geometry(graph, point))
    stations = ds["STATION"].values
    ds = ds.assign_coords(NAME=("STATION", stations))
    ds["STATION"] = closest_node_per_location
    return ds


def overwrite_data_on_node_with_data_from_another_node(ds, source_station, target_station, variables=None):
    """
    Copy data from one station to another in an xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        The dataset containing station data.
    source_station : str
        Name of the station to copy data from.
    target_station : str
        Name of the station to overwrite.
    variables : list of str, optional
        List of variables to copy. If None, all variables are copied.

    Returns:
    --------
    xr.Dataset
        Dataset with updated target station data.
    """
    # Determine which variables to copy
    if variables is None:
        variables = list(ds.data_vars)

    # Check if the stations exist
    if source_station not in ds.STATION.values:
        raise ValueError(f"Source station '{source_station}' not found in dataset.")
    if target_station not in ds.STATION.values:
        raise ValueError(f"Target station '{target_station}' not found in dataset.")

    # Copy data
    for var in variables:
        if var not in ds.data_vars:
            raise ValueError(f"Variable '{var}' not found in dataset.")
        ds[var].loc[dict(STATION=target_station)] = ds[var].sel(STATION=source_station)

    return ds


def set_station_value(ds, station, variable, value):
    """
    Replace all data of a variable at a specific station with a single value.

    Parameters:
    -----------
    ds : xr.Dataset
        The xarray Dataset.
    station : str
        The station name where the data should be replaced.
    variable : str
        The variable name to modify.
    value : numeric or str
        The value to set.

    Returns:
    --------
    xr.Dataset
        Dataset with updated values.
    """
    if variable not in ds.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset.")
    if station not in ds.STATION.values:
        raise ValueError(f"Station '{station}' not found in dataset.")

    # Set the value
    ds[variable].loc[dict(STATION=station)] = value

    return ds
