import pandas as pd
import numpy as np
import datetime
import xarray as xr

from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager

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
