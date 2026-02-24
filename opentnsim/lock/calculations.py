"""This module contains functions to perform calculations for lock operations."""
import math
import networkx as nx
import numpy as np
import pandas as pd
import pyzsf
from shapely.geometry import Point, Polygon
import datetime
from opentnsim.utils import time_to_numpy
from opentnsim.constants import gravitational_acceleration
from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from opentnsim.graph.calculations import transform_geometry
from opentnsim.graph.utils import (
    get_sailing_information_on_edge_to_distance_on_another_edge,
    get_sailing_time,
    check_graph_is_multidigraph_type,
    get_edge,
    get_length_of_edge,
    get_geometry_of_edge,
)
from opentnsim.lock.utils import (
    _get_waiting_area,
    _get_lineup_area,
    _check_if_vessel_is_first_vessel,
    _get_vessel_sailing_in_speed,
    _get_vessel_sailing_out_speed,
    _get_vessel_sailing_speed_in_lock,
    _get_vessel_sailing_speed_out_lock,
    _get_distance_to_lock,
    _get_vessels_from_planned_operation,
    _get_first_vessel_of_lock_operation,
    _get_last_vessel_of_lock_operation,
    _get_route_to_lock,
    _get_water_levels_before_and_after_levelling,
    _get_information_for_lock_operation,
    _update_lock_vessel_planning,
    _update_lock_operation_planning,
    _get_lock_operation_to_and_from_node,
    _correct_lock_operation_start_time_if_outside_of_operational_hours,
    _update_vessel_planning_for_delayed_deparature,
    _update_future_lock_operations_by_lock_delay_previous_operation,
)


def calculate_z(
    t,
    t_start,
    direction,
    wlev_init,
    operation_index,
    operation_planning,
    start_node,
    end_node,
    node_open,
    epoch,
):
    # set default time and water level difference series
    z = np.zeros_like(t)

    # convert given t_start into np.datetime64 (this is required to communicate with the hydrodynamic data via the NetCDF package)
    t_start = time_to_numpy(t_start)

    hydromanager = HydrodynamicDataManager()
    # determine the actual water levels
    time_index = hydromanager._get_time_index_of_hydrodynamic_data(t_start)
    t_simulation_start = np.datetime64(epoch)
    H_A = hydromanager._get_hydrodynamic_data_series(t_simulation_start, start_node, "Water level")
    H_B = hydromanager._get_hydrodynamic_data_series(t_simulation_start, end_node, "Water level")
    H_A_init = H_A[time_index]
    H_B_init = H_B[time_index]

    if wlev_init is None:
        last_operations = operation_planning[operation_planning.index < operation_index]
        if not last_operations.empty:
            last_operation = last_operations.iloc[-1]
            last_operation_direction = last_operation.direction
            if not last_operation_direction:
                wlev_init = H_B_init
            else:
                wlev_init = H_A_init

        elif node_open == start_node:
            wlev_init = H_A_init
        else:
            wlev_init = H_B_init

    if not direction:
        z[0] = H_B_init - wlev_init

    else:
        z[0] = H_A_init - wlev_init

    return z, H_A, H_B


def levelling_time_equation(
    t,
    z,
    lock_length,
    lock_width,
    disch_coeff,
    gate_opening_time,
    opening_area,
    t_start,
    dt,
    direction,
    water_level_difference_limit_to_open_gate,
    H_A,
    H_B,
):
    """Calculates the levelling time of a lock operation based on Eq. 4.64 of Ports and Waterways Open Textbook (https://books.open.tudelft.nl/home/catalog/book/204)
    This function is called by get_levelling_time()
    Returns
    -------
    levelling_time : float
        the time duration of the levelling process
    t : list of float
        the time series of the levelling process
    z : list of float
        the water level difference series over the time of the levelling process
    """
    t_start = time_to_numpy(t_start)
    A_ch = lock_length * lock_width  # surface area of the lock chamber [m^2] (constant over time)
    m = disch_coeff  # discharge coefficient [-] (constant over time)
    g = gravitational_acceleration  # gravitational acceleration [m/(s^2)] (constant over time)
    T1 = gate_opening_time  # time to open the gate [s] (constant over time)
    A_s = np.linspace(0, opening_area, int(T1 / float(dt)))  # sluice opening area over time when opening [m^2] (time-dependent)
    A_s = np.append(A_s, [opening_area] * (len(z) - len(A_s)))  # sluice opening over full levelling process [m^2] (time-dependent)
    hydromanager = HydrodynamicDataManager()
    H_time = hydromanager.hydrodynamic_data['TIME'].values.astype(float)  # time series of the hydrodynamic data [s]

    # time-integration by (sefl-coded) Euler's method TODO Checken of we een standaard solver kunnen gebruiken. En of we dit algoritme los kunnen maken van de klasse.
    for i in range(len(t) - 1):
        H_Ai = np.interp(
            (np.timedelta64(int(i * float(dt) * 10**6), "us") + t_start - np.datetime64("1970-01-01")) / np.timedelta64(1, "us"),
            H_time,
            H_A,
        )  # water level at side A at time = i
        H_Aii = np.interp(
            (np.timedelta64(int((i + 1) * float(dt) * 10**6), "us") + t_start - np.datetime64("1970-01-01"))
            / np.timedelta64(1, "us"),
            H_time,
            H_A,
        )  # water level at side A at time = i + 1
        H_Bi = np.interp(
            (np.timedelta64(int(i * float(dt) * 10**6), "us") + t_start - np.datetime64("1970-01-01")) / np.timedelta64(1, "us"),
            H_time,
            H_B,
        )  # water level at side B at time = i
        H_Bii = np.interp(
            (np.timedelta64(int((i + 1) * float(dt) * 10**6), "us") + t_start - np.datetime64("1970-01-01"))
            / np.timedelta64(1, "us"),
            H_time,
            H_B,
        )  # water level at side B at time = i + 1
        deltaH_A = H_Aii - H_Ai  # water level difference at side A between time = i and time = i + 1
        deltaH_B = H_Bii - H_Bi  # water level difference at side B between time = i and time = i + 1

        # determine the contribution to the change in water level difference outside of the lock (i.e., due to tides) in the water level difference at time = i + 1
        if not direction:
            to_wlev_change = -deltaH_B
        else:
            to_wlev_change = -deltaH_A

        # calculate change in water level difference between time = i and time = i + 1
        z_i = abs(z[i])  # absolute water level difference at time = i

        dz_dt = -m * A_s[i] * np.sqrt(2 * g * np.max([0, z_i])) / A_ch  # change in water level difference over time [m/s]
        if z[i] < 0:  # correct if water level difference is negative
            dz_dt = -dz_dt
        dz = dz_dt * float(dt) + to_wlev_change

        # calculate the new water level difference at time = i + 1
        z[i + 1] = z[i] + dz
        if np.sign(z[i + 1]) != np.sign(z[i]):  # prevents overshooting of the water level difference
            z[i + 1] = 0

        if (
            np.abs(z[i + 1]) <= water_level_difference_limit_to_open_gate
        ):  # breaks the integration if the water level difference is smaller than a default 5 cm (the last 5 cm of water level difference takes long to overcome, so lock master opens gate)
            z[(i + 1) :] = np.nan  # set all next values of the water level series to nan
            break

    # determining levelling time based on the first nan of the series TODO: Class-functie maken _get_levelling_time()
    if len(np.argwhere(np.isnan(z))):
        levelling_time = t[np.argwhere(np.isnan(z))[0]][0]
    else:
        levelling_time = t[-1]

    return levelling_time, t, z


def calculate_levelling_time(lock_chamber, t_start, direction, wlev_init=None, operation_index=0, prediction=False):
    """
    Calculates the levelling time of a lock operation

    Parameters
    ----------
    t_start :
        the start time of the levelling process
    direction : int
        the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    wlev_init : float
        initial water level in the lock chamber
    same_direction : bool
        states if the levelling process is predicted in the same direction as the last lock operation (True) or not (False)
    prediction : bool
        states if the levelling process is only predicted (True) or executed (False)

    Returns
    -------
    levelling_time : float
        the time duration of the levelling process
    t : list of float
        the time series of the levelling process
    z : list of float
        the water level difference series over the time of the levelling process
    """
    # TODO: functie maken om tstart om te zetten (met _to_array)
    # TODO: Bij andere klasses altijd checken of iets een datetime.datetime is. En als dit niet zo is een error inbouwen of hem gelijk omzetten.
    dt = lock_chamber.time_step
    t_final = 3600  # maximum levelling time has been set to an hour
    t = np.arange(0, t_final + float(dt), float(dt))
    t_start = time_to_numpy(t_start)
    # if there is no hydrodynamic data included in the run, use the constant levelling time included in the lock object
    # if there is no hydrodynamic data included in the run, use the constant levelling time included in the lock object
    if not hasattr(lock_chamber.env,'hydrodynamics'):
        levelling_time = lock_chamber.levelling_time
        z = np.zeros(len(t))
        return levelling_time, t, z

    z, H_A, H_B = calculate_z(
        t=t,
        t_start=t_start,
        direction=direction,
        wlev_init=wlev_init,
        operation_index=operation_index,
        operation_planning=lock_chamber.lock_complex.operation_planning,
        start_node=lock_chamber.start_node,
        end_node=lock_chamber.end_node,
        node_open=lock_chamber.gate_open,
        epoch=lock_chamber.env.epoch,
    )

    # if a function has been included to predict the levelling time based on the water level difference: calculate the levelling time based on the initial water level difference
    if callable(lock_chamber.levelling_time):
        levelling_time = lock_chamber.levelling_time(z[0])
        return levelling_time, t, z

    # if no function has been included: compute the levelling time based on Eq. 4.64 of Ports and Waterways Open Textbook (https://books.open.tudelft.nl/home/catalog/book/204)
    levelling_time, t, z = levelling_time_equation(
        t=t,
        z=z,
        lock_length=lock_chamber.lock_length,
        lock_width=lock_chamber.lock_width,
        disch_coeff=lock_chamber.disch_coeff,
        gate_opening_time=lock_chamber.gate_opening_time,
        opening_area=lock_chamber.opening_area,
        t_start=t_start,
        dt=dt,
        direction=direction,
        water_level_difference_limit_to_open_gate=lock_chamber.water_level_difference_limit_to_open_gate,
        H_A=H_A,
        H_B=H_B,
    )

    # if this function was not ran as a prediction, but rather as the actual levelling event: update the water level time series of the lock chamber
    if not prediction:
        # TODO: de lock.water_level wordt niet gebruikt, maar is wel leuk om als logging terug te zien na een berekening. Nadenken of we dat zo willen laten, of anders willen bijhouden.
        t_final = t_start + np.timedelta64(int(levelling_time))
        hydromanager = HydrodynamicDataManager()
        t_index_final = hydromanager._get_time_index_of_hydrodynamic_data(t_final)
        if not direction:
            lock_chamber.water_level[t_index_final:] = H_B[t_index_final:].copy()
        else:
            lock_chamber.water_level[t_index_final:] = H_A[t_index_final:].copy()

    return levelling_time, t, z


def calculate_lock_operation_start_information(lock_chamber, vessel, operation_index, direction):
    """Determines the

    Parameters
    ----------
    vessel :

    operation_index :

    direction :

    :return:
    """
    # determine the index of the vessel in the lock master's vessel planning
    lock_complex = lock_chamber.lock_complex
    operation_planning = lock_complex.operation_planning

    # determine the time that the lock operation can start (operation perspective)
    time_lock_operation_start = calculate_lock_operation_start_time(lock_chamber, vessel, operation_index, direction)

    # correct the start time of the lock operation if it will fall outside of the operation hours of the lock complex
    time_lock_operation_start = _correct_lock_operation_start_time_if_outside_of_operational_hours(lock_chamber, time_lock_operation_start)

    # determine the time that vessel can start entering the lock
    time_lock_entry_start = calculate_lock_entry_start_time(lock_chamber, vessel, operation_index, direction, time_lock_operation_start)

    # determine the minimum time that gate should be opened in advance of a vessel arrival and add this to the vessel planning
    minimum_advance_to_open_gate = lock_chamber.minimum_advance_to_open_gate
    time_potential_lock_gate_opening_stop = time_lock_entry_start - minimum_advance_to_open_gate

    previous_planned_operations = operation_planning[operation_planning.index < operation_index]
    if not previous_planned_operations.empty:
        previous_operation = previous_planned_operations.iloc[-1]
        if not len(previous_operation.vessels):
            if time_potential_lock_gate_opening_stop < previous_operation.time_lock_operation_stop:
                operation_delay = previous_operation.time_lock_operation_stop - time_potential_lock_gate_opening_stop
                time_lock_operation_start += operation_delay
                time_lock_entry_start += operation_delay
                time_potential_lock_gate_opening_stop += operation_delay

    # determine the lock entry stop and gate opening stop time
    time_lock_entry_stop = calculate_lock_entry_stop_time(lock_chamber, vessel, operation_index, direction, time_lock_operation_start)

    # determine the delay time for the vessel to enter the lock
    delay = pd.Timedelta(seconds=0.)
    if vessel is not None:
        delay = calculate_sailing_in_time_delay(lock_chamber, vessel, operation_index, time_lock_entry_start)

    arrival_information = {"time_potential_lock_gate_opening_stop": time_potential_lock_gate_opening_stop + delay,
                           "time_lock_operation_start": time_lock_operation_start + delay,
                           "time_lock_entry_start": time_lock_entry_start + delay,
                           "time_lock_entry_stop": time_lock_entry_stop + delay}

    return arrival_information


def calculate_lock_departure_information(lock_chamber, vessel, operation_index, direction, levelling_information):
    time_lock_departure_start = calculate_vessel_departure_start_time(lock_chamber, vessel, operation_index, levelling_information["time_gate_opening_stop"])
    time_lock_departure_stop = calculate_vessel_departure_stop_time(lock_chamber, vessel, operation_index, levelling_information["time_gate_opening_stop"])
    time_lock_operation_stop = calculate_lock_operation_stop_time(lock_chamber, vessel, operation_index, direction, levelling_information["time_gate_opening_stop"])
    time_lock_gate_closing_start = calculate_lock_gate_closing_time(lock_chamber, vessel, operation_index, levelling_information["time_gate_opening_stop"])
    departure_information = {"time_lock_departure_start":time_lock_departure_start,
                             "time_lock_departure_stop":time_lock_departure_stop,
                             "time_lock_operation_stop":time_lock_operation_stop,
                             "time_lock_gate_closing_start":time_lock_gate_closing_start,
                             "time_potential_lock_gate_closure_start": time_lock_gate_closing_start,}
    return departure_information


def calculate_sailing_information_on_route_to_lock_complex(lock_complex, vessel, lock_end_node):
    """
    Calculates the sailing information (i.e., duration, distance, and speed) of the vessel per edge of its route between its current location and the lock gate

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    lock_end_node : str
        the node name that forms the end node of the lock complex given the direction of the vessel

    Returns
    -------
    sailing_time : pd.DataFrame
        sailing information (i.e., duration, distance, and speed) per edge of the route of the vessel between its current location and the lock gate
    """

    # unpacks the logbook of the vessel
    vessel_df = pd.DataFrame(vessel.logbook)
    if vessel_df.empty:
        return pd.DataFrame()

    # determine the sailing time already based on the current edge (if registration node is not coupled to node, but instead is somewhere along the edge: not sure if this is already implemented)
    current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
    reversed_vessel_df = vessel_df.iloc[::-1]
    for index,message in reversed_vessel_df.iterrows():
        if 'node' in message.Message:
            break
    passed_time = (current_time - message.Timestamp).total_seconds()

    # determines the distance from the node of the edge to the lock gate (depending on the direction of the vessel)
    distance = lock_complex.distance_from_start_node_to_lock_gate_A
    if lock_end_node != lock_complex.end_node:
        distance = lock_complex.distance_from_end_node_to_lock_gate_B

    # determine the sailing time from its current node to the end of the lock complex (depending on the direction of the vessel)
    route_vessel = vessel.route_ahead
    route_index_current_node = route_vessel.index(vessel.current_node)
    route_index_end_of_lock_complex = route_vessel.index(lock_end_node)
    route_vessel_to_pass_lock_complex = route_vessel[route_index_current_node:route_index_end_of_lock_complex]
    _, sailing_information = get_sailing_time(vessel, route_vessel_to_pass_lock_complex) #TODO: maybe rename this function in the VTS, because it provides a dataframe of the sailing information (i.e., time, speed, and distance) per edge over the route of the vessel

    # correct the sailing time at the lock complex edge to the distance on that edge from the node to the lock gate (depending on the direction of the vessel)
    last_sailing_index = sailing_information.iloc[-1].index
    sailing_information.loc[last_sailing_index, 'distance'] = distance
    sailing_information.loc[last_sailing_index, 'time'] = distance / sailing_information.loc[last_sailing_index, 'speed']

    # if there are overruled speeds implemented, correct the above speeds and sailing times
    if not vessel.overruled_speed.empty:
        for edge, overruled_speed in vessel.overruled_speed.iterrows():
            edge_index_mask = sailing_information.index == edge
            sailing_information.loc[edge_index_mask, 'speed'] = overruled_speed.speed
            sailing_information.loc[edge_index_mask, 'time'] = sailing_information.loc[edge_index_mask, 'distance'] / sailing_information.loc[edge_index_mask, 'speed']

    # determine the index of the first edge in the sailing time dataframe to correct the sailing distance and sailing time of this edge with the already passed time and passed distance by this ship over this edge
    index_sailing_on_first_edge = (sailing_information[sailing_information.index.isin([(vessel.current_node, route_vessel_to_pass_lock_complex[1], 0)])].iloc[0].name)
    index_mask = sailing_information.index == index_sailing_on_first_edge
    interpolation = 1 - passed_time / sailing_information.loc[index_mask].Time
    sailing_information.loc[sailing_information[index_mask].index, 'distance'] = sailing_information.loc[sailing_information[index_mask].index, 'distance'] * interpolation
    sailing_information.loc[sailing_information[index_mask].index, 'time'] = sailing_information.loc[sailing_information[index_mask].index, 'time'] * interpolation
    sailing_information['speed'] = sailing_information['speed'].astype(float)

    return sailing_information

def calculate_sailing_time_to_waiting_area(waiting_area, vessel):
    """TODO: note that this function looks a lot like other 'calculate_sailing_time_to'-functions below, so maybe we can investigate to combine the functions
    Calculates the sailing time of a vessel from its location to the waiting area

    Parameters
    -------------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput

    Returns
    -------
    sailing_to_waiting_area_time: pd.Timedelta
        sailing time to the waiting area in [s]
    sailing_distance: float
        sailing distance to the waiting area in [m]
    average_sailing_speed: float
        average sailing speed to the lock chambers's waiting area in [m/s]

    """
    if vessel is None:
        return 0., _, _
    # determine route to the start node of the edge at which the waiting area is located
    route_to_waiting_area = nx.dijkstra_path(vessel.env.graph, vessel.current_node, waiting_area.edge[1]) #TODO: check this -> waiting area should be assigned to vessel

    # determine the distance that the vessel has to sail on the edge at which the waiting area is located (from the start node of the edge)
    distance_to_waiting_area_on_last_edge = waiting_area.distance_from_edge_start

    # calculation of the sailing information (time, distance, speed) per edge on route to the waiting area
    sailing_to_waiting_area = get_sailing_information_on_edge_to_distance_on_another_edge(vessel, route_to_waiting_area, 0., distance_to_waiting_area_on_last_edge)

    # calculation of the sailing time, distance, and average speed to the waiting area
    sailing_to_waiting_area_time = pd.Timedelta(seconds=sailing_to_waiting_area['time'].sum())
    sailing_distance = sailing_to_waiting_area['distance'].sum()
    average_sailing_speed = sailing_to_waiting_area['speed']
    if sailing_to_waiting_area_time.total_seconds():
        average_sailing_speed = sailing_distance / sailing_to_waiting_area['time'].sum()

    return sailing_to_waiting_area_time, sailing_distance, average_sailing_speed

def calculate_sailing_time_to_lineup_area(lineup_area, vessel):
    """
    Calculates the sailing time of a vessel from its location to the line-up area

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput

    Returns
    -------
    sailing_to_lineup_area_time : pd.Timedelta
        sailing time to the lock chambers's line-up area in [s]

    """
    if vessel is None:
        return pd.Timedelta(seconds = 0.)

    current_node = vessel.current_node
    # unpack first encountered line-up area
    if lineup_area is None:
        return pd.Timedelta('NaT')

    # determine the route of the vessel to the line-up area edge
    route_to_lineup_area = nx.dijkstra_path(lock_complex.env.graph, current_node, lineup_area_approach.end_node)

    # determine the distance that the vessel has to sail on the edge at which the line-up area is located (from the start node of the edge)
    distance_to_lineup_area_from_last_node = lineup_area_approach.distance_from_start_edge

    # calculation of the sailing information (time, distance, speed) per edge on route to the line-up area
    sailing_to_lineup_area = get_sailing_information_on_edge_to_distance_on_another_edge(vessel, route_to_lineup_area, 0., distance_to_lineup_area_from_last_node)

    # calculation of the sailing time to the line-up area
    sailing_to_lineup_area_time = pd.Timedelta(seconds=sailing_to_lineup_area['time'].sum())

    return sailing_to_lineup_area_time

def calculate_sailing_time_to_approach_point(lock_chamber, vessel, direction):
    """
    Calculates the sailing time of a vessel from its location to the approach point

    The approach point is the closest location in front of the lock gate where the outdirection vessel(s) can pass the indirection vessel waiting to enter the lock.
    The point is located in between the line-up area and the lock gate.

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    direction : int
        the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    current_node : str
        the node name (which has to be in the graph) at which the vessel is currently sailing

    Returns
    -------
    sailing_to_lineup_area_time : pd.Timedelta
        sailing time to the lock chambers's line-up area in [s]

    """
    if vessel is None:
        return pd.Timedelta(seconds=0)
    # unpack sailing distance from crossing point to lock gate
    sailing_distance_from_entry = lock_chamber.sailing_distance_to_crossing_point

    # determine the time of entering the lock
    sailing_speed_during_entry = _get_vessel_sailing_in_speed(lock_chamber, vessel, direction)
    sailing_time_entry = pd.Timedelta(seconds=sailing_distance_from_entry / sailing_speed_during_entry)

    # determine the time of the vessel to its first encountered waiting area and lock_gate
    sailing_time_to_waiting_area = pd.Timedelta(seconds=0)
    sailing_time_to_lock_gate = calculate_sailing_time_to_lock_gate(lock_chamber, vessel, direction)

    # determine the sailing time to the approach point
    sailing_time_to_start_approach = sailing_time_to_lock_gate - sailing_time_entry - sailing_time_to_waiting_area
    return sailing_time_to_start_approach

def calculate_delay_until_arrival_within_operational_hours(lock_complex, time_sailing_to_lock_start):
    operational_hours = lock_complex.operational_hours
    within_operation_hours = operational_hours[(time_sailing_to_lock_start >= operational_hours.start_time) &
                                               (time_sailing_to_lock_start <= operational_hours.stop_time)]
    delay = pd.Timedelta(seconds=0)
    if within_operation_hours.empty:
        first_available_hour = operational_hours[operational_hours.start_time >= time_sailing_to_lock_start].iloc[0]
        delay = first_available_hour.start_time - time_sailing_to_lock_start
    return delay


def calculate_sailing_time_to_lock_gate(lock_chamber, vessel, direction):
    """
    Calculates the sailing time of a vessel from its location to the first lock gate that it will encounter

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    direction : int
        the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Returns
    -------
    sailing_to_lineup_area_time : pd.Timedelta
        sailing time to the lock chambers's line-up area in [s]

    """
    if vessel is None:
        return pd.Timedelta(seconds=0.)
    # determine the end node of the lock complex from the perspective of the vessel and the distance from the start node of the lock complex to the lock gate
    distance_to_lock = _get_distance_to_lock(lock_chamber, direction)

    # determine the route of the vessel to the end node of the lock complex from the perspective of the vessel
    route_to_lock_chamber = _get_route_to_lock(vessel, lock_chamber)

    # calculate sailing time to the start node of the edge of lock complex from the perspective of the vessel
    _, sailing_to_lock_chamber = get_sailing_time(vessel, route_to_lock_chamber)
    sailing_to_lock_chamber_distance = sailing_to_lock_chamber['distance'].sum()
    sailing_to_lock_chamber_time = sailing_to_lock_chamber['time'].sum()

    # add sailing distance and time to the lock gate on the edge of the lock complex to sailing information to the start node of this edge
    sailing_to_lock_chamber_distance += distance_to_lock
    sailing_to_lock_chamber_time += distance_to_lock / _get_vessel_sailing_in_speed(lock_chamber, vessel, direction)
    sailing_to_lock_chamber_time = pd.Timedelta(seconds=sailing_to_lock_chamber_time)

    return sailing_to_lock_chamber_time


def calculate_sailing_time_to_pass_lock(lock_chamber, vessel):
    if vessel is None:
        return pd.Timedelta(seconds=0.)

    route_to_lock_chamber = _get_route_to_lock(vessel, lock_chamber, True)

    # calculate sailing time to the start node of the edge of lock complex from the perspective of the vessel
    _, sailing_to_lock_chamber = get_sailing_time(vessel, route_to_lock_chamber)
    sailing_to_lock_chamber_time = sailing_to_lock_chamber['time'].sum()

    return pd.Timedelta(seconds=sailing_to_lock_chamber_time)


def calculate_sailing_time_in_lock(lock_chamber, vessel, operation_index):
    """
    Calculates the time duration that a vessel needs to enter the lock until laying still

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Returns
    -------
    sailing_time_into_lock : pd.Timedelta
        the time duration of the process of sail in the lock [s]

    """
    # determine the vessels assigned to the lock operation (that are already in the lock)
    lock_complex = lock_chamber.lock_complex
    vessels = _get_vessels_from_planned_operation(lock_complex, operation_index=operation_index)

    # determine the sailing distance from the lock gate to the position assigned to the vessel
    vessel_index = vessels.index(vessel)
    sailing_distance_from_lock_gate = (lock_chamber.lock_length - np.sum([vessel.L for vessel in vessels[:vessel_index]])) - 0.5 * vessel.L


    # determine the sailing speed of the vessel in the lock
    sailing_speed_into_lock = _get_vessel_sailing_speed_in_lock(lock_chamber, vessel)

    # calculate the time required to complete the process of sailing from the lock gate to laying still in the lock chamber on the assigned longitudinal coordinate (x)
    sailing_time_into_lock = pd.Timedelta(seconds=sailing_distance_from_lock_gate / sailing_speed_into_lock)

    return sailing_time_into_lock


def calculate_sailing_in_time_delay(lock_chamber, vessel, operation_index, time_lock_entry_start = None):
    """
    Calculates the minimum required time gap between two entering vessels for safety, resulting in a delay

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        the index of the lock operation in the operation planning dataframe
    direction : int
        the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    minimum_difference_with_previous_vessel : bool
        .

    Returns
    -------
    sailing_in_time_delay : pd.Timedelta
        time delay because of waiting for the vessel to sail entering the lock [s]

    """

    delay_to_entry = pd.Timedelta(seconds=0)
    if vessel is None:
        return delay_to_entry

    lock_complex = lock_chamber.lock_complex
    vessel_planning = lock_complex.vessel_planning
    vessels = _get_vessels_from_planned_operation(lock_complex, operation_index=operation_index)
    vessel_index = vessels.index(vessel)

    previous_vessels = vessels[:(vessel_index + 1)]
    for vessel_0, vessel_1 in zip(previous_vessels[:-1], previous_vessels[1:]):
        index_v0 = vessel_planning[vessel_planning.id == vessel_0.id].iloc[0].name
        index_v1 = vessel_planning[vessel_planning.id == vessel_1.id].iloc[0].name
        sailing_in_start_v0 = vessel_planning.loc[index_v0, 'time_lock_entry_start']
        sailing_in_start_v1 = vessel_planning.loc[index_v1, 'time_lock_entry_start']
        if time_lock_entry_start is not None and vessel_1.id == vessel.id:
            sailing_in_start_v1 = time_lock_entry_start
        sailing_in_gap = sailing_in_start_v1 - sailing_in_start_v0
        if sailing_in_gap < lock_chamber.sailing_in_time_gap_through_gate:
            delay_to_entry += lock_chamber.sailing_in_time_gap_through_gate - sailing_in_gap
    return delay_to_entry

def calculate_vessel_entry_duration(lock_chamber, vessel, direction):
    """
    Calculates the time duration required for a vessel starts entering the lock (from approach point to first encountered lock gate)

    Parameters
    ----------
    vessel : type [optional]
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    direction : int
        the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Returns
    -------
    sailing_time_entry : pd.Timedelta
        the time duration required for a vessel starts entering the lock [s]

    """
    # determine the distance from the lock gate to the approach point
    if vessel is None:
        return pd.Timedelta(seconds=0.)
    sailing_distance_from_entry = lock_chamber.sailing_distance_to_crossing_point

    # determine the vessel speed when entering the lock
    sailing_speed_during_entry = _get_vessel_sailing_in_speed(lock_chamber, vessel, direction)

    # determine the time of the process of entering
    sailing_time_entry = pd.Timedelta(seconds=sailing_distance_from_entry / sailing_speed_during_entry)

    return sailing_time_entry

def calculate_vessel_passing_start_time(lock_chamber, vessel, operation_index, direction):
    """
    Calculates the start time that a vessel can start its manoeuvre of entering the lock

    Parameters
    ----------
    vessel : type [optional]
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        the index of the lock operation in the operation planning of the lock complex master
    direction : int
        the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Returns
    -------
    vessel_passing_start_timestamp : pd.Timestamp
        the moment in time that a vessel starts entering the lock from the approach point

    """
    # determines the current time
    current_time = pd.Timestamp(datetime.datetime.fromtimestamp(lock_chamber.env.now))

    # calculate the sailing time durations to the lock gate, the approach point and if there is any form of delay for this
    sailing_time_to_lock = calculate_sailing_time_to_lock_gate(lock_chamber, vessel, direction)
    sailing_time_entry = calculate_vessel_entry_duration(lock_chamber, vessel, direction)
    sailing_in_delay = calculate_sailing_in_time_delay(lock_chamber, vessel, operation_index)

    # calculate time that the vessel can start passing the lock
    vessel_passing_start_timestamp = current_time + (sailing_time_to_lock - sailing_time_entry) + sailing_in_delay

    return vessel_passing_start_timestamp

def calculate_lock_operation_start_time(lock_chamber, vessel, operation_index, direction):
    """
    Calculates the new earliest possible start time of a lock operation

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        the index of the lock operation in the operation planning of the lock complex master
    direction : int
        the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Returns
    -------
    lock_operation_start_time : pd.Timestamp
        the moment in time of the start of the lock operation

    """
    # unpacks the lock complex master's operation planning
    lock_complex = lock_chamber.lock_complex
    operation_planning = lock_complex.operation_planning

    # determines the lock operation start time based on the first vessel that was assigned to this lock operation
    first_vessel = _get_first_vessel_of_lock_operation(lock_complex, vessel, operation_index)
    lock_operation_start_time = calculate_vessel_passing_start_time(lock_chamber, first_vessel, operation_index, direction)

    # determines the lock_operation_start_time based on whether it fits given the previous lock operations (should not be overlapping)
    previous_operations = operation_planning[operation_planning.index < operation_index]
    if not previous_operations.empty:
        previous_operation = previous_operations.iloc[-1]
        previous_lock_operation_stop_time = previous_operation.time_lock_operation_stop
        if lock_operation_start_time < previous_lock_operation_stop_time:
            lock_operation_start_time = previous_lock_operation_stop_time

    return lock_operation_start_time

def calculate_lock_gate_opening_time(lock_chamber, vessel, operation_index, direction, operation_start_time):
    """
    Calculates the time at which the lock gate can open before an vessel arrival

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        the index of the lock operation in the operation planning of the lock complex master
    direction : int
        the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Returns
    -------
    lock_entry_start_time : pd.Timestamp
        the time at which the lock gate can open before an vessel arrival
    """
    lock_complex = lock_chamber.lock_complex
    first_vessel = _get_first_vessel_of_lock_operation(lock_complex, vessel, operation_index)
    lock_entry_start_duration = calculate_vessel_entry_duration(lock_chamber, first_vessel, direction)
    lock_entry_start_duration -= lock_chamber.minimum_advance_to_open_gate
    lock_entry_start_time = lock_entry_start_duration + operation_start_time
    return lock_entry_start_time

def calculate_lock_entry_start_time(lock_chamber, vessel, operation_index, direction, operation_start_time):
    """
    Calculates the time at which the vessel can start sailing into to lock

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        the index of the lock operation in the operation planning of the lock complex master
    direction : int
        the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Returns
    -------
    lock_entry_start_time : pd.Timestamp
        the time at which the vessel can start sailing into to lock
    """
    lock_complex = lock_chamber.lock_complex
    first_vessel = _get_first_vessel_of_lock_operation(lock_complex, vessel, operation_index)
    lock_entry_start_duration = calculate_vessel_entry_duration(lock_chamber, first_vessel, direction)
    lock_entry_start_time = lock_entry_start_duration + operation_start_time
    return lock_entry_start_time

def calculate_vessel_entry_stop_time(lock_chamber, vessel, operation_index, direction):
    """
    Calculates the moment in time that a vessel finished its lock entry process

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Returns
    -------
    vessel_entry_stop_time : pd.Timestamp
         the moment in time that the vessel stops entering the lock
    """

    # determine the moment in time that the vessel starts to enter the lock
    vessel_entry_start_time = calculate_vessel_entry_duration(lock_chamber, vessel, direction)

    # determine the time duration of the vessel in the lock
    sailing_time_in_lock = calculate_sailing_time_in_lock(lock_chamber, vessel, operation_index)

    # calculate the moment in time that the vessel stops entering the lock
    vessel_entry_stop_time = vessel_entry_start_time + sailing_time_in_lock

    return vessel_entry_stop_time


def calculate_lock_entry_stop_time(lock_chamber, vessel, operation_index, direction, lock_entry_start_time):
    """
    Calculates the moment in time that a lock operation entry process of a lock operation is finished (all vessels are in lock chamber)

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Returns
    -------
    lock_entry_stop_time : pd.Timestamp
        the time that a lock operation entry process is finished
    """

    # determine the last assigned vessel of the lock operation to determine the lock entry stop time
    if vessel is None:
        return lock_entry_start_time
    lock_complex = lock_chamber.lock_complex
    last_vessel = _get_last_vessel_of_lock_operation(lock_complex, operation_index)
    lock_entry_stop_duration = calculate_vessel_entry_stop_time(lock_chamber, last_vessel, operation_index, direction)
    lock_entry_stop_time = lock_entry_stop_duration + lock_entry_start_time
    return lock_entry_stop_time

def calculate_lock_operation_times(lock_chamber, operation_index, start_time, vessel = None, direction=None):
    """
    Calculates the moments in time of the start and stop of the operation steps of the lock: (1) gate closing, (2) levelling, (3) gate opening

    Parameters
    ----------
    operation_index : int
        the index of the lock operation in the operation planning of the lock complex master
    last_entering_time : pd.Timestamp
        the time that the last vessel entered the lock
    start_time : pd.Timestamp
        the start time of the lock operation (i.e., for the gate to close)
    vessel : type [optional]
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    direction : int [optional]
        the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Returns
    -------
    levelling_information : dict
        including:
            time_gate_closing_start : pd.Timestamp
                the time that the lock gate are planned to start closing
            time_gate_closing_stop : pd.Timestamp
                the time that the lock gate are planned to stop closing
            time_levelling_start : pd.Timestamp
                the time that the lock chamber is planned to start levelling
            time_levelling_stop : pd.Timestamp
                the time that the lock chamber is planned to stop levelling
            time_gate_opening_start : pd.Timestamp
                the time that the lock gate are planned to start opening
            time_gate_opening_stop : pd.Timestamp
                the time that the lock gate are planned to stop opening
    """
    # unpack the lock complex master's vessel and operation plannings
    lock_complex = lock_chamber.lock_complex
    vessel_planning = lock_complex.vessel_planning
    operation_planning = lock_complex.operation_planning
    try:
        vessel_goes_with_previous_operation = start_time < operation_planning.loc[operation_index, "time_gate_closing_start"]
    except:
        vessel_goes_with_previous_operation = False

    if vessel_goes_with_previous_operation:
        time_gate_closing_start = operation_planning.loc[operation_index, "time_gate_closing_start"]
        time_gate_closing_stop = operation_planning.loc[operation_index, "time_gate_closing_stop"]
        time_levelling_start = operation_planning.loc[operation_index, "time_levelling_start"]
        time_levelling_stop = operation_planning.loc[operation_index, "time_levelling_stop"]
        time_gate_opening_start = operation_planning.loc[operation_index, "time_gate_opening_start"]
        time_gate_opening_stop = operation_planning.loc[operation_index, "time_gate_opening_stop"]

    else:
        # set default time gate closing start as start time
        time_gate_closing_start = start_time

        # overwrite the time gate closing start if there is a rule that the gate can close before a vessel is laying still and there are vessels in the lock
        # if lock_chamber.close_gate_before_vessel_is_laying_still and vessel is not None:
        #     time_gate_closing_start = last_entering_time + lock_chamber.minimum_delay_to_close_gate

        # determine the new closing stop times of the gate and the time that the levelling can hence start
        time_gate_closing_stop = time_gate_closing_start + pd.Timedelta(seconds=lock_chamber.gate_closing_time)
        time_levelling_start = time_gate_closing_stop

        # overwrite the time of levelling start if there is a rule that the gate can close before a vessel is laying still and there are vessels in the lock (the vessel always has to lay still before levelling can start)
        if lock_chamber.close_gate_before_vessel_is_laying_still and vessel is not None:
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            if not isinstance(vessel_planning.loc[vessel_planning_index,'time_lock_entry_stop'],float):
                time_levelling_start = np.max([vessel_planning.loc[vessel_planning_index,'time_lock_entry_stop'],time_levelling_start])
            else:
                time_levelling_start = time_levelling_start

        # determine levelling stop time and gate opening start and stop times
        time_levelling_stop,_,_ = calculate_levelling_time(lock_chamber, t_start=time_levelling_start, operation_index=operation_index, direction=direction, prediction=True)
        time_levelling_stop = time_levelling_start + pd.Timedelta(seconds=time_levelling_stop)
        time_gate_opening_start = time_levelling_stop
        time_gate_opening_stop = time_levelling_stop + pd.Timedelta(seconds=lock_chamber.gate_opening_time)

    wlev_A, wlev_B = _get_water_levels_before_and_after_levelling(lock_chamber, time_levelling_start, time_levelling_stop, direction)

    levelling_information = {"time_gate_closing_start":time_gate_closing_start,
                             "time_gate_closing_stop":time_gate_closing_stop,
                             "time_levelling_start":time_levelling_start,
                             "time_levelling_stop":time_levelling_stop,
                             "time_gate_opening_start":time_gate_opening_start,
                             "time_gate_opening_stop":time_gate_opening_stop,
                             "wlev_A":wlev_A,
                             "wlev_B":wlev_B}
    return levelling_information


def calculate_vessel_departure_start_delay(lock_chamber, vessel, operation_index, operation_stop_time):
    """
    Calculates the delay for a vessel to start leaving the lock

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Returns
    -------
    delay_to_departure : pd.Timestamp
        the delay of a vessel to start its departure process out of the lock
    """
    delay_to_departure = pd.Timedelta(seconds=0)
    if vessel is None:
        return delay_to_departure

    lock_complex = lock_chamber.lock_complex
    vessel_planning = lock_complex.vessel_planning
    vessels = _get_vessels_from_planned_operation(lock_complex, operation_index=operation_index)
    vessel_index = vessels.index(vessel)

    previous_vessels = vessels[:(vessel_index+1)]
    for vessel_0, vessel_1 in zip(previous_vessels[:-1],previous_vessels[1:]):
        index_v0 = vessel_planning[vessel_planning.id == vessel_0.id].iloc[0].name
        index_v1 = vessel_planning[vessel_planning.id == vessel_1.id].iloc[0].name
        sailing_out_start_v0 = vessel_planning.loc[index_v0, 'time_lock_departure_start']
        sailing_out_stop_v0 = vessel_planning.loc[index_v0, 'time_lock_departure_stop']
        sailing_out_stop_v1 = vessel_planning.loc[index_v1, 'time_lock_departure_stop']
        if vessel_1.id == vessel.id:
            sailing_time_out_of_lock = calculate_vessel_sailing_time_out_of_lock(lock_chamber, vessel, operation_index)
            sailing_out_stop_v1 = operation_stop_time + sailing_time_out_of_lock
            #delay_to_departure = operation_stop_time

        sailing_out_gap = sailing_out_stop_v1 - sailing_out_stop_v0
        if sailing_out_gap < lock_chamber.sailing_out_time_gap_through_gate:
            delay_to_departure += lock_chamber.sailing_out_time_gap_through_gate - sailing_out_gap
    return delay_to_departure


def calculate_vessel_departure_start_time(lock_chamber, vessel, operation_index, operation_stop_time):
    """
    Calculates the moment in time that a vessel can start leaving the lock

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Returns
    -------
    departure_start_time : pd.Timestamp
        the time that a vessel's departure process out of the lock can start
    """
    delay_to_departure = calculate_vessel_departure_start_delay(lock_chamber, vessel, operation_index, operation_stop_time)
    departure_start_time = operation_stop_time + delay_to_departure
    return departure_start_time


def calculate_lock_departure_start_time(lock_chamber, vessel, operation_index, operation_stop_time):
    """
    Calculates the moment in time the departure can start of a lock operation

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    operation_stop_time : pd.Timestamp
        the time that the lock operation has stopped (i.e., gate have been opened again)

    Returns
    -------
    departure_start_time : pd.Timestamp
        the time that a lock operation's departure process can start
    """
    lock_complex = lock_chamber.lock_complex
    first_vessel = _get_first_vessel_of_lock_operation(lock_complex, vessel, operation_index)
    time_departure_start = calculate_vessel_departure_start_time(lock_chamber, first_vessel, operation_index, operation_stop_time)
    return time_departure_start


def calculate_vessel_sailing_time_out_of_lock(lock_chamber, vessel, operation_index):
    """
    Calculates the sailing time for a vessel to sail from its position in the lock to the lock gate that have to be passed to sail out of the lock

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation

    Returns
    -------
    departure_start_time : pd.Timestamp
        the time that a lock operation's departure process can start
    """
    lock_complex = lock_chamber.lock_complex
    vessels = _get_vessels_from_planned_operation(lock_complex, operation_index=operation_index,)
    vessel_index = vessels.index(vessel)
    distance_to_lock = np.sum([vessel.L for vessel in vessels[:vessel_index]]) + 0.5 * vessel.L
    vessel_speed = _get_vessel_sailing_speed_out_lock(lock_chamber, vessel)
    sailing_out_time = pd.Timedelta(seconds=0.)
    if vessel_speed:
        sailing_out_time = pd.Timedelta(seconds=distance_to_lock / vessel_speed)
    return sailing_out_time


def calculate_vessel_departure_stop_time(lock_chamber, vessel, operation_index, operation_stop_time):
    """
    Calculates the moment in time the departure process of a vessel stops

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    operation_stop_time : pd.Timestamp
        the time that the lock operation has stopped (i.e., gate have been opened again)

    Returns
    -------
    time_departure_stop : pd.Timestamp
        the moment in time that a vessel's departure process stops
    """
    if vessel is None:
        return operation_stop_time
    time_departure_start = calculate_vessel_departure_start_time(lock_chamber, vessel, operation_index, operation_stop_time)
    sailing_out_time = calculate_vessel_sailing_time_out_of_lock(lock_chamber, vessel, operation_index)
    time_departure_stop = time_departure_start + sailing_out_time
    return time_departure_stop


def calculate_lock_departure_stop_time(lock_chamber, vessel, operation_index, operation_stop_time):
    """
    Calculates the moment in time the departure process of a lock operation stops

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    operation_stop_time : pd.Timestamp
        the time that the lock operation has stopped (i.e., gate have been opened again)

    Returns
    -------
    time_departure_stop : pd.Timestamp
        the moment in time that a lock operation's departure process stops
    """
    if vessel is None:
        return operation_stop_time
    lock_complex = lock_chamber.lock_complex
    last_vessel = _get_last_vessel_of_lock_operation(lock_complex, operation_index)
    time_departure_stop = calculate_vessel_departure_stop_time(lock_chamber, last_vessel, operation_index, operation_stop_time)
    return time_departure_stop


def calculate_vessel_passing_stop_time(lock_chamber, vessel, operation_index, direction, operation_stop_time):
    """
    Calculates the moment in time the vessel has reached the approach point at the other side of the lock (while sailing away from the lock)

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    operation_stop_time : pd.Timestamp
        the time that the lock operation has stopped (i.e., gate have been opened again)

    Returns
    -------
    time_departure_stop : pd.Timestamp
        the moment in time the vessel has reached the approach point at the other side of the lock
    """
    time_departure_stop = calculate_vessel_departure_stop_time(lock_chamber, vessel, operation_index, operation_stop_time)
    vessel_speed = _get_vessel_sailing_out_speed(lock_chamber, vessel, direction)
    if vessel_speed:
        time_departure_stop += pd.Timedelta(seconds = lock_chamber.sailing_distance_to_crossing_point/vessel_speed)
    return time_departure_stop


def calculate_lock_operation_stop_time(lock_chamber, vessel, operation_index, direction, operation_stop_time):
    """
    Calculates the moment in time a new lock operation can start

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    operation_stop_time : pd.Timestamp
        the time that the lock operation has stopped (i.e., gate have been opened again)

    Returns
    -------
    time_operation_stop : pd.Timestamp
        the moment in time a new lock operation can start
    """
    if vessel is None:
        return operation_stop_time
    lock_complex = lock_chamber.lock_complex
    last_vessel = _get_last_vessel_of_lock_operation(lock_complex, operation_index)
    time_operation_stop = calculate_vessel_passing_stop_time(lock_chamber, last_vessel, operation_index, direction, operation_stop_time)
    return time_operation_stop


def calculate_lock_gate_closing_time(lock_chamber, vessel, operation_index, operation_stop_time):
    """
    Calculates the moment in time a new lock operation can start

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    operation_stop_time : pd.Timestamp
        the time that the lock operation has stopped (i.e., gate have been opened again)

    Returns
    -------
    time_operation_stop : pd.Timestamp
        the moment in time a new lock operation can start
    """
    lock_gate_closing_time = calculate_lock_departure_stop_time(lock_chamber, vessel, operation_index, operation_stop_time)
    return lock_gate_closing_time


def calculate_delay_previous_vessel_to_optimize_sailing_in_process(lock_chamber, vessel, previous_vessel):
    # unpack the lock master's vessel planning
    lock_complex = lock_chamber.lock_complex
    vessel_planning = lock_complex.vessel_planning

    # information of vessel
    planning_index_vessel = vessel_planning[vessel_planning.index == vessel.id].iloc[-1].name
    time_lock_entry_vessel = vessel_planning.loc[planning_index_vessel, 'time_lock_entry_start']

    # information of previous vessel
    planning_index_previous_vessel = vessel_planning[vessel_planning.id == previous_vessel.id].iloc[-1].name
    time_lock_entry_previous_vessel = vessel_planning.loc[planning_index_previous_vessel, 'time_lock_entry_start']
    sailing_information_previous_vessel = calculate_sailing_information_on_route_to_lock_complex(lock_complex,
                                                                                                 previous_vessel,
                                                                                                 lock_end_node)
    total_time_to_lock_previous_vessel = sailing_information_previous_vessel.Time.sum()
    total_distance_to_lock_previous_vessel = sailing_information_previous_vessel.Distance.sum()
    if sailing_information_previous_vessel.empty or total_time_to_lock_previous_vessel <= 0.0:
        return  0.

    # planning strategy information
    minimum_sailing_in_time_gap = datetime.timedelta(seconds=lock_chamber.sailing_in_time_gap_through_gate)

    # calculate gap in arrival time between vessels
    extra_sailing_in_time_gap = time_lock_entry_vessel - time_lock_entry_previous_vessel - minimum_sailing_in_time_gap
    if extra_sailing_in_time_gap.total_seconds() <= 0.0:
        return  0.

    # determine the optimum speed of the previous vessel to delay its arrival time
    average_speed = total_distance_to_lock_previous_vessel / total_time_to_lock_other_vessel
    overruled_speed = np.max([lock_chamber.minimum_manoeuvrability_speed, total_distance_to_lock_previous_vessel /
                              (extra_sailing_in_time_gap.total_seconds() + total_time_to_lock_other_vessel)])

    # calculate delay in arrival time of previous vessel
    arrival_delay_previous_vessel = total_distance_to_lock_previous_vessel / overruled_speed - \
                                    total_distance_to_lock_previous_vessel / average_speed

    return arrival_delay_previous_vessel


def calculate_optimal_approach_speed_information(lock_chamber, vessel, lock_end_node, waiting_time):
    # determines the sailing information of the vessel (i.e., speed, distance, time) over the edges from its current location to its first encountered lock gate
    lock_complex = lock_chamber.lock_complex
    sailing_information = calculate_sailing_information_on_route_to_lock_complex(lock_complex, vessel, lock_end_node)

    # skip function if no sailing information is available
    if sailing_information.empty:
        return pd.DataFrame()

    # determines the average speed of the vessel over its route and calculate the overruled speed of the vessel based on the waiting time
    average_speed = sailing_information.loc[:, 'distance'].sum() / sailing_information.loc[:, 'time'].sum()
    overruled_speed = np.max([lock_chamber.minimum_manoeuvrability_speed, sailing_information.loc[:, 'distance'].sum() / (
                sailing_information.loc[:, 'time'].sum() + waiting_time)])
    reversed_sailing_information = sailing_information.iloc[::-1]

    # loops over the sailing information of the edges to adhere to the overruled speed (averaged over the route), the stops if too much iterations are required or when the difference between the new average speed and the overruled speed are sufficiently close to each other or when there are no speeds to be reduced
    iteration = 0
    speed_mask = reversed_sailing_information.speed < lock_chamber.minimum_manoeuvrability_speed
    while not np.abs(average_speed - overruled_speed) <= 0.01 and not reversed_sailing_information[
        speed_mask].empty:
        if iteration == 100:
            break

        # the difference in new average speed and overrulled speed
        speed_difference = average_speed - overruled_speed

        # identifies all speeds that are still greater than the minimum required speed for manoevrability (safety), so that these speeds can be reduced -> adjust the speed and time
        speed_mask = reversed_sailing_information.speed > lock_chamber.minimum_manoeuvrability_speed
        reversed_sailing_information.loc[
            reversed_sailing_information[speed_mask].index, 'speed'] -= speed_difference
        reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'time'] = \
            reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'distance'] / \
            reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'speed']

        # if in the previous steps speeds have been reduced to less than the minimum manoevrability speed, then change these speeds to this minimum -> adjust again the speed and time
        speed_mask = reversed_sailing_information.speed < lock_chamber.minimum_manoeuvrability_speed
        reversed_sailing_information.loc[
            reversed_sailing_information[speed_mask].index, 'speed'] = lock_chamber.minimum_manoeuvrability_speed
        reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'time'] = \
            reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'distance'] / \
            reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'speed']

        # calculate the new average speed and increase the iteration number by one
        average_speed = reversed_sailing_information.Distance.sum() / reversed_sailing_information.Time.sum()
        iteration += 1
    return reversed_sailing_information_info.speed


def calculate_empty_lock_operation_information_and_update_planning(lock_chamber, operation_index, direction):
    lock_operation_information = calculate_lock_operation_information_and_update_planning(lock_chamber, None, operation_index, direction)
    return lock_operation_information


def calculate_lock_operation_information_and_update_planning(lock_chamber, vessel, operation_index, direction):
    vessel_planning_index = 0
    lock_complex = lock_chamber.lock_complex
    is_first_vessel = True
    if vessel is not None:
        is_first_vessel = _check_if_vessel_is_first_vessel(lock_complex, vessel, operation_index)
        vessel_planning_index = lock_complex.vessel_planning[lock_complex.vessel_planning.id == vessel.id].iloc[-1].name
        vessel_planning_info = lock_complex.vessel_planning.loc[vessel_planning_index]
        lock_complex.vessel_planning.loc[vessel_planning_index, 'operation_index'] = operation_index

    lock_operation_information = _get_information_for_lock_operation(lock_chamber, operation_index, direction)
    if vessel is not None:
        lock_operation_information["capacity_L"] -= vessel.L
        other_vessels_in_lock = lock_operation_information["vessels"]
        vessels = other_vessels_in_lock.copy()
        vessels.append(vessel)
        lock_operation_information["vessels"] = vessels
        _update_lock_vessel_planning(lock_complex, vessel_planning_index, lock_operation_information)
    _update_lock_operation_planning(lock_complex, operation_index, lock_operation_information)

    arrival_information = calculate_lock_operation_start_information(lock_chamber, vessel, operation_index, direction)
    if vessel is not None:
        sailing_time_from_approach_point_to_lock = vessel_planning_info.time_lock_entry_start - \
                                                   vessel_planning_info.time_arrival_at_approach_point
        arrival_information['time_arrival_at_approach_point'] = arrival_information["time_lock_entry_start"] - \
                                                                sailing_time_from_approach_point_to_lock

    if vessel is not None:
        _update_lock_vessel_planning(lock_complex, vessel_planning_index, arrival_information)

    if not is_first_vessel:
        for info in ['time_potential_lock_gate_opening_stop', 'time_lock_operation_start', 'time_lock_entry_start']:
            arrival_information.pop(info, None)

    _update_lock_operation_planning(lock_complex, operation_index, arrival_information)

    levelling_information = calculate_lock_operation_times(lock_chamber,
                                                           operation_index=operation_index,
                                                           start_time=arrival_information["time_lock_entry_stop"],
                                                           vessel=vessel,
                                                           direction=direction)

    if vessel is not None:
        _update_lock_vessel_planning(lock_complex, vessel_planning_index, levelling_information)

    _update_lock_operation_planning(lock_complex, operation_index, levelling_information)

    vessel_planning = lock_complex.vessel_planning
    operation_info = lock_complex.operation_planning.loc[operation_index]
    try:
        delay = levelling_information['time_gate_opening_stop'] - operation_info['time_lock_departure_start']
    except:
        delay = pd.Timedelta(seconds = 0)

    if delay > pd.Timedelta(seconds = 0):
        for other_vessel in other_vessels_in_lock:
            other_vessel_planning_index = vessel_planning[vessel_planning.id == other_vessel.id].iloc[-1].name
            vessel_planning.loc[other_vessel_planning_index, 'time_lock_departure_start'] += delay
            vessel_planning.loc[other_vessel_planning_index, 'time_lock_departure_stop'] += delay
            vessel_planning.loc[other_vessel_planning_index, 'time_lock_operation_stop'] += delay
            vessel_planning.loc[other_vessel_planning_index, 'time_potential_lock_gate_closure_start'] += delay

    departure_information = calculate_lock_departure_information(lock_chamber, vessel, operation_index, direction,
                                                                 levelling_information)
    if vessel is not None:
        _update_lock_vessel_planning(lock_complex, vessel_planning_index, departure_information)

    if not is_first_vessel and delay > pd.Timedelta(seconds = 0):
        first_vessel_index = vessel_planning[vessel_planning.id == other_vessels_in_lock[0].id].iloc[-1].name
        departure_information['time_lock_departure_start'] = vessel_planning.loc[first_vessel_index, 'time_lock_departure_start']

    _update_lock_operation_planning(lock_complex, operation_index, departure_information)
    _update_future_lock_operations_by_lock_delay_previous_operation(lock_chamber, operation_index,
                                                                    departure_information)

    lock_operation_information = {**lock_operation_information, **arrival_information,
                                  **levelling_information, **departure_information}
    return lock_operation_information


def calculate_vessel_approach_information(lock_complex, vessel, direction):
    vessel_planning = lock_complex.vessel_planning
    vessel_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
    lineup_area = None
    lock_chamber = lock_complex.lock_chambers[vessel_planning.loc[vessel_index, 'lock_chamber']]
    if len(lock_complex.lineup_areas):
        lineup_area = lock_complex.lineup_areas[vessel_planning.loc[vessel_index, 'lineup_area']]
    waiting_area = lock_complex.waiting_areas[vessel_planning.loc[vessel_index, 'waiting_area']]

    node_from, node_to = _get_lock_operation_to_and_from_node(lock_chamber, direction)
    current_time = datetime.datetime.fromtimestamp(vessel.env.now)
    sailing_time_to_waiting_area = calculate_sailing_time_to_waiting_area(waiting_area, vessel)[0]
    sailing_time_to_lineup_area = calculate_sailing_time_to_lineup_area(lineup_area, vessel)
    sailing_time_to_approach = calculate_sailing_time_to_approach_point(lock_chamber, vessel, direction)
    sailing_time_to_lock = calculate_sailing_time_to_lock_gate(lock_chamber, vessel, direction)
    sailing_time_to_pass_lock = calculate_sailing_time_to_pass_lock(lock_chamber, vessel)
    delay = calculate_delay_until_arrival_within_operational_hours(lock_chamber, current_time + sailing_time_to_lock)

    vessel_information = {}
    vessel_information['time_of_registration'] = current_time
    vessel_information['time_of_acceptance'] = current_time
    vessel_information['node_from'] = node_from
    vessel_information['node_to'] = node_to
    vessel_information['time_arrival_at_waiting_area'] = current_time + sailing_time_to_waiting_area + delay
    vessel_information['time_arrival_at_lineup_area'] = current_time + sailing_time_to_lineup_area + delay
    vessel_information['time_arrival_at_approach_point'] = current_time + sailing_time_to_approach + delay
    vessel_information['time_lock_entry_start'] = current_time + sailing_time_to_lock + delay
    vessel_information['time_to_traverse_waterway_without_lock'] = current_time + sailing_time_to_pass_lock
    vessel_information['delay'] = delay
    return vessel_information


def calculate_lock_distances_to_nodes_of_edge_from_geometry(lock_chamber, m = False):
    if not m and not isinstance(lock_chamber.geometry, Polygon):
        raise ValueError('Given lock geometry is not a Polygon')
    if m and not isinstance(lock_chamber.geometry_m, Polygon):
        raise ValueError('Given lock geometry is not a Polygon')

    edge_geometry = get_geometry_of_edge(lock_chamber.env.graph,lock_chamber.edge)
    edge_geometry_m = transform_geometry(edge_geometry, epsg_out = lock_chamber.crs_m)
    if not m:
        lock_chamber.geometry_m = transform_geometry(lock_chamber.geometry, epsg_out = lock_chamber.crs_m)
    intersected_edge = edge_geometry_m.intersection(lock_chamber.geometry_m)
    locations_of_lock_gate = [Point(coords) for coords in intersected_edge.coords]
    if len(locations_of_lock_gate) < 2:
        raise ValueError(f'Given lock geometry is not valid -> leads to {len(locations_of_lock_gate)} intersection points (which should be 2 at minimum).')
    locations_of_lock_gate = [locations_of_lock_gate[0],locations_of_lock_gate[-1]]
    distance_from_start_node_to_lock_gate_A = 0
    distance_from_end_node_to_lock_gate_B = 0
    for index, location_of_lock_gate in enumerate(locations_of_lock_gate):
        distance_from_start_node = edge_geometry_m.project(location_of_lock_gate)
        if not index:
            distance_from_start_node_to_lock_gate_A = distance_from_start_node
        else:
            distance_from_end_node_to_lock_gate_B = edge_geometry_m.length - distance_from_start_node
    return distance_from_start_node_to_lock_gate_A, distance_from_end_node_to_lock_gate_B


def calculate_lock_dimensions_from_geometry(lock_chamber, m = False):
    if not m:
        lock_chamber.geometry_m = transform_geometry(lock_chamber.geometry, epsg_out=lock_chamber.crs_m)
    lock_chamber.geometry_m_rectangle = lock_chamber.geometry_m.minimum_rotated_rectangle
    coords = list(lock_chamber.geometry_m_rectangle.exterior.coords)[:-1]
    edges = []
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        edges.append(math.hypot(x2 - x1, y2 - y1))

    unique = sorted(set(round(e, 8) for e in edges), reverse=True)

    length = unique[0]
    width = unique[1]
    return length, width


def calculate_and_check_lock_dimensions(lock_chamber):
    if not lock_chamber.lock_depth:
        raise ValueError(f'Invalid lock depth: {lock_chamber.lock_depth} (should be > 0).')
    if not lock_chamber.lock_length and not (lock_chamber.geometry is None or lock_chamber.geometry_m is None):
        raise ValueError(f'Invalid lock length: {lock_chamber.lock_length} (should be > 0).')
    if not lock_chamber.lock_width and not (lock_chamber.geometry is None or lock_chamber.geometry_m is None):
        raise ValueError(f'Invalid lock length: {lock_chamber.lock_width} (should be > 0).')
    if (not lock_chamber.lock_length or not lock_chamber.lock_width) and \
            (lock_chamber.geometry is not None or lock_chamber.geometry_m is not None):
        if lock_chamber.geometry is not None:
            m = False
        else:
            m = True
        lock_length, lock_width = calculate_lock_dimensions_from_geometry(lock_chamber, m=m)
        if not lock_chamber.lock_length:
            lock_chamber.lock_length = lock_length
        if not lock_chamber.lock_width:
            lock_chamber.lock_width = lock_width
            
            
def calculate_time_to_open_gate(lock_chamber, operation_index, direction, gate_required_to_be_open):
    """
    Determines the time to finish the levelling process and the gate opening process

    Parameters
    ----------
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    gate_required_to_be_open : pd.Timestamp
        the moment in time that the gate are required to be opened

    Returns
    -------
    operation_time : pd.Timedelta
        the time to finish the levelling process and the gate opening process
    """
    operation_start_time = gate_required_to_be_open - pd.Timedelta(seconds=lock_chamber.gate_opening_time)
    levelling_information = calculate_lock_operation_times(
        lock_chamber,
        operation_index=operation_index,
        start_time=operation_start_time,
        direction=direction,
    )

    levelling_time = levelling_information["time_levelling_stop"] - levelling_information["time_levelling_start"]
    wlev_before, wlev_after = levelling_information["wlev_A"], levelling_information["wlev_B"]

    levelling_required = True
    if abs(wlev_after - wlev_before) < 0.1:
        levelling_required = False

    if not levelling_required:
        levelling_time = pd.Timedelta(seconds=0.0)

    operation_time = levelling_time + pd.Timedelta(seconds=lock_chamber.gate_opening_time)
    return operation_time


def calculate_ZSF_eventttable(lock_chamber):
    lock_df = pd.DataFrame(lock_chamber.logbook)

    zsf_events = pd.DataFrame(
        columns=['time_start', 'time_stop', 'head_sea', 'head_lake', 'routine', 'salinity_sea', 'salinity_lake',
                 'ship_volume_lake_to_sea', 'ship_volume_sea_to_lake', 't_level', 't_open_lake', 't_open_sea',
                 'temperature_lake', 'temperature_sea'])

    head_sea = np.nan
    head_lake = np.nan
    salinity_sea = np.nan
    salinity_lake = np.nan
    temperature_sea = np.nan
    temperature_lake = np.nan
    for index, info in lock_df.iterrows():
        t_level = 0
        t_open_lake = 0
        t_open_sea = 0
        if not index and info.Message == "Lock gate closing start":
            time_stop = info.Timestamp + pd.Timedelta(seconds=lock_chamber.gate_opening_time) / 2
            duration = (time_stop - lock_chamber.env.simulation_start).total_seconds()
            if info.Geometry == lock_chamber.start_node:
                routine = 4
                t_open_sea = duration
            else:
                routine = 2
                t_open_lake = duration
            time_start = lock_chamber.env.simulation_start

        elif info.Message == "Lock gate opening start":
            future_lock_df = lock_df.loc[index:]
            future_gate_closing_df = future_lock_df[future_lock_df.Message == "Lock gate closing stop"]
            time_start = info.Timestamp + pd.Timedelta(seconds=lock_chamber.gate_opening_time) / 2
            if future_gate_closing_df.empty:
                duration = (lock_chamber.env.simulation_stop - time_start).total_seconds()
                time_stop = lock_chamber.env.simulation_stop
            else:
                time_stop = future_gate_closing_df.iloc[0].Timestamp - pd.Timedelta(
                    seconds=lock_chamber.gate_closing_time) / 2
                duration = (time_stop - time_start).total_seconds()

            if info.Geometry == lock_chamber.start_node:
                routine = 4
                t_open_sea = duration
            else:
                routine = 2
                t_open_lake = duration

        elif info.Message == "Lock chamber converting stop":
            past_lock_df = lock_df.loc[:index]
            levelling_start_info = past_lock_df[past_lock_df.Message == "Lock chamber converting start"].iloc[-1]
            time_start = levelling_start_info.Timestamp
            time_stop = info.Timestamp
            duration = (time_stop - time_start).total_seconds()
            if info.Geometry == lock_chamber.start_node:
                routine = 3
                t_level = duration
            else:
                routine = 1
                t_level = duration

        else:
            continue

        event_idx = len(zsf_events)
        hydromanager = HydrodynamicDataManager()
        time_start = np.datetime64(time_start)
        time_stop = np.datetime64(time_stop)
        head_sea = hydromanager._get_hydrodynamic_data_value(time_start, lock_chamber.start_node, 'Water level')
        head_lake = hydromanager._get_hydrodynamic_data_value(time_start, lock_chamber.end_node, 'Water level')
        salinity_sea = hydromanager._get_hydrodynamic_data_value(time_start, lock_chamber.start_node, 'Salinity')
        salinity_lake = hydromanager._get_hydrodynamic_data_value(time_start, lock_chamber.end_node, 'Salinity')
        temperature_sea = hydromanager._get_hydrodynamic_data_value(time_start, lock_chamber.start_node, 'Temperature')
        temperature_lake = hydromanager._get_hydrodynamic_data_value(time_start, lock_chamber.end_node, 'Temperature')
        ship_volume_lake_to_sea = 0.
        ship_volume_sea_to_lake = 0.
        zsf_events.loc[event_idx, :] = [time_start, time_stop, head_sea, head_lake, routine, salinity_sea,
                                        salinity_lake, ship_volume_lake_to_sea, ship_volume_sea_to_lake, t_level,
                                        t_open_lake, t_open_sea, temperature_lake, temperature_sea]

    # #Correcting events for varying water levels
    for index in zsf_events[zsf_events.routine == 3].index:
        iloc = zsf_events.index.get_loc(index)
        zsf_events.loc[index, 'head_sea'] = zsf_events.iloc[iloc + 1]['head_sea']

    for index in zsf_events[zsf_events.routine == 1].index:
        iloc = zsf_events.index.get_loc(index)
        zsf_events.loc[index, 'head_lake'] = zsf_events.iloc[iloc + 1]['head_lake']

    zsf_events = zsf_events.set_index(['time_start','time_stop'])
    return zsf_events


def calculate_water_exchange_fluxes(lock_chamber, init_salinity_lock = 15.0, init_head_lock = 0.0):
    zsf_events = calculate_ZSF_eventttable(lock_chamber)

    lock_parameters = {
        "lock_length": lock_chamber.lock_length,
        "lock_width": lock_chamber.lock_width,
        "lock_bottom": -lock_chamber.lock_depth, }

    mitigation_parameters = {
        "density_current_factor_lake": 1.0,
        "density_current_factor_sea": 1.0,
        "distance_door_bubble_screen_lake": 0.0,
        "distance_door_bubble_screen_sea": 0.0,
        "flushing_discharge_high_tide": 0.0,
        "flushing_discharge_low_tide": 0.0,
        "sill_height_lake": 0.0,
        "sill_height_sea": 0.0,
    }

    first_event = zsf_events.iloc[0]
    if first_event.routine == 2:
        head_lock = first_event.head_lake
    elif first_event.routine == 4:
        head_lock = first_event.head_sea
    ZSF = pyzsf.ZSFUnsteady(sal_lock=init_salinity_lock, head_lock=head_lock, **lock_parameters, **mitigation_parameters)
    lockages = list(zsf_events.to_dict("records"))

    all_results = []
    for parameters in lockages:
        routine = int(parameters.pop("routine"))
        t_open_lake = parameters.pop("t_open_lake")
        t_open_sea = parameters.pop("t_open_sea")
        t_level = parameters.pop("t_level")
        parameters["ship_volume_sea_to_lake"] = 0.0
        parameters["ship_volume_lake_to_sea"] = 0.0

        initial_ZSF_state = ZSF.state.copy()
        initial_ZSF_state = {'salinity_lock_start': initial_ZSF_state['salinity_lock'],
                             'saltmass_lock_start': initial_ZSF_state['saltmass_lock'],
                             'volume_ship_in_lock_start': initial_ZSF_state['volume_ship_in_lock'],
                             'head_lock_start': initial_ZSF_state['head_lock']}

        if routine == 1:
            assert t_level > 0
            results = ZSF.step_phase_1(t_level, **parameters)
        elif routine == 2:
            ZSF.state["head_lock"] = parameters['head_lake']
            results = ZSF.step_phase_2(t_open_lake, **parameters)
        elif routine == 3:
            assert t_level > 0
            results = ZSF.step_phase_3(t_level, **parameters)
        elif routine == 4:
            ZSF.state["head_lock"] = parameters['head_sea']
            results = ZSF.step_phase_4(t_open_sea, **parameters)
        elif routine in {-2, -4}:
            results = ZSF.step_flush_doors_closed(t_flushing, **parameters)
        else:
            raise Exception(f"Unknown routine '{routine}'")

        for key,value in initial_ZSF_state.items():
            results[key] = value

        results['salinity_lock_stop'] = ZSF.state['salinity_lock']
        results['saltmass_lock_stop'] = ZSF.state['saltmass_lock']
        results['volume_ship_in_lock_stop'] = ZSF.state['volume_ship_in_lock']
        results['head_lock_stop'] = ZSF.state['head_lock']
        all_results.append(results)

    ZSF_results = pd.concat([zsf_events,pd.DataFrame(all_results, index=zsf_events.index)],axis=1)
    ZSF_results = ZSF_results.astype(float)
    ZSF_results = ZSF_results.reset_index()
    return ZSF_results


def calculate_aggregated_water_exchange_fluxes(lock_chamber, ZSF_results):
    # Aggregate results
    duration_ts = (lock_chamber.env.simulation_stop - lock_chamber.env.simulation_start)
    duration = duration_ts.total_seconds()

    overall_results = {}
    overall_mass_to_sea = 0.0
    overall_mass_to_lake = 0.0

    for results in ZSF_results.to_dict("records"):
        for k, v in results.items():
            if k.startswith(("volume_", "mass_")):
                overall_results[k] = overall_results.get(k, 0.0) + v

        overall_mass_to_sea += results["volume_to_sea"] * results["salinity_to_sea"]
        overall_mass_to_lake += results["volume_to_lake"] * results["salinity_to_lake"]

    overall_results["salinity_to_sea"] = overall_mass_to_sea / overall_results["volume_to_sea"]
    overall_results["salinity_to_lake"] = overall_mass_to_lake / overall_results["volume_to_lake"]

    overall_discharges = {}
    for k, v in overall_results.items():
        if k.startswith("volume_"):
            overall_discharges[f"discharge_{k[7:]}"] = v / duration
    overall_results.update(overall_discharges)

    return overall_results