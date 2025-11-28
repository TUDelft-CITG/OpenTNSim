"""This module contains functions to perform calculations for lock operations."""

import numpy as np
from opentnsim.utils import time_to_numpy
from opentnsim.constants import gravitational_acceleration
from opentnsim.vessel_traffic_service.hydrodanamic_data_manager import HydrodynamicDataManager


def calculate_z(
    t,
    t_start,
    direction,
    wlev_init,
    operation_index,
    operation_planning,
    hydrodynamic_information_path,
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
    time_index = hydromanager._get_time_index_of_hydrodynamic_data(hydrodynamic_information_path, t_start)
    t_simulation_start = np.datetime64(epoch)
    H_A = hydromanager._get_hydrodynamic_data_series(hydrodynamic_information_path, t_simulation_start, start_node, "Water level")
    H_B = hydromanager._get_hydrodynamic_data_series(hydrodynamic_information_path, t_simulation_start, end_node, "Water level")
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
    water_level_difference_limit_to_open_doors,
    prediction,
    H_A,
    H_B,
):
    """Calculates the levelling time of a lock operation based on Eq. 4.64 of Ports and Waterways Open Textbook (https://books.open.tudelft.nl/home/catalog/book/204)
    This function is called by determine_levelling_time()
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
    H_time = HydrodynamicDataManager().hydrodynamic_times.astype(float)  # time series of the hydrodynamic data [s]

    # time-integration by (self-coded) Euler's method TODO Checken of we een standaard solver kunnen gebruiken. En of we dit algoritme los kunnen maken van de klasse.
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
            np.abs(z[i + 1]) <= water_level_difference_limit_to_open_doors
        ):  # breaks the integration if the water level difference is smaller than a default 5 cm (the last 5 cm of water level difference takes long to overcome, so lock master opens doors)
            z[(i + 1) :] = np.nan  # set all next values of the water level series to nan
            break

    # determining levelling time based on the first nan of the series TODO: Class-functie maken _determine_levelling_time()
    if len(np.argwhere(np.isnan(z))):
        levelling_time = t[np.argwhere(np.isnan(z))[0]][0]
    else:
        levelling_time = t[-1]

    return levelling_time, t, z
