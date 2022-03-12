# Importing libraries

# Used for mathematical functions
# package(s) related to time, space and id
import itertools

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import pandas as pd

# package(s) for data handling
import numpy as np
import tqdm

# OpenTNSim
import opentnsim

import pytest

# Creating the test objects

# Actual testing starts here
# - tests 3 fixed velocities to return the right P_tot
# - tests 3 fixed power to return indeed the same P_tot
# - tests 3 fixed power to return indeed the same v
# todo: current tests do work with vessel.h_squat=True ... issues still for False
def test_simulation():
    # Make your preferred class out of available mix-ins.
    TransportResource = type(
        "Vessel",
        (
            opentnsim.core.Identifiable,
            opentnsim.core.Movable,
            opentnsim.core.VesselProperties,
            opentnsim.core.ConsumesEnergy,
            opentnsim.core.ExtraMetadata,
        ),
        {},
    )

    # Create a dict with all important settings
    data_vessel = {"env": None,
                   "name": 'Vessel M9',
                   "route": None,
                   "geometry": None,
                   "v": None,  # m/s
                   "type": None,
                   "B": 11.45,
                   "L": 135,
                   "H_e": None,
                   "H_f": None,
                   "T": 2.75,
                   "safety_margin": 0.3,  # for tanker vessel with rocky bed the safety margin is recommended as 0.3 m
                   "h_squat": True,  # if consider the ship squatting while moving, set to True, otherwise set to False
                   "payload": None,
                   "vessel_type": "Tanker", # vessel types: "Container","Dry_SH","Dry_DH","Barge","Tanker". ("Dry_SH" means dry bulk single hull, "Dry_DH" means dry bulk double hull)
                   "P_installed": 2200.0,
                   "P_tot_given": None,  # kW
                   "bulbous_bow": False,  # if a vessel has no bulbous_bow, set to False; otherwise set to True.
                   "P_hotel_perc": 0.05,
                   "P_hotel": None,  # None: calculate P_hotel from percentage
                   "L_w": 3.0,
                   "C_B": 0.85,
                   "C_year": 1990,
                   }

    # todo: for the test we can reduce the sampling number to save time (reduced 200 to 10)
    V_s = [0.1, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # ship sailing speeds to water, (m/s)
    h_0 = [10, 7.5, 5, 3.5]  # water depths,(m)
    # todo: perhaps if we want one test to only test the resistance factors we don't need to vary engineages (make other test for that?)
    C_year = [1970, 1980, 1990, 2000, 2010, 2020]  # engine construction years

    # prepare the work to be done
    # create a list of all combinations
    work = list(itertools.product(C_year, h_0, V_s))

    # prepare a list of dictionaries for pandas
    rows = []
    for item in work:
        row = {"C_year": item[0], "h_0": item[1], "V_s": item[2]}
        rows.append(row)

    # these are all the simulations that we want to run
    # convert them to dataframe, so that we can apply a function and monitor progress
    work_df = pd.DataFrame(rows)

    results = []

    for i, row in tqdm.tqdm(work_df.iterrows()):
        # create a new vessel, like the one above (so that it also has L)
        C_year = row['C_year']
        data_vessel_i = data_vessel.copy()
        data_vessel_i['C_year'] = C_year
        vessel = TransportResource(**data_vessel_i)

        V_s = row['V_s']
        h_0 = row['h_0']
        vessel.calculate_properties()  # L is used here in the computation of L_R
        h_0 = vessel.calculate_h_squat(v=V_s, h_0=h_0)

        R_f = vessel.calculate_frictional_resistance(V_s, h_0)
        R_f_one_k1 = vessel.calculate_viscous_resistance()
        R_APP = vessel.calculate_appendage_resistance(V_s)
        R_W = vessel.calculate_wave_resistance(V_s, h_0)
        R_res = vessel.calculate_residual_resistance(V_s, h_0)
        R_T = vessel.calculate_total_resistance(V_s, h_0)

        # todo: if we want to make this test only about resistance components, we may want to exclude this
        P_tot = vessel.calculate_total_power_required(V_s)
        vessel.emission_factors_general()
        vessel.correction_factors(V_s)
        vessel.calculate_emission_factors_total(V_s)
        Fuel_g_m = vessel.calculate_fuel_use_g_m(V_s)
        [emission_g_m_CO2, emission_g_m_PM10, emission_g_m_NOX] = vessel.calculate_emission_rates_g_m(V_s)

        result = {}
        result.update(row)
        result['P_installed'] = vessel.P_installed
        result['R_f_one_k1'] = R_f_one_k1
        result['R_APP'] = R_APP
        result['R_W'] = R_W
        result['R_res'] = R_res
        result['R_T'] = R_T

        # todo: if we want to make this test only about resistance components, we may want to exclude this
        result['P_tot'] = P_tot
        result['Fuel_g_km'] = Fuel_g_m * 1000
        result['emission_g_km_CO2'] = emission_g_m_CO2 * 1000
        result['emission_g_km_PM10'] = emission_g_m_PM10 * 1000
        result['emission_g_km_NOX'] = emission_g_m_NOX * 1000
        results.append(result)

    # collect info dataframe
    plot_df = pd.DataFrame(results)

    # convert from meters per second to km per hour
    ms_to_kmh = 3.6
    plot_df['V_s_km'] = plot_df['V_s'] * ms_to_kmh

    # todo: Here there should be quite a list of assert statements to test a whole bunch of numbers
    # e.g. speeds [0.1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] x depths [10, 7.5, 5, 3.5] = 40
    # also we may want to test various resistance components (40 x 5  could be a list of potentially 200 assert statements)
    # maybe we can reduce by testing less speeds

    np.testing.assert_almost_equal(0.049849,  plot_df.R_T[0], decimal=3, err_msg='not almost equal', verbose=True)


