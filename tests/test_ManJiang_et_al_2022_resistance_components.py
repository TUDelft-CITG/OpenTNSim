"""Here we test the total resistance with all its resistance components for inland ships, which includes
 R_f, R_f_one_k1, R_APP, R_W, R_res. With input Vs=[0.1, 1, 2, 3, 4] m/s, h_0 = 5 m, C_year = 2000, do not consider squat.

In the future it is nice to include another test-- resistance component R_B for seagoing ships which has a bulbous bow,
and test the switching between inland ship and seagoing ship resistance calculation
"""

# Importing libraries

# Used for mathematical functions
# package(s) related to time, space and id
import itertools
import pathlib

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import pandas as pd

# package(s) for data handling
import tqdm

# OpenTNSim
import opentnsim

import pytest
import utils


@pytest.fixture
def expected_df():
    path = pathlib.Path(__file__)
    return utils.get_expected_df(path)


# Creating the test objects


# Actual testing starts here
# - tests 3 fixed velocities to return the right P_tot
# - tests 3 fixed power to return indeed the same P_tot
# - tests 3 fixed power to return indeed the same v
# todo: current tests do work with vessel.h_squat=True ... issues still for False
@pytest.mark.skip("Regression tests need to be updated after fix in energy module")
def test_simulation(expected_df):
    # Make your preferred class out of available mix-ins.
    TransportResource = type(
        "Vessel",
        (
            opentnsim.core.Identifiable,
            opentnsim.core.Movable,
            opentnsim.core.VesselProperties,
            opentnsim.energy.ConsumesEnergy,
            opentnsim.core.ExtraMetadata,
        ),
        {},
    )

    # Create a dict with all important settings
    data_vessel = {
        "env": None,
        "name": "Vessel M9",
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
        "h_squat": False,  # if consider the ship squatting while moving, set to True, otherwise set to False
        "P_installed": 2200.0,
        "P_tot_given": None,  # kW
        "bulbous_bow": False,  # if a vessel has no bulbous_bow, set to False; otherwise set to True.
        "P_hotel_perc": 0.05,
        "P_hotel": None,  # None: calculate P_hotel from percentage
        "x": 2,  # number of propellers
        "L_w": 3.0,
        "C_B": 0.85,
        "C_year": 2000,
    }
    # input
    V_s = [0.1, 1, 2, 3, 4]  # ship sailing speeds to water, (m/s)
    h_0 = [5]  # water depths,(m)
    C_year = [2000]  # engine construction years

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

    for i, row in tqdm.tqdm(work_df.iterrows(), disable=True):
        # create a new vessel, like the one above (so that it also has L)
        C_year = row["C_year"]
        data_vessel_i = data_vessel.copy()
        data_vessel_i["C_year"] = C_year
        vessel = TransportResource(**data_vessel_i)

        V_s = row["V_s"]
        h_0 = row["h_0"]
        vessel.calculate_properties()  # L is used here in the computation of L_R
        h_0 = vessel.calculate_h_squat(v=V_s, h_0=h_0)

        _ = vessel.calculate_frictional_resistance(V_s, h_0)
        R_f_one_k1 = vessel.calculate_viscous_resistance()
        R_APP = vessel.calculate_appendage_resistance(V_s)
        R_W = vessel.calculate_wave_resistance(V_s, h_0)
        R_res = vessel.calculate_residual_resistance(V_s, h_0)
        R_T = vessel.calculate_total_resistance(V_s, h_0)

        result = {}
        result.update(row)
        result["h_0"] = h_0
        result["R_f_one_k1"] = R_f_one_k1
        result["R_APP"] = R_APP
        result["R_W"] = R_W
        result["R_res"] = R_res
        result["R_T"] = R_T
        results.append(result)

    # collect info dataframe
    plot_df = pd.DataFrame(results)

    # convert from meters per second to km per hour
    ms_to_kmh = 3.6
    plot_df["V_s_km"] = plot_df["V_s"] * ms_to_kmh

    # utils.create_expected_df(path=pathlib.Path(__file__), df=plot_df)
    columns_to_test = [column for column in plot_df.columns]
    pd.testing.assert_frame_equal(expected_df[columns_to_test], plot_df[columns_to_test], check_exact=False)
