"""Here we test the payload calculation from Van Dorsser et al's method for a range of settings (for a few vessel types, and a few payload – draft combinations)"""
# To do: add more asserts for a range of settings

import numpy as np
import pandas as pd
import opentnsim.core
import opentnsim.strategy
import itertools
import tqdm

import pytest

# Make your preferred class out of available mix-ins.
def test_simulation():
    TransportResource = type(
        "Vessel",
        (
            opentnsim.core.Identifiable,
            opentnsim.core.Movable,
            opentnsim.core.Routeable,
            opentnsim.core.VesselProperties,  # needed to add vessel properties
            opentnsim.energy.ConsumesEnergy,
            opentnsim.core.ExtraMetadata,
        ),
        {},
    )
    # Create a dict with all important settings
    data_vessel = {
        "env": None,
        "name": None,
        "route": None,
        "geometry": None,
        "v": 3.5,  # m/s
        "type": None,
        "B": 11.4,
        "L": 110,
        "H_e": None,
        "H_f": None,
        "T": 3.5,
        "safety_margin": 0.3,  # for tanker vessel with rocky bed the safety margin is recommended as 0.3 m
        "h_squat": True,  # if consider the ship squatting while moving, set to True, otherwise set to False
        "P_installed": 1750.0,  # kW
        "P_tot_given": None,  # kW
        "bulbous_bow": False,  # if a vessel has no bulbous_bow, set to False; otherwise set to True.
        "P_hotel_perc": 0.05,
        "P_hotel": None,  # None: calculate P_hotel from percentage
        "L_w": 3.0,
        "C_B": 0.85,
        "C_year": 1990,
    }

    T_strategy = [3.5, 3, 2.2, 2.1, 2.0]
    # prepare the work to be done
    # create a list of all combinations
    work = list(itertools.product(T_strategy))
    # prepare a list of dictionaries for pandas
    rows = []
    for item in work:
        row = {"T_strategy": item[0]}
        rows.append(row)

        # these are all the simulations that we want to run
    # convert them to dataframe, so that we can apply a function and monitor progress
    work_df = pd.DataFrame(rows)
    Strategies = []
    for i, row in tqdm.tqdm(work_df.iterrows()):
        T_strategy = row["T_strategy"]
        vessel = TransportResource(**data_vessel)

        Payload = opentnsim.strategy.T2Payload(vessel, T_strategy, vessel_type="Tanker")

        Strategy = {}
        Strategy.update(row)

        Strategy["Payload_strategy_tanker (ton)"] = Payload
        Strategies.append(Strategy)

    Strategies_df = pd.DataFrame(Strategies)

    # Test if the output of "Tanker" vessel are the same as calculated by strategy.py
    np.testing.assert_almost_equal(
        2681.7553,
        Strategies_df["Payload_strategy_tanker (ton)"][0],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        2152.3154,
        Strategies_df["Payload_strategy_tanker (ton)"][1],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        1227.2542,
        Strategies_df["Payload_strategy_tanker (ton)"][2],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        1112.7661,
        Strategies_df["Payload_strategy_tanker (ton)"][3],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
    np.testing.assert_almost_equal(
        998.5323,
        Strategies_df["Payload_strategy_tanker (ton)"][4],
        decimal=2,
        err_msg="not almost equal",
        verbose=True,
    )
