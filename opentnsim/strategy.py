"""Sailing strategies in OpenTNSim.

The vessel-waterway interaction is complex, since the vessel's sailing behaviors are influenced by not only the changing waterway situations along the route, but also the varation of its own payload, fuel weight, velocity, engine power, etc. Therefore, to sail wisely and safely, it is necessary to formulate optimal sailing stratigies to achieve various sailing goals according to different needs.

This package combine with "optimal sailing stratigies notebook" provides stratigies for preventing ship grounding, optimizing cargo capacity, optimizing fuel usage, reducing emissions, considering sailing duration, etc.
"""

# To Do in this pacakge:
# 1) add "burning lighter" function to monitor the fuel weight decreasing along the route. For the battery-container or electricity powered vessel, the "fuel weight" is constant.
# 2）add "refueling heavier" function to show the fuel weight increased again at the refueling stations. For the battery-container or electricy powered vessel, the "fuel weight" is constant.
# 3) add "get_fuel_weight" function which call both the "burning lighter" and "refueling heavier" functions to take into account the influence of the variation of fuel weight to the actual draught and payload.
# 4) add "get_refueling_duration" function. For the battery-container, the duration is the unloading and loading time for the battery-containers.
# 5) add "get_optimal_refueling_amount" function. It's not always beneficial to be fully refueled with fuel for sailing, since more fuel on board leads to less cargo and there might still be residual fuel in the tank after a round trip if refuel too much. Therefore, it's needed to calculate the optimal refuling amount for each unique sailing case (route, vessel size & type, payload, time plan, refueling spots along the route).  The optimal refueling amount for a transport case determined by both the fuel consumption in time and space and the locations of refuling spots.
# 6) consider writting the "fix power or fix speed" example which is in the paper as a function into this pacakge in the future or Figure 10 -12 notebooks are enough already? What might it benefit if adds this function?


import functools
import itertools

import logging

import pandas as pd
import numpy as np
import scipy.optimize

import tqdm

logger = logging.getLogger(__name__)


# To know the corresponding Payload for each T_strategy
def T2Payload(vessel, T_strategy, vessel_type):
    """Calculate the corresponding payload for each T_strategy
    the calculation is based on Van Dorsser et al's method (2020) (https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)


    input:
    - T_strategy: user given possible draught
    - vessel types: "Container","Dry_SH","Dry_DH","Barge","Tanker". ("Dry_SH" means dry bulk single hull, "Dry_DH" means dry bulk double hull)

    output:
    - Payload_comupted: corresponding payload for the T_strategy for different vessel types

    """
    # Design draft T_design, refer to Table 5

    Tdesign_coefs = dict(
        {
            "intercept": 0,
            "c1": 1.7244153371,
            "c2": 2.2767179246,
            "c3": 1.3365379898,
            "c4": -5.9459308905,
            "c5": 6.2902305560 * 10**-2,
            "c6": 7.7398861528 * 10**-5,
            "c7": 9.0052384439 * 10**-3,
            "c8": 2.8438560877,
        }
    )

    assert vessel_type in [
        "Container",
        "Dry_SH",
        "Dry_DH",
        "Barge",
        "Tanker",
    ], 'Invalid value vessel_type, should be "Container","Dry_SH","Dry_DH","Barge" or "Tanker"'
    if vessel_type == "Container":
        [dum_container, dum_dry, dum_barge, dum_tanker] = [1, 0, 0, 0]
    elif vessel_type == "Dry_SH":
        [dum_container, dum_dry, dum_barge, dum_tanker] = [0, 1, 0, 0]
    elif vessel_type == "Dry_DH":
        [dum_container, dum_dry, dum_barge, dum_tanker] = [0, 1, 0, 0]
    elif vessel_type == "Barge":
        [dum_container, dum_dry, dum_barge, dum_tanker] = [0, 0, 1, 0]
    elif vessel_type == "Tanker":
        [dum_container, dum_dry, dum_barge, dum_tanker] = [0, 0, 0, 1]

    T_design = (
        Tdesign_coefs["intercept"]
        + (dum_container * Tdesign_coefs["c1"])
        + (dum_dry * Tdesign_coefs["c2"])
        + (dum_barge * Tdesign_coefs["c3"])
        + (dum_tanker * Tdesign_coefs["c4"])
        + (Tdesign_coefs["c5"] * dum_container * vessel.L**0.4 * vessel.B**0.6)
        + (Tdesign_coefs["c6"] * dum_dry * vessel.L**0.7 * vessel.B**2.6)
        + (Tdesign_coefs["c7"] * dum_barge * vessel.L**0.3 * vessel.B**1.8)
        + (Tdesign_coefs["c8"] * dum_tanker * vessel.L**0.1 * vessel.B**0.3)
    )

    # Empty draft T_empty, refer to Table 4
    Tempty_coefs = dict(
        {
            "intercept": 7.5740820927 * 10**-2,
            "c1": 1.1615080992 * 10**-1,
            "c2": 1.6865973494 * 10**-2,
            "c3": -2.7490565381 * 10**-2,
            "c4": -5.1501240744 * 10**-5,
            "c5": 1.0257551153 * 10**-1,
            "c6": 2.4299435211 * 10**-1,
            "c7": -2.1354295627 * 10**-1,
        }
    )

    if vessel_type == "Container":
        [dum_container, dum_dry, dum_barge, dum_tanker] = [1, 0, 0, 0]
    elif vessel_type == "Dry_SH":
        [dum_container, dum_dry, dum_barge, dum_tanker] = [0, 0, 0, 0]
    elif vessel_type == "Dry_DH":
        [dum_container, dum_dry, dum_barge, dum_tanker] = [0, 1, 0, 0]
    elif vessel_type == "Barge":
        [dum_container, dum_dry, dum_barge, dum_tanker] = [0, 0, 1, 0]

    elif vessel_type == "Tanker":
        [dum_container, dum_dry, dum_barge, dum_tanker] = [0, 0, 0, 1]

    # dum_container and dum_dry use the same "c5"
    T_empty = (
        Tempty_coefs["intercept"]
        + (Tempty_coefs["c1"] * vessel.B)
        + (Tempty_coefs["c2"] * ((vessel.L * T_design) / vessel.B))
        + (Tempty_coefs["c3"] * (np.sqrt(vessel.L * vessel.B)))
        + (Tempty_coefs["c4"] * (vessel.L * vessel.B * T_design))
        + (Tempty_coefs["c5"] * dum_container)
        + (Tempty_coefs["c5"] * dum_dry)
        + (Tempty_coefs["c6"] * dum_tanker)
        + (Tempty_coefs["c7"] * dum_barge)
    )

    # Capacity indexes, refer to Table 3 and eq 2
    CI_coefs = dict(
        {
            "intercept": 2.0323139721 * 10**1,
            "c1": -7.8577991460 * 10**1,
            "c2": -7.0671612519 * 10**0,
            "c3": 2.7744056480 * 10**1,
            "c4": 7.5588609922 * 10**-1,
            "c5": 3.6591813315 * 10**1,
        }
    )
    # Capindex_1 related to actual draft (especially used for shallow water)
    Capindex_1 = (
        CI_coefs["intercept"]
        + (CI_coefs["c1"] * T_empty)
        + (CI_coefs["c2"] * T_empty**2)
        + (CI_coefs["c3"] * T_strategy)
        + (CI_coefs["c4"] * T_strategy**2)
        + (CI_coefs["c5"] * (T_empty * T_strategy))
    )
    # Capindex_2 related to design draft
    Capindex_2 = (
        CI_coefs["intercept"]
        + (CI_coefs["c1"] * T_empty)
        + (CI_coefs["c2"] * T_empty**2)
        + (CI_coefs["c3"] * T_design)
        + (CI_coefs["c4"] * T_design**2)
        + (CI_coefs["c5"] * (T_empty * T_design))
    )

    # DWT design capacity, refer to Table 6 and eq 3
    capacity_coefs = dict(
        {
            "intercept": -1.6687441313 * 10**1,
            "c1": 9.7404521380 * 10**-1,
            "c2": -1.1068568208,
        }
    )

    DWT_design = (
        capacity_coefs["intercept"]
        + (capacity_coefs["c1"] * vessel.L * vessel.B * T_design)
        + (capacity_coefs["c2"] * vessel.L * vessel.B * T_empty)
    )  # designed DWT
    DWT_actual = (Capindex_1 / Capindex_2) * DWT_design  # actual DWT of shallow water

    if T_strategy < T_design:
        DWT_final = DWT_actual
        # Consumables represents the persentage of fuel weight,which is 4-6% of designed DWT
        # 4% for shallow water (Van Dosser  et al. Chapter 8,pp.68).
        # Based on personal communication with experts we lowered this to 0.005.
        # This should match with the current practice.
        consumables = 0.005

    else:
        DWT_final = DWT_design
        # consumables represents the persentage of fuel weight,which is 4-6% of designed DWT
        # 4% for shallow water (Van Dosser  et al. Chapter 8,pp.68).
        # Based on personal communication with experts we lowered this to 0.005.
        # This should match with the current practice.
        consumables = 0.005

    fuel_weight = DWT_design * consumables  # (Van Dosser et al. Chapter 8, pp.68).

    Payload_computed = DWT_final - fuel_weight  # payload=DWT-fuel_weight

    # DWT_final covers the situations of the DWT at maximum draught and the DWT at adjusted draught
    # We include DWT_final for calculating cargo-fuel trade off by function 'get_adjusted_cargo_amount'.
    return Payload_computed


def Payload2T(vessel, Payload_strategy, vessel_type, bounds=(0, 5)):
    """Calculate the corresponding draught (T_Payload2T) for each Payload_strategy
    the calculation is based on Van Dorsser et al's method (2020) (https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships), which applyies for inland shipping.


    input:
    - Payload_strategy: user given payload
    - vessel types: "Container","Dry_SH","Dry_DH","Barge","Tanker". ("Dry_SH" means dry bulk single hull, "Dry_DH" means dry bulk double hull)
    - bounds: the searching range for draught. As this method which based on Van Dorsser et al (2020) is for inland vessels, of which the draughts are no larger than 5 meter, we set the upper bound as 5 m as default value.

    output:
    - T_Payload2T: corresponding draught for each payload for different vessel types

    """

    def seek_T_given_Payload(T_Payload2T, vessel, vessel_type):
        """function to optimize"""
        Payload_computed = T2Payload(vessel=vessel, T_strategy=T_Payload2T, vessel_type=vessel_type)
        # compute difference between a given payload (Payload_strategy) and a computed payload (Payload_computed)
        diff = Payload_strategy - Payload_computed
        return diff**2

    # fill in some of the parameters that we already know
    fun = functools.partial(seek_T_given_Payload, vessel=vessel, vessel_type=vessel_type)

    # lookup a minimum
    fit = scipy.optimize.minimize_scalar(fun, bounds=bounds, method="bounded")

    # check if we found a minimum
    if not fit.success:
        raise ValueError(fit)

    # the value of fit.x within the bound (0,5) is the draught we find where the diff**2 reach a minimum (zero).
    T_Payload2T = fit.x

    return T_Payload2T


def get_v(vessel, width, depth, margin, bounds):
    """for a waterway section with a given width and depth, compute the velocity that can be
    reached given a vessel's T and a safety margin."""

    def seek_v_given_z(v, vessel, width, depth, margin):
        # calculate sinkage

        z_computed = (vessel.C_B * ((vessel.B * vessel._T) / (width * depth)) ** 0.81) * ((v * 1.94) ** 2.08) / 20

        # calculate available underkeel clearance (vessel in rest)
        z_given = depth - vessel._T

        # compute difference between the sinkage and the space available for sinkage (including safety margin)
        diff = z_given - z_computed - margin

        return diff**2

    # goalseek to minimize
    fun = functools.partial(seek_v_given_z, vessel=vessel, width=width, depth=depth, margin=margin)
    fit = scipy.optimize.minimize_scalar(fun, bounds=bounds, method="bounded")

    # check if we found a minimum
    if not fit.success:
        raise ValueError(fit)

    # the value of fit.x within the bound (0,20) is the velocity we find where the diff**2 reach a minimum (zero).
    v = fit.x

    # calculate the key values again for the resulting speed
    z_computed = (vessel.C_B * ((vessel.B * vessel._T) / (width * depth)) ** 0.81) * ((v * 1.94) ** 2.08) / 20
    depth_squat = depth - z_computed

    return v, depth, depth_squat, z_computed, margin


def get_upperbound_for_power2v(vessel, width, depth, margin=0, bounds=(0, 20)):
    """for a waterway section with a given width and depth, compute a maximum installed-
    power-allowed velocity, considering squat. This velocity can be used as upperbound in the
    power2v function in energy.py. The "upperbound" is the maximum value in velocity searching
    range."""

    # estimate the grounding velocity
    grounding_v, depth, depth_squat, z_computed, margin = get_v(vessel, width, depth, margin=0, bounds=bounds)

    # find max power velocity
    velocity = np.linspace(0.01, grounding_v, 100)
    task = list(itertools.product(velocity[0:-1]))

    # prepare a list of dictionaries for pandas
    rows = []
    for item in task:
        row = {"velocity": item[0]}
        rows.append(row)

    # convert simulations to dataframe, so that we can apply a function and monitor progress
    task_df = pd.DataFrame(rows)

    # creat a results empty list to collect the below results
    results = []
    for i, row in tqdm.tqdm(task_df.iterrows(), disable=True):
        # calculate squat and the waterdepth after squat
        # todo: check if we can replace this with a squat method
        z_computed = (vessel.C_B * ((vessel.B * vessel._T) / (width * depth)) ** 0.81) * ((row["velocity"] * 1.94) ** 2.08) / 20

        # calculate squatted waterdepth
        h_0 = depth - z_computed

        # for the squatted water depth calculate resistance and power
        vessel.calculate_total_resistance(v=row["velocity"], h_0=h_0)
        vessel.calculate_total_power_required(v=row["velocity"], h_0=h_0)

        # prepare a row
        result = {}
        result.update(row)
        result["Powerallowed_v"] = row["velocity"]
        result["P_tot"] = vessel.P_tot
        result["P_installed"] = vessel.P_installed
        result["h_0"] = depth
        result["z_computed"] = z_computed
        result["h_squat"] = h_0

        # update resulst dict
        results.append(result)

    results_df = pd.DataFrame(results)

    # return only dataframe up to and including the first time that P_tot == P_installed
    selected = results_df[0 : (results_df.P_installed >= results_df.P_tot).idxmin()]

    upperbound = selected.Powerallowed_v.max()

    return upperbound, selected, results_df
