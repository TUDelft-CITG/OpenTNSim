"""Module with alogorithms necessary for the energy module"""

# %% IMPORT DEPENENDENCIES
# generic
import pathlib
import logging
import functools
import numpy as np
import pandas as pd
import scipy.optimize

# OpenTNSim
import opentnsim

# logging
logger = logging.getLogger(__name__)


def power2v(vessel, edge, upperbound):
    """Compute vessel velocity given an edge and power (P_tot_given)

    bounds is the limits where to look for a solution for the velocity [m/s]
    returns velocity [m/s]
    """

    assert isinstance(vessel, opentnsim.core.vessel_properties.VesselProperties), "vessel should be an instance of VesselProperties"
    assert vessel.C_B is not None, "C_B cannot be None"

    def seek_v_given_power(v, vessel, edge):
        """function to optimize"""
        # TODO: check it this needs to be made more general, now relies on ['Info'] to be present
        # water depth from the edge
        h_0 = edge["Info"]["GeneralDepth"]
        try:
            h_0 = vessel.calculate_h_squat(v, h_0)
        except AttributeError:
            # no squat available
            pass
        # TODO: consider precomputing a range v/h combinations for the ship before the simulation starts
        vessel.calculate_total_resistance(v, h_0)

        # compute total power given
        P_given = vessel.calculate_total_power_required(v=v, h_0=h_0)
        if isinstance(vessel.P_tot, complex):
            raise ValueError(f"P tot is complex: {vessel.P_tot}")

        # compute difference between power setting by captain (incl hotel) and power needed for velocity (incl hotel)
        diff = vessel.P_tot_given - P_given  # vessel.P_tot
        logger.debug(f"optimizing for v: {v}, P_tot_given: {vessel.P_tot_given}, P_tot {vessel.P_tot}, P_given {P_given}")

        return diff**2

    # fill in some of the parameters that we already know
    fun = functools.partial(seek_v_given_power, vessel=vessel, edge=edge)
    # lookup a minimum
    fit = scipy.optimize.minimize_scalar(fun, bounds=(0, upperbound), method="bounded", options=dict(xatol=0.0000001))

    # check if we found a minimum
    if not fit.success:
        raise ValueError(fit)
    logger.debug(f"fit: {fit}")

    return fit.x
