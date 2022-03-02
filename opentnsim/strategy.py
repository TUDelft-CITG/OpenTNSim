"""Sailing strategies for OpenTNSim. This is the place to put logic like sailing limited by engine, draught, velocity, fuel usage."""
import datetime
import functools
import time
import logging

import numpy as np
import scipy.optimize
import simpy

logger = logging.getLogger(__name__)


def T2v(vessel, h_min, bounds=(0, 10)):
    """Compute vessel velocity given the minimum water depth and possible actual draught

    bounds is the limits where to look for a solution for the velocity [m/s]
    returns velocity [m/s]
    """

    def seek_v_given_T(v, vessel, h_min):
        """function to optimize"""

        # compute the maximum draught a vessel can have to pass the minimum water depth section,
        # considering the maximum squat while sailing in limited water depth.
        h_min = vessel.h_min
        #z = (vessel.C_B * (1.94 * v)**2) * (6 * vessel.B * vessel.T / (150 * vessel.h_min) + 0.4) / 100 # vessel.T is the computed T
        # Here we use the standard width 150 m as the waterway width
        z = (vessel.C_B * ((vessel.B * vessel.T) / (150 * vessel.h_min)) ** 0.81) * (v ** 2.08) / 20
        # print('z: {:.2f} m'.format(z))
        # compute difference between given draught (T_strategy) and computed draught (T_computed)
        T_strategy = vessel._T  # the user provided T
        T_computed = vessel.h_min - z - vessel.safety_margin
        # T_computed = vessel.h_min - z
        diff = T_strategy - T_computed

        # print('T_strategy: {:.2f} m'.format(T_strategy))
        # print('T_computed: {:.2f} m'.format(T_computed))
        # print('T_strategy - T_computed: {:.2f} m'.format(diff))

#        logger.debug(f'optimizing for v: {v}, T_strategy: {T_strategy}, T: {vessel.T}'T_computed: {:.2f} m'.format(T_computed), T: {vessel.T}')

        return diff ** 2

#    print('')
#    print('I am in T2v')

    # fill in some of the parameters that we already know
    fun = functools.partial(seek_v_given_T, vessel=vessel, h_min = vessel.h_min)

    # lookup a minimum
    fit = scipy.optimize.minimize_scalar(fun, bounds=bounds, method='bounded')

    # check if we found a minimum
    if not fit.success:
        raise ValueError(fit)

    v_T2v =  fit.x


    return v_T2v



def P2v(vessel, h_min, bounds=(0, 10)):
    """Compute vessel velocity given an edge and power (P_tot_given)

    bounds is the limits where to look for a solution for the velocity [m/s]
    returns velocity [m/s]
    """

    def seek_v_given_P_installed(v, vessel, h_min):
        """function to optimize"""
        # water depth from the edge
        h_min = vessel.h_min
        # TODO: consider precomputing a range v/h combinations for the ship before the simulation starts
        vessel.calculate_total_resistance(v, h_min)
        # compute total power given
        vessel.calculate_total_power_required(v=v)
        if isinstance(vessel.P_tot, complex):
            raise ValueError(f"P tot is complex: {vessel.P_tot}")

        # compute difference between power setting by captain and power needed for velocity
        diff = vessel.P_installed - vessel.P_tot
        #logger.debug(f'optimizing for v: {v}, P_tot_given: {vessel.P_tot_given}, P_tot {vessel.P_tot}, P_given {P_given}')
        return diff ** 2

    # fill in some of the parameters that we already know
    fun = functools.partial(seek_v_given_P_installed, vessel=vessel, h_min = vessel.h_min)
    # lookup a minimum
    fit = scipy.optimize.minimize_scalar(fun, bounds=bounds, method='bounded')
    v_P2v = fit.x

    # check if we found a minimum
    if not fit.success:
        raise ValueError(fit)
    logger.debug(f"fit: {fit}")
    # return fit.x
    return v_P2v

# To get maximum velocity (v_computed) for each T_strategy considering maximum sinkage and safety margin
# here we need Graph (FG) and path (overall) to know the minimum waterdepth h_min along the route for planning strategies
def formulate_sailing_strategies(FG, path, vessel, T_strategy):
    """TODO: add docstring here..."""

    # Start simpy environment
    simulation_start = datetime.datetime.now()
    env = simpy.Environment(initial_time = time.mktime(simulation_start.timetuple()))
    env.epoch = time.mktime(simulation_start.timetuple())

    # Add graph to environment
    env.FG = FG


    # get T2v
    v_T2v  = T2v(vessel, h_min = vessel.h_min)
    # get P2v
    v_P2v = P2v(vessel, h_min = vessel.h_min)
    #print('v_computed: {:.2f} m/s'.format(v_computed))

    #print('z: {:.2f} m'.format(z))
    # Start the simulation
    env.process(vessel.move())
    env.run()


    return v_T2v, v_P2v


# To know the corresponding Payload for each T_strategy
def T2Payload(vessel, T_strategy, vessel_type):
    """ Calculate payload based on Van Dorsser et al
    (https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)

    Given a route we get several possible draught (T_strategy) with veolicity (v_computed) combinations for the moving vessel, this step is done via T2v simulation.
    At this step, we calculate the payload for the possible draughts (T_strategy).

    Input:
    - T_strategy
    - vessel types: "Container","Dry_SH","Dry_DH","Barge","Tanker"

    Output:
    - corresponding payload for the T_strategy for different vessel types

    """
    #Design draft T_design, refer to Table 5

    Tdesign_coefs = dict({"intercept":0,
                     "c1": 1.7244153371,
                     "c2": 2.2767179246,
                     "c3": 1.3365379898,
                     "c4": -5.9459308905,
                     "c5": 6.2902305560*10**-2,
                     "c6": 7.7398861528*10**-5,
                     "c7": 9.0052384439*10**-3,
                     "c8": 2.8438560877
                     })

    assert vessel_type in ["Container","Dry_SH","Dry_DH","Barge","Tanker"],'Invalid value vessel_type, should be "Container","Dry_SH","Dry_DH","Barge" or "Tanker"'
    if vessel_type == "Container":
        [dum_container, dum_dry,
        dum_barge, dum_tanker] = [1,0,0,0]
    elif vessel_type == "Dry_SH":
        [dum_container, dum_dry,
        dum_barge, dum_tanker] = [0,1,0,0]
    elif vessel_type == "Dry_DH":
        [dum_container, dum_dry,
        dum_barge, dum_tanker] = [0,1,0,0]
    elif vessel_type == "Barge":
        [dum_container, dum_dry,
        dum_barge, dum_tanker] = [0,0,1,0]
    elif vessel_type == "Tanker":
        [dum_container, dum_dry,
        dum_barge, dum_tanker] = [0,0,0,1]

    T_design = Tdesign_coefs['intercept'] + (dum_container * Tdesign_coefs['c1']) + \
                                            (dum_dry * Tdesign_coefs['c2']) + \
                                            (dum_barge * Tdesign_coefs['c3']) +\
                                            (dum_tanker * Tdesign_coefs['c4']) +\
                                            (Tdesign_coefs['c5'] * dum_container * vessel.L**0.4 * vessel.B**0.6) +\
                                            (Tdesign_coefs['c6'] * dum_dry * vessel.L**0.7 * vessel.B**2.6)+\
                                            (Tdesign_coefs['c7'] * dum_barge * vessel.L**0.3 * vessel.B**1.8) +\
                                            (Tdesign_coefs['c8'] * dum_tanker * vessel.L**0.1 * vessel.B**0.3)

    #Empty draft T_empty, refer to Table 4

    Tempty_coefs = dict({"intercept": 7.5740820927*10**-2,
                "c1": 1.1615080992*10**-1,
                "c2": 1.6865973494*10**-2,
                "c3": -2.7490565381*10**-2,
                "c4": -5.1501240744*10**-5,
                "c5": 1.0257551153*10**-1,
                "c6": 2.4299435211*10**-1,
                "c7": -2.1354295627*10**-1,
                })


    if vessel_type == "Container":
        [dum_container, dum_dry,
        dum_barge, dum_tanker] = [1,0,0,0]
    elif vessel_type == "Dry_SH":
        [dum_container, dum_dry,
        dum_barge, dum_tanker] = [0,0,0,0]
    elif vessel_type == "Dry_DH":
        [dum_container, dum_dry,
        dum_barge, dum_tanker] = [0,1,0,0]
    elif vessel_type == "Barge":
        [dum_container, dum_dry,
        dum_barge, dum_tanker] = [0,0,1,0]

    elif vessel_type == "Tanker":
        [dum_container, dum_dry,
        dum_barge, dum_tanker] = [0,0,0,1]

    # dum_container and dum_dry use the same "c5"
    T_empty = Tempty_coefs['intercept']  + (Tempty_coefs['c1'] * vessel.B) + \
                                           (Tempty_coefs['c2'] * ((vessel.L * T_design) / vessel.B)) + \
                                           (Tempty_coefs['c3'] * (np.sqrt(vessel.L * vessel.B)))  + \
                                           (Tempty_coefs['c4'] * (vessel.L * vessel.B * T_design)) +  \
                                           (Tempty_coefs['c5'] * dum_container) + \
                                           (Tempty_coefs['c5'] * dum_dry)   + \
                                           (Tempty_coefs['c6'] * dum_tanker) + \
                                           (Tempty_coefs['c7'] * dum_barge)



    #Capacity indexes, refer to Table 3 and eq 2
    CI_coefs = dict({"intercept": 2.0323139721 * 10**1,

            "c1": -7.8577991460 * 10**1,
            "c2": -7.0671612519 * 10**0,
            "c3": 2.7744056480 * 10**1,
            "c4": 7.5588609922 * 10**-1,
            "c5": 3.6591813315 * 10**1
            })
    # Capindex_1 related to actual draft (especially used for shallow water)
    Capindex_1 = CI_coefs["intercept"] + (CI_coefs["c1"] * T_empty) + (CI_coefs["c2"] * T_empty**2)  +  (
    CI_coefs["c3"] * T_strategy) + (CI_coefs["c4"] * T_strategy**2)   + ( CI_coefs["c5"] * (T_empty * T_strategy))
    # Capindex_2 related to design draft
    Capindex_2 = CI_coefs["intercept"] + (CI_coefs["c1"] * T_empty) + (CI_coefs["c2"] * T_empty**2)   + (
    CI_coefs["c3"] * T_design) + (CI_coefs["c4"] * T_design**2)  + (CI_coefs["c5"] * (T_empty * T_design))

    #DWT design capacity, refer to Table 6 and eq 3
    capacity_coefs = dict({"intercept": -1.6687441313*10**1,
         "c1": 9.7404521380*10**-1,
         "c2": -1.1068568208,
         })

    DWT_design = capacity_coefs['intercept'] + (capacity_coefs['c1'] * vessel.L * vessel.B * T_design) + (
     capacity_coefs['c2'] * vessel.L * vessel.B * T_empty) # designed DWT
    DWT_actual = (Capindex_1/Capindex_2)*DWT_design # actual DWT of shallow water


    if T_strategy < T_design:
        consumables=0.04 #consumables represents the persentage of fuel weight,which is 4-6% of designed DWT
                          # 4% for shallow water (Van Dosser  et al. Chapter 8,pp.68).

    else:
        consumables=0.06 #consumables represents the persentage of fuel weight,which is 4-6% of designed DWT
                          # 6% for deep water (Van Dosser et al. Chapter 8, pp.68).

    fuel_weight=DWT_design*consumables #(Van Dosser et al. Chapter 8, pp.68).
    Payload_strategy = DWT_actual-fuel_weight # payload=DWT-fuel_weight


    return Payload_strategy
