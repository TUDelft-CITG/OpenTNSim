"""Sailing strategies for OpenTNSim. This is the place to put logic like sailing limited by engine, draught, velocity, fuel usage."""
import datetime
import functools
import time
import logging

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
