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
# 7) consider moving h_min calculation from core.py into strategy.py, since h_min is for strategy. But is it do-able? Any undesired influence to other calculations?


import datetime
import functools
import time
import logging

import numpy as np
import scipy.optimize
import simpy
import opentnsim
logger = logging.getLogger(__name__)



# 6) make the functions elegant (explicy name, dry code, etc.) in strategy.py and "optimal sailing stratigies notebook", then test again.

# 8)fix error of payload_2_T functions in strategy.py 



def T2v(vessel, h_min, bounds=(0, 10)):
    """Compute the maximum velocity a vessel can have without touching the safety margin above bed in the bottleneck section

    bounds is the limits where to look for a solution for the velocity [m/s]
    returns velocity [m/s]

    the maximum velocity a ship should not exceed v_T2v to prevent grounding
    """

    def seek_v_given_T(v, vessel, h_min):
        """function to optimize"""

        # calculate the maximum ship sinkage (z) in the waterway section with minimum water depth
        h_min = vessel.h_min
        z = (vessel.C_B * ((vessel.B * vessel.T) / (150 * vessel.h_min)) ** 0.81) * ((v*1.94) ** 2.08) / 20

        # compute difference between a given draught (T_strategy) and a computed draught (T_computed)
        T_strategy = vessel._T  # the user provided draught, which is stored in the vessel properties
        T_computed = vessel.h_min - z - vessel.safety_margin # the comupted draught
        diff = T_strategy - T_computed


        return diff ** 2

    # fill in some of the parameters that we already know
    fun = functools.partial(seek_v_given_T, vessel=vessel, h_min = vessel.h_min)

    # lookup a minimum
    fit = scipy.optimize.minimize_scalar(fun, bounds=bounds, method='bounded')

    # check if we found a minimum
    if not fit.success:
        raise ValueError(fit)

    # the value of fit.x within the bound (0,10) is the velocity we find where the diff**2 reach a minimum (zero).
    v_T2v =  fit.x


    return v_T2v

def get_upperbound_for_power2v(vessel, width, depth, bounds=(0,20)):
    """ for a waterway section with a given width and depth, compute a maximum installed-
    power-allowed velocity, considering squat. This velocity is set as upperbound in the 
    power2v function in energy.py "upperbound" is the maximum value in velocity searching 
    range.
    """
    
    def get_grounding_v(vessel, width, depth, bounds):
        
        def seek_v_given_z(v, vessel, width, depth):
            # calculate sinkage
            z_computed = (vessel.C_B * ((vessel.B * vessel._T) / (width * depth)) ** 0.81) * ((v*1.94) ** 2.08) / 20
            
            # calculate available underkeel clearance (vessel in rest)
            z_given = depth - vessel._T
            
            # compute difference between the sinkage and the space available for sinkage
            diff = z_given - z_computed

            return diff ** 2
        
        # goalseek to minimize
        fun = functools.partial(seek_v_given_z, vessel=vessel, width=width, depth=depth)
        fit = scipy.optimize.minimize_scalar(fun, bounds=bounds, method='bounded')
        
        # check if we found a minimum
        if not fit.success:
            raise ValueError(fit)

        # the value of fit.x within the bound (0,20) is the velocity we find where the diff**2 reach a minimum (zero).
        grounding_v =  fit.x
        
        print('grounding velocity {:.2f} m/s'.format(grounding_v))
        
        return grounding_v                      

    # create a large velocity[m/s] range for both inland shipping and seagoing shipping
    grounding_v = get_grounding_v(vessel, width, depth, bounds)
    velocity = np.linspace(0.01, grounding_v, 1000) 
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
    for i, row in tqdm.tqdm(task_df.iterrows()):
        h_0 = depth      
        velocity = row['velocity']
        
        # calculate squat and the waterdepth after squat
        z_computed = (vessel.C_B * ((vessel.B * vessel._T) / (150 * h_0)) ** 0.81) * ((velocity*1.94) ** 2.08) / 20
        h_0 = depth - z_computed
        
        # for the squatted water depth calculate resistance and power
        # vessel.calculate_properties()
        # vessel.calculate_frictional_resistance(v=velocity, h_0=h_0)
        vessel.calculate_total_resistance(v=velocity, h_0=h_0)
        P_tot = vessel.calculate_total_power_required(v=velocity)
        
        # prepare a row
        result = {}
        result.update(row)
        result['Powerallowed_v'] = velocity
        result['P_tot'] = P_tot
        result['P_installed'] = vessel.P_installed
        
        # update resulst dict
        results.append(result)
    
    results_df = pd.DataFrame(results)

    selected = results_df.query('P_tot < P_installed')
    upperbound = max(selected['Powerallowed_v'])
    
    return upperbound



def P2v(vessel, h_min , upperbound):
    """Compute the maximum vessel velocity limited by the installed engine power (P_installed) in the bottleneck section

    bounds is the limits where to look for a solution for the velocity [m/s]
    returns velocity [m/s]

    the maximum velocity a ship can sail should be no larger than v_P2v due to the installed engine power limitation
    """

    def seek_v_given_P_installed(v, vessel, h_min):
        """function to optimize"""

        # to get vessel.P_tot, we need to comupute total resistance and total power required first
        # compute total resistance with the minimum water depth
        h_min = vessel.h_min
        h_min = vessel.calculate_h_squat(v, h_min)
        vessel.calculate_total_resistance(v, h_min)
        # compute total power given
        vessel.calculate_total_power_required(v=v)

        if isinstance(vessel.P_tot, complex):
            raise ValueError(f"P tot is complex: {vessel.P_tot}")

        # calculate difference between installed engine power(already given by user in vessel properties) and the computed total power
        diff = vessel.P_installed - vessel.P_tot

        return diff ** 2

    # fill in some of the parameters that we already know
    fun = functools.partial(seek_v_given_P_installed, vessel=vessel, h_min = vessel.h_min)
    # lookup a minimum
    fit = scipy.optimize.minimize_scalar(fun, bounds=(0, upperbound), method='bounded')

    # the value of fit.x within the bound (0,10) is the velocity we find where the diff**2 reach a minimum (zero).
    v_P2v = fit.x

    # check if we found a minimum
    if not fit.success:
        raise ValueError(fit)
    logger.debug(f"fit: {fit}")

    return v_P2v


def get_v_max_for_bottleneck(FG, path, vessel, T_strategy):
    """Compute the maximum speed a vessel can sail at with a given draught to pass a waterway section with limited water depth (bottleneck section).

    assumption:
    - the fuel weight remains a same value thus doesn't influence ship's actual draught while sailing. Later we will add the "burning lighter" and "refueling heavier" functions to take into account the influence of the variation of fuel weight to the actual draught and cargo capacity (payload).

    input:
    - FG: a graph with nodes and edges. It is an 1D network along which the vessel will move.
    - path: the route on which the vessel will move. It's a list of nodes pairs.
    - vessel: a vessel object with vessel properties and TransportResource (ConsumesEnergy mixin classes)
    - T_strategy: possible draughts given by the user considering the vessel size and water depth

    output:
    - v_T2v: the maximum speed a vessel can sail at without touching the safety margin above the bed
    - v_P2v: the maximum speed a vessel can sail at within the installed engine power
    - v_max_final: the final maximum speed calculated for passing the bottleneck section, which is the smaller one between v_T2v and v_P2v
    """

    # Start simpy environment
    simulation_start = datetime.datetime.now()
    env = simpy.Environment(initial_time = time.mktime(simulation_start.timetuple()))
    env.epoch = time.mktime(simulation_start.timetuple())

    # Add graph to environment
    env.FG = FG

    # get v_T2v in the bottleneck section
    v_T2v  = T2v(vessel, h_min = vessel.h_min)
    # get v_P2v in the bottleneck section
    upperbound = opentnsim.energy.get_upperbound_for_power2v(vessel, width=150, depth=2.5)
    v_P2v = P2v(vessel, h_min = vessel.h_min, upperbound = upperbound)
    # get final maximum velocity in the bottleneck section
    v_max_final = min(v_T2v, v_P2v)

    # Start the simulation
    env.process(vessel.move())
    env.run()

    return v_T2v, v_P2v, v_max_final


# To know the corresponding Payload for each T_strategy
def T2Payload(vessel, T_strategy, vessel_type):
    """ Calculate the corresponding payload for each T_strategy
    the calculation is based on Van Dorsser et al's method (2020) (https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)


    input:
    - T_strategy: user given possible draught
    - vessel types: "Container","Dry_SH","Dry_DH","Barge","Tanker". ("Dry_SH" means dry bulk single hull, "Dry_DH" means dry bulk double hull)

    output:
    - Payload_comupted: corresponding payload for the T_strategy for different vessel types

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
    Payload_comupted = DWT_actual-fuel_weight # payload=DWT-fuel_weight


    return Payload_comupted


# def Payload2T(vessel, Payload_strategy, vessel_type, bounds=(0, 40)):
#     """ Calculate the corresponding draught (T_Payload2T) for each Payload_strategy
#     the calculation is based on Van Dorsser et al's method (2020) (https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)


#     input:
#     - Payload_strategy: user given payload
#     - vessel types: "Container","Dry_SH","Dry_DH","Barge","Tanker". ("Dry_SH" means dry bulk single hull, "Dry_DH" means dry bulk double hull)

#     output:
#     - T_Payload2T: corresponding draught for each payload for different vessel types

#     """

#     def seek_T_given_Payload(Payload_strategy, vessel, vessel_type):
#         """function to optimize"""

#         Payload_computed = T2Payload(vessel, T_strategy, vessel_type)
#         # compute difference between a given payload (Payload_strategy) and a computed payload (Payload_computed)
#         diff = Payload_strategy - Payload_computed

#         return diff ** 2

#     # fill in some of the parameters that we already know
#     fun = functools.partial(seek_T_given_Payload, vessel=vessel)

#     # lookup a minimum
#     fit = scipy.optimize.minimize_scalar(fun, bounds=bounds, method='bounded')

#     # check if we found a minimum
#     if not fit.success:
#         raise ValueError(fit)

#     # the value of fit.x within the bound (0,10) is the velocity we find where the diff**2 reach a minimum (zero).
#         T_Payload2T =  fit.x


#     return T_Payload2T
