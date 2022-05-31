# package(s) related to the simulation
import numpy as np
import bisect
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt

# spatial libraries
import pyproj

class VesselTrafficService:
    """Class: a collection of functions that processes requests of vessels regarding the nautical processes on ow to enter the port safely"""

    def provide_sailing_direction(vessel,edge):
        initial_bound = vessel.bound
        network = vessel.env.FG
        phase_lag = network.edges[edge]['Info']['Tidal phase lag']
        if phase_lag < 0: bound = 'outbound'
        elif phase_lag > 0: bound = 'inbound'
        else: bound = initial_bound
        return bound

    def provide_ukc_clearance(vessel,node,delay=0,components_calc = False):
        """ Function: calculates the sail-in-times for a specific vssel with certain properties and a pre-determined route and provides this information to the vessel

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node:
                - components_calc:

        """

        interp_wdep, ukc_s, ukc_p, fwa = VesselTrafficService.provide_sail_in_times_tidal_window(vessel, [node], ukc_calc=True)
        net_ukc = interp_wdep(vessel.env.now+delay) - (vessel.T_f + vessel.metadata['ukc'])

        if components_calc == False:
            return net_ukc
        else:
            if net_ukc < ukc_s+ukc_p+fwa:
                return ukc_s+ukc_p+fwa
            else:
                return net_ukc

    def provide_sail_in_times_tidal_window(vessel,route,plot=False,sailing_time_correction=True,visualization_calculation=False,ukc_calc=False):
        """ Function: calculates the sail-in-times for a specific vessel with certain properties and a pre-determined route and provides this information to the vessel

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                - plot: provide a visualization of the calculation for each vessel
                - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)
                - visualization_calculation: a bool that indicates whether the calculation should be made for a single node or not (for a route with multiple nodes)

        """

        # Functions used to calculate the sail-in-times for a specific vessel
        def tidal_window_restriction_determinator(vessel, route, types, specifications, node, sailing_time_to_next_node):
            """ Function: determines which tidal window restriction applies to the vessel at the specific node

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route:
                    - types: the type of the restriction
                    - specifications: the specific data regarding the properties for which the restriction holds
                    - node: a string that defines the node of the tidal window restriction

            """

            # Predefined bool
            boolean = True
            no_tidal_window = True
            # Determining if and which restriction applies for the vessel by looping over the restriction class
            for restriction_class in enumerate(specifications[0]):
                # - if restriction does not apply to vessel because it is for vessels sailing in the opposite direction: continue loop
                if vessel.bound != specifications[2][restriction_class[0]]:
                    continue
                if specifications[5][restriction_class[0]] != [] and (specifications[5][restriction_class[0]][0] not in route or specifications[5][restriction_class[0]][1] not in route):
                    continue
                # - else: looping over the restriction criteria
                for restriction_type in enumerate(restriction_class[1]):
                    # - if previous condition is not met and there are no more restriction criteria and the previous condition has an 'AND' boolean statement: continue loop
                    if not boolean and restriction_type[0] == len(restriction_class[1]) - 1 and specifications[-1][restriction_class[0]][restriction_type[0] - 1] == 'and':
                        continue
                    # - if previous condition is not met and there are more restriction criteria and the next condition has an 'AND' boolean statement: continue loop
                    if not boolean and restriction_type[0] != len(restriction_class[1]) - 1 and specifications[-1][restriction_class[0]][restriction_type[0]] == 'and':
                        continue
                    # - if previous condition is not met and the next condition has an 'OR' boolean statement: continue loop with predefined boolean
                    if not boolean and restriction_type[0] != len(restriction_class[1]) - 1 and specifications[-1][restriction_class[0]][restriction_type[0]] == 'or':
                        boolean = True
                        continue

                    # Extracting the correct vessel property for the restriction type
                    if restriction_type[1].find('Length') != -1: value = getattr(vessel, 'L')
                    if restriction_type[1].find('Draught') != -1: value = getattr(vessel, 'T_f')
                    if restriction_type[1].find('Beam') != -1: value = getattr(vessel, 'B')
                    if restriction_type[1].find('UKC') != -1: value = VesselTrafficService.provide_ukc_clearance(vessel,node,sailing_time_to_next_node,components_calc = True)
                    if restriction_type[1].find('Type') != -1: value = getattr(vessel, 'type')
                    # Determine if the value for the property satisfies the condition of the restriction type
                    df = pd.DataFrame({'Value': [value],'Restriction': [specifications[1][restriction_class[0]][restriction_type[0]]]})
                    boolean = df.eval('Value' + specifications[3][restriction_class[0]][restriction_type[0]] + 'Restriction')[0]

                    # - if condition is not met: continue loop
                    if not boolean and restriction_type[0] != len(restriction_class[1]) - 1:
                        continue

                    # - if one of the conditions is met and the restriction contains an 'OR' boolean statement:
                    if boolean and restriction_type[0] != len(restriction_class[1]) - 1 and specifications[-1][restriction_class[0]][restriction_type[0]] == 'or':
                        no_tidal_window = False
                        break
                    # - elif all the conditions are met and the restriction contains solely 'AND' boolean statements:
                    elif boolean and restriction_type[0] == len(restriction_class[1]) - 1:
                        no_tidal_window = False
                        break

                # - if condition is met: break the loop
                if boolean == True:
                    break

                # - else: restart the loop with predefined bool
                else:
                    boolean = True

            return restriction_class[0], no_tidal_window

        def times_vertical_tidal_window(vessel,route,axis=[],plot=False,sailing_time_correction=True,ukc_calc=False):
            """ Function: calculates the windows available to sail-in and -out of the port given the vertical tidal restrictions according to the tidal window policy.

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                    - plot: provide a visualization of the calculation for each vessel
                    - axis: axes class from the matplotlib package
                    - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

            """

            # Function used in the determination of the calculation of the sail-in-times given the policy for determining the vertical tidal windows
            def minimum_available_water_depth_along_route(vessel, route, axis=[], plot=False, sailing_time_correction=True, ukc_calc=False):
                """ Function: calculates the minimum available water depth (predicted/modelled/measured water level minus the local maintained bed level) along the route over time,
                              subtracted with the difference between the gross ukc and net ukc (hence: subtracted with additional safety margins consisting of vessel-related factors
                              and water level factors). The bottom-related factors are already accounted for in the use of the MBL instead of the actual depth.

                    Input:
                        - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                        - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                        - plot: provide a visualization of the calculation for each vessel
                        - axis: axes class from the matplotlib package
                        - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

                """

                # Continuation of the calculation of the minimum available water depth along the route of the vessel by predefining some parameters
                network = vessel.env.FG
                distance_to_next_node = 0
                ukc = []
                wdep_nodes = []
                t_min_wdep = network.nodes[route[0]]['Info']['Water level'][0]
                min_wdep = []

                # Start of calculation by looping over the nodes of the route
                for nodes in enumerate(route):
                    # - if a correction for the sailing time should be applied: the total distance should be keep track of
                    if sailing_time_correction and not ukc_calc:
                        if not nodes[1] == route[0]:
                            distance_to_next_node += pyproj.Geod(ellps='WGS84').inv(network.nodes[route[nodes[0] - 1]]['geometry'].x,
                                                                                    network.nodes[route[nodes[0] - 1]]['geometry'].y,
                                                                                    network.nodes[route[nodes[0]]]['geometry'].x,
                                                                                    network.nodes[route[nodes[0]]]['geometry'].y)[2]

                    # Sailing time is calculated by the distance of the route up to the specific node, divided by the sailing speed (if no correction for the sailing time should be applied: distance is always 0)
                    sailing_time_to_next_node = distance_to_next_node / vessel.v

                    # Importing some node specific data on the time series of the water levels, depth, and ukc policy (corrected for the sailing time if correction was applied)
                    t_wlev = [t - sailing_time_to_next_node for t in network.nodes[route[nodes[0]]]['Info']['Water level'][0]]
                    wlev = network.nodes[route[nodes[0]]]['Info']['Water level'][1]
                    depth = network.nodes[route[nodes[0]]]['Info']['MBL']
                    types = network.nodes[nodes[1]]['Info']['Vertical tidal restriction']['Type']
                    specifications = network.nodes[nodes[1]]['Info']['Vertical tidal restriction']['Specification']

                    # Determine which restriction applies to vessel
                    restriction_index, _ = tidal_window_restriction_determinator(vessel, route, types, specifications, nodes[1], sailing_time_to_next_node)

                    # Calculate ukc policy based on the applied restriction
                    ukc_s = types[0][restriction_index][0]
                    ukc_p = types[1][restriction_index][0] * vessel.T_f
                    fwa = types[2][restriction_index][0] * vessel.T_f

                    # Interpolation of the available water depth over time and append data to list of the available water depths for the whole route
                    interp_wdep = sc.interpolate.CubicSpline(t_wlev, [y + depth - ukc_s - ukc_p - fwa for y in wlev])
                    if ukc_calc:
                        return interp_wdep, ukc_s, ukc_p, fwa
                    wdep = interp_wdep(t_min_wdep)
                    wdep_nodes.append(wdep)

                    # Add to axes of the plot (if plot is requested): the available water depths at all the nodes of the route
                    if plot:
                        axis.plot(t_min_wdep, wdep, color='lightskyblue', alpha=0.4)

                # Pick the minimum of the water depths for each time and each node
                min_wdep = [min(idx) for idx in zip(*wdep_nodes)]

                return t_min_wdep, min_wdep

            #Continuation of the calculation of the windows given the vertical tidal restrictions by setting some parameters
            if not ukc_calc:
                times_vertical_tidal_window = []
                water_depth_required = vessel.T_f + vessel.metadata['ukc'] #vessel draught + additional vessel-related factors (if applicable)

            #Calculation of the minimum available water depth along the route of the vessel
            if not ukc_calc:
                new_t, min_wdep = minimum_available_water_depth_along_route(vessel, route, axis, plot, sailing_time_correction, ukc_calc)
            else:
                interp_wdep, ukc_s, ukc_p, fwa = minimum_available_water_depth_along_route(vessel, route, ukc_calc=True)
                return interp_wdep, ukc_s, ukc_p, fwa

            #Interpolation of the net ukc
            root_interp_water_level_at_edge = sc.interpolate.CubicSpline(new_t,[x-water_depth_required for x in min_wdep])

            #If there is not enough available water depth over time: no tidal window
            if np.max([x - water_depth_required for x in min_wdep]) < 0:
                times_vertical_tidal_window.append([vessel.waiting_time_start, 'Start']) #tidal restriction starts at t_start
                times_vertical_tidal_window.append([vessel.env.now+vessel.metadata['max_waiting_time'], 'Stop']) #tidal restriction ends at t_end
            #Else if there is enough available water depth at certain moments: calculate tidal windows by looping over the roots of the interpolation
            else:
                for root in root_interp_water_level_at_edge.roots():
                    #-if root falls within time series of the data on the net ukc, and the net ukc a moment later than the root is higher than the required water depth:
                    if root > new_t[0] and root < new_t[-1] and min_wdep[bisect.bisect_right(new_t,root)] > water_depth_required:
                        # -if there are no values in the list of vertical tidal windows yet or there are values and the last value indicates that the tidal restriction has started and not ended and the net ukc a moment ealier than the root is less than the required water depth
                        if (times_vertical_tidal_window == [] or (times_vertical_tidal_window != [] and times_vertical_tidal_window[-1][1] != 'Stop')) and min_wdep[bisect.bisect_right(new_t,root)-1] < water_depth_required:
                            times_vertical_tidal_window.append([root, 'Stop']) #tidal restriction ends at t=root
                    # -if root falls within time series of the data on the net ukc, and the net ukc a moment later than the root is less than the required water depth:
                    elif root > new_t[0] and root < new_t[-1] and min_wdep[bisect.bisect_right(new_t,root)] < water_depth_required:
                        # -if there are no values in the list of vertical tidal windows yet or there are values and the last value indicates that the tidal restriction has ended and not ended and the net ukc a moment ealier than the root is higher than the required water depth
                        if (times_vertical_tidal_window == [] or (times_vertical_tidal_window != [] and times_vertical_tidal_window[-1][1] != 'Start')) and min_wdep[bisect.bisect_right(new_t,root)-1] > water_depth_required:
                            times_vertical_tidal_window.append([root, 'Start']) #tidal restriction starts at t=root

                #If the sail-in or -out-times given the vertical tidal restrictions are not empty: set the initial value at t=0
                if times_vertical_tidal_window != []:
                    #-if the first value in the list indicates that the tidal restriction starts: append that the tidal restriction ends at t=0
                    if times_vertical_tidal_window[0][1] == 'Start' and times_vertical_tidal_window[0][0] > vessel.waiting_time_start:
                            times_vertical_tidal_window.insert(0,[vessel.waiting_time_start, 'Stop'])
                    # -if the first value in the list indicates that the tidal restriction stop: append that the tidal restriction starts at t=0
                    if times_vertical_tidal_window[0][1] == 'Stop' and times_vertical_tidal_window[0][0] > vessel.waiting_time_start:
                            times_vertical_tidal_window.insert(0,[vessel.waiting_time_start, 'Start'])

            #If there are still no sail-in or -out-times (meaning the vessel can enter regardless the vertical tidal restriction): set start- and end-times
            if times_vertical_tidal_window == []:
                times_vertical_tidal_window.append([vessel.waiting_time_start, 'Stop']) #tidal restriction stops at t_start
                times_vertical_tidal_window.append([vessel.env.now+vessel.metadata['max_waiting_time'], 'Start']) #tidal restriction starts at t_end

            #If plot is requested: plot the minimum available water depth, the required water depth of the vessel, the resulting vertical tidal windows, and add some lay-out
            if plot:
                axis.plot(new_t,min_wdep,color='deepskyblue')
                axis.plot([x[0] for x in times_vertical_tidal_window], (vessel.metadata['ukc']+vessel.T_f) * np.ones(len(times_vertical_tidal_window)), color='deepskyblue',marker='o',linestyle='None')
                axis.text(vessel.env.now+vessel.metadata['max_waiting_time'], 1.01*(vessel.metadata['ukc']+vessel.T_f), 'Required water depth', color='deepskyblue',horizontalalignment='center')
                axis.axhline((vessel.metadata['ukc']+vessel.T_f), color='deepskyblue', linestyle='--')
                axis.set_ylim([0,vessel.T_f+5])

            #Return the sail-in or -out-times given the vertical tidal restrictions over the route of the vessel
            return times_vertical_tidal_window

        def times_horizontal_tidal_window(vessel,route,axis=[],plot=plot,sailing_time_correction=True,visualization_calculation=False):
            """ Function: calculates the windows available to sail-in and -out of the port given the horizontal tidal restrictions according to the tidal window policy.

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                    - plot: provide a visualization of the calculation for each vessel
                    - axis: axes class from the matplotlib package
                    - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

            """

            def tidal_windows_tightest_current_restriction_along_route(vessel, route, axis=[], plot=False, sailing_time_correction=True, visualization_calculation=False):
                """ Function: calculates the normative current restrictions along the route over time and calculates the resulting horizontal tidal windows from these locations.

                    Input:
                        - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                        - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                        - plot: provide a visualization of the calculation for each vessel
                        - axis: axes class from the matplotlib package
                        - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

                """

                # Functions used for the calculation of the normative restriction along the route and the subsequent horizontal tidal windows over time
                def calculate_and_store_normative_current_velocity_data(node, data_nodes, sailing_time_to_next_node, t_ccur_raw, ccur, critcur_nodes, mccur_nodes, method, crit_ccur=[], crit_ccur_flood=0, crit_ccur_ebb=0):
                    """ Function: imports the current velocity data time series, corrects the time for the sailing time of the vessel, and subtracts the velocity data with the critical
                                  cross-current in order to find the times when the velocity exceeds the limit (by interpolation). The correct critical cross-current limit is picked by
                                  determining to what tidal period part of the time series belongs to (by determining what is the next tidal period). It appends the corrected data and
                                  the corresponding limits to separate lists.

                        Input:
                            - node: the name string of the node in the given network
                            - data_nodes: a list containing the two name string of the node in the given network the data is located. These locations depend on the direction of the vessel around the node of the restriction.
                            - sailing_time_to_next_node: time in s that the vessel needs to reach the node at which the restiction holds
                            - t_ccur_raw: a list of the raw, uncorrected timestamps of the time series for the current velocity
                            - ccur: a list to which the corrected current data will be appended
                            - critcur_nodes: a list to which the list of critical cross-currents corresponding to the maximum corrected current data will be appended to (for each node at which a horizontal tidal restriction is installed separately)
                            - mccur_nodes: a list to which the list of maximum corrected current data will be appended to (for each node at which a horizontal tidal restriction is installed separately)
                            - method: type of horizontal tidal restriction (method used)
                            - crit_ccur: a list to which the critical current velocity limit corresponding to the current data will be appended
                            - crit_ccur_flood: critical cross-current limit applied during flood in m/s
                            - crit_ccur_ebb: critical cross-current limit applied during ebb in m/s

                    """

                    # Setting some default bools
                    critical_cross_current_method_in_restrictions = False
                    point_based_method_in_restrictions = False
                    # Looping over the two nodes, one prior and one after the node at which the horizontal tidal restriction is installed, determining the direction of the vessel
                    for rep in range(2):
                        # Import current data and correct time series with sailing time, and import tidal periods
                        t_ccur = [t - sailing_time_to_next_node for t in network.nodes[route[node]]['Info']['Horizontal tidal restriction']['Data'][data_nodes[rep]][0]]
                        cur = network.nodes[route[node]]['Info']['Horizontal tidal restriction']['Data'][data_nodes[rep]][1]
                        times_tidal_periods = [z[0] - sailing_time_to_next_node for z in network.nodes[route[node]]['Info']['Tidal periods']]
                        tidal_periods = [z[1] for z in network.nodes[route[node]]['Info']['Tidal periods']]

                        # Different procedure for the methods:
                        if method == 'Critical cross-current':
                            critical_cross_current_method_in_restrictions = True
                            # Determine the next tidal period of the time series
                            t_ccur_tidal_periods = [tidal_periods[bisect.bisect_right(times_tidal_periods,y)] if y < times_tidal_periods[-1] else tidal_periods[-1] for y in t_ccur]
                            # Interpolate new time series
                            interp_ccur = sc.interpolate.CubicSpline(t_ccur, cur)
                            # If the next tidal period is ebb, then it means that the current period is flood. Hence, append interpolated current data subtracted with the critical cross-current, and the critical cross-current itself to the predefined lists
                            ccur.append([y - crit_ccur_flood if t_ccur_tidal_periods[x] == 'Ebb Start' else y - crit_ccur_ebb for x, y in enumerate(interp_ccur(t_ccur))])
                            crit_ccur.append([crit_ccur_flood if t_ccur_tidal_periods[x] == 'Ebb Start' else crit_ccur_ebb for x, y in enumerate(network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Data'][data_nodes[rep]][1])])

                        elif method == 'Point-based':
                            point_based_method_in_restrictions = True
                            # Just append the raw current data to the predefined list
                            ccur.append(network.nodes[route[node]]['Info']['Horizontal tidal restriction']['Data'][data_nodes[rep]][1])

                    # Take the maximum of both lists with the exceedance of the cross-current (positive values)
                    mccur = [max(idx) for idx in zip(*ccur)]

                    if method == 'Critical cross-current':
                        # Take the cross-currents of both lists which belong to the maximum cross-current limit exceeding velocities
                        critcur = [crit_ccur[np.argmax(val)][idx] for idx, val in enumerate(zip(*ccur))]
                        critcur_nodes.append(critcur)
                        mccur_nodes.append(mccur)

                    if method == 'Point-based':
                        # Interpolate new time series
                        interp_ccur = sc.interpolate.CubicSpline(t_ccur, mccur)
                        mccur = interp_ccur(t_ccur)
                        critcur = 10*np.ones(len(mccur))
                        critcur_nodes.append(critcur)
                        mccur_nodes.append(-10*np.ones(len(mccur)))

                    #Returns the determined tidal periods which may be used in a later stage of the calculation
                    return critical_cross_current_method_in_restrictions, point_based_method_in_restrictions, t_ccur, mccur, critcur, times_tidal_periods, tidal_periods

                def calculate_and_append_individual_horizontal_tidal_windows_critical_cross_current_method(new_t,max_ccur,max_crit_ccur):
                    """ Function: calculates and appends the horizontal tidal windows (available sail-in and -out times given the horizontal tidal restrictions) and corresponding critical cross-current
                                  velocities to the appropriate lists for the 'critical cross-curent'-method

                        Input:
                            - new_t: time of the time series for the current velocity corrected for the sailing time
                            - max_ccur: a list of the governing current velocities of the time series of the governing cross-currents in m/s
                            - max_crit_ccur: a list containing the critical-cross currents corrspeonding to the governing current velocities of the time series of the governing cross-currents in m/s

                    """

                    times_horizontal_tidal_window = []
                    crit_ccurs_horizontal_tidal_window = []

                    # Interpolate the data of the exceedance of the critical cross-current velocity
                    interp_max_cross_current = sc.interpolate.CubicSpline(new_t, [x for x in max_ccur])
                    # Loop over the roots (crossings where the cross-current velocity exceeds the cross-current velocity limit)
                    for root in interp_max_cross_current.roots():
                        # If the root falls within the time series and the next cross-current does not exceed the limit and the list of tidal restrictions is still empty or there are some restrictions identified but the previous restriction is not a stopping criteria
                        if root > new_t[0] and root < new_t[-1] and max_ccur[bisect.bisect_right(new_t, root)] < 0:
                            if times_horizontal_tidal_window == [] or (times_horizontal_tidal_window != [] and times_horizontal_tidal_window[-1][1] != 'Stop'):
                                # Append a stopping time of the restriction and the corresponding critical cross-current
                                times_horizontal_tidal_window.append([root, 'Stop'])
                                crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[bisect.bisect_right(new_t, root)])
                        # If the root falls within the time series and the next cross-current exceeds the limit as well and the list of tidal restrictions is still empty or there are some restrictions identified but the previous restriction is not a starting criteria
                        elif root > new_t[0] and root < new_t[-1] and max_ccur[bisect.bisect_right(new_t, root)] > 0:
                            if times_horizontal_tidal_window == [] or (times_horizontal_tidal_window != [] and times_horizontal_tidal_window[-1][1] != 'Start'):
                                # Append a starting time of the restriction and the corresponding critical cross-current
                                times_horizontal_tidal_window.append([root, 'Start'])
                                crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[bisect.bisect_right(new_t, root)])

                    # If the list of tidal windows are still empty: there is no horizontal tidal restriction for the particular vessel, so restriction stops at t_start and ends at t_end of the simulation
                    if times_horizontal_tidal_window == []:
                        times_horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                        crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[bisect.bisect_right(new_t, vessel.waiting_time_start)])
                        times_horizontal_tidal_window.append([np.max(new_t), 'Start'])
                        crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[-1])

                    # If the last restriction contains a starting time and is smaller than the end of the time series: add a stopping time restriction at the t_end of the simulation
                    if times_horizontal_tidal_window[-1][1] == 'Start' and times_horizontal_tidal_window[-1][0] < np.max(new_t):
                        times_horizontal_tidal_window.append([np.max(new_t), 'Stop'])
                        crit_ccurs_horizontal_tidal_window.append(max_crit_ccur[-1])

                    return times_horizontal_tidal_window, crit_ccurs_horizontal_tidal_window

                def finding_current_velocity_bounds_point_based_method(t_ccur, mccur, p_range, crit_ccur_tidal_period, times_tidal_periods, tidal_periods, sailing_time_to_next_node):
                    """ Function: defines the bounds around the point-based current velocity and finds the times at which the current velocity crosses these bounds by solving the roots of the interpolation.
                                  Furthermore, the tidal periods for each time the current velocity intersects with the bounds around the point-based current velocity.

                        Input:
                            - t_ccur: time of the time series for the current velocity corrected for the sailing time
                            - mccur: a list of times of the governing cross-currents in m/s
                            - p_range: the range in % around the point-based current velocity
                            - crit_ccur_tidal_period: critical cross-current limit applied during the tidal period in m/s
                            - times_tidal_periods: the starting times of the tidal periods
                            - tidal_periods: the definition of the tidal periods (Ebb or Flood Start)
                            - sailing_time_to_next_node: time in s that the vessel needs to reach the node at which the restiction holds

                    """

                    # Interpolate the interpolated cross-current velocity, corrected for the sailing time, subtracted with the critical cross-current for the upper and lower bound separately and calculate roots, and corresponding next tidal periods
                    crit_ccur_upper_bound = crit_ccur_tidal_period * (1 + p_range)
                    crit_ccur_lower_bound = crit_ccur_tidal_period * (1 - p_range)
                    interp_ccur_upper_bound = sc.interpolate.CubicSpline(t_ccur, [ccur - crit_ccur_upper_bound for ccur in mccur])
                    interp_ccur_lower_bound = sc.interpolate.CubicSpline(t_ccur, [ccur - crit_ccur_lower_bound for ccur in mccur])
                    roots_upper_bound = interp_ccur_upper_bound.roots()
                    roots_lower_bound = interp_ccur_lower_bound.roots()
                    t_ccur_tidal_periods_upper_bound = [tidal_periods[bisect.bisect_right(times_tidal_periods,y)] if y <= times_tidal_periods[-1] else tidal_periods[-1] for y in roots_upper_bound]
                    t_ccur_tidal_periods_lower_bound = [tidal_periods[bisect.bisect_right(times_tidal_periods,y)] if y <= times_tidal_periods[-1] else tidal_periods[-1] for y in roots_lower_bound]
                    return crit_ccur_upper_bound, crit_ccur_lower_bound, roots_upper_bound ,roots_lower_bound, t_ccur_tidal_periods_upper_bound, t_ccur_tidal_periods_lower_bound

                def calculate_and_append_individual_horizontal_tidal_windows_point_based_method(previous_time, next_time, next_tidal_period, sailing_time_to_next_node, crit_ccur_bound_tidal_period, roots_bounds_tidal_period, t_ccur_tidal_periods_bounds_period, crit_ccur_tidal_periods, p_tidal_periods, t_ccur, mccur, horizontal_tidal_window, crit_vel_horizontal_tidal_window):
                    """ Function: calculates and appends the horizontal tidal windows (available sail-in and -out times given the horizontal tidal restrictions) and corresponding critical cross-current
                                  velocities to the appropriate lists for the 'point-based'-method

                        Input:
                            - previous_time: stopping (starting) time of the previous (current) tidal window
                            - next_time:
                            - next_tidal_period: the string containing information about the next tidal period (Flood/Ebb Start)
                            - sailing_time_to_next_node: time in s that the vessel needs to reach the node at which the restiction holds
                            - crit_ccur_bound_tidal_period: a list containing the upper and lower bounds around the critical cross-currents in m/s respectively
                            - roots_bounds_tidal_period: a list containing the lists with the times that the current velocity intersects with the upper and lower bound around the point-based current velocity respectively
                            - t_ccur_tidal_periods_bounds_period: a list containing the lists with the tidal periods for each time the current velocity intersects with the upper and lower bound around the point-based current velocity respectively
                            - crit_ccur_tidal_periods: a list containing the critical cross-currents in m/s during flood and ebb respectively
                            - p_tidal_periods: a list containing the ranges around the critical cross-currents in [-] during flood and ebb respectively
                            - t_ccur: a list of times of the times series of the governing cross-currents in s
                            - mccur: a list of the current velocities of the time series of the governing cross-currents in m/s
                            - horizontal_tidal_window: a list to which the horizontal tidal windows should be appended
                            - crit_vel_horizontal_tidal_window: a list to which the critical cross-currents corresponding to the horizontal tidal windows should be appended

                    """

                    # Determine the time index for the boundary of the next tidal period and if index is smaller than 0, then set index to 0
                    next_previous_time_tidal_periods = times_tidal_periods.index(next_time)-1
                    if next_previous_time_tidal_periods < 0: next_previous_time_tidal_periods = 0
                    # Calculate the time index of the starting point and ending point of the current tidal period (corrected for the sailing time)
                    start_time = previous_time
                    stop_time = next_time
                    index_previous_time = bisect.bisect_right(t_ccur, start_time)-1
                    index_now = bisect.bisect_right(t_ccur, vessel.waiting_time_start)
                    index_next = bisect.bisect_right(t_ccur, stop_time)
                    # If the flood tide is freely accessible (critical cross-current limit is -1)
                    tide_index = 0
                    next_tide_index = 1
                    if next_tidal_period == 'Flood Start': tide_index = 1
                    next_tide_index = next_tide_index - tide_index
                    if crit_ccur_tidal_periods[tide_index] == -1:
                        # If the accessibility of the ebb tide is not limited to the slack tides: add stop of tidal restriction at the start of flood tide and append 0 m/s as critical cross-current to list
                        if crit_ccur_tidal_periods[next_tide_index] != 'min' and index_previous_time != 0:
                            horizontal_tidal_window.append([times_tidal_periods[next_previous_time_tidal_periods], 'Stop'])
                            crit_vel_horizontal_tidal_window.append(0)
                        # And add the start of the tidal restriction at the end of flood tide and append 0 m/s as critical cross-current to list
                        horizontal_tidal_window.append([next_time, 'Start'])
                        crit_vel_horizontal_tidal_window.append(0)

                    # Else if the accessibility during the ebb tide is limited to the slack waters and there is current data within this tidal period
                    elif crit_ccur_tidal_periods[tide_index] == 'min':
                        if mccur[index_previous_time:index_next] != []:
                            # Interpolation of the maximum current velocities during this specific tidal period subtracted with the critical point-based cross-current and calculate the times at which the cross-current equals this critical limit
                            interp = sc.interpolate.CubicSpline(t_ccur[index_previous_time:index_next],[y - p_tidal_periods[tide_index] for y in mccur[index_previous_time:index_next]])
                            times = [x for x in interp.roots() if (x >= previous_time and x <= next_time)]
                            # If there are no crossings (minimum number of crossings should be at least two)
                            if len(times) < 1: times = [t_ccur[index_previous_time], t_ccur[index_next]]
                            # Take the first and last value of the crossings, and append to tidal windows and critical cross-currents: restriction starts at time of first crossing and ends at time of last crossing
                            time = next(item for item in times if item is not None)
                            if time < t_ccur[np.argmax(mccur[index_previous_time:index_next]) + index_previous_time]:
                                horizontal_tidal_window.append([next(item for item in times if item is not None),'Start'])
                                crit_vel_horizontal_tidal_window.append(p_tidal_periods[tide_index])
                                horizontal_tidal_window.append([next(item for item in reversed(times) if item is not None),'Stop'])
                                crit_vel_horizontal_tidal_window.append(p_tidal_periods[tide_index])
                            else:
                                horizontal_tidal_window.append([next(item for item in times if item is not None),'Stop'])
                                crit_vel_horizontal_tidal_window.append(p_tidal_periods[tide_index])
                            if index_previous_time == 0: #next_previous_time_tidal_periods
                                if mccur[index_now] <= p_tidal_periods[tide_index]:
                                    horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                                    crit_vel_horizontal_tidal_window.append(0)
                                else:
                                    horizontal_tidal_window.append([vessel.waiting_time_start,'Start'])
                                    crit_vel_horizontal_tidal_window.append(0)

                    # Else (flood tide is restricted and accessilibity of ebb tide is not limited to the slack waters) if there is current data within this tidal period
                    elif mccur[index_previous_time:index_next] != []:
                        # If the maximum cross-current in the tidal period does not exceed the upper bound limit: append to list that restriction stops at the maximum cross-current, with the corresponding maximum cross-current
                        if np.max(mccur[index_previous_time:index_next]) < crit_ccur_bound_tidal_period[0]:
                            if p_tidal_periods[tide_index] != 0:
                                if crit_ccur_tidal_periods[next_tide_index] != 0:
                                    horizontal_tidal_window.append([t_ccur[np.argmax(mccur[index_previous_time:index_next]) + index_previous_time],'Stop'])
                                    crit_vel_horizontal_tidal_window.append(np.max(mccur[index_previous_time:index_next]))
                                else:
                                    horizontal_tidal_window.append([t_ccur[np.argmax(mccur[index_previous_time:index_next]) + index_previous_time],'Start'])
                                    crit_vel_horizontal_tidal_window.append(np.max(mccur[index_previous_time:index_next]))
                            else:
                                if index_previous_time == 0: #next_previous_time_tidal_periods
                                    if mccur[index_now] <= p_tidal_periods[tide_index]:
                                        horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                                        crit_vel_horizontal_tidal_window.append(0)
                                    else:
                                        horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                                        crit_vel_horizontal_tidal_window.append(0)
                                    #horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                                    #crit_vel_horizontal_tidal_window.append(0)
                                else:
                                    if mccur[index_now] <= p_tidal_periods[tide_index]:
                                        horizontal_tidal_window.append([t_ccur[index_previous_time], 'Stop'])
                                        crit_vel_horizontal_tidal_window.append(0)
                                    else:
                                        horizontal_tidal_window.append([t_ccur[index_previous_time], 'Start'])
                                        crit_vel_horizontal_tidal_window.append(0)
                                    #horizontal_tidal_window.append([t_ccur[index_previous_time], 'Stop'])
                                    #crit_vel_horizontal_tidal_window.append(0)

                        # Limit the analysis to the tidal period from this time to the end of the period
                        index_previous_time2 = np.argmax(mccur[index_previous_time:index_next]) + index_previous_time
                        # If the maximum cross-current in the remainder of the tidal period does not exceed the lower bound limit and flood is accessible (crit-current is not 0 m/s and the ebb accessibility is not limited to the slack waters (critical cross-current is not 'min'): append to list that restriction starts at the minimum cross-current, with the corresponding minimum cross-current
                        if np.min(mccur[index_previous_time2:index_next]) > crit_ccur_bound_tidal_period[1] and crit_ccur_tidal_periods[tide_index] != 0 and p_tidal_periods[tide_index] != 0 and crit_ccur_tidal_periods[1] != 'min':
                            horizontal_tidal_window.append([t_ccur[np.argmin(mccur[index_previous_time2:index_next]) + index_previous_time2],'Start'])
                            crit_vel_horizontal_tidal_window.append(np.min(mccur[index_previous_time2:index_next]))
                        # Else if the flood tide is not accessible and the accessibility of the ebb tide is not limited to the slack waters: only append the start time of the tidal window and the corresponding cross-current (= 0 m/s)
                        elif p_tidal_periods[tide_index] == 0 and crit_ccur_tidal_periods[next_tide_index] != 'min':
                            horizontal_tidal_window.append([next_time, 'Start'])
                            crit_vel_horizontal_tidal_window.append(0)
                            if index_previous_time == 0 and p_tidal_periods[next_tide_index] != 0 and crit_ccur_tidal_periods[next_tide_index] != 0: #next_previous_time_tidal_periods
                                if mccur[index_now] <= p_tidal_periods[next_tide_index]:
                                    horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                                    crit_vel_horizontal_tidal_window.append(0)
                                else:
                                    horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                                    crit_vel_horizontal_tidal_window.append(0)
                            if next_previous_time_tidal_periods >= len(times_tidal_periods)-2:
                                horizontal_tidal_window.append([next_time, 'Stop'])
                                crit_vel_horizontal_tidal_window.append(0)

                        # Else if the accessibility of the ebb tide is limited to the slack waters: only append the start time of the tidal window and the corresponding cross-current (= critical cross-current during ebb)
                        elif crit_ccur_tidal_periods[next_tide_index] == 'min':
                            # Interpolation of the maximum current velocities during this specific tidal period subtracted with the critical point-based cross-current for ebb tide and calculate the times at which the cross-current equals this critical limit
                            interp = sc.interpolate.CubicSpline(t_ccur[index_previous_time:index_next],[y - p_tidal_periods[next_tide_index] for y in mccur[index_previous_time:index_next]])
                            times = [x for x in interp.roots() if (x >= previous_time and x <= next_time)]
                            # If there are no crossings (minimum number of crossings should be at least two)
                            if len(times) < 1: times = [t_ccur[index_previous_time], t_ccur[index_next]]
                            # Take the first and last value of the crossings, and append to tidal windows and critical cross-currents: restriction starts at time of first crossing
                            time = next(item for item in times if item is not None)
                            if time < t_ccur[np.argmax(mccur[index_previous_time:index_next])+index_previous_time]:
                                horizontal_tidal_window.append([time,'Start'])
                                crit_vel_horizontal_tidal_window.append(p_tidal_periods[1])
                            if index_previous_time == 0: #next_previous_time_tidal_periods
                                if (mccur[index_previous_time] <= crit_ccur_bound_tidal_period[1] and p_tidal_periods[tide_index] != 0) or (mccur[index_previous_time] > crit_ccur_bound_tidal_period[1] and p_tidal_periods[tide_index] == 0):
                                    horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                                    crit_vel_horizontal_tidal_window.append(0)
                                else:
                                    horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                                    crit_vel_horizontal_tidal_window.append(0)

                        if crit_ccur_tidal_periods[tide_index] == 0 and index_previous_time == 0:
                            horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                            crit_vel_horizontal_tidal_window.append(0)

                        elif crit_ccur_tidal_periods[next_tide_index] == 0 and index_previous_time == 0:
                            if mccur[index_now] < crit_ccur_bound_tidal_period[1] and p_tidal_periods[tide_index] != 0:
                                horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                                crit_vel_horizontal_tidal_window.append(0)
                            elif mccur[index_now] > crit_ccur_bound_tidal_period[0] and p_tidal_periods[tide_index] != 0:
                                horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                                crit_vel_horizontal_tidal_window.append(0)
                            elif mccur[index_now] > crit_ccur_bound_tidal_period[0] and p_tidal_periods[tide_index] == 0:
                                horizontal_tidal_window.append([vessel.waiting_time_start, 'Start'])
                                crit_vel_horizontal_tidal_window.append(0)

                    # If none of the above applies and hence the regular cross-current velocity bounds (resulting from finding_current_velocity_bounds_point_based_method) apply:
                    if type(crit_ccur_tidal_periods[tide_index]) != str and crit_ccur_tidal_periods[tide_index] != -1 and crit_ccur_tidal_periods[tide_index] != 0:
                        for bound_index in [0,1]:
                            bool_restriction = 'Stop'
                            if bound_index == 1: bool_restriction = 'Start'
                            for root in enumerate(roots_bounds_tidal_period[bound_index]):
                                if root[1] >= start_time and root[1] <= stop_time:
                                    # calculate the time index of the root
                                    index = bisect.bisect_right(t_ccur, root[1])
                                    # if the index is greater than the length of the time series: index = -1
                                    if index >= len(t_ccur): index -= 1
                                    # if the critical cross-current at the index is lower than the upper bound of the critical cross-current and is located in the flood tidal period and the accessibility during flood tide is not limited to the slack waters: append root (corrected for the sailing time) and the corresponding cross-current to the stopping times of the tidal restriction
                                    if mccur[index] < crit_ccur_bound_tidal_period[bound_index] and t_ccur_tidal_periods_bounds_period[bound_index][root[0]] == next_tidal_period:
                                        if crit_ccur_tidal_periods[tide_index] != -1:
                                            horizontal_tidal_window.append([root[1], bool_restriction])
                                            crit_vel_horizontal_tidal_window.append(crit_ccur_bound_tidal_period[bound_index])

                    #Data appended to the lists, nothing has to be returned
                    return

                # Predefining some parameters
                network = vessel.env.FG
                distance_to_next_node = 0
                mccur_nodes = []
                t_ccur = network.nodes[route[0]]['Info']['Current velocity'][0]
                max_ccur = []
                list_of_nodes = []
                critcur_nodes = []
                times_horizontal_tidal_windows = []
                crit_ccurs_horizontal_tidal_windows = []
                list_of_list_indexes = []
                critical_cross_current_method_in_restrictions = False
                point_based_method_in_restrictions = False
                t_ccur_raw = network.nodes[route[0]]['Info']['Current velocity'][0]

                # Start of calculation by looping over the nodes of the route
                for nodes in enumerate(route):
                    # - if a correction for the sailing time should be applied: the total distance should be keep track of
                    if nodes[0] == 0:
                        if 'bound' not in dir(vessel): vessel.bound = 'inbound'  # to be removed later: should be part of the vessel generator in which the user should define the initial direction of the vessel
                    else:
                        vessel.bound = VesselTrafficService.provide_sailing_direction(vessel, [route[route.index(nodes[1])-1], nodes[1]])

                    if sailing_time_correction:
                        if not nodes[1] == route[0]:
                            distance_to_next_node += pyproj.Geod(ellps='WGS84').inv(network.nodes[route[nodes[0] - 1]]['geometry'].x,
                                                                                    network.nodes[route[nodes[0] - 1]]['geometry'].y,
                                                                                    network.nodes[route[nodes[0]]]['geometry'].x,
                                                                                    network.nodes[route[nodes[0]]]['geometry'].y)[2]

                    # Sailing time is calculated by the distance of the route up to the specific node, divided by the sailing speed (if no correction for the sailing time should be applied: distance is always 0)
                    sailing_time_to_next_node = distance_to_next_node / vessel.v

                    # If there is no horizontal tidal restriction at the specific node in the route: continue loop
                    if network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction'] == {}:
                        continue

                    # Else: importing some node specific data on the type and specifications of the horizontal tidal restriction and predefining some bools
                    types = network.nodes[nodes[1]]['Info']['Horizontal tidal restriction']['Type']
                    specifications = network.nodes[nodes[1]]['Info']['Horizontal tidal restriction']['Specification']

                    # Determine which restriction applies to vessel
                    restriction_index, no_tidal_window = tidal_window_restriction_determinator(vessel, route, types, specifications, nodes[1], sailing_time_to_next_node)
                    # If no horizontal tidal window applies to vessel at the specific node: continue loop over nodes of the route of the vessel
                    if no_tidal_window:
                        continue

                    # Else if there applies a horizontal tidal window: continue tidal window calculation by predefining some parameters and importing critical cross-currents
                    ccur = []
                    crit_ccur = []
                    times_horizontal_tidal_window = []
                    crit_vel_horizontal_tidal_window = []
                    crit_ccur_flood_old = types[1][restriction_index][0]
                    crit_ccur_ebb_old = types[1][restriction_index][1]

                    # Determination of the direction of the vessel and the nodes that the vessel passes by in order, in order to extract the correct data
                    restriction_on_route = False
                    # -if the calculation is requested to be performed for visualization purposes (no route, but for single node), then use the given direction of the specified vessel
                    if visualization_calculation:
                        restriction_on_route = True
                        if vessel.bound == 'inbound':
                            n1 = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][0]
                            n2 = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][1]
                        elif vessel.bound == 'outbound':
                            n2 = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][0]
                            n1 = network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][1]
                    # -else: select the nodes by the modelled direction of the vessel by nested looping over the nodes in the route
                    else:
                        for n1 in route[:nodes[0]]:
                            if n1 == network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][0]:
                                for n2 in route[nodes[0]:]:
                                    if n2 == network.nodes[route[nodes[0]]]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][1]:
                                        restriction_on_route = True
                                        break
                                break

                    #Determining critical cross-current limits based on critical cross-current limit input (potentially integers [-1,0] or strings) by subtracting it with vessel-specific cross-current (if applicable)
                    # if passing the node during flood period is not restricted to the slack water (critical cross-current = type<str> = 'min') or freely or never accessible, then import cricital cross-current for flood and subtract it with vessel-specific cross-current (if applicable)
                    if crit_ccur_flood_old != -1 and crit_ccur_flood_old != 0 and type(crit_ccur_flood_old) != str:
                        crit_ccur_flood = crit_ccur_flood_old - vessel.metadata['max_cross_current']
                    # else if flood is freely accessible: set critical cross-current for flood at the very high level of 10 m/s
                    elif crit_ccur_flood_old == -1:
                        crit_ccur_flood = -1
                    elif crit_ccur_flood_old == 0:
                        crit_ccur_flood = 0
                    # if passing the node during ebb period is not restricted to the slack water (critical cross-current = type<str> = 'min') or freely or never accessible, then import cricital cross-current for ebb and subtract it with vessel-specific cross-current (if applicable)
                    if crit_ccur_ebb_old != -1 and crit_ccur_ebb_old != 0 and type(crit_ccur_ebb_old) != str:
                        crit_ccur_ebb = crit_ccur_ebb_old - vessel.metadata['max_cross_current']
                    # else if flood is freely accessible: set critical cross-current for ebb at the very high level of 10 m/s
                    elif crit_ccur_ebb_old == -1:
                        crit_ccur_ebb = -1
                    elif crit_ccur_ebb_old == 0:
                        crit_ccur_ebb = 0

                    #Import and store data
                    if types[0][restriction_index] == 'Critical cross-current':
                        critical_cross_current_method_in_restrictions,_,t_ccur,mccur,critcur,_,_ = calculate_and_store_normative_current_velocity_data(nodes[0],[n1,n2],sailing_time_to_next_node,t_ccur_raw,ccur,critcur_nodes,mccur_nodes,'Critical cross-current',crit_ccur,crit_ccur_flood,crit_ccur_ebb)
                        # If a plot is requested, then plot the cross-current velocities and the corresponding critical cross-currents
                        if plot:
                            axis.plot(t_ccur, [y + critcur[x] for x, y in enumerate(mccur)], color='lightcoral', alpha=0.4)
                            axis.plot(t_ccur, [y if y != 10 else None for y in critcur], color='lightcoral', linestyle='--', alpha=0.4)

                    elif types[0][restriction_index] == 'Point-based':
                        _,point_based_method_in_restrictions,t_ccur,mccur,critcur,times_tidal_periods,tidal_periods = calculate_and_store_normative_current_velocity_data(nodes[0],[n1,n2],sailing_time_to_next_node,t_ccur_raw,ccur,critcur_nodes,mccur_nodes,'Point-based')
                        # Setting some default parameters and lists and importing some information about the range outside the point of entry of vessels
                        p_flood = types[2][restriction_index][0]
                        p_ebb = types[2][restriction_index][1]

                        # if passing the node during flood period is not restricted to the slack water (critical cross-current = type<str> = 'min')
                        if type(crit_ccur_flood_old) != str:
                            # Interpolate the interpolated cross-current velocity, corrected for the sailing time, subtracted with the critical cross-current for the upper and lower bound separately and calculate roots, and corresponding next tidal periods
                            [crit_ccur_upper_bound_flood,
                             crit_ccur_lower_bound_flood,
                             roots_upper_bound_flood,
                             roots_lower_bound_flood,
                             t_ccur_tidal_periods_upper_bound_flood,
                             t_ccur_tidal_periods_lower_bound_flood] = finding_current_velocity_bounds_point_based_method(t_ccur, mccur, p_flood, crit_ccur_flood, times_tidal_periods, tidal_periods, sailing_time_to_next_node)
                        else:
                            [crit_ccur_upper_bound_flood,
                             crit_ccur_lower_bound_flood,
                             roots_upper_bound_flood,
                             roots_lower_bound_flood,
                             t_ccur_tidal_periods_upper_bound_flood,
                             t_ccur_tidal_periods_lower_bound_flood] = [crit_ccur_flood_old,crit_ccur_flood_old,[],[],[],[]]
                            crit_ccur_flood = crit_ccur_upper_bound_flood
                        # if passing the node during ebb period is not restricted to the slack water (critical cross-current = type<str> = 'min')
                        if type(crit_ccur_ebb_old) != str:
                            # Interpolate the interpolated cross-current velocity, corrected for the sailing time, subtracted with the critical cross-current for the upper and lower bound separately and calculate roots, and corresponding next tidal periods
                            [crit_ccur_upper_bound_ebb,
                             crit_ccur_lower_bound_ebb,
                             roots_upper_bound_ebb,
                             roots_lower_bound_ebb,
                             t_ccur_tidal_periods_upper_bound_ebb,
                             t_ccur_tidal_periods_lower_bound_ebb] = finding_current_velocity_bounds_point_based_method(t_ccur, mccur, p_ebb, crit_ccur_ebb, times_tidal_periods, tidal_periods,sailing_time_to_next_node)
                        else:
                            [crit_ccur_upper_bound_ebb,
                             crit_ccur_lower_bound_ebb,
                             roots_upper_bound_ebb,
                             roots_lower_bound_ebb,
                             t_ccur_tidal_periods_upper_bound_ebb,
                             t_ccur_tidal_periods_lower_bound_ebb] = [crit_ccur_ebb_old,crit_ccur_ebb_old,[],[],[],[]]
                            crit_ccur_ebb = crit_ccur_upper_bound_ebb

                        #Setting a a default starting time for the tidal period and looping over each tidal period to determine whether there are special cases (special restrictions or currents within tidal periods that do not reach the lower and/or upper bound)
                        previous_time = t_ccur[0]
                        for period in enumerate(times_tidal_periods):
                            #If next tidal period is ebb, hence current tidal period is flood:
                            if tidal_periods[period[0]] == 'Ebb Start':
                                calculate_and_append_individual_horizontal_tidal_windows_point_based_method(previous_time,
                                                                                                            period[1],
                                                                                                            tidal_periods[period[0]],
                                                                                                            sailing_time_to_next_node,
                                                                                                            [crit_ccur_upper_bound_flood,crit_ccur_lower_bound_flood],
                                                                                                            [roots_upper_bound_flood,roots_lower_bound_flood],
                                                                                                            [t_ccur_tidal_periods_upper_bound_flood,t_ccur_tidal_periods_lower_bound_flood],
                                                                                                            [crit_ccur_flood,crit_ccur_ebb],
                                                                                                            [p_flood,p_ebb],
                                                                                                            t_ccur,
                                                                                                            mccur,
                                                                                                            times_horizontal_tidal_window,
                                                                                                            crit_vel_horizontal_tidal_window)

                            # Else if next tidal period is flood, hence current tidal period is ebb:
                            elif tidal_periods[period[0]] == 'Flood Start':
                                calculate_and_append_individual_horizontal_tidal_windows_point_based_method(previous_time,
                                                                                                            period[1],
                                                                                                            tidal_periods[period[0]],
                                                                                                            sailing_time_to_next_node,
                                                                                                            [crit_ccur_upper_bound_ebb,crit_ccur_lower_bound_ebb],
                                                                                                            [roots_upper_bound_ebb,roots_lower_bound_ebb],
                                                                                                            [t_ccur_tidal_periods_upper_bound_ebb,t_ccur_tidal_periods_lower_bound_ebb],
                                                                                                            [crit_ccur_flood,crit_ccur_ebb],
                                                                                                            [p_flood,p_ebb],
                                                                                                            t_ccur,
                                                                                                            mccur,
                                                                                                            times_horizontal_tidal_window,
                                                                                                            crit_vel_horizontal_tidal_window)

                            # Setting new starting time for the tidal period equalling the end time of the previous tidal period
                            previous_time = period[1]

                        # Combine the lists for the restrictions and corresponding critical cross-currents and sort on the starting and stopping times of the restrictions
                        zipped_lists = zip(times_horizontal_tidal_window, crit_vel_horizontal_tidal_window)
                        sorted_pairs = sorted(zipped_lists)
                        tuples = zip(*sorted_pairs)
                        times_horizontal_tidal_window, crit_vel_horizontal_tidal_window = [list(tuple) for tuple in tuples]

                        # Set a default list of indexes in the restrictions list that should be removed based on the following conditions by looping over all the times
                        indexes_to_be_removed = []
                        for time in range(len(times_horizontal_tidal_window)):
                            # Skip the first index
                            if time == 0:
                                continue
                            # If there are two stopping restrictions in sequence in the list, add the index of the previous stopping time of the restriction to the list of indexes to be removed
                            elif times_horizontal_tidal_window[time][1] == 'Stop' and times_horizontal_tidal_window[time - 1][1] == 'Stop':
                                indexes_to_be_removed.append(time - 1)
                            # If there are two starting restrictions in sequence in the list, add the index of the next starting time of the restriction to the list of indexes to be removed
                            elif times_horizontal_tidal_window[time][1] == 'Start' and times_horizontal_tidal_window[time - 1][1] == 'Start':
                                indexes_to_be_removed.append(time)

                        # Remove the indexes in the list of the restrictions and corresponding cross-currents
                        for remove_index in list(reversed(indexes_to_be_removed)):
                            times_horizontal_tidal_window.pop(remove_index)
                            crit_vel_horizontal_tidal_window.pop(remove_index)

                        # Add the horizontal tidal window to the list of all horizontal tidal windows along the route
                        times_horizontal_tidal_windows.extend(times_horizontal_tidal_window)
                        crit_ccurs_horizontal_tidal_windows.extend(crit_vel_horizontal_tidal_window)
                        list_of_list_indexes.extend(np.ones(len(times_horizontal_tidal_window)))

                        # If a plot is requested: plot the maximum cross-currents and the critical point-based cross-currents
                        if plot:
                            axis.plot(t_ccur, [y for x, y in enumerate(mccur)], color='rosybrown')
                            axis.plot([x[0] for x in times_horizontal_tidal_window], crit_vel_horizontal_tidal_window,color='sienna', marker='o', linestyle='None')

                # Pick the maximum cross-current and cross-current limits of all the tidal restrictions
                max_ccur = [max(idx) for idx in zip(*mccur_nodes)]
                max_critcur = [critcur_nodes[np.argmax(val)][idx] for idx, val in enumerate(zip(*mccur_nodes))]

                if critical_cross_current_method_in_restrictions:
                    [times_horizontal_tidal_window,
                     crit_ccurs_horizontal_tidal_window] = calculate_and_append_individual_horizontal_tidal_windows_critical_cross_current_method(t_ccur,max_ccur,max_critcur)
                    times_horizontal_tidal_windows.extend(times_horizontal_tidal_window)
                    crit_ccurs_horizontal_tidal_windows.extend(crit_ccurs_horizontal_tidal_window)
                    list_of_list_indexes.extend(np.zeros(len(times_horizontal_tidal_window)))

                # If a plot is requested and the list of maximum cross-current limits is not empty: plot these maximum cross-currents except for when the tidal period is fully accessible (cross-current limit was set to 10 m/s)
                if plot and max_critcur != []:
                    axis.plot(t_ccur, [y if y != 10 else None for y in max_critcur], color='lightcoral', alpha=0.4,linestyle='--')

                # Return the modified time series for the cross-currents, critical cross-currents and if applicable the calculated horizontal tidal windows
                return t_ccur, max_ccur, max_critcur, times_horizontal_tidal_windows, crit_ccurs_horizontal_tidal_windows, list_of_list_indexes

            #Continuation of the calculation by defining the network
            network = vessel.env.FG

            # Calculate the maximum current-velocity at each node of the route in order to set the y-lim of the plot properly
            if plot:
                max_cur = []
                for node in route:
                    max_cur.append(np.max(network.nodes[node]['Info']['Current velocity'][1]))

            # Set some default parameters
            times_horizontal_tidal_window = []
            crit_ccurs_horizontal_tidal_window = []
            list_of_list_indexes = []
            list_indexes = [0, 1]

            # Determine the data of the governing horizontal tidal restriction (critical cross-current method) and/or the governing horizontal tidal restriction (point-based method)
            [t_ccur, max_ccur, max_crit_ccur, times_horizontal_tidal_window, crit_ccurs_horizontal_tidal_window, list_of_list_indexes] = tidal_windows_tightest_current_restriction_along_route(vessel, route, axis, plot, sailing_time_correction, visualization_calculation)

            # If the list of tidal windows are still empty: there is no horizontal tidal restriction for the particular vessel, so restriction stops at t_start and ends at t_end of the simulation
            if times_horizontal_tidal_window == []:
                times_horizontal_tidal_window.append([vessel.waiting_time_start, 'Stop'])
                crit_ccurs_horizontal_tidal_window.append(0)
                list_of_list_indexes.append(0)  # integer = 0 meaning that the restriction is of type critical cross-current
                times_horizontal_tidal_window.append([np.max(t_ccur), 'Start'])
                crit_ccurs_horizontal_tidal_window.append(0)
                list_of_list_indexes.append(0)  # integer = 0 meaning that the restriction is of type critical cross-current

            # Combine the list of the horizontal tidal windows and the restriction types and critical cross-currents respectively and sort it chronologically
            list_of_list_indexes = [x for _, x in sorted(zip(times_horizontal_tidal_window, list_of_list_indexes))]
            crit_ccurs_horizontal_tidal_window = [x for _, x in sorted(zip(times_horizontal_tidal_window, crit_ccurs_horizontal_tidal_window))]
            times_horizontal_tidal_window.sort()

            # Set a default list of indexes in the restrictions list that should be removed based on the following conditions by nested looping over the restriction types and all the starting and stopping times of the restriction
            indexes_to_be_removed = []
            for list_index in list_indexes:
                for time1 in range(len(times_horizontal_tidal_window)):
                    # If the time is a starting time of a restriction and is the same as the selected restriction type: loop over the times again and find the next stopping time of the restriction with the same restriction type
                    if times_horizontal_tidal_window[time1][1] == 'Start' and list_of_list_indexes[time1] == list_index:
                        for time2 in range(len(times_horizontal_tidal_window)):
                            if time2 > time1 and times_horizontal_tidal_window[time2][1] == 'Stop' and list_of_list_indexes[time2] == list_index:
                                # select all the integers between the starting and stopping time of the restriction and add them to the indexes to be removed
                                indexes = np.arange(time1 + 1, time2, 1)
                                for index in indexes:
                                    indexes_to_be_removed.append(index)
                                break

            # Sort the indexes to be removed and remove the duplicates
            indexes_to_be_removed.sort()
            indexes_to_be_removed = list(dict.fromkeys(indexes_to_be_removed))

            # Remove the times of the restriction, the corresponding type of restriction and critical cross-current of the restriction by looping over the indexes to be removed
            for remove_index in list(reversed(indexes_to_be_removed)):
                times_horizontal_tidal_window.pop(remove_index)
                list_of_list_indexes.pop(remove_index)
                crit_ccurs_horizontal_tidal_window.pop(remove_index)

            # Set a new default list of indexes in the restrictions list that should be removed based on the following conditions by looping over all the times
            indexes_to_be_removed = []
            for time in range(len(times_horizontal_tidal_window)):
                # Skip the first index
                if time == 0:
                    continue
                # If there are two stopping restrictions in sequence in the list, add the index of the previous stopping time of the restriction to the list of indexes to be removed
                elif times_horizontal_tidal_window[time][1] == 'Stop' and times_horizontal_tidal_window[time - 1][1] == 'Stop':
                    indexes_to_be_removed.append(time - 1)
                # If there are two starting restrictions in sequence in the list, add the index of the next starting time of the restriction to the list of indexes to be removed
                elif times_horizontal_tidal_window[time][1] == 'Start' and times_horizontal_tidal_window[time - 1][1] == 'Start':
                    indexes_to_be_removed.append(time)

            # Again remove the times of the restriction, the corresponding type of restriction and critical cross-current of the restriction by looping over the indexes to be removed
            for remove_index in list(reversed(indexes_to_be_removed)):
                times_horizontal_tidal_window.pop(remove_index)
                list_of_list_indexes.pop(remove_index)
                crit_ccurs_horizontal_tidal_window.pop(remove_index)

            # If the first restriction in the list is a starting time and is greater than the starting time of the waiting time of the vessel: add a stopping condition at the starting time of the vessel and add the corresponding critical cross-current
            if times_horizontal_tidal_window[0][1] == 'Start' and times_horizontal_tidal_window[0][0] > vessel.waiting_time_start:
                times_horizontal_tidal_window.insert(0,[vessel.waiting_time_start, 'Stop'])
                crit_ccurs_horizontal_tidal_window.insert(0,max_crit_ccur[bisect.bisect_right(t_ccur, vessel.waiting_time_start)])
            # If the first restriction in the list is a stopping time and is greater than the starting time of the waiting time of the vessel: add a starting condition at the starting time of the vessel and add the corresponding critical cross-current
            elif times_horizontal_tidal_window[0][1] == 'Stop' and times_horizontal_tidal_window[0][0] > vessel.waiting_time_start:
                times_horizontal_tidal_window.insert(0,[vessel.waiting_time_start, 'Start'])
                crit_ccurs_horizontal_tidal_window.insert(0,max_crit_ccur[bisect.bisect_right(t_ccur, vessel.waiting_time_start)])

            # If a plot is requested: plot the governing maximum cross-current limits and cross-current with a corresponding label and add a lay-out
            if plot and len(max_crit_ccur) != 0:
                axis.plot(t_ccur,[y if y != 10 else None for y in max_crit_ccur], color='indianred', linestyle='--')
                axis.plot([t for t in t_ccur], [y + max_crit_ccur[i] for i,y in enumerate(max_ccur)],color='indianred')
                axis.set_ylabel('Cross-current velocity [m/s]', color='indianred')
                axis.set_ylim([0, 1.05*np.max(max_cur)])
                y_loc = [y if y != 10 else None for y in max_crit_ccur][bisect.bisect_right(t_ccur, vessel.env.now + 0.5*vessel.metadata['max_waiting_time'])]
                # If the location of the label for the critical cross-current is not visible in the plot (exceeds the y-lim): calculate a new location for the label when the critical cross-current is visible
                if True in [y < 1.05*np.max(max_cur) for y in max_crit_ccur[bisect.bisect_right(t_ccur, vessel.env.now + 0.5*vessel.metadata['max_waiting_time']):bisect.bisect_right(t_ccur, vessel.env.now + 1.5*vessel.metadata['max_waiting_time'])]] and y_loc == None:
                    y_loc = next(item for item in [y if y < 1.05*np.max(max_cur) else None for y in max_crit_ccur][bisect.bisect_right(t_ccur, vessel.env.now + vessel.metadata['max_waiting_time']):] if item is not None)
                if y_loc != None and y_loc < 1.05*np.max(max_cur): #if a new value is found: plot the critical cross-currents limits corresponding to the horizontal tidal windows
                    axis.plot([x[0] for x in times_horizontal_tidal_window],[y for y in crit_ccurs_horizontal_tidal_window], color='indianred', marker='o',linestyle='None')
                    axis.text(vessel.env.now + vessel.metadata['max_waiting_time'],y_loc, 'Critical cross-current', color='indianred',horizontalalignment='center')

            # Return the horizontal tidal windows
            return times_horizontal_tidal_window

        def times_tidal_window(vessel,route,axes=[[],[]],plot=False,sailing_time_correction=True,visualization_calculation=False,ukc_calc=False):
            """ Function: calculates the windows available to sail-in and -out of the port by combining the tidal windows of the horizontal and vertical tidal restrictions given the tidal window polciy

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                    - plot: provide a visualization of the calculation for each vessel
                    - axes: list of two axes classes from the matplotlib package
                    - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

            """

            # If just a calculation of the net_ukc is required:
            if ukc_calc:
                interp_wdep, ukc_s, ukc_p, fwa = times_vertical_tidal_window(vessel, route, ukc_calc=ukc_calc)
                return interp_wdep, ukc_s, ukc_p, fwa

            # Else: calculate the tidal window restriction for the vertical and horizontal tide respectively
            list_of_times_vertical_tidal_window = times_vertical_tidal_window(vessel,route,axes[0],plot,sailing_time_correction,ukc_calc)
            list_of_times_horizontal_tidal_window = times_horizontal_tidal_window(vessel,route,axes[1],plot,sailing_time_correction,visualization_calculation)

            # Set some default parameters
            list_indexes = [0,1]
            times_tidal_window = []
            list_of_list_indexes = []

            #Append data to the lists and append a corresponding indentification index (0=vertical, 1=horizontal)
            for time in list_of_times_vertical_tidal_window:
                times_tidal_window.append(time)
                list_of_list_indexes.append(0)
            for time in list_of_times_horizontal_tidal_window:
                times_tidal_window.append(time)
                list_of_list_indexes.append(1)

            #Sort lists on time
            list_of_list_indexes = [x for _, x in sorted(zip(times_tidal_window, list_of_list_indexes))]
            times_tidal_window.sort()

            #Remove values that fall within restrictive periods of both tidal policies.
            indexes_to_be_removed = []
            for list_index in list_indexes:
                for time1 in range(len(times_tidal_window)):
                    if times_tidal_window[time1][1] == 'Start' and list_of_list_indexes[time1] == list_index:
                        for time2 in range(len(times_tidal_window)):
                            if time2 > time1 and times_tidal_window[time2][1] == 'Stop' and list_of_list_indexes[time2] == list_index:
                                indexes = np.arange(time1 + 1, time2, 1)
                                for index in indexes:
                                    indexes_to_be_removed.append(index)
                                break

            #Sort the indexes that should be removed and remove duplicates
            indexes_to_be_removed.sort()
            indexes_to_be_removed = list(dict.fromkeys(indexes_to_be_removed))

            #Remove the values on the indexes that should be removed
            for remove_index in list(reversed(indexes_to_be_removed)):
                times_tidal_window.pop(remove_index)
                list_of_list_indexes.pop(remove_index)

            # Set a new default list of indexes in the restrictions list that should be removed based on the following conditions by looping over all the times
            indexes_to_be_removed = []
            for time in range(len(times_tidal_window)):
                if time == 0:
                    continue
                # If there are two stopping restrictions in sequence in the list, add the index of the previous stopping time of the restriction to the list of indexes to be removed
                elif times_tidal_window[time][1] == 'Stop' and times_tidal_window[time - 1][1] == 'Stop':
                    indexes_to_be_removed.append(time - 1)
                # If the first restriction in the list is a stopping time and is greater than the starting time of the waiting time of the vessel: add a starting condition at the starting time of the vessel and add the corresponding critical cross-current
                elif times_tidal_window[time][1] == 'Start' and times_tidal_window[time - 1][1] == 'Start':
                    indexes_to_be_removed.append(time)

            # Again remove the values on the indexes that should be removed
            for remove_index in list(reversed(indexes_to_be_removed)):
                times_tidal_window.pop(remove_index)
                list_of_list_indexes.pop(remove_index)

            # Return the final tidal windows
            return times_tidal_window

        # Continuation of the calculation of the available sail-in-times by setting the starting time and some lists
        if not ukc_calc:
            vessel.waiting_time_start = vessel.env.now
            axes = [[],[]]
        else:
            interp_wdep, ukc_s, ukc_p, fwa = times_tidal_window(vessel, route, ukc_calc=ukc_calc)
            return interp_wdep, ukc_s, ukc_p, fwa

        # If plot requested: create an empty figure
        if plot:
            fig, ax1 = plt.subplots(figsize=[10, 10])
            ax2 = ax1.twinx()
            axes = [ax1,ax2]

        # Running the above functions to determine the available-sail-in-times
        available_sail_in_times = times_tidal_window(vessel,route,axes,plot,sailing_time_correction,visualization_calculation,ukc_calc)

        # If plot requested: add the following to the plot
        if plot:
            # Predefining some lists
            line = []
            linelist = []

            # Appending the tidal windows to the list by looping over the resulting available sail-in-times
            for t2 in range(len(available_sail_in_times)):
                if available_sail_in_times[t2][1] == 'Start' and t2 != 0:
                    linelist.append([available_sail_in_times[t2][0], available_sail_in_times[t2 - 1][0]])

            # Visualization of the list of tidal windows
            for lines in linelist:
                line, = ax1.plot([lines[1], lines[0]],[0.05,0.05],color='darkslateblue', marker='o')

            # Lay-out of the plot
            plt.title('Tidal window calculation for a '+str(vessel.name) + '-class ' +str(vessel.type) + ', sailing ' + str(vessel.bound))
            plt.xlim([vessel.env.now-0.5*vessel.metadata['max_waiting_time'], vessel.env.now + 1.5*vessel.metadata['max_waiting_time']])
            plt.legend([line], ['Tidal window'])
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('Experienced water depth [m]', color='deepskyblue')
            ax2.spines['left'].set_color('deepskyblue')
            ax2.spines['right'].set_color('indianred')
            plt.show()

        return available_sail_in_times