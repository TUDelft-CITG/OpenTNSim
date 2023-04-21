# package(s) related to the simulation
import numpy as np
import bisect
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx
import time as timepy
import datetime

from shapely.ops import transform
# spatial libraries
import pyproj

class VesselTrafficService:
    """Class: a collection of functions that processes requests of vessels regarding the nautical processes on ow to enter the port safely"""

    def __init__(self,hydrodynamic_data,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.hydrodynamic_information = hydrodynamic_data

    def provide_sailing_time(self,vessel,route):
        distance_to_terminal = 0
        for u, v in zip(route[:-1], route[1:]):
            k = sorted(vessel.env.FG[u][v], key=lambda x: vessel.env.FG[u][v][x]['geometry'].length)[0]
            distance_to_terminal += vessel.env.FG.edges[u,v,k]['Info']['length']
        sailing_time = distance_to_terminal / vessel.v
        return sailing_time

    def provide_nearest_anchorage_area(self,vessel,node):
        nodes_of_anchorages = []
        capacity_of_anchorages = []
        users_of_anchorages = []
        sailing_times_to_anchorages = []
        # Loop over the nodes of the network and identify all the anchorage areas:
        for node_anchorage in vessel.env.FG.nodes:
            if 'Anchorage' in vessel.env.FG.nodes[node_anchorage]:
                # Determine if the anchorage area can be reached
                anchorage_reachable = True
                route_to_anchorage = nx.dijkstra_path(vessel.env.FG, node, node_anchorage)
                for node_on_route in route_to_anchorage:
                    station_index = list(self.hydrodynamic_information['Stations']).index(node_on_route)
                    min_water_level = np.min(self.hydrodynamic_information['Water level'][station_index].values)
                    MBL = self.hydrodynamic_information['MBL'][station_index].values
                    _, ukc_s, ukc_p, ukc_r, fwa = self.provide_sail_in_times_tidal_window(vessel,[node_on_route],ukc_calc=True)
                    if min_water_level + MBL < (vessel.T_f + ukc_s + ukc_p + ukc_r + fwa):
                        anchorage_reachable = False
                        break

                if not anchorage_reachable:
                    continue

                # Extract information over the individual anchorage areas: capacity, users, and the sailing distance to the anchorage area from the designated terminal the vessel is planning to call
                nodes_of_anchorages.append(node_anchorage)
                capacity_of_anchorages.append(vessel.env.FG.nodes[node_anchorage]['Anchorage'][0].resource.capacity)
                users_of_anchorages.append(len(vessel.env.FG.nodes[node_anchorage]['Anchorage'][0].resource.users))
                route_from_anchorage = nx.dijkstra_path(vessel.env.FG, node_anchorage, vessel.route[-1])
                sailing_time_to_anchorage = vessel.env.vessel_traffic_service.provide_sailing_time(vessel,route_from_anchorage)
                sailing_times_to_anchorages.append(sailing_time_to_anchorage)

        # Sort the lists based on the sailing distance to the anchorage area from the designated terminal the vessel is planning to call
        sorted_nodes_anchorages = [nodes for (distances, nodes) in sorted(zip(sailing_times_to_anchorages, nodes_of_anchorages))]
        sorted_users_of_anchorages = [nodes for (distances, nodes) in sorted(zip(sailing_times_to_anchorages, users_of_anchorages))]
        sorted_capacity_of_anchorages = [nodes for (distances, nodes) in sorted(zip(sailing_times_to_anchorages, capacity_of_anchorages))]

        # Take the anchorage area that is closest to the designated terminal the vessel is planning to call if there is sufficient capacity:
        node_anchorage = 0
        for anchorage_index,node_anchorage in enumerate(sorted_nodes_anchorages):
            if sorted_users_of_anchorages[anchorage_index] < sorted_capacity_of_anchorages[anchorage_index]:
                node_anchorage
                break

        return node_anchorage

    def provide_ukc_clearance(self,vessel,node):
        """ Function: calculates the sail-in-times for a specific vssel with certain properties and a pre-determined route and provides this information to the vessel

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node:
                - components_calc:

        """

        _, ukc_s, ukc_p, ukc_r, fwa = self.provide_sail_in_times_tidal_window(vessel,[node],ukc_calc=True)
        node_index = list(self.hydrodynamic_information['Stations'].values).index(node)
        time_index = bisect.bisect_left(self.hydrodynamic_information['Times'].values,vessel.env.now)
        wlev = self.hydrodynamic_information['Water level'].values[node_index][time_index]
        depth = self.hydrodynamic_information['MBL'].values[node_index]
        net_ukc = wlev + depth - (vessel.T_f + ukc_s + ukc_p + ukc_r + fwa + vessel.metadata['ukc'])
        return net_ukc

    def provide_governing_current_velocity(self,vessel,node,start_node,end_node,data=None,method=None):
        station_index = list(self.hydrodynamic_information['Stations'].values).index(node)
        times = self.hydrodynamic_information['Times'][start_node:end_node].values
        relative_layer_height = self.hydrodynamic_information['Relative layer height'].values
        current_velocity = self.hydrodynamic_information['Current velocity'][station_index].transpose().values[start_node:end_node]

        def depth_averaged_current_velocity(interpolation_depth,times,relative_layer_height,current_velocity,start_node,end_node,station_index):
            layer_boundaries = []
            average_current_velocity = []
            number_of_layers = len(relative_layer_height)
            water_depth = (self.hydrodynamic_information['MBL'][station_index] + self.hydrodynamic_information['Water level'][station_index][start_node:end_node])
            relative_water_depth = water_depth * self.hydrodynamic_information['Relative layer height']
            cumulative_water_depth = relative_water_depth.cumsum('LAYER').values

            for ti in range(len(times)):
                layer_boundaries.append(np.interp(interpolation_depth, cumulative_water_depth[ti], np.arange(0, number_of_layers, 1)))

            layer_boundary = np.floor(layer_boundaries)
            relative_boundary_layer_thickness = layer_boundaries - layer_boundary

            for ti in range(len(times)):
                if int(layer_boundary[ti]) + 2 < len(relative_layer_height):
                    rel_layer_heights = relative_layer_height[0:int(layer_boundary[ti]) + 2]
                    rel_layer_heights[-1] = rel_layer_heights[-1] * relative_boundary_layer_thickness[ti]
                    average_current_velocity.append(np.average(current_velocity[ti][0:int(layer_boundary[ti]) + 2], weights=rel_layer_heights))
                else:
                    average_current_velocity.append(np.average(current_velocity[ti], weights=relative_layer_height))

            return average_current_velocity

        if 'LAYER' in list(self.hydrodynamic_information['Current velocity'].dims):
            if vessel.T_f <= 5:
                current_velocity = depth_averaged_current_velocity(5,times,relative_layer_height,current_velocity,start_node,end_node,station_index)
            elif vessel.T_f <= 15:
                current_velocity = depth_averaged_current_velocity(vessel.T_f,times,relative_layer_height,current_velocity,start_node,end_node,station_index)
            else:
                current_velocity = [np.average(current_velocity[t], weights=relative_layer_height) for t in range(len(times))]

        else:
            current_velocity = current_velocity

        return current_velocity

    def provide_sail_in_times_tidal_window(self,vessel,route,delay=0,plot=False,ukc_calc=False):
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
                    if restriction_type[1].find('UKC') != -1: value = self.provide_ukc_clearance(vessel,node)
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

        def times_vertical_tidal_window(vessel,route,axis=[],plot=False,ukc_calc=False,delay=0):
            """ Function: calculates the windows available to sail-in and -out of the port given the vertical tidal restrictions according to the tidal window policy.

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                    - plot: provide a visualization of the calculation for each vessel
                    - axis: axes class from the matplotlib package
                    - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

            """

            # Function used in the determination of the calculation of the sail-in-times given the policy for determining the vertical tidal windows
            def minimum_available_water_depth_along_route(vessel, route, axis=[], plot=False, ukc_calc=False, delay=0):
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
                twdep_nodes = []
                wdep_nodes = []
                selected_route = []
                starting_node_vertical_tidal_window = None
                start_time_index = np.max([0,bisect.bisect_right(list(self.hydrodynamic_information['Times'].values),vessel.env.now + delay)-1])
                end_time_index = bisect.bisect_left(list(self.hydrodynamic_information['Times'].values),vessel.env.now + delay + vessel.sailing_time_to_terminal + vessel.metadata['max_waiting_time'])

                # Start of calculation by looping over the nodes of the route
                for node_index, node_name in enumerate(route):
                    node_index = list(self.hydrodynamic_information['Stations'].values).index(node_name)
                    selected_route.append(node_name)
                    sailing_time_to_next_node = vessel.env.vessel_traffic_service.provide_sailing_time(vessel,selected_route)

                    # Importing some node specific data on the time series of the water levels, depth, and ukc policy (corrected for the sailing time if correction was applied)
                    t_wlev = self.hydrodynamic_information['Times'].values[start_time_index:end_time_index]
                    t_step = t_wlev[1] - t_wlev[0]

                    time_correction_index = int(np.round(sailing_time_to_next_node / t_step))
                    wlev = self.hydrodynamic_information['Water level'][node_index].values[start_time_index:end_time_index]
                    depth = self.hydrodynamic_information['MBL'][node_index].values
                    wdep = [y + depth for y in wlev]

                    if not vessel.env.FG.nodes[node_name]['Info']['Vertical tidal restriction']:
                        if ukc_calc:
                            return wdep, 0, 0, 0
                        continue

                    types = vessel.env.FG.nodes[node_name]['Info']['Vertical tidal restriction']['Type']
                    specifications = vessel.env.FG.nodes[node_name]['Info']['Vertical tidal restriction']['Specification']

                    # Determine which restriction applies to vessel
                    restriction_index, _ = tidal_window_restriction_determinator(vessel, route, types, specifications, node_name, sailing_time_to_next_node)

                    # Calculate ukc policy based on the applied restriction
                    ukc_s = types[0][restriction_index]
                    ukc_p = types[1][restriction_index] * vessel.T_f
                    ukc_r = types[2][restriction_index][0] * (vessel.T_f-types[2][restriction_index][1])
                    fwa = types[3][restriction_index] * vessel.T_f

                    if ukc_calc:
                        return wdep, ukc_s, ukc_p, ukc_r, fwa

                    if not starting_node_vertical_tidal_window and np.min(wdep) < vessel.T_f+ukc_s+ukc_p+ukc_r+fwa:
                        starting_node_vertical_tidal_window = (node_index, node_name)

                    twdep_nodes.append([t - time_correction_index*t_step for t in t_wlev])
                    wdep_nodes.append([wd-(vessel.T_f+vessel.metadata['ukc']+ukc_s + ukc_p + ukc_r + fwa) for wd in wdep])

                    # Add to axes of the plot (if plot is requested): the available water depths at all the nodes of the route
                    if plot:
                        axis.plot(t_wlev[:-np.max([time_correction_index,1])], [wd-vessel.T_f for wd in np.array(wdep[np.max([time_correction_index,1]):])-(ukc_p+ukc_s+ukc_r+fwa-vessel.metadata['ukc'])], color='lightskyblue', alpha=0.4)

                # Pick the minimum of the water depths for each time and each node
                for i in range(len(wdep_nodes)):
                    index = twdep_nodes[i].index(t_wlev[0])
                    wdep_nodes[i] = wdep_nodes[i][index:]

                min_wdep = []
                for i in range(len(wdep_nodes[0])):
                    min_wdep.append(min(row[i] for row in wdep_nodes if i < len(row)))
                t_wlev = t_wlev[:len(min_wdep)]

                return t_wlev, min_wdep, starting_node_vertical_tidal_window, ukc_s, ukc_p, ukc_r, fwa

            #Continuation of the calculation of the windows given the vertical tidal restrictions by setting some parameters and calculation of the minimum available water depth along the route of the vessel
            if not ukc_calc:
                times_vertical_tidal_window = []
                new_t, min_wdep, starting_node_vertical_tidal_window, _, _, _, _ = minimum_available_water_depth_along_route(vessel,route,axis,plot,ukc_calc,delay)

            else:
                interp_wdep, ukc_s, ukc_p, ukc_r, fwa = minimum_available_water_depth_along_route(vessel, route, ukc_calc=True)
                return interp_wdep, ukc_s, ukc_p, ukc_r, fwa

            #If there is not enough available water depth over time: no tidal window
            if np.max(min_wdep) < 0:
                times_vertical_tidal_window.append([vessel.waiting_time_start+delay, 'Start']) #tidal restriction starts at t_start
                times_vertical_tidal_window.append([vessel.env.now+vessel.metadata['max_waiting_time']+delay, 'Stop']) #tidal restriction ends at t_end

            #Else if there is enough available water depth at certain moments: calculate tidal windows by looping over the roots of the interpolation
            else:
                zero_crossings = np.where(np.diff(np.sign(min_wdep)))[0]
                for root in [new_t[i] for i in zero_crossings]:
                    #-if root falls within time series of the data on the net ukc, and the net ukc a moment later than the root is higher than the required water depth:
                    if root > new_t[0] and root < new_t[-1] and min_wdep[bisect.bisect_right(new_t,root)] > 0:
                        # -if there are no values in the list of vertical tidal windows yet or there are values and the last value indicates that the tidal restriction has started and not ended and the net ukc a moment ealier than the root is less than the required water depth
                        if (times_vertical_tidal_window == [] or (times_vertical_tidal_window != [] and times_vertical_tidal_window[-1][1] != 'Stop')) and min_wdep[bisect.bisect_right(new_t,root)-1] < 0:
                            times_vertical_tidal_window.append([root, 'Stop']) #tidal restriction ends at t=root
                    # -if root falls within time series of the data on the net ukc, and the net ukc a moment later than the root is less than the required water depth:
                    elif root > new_t[0] and root < new_t[-1] and min_wdep[bisect.bisect_right(new_t,root)] < 0:
                        # -if there are no values in the list of vertical tidal windows yet or there are values and the last value indicates that the tidal restriction has ended and not ended and the net ukc a moment ealier than the root is higher than the required water depth
                        if (times_vertical_tidal_window == [] or (times_vertical_tidal_window != [] and times_vertical_tidal_window[-1][1] != 'Start')) and min_wdep[bisect.bisect_right(new_t,root)-1] > 0:
                            times_vertical_tidal_window.append([root, 'Start']) #tidal restriction starts at t=root

                #If the sail-in or -out-times given the vertical tidal restrictions are not empty: set the initial value at t=0
                if times_vertical_tidal_window != []:
                    #-if the first value in the list indicates that the tidal restriction starts: append that the tidal restriction ends at t=0
                    if times_vertical_tidal_window[0][1] == 'Start' and times_vertical_tidal_window[0][0] > vessel.waiting_time_start+delay:
                            times_vertical_tidal_window.insert(0,[vessel.waiting_time_start+delay, 'Stop'])
                    # -if the first value in the list indicates that the tidal restriction stop: append that the tidal restriction starts at t=0
                    if times_vertical_tidal_window[0][1] == 'Stop' and times_vertical_tidal_window[0][0] > vessel.waiting_time_start+delay:
                            times_vertical_tidal_window.insert(0,[vessel.waiting_time_start+delay, 'Start'])

            #If there are still no sail-in or -out-times (meaning the vessel can enter regardless the vertical tidal restriction): set start- and end-times
            if times_vertical_tidal_window == []:
                times_vertical_tidal_window.append([vessel.waiting_time_start+delay, 'Stop']) #tidal restriction stops at t_start
                times_vertical_tidal_window.append([vessel.env.now+vessel.metadata['max_waiting_time']+delay, 'Start']) #tidal restriction starts at t_end

            #If plot is requested: plot the minimum available water depth, the required water depth of the vessel, the resulting vertical tidal windows, and add some lay-out
            if plot:
                axis.plot(new_t,min_wdep,color='deepskyblue')
                axis.plot([x[0] for x in times_vertical_tidal_window], np.zeros(len(times_vertical_tidal_window)), color='deepskyblue',marker='o',linestyle='None')
                axis.set_ylim([-2,2])

            #Return the sail-in or -out-times given the vertical tidal restrictions over the route of the vessel
            return times_vertical_tidal_window, starting_node_vertical_tidal_window

        def times_horizontal_tidal_window(vessel,route, axis=[],plot=plot):
            """ Function: calculates the windows available to sail-in and -out of the port given the horizontal tidal restrictions according to the tidal window policy.

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                    - plot: provide a visualization of the calculation for each vessel
                    - axis: axes class from the matplotlib package
                    - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

            """

            def tidal_windows_tightest_current_restriction_along_route(vessel, route, axis=[], plot=False):
                """ Function: calculates the normative current restrictions along the route over time and calculates the resulting horizontal tidal windows from these locations.

                    Input:
                        - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                        - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
                        - plot: provide a visualization of the calculation for each vessel
                        - axis: axes class from the matplotlib package
                        - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

                """

                # Functions used for the calculation of the normative restriction along the route and the subsequent horizontal tidal windows over time
                def calculate_and_store_normative_current_velocity_data(vessel, node_horizontal_tidal_window, data_nodes, t_ccur, ccur, method, sailing_time_to_next_node,crit_ccur=[], crit_ccur_flood=0, crit_ccur_ebb=0):
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

                    start_time_index = np.max([0,bisect.bisect_right(list(self.hydrodynamic_information['Times'].values),vessel.env.now + delay)-1])
                    end_time_index = bisect.bisect_left(list(self.hydrodynamic_information['Times'].values),vessel.env.now + delay + sailing_time_to_next_node + vessel.metadata['max_waiting_time'])
                    t_step = t_ccur[1] - t_ccur[0]
                    time_correction_index = int(np.round(sailing_time_to_next_node / t_step))

                    tidal_periods = []

                    # Looping over the two nodes, one prior and one after the node at which the horizontal tidal restriction is installed, determining the direction of the vessel
                    for node_number in range(len(data_nodes)):
                        # Import current data and correct time series with sailing time, and import tidal periods
                        node_index = list(self.hydrodynamic_information['Stations'].values).index(node_horizontal_tidal_window)
                        cur = self.provide_governing_current_velocity(vessel,node_horizontal_tidal_window,start_time_index,end_time_index,data=network.nodes[node_horizontal_tidal_window]['Info']['Horizontal tidal restriction']['Data'][data_nodes[node_number]])
                        tidal_periods = [z[1] for z in list(self.hydrodynamic_information['Observed horizontal tidal periods'][node_index].values) if z[0] != 'nan']
                        times_tidal_periods = [list(self.hydrodynamic_information['Times'].values)[int(z[0])] - sailing_time_to_next_node for z in list(self.hydrodynamic_information['Observed horizontal tidal periods'][node_index].values) if z[0] != 'nan' and int(z[0]) < len(list(self.hydrodynamic_information['Times'].values))]
                        # Different procedure for the methods:
                        if method == 'Critical cross-current':
                            # If the next tidal period is ebb, then it means that the current period is flood. Hence, append interpolated current data subtracted with the critical cross-current, and the critical cross-current itself to the predefined lists
                            t_ccur_tidal_periods = [tidal_periods[bisect.bisect_left(times_tidal_periods, t)] if t < times_tidal_periods[-1] else tidal_periods[-1] for t in t_ccur[start_time_index:end_time_index]]
                            ccur.append([vel - crit_ccur_flood if t_ccur_tidal_periods[i] == 'Ebb Start' else vel - crit_ccur_ebb for i, vel in enumerate(cur)])
                            crit_ccur.append([crit_ccur_flood if t_ccur_tidal_periods[i] == 'Ebb Start' else crit_ccur_ebb for i, _ in enumerate(cur)])

                        elif method == 'Point-based':
                            # Just append the raw current data to the predefined list
                            ccur.append(cur)

                    # Take the maximum of both lists with the exceedance of the cross-current (positive values)
                    mccur = [max(idx) for idx in zip(*ccur)]
                    tccur = [t-time_correction_index*t_step for t in t_ccur[start_time_index:end_time_index]]

                    if method == 'Point-based':
                        # Interpolate new time series
                        interp_ccur = sc.interpolate.CubicSpline(t_ccur, mccur)
                        mccur = interp_ccur(t_ccur)

                    #Returns the determined tidal periods which may be used in a later stage of the calculation
                    return tccur, mccur, tidal_periods, start_time_index, end_time_index

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
                    zero_crossings = np.where(np.diff(np.sign(max_ccur)))[0]
                    # Loop over the roots (crossings where the cross-current velocity exceeds the cross-current velocity limit)
                    for root in [new_t[i] for i in zero_crossings]:
                        # If the root falls within the time series and the next cross-current does not exceed the limit and the list of tidal restrictions is still empty or there are some restrictions identified but the previous restriction is not a stopping criteria
                        index = new_t.index(root)
                        if max_ccur[index+1] < 0:
                            if times_horizontal_tidal_window == [] or (times_horizontal_tidal_window != [] and times_horizontal_tidal_window[-1][1] != 'Stop'):
                                # Append a stopping time of the restriction and the corresponding critical cross-current
                                times_horizontal_tidal_window.append([root, 'Stop'])
                                crit_ccurs_horizontal_tidal_window.append(0)

                        # If the root falls within the time series and the next cross-current exceeds the limit as well and the list of tidal restrictions is still empty or there are some restrictions identified but the previous restriction is not a starting criteria
                        elif max_ccur[index+1] > 0:
                            if times_horizontal_tidal_window == [] or (times_horizontal_tidal_window != [] and times_horizontal_tidal_window[-1][1] != 'Start'):
                                # Append a starting time of the restriction and the corresponding critical cross-current
                                times_horizontal_tidal_window.append([root, 'Start'])
                                crit_ccurs_horizontal_tidal_window.append(0)

                    # If the list of tidal windows are still empty: there is no horizontal tidal restriction for the particular vessel, so restriction stops at t_start and ends at t_end of the simulation
                    if times_horizontal_tidal_window == []:
                        times_horizontal_tidal_window.append([vessel.waiting_time_start+delay, 'Stop'])
                        crit_ccurs_horizontal_tidal_window.append(0)
                        times_horizontal_tidal_window.append([vessel.env.now + vessel.metadata['max_waiting_time'] + delay, 'Start'])
                        crit_ccurs_horizontal_tidal_window.append(0)

                    # If the last restriction contains a starting time: add a stopping time restriction at the t_end of the simulation
                    if times_horizontal_tidal_window[-1][1] == 'Start':
                        times_horizontal_tidal_window.append([vessel.env.now + vessel.metadata['max_waiting_time'] + delay, 'Stop'])
                        crit_ccurs_horizontal_tidal_window.append(0)

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
                    t_ccur_tidal_periods_upper_bound = [tidal_periods[bisect.bisect_left(times_tidal_periods,y)] if y <= times_tidal_periods[-1] else tidal_periods[-1] for y in roots_upper_bound]
                    t_ccur_tidal_periods_lower_bound = [tidal_periods[bisect.bisect_left(times_tidal_periods,y)] if y <= times_tidal_periods[-1] else tidal_periods[-1] for y in roots_lower_bound]
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
                    index_now = bisect.bisect_right(t_ccur, vessel.waiting_time_start)-1
                    if index_now == len(t_ccur):
                        index_now = index_now-2
                    index_next = bisect.bisect_left(t_ccur, stop_time)
                    if index_next == len(t_ccur):
                        index_next = index_next-1
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
                                    index = bisect.bisect_left(t_ccur, root[1])
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
                mccur_nodes = []
                tccur_nodes = []
                t_ccur = list(self.hydrodynamic_information['Times'].values)
                critcur_nodes = []
                times_horizontal_tidal_windows = []
                crit_ccurs_horizontal_tidal_windows = []
                list_of_list_indexes = []
                critical_cross_current_method_in_restrictions = False
                t_ccur_raw = list(self.hydrodynamic_information['Times'].values)
                starting_node_horizontal_tidal_window = None
                sailing_times_critcur_nodes = []

                # Start of calculation by looping over the nodes of the route
                selected_route = []
                sailing_time_to_next_node = 0

                for node_index,node_name in enumerate(route):
                    selected_route.append(node_name)
                    sailing_time_to_next_node = vessel.env.vessel_traffic_service.provide_sailing_time(vessel,selected_route)

                    # If there is no horizontal tidal restriction at the specific node in the route: continue loop
                    if 'Horizontal tidal restriction' not in network.nodes[route[node_index]]['Info'].keys():
                        continue

                    index_restriction = node_index
                    node_restriction = node_name

                    # Else: importing some node specific data on the type and specifications of the horizontal tidal restriction and predefining some bools
                    types = network.nodes[node_restriction]['Info']['Horizontal tidal restriction']['Type']
                    specifications = network.nodes[node_restriction]['Info']['Horizontal tidal restriction']['Specification']

                    # Determine which restriction applies to vessel
                    restriction_index, no_tidal_window = tidal_window_restriction_determinator(vessel, route, types, specifications, node_restriction, sailing_time_to_next_node)
                    # If no horizontal tidal window applies to vessel at the specific node: continue loop over nodes of the route of the vessel
                    if no_tidal_window:
                        continue

                    elif not starting_node_horizontal_tidal_window:
                        starting_node_horizontal_tidal_window = (node_index,node_name)

                    # Else if there applies a horizontal tidal window: continue tidal window calculation by predefining some parameters and importing critical cross-currents
                    ccur = []
                    crit_ccur = []
                    times_horizontal_tidal_window = []
                    crit_vel_horizontal_tidal_window = []
                    crit_ccur_flood_old = types[1][restriction_index][0]
                    crit_ccur_ebb_old = types[1][restriction_index][1]

                    # Determination of the direction of the vessel and the nodes that the vessel passes by in order, in order to extract the correct data
                    for approaching_node in route[:index_restriction]:
                        if approaching_node != network.nodes[node_restriction]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][0]:
                            continue

                        for departing_node in route[index_restriction:]:
                            if departing_node != network.nodes[node_restriction]['Info']['Horizontal tidal restriction']['Specification'][5][restriction_index][1]:
                                continue

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
                        critical_cross_current_method_in_restrictions = True
                        tccur,mccur,_,start_time_index,end_time_index = calculate_and_store_normative_current_velocity_data(vessel, starting_node_horizontal_tidal_window[1],[approaching_node,departing_node],t_ccur,ccur,'Critical cross-current',sailing_time_to_next_node,crit_ccur,crit_ccur_flood,crit_ccur_ebb)
                        sailing_times_critcur_nodes.append([sailing_time_to_next_node for _ in range(len(mccur))])
                        t_step = list(self.hydrodynamic_information['Times'].values)[1] - list(self.hydrodynamic_information['Times'].values)[0]
                        time_correction_index = np.max([1,int(np.round(sailing_time_to_next_node / t_step))])
                        corrected_time = list(self.hydrodynamic_information['Times'].values)[start_time_index:end_time_index]
                        corrected_mccur = mccur[time_correction_index:]
                        # If a plot is requested, then plot the cross-current velocities and the corresponding critical cross-currents
                        if plot:
                            axis.plot(corrected_time[:-time_correction_index], corrected_mccur, color='lightcoral', alpha=0.4)
                        mccur_nodes.append(mccur)
                        tccur_nodes.append(tccur)

                    elif types[0][restriction_index] == 'Point-based':
                        tccur,mccur,times_tidal_periods,tidal_periods,start_time_index,end_time_index = calculate_and_store_normative_current_velocity_data(vessel, starting_node_horizontal_tidal_window[1],[approaching_node,departing_node],sailing_time_to_next_node,t_ccur_raw,ccur,critcur_nodes,mccur_nodes,'Point-based',sailing_time_to_next_node)
                        mccur_nodes.append(mccur)
                        tccur_nodes.append(tccur)
                        sailing_times_critcur_nodes.append([sailing_time_to_next_node for _ in range(len(critcur))])
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
                            # If there are two stopping restrictions in sequence in the list, add the index of the previous stopping time of the restriction to the list of indeMinDistanceGeneralizerxes to be removed
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
                            axis.plot(corrected_time[start_time_index:end_time_index], mccur, color='rosybrown')
                            axis.plot([x[0] for x in times_horizontal_tidal_window], crit_vel_horizontal_tidal_window,color='sienna', marker='o', linestyle='None')

                # Pick the maximum cross-current and cross-current limits of all the tidal restrictions
                max_ccur = np.zeros(len(t_ccur))
                t_max_ccur = t_ccur
                sailing_times_critcur = 0
                if mccur_nodes:
                    max_ccur = []
                    for i in range(len(mccur_nodes[0])):
                        max_ccur.append(min(row[i] for row in mccur_nodes if i < len(row)))
                    t_max_ccur = tccur_nodes[0]
                    sailing_times_critcur = [sailing_times_critcur_nodes[np.argmax(val)][idx] for idx, val in enumerate(zip(*mccur_nodes))]

                if critical_cross_current_method_in_restrictions:
                    [times_horizontal_tidal_window,
                     crit_ccurs_horizontal_tidal_window] = calculate_and_append_individual_horizontal_tidal_windows_critical_cross_current_method(t_max_ccur,max_ccur,[crit_ccur_flood,crit_ccur_ebb])
                    times_horizontal_tidal_windows.extend(times_horizontal_tidal_window)
                    crit_ccurs_horizontal_tidal_windows.extend(crit_ccurs_horizontal_tidal_window)
                    list_of_list_indexes.extend(np.zeros(len(times_horizontal_tidal_window)))

                start_time_index = np.max([0,bisect.bisect_right(list(self.hydrodynamic_information['Times'].values),vessel.env.now + delay)-1])
                end_time_index = bisect.bisect_left(list(self.hydrodynamic_information['Times'].values),vessel.env.now + delay + np.max(sailing_times_critcur) + vessel.metadata['max_waiting_time'])

                # Return the modified time series for the cross-currents, critical cross-currents and if applicable the calculated horizontal tidal windows
                return t_max_ccur, max_ccur, times_horizontal_tidal_windows, crit_ccurs_horizontal_tidal_windows, list_of_list_indexes, starting_node_horizontal_tidal_window, start_time_index, end_time_index

            # Set some default parameters
            list_indexes = [0, 1]

            # Determine the data of the governing horizontal tidal restriction (critical cross-current method) and/or the governing horizontal tidal restriction (point-based method)
            [t_max_ccur, max_ccur, times_horizontal_tidal_window, crit_ccurs_horizontal_tidal_window, list_of_list_indexes, starting_node_horizontal_tidal_window, start_time_index, end_time_index] = tidal_windows_tightest_current_restriction_along_route(vessel, route, axis, plot)

            # If the list of tidal windows are still empty: there is no horizontal tidal restriction for the particular vessel, so restriction stops at t_start and ends at t_end of the simulation
            if times_horizontal_tidal_window == []:
                times_horizontal_tidal_window.append([vessel.waiting_time_start+delay, 'Stop'])
                crit_ccurs_horizontal_tidal_window.append(0)
                list_of_list_indexes.append(0)  # integer = 0 meaning that the restriction is of type critical cross-current
                times_horizontal_tidal_window.append([np.max(t_max_ccur), 'Start'])
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
            if times_horizontal_tidal_window[0][1] == 'Start' and times_horizontal_tidal_window[0][0] > vessel.waiting_time_start+delay:
                times_horizontal_tidal_window.insert(0,[vessel.waiting_time_start+delay, 'Stop'])
                crit_ccurs_horizontal_tidal_window.insert(0,0)
            # If the first restriction in the list is a stopping time and is greater than the starting time of the waiting time of the vessel: add a starting condition at the starting time of the vessel and add the corresponding critical cross-current
            elif times_horizontal_tidal_window[0][1] == 'Stop' and times_horizontal_tidal_window[0][0] > vessel.waiting_time_start+delay:
                times_horizontal_tidal_window.insert(0,[vessel.waiting_time_start+delay, 'Start'])
                crit_ccurs_horizontal_tidal_window.insert(0,0)

            # If a plot is requested: plot the governing maximum cross-current limits and cross-current with a corresponding label and add a lay-out
            if plot:
                axis.plot(t_max_ccur, max_ccur,color='indianred')
                axis.plot([x[0] for x in times_horizontal_tidal_window],[y for y in crit_ccurs_horizontal_tidal_window], color='indianred', marker='o',linestyle='None')
                axis.set_ylim([-2, 2])
                axis.set_ylabel('Exceedance of critical current velocity [m/s]', color='indianred')

            # Return the horizontal tidal windows
            return times_horizontal_tidal_window, starting_node_horizontal_tidal_window, start_time_index, end_time_index

        def times_tidal_window(vessel,route,axes=[[],[]],plot=False,ukc_calc=False,delay=0):
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
                interp_wdep, ukc_s, ukc_p, ukc_r, fwa = times_vertical_tidal_window(vessel, route, ukc_calc=ukc_calc)
                return interp_wdep, ukc_s, ukc_p, ukc_r, fwa

            # Else: calculate the tidal window restriction for the vertical and horizontal tide respectively
            list_of_times_vertical_tidal_window, starting_node_vertical_tidal_window = times_vertical_tidal_window(vessel,route,axes[0],plot,ukc_calc,delay)
            list_of_times_horizontal_tidal_window, starting_node_horizontal_tidal_window, start_time_index, end_time_index  = times_horizontal_tidal_window(vessel,route,axes[1],plot)

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

            starting_nodes_tidal_window = [starting_node_vertical_tidal_window,starting_node_horizontal_tidal_window]
            starting_node_indexes_tidal_window = [i[0] for i in starting_nodes_tidal_window if i]
            if starting_node_indexes_tidal_window != []:
                starting_node_index_tidal_window = np.min(starting_node_indexes_tidal_window)
                starting_node_tidal_window = [i[1] for i in starting_nodes_tidal_window if i and i[0] == np.min(starting_node_index_tidal_window)][0]
            else:
                starting_node_tidal_window = None

            # Return the final tidal windows
            return times_tidal_window, starting_node_tidal_window, start_time_index, end_time_index

        # Continuation of the calculation of the available sail-in-times by setting the starting time and some lists
        if not ukc_calc:
            vessel.waiting_time_start = vessel.env.now
            axes = [[],[]]
        else:
            interp_wdep, ukc_s, ukc_p, ukc_r, fwa = times_tidal_window(vessel, route, ukc_calc=ukc_calc)
            return interp_wdep, ukc_s, ukc_p, ukc_r, fwa

        # If plot requested: create an empty figure
        if plot:
            fig, ax1 = plt.subplots(figsize=[10, 10])
            ax2 = ax1.twinx()
            axes = [ax1,ax2]

        # Running the above functions to determine the available-sail-in-times
        available_sail_in_times, starting_node_tidal_window, start_time_index, end_time_index = times_tidal_window(vessel,route, axes, plot, ukc_calc, delay)

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
            ax2.axhline(0, color='k', linewidth=0.5)
            for lines in linelist:
                line, = ax2.plot([lines[1], lines[0]],[0,0],color='darkslateblue', marker='o',linewidth=2.5)

            # Lay-out of the plot
            plt.title('Tidal window calculation for a '+str(vessel.name) + '-class ' +str(vessel.type) + ',\n sailing ' + str(vessel.bound) + ' from ' + route[0] + ' to ' + route[-1])
            plt.legend([line], ['Tidal window'])
            xlim = [vessel.env.now+delay,vessel.env.now+delay+ vessel.metadata['max_waiting_time']]
            plt.xlim(xlim)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Net UKC [m]', color='deepskyblue')

            ax2.spines['left'].set_color('deepskyblue')
            ax2.spines['right'].set_color('indianred')
            xticks = np.arange(np.ceil((xlim[0]-3600) / 86400) * 86400, xlim[1], 86400)
            ax2.set_xticks(xticks)
            xtickslabels = [datetime.datetime.fromtimestamp(xtick).date() for xtick in xticks]
            ax2.set_xticklabels(xtickslabels)
            plt.show()

        return available_sail_in_times, starting_node_tidal_window