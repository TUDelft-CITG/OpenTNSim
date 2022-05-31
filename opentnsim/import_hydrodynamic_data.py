# package(s) related to the simulation
import numpy as np
import math
import bisect
import scipy as sc
import scipy.signal
import pandas as pd
import hatyan
import sys
import os

from datetime import timedelta

# spatial libraries
import pyproj

# additional packages
import datetime

class NetworkProperties:
    """Class: a collection of functions that append properties to the network"""

    def __init__(self,
                 *args,
                 **kwargs
    ):
        super().__init__(*args, **kwargs)

    def append_data_to_nodes(network,data):
        """ Function: appends geometric and hydrodynamic data to the nodes

            Input:
                - network: a graph constructed with the DiGraph class of the networkx package
                - data: xarray-dataset, containting:
                    - W: a list with [[],[]]-format containing the geometry of the individual nodes imported as points constructed with
                         the geometry class of the shapely package in the first column and the actual waterway widths in [m] as floats
                         at the nodes in the second column. The indexes should be matching, containing information about the same node.
                    - D: a list with [[],[]]-format containing the geometry of the nodes imported as points constructed with
                         the geometry class of the shapely package in the first column and the actual waterway depths in [m] as floats
                         at the nodes in the second column. The indexes should be matching, containing information about the same node.
                    - MBL: a list with [[],[]]-format containing the geometry of the individual nodes imported as points constructed with
                           the geometry class of the shapely package in the first column and the actual maintained water depths of
                           the waterays in [m] as floats in the second column. The indexes should be matching, containing information about
                           the same node.
                    - eta: a list with [[],[]]-format containing the geometry of the individual nodes imported as points constructed with
                           the geometry class of the shapely package in the first column and a list with [[],[]]-format of time
                           series of the actual water levels in [m] in the second column. The indexes should be matching, containing information
                           about the same node. The list in the second column should consist of the timestamps of the data in the first column
                           and the values of the property as floats in the second column. The time series should be in chronological order.
                    - vmag: a list with [[],[]]-format containing the geometry of the individual nodes imported as points constructed with
                            the geometry class of the shapely package in the first column and a list with [[],[]]-format of time
                            series of the actual current velocity magnitudes in [m/s] in the second column. The indexes should be matching,
                            containing information about the same node. The list in the second column should consist of the timestamps of the
                            data in the first column and the values of the property as floats in the second column. The time series should be
                            in chronological order.
                    - vdir: a list with [[],[]]-format containing the geometry of the individual nodes imported as points constructed with
                            the geometry class of the shapely package in the first column and a list with [[],[]]-format of time
                            series of the actual current directions in [rad] in the second column. The indexes should be matching, containing
                            information about the same node. The list in the second column should consist of the timestamps of the data in the
                            first column and the values of the property as floats in the second column. The time series should be in chronological
                            order. Additionally, the reference system of the current direction should be in [0,2pi]-form with the northern direction
                            corresponding to 0 and 2pi radians.
        """

        def astronomical_tide(signal_time, signal_values):
            old_stdout = sys.stdout  # backup current stdout
            sys.stdout = open(os.devnull, "w")
            signal_datetime = [datetime.datetime.fromtimestamp(y) for y in signal_time]
            duration = signal_datetime[-1] - signal_datetime[0]
            periods = [[timedelta(minutes=25, hours=12), 'tidalcycle'],
                       [timedelta(days=1), 'day'],
                       [timedelta(days=14), 'springneap'],
                       [timedelta(days=31), 'month'],
                       [timedelta(days=365 / 2), 'halfyear'],
                       [timedelta(days=365), 'year']]
            indexes = [deltatime[0] >= duration for deltatime in periods]
            if True not in indexes: index = -1
            elif duration < timedelta(minutes=25, hours=12): return [signal_time, signal_values]
            const_list = hatyan.get_const_list_hatyan('springneap')#periods[indexes.index(True)][1])
            ts_meas = pd.DataFrame({'values': signal_values}, index=signal_datetime)
            ts_meas = hatyan.crop_timeseries(ts=ts_meas, times_ext=signal_datetime);
            comp_frommeas, comp_allyears = hatyan.get_components_from_ts(ts=ts_meas, const_list=const_list,nodalfactors=True, return_allyears=True,fu_alltimes=True)
            ts_prediction = hatyan.prediction(comp=comp_frommeas, nodalfactors=True, xfac=True, fu_alltimes=True,times_ext=signal_datetime, timestep_min=10)
            sys.stdout = old_stdout  # reset old stdout
            return [[index.timestamp()-3600 for index in ts_prediction.index], [value for value in ts_prediction['values']]]

        def H99(t,wlev,node):
            """ Function: calculates the water level which is exceeded 99% of the tides for a given node in the network.

                Input:
                    - t: a list containing the chronological sequence of timestamps for the specific node, derived from the
                         list of the time series containing the water level at the specific node
                    - wlev: a list containing the chronological sequence of water levels for the specific node, derived from the
                            list of the time series containing the water level at the specific node
                    - node: the name string of the node in the given network
            """

            # Deriving some properties
            mean_wlev = np.mean(wlev)
            deltat = t[1] - t[0]

            # Interpolation of the variation of the water level and calculation of the zero crossing times
            intp = sc.interpolate.CubicSpline(t, [w - mean_wlev for w in wlev])
            roots = []
            for root in intp.roots():
                if root < network.nodes[node]['Info']['Water level'][0][0] or root > network.nodes[node]['Info']['Water level'][0][-1]:
                    continue
                roots.append(root)

            # Calculation of the maximum water level for each tidal cycle (when water level is greater than mean water level)
            max_index = []
            max_t = []
            max_wlev = []
            prev_root = t[0]
            for root in enumerate(roots):
                if wlev[0] <= mean_wlev:
                    if root[0] % 2 == 0:
                        prev_root = root[1]
                        continue
                else:
                    if root[0] % 2 == 1:
                        prev_root = root[1]
                        continue

                index_range_min = bisect.bisect_right(t, prev_root)
                if t[index_range_min] - prev_root < deltat / 2: index_range_min = index_range_min - 1
                index_range_max = bisect.bisect_right(t, root[1])
                if index_range_max == len(t) or t[index_range_max] - root[1] < deltat / 2: index_range_max = index_range_max - 1
                if index_range_min == index_range_max: continue
                max_index.append(np.argmax(wlev[index_range_min:index_range_max]) + index_range_min)
                max_t.append(t[max_index[-1]])
                max_wlev.append(wlev[max_index[-1]])

            # Sorting lists from highest to lowest maximum water level
            zipped_lists = zip(max_wlev, max_index, max_t)
            sorted_pairs = sorted(zipped_lists)
            tuples = zip(*sorted_pairs)
            mwlev, mindex, mt = [list(tuple) for tuple in tuples]

            # Returning the water level in the list closest to the 99% value
            return mwlev[round(len(max_wlev) * 0.01)]

        def tidal_periods(t_astro_tide_wlev, astro_tide_wlev):
            """ Function: calculates the water level which is exceeded 99% of the tides for a given node in the network.

                Input:
                    - t_astro_wlev: a list containing the chronological sequence of timestamps for the specific node, derived from the
                                    list of the time series containing the water level at the specific node
                    - astro_wlev: a list containing the chronological sequence of water levels for the specific node, derived from the
                                  list of the time series containing the water level at the specific node
            """

            # Deriving some properties
            mean_astro_wlev = np.mean(astro_tide_wlev)

            # Interpolation of the variation of the water level and calculation of the zero crossing times
            cor_astro_tide = [astro_wlev - mean_astro_wlev for astro_wlev in astro_tide_wlev]
            intp = sc.interpolate.CubicSpline(t_astro_tide_wlev, cor_astro_tide)
            times_tidal_periods = []
            index_prev_root = 0
            for root in intp.roots():
                index_current_root = bisect.bisect_right(t_astro_tide_wlev, root)
                if root >= t_astro_tide_wlev[0] and root <= t_astro_tide_wlev[-1] and np.max(cor_astro_tide[index_prev_root:index_current_root]) > 0:
                    index = cor_astro_tide[index_prev_root:index_current_root].index(np.max(cor_astro_tide[index_prev_root:index_current_root]))
                    time_start_next_tidal_period = t_astro_tide_wlev[index + index_prev_root]
                    times_tidal_periods.append([time_start_next_tidal_period, 'Ebb Start'])
                    index_prev_root = index_current_root
                elif root >= t_astro_tide_wlev[0] and root <= t_astro_tide_wlev[-1] and np.min(cor_astro_tide[index_prev_root:index_current_root]) < 0:
                    index = cor_astro_tide[index_prev_root:index_current_root].index(np.min(cor_astro_tide[index_prev_root:index_current_root]))
                    time_start_next_tidal_period = t_astro_tide_wlev[index + index_prev_root]
                    times_tidal_periods.append([time_start_next_tidal_period, 'Flood Start'])
                    index_prev_root = index_current_root

            return times_tidal_periods

        def append_info_to_edges(network):
            """ Function: appends nodal data to the edges of the network for visualisation purposes only

                Input:
                    - network: a graph constructed with the DiGraph class of the networkx package

            """

            #Function that is used in the calculation of info that needs to be appended to the edges
            def lag_finder(time_signal, signal1, signal2):
                nsamples = len(signal2)
                b, a = sc.signal.butter(2, 0.1)
                signal2 = sc.signal.filtfilt(b, a, signal2)
                # regularize datasets by subtracting mean and dividing by s.d.
                signal2 -= np.mean(signal2);
                signal2 /= np.std(signal2)
                signal1 -= np.mean(signal1);
                signal1 /= np.std(signal1)
                # Find cross-correlation
                xcorr = sc.signal.correlate(signal2, signal1)

                # delta time array to match xcorr
                dt = np.arange(1 - nsamples, nsamples)
                time_step = time_signal[1] - time_signal[0]
                delay = dt[xcorr.argmax()] * time_step
                return delay

            # Loops over the edges of the network
            for edge in enumerate(network.edges):
                # Adds parameters to the dictionary
                network.edges[edge[1]]['Info']['Width'] = []
                network.edges[edge[1]]['Info']['Depth'] = []
                network.edges[edge[1]]['Info']['MBL'] = []
                network.edges[edge[1]]['Info']['Tidal phase lag'] = []

                #Appends data to the edges
                network.edges[edge[1]]['Info']['Width'] = np.min([network.nodes[edge[1][0]]['Info']['Width'], network.nodes[edge[1][1]]['Info']['Width']])
                network.edges[edge[1]]['Info']['Tidal phase lag'] = lag_finder(network.nodes[edge[1][0]]['Info']['Astronomical tide'][0],
                                                                               network.nodes[edge[1][0]]['Info']['Astronomical tide'][1],
                                                                               network.nodes[edge[1][1]]['Info']['Astronomical tide'][1])

                #If there is a terminal in the edge, the greatest value for the MBL or depth of the two nodes creating the edge are used
                if 'Terminal' in network.edges[edge[1]]:
                    network.edges[edge[1]]['Info']['Depth'] = np.max([network.nodes[edge[1][0]]['Info']['Depth'], network.nodes[edge[1][1]]['Info']['Depth']])
                    network.edges[edge[1]]['Info']['MBL'] = np.max([network.nodes[edge[1][0]]['Info']['MBL'], network.nodes[edge[1][1]]['Info']['MBL']])
                else:
                    network.edges[edge[1]]['Info']['Depth'] = np.min([network.nodes[edge[1][0]]['Info']['Depth'], network.nodes[edge[1][1]]['Info']['Depth']])
                    network.edges[edge[1]]['Info']['MBL'] = np.min([network.nodes[edge[1][0]]['Info']['MBL'], network.nodes[edge[1][1]]['Info']['MBL']])

        def calculate_and_append_current_components_to_nodes(network):
            """ Function: calculates the longitudinal and lateral (cross-) current velocity with respect to the sailing path of
                          the vessel (alignment of the edge) for each node (there can be multiple time series for each node as
                          multiple edges can be connected at this node) and appends this information to the nodes of the network.

                Input:
                    - network: a graph constructed with the DiGraph class of the networkx package
            """

            #Loops over the nodes of the network and finds the edges attached to this node by a second loop over the nodes
            for node1 in network.nodes:
                nodes = []
                for node2 in network.nodes:
                    if (node1, node2) in network.edges:
                        nodes.append(node2)

                #If there is only one edge connected to the node (boundary nodes): longitudinal and cross-currents are assumed 0
                if len(nodes) == 1:
                    network.nodes[node1]['Info']['Cross-current'][nodes[0]] = [[], []]
                    network.nodes[node1]['Info']['Longitudinal current'][nodes[0]] = [[], []]
                    time_current_velocity = network.nodes[node1]['Info']['Current velocity'][0]
                    for t in range(len(time_current_velocity)):
                        network.nodes[node1]['Info']['Cross-current'][nodes[0]][0].append(time_current_velocity[t])
                        network.nodes[node1]['Info']['Cross-current'][nodes[0]][1].append(0)
                        network.nodes[node1]['Info']['Longitudinal current'][nodes[0]][0].append(time_current_velocity[t])
                        network.nodes[node1]['Info']['Longitudinal current'][nodes[0]][1].append(0)
                    continue

                # Else if there is multiple edges connected to the node: loop over the connected edges
                for node2 in nodes:
                    network.nodes[node1]['Info']['Cross-current'][node2] = [[], []]
                    network.nodes[node1]['Info']['Longitudinal current'][node2] = [[], []]
                    time_current = network.nodes[node1]['Info']['Current velocity'][0]
                    current_velocity = network.nodes[node1]['Info']['Current velocity'][1]
                    current_direction = network.nodes[node1]['Info']['Current direction'][1]

                    #Calculation of the orientation of the edge
                    origin_lat = network.nodes[node2]['geometry'].x
                    origin_lon = network.nodes[node2]['geometry'].y
                    node_lat = network.nodes[node1]['geometry'].x
                    node_lon = network.nodes[node1]['geometry'].y
                    course, _, _ = pyproj.Geod(ellps="WGS84").inv(origin_lat, origin_lon, node_lat, node_lon)
                    if course < 0:
                        course = 360 + course

                    #Calculation of the current velocity components
                    for t in range(len(time_current)):
                        network.nodes[node1]['Info']['Cross-current'][node2][0].append(time_current[t])
                        network.nodes[node1]['Info']['Cross-current'][node2][1].append(abs(current_velocity[t] * np.sin((current_direction[t] - course) / 180 * math.pi)))
                        network.nodes[node1]['Info']['Longitudinal current'][node2][0].append(time_current[t])
                        network.nodes[node1]['Info']['Longitudinal current'][node2][1].append(abs(current_velocity[t] * np.cos((current_direction[t] - course) / 180 * math.pi)))

        #Continuation of the append_data_to_nodes function by looping over all the nodes
        for node in enumerate(network.nodes):
            #Creating a dictionary attached to the nodes of the network
            network.nodes[node[1]]['Info'] = {'Tidal periods': [],'Horizontal tidal restriction': {},'Vertical tidal restriction': {}, 'Width': [], 'Depth': [], 'MBL': [], 'Astronomical tide': [], 'H_99%': [], 'Water level': [[],[]],  'Current velocity': [[],[]], 'Current direction': [[],[]], 'Cross-current': {}, 'Longitudinal current': {}}

            network.nodes[node[1]]['Info']['MBL'] = float(data['MBL'][node[0]].values)
            network.nodes[node[1]]['Info']['Width'] = float(data['Width'][node[0]].values)
            network.nodes[node[1]]['Info']['Depth'] = float(data['Depth'][node[0]].values)
            network.nodes[node[1]]['Info']['Water level'][0] = list(data['Times'].values)
            network.nodes[node[1]]['Info']['Water level'][1] = list(data['Water level'][node[0]].values)
            network.nodes[node[1]]['Info']['Current velocity'][0] = list(data['Times'].values)
            network.nodes[node[1]]['Info']['Current velocity'][1] = list(data['Current velocity'][node[0]].values) #.transpose('STATION', 'TIME', 'LAYER')
            network.nodes[node[1]]['Info']['Current direction'][0] = list(data['Times'].values)
            network.nodes[node[1]]['Info']['Current direction'][1] = list(data['Current direction'][node[0]].values) #.transpose('STATION', 'TIME', 'LAYER')

            # Calculation of the water level which is exceeded 99% of the tides
            network.nodes[node[1]]['Info']['Astronomical tide'] = astronomical_tide(network.nodes[node[1]]['Info']['Water level'][0],network.nodes[node[1]]['Info']['Water level'][1])
            network.nodes[node[1]]['Info']['H_99%'] = H99(network.nodes[node[1]]['Info']['Astronomical tide'][0],network.nodes[node[1]]['Info']['Astronomical tide'][1],node[1])
            network.nodes[node[1]]['Info']['Tidal periods'] = tidal_periods(network.nodes[node[1]]['Info']['Astronomical tide'][0],                                                              network.nodes[node[1]]['Info']['Astronomical tide'][1])

        # Appending longitudinal and cross-current velocity components to the nodes of the network
        calculate_and_append_current_components_to_nodes(network)

        # Appending static data to the edges
        append_info_to_edges(network)

    def append_vertical_tidal_restriction_to_network(network,node,vertical_tidal_window_input):
        """ Function: appends vertical tidal restrictions to the node of the network

            Input:
                - network: a graph constructed with the DiGraph class of the networkx package
                - node: the name string of the node in the given network
                - vertical_tidal_window_input: assembly of specific information that defines the restriction (see specific input classes in the notebook)
        """

        #Specifies two parameters in the dictionary with a corresponding data structure of lists
        network.nodes[node]['Info']['Vertical tidal restriction']['Type'] = [[], [], []]
        network.nodes[node]['Info']['Vertical tidal restriction']['Specification'] = [[], [], [], [], [], []]

        #Loops over the number of types of restrictions that may hold for different classes of vessels
        for input_data in vertical_tidal_window_input:
            # Unpacks the data for flood and ebb and appends it to a list
            ukc_p = []
            ukc_s = []
            fwa = []
            for info in input_data.window_specifications.ukc_p:
                ukc_p.append(input_data.window_specifications.ukc_p[info])
            for info in input_data.window_specifications.ukc_s:
                ukc_s.append(input_data.window_specifications.ukc_s[info])
            for info in input_data.window_specifications.fwa:
                fwa.append(input_data.window_specifications.fwa[info])

            # Appends the specific data regarding the type of the restriction to data structure
            network.nodes[node]['Info']['Vertical tidal restriction']['Type'][0].append(ukc_s)
            network.nodes[node]['Info']['Vertical tidal restriction']['Type'][1].append(ukc_p)
            network.nodes[node]['Info']['Vertical tidal restriction']['Type'][2].append(fwa)

            # Unpacks the data for the different vessel criteria and appends it to a list
            vessel_characteristics_type = []
            vessel_characteristics_spec = []
            vessel_characteristics_value = []
            vessel_characteristics = input_data.vessel_specifications.characteristic_dicts()
            for info in vessel_characteristics:
                vessel_characteristics_type.append(info)
                vessel_characteristics_spec.append(vessel_characteristics[info][0])
                vessel_characteristics_value.append(vessel_characteristics[info][1])

            # Unravels the boolean operators between the restrictions and appends it to a list
            vessel_method_list = []
            sign_list = input_data.vessel_specifications.vessel_method.split()
            for sign in sign_list:
                if sign[0] != '(' and sign[-1] != ')' and sign != 'x':
                    vessel_method_list.append(sign)

            # Appends the specific data regarding the properties for which the restriction holds to data structure
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][0].append(vessel_characteristics_type)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][1].append(vessel_characteristics_value)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][2].append(input_data.vessel_specifications.vessel_direction)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][3].append(vessel_characteristics_spec)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][3].append(vessel_characteristics_spec)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][4].append(vessel_method_list)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][5].append([])

    def append_horizontal_tidal_restriction_to_network(network,node,horizontal_tidal_window_input):
        """ Function: appends horizontal tidal restrictions to the node of the network

            Input:
                - network: a graph constructed with the DiGraph class of the networkx package
                - node: the name string of the node in the given network
                - horizontal_tidal_window_input: assembly of specific information that defines the restriction (see specific input classes in the notebook)
        """

        # Specifies two parameters in the dictionary with a corresponding data structure of lists
        network.nodes[node]['Info']['Horizontal tidal restriction']['Type'] = [[], [], []]
        network.nodes[node]['Info']['Horizontal tidal restriction']['Data'] = {}
        network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'] = [[], [], [], [], [], []]

        # Loops over the number of types of restrictions that may hold for different classes of vessels
        for input_data in enumerate(horizontal_tidal_window_input):
            # Unpacks the data for flood and ebb and appends it to a list
            current_velocity_value = []
            for info in input_data[1].window_specifications.current_velocity_values:
                current_velocity_value.append(input_data[1].window_specifications.current_velocity_values[info])

            # Dependent on the type of restriction, additional information should be unpacked and appended to a list
            current_velocity_range = []
            # - if no necessary:
            if input_data[1].window_specifications.current_velocity_ranges == dict:
                current_velocity_range = []
            # - elif necessary:
            else:
                for info in input_data[1].window_specifications.current_velocity_ranges:
                    current_velocity_range.append(input_data[1].window_specifications.current_velocity_ranges[info])

            # Appends the specific data regarding the type of the restriction to data structure
            network.nodes[node]['Info']['Horizontal tidal restriction']['Type'][0].append(input_data[1].window_specifications.window_method)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Type'][1].append(current_velocity_value)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Type'][2].append(current_velocity_range)

            # Unpacks the data for the different vessel criteria and appends it to a list
            vessel_characteristics_type = []
            vessel_characteristics_spec = []
            vessel_characteristics_value = []
            vessel_characteristics = input_data[1].vessel_specifications.characteristic_dicts()
            for info in vessel_characteristics:
                vessel_characteristics_type.append(info)
                vessel_characteristics_spec.append(vessel_characteristics[info][0])
                vessel_characteristics_value.append(vessel_characteristics[info][1])

            # Unravels the boolean operators between the restrictions and appends it to a list
            vessel_method_list = []
            sign_list = input_data[1].vessel_specifications.vessel_method.split()
            for sign in sign_list:
                if sign[0] != '(' and sign[-1] != ')' and sign != 'x':
                    vessel_method_list.append(sign)

            # Appends the specific data regarding the properties for which the restriction holds to data structure
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][0].append(vessel_characteristics_type)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][1].append(vessel_characteristics_value)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][2].append(input_data[1].vessel_specifications.vessel_direction)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][3].append(vessel_characteristics_spec)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][4].append(vessel_method_list)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][5].append([input_data[1].condition['Origin'], input_data[1].condition['Destination']])
            # Unpacks the specific current velocity data used for the restriction and appends it to the data structure
            # - if pre-calculated data from the model should be used:
            if type(input_data[1].data[1]) == str:
                # - if raw current velocity data should be used:
                if input_data[1].data[1] == 'Current velocity':
                    for n in network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][5][input_data[0]]:
                        network.nodes[node]['Info']['Horizontal tidal restriction']['Data'][n] = network.nodes[input_data[1].data[0]]['Info'][input_data[1].data[1]]
                # - elif longitudinal or cross-current velocity should be used:
                else:
                    network.nodes[node]['Info']['Horizontal tidal restriction']['Data'] = network.nodes[input_data[1].data[0]]['Info'][input_data[1].data[1]]
            # - elif manual data should be used:
            else:
                for n in nodes:
                    network.nodes[node]['Info']['Horizontal tidal restriction']['Data'][n][0] = input_data[1].data[node][0]
                    network.nodes[node]['Info']['Horizontal tidal restriction']['Data'][n][1] = input_data[1].data[node][1]