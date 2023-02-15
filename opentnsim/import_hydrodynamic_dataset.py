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
import time as timepy

import matplotlib.pyplot as plt

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

    def append_data_to_nodes(network,data,simulation_type='Default'):
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
                if root < t[0] or root > t[-1]:
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
                if index_range_min >= index_range_max: continue
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

        def append_info_to_edges(network,simulation_type):
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
                if type(network.edges[edge[1]]) != dict:
                    network.edges[edge[1]] = {}
                if 'Info' not in list(network.edges[edge[1]].keys()):
                    network.edges[edge[1]]['Info'] = {}
                network.edges[edge[1]]['Info']['Width'] = []
                network.edges[edge[1]]['Info']['Depth'] = []
                network.edges[edge[1]]['Info']['MBL'] = []

                #Appends data to the edges
                if 'Width' in list(network.nodes[node[1]]['Info'].keys()):
                    network.edges[edge[1]]['Info']['Width'] = np.min([network.nodes[edge[1][0]]['Info']['Width'], network.nodes[edge[1][1]]['Info']['Width']])

                if simulation_type == 'Accessibility':
                    if 'Tidal phase lag' not in list(network.nodes[node[1]]['Info'].keys()):
                        network.edges[edge[1]]['Info']['Tidal phase lag'] = []
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
                network.nodes[node1]['Info']['Cross-current'] = {}
                network.nodes[node1]['Info']['Longitudinal current'] = {}
                for node2 in network.nodes:
                    if (node1, node2) in network.edges:
                        nodes.append(node2)

                #If there is only one edge connected to the node (boundary nodes): longitudinal and cross-currents are assumed 0
                if len(nodes) == 1:
                    network.nodes[node1]['Info']['Cross-current'][nodes[0]] = []
                    network.nodes[node1]['Info']['Longitudinal current'][nodes[0]] = []
                    for t in range(len(network.nodes[node1]['Info']['Times'])):
                        network.nodes[node1]['Info']['Cross-current'][nodes[0]].append(0)
                        network.nodes[node1]['Info']['Longitudinal current'][nodes[0]].append(0)
                    continue

                # Else if there is multiple edges connected to the node: loop over the connected edges
                for node2 in nodes:
                    network.nodes[node1]['Info']['Cross-current'][node2] = []
                    network.nodes[node1]['Info']['Longitudinal current'][node2] = []
                    current_velocity = network.nodes[node1]['Info']['Current velocity']
                    current_direction = network.nodes[node1]['Info']['Current direction']

                    #Calculation of the orientation of the edge
                    origin_lat = network.nodes[node2]['geometry'].x
                    origin_lon = network.nodes[node2]['geometry'].y
                    node_lat = network.nodes[node1]['geometry'].x
                    node_lon = network.nodes[node1]['geometry'].y
                    course, _, _ = pyproj.Geod(ellps="WGS84").inv(origin_lat, origin_lon, node_lat, node_lon)
                    if course < 0:
                        course = 360 + course

                    #Calculation of the current velocity components
                    # network.nodes[node1]['Info']['Cross-current'][node2] = [([abs(current_velocity[l][t] * np.sin((current_direction[l][t] - course) / 180 * math.pi)) for t in range(len(network.nodes[node1]['Info']['Times']))]) for l in range(len(network.nodes[node1]['Info']['Layers']))]
                    # network.nodes[node1]['Info']['Longitudinal current'][node2] = [([abs(current_velocity[l][t] * np.cos((current_direction[l][t] - course) / 180 * math.pi)) for t in range(len(network.nodes[node1]['Info']['Times']))]) for l in range(len(network.nodes[node1]['Info']['Layers']))]

        #Continuation of the append_data_to_nodes function by looping over all the nodes
        for node in enumerate(network.nodes):
            #Creating a dictionary attached to the nodes of the network
            if type(network.nodes[node[1]]) != dict:
                network.nodes[node[1]] = {}
            if 'Info' not in list(network.nodes[node[1]].keys()):
                network.nodes[node[1]]['Info'] = {}

            for data_name in list(data.keys()):
                network.nodes[node[1]]['Info'][data_name] = []

            #network.nodes[node[1]]['Info']['Times'] = [(time - np.datetime64('1970-01-01T00:00:00.000000')) / np.timedelta64(1, 's') for time in list(data['Times'].values)]
            network.nodes[node[1]]['Info']['Times'] = list(data['Times'].values)
            if 'Relative layer height' in list(network.nodes[node[1]]['Info'].keys()):
                network.nodes[node[1]]['Info']['Layers'] = list(data['Relative layer height'].values)
            else:
                network.nodes[node[1]]['Info']['Layers'] = [1]

            if 'Width' in list(network.nodes[node[1]]['Info'].keys()):
                network.nodes[node[1]]['Info']['Width'] = float(data['Width'][node[0]].values)
            if 'Depth' in list(network.nodes[node[1]]['Info'].keys()):
                network.nodes[node[1]]['Info']['Depth'] = float(data['Depth'][node[0]].values)
            if 'MBL' in list(network.nodes[node[1]]['Info'].keys()):
                if node[1] == '8867980' or node[1] == '8861158' or node[1] == '8867547':
                    network.nodes[node[1]]['Info']['MBL'] = 16.4
                elif node[1] == '8866999' or node[1] == '8866859':
                    network.nodes[node[1]]['Info']['MBL'] = 25.0
                else:
                    network.nodes[node[1]]['Info']['MBL'] = float(data['MBL'][node[0]].values)
            else:
                if node[1] == '8867980' or node[1] == '8861158' or node[1] == '8867547':
                    network.nodes[node[1]]['Info']['MBL'] = 16.4
                elif node[1] == '8866999' or node[1] == '8866859':
                    network.nodes[node[1]]['Info']['MBL'] = 25.0
                else:
                    network.nodes[node[1]]['Info']['MBL'] = float(data['Depth'][node[0]].values)
            if 'Water level' in list(network.nodes[node[1]]['Info'].keys()):
                network.nodes[node[1]]['Info']['Water level'] = list(data['Water level'][node[0]].values)
            if 'Current velocity' in list(network.nodes[node[1]]['Info'].keys()):
                network.nodes[node[1]]['Info']['Current velocity'] = list(data['Current velocity'][node[0]].values) #.transpose('STATION', 'TIME', 'LAYER')
            if 'Current direction' in list(network.nodes[node[1]]['Info'].keys()):
                network.nodes[node[1]]['Info']['Current direction'] = list(data['Current direction'][node[0]].values) #.transpose('STATION', 'TIME', 'LAYER')
            if 'Salinity' in list(network.nodes[node[1]]['Info'].keys()):
                network.nodes[node[1]]['Info']['Salinity'] = list(data['Salinity'][node[0]].values) #.transpose('STATION', 'TIME', 'LAYER')
            if 'Tidal phase lag' in list(network.nodes[node[1]]['Info'].keys()):
                network.nodes[node[1]]['Info']['Tidal phase lag'] = float(data['Phase lag'][node[0]].values)

            if simulation_type == 'Accessibility':
                # Calculation of the water level which is exceeded 99% of the tides
                network.nodes[node[1]]['Info']['Vertical tidal restriction'] = {}
                network.nodes[node[1]]['Info']['Horizontal tidal restriction'] = {}
                if 'Astronomic water level' not in dir(data):
                    network.nodes[node[1]]['Info']['Astronomical tide'] = astronomical_tide(network.nodes[node[1]]['Info']['Times'],network.nodes[node[1]]['Info']['Water level'])
                else:
                    network.nodes[node[1]]['Info']['Astronomical tide'] = [network.nodes[node[1]]['Info']['Times'],list(data['Astronomic water level'][node[0]].values)]

                network.nodes[node[1]]['Info']['H_99%'] = H99(network.nodes[node[1]]['Info']['Astronomical tide'][0],network.nodes[node[1]]['Info']['Astronomical tide'][1],node[1])

                if 'Observed horizontal tidal periods' not in dir(data):
                    network.nodes[node[1]]['Info']['Tidal periods'] = tidal_periods(network.nodes[node[1]]['Info']['Astronomical tide'][0],network.nodes[node[1]]['Info']['Astronomical tide'][1])
                else:
                    network.nodes[node[1]]['Info']['Tidal periods'] = list(data['Observed horizontal tidal periods'][node[0]].values)

        # Appending longitudinal and cross-current velocity components to the nodes of the network
        if simulation_type == 'Accessibility':
            calculate_and_append_current_components_to_nodes(network)

        # Appending static data to the edges
        append_info_to_edges(network,simulation_type)