from opentnsim.core import SimpyObject, Identifiable, Movable
from opentnsim.graph import OnNode
from opentnsim.tidal_accessibility import check_if_route_contains_restrictions

import datetime
import pandas as pd
import numpy as np
import networkx as nx
import simpy

class IsPartofPort:
    def __init__(self, port, *args, **kwargs):
        if not isinstance(port,IsPort):
            raise ValueError("'port' should be an IsPort-object")
        self.port = port
        super().__init__(*args, **kwargs)

class HasPortAccess(Movable):
    def __init__(self, bound, *args, **kwargs):
        self.bound = bound
        super().__init__(*args, **kwargs)
        self.on_pass_node_functions.append(self.request_port_entry)


    def determine_sailing_time(self):
        route = self.route
        sailing_information = self.env.vessel_traffic_service.provide_sailing_time(self, route)
        sailing_time = sailing_information["Time"].cumsum().values[0]
        return sailing_time


    def determine_new_route_after_terminal(self):
        new_route = None
        origin = self.route[-1]
        if self.next_destination is not None:
            destination = self.next_destination
            new_route = nx.dijkstra_path(self.env.graph,origin,destination)
        elif len(self.next_terminals):
            next_terminal = self.next_terminals[-1]
            self.next_terminals = self.next_terminals[1:]
            next_terminal.request_terminal_access(vessel, origin)
        return new_route

    def request_port_entry(self, origin, at_terminal = False, leaving_port = False):
        # Request for a terminal
        if not at_terminal:
            if 'Port Entry' not in self.env.graph.nodes[origin].keys():
                return

            port = self.env.graph.nodes[origin]['Port Entry'].port
            if not self.terminal.port == port:
                return
            elif 'port_accessed' in dir(self) and self.port_accessed == port:
                return

        if leaving_port:
            route = self.determine_new_route_after_terminal()
            self.route = route
            self.bound = 'outbound'

        df_entry = self.get_port_access_info(leaving_port=leaving_port)
        if leaving_port:
            port = self.terminal.port
            _, _, waiting_times, waiting_causes = port.check_vessel_entry(self, df_entry)
            for index, (waiting_time, waiting_cause) in enumerate(zip(waiting_times, waiting_causes)):
                if waiting_cause == 'Tide':
                    yield from self.wait_for_tidal_window(origin, waiting_time)
            return

        available_berth_time_slots = self.terminal.determine_berth_availability(self, origin)
        berth, berth_name, waiting_times, waiting_causes = port.check_vessel_entry(self, df_entry, available_berth_time_slots)
        self.terminal.assign_berth_to_vessel(self, berth)

        self.port_accessed = port
        for index,(waiting_time,waiting_cause) in enumerate(zip(waiting_times,waiting_causes)):
            waiting_time = waiting_time.total_seconds()
            if not index:
                departure_time_to_anchorage = self.env.now
                yield from self.sail_to_anchorage(origin)
                arrival_time_at_anchorage = self.env.now
                correction_waiting_time = arrival_time_at_anchorage - departure_time_to_anchorage
                waiting_time = waiting_time - correction_waiting_time

            if waiting_cause == berth_name:
                yield from self.wait_for_berth_availability(origin, waiting_time)

            elif waiting_cause == 'Tide':
                yield from self.wait_for_tidal_window(origin, waiting_time)


    def request_port_exit(self, origin):
        yield from self.request_port_entry(origin, at_terminal = True, leaving_port = True)


    def get_port_access_info(self, leaving_port = False):
        port = self.terminal.port
        df_tidal_availability = port.check_tidal_availability(self)
        df_berth_availability = pd.DataFrame()
        if not leaving_port:
            df_berth_availability = port.check_terminal_accessibility(self)

        #Combine the dataframes
        df_entry = pd.DataFrame()
        df_entry['Tide'] = df_tidal_availability['Accessibility'] == 'Accessible'
        df_entry = pd.concat([df_entry,df_berth_availability],axis=1)
        df_entry = df_entry.sort_index()
        with pd.option_context("future.no_silent_downcasting", True):
            df_entry = df_entry.ffill()

        return df_entry


class IsPort(SimpyObject, Identifiable):
    def __init__(self, port_entry_nodes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.port_entry_nodes = port_entry_nodes
        if not len(self.port_entry_nodes):
            raise Warning('The port will not be accessible for vessels. It needs port entry_nodes')
        for port_entry_node in self.port_entry_nodes:
            IsPortEntry(env=self.env,node=port_entry_node,port=self)
        self.anchorage_areas = []
        self.terminals = []
        if 'ports' not in dir(self.env):
            self.env.ports = []
        self.env.ports.append(self)


    def check_terminal_accessibility(self, vessel):
        df_berth_availability = vessel.terminal.determine_terminal_availability(vessel)
        return df_berth_availability


    def check_tidal_availability(self, vessel):
        tide_bound = check_if_route_contains_restrictions(vessel)
        route = vessel.route
        time_start = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now))
        sailing_time = vessel.determine_sailing_time()
        sailing_time = np.max([pd.Timedelta(seconds=sailing_time), pd.Timedelta(hours=48)])
        time_end = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now) + sailing_time)
        if tide_bound:
            df_tidal_availability = vessel.env.vessel_traffic_service.provide_tidal_windows(vessel, route, time_start, time_end, plot=True)[0]
        return df_tidal_availability


    def check_port_accessibility(self, df_entry_future_to_entry_time, leaving_port):
        waiting_stops_info = df_entry_future_to_entry_time[df_entry_future_to_entry_time.Combined == True]
        waiting_starts_info = df_entry_future_to_entry_time[df_entry_future_to_entry_time.Combined == False]
        if len(waiting_starts_info) and not len(waiting_stops_info):
            if leaving_port:
                raise Warning("The port will not be accessible for the vessel's outbound trip")
            else:
                raise Warning("The port will not be accessible for the vessel's inbound trip")
            raise simpy.exceptions.Interrupt('Vessel trip not possible.')

    def check_vessel_entry(self, vessel, df_entry, available_berths = None, leaving_port = False):
        best_available_berth = None
        best_available_berth_name = None
        waiting_times = None
        waiting_causes = None
        if available_berths is not None:
            minimum_waiting_time = available_berths.Waiting_time.min()
            berths_with_minimum_waiting_time = available_berths[available_berths.Waiting_time == minimum_waiting_time]
            minimum_berth_length = berths_with_minimum_waiting_time.Berth_length.min()
            best_available_berths = berths_with_minimum_waiting_time[berths_with_minimum_waiting_time.Berth_length == minimum_berth_length]
            best_available_berth = best_available_berths.iloc[0]
            best_available_berth_name = best_available_berth.name

        current_time = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now))
        df_entry_future_to_entry_time = df_entry[(df_entry.index >= current_time)]
        if available_berths is None:
            if not 'Tide' in df_entry_future_to_entry_time.columns:
                return best_available_berth, best_available_berth_name, waiting_times, waiting_causes
            else:
                df_entry_future_to_entry_time = df_entry_future_to_entry_time[['Tide']]
        else:
            if 'Tide' in df_entry_future_to_entry_time.columns:
                df_entry_future_to_entry_time = df_entry_future_to_entry_time[['Tide', best_available_berth_name]]
            else:
                df_entry_future_to_entry_time = df_entry_future_to_entry_time[[best_available_berth_name]]

        df_entry_future_to_entry_time['Combined'] = df_entry_future_to_entry_time.all(axis=1)
        df_entry_future_to_entry_time['Redundant'] = df_entry_future_to_entry_time.any(axis=1) == False
        df_entry_future_to_entry_time = df_entry_future_to_entry_time[df_entry_future_to_entry_time['Redundant'] == False]
        df_entry_future_to_entry_time = df_entry_future_to_entry_time.drop('Redundant', axis=1)
        df_entry_future_to_entry_time["Reason"] = df_entry_future_to_entry_time.apply(lambda row: row.index[row.eq(False)][0] if any(row.eq(False)) else None, axis=1)
        with pd.option_context("future.no_silent_downcasting", True):
            df_entry_future_to_entry_time = df_entry_future_to_entry_time.ffill()
        waiting_stops_info = df_entry_future_to_entry_time[df_entry_future_to_entry_time.Combined == True]
        waiting_starts_info = df_entry_future_to_entry_time[df_entry_future_to_entry_time.Combined == False]
        self.check_port_accessibility(df_entry_future_to_entry_time, leaving_port=leaving_port)
        waiting_times = []
        waiting_causes = []
        for (waiting_start_time, waiting_start_info), (waiting_stop_time, waiting_stop_info) in zip(waiting_starts_info.iterrows(), waiting_stops_info.iterrows()):
            waiting_times.append(waiting_stop_time - waiting_start_time)
            waiting_causes.append(waiting_start_info.Reason)
            break

        if available_berths is not None:
            best_available_berth = vessel.terminal.select_berth_based_on_name(best_available_berth_name)
        return best_available_berth, best_available_berth_name, waiting_times, waiting_causes

    def check_vessel_exit(self, vessel, df_entry):
        return check_vessel_entry(self, vessel, df_entry, leaving_port = True)

class IsPortEntry(SimpyObject, OnNode, IsPartofPort):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env.graph.nodes[self.node]['Port Entry'] = self