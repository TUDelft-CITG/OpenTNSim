from opentnsim.core import SimpyObject, Identifiable, Movable
from opentnsim.graph import OnNode
from opentnsim.tidal_accessibility import check_if_route_contains_restrictions

import datetime
import pandas as pd
import numpy as np
import networkx as nx
import simpy
import warnings


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
        else:
            port = self.terminal.port
        yield from port.communicate_port_accessibility_info(self, origin, leaving_port=leaving_port)

    def request_port_exit(self, origin):
        yield from self.request_port_entry(origin, at_terminal = True, leaving_port = True)


class IsPort(SimpyObject, Identifiable):
    def __init__(self, port_entry_nodes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.port_entry_nodes = port_entry_nodes
        if not len(self.port_entry_nodes):
            raise ValueError('The port will not be accessible for vessels. It needs port entry_nodes')
        for port_entry_node in self.port_entry_nodes:
            IsPortEntry(env=self.env,node=port_entry_node,port=self)
        self.anchorage_areas = []
        self.terminals = []
        if 'ports' not in dir(self.env):
            self.env.ports = []
        self.env.ports.append(self)


    def communicate_port_accessibility_info(self, vessel, origin, leaving_port = False):
        if leaving_port:
            route = self.plan_new_route_for_vessel(vessel)
            vessel.route = route
            vessel.bound = 'outbound'

        df_entry = self.get_accessibility_info(vessel, leaving_port=leaving_port)

        berth = None
        if not leaving_port:
            berth = self.request_terminal_access(vessel, origin, df_entry)

        waiting_times, waiting_causes = self.determine_potential_waiting_time_for_vessel(vessel, df_entry, berth, leaving_port)

        if leaving_port:
            for index, (waiting_time, waiting_cause) in enumerate(zip(waiting_times, waiting_causes)):
                waiting_time = waiting_time.total_seconds()
                if waiting_cause == 'Tide':
                    #TODO: vessel needs to go to internal waiting area, or if not available/accessible, wait at berth -> schedule should be updated
                    yield from vessel.wait_for_tidal_window(origin, waiting_time)

        for index, (waiting_time, waiting_cause) in enumerate(zip(waiting_times, waiting_causes)):
            waiting_time = waiting_time.total_seconds()
            if not index:
                waiting_time = yield from self.communicate_vessel_to_sail_to_anchorage(vessel, origin, waiting_time)

            if waiting_cause == berth.name:
                yield from self.communicate_vessel_to_wait_for_berth_availability(vessel, origin, waiting_time)

            elif waiting_cause == 'Tide':
                yield from self.communicate_vessel_to_wait_for_tidal_window(vessel, origin, waiting_time)

        yield from self.communicate_vessel_to_continue_trip_to_terminal()


    def determine_potential_waiting_time_for_vessel(self, vessel, df_entry, berth = None, leaving_port = False):
        waiting_times = []
        waiting_causes = []

        if (not leaving_port and berth is None) or df_entry.empty:
            self.communicate_trip_not_possible(vessel, leaving_port)

        df_entry['Combined'] = df_entry.all(axis=1)
        df_entry['Redundant'] = df_entry['Combined'] == df_entry['Combined'].shift(1)
        df_entry = df_entry[df_entry['Redundant'] == False]
        df_entry = df_entry.drop('Redundant', axis=1)
        df_entry["Reason"] = df_entry.apply(lambda row: row.index[row.eq(False)][0] if any(row.eq(False)) else None, axis=1)
        with pd.option_context("future.no_silent_downcasting", True):
            df_entry = df_entry.ffill()
        waiting_stops_info = df_entry[df_entry.Combined == True]

        waiting_start_time = datetime.datetime.fromtimestamp(vessel.env.now)
        for (waiting_stop_time, waiting_info) in waiting_stops_info.iterrows():
            waiting_times.append(waiting_stop_time - waiting_start_time)
            waiting_causes.append(waiting_info.Reason)
            break

        vessel.port_accessed = self
        return waiting_times, waiting_causes


    def communicate_vessel_to_sail_to_anchorage(self, vessel, origin, waiting_time):
        departure_time_to_anchorage = vessel.env.now
        yield from vessel.sail_to_anchorage(origin)
        arrival_time_at_anchorage = vessel.env.now
        correction_waiting_time = arrival_time_at_anchorage - departure_time_to_anchorage
        waiting_time = waiting_time - correction_waiting_time
        return waiting_time


    def communicate_vessel_to_wait_for_berth_availability(self, vessel, origin, waiting_time):
        yield from vessel.wait_for_berth_availability(origin, waiting_time)


    def communicate_vessel_to_wait_for_tidal_window(self, vessel, origin, waiting_time):
        yield from vessel.wait_for_tidal_window(origin, waiting_time)


    def communicate_vessel_to_continue_trip_to_terminal(self):
        yield from []


    def communicate_trip_not_possible(self, vessel, leaving_port):
        if leaving_port:
            warnings.warn(f"The port is not accessible for the outbound trip of vessel with id: {vessel.id}")
        else:
            warnings.warn(f"The port is not accessible for the inbound trip of vessel with id: {vessel.id}")
        raise simpy.exceptions.Interrupt('Vessel trip not possible.')


    def plan_new_route_for_vessel(self, vessel):
        new_route = None
        origin = vessel.route[-1]
        if vessel.next_destination is not None:
            destination = vessel.next_destination
            new_route = nx.dijkstra_path(vessel.env.graph,origin,destination)
        elif len(vessel.next_terminals):
            next_terminal = vessel.next_terminals[-1]
            vessel.next_terminals = vessel.next_terminals[1:]
            berth = next_terminal.request_terminal_access(vessel, origin)
        return new_route


    def request_terminal_access(self, vessel, origin, df_entry):
        available_berth_time_slots = vessel.terminal.determine_berth_availability(vessel, origin, df_entry)
        berth = vessel.terminal.select_berth_for_vessel(available_berth_time_slots)
        vessel.terminal.assign_berth_to_vessel(vessel, berth)
        return berth


    def get_accessibility_info(self, vessel, leaving_port = False):
        df_tidal_availability = self.get_tidal_availability_info(vessel)
        df_berth_availability = self.get_terminal_availability_info(vessel, leaving_port)

        #Combine the dataframes
        df_entry = pd.DataFrame()
        df_entry['Tide'] = df_tidal_availability['Accessibility'] == 'Accessible'

        df_entry = pd.concat([df_entry,df_berth_availability],axis=1)
        df_entry = df_entry.sort_index()
        with pd.option_context("future.no_silent_downcasting", True):
            df_entry = df_entry.ffill()

        return df_entry


    def get_terminal_availability_info(self, vessel, leaving_port = False):
        df_berth_availability = pd.DataFrame()
        if not leaving_port:
            df_berth_availability = vessel.terminal.provide_terminal_availability_info(vessel)
        return df_berth_availability


    def get_tidal_availability_info(self, vessel):
        tide_bound = check_if_route_contains_restrictions(vessel)
        route = vessel.route
        time_start = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now))
        sailing_time = vessel.determine_sailing_time()
        sailing_time = np.max([pd.Timedelta(seconds=sailing_time), pd.Timedelta(hours=48)])
        time_end = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now) + sailing_time)
        if tide_bound:
            df_tidal_availability = vessel.env.vessel_traffic_service.provide_tidal_windows(vessel, route, time_start, time_end)[0]
        return df_tidal_availability


class IsPortEntry(SimpyObject, OnNode, IsPartofPort):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env.graph.nodes[self.node]['Port Entry'] = self