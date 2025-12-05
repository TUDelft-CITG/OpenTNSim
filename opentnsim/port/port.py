from opentnsim.core import SimpyObject, Identifiable, Movable
from opentnsim.graph import OnNode
from opentnsim.tidal_accessibility import check_if_route_contains_restrictions

import datetime
import pandas as pd
import numpy as np

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
        self.on_pass_node_functions.append(self.request_port_access)


    def determine_sailing_time(self):
        route = self.route
        sailing_information = self.env.vessel_traffic_service.provide_sailing_time(self, route)
        sailing_time = sailing_information["Time"].cumsum().values[0]
        return sailing_time


    def request_port_passage(self, origin):
        port = self.terminal.port
        yield from self.request_port_access(origin,port)

        self.env.process(self.move())
        raise simpy.exceptions.Interrupt('Route of vessel has changed.')


    def get_port_access_info(self, origin):
        tide_bound = check_if_route_contains_restrictions(self)
        route = self.route
        time_start = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))
        sailing_time = self.determine_sailing_time()
        sailing_time = np.max([pd.Timedelta(seconds=sailing_time),pd.Timedelta(hours=48)])
        time_end = np.datetime64(datetime.datetime.fromtimestamp(self.env.now) + sailing_time)
        if tide_bound:
            df_entry_times = self.env.vessel_traffic_service.provide_tidal_windows(self, route, time_start, time_end, plot=True)[0]

        df_entry = pd.DataFrame()
        df_entry['Tide'] = df_entry_times['Accessibility'] == 'Accessible'
        df_berth_availability = self.terminal.determine_terminal_availability(vessel=self)
        df_entry = pd.concat([df_entry,df_berth_availability],axis=1)
        df_entry = df_entry.sort_index()
        with pd.option_context("future.no_silent_downcasting", True):
            df_entry = df_entry.ffill()

        #Terminal makes decision -> selected_berth, entry_time
        berth, berth_name, waiting_times, waiting_causes = self.terminal.request_terminal_access(vessel=self, origin=origin, df_entry=df_entry)
        return  berth, berth_name, waiting_times, waiting_causes


    def request_port_access(self, origin, berth = None, berth_name = None, waiting_times = None, waiting_causes = None):
        # Request for a terminal
        if berth is None:
            if 'Port Entry' not in self.env.graph.nodes[origin].keys():
                return

            port = self.env.graph.nodes[origin]['Port Entry'].port
            if not self.terminal.port == port:
                return
            elif 'port_accessed' in dir(self) and self.port_accessed == port:
                return

            berth, berth_name, waiting_times, waiting_causes = self.get_port_access_info(origin)

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


class IsPortEntry(SimpyObject, OnNode, IsPartofPort):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env.graph.nodes[self.node]['Port Entry'] = self