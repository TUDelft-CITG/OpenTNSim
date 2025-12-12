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

        berth = None
        if not leaving_port:
            berth = self.request_terminal_access(origin)
        yield from port.communicate_port_accessibility_info(self, origin, berth, leaving_port=leaving_port)


    def request_terminal_access(self, origin):
        available_berth_time_slots = self.terminal.determine_berth_availability(self, origin)
        berth = self.terminal.select_berth_for_vessel(available_berth_time_slots)
        return berth


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


    def communicate_port_accessibility_info(self, vessel, origin, berth = None, leaving_port = False):
        if leaving_port:
            route = self.plan_new_route_for_vessel(vessel)
            vessel.route = route
            vessel.bound = 'outbound'

        port_availability_df = self.get_accessibility_info(vessel, origin, berth, leaving_port=leaving_port)
        waiting_events = self.determine_vessel_waiting_events(vessel, port_availability_df, leaving_port)

        if leaving_port:
            pass
            # for index, (waiting_time, waiting_cause) in enumerate(zip(waiting_times, waiting_causes)):
            #     waiting_time = waiting_time.total_seconds()
            #     if waiting_cause == 'Tide':
            #         #TODO: vessel needs to go to internal waiting area, or if not available/accessible, wait at berth -> schedule should be updated
            #         yield from self.communicate_vessel_to_wait_for_tidal_window(vessel, origin, waiting_time)

        else:
            #Assign vessel to berth
            vessel.terminal.assign_berth_to_vessel(vessel, origin, berth)

            #Move vessel to the anchorage area if required
            if len(waiting_events):
                total_waiting_time = sum(waiting_events.values())
                required_to_sail_to_anchorage_area = self.determine_if_vessel_needs_to_sail_to_the_anchorage_area(vessel, origin, total_waiting_time)
                if required_to_sail_to_anchorage_area:
                    yield from self.communicate_vessel_to_sail_to_anchorage(vessel, origin)

                yield from self.communicate_vessel_to_wait(vessel, origin, waiting_events)

        yield from self.communicate_vessel_to_continue_trip()


    def communicate_vessel_to_wait(self, vessel, origin, waiting_events):
        waiting_event = self.env.event()
        waiting_event.succeed()

        for waiting_event_reason, waiting_event_time in waiting_events.items():
            vessel.log_entry_v0(f"Waiting for {waiting_event_reason} start",
                                vessel.env.now,
                                vessel.distance,
                                vessel.env.graph.nodes[origin]["geometry"])
            yield vessel.env.timeout(waiting_event_time)
            vessel.log_entry_v0(f"Waiting for {waiting_event_reason} stop",
                                vessel.env.now,
                                vessel.distance,
                                vessel.env.graph.nodes[origin]["geometry"])

        yield waiting_event


    def determine_if_vessel_needs_to_sail_to_the_anchorage_area(self, vessel, origin, waiting_time):
        sail_to_anchorage_area = False
        vessel_traffic_service = self.env.vessel_traffic_service
        nearest_anchorage_area = vessel.find_nearest_anchorage_area(origin)
        route_to_anchorage_area = nx.dijkstra_path(self.env.graph, origin, nearest_anchorage_area.node)
        route_after_anchorage_area = nx.dijkstra_path(self.env.graph, nearest_anchorage_area.node, vessel.route[-1])
        sailing_time_to_terminal = vessel_traffic_service.provide_sailing_time(vessel, vessel.route)["Time"].sum()
        sailing_time_to_anchorage_area = vessel.determine_sailing_time_to_anchorage_area(route_to_anchorage_area)
        new_sailing_time_to_terminal = vessel_traffic_service.provide_sailing_time(vessel, route_after_anchorage_area)["Time"].sum()
        delay_of_sailing_to_terminal = new_sailing_time_to_terminal - sailing_time_to_terminal + sailing_time_to_anchorage_area
        if delay_of_sailing_to_terminal <= waiting_time:
            sail_to_anchorage_area = True
        return sail_to_anchorage_area


    def determine_vessel_waiting_events(self, vessel, port_availability_df, leaving_port = False):
        port_availability_df['Combined'] = port_availability_df.all(axis=1)
        with pd.option_context("future.no_silent_downcasting", True):
            port_availability_df = port_availability_df.ffill()

        def get_waiting_time_reason(lst):
            if not lst:
                return ""  # No False columns
            elif len(lst) == 1:
                return lst[0]
            else:
                return ", ".join(lst[:-1]) + " and " + lst[-1]

        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        previous_events = port_availability_df[port_availability_df.index <= current_time]
        future_events = port_availability_df[port_availability_df.index > current_time]
        last_previous_event_index = previous_events.index.max()
        if pd.isna(last_previous_event_index):
            previous_event = port_availability_df.iloc[0:0]
        else:
            previous_event = port_availability_df.loc[[last_previous_event_index]]
        port_availability_df = pd.concat([previous_event, future_events])
        port_availability_df.index.values[0] = current_time

        cols_to_check = port_availability_df.columns.drop('Combined')
        port_availability_df['Reason'] = port_availability_df[cols_to_check].apply(
            lambda row: get_waiting_time_reason(list(row[row.eq(False)].index)),
            axis=1
        )


        port_available_df = port_availability_df[port_availability_df['Combined'] == True]
        if port_available_df.empty:
            self.communicate_trip_not_possible(vessel, leaving_port)
            return waiting_time

        waiting_time_end = port_available_df.iloc[0].name
        waiting_events = port_availability_df.loc[:waiting_time_end]
        waiting_reasons = waiting_events['Reason'][:-1]
        waiting_times = (waiting_events.index.to_series().shift(-1) - waiting_events.index).apply(lambda x: x.total_seconds())

        waiting_events = {}
        for waiting_reason, waiting_time in zip(waiting_reasons,waiting_times):
            waiting_events[waiting_reason] = waiting_time

        vessel.port_accessed = self
        return waiting_events


    def communicate_vessel_to_sail_to_anchorage(self, vessel, origin):
        yield from vessel.sail_to_anchorage(origin)


    def communicate_vessel_to_wait_for_berth_availability(self, vessel, origin, waiting_time):
        yield from vessel.wait_for_berth_availability(origin, waiting_time)


    def communicate_vessel_to_wait_for_tidal_window(self, vessel, origin, waiting_time):
        yield from vessel.wait_for_tidal_window(origin, waiting_time)


    def communicate_vessel_to_continue_trip(self):
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
            berth = vessel.request_terminal_access(vessel, origin)
        return new_route


    def get_accessibility_info(self, vessel, origin, berth = None, leaving_port = False):
        df_tidal_availability = self.get_tidal_availability_info(vessel)
        df_terminal_availability = self.get_terminal_availability_info(vessel, origin, berth, leaving_port)

        #Combine the dataframes
        port_availability_df = pd.concat([df_tidal_availability,df_terminal_availability],axis=1)
        port_availability_df = port_availability_df.sort_index()
        with pd.option_context("future.no_silent_downcasting", True):
            port_availability_df = port_availability_df.ffill()
            port_availability_df = port_availability_df.ffill()
        return port_availability_df


    def get_terminal_availability_info(self, vessel, origin, berth = None, leaving_port = False):
        df_terminal_availability = pd.DataFrame()
        if not leaving_port:
            df_terminal_availability = vessel.terminal.provide_terminal_availability_info(vessel, origin, berth)
        return df_terminal_availability


    def get_tidal_availability_info(self, vessel):
        tide_bound = check_if_route_contains_restrictions(vessel)
        route = vessel.route
        time_start = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now))
        sailing_time = vessel.determine_sailing_time()
        sailing_time = np.max([pd.Timedelta(seconds=sailing_time), pd.Timedelta(hours=48)])
        time_end = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now) + sailing_time)
        df_tidal_availability = pd.DataFrame(columns=['Accessibility'])
        if tide_bound:
            df_tidal_availability = vessel.env.vessel_traffic_service.provide_tidal_windows(vessel, route, time_start, time_end)[0]
        df_tidal_availability['Tide'] = df_tidal_availability['Accessibility'] == 'Accessible'
        return df_tidal_availability[['Tide']]


class IsPortEntry(SimpyObject, OnNode, IsPartofPort):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env.graph.nodes[self.node]['Port Entry'] = self