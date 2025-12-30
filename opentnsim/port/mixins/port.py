from opentnsim.core import SimpyObject, Identifiable, Movable, Log
from opentnsim.graph.mixins import OnNode
from opentnsim.port.utils import get_accessibility_info, update_terminal_planning, determine_vessel_waiting_events, \
    determine_if_vessel_needs_to_sail_to_the_anchorage_area, check_if_route_contains_restrictions, get_vessel_from_id, \
    find_waterways_to_be_passed
from opentnsim.port.visualizations import plot_vessels_over_route, plot_time_distance_diagram
from opentnsim.port.calculations import calculate_total_waiting_time

import datetime
import pandas as pd
import numpy as np
import networkx as nx
import simpy
import warnings
from IPython.display import display

import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

class IsPortComponent:
    def __init__(self, port, *args, **kwargs):
        if not isinstance(port,IsPort):
            raise ValueError("'port' should be an IsPort-object")
        self.port = port
        super().__init__(*args, **kwargs)


class HasPortAccess(Movable, Log):
    def __init__(self, bound, priority = 0, *args, **kwargs):
        self.bound = bound
        self.priority = priority
        self.routes_sailed = []
        self.trip_index = 0
        super().__init__(*args, **kwargs)
        self.on_pass_node_functions.append(self.request_port_entry)
        self.env.vessels.append(self)


    def determine_sailing_time(self):
        route = self.route
        sailing_information = self.env.vessel_traffic_service.provide_sailing_time(self, route)
        sailing_time = sailing_information["Time"].cumsum().values[0]
        return sailing_time


    def request_port_entry(self, origin, at_terminal = False, leaving_port = False, parallel_process = None, process_stop_time = pd.Timestamp('NaT')):
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

        berth = self.select_berth(origin)
        yield from port.communicate_port_accessibility_info(self, origin, berth, leaving_port=leaving_port,
                                                            parallel_process=parallel_process, process_stop_time=process_stop_time)


    def select_berth(self, origin, leaving_port = False):
        berth = None
        if not leaving_port:
            if hasattr(self,'berth'):
                return self.berth
            available_berth_time_slots = self.terminal.determine_berth_availability(self, origin)
            berth = self.terminal.select_berth_for_vessel(available_berth_time_slots)
        return berth


    def request_port_exit(self, origin, parallel_process = None, process_stop_time = pd.Timestamp('NaT')):
        try:
            yield from self.request_port_entry(origin, at_terminal = True, leaving_port = True, parallel_process=parallel_process, process_stop_time = process_stop_time)
        except simpy.Interrupt:
            return

    def plot_time_distance_diagram(self):
        fig = plot_time_distance_diagram(self)
        return fig


class IsPortAuthority:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def plan_vessel_trip(self, vessel, origin, berth=None, leaving_port=False, process_stop_time = pd.Timestamp('NaT')):
        port_availability_df, priority = get_accessibility_info(vessel, origin, berth, leaving_port=leaving_port)
        waiting_events = determine_vessel_waiting_events(self, vessel, port_availability_df)
        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        total_waiting_time = calculate_total_waiting_time(waiting_events)
        if leaving_port and not pd.isna(process_stop_time):
            total_waiting_time = (process_stop_time-current_time).total_seconds()
        passing_waterways = find_waterways_to_be_passed(vessel)
        for waterway in passing_waterways.values():
            waterway.add_vessel_to_passing_vessels(vessel, origin, delay=total_waiting_time, priority=priority)
        return port_availability_df, priority, waiting_events, total_waiting_time


    def replan_vessel_trip(self, vessel, origin, berth=None, leaving_port=False):
        if not leaving_port:
            vessel.terminal.replan_vessels_terminal_berths(vessel)
        port_availability_df, priority, waiting_events, total_waiting_time = self.plan_vessel_trip(vessel, origin, berth, leaving_port)
        return port_availability_df, priority, waiting_events, total_waiting_time


    def communicate_port_accessibility_info(self, vessel, origin, berth = None, leaving_port = False, parallel_process = None, process_stop_time = pd.Timestamp('NaT')):
        port_availability_df, priority, waiting_events, total_waiting_time = self.plan_vessel_trip(vessel, origin, berth, leaving_port, process_stop_time)

        if not parallel_process is None:
            yield from self.communicate_vessel_to_hold_position(vessel, origin, parallel_process,leaving_port=leaving_port, process_stop_time = process_stop_time)

        # if trip is not possible: stop vessel
        if waiting_events is None:
            self.communicate_trip_not_possible(vessel, leaving_port)


        if not leaving_port:
            arrival_time_at_berth = vessel.terminal.assign_vessel_to_berth(vessel, origin, berth, delay=total_waiting_time)
            vessel.terminal.assign_vessel_to_queue(vessel, arrival_time_at_berth, waiting_time=total_waiting_time, berth=berth)

        # if the vessel should wait
        if total_waiting_time:
            vessel.waiting_event = vessel.env.process(self.communicate_vessel_to_wait(vessel, origin, waiting_events, total_waiting_time, berth, leaving_port))
            if not leaving_port: #check if vessel needs to wait in an anchorage area -> move vessel to anchorage area
                required_to_sail_to_anchorage_area = determine_if_vessel_needs_to_sail_to_the_anchorage_area(self.env, vessel, origin, total_waiting_time)
                if required_to_sail_to_anchorage_area:
                    yield from self.communicate_vessel_to_sail_to_anchorage(vessel, origin)

            yield vessel.waiting_event

            if not leaving_port:
                vessel.terminal.remove_vessel_from_queue(vessel)

        # vessel can continue trip
        yield from self.communicate_vessel_to_continue_trip(vessel, origin)


    def communicate_trip_not_possible(self, vessel, leaving_port):
        if leaving_port:
            warnings.warn(f"The port is not accessible for the outbound trip of vessel with id: {vessel.id}")
        else:
            warnings.warn(f"The port is not accessible for the inbound trip of vessel with id: {vessel.id}")
        raise simpy.exceptions.Interrupt('Vessel trip not possible.')


    def communicate_vessel_to_sail_to_anchorage(self, vessel, origin):
        yield from vessel.sail_to_anchorage(origin)


    def communicate_vessel_to_wait(self, vessel, origin, waiting_events, total_waiting_time = 0., berth=None, leaving_port=False):
        interrupted = False
        old_waiting_event_reason = None
        while True:
            for waiting_event_reason, waiting_event_time in waiting_events.items():
                waiting_event_reason = waiting_event_reason.split(' (')[0]
                if interrupted and waiting_event_reason != old_waiting_event_reason:
                    vessel.log_entry_v0(f"Waiting for {old_waiting_event_reason} stop",
                                        vessel.env.now,
                                        vessel.distance,
                                        vessel.env.graph.nodes[origin]["geometry"])

                if not interrupted or waiting_event_reason != old_waiting_event_reason:
                    vessel.log_entry_v0(f"Waiting for {waiting_event_reason} start",
                                        vessel.env.now,
                                        vessel.distance,
                                        vessel.env.graph.nodes[origin]["geometry"])

                elif not leaving_port:
                    arrival_time_at_berth = vessel.terminal.assign_vessel_to_berth(vessel, origin, berth,delay=total_waiting_time)
                    vessel.terminal.update_queue(vessel, arrival_time_at_berth, waiting_time=total_waiting_time,berth=berth)

                interrupted = False
                try:
                    yield vessel.env.timeout(waiting_event_time)
                except simpy.Interrupt:
                    interrupted = True
                    port_availability_df, priority, waiting_events, total_waiting_time = self.replan_vessel_trip(vessel, origin, berth, leaving_port)
                    old_waiting_event_reason = waiting_event_reason
                    break

                vessel.log_entry_v0(f"Waiting for {waiting_event_reason} stop",
                                    vessel.env.now,
                                    vessel.distance,
                                    vessel.env.graph.nodes[origin]["geometry"])

            if not interrupted:
                break


    def communicate_vessel_to_hold_position(self, vessel, origin, parallel_process, leaving_port=False, process_stop_time = pd.Timestamp('NaT')):
        while not parallel_process.processed:
            port_availability_df, _ = get_accessibility_info(vessel, origin, leaving_port=leaving_port)
            port_availability_df['Combined'] = port_availability_df.all(axis=1)
            if pd.isna(process_stop_time):
                process_stop_time = datetime.datetime.fromtimestamp(vessel.env.now)
            current_time = datetime.datetime.fromtimestamp(vessel.env.now)
            future_events = port_availability_df[(port_availability_df.index >= process_stop_time)&(port_availability_df.Combined)]
            waiting_time = pd.Timedelta(seconds=3600.)
            if not future_events.empty:
                future_event = future_events.iloc[0]
                waiting_time = future_event.name - current_time
                if len(future_events) > 1:
                    vessel.berth.update_planning(vessel, new_release_time = future_event.name)
                    update_terminal_planning(vessel, delay = waiting_time.total_seconds())

            yield vessel.env.timeout(waiting_time.total_seconds())


    def communicate_vessel_to_continue_trip(self, vessel, origin):
        vessel.routes_sailed.append(vessel.route)
        passing_waterways = {}
        for node in vessel.route:
            waterway = None
            if "Waterway" in self.env.graph.nodes[node]:
                waterway = self.env.graph.nodes[node]["Waterway"]

            if waterway and waterway.name not in passing_waterways.keys():
                passing_waterways[waterway.name] = waterway
        vessel.trip_index += 1
        # for waterway in passing_waterways.values():
        #     availability_df = waterway.get_waterway_availability_info(vessel, origin)

        yield from []


class IsPort(IsPortAuthority, SimpyObject, Identifiable):
    def __init__(self, port_entry_nodes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.port_entry_nodes = port_entry_nodes
        if not len(self.port_entry_nodes):
            raise ValueError('The port will not be accessible for vessels. It needs port entry_nodes')
        for port_entry_node in self.port_entry_nodes:
            IsPortEntry(env=self.env,node=port_entry_node,port=self)
        self.anchorage_areas = []
        self.terminals = []
        self.env.vessels = []
        if 'ports' not in dir(self.env):
            self.env.ports = []
        self.env.ports.append(self)

    def plot_vessels(self, node_start, node_stop, *args, **kwargs):
        fig = plot_vessels_over_route(self.env, node_start, node_stop, self.env.vessels, *args, **kwargs)
        return fig


class IsPortEntry(SimpyObject, OnNode, IsPortComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env.graph.nodes[self.node]['Port Entry'] = self
