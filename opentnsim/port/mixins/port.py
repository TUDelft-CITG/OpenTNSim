from opentnsim.core import SimpyObject, Identifiable, Movable
from opentnsim.graph.mixins import OnNode
from opentnsim.graph.utils import get_sailing_time
from opentnsim.port.utils import (
    get_accessibility_info, 
    update_terminal_planning, 
    determine_vessel_waiting_events,
    determine_if_vessel_needs_to_sail_to_the_anchorage_area, 
    find_waterways_to_be_passed,
    get_oriented_waterway_route
)
from opentnsim.port.visualizations import plot_vessels_over_route, plot_time_distance_diagram
from opentnsim.port.calculations import calculate_total_waiting_time

import datetime
import pandas as pd
import simpy
import warnings
import numpy as np
pd.options.mode.chained_assignment = None

class IsPortComponent:
    def __init__(self, port, *args, **kwargs):
        if not isinstance(port,IsPort):
            raise ValueError("'port' should be an IsPort-object")
        self.port = port
        super().__init__(*args, **kwargs)


class HasPortAccess(Movable, Identifiable):
    def __init__(self, bound, priority = 0, *args, **kwargs):
        self.bound = bound
        self.priority = priority
        self.routes_sailed = []
        self.trip_index = 0
        super().__init__(*args, **kwargs)
        self.on_pass_node_functions.append(self.request_port_entry)
        self.env.vessels[self.id] = self
        self.waiting = False
        self.accessibility_info = pd.DataFrame()
        self.waiting_df = pd.DataFrame(columns = ['time_start', 'time_stop', 'reason', 'conflict_edge', 'conflict_type', 'conflict_vessels', 'conflict_rule', 'conflict_downtime'])

    def request_port_entry(self, origin, at_terminal = False, leaving_port = False, parallel_process = None, process_stop_time = pd.Timestamp('NaT')):
        if not at_terminal:
            if hasattr(self,'port_accessed') and self.port_accessed:
                return
            
            if 'Port Entry' in self.env.graph.nodes[origin].keys():
                port = self.env.graph.nodes[origin]['Port Entry'].port
            elif 'Anchorage' in self.env.graph.nodes[origin].keys():
                port = self.env.graph.nodes[origin]['Anchorage'][0].port
            else:
                return
            
        else:
            port = self.terminal.port

        self.port = port
        berth = None
        if hasattr(self, 'terminal'):
            berth = self.select_berth(origin)
        if parallel_process is not None and not pd.isna(process_stop_time):
            try:
                yield from port.communicate_port_accessibility_info(self, origin, berth, leaving_port=leaving_port,
                                                                    parallel_process=parallel_process, process_stop_time=process_stop_time)
            except simpy.exceptions.Interrupt as e:
                raise e
        else:
            yield from port.communicate_port_accessibility_info(
                self, 
                origin, 
                berth, 
                leaving_port=leaving_port,
            )
            self.port_accessed = True
        self.routes_sailed.append(self.route)


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

    def plot_time_distance_diagram(self, route):
        fig = plot_time_distance_diagram(self, route)
        return fig


class IsPortAuthority:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def replan_vessel_trip(
            self, 
            vessel, 
            origin, 
            berth=None, 
            leaving_port=False):

        (
            port_availability_df_per_waterway,
            waiting_events_per_waterway,
            total_waiting_time_per_waterway,
            traffic_conflicts_edge_per_waterway,
            traffic_conflicts_type_per_waterway,
            traffic_conflicts_vessels_per_waterway,
            traffic_conflicts_rules_per_waterway,
            traffic_conflicts_downtimes_per_waterway,
            ) = self.plan_vessel_trip(
                vessel = vessel,
                origin = origin,
                berth = berth,
                leaving_port = leaving_port,
                )
        return (
            port_availability_df_per_waterway,
            waiting_events_per_waterway,
            total_waiting_time_per_waterway,
            traffic_conflicts_edge_per_waterway,
            traffic_conflicts_type_per_waterway,
            traffic_conflicts_vessels_per_waterway,
            traffic_conflicts_rules_per_waterway,
            traffic_conflicts_downtimes_per_waterway,
        )

    def plan_vessel_trip(
            self, 
            vessel, 
            origin, 
            berth=None, 
            leaving_port=False, 
        ):
        passing_waterways = find_waterways_to_be_passed(vessel)
        port_availability_df_per_waterway, conflicts_dfs = get_accessibility_info(
            vessel, origin, berth, leaving_port=leaving_port,
        )

        waiting_events_per_waterway = {}
        total_waiting_time_per_waterway = {}
        traffic_conflicts_edge_per_waterway = {}
        traffic_conflicts_type_per_waterway = {}
        traffic_conflicts_vessels_per_waterway = {}
        traffic_rules_type_per_waterway = {}
        traffic_downtimes_vessels_per_waterway = {}
        total_foreseen_waiting_time = 0.
        if len(passing_waterways):
            for index, (waterway_name, waterway) in enumerate(passing_waterways.items()):
                conflict_df = conflicts_dfs[index]
                for col in conflict_df.columns:
                    conflict_df[col] = conflict_df[col].apply(lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x)
                conflict_df = conflict_df.dropna(how="all")

                waterway_start_node = get_oriented_waterway_route(waterway, vessel)[0]
                waterway_route_start_index = vessel.route.index(waterway_start_node)
                current_time_start_index = vessel.position_on_route
                route_to_waterway = vessel.route[current_time_start_index:(waterway_route_start_index+1)]
                edge_route = list(zip(route_to_waterway[:-1],route_to_waterway[1:]))
                sailing_time_to_waterway, _ = get_sailing_time(vessel, edge_route)

                current_time = datetime.datetime.fromtimestamp(self.env.now)
                last_message = pd.DataFrame(vessel.logbook).iloc[-1] if len(vessel.logbook) > 0 else None
                if last_message is not None and 'Sailing' in last_message.Message:
                    start_time_sailing_on_current_node = last_message.Timestamp
                    sailing_time_on_current_edge = current_time - start_time_sailing_on_current_node
                    sailing_time_to_waterway -= sailing_time_on_current_edge.total_seconds()

                delay = sailing_time_to_waterway + total_foreseen_waiting_time
                port_availability_df = port_availability_df_per_waterway[waterway_name]
                waiting_events, conflict_edges, conflicts_type, vessels_in_conflict, rules, downtimes = determine_vessel_waiting_events(
                    self, vessel, port_availability_df, conflict_df, delay
                )

                total_waiting_time = calculate_total_waiting_time(waiting_events)
                total_foreseen_waiting_time += total_waiting_time
                waterway.add_vessel_to_passing_vessels(
                    vessel, origin, delay=total_foreseen_waiting_time
                )
                
                (port_availability_df, 
                waiting_events, 
                total_waiting_time,
                conflict_edges,
                conflicts_type,
                vessels_in_conflict,
                rules,
                downtimes) = (
                    waterway.update_passing_vessels_planning(
                        vessel,
                        port_availability_df,
                        waiting_events,
                        total_waiting_time,
                        conflict_edges,
                        conflicts_type,
                        vessels_in_conflict,
                        rules,
                        downtimes,
                    )
                )

                # store per waterway results
                port_availability_df_per_waterway[waterway_name] = port_availability_df
                waiting_events_per_waterway[waterway_name] = waiting_events
                traffic_conflicts_edge_per_waterway[waterway_name] = conflict_edges
                traffic_conflicts_type_per_waterway[waterway_name] = conflicts_type
                traffic_conflicts_vessels_per_waterway[waterway_name] = vessels_in_conflict
                total_waiting_time_per_waterway[waterway_name] = total_waiting_time
                traffic_rules_type_per_waterway[waterway_name] = rules
                traffic_downtimes_vessels_per_waterway[waterway_name] = downtimes
        else:
            delay = 0.
            port_availability_df = port_availability_df_per_waterway
            conflict_df = pd.DataFrame()
            waiting_events, conflict_edges, conflicts_type, vessels_in_conflict, rules, downtimes = determine_vessel_waiting_events(
                self, vessel, port_availability_df, conflict_df, delay
            )

            total_waiting_time = calculate_total_waiting_time(waiting_events)
            total_foreseen_waiting_time += total_waiting_time
            
            # store per waterway results
            port_availability_df_per_waterway = port_availability_df
            waiting_events_per_waterway = waiting_events
            traffic_conflicts_edge_per_waterway = conflict_edges
            traffic_conflicts_type_per_waterway = conflicts_type
            traffic_conflicts_vessels_per_waterway = vessels_in_conflict
            total_waiting_time_per_waterway= total_waiting_time
            traffic_rules_type_per_waterway = rules
            traffic_downtimes_vessels_per_waterway = downtimes

        return (
            port_availability_df_per_waterway,
            waiting_events_per_waterway,
            total_waiting_time_per_waterway,
            traffic_conflicts_edge_per_waterway,
            traffic_conflicts_type_per_waterway,
            traffic_conflicts_vessels_per_waterway,
            traffic_rules_type_per_waterway,
            traffic_downtimes_vessels_per_waterway
        )


    def communicate_port_accessibility_info(
        self,
        vessel,
        origin,
        berth=None,
        leaving_port=False,
        parallel_process = None, 
        process_stop_time = pd.Timestamp('NaT')
    ):

        (df, 
         waiting_events_per_waterway, 
         total_waiting_time_per_waterway,
         traffic_conflicts_edge_per_waterway,
         traffic_conflicts_type_per_waterway,
         traffic_conflicts_vessels_per_waterway,
         traffic_conflicts_rules_per_waterway,
         traffic_conflicts_downtimes_per_waterway,)  = (
            self.plan_vessel_trip(
                vessel, origin, berth, leaving_port
            )
        )
        if not parallel_process is None:
            try:
                yield from self.communicate_vessel_to_hold_position(vessel, origin, parallel_process,leaving_port=leaving_port, process_stop_time = process_stop_time)
            except simpy.exceptions.Interrupt as e:
                raise e  

        # if trip is not possible: stop vessel
        for _, waiting_events in waiting_events_per_waterway.items():
            if waiting_events is None:
                self.communicate_trip_not_possible(vessel, leaving_port)

        through_waterway = False
        try:
            total_waiting_time = sum(total_waiting_time_per_waterway.values())
            through_waterway = True
        except:
            total_waiting_time = total_waiting_time_per_waterway

        if not leaving_port and berth is not None:
            arrival_time_at_berth = vessel.terminal.assign_vessel_to_berth(
                vessel, origin, berth, delay=total_waiting_time
            )
            vessel.terminal.assign_vessel_to_queue(
                vessel,
                arrival_time_at_berth,
                waiting_time=total_waiting_time,
                berth=berth,
            )

        # if the vessel should wait
        if through_waterway:
            for waterway_name in waiting_events_per_waterway.keys():
                total_waiting_time = total_waiting_time_per_waterway[waterway_name]
                waiting_events = waiting_events_per_waterway[waterway_name]
                traffic_conflicts_edge = traffic_conflicts_edge_per_waterway[waterway_name]
                traffic_conflicts_type = traffic_conflicts_type_per_waterway[waterway_name]
                traffic_conflicts_vessels = traffic_conflicts_vessels_per_waterway[waterway_name]
                traffic_conflicts_rules = traffic_conflicts_rules_per_waterway[waterway_name]
                traffic_conflicts_downtimes = traffic_conflicts_downtimes_per_waterway[waterway_name]
                waterway = self.waterways[waterway_name]
                waterway_route = get_oriented_waterway_route(waterway, vessel)
                if total_waiting_time:
                    vessel.on_pass_node_functions.append(
                        self.communicate_vessel_to_wait(
                            vessel = vessel,
                            waiting_node = waterway_route[0],
                            waiting_events = waiting_events,
                            conflict_edges = traffic_conflicts_edge,
                            conflict_types = traffic_conflicts_type,
                            vessels_in_conflict = traffic_conflicts_vessels,
                            conflict_rules = traffic_conflicts_rules,
                            conflict_downtimes = traffic_conflicts_downtimes,
                            total_waiting_time = total_waiting_time,
                            berth = berth,
                            leaving_port = leaving_port,
                        )
                    )
        elif total_waiting_time:
            import string
            waiting_events = waiting_events_per_waterway
            n = len(waiting_events.keys())
            vessel.on_pass_node_functions.append(
                self.communicate_vessel_to_wait(
                    vessel = vessel,
                    waiting_node = origin,
                    waiting_events = waiting_events,
                    conflict_edges = dict.fromkeys(string.ascii_lowercase[:n], np.nan),
                    conflict_types = dict.fromkeys(string.ascii_lowercase[:n], np.nan),
                    vessels_in_conflict = dict.fromkeys(string.ascii_lowercase[:n], np.nan),
                    conflict_rules = dict.fromkeys(string.ascii_lowercase[:n], np.nan),
                    conflict_downtimes = dict.fromkeys(string.ascii_lowercase[:n], np.nan),
                    total_waiting_time = total_waiting_time,
                    berth = berth,
                    leaving_port = leaving_port,
                )
            )
        vessel.accessibility_info = df

    def communicate_vessel_to_hold_position(self, vessel, origin, parallel_process, leaving_port=False, process_stop_time = pd.Timestamp('NaT')):
        while not parallel_process.processed:
            port_availability_df, _ = get_accessibility_info(vessel, origin, leaving_port=leaving_port)
            port_availability_df['Combined'] = port_availability_df.all(axis=1)
            if pd.isna(process_stop_time):
                process_stop_time = datetime.datetime.fromtimestamp(vessel.env.now)
            current_time = datetime.datetime.fromtimestamp(vessel.env.now)
            future_events = port_availability_df[
                (port_availability_df.index >= process_stop_time)&(port_availability_df.Combined)
                ]
            waiting_time = pd.Timedelta(seconds=3600.)
            if not future_events.empty:
                future_event = future_events.iloc[0]
                waiting_time = future_event.name - current_time
                if len(future_events) > 1:
                    vessel.berth.update_planning(vessel, new_release_time = future_event.name)
                    update_terminal_planning(vessel, delay = waiting_time.total_seconds())
            try:
                yield vessel.env.timeout(waiting_time.total_seconds())
            except simpy.exceptions.Interrupt as e:
                raise e

    def communicate_trip_not_possible(self, vessel, leaving_port):
        if leaving_port:
            warnings.warn(f"The port is not accessible for the outbound trip of vessel with id: {vessel.id}")
        else:
            warnings.warn(f"The port is not accessible for the inbound trip of vessel with id: {vessel.id}")
        raise simpy.exceptions.Interrupt('Vessel trip not possible.')


    def communicate_vessel_to_sail_to_anchorage(self, vessel, origin):
        try:
            yield from vessel.sail_to_anchorage(origin)
        except simpy.exceptions.Interrupt as e:
            raise e


    def communicate_vessel_to_wait(
        self,
        vessel,
        waiting_node,
        waiting_events,
        conflict_edges,
        conflict_types,
        conflict_rules,
        conflict_downtimes,
        vessels_in_conflict,
        total_waiting_time=0.,
        berth=None,
        leaving_port=False
    ):
        def wait(node):

            # only trigger at the correct node
            if node != waiting_node:
                return

            origin = node

            if not leaving_port:

                required_to_sail_to_anchorage_area = (
                    determine_if_vessel_needs_to_sail_to_the_anchorage_area(
                        self.env,
                        vessel,
                        origin,
                        total_waiting_time
                    )
                )

                if required_to_sail_to_anchorage_area:
                    yield from self.communicate_vessel_to_sail_to_anchorage(
                        vessel,
                        origin
                    )

            if not leaving_port and hasattr(vessel, "terminal"):

                arrival_time_at_berth = vessel.terminal.assign_vessel_to_berth(
                    vessel,
                    origin,
                    berth,
                    delay=total_waiting_time
                )

                vessel.terminal.update_queue(
                    vessel,
                    arrival_time_at_berth,
                    waiting_time=total_waiting_time,
                    berth=berth
                )

            while True:
                vessel.waiting = True
                for ((reason, wait_time), conflict_edge, conflict_type, conflict_vessels, conflict_rule, conflict_downtime) in (
                    zip(waiting_events.items(), conflict_edges.values(), conflict_types.values(), vessels_in_conflict.values(), conflict_rules.values() ,conflict_downtimes.values(),)):
                    clean_reason = reason.split(" (")[0]
                    time_start = vessel.env.now

                    vessel.log_entry_v0(
                        f"Waiting for {clean_reason} start",
                        time_start,
                        vessel.distance,
                        vessel.env.graph.nodes[origin]["geometry"]
                    )

                    waiting_index = len(vessel.waiting_df)
                    vessel.waiting_df.loc[waiting_index, 'time_start'] = datetime.datetime.fromtimestamp(time_start)
                    vessel.waiting_df.loc[waiting_index, 'reason'] = clean_reason
                    vessel.waiting_df.loc[waiting_index, 'conflict_edge'] = conflict_edge
                    vessel.waiting_df.loc[waiting_index, 'conflict_type'] = conflict_type
                    vessel.waiting_df.loc[waiting_index, 'conflict_vessels'] = conflict_vessels
                    vessel.waiting_df.loc[waiting_index, 'conflict_rule'] = conflict_rule
                    vessel.waiting_df.loc[waiting_index, 'conflict_downtime'] = conflict_downtime

                    try:
                        yield vessel.env.timeout(np.max([wait_time,0.]))
                    except simpy.exceptions.Interrupt as e:
                        time_stop = vessel.env.now
                        vessel.log_entry_v0(
                            f"Waiting for {clean_reason} stop",
                            time_stop,
                            vessel.distance,
                            vessel.env.graph.nodes[origin]["geometry"]
                        )
                        vessel.waiting_df.loc[waiting_index, 'time_stop'] = datetime.datetime.fromtimestamp(time_stop)
                        break

                    time_stop = vessel.env.now
                    vessel.log_entry_v0(
                        f"Waiting for {clean_reason} stop",
                        time_stop,
                        vessel.distance,
                        vessel.env.graph.nodes[origin]["geometry"]
                    )
                    vessel.waiting_df.loc[waiting_index, 'time_stop'] = datetime.datetime.fromtimestamp(time_stop)

                break

            if not leaving_port and hasattr(vessel, "terminal"):
                vessel.terminal.remove_vessel_from_queue(vessel)

            if hasattr(vessel, "route_to_anchorage_area"):
                yield from vessel.pass_anchorage(
                    vessel.route_to_anchorage_area[-1]
                )
            vessel.waiting = False

        return wait


class IsPort(IsPortAuthority, SimpyObject, Identifiable):
    def __init__(self, port_entry_nodes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.port_entry_nodes = port_entry_nodes
        if not len(self.port_entry_nodes):
            raise ValueError('The port will not be accessible for vessels. It needs port entry_nodes')
        for port_entry_node in self.port_entry_nodes:
            IsPortEntry(env=self.env,node=port_entry_node,port=self)
        self.anchorage_areas = {}
        self.terminals = {}
        self.waterways = {}
        self.env.vessels = {}
        if 'ports' not in dir(self.env):
            self.env.ports = []
        self.env.ports.append(self)

    def plot_vessels(self, node_start, node_stop, *args, **kwargs):
        fig = plot_vessels_over_route(self.env, node_start, node_stop, self.env.vessels.values(), *args, **kwargs)
        return fig


class IsPortEntry(OnNode, IsPortComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env.graph.nodes[self.node]['Port Entry'] = self
