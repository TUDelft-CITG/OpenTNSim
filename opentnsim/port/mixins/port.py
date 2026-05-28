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
import networkx as nx
pd.options.mode.chained_assignment = None
8
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

    def request_port_entry(self, origin, at_terminal = False, leaving_port = False):
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
        
        port.communicate_port_accessibility_info(
            self, 
            origin, 
            berth, 
            leaving_port=leaving_port,
        )
        self.port_accessed = True
        yield from []


    def select_berth(self, origin, leaving_port = False):
        berth = None
        if not leaving_port:
            if hasattr(self,'berth'):
                return self.berth
            available_berth_time_slots = self.terminal.determine_berth_availability(self, origin)
            berth = self.terminal.select_berth_for_vessel(available_berth_time_slots)
        return berth


    def request_port_exit(self, origin):
        try:
            yield from self.request_port_entry(origin, at_terminal = True, leaving_port = True)
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
        )

    def plan_vessel_trip(
            self, 
            vessel, 
            origin, 
            berth=None, 
            leaving_port=False, 
        ):
        passing_waterways = find_waterways_to_be_passed(vessel)
        port_availability_df_per_waterway = get_accessibility_info(
            vessel, origin, berth, leaving_port=leaving_port,
        )

        waiting_events_per_waterway = {}
        total_waiting_time_per_waterway = {}
        total_foreseen_waiting_time = 0.
        for waterway_name, waterway in passing_waterways.items():
            waterway_start_node = get_oriented_waterway_route(waterway, vessel)[0]
            waterway_route_start_index = vessel.route.index(waterway_start_node)
            current_time_start_index = vessel.position_on_route
            route_to_waterway = vessel.route[current_time_start_index:(waterway_route_start_index+1)]
            edge_route = list(zip(route_to_waterway[:-1],route_to_waterway[1:]))
            sailing_time_to_waterway, _ = get_sailing_time(vessel, edge_route)
            delay = sailing_time_to_waterway + total_foreseen_waiting_time
            port_availability_df = port_availability_df_per_waterway[waterway_name]
            waiting_events = determine_vessel_waiting_events(
                self, vessel, port_availability_df, delay
            )

            total_waiting_time = calculate_total_waiting_time(waiting_events)

            waterway.add_vessel_to_passing_vessels(
                vessel, origin, delay=total_waiting_time
            )
            
            port_availability_df, waiting_events, total_waiting_time = (
                waterway.update_passing_vessels_planning(
                    vessel,
                    port_availability_df,
                    waiting_events,
                    total_waiting_time,
                )
            )

            # store per waterway results
            port_availability_df_per_waterway[waterway_name] = port_availability_df
            waiting_events_per_waterway[waterway_name] = waiting_events
            total_waiting_time_per_waterway[waterway_name] = total_waiting_time
            total_foreseen_waiting_time += total_waiting_time
        return (
            port_availability_df_per_waterway,
            waiting_events_per_waterway,
            total_waiting_time_per_waterway,
        )


    def communicate_port_accessibility_info(
        self,
        vessel,
        origin,
        berth=None,
        leaving_port=False,
    ):
        df, waiting_events_per_waterway, total_waiting_time_per_waterway = (
            self.plan_vessel_trip(
                vessel, origin, berth, leaving_port
            )
        )

        # if trip is not possible: stop vessel
        for _, waiting_events in waiting_events_per_waterway.items():
            if waiting_events is None:
                self.communicate_trip_not_possible(vessel, leaving_port)

        total_waiting_time = sum(total_waiting_time_per_waterway.values())

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
        for waterway_name in waiting_events_per_waterway.keys():
            total_waiting_time = total_waiting_time_per_waterway[waterway_name]
            waiting_events = waiting_events_per_waterway[waterway_name]
            waterway = self.waterways[waterway_name]
            waterway_route = get_oriented_waterway_route(waterway, vessel)
            if total_waiting_time:
                vessel.on_pass_node_functions.append(
                    self.communicate_vessel_to_wait(
                        vessel = vessel,
                        waiting_node = waterway_route[0],
                        waiting_events = waiting_events,
                        total_waiting_time = total_waiting_time,
                        berth = berth,
                        leaving_port = leaving_port,
                    )
                )
        vessel.accessibility_info = df


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
                for reason, wait_time in waiting_events.items():

                    clean_reason = reason.split(" (")[0]

                    vessel.log_entry_v0(
                        f"Waiting for {clean_reason} start",
                        vessel.env.now,
                        vessel.distance,
                        vessel.env.graph.nodes[origin]["geometry"]
                    )

                    try:
                        yield vessel.env.timeout(wait_time)
                    except simpy.exceptions.Interrupt as e:
                        vessel.log_entry_v0(
                            f"Waiting for {clean_reason} stop",
                            vessel.env.now,
                            vessel.distance,
                            vessel.env.graph.nodes[origin]["geometry"]
                        )
                        break

                    vessel.log_entry_v0(
                        f"Waiting for {clean_reason} stop",
                        vessel.env.now,
                        vessel.distance,
                        vessel.env.graph.nodes[origin]["geometry"]
                    )

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
        fig = plot_vessels_over_route(self.env, node_start, node_stop, self.env.vessels, *args, **kwargs)
        return fig


class IsPortEntry(OnNode, IsPortComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env.graph.nodes[self.node]['Port Entry'] = self
