from opentnsim.core import SimpyObject, Identifiable, Movable
from opentnsim.graph.mixins import OnNode
from opentnsim.graph.utils import get_sailing_time, node_path_to_edge_path, get_shortest_route
from opentnsim.port.calculations import (
    calculate_total_waiting_time_of_events, 
    calculate_estimated_arrival_and_departure_times_at_port_and_berth,
)
from opentnsim.port.utils import (
    get_accessibility_info, 
    determine_vessel_waiting_events,
    determine_if_vessel_needs_to_sail_to_the_anchorage_area, 
    find_waterways_to_be_passed,
    get_waterway_orientation,
    check_vessel_is_tide_bound,
    check_berth_can_host_vessel,
    sort_port_trips,
    find_berth_based_on_name,
    determine_nearest_anchorage_area,
)
from opentnsim.port.visualizations import plot_vessels_over_route, plot_time_distance_diagram

import datetime
import pandas as pd
import simpy
import warnings
import numpy as np
import uuid
pd.options.mode.chained_assignment = None

class IsPortComponent:
    def __init__(self, port, *args, **kwargs):
        if not isinstance(port, IsPort):
            raise ValueError("'port' should be an IsPort-object")
        self.port = port
        super().__init__(*args, **kwargs)


class HasPortAccess(Movable, Identifiable):
    def __init__(self, bound, *args, **kwargs):
        self.bound = bound
        self.routes_sailed = []
        self.trip_index = 0
        self.trip_id = str(uuid.uuid4())
        super().__init__(*args, **kwargs)
        self.on_pass_node_functions.append(self.request_port_entry)
        self.env.vessels[self.id] = self
        self.waiting = False
        self.accessibility_info = pd.DataFrame()
        self.trips = pd.DataFrame(
            columns = [
                'trip_id',
                'vessel_id',
                'vessel_name',
                'vessel_L', 
                'vessel_B', 
                'vessel_T', 
                'current_node',
                'origin', 
                'destination', 
                'terminal',
                'berth', 
                'time_arrival_at_port',
                'time_trip_start',
                'status',
                'tide_bound', 
                'pilots_required', 
                'tugs_required',
            ]
        )
        self.waiting_df = pd.DataFrame(
            columns = [
                'time_start', 
                'time_stop', 
                'reason', 
                'conflict_edge', 
                'conflict_type', 
                'conflict_vessels', 
                'conflict_rule', 
                'conflict_downtime',
            ]
        )

    def request_port_entry(
            self, 
            origin, 
            parallel_process = None,
            delay = pd.Timedelta(seconds = 0),
        ):
        
        # set current time
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))

        # set default parameters
        berth = None
        terminal = None

        # set destination
        destination = self.route[-1]

        # get infrastructure and determine if vessel has entered the port and has already been assigned a berth
        infrastructure_at_node = self.env.graph.nodes[origin]
        vessel_sailing_to_anchorage_area = hasattr(self, 'anchorage_area_request')
        vessel_entered_port = hasattr(self, 'port')
        vessel_assigned_to_berth = hasattr(self, 'berth')

        # determine port and prevent vessel from requesting port when not at port entry node or at berth
        vessel_at_berth = False
        if 'Port Entry' in infrastructure_at_node:
            port = infrastructure_at_node['Port Entry'].port
            if (vessel_entered_port and self.port == port) or (vessel_sailing_to_anchorage_area and self.anchorage_area_request.anchorage_area in port.anchorage_areas.values()):
                return
        elif 'Berth' in infrastructure_at_node and vessel_assigned_to_berth:
            terminal = self.berth.terminal
            port = self.berth.terminal.port
            vessel_at_berth = True
            if parallel_process is None:
                return
        else:
            return

        # establish and check vessel trip
        if not vessel_at_berth:
            # if vessel at entry point find a trip to a berth
            possible_trips = port.find_possible_trips(self, origin)
            trip_possible, berth, is_tide_bound, pilots_required, tugs_required, waiting_time, tidal_window_results, operation_waiting_for = port.select_trip(possible_trips)
            
            # if berth could not be found -> port not accessible
            if berth is None:
                port.communicate_trip_not_possible(self, vessel_at_berth)
                return
            # else set destination and terminal
            destination = berth.node
            terminal = berth.terminal  
        else:
            start_time = current_time + pd.Timedelta(seconds = self.loading_time*3600 + self.deberthing_time*60)
            trip_possible, is_tide_bound, pilots_required, tugs_required, waiting_time, tidal_window_results = port.check_trip(self, start_time = start_time)
            waiting_time += (start_time - current_time)

        # add tidal window calculation to vessel (for quick access, preventing recalculation)
        self.tidal_window_calculations[self.trip_id] = tidal_window_results

        # if trip is not possible -> port not accessible
        if not trip_possible:
            port.communicate_trip_not_possible(self, vessel_at_berth)
            return
        
        # set trip start time
        possible_trip_start_time = (current_time + waiting_time).round('s')

        # add vessel to berth / update berth planning
        if berth is not None:
            # calculate estimated time of arrival at port and berth, and estimated time of departure from berth
            eta_port, eta_berth, etd_berth = calculate_estimated_arrival_and_departure_times_at_port_and_berth(
                self, origin, berth, waiting_time)
            berth.terminal.add_vessel_to_berth_planning(self, berth, eta_port, eta_berth, etd_berth, operation_waiting_for)
        elif vessel_at_berth:
            self.berth.terminal.update_berth_planning(self, origin, possible_trip_start_time)

        # add vessel to the port planning
        status = 'planned'
        if self.trip_id not in port.vessel_trips.trip_id.values:
            trip_index = len(self.trips)
            port_trip_index = len(port.vessel_trips)
            trip_info = [
                self.trip_id,
                self.id,
                self.name,
                self.L,
                self.B,
                self.T,
                origin,
                origin, #for now: origin (within port) = origin (port entry point / berth location)
                destination,
                terminal.name if terminal else None,
                berth.name if berth else None,
                current_time.round('s'), #for now: ETA at port = time at port entry point / berth location
                possible_trip_start_time,
                status,
                is_tide_bound,
                pilots_required,
                tugs_required,
            ]
            self.trips.loc[trip_index, :] = trip_info
            port.vessel_trips.loc[port_trip_index, :] = trip_info

        # start port process
        try:
            yield from port.grant_vessel_port_access(
                self, 
                origin, 
                terminal,
                berth, 
                parallel_process, 
                delay,
            )
        except simpy.exceptions.Interrupt as e:
            raise e


    def request_port_exit(
            self, 
            origin, 
            parallel_process = None,
            delay = pd.Timedelta(seconds = 0)
        ):

        try:
            yield from self.request_port_entry(
                origin, 
                parallel_process,
                delay,
            )
        except simpy.Interrupt:
            return
        

    def plot_time_distance_diagram(self, route):
        fig = plot_time_distance_diagram(self, route)
        return fig


class IsPortAuthority:
    def __init__(self, *args, **kwargs):
        self.vessel_trips = pd.DataFrame(
            columns = [
                'trip_id',
                'vessel_id',
                'vessel_name',
                'vessel_L', 
                'vessel_B', 
                'vessel_T', 
                'current_node',
                'origin', 
                'destination', 
                'terminal',
                'berth', 
                'time_arrival_at_port',
                'time_trip_start',
                'status',
                'tide_bound', 
                'pilots_required', 
                'tugs_required',
            ]
        )
        super().__init__(*args, **kwargs)


    def check_trip(self, vessel, berth=None, route = None, start_time = pd.Timestamp('NaT')):
        # check berth
        berth_can_host_vessel = True
        if berth is not None:
            berth_can_host_vessel = check_berth_can_host_vessel(berth, vessel)

        # check environmental conditions
        vessel_can_reach_berth, is_tide_bound, tidal_waiting_time, tidal_window_results = check_vessel_is_tide_bound(vessel, route, start_time)

        # check pilot assistence
        pilots_required = 0 #[int] for now: no pilots

        # check tug assistence
        tugs_required = 0 #[int] for now: no tugs

        trip_possible = (berth_can_host_vessel and vessel_can_reach_berth)

        return trip_possible, is_tide_bound, pilots_required, tugs_required, tidal_waiting_time, tidal_window_results


    def find_possible_trips(self, vessel, origin, time_start = pd.Timestamp('NaT')):
        if pd.isna(time_start):
            time_start = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        terminal = vessel.terminal
        availability_info_per_berth = terminal.provide_availability_info_per_berth(vessel, origin)
        possible_trips = pd.DataFrame(columns = ['berth', 'trip_possible', 'is_tide_bound', 'pilots_required', 'tugs_required', 'waiting_time', 'tidal_window_results', 'operation_waiting_for'])
        for berth_name, berth_info in availability_info_per_berth.iterrows():
            berth = find_berth_based_on_name(terminal, berth_name)
            destination = berth.node
            route = get_shortest_route(self.env.graph, origin, destination, vessel)
            start_time = time_start + berth_info.waiting_time
            trip_possible, is_tide_bound, pilots_required, tugs_required, waiting_time, tidal_window_results = self.check_trip(vessel, berth, route, start_time)
            waiting_time += berth_info.waiting_time
            operation_waiting_for = berth_info.operation_waiting_for
            possible_trips.loc[berth_name,:] = [berth, trip_possible, is_tide_bound, pilots_required, tugs_required, waiting_time, tidal_window_results, operation_waiting_for]
        return possible_trips


    def select_trip(self, possible_trips):
        possible_trips = possible_trips[possible_trips.trip_possible]
        trip_possible = True
        is_tide_bound = False
        pilots_required = 0
        tugs_required = 0
        waiting_time = pd.Timedelta(seconds=0)
        tidal_window_results = None
        berth = None
        operation_waiting_for = None
        if not possible_trips.empty:
            trip_possible = True
            berth_with_least_waiting_time = possible_trips.sort_values(by="waiting_time").iloc[0]
            berth = berth_with_least_waiting_time.berth
            is_tide_bound = berth_with_least_waiting_time["is_tide_bound"]
            pilots_required = berth_with_least_waiting_time["pilots_required"]
            tugs_required = berth_with_least_waiting_time["tugs_required"]
            waiting_time = berth_with_least_waiting_time["waiting_time"]
            tidal_window_results = berth_with_least_waiting_time["tidal_window_results"]
            operation_waiting_for = berth_with_least_waiting_time["operation_waiting_for"]
        return trip_possible, berth, is_tide_bound, pilots_required, tugs_required, waiting_time, tidal_window_results, operation_waiting_for

    
    def replan_vessel_trip(
            self, 
            vessel, 
            origin, 
            terminal,
            berth=None,
            ):
        trip_info = self.plan_vessel_trip(
            vessel = vessel,
            origin = origin,
            terminal = terminal,
            berth = berth,
            )
        return trip_info


    def plan_vessel_trip(
            self, 
            vessel, 
            origin, 
            terminal,
            berth=None, 
            delay = pd.Timedelta(seconds = 0),
        ):

        waterways_to_be_passed = find_waterways_to_be_passed(vessel)

        port_availability_df_per_waterway, conflicts_dfs = get_accessibility_info(
            vessel, origin, waterways_to_be_passed, terminal, berth, delay=delay,
        )

        waiting_events_per_waterway = {}
        total_waiting_time_per_waterway = {}
        total_foreseen_waiting_time = delay.total_seconds()
        for index, waterway_info in waterways_to_be_passed.iterrows():
            conflict_df = conflicts_dfs[index]
            for col in conflict_df.columns:
                conflict_df[col] = conflict_df[col].apply(lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x)
            conflict_df = conflict_df.dropna(how="all")
            waterway = waterway_info.waterway
            waterway_name = waterway_info.waterway_name
            sailing_time_to_waterway = waterway_info.sailing_time_to_waterway
            delay = sailing_time_to_waterway.total_seconds() + total_foreseen_waiting_time
            port_availability_df = port_availability_df_per_waterway[waterway_name]
            waiting_events = determine_vessel_waiting_events(self, port_availability_df, delay)

            total_waiting_time = 0
            if None in waiting_events.values():
                port_availability_df_per_waterway = {}
                port_availability_df_per_waterway[waterway_name]  = port_availability_df
                waiting_events_per_waterway = {}
                waiting_events_per_waterway[waterway_name] = waiting_events
                break

            total_waiting_time = calculate_total_waiting_time_of_events(waiting_events)
            total_foreseen_waiting_time += total_waiting_time
            if waterway is not None:
                waterway.add_vessel_to_passage_planning(
                    vessel, 
                    origin, 
                    delay=pd.Timedelta(seconds=total_foreseen_waiting_time),
                )
            
            # store per waterway results
            port_availability_df_per_waterway[waterway_name] = port_availability_df
            waiting_events_per_waterway[waterway_name] = waiting_events
            total_waiting_time_per_waterway[waterway_name] = total_waiting_time
      
        trip_info = (
            port_availability_df_per_waterway,
            waiting_events_per_waterway,
            total_waiting_time_per_waterway,
        )
        return trip_info


    def grant_vessel_port_access(
        self,
        vessel,
        origin,
        terminal,
        berth=None,
        parallel_process = None, 
        delay = pd.Timedelta(seconds=0),
    ):
        at_berth = False
        if isinstance(parallel_process, simpy.events.Timeout):
            at_berth = True
            delay += pd.Timedelta(seconds = parallel_process._delay)

        # optimize: reshuffle planning
        self.vessel_trips, changed_vessel_trips_df = sort_port_trips(self, self.terminals)
        for _, vessel_trip_info in changed_vessel_trips_df.iterrows():
            vessel_to_plan = self.env.vessels[vessel_trip_info.vessel_id]
            if vessel_to_plan.trip_id != vessel.trip_id:
               vessel_to_plan.mission.interrupt('Changed waiting event')
            else:
                trip_info = self.plan_vessel_trip(vessel, origin, terminal, berth, delay)
        
        if changed_vessel_trips_df.empty:
            trip_info = self.plan_vessel_trip(vessel, origin, terminal, berth, delay)

        df = trip_info[0]
        waiting_events_per_waterway = trip_info[1]
        total_waiting_time_per_waterway = trip_info[2]
        
        vessel.accessibility_info = df
        if not parallel_process is None:
            yield parallel_process

        # if trip is not possible: stop vessel
        for waterway_name in waiting_events_per_waterway.keys():
            if None in waiting_events_per_waterway[waterway_name].values():
                self.communicate_trip_not_possible(vessel, at_berth)
                break

        total_waiting_time = sum(total_waiting_time_per_waterway.values())
        for waterway_name in waiting_events_per_waterway.keys():
            total_waiting_time = total_waiting_time_per_waterway[waterway_name]
            waiting_events = waiting_events_per_waterway[waterway_name]
            if waterway_name in self.waterways:
                waterway = self.waterways[waterway_name]
                _, waterway_route, _ = get_waterway_orientation(waterway, vessel)
            else:
                waterway_route = vessel.route
            if total_waiting_time:
                if parallel_process is None:
                    yield from self.communicate_vessel_to_wait(
                        vessel = vessel,
                        waiting_node = waterway_route[0],
                        waiting_events = waiting_events,
                        terminal = terminal,
                        berth = berth,
                        at_berth = at_berth,
                        total_waiting_time = total_waiting_time,
                    )(vessel.current_node)
                    return
                else:
                    yield from self.communicate_vessel_to_wait(
                        vessel,
                        vessel.current_node,
                        waiting_events,
                        terminal = terminal,
                        berth=berth,
                        at_berth = at_berth,
                        total_waiting_time=total_waiting_time,
                    )(vessel.current_node)

        self.update_vessel_trip_status(vessel, new_status = 'accepted')
        vessel.terminal.update_vessel_berth_planning_status(vessel, new_status = 'accepted')
        vessel.port = self
        vessel.port_accessed = True
        vessel.routes_sailed.append(vessel.route)


    def communicate_trip_not_possible(self, vessel, at_berth):
        if at_berth:
            warnings.warn(f"The port is not accessible for the outbound trip of vessel with id: {vessel.id}")
        else:
            warnings.warn(f"The port is not accessible for the inbound trip of vessel with id: {vessel.id}")
        raise simpy.exceptions.Interrupt('Vessel trip not possible.')


    def communicate_vessel_to_sail_to_anchorage(self, vessel, origin, anchorage_area):
        try:
            yield from vessel.sail_to_anchorage(origin, anchorage_area)
        except simpy.exceptions.Interrupt as e:
            raise e

    def communicate_vessel_to_wait(
        self,
        vessel,
        waiting_node,
        waiting_events,
        terminal,
        berth=None,
        at_berth = False,
        total_waiting_time=0.,
    ):
        def wait(node):

            # Only trigger at the correct node
            if node != waiting_node:
                return

            origin = node
            if not at_berth:
                nearest_anchorage_area = determine_nearest_anchorage_area(vessel, origin, self)
                required_to_sail_to_anchorage_area = (
                    determine_if_vessel_needs_to_sail_to_the_anchorage_area(
                        vessel,
                        origin,
                        nearest_anchorage_area,
                        total_waiting_time
                    )
                )

                if required_to_sail_to_anchorage_area:
                    try:
                        yield from self.communicate_vessel_to_sail_to_anchorage(
                            vessel,
                            origin,
                            nearest_anchorage_area,
                        )
                    except simpy.Interrupt as e:
                        pass

            # Keep the original input arguments unchanged
            current_waiting_events = waiting_events

            while True:
                vessel.waiting = True
                interrupted = False

                for reason, wait_time in current_waiting_events.items():
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

                    try:
                        yield vessel.env.timeout(
                            np.max([wait_time, 0.])
                        )

                    except simpy.exceptions.Interrupt:
                        interrupted = True
                        time_stop = vessel.env.now

                        vessel.log_entry_v0(
                            f"Waiting for {clean_reason} stop",
                            time_stop,
                            vessel.distance,
                            vessel.env.graph.nodes[origin]["geometry"]
                        )

                        vessel.waiting_df.loc[
                            waiting_index, 'time_stop'
                        ] = datetime.datetime.fromtimestamp(time_stop)

                        break

                    time_stop = vessel.env.now

                    vessel.log_entry_v0(
                        f"Waiting for {clean_reason} stop",
                        time_stop,
                        vessel.distance,
                        vessel.env.graph.nodes[origin]["geometry"]
                    )

                    vessel.waiting_df.loc[
                        waiting_index, 'time_stop'
                    ] = datetime.datetime.fromtimestamp(time_stop)

                if not interrupted:
                    break

                # Recalculate the waiting conditions after an interruption
                trip_info = self.plan_vessel_trip(vessel, origin, terminal, berth)
                waiting_events_per_waterway = trip_info[1]

                first_waterway_name = list(waiting_events_per_waterway.keys())[0]
                current_waiting_events = waiting_events_per_waterway[first_waterway_name]
                if None in current_waiting_events.values():
                    self.communicate_trip_not_possible(vessel, at_berth)
                    break

            new_sailing_event_has_started = False
            if hasattr(vessel, "route_to_anchorage_area"):
                try:
                    yield from vessel.pass_anchorage(
                        vessel.route_to_anchorage_area[-1]
                    )
                except simpy.exceptions.Interrupt:
                    new_sailing_event_has_started = True
                    pass

            vessel.waiting = False

            self.update_vessel_trip_status(vessel, new_status='accepted')
            vessel.terminal.update_vessel_berth_planning_status(vessel, new_status='accepted')
            vessel.port = self
            vessel.port_accessed = True
            if not at_berth:
                vessel.routes_sailed.append(vessel.route)
            if new_sailing_event_has_started:
                raise simpy.Interrupt('New sailing event has started')

        return wait


    def determine_sailing_time_to_port(self, vessel, origin, berth=None):
        if berth is not None:
            destination = berth.node
            route = get_shortest_route(self.env.graph, origin, destination, vessel)
            destinations = []
            for node in route:
                if 'Port Entry' in self.env.graph.nodes[node]:
                    break
            destinations = [node]
        else:
            destinations = self.port_entry_nodes

        sailing_time = pd.Timedelta('NaT')
        for destination in destinations:
            route = get_shortest_route(self.env.graph, origin, destination, vessel)
            edge_route = node_path_to_edge_path(vessel.env.graph, route)
            sailing_time, _ = get_sailing_time(vessel, edge_route)
            sailing_time = pd.Timedelta(seconds=sailing_time)
        return sailing_time

    
    def update_vessel_trip_status(self, vessel, new_status):
        trip_info = self.vessel_trips[self.vessel_trips.trip_id == vessel.trip_id]
        vessel.trips.loc[vessel.trip_index, 'status'] = new_status
        if not trip_info.empty:
            index = trip_info.iloc[-1].name
            self.vessel_trips.loc[index, 'status'] = new_status


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
        fig = plot_vessels_over_route(self.env, node_start, node_stop, *args, **kwargs)
        return fig


class IsPortEntry(OnNode, IsPortComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env.graph.nodes[self.node]['Port Entry'] = self
