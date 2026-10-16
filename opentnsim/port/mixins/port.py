from IPython.display import display

from opentnsim.core import SimpyObject, Identifiable, Movable
from opentnsim.graph.mixins import OnNode
from opentnsim.graph.utils import get_sailing_time, node_path_to_edge_path
from opentnsim.port.utils import (
    get_accessibility_info, 
    determine_vessel_waiting_events,
    determine_if_vessel_needs_to_sail_to_the_anchorage_area, 
    find_waterways_to_be_passed,
    get_oriented_waterway_route,
    check_vessel_is_tide_bound,
    check_berth_can_host_vessel,
    sort_port_trips,
    select_berth_based_on_name,
)

from opentnsim.port.visualizations import plot_vessels_over_route, plot_time_distance_diagram
from opentnsim.port.calculations import calculate_total_waiting_time_of_events

import datetime
import pandas as pd
import simpy
import warnings
import numpy as np
import networkx as nx
import uuid
pd.options.mode.chained_assignment = None

class IsPortComponent:
    def __init__(self, port, *args, **kwargs):
        if not isinstance(port, IsPort):
            raise ValueError("'port' should be an IsPort-object")
        self.port = port
        super().__init__(*args, **kwargs)


class HasPortAccess(Movable, Identifiable):
    def __init__(self, bound, priority = 0, *args, **kwargs):
        self.bound = bound
        self.priority = priority
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
            at_terminal = False, 
            leaving_port = False, 
            parallel_process = None,
            process_stop_time = pd.Timestamp('NaT'),
        ):

        # determine port and prevent vessel from requesting port entry double
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

        # review trip / select berth
        current_time = datetime.datetime.fromtimestamp(self.env.now)
        berth = None
        berth_name = None
        terminal_name = None
        destination = self.route[-1]
        if hasattr(self, 'terminal') and not leaving_port:
            possible_trips = port.find_possible_trips(self, origin)
            trip_possible, berth, is_tide_bound, pilots_required, tugs_required, waiting_time, tidal_window_results, operation_waiting_for = port.select_trip(possible_trips)
            destination = berth.node
            terminal_name = berth.terminal.name
            berth_name = berth.name
        else:
            trip_possible, is_tide_bound, pilots_required, tugs_required, waiting_time, tidal_window_results = port.check_trip(self)
        self.tidal_window_calculations[self.trip_id] = tidal_window_results
        earliest_possible_trip_start = pd.Timestamp(current_time)
        if waiting_time:
            earliest_possible_trip_start += waiting_time
        earliest_possible_trip_start = earliest_possible_trip_start.round('s')

        # kill vessel if it cannot enter the port
        if not trip_possible:
            self.communicate_trip_not_possible(self, leaving_port)
            return

        # add vessel to berth / modify berth planning
        if berth is not None:
            _ = berth.terminal.assign_vessel_to_berth(self, origin, berth, operation_waiting_for, waiting_time.total_seconds())
        elif at_terminal:
            berth_planning = self.berth.terminal.berth_planning
            berth_info = berth_planning[berth_planning.vessel_id == self.id]
            if not berth_info.empty:
                index = berth_info.iloc[-1].name
                berth_planning.loc[index, 'ETD_berth'] = earliest_possible_trip_start

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
                terminal_name,
                berth_name,
                pd.Timestamp(current_time).round('s'), #for now: ETA at port = time at port entry point / berth location
                earliest_possible_trip_start,
                status,
                is_tide_bound,
                pilots_required,
                tugs_required,
            ]
            self.trips.loc[trip_index, :] = trip_info
            port.vessel_trips.loc[port_trip_index, :] = trip_info

        # start port process
        try:
            yield from port.grant_port_access(
                self, 
                origin, 
                berth, 
                leaving_port=leaving_port, 
                parallel_process=parallel_process, 
                process_stop_time=process_stop_time,
            )
        except simpy.exceptions.Interrupt as e:
            raise e

        self.port = port
        self.port_accessed = True
        self.routes_sailed.append(self.route)


    def request_port_exit(
            self, 
            origin, 
            parallel_process = None, 
            process_stop_time = pd.Timestamp('NaT'),
        ):

        try:
            yield from self.request_port_entry(
                origin, 
                at_terminal = True, 
                leaving_port = True, 
                parallel_process=parallel_process, 
                process_stop_time = process_stop_time,
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
        available_berths = terminal.determine_berth_availability(vessel, origin)
        possible_trips = pd.DataFrame(columns = ['berth', 'trip_possible', 'is_tide_bound', 'pilots_required', 'tugs_required', 'waiting_time', 'tidal_window_results', 'operation_waiting_for'])
        for berth_name, berth_info in available_berths.iterrows():
            berth = select_berth_based_on_name(terminal, berth_name)
            destination = berth.node
            route = nx.dijkstra_path(self.env.graph, origin, destination)
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
            berth=None, 
            leaving_port=False):
        trip_info = self.plan_vessel_trip(
            vessel = vessel,
            origin = origin,
            berth = berth,
            leaving_port = leaving_port,
            )
        return trip_info

    def plan_vessel_trip(
            self, 
            vessel, 
            origin, 
            berth=None, 
            leaving_port=False, 
            delay = pd.Timedelta(seconds = 0),
        ):

        waterways_to_be_passed = find_waterways_to_be_passed(vessel)

        port_availability_df_per_waterway, conflicts_dfs = get_accessibility_info(
            vessel, origin, waterways_to_be_passed, berth, leaving_port=leaving_port, delay=delay,
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
            destination = vessel.route[-1]
            if waterway is not None:
                waterway.add_vessel_to_passage_planning(vessel, origin, destination, delay=total_foreseen_waiting_time)
            
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


    def grant_port_access(
        self,
        vessel,
        origin,
        berth=None,
        leaving_port=False,
        parallel_process = None, 
        process_stop_time = pd.Timestamp('NaT')
    ):

        delay = pd.Timedelta(seconds=0)
        if not pd.isna(process_stop_time):
            delay = process_stop_time - datetime.datetime.fromtimestamp(vessel.env.now)

        # optimize: reshuffle planning
        self.vessel_trips, changed_vessel_trips_df = sort_port_trips(self, self.terminals)
        for _, vessel_trip_info in changed_vessel_trips_df.iterrows():
            vessel_to_plan = self.env.vessels[vessel_trip_info.vessel_id]
            if vessel_to_plan.id != vessel.id:
               vessel_to_plan.mission.interrupt('Changed waiting event')
            else:
                trip_info = self.plan_vessel_trip(vessel, origin, berth, leaving_port, delay)
        if changed_vessel_trips_df.empty:
            trip_info = self.plan_vessel_trip(vessel, origin, berth, leaving_port, delay)

        df = trip_info[0]
        waiting_events_per_waterway = trip_info[1]
        total_waiting_time_per_waterway = trip_info[2]
        
        vessel.accessibility_info = df
        if not parallel_process is None:
            yield parallel_process

        # if trip is not possible: stop vessel
        for waterway_name in waiting_events_per_waterway.keys():
            if None in waiting_events_per_waterway[waterway_name].values():
                self.communicate_trip_not_possible(vessel, leaving_port)
                break

        total_waiting_time = sum(total_waiting_time_per_waterway.values())
        for waterway_name in waiting_events_per_waterway.keys():
            total_waiting_time = total_waiting_time_per_waterway[waterway_name]
            waiting_events = waiting_events_per_waterway[waterway_name]
            if waterway_name in self.waterways:
                waterway = self.waterways[waterway_name]
                waterway_route = get_oriented_waterway_route(waterway, vessel)
            else:
                waterway_route = vessel.route
            if total_waiting_time:
                if parallel_process is None:
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
                    return
                else:
                    yield from self.communicate_vessel_to_wait(
                        vessel,
                        vessel.current_node,
                        waiting_events,
                        total_waiting_time=total_waiting_time,
                        berth=berth,
                        leaving_port=leaving_port,
                    )(vessel.current_node)

        self.update_vessel_trip_status(vessel, new_status = 'accepted')
        vessel.terminal.update_vessel_berth_planning_status(vessel, new_status = 'accepted')
        vessel.port_accessed = self


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

            # Only trigger at the correct node
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
                trip_info = self.plan_vessel_trip(vessel, origin, berth, leaving_port)
                waiting_events_per_waterway = trip_info[1]

                first_waterway_name = list(waiting_events_per_waterway.keys())[0]
                current_waiting_events = waiting_events_per_waterway[first_waterway_name]
                if None in current_waiting_events.values():
                    self.communicate_trip_not_possible(vessel, leaving_port)
                    break

            if not leaving_port and hasattr(vessel, "terminal"):
                vessel.terminal.remove_vessel_from_queue(vessel)

            if hasattr(vessel, "route_to_anchorage_area"):
                yield from vessel.pass_anchorage(
                    vessel.route_to_anchorage_area[-1]
                )

            vessel.waiting = False

            self.update_vessel_trip_status(vessel, new_status='accepted')
            vessel.terminal.update_vessel_berth_planning_status(vessel, new_status='accepted')
            vessel.port_accessed = self

        return wait


    def determine_sailing_time_to_port(self, vessel, origin, berth=None):
        if berth is not None:
            destination = berth.node
            route = nx.dijkstra_path(self.env.graph,origin,destination)
            destinations = []
            for node in route:
                if 'Port Entry' in self.env.graph.nodes[node]:
                    break
            destinations = [node]
        else:
            destinations = self.port_entry_nodes

        sailing_time = pd.Timedelta('NaT')
        for destination in destinations:
            route = nx.dijkstra_path(self.env.graph,origin,destination)
            edge_route = node_path_to_edge_path(vessel.env.graph, route)
            sailing_time, _ = get_sailing_time(vessel, edge_route)
            sailing_time = pd.Timedelta(seconds=sailing_time)
        return sailing_time

    # def update_passage_planning(
    #         self, 
    #         waterway,
    #         last_added_vessel,
    #         port_availability_df, 
    #         waiting_events, 
    #         total_waiting_time,
    #         traffic_conflicts_edge,
    #         traffic_conflicts_type,
    #         traffic_conflicts_vessels,
    #         traffic_conflict_rules,
    #         traffic_conflict_downtimes,):
        
    #     df = waterway.passage_planning
    #     vessel_ids_to_pass_waterway = df[~df.is_passing].index.to_list()

    #     if last_added_vessel.id not in vessel_ids_to_pass_waterway:
    #         return (port_availability_df, 
    #             waiting_events, 
    #             total_waiting_time, 
    #             traffic_conflicts_edge, 
    #             traffic_conflicts_type,
    #             traffic_conflicts_vessels,
    #             traffic_conflict_rules,
    #             traffic_conflict_downtimes,)
        
    #     vessel_ids_to_pass_waterway.remove(last_added_vessel.id)
    #     if not len(vessel_ids_to_pass_waterway):
    #         return (port_availability_df, 
    #             waiting_events, 
    #             total_waiting_time, 
    #             traffic_conflicts_edge, 
    #             traffic_conflicts_type,
    #             traffic_conflicts_vessels,
    #             traffic_conflict_rules,
    #             traffic_conflict_downtimes,)
        
    #     priority_last_added_vessel = waterway.priority_rules(last_added_vessel) if waterway.priority_rules else 0
    #     last_added_vessel_pos = df.index.get_loc(last_added_vessel.id)
    #     prior_df = df.iloc[:last_added_vessel_pos]
    #     prior_vessel_priorities_to_pass_waterway = [math.inf]
    #     if not prior_df.empty:
    #         prior_vessel_priorities_to_pass_waterway = prior_df['priority'].to_list()
    #     if np.min(prior_vessel_priorities_to_pass_waterway) >= priority_last_added_vessel:
    #         return (port_availability_df, 
    #             waiting_events, 
    #             total_waiting_time, 
    #             traffic_conflicts_edge, 
    #             traffic_conflicts_type,
    #             traffic_conflicts_vessels,
    #             traffic_conflict_rules,
    #             traffic_conflict_downtimes,)
        
    #     start_time_last_added_vessel = df.loc[last_added_vessel.id, 'time_passage_start']
    #     vessel_stop_times_to_pass_waterway = df.loc[vessel_ids_to_pass_waterway, 'time_passage_stop'].to_list()
    #     if np.min(vessel_stop_times_to_pass_waterway) <= start_time_last_added_vessel:
    #         return (port_availability_df, 
    #             waiting_events, 
    #             total_waiting_time, 
    #             traffic_conflicts_edge, 
    #             traffic_conflicts_type,
    #             traffic_conflicts_vessels,
    #             traffic_conflict_rules,
    #             traffic_conflict_downtimes,)
        
    #     vessel_ids_to_pass_waterway = df[~df.is_passing].index.to_list()
    #     modifiable_planning_df = df.loc[vessel_ids_to_pass_waterway,:]
    #     last_added_vessel_pos = modifiable_planning_df.index.get_loc(last_added_vessel.id)
    #     vessel_selection_df = modifiable_planning_df.iloc[:last_added_vessel_pos].loc[lambda x: x["priority"] < priority_last_added_vessel]
    #     modifiable_planning_df = pd.concat([vessel_selection_df, modifiable_planning_df.loc[[last_added_vessel.id]]])
    #     is_first = modifiable_planning_df.index[0] == last_added_vessel.id
    #     if is_first:
    #         return (port_availability_df, 
    #             waiting_events, 
    #             total_waiting_time, 
    #             traffic_conflicts_edge, 
    #             traffic_conflicts_type,
    #             traffic_conflicts_vessels,
    #             traffic_conflict_rules,
    #             traffic_conflict_downtimes,)

    #     modifiable_planning_df = modifiable_planning_df.sort_values(by=["priority", "time_passage_start"], ascending=[False, True])
    #     waterway.passing_vessels_per_edge = waterway.passing_vessels_per_edge[(~waterway.passing_vessels_per_edge.vessel_id.isin(modifiable_planning_df.index))].sort_values('time_start').reset_index(drop=True)
    #     for vessel_id, _ in modifiable_planning_df.iterrows():
    #         vessel = self.env.vessels[vessel_id]
    #         if vessel_id != last_added_vessel.id:   
    #             if not vessel.waiting and "communicate_vessel_to_wait" in repr(vessel.on_pass_node_functions): 
    #                 vessel.on_pass_node_functions = [cb for cb in vessel.on_pass_node_functions if "communicate_vessel_to_wait" not in repr(cb)]
    #             vessel.port.grant_port_access(vessel, vessel.current_node)
    #             if vessel.waiting:
    #                 vessel.mission.interrupt('Changed waiting event')
    #         else:
    #             (
    #                 port_availability_df_per_waterway, 
    #                 waiting_events_per_waterway, 
    #                 total_waiting_time_per_waterway,
    #                 traffic_conflicts_edge_per_waterway,
    #                 traffic_conflicts_type_per_waterway,
    #                 traffic_conflicts_vessels_per_waterway,
    #                 traffic_conflict_rules_per_waterway,
    #                 traffic_conflict_downtimes_per_waterway
    #                 ) = vessel.port.replan_vessel_trip(vessel, vessel.current_node)
    #             port_availability_df = port_availability_df_per_waterway[waterway.name]
    #             waiting_events = waiting_events_per_waterway[waterway.name]
    #             total_waiting_time = total_waiting_time_per_waterway[waterway.name]
    #             traffic_conflicts_edge = traffic_conflicts_edge_per_waterway[waterway.name]
    #             traffic_conflicts_type = traffic_conflicts_type_per_waterway[waterway.name]
    #             traffic_conflicts_vessels = traffic_conflicts_vessels_per_waterway[waterway.name]
    #             traffic_conflict_rules = traffic_conflict_rules_per_waterway[waterway.name]
    #             traffic_conflict_downtimes = traffic_conflict_downtimes_per_waterway[waterway.name]
    #             waterway.passage_planning.sort_values('priority', ascending=False, inplace=True)

    #     waterway.passage_planning.sort_values(by=['tme_of_registration','priority', 'time_passage_start'], ascending=[True, False, True], inplace=True)
    #     return (port_availability_df, 
    #             waiting_events, 
    #             total_waiting_time, 
    #             traffic_conflicts_edge, 
    #             traffic_conflicts_type,
    #             traffic_conflicts_vessels,
    #             traffic_conflict_rules,
    #             traffic_conflict_downtimes)
    
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
