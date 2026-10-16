from opentnsim.core import  Identifiable, Log, Movable, PriorityFilterStore, VesselProperties
from opentnsim.graph.utils import get_sailing_time, node_path_to_edge_path
from opentnsim.port.mixins.port import IsPortComponent
from opentnsim.port.mixins.berth import IsQuay, IsJetty, IsBerth
from opentnsim.port.utils import determine_new_route_for_vessel, select_berth_based_on_name

import numpy as np
import pandas as pd
import datetime
import networkx as nx
import simpy
import uuid

class TerminalHandable(Movable, Identifiable, VesselProperties):
    def __init__(self,
                 terminal,
                 berthing_time,
                 loading_time,
                 deberthing_time,
                 next_destination = None,
                 next_terminals = [],
                 next_berthing_times=[],
                 next_loading_times = [],
                 next_deberthing_times = [],
                 *args,
                 **kwargs):
        self.terminal = terminal
        self.berthing_time = berthing_time
        self.loading_time = loading_time
        self.deberthing_time = deberthing_time
        if next_destination is None and not len(next_terminals):
            raise ValueError('The vessel has no destination after the terminal.')
        self.next_destination = next_destination
        self.next_berthing_times = next_berthing_times
        self.next_terminals = next_terminals
        self.next_loading_times = next_loading_times
        self.next_deberthing_times = next_deberthing_times
        self.berth_operation_id = None
        super().__init__(*args, **kwargs)
        self.on_complete_pass_edge_functions.append(self.pass_terminal)


    def pass_terminal(self, destination):
        if 'Berth' not in self.env.graph.nodes[destination].keys():
            return

        berths = self.env.graph.nodes[destination]['Berth']
        if not self.berth in berths:
            return

        self.port_accessed = False
        yield from self.request_berth_access()
        yield from self.berthing(destination)
        negotiate_port_exit = yield from self.loading(destination)
        yield negotiate_port_exit
        yield from self.deberthing(destination)
        yield from self.release_berth_access(destination)
        self.env.process(self.move())
        raise simpy.exceptions.Interrupt('Route of vessel has changed.')

    
    def request_berth_access(self):
        yield from self.berth.request_access(self)
        berth_planning = self.berth.terminal.berth_planning
        berthing_info = berth_planning[berth_planning.operation_id == self.berth_operation_id]
        if not berthing_info.empty:
            index = berthing_info.iloc[-1].name
            berth_planning.loc[index, 'status'] = 'at berth'


    def release_berth_access(self, origin):
        yield from self.berth.release_access(self)
        self.berth.remove_vessel_from_occupants_df(self)
        berth_planning = self.berth.terminal.berth_planning
        berthing_info = berth_planning[berth_planning.vessel_id == self.id]
        if not berthing_info.empty:
            index = berthing_info.iloc[-1].name
            berth_planning.loc[index, 'status'] = 'completed'


    def berthing(self, origin):
        self.log_entry_v0("Berthing start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        berthing_process = self.env.timeout(self.berthing_time*60)
        current_time = datetime.datetime.fromtimestamp(self.env.now)
        berthing_stop_time = current_time + pd.Timedelta(seconds=self.berthing_time*60)
        while True:
            try:
                yield berthing_process
                break
            except simpy.Interrupt:
                current_time = datetime.datetime.fromtimestamp(self.env.now)
                remaining_berthing_time = berthing_stop_time - current_time
                berthing_process = self.env.timeout(remaining_berthing_time.total_seconds())
        self.log_entry_v0("Berthing stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        if len(self.next_berthing_times):
            self.berthing_time = self.next_berthing_times[0]
            self.next_berthing_times = self.next_berthing_times[1:]
        self.trip_index += 1
        self.trip_id = str(uuid.uuid4())


    def loading(self, origin):
        self.log_entry_v0("Loading start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        new_route = determine_new_route_for_vessel(self)
        self.route = new_route
        self.edge_route = node_path_to_edge_path(self.env.graph, new_route)
        self.bound = 'outbound'
        current_time = datetime.datetime.fromtimestamp(self.env.now)
        loading_time = pd.Timedelta(seconds=self.loading_time*3600)
        loading_stop_time = current_time + loading_time
        time_at_berth = loading_time + pd.Timedelta(seconds=self.deberthing_time*60)
        time_at_berth_stop = current_time + time_at_berth
        loading_process = self.env.timeout(self.loading_time*3600)
        request_port_exit = self.request_port_exit(
            origin, 
            parallel_process = loading_process, 
            process_stop_time = time_at_berth_stop
            )

        negotiate_port_exit = self.env.process(request_port_exit)
        while True:
            try:
                yield loading_process
                break
            except simpy.Interrupt:
                current_time = datetime.datetime.fromtimestamp(self.env.now)
                remaining_loading_time = loading_stop_time - current_time
                loading_process = self.env.timeout(remaining_loading_time.total_seconds())
                
        self.log_entry_v0("Loading stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        if len(self.next_loading_times):
            self.loading_time = self.next_loading_times[0]
            self.next_loading_times = self.next_loading_times[1:]

        return negotiate_port_exit


    def deberthing(self, origin):
        self.log_entry_v0("Deberthing start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        deberthing_process = self.env.timeout(self.deberthing_time*60)
        current_time = datetime.datetime.fromtimestamp(self.env.now)
        deberthing_stop_time = current_time + pd.Timedelta(seconds=self.deberthing_time*60)
        while True:
            try:
                yield deberthing_process
                break
            except simpy.Interrupt:
                current_time = datetime.datetime.fromtimestamp(self.env.now)
                remaining_deberthing_time = deberthing_stop_time - current_time
                deberthing_process = self.env.timeout(remaining_deberthing_time.total_seconds())
        self.log_entry_v0("Deberthing stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        if len(self.next_deberthing_times):
            self.deberthing_time = self.next_deberthing_times[0]
            self.next_deberthing_times = self.next_deberthing_times[1:]


class HasBerthPlanning:

    def __init__(self, *args, **kwargs):
        berth_names = [berth.name for berth in self.berths.items]
        berth_capacities = [
            berth.berth_length if isinstance(berth, IsQuay) else 1 for berth in self.berths.items
            ]
        self.berth_availability = pd.DataFrame(columns=berth_names)
        self.berth_availability.loc[pd.Timestamp(self.env.simulation_start)] = berth_capacities
        self.berth_availability.loc[pd.Timestamp(self.env.simulation_stop)] = berth_capacities
        quay_berths = [berth.name for berth in self.berths.items if isinstance(berth, IsQuay)]
        self.berth_availability[quay_berths] = self.berth_availability[quay_berths].astype(float)
        jetty_berths = [berth.name for berth in self.berths.items if isinstance(berth, IsJetty)]
        self.berth_availability[jetty_berths] = self.berth_availability[jetty_berths].astype(int)
        self.berth_planning = pd.DataFrame(
            columns=[
                'operation_id', 
                'vessel_id', 
                'vessel_name', 
                'vessel_L', 
                'vessel_B', 
                'vessel_T', 
                'berth', 
                'ETA_port', 
                'ETA_berth', 
                'ETD_berth', 
                'priority',
                'status',
                'operation_waiting_for'
                ]
            )
        super().__init__(*args, **kwargs)


class IsTerminal(Log, Identifiable, HasBerthPlanning, IsPortComponent):

    def __init__(self, env, berths, *args,**kwargs):
        """ Creates a terminal

        Input
        -----
        berths : a list of IsBerth, IsJetty and/or IsQuay objects
        """
        self.env = env
        self.berths = PriorityFilterStore(env=env)
        for berth in berths:
            berth.terminal = self
            self.berths.put(berth)
        super().__init__(env=env, *args, **kwargs)
        self.port.terminals[self.name] = self
        self.queue = pd.DataFrame(
            columns=[
                "Vessel_L",
                "Vessel_B",
                "Vessel_T",
                "Berth",
                "Waiting_start_time",
                "Waiting_stop_time",
                "Arrival_time_at_berth"
                ]
            )


    def assign_vessel_to_queue(self, vessel, arrival_time_at_berth, waiting_time=0.,berth=None):
        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        waiting_stop = current_time + pd.Timedelta(seconds=waiting_time)
        berth_name = None
        if isinstance(berth, IsBerth):
            berth_name = berth.name
            berth.add_vessel_to_queue(vessel, current_time, waiting_stop, arrival_time_at_berth)
        self.queue.loc[vessel.id] = [
            vessel.L, vessel.B, vessel.T, berth_name, current_time, waiting_stop, arrival_time_at_berth
            ]


    def add_vessel_to_berth_planning(
            self, vessel, berth, eta_port, eta_berth, etd_berth, operation_waiting_for = None
            ):
        berth_availability = self.berth_availability.copy()
        if isinstance(berth, IsQuay):
            berth_availability.loc[eta_berth, berth.name] = np.nan
            berth_availability.loc[etd_berth, berth.name] = np.nan
        elif isinstance(berth, IsJetty):
            berth_availability.loc[eta_berth, berth.name] = 0
            berth_availability.loc[etd_berth, berth.name] = 1
            mask = (berth_availability.index > eta_berth)&(berth_availability.index < etd_berth)
            berth_availability.loc[berth_availability[mask].index, berth.name] = 0
            berth_availability[berth.name] = berth_availability[berth.name].astype(int)

        berth_availability = berth_availability.sort_index()
        with pd.option_context("future.no_silent_downcasting", True):
            berth_availability = berth_availability.ffill().infer_objects(copy=False)
            berth_availability = berth_availability.bfill().infer_objects(copy=False)

        if isinstance(berth, IsQuay):
            mask = (berth_availability.index >= eta_berth)&(berth_availability.index < etd_berth)
            berth_availability.loc[berth_availability[mask].index, berth.name] -= vessel.L

        berth_priority = 0 # for now: all vessel priorities are equal
        status = 'planned'
        index = len(self.berth_planning)
        vessel.berth_operation_id = str(uuid.uuid4())
        self.berth_planning.loc[index, :] = [
            vessel.berth_operation_id, 
            vessel.id, 
            vessel.name, 
            vessel.L, 
            vessel.B, 
            vessel.T, 
            berth.name, 
            eta_port, 
            eta_berth, 
            etd_berth, 
            berth_priority, 
            status, 
            operation_waiting_for
            ]
        self.berth_availability = berth_availability


    def assign_vessel_to_berth(self, vessel, origin, berth, operation_waiting_for = None, delay = 0.):
        waiting_time = pd.Timedelta(seconds = delay)
        vessel.berth = berth
        sailing_time_to_port = self.port.determine_sailing_time_to_port(vessel, origin, berth)
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
        eta_port = current_time + sailing_time_to_port
        sailing_time_to_berth = self.determine_sailing_time_to_berth(vessel, origin, berth)
        eta_berth = current_time + sailing_time_to_berth + waiting_time
        time_at_berth = vessel.berthing_time*60 + vessel.loading_time*3600 + vessel.deberthing_time*60
        etd_berth = eta_berth + pd.Timedelta(seconds=time_at_berth)
        eta_port, eta_berth, etd_berth = eta_port.round('s'), eta_berth.round('s'), etd_berth.round('s')
        self.add_vessel_to_berth_planning(vessel, berth, eta_port, eta_berth, etd_berth, operation_waiting_for)
        berth.add_vessel_to_occupants_df(vessel, eta_berth, etd_berth)
        return eta_berth


    def determine_berth_availability(self, vessel, origin, berth = None, delay = 0.):
        berthing_time = vessel.berthing_time*60
        loading_time = vessel.loading_time*3600
        deberthing_time = vessel.deberthing_time*60
        occupation_duration = pd.Timedelta(seconds=berthing_time + loading_time + deberthing_time)
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now)) + pd.Timedelta(seconds = delay)
        potential_berths = self.provide_berth_availability_info(vessel, berth)
        df_berth_time_slot = pd.DataFrame(columns=[
            'time_start','time_stop','waiting_time','berth_length','operation_waiting_for'
            ])
        berths = [berth]
        if berth is None:
            berths =  self.berths.items
        for berth in berths:
            berth_name = berth.name
            if berth_name not in potential_berths.columns:
                continue

            berth_available = potential_berths[berth_name]
            mask_berth_available = (berth_available == True)
            berth_availability_start_times = berth_available[mask_berth_available].index.to_list()
            if not len(berth_availability_start_times):
                 continue

            for berth_availability_start_time in berth_availability_start_times:
                berth_availability_stop_times = berth_available[
                    (berth_available.index > berth_availability_start_time)&(berth_available == False)]
                if not len(berth_availability_stop_times):
                    berth_availability_stop_time = berth_available.index[-1]
                else:
                    berth_availability_stop_time = berth_availability_stop_times.index[0]

                sailing_time_to_berth = self.determine_sailing_time_to_berth(vessel, origin, berth)
                actual_berth_availability_start_time = np.max([
                    berth_availability_start_time, current_time + sailing_time_to_berth])
                operation_waiting_for = None
                operation_waiting_for_df = self.berth_planning[
                    self.berth_planning.ETD_berth == actual_berth_availability_start_time]
                if not operation_waiting_for_df.empty:
                    operation_waiting_for = operation_waiting_for_df.iloc[0].operation_id
                berth_availability_duration = berth_availability_stop_time - actual_berth_availability_start_time
                if berth_availability_duration < occupation_duration:
                    continue

                waiting_time = berth_availability_start_time - current_time - sailing_time_to_berth
                waiting_time = np.max([pd.Timedelta(seconds=0), waiting_time])
                if waiting_time: # prevent overlap in planning
                    berth_availability_start_time += pd.Timedelta(seconds = 1)
                    berth_availability_stop_time += pd.Timedelta(seconds = 1)
                    waiting_time += pd.Timedelta(seconds = 1) 
                df_berth_time_slot.loc[berth_name,:] = [
                    berth_availability_start_time,
                    berth_availability_stop_time,
                    waiting_time,
                    berth.berth_length,
                    operation_waiting_for]
                break
        return df_berth_time_slot


    def determine_sailing_time_to_berth(self, vessel, origin, berth):
        destination = berth.node
        route = nx.dijkstra_path(self.env.graph,origin,destination)
        edge_route = node_path_to_edge_path(vessel.env.graph, route)
        sailing_time, _ = get_sailing_time(vessel, edge_route)
        sailing_time = pd.Timedelta(seconds=sailing_time)
        return sailing_time


    def determine_potential_berth_availability(self, df_potential_available_berths, vessel):
        df_availability = pd.DataFrame()
        for berth_name in df_potential_available_berths.columns:
            berth = select_berth_based_on_name(self, berth_name)
            if isinstance(berth, IsQuay):
                berth_available = (df_potential_available_berths[berth_name] >= vessel.L)
            else:
                berth_available = (df_potential_available_berths[berth_name] > 0)
            df_availability[berth_name] = berth_available
        return df_availability


    def determine_potential_available_berths(self, vessel, berth):
        berth_availability = self.berth_availability.copy()
        if vessel.berth_operation_id in self.berth_planning.operation_id.values:
            index = self.berth_planning[self.berth_planning.operation_id == vessel.berth_operation_id].iloc[-1].name
            eta_berth = (self.berth_planning.loc[index, 'ETA_berth']).round('s')
            assigned_berth_name = self.berth_planning.loc[index, 'berth']
            assigned_berth = select_berth_based_on_name(self, assigned_berth_name)
            if isinstance(assigned_berth, IsQuay):
                berth_availability.loc[eta_berth, assigned_berth_name] += vessel.L
            elif isinstance(assigned_berth, IsJetty):
                berth_availability.loc[eta_berth, assigned_berth_name] += 1

        terminal_berths = self.berths.items
        if berth is not None:
            terminal_berths = [berth]
        fit_berths_names = [
            berth.name for berth in terminal_berths if (berth.depth >= vessel.T) and (berth.berth_length >= vessel.L)]
        current_time = datetime.datetime.fromtimestamp(self.env.now)
        fit_berth_availability = berth_availability[fit_berths_names]
        previous_events = fit_berth_availability[fit_berth_availability.index <= current_time]
        future_events = fit_berth_availability[fit_berth_availability.index > current_time]
        last_previous_event_index = previous_events.index.max()

        if pd.isna(last_previous_event_index):
            previous_event = fit_berth_availability.iloc[0:0]
        else:
            previous_event = fit_berth_availability.loc[[last_previous_event_index]]

        fit_berth_availability = pd.concat([previous_event, future_events])
        return fit_berth_availability


    def provide_berth_availability_info(self, vessel, berth):
        berth_availability = self.determine_potential_available_berths(vessel, berth)
        df_berth_availability = self.determine_potential_berth_availability(berth_availability, vessel)
        return df_berth_availability


    def provide_terminal_availability_info(self, vessel, origin, berth = None, delay = pd.Timedelta(seconds=0)):
        df_berth_availability = self.provide_berth_availability_info(vessel, berth)
        df_terminal_availability = pd.DataFrame()
        sailing_time_to_berth = pd.Timedelta(seconds=0.) + delay
        if berth is None:
            df_terminal_availability['Terminal'] = df_berth_availability.any(axis=1)
        else:
            sailing_time_to_berth = self.determine_sailing_time_to_berth(vessel, origin, berth)
            sailing_time_to_berth = (sailing_time_to_berth + delay).round('s')
            df_terminal_availability['Terminal'] = df_berth_availability[berth.name]
        if not df_terminal_availability.empty:
            df_terminal_availability.index -= sailing_time_to_berth
        else:
            df_terminal_availability.loc[pd.Timestamp(self.env.simulation_start),'Terminal'] = False
            df_terminal_availability.loc[pd.Timestamp(self.env.simulation_stop), 'Terminal'] = False
        return df_terminal_availability


    def remove_vessel_from_queue(self, vessel):
        if vessel.id in self.queue.index:
            self.queue = self.queue.drop(vessel.id)
    

    def select_berth_for_vessel(self, vessel, origin, delay=0.):
        best_available_berth = None
        available_berths = self.determine_berth_availability(vessel, origin, delay=delay)
        minimum_waiting_time = available_berths.waiting_time.min()
        berths_with_minimum_waiting_time = available_berths[available_berths.waiting_time == minimum_waiting_time]
        minimum_berth_length = berths_with_minimum_waiting_time.berth_length.min()
        best_available_berths = berths_with_minimum_waiting_time[
            berths_with_minimum_waiting_time.berth_length == minimum_berth_length]
        if len(best_available_berths):
            best_available_berth = best_available_berths.iloc[0]
            best_available_berth_name = best_available_berth.name
            operation_waiting_for = best_available_berth.operation_waiting_for
            best_available_berth = select_berth_based_on_name(self, best_available_berth_name)
        return best_available_berth, operation_waiting_for


    def update_vessel_berth_planning_status(self, vessel, new_status):
        if hasattr(vessel, 'berth') and vessel.berth is not None:
            berth_planning = self.berth_planning
            berth_info = berth_planning.loc[berth_planning.vessel_id == vessel.id]
            if not berth_info.empty:
                berth_info_index = berth_info.iloc[-1].name
                berth_planning.loc[berth_info_index, 'status'] = new_status