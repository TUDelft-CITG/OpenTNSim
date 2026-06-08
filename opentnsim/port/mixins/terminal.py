from opentnsim.core import  Identifiable, Log, Movable, PriorityFilterStore, VesselProperties
from opentnsim.graph.utils import get_sailing_time, node_path_to_edge_path
from opentnsim.port.mixins.port import IsPortComponent
from opentnsim.port.mixins.berth import IsQuay, IsJetty, IsBerth
from opentnsim.port.utils import determine_new_route_for_vessel, get_vessel_from_id

import numpy as np
import pandas as pd
import datetime
import networkx as nx
import simpy


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
        yield from self.loading(destination)
        yield from self.request_port_exit(destination)
        yield from self.deberthing(destination)
        yield from self.release_berth_access(destination)
        self.env.process(self.move())
        raise simpy.exceptions.Interrupt('Route of vessel has changed.')


    def request_berth_access(self):
        yield from self.berth.request_access(self)


    def release_berth_access(self, origin):
        yield from self.berth.release_access(self)
        self.berth.remove_vessel_from_occupants_df(self)


    def berthing(self, origin):
        self.log_entry_v0("Berthing start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        yield self.env.timeout(self.berthing_time*60)
        self.log_entry_v0("Berthing stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        if len(self.next_berthing_times):
            self.berthing_time = self.next_berthing_times[0]
            self.next_berthing_times = self.next_berthing_times[1:]


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
        process_stop_time = current_time + pd.Timedelta(seconds=self.loading_time*3600 + self.deberthing_time*60)
        loading_process = self.env.timeout(self.loading_time*3600)
        negotiate_port_exit = self.env.process(self.request_port_exit(origin, parallel_process=loading_process, process_stop_time = process_stop_time))
        yield loading_process | negotiate_port_exit
        if negotiate_port_exit.is_alive:
            negotiate_port_exit.interrupt()
        self.log_entry_v0("Loading stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        if len(self.next_loading_times):
            self.loading_time = self.next_loading_times[0]
            self.next_loading_times = self.next_loading_times[1:]


    def deberthing(self, origin):
        self.log_entry_v0("Deberthing start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        yield self.env.timeout(self.deberthing_time*60)
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
        berth_capacities = [berth.berth_length if isinstance(berth, IsQuay) else 1 for berth in self.berths.items]
        self.berth_planning = pd.DataFrame(columns=berth_names)
        self.berth_planning.loc[self.env.simulation_start] = berth_capacities
        self.berth_planning.loc[self.env.simulation_stop] = berth_capacities
        quay_berths = [berth.name for berth in self.berths.items if isinstance(berth, IsQuay)]
        self.berth_planning[quay_berths] = self.berth_planning[quay_berths].astype(float)
        jetty_berths = [berth.name for berth in self.berths.items if isinstance(berth, IsJetty)]
        self.berth_planning[jetty_berths] = self.berth_planning[jetty_berths].astype(int)
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
        self.queue = pd.DataFrame(columns=["Vessel_L","Vessel_B","Vessel_T","Berth","Waiting_start_time","Waiting_stop_time","Arrival_time_at_berth"])


    def assign_vessel_to_queue(self, vessel, arrival_time_at_berth, waiting_time=0.,berth=None):
        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        waiting_stop = current_time + pd.Timedelta(seconds=waiting_time)
        berth_name = None
        if isinstance(berth, IsBerth):
            berth_name = berth.name
            berth.add_vessel_to_queue(vessel, current_time, waiting_stop, arrival_time_at_berth)
        self.queue.loc[vessel.id] = [vessel.L, vessel.B, vessel.T, berth_name, current_time, waiting_stop, arrival_time_at_berth]


    def update_queue(self, vessel, arrival_time_at_berth, waiting_time=0.,berth=None):
        self.remove_vessel_from_queue(vessel)
        self.assign_vessel_to_queue(vessel, arrival_time_at_berth, waiting_time, berth)
        self.queue = self.queue.sort_values('Waiting_start_time')
        if isinstance(berth, IsBerth):
            waiting_start = berth.queue.loc[vessel.id, 'Waiting_start_time']
            current_time = datetime.datetime.fromtimestamp(vessel.env.now)
            waiting_stop = current_time + pd.Timedelta(seconds=waiting_time)
            berth.update_queue(vessel, waiting_start, waiting_stop, arrival_time_at_berth)


    def remove_vessel_from_queue(self, vessel):
        if vessel.id in self.queue.index:
            self.queue = self.queue.drop(vessel.id)


    def find_suitable_berths(self, vessel):
        suitable_berths = []
        for berth in self.berths.items:
            if berth.depth > vessel.T:
                suitable_berths.append(berth_name)
        return suitable_berths


    def determine_sailing_time_to_berth(self, vessel, origin, berth):
        destination = berth.node
        route = nx.dijkstra_path(self.env.graph,origin,destination)
        edge_route = node_path_to_edge_path(vessel.env.graph, route)
        sailing_time, _ = get_sailing_time(vessel, edge_route)
        return sailing_time


    def select_berth_based_on_name(self, berth_name):
        selected_berth = None
        for berth in self.berths.items:
            if berth.name == berth_name:
                selected_berth = berth
                break
        return selected_berth


    def determine_potential_berth_availability(self, df_potential_available_berths, vessel):
        df_availability = pd.DataFrame()
        for berth_name in df_potential_available_berths.columns:
            berth = self.select_berth_based_on_name(berth_name)
            if isinstance(berth, IsQuay):
                berth_available = (df_potential_available_berths[berth_name] >= vessel.L)
            else:
                berth_available = (df_potential_available_berths[berth_name] > 0)
            df_availability[berth_name] = berth_available
        return df_availability


    def determine_potential_available_berths(self, vessel):
        berth_planning = self.berth_planning
        terminal_berths = self.berths.items
        fit_berths_names = [berth.name for berth in terminal_berths if (berth.depth >= vessel.T) and (berth.berth_length >= vessel.L)]
        current_time = datetime.datetime.fromtimestamp(self.env.now)
        fit_berth_planning_availability = berth_planning[fit_berths_names]
        previous_events = fit_berth_planning_availability[fit_berth_planning_availability.index <= current_time]
        future_events = fit_berth_planning_availability[fit_berth_planning_availability.index > current_time]
        last_previous_event_index =  previous_events.index.max()

        if pd.isna(last_previous_event_index):
            previous_event = fit_berth_planning_availability.iloc[0:0]
        else:
            previous_event = fit_berth_planning_availability.loc[[last_previous_event_index]]

        fit_berth_planning_availability = pd.concat([previous_event, future_events])
        return fit_berth_planning_availability


    def provide_berth_availability_info(self, vessel):
        berth_planning_availability = self.determine_potential_available_berths(vessel)
        df_berth_availability = self.determine_potential_berth_availability(berth_planning_availability, vessel)
        return df_berth_availability


    def provide_terminal_availability_info(self, vessel, origin, berth = None):
        df_berth_availability = self.provide_berth_availability_info(vessel)
        df_terminal_availability = pd.DataFrame()
        sailing_time_to_berth = pd.Timedelta(seconds=0.)
        if berth is None:
            df_terminal_availability['Terminal'] = df_berth_availability.any(axis=1)
        else:
            sailing_time_to_berth = pd.Timedelta(seconds=self.determine_sailing_time_to_berth(vessel, origin, berth))
            df_terminal_availability['Terminal'] = df_berth_availability[berth.name]
        if not df_terminal_availability.empty:
            df_terminal_availability.index -= sailing_time_to_berth
        else:
            df_terminal_availability.loc[self.env.simulation_start,'Terminal'] = False
            df_terminal_availability.loc[self.env.simulation_stop, 'Terminal'] = False
        return df_terminal_availability


    def determine_berth_availability(self, vessel, origin):
        berthing_time = vessel.berthing_time*60
        loading_time = vessel.loading_time*3600
        deberthing_time = vessel.deberthing_time*60
        occupation_duration = berthing_time + loading_time + deberthing_time
        current_time = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))
        potential_berths = self.provide_berth_availability_info(vessel)
        df_berth_time_slot = pd.DataFrame(columns=['Time_start','Time_stop','Waiting_time','Berth_length'])
        for berth in self.berths.items:
            berth_name = berth.name
            if berth_name not in potential_berths.columns:
                continue


            berth_available = potential_berths[berth_name]
            mask_berth_available = (berth_available == True)
            berth_availability_start_times = np.array(berth_available[mask_berth_available].index)
            if not len(berth_availability_start_times):
                 continue

            for berth_availability_start_time in berth_availability_start_times:
                berth_availability_stop_times = berth_available[(berth_available.index > berth_availability_start_time)&(berth_available == False)]
                if not len(berth_availability_stop_times):
                    berth_availability_stop_time = berth_available.index[-1]
                else:
                    berth_availability_stop_time = berth_availability_stop_times.index[0]

                sailing_time_to_berth = pd.Timedelta(seconds=self.determine_sailing_time_to_berth(vessel, origin, berth))
                actual_berth_availability_start_time = np.max([berth_availability_start_time,current_time + sailing_time_to_berth])
                berth_availability_duration = (berth_availability_stop_time - actual_berth_availability_start_time) / np.timedelta64(1, 's')
                if berth_availability_duration < occupation_duration:
                    continue

                waiting_time = berth_availability_start_time - current_time - sailing_time_to_berth
                waiting_time = np.max([pd.Timedelta(seconds=0),waiting_time])
                df_berth_time_slot.loc[berth_name,:] = [berth_availability_start_time,berth_availability_stop_time,waiting_time,berth.berth_length]
                break
        return df_berth_time_slot


    def select_berth_for_vessel(self, available_berths):
        best_available_berth = None
        minimum_waiting_time = available_berths.Waiting_time.min()
        berths_with_minimum_waiting_time = available_berths[available_berths.Waiting_time == minimum_waiting_time]
        minimum_berth_length = berths_with_minimum_waiting_time.Berth_length.min()
        best_available_berths = berths_with_minimum_waiting_time[berths_with_minimum_waiting_time.Berth_length == minimum_berth_length]
        if len(best_available_berths):
            best_available_berth = best_available_berths.iloc[0]
            best_available_berth_name = best_available_berth.name
            best_available_berth = self.select_berth_based_on_name(best_available_berth_name)
        return best_available_berth


    def calculate_time_at_berth(self, vessel):
        time_at_berth = vessel.berthing_time*60 + vessel.loading_time*3600 + vessel.deberthing_time*60
        return time_at_berth


    def assign_vessel_to_berth(self, vessel, origin, berth, delay = 0.):
        delay = pd.Timedelta(seconds = delay)
        vessel.berth = berth
        sailing_time_to_berth = pd.Timedelta(seconds=self.determine_sailing_time_to_berth(vessel, origin, berth))
        time_start = datetime.datetime.fromtimestamp(vessel.env.now) + sailing_time_to_berth + delay
        time_at_berth = self.calculate_time_at_berth(vessel)
        time_stop = time_start + pd.Timedelta(seconds=time_at_berth)
        self.add_vessel_to_berth_planning(vessel, berth, time_start, time_stop)
        if isinstance(berth, IsBerth):
            berth.add_vessel_to_occupants_df(vessel, time_start, time_stop)
        return time_start


    def add_vessel_to_berth_planning(self, vessel, berth, time_start, time_stop):
        berth_planning = self.berth_planning.copy()
        if isinstance(berth, IsQuay):
            berth_planning.loc[time_start, berth.name] = np.nan
            berth_planning.loc[time_stop, berth.name] = np.nan
        elif isinstance(berth, IsJetty):
            berth_planning.loc[time_start, berth.name] = 0
            berth_planning.loc[time_stop, berth.name] = 1
            mask = (berth_planning.index > time_start)&(berth_planning.index < time_stop)
            berth_planning.loc[berth_planning[mask].index, berth.name] = 0
            berth_planning[berth.name] = berth_planning[berth.name].astype(int)

        berth_planning = berth_planning.sort_index()
        with pd.option_context("future.no_silent_downcasting", True):
            berth_planning = berth_planning.ffill().infer_objects(copy=False)
            berth_planning = berth_planning.bfill().infer_objects(copy=False)

        if isinstance(berth, IsQuay):
            mask = (berth_planning.index >= time_start)&(berth_planning.index < time_stop)
            berth_planning.loc[berth_planning[mask].index, berth.name] -= vessel.L
        self.berth_planning = berth_planning


    def replan_vessels_terminal_berths(self, vessel, delay = 0.):
        self.berth_planning = self.berth_planning.iloc[[0,-1]]
        for berth in self.berths.items:
            occupying_vessels = berth.occupying_vessels
            if vessel.id in occupying_vessels.index:
                occupying_vessels = occupying_vessels.drop(vessel.id)
            for occupying_vessel_id, occupying_vessel_info in occupying_vessels.iterrows():
                occupying_vessel = get_vessel_from_id(self.env, [occupying_vessel_id])[0]
                time_start = occupying_vessel_info.Time_at_berth_start
                time_stop = occupying_vessel_info.Time_at_berth_stop + pd.Timedelta(seconds=delay)
                self.add_vessel_to_berth_planning(occupying_vessel, berth, time_start, time_stop)