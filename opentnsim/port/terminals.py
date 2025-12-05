from opentnsim.core import  HasResource, Identifiable, Locatable, Log, HasLength, HasResource, Movable, SimpyObject
from opentnsim.output import HasOutput
from opentnsim.graph import OnNode, OnEdge
from opentnsim.core.capacity import PriorityFilterStore
from opentnsim.port.port import IsPartofPort
from opentnsim.tidal_accessibility import check_if_route_contains_restrictions

import numpy as np
import pandas as pd
import datetime
import networkx as nx

class HasTerminal(Movable):
    def __init__(self,
                 terminal,
                 berthing_time,
                 loading_time,
                 deberthing_time,
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
        self.next_berthing_times = next_berthing_times
        self.next_terminals = next_terminals
        self.next_loading_times = next_loading_times
        self.next_deberthing_times = next_deberthing_times
        super().__init__(*args, **kwargs)
        self.on_complete_edge_functions.append(self.pass_terminal)


    def pass_terminal(self, origin, destination):
        # origin (although not used) is needed as this is a 'complete edge'-function
        if 'Berth' not in self.env.graph.nodes[destination].keys():
            return

        berths = self.env.graph.nodes[destination]['Berth']
        if not self.berth in berths:
            return

        yield from self.request_berth_access()
        yield from self.berthing(destination)
        yield from self.loading(destination)
        yield from self.release_berth_access()
        yield from self.deberthing(destination)


    def determine_new_route(self):
        origin = self.route[-1]
        if len(self.next_terminals):
            next_terminal = self.next_terminals[-1]
            self.next_terminals = self.next_terminals[1:]
        next_terminal.request_terminal_access(vessel, origin)

    def request_berth_access(self):
        if isinstance(self.berth, IsQuay):
            yield from self.berth.request_quay_access(self)
        elif isinstance(self.berth, IsJetty):
            yield from []


    def release_berth_access(self):
        if isinstance(self.berth, IsQuay):
            yield from self.berth.release_quay_access(self)
        elif isinstance(self.berth, IsJetty):
            yield from []


    def berthing(self, origin):
        self.log_entry_v0("Berthing start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        yield self.env.timeout(self.berthing_time)
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
        yield self.env.timeout(self.loading_time)
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
        yield self.env.timeout(self.deberthing_time)
        self.log_entry_v0("Deberthing stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        if len(self.next_deberthing_times):
            self.deberthing_time = self.next_deberthing_times[0]
            self.next_deberthing_times = self.next_deberthing_times[1:]


    def wait_for_berth_availability(self, origin, waiting_time):
        self.log_entry_v0("Waiting for berth availability start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        yield self.env.timeout(waiting_time)
        self.log_entry_v0("Waiting for berth availability stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])


class HasBerthPlanning:

    def __init__(self, *args, **kwargs):
        berth_names = [berth.name for berth in self.berths.items]
        berth_capacities = [berth.length for berth in self.berths.items]
        self.berth_planning = pd.DataFrame(columns=berth_names)
        self.berth_planning.loc[self.env.simulation_start] = berth_capacities
        self.berth_planning.loc[self.env.simulation_stop] = berth_capacities
        super().__init__(*args, **kwargs)


class IsTerminal(Log, Identifiable, HasBerthPlanning, IsPartofPort, HasOutput):

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
        self.port.terminals.append(self)

    def find_suitable_berths(self, vessel):
        suitable_berths = []
        for berth in self.berths.items:
            if berth.depth > vessel.T:
                suitable_berths.append(berth_name)
        return suitable_berths


    def determine_potential_available_berths(self, vessel):
        berth_planning = self.berth_planning
        terminal_berths = self.berths.items
        fit_berths_names = [berth.name for berth in terminal_berths if (berth.depth > vessel.T) and (berth.length > vessel.L)]
        current_time = datetime.datetime.fromtimestamp(self.env.now)
        fit_berth_planning = berth_planning[fit_berths_names]
        fit_berth_planning_availability = fit_berth_planning[fit_berth_planning.index >= current_time]
        return fit_berth_planning_availability


    def determine_sailing_time_to_berth(self, vessel, origin, berth):
        destination = berth.node
        route = nx.dijkstra_path(self.env.graph,origin,destination)
        sailing_information = self.env.vessel_traffic_service.provide_sailing_time(vessel, route)
        sailing_time = sailing_information["Time"].cumsum().values[0]
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
            berth_available = (df_potential_available_berths[berth_name] > vessel.L)
            df_availability[berth_name] = berth_available
        return df_availability


    def determine_berth_availability(self, vessel, origin, df_entry):
        berthing_time = vessel.berthing_time
        loading_time = vessel.loading_time
        deberthing_time = vessel.deberthing_time
        occupation_duration = berthing_time + loading_time + deberthing_time
        current_time = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))

        df_berth_availability = df_entry.copy()
        if 'Tide' in df_entry.columns:
            df_tidal_windows = df_entry['Tide'].copy()
            df_berth_availability = df_berth_availability.drop('Tide', axis=1)
        else:
            df_entry['Tide'] = True
            df_tidal_windows = df_entry['Tide'].copy()

        df_berth_time_slot = pd.DataFrame(columns=['Time_start','Time_stop','Waiting_time','Berth_length'])
        for berth_name in df_berth_availability.columns:
            berth = self.select_berth_based_on_name(berth_name)
            berth_available = df_berth_availability[berth_name]
            mask_berth_available_and_reachable = (berth_available == True) & (df_tidal_windows == True)
            berth_availability_start_times = np.array(berth_available[mask_berth_available_and_reachable].index)
            if not len(berth_availability_start_times):
                 continue
            for berth_availability_start_time in berth_availability_start_times:
                berth_availability_stop_times = berth_available[(berth_available.index > berth_availability_start_time)&(berth_available == False)]
                if not len(berth_availability_stop_times):
                    berth_availability_stop_time = berth_available.index[-1]
                else:
                    berth_availability_stop_time = berth_availability_stop_times[0]
                berth_availability_duration = (berth_availability_stop_time - berth_availability_start_time) / np.timedelta64(1, 's')
                if berth_availability_duration < occupation_duration:
                    continue
                sailing_time_to_berth = self.determine_sailing_time_to_berth(vessel, origin, berth)
                waiting_time = berth_availability_start_time - current_time - pd.Timedelta(seconds=sailing_time_to_berth)
                waiting_time = np.max([pd.Timedelta(seconds=0),waiting_time])
                df_berth_time_slot.loc[berth_name,:] = [berth_availability_start_time,berth_availability_stop_time,waiting_time,berth.length]
                break

        return df_berth_time_slot

    def pick_best_available_berth(self, df_berth_time_slot, df_entry):
        minimum_waiting_time = df_berth_time_slot.Waiting_time.min()
        berths_with_minimum_waiting_time = df_berth_time_slot[df_berth_time_slot.Waiting_time == minimum_waiting_time]
        minimum_berth_length = berths_with_minimum_waiting_time.Berth_length.min()
        best_available_berths = berths_with_minimum_waiting_time[berths_with_minimum_waiting_time.Berth_length == minimum_berth_length]
        best_available_berth = best_available_berths.iloc[0]
        best_available_berth_name = best_available_berth.name
        entry_time = best_available_berth.Time_start
        current_time = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))
        df_entry_future_to_entry_time = df_entry[(df_entry.index >= current_time) & (df_entry.index <= entry_time)]
        if 'Tide' in df_entry_future_to_entry_time.columns:
            df_entry_future_to_entry_time = df_entry_future_to_entry_time[['Tide',best_available_berth_name]]
        else:
            df_entry_future_to_entry_time = df_entry_future_to_entry_time[[best_available_berth_name]]
        df_entry_future_to_entry_time['Combined'] = df_entry_future_to_entry_time.all(axis=1)
        df_entry_future_to_entry_time['Redundant'] = df_entry_future_to_entry_time.any(axis=1) == False
        df_entry_future_to_entry_time = df_entry_future_to_entry_time[df_entry_future_to_entry_time['Redundant'] == False]
        df_entry_future_to_entry_time = df_entry_future_to_entry_time.drop('Redundant', axis=1)
        df_entry_future_to_entry_time["Reason"] = df_entry_future_to_entry_time.apply(lambda row: row.index[row.eq(False)][0] if any(row.eq(False)) else None, axis=1)
        with pd.option_context("future.no_silent_downcasting", True):
            df_entry_future_to_entry_time = df_entry_future_to_entry_time.ffill()
        waiting_stops_info = df_entry_future_to_entry_time[df_entry_future_to_entry_time.Combined == True]
        waiting_starts_info = df_entry_future_to_entry_time[df_entry_future_to_entry_time.Combined == False]
        waiting_times = []
        waiting_causes = []
        for (waiting_start_time, waiting_start_info), (waiting_stop_time, waiting_stop_info) in zip(waiting_starts_info.iterrows(), waiting_stops_info.iterrows()):
            waiting_times.append(waiting_stop_time - waiting_start_time)
            waiting_causes.append(waiting_start_info.Reason)
        best_available_berth = self.select_berth_based_on_name(best_available_berth_name)
        return best_available_berth, best_available_berth_name, waiting_times, waiting_causes


    def assign_berth_to_vessel(self, vessel, berth):
        vessel.berth = berth


    def request_terminal_access(self, vessel, origin, df_entry):
        available_berth_time_slots = self.determine_berth_availability(vessel, origin, df_entry)
        berth, berth_name, waiting_times, waiting_causes = self.pick_best_available_berth(available_berth_time_slots, df_entry)
        self.assign_berth_to_vessel(vessel, berth)
        return berth, berth_name, waiting_times, waiting_causes


    def determine_terminal_availability(self, vessel):
        fit_berth_planning_availability = self.determine_potential_available_berths(vessel)
        df_availability = self.determine_potential_berth_availability(fit_berth_planning_availability, vessel)
        return df_availability


def add_berth_to_graph(berth):
    node_info = berth.env.graph.nodes[berth.node]
    if 'Berth' not in node_info.keys():
        node_info['Berth'] = []
    node_info['Berth'].append(berth)


class IsJetty(OnNode, HasResource, Identifiable, Log):
    def __init__(self, length, depth, capacity=1, *args, **kwargs):
        super().__init__(nr_resources=capacity, *args, **kwargs)
        add_berth_to_graph(self)
        self.length = length
        self.depth = depth
        self.capacity = capacity


class IsQuay(OnNode, HasLength, Identifiable, Log):
    def __init__(self, length, depth, *args, **kwargs):
        super().__init__(length=length, remaining_length=length,*args, **kwargs)
        add_berth_to_graph(self)
        self.length = length
        self.depth = depth
        self.capacity = np.inf
        self.availability_quay_positions = pd.DataFrame(data=[[0, length, length, None]],columns=['Distance_start','Distance_stop','Length','Occupant'])


    def request_quay_access(self, vessel):
        quay_position = self.select_quay_position(vessel)
        yield from self.adjust_availability_quay_positions(vessel, quay_position)


    def release_quay_access(self, vessel):
        quay_position = self.find_quay_position(vessel)
        yield from self.readjust_availability_quay_positions(quay_position)



    def find_quay_position(self, vessel):
        quay_position = self.availability_quay_positions[self.availability_quay_positions.Occupant == vessel].index[0]
        return quay_position


    def calculate_quay_length_level(self):
        """ Function that keeps track of the maximum length that is available at the quay. """
        new_level = np.max(self.availability_quay_positions[self.availability_quay_positions.Occupant.isna()]['Length'])
        return new_level


    def select_quay_position(self, vessel):
        """ Function that claims a length along the quay equal to the length of the vessel itself and calculates the relative position of the vessel along the quay. If there are multiple
            relative positions possible, the vessel claims the first position. If there is no suitable position availalble (vessel does not fit), then it returns the action
            of moving to the anchorage area.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties

        """

        potential_quay_positions = self.availability_quay_positions[self.availability_quay_positions.Length >= vessel.L]
        quay_position = potential_quay_positions['Length'].idxmin()
        return quay_position


    def adjust_availability_quay_positions(self, vessel, quay_position):
        """ Function that adjusts the available quay lenghts and positions given a honored request of a vessel at a given position

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - index_quay_position: quay position index at which the vessel is located at the quay with respect to the other vessels

        """
        # Determine the current maximum available length of the terminal
        old_level = self.calculate_quay_length_level()

        # Add vessel to layout
        quay_position_info = self.availability_quay_positions.loc[quay_position].copy()
        self.availability_quay_positions.loc[quay_position, 'Distance_stop'] = quay_position_info.Distance_start + vessel.L
        self.availability_quay_positions.loc[quay_position, 'Length'] = vessel.L
        self.availability_quay_positions.loc[quay_position, 'Occupant'] = vessel

        # Add additional row with leftover quay length
        position = self.availability_quay_positions.index.get_loc(quay_position)+1
        if quay_position_info.Length != vessel.L:
            distance_start = self.availability_quay_positions.loc[quay_position, 'Distance_stop']
            distance_stop = quay_position_info.Distance_stop
            length = distance_stop - distance_start
            new_position = pd.DataFrame({'Distance_start':[distance_start],
                                         'Distance_stop':[distance_stop],
                                         'Length':[length],
                                         'Occupant':[None]})
            self.availability_quay_positions = pd.concat([self.availability_quay_positions.iloc[:position], new_position, self.availability_quay_positions.iloc[position:]])
        self.availability_quay_positions = self.availability_quay_positions.reset_index(drop=True)

        # Determine the new current maximum available length of the terminal
        new_level = self.calculate_quay_length_level()

        # Claim length of resource so that the level equals the berth position with the largest length (unless it has not changed)
        if old_level != new_level:
            yield self.resource.get(old_level - new_level)


    def readjust_availability_quay_positions(self, position):
        """ Function that readjusts the available quay lenghts and positions given a release of a request of a vessel at a given position

            Input:
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class
                - position: quay position index at which the vessel is located at the quay with respect to the other vessels

        """
        # Determine the current maximum available length of the terminal
        old_level = self.calculate_quay_length_level()

        # Drop vessel from layout
        self.availability_quay_positions.loc[position,'Occupant'] = None

        # Combine subsequent empty positions
        self.availability_quay_positions['Occupant_filled'] = self.availability_quay_positions['Occupant'].fillna('NA')
        self.availability_quay_positions['group'] = (self.availability_quay_positions['Occupant_filled'] != self.availability_quay_positions['Occupant_filled'].shift()).cumsum()
        self.availability_quay_positions = self.availability_quay_positions.groupby('group', as_index=False).agg({'Distance_start': 'first',
                                                                                                                  'Distance_stop': 'last',
                                                                                                                  'Length':'sum',
                                                                                                                  'Occupant': 'first'})
        self.availability_quay_positions = self.availability_quay_positions.drop(columns=['group'])

        # Determine the new current maximum available length of the terminal
        new_level = self.calculate_quay_length_level()

        # Put back length to resource so that the level equals the berth position with the largest length (unless it has not changed)
        if old_level != new_level:
            yield self.resource.put(new_level - old_level)





