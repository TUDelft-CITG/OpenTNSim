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

        print('hi, continue here: berthing process, loading process, deberthing process, while updating schedule')
        yield from []


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
        super().__init__(length=length, *args, **kwargs)
        add_berth_to_graph(self)
        self.length = length
        self.depth = depth
        self.capacity = np.inf

    def calculate_quay_length_level(self):
        """ Function that keeps track of the maximum length that is available at the quay

            Input:
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class

        """

        # Set default parameters
        aql = self.available_quay_lengths
        new_level = np.max(aql)
        available_quay_lengths = [0]

        # Loop over the position indexes
        for index in range(len(aql)):
            # If the index of the locaton is 0, or if the previous location is the same as the current location (and hence the index of the location is not 0) or if the location is not available (value = 1):
            if index == 0 or aql[index][1] == aql[index - 1][1] or aql[index][0] == 1:
                # Continue, else if its the last index and there is not yet a suitable index found for an available location: return that available length is the last one in the list (=0)
                if index == len(aql) - 1:
                    new_level = available_quay_lengths[-1]
                continue

            # If there is an available location: append length to list and return the maximum of the list
            available_quay_lengths.append(aql[index][1] - aql[index - 1][1])
            new_level = np.max(available_quay_lengths)
        return new_level

    def adjust_available_quay_lengths(self, vessel, index_quay_position):
        """ Function that adjusts the available quay lenghts and positions given a honored request of a vessel at a given position

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - index_quay_position: quay position index at which the vessel is located at the quay with respect to the other vessels

        """

        # Import the locations of the current configuration of vessels located at the quay
        aql = self.available_quay_lengths

        # Determine the current maximum available length of the terminal
        old_level = self.calculate_quay_length_level()

        # If the value of the position index before the honered quay position (start of the available position) is still available (=0), change it to 1
        if aql[index_quay_position - 1][0] == 0:
            aql[index_quay_position - 1][0] = 1

        # If the value of the honered quay position (end of the available position) is still available (=0) and the end of this position equals the start of the position added with the vessel length, change it to 1
        if aql[index_quay_position][0] == 0 and aql[index_quay_position][1] == aql[index_quay_position - 1][1] + vessel.L:
            aql[index_quay_position][0] = 1

        # Else insert a new stopping location in the locations of the current configuration of vessels located at the quay by twice adding the vessel length to the start position of the location, once with a occupied value (=1), followed by a available value (=0)
        else:
            aql.insert(index_quay_position, [1, vessel.L + aql[index_quay_position - 1][1]])
            aql.insert(index_quay_position + 1, [0, vessel.L + aql[index_quay_position - 1][1]])

        # Replace the list of the locations of the current configuration of vessels located at the quay of the terminal
        self.available_quay_lengths = aql
        # Calculate the quay position and append to the vessel (mid-length of the vessel + starting length of the position)
        vessel.quay_position = 0.5 * vessel.L + aql[index_quay_position - 1][1]
        # Determine the new current maximum available length of the terminal
        new_level = self.calculate_quay_length_level()
        # If the old level does not equal (is greater than) the new level and the vessel does not have to wait in the anchorage first: then claim the difference between these lengths
        if old_level != new_level and vessel.waiting_in_anchorage != True:
            self.length.get(old_level - new_level)
        # Else if the vessel has to wait in the anchorage first: calculate the difference between the lengths corrected by the vessel length to be claimed by the vessel (account for this vessel, so that it has priority over new vessels)
        elif vessel.waiting_in_anchorage == True:
            new_level = old_level-vessel.L-new_level
            # If this difference is negative: give absolute length back to terminal
            if new_level < 0:
                self.length.put(-new_level)
            # Else if this difference is positive: claim this length of the terminal
            elif new_level > 0:
                self.length.get(new_level)
        return

    def readjust_available_quay_lengths(self, position):
        """ Function that readjusts the available quay lenghts and positions given a release of a request of a vessel at a given position

            Input:
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class
                - position: quay position index at which the vessel is located at the quay with respect to the other vessels

        """

        # Import the locations of the current configuration of vessels located at the quay
        aql = self.available_quay_lengths
        # Loop over the position indexes
        for index in range(len(aql)):
            # Skip the first position index
            if index == 0:
                continue
            # If the position of the vessel falls within the position bounds in the current configuration: break loop (save index)
            if aql[index - 1][1] < position and aql[index][1] > position:
                break

        # Set both values of these position bounds to zero (available again)
        aql[index - 1][0] = 0
        aql[index][0] = 0

        # Set a default list of redundant indexes to be removed
        to_remove = []
        # Nested loop over the position indexes
        for index in enumerate(aql):
            for jndex in enumerate(aql):
                # If the two indexes are not equal and the value at position index 1 and index 2 are both zero (available) and the locations of the two indexes are equal: remove the first positional index
                if index[0] != jndex[0] and index[1][0] == 0 and jndex[1][0] == 0 and index[1][1] == jndex[1][1]:
                    to_remove.append(index[0])

        # If there are indexes to be removed, loop over these indexes and remove them
        for index in list(reversed(to_remove)):
            aql.pop(index)

        # Return the locations of the new configuration of vessels located at the quay
        return aql





