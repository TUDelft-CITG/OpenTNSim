from opentnsim.core import  HasResource, Identifiable, Locatable, Log, HasLength, HasResource, Movable, SimpyObject
from opentnsim.output import HasOutput
from opentnsim.graph import OnNode, OnEdge
from opentnsim.core.capacity import PriorityFilterStore
from opentnsim.port.port import IsPartofPort

import numpy as np
import pandas as pd
import datetime
import networkx as nx
import simpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial import ConvexHull

class HasTerminal(Movable):
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
        yield from self.release_berth_access(destination)
        yield from self.deberthing(destination)
        self.env.process(self.move())
        raise simpy.exceptions.Interrupt('Route of vessel has changed.')


    def request_berth_access(self):
        if isinstance(self.berth, IsQuay):
            yield from self.berth.request_quay_access(self)
        elif isinstance(self.berth, IsJetty):
            yield from []


    def release_berth_access(self, origin):
        if isinstance(self.berth, IsQuay):
            yield from self.berth.release_quay_access(self)
        elif isinstance(self.berth, IsJetty):
            yield from []

        # if going to new terminal -> request port passage
        pass
        # if leaving the port -> request port exit
        yield from self.request_port_exit(origin)

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
        yield self.env.timeout(self.loading_time*3600)
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
        berth_capacities = [berth.length if isinstance(berth, IsQuay) else 1 for berth in self.berths.items]
        self.berth_planning = pd.DataFrame(columns=berth_names)
        self.berth_planning.loc[self.env.simulation_start] = berth_capacities
        self.berth_planning.loc[self.env.simulation_stop] = berth_capacities
        quay_berths = [berth.name for berth in self.berths.items if isinstance(berth, IsQuay)]
        self.berth_planning[quay_berths] = self.berth_planning[quay_berths].astype(float)
        jetty_berths = [berth.name for berth in self.berths.items if isinstance(berth, IsJetty)]
        self.berth_planning[jetty_berths] = self.berth_planning[jetty_berths].astype(int)
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
        self.vessels_waiting_df = pd.DataFrame(columns=["Waiting_start_time","Vessel_length","Cargo_volume","Event"])


    def find_suitable_berths(self, vessel):
        suitable_berths = []
        for berth in self.berths.items:
            if berth.depth > vessel.T:
                suitable_berths.append(berth_name)
        return suitable_berths


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
            berth = self.select_berth_based_on_name(berth_name)
            if isinstance(berth, IsQuay):
                berth_available = (df_potential_available_berths[berth_name] > vessel.L)
            else:
                berth_available = (df_potential_available_berths[berth_name] > 0)
            df_availability[berth_name] = berth_available
        return df_availability


    def determine_potential_available_berths(self, vessel):
        berth_planning = self.berth_planning
        terminal_berths = self.berths.items
        fit_berths_names = [berth.name for berth in terminal_berths if (berth.depth >= vessel.T) and (berth.length >= vessel.L)]
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
        df_terminal_availability.index -= sailing_time_to_berth
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
                df_berth_time_slot.loc[berth_name,:] = [berth_availability_start_time,berth_availability_stop_time,waiting_time,berth.length]
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


    def assign_berth_to_vessel(self, vessel, origin, berth):
        vessel.berth = berth
        sailing_time_to_berth = self.determine_sailing_time_to_berth(vessel, origin, berth)
        time_start = datetime.datetime.fromtimestamp(vessel.env.now) + pd.Timedelta(seconds=sailing_time_to_berth)
        time_at_berth = self.calculate_time_at_berth(vessel)
        time_stop = time_start + pd.Timedelta(seconds=time_at_berth)
        self.update_berth_planning(vessel, berth, time_start, time_stop)


    def update_berth_planning(self, vessel, berth, time_start, time_stop):
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

        if isinstance(berth, IsQuay):
            mask = (berth_planning.index > time_start)&(berth_planning.index < time_stop)
            berth_planning.loc[berth_planning[mask].index, berth.name] -= vessel.L

        self.berth_planning = berth_planning


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
        self.availability_quay_positions = pd.DataFrame(data=[[0, length, length, None]],columns=['Distance_start','Distance_stop','Length_available','Occupant'])
        self.historic_quay_planning = pd.DataFrame(data={0: [None,None], length: [None,None]},
                                                   index=[self.env.simulation_start,self.env.simulation_stop])


    def add_quay_position_to_historic_planning(self, vessel, quay_position_nr, add=True):

        def horizontal_fill_between_equals(row):
            row_values = row.values
            for i in range(len(row_values)):
                if pd.notna(row_values[i]):
                    j = i + 1
                    while j < len(row_values) and pd.isna(row_values[j]):
                        if j + 1 < len(row_values) and pd.notna(row_values[j + 1]) and row_values[j + 1] == row_values[i]:
                            row_values[i + 1:j + 1] = row_values[i]
                            break
                        j += 1
            return pd.Series(row_values, index=row.index)

        quay_position = self.availability_quay_positions.loc[quay_position_nr]
        current_time = datetime.datetime.fromtimestamp(self.env.now)
        quay_position_start = quay_position.Distance_start + 0.001
        quay_position_stop = quay_position.Distance_stop - 0.001

        if add:
            if quay_position.Distance_start not in self.historic_quay_planning.columns:
                self.historic_quay_planning.loc[:,quay_position_start] = None
            if quay_position.Distance_stop not in self.historic_quay_planning.columns:
                self.historic_quay_planning.loc[:,quay_position_stop] = None

        self.historic_quay_planning.loc[current_time, quay_position_start] = vessel.id
        self.historic_quay_planning.loc[current_time, quay_position_stop] = vessel.id

        if not add:
            for col in [quay_position_start,quay_position_stop]:
                notna = self.historic_quay_planning[col].notna()
                if not notna.any():
                    continue

                first_idx = notna.idxmax()
                last_idx = notna[::-1].idxmax()

                mask = (self.historic_quay_planning.index >= first_idx) & (self.historic_quay_planning.index <= last_idx)

                self.historic_quay_planning.loc[mask, col] = self.historic_quay_planning.loc[mask, col].bfill()

            self.historic_quay_planning = self.historic_quay_planning.apply(horizontal_fill_between_equals, axis=1)

        self.historic_quay_planning = self.historic_quay_planning.sort_index()
        self.historic_quay_planning = self.historic_quay_planning.reindex(sorted(self.historic_quay_planning.columns), axis=1)
        self.historic_quay_planning[self.historic_quay_planning.isna()] = None

    def plot_historic_quay_planning(self):
        fig, ax = plt.subplots()
        historic_quay_planning_plot = self.historic_quay_planning.stack().reset_index()
        historic_quay_planning_plot.columns = ['timestamp', 'column', 'id']
        historic_quay_planning_plot = historic_quay_planning_plot[historic_quay_planning_plot['id'].notna()]
        historic_quay_planning_plot_id_mapping = {id_: list(zip(sub_df['timestamp'], sub_df['column'])) for id_, sub_df
                                                  in historic_quay_planning_plot.groupby('id', sort=False)}
        for vessel_id in historic_quay_planning_plot_id_mapping.keys():
            vessel_occupancy = historic_quay_planning_plot_id_mapping[vessel_id]
            timestamps, quay_position = zip(*vessel_occupancy)
            quay_position = list(quay_position)
            quay_position = np.array(quay_position, dtype=float)
            timestamps = list(timestamps)
            timestamps = mdates.date2num(timestamps)
            quay_position_over_time = np.column_stack((quay_position, timestamps))
            quay_position_over_time_polygons = ConvexHull(quay_position_over_time)
            ax.fill(quay_position_over_time[quay_position_over_time_polygons.vertices, 0],
                    quay_position_over_time[quay_position_over_time_polygons.vertices, 1], )

        plt.gca().yaxis_date()
        plt.xlim(0,self.length)
        plt.ylabel("Time")
        plt.xlabel("Quay length [m]")
        plt.close()
        return fig

    def request_quay_access(self, vessel):
        quay_position = self.select_quay_position(vessel)
        yield from self.adjust_availability_quay_positions(vessel, quay_position)
        self.add_quay_position_to_historic_planning(vessel, quay_position)


    def release_quay_access(self, vessel):
        quay_position = self.find_quay_position(vessel)
        self.add_quay_position_to_historic_planning(vessel, quay_position, add=False)
        yield from self.readjust_availability_quay_positions(quay_position)


    def find_quay_position(self, vessel):
        quay_position = self.availability_quay_positions[self.availability_quay_positions.Occupant == vessel.id].index[0]
        return quay_position


    def calculate_quay_length_level(self):
        """ Function that keeps track of the maximum length that is available at the quay. """
        new_level = np.max(self.availability_quay_positions[self.availability_quay_positions.Occupant.isna()]['Length_available'])
        return new_level


    def select_quay_position(self, vessel):
        """ Function that claims a length along the quay equal to the length of the vessel itself and calculates the relative position of the vessel along the quay. If there are multiple
            relative positions possible, the vessel claims the first position. If there is no suitable position availalble (vessel does not fit), then it returns the action
            of moving to the anchorage area.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties

        """

        potential_quay_positions = self.availability_quay_positions[self.availability_quay_positions.Length_available >= vessel.L]
        quay_position = potential_quay_positions['Length_available'].idxmin()
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
        self.availability_quay_positions.loc[quay_position, 'Length_available'] = 0.0
        self.availability_quay_positions.loc[quay_position, 'Occupant'] = vessel.id

        quay_claimed = self.availability_quay_positions[self.availability_quay_positions.Length_available == 0.0]
        length_claimed = (quay_claimed.Distance_stop - quay_claimed.Distance_start).sum()

        next_quay_position = self.availability_quay_positions.index.get_loc(quay_position) + 1
        if next_quay_position in self.availability_quay_positions.index:
            distance_stop = self.availability_quay_positions.loc[quay_position, 'Distance_stop']
            distance_start = self.availability_quay_positions.loc[next_quay_position, 'Distance_start']
            if distance_start != distance_stop:
                length = distance_stop - distance_start
                new_position = pd.DataFrame({'Distance_start': [distance_stop],
                                             'Distance_stop': [distance_start],
                                             'Length_available': [length],
                                             'Occupant': [None]})
                self.availability_quay_positions = pd.concat([self.availability_quay_positions.iloc[:next_quay_position], new_position, self.availability_quay_positions.iloc[next_quay_position:]])

        elif length_claimed != self.length:
            length = self.length - length_claimed
            new_position = pd.DataFrame({'Distance_start': self.availability_quay_positions.loc[quay_position, 'Distance_stop'],
                                         'Distance_stop':  self.length,
                                         'Length_available': [length],
                                         'Occupant': [None]})
            self.availability_quay_positions = pd.concat([self.availability_quay_positions.iloc[:next_quay_position], new_position])

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
                                                                                                                  'Length_available':'sum',
                                                                                                                  'Occupant': 'first'})
        self.availability_quay_positions = self.availability_quay_positions.drop(columns=['group'])

        # Determine the new current maximum available length of the terminal
        new_level = self.calculate_quay_length_level()

        # Put back length to resource so that the level equals the berth position with the largest length (unless it has not changed)
        if old_level != new_level:
            yield self.resource.put(new_level - old_level)





