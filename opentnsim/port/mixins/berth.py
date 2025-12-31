from opentnsim.core import  HasResource, Identifiable, Log, HasLength, Movable
from opentnsim.graph.mixins import OnNode, OnEdge

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial import ConvexHull

def add_berth_to_graph(berth):
    node_info = berth.env.graph.nodes[berth.node]
    if 'Berth' not in node_info.keys():
        node_info['Berth'] = []
    node_info['Berth'].append(berth)

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

class IsBerth(OnNode, Identifiable, Log):

    def __init__(self, berth_length, depth, *args, **kwargs):
        self.depth = depth
        self.berth_length = berth_length
        super().__init__(*args, **kwargs)
        add_berth_to_graph(self)
        self.historic_berth_planning = pd.DataFrame(data={0: [None, None], berth_length: [None, None]},
                                                    index=[self.env.simulation_start, self.env.simulation_stop])
        self.occupying_vessels = pd.DataFrame(columns=["Vessel_L", "Vessel_B", "Vessel_T", "Time_at_berth_start", "Time_at_berth_stop"])
        self.queue = pd.DataFrame(columns=["Vessel_L", "Vessel_B", "Vessel_T", "Waiting_start_time", "Waiting_stop_time","Arrival_time_at_berth"])

    def add_berth_position_to_historic_planning(self, vessel, quay_position=0, delay=0, leaving=False):
        current_time = datetime.datetime.fromtimestamp(self.env.now) + pd.Timedelta(seconds=delay)
        if isinstance(self,IsQuay):
            quay_position = self.availability_quay_positions.loc[quay_position]
            position_start = quay_position.Distance_start + 0.001
            position_stop = quay_position.Distance_stop - 0.001
        elif isinstance(self,IsJetty):
            gap = (self.berth_length - vessel.L) / 2
            position_start = gap
            position_stop = self.berth_length - gap

        if not leaving:
            if position_start not in self.historic_berth_planning.columns:
                self.historic_berth_planning.loc[:, position_start] = None
            if position_stop not in self.historic_berth_planning.columns:
                self.historic_berth_planning.loc[:, position_stop] = None

        self.historic_berth_planning.loc[current_time, position_start] = vessel.id
        self.historic_berth_planning.loc[current_time, position_stop] = vessel.id

        if leaving:
            for position_columm in [position_start, position_stop]:
                notna = self.historic_berth_planning[position_columm].notna()
                if not notna.any():
                    continue
                first_idx = notna.idxmax()
                last_idx = notna[::-1].idxmax()
                mask = (self.historic_berth_planning.index >= first_idx) & (self.historic_berth_planning.index <= last_idx)
                self.historic_berth_planning.loc[mask, position_columm] = self.historic_berth_planning.loc[mask, position_columm].bfill()
            self.historic_berth_planning = self.historic_berth_planning.apply(horizontal_fill_between_equals, axis=1)

        self.historic_berth_planning = self.historic_berth_planning.sort_index()
        self.historic_berth_planning = self.historic_berth_planning.reindex(sorted(self.historic_berth_planning.columns), axis=1)
        self.historic_berth_planning[self.historic_berth_planning.isna()] = None
        self.historic_berth_planning = self.historic_berth_planning.sort_index()

    def add_vessel_to_occupants_df(self, vessel, time_at_berth_start, time_at_berth_stop):
        self.occupying_vessels.loc[vessel.id, :] = [vessel.L, vessel.B, vessel.T, time_at_berth_start, time_at_berth_stop]

    def add_vessel_to_queue(self, vessel, time_waiting_start, time_waiting_stop, arrival_time_at_berth):
        self.queue.loc[vessel.id, :] = [vessel.L, vessel.B, vessel.T, time_waiting_start, time_waiting_stop, arrival_time_at_berth]

    def remove_vessel_from_occupants_df(self, vessel):
        self.occupying_vessels = self.occupying_vessels.drop(vessel.id)

    def remove_vessel_from_queue(self, vessel):
        self.queue = self.queue.drop(vessel.id)

    def update_occupants_df(self, vessel, time_at_berth_start = None, time_at_berth_stop = None):
        if time_at_berth_start is None:
            time_at_berth_start = self.occupying_vessels.loc[vessel.id, "Time_at_berth_start"]
        if time_at_berth_stop is None:
            time_at_berth_stop = self.occupying_vessels.loc[vessel.id, "Time_at_berth_stop"]
        self.add_vessel_to_occupants_df(vessel, time_at_berth_start, time_at_berth_stop)

    def update_queue(self, vessel, time_waiting_start, time_waiting_stop, arrival_time_at_berth):
        self.add_vessel_to_queue(vessel, time_waiting_start, time_waiting_stop, arrival_time_at_berth)
        self.queue = self.queue.sort_values('Waiting_start_time')

    def plot_historic_berth_planning(self):
        fig, ax = plt.subplots()
        historic_quay_planning_plot = self.historic_berth_planning.stack().reset_index()
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
        plt.xlim(0,self.berth_length)
        plt.ylabel("Time")
        if isinstance(self, IsQuay):
            plt.xlabel("Quay length [m]")
        elif isinstance(self, IsBerth):
            plt.xlabel("Berth length [m]")
        plt.close()
        return fig


class IsJetty(IsBerth, HasResource):
    def __init__(self, length, capacity=1, *args, **kwargs):
        self.capacity = capacity
        super().__init__(berth_length=length,nr_resources=capacity, *args, **kwargs)

    def request_access(self, vessel):
        self.add_berth_position_to_historic_planning(vessel, delay = vessel.berthing_time)
        yield from []


    def release_access(self, vessel):
        self.add_berth_position_to_historic_planning(vessel, leaving = True)
        yield from []


    def update_planning(self, vessel, new_release_time):
        pass


class IsQuay(IsBerth, HasLength):
    def __init__(self, length, capacity = np.inf, *args, **kwargs):
        self.capacity = capacity
        super().__init__(berth_length=length,length=length,remaining_length=length,*args, **kwargs)
        self.availability_quay_positions = pd.DataFrame(data=[[0, length, length, None, None, None]],
                                                        columns=['Distance_start','Distance_stop','Length_available','Occupant','Time_start','Time_stop'])

    def request_access(self, vessel):
        quay_position = self.select_quay_position(vessel)
        time_start = datetime.datetime.fromtimestamp(vessel.env.now)
        time_stop = time_start + pd.Timedelta(seconds = vessel.berthing_time*60 +
                                                        vessel.loading_time*3600 +
                                                        vessel.deberthing_time*60)
        yield from self.adjust_availability_quay_positions(vessel, quay_position, time_start, time_stop)
        self.add_berth_position_to_historic_planning(vessel, quay_position, delay = vessel.berthing_time)


    def release_access(self, vessel):
        quay_position = self.find_quay_position(vessel)
        self.add_berth_position_to_historic_planning(vessel, quay_position, leaving=True)
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


    def update_planning(self, vessel, new_release_time):
        new_release_time += pd.Timedelta(seconds=vessel.deberthing_time*60)
        vessel.terminal.berth_planning.loc[new_release_time, self.name] = np.nan
        with pd.option_context("future.no_silent_downcasting", True):
            vessel.terminal.berth_planning = vessel.terminal.berth_planning.ffill().infer_objects(copy=False)

        vessel.terminal.berth_planning =  vessel.terminal.berth_planning.sort_index()
        quay_position = self.find_quay_position(vessel)

        old_release_time = self.availability_quay_positions.loc[quay_position, 'Time_stop']
        delay = (new_release_time-old_release_time).total_seconds()
        self.availability_quay_positions.loc[quay_position, 'Time_stop'] += pd.Timedelta(seconds=delay)

        current_level = self.calculate_quay_length_level()
        availability_quay_positions_copy = self.availability_quay_positions.copy()
        mask = availability_quay_positions_copy.Occupant.notna()
        time_start_min = availability_quay_positions_copy[mask]['Time_start'].min()
        times = np.sort(np.unique(availability_quay_positions_copy[mask]['Time_stop'].values.flatten()))

        mask = vessel.terminal.berth_planning.index >= time_start_min
        vessel.terminal.berth_planning.loc[vessel.terminal.berth_planning[mask].index, self.name] = current_level
        for time in times:
            vessels_leaving = availability_quay_positions_copy[availability_quay_positions_copy.Time_stop == time]
            for quay_position,vessel_leaving in vessels_leaving.iterrows():
                self.remove_vessel_from_planning(quay_position)
            new_level = self.calculate_quay_length_level()
            mask = vessel.terminal.berth_planning.index >= time
            vessel.terminal.berth_planning.loc[vessel.terminal.berth_planning[mask].index, self.name] = new_level
        self.availability_quay_positions = availability_quay_positions_copy
        self.update_occupants_df(vessel, time_at_berth_stop = new_release_time)


    def adjust_availability_quay_positions(self, vessel, quay_position, time_start, time_stop):
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
        self.availability_quay_positions.loc[quay_position, 'Time_start'] = time_start
        self.availability_quay_positions.loc[quay_position, 'Time_stop'] = time_stop

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
                                             'Occupant': [None],
                                             'Time_start': [None],
                                             'Time_stop': [None]})
                self.availability_quay_positions = pd.concat([self.availability_quay_positions.iloc[:next_quay_position], new_position, self.availability_quay_positions.iloc[next_quay_position:]])

        elif length_claimed != self.berth_length:
            length = self.berth_length - length_claimed
            new_position = pd.DataFrame({'Distance_start': self.availability_quay_positions.loc[quay_position, 'Distance_stop'],
                                         'Distance_stop':  self.berth_length,
                                         'Length_available': [length],
                                         'Occupant': [None],
                                         'Time_start': [None],
                                         'Time_stop': [None]})
            self.availability_quay_positions = pd.concat([self.availability_quay_positions.iloc[:next_quay_position], new_position])

        self.availability_quay_positions = self.availability_quay_positions.reset_index(drop=True)

        # Determine the new current maximum available length of the terminal
        new_level = self.calculate_quay_length_level()

        # Claim length of resource so that the level equals the berth position with the largest length (unless it has not changed)
        if old_level != new_level:
            yield self.length.get(old_level - new_level)


    def remove_vessel_from_planning(self, position):
        # Drop vessel from layout
        self.availability_quay_positions.loc[position, 'Occupant'] = None

        # Combine subsequent empty positions
        self.availability_quay_positions['Occupant_filled'] = self.availability_quay_positions['Occupant'].fillna('NA')
        self.availability_quay_positions['group'] = (
                    self.availability_quay_positions['Occupant_filled'] != self.availability_quay_positions[
                'Occupant_filled'].shift()).cumsum()
        self.availability_quay_positions = self.availability_quay_positions.groupby('group', as_index=False).agg(
            {'Distance_start': 'first',
             'Distance_stop': 'last',
             'Length_available': 'sum',
             'Occupant': 'first',
             'Time_start': 'first',
             'Time_stop': 'first', })
        mask = self.availability_quay_positions.Occupant.isna()
        available_lenghts = self.availability_quay_positions[mask].apply(lambda x: x.Distance_stop - x.Distance_start,
                                                                         axis=1)
        self.availability_quay_positions.loc[
            self.availability_quay_positions[mask].index, 'Length_available'] = available_lenghts
        self.availability_quay_positions.loc[self.availability_quay_positions[mask].index, 'Time_start'] = None
        self.availability_quay_positions.loc[self.availability_quay_positions[mask].index, 'Time_stop'] = None
        self.availability_quay_positions = self.availability_quay_positions.drop(columns=['group'])


    def readjust_availability_quay_positions(self, position):
        """ Function that readjusts the available quay lenghts and positions given a release of a request of a vessel at a given position

            Input:
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class
                - position: quay position index at which the vessel is located at the quay with respect to the other vessels

        """
        # Determine the current maximum available length of the terminal
        old_level = self.calculate_quay_length_level()

        self.remove_vessel_from_planning(position)

        # Determine the new current maximum available length of the terminal
        new_level = self.calculate_quay_length_level()

        # Put back length to resource so that the level equals the berth position with the largest length (unless it has not changed)
        if old_level != new_level:
            yield self.length.put(new_level - old_level)