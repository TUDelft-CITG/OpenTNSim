from opentnsim.core import  HasResource, Identifiable, Log, HasLength
from opentnsim.graph.mixins import OnNode
from opentnsim.port.visualizations import plot_berth_occupation, plot_vessel_arrival_timeline
from opentnsim.port.calculations import calculate_berth_performance, calculate_availability_quay_positions

import numpy as np
import pandas as pd
import datetime

class IsBerth(OnNode, Identifiable, Log):
    """Mixin class: Berth object
    
    Used to create a berth at a node (as part of a terminal)

    Parameters
    ----------
    berth_length: float
        the length of the berth [in m]
    depth: float
        the depth at the berth [in m]

    Attributes
    ----------
    berth_occupation: pandas.DataFrame
        dataframe that keeps track of the vessels that visited the berth

    Methods
    -------
    add_vessel_to_berth_occupation:
        adds a vessel to the berth occupation dataframe
    plot_berth_occupation:
        creates a visualization of the berth occupation
    plot_vessel_arrival_timeline:
        creates a visualization of the timelines of the vessels that visited the berth
    get_performance:
        provides a table with KPIs of the performance of the berth
    """
    def __init__(self, berth_length, depth, *args, **kwargs):
        """Initialization"""
        self.depth = depth
        self.berth_length = berth_length
        super().__init__(*args, **kwargs)

        # add berth to the graph
        node_info = self.env.graph.nodes[self.node]
        if 'Berth' not in node_info.keys():
            node_info['Berth'] = []
        node_info['Berth'].append(self)

        # create berth availability dataframe
        self.berth_occupation = pd.DataFrame(
            columns = [
                'vessel_id', 
                'operation_id', 
                'time_start', 
                'time_stop', 
                'position_start', 
                'position_stop',
            ]
        )

    def add_vessel_to_berth_occupation(self, vessel, quay_position=None, delay=0, leaving=False):
        """Adds vessel to the berth occupation dataframe
        
        Parameters
        ----------
        vessel: obj
            vessel type object that is TerminalHandable
        quay_position: pd.Series
            instance of self.berth_occupation assigned to the vessel
        delay: float, optional
            delay of the vessel [in s]
        leaving: bool, optional
            indicates if the vessel is leavnig the berth
        """
        # set the current time
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now)) + pd.Timedelta(seconds=delay)

        # determine index
        index = len(self.berth_occupation)
        if vessel.berth_operation_id in self.berth_occupation.operation_id.values:
            index = self.berth_occupation[self.berth_occupation.operation_id == vessel.berth_operation_id].iloc[-1].name

        # if vessel is not leaving -> add vessel to berth_occupation->dataframe
        if not leaving:
            # get the vessl start and stop position at the berth
            if isinstance(self,IsQuay):
                position_start = quay_position.position_start
                position_stop = quay_position.position_stop
            elif isinstance(self,IsJetty):
                gap = (self.berth_length - vessel.L) / 2 #determine gap at sides of vessel (when vessel is smaller then jetty and assuming vessel is centered)
                position_start = gap
                position_stop = self.berth_length - gap

            # calculate estimated time at berth
            time_at_berth = pd.Timedelta(seconds = vessel.berthing_time*60 + vessel.loading_time*3600 + vessel.deberthing_time*60)
            etd = current_time + time_at_berth

            # add info to berth occupation dataframe
            self.berth_occupation.loc[index, :] = [
                vessel.id, 
                vessel.berth_operation_id, 
                current_time, 
                etd, 
                position_start, 
                position_stop,
            ]
        # if vessel is leaving -> update time_stop in berth_occupation-dataframe
        else:
            self.berth_occupation.loc[index, 'time_stop'] = current_time

    def plot_berth_occupation(self, time_start = None, time_stop = None, style = 'matplotlib'):
        """Creates a plot of vessels occupying the berth length (x-axis) over time (y-axis)
        
        Parameters
        ----------
        time_start: pandas.Timestamp, optional
            start time of the plot
        time_stop:pandas.Timestamp, optional
            stop time of the plot
        style: str
            matplotlib or plotly

        Returns
        -------
        fig: matplotlib.figure.Figure or plotly.graph_objs._figure.Figure
            the figure
        """
        fig = plot_berth_occupation(self, time_start, time_stop, style)
        return fig

    def plot_vessel_arrival_timeline(self):
        """Creates a plotly-plot of the timeline of the vessels that arrived at the berth, including sailing, waiting, berthing, and loading processes
        
        Returns
        -------
        fig: plotly.graph_objs._figure.Figure
            the figure
        """
        fig = plot_vessel_arrival_timeline(self)
        return fig

    def get_berth_performance(self, time_start = None, time_stop = None):
        """Calculates and provides various KPIs of the berth's performance

        Returns
        -------
        results: pd.DataFrame
            dataframe with KPIs related to the performance of the berth
        """
        results = calculate_berth_performance(self, time_start, time_stop)
        return results


class IsJetty(IsBerth, HasResource):
    """Mixin class: Jetty object
    
    Used to create a jetty at a node within a terminal. 
    Unlike a quay, which has a floating-point capacity representing its available length, a jetty has an integer capacity representing the number of vessels it can accommodate.

    Parameters
    ----------
    length: float
        the length of the jetty [in m]
    capacity: int, optional
        the capacity in numbers of vessels of the jetty (normally 1 as jetty can serve only one vessel at the time)

    Attributes
    ----------
    capacity: int
        the capacity in numbers of vessels of the jetty (normally 1 as jetty can serve only one vessel at the time)

    Methods
    -------
    grant_vessel_access:
        granting a vessel access to the berth
    release_vessel_access:
        releasing the vessel from the berth
    """
    def __init__(self, length, capacity=1, *args, **kwargs):
        """Initialization"""
        self.capacity = capacity
        super().__init__(berth_length=length,nr_resources=capacity, *args, **kwargs)

    def grant_vessel_access(self, vessel):
        """Granting vessel access to the berth"""
        # requesting the jetty's capacity (simpy.Resource)
        vessel.berth_request = self.resource.request()
        yield vessel.berth_request

        # adding the vessel's arrival to the berth occupation
        self.add_vessel_to_berth_occupation(vessel, delay = vessel.berthing_time)

    def release_vessel_access(self, vessel):
        """Releasing vessel access to the berth"""
        # adding the vessel's departure to the berth occupation
        self.add_vessel_to_berth_occupation(vessel, leaving = True)

        # releaing the jetty's claimed capacity (simpy.Resource)
        yield self.resource.release(vessel.berth_request)


class IsQuay(IsBerth, HasLength):
    """Mixin class: Quay object
        
    Used to create a quay at a node within a terminal. 
    Unlike a jetty, which has an integer capacity representing the number of vessels it can accommodate, a quay has a floating-point capacity representing its available length.

    Parameters
    ----------
    length: float
        the length of the jetty [in m]
    capacity: int, optional
        the capacity in numbers of vessels of the jetty (normally infinite as capacity is variable and depends on the length of the quay)

    Attributes
    ----------
    capacity: int
        the capacity in numbers of vessels of the jetty (normally infinite as capacity is variable and depends on the length of the quay)

    Methods
    -------
    grant_vessel_access:
        granting a vessel access to the berth
    release_vessel_access:
        releasing the vessel from the berth
    provide_quay_location_for_vessel:
        provides a quay location for the vessel in question
    return_quay_location_of_vessel:
        returns the quay location for the vessel in question
    """
    def __init__(self, length, capacity = np.inf, *args, **kwargs):
        """Initialization"""
        self.capacity = capacity
        super().__init__(berth_length=length, length=length, remaining_length=length, *args, **kwargs)

    def grant_vessel_access(self, vessel):
        """Granting vessel access to the berth
        
        Parameters
        ----------
        vessel: obj
            vessel type object that is TerminalHandable
        """
        # provide a quay position for the vessel
        quay_position, length_to_claim = self.provide_quay_location_for_vessel(vessel)

        # requesting part of the quay's length so that simpy.Container.level equals position with largest available length 
        if length_to_claim:
            yield self.length.get(length_to_claim)

        # adding the vessel's arrival to the berth occupation
        self.add_vessel_to_berth_occupation(vessel, quay_position, delay = vessel.berthing_time)

    def release_vessel_access(self, vessel):
        """Releasing vessel access to the berth
        
        Parameters
        ----------
        vessel: obj
            vessel type object that is TerminalHandable
        """
        # adding the vessel's departure to the berth occupation
        self.add_vessel_to_berth_occupation(vessel, leaving=True)
        
        # releasing part of the quay's length so that simpy.Container.level equals position with largest available length 
        _, length_to_return = self.return_quay_location_of_vessel(vessel)
        if length_to_return:
            yield self.length.put(length_to_return)

    def provide_quay_location_for_vessel(self, vessel, time_start = None):
        """Provide a quay location for the vessel in question
        
        Parameters
        ----------
        vessel: obj
            vessel type object that is TerminalHandable
        time_start: pd.Timestamp, optional
            time at which the quay should be available
        """
        # set the time start and stop of the occupation of the position
        if time_start is None:
            time_start = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        minimum_time_at_berth = pd.Timedelta(seconds = vessel.berthing_time*60 + vessel.loading_time*3600 + vessel.deberthing_time*60)
        time_stop = time_start + minimum_time_at_berth

        # calculate the available quay positions
        available_quay_positions = calculate_availability_quay_positions(self, time_start)

        # determine the available quay positions for the vessel
        available_quay_positions_for_vessel = available_quay_positions[
            (available_quay_positions['length_available'] >= vessel.L)&
            (available_quay_positions.time_stop >= time_stop)
        ]

        # get the first available quay position with the least available length
        quay_position = available_quay_positions_for_vessel.sort_values(['time_start', 'length_available'], ascending=[True, True]).iloc[0]

        # set the start and stop position (for now only stop position is modified)
        if quay_position.position_stop - quay_position.position_start > vessel.L:
            quay_position.position_stop = quay_position.position_start + vessel.L

        # set the parameters for the claimed quay position
        available_quay_positions.loc[quay_position.name, 'position_start'] = quay_position.position_start
        available_quay_positions.loc[quay_position.name, 'position_stop'] = quay_position.position_stop
        available_quay_positions.loc[quay_position.name, 'length_available'] = quay_position.length_available - vessel.L

        # get the current available length of the quay (i.e. the length of the position with the maximum length available before the vessel in question has berthed)
        available_length_old = self.length.level

        # determine the new available length of the quay (i.e. the length of the position with the maximum length available after the vessel in question has berthed)
        available_length_new = available_quay_positions.length_available.max()

        # determine the length the vessel should claim from the quay
        length_to_claim = available_length_old - available_length_new
        return quay_position, length_to_claim

    def return_quay_location_of_vessel(self, vessel, time = None):
        """Returns the quay location of the vessel in question to the berth availability
                
        Parameters
        ----------
        vessel: obj
            vessel type object that is TerminalHandable
        time: pd.Timestamp, optional
            time at which the quay is available again
        """
        # if vessel not in berth_occupation-dataframe -> error (should not be possible)
        if vessel.berth_operation_id not in self.berth_occupation.operation_id.values:
            raise ValueError(
                f"Berth operation ID {vessel.berth_operation_id} for vessel "
                f"{vessel} not found in berth occupation."
            )
        
        # set the time
        if time is None:
            time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))

        # calculate the available quay positions
        available_quay_positions = calculate_availability_quay_positions(self)

        # get the current available length of the quay (i.e. the length of the position with the maximum length available before the vessel in question has berthed)
        available_length_old = self.length.level

        # get the information of the berth operation of the vessel in question
        berth_operation_info = self.berth_occupation[
            self.berth_occupation.operation_id == vessel.berth_operation_id
        ].iloc[-1]

        # set the new available quay position
        # - if newly available position lies in continuation of another current available position -> modify this last position
        if berth_operation_info.position_start in available_quay_positions.position_stop.values:
            position = available_quay_positions[
                available_quay_positions.position_stop == berth_operation_info.position_start
            ].iloc[-1]
            position.position_stop = berth_operation_info.position_stop
        # - else, if newly available position lies directly ahead of another current available position -> modify this last position
        elif berth_operation_info.position_stop in available_quay_positions.position_start.values:
            position = available_quay_positions[
                available_quay_positions.position_start == berth_operation_info.position_stop
            ].iloc[-1]
            position.position_start = berth_operation_info.position_start
        # - else: newly available position is isolated -> create add position with new index
        else:
            new_index = available_quay_positions.index.max() + 1
            position = pd.Series(
                {'position_start': berth_operation_info.position_start, 
                 'position_stop': berth_operation_info.position_stop, 
                 'time_start': time,
                 'time_stop': vessel.env.simulation_stop,
                 'length_available': vessel.L,},
                name=new_index
            )

        # set new position information
        position.time_start = time
        position.time_stop = vessel.env.simulation_stop
        position.length_available = position.position_stop - position.position_start

        # add position to the available quay position
        available_quay_positions.loc[position.name] = position

        # determine the new available length of the quay (i.e. the length of the position with the maximum length available after the vessel in question has berthed)
        available_length_new = available_quay_positions.length_available.max()

        # determine the length the vessel should release from the quay
        length_to_return = available_length_new - available_length_old
        return available_quay_positions, length_to_return