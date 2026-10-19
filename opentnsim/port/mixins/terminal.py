from opentnsim.core import  Identifiable, Log, Movable, PriorityFilterStore, VesselProperties
from opentnsim.graph.utils import node_path_to_edge_path
from opentnsim.port.mixins.port import IsPortComponent
from opentnsim.port.mixins.berth import IsQuay, IsJetty
from opentnsim.port.calculations import (
    calculate_estimated_arrival_and_departure_times_at_port_and_berth, 
    calculate_sailing_time_to_berth,
)
from opentnsim.port.utils import (
    determine_new_route_for_vessel, 
    get_terminal_berths_availability,
    get_berth_availability_without_vessel
)

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
        loading_process = self.env.timeout(self.loading_time*3600)
        request_port_exit = self.request_port_exit(
            origin, 
            parallel_process = loading_process, 
            delay = pd.Timedelta(seconds = self.deberthing_time*60),
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


class IsTerminal(Log, Identifiable, IsPortComponent):

    def __init__(self, berths, *args,**kwargs):
        """ Creates a terminal object

        Input
        -----
        berths: 
            a list of IsBerth (IsJetty and/or IsQuay) objects
        """
        super().__init__(*args, **kwargs)

        # add terminal to port terminals
        self.port.terminals[self.name] = self

        # add berths as filterstore
        self.berths = PriorityFilterStore(env=self.env)
        for berth in berths:
            berth.terminal = self
            self.berths.put(berth)

        # get berth_names and berth_capacities
        berth_names = [berth.name for berth in self.berths.items]
        berth_capacities = [berth.berth_length if isinstance(berth, IsQuay) else 1 for berth in self.berths.items]

        # get quays and jetties type berths
        quay_berths = [berth.name for berth in self.berths.items if isinstance(berth, IsQuay)]
        jetty_berths = [berth.name for berth in self.berths.items if isinstance(berth, IsJetty)]

        # berth availability dataframe
        self.berth_availability = pd.DataFrame(columns=berth_names)
        self.berth_availability.loc[pd.Timestamp(self.env.simulation_start)] = berth_capacities
        self.berth_availability.loc[pd.Timestamp(self.env.simulation_stop)] = berth_capacities
        self.berth_availability[quay_berths] = self.berth_availability[quay_berths].astype(float)
        self.berth_availability[jetty_berths] = self.berth_availability[jetty_berths].astype(int)

        # berth planning
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
        

    def add_vessel_to_berth_planning(
            self, 
            vessel, 
            berth, 
            eta_port,
            eta_berth,
            etd_berth,
            operation_waiting_for = None,
            ):
        """Add a vessel to the berth planning of the terminal
        
        Input
        -----
        vessel: Vessel object that is TerminalHandable
            the vessel in question
        berth: IsBerth (IsJetty or IsQuay) object
            the berth the vessel is added to 
        eta_port: pandas.Timestamp
            estimated time of arrival at the port
        eta_berth: pandas.Timestamp
            estimated time of arrival at berth
        etd_berth: pandas.Timestamp
            estimated time of departure from berth
        operation_waiting_for: str
            the id of the berthing operation to which the vessel has to wait
        """
        # create initial space for vessel in planning (delay guarantees that vessel fits in this planning)
        berth_availability = self.berth_availability.copy()
        if isinstance(berth, IsQuay): # quays: nans will be later filled by other algorithm (prevents vessel from provide faulty length values)
            berth_availability.loc[eta_berth, berth.name] = np.nan
            berth_availability.loc[etd_berth, berth.name] = np.nan
        elif isinstance(berth, IsJetty): # jetties: 0 = not available, 1 = available
            berth_availability.loc[eta_berth, berth.name] = 0
            berth_availability.loc[etd_berth, berth.name] = 1
            mask = (berth_availability.index > eta_berth)&(berth_availability.index < etd_berth)
            berth_availability.loc[berth_availability[mask].index, berth.name] = 0
            berth_availability[berth.name] = berth_availability[berth.name].astype(int)

        # update planning by ffill and bfill values (guarantees that planning did not result in nans, extrapolates existing quay length values)
        berth_availability = berth_availability.sort_index()
        with pd.option_context("future.no_silent_downcasting", True):
            berth_availability = berth_availability.ffill().infer_objects(copy=False)
            berth_availability = berth_availability.bfill().infer_objects(copy=False)

        # subtracts vessel length from existing quay length values 
        if isinstance(berth, IsQuay):
            mask = (berth_availability.index >= eta_berth)&(berth_availability.index < etd_berth)
            berth_availability.loc[berth_availability[mask].index, berth.name] -= vessel.L
        
        # set parameters and determine planning index 
        berth_priority = 0 # for now: all vessel priorities are equal
        status = 'planned'
        if hasattr(vessel, 'berth_operation_id') and vessel.berth_operation_id not in self.berth_planning.operation_id.values:
            # assign berth and berth operation id to vessel
            vessel.berth = berth
            vessel.berth_operation_id = str(uuid.uuid4())
            index = len(self.berth_planning)
        else:
            index = self.berth_planning[self.berth_planning.operation_id == vessel.berth_operation_id].iloc[-1].name

        # add vessel to the berth planning
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
        berth.add_vessel_to_occupants_df(vessel, eta_berth, etd_berth)

        # override berth availability
        self.berth_availability = berth_availability


    def provide_availability_info_per_berth(self, vessel, origin, delay = 0.):
        """Determines availability information per fit berth for the operation of the vessel in question
        
        Input
        -----
        vessel: Vessel object that is TerminalHandable
            the vessel in question
        origin: str
            the node of the vessel in env.graph
        delay: pandas.Timedelta
            the delay of the vessel

        Returns
        -------
        availability_info_per_berth: pandas.DataFrame
            information per berth of its first available slot for the vessel
        """
        # set current time
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now)) + pd.Timedelta(seconds = delay)

        # determine vessel's occupation time at berth
        berthing_time = vessel.berthing_time*60
        loading_time = vessel.loading_time*3600
        deberthing_time = vessel.deberthing_time*60
        occupation_duration = pd.Timedelta(seconds=berthing_time + loading_time + deberthing_time)

        # create dataframe
        availability_info_per_berth = pd.DataFrame(columns=[
            'time_start','time_stop','waiting_time','berth_length','operation_waiting_for'
            ])

        # determine availability of berths that are accessible for the vessel (fit)
        terminal_berths = self.berths.items
        fit_berths = [berth for berth in terminal_berths if (berth.depth >= vessel.T) and (berth.berth_length >= vessel.L)]
        fit_berth_availability = get_terminal_berths_availability(vessel, origin, self)

        # loop over berths
        for berth in fit_berths:
            berth_name = berth.name

            # get availability of berth in question
            berth_available = fit_berth_availability[berth_name]

            # determine start time at which berth becomes available
            mask_berth_available = (berth_available == True)
            berth_availability_start_times = berth_available[mask_berth_available].index.to_list()
            if not len(berth_availability_start_times):
                 continue

            # loop over availability start time of berth in qeustion
            for berth_availability_start_time in berth_availability_start_times:                       
                # determine berth availability stop time
                berth_availability_stop_times = berth_available[
                    (berth_available.index > berth_availability_start_time)&(berth_available == False)]
                if not len(berth_availability_stop_times):
                    berth_availability_stop_time = berth_available.index[-1]
                else:
                    berth_availability_stop_time = berth_availability_stop_times.index[0]

                # skip berth if window of availability is smaller than the operation of the vessel requires
                berth_availability_duration = berth_availability_stop_time - berth_availability_start_time
                if berth_availability_duration < occupation_duration:
                    continue
                    
                # determine if vessel will wait for another terminal operation to end
                operation_waiting_for = None
                operation_waiting_for_df = self.berth_planning[
                    self.berth_planning.ETD_berth == berth_availability_start_time]
                if not operation_waiting_for_df.empty:
                    operation_waiting_for = operation_waiting_for_df.iloc[0].operation_id

                # determine the waiting time for the vessel
                waiting_time = berth_availability_start_time - current_time
                waiting_time = np.max([pd.Timedelta(seconds=0), waiting_time])
                if waiting_time: # prevent overlap in planning
                    berth_availability_start_time += pd.Timedelta(seconds = 1)
                    berth_availability_stop_time += pd.Timedelta(seconds = 1)
                    waiting_time += pd.Timedelta(seconds = 1) 

                # add potential slot to dataframe
                availability_info_per_berth.loc[berth_name,:] = [
                    berth_availability_start_time,
                    berth_availability_stop_time,
                    waiting_time,
                    berth.berth_length,
                    operation_waiting_for]
                
                # assess next berth
                break

        return availability_info_per_berth


    def provide_terminal_availability_info(self, vessel, origin, berth = None, delay = pd.Timedelta(seconds = 0)):
        """Provides the availability of the terminal
                                
        Input
        -----
        vessel: Vessel object that is TerminalHandable
            the vessel in question
        origin: str
            the node of the vessel in env.graph
        berth: IsBerth (IsJetty or IsQuay) object
            the berth the vessel is added to 
        delay: pandas.Timedelta
            the delay of the vessel

        Returns
        -------
        terminal_availability : pandas.DataFrame
            availability of the terminal
        """
        # get terminal availability of fit berth
        terminal_berths_availability = get_terminal_berths_availability(vessel, origin, self, delay)   

        # get list of fit berths (or fit berth if berth has been selected)
        terminal_berths = self.berths.items
        if berth is None:
            fit_berths = [berth.name for berth in terminal_berths if (berth.depth >= vessel.T) and (berth.berth_length >= vessel.L)]
        else:
            fit_berths = [berth.name]

        terminal_fit_berths_availability = terminal_berths_availability[fit_berths]

        # get terminal availability
        terminal_availability = pd.DataFrame({'Terminal': terminal_fit_berths_availability.any(axis=1)})
        return terminal_availability


    def update_berth_planning(self, vessel, origin, etd_berth, delay = pd.Timedelta(seconds = 0)):
        """Updates the berth planning with new information
                                
        Input
        -----
        vessel: Vessel object that is TerminalHandable
            the vessel in question
        origin: str
            the node of the vessel in env.graph
        etd_berth: pandas.Timestamp
            the new estimated time of departure from the berth
        delay: pandas.Timedelta
            the delay of the vessel
        """
        berth = vessel.berth
        berth_planning = self.berth_planning
        berth_info = berth_planning[berth_planning.operation_id == vessel.berth_operation_id]
        if berth_info.empty:
            return

        index = berth_info.iloc[-1].name
        self.berth_availability = get_terminal_berths_availability(vessel, origin, self, delay, bool = False)
        eta_port = berth_planning.loc[index, 'ETA_port']
        eta_berth = berth_planning.loc[index, 'ETA_berth']
        operation_waiting_for = berth_planning.loc[index, 'operation_waiting_for']
        self.add_vessel_to_berth_planning(vessel, berth, eta_port, eta_berth, etd_berth, operation_waiting_for)   


    def update_vessel_berth_planning_status(self, vessel, new_status):
        """Updates the status of the vessel in the berth planning
                        
        Input
        -----
        vessel: Vessel object that is TerminalHandable
            the vessel in question
        new_status: str
            the new status of the vessel
        """
        if not hasattr(vessel, 'berth') or vessel.berth is None:
            return

        berth_planning = self.berth_planning
        berth_info = berth_planning.loc[berth_planning.vessel_id == vessel.id]
        if not berth_info.empty:
            berth_info_index = berth_info.iloc[-1].name
            berth_planning.loc[berth_info_index, 'status'] = new_status