from opentnsim.core import  Identifiable, Log, Movable, PriorityFilterStore, VesselProperties
from opentnsim.graph.utils import node_path_to_edge_path
from opentnsim.port.mixins.port import IsPortComponent
from opentnsim.port.mixins.berth import IsQuay, IsJetty
from opentnsim.port.utils import (
    determine_new_route_for_vessel, 
    get_terminal_berths_availability,
)

import numpy as np
import pandas as pd
import datetime
import simpy
import uuid

class TerminalHandable(Movable, Identifiable, VesselProperties):
    """Mixin class: Something can be operated by a terminal
    
    Used for object that calls at a terminal

    Parameters
    ----------
    terminal: IsTerminal object
        speed of the object (in m/s)
    berthing_time: float
        time for the object to berth at a terminal [in minutes]
    loading_time: float
        time for the object to perform loading operations at a terminal [in hours]
    deberthing_time: float
        time for the object to berth at a terminal [in minutes]
    next_destination: str, optional
        the next destination of the vessel as a node in the graph
    next_terminals: list of IsTerminal objects
        next terminals the object will call at after finishing being handled at the defined terminal
    next_berthing_times: list of floats, optional
        next berthing times [in minutes]
    next_loading_times: list of floats, optional
        next loading times [in hours]
    next_deberthing_times: list of floats, optional
        next deberthing times [in minutes]

    Attributes
    ----------
    terminal: IsTerminal object
        speed of the object (in m/s)
    berthing_time: float
        time for the object to berth at a terminal [in minutes]
    loading_time: float
        time for the object to perform loading operations at a terminal [in hours]
    deberthing_time: float
        time for the object to berth at a terminal [in minutes]
    next_destination: str, optional
        the next destination of the vessel as a node in the graph
    next_terminals: list of IsTerminal objects
        next terminals the object will call at after finishing being handled at the defined terminal
    next_berthing_times: list of floats, optional
        next berthing times [in minutes]
    next_loading_times: list of floats, optional
        next loading times [in hours]
    next_deberthing_times: list of floats, optional
        next deberthing times [in minutes]

    Methods
    -------
    pass_terminal:
        general terminal operation
    request_berth_access: 
        requesting berth access
    release_berth_access: 
        releasting the berth access request
    berthing:
        the berthing operation
    loading: 
        the loading operation
    deberthing: 
        the deberthing operation
    """
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
        """Initialization"""
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

    def pass_terminal(self, edge):
        """Stepwise execution of operations to pass a terminal
                                        
        Input
        -----
        edge: tuple of two strings
            edge in the environment.graph.edges
        """
        # get destination
        destination = edge[1]

        # if vessel not at a berth -> do nothing (return)
        if 'Berth' not in self.env.graph.nodes[destination].keys():
            return

        # if vessel not at its assigned berth -> do nothing (return)
        berths = self.env.graph.nodes[destination]['Berth']
        if not self.berth in berths:
            return

        # 1) requesting berth access (formality -> berth has been assigned to the vessel before)
        yield from self.request_berth_access()

        # 2) berthing operation
        yield from self.berthing(destination)

        # 3) loading operation (including request to port authority for leaving terminal after loading operation)
        # - set new trip index and id
        self.trip_index += 1
        self.trip_id = str(uuid.uuid4())
        # - perform loading operation
        request_to_leave_terminal = yield from self.loading(destination)
        yield request_to_leave_terminal

        # 4) deberthing operation
        yield from self.deberthing(destination)

        # 5) releasing berth request (formality -> always possible)
        yield from self.release_berth_access()

        # 6) prepare next trip
        # - set next berthing time (if vessel will move to other terminal)
        if len(self.next_berthing_times):
            self.berthing_time = self.next_berthing_times[0]
            self.next_berthing_times = self.next_berthing_times[1:]
            self.loading_time = self.next_loading_times[0]
            self.next_loading_times = self.next_loading_times[1:]
            self.deberthing_time = self.next_deberthing_times[0]
            self.next_deberthing_times = self.next_deberthing_times[1:]

        # 7) starting next trip
        self.env.process(self.move())
        raise simpy.exceptions.Interrupt('Route of vessel has changed.')

    def request_berth_access(self):
        """Requesting a berth"""
        # requesting berth access
        yield from self.berth.grant_vessel_access(self)

        # change berth planning status to 'at berth'
        self.berth.terminal.update_vessel_berth_planning_status(self, 'at berth')

    def release_berth_access(self):
        """Releasing the berth access"""
        # releasing berth access
        yield from self.berth.release_vessel_access(self)

        # change berth planning status to 'completed'
        self.berth.terminal.update_vessel_berth_planning_status(self, 'completed')

    def berthing(self, origin):
        """Berthing operation of a vessel
                                
        Input
        -----
        origin: str
            node in the environment.graph
        """
        # set current time
        current_time = datetime.datetime.fromtimestamp(self.env.now)
        
        # log berthing start
        self.log_entry_v0("Berthing start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])

        # set berthing time and berthing stop time    
        berthing_time =  pd.Timedelta(seconds=self.berthing_time*60)    
        berthing_stop_time = current_time + berthing_time
        
        # create berthing process
        berthing_process = self.env.timeout(berthing_time.total_seconds())
        
        # yield berthing process (when interrupted, continues where left)
        while True:
            try:
                # yields berthing process (when done breaks while True loop)
                yield berthing_process
                break
            except simpy.Interrupt:
                # Determines current time
                current_time = datetime.datetime.fromtimestamp(self.env.now)
                # Determines remaining time for vessel to berth
                remaining_berthing_time = berthing_stop_time - current_time
                # Creates new berthing process
                berthing_process = self.env.timeout(remaining_berthing_time.total_seconds())

        # log berthing stop
        self.log_entry_v0("Berthing stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])

    def loading(self, origin):
        """Loading operation of a vessel
                        
        Input
        -----
        origin: str
            node in the environment.graph

        Returns
        -------
        request_to_leave_terminal: simpy.env.process
            request the port authority to leave the terminal after handling
        """
        # set current time
        current_time = datetime.datetime.fromtimestamp(self.env.now)

        # log loading start
        self.log_entry_v0("Loading start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])

        # determine loading and deberthing durations, and loading stop time
        loading_time = pd.Timedelta(seconds=self.loading_time*3600)
        deberthing_time = pd.Timedelta(seconds = self.deberthing_time*60)
        loading_stop_time = current_time + loading_time

        # create loading process
        loading_process = self.env.timeout(loading_time.total_seconds())

        # create process for request_to_leave_terminal
        new_route = determine_new_route_for_vessel(self)
        self.route = new_route
        self.edge_route = node_path_to_edge_path(self.env.graph, new_route)
        self.bound = 'outbound'
        request_to_leave_terminal = self.env.process(
            self.request_port_exit(
                origin, 
                parallel_process = loading_process, 
                delay = deberthing_time,
            )
        )

        # yield loading process (when interrupted, continues where left)
        while True:
            try:
                yield loading_process
                break
            except simpy.Interrupt:
                current_time = datetime.datetime.fromtimestamp(self.env.now)
                remaining_loading_time = loading_stop_time - current_time
                loading_process = self.env.timeout(remaining_loading_time.total_seconds())

        # Log loading stop  
        self.log_entry_v0("Loading stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])

        return request_to_leave_terminal

    def deberthing(self, origin):
        """Deberthing operation of a vessel
                
        Input
        -----
        origin: str
            node in the environment.graph
        """
        # set current time
        current_time = datetime.datetime.fromtimestamp(self.env.now)

        # log deberthing start
        self.log_entry_v0("Deberthing start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])

        # set deberthing duration and deberthing stop time
        deberthing_time = pd.Timedelta(seconds=self.deberthing_time*60)
        deberthing_stop_time = current_time + deberthing_time

        # creating deberthing process
        deberthing_process = self.env.timeout(deberthing_time.total_seconds())
        
        # yield deberhting process (when interrupted, continues where left)
        while True:
            try:
                yield deberthing_process
                break
            except simpy.Interrupt:
                current_time = datetime.datetime.fromtimestamp(self.env.now)
                remaining_deberthing_time = deberthing_stop_time - current_time
                deberthing_process = self.env.timeout(remaining_deberthing_time.total_seconds())

        # log deberthing stop    
        self.log_entry_v0("Deberthing stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])



class IsTerminal(Log, Identifiable, IsPortComponent):
    """Mixin class: Terminal object
        
    Used to create a terminal

    Parameters 
    ----------
    berths: list of IsBerth (IsQuay and/or IsJetty) objects
        the berths that the terminal operate


    Attributes
    ----------
    berths: simpy.FilterStore
        storage of all the berths the terminal operates
    berth_availability: pandas.DataFrame
        available quay lenghts and berths over time
    berth_planning: pandas.DataFrame
        planning of the vessel calls at the berths that the terminal operates

    Methods
    -------        
    add_vessel_to_berth_planning:
        adds a vessel to the berth planning
    provide_availability_info_per_berth: 
        provides the first instances of berths being available for a vessel
    provide_terminal_availability_info: 
        provides the availability of a terminal as a whole for a vessel
    update_berth_planning:
        updates the berth planning
    update_vessel_berth_planning_status:
        updates the status of a vessel in the berth planning
    """
    def __init__(self, berths, *args,**kwargs):
        """Initialization"""
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
        """Updates the berth planning and availability with a new estimated time of departure
                                
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
        # set berth and obtain berth planning and berthing information of the vessel
        berth = vessel.berth
        berth_planning = self.berth_planning
        berth_info = berth_planning[berth_planning.operation_id == vessel.berth_operation_id]

        # if vessel not in berth planning -> do nothing (return)
        if berth_info.empty:
            return

        # determines index in berth planning df
        index = berth_info.iloc[-1].name

        # overwrite berth availability with actual availability (minus vessel in question)
        self.berth_availability = get_terminal_berths_availability(vessel, origin, self, delay, bool = False)

        # get ETA at port and berth, and the operation the vessel may be waiting for (this information did not change)
        eta_port = berth_planning.loc[index, 'ETA_port']
        eta_berth = berth_planning.loc[index, 'ETA_berth']
        operation_waiting_for = berth_planning.loc[index, 'operation_waiting_for']

        # add vessel to berth planning and berth_availability with updated ETD at berth
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
        # if no berth -> do nothing (return)
        if not hasattr(vessel, 'berth') or vessel.berth is None:
            return

        # change the status by accessing the berth planning and berth info of the vessel
        berth_planning = self.berth_planning
        berth_info = berth_planning.loc[berth_planning.operation_id == vessel.berth_operation_id]
        if not berth_info.empty:
            berth_info_index = berth_info.iloc[-1].name
            berth_planning.loc[berth_info_index, 'status'] = new_status
