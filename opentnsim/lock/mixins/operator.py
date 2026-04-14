import datetime
import functools
import math
import numpy as np
import pandas as pd
import simpy

from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from opentnsim.lock.calculations import (
    calculate_levelling_time,
    calculate_time_to_open_gate,
    calculate_sailing_time_to_approach_point,
    calculate_lock_salinity_and_saltmass
)
from opentnsim.lock.utils import (
    _get_operation_info,
    _get_next_operations,
    _get_vessel_departure_start_delay,
    _check_if_vessel_is_first_vessel,
    _check_if_vessel_is_last_vessel,
    _get_vessel_sailing_in_speed,
    _get_lock_operation_to_and_from_node,
    _check_if_lock_chamber_is_next_lock_complex_object,
    determine_if_gate_is_closed,
    determine_if_gate_can_be_closed,
)

class IsLockChamberOperator:
    """The lock chamber operator operates one chamber of the lock."""

    def __init__(self,
                 close_gate_before_vessel_is_laying_still = False,
                 min_vessels_in_operation=0,
                 max_vessels_in_operation=math.inf,
                 clustering_time=0.5 * 60 * 60,
                 water_level_difference_limit_to_open_gate=0.05,
                 start_sailing_out_time_after_gate_have_been_opened=0.0, # a float that is the time that the vessel wait to start sailing out of the lock after the gate have been opened after levelling [s]
                 minimum_advance_to_open_gate=600.0, # a float that is the time that the gate are opened before a vessel arrives at the gate [s]
                 minimum_delay_to_close_gate=120.0, # a float that is the time that the gate are closed after a vessel has sailed through the gate [s]
                 minimum_time_between_operations=0.0, # a float that is the minimum required time between lock operations that the lock gate can be both closed (to reduce salt intrusion) [s]
                 sailing_in_time_gap_through_gate=180.0, # a float that is the time gap after which the next vessel can sail into the lock through the lock gate (after another vessel has sailed through to enter the lock) [s]
                 sailing_out_time_gap_through_gate=180.0, # a float that is the time gap after which the next vessel can sail out of the lock through the lock gate (after another vessel has sailed through to leave the lock)[s]
                 sailing_in_time_gap_after_berthing_previous_vessel=0.0, # a float that is the time gap after which the next vessel can sail into the lock (after another vessel has berthed) [s]
                 sailing_out_time_gap_after_berthing_previous_vessel=0.0, # a float that is the time gap after which the next vessel can sail out of the lock (after another vessel has deberthed) [s]
                 minimize_gate_open_times=False,
                 closing_gate_in_between_operations=False,
                 closing_gate_in_between_arrivals=False,
                 *args, **kwargs):

        self.min_vessels_in_operation = min_vessels_in_operation
        self.max_vessels_in_operation = max_vessels_in_operation
        self.clustering_time = clustering_time
        self.minimize_gate_open_times = minimize_gate_open_times
        self.closing_gate_in_between_operations = closing_gate_in_between_operations
        self.closing_gate_in_between_arrivals = closing_gate_in_between_arrivals
        self.close_gate_before_vessel_is_laying_still = close_gate_before_vessel_is_laying_still
        self.water_level_difference_limit_to_open_gate = water_level_difference_limit_to_open_gate
        self.start_sailing_out_time_after_gate_have_been_opened = start_sailing_out_time_after_gate_have_been_opened
        self.minimum_delay_to_close_gate = pd.Timedelta(seconds=minimum_delay_to_close_gate)
        self.minimum_advance_to_open_gate = pd.Timedelta(seconds=minimum_advance_to_open_gate)
        self.minimum_time_between_operations = pd.Timedelta(seconds=minimum_time_between_operations)
        self.sailing_in_time_gap_through_gate = pd.Timedelta(seconds=sailing_in_time_gap_through_gate)
        self.sailing_out_time_gap_through_gate = pd.Timedelta(seconds=sailing_out_time_gap_through_gate)
        self.sailing_in_time_gap_after_berthing_previous_vessel = pd.Timedelta(seconds=sailing_in_time_gap_after_berthing_previous_vessel)
        self.sailing_out_time_gap_after_berthing_previous_vessel = pd.Timedelta(seconds=sailing_out_time_gap_after_berthing_previous_vessel)
        super().__init__(*args, **kwargs)
        self.wait_for_other_vessels = simpy.FilterStore(env=self.env)
        self.wait_for_levelling = simpy.FilterStore(env=self.env)


    def communicate_vessel_to_start_approaching_lock_chamber(self, vessel, waiting_area, direction):
        yield waiting_area.resource.release(vessel.waiting_area_request)

        # add processes to the vessel that interact with the lock chamber
        allow_vessel_to_sail_into_lock = functools.partial(self.allow_vessel_to_sail_into_lock,
                                                           vessel=vessel,
                                                           waiting_area=waiting_area)
        allow_vessel_to_be_locked = functools.partial(self.allow_vessel_to_be_locked, vessel=vessel)
        allow_vessel_to_sail_out_of_lock = functools.partial(self.allow_vessel_to_sail_out_of_lock,
                                                             vessel=vessel)

        vessel.on_pass_edge_functions.append(allow_vessel_to_sail_into_lock)
        vessel.on_pass_edge_functions.append(allow_vessel_to_be_locked)
        vessel.on_pass_edge_functions.append(allow_vessel_to_sail_out_of_lock)

        # correct distance left on edge with the already covered distance through this function
        vessel.overruled_speed.loc[waiting_area.edge, 'speed'] = _get_vessel_sailing_in_speed(self, vessel, direction)
        distance_left_on_edge = waiting_area.distance_waiting_area_to_end_edge
        vessel.distance_left_on_edge = distance_left_on_edge


    def communicate_vessel_to_sail_to_lock_chamber(self, waiting_area, vessel, direction):
        yield from vessel.sail_to_lock_chamber(self, waiting_area, direction)


    def communicate_vessel_to_sail_to_position_in_lock_chamber(self, vessel, direction):
        yield from vessel.sail_to_position_in_lock_chamber(self, direction)


    def communicate_vessel_to_sail_out_of_lock_chamber(self, vessel, direction):
        yield from vessel.sail_out_of_lock_chamber(self, direction)

        # remove functions specific to passing the lock chamber
        remove_functions = [self.allow_vessel_to_sail_into_lock,
                            self.allow_vessel_to_be_locked,
                            self.allow_vessel_to_sail_out_of_lock]
        remove_on_pass_edge_functions = []
        for index, function in enumerate(vessel.on_pass_edge_functions):
            if isinstance(function, functools.partial):
                if function.func in remove_functions:
                    remove_on_pass_edge_functions.append(function)
            elif function in remove_functions:
                remove_on_pass_edge_functions.append(function)
        for function in remove_on_pass_edge_functions:
            vessel.on_pass_edge_functions.remove(function)


    def instruct_vessel_to_wait_in_lock_chamber_before_sailing_out(self, vessel, sailing_out_delay):
        delay_start = vessel.env.now
        had_delay = False
        if sailing_out_delay:
            had_delay = True
            location_of_vessel = pd.DataFrame(vessel.logbook).iloc[-1]['Geometry']
            vessel.log_entry_v0(
                "Waiting for other vessels to leave lock start", self.env.now, vessel.output.copy(), location_of_vessel)
        while sailing_out_delay:
            try:
                yield vessel.env.timeout(sailing_out_delay)
                sailing_out_delay = 0
            except simpy.Interrupt as e:
                sailing_out_delay -= vessel.env.now - delay_start
        if had_delay:
            location_of_vessel = pd.DataFrame(vessel.logbook).iloc[-1]['Geometry']
            vessel.log_entry_v0(
                "Waiting for other vessels to leave lock stop", self.env.now, vessel.output.copy(), location_of_vessel)


    def instruct_vessel_to_wait_in_waiting_area(self, vessel, sailing_out_delay):
        delay_start = vessel.env.now
        while sailing_out_delay:
            try:
                yield vessel.env.timeout(sailing_out_delay)
                sailing_out_delay = 0
            except simpy.Interrupt as e:
                sailing_out_delay -= vessel.env.now - delay_start


    def communicate_vessel_to_leave_lock_complex(self, vessel, direction):
        yield from vessel.leave_lock_complex(self, direction)


    def allow_vessel_to_sail_into_lock(self, edge, waiting_area, vessel=None):
        """Allows the vessel to sail into the lock chamber

        Parameters
        ----------
        origin : str
            node name (that has to be in the graph) on which the vessel is currently sailing, to navigate an edge should form an edge with the origin)
        destination : str
            node name (that has to be in the graph) on which the vessel is currently sailing to, to navigate an edge (should form an edge with the origin)
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        """
        lock_chamber_is_next_up = _check_if_lock_chamber_is_next_lock_complex_object(self, edge)
        if not lock_chamber_is_next_up:
            return

        # unpacks the vessel planning
        vessel_planning = self.lock_complex.vessel_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index, "operation_index"]

        # determines information of the lock operation
        direction = vessel_planning.loc[vessel_planning_index, "direction"]

        # actions to be undertaken
        yield from self.communicate_vessel_to_sail_to_lock_chamber(waiting_area, vessel, direction)
        if self.close_gate_before_vessel_is_laying_still:
            self.close_gate_between_arrivals_if_necessary(vessel, direction, operation_index)
        vessel.request_to_enter_lock = self.resource.request()
        vessel.request_to_enter_lock.vessel = vessel
        yield vessel.request_to_enter_lock
        yield self.length.get(vessel.L)
        yield from self.communicate_vessel_to_sail_to_position_in_lock_chamber(vessel, direction)
        if not self.close_gate_before_vessel_is_laying_still:
            self.close_gate_between_arrivals_if_necessary(vessel, direction, operation_index)


    def allow_vessel_to_sail_out_of_lock(self, edge, vessel=None):
        """Allows the vessel to sail out of the lock chamber

        Parameters
        ----------
        origin : str
            node name (that has to be in the graph) on which the vessel is currently sailing, to navigate an edge should form an edge with the origin)
        destination : str
            node name (that has to be in the graph) on which the vessel is currently sailing to, to navigate an edge (should form an edge with the origin)
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput

        Yields
        ------
        Vessel to sail to the end of the edge at which the lock chamber is located, and initiates new processes: i.e. closing gate or empty lock operation
        """
        lock_chamber_is_next_up = _check_if_lock_chamber_is_next_lock_complex_object(self, edge)
        if not lock_chamber_is_next_up:
            return

        # unpacks the vessel planning
        vessel_planning = self.lock_complex.vessel_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index, "operation_index"]
        direction = vessel_planning.loc[vessel_planning_index, "direction"]
        yield from self.communicate_vessel_to_sail_out_of_lock_chamber(vessel, direction)
        yield self.length.put(vessel.L)
        yield self.resource.release(vessel.request_to_enter_lock)

        # determine if the lock has to be levelled
        self.prepare_next_lock_operation(operation_index, direction, vessel)

        yield from self.communicate_vessel_to_leave_lock_complex(vessel, direction)


    def allow_vessel_to_pass_waiting_area(self, vessel, lock_chamber, waiting_area):
        """
        Let the vessel pass the waiting area

        Parameters
        ----------
        waiting_area : class
            the waiting area of the lock chamber (IsLockWaitingArea-class)

        Yields
        ------
            waiting time in the waiting area: (1) for another vessel and (2) for the start of the assigned lock operation
        """
        # unpacks the lock complex master's vessel planning and the vessel index in this planning
        vessel_planning = self.lock_complex.vessel_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        direction = vessel_planning.loc[vessel_planning_index, 'direction']

        # unpack the vessel and lock operation planning of the lock and the vessel index and operation index
        vessel_planning = self.lock_complex.vessel_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_planning = self.lock_complex.operation_planning
        operation_index = vessel_planning.loc[vessel_planning_index, 'operation_index']
        operation_planning_lock = operation_planning[operation_planning.lock_chamber == lock_chamber.name]
        vessels_in_operation = operation_planning_lock[operation_planning_lock.operation_index == operation_index].iloc[-1]['vessels']

        if len(vessels_in_operation) < self.min_vessels_in_operation:
            yield from self.let_vessel_wait_for_other_vessels_in_waiting_area(vessel)

        # determines the sailing time to reach the approach point of the lock complex
        distance_sailed = waiting_area.distance_from_edge_start
        sailing_to_approach = calculate_sailing_time_to_approach_point(self, vessel, direction, distance_sailed)

        # determine the current time (after waiting for another vessel, or not) and the time that the vessel will be at the approach point if it will continue and what was planned before
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        time_at_approach = current_time + sailing_to_approach
        planned_start_time_entering_lock = vessel_planning.loc[vessel_planning_index, 'time_lock_operation_start']

        # determine (additional) waiting time for the vessel
        waiting_time = planned_start_time_entering_lock - time_at_approach

        # determine the waiting time that a vessel can do by decreasing it sailing speed and the waiting time that the vessel has to wait stationary in the waiting area (due to a minimum required speed for safe manoeuvrability)
        remaining_static_waiting_time = waiting_time.total_seconds()
        if remaining_static_waiting_time > 0.:
            yield from self.let_vessel_wait_for_available_lock_operation_in_waiting_area(vessel, waiting_area)

        # release vessel from waiting area and let vessel continue
        yield from self.communicate_vessel_to_start_approaching_lock_chamber(vessel, waiting_area, direction)
        self.prepare_lock_operation(vessel)


    def let_vessel_wait_for_available_lock_operation_in_waiting_area(self, vessel, waiting_area):
        # unpacks the lock complex master's vessel planning and the vessel index in this planning
        vessel_planning = self.lock_complex.vessel_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        direction = vessel_planning.loc[vessel_planning_index, 'direction']
        distance_sailed = waiting_area.distance_from_edge_start

        # determines the sailing time to reach the approach point of the lock complex
        sailing_to_approach = calculate_sailing_time_to_approach_point(self, vessel, direction, distance_sailed)

        # set the moment in time that the waiting in the waiting area has started
        waiting_start = vessel.env.now

        # determine the current time (after waiting for another vessel, or not) and the time that the vessel will be at the approach point if it will continue and what was planned before
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        time_at_approach = current_time + sailing_to_approach
        planned_start_time_entering_lock = vessel_planning.loc[vessel_planning_index, 'time_lock_operation_start']

        # determine (additional) waiting time for the vessel
        waiting_time = planned_start_time_entering_lock - time_at_approach

        # determine the waiting time that a vessel can do by decreasing it sailing speed and the waiting time that the vessel has to wait stationary in the waiting area (due to a minimum required speed for safe manoeuvrability)
        remaining_static_waiting_time = waiting_time.total_seconds()

        # if there is stationary waiting time -> let vessel wait (longer) in the waiting area
        if remaining_static_waiting_time > 0.:
            # log the start of the waiting process
            vessel.log_entry_v0("Waiting for lock operation start",
                              vessel.env.now, vessel.output.copy(), vessel.logbook[-1]['Geometry'], )
            # waiting in the waiting area, if request is interrupted, the vessel keeps waiting but time that vessel already has waited is subtracted
            while remaining_static_waiting_time > 0.:
                try:
                    yield self.env.timeout(remaining_static_waiting_time)
                    time_at_approach += pd.Timedelta(seconds=remaining_static_waiting_time)
                    remaining_static_waiting_time = 0.
                    time_operation_start = vessel_planning.loc[vessel_planning_index, 'time_lock_operation_start']
                    remaining_static_waiting_time = (time_operation_start - time_at_approach).total_seconds()
                except simpy.Interrupt as e:
                    remaining_static_waiting_time -= self.env.now - waiting_start

            # log the stop of the waiting process
            vessel.log_entry_v0("Waiting for lock operation stop", vessel.env.now, vessel.output.copy(),
                                vessel.logbook[-1]['Geometry'],)


    def let_vessel_wait_for_other_vessels_in_waiting_area(self, vessel):
        # unpack the vessel and lock operation planning of the lock and the vessel index and operation index
        lock_complex = self.lock_complex
        vessel_planning = self.lock_complex.vessel_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_planning = self.lock_complex.operation_planning
        operation_index = vessel_planning.loc[vessel_planning_index, 'operation_index']

        waiting_start = vessel.env.now
        # log the waiting event
        vessel.log_entry_v0("Waiting for other vessel for lock operation start",
                          waiting_start, vessel.output.copy(), vessel.logbook[-1]['Geometry'])

        # create a request to wait for another vessel (this is a request for a filter store: only if there are enough vessels the operation will be assigned to the store and all vessels will continue to the lock chamber)
        request = self.wait_for_other_vessel_to_arrive.get(lambda operation: operation.operation_index == operation_index)

        # waiting in the waiting area, if request is interrupted, the vessel keeps waiting
        while len(operation_planning.loc[operation_index, 'vessels']) < self.min_vessels_in_operation:
            try:
                yield request
            except simpy.Interrupt as e:
                pass

        # determine the moment in time that the waiting has stopped
        waiting_stop = vessel.env.now

        # if the moment of the vessel starting to enter the lock has shifted, then update the vessel planning and the operation planning if it is the first assigned vessel to the lock
        vessel_planning_info =  vessel_planning.loc[vessel_planning_index]
        operation_planning_info = operation_planning.loc[operation_index]
        passage_information = {'time_lock_operation_start': vessel_planning_info['time_lock_operation_start'],
                               'time_lock_entry_start': vessel_planning_info['time_lock_entry_start'],
                               'time_lock_entry_stop':  vessel_planning_info['time_lock_entry_stop'],
                               'time_arrival_at_lineup_area': vessel_planning_info['time_arrival_at_lineup_area']}
        operation_information = {'time_entry_start': operation_planning_info['time_entry_start']}

        delay = pd.Timedelta(seconds=waiting_stop - waiting_start)
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(waiting_stop))
        if current_time + sailing_to_approach > start_time_entering_lock:
            delay = delay.round("us")
            passage_information['time_lock_operation_start'] += delay
            passage_information['time_lock_entry_start'] += delay
            passage_information['time_lock_entry_stop'] += delay

            if _check_if_vessel_is_first_vessel(lock_complex, vessel, operation_index):
                operation_information['time_entry_start'] += delay
            vessel_planning.loc[vessel_planning_index, 'time_arrival_at_lineup_area'] += delay
            _update_lock_vessel_planning(lock_complex, vessel_planning_index, passage_information)
            _update_lock_operation_planning(lock_complex, operation_index, operation_information)

        # log that the waiting has stopped
        vessel.log_entry_v0("Waiting for other vessel for lock operation stop",
                          vessel.env.now, vessel.output.copy(),vessel.logbook[-1]['Geometry'],)


    def wait_for_other_vessels_to_start_lock_operation(self,  vessel):
        # Wait for last assigned vessel of lock operation
        waiting_for_other_vessels = True
        lock_position = vessel.position_in_lock
        vessel.log_entry_v0("Waiting for other vessels to enter lock start", self.env.now, vessel.output.copy(), lock_position)
        while waiting_for_other_vessels:
            try:
                yield self.wait_for_other_vessels.get(filter=(lambda request: request.id == vessel.id))
                waiting_for_other_vessels = False
            except simpy.Interrupt as e:
                waiting_for_other_vessels = True
        vessel.log_entry_v0("Waiting for other vessels to enter lock stop", self.env.now, vessel.output.copy(), lock_position)

        # Follow the converting lock chamber
        waiting_for_levelling = True
        while waiting_for_levelling:
            try:
                yield self.wait_for_levelling.get(filter=(lambda request: request.id == vessel.id))
                waiting_for_levelling = False
            except simpy.Interrupt as e:
                waiting_for_levelling = True


    def prepare_lock_operation(self, vessel):
        vessel_planning = self.lock_complex.vessel_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index].operation_index
        direction = vessel_planning.loc[vessel_planning_index].direction
        level = self.start_node
        if direction:
            level = self.end_node

        # on continuing sailing to the lock complex, determine the current time and whether the vessel is the first vessel or will arrive after another vessel
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
        first_in_lock = _check_if_vessel_is_first_vessel(self.lock_complex, vessel, operation_index)
        between_arrivals = False
        if not first_in_lock:
            between_arrivals = True

        gateinfo = determine_if_gate_is_closed(self, operation_index, direction, vessel, first_in_lock,between_arrivals)
        gate_is_closed, gate_required_to_be_open, operation_time = gateinfo

        # if gate is open, then the vessel can continue normally
        if not gate_is_closed:
            return

        # if not, and if the time that the gate will be open lies ahead of the current time -> create a gate open request with a delay so that the gate are open at the right moment (according to the lock master's policy)
        gate_open_delay = ((gate_required_to_be_open - operation_time) - current_time).total_seconds()
        if gate_open_delay > 0.:
            open_gate_process = self.open_gate(to_level=lock_start_node, delay=gate_open_delay, vessel=vessel)
            self.env.process(open_gate_process)

        # if it is already too late, the gate should open immediately -> determine the time that the gate are required to be opened again (this can include a new levelling process in case of tidal water levels)
        levelling_required = False
        if operation_time > pd.Timedelta(seconds=self.gate_closing_time):
            levelling_required = True

        # log the gate open process and the lock levelling process if this is required
        if levelling_required:
            levelling_process = self.convert_chamber(level, 1 - direction, operation_index=None,
                                                     vessel=None, delay = 0.)
            self.env.process(levelling_process)


    def prepare_next_lock_operation(self, operation_index, direction, vessel):
        """Lock operator checks and (if required) initiates an empty lock operation or closes the gate if there is sufficient time with respect to the next operation's start time

        Parameters
        ----------
        lock_chamber : object
            the lock chamber object generated with IsLockChamber
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (A -> B) or 1 (B -> A)
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        """
        # get variables of the last lock operation: do nothing if it is not the last vessel that is sailing out of the lock
        made_operation = _get_operation_info(self, operation_index)
        vessels_in_last_operation = made_operation.vessels
        is_last_vessel_sailing_out = vessels_in_last_operation[-1] == vessel

        if not is_last_vessel_sailing_out:
            return

        # get the current time, and the information of the next operation
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
        _, to_node = _get_lock_operation_to_and_from_node(self, 1 - direction)
        next_operations = _get_next_operations(self, operation_index)

        # determine if the gate can be closed after the considered vessel has sailed out of the lock
        gate_can_be_closed = determine_if_gate_can_be_closed(self, vessel, direction, operation_index)

        # determine if the next operation is empty
        next_lockage_is_empty = False
        next_operation = None

        if not next_operations.empty:
            next_operation = next_operations.iloc[0]
            if not len(next_operation.vessels):
                next_lockage_is_empty = True

        # an action should be done if the gate can be closed in between operations, or if the next lock operation is empty
        if gate_can_be_closed and self.closing_gate_in_between_operations:
            gate_closing_start_time = last_operation.time_potential_lock_gate_closure_start
            delay = np.max([self.minimum_delay_to_close_gate.total_seconds(),
                            (gate_closing_start_time - current_time).total_seconds()])

            # close the gate with the correct delay
            vessel.env.process(self.close_gate(delay=delay))

        elif next_lockage_is_empty:
            gate_closing_start_time = next_operation.time_gate_closing_start
            closing_delay = np.max([self.minimum_delay_to_close_gate.total_seconds(),
                                    (gate_closing_start_time - current_time).total_seconds()])

            # if there is an empty lock operation and no policy that gate are closed in between operations is active -> close gate and convert chamber afterwards
            if not self.closing_gate_in_between_operations:
                convert_chamber_delay = closing_delay
                closing_gate = True
            # if there is an empty lock operation but the policy that gate are closed in between operations is active -> close gate and convert chamber later, or convert chamber immediately if there is insufficient time
            else:
                next_operation = next_operations.iloc[1]
                gate_opening_start_time = next_operation.time_potential_lock_gate_opening_stop
                lock_operation_duration = calculate_time_to_open_gate(self, vessel_operation_index + 1,
                                                                      1 - direction, gate_opening_start_time)
                opening_delay = np.max([0, (gate_opening_start_time - current_time).total_seconds()])
                opening_delay -= lock_operation_duration.total_seconds()
                if opening_delay > (closing_delay + self.gate_closing_time):
                    convert_chamber_delay = opening_delay
                    closing_gate = False
                    vessel.env.process(self.close_gate(delay=closing_delay))
                else:
                    convert_chamber_delay = closing_delay
                    closing_gate = True

            # convert the lock chamber with the correct delay and if the gate should first be closed
            vessel.env.process(self.convert_chamber(operation_index = operation_index + 1,
                                                    new_level = to_node,
                                                    vessel = None,
                                                    close_gate = closing_gate,
                                                    delay = convert_chamber_delay,
                                                    direction = 1 - direction))


    def allow_vessel_to_be_locked(self, edge, vessel=None):
        """
        Initiates levelling process as function that can be added to a vessel

        Parameters
        ----------
        origin : str
            node name (that has to be in the graph) on which the vessel is currently sailing, to navigate an edge should form an edge with the origin)
        destination : str
            node name (that has to be in the graph) on which the vessel is currently sailing to, to navigate an edge (should form an edge with the origin)
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        k : int
            identifier of the edge between two nodes in a multidigraph network


        """
        lock_chamber_is_next_up = _check_if_lock_chamber_is_next_lock_complex_object(self, edge)
        if not lock_chamber_is_next_up:
            return

        # unpack the lock complex master's vessel planning and determine the operation index
        vessel_planning = self.lock_complex.vessel_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index, "operation_index"]

        # determine the direction to the lock chamber is currently levelled to, and to which node the lock chamber will level
        current_node = self.gate_open_at_node
        if current_node == self.start_node:
            direction = 0
            next_node = self.end_node
        else:
            direction = 1
            next_node = self.start_node

        # initiate levelling if vessel is the last assigned vessel in the lock
        is_last_vessel = _check_if_vessel_is_last_vessel(self, vessel, operation_index)
        if is_last_vessel:
            yield from self.convert_chamber(next_node, direction, operation_index=operation_index, vessel=vessel)
        else:
            yield from self.wait_for_other_vessels_to_start_lock_operation(vessel)

        # determine and yield sailing out delay
        sailing_out_delay = _get_vessel_departure_start_delay(self, vessel, operation_index).total_seconds()

        if sailing_out_delay > 0.: #TODO: delay should never be smaller than 0, but it still occurs
            yield from self.instruct_vessel_to_wait_in_lock_chamber_before_sailing_out(vessel, sailing_out_delay)


    def convert_chamber(self, new_level, direction, operation_index=None, vessel=None, delay = 0., close_gate = None):
        """
        Converts the lock chamber and logs this event

        Parameters
        ----------
        new_level : str
            node that represents the side at which the lock is currently levelled
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        close_gate : bool
            if the gate have to be closed: yes (True) or no (False)
        delay : float
            a delay before lock levelling [s]

        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Yields
        ------
        The levelling of the lock chamber
        """
        vessels = []
        this_operation = None
        if operation_index is not None:
            this_operation = _get_operation_info(self, operation_index)
            vessels = this_operation.vessels

        vessel_planning_index = None
        vessel_planning_info = None
        vessel_planning = self.lock_complex.vessel_planning
        if vessel is not None:
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            vessel_planning_info = vessel_planning.loc[vessel_planning_index]

        # liberate the vessels that were requested to wait for the last vessel
        for other_vessel in vessels[:-1]:
            terminate_waiting_time_for_other_vessel = False
            while not terminate_waiting_time_for_other_vessel:
                try:
                    yield self.wait_for_other_vessels.put(other_vessel)
                    terminate_waiting_time_for_other_vessel = True
                except simpy.Interrupt as e:
                    terminate_waiting_time_for_other_vessel = False

        # Wait for other vessels to lay still
        if not delay and operation_index is not None:
            delay = (this_operation.time_gate_closing_start.to_pydatetime(warn = False).timestamp() - self.env.now)

        # Convert lock chamber
        if close_gate is None:
            if delay > 0:
                yield self.env.timeout(delay)
                delay = 0
            close_gate = True
            if operation_index is not None and vessel_planning_index is not None:
                gate_can_be_closed = this_operation.time_gate_closing_start > vessel_planning_info.time_lock_entry_stop
                if self.close_gate_before_vessel_is_laying_still and not gate_can_be_closed:
                    close_gate = False

        # if there is a delay -> yield time out
        start_delay = self.env.now
        while delay > 0:
            try:
                yield self.env.timeout(delay)
                delay = 0
            except simpy.Interrupt as e:
                delay -= self.env.now - start_delay

        # close the gate or make sure that lock is not performing another process
        if close_gate:
            yield from self.close_gate(delay=delay)
        else:
            hold_gate_A = self.gate_A.resource.request()
            hold_levelling = self.levelling.resource.request()
            hold_gate_B = self.gate_B.resource.request()
            yield hold_gate_A
            yield hold_levelling
            yield hold_gate_B
            self.gate_A.resource.release(hold_gate_A)
            self.levelling.resource.release(hold_levelling)
            self.gate_B.resource.release(hold_gate_B)

        # level lock and open the gate afterwards
        yield from self.level_lock(new_level, direction, operation_index=operation_index)
        yield from self.open_gate()

        # Liberate waiting vessels in lock chamber
        for other_vessel in vessels[:-1]:
            terminate_levelling_for_other_vessel = False
            while not terminate_levelling_for_other_vessel:
                try:
                    yield self.wait_for_levelling.put(other_vessel)
                    terminate_levelling_for_other_vessel = True
                except simpy.Interrupt as e:
                    terminate_levelling_for_other_vessel = False


    def close_gate(self, delay=0.):
        """
        Lock operator closes the lock gate

        Parameters
        ----------
        delay : float
            a delay before gate opening [s]

        Yields
        ------
        The closing of the gate
        """
        if self.has_salinity:
            calculate_lock_salinity_and_saltmass(self)

        # if there is a delay -> yield time out
        start_delay = self.env.now
        while delay > 0:
            try:
                yield self.env.timeout(delay)
                delay = 0
            except simpy.Interrupt as e:
                delay -= self.env.now - start_delay

        # make sure that all lock elements are requested, so only one process is occurring
        hold_gate_A = self.gate_A.resource.request()
        hold_levelling = self.levelling.resource.request()
        hold_gate_B = self.gate_B.resource.request()
        yield hold_gate_A
        yield hold_levelling
        yield hold_gate_B

        # log the start of the event
        self.log_entry_v0("Lock gate closing start", self.env.now, self.output.copy(), self.gate_open_at_node)
        for request in self.resource.users:
            user = request.vessel
            location_of_vessel = pd.DataFrame(user.logbook).iloc[-1]['Geometry']
            user.log_entry_v0("Waiting for lock gate closing start", self.env.now, user.output.copy(), location_of_vessel)

        # timeout event of the gate closing
        remaining_gate_closing_time = self.gate_closing_time
        start_time_closing = self.env.now
        while remaining_gate_closing_time:
            try:
                yield self.env.timeout(remaining_gate_closing_time)
                remaining_gate_closing_time = 0
            except simpy.Interrupt as e:
                remaining_gate_closing_time -= self.env.now - start_time_closing

        # set water level to the side at which the gate has been closed
        time = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))
        if self.gate_open_at_node == self.start_node:
            node = self.start_node
        else:
            node = self.end_node

        if self.has_water_level:
            hydromanager = HydrodynamicDataManager()
            time_index = np.abs(self.time - time).argmin()
            new_water_level = hydromanager._get_hydrodynamic_data_value(time, node, "Water level")
            self.water_level[time_index:] = new_water_level

        # log the end of the event
        self.log_entry_v0("Lock gate closing stop", self.env.now, self.output.copy(), self.gate_open_at_node)
        for request in self.resource.users:
            user = request.vessel
            location_of_vessel = pd.DataFrame(user.logbook).iloc[-1]['Geometry']
            user.log_entry_v0("Waiting for lock gate closing stop", self.env.now, user.output.copy(), location_of_vessel)

        if self.gate_open_at_node == self.start_node:
            self.gate_A_open = False
        else:
            self.gate_B_open = False

        # release all lock elements that were requested, so the next process can start
        self.gate_A.resource.release(hold_gate_A)
        self.levelling.resource.release(hold_levelling)
        self.gate_B.resource.release(hold_gate_B)


    def close_gate_between_arrivals_if_necessary(self, vessel, direction, operation_index):
        # calculate delay to close gate
        vessel_planning = self.lock_complex.vessel_planning
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        vessel_planning_info = vessel_planning.loc[vessel_planning_index]
        delay_to_close_gate = vessel_planning_info["time_potential_lock_gate_closure_start"] - current_time
        gate_can_be_closed_between_vessel_arrivals = determine_if_gate_can_be_closed(self, vessel, direction,
                                                                                     operation_index, True)
        if not self.close_gate_before_vessel_is_laying_still:
            delay_to_close_gate = pd.Timedelta(seconds = 0.)

        if gate_can_be_closed_between_vessel_arrivals:
            vessel.env.process(self.close_gate(delay=delay_to_close_gate.total_seconds()))


    def level_lock(self, new_level, direction, operation_index=None):
        """
        Lock operator levels the water level of the lock chamber to the harbour side of the direction of the lock operation

        new_level : str
            node of the edge of lock complex to which the lock chamber is levelling
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        same_direction : bool


        Yields
        ------
        Levelling of the lock chamber
        """

        # make sure that all lock elements are requested, so only one process is occurring
        hold_gate_A = self.gate_A.resource.request()
        hold_levelling = self.levelling.resource.request()
        hold_gate_B = self.gate_B.resource.request()
        yield hold_gate_A
        yield hold_levelling
        yield hold_gate_B

        # determine the levelling time
        levelling_time, _, _ = calculate_levelling_time(self, self.env.now, direction, operation_index=operation_index)

        # log the start of the event
        self.log_entry_v0("Lock levelling start", self.env.now, self.output.copy(), self.gate_open_at_node, )
        for request in self.resource.users:
            user = request.vessel
            location_of_vessel = pd.DataFrame(user.logbook).iloc[-1]['Geometry']
            user.log_entry_v0("Waiting for lock levelling start", self.env.now, user.output.copy(), location_of_vessel)

        # set new node to which the gate will be opened
        self.gate_open_at_node = new_level

        # timeout
        remaining_levelling_time = levelling_time
        start_levelling = self.env.now
        while remaining_levelling_time:
            try:
                yield self.env.timeout(remaining_levelling_time)
                remaining_levelling_time = 0
            except simpy.Interrupt as e:
                remaining_levelling_time -= self.env.now - start_levelling

        # log the end of the event
        self.log_entry_v0("Lock levelling stop", self.env.now, self.output.copy(), self.gate_open_at_node, )
        for request in self.resource.users:
            user = request.vessel
            location_of_vessel = pd.DataFrame(user.logbook).iloc[-1]['Geometry']
            user.log_entry_v0("Waiting for lock levelling stop", self.env.now, user.output.copy(), location_of_vessel)

        # release all lock elements that were requested, so the next process can start
        self.gate_A.resource.release(hold_gate_A)
        self.levelling.resource.release(hold_levelling)
        self.gate_B.resource.release(hold_gate_B)


    def open_gate(self, to_level=None, vessel=None, delay=0.):
        """
        Lock operator opens the lock gate TODO: attribute for lock operator

        Parameters
        ----------
        to_level : str
            node of the edge of lock complex to which the lock chamber opens
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        delay : float
            a delay before gate opening

        Yields
        ------
        The opening of the gate
        """

        # if there is a delay -> yield time out
        start_delay = self.env.now
        while delay > 0:
            try:
                yield self.env.timeout(delay)
                delay = 0
            except (
                simpy.Interrupt
            ) as e:  # if there is a delay -> yield time out with new delay (remaining delay added with a delay equal to the exception)
                delay -= self.env.now - start_delay
                if vessel is not None:
                    if e.cause is not None:
                        delay += float(e.cause)

        # determine to_level
        if to_level is None:
            to_level = self.gate_open_at_node

        #lock at new location
        self.gate_open_at_node = to_level

        if self.has_water_level:
            hydromanager = HydrodynamicDataManager()
            time = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))
            time_index_lock = np.abs(self.time - time).argmin()
            interp_time = self.time[time_index_lock:]
            time_series = hydromanager.hydrodynamic_data.TIME.values
            wlev_series_node_gate_open = hydromanager._get_hydrodynamic_data_series(time, self.gate_open_at_node, "Water level")
            time_index_harbour = hydromanager._get_time_index_of_hydrodynamic_data(time)
            self.water_level[time_index_lock:] = np.interp(interp_time.astype('datetime64[ns]').astype('int64') / 1e9,
                                                           time_series[time_index_harbour:].astype('int64') / 1e9,
                                                           wlev_series_node_gate_open)

        # make sure that all lock elements are requested, so only one process is occurring
        hold_gate_A = self.gate_A.resource.request()
        hold_levelling = self.levelling.resource.request()
        hold_gate_B = self.gate_B.resource.request()
        yield hold_gate_A
        yield hold_levelling
        yield hold_gate_B

        # log the process start
        self.log_entry_v0("Lock gate opening start", self.env.now, self.output.copy(), self.gate_open_at_node)
        for request in self.resource.users:
            user = request.vessel
            location_of_vessel = pd.DataFrame(user.logbook).iloc[-1]['Geometry']
            user.log_entry_v0("Waiting for lock gate opening start", self.env.now, user.output.copy(), location_of_vessel)

        # timeout
        remaining_gate_opening_time = self.gate_opening_time
        start_time_opening = self.env.now
        while remaining_gate_opening_time:
            try:
                yield self.env.timeout(remaining_gate_opening_time)
                remaining_gate_opening_time = 0
            except simpy.Interrupt as e:
                remaining_gate_opening_time -= self.env.now - start_time_opening

        # log the process stop
        self.log_entry_v0("Lock gate opening stop", self.env.now, self.output.copy(), self.gate_open_at_node,)
        for request in self.resource.users:
            user = request.vessel
            location_of_vessel = pd.DataFrame(user.logbook).iloc[-1]['Geometry']
            user.log_entry_v0("Waiting for lock gate opening stop", self.env.now, user.output.copy(), location_of_vessel)

        # determine which side the gate is open to
        if self.gate_open_at_node == self.start_node:
            self.gate_A_open = True
        else:
            self.gate_B_open = True

        # release all lock elements that were requested, so the next process can start
        self.gate_A.resource.release(hold_gate_A)
        self.levelling.resource.release(hold_levelling)
        self.gate_B.resource.release(hold_gate_B)

        if self.has_salinity:
            calculate_lock_salinity_and_saltmass(self)
