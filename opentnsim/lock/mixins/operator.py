import datetime
import functools
import math
import numpy as np
import pandas as pd
import simpy

from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from opentnsim.lock.calculations import (
    calculate_levelling_time,
    calculate_lock_operation_times,
    calculate_vessel_departure_start_delay
)
from opentnsim.lock.utils import (
    _get_lock_operation_to_and_from_node,
    _get_waiting_area,
    _get_distance_to_lock,
)
from opentnsim.graph.calculations import calculate_location_over_edges
from opentnsim.graph.utils import check_graph_is_multidigraph_type, get_edge
from IPython.display import display

class IsLockChamberOperator:
    """The lock chamber operator operates one chamber of the lock."""

    def __init__(self,
                 close_gate_before_vessel_is_laying_still = False,
                 min_vessels_in_operation=0,
                 max_vessels_in_operation=math.inf,
                 clustering_time=0.5 * 60 * 60,
                 water_level_difference_limit_to_open_gate=0.05,
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
        super().__init__(*args, **kwargs)

    def initiate_levelling(self, origin, destination, lock_chamber, vessel=None):
        """
        Initiates levelling process as function that can be added to a vessel TODO: preferably you don't want to add this process to the vessel but let the lock master / operator handle this

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
        is_multidigraph = check_graph_is_multidigraph_type(self.env.graph)
        edge = get_edge(self.env.graph, (origin, destination, lock_chamber.k), is_multidigraph)
        if not 'Lock chamber' in self.env.graph.edges[edge].keys():
            return

        lock_chambers_on_edge = self.env.graph.edges[edge]['Lock chamber']
        lock_chamber_found = False
        for lock_chamber_on_edge in lock_chambers_on_edge:
            if lock_chamber == lock_chamber_on_edge:
                lock_chamber_found = True
                break

        if not lock_chamber_found:
            return

        # unpack the lock complex master's vessel and lock operation plannings
        vessel_planning = lock_chamber.lock_complex.vessel_planning
        operation_planning = lock_chamber.lock_complex.operation_planning

        # determine the index of the vessel and the lock operation to which it is assigned to and the index of this operation
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index, "operation_index"]
        this_operation = operation_planning.loc[operation_index]

        # determine the direction to the lock chamber is currently levelled to, and to which node the lock chamber will level
        current_node = lock_chamber.gate_open
        if current_node == lock_chamber.start_node:
            direction = 0
            next_node = lock_chamber.end_node
        else:
            direction = 1
            next_node = lock_chamber.start_node

        # determine the vessels that are assigned to the lock operation to which the vessel is assigned
        vessels = this_operation.vessels
        # initiate levelling if vessel is the last assigned vessel in the lock
        if vessel == vessels[-1]:
            # liberate the vessels that were requested to wait for the last vessel
            for other_vessel in vessels[:-1]:
                terminate_waiting_time_for_other_vessel = False
                while not terminate_waiting_time_for_other_vessel:
                    try:
                        yield lock_chamber.wait_for_other_vessels.put(other_vessel)
                        terminate_waiting_time_for_other_vessel = True
                    except simpy.Interrupt as e:
                        terminate_waiting_time_for_other_vessel = False

            # Wait for other vessels to lay still
            delay = (operation_planning.loc[operation_index].time_gate_closing_start.round("s").to_pydatetime().timestamp() - lock_chamber.env.now)
            if delay > 0:
                yield lock_chamber.env.timeout(delay)

            # Convert lock chamber
            close_gate = True
            if (lock_chamber.close_gate_before_vessel_is_laying_still
                and this_operation.time_gate_closing_start < vessel_planning.loc[vessel_planning_index, "time_lock_entry_stop"]):
                close_gate = False

            lock_chamber.lock_complex.operation_planning.loc[operation_index, 'status'] = 'unavailable'
            yield from lock_chamber.convert_chamber(next_node, direction, operation_index=operation_index, vessel=vessel, close_gate=close_gate)

            # Liberate waiting vessels in lock chamber
            for other_vessel in vessels[:-1]:
                terminate_levelling_for_other_vessel = False
                while not terminate_levelling_for_other_vessel:
                    try:
                        yield lock_chamber.wait_for_levelling.put(other_vessel)
                        terminate_levelling_for_other_vessel = True
                    except simpy.Interrupt as e:
                        terminate_levelling_for_other_vessel = False

        # If vessel is not the last assigned vessel
        else:
            # Wait for last assigned vessel of lock operation
            waiting_for_other_vessels = True
            last_location = vessel.logbook[-1]["Geometry"]
            vessel.log_entry_v0("Waiting for other vessels in lock start", self.env.now, self.output.copy(), last_location)
            while waiting_for_other_vessels:
                try:
                    yield lock_chamber.wait_for_other_vessels.get(filter=(lambda request: request.id == vessel.id))
                    waiting_for_other_vessels = False
                except simpy.Interrupt as e:
                    waiting_for_other_vessels = True
            vessel.log_entry_v0("Waiting for other vessels in lock stop", self.env.now, self.output.copy(),last_location)

            # Follow the converting lock chamber
            vessel.log_entry_v0(
                "Levelling start",
                vessel.env.now,
                vessel.output.copy(),
                vessel.position_in_lock,
            )
            waiting_for_levelling = True
            while waiting_for_levelling:
                try:
                    yield lock_chamber.wait_for_levelling.get(filter=(lambda request: request.id == vessel.id))
                    waiting_for_levelling = False
                except simpy.Interrupt as e:
                    waiting_for_levelling = True
            vessel.log_entry_v0(
                "Levelling stop",
                vessel.env.now,
                vessel.output.copy(),
                vessel.position_in_lock,
            )

        # determine and yield sailing out delay
        sailing_out_delay = calculate_vessel_departure_start_delay(lock_chamber, vessel, operation_index).total_seconds()
        delay_start = vessel.env.now
        while sailing_out_delay:
            try:
                yield vessel.env.timeout(sailing_out_delay)
                sailing_out_delay = 0
            except simpy.Interrupt as e:
                sailing_out_delay -= vessel.env.now - delay_start


    def prepare_next_lock_operation(self, lock_chamber, operation_index, direction, vessel):
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
        lock_complex = lock_chamber.lock_complex
        operation_planning = lock_complex.operation_planning
        last_operation = operation_planning.loc[operation_index]
        vessels_in_last_operation = last_operation.vessels
        is_last_vessel_sailing_out = vessels_in_last_operation[-1] == vessel
        if not is_last_vessel_sailing_out:
            return

        # get the current time, and the information of the next operation
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
        _, to_node = _get_lock_operation_to_and_from_node(self, 1 - direction)
        next_operations = operation_planning[operation_planning.index > operation_index]

        # determine if the gate can be closed after the considered vessel has sailed out of the lock
        gate_can_be_closed = lock_chamber.determine_if_gate_can_be_closed(vessel, direction, operation_index)

        # determine if the next operation is empty
        next_lockage_is_empty = False
        if not next_operations.empty:
            next_operation = next_operations.iloc[0]
            if not len(next_operation.vessels):
                next_lockage_is_empty = True

        # an action should be done if the gate can be closed in between operations, or if the next lock operation is empty
        if gate_can_be_closed and self.closing_gate_in_between_operations:
            gate_closing_start_time = last_operation.time_potential_lock_gate_closure_start
            delay = np.max([self.sailing_time_before_closing_lock_gate, (gate_closing_start_time - current_time).total_seconds()])

            # close the gate with the correct delay
            vessel.env.process(lock_chamber.close_gate(delay=delay))

        elif next_lockage_is_empty:
            gate_closing_start_time = next_operation.time_gate_closing_start
            closing_delay = np.max([self.sailing_time_before_closing_lock_gate, (gate_closing_start_time - current_time).total_seconds()])

            # if there is an empty lock operation and no policy that gate are closed in between operations is active -> close gate and convert chamber afterwards
            if not self.closing_gate_in_between_operations:
                convert_chamber_delay = closing_delay
                closing_gate = True
            # if there is an empty lock operation but the policy that gate are closed in between operations is active -> close gate and convert chamber later, or convert chamber immediately if there is insufficient time
            else:
                next_operation = next_operations.iloc[1]
                gate_opening_start_time = next_operation.time_potential_lock_gate_opening_stop
                lock_operation_duration = self.determine_time_to_open_gate(operation_index = vessel_operation_index + 1,
                                                                           direction =1 - direction,
                                                                           gate_required_to_be_open = gate_opening_start_time)
                opening_delay = (np.max([0, (gate_opening_start_time - current_time).total_seconds()]) - lock_operation_duration.total_seconds())
                if opening_delay > (closing_delay + self.gate_closing_time):
                    convert_chamber_delay = opening_delay
                    closing_gate = False
                    vessel.env.process(lock_chamber.close_gate(delay=closing_delay))
                else:
                    convert_chamber_delay = closing_delay
                    closing_gate = True

            # convert the lock chamber with the correct delay and if the gate should first be closed
            vessel.env.process(lock_chamber.convert_chamber(operation_index = operation_index + 1,
                                                            new_level = to_node,
                                                            vessel = None,
                                                            close_gate = closing_gate,
                                                            delay = convert_chamber_delay,
                                                            direction = 1 - direction))

    def allow_vessel_to_sail_out_of_lock(self, origin, destination, lock_chamber, vessel=None):
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
        is_multidigraph = check_graph_is_multidigraph_type(self.env.graph)
        edge = get_edge(self.env.graph, (origin, destination, lock_chamber.k), is_multidigraph)
        if not 'Lock chamber' in self.env.graph.edges[edge].keys():
            return

        lock_chambers_on_edge = self.env.graph.edges[edge]['Lock chamber']
        lock_chamber_found = False
        for lock_chamber_on_edge in lock_chambers_on_edge:
            if lock_chamber == lock_chamber_on_edge:
                lock_chamber_found = True
                break

        if not lock_chamber_found:
            return

        # unpacks the vessel planning
        vessel_planning = lock_chamber.lock_complex.vessel_planning

        # determines information of the lock operation
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index, "operation_index"]
        direction = vessel_planning.loc[vessel_planning_index, "direction"]

        # determines the distance from the vessel to the lock gate that have to be passed
        distance_in_lock_from_position = lock_chamber.lock_length - vessel.distance_position_from_first_lock_gate

        # determines the geometry objects of the lock based on the direction of the vessel TODO: function?
        if not direction:
            second_lock_gate_position = lock_chamber.gate_B.geometry
            remaining_distance = lock_chamber.distance_from_end_node_to_lock_gate_B
            exit_geom = vessel.env.graph.nodes[lock_chamber.end_node]["geometry"]
        else:
            second_lock_gate_position = lock_chamber.gate_A.geometry
            remaining_distance = lock_chamber.distance_from_start_node_to_lock_gate_A
            exit_geom = vessel.env.graph.nodes[lock_chamber.start_node]["geometry"]

        # releasing the length of the lock TODO: only the yield statement should be kept: now it is prevented that vessels need to wait to put back their length, but in principle this should not occur although it sometimes occurs due to bugs
        release_lock_access = False
        while not release_lock_access:
            try:
                yield lock_chamber.length.put(vessel.L)
                release_lock_access = True
            except simpy.Interrupt as e:
                release_lock_access = True

        # determine the waiting time to sail out of the lock TODO: have another algorithm to determine this time: now the vessel planning is used, but this should be prevented -> the lock planning might be off by a few seconds/minutes due to uncertainties/errors in predictions and unforeseen circumstances
        waiting_to_sail_out_time = (vessel_planning.loc[vessel_planning_index, "time_lock_departure_start"] -
                                    pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))).total_seconds()

        # let the vessel wait to sail out of the lock (vessels may have to give other vessels priority to sail out to later sail out of the lock in a safe manner with sufficient distance to the vessel ahead -> i.e., if they sailed into the lock ahead of the considered vessel blocking the sailing out path)
        waiting_to_sail_out_time_start = vessel.env.now
        while waiting_to_sail_out_time > 0:
            try:
                yield vessel.env.timeout(waiting_to_sail_out_time)
                waiting_to_sail_out_time = 0
            except simpy.Interrupt as e:
                waiting_to_sail_out_time -= vessel.env.now - waiting_to_sail_out_time_start

        # log that the vessel can start sailing out of the lock (up to the lock gate)
        vessel.log_entry_v0("Sailing to second lock gate start", vessel.env.now, vessel.output.copy(), vessel.position_in_lock,)

        # determine the process of sailing to the lock gate that have to be passed (distance to these gate divided by the sailing out speed of the vessel)
        vessel_speed = lock_chamber.vessel_sailing_speed_out_lock(vessel)
        sailing_out_time = distance_in_lock_from_position / vessel_speed
        sailing_out_start = vessel.env.now
        while sailing_out_time:
            try:
                yield vessel.env.timeout(sailing_out_time)
                sailing_out_time = 0
            except simpy.Interrupt as e:
                sailing_out_time -= vessel.env.now - sailing_out_start

        # log that the vessel can stops sailing out of the lock (up to the lock gate)
        vessel.log_entry_v0("Sailing to second lock gate stop", vessel.env.now, vessel.output.copy(), second_lock_gate_position,)

        # remove functions specific to passing the lock chamber
        remove_functions = [lock_chamber.allow_vessel_to_sail_into_lock,
                            lock_chamber.initiate_levelling,
                            lock_chamber.allow_vessel_to_sail_out_of_lock]
        remove_on_pass_edge_functions = []
        for index, function in enumerate(vessel.on_pass_edge_functions):
            if isinstance(function, functools.partial):
                if function.func in remove_functions:
                    remove_on_pass_edge_functions.append(function)
            elif function in remove_functions:
                remove_on_pass_edge_functions.append(function)
        for function in remove_on_pass_edge_functions:
            vessel.on_pass_edge_functions.remove(function)

        # determine if the lock has to be levelled
        self.prepare_next_lock_operation(lock_chamber, operation_index, direction, vessel)

        # log that sailing out of the lock complex is starting
        vessel.log_entry_v0("Sailing to lock complex exit start", vessel.env.now, vessel.output.copy(), second_lock_gate_position)

        # let the vessel sail to the end of the lock complex
        vessel_speed = lock_chamber.vessel_sailing_out_speed(vessel, direction)
        sailing_out_time = remaining_distance / vessel_speed
        sailing_out_start = vessel.env.now
        while sailing_out_time:
            try:
                yield vessel.env.timeout(sailing_out_time)
                sailing_out_time = 0
            except simpy.Interrupt as e:
                sailing_out_time -= vessel.env.now - sailing_out_start
                remaining_sailing_distance = vessel_speed * sailing_out_time
                sailing_out_time = remaining_sailing_distance / vessel.current_speed

        # log that sailing out of the lock complex is stopping and set that no distance has to be sailed along the edge (vessel is at end of lock complex)
        vessel.log_entry_v0("Sailing to lock complex exit stop", vessel.env.now, vessel.output.copy(), exit_geom,)
        vessel.distance_left_on_edge = 0

    def allow_vessel_to_sail_into_lock(self, origin, destination, lock_chamber, waiting_area, vessel=None):
        """Allows the vessel to sail into the lock chamber

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

        is_multidigraph = check_graph_is_multidigraph_type(self.env.graph)
        edge = get_edge(self.env.graph, (origin, destination, lock_chamber.k), is_multidigraph)
        if not 'Lock chamber' in self.env.graph.edges[edge].keys():
            return

        lock_chambers_on_edge = self.env.graph.edges[edge]['Lock chamber']
        lock_chamber_found = False
        for lock_chamber_on_edge in lock_chambers_on_edge:
            if lock_chamber == lock_chamber_on_edge:
                lock_chamber_found = True
                break

        if not lock_chamber_found:
            return

        # unpacks the vessel planning
        vessel_planning = lock_chamber.lock_complex.vessel_planning

        # determines information of the lock operation
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index, "operation_index"]
        direction = vessel_planning.loc[vessel_planning_index, "direction"]
        current_time = vessel.env.now

        # determines the geometry objects of the lock based on the direction of the vessel TODO: function?
        distance_to_lock = _get_distance_to_lock(self, direction)
        if not direction:
            first_lock_gate_position = lock_chamber.gate_A.geometry
        else:
            first_lock_gate_position = lock_chamber.gate_B.geometry

        # correct the distance to the lock gate if the vessel is in the waiting area, located at the same edge of the lock
        lock_start_node, lock_end_node = _get_lock_operation_to_and_from_node(self, direction)
        if (lock_start_node, lock_end_node) == waiting_area.edge:
            distance_to_lock -= waiting_area.distance_from_edge_start

        # log the start of sailing to the lock gate
        last_position_vessel = vessel.logbook[-1]["Geometry"]
        vessel.log_entry_v0("Sailing to first lock gate start", vessel.env.now, vessel.output.copy(), last_position_vessel,)

        # let vessel sail to the lock gate
        vessel_speed = lock_chamber.vessel_sailing_in_speed(vessel, direction)
        remaining_sailing_time = distance_to_lock / vessel_speed
        while remaining_sailing_time > 0:
            try:
                yield vessel.env.timeout(remaining_sailing_time)
                remaining_sailing_time = 0
            except simpy.Interrupt as e:
                remaining_sailing_time -= vessel.env.now - current_time
                remaining_sailing_distance = vessel_speed * remaining_sailing_time
                remaining_sailing_time = remaining_sailing_distance / vessel.current_speed

        # vessel entering now the lock -> delete the overruled speeds imposed on the vessel
        vessel.overruled_speed = vessel.overruled_speed.iloc[0:0]

        # claim the lock length (this should not lead to waiting time)
        yield lock_chamber.length.get(vessel.L)

        # log the stop of sailing to the lock gate
        vessel.log_entry_v0("Sailing to first lock gate stop", vessel.env.now, vessel.output.copy(), first_lock_gate_position,)

        # Checks if gate should be closed intermediately
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

        # calculate delay to close gate
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
        delay_to_close_gate = vessel_planning.loc[vessel_planning_index, "time_potential_lock_gate_closure_start"] - current_time

        # close gate if gate can be closed in between vessel arrivals or if vessel is last vessel to enter the lock
        gate_can_be_closed_between_vessel_arrivals = lock_chamber.determine_if_gate_can_be_closed(vessel, direction, operation_index, between_arrivals=True)
        if lock_chamber.close_gate_before_vessel_is_laying_still and gate_can_be_closed_between_vessel_arrivals:
            vessel.env.process(lock_chamber.close_gate(delay=delay_to_close_gate.total_seconds()))

        # log the start of sailing to the position within the lock chamber
        vessel.log_entry_v0("Sailing to position in lock start", vessel.env.now, vessel.output.copy(), first_lock_gate_position, )

        # determine position in the lock chamber and distance to sail to this location
        vessel.distance_position_from_first_lock_gate = lock_chamber.length.level + 0.5 * vessel.L
        if not direction:
            distance_to_position_in_lock = lock_chamber.distance_from_start_node_to_lock_gate_A + vessel.distance_position_from_first_lock_gate
        else:
            distance_to_position_in_lock = lock_chamber.distance_from_end_node_to_lock_gate_B + vessel.distance_position_from_first_lock_gate

        vessel.position_in_lock = calculate_location_over_edges(self.env.graph, edge, distance_to_position_in_lock, crs_m = self.crs_m)

        # let vessel sail to the assigned location in the lock chamber
        vessel_speed = lock_chamber.vessel_sailing_speed_in_lock(vessel)
        remaining_sailing_time = vessel.distance_position_from_first_lock_gate / vessel_speed
        while remaining_sailing_time > 0:
            try:
                yield vessel.env.timeout(remaining_sailing_time)
                remaining_sailing_time = 0
            except simpy.Interrupt as e:
                remaining_sailing_time -= vessel.env.now - start_sailing

        # log the stop of the sailing event to the assigned locaiton in the lock chamber
        vessel.log_entry_v0("Sailing to position in lock stop", vessel.env.now, vessel.output.copy(), vessel.position_in_lock,)

        # close gate if gate can be closed between vessel arrivals and gate have not already been closed before
        gate_can_be_closed_between_vessel_arrivals = lock_chamber.determine_if_gate_can_be_closed(vessel, direction, operation_index, between_arrivals=True)
        if not lock_chamber.close_gate_before_vessel_is_laying_still and gate_can_be_closed_between_vessel_arrivals:
            vessel.env.process(lock_chamber.close_gate())

    def convert_chamber(self, new_level, direction, operation_index=None, vessel=None, close_gate=True, delay=0.0):
        """
        Converts the lock chamber and logs this event. TODO: attribute for lock operator

        Parameters
        ----------
        new_level : str
            node that represents the side at which the lock is currently levelled
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        close_gate : bool
            if the gate have to be closed: yes (True) or no (False)
        delay : float
            a delay before lock conversion [s]

        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Yields
        ------
        The conversion of the lock chamber
        """
        if operation_index is not None:
            self.lock_complex.operation_planning.loc[operation_index, "status"] = "unavailable"

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
        yield from self.level_lock(new_level, direction, operation_index=operation_index, vessel=vessel)
        yield from self.open_gate()

    def close_gate(self, delay=0.0):
        """
        Lock operator closes the lock gate TODO: attribute for lock operator

        Parameters
        ----------
        delay : float
            a delay before gate opening [s]

        Yields
        ------
        The closing of the gate
        """

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
        self.log_entry_v0("Lock gate closing start", self.env.now, self.output.copy(), self.gate_open)

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
        if self.gate_open == self.start_node:
            node = self.start_node
        else:
            node = self.end_node

        hydromanager = HydrodynamicDataManager()
        time_index = hydromanager._get_time_index_of_hydrodynamic_data(time)
        new_water_level = hydromanager._get_hydrodynamic_data_value(time, node, "Water level")
        self.water_level[time_index:] = new_water_level

        # log the end of the event
        self.log_entry_v0("Lock gate closing stop", self.env.now, self.output.copy(), self.gate_open)
        if self.gate_open == self.start_node:
            self.gate_A_open = False
        else:
            self.gate_B_open = False

        # release all lock elements that were requested, so the next process can start
        self.gate_A.resource.release(hold_gate_A)
        self.levelling.resource.release(hold_levelling)
        self.gate_B.resource.release(hold_gate_B)

    def level_lock(self, new_level, direction, operation_index=None, vessel=None):
        """
        Lock operator levels the water level of the lock chamber to the harbour side of the direction of the lock operation TODO: attribute for lock operator

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
        levelling_time, _, _ = calculate_levelling_time(self, t_start=self.env.now, direction=direction, operation_index=operation_index)

        # log the start of the event
        if vessel is not None:
            vessel.log_entry_v0(
                "Levelling start",
                vessel.env.now,
                vessel.output.copy(),
                vessel.position_in_lock,
            )
        self.log_entry_v0(
            "Lock chamber converting start",
            self.env.now,
            self.output.copy(),
            self.gate_open,
        )

        # set new node to which the gate will be opened
        self.gate_open = new_level

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
        self.log_entry_v0(
            "Lock chamber converting stop",
            self.env.now,
            self.output.copy(),
            self.gate_open,
        )
        if vessel is not None:
            vessel.log_entry_v0(
                "Levelling stop",
                vessel.env.now,
                vessel.output.copy(),
                vessel.position_in_lock,
            )

        # release all lock elements that were requested, so the next process can start
        self.gate_A.resource.release(hold_gate_A)
        self.levelling.resource.release(hold_levelling)
        self.gate_B.resource.release(hold_gate_B)

    def open_gate(self, to_level=None, vessel=None, delay=0.0):
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

        # delete attribute as form of communication of the vessel TODO: a bit complex, better do it in another way
        if vessel is not None:
            delattr(vessel, "gate_open_request")

        hydromanager = HydrodynamicDataManager()

        # determine the water level in the lock chamber
        time = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))
        time_index = hydromanager._get_time_index_of_hydrodynamic_data(time)
        wlev_chamber = self.water_level[time_index]

        # determine to_level
        if to_level is None:
            to_level = self.gate_open

        # determine the water level in the harbour
        wlev_harbour = hydromanager._get_hydrodynamic_data_value(time, to_level, "Water level")

        # determine the direction to which the vessels are sailing out
        if to_level == self.start_node:
            direction = 1
        else:
            direction = 0

        # if the water levels in the chamber and harbour are not aligned -> level lock again
        if wlev_chamber is not None and wlev_harbour is not None and np.abs(wlev_chamber - wlev_harbour) >= 0.1:
            yield from self.level_lock(to_level, direction=direction)
        else:
            self.gate_open = to_level

        time = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))
        time_index = hydromanager._get_time_index_of_hydrodynamic_data(time)
        wlev_series_node_gate_open = hydromanager._get_hydrodynamic_data_series(time, self.gate_open, "Water level")
        self.water_level[time_index:] = wlev_series_node_gate_open

        # make sure that all lock elements are requested, so only one process is occurring
        hold_gate_A = self.gate_A.resource.request()
        hold_levelling = self.levelling.resource.request()
        hold_gate_B = self.gate_B.resource.request()
        yield hold_gate_A
        yield hold_levelling
        yield hold_gate_B

        # log the process start
        self.log_entry_v0("Lock gate opening start", self.env.now, self.output.copy(), self.gate_open)

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
        self.log_entry_v0(
            "Lock gate opening stop",
            self.env.now,
            self.output.copy(),
            self.gate_open,
        )

        # determine which side the gate is open to
        if self.gate_open == self.start_node:
            self.gate_A_open = True
        else:
            self.gate_B_open = True

        # release all lock elements that were requested, so the next process can start
        self.gate_A.resource.release(hold_gate_A)
        self.levelling.resource.release(hold_levelling)
        self.gate_B.resource.release(hold_gate_B)

    @property
    def minimum_delay_to_close_gate(self):
        """
        Calculates the time delay (in seconds) between when the last vessel has entered the lock and when the lock gate can be closed

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput

        Returns
        -------
        minimum_delay_to_close_gate : pd.Timedelta
            the minimum time delay that the lock gate can be closed after a vessel has entered the lock
        """
        minimum_delay_to_close_gate = pd.Timedelta(seconds=self.sailing_time_before_closing_lock_gate)
        return minimum_delay_to_close_gate

    @property
    def minimum_advance_to_open_gate(self):
        """
        Determines the minimum time in advance that a lock gate should be opened

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)


        Returns
        -------
        minimum_advance_to_open_gate : pd.Timedelta
            the minimum time in advance that a lock gate should be opened [s]

        """
        minimum_advance_to_open_gate = pd.Timedelta(seconds=self.sailing_time_before_opening_lock_gate)
        # minimum_advance_to_open_gate += pd.Timedelta(seconds=vessel.L/self.vessel_sailing_in_speed(vessel,direction))
        # TODO: take into account the vessels' bows and sterns to determine the time before and after which the gate should be respectively opened and closed
        return minimum_advance_to_open_gate

    def determine_if_gate_can_be_closed(self, vessel, direction, operation_index, between_arrivals=False):
        """
        Determines if the gate can be closed in between operations or vessel arrivals

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        between_arrivals : bool
            if the function is run to determine if the gate can be closed in between vessel arrivals (True) or not (False)

        Returns
        -------
        gate_can_be_closed : bool
            gate can be closed (True) or not (False)
        """
        operation_planning = self.lock_complex.operation_planning
        vessel_planning = self.lock_complex.vessel_planning

        this_operation = operation_planning.loc[operation_index]
        vessels_in_operation = this_operation.vessels
        last_vessel_to_enter_lock = vessels_in_operation[-1] == vessel

        gate_can_be_closed = False
        if not between_arrivals and not self.closing_gate_in_between_operations:
            return gate_can_be_closed
        if between_arrivals and (not self.closing_gate_in_between_arrivals or not last_vessel_to_enter_lock):
            return gate_can_be_closed
        gate_can_be_closed = True

        if not between_arrivals:
            last_time_gate_closed = operation_planning.loc[operation_index, "time_potential_lock_gate_closure_start"]
        else:
            last_time_gate_closed = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        last_time_gate_closed += pd.Timedelta(seconds=self.gate_closing_time)

        next_operations = operation_planning[operation_planning.index > operation_index]
        vessel_index = operation_planning.loc[operation_index, "vessels"].index(vessel)
        vessels_in_operation = operation_planning.loc[operation_index, "vessels"]

        operation_step = 1
        if between_arrivals and vessel_index != len(vessels_in_operation) - 1:
            next_vessel = vessels_in_operation[vessel_index + 1]
            next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name
            gate_required_to_be_open = vessel_planning.loc[next_vessel_planning_index, "time_potential_lock_gate_opening_stop"]
            same_direction = True
        elif not next_operations.empty:
            next_operation = next_operations.iloc[0]
            if not len(next_operation.vessels):
                next_operation = next_operations.iloc[1]
                operation_step += 1
            gate_required_to_be_open = next_operation.time_potential_lock_gate_opening_stop
            same_direction = direction != next_operation.direction
        else:
            return gate_can_be_closed

        if same_direction:
            direction = 1 - direction

        gate_opening_time = self.determine_time_to_open_gate(operation_index + operation_step, direction, gate_required_to_be_open)

        if (
            gate_required_to_be_open - gate_opening_time < last_time_gate_closed
            or gate_required_to_be_open - last_time_gate_closed
            < self.minimum_time_between_operations_for_intermediate_gate_closure
        ):
            gate_can_be_closed = False
        return gate_can_be_closed

    def determine_if_gate_is_closed(self, vessel, operation_index, direction, first_in_lock=False, between_arrivals=False):
        """
        Determines if the gate are closed

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        first_in_lock : bool
            if the function is run for the first vessel assigned to the lock operation (True) or not (False)
        between_arrivals : bool
            if the function is run to determine if the gate can be closed in between vessel arrivals (True) or not (False)

        Returns
        -------
        gate_are_closed : bool
            gate are closed (True) or not (False)
        gate_required_to_be_open : pd.Timestamp
            moment in time when the gate need to be opened
        operation_time : pd.Timedelta
            the time duration required to perform the lock operation
        """
        operation_planning = self.lock_complex.operation_planning
        vessel_planning = self.lock_complex.vessel_planning
        vessels = operation_planning.loc[operation_index, "vessels"]
        vessel_index = vessels.index(vessel)

        if between_arrivals and not self.closing_gate_in_between_arrivals:
            return False, None, None

        if not between_arrivals and not self.closing_gate_in_between_operations:
            return False, None, None

        last_lockage_was_empty = False
        if operation_index - 2 in operation_planning.index:
            last_lockage_was_empty = len(operation_planning.loc[operation_index - 1, "vessels"]) == 0
        if last_lockage_was_empty:
            return False, None, None

        if not first_in_lock and vessel_index:
            previous_vessel_planning_index = (
                vessel_planning[vessel_planning.id == operation_planning.loc[operation_index, "vessels"][vessel_index - 1].id]
                .iloc[-1]
                .name
            )
            last_time_gate_closed = vessel_planning.loc[
                previous_vessel_planning_index, "time_potential_lock_gate_closure_start"
            ] + pd.Timedelta(seconds=self.gate_closing_time)
        elif operation_index == 0:
            last_time_gate_closed = datetime.datetime.fromtimestamp(self.env.now)
        else:
            last_time_gate_closed = operation_planning.loc[
                operation_index - 1
            ].time_potential_lock_gate_closure_start + pd.Timedelta(seconds=self.gate_closing_time)

        if first_in_lock:
            gate_required_to_be_open = operation_planning.loc[operation_index, "time_potential_lock_gate_opening_stop"]
        else:
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            gate_required_to_be_open = vessel_planning.loc[vessel_planning_index, "time_potential_lock_gate_opening_stop"]

        operation_time = self.determine_time_to_open_gate(operation_index, direction, gate_required_to_be_open)
        gate_are_closed = False

        if (
            gate_required_to_be_open - operation_time > last_time_gate_closed
            and gate_required_to_be_open - last_time_gate_closed
            > self.minimum_time_between_operations_for_intermediate_gate_closure
        ):
            gate_are_closed = True

        return gate_are_closed, gate_required_to_be_open, operation_time

    def determine_time_to_open_gate(self, operation_index, direction, gate_required_to_be_open):
        """
        Determines the time to finish the levelling process and the gate opening process

        Parameters
        ----------
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        gate_required_to_be_open : pd.Timestamp
            the moment in time that the gate are required to be opened

        Returns
        -------
        operation_time : pd.Timedelta
            the time to finish the levelling process and the gate opening process
        """
        last_entering_time = gate_required_to_be_open - pd.Timedelta(seconds=self.gate_opening_time)
        operation_start_time = gate_required_to_be_open - pd.Timedelta(seconds=self.gate_opening_time)
        levelling_information = calculate_lock_operation_times(
            self,
            operation_index=operation_index,
            last_entering_time=last_entering_time,
            start_time=operation_start_time,
            direction=direction,
        )

        levelling_time = levelling_information["time_levelling_stop"] - levelling_information["time_levelling_start"]
        wlev_before, wlev_after = levelling_information["wlev_A"], levelling_information["wlev_B"]

        levelling_required = True
        if abs(wlev_after - wlev_before) < 0.1:
            levelling_required = False

        if not levelling_required:
            levelling_time = pd.Timedelta(seconds=0.0)

        operation_time = levelling_time + pd.Timedelta(seconds=self.gate_opening_time)
        return operation_time