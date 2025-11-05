"""Contains the mixin for lock chambers. Also contains the parent class LockChamberOperatior."""
import simpy
import numpy as np
import pandas as pd
import datetime
import math
import functools

from opentnsim.core import HasResource, Identifiable, Log, HasLength, ExtraMetadata
from opentnsim.utils import time_to_numpy
from opentnsim.lock.calculations import calculate_z, levelling_time_equation
from opentnsim.lock.utils import _get_lock_operation_to_and_from_node
from opentnsim.vessel_traffic_service.hydrodanamic_data_manager import HydrodynamicDataManager
from opentnsim.output import HasOutput
from opentnsim.graph.mixins import HasMultiDiGraph, get_length_of_edge
from opentnsim.constants import knots


class IsLockChamberOperator:
    """The lock chamber operator operates one chamber of the lock.

    The operator communicates with the lock master through self.lock_master
    to coordinate vessel operations and lock state changes.
    """

    def __init__(self, lock_master, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock_master = lock_master

    def _get_appropriate_waiting_area(self, direction):
        """Get appropriate waiting area through lock master.

        Parameters
        ----------
        direction : int
            the direction of the vessel: 0 (A -> B) or 1 (B -> A)
        """
        if self.lock_master:
            return self.lock_master._get_appropriate_waiting_area(direction)
        else:
            raise AttributeError("No lock master set for accessing waiting areas")

    @property
    def closing_doors_in_between_arrivals(self):
        """Get policy on closing doors in between arrivals through lock master."""
        if self.lock_master:
            return self.lock_master.closing_doors_in_between_arrivals
        else:
            raise AttributeError("No lock master set for accessing door closing policy")

    @property
    def closing_doors_in_between_operations(self):
        """Get policy on closing doors in between operations through lock master."""
        if self.lock_master:
            return self.lock_master.closing_doors_in_between_operations
        else:
            raise AttributeError("No lock master set for accessing door closing policy")

    def close_doors_before_vessel_is_laying_still(self, operation_index):
        """Get policy on closing doors before vessel is laying still through lock master.

        Parameters
        ----------
        operation_index : int
            index of the lock operation
        """
        if self.lock_master:
            return self.lock_master.close_doors_before_vessel_is_laying_still(operation_index)
        else:
            raise AttributeError("No lock master set for accessing door closing before vessel laying still policy")

    @property
    def _distance_to_lock(self):
        """Get distance to lock through lock master."""
        if self.lock_master:
            return self.lock_master._distance_to_lock
        else:
            raise AttributeError("No lock master set for accessing distance to lock")

    @property
    def vessel_planning(self):
        """Get vessel planning through lock master."""
        if self.lock_master:
            return self.lock_master.vessel_planning
        else:
            raise AttributeError("No lock master set for accessing vessel planning")

    @property
    def operation_planning(self):
        """Get operation planning through lock master."""
        if self.lock_master:
            return self.lock_master.operation_planning
        else:
            raise AttributeError("No lock master set for accessing operation planning")

    @property
    def waiting_area_B(self):
        """Get waiting area B through lock master."""
        if self.lock_master:
            return self.lock_master.waiting_area_B
        else:
            raise AttributeError("No lock master set for accessing waiting area B")

    @property
    def waiting_area_A(self):
        """Get waiting area A through lock master."""
        if self.lock_master:
            return self.lock_master.waiting_area_A
        else:
            raise AttributeError("No lock master set for accessing waiting area A")

    def register_vessel(self, vessel):
        """Register vessel through lock master.

        Parameters
        ----------
        vessel : type
            a type including the following mixins: PassesLockComplex,Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        """
        if self.lock_master:
            yield from self.lock_master.register_vessel(vessel)

    def calculate_sailing_time_to_waiting_area(self, vessel, direction, current_node=None, prognosis=False, overwrite=True):
        """Calculate sailing time to waiting area through lock master.

        Parameters
        ----------
        vessel : type
            a type including the following mixins: PassesLockComplex,Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (A -> B) or 1 (B -> A)
        current_node : str
            node name (that has to be in the graph) on which the vessel is currently sailing, to navigate an edge should form an edge with the origin)
        prognosis : bool
            if the sailing time is calculated for prognosis purposes (True) or for actual sailing (False)
        overwrite : bool
            if existing sailing time in the vessel planning should be overwritten (True) or not (False)
        """

        if self.lock_master:
            return self.lock_master.calculate_sailing_time_to_waiting_area(vessel, direction, current_node, prognosis, overwrite)

    def calculate_vessel_departure_start_delay(self, vessel, operation_index, direction, prognosis=False):
        """Calculate vessel departure start delay through lock master.

        Parameters
        ----------
        vessel : type
            a type including the following mixins: PassesLockComplex,Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the vessel: 0 (A -> B) or 1 (B -> A)
        prognosis : bool
            if the delay is calculated for prognosis purposes (True) or for actual sailing (False)
        overwrite : bool
            if existing delay in the vessel planning should be overwritten (True) or not (False)
        """
        if self.lock_master:
            return self.lock_master.calculate_vessel_departure_start_delay(vessel, operation_index, direction, prognosis)

    def initiate_levelling(self, origin, destination, vessel=None, k=0):
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
        # TODO: Moeten de origin en destination hier naast elkaar liggen? Zoja, toevoegen in documentatie.

        # determine if there is a lock on the edge
        if "Lock" not in vessel.multidigraph.edges[origin, destination, k].keys():
            return

        # get the lock complex object
        lock = vessel.multidigraph.edges[origin, destination, k]["Lock"][0]

        # unpack the lock complex master's vessel and lock operation plannings
        vessel_planning = lock.vessel_planning
        operation_planning = lock.operation_planning

        # determine the index of the vessel and the lock operation to which it is assigned to and the index of this operation
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index, "operation_index"]
        this_operation = operation_planning.loc[operation_index]

        # determine the direction to the lock chamber is currently levelled to, and to which node the lock chamber will level
        current_node = lock.node_open
        if current_node == lock.start_node:
            direction = 0
            next_node = lock.end_node
        else:
            direction = 1
            next_node = lock.start_node

        # determine the vessels that are assigned to the lock operation to which the vessel is assigned
        vessels = this_operation.vessels

        # initiate levelling if vessel is the last assigned vessel in the lock
        if vessel == vessels[-1]:
            # liberate the vessels that were requested to wait for the last vessel
            for other_vessel in vessels[:-1]:
                terminate_waiting_time_for_other_vessel = False
                while not terminate_waiting_time_for_other_vessel:
                    try:
                        yield lock.wait_for_other_vessels.put(other_vessel)
                        terminate_waiting_time_for_other_vessel = True
                    except simpy.Interrupt as e:
                        terminate_waiting_time_for_other_vessel = False

            # Wait for other vessels to lay still
            delay = (operation_planning.loc[operation_index].time_door_closing_start.round("s").to_pydatetime().timestamp() - lock.env.now)
            if delay > 0:
                yield lock.env.timeout(delay)

            # Convert lock chamber
            close_doors = True
            if (lock.close_doors_before_vessel_is_laying_still
                and this_operation.time_door_closing_start < vessel_planning.loc[vessel_planning_index, "time_lock_entry_stop"]):
                close_doors = False

            lock.operation_planning.loc[operation_index, 'status'] = 'unavailable'
            yield from lock.convert_chamber(next_node, direction, operation_index=operation_index, vessel=vessel, close_doors=close_doors)

            # Liberate waiting vessels in lock chamber
            for other_vessel in vessels[:-1]:
                terminate_levelling_for_other_vessel = False
                while not terminate_levelling_for_other_vessel:
                    try:
                        yield lock.wait_for_levelling.put(other_vessel)
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
                    yield lock.wait_for_other_vessels.get(filter=(lambda request: request.id == vessel.id))
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
                    yield lock.wait_for_levelling.get(filter=(lambda request: request.id == vessel.id))
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
        sailing_out_delay = lock.calculate_vessel_departure_start_delay(vessel, operation_index, direction).total_seconds()
        delay_start = vessel.env.now
        while sailing_out_delay:
            try:
                yield vessel.env.timeout(sailing_out_delay)
                sailing_out_delay = 0
            except simpy.Interrupt as e:
                sailing_out_delay -= vessel.env.now - delay_start

    def prepare_next_lock_operation(self, lock, operation_index, direction, vessel):
        """Lock operator checks and (if required) initiates an empty lock operation or closes the doors if there is sufficient time with respect to the next operation's start time

        Parameters
        ----------
        lock : object
            the lock chamber object generated with IsLockChamber
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (A -> B) or 1 (B -> A)
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        """
        # get variables of the last lock operation: do nothing if it is not the last vessel that is sailing out of the lock
        operation_planning = self.lock_master.operation_planning
        last_operation = operation_planning.loc[operation_index]
        vessels_in_last_operation = last_operation.vessels
        is_last_vessel_sailing_out = vessels_in_last_operation[-1] == vessel
        if not is_last_vessel_sailing_out:
            return

        # get the current time, and the information of the next operation
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
        _, to_node = _get_lock_operation_to_and_from_node(self, 1 - direction)
        next_operations = operation_planning[operation_planning.index > operation_index]

        # determine if the doors can be closed after the considered vessel has sailed out of the lock
        doors_can_be_closed = lock.determine_if_door_can_be_closed(vessel, direction, operation_index)

        # determine if the next operation is empty
        next_lockage_is_empty = False
        if not next_operations.empty:
            next_operation = next_operations.iloc[0]
            if not len(next_operation.vessels):
                next_lockage_is_empty = True

        # an action should be done if the doors can be closed in between operations, or if the next lock operation is empty
        if doors_can_be_closed and lock.closing_doors_in_between_operations:
            door_closing_start_time = last_operation.time_potential_lock_door_closure_start
            delay = np.max([self.sailing_time_before_closing_lock_doors, (door_closing_start_time - current_time).total_seconds()])

            # close the doors with the correct delay
            vessel.env.process(lock.close_door(delay=delay))

        elif next_lockage_is_empty:
            door_closing_start_time = next_operation.time_door_closing_start
            closing_delay = np.max([self.sailing_time_before_closing_lock_doors, (door_closing_start_time - current_time).total_seconds()])

            # if there is an empty lock operation and no policy that doors are closed in between operations is active -> close doors and convert chamber afterwards
            if not lock.closing_doors_in_between_operations:
                convert_chamber_delay = closing_delay
                closing_doors = True
            # if there is an empty lock operation but the policy that doors are closed in between operations is active -> close doors and convert chamber later, or convert chamber immediately if there is insufficient time
            else:
                next_operation = next_operations.iloc[1]
                door_opening_start_time = next_operation.time_potential_lock_door_opening_stop
                lock_operation_duration = self.determine_time_to_open_door(operation_index = vessel_operation_index + 1,
                                                                           direction =1 - direction,
                                                                           doors_required_to_be_open = door_opening_start_time)
                opening_delay = (np.max([0, (door_opening_start_time - current_time).total_seconds()]) - lock_operation_duration.total_seconds())
                if opening_delay > (closing_delay + self.lock_master.doors_closing_time):
                    convert_chamber_delay = opening_delay
                    closing_doors = False
                    vessel.env.process(lock.close_door(delay=closing_delay))
                else:
                    convert_chamber_delay = closing_delay
                    closing_doors = True

            # convert the lock chamber with the correct delay and if the doors should first be closed
            vessel.env.process(lock.convert_chamber(operation_index = operation_index + 1,
                                                    new_level = to_node,
                                                    vessel = None,
                                                    close_doors = closing_doors,
                                                    delay = convert_chamber_delay,
                                                    direction = 1 - direction))

    def allow_vessel_to_sail_out_of_lock(self, origin, destination, vessel=None, k=0):
        """Allows the vessel to sail out of the lock chamber

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

        Yields
        ------
        Vessel to sail to the end of the edge at which the lock chamber is located, and initiates new processes: i.e. closing doors or empty lock operation
        """
        # checks if lock is present on the edge
        if "Lock" not in vessel.multidigraph.edges[origin, destination, k].keys():
            return

        # unpacks the lock and vessel and operation planning
        lock = vessel.multidigraph.edges[origin, destination, k]["Lock"][0]
        vessel_planning = lock.vessel_planning

        # determines information of the lock operation
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index, "operation_index"]
        direction = vessel_planning.loc[vessel_planning_index, "direction"]

        # determines the distance from the vessel to the lock doors that have to be passed
        distance_in_lock_from_position = lock.lock_length - vessel.distance_position_from_first_lock_doors

        # determines the geometry objects of the lock based on the direction of the vessel TODO: function?
        if not direction:
            second_lock_doors_position = lock.location_lock_doors_B
            remaining_distance = lock.distance_from_end_node_to_lock_doors_B
            exit_geom = vessel.env.graph.nodes[lock.end_node]["geometry"]
        else:
            second_lock_doors_position = lock.location_lock_doors_A
            remaining_distance = lock.distance_from_start_node_to_lock_doors_A
            exit_geom = vessel.env.graph.nodes[lock.start_node]["geometry"]

        # releasing the length of the lock TODO: only the yield statement should be kept: now it is prevented that vessels need to wait to put back their length, but in principle this should not occur although it sometimes occurs due to bugs
        release_lock_access = False
        while not release_lock_access:
            try:
                yield lock.length.put(vessel.L)
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

        # log that the vessel can start sailing out of the lock (up to the lock doors)
        vessel.log_entry_v0("Sailing to second lock doors start", vessel.env.now, vessel.output.copy(), vessel.position_in_lock,)

        # determine the process of sailing to the lock doors that have to be passed (distance to these doors divided by the sailing out speed of the vessel)
        vessel_speed = lock.vessel_sailing_speed_out_lock(vessel)
        sailing_out_time = distance_in_lock_from_position / vessel_speed
        sailing_out_start = vessel.env.now
        while sailing_out_time:
            try:
                yield vessel.env.timeout(sailing_out_time)
                sailing_out_time = 0
            except simpy.Interrupt as e:
                sailing_out_time -= vessel.env.now - sailing_out_start

        # log that the vessel can stops sailing out of the lock (up to the lock doors)
        vessel.log_entry_v0("Sailing to second lock doors stop", vessel.env.now, vessel.output.copy(), second_lock_doors_position,)

        # remove functions specific to passing the lock chamber
        remove_functions = [lock.allow_vessel_to_sail_into_lock, lock.initiate_levelling, lock.allow_vessel_to_sail_out_of_lock]
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
        self.prepare_next_lock_operation(lock, operation_index, direction, vessel)

        # log that sailing out of the lock complex is starting
        vessel.log_entry_v0("Sailing to lock complex exit start", vessel.env.now, vessel.output.copy(), second_lock_doors_position)

        # let the vessel sail to the end of the lock complex
        vessel_speed = lock.vessel_sailing_out_speed(vessel, direction)
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

    def allow_vessel_to_sail_into_lock(self, origin, destination, vessel=None, k=0):
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

        # checks if lock is present on the edge
        if "Lock" not in vessel.multidigraph.edges[origin, destination, k].keys():
            return

        # unpacks the lock and vessel and operation planning
        lock = vessel.multidigraph.edges[origin, destination, k]["Lock"][0]
        vessel_planning = lock.vessel_planning

        # determines information of the lock operation
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index, "operation_index"]
        direction = vessel_planning.loc[vessel_planning_index, "direction"]
        current_time = vessel.env.now

        # determines the geometry objects of the lock based on the direction of the vessel TODO: function?
        waiting_area = lock._get_appropriate_waiting_area(direction)
        distance_to_lock = lock._distance_to_lock(direction)
        if not direction:
            first_lock_door_position = lock.location_lock_doors_A
        else:
            first_lock_door_position = lock.location_lock_doors_B

        # correct the distance to the lock doors if the vessel is in the waiting area, located at the same edge of the lock
        lock_start_node, lock_end_node = _get_lock_operation_to_and_from_node(self, direction)
        if (lock_start_node, lock_end_node) == waiting_area.edge:
            distance_to_lock -= waiting_area.distance_from_edge_start

        # log the start of sailing to the lock doors
        last_position_vessel = vessel.logbook[-1]["Geometry"]
        vessel.log_entry_v0("Sailing to first lock doors start", vessel.env.now, vessel.output.copy(), last_position_vessel,)

        # let vessel sail to the lock doors
        vessel_speed = lock.vessel_sailing_in_speed(vessel, direction)
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
        yield lock.length.get(vessel.L)

        # log the stop of sailing to the lock doors
        vessel.log_entry_v0("Sailing to first lock doors stop", vessel.env.now, vessel.output.copy(), first_lock_door_position,)

        # Checks if door should be closed intermediately
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

        # calculate delay to close doors
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
        delay_to_close_doors = vessel_planning.loc[vessel_planning_index, "time_potential_lock_door_closure_start"] - current_time

        # close doors if doors can be closed in between vessel arrivals or if vessel is last vessel to enter the lock
        doors_can_be_closed_between_vessel_arrivals = lock.determine_if_door_can_be_closed(vessel, direction, operation_index, between_arrivals=True)
        if lock.close_doors_before_vessel_is_laying_still and doors_can_be_closed_between_vessel_arrivals:
            vessel.env.process(lock.close_door(delay=delay_to_close_doors.total_seconds()))

        # log the start of sailing to the position within the lock chamber
        vessel.log_entry_v0("Sailing to position in lock start", vessel.env.now, vessel.output.copy(), first_lock_door_position, )

        # determine position in the lock chamber and distance to sail to this location
        vessel.distance_position_from_first_lock_doors = lock.length.level + 0.5 * vessel.L
        if not direction:
            distance_to_position_in_lock = lock.distance_from_start_node_to_lock_doors_A + vessel.distance_position_from_first_lock_doors
        else:
            distance_to_position_in_lock = lock.distance_from_end_node_to_lock_doors_B + vessel.distance_position_from_first_lock_doors
        vessel.position_in_lock = vessel.env.vessel_traffic_service.provide_location_over_edges(lock_start_node, lock_end_node, distance_to_position_in_lock)

        # let vessel sail to the assigned location in the lock chamber
        vessel_speed = lock.vessel_sailing_speed_in_lock(vessel)
        remaining_sailing_time = vessel.distance_position_from_first_lock_doors / vessel_speed
        while remaining_sailing_time > 0:
            try:
                yield vessel.env.timeout(remaining_sailing_time)
                remaining_sailing_time = 0
            except simpy.Interrupt as e:
                remaining_sailing_time -= vessel.env.now - start_sailing

        # log the stop of the sailing event to the assigned locaiton in the lock chamber
        vessel.log_entry_v0("Sailing to position in lock stop", vessel.env.now, vessel.output.copy(), vessel.position_in_lock,)

        # close doors if doors can be closed between vessel arrivals and doors have not already been closed before
        doors_can_be_closed_between_vessel_arrivals = lock.determine_if_door_can_be_closed(vessel, direction, operation_index, between_arrivals=True)
        if not lock.close_doors_before_vessel_is_laying_still and doors_can_be_closed_between_vessel_arrivals:
            vessel.env.process(lock.close_door())

    def convert_chamber(self, new_level, direction, operation_index=None, vessel=None, close_doors=True, delay=0.0):
        """
        Converts the lock chamber and logs this event. TODO: attribute for lock operator

        Parameters
        ----------
        new_level : str
            node that represents the side at which the lock is currently levelled
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        close_doors : bool
            if the doors have to be closed: yes (True) or no (False)
        delay : float
            a delay before lock conversion [s]

        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Yields
        ------
        The conversion of the lock chamber
        """
        if operation_index is not None:
            self.lock_master.operation_planning.loc[operation_index, "status"] = "unavailable"

        # if there is a delay -> yield time out
        start_delay = self.env.now
        while delay > 0:
            try:
                yield self.env.timeout(delay)
                delay = 0
            except simpy.Interrupt as e:
                delay -= self.env.now - start_delay

        # close the doors or make sure that lock is not performing another process
        if close_doors:
            yield from self.close_door(delay=delay)
        else:
            hold_door_A = self.door_A.request()
            hold_levelling = self.levelling.request()
            hold_door_B = self.door_B.request()
            yield hold_door_A
            yield hold_levelling
            yield hold_door_B
            self.door_A.release(hold_door_A)
            self.levelling.release(hold_levelling)
            self.door_B.release(hold_door_B)

        # level lock and open the doors afterwards
        yield from self.level_lock(new_level, direction, operation_index=operation_index, vessel=vessel)
        yield from self.open_door()

    def close_door(self, delay=0.0):
        """
        Lock operator closes the lock doors TODO: attribute for lock operator

        Parameters
        ----------
        delay : float
            a delay before door opening [s]

        Yields
        ------
        The closing of the door
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
        hold_door_A = self.door_A.request()
        hold_levelling = self.levelling.request()
        hold_door_B = self.door_B.request()
        yield hold_door_A
        yield hold_levelling
        yield hold_door_B

        # log the start of the event
        self.log_entry_v0("Lock doors closing start", self.env.now, self.output.copy(), self.node_open)

        # timeout event of the doors closing
        remaining_doors_closing_time = self.doors_closing_time
        start_time_closing = self.env.now
        while remaining_doors_closing_time:
            try:
                yield self.env.timeout(remaining_doors_closing_time)
                remaining_doors_closing_time = 0
            except simpy.Interrupt as e:
                remaining_doors_closing_time -= self.env.now - start_time_closing

        # set water level to the side at which the door has been closed
        time = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))
        if self.node_open == self.start_node:
            node = self.start_node
        else:
            node = self.end_node
        time_index = HydrodynamicDataManager()._get_time_index_of_hydrodynamic_data(
            self.env.vessel_traffic_service.hydrodynamic_information_path, time
        )
        new_water_level = HydrodynamicDataManager()._get_hydrodynamic_data_value(
            self.env.vessel_traffic_service.hydrodynamic_information_path, time, node, "Water level"
        )
        self.water_level[time_index:] = new_water_level

        # log the end of the event
        self.log_entry_v0("Lock doors closing stop", self.env.now, self.output.copy(), self.node_open)
        if self.node_open == self.start_node:
            self.door_A_open = False
        else:
            self.door_B_open = False

        # release all lock elements that were requested, so the next process can start
        self.door_A.release(hold_door_A)
        self.levelling.release(hold_levelling)
        self.door_B.release(hold_door_B)

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
        hold_door_A = self.door_A.request()
        hold_levelling = self.levelling.request()
        hold_door_B = self.door_B.request()
        yield hold_door_A
        yield hold_levelling
        yield hold_door_B

        # determine the levelling time
        levelling_time, _, _ = self.determine_levelling_time(t_start=self.env.now, direction=direction, operation_index=operation_index)

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
            self.node_open,
        )

        # set new node to which the doors will be opened
        self.node_open = new_level

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
            self.node_open,
        )
        if vessel is not None:
            vessel.log_entry_v0(
                "Levelling stop",
                vessel.env.now,
                vessel.output.copy(),
                vessel.position_in_lock,
            )

        # release all lock elements that were requested, so the next process can start
        self.door_A.release(hold_door_A)
        self.levelling.release(hold_levelling)
        self.door_B.release(hold_door_B)

    def open_door(self, to_level=None, vessel=None, delay=0.0):
        """
        Lock operator opens the lock doors TODO: attribute for lock operator

        Parameters
        ----------
        to_level : str
            node of the edge of lock complex to which the lock chamber opens
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        delay : float
            a delay before door opening

        Yields
        ------
        The opening of the door
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
            delattr(vessel, "door_open_request")

        hydromanager = HydrodynamicDataManager()

        # determine the water level in the lock chamber
        time = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))
        time_index = hydromanager._get_time_index_of_hydrodynamic_data(
            self.env.vessel_traffic_service.hydrodynamic_information_path, time
        )
        wlev_chamber = self.water_level[time_index]

        # determine to_level
        if to_level is None:
            to_level = self.node_open

        # determine the water level in the harbour
        wlev_harbour = hydromanager._get_hydrodynamic_data_value(
            self.env.vessel_traffic_service.hydrodynamic_information_path, time, to_level, "Water level"
        )

        # determine the direction to which the vessels are sailing out
        if to_level == self.start_node:
            direction = 1
        else:
            direction = 0

        # if the water levels in the chamber and harbour are not aligned -> level lock again
        if not math.isnan(wlev_chamber) and not math.isnan(wlev_harbour) and np.abs(wlev_chamber - wlev_harbour) >= 0.1:
            yield from self.level_lock(to_level, direction=direction)
        else:
            self.node_open = to_level

        time = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))
        time_index = hydromanager._get_time_index_of_hydrodynamic_data(
            self.env.vessel_traffic_service.hydrodynamic_information_path, time
        )
        wlev_series_node_door_open = hydromanager._get_hydrodynamic_data_series(
            self.env.vessel_traffic_service.hydrodynamic_information_path, time, self.node_open, "Water level"
        )
        self.water_level[time_index:] = wlev_series_node_door_open

        # make sure that all lock elements are requested, so only one process is occurring
        hold_door_A = self.door_A.request()
        hold_levelling = self.levelling.request()
        hold_door_B = self.door_B.request()
        yield hold_door_A
        yield hold_levelling
        yield hold_door_B

        # log the process start
        self.log_entry_v0("Lock doors opening start", self.env.now, self.output.copy(), self.node_open)

        # timeout
        remaining_doors_opening_time = self.doors_opening_time
        start_time_opening = self.env.now
        while remaining_doors_opening_time:
            try:
                yield self.env.timeout(remaining_doors_opening_time)
                remaining_doors_opening_time = 0
            except simpy.Interrupt as e:
                remaining_doors_opening_time -= self.env.now - start_time_opening

        # log the process stop
        self.log_entry_v0(
            "Lock doors opening stop",
            self.env.now,
            self.output.copy(),
            self.node_open,
        )

        # determine which side the door is open to
        if self.node_open == self.start_node:
            self.door_A_open = True
        else:
            self.door_B_open = True

        # release all lock elements that were requested, so the next process can start
        self.door_A.release(hold_door_A)
        self.levelling.release(hold_levelling)
        self.door_B.release(hold_door_B)

    def minimum_delay_to_close_doors(self):
        """
        Calculates the time delay (in seconds) between when the last vessel has entered the lock and when the lock doors can be closed

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput

        Returns
        -------
        minimum_delay_to_close_doors : pd.Timedelta
            the minimum time delay that the lock doors can be closed after a vessel has entered the lock
        """
        minimum_delay_to_close_doors = pd.Timedelta(seconds=self.sailing_time_before_closing_lock_doors)
        return minimum_delay_to_close_doors

    def minimum_advance_to_open_doors(self):
        """
        Determines the minimum time in advance that a lock door should be opened

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)


        Returns
        -------
        minimum_advance_to_open_doors : pd.Timedelta
            the minimum time in advance that a lock door should be opened [s]

        """
        minimum_advance_to_open_doors = pd.Timedelta(seconds=self.sailing_time_before_opening_lock_doors)
        # minimum_advance_to_open_doors += pd.Timedelta(seconds=vessel.L/self.vessel_sailing_in_speed(vessel,direction))
        # TODO: take into account the vessels' bows and sterns to determine the time before and after which the door should be respectively opened and closed
        return minimum_advance_to_open_doors

    def determine_if_door_can_be_closed(self, vessel, direction, operation_index, between_arrivals=False):
        """
        Determines if the doors can be closed in between operations or vessel arrivals

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        between_arrivals : bool
            if the function is run to determine if the doors can be closed in between vessel arrivals (True) or not (False)

        Returns
        -------
        doors_can_be_closed : bool
            doors can be closed (True) or not (False)
        """
        operation_planning = self.lock_master.operation_planning
        this_operation = operation_planning.loc[operation_index]
        vessels_in_operation = this_operation.vessels
        last_vessel_to_enter_lock = vessels_in_operation[-1] == vessel

        doors_can_be_closed = False
        if not between_arrivals and not self.lock_master.closing_doors_in_between_operations:
            return doors_can_be_closed
        if between_arrivals and (not self.closing_doors_in_between_arrivals or not last_vessel_to_enter_lock):
            return doors_can_be_closed
        doors_can_be_closed = True

        operation_planning = self.lock_master.operation_planning
        vessel_planning = self.lock_master.vessel_planning

        if not between_arrivals:
            last_time_doors_closed = operation_planning.loc[operation_index, "time_potential_lock_door_closure_start"]
        else:
            last_time_doors_closed = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        last_time_doors_closed += pd.Timedelta(seconds=self.doors_closing_time)

        next_operations = operation_planning[operation_planning.index > operation_index]
        vessel_index = operation_planning.loc[operation_index, "vessels"].index(vessel)
        vessels_in_operation = operation_planning.loc[operation_index, "vessels"]

        operation_step = 1
        if between_arrivals and vessel_index != len(vessels_in_operation) - 1:
            next_vessel = vessels_in_operation[vessel_index + 1]
            next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name
            doors_required_to_be_open = vessel_planning.loc[next_vessel_planning_index, "time_potential_lock_door_opening_stop"]
            same_direction = True
        elif not next_operations.empty:
            next_operation = next_operations.iloc[0]
            if not len(next_operation.vessels):
                next_operation = next_operations.iloc[1]
                operation_step += 1
            doors_required_to_be_open = next_operation.time_potential_lock_door_opening_stop
            same_direction = direction != next_operation.direction
        else:
            return doors_can_be_closed

        if same_direction:
            direction = 1 - direction

        door_opening_time = self.determine_time_to_open_door(operation_index + operation_step, direction, doors_required_to_be_open)

        if (
            doors_required_to_be_open - door_opening_time < last_time_doors_closed
            or doors_required_to_be_open - last_time_doors_closed
            < self.minimum_time_between_operations_for_intermediate_door_closure
        ):
            doors_can_be_closed = False
        return doors_can_be_closed

    def determine_if_door_is_closed(self, vessel, operation_index, direction, first_in_lock=False, between_arrivals=False):
        """
        Determines if the doors are closed

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
            if the function is run to determine if the doors can be closed in between vessel arrivals (True) or not (False)

        Returns
        -------
        doors_are_closed : bool
            doors are closed (True) or not (False)
        doors_required_to_be_open : pd.Timestamp
            moment in time when the doors need to be opened
        operation_time : pd.Timedelta
            the time duration required to perform the lock operation
        """
        operation_planning = self.lock_master.operation_planning
        vessel_planning = self.lock_master.vessel_planning
        vessels = operation_planning.loc[operation_index, "vessels"]
        vessel_index = vessels.index(vessel)

        if between_arrivals and not self.closing_doors_in_between_arrivals:
            return False, None, None

        if not between_arrivals and not self.lock_master.closing_doors_in_between_operations:
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
            last_time_doors_closed = vessel_planning.loc[
                previous_vessel_planning_index, "time_potential_lock_door_closure_start"
            ] + pd.Timedelta(seconds=self.doors_closing_time)
        elif operation_index == 0:
            last_time_doors_closed = datetime.datetime.fromtimestamp(self.env.now)
        else:
            last_time_doors_closed = operation_planning.loc[
                operation_index - 1
            ].time_potential_lock_door_closure_start + pd.Timedelta(seconds=self.doors_closing_time)

        if first_in_lock:
            doors_required_to_be_open = operation_planning.loc[operation_index, "time_potential_lock_door_opening_stop"]
        else:
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            doors_required_to_be_open = vessel_planning.loc[vessel_planning_index, "time_potential_lock_door_opening_stop"]

        operation_time = self.determine_time_to_open_door(operation_index, direction, doors_required_to_be_open)
        doors_are_closed = False

        if (
            doors_required_to_be_open - operation_time > last_time_doors_closed
            and doors_required_to_be_open - last_time_doors_closed
            > self.minimum_time_between_operations_for_intermediate_door_closure
        ):
            doors_are_closed = True

        return doors_are_closed, doors_required_to_be_open, operation_time

    def determine_time_to_open_door(self, operation_index, direction, doors_required_to_be_open):
        """
        Determines the time to finish the levelling process and the door opening process

        Parameters
        ----------
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        doors_required_to_be_open : pd.Timestamp
            the moment in time that the doors are required to be opened

        Returns
        -------
        operation_time : pd.Timedelta
            the time to finish the levelling process and the door opening process
        """
        last_entering_time = doors_required_to_be_open - pd.Timedelta(seconds=self.doors_opening_time)
        operation_start_time = doors_required_to_be_open - pd.Timedelta(seconds=self.doors_opening_time)
        levelling_information = self.calculate_lock_operation_times(
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

        operation_time = levelling_time + pd.Timedelta(seconds=self.doors_opening_time)
        return operation_time

    def determine_water_levels_before_and_after_levelling(self, levelling_start, levelling_stop, direction):
        """
        Determines the water level at both sides of the lock

        Parameters
        ----------
        levelling_start : pd.Timestamp
            the start time of the levelling process
        levelling_stop : pd.Timestamp
            the stop time of the levelling process
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Returns
        ------
        wlev_A :
            the water level at side A [m] before or after the levelling process (depending on the direction of the operation)
        wlev_B :
            the actual water level at side B [m] before or after the levelling process (depending on the direction of the operation)

        """
        hydromanager = HydrodynamicDataManager()
        t_start = np.datetime64(levelling_start)
        t_stop = np.datetime64(levelling_stop)
        if not direction:
            wlev_A = hydromanager._get_hydrodynamic_data_value(
                self.env.vessel_traffic_service.hydrodynamic_information_path, t_start, self.start_node, "Water level"
            )
            wlev_B = hydromanager._get_hydrodynamic_data_value(
                self.env.vessel_traffic_service.hydrodynamic_information_path, t_stop, self.end_node, "Water level"
            )
        else:
            wlev_A = hydromanager._get_hydrodynamic_data_value(
                self.env.vessel_traffic_service.hydrodynamic_information_path, t_stop, self.start_node, "Water level"
            )
            wlev_B = hydromanager._get_hydrodynamic_data_value(
                self.env.vessel_traffic_service.hydrodynamic_information_path, t_start, self.end_node, "Water level"
            )

        return wlev_A, wlev_B

    def determine_levelling_time(self, t_start, direction, wlev_init=None, operation_index=0, prediction=False):
        """
        Calculates the levelling time of a lock operation

        Parameters
        ----------
        t_start :
            the start time of the levelling process
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        wlev_init : float
            initial water level in the lock chamber
        same_direction : bool
            states if the levelling process is predicted in the same direction as the last lock operation (True) or not (False)
        prediction : bool
            states if the levelling process is only predicted (True) or executed (False)

        Returns
        -------
        levelling_time : float
            the time duration of the levelling process
        t : list of float
            the time series of the levelling process
        z : list of float
            the water level difference series over the time of the levelling process
        """
        # TODO: functie maken om tstart om te zetten (met _to_array)
        # TODO: Bij andere klasses altijd checken of iets een datetime.datetime is. En als dit niet zo is een error inbouwen of hem gelijk omzetten.
        dt = self.time_step
        t_final = 3600  # maximum levelling time has been set to an hour
        t = np.arange(0, t_final + float(dt), float(dt))
        t_start = time_to_numpy(t_start)
        # if there is no hydrodynamic data included in the run, use the constant levelling time included in the lock object
        # if there is no hydrodynamic data included in the run, use the constant levelling time included in the lock object
        if self.env.vessel_traffic_service.hydrodynamic_information_path is None:
            levelling_time = self.levelling_time
            z = np.zeros(len(t))
            return levelling_time, t, z

        z, H_A, H_B = calculate_z(
            t=t,
            t_start=t_start,
            direction=direction,
            wlev_init=wlev_init,
            operation_index=operation_index,
            operation_planning=self.lock_master.operation_planning,
            hydrodynamic_information_path=self.env.vessel_traffic_service.hydrodynamic_information_path,
            start_node=self.start_node,
            end_node=self.end_node,
            node_open=self.node_open,
            epoch=self.env.epoch,
        )

        # if a function has been included to predict the levelling time based on the water level difference: calculate the levelling time based on the initial water level difference
        if callable(self.levelling_time):
            levelling_time = self.levelling_time(z[0])
            return levelling_time, t, z

        # if no function has been included: compute the levelling time based on Eq. 4.64 of Ports and Waterways Open Textbook (https://books.open.tudelft.nl/home/catalog/book/204)
        levelling_time, t, z = levelling_time_equation(
            t=t,
            z=z,
            lock_length=self.lock_length,
            lock_width=self.lock_width,
            disch_coeff=self.disch_coeff,
            gate_opening_time=self.gate_opening_time,
            opening_area=self.opening_area,
            t_start=t_start,
            dt=dt,
            direction=direction,
            water_level_difference_limit_to_open_doors=self.lock_master.water_level_difference_limit_to_open_doors,
            prediction=prediction,
            H_A=H_A,
            H_B=H_B,
        )

        # if this function was not ran as a prediction, but rather as the actual levelling event: update the water level time series of the lock chamber
        if not prediction:
            # TODO: de self.water_level wordt niet gebruikt, maar is wel leuk om als logging terug te zien na een berekening. Nadenken of we dat zo willen laten, of anders willen bijhouden.
            t_final = t_start + np.timedelta64(int(levelling_time))
            t_index_final = HydrodynamicDataManager()._get_time_index_of_hydrodynamic_data(
                self.env.vessel_traffic_service.hydrodynamic_information_path, t_final
            )
            if not direction:
                self.water_level[t_index_final:] = H_B[t_index_final:].copy()
            else:
                self.water_level[t_index_final:] = H_A[t_index_final:].copy()

        return levelling_time, t, z


class IsLockChamber(IsLockChamberOperator, HasResource, HasLength, Identifiable, Log, HasOutput, HasMultiDiGraph, ExtraMetadata):
    """Mixin class: lock complex has a lock chamber:

    creates a lock chamber with a resource which is requested when a vessels wants to enter the area with limited capacity

    Attributes
    ----------
    vessel_sailing_speed_in_lock :
        calculates the average speed in the lock when entering
    vessel_sailing_speed_out_lock :
        calculates the average speed in the lock when leaving
    vessel_sailing_in_speed :
        Calculates the average speed when sailing towards the lock chamber
    vessel_sailing_out_speed :
        Calculates the average speed when sailing away from the lock chamber
    determine_levelling_time :
        calculates the levelling time of a lock operation

    """

    def __init__(
        self,
        start_node,  # a string which indicates the location of the first pair of lock doors
        end_node,  # a string which indicates the location of the second pair of lock doors
        lock_length,  # a float which contains the length of the lock chamber
        lock_width,  # a float which contains the width of the lock chamber
        lock_depth,  # a float which contains the depth of the lock chamber
        lock_master=None,
        k=0,  # a int which is the identifier of the edge between two nodes in a multidigraph network
        distance_from_start_node_to_lock_doors_A=0.0,  # a float that is the distance between the start_node of the edge and the lock doors A [m]
        distance_from_end_node_to_lock_doors_B=0.0,  # a float that is the distance between the end_node of the edge and the lock doors B [m]
        registration_nodes=[],  # a list of str with the node names at which the vessels request registration to the lock complex master
        doors_opening_time=300.0,  # a float which contains the time it takes to open the doors [s]
        doors_closing_time=300.0,  # a float which contains the time it takes to close the doors [s]
        disch_coeff=0.4,  # a float which contains the discharge coefficient of filling system
        opening_area=12.0,  # a float which contains the cross-sectional area of filling system [m^2]
        opening_depth=None,  # a float which contains the depth at which filling system is located [m^2]
        speed_reduction_factor_lock_chamber=0.3,  # a float that is the reduction factor for the vessel speed from its original speed when entering the lock
        start_sailing_out_time_after_doors_have_been_opened=0.0,  # a float that is the time that the vessel wait to start sailing out of the lock after the doors have been opened after levelling [s]
        sailing_time_before_opening_lock_doors=600.0,  # a float that is the time that the doors are opened before a vessel arrives at the doors [s]
        sailing_time_before_closing_lock_doors=120.0,  # a float that is the time that the doors are closed after a vessel has sailed through the doors [s]
        minimum_time_between_operations_for_intermediate_door_closure=0.0,  # a float that is the minimum required time between lock operations that the lock doors can be both closed (to reduce salt intrusion) [s]
        sailing_distance_to_crossing_point=500.0,  # a float that is the distance at which vessels can safely pass each other in front of the lock (last vessel that sails out and first vessel that sails in) [m]
        passage_time_door=300.0,  # a float [s] ?
        sailing_in_time_gap_through_doors=180.0,  # a float that is the time gap after which the next vessel can sail into the lock through the lock doors (after another vessel has sailed through to enter the lock) [s]
        sailing_out_time_gap_through_doors=180.0,  # a float that is the time gap after which the next vessel can sail out of the lock through the lock doors (after another vessel has sailed through to leave the lock)[s]
        sailing_in_time_gap_after_berthing_previous_vessel=0.0,  # a float that is the time gap after which the next vessel can sail into the lock (after another vessel has berthed) [s]
        sailing_out_time_gap_after_berthing_previous_vessel=0.0,  # a float that is the time gap after which the next vessel can sail out of the lock (after another vessel has deberthed) [s]
        sailing_in_speed_A=2 * knots,  # a float that is the speed at which the vessel sails into the lock to the sea side [m/s]
        sailing_out_speed_A=2 * knots,  # a float that is the speed at which the vessel sails out of the lock to the sea side [m/s]
        sailing_in_speed_B=2 * knots,  # a float that is the speed at which the vessel sails into the lock to the canal side [m/s]
        sailing_out_speed_B=2
        * knots,  # a float that is the speed at which the vessel sails out of the lock to the canal side [m/s]
        minimum_manoeuvrability_speed=2
        * knots,  # a float that is the minimum speed at which the vessel is still safely manoeuvrable [m/s]
        levelling_time=600.0,  # a float that fixates the levelling time [s]
        time_step=10.0,  # a float that is the integration time step to determine the levelling time [s]
        gate_opening_time=60.0,  # a float that is the time it takes for the levelling gate to open [s]
        node_open=None,  # a string that is the node name to which the lock was last levelled to at the initial time of simulation (either start_node or end_node)
        conditions=None,  # maybe obsolete ???
        priority_rules=None,  # maybe obsolete ???
        used_as_one_way_traffic_regulation=False,  # maybe obsolete ???
        seed_nr=None,  # a int for the seed to fix the determination of the node_open when node_open is None
        *args,
        **kwargs,
    ):
        """Initialization"""
        # TODO: checken of alle inputs nodig zijn
        # TODO: checken of alle parents nodig zijn
        # TODO: parentklasse Lockmaster toevoegen

        # set input parameters as properties
        self.lock_length = lock_length
        self.lock_width = lock_width
        # TODO: @Floor lock_depth wordt niet gebruikt... Willen we die houden?
        self.lock_depth = lock_depth
        # TODO @Floor, is deze coefficient afhankelijk van de lock, of is dit een standaard coefficient die we ergens anders kunnen opslaan?
        self.disch_coeff = disch_coeff #0.4

        self.opening_area = opening_area
        if opening_depth is None:
            opening_depth = lock_depth/2
        self.opening_depth = opening_depth
        self.levelling_time = levelling_time
        self.start_sailing_out_time_after_doors_have_been_opened = start_sailing_out_time_after_doors_have_been_opened
        self.sailing_time_before_opening_lock_doors = sailing_time_before_opening_lock_doors
        self.sailing_time_before_closing_lock_doors = sailing_time_before_closing_lock_doors
        self.minimum_time_between_operations_for_intermediate_door_closure = minimum_time_between_operations_for_intermediate_door_closure
        self.sailing_in_time_gap_after_berthing_previous_vessel = sailing_in_time_gap_after_berthing_previous_vessel
        self.sailing_out_time_gap_after_berthing_previous_vessel = sailing_out_time_gap_after_berthing_previous_vessel
        self.sailing_in_speed_A = sailing_in_speed_A
        self.sailing_out_speed_A = sailing_out_speed_A
        self.sailing_in_speed_B = sailing_in_speed_B
        self.sailing_out_speed_B = sailing_out_speed_B
        self.sailing_distance_to_crossing_point = sailing_distance_to_crossing_point
        self.sailing_in_time_gap_through_doors = sailing_in_time_gap_through_doors
        self.sailing_out_time_gap_through_doors = sailing_out_time_gap_through_doors
        self.speed_reduction_factor = speed_reduction_factor_lock_chamber
        self.passage_time_door = passage_time_door
        self.start_node = start_node
        self.end_node = end_node
        self.k = k
        self.minimum_manoeuvrability_speed = minimum_manoeuvrability_speed
        self.node_open = node_open
        self.conditions = conditions
        self.time_step = time_step
        self.priority_rules = priority_rules
        self.registration_nodes = registration_nodes
        self.gate_opening_time = gate_opening_time
        self.door_A_open = True
        self.door_B_open = True
        if not registration_nodes:
            self.registration_nodes = [start_node,end_node]
        self.distance_from_start_node_to_lock_doors_A = distance_from_start_node_to_lock_doors_A
        self.distance_from_end_node_to_lock_doors_B = distance_from_end_node_to_lock_doors_B
        self.used_as_one_way_traffic_regulation = used_as_one_way_traffic_regulation
        self.converting_chamber = False

        # TODO: checken of de seed_nr en de random functie worden gebruikt.
        if seed_nr is not None:
            np.random.seed(seed_nr)

        # TODO: als lockmaster een parent klasse is, zou lock_complex=self weg moeten kunnen.
        # TODO: capaciteit = 100. checken of deze info overbodig is doordat er al een lock_length is. En anders kijken of de capaciteit op oneindig kan.
        super().__init__(
            lock_master=lock_master,
            capacity=100,
            length=lock_length,
            remaining_length=lock_length,
            *args,
            **kwargs,
        )

        self._verify_node_AB()

        if self.env.vessel_traffic_service.hydrodynamic_information_path is not None:
            hydro_manager = HydrodynamicDataManager()
            if isinstance(self.env.vessel_traffic_service.hydrodynamic_information_path,str):
                hydro_manager.hydrodynamic_data = Dataset(self.env.vessel_traffic_service.hydrodynamic_information_path)
            else:
                hydro_manager.hydrodynamic_data = self.env.vessel_traffic_service.hydrodynamic_information
            if isinstance(self.env.vessel_traffic_service.hydrodynamic_information_path, str):
                hydro_manager.hydrodynamic_times = (
                    hydro_manager.hydrodynamic_data["TIME"][:].data.astype("timedelta64[m]")
                    + self.env.vessel_traffic_service.hydrodynamic_start_time
                )
            else:
                hydro_manager.hydrodynamic_times = hydro_manager.hydrodynamic_data["TIME"][:]

        if self.node_open is None:
            self.node_open = np.random.choice([start_node, end_node])

        if self.closing_doors_in_between_operations:
            self.door_A_open = False
            self.door_B_open = False
        elif self.node_open == self.start_node:
            self.door_B_open = False
        else:
            self.door_A_open = False

        # Geometry on edge
        edge = (start_node, end_node, 0)
        edge_info = self.multidigraph.edges[edge]
        # TODO Checken of de distance bepalen werkt, en misschien automatiseren op basis van geometrie
        # TODO: nodes verwijderen uit graaf als die precies op de sluis liggen. (wellicht als voorbewerking van de graaf)
        # TODO: losse klasse maken van de lock-doors die locatable, hasresource (capacity=1) en identifiable is en eigenschap open/dicht heeft.

        edge_aligned_with_edge_geometry = self.env.vessel_traffic_service.check_if_geometry_is_aligned_with_edge(edge)
        start_node_geometry = start_node
        end_node_geometry = end_node
        distance_from_start_node_geometry_to_lock_doors_A = self.distance_from_start_node_to_lock_doors_A
        distance_from_start_node_geometry_to_lock_doors_B = self.distance_from_start_node_to_lock_doors_A + lock_length
        if not edge_aligned_with_edge_geometry:
            start_node_geometry = end_node
            end_node_geometry = start_node
            distance_from_start_node_geometry_to_lock_doors_B = self.distance_from_end_node_to_lock_doors_B
            distance_from_start_node_geometry_to_lock_doors_A = self.distance_from_end_node_to_lock_doors_B + lock_length

        self.location_lock_doors_A = self.env.vessel_traffic_service.provide_location_over_edges(start_node_geometry, end_node_geometry, distance_from_start_node_geometry_to_lock_doors_A)
        self.location_lock_doors_B = self.env.vessel_traffic_service.provide_location_over_edges(start_node_geometry, end_node_geometry, distance_from_start_node_geometry_to_lock_doors_B)

        self.lock_pos_length = simpy.Container(self.env, capacity=lock_length, init=lock_length)
        self.door_A= simpy.PriorityResource(self.env, capacity = 1)
        self.levelling = simpy.Resource(self.env, capacity=1)
        self.door_B = simpy.PriorityResource(self.env, capacity = 1)

        # TODO: kijken of onderstaande eigenschappen nodig zijn. en capacity op infinity zetten als mogelijk.
        self.wait_for_other_vessel_to_arrive = simpy.FilterStore(self.env,capacity=100000000)
        self.wait_for_levelling = simpy.FilterStore(self.env,capacity=100000000)
        self.wait_for_other_vessels = simpy.FilterStore(self.env,capacity=100000000)

        # Operating
        self.doors_opening_time = doors_opening_time
        self.doors_closing_time = doors_closing_time

        # TODO: maak ene functie _test_input() die bijvoorbeeld checkt of de nodes na elkaar liggen.
        # Water level
        assert start_node != end_node

        time = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))
        wlev_series = HydrodynamicDataManager()._get_hydrodynamic_data_series(
            self.env.vessel_traffic_service.hydrodynamic_information_path, time, self.node_open, "Water level"
        )
        self.water_level = wlev_series

        # TODO: in functie zetten.
        # TODO: In de documentatie zetten dat detecotr nodes op volgorde moeten komen. En ook een assert maken.
        for registration_node, lock_edge in zip(self.registration_nodes,[(self.start_node,self.end_node,self.k),(self.end_node,self.start_node,self.k)]):
            if 'Lock_registration_node' not in self.multidigraph.nodes[registration_node]:
                self.multidigraph.nodes[registration_node]['Lock_registration_node'] = lock_edge

        # Add to the graph:
        # TODO: In losse functie (add_lock_to_graph)
        if "graph" in dir(self.env):
            k = sorted(self.multidigraph[self.start_node][self.end_node], key=lambda x: get_length_of_edge(self.multidigraph,(self.start_node, self.end_node, x)))[0]
            # Add the lock to the edge or append it to the existing list
            if "Lock" not in self.multidigraph.edges[self.start_node, self.end_node, k].keys():
                self.multidigraph.edges[self.start_node, self.end_node, k]["Lock"] = [self]
                self.multidigraph.edges[self.end_node, self.start_node, k]["Lock"] = [self]
            else:
                self.multidigraph.edges[self.start_node, self.end_node, k]["Lock"].append(self)
                self.multidigraph.edges[self.end_node, self.start_node, k]["Lock"].append(self)

    def _verify_node_AB(self):
        """Function to verify if nodes A and B are part of the graph, and have an edge between them."""
        if self.start_node not in self.env.graph.nodes or self.end_node not in self.env.graph.nodes:
            raise ValueError(
                f"Lock chamber {self.name} has invalid node_A {self.start_node} or node_B {self.end_node} which are not part of the graph."
            )
        if not self.env.graph.has_edge(self.start_node, self.end_node):
            raise ValueError(
                f"Lock chamber {self.name} does not have an edge between node A {self.start_node} and node B {self.end_node}."
            )

    def vessel_sailing_speed_in_lock(self, vessel):
        """
        Calculates the average speed in the lock when entering

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        x_location_lock : float
            logintudinal coordinate in the lock to which the vessel is assigned [m]
        P_used : float
            the breaking power used by the vessel to gradually decelerate [kW]

        Returns
        -------
        speed : float
            the average speed in the lock from the lock doors to the location of berthing

        """
        # TODO: sailing_in_speed_B zou A of B moeten zijn. Checken of deze eigenschap vaker voorkomt.
        speed = self.sailing_in_speed_B
        if vessel.bound == 'inbound':
            speed = self.sailing_in_speed_A

        return speed

    def vessel_sailing_speed_out_lock(self, vessel):
        """
        Calculates the average speed to in the lock when leaving

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        x_location_lock : float
            logintudinal coordinate in the lock to which the vessel is assigned [m]
        P_used : float
            the breaking power used by the vessel to gradually decelerate [kW]

        Returns
        -------
        speed : float
            the average speed in the lock from the lock doors to the location of berthing

        """
        speed = self.sailing_out_speed_A
        if vessel.bound == 'inbound':
            speed = self.sailing_out_speed_B

        return speed

    def vessel_sailing_in_speed(self, vessel, direction):
        """
        Calculates the average speed when sailing towards the lock chamber

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        x_location_lock : float
            logintudinal coordinate in the lock to which the vessel is assigned [m]
        P_used : float
            the breaking power used by the vessel to gradually decelerate [kW]

        Returns
        -------
        speed : float
            the average speed in the lock from the lock doors to the location of berthing

        """
        # determine the edge on which the vessel is sailing and the distance to the lock doors
        edge = self._directional_edge(direction)

        # determine the speed of the vessel over the edge
        speed = vessel._compute_velocity_on_edge(edge[0], edge[1])

        # if there is an overruled speed on the edge, use this speed
        if "overruled_speed" in dir(vessel) and edge in vessel.overruled_speed.index:
            speed = vessel.overruled_speed.loc[edge, "Speed"]

        return speed

    def vessel_sailing_out_speed(self, vessel, direction, P_used=None, h0=17, until_crossing_point=False):
        """
        Calculates the average speed when sailing away from the lock chamber

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        P_used : float
            the breaking power used by the vessel to gradually decelerate [kW]
        until_crossing_point : bool


        Returns
        -------
        speed : float
            the average speed in the lock from the lock doors to the location of berthing

        """
        # determine the edge on which the vessel is sailing and the distance to the lock doors
        edge = self._directional_edge(direction)

        # determine the speed of the vessel over the edge
        speed = vessel._compute_velocity_on_edge(edge[0], edge[1])

        # if there is an overruled speed on the edge, use this speed
        if 'overruled_speed' in dir(vessel) and edge in vessel.overruled_speed.index:
            speed = vessel.overruled_speed.loc[edge, 'Speed']

        return speed

    def _directional_edge(self, direction):
        """get the edge of the lock chamber in the correct direction"""
        if not direction:
            return (self.start_node, self.end_node)
        else:
            return (self.end_node, self.start_node)
