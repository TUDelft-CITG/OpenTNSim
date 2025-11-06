"""This is the lock module as part of the OpenTNSim package. See the locking examples in the book for detailed descriptions."""

# package(s) related to the simulation
import datetime

import networkx as nx
import numpy as np
import pandas as pd
import functools
from shapely.ops import transform
import pyproj
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math

import warnings

# spatial libraries
from collections import namedtuple
import simpy

# from netCDF4 import Dataset
from IPython.display import display

from opentnsim.utils import inherit_docstring
from opentnsim.core import HasResource, Identifiable, Log, Movable, HasLength, SimpyObject, ExtraMetadata
from opentnsim.graph.mixins import HasMultiDiGraph, get_length_of_edge
from opentnsim.output import HasOutput
from opentnsim.vessel_traffic_service.hydrodanamic_data_manager import HydrodynamicDataManager
from opentnsim.lock.utils import (
    _get_lock_object_on_registration_node,
    _update_lock_operation_planning,
    _update_lock_vessel_planning,
    _get_lock_operation_to_and_from_node,
    determine_route_to_closest_waiting_area,
)
from opentnsim.lock.lock_chamber import IsLockChamber


@inherit_docstring
class HasLockPlanning:
    """This class keeps track of the lock-planning of a lock-master."""

    def __init__(self, *args, **kwargs):
        self.vessel_planning = pd.DataFrame(
            index=pd.Index([]),
            columns=[
                "id",
                "node_from",
                "node_to",
                "lock_chamber",
                "L",
                "B",
                "T",
                "operation_index",
                "time_of_registration",
                "time_of_acceptance",
                "time_potential_lock_door_opening_stop",
                "time_arrival_at_waiting_area",
                "time_arrival_at_lineup_area",
                "time_lock_passing_start",
                "time_lock_entry_start",
                "time_lock_entry_stop",
                "time_lock_departure_start",
                "time_lock_departure_stop",
                "time_lock_passing_stop",
                "time_potential_lock_door_closure_start",
            ],
        )
        self.operation_planning = pd.DataFrame(
            index=pd.Index([], name="lock_operation"),
            columns=[
                "node_from",
                "node_to",
                "direction",
                "lock_chamber",
                "vessels",
                "capacity_L",
                "capacity_B",
                "time_potential_lock_door_opening_stop",
                "time_operation_start",  # See comments below
                "time_entry_start",  # See comments below
                "time_entry_stop",
                "time_door_closing_start",
                "time_door_closing_stop",
                "time_levelling_start",
                "time_levelling_stop",
                "time_door_opening_start",
                "time_door_opening_stop",
                "time_departure_start",
                "time_departure_stop",  # Note that start and stop times of different operations can overlap, but entry start and departure stop can not
                "time_operation_stop",  # Operation start and stop times are solely required when leaving and entering vessels need to pass each other at the safe crossing point
                "time_potential_lock_door_closure_start",
                "wlev_A",
                "wlev_B",
                "maximum_individual_delay",
                "total_delay",
                "status",
            ],
        )

    # TODO Add self.operations_performed. Hiermee kan je in de historie kijken wat er allemaal al is gebeurt.

    def get_vessel_from_planned_operation(self, operation_index):
        """
        Gets the vessels that are assigned to a certain lock operation in the operation planning of the lock master

        Parameters
        ----------
        operation_index : int
            index of the lock operation

        Returns
        -------
        vessels : list of vessel type objects
            the vessels that have been assigned to the specified lock operation (a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput)

        """
        # set default list of vessels (empty)
        vessels = []

        # determines the vessels in the lock operation
        selected_operation = self.lock_complex.operation_planning[self.lock_complex.operation_planning.index == operation_index]
        if not selected_operation.empty:
            vessels = selected_operation.loc[operation_index, "vessels"].copy()
        return vessels

    def _correct_lock_operation_start_time_if_outside_of_operational_hours(self, time_lock_operation_start):
        """Corrects the start time of the lock operation if it falls outside of the operational hours of the lock complex

        Parameters
        ----------
        time_lock_operation_start : pd.Timestamp
            the time when the operation is planned to start

        Returns
        -------
        time_lock_operation_start : pd.Timestamp
            the time when the operation will start
        """
        operational_hours = self.operational_hours
        within_operation_hours = operational_hours[(time_lock_operation_start >= operational_hours.start_time) & (time_lock_operation_start <= operational_hours.stop_time)]
        if within_operation_hours.empty:
            first_available_hour = operational_hours[operational_hours.start_time >= time_lock_operation_start].iloc[0]
            time_lock_operation_start = first_available_hour.start_time
        return time_lock_operation_start

    def _plan_or_execute_empty_lock_operation_if_required(self, operation_index, direction):
        """Plans an empty lock operation if this is required and executes it if it is the first lock operation

        Parameters
        ----------
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Returns
        -------
        operation_index : int
            index of the lock operation (+1 if an empty lock operation was required)
        """
        node_of_approach, to_node = _get_lock_operation_to_and_from_node(self, direction)
        previous_planned_operations = self.operation_planning[self.operation_planning.index < operation_index]
        if not previous_planned_operations.empty:
            previous_planned_operation = previous_planned_operations.iloc[-1]
            if previous_planned_operation.direction == direction:
                self.add_empty_lock_operation_to_planning(operation_index, 1 - direction)
                operation_index += 1  # the new operation index lies now one ahead
        elif self.node_open != node_of_approach:
            self.add_empty_lock_operation_to_planning(operation_index, 1 - direction)
            self.env.process(
                self.lock_chamber.convert_chamber(new_level=node_of_approach, direction=1 - direction)
            )  # TODO: check if this can be done differently with Fedor
            operation_index += 1  # the new operation index lies now one ahead
        return operation_index

    def _determine_lock_operation_start_information(self, vessel, operation_index, direction):
        """Determines the

        Parameters
        ----------
        vessel :

        operation_index :

        direction :

        :return:
        """
        # determine the index of the vessel in the lock master's vessel planning
        vessel_planning_index = self.vessel_planning[self.vessel_planning.id == vessel.id].iloc[-1].name

        # determine the earlier possible arrival time of the vessel (vessel perspective)
        earliest_possible_time_lock_entry_start = self.vessel_planning.loc[vessel_planning_index, "time_lock_entry_start"]

        # determine the time that the lock operation can start (operation perspective)
        time_lock_operation_start = self.calculate_lock_operation_start_time(vessel, operation_index, direction, prognosis=True)

        # correct the start time of the lock operation if it will fall outside of the operation hours of the lock complex
        time_lock_operation_start = self._correct_lock_operation_start_time_if_outside_of_operational_hours(time_lock_operation_start)

        # determine the time that vessel can start entering the lock
        time_lock_entry_start = self.calculate_lock_entry_start_time(vessel, operation_index, direction, time_lock_operation_start)

        # determine the minimum time that doors should be opened in advance of a vessel arrival and add this to the vessel planning
        minimum_advance_to_open_doors = self.lock_chamber.minimum_advance_to_open_doors()
        time_potential_lock_door_opening_stop = time_lock_entry_start - minimum_advance_to_open_doors

        previous_planned_operations = self.operation_planning[self.operation_planning.index < operation_index]
        if not previous_planned_operations.empty:
            previous_operation = previous_planned_operations.iloc[-1]
            if not len(previous_operation.vessels):
                if time_potential_lock_door_opening_stop < previous_operation.time_operation_stop:
                    operation_delay = previous_operation.time_operation_stop - time_potential_lock_door_opening_stop
                    time_lock_operation_start += operation_delay
                    time_lock_entry_start += operation_delay
                    time_potential_lock_door_opening_stop += operation_delay

        # determine the lock entry stop and door opening stop time
        time_lock_entry_stop = self.calculate_lock_entry_stop_time(vessel, operation_index, direction, time_lock_operation_start, prognosis=True)
        time_lock_door_opening_stop = self.calculate_lock_door_opening_time(vessel, operation_index, direction, time_lock_operation_start)

        # determine the delay time for the vessel to enter the lock
        vessel_entry_delay = time_lock_entry_start - earliest_possible_time_lock_entry_start

        return (time_lock_operation_start,
                time_lock_entry_start,
                minimum_advance_to_open_doors,
                time_potential_lock_door_opening_stop,
                time_lock_entry_stop,
                time_lock_door_opening_stop,
                vessel_entry_delay)

    def add_vessel_to_new_lock_operation(self, vessel, operation_index, direction):
        """
        Adds a vessel to a newly to be planned lock operation

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        """
        # unpack the lock master's vessel and lock operation plannings
        vessel_planning = self.lock_complex.vessel_planning
        operation_planning = self.lock_complex.operation_planning
        node_of_approach, to_node = _get_lock_operation_to_and_from_node(self,direction)

        # determine if the new lock operation should follow a empty lock operation (when the new lock operation has the same direction as the previous lock operation)
        operation_index = self._plan_or_execute_empty_lock_operation_if_required(operation_index, direction)

        # determine the index of the vessel in the lock master's vessel planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

        # add operation to the planning with information
        vessel_passage_information = {"operation_index":operation_index}
        _update_lock_vessel_planning(self, vessel_planning_index, vessel_passage_information)

        lock_operation_information = {"node_from":node_of_approach,
                                      "node_to":to_node,
                                      "direction":direction,
                                      "lock_chamber":self.lock_complex.name,
                                      "vessels":[],  # leave vessels empty for now
                                      "capacity_L":self.lock_complex.lock_length - vessel.L,
                                      "capacity_B":self.lock_complex.lock_width - vessel.B,}
        _update_lock_operation_planning(self, operation_index, lock_operation_information)

        (time_lock_operation_start,
         time_lock_entry_start,
         minimum_advance_to_open_doors,
         time_potential_lock_door_opening_stop,
         time_lock_entry_stop,
         time_lock_door_opening_stop,
         vessel_entry_delay) = self._determine_lock_operation_start_information(vessel, operation_index, direction)

        # determine the moments in time of the lock operation process steps starts and stops
        levelling_information = self.calculate_lock_operation_times(operation_index=operation_index,
                                                                    last_entering_time=time_lock_entry_start,
                                                                    start_time=time_lock_entry_stop,
                                                                    vessel=vessel,
                                                                    direction=direction)

        # determine the water levels and set the list of vessels
        wlev_A, wlev_B = levelling_information["wlev_A"], levelling_information["wlev_B"]

        # determine the moments in time of the vessel's departure from the lock (steps starts and stops) and the time the operation has stopped and the doors can close again
        time_lock_departure_start = self.calculate_lock_departure_start_time(vessel, operation_index, direction, levelling_information["time_door_opening_stop"], prognosis=True)
        time_lock_departure_stop = self.calculate_lock_departure_stop_time(vessel, operation_index, direction, levelling_information["time_door_opening_stop"], prognosis=True)
        time_lock_operation_stop = self.calculate_lock_operation_stop_time(vessel, operation_index, direction, levelling_information["time_door_opening_stop"], prognosis=True)
        time_lock_door_closing_start = self.calculate_lock_door_closing_time(vessel, operation_index, direction, levelling_information["time_door_opening_stop"], prognosis=True)

        vessels = [vessel]

        # include the update of the lock operation, if there is a rule of a required minumum number of vessels, then wait, otherwise the lock operation is ready
        status = "available"
        if len(vessels) == self.max_vessels_in_operation:
            status = "unavailable"

        # determine the time that the doors can start closing after the vessel has entered the lock (depending on whether the doors can close before the vessel has berthed), and add this to vessel planning
        if self.close_doors_before_vessel_is_laying_still:
            time_potential_lock_door_closure_start = time_lock_entry_start + self.lock_chamber.minimum_delay_to_close_doors()
        else:
            time_potential_lock_door_closure_start = time_lock_entry_stop

        # determine the new vessel delay
        delay = vessel_planning.loc[vessel_planning_index, "delay"]
        delay += vessel_entry_delay

        # store above information in dictionaries
        vessel_passage_information = {"time_potential_lock_door_opening_stop":time_potential_lock_door_opening_stop,
                                      "time_lock_passing_start":time_lock_operation_start,
                                      "time_lock_entry_start":time_lock_entry_start,
                                      "time_lock_entry_stop":time_lock_entry_stop,
                                      "time_potential_lock_door_closure_start":time_potential_lock_door_closure_start,
                                      "time_lock_departure_start":time_lock_departure_start,
                                      "time_lock_departure_stop":time_lock_departure_stop,
                                      "time_lock_passing_stop":time_lock_operation_stop,
                                      "delay":delay}
        _update_lock_vessel_planning(self, vessel_planning_index, vessel_passage_information)

        lock_operation_information = {"time_operation_start":time_lock_operation_start,
                                      "time_potential_lock_door_opening_stop":time_lock_door_opening_stop,
                                      "time_entry_start":time_lock_entry_start,
                                      "time_entry_stop":time_lock_entry_stop,
                                      "vessels":vessels,
                                      "time_door_closing_start":levelling_information["time_door_closing_start"],
                                      "time_door_closing_stop":levelling_information["time_door_closing_stop"],
                                      "time_levelling_start": levelling_information["time_levelling_start"],
                                      "time_levelling_stop": levelling_information["time_levelling_stop"],
                                      "time_door_opening_start": levelling_information["time_door_opening_start"],
                                      "time_door_opening_stop": levelling_information["time_door_opening_stop"],
                                      "time_departure_start": time_lock_departure_start,
                                      "time_departure_stop": time_lock_departure_stop,
                                      "time_operation_stop": time_lock_operation_stop,
                                      "time_potential_lock_door_closure_start": time_lock_door_closing_start,
                                      "wlev_A": wlev_A,
                                      "wlev_B": wlev_B,
                                      "status": status}
        _update_lock_operation_planning(self, operation_index, lock_operation_information)

        # if there is another lock operation is planned after this newly planned operation, check if an additional empty lock operation is required (not if there is a policy that both lock doors are closed in between operations)
        later_planned_operations = operation_planning[operation_planning.index > operation_index]
        if not later_planned_operations.empty and not self.closing_doors_in_between_operations:
            next_operation = later_planned_operations.iloc[0]
            if node_from == next_operation["node_from"]:
                self.add_empty_lock_operation_to_planning(operation_index, 1 - direction)

        yield from []

    def _process_delay_in_vessel_planning(self, operation_start_delay, other_vessels_in_operation):
        vessel_planning = self.vessel_planning
        if operation_start_delay == pd.Timedelta(seconds = 0):
            return

        for vessel_index, other_vessel in enumerate(other_vessels_in_operation):
            other_vessel_planning_index = vessel_planning[vessel_planning.id == other_vessel.id].iloc[-1].name
            vessel_planning.loc[other_vessel_planning_index, "time_potential_lock_door_opening_stop"] += operation_start_delay
            vessel_planning.loc[other_vessel_planning_index, "time_potential_lock_door_closure_start"] += operation_start_delay
            vessel_planning.loc[other_vessel_planning_index, "time_arrival_at_waiting_area"] += operation_start_delay
            vessel_planning.loc[other_vessel_planning_index, "time_arrival_at_lineup_area"] += operation_start_delay
            vessel_planning.loc[other_vessel_planning_index, "time_lock_passing_start"] += operation_start_delay
            vessel_planning.loc[other_vessel_planning_index, "time_lock_entry_start"] += operation_start_delay
            vessel_planning.loc[other_vessel_planning_index, "time_lock_entry_stop"] += operation_start_delay
            vessel_planning.loc[other_vessel_planning_index, "delay"] += operation_start_delay
            if vessel_index < len(other_vessels_in_operation) - 1:
                next_vessel = other_vessels_in_operation[vessel_index + 1]
                next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name

                # if there is slack in planning, plan two subsequent entering vessels closer to each other by adjusting the 'operation start' delay
                operation_start_delay = (vessel_planning.loc[other_vessel_planning_index, "time_lock_entry_start"] -
                                         vessel_planning.loc[next_vessel_planning_index, "time_lock_entry_start"])

    def _determine_lock_departure_information(self, vessel, operation_index, direction, levelling_information):
        time_lock_departure_start = self.calculate_lock_departure_start_time(vessel, operation_index, direction, levelling_information["time_door_opening_stop"])
        time_vessel_departure_start = self.calculate_vessel_departure_start_time(vessel, operation_index, direction, levelling_information["time_door_opening_stop"])
        time_lock_departure_stop = self.calculate_lock_departure_stop_time(vessel, operation_index, direction, levelling_information["time_door_opening_stop"])
        time_vessel_departure_stop = self.calculate_vessel_departure_stop_time(vessel, operation_index, direction, levelling_information["time_door_opening_stop"])
        time_lock_operation_stop = self.calculate_lock_operation_stop_time(vessel, operation_index, direction, levelling_information["time_door_opening_stop"])
        time_vessel_passing_stop = self.calculate_vessel_passing_stop_time(vessel, operation_index, direction, levelling_information["time_door_opening_stop"])
        time_lock_door_closing_start = self.calculate_lock_door_closing_time(vessel, operation_index, direction, levelling_information["time_door_opening_stop"])

        departure_information = {"time_lock_departure_start":time_lock_departure_start,
                                 "time_vessel_departure_start":time_vessel_departure_start,
                                 "time_lock_departure_stop":time_lock_departure_stop,
                                 "time_vessel_departure_stop":time_vessel_departure_stop,
                                 "time_lock_operation_stop":time_lock_operation_stop,
                                 "time_vessel_passing_stop":time_vessel_passing_stop,
                                 "time_lock_door_closing_start":time_lock_door_closing_start}

        return departure_information

    def _update_future_lock_operations_by_lock_delay_previous_operation(self, operation_index, lock_departure_information):
        """Updates the lock operation and vessel plannings based on a delay in a previous planned operation

        Parameters
        ----------
        operation_index : int
            index of the lock operation
        lock_departure_information : dict
            information with start and stop times of events that make up the departure of vessels from the lock operation
            required keys: "time_lock_door_closing_start", "time_lock_operation_stop"
        """
        operation_planning = self.operation_planning
        vessel_planning = self.vessel_planning

        # update the next lock operations if the previous lock operation caused a delay
        next_planned_operations = operation_planning[operation_planning.index > operation_index]
        for next_operation_index, next_operation_info in next_planned_operations.iterrows():

            # determine time delay of the process of sailing into the lock if the next operation in the planning confict with the delayed operation
            sailing_in_delay = pd.Timedelta(seconds=0)
            if not len(next_operation_info) and lock_departure_information["time_lock_door_closing_start"] > next_operation_info.time_potential_lock_door_opening_stop:
                sailing_in_delay = lock_departure_information["time_lock_door_closing_start"] - next_operation_info.time_potential_lock_door_opening_stop
            elif len(next_operation_info) and lock_departure_information["time_lock_operation_stop"] > next_operation_info.time_operation_start:
                sailing_in_delay = lock_departure_information["time_lock_operation_stop"] - next_operation_info.time_operation_start

            # determine the new start time of the next operation (dependening on whether it will fall withing the operation hours)
            new_operation_start = operation_planning.loc[next_operation_index, "time_operation_start"] + sailing_in_delay
            # within_operation_hours = operational_hours[(new_operation_start >= operational_hours.start_time) & (new_operation_start <= operational_hours.stop_time)]
            # if within_operation_hours.empty:
            #     first_available_hour = operational_hours[operational_hours.start_time >= new_operation_start].iloc[0]
            #     sailing_in_delay += first_available_hour.start_time - new_operation_start

            # break loop if there is no delay (next operations will then also not experience a delay)
            if not sailing_in_delay.total_seconds() > 0:
                break

            # update the operation planning if there is a delay
            operation_planning.loc[next_operation_index, "time_potential_lock_door_opening_stop"] += sailing_in_delay
            operation_planning.loc[next_operation_index, "time_operation_start"] += sailing_in_delay
            operation_planning.loc[next_operation_index, "time_entry_start"] += sailing_in_delay
            operation_planning.loc[next_operation_index, "time_entry_stop"] += sailing_in_delay

            # update the vessel planning
            next_vessel = None
            next_vessels = next_operation_info.vessels
            next_direction = next_operation_info.direction
            last_vessel_entering_time = operation_planning.loc[next_operation_index, "time_entry_start"]
            for next_vessel_index, next_vessel in enumerate(next_vessels):
                next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name
                vessel_planning.loc[next_vessel_planning_index, "time_potential_lock_door_opening_stop"] += sailing_in_delay
                vessel_planning.loc[next_vessel_planning_index, "time_potential_lock_door_closure_start"] += sailing_in_delay
                vessel_planning.loc[next_vessel_planning_index, "time_arrival_at_lineup_area"] += sailing_in_delay
                vessel_planning.loc[next_vessel_planning_index, "time_lock_passing_start"] += sailing_in_delay
                vessel_planning.loc[next_vessel_planning_index, "time_lock_entry_start"] += sailing_in_delay
                vessel_planning.loc[next_vessel_planning_index, "time_lock_entry_stop"] += sailing_in_delay
                last_vessel_entering_time = vessel_planning.loc[next_vessel_planning_index, "time_lock_entry_start"]
                if next_vessel_index != len(next_vessels) - 1:
                    next_next_vessel = next_vessels[next_vessel_index + 1]
                    next_next_vessel_planning_index = vessel_planning[vessel_planning.id == next_next_vessel.id].iloc[-1].name

                    # determine sailing in delay for next vessel (it can be that there is some slack time between two vessel arrivals)
                    sailing_in_delay = pd.Timedelta(seconds=0)
                    entry_start_previous_vessel = vessel_planning.loc[next_vessel_planning_index, "time_lock_entry_start"]
                    entry_start_next_vessel = vessel_planning.loc[next_next_vessel_planning_index, "time_lock_entry_start"]
                    if entry_start_next_vessel < entry_start_previous_vessel:
                        sailing_in_delay = entry_start_previous_vessel - entry_start_next_vessel
                        extra_delay = self.calculate_sailing_in_time_delay(next_next_vessel, next_operation_index,
                                                                           next_direction,
                                                                           minimum_difference_with_previous_vessel=True,
                                                                           overwrite=False)
                        sailing_in_delay += extra_delay

            # determine the new start and stop times of the lock operation (i.e., door-closing, levelling, door-opening) as it can be that the levelling time is now changed due to the shift of this operation in time (i.e., due to tides)
            time_doors_closing = operation_planning.loc[next_operation_index, "time_entry_stop"]
            levelling_information = self.calculate_lock_operation_times(operation_index=next_operation_index,
                                                                        last_entering_time=last_vessel_entering_time,
                                                                        start_time=time_doors_closing,
                                                                        vessel=next_vessel,
                                                                        direction=next_direction,)
            # update the operation planning accordingly
            operation_planning.loc[next_operation_index, "time_door_closing_start"] = levelling_information["time_door_closing_start"]
            operation_planning.loc[next_operation_index, "time_door_closing_stop"] = levelling_information["time_door_closing_stop"]
            operation_planning.loc[next_operation_index, "time_levelling_start"] = levelling_information["time_levelling_start"]
            delay_after_levelling = levelling_information["time_levelling_stop"] - operation_planning.loc[next_operation_index, "time_levelling_stop"]
            operation_planning.loc[next_operation_index, "time_levelling_stop"] = levelling_information["time_levelling_stop"]
            operation_planning.loc[next_operation_index, "time_door_opening_start"] = levelling_information["time_door_opening_start"]
            operation_planning.loc[next_operation_index, "time_door_opening_stop"] = levelling_information["time_door_opening_stop"]
            if delay_after_levelling > pd.Timedelta(seconds=0):
                operation_planning.loc[next_operation_index, "time_departure_start"] += delay_after_levelling
                operation_planning.loc[next_operation_index, "time_departure_stop"] += delay_after_levelling
                operation_planning.loc[next_operation_index, "time_operation_stop"] += delay_after_levelling
                operation_planning.loc[next_operation_index, "time_potential_lock_door_closure_start"] += delay_after_levelling
                operation_planning.loc[next_operation_index, "total_delay"] += delay_after_levelling * len(next_vessels)
                operation_planning.loc[next_operation_index, "maximum_individual_delay"] += delay_after_levelling

            # update also the departure information of the affected vessels
            for vessel_index, next_vessel in enumerate(next_vessels):
                next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name
                vessel_planning.loc[next_vessel_planning_index, "time_lock_departure_start"] += delay_after_levelling
                vessel_planning.loc[next_vessel_planning_index, "time_lock_departure_stop"] += delay_after_levelling
                vessel_planning.loc[next_vessel_planning_index, "time_lock_passing_stop"] += delay_after_levelling
                vessel_planning.loc[next_vessel_planning_index, "delay"] += delay_after_levelling

    def add_vessel_to_planned_lock_operation(self, vessel, operation_index, direction):
        """
        Add vessel to a planned lock operation

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            the index of the already planned lock operation to which the vessel is added to
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Returns
        -------
        operation_planning : pd.DataFrame
            the lock complex master's new planning of lock operations

        """
        # unpack the lock master's vessel and lock operation plannings
        vessel_planning = self.lock_complex.vessel_planning
        operation_planning = self.lock_complex.operation_planning

        # determine the vessel index in the lock complex master's vessel planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

        # set sailing in time gap and vessel entry delay time
        sailing_in_gap = pd.Timedelta(seconds=0)
        vessel_entry_delay = pd.Timedelta(seconds=0)

        # determine the number of vessels that are already assigned to the lock operation to which the vessels is/will be added
        vessels_in_operation = operation_planning.loc[operation_index, "vessels"]

        # add vessel to the operation if it is not yet part of it
        if vessel not in vessels_in_operation:
            vessels_in_operation.append(vessel)
            operation_planning.loc[operation_index, "vessels"] = (vessels_in_operation)  # TODO: is they redundant? or do we need to overwrite the information in the operation planning dataframe again
            self.calculate_sailing_time_to_approach_point(vessel, direction, operation_index=operation_index)  # TODO: can this be removed?

            # if there is a rule that prescribes a minimum amount of vessels in the lock operation and this condition is satisfied, put an operation-object in the FilterStore to communicate that the earlier waiting vessels do not have to wait any longer
            if self.min_vessels_in_operation and len(vessels_in_operation) == self.min_vessels_in_operation:
                Operation = namedtuple("Operation", "operation_index")
                operation = Operation(operation_index)
                yield self.wait_for_other_vessel_to_arrive.put(operation)

                # calculate the required sailing in time delay
                sailing_in_gap = self.calculate_sailing_in_time_delay(vessel, operation_index, direction, prognosis=False, overwrite=False)

        # calculate the new arrival time at the lock entry
        time_arrival_time_at_lock_entry = vessel_planning.loc[vessel_planning_index, "time_lock_passing_start"] + sailing_in_gap

        # if the condition of minimum amount of vessels in the lock operation is satisfied, change status of lock operation to ready
        if len(vessels_in_operation) == self.max_vessels_in_operation:
            operation_planning.loc[operation_index, "status"] = "unavailable"

        # update capacity parameters
        operation_planning.loc[operation_index, "capacity_L"] -= vessel.L
        operation_planning.loc[operation_index, "capacity_B"] -= vessel.B

        # determine the other vessels in the lock and the planned times to start the operation and the time that the lock door has been opened
        other_vessels_in_operation = operation_planning.loc[operation_index, "vessels"][:-1]
        time_lock_operation_start = operation_planning.loc[operation_index, "time_operation_start"]
        potential_lock_door_opening_stop = operation_planning.loc[operation_index, "time_potential_lock_door_opening_stop"]

        # determine the time that the vessel can be as fast as at the approach point
        time_first_vessel_required_to_be_at_lock_approach = time_arrival_time_at_lock_entry + vessel_entry_delay

        # correct start time of lock operation if there are no other vessels scheduled in the lock and the approach start time lies beyond the earlier estimated operation start time
        if time_first_vessel_required_to_be_at_lock_approach > operation_planning.loc[operation_index, "time_operation_start"] and not len(other_vessels_in_operation):
            time_lock_operation_start = time_first_vessel_required_to_be_at_lock_approach

        # add to vessel entry delay if the time of starting the approach lies ahead of the operation start time
        elif time_first_vessel_required_to_be_at_lock_approach < operation_planning.loc[operation_index, "time_operation_start"]:
            vessel_entry_delay += (operation_planning.loc[operation_index, "time_operation_start"] - time_first_vessel_required_to_be_at_lock_approach)

        if not len(other_vessels_in_operation) and time_lock_operation_start < operation_planning.loc[operation_index-1, "time_operation_stop"]:
            vessel_entry_delay += operation_planning.loc[operation_index-1, "time_operation_stop"] - time_lock_operation_start

        # add the delay to the expected time of lock entry to the vessel
        if vessel_entry_delay > pd.Timedelta(seconds=0):
            time_arrival_time_at_lock_entry += vessel_entry_delay

        # update the vessel planning based on the above delays
        time_vessel_entry_start = self.calculate_vessel_entry_duration(vessel, direction) + time_arrival_time_at_lock_entry
        time_lock_entry_stop = self.calculate_lock_entry_stop_time(vessel, operation_index, direction, time_arrival_time_at_lock_entry)
        vessel_planning.loc[vessel_planning_index, "operation_index"] = operation_index
        vessel_planning.loc[vessel_planning_index, "lock_chamber"] = self.lock_complex.name
        vessel_planning.loc[vessel_planning_index, "time_lock_passing_start"] = time_arrival_time_at_lock_entry
        vessel_planning.loc[vessel_planning_index, "time_lock_entry_start"] = time_vessel_entry_start
        vessel_planning.loc[vessel_planning_index, "time_lock_entry_stop"] = time_lock_entry_stop

        # determine the operation start delay
        operation_start_delay = time_lock_operation_start - operation_planning.loc[operation_index, "time_operation_start"]

        # update the lock master's vessel and lock operation planning by adding the operation start and vessel entry delay
        operation_planning.loc[operation_index, "time_operation_start"] += operation_start_delay
        if vessel_entry_delay > pd.Timedelta(seconds=0):
            vessel_planning.loc[vessel_planning_index, "delay"] += vessel_entry_delay
        operation_planning.loc[operation_index, "time_potential_lock_door_opening_stop"] += operation_start_delay

        # update the values of the entry start, and (if there are no other vessels) overwrite the operation start
        if not len(other_vessels_in_operation):
            time_entry_start = time_vessel_entry_start
        else:
            time_entry_start = operation_planning.loc[operation_index, "time_entry_start"]
            potential_lock_door_opening_stop = operation_planning.loc[operation_index, "time_potential_lock_door_opening_stop"]
            time_lock_operation_start = operation_planning.loc[operation_index, "time_operation_start"]
            time_entry_start += operation_start_delay

        # if there is a delay in the start op the operation: update the vessel planning of the previous arriving vessels of this operation
        self._process_delay_in_vessel_planning(operation_start_delay, other_vessels_in_operation)

        # determine the times of door closing, levelling and door opening: if lock entry stop time or extract them when the new lock entry stop time is ahead of the door closing start time TODO: check if this is correct
        levelling_information = self.calculate_lock_operation_times(operation_index=operation_index,
                                                                    last_entering_time=time_vessel_entry_start,
                                                                    start_time=time_lock_entry_stop,
                                                                    vessel=vessel,
                                                                    direction=direction,)

        # determine water levels to be included in the planning
        wlev_A, wlev_B = levelling_information["wlev_A"], levelling_information["wlev_B"]

        # if there is a delay in the departure of the vessels, also include that in the planning
        additional_sailing_out_delay = levelling_information["time_door_opening_stop"] - operation_planning.loc[operation_index, "time_door_opening_stop"]
        if additional_sailing_out_delay > pd.Timedelta(seconds=0):
            for other_vessel in other_vessels_in_operation:
                other_vessel_planning_index = vessel_planning[vessel_planning.id == other_vessel.id].iloc[-1].name
                vessel_planning.loc[other_vessel_planning_index, "time_lock_departure_start"] += additional_sailing_out_delay
                vessel_planning.loc[other_vessel_planning_index, "time_lock_departure_stop"] += additional_sailing_out_delay
                vessel_planning.loc[other_vessel_planning_index, "time_lock_passing_stop"] += additional_sailing_out_delay
                vessel_planning.loc[other_vessel_planning_index, "delay"] += additional_sailing_out_delay

        # update the operation planning with the above information
        operation_planning.loc[operation_index, "time_potential_lock_door_opening_stop"] = potential_lock_door_opening_stop
        operation_planning.loc[operation_index, "time_operation_start"] = time_lock_operation_start
        operation_planning.loc[operation_index, "time_entry_start"] = time_entry_start
        operation_planning.loc[operation_index, "time_entry_stop"] = time_lock_entry_stop
        operation_planning.loc[operation_index, "time_door_closing_start"] = levelling_information["time_door_closing_start"]
        operation_planning.loc[operation_index, "time_door_closing_stop"] = levelling_information["time_door_closing_stop"]
        operation_planning.loc[operation_index, "time_levelling_start"] = levelling_information["time_levelling_start"]
        operation_planning.loc[operation_index, "time_levelling_stop"] = levelling_information["time_levelling_stop"]
        operation_planning.loc[operation_index, "time_door_opening_start"] = levelling_information["time_door_opening_start"]
        operation_planning.loc[operation_index, "time_door_opening_stop"] = levelling_information["time_door_opening_stop"]
        operation_planning.loc[operation_index, "maximum_individual_delay"] = np.max(vessel_planning[vessel_planning.operation_index == operation_index].delay)
        operation_planning.loc[operation_index, "total_delay"] = np.sum(vessel_planning[vessel_planning.operation_index == operation_index].delay)

        # determine the new departure and operation start and stop times
        lock_departure_information = self._determine_lock_departure_information(vessel, operation_index, direction, levelling_information)
        if self.close_doors_before_vessel_is_laying_still:
            time_potential_lock_door_closure_start = time_vessel_entry_start + self.lock_chamber.minimum_delay_to_close_doors()
        else:
            time_potential_lock_door_closure_start = levelling_information["time_door_closing_start"]

        # update vessel and operation plannings accordingly
        lock_operation_information = {"time_departure_start":lock_departure_information["time_lock_departure_start"],
                                      "time_departure_stop":lock_departure_information["time_lock_departure_stop"],
                                      "time_operation_stop":lock_departure_information["time_lock_operation_stop"],
                                      "time_potential_lock_door_closure_start":lock_departure_information["time_lock_door_closing_start"],
                                      "wlev_A":wlev_A,
                                      "wlev_B":wlev_B}
        _update_lock_operation_planning(self,operation_index,lock_operation_information)

        vessel_passage_information = {
            "time_potential_lock_door_closure_start": time_potential_lock_door_closure_start,
            "time_potential_lock_door_opening_stop": (time_vessel_entry_start - self.lock_chamber.minimum_advance_to_open_doors()),
            "time_lock_departure_start": lock_departure_information["time_vessel_departure_start"],
            "time_lock_departure_stop": lock_departure_information["time_vessel_departure_stop"],
            "time_lock_passing_stop": lock_departure_information["time_vessel_passing_stop"],
        }
        _update_lock_vessel_planning(self,vessel_planning_index,vessel_passage_information)

        # update the next lock operations if the previous lock operation caused a delay
        self._update_future_lock_operations_by_lock_delay_previous_operation(operation_index, lock_departure_information)
        return operation_planning

    def update_operation_planning(self, vessel, direction, operation_index, add_operation):
        """
        Updates the lock master's lock operation planning

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        operation_index : int
            index of the lock operation
        add_operation : bool
            expresses whether the vessel should be added to a new lock operation planning: yes [True] or no [False]

        Yields
        -------
        Adds vessel to new or planned lock operation

        """
        # unpack the lock master's vessel and lock operation plannings
        operation_planning = self.lock_complex.operation_planning
        vessel_planning = self.lock_complex.vessel_planning

        # add vessel to a new lock operation or to a planned one
        if operation_planning.empty or add_operation:
            yield from self.add_vessel_to_new_lock_operation(vessel, operation_index, direction)
        else:
            yield from self.add_vessel_to_planned_lock_operation(vessel, operation_index, direction)

        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index, "operation_index"]
        operation_planning.loc[operation_index, "maximum_individual_delay"] = np.max(vessel_planning[vessel_planning.operation_index == operation_index].delay)

        operation_planning.loc[operation_index, "total_delay"] = np.sum(vessel_planning[vessel_planning.operation_index == operation_index].delay)

    def assign_vessel_to_lock_operation(self, vessel, direction):
        """
        Function that adds a vessel to the lock operation planning

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Returns
        -------
        operation_index : int
            index of the lock operation to which the vessel can be added (can either be an existing or a new lock operation)
        add_operation : bool
            determines if a new lock operation should be added (True) or not (False)
        available_operations : pd.DataFrame
            the available lock operations to which the vessel can be assigned including their information

        """
        # unpack the lock complex' vessel and operations planning
        operation_planning = self.lock_complex.operation_planning
        vessel_planning = self.lock_complex.vessel_planning

        # determine the index of the vessel in the vessel planning to determine when the vessel is estimated to pass the approach point and enters the lock#TODO: write a test that the vessel has indeed earlier be included in the vessel planning (the 'add_vessel_to_vessel_planning'-function should always be ran before this function)
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        time_lock_passing_start = vessel_planning.loc[vessel_planning_index, "time_lock_passing_start"]
        time_lock_entry_start = vessel_planning.loc[vessel_planning_index, "time_lock_entry_start"]

        # add to the vessel planning that the vessel has a delay (which is still 0 [s])
        vessel_planning.loc[vessel_planning_index, "delay"] = pd.Timedelta(seconds=0)

        # determine whether the planned approach fits within the operation hours of the lock and add a delay to the planned approach of the vessel when it is outside of the operational hours
        operational_hours = self.operational_hours
        within_operation_hours = operational_hours[(time_lock_passing_start >= operational_hours.start_time) & (time_lock_passing_start <= operational_hours.stop_time)]
        if within_operation_hours.empty:
            first_available_hour = operational_hours[operational_hours.start_time >= time_lock_passing_start].iloc[0]
            delay = first_available_hour.start_time - time_lock_passing_start
            time_lock_entry_start += delay
            vessel_planning.loc[vessel_planning_index, "time_arrival_at_waiting_area"] += delay
            vessel_planning.loc[vessel_planning_index, "time_arrival_at_lineup_area"] += delay
            vessel_planning.loc[vessel_planning_index, "time_lock_passing_start"] += delay
            vessel_planning.loc[vessel_planning_index, "time_lock_entry_start"] += delay
            vessel_planning.loc[vessel_planning_index, "time_of_acceptance"] += delay
            vessel_planning.loc[vessel_planning_index, "delay"] += delay

        # determine the maximum delay of an individual vessel in all the planned lock operation if the vessel is assigned to that operation
        maximum_individual_delay = operation_planning.maximum_individual_delay + (time_lock_entry_start - operation_planning.time_entry_stop)

        # filter the planned lock operations based on the following criteria to select available operations to which the vessel can be assigned
        mask_direction = operation_planning.direction == direction  # lock operations in the same direction as the vessel
        mask_available = operation_planning.status == "available"  # lock operations that are not unavailable
        mask_capacity_L = (operation_planning.capacity_L >= vessel.L)  # lock operations that have a capacity in which the vessel fits longitudinally (based on the vessel's length)
        mask_capacity_B = (operation_planning.capacity_B >= vessel.B)  # lock operations that have a capacity in which the vessel fits laterally (based on the vessel's beam) TODO: implement this later
        mask_max_waiting_time = maximum_individual_delay < pd.Timedelta(seconds=self.lock_complex.clustering_time)  # lock operations that will not exceed the maximum set waiting time for individual vessels
        mask_empty_lock = operation_planning.vessels.apply(len) == 0  # lock operations that are still empty

        # max vessels mask: lock operations that do not exceed a maximum number of vessels
        mask_max_vessels = mask_available
        if self.max_vessels_in_operation:
            mask_max_vessels = operation_planning.vessels.apply(len) < self.max_vessels_in_operation

        # future operations mask: lock operations that still have to take place
        mask_future_operations = operation_planning.time_levelling_start >= time_lock_entry_start

        # combinations of the masks TODO: this part of the code should be improved in clarity
        mask_max_waiting_time = (mask_max_waiting_time & ~mask_empty_lock)  # non-empty lock operations with non-exceedance of the maximum waiting time
        if self.min_vessels_in_operation:
            mask_min_vessels = operation_planning.vessels.apply(len) < self.min_vessels_in_operation
        else:
            mask_min_vessels = operation_planning.vessels.apply(len) >= self.min_vessels_in_operation

        mask_empty_available_lock = mask_empty_lock & mask_future_operations

        # select available operations TODO: this part of the code should be improved in clarity and readability
        available_operations = operation_planning[
            mask_available
            & mask_direction
            & mask_min_vessels
            & mask_max_vessels
            & mask_capacity_L
            & (mask_future_operations | mask_max_waiting_time | mask_empty_available_lock)
        ].copy()
        # TODO: include mask_capacity_B for 2D implementation
        # TODO: create a selection method that can pick the lock operation based on minimizing expected delay or freshwater loss/saltwater intrusion

        # determine if vessel can be added to an existing lock operation planning and (if yes) to which one, or should be added to a new lock operation
        add_operation = False
        if not available_operations.empty:
            operation_index = available_operations.iloc[0].name
        else:
            operation_index = len(operation_planning)
            add_operation = True

        return operation_index, add_operation, available_operations


@inherit_docstring
class PassesLockComplex(Movable, HasMultiDiGraph):
    """Mixin class: Something that passes a lock complex (i.e., can be added to a vessel-object)

    Pre-requisites
    --------------
    arrival_time:
        the vessel should have an arrival_time in its metadata


    Attributes
    -----------
    register_to_lock_master: generator
        vessel requests registration of itself to the lock master of the lock complex (for short-term planning)
    sail_to_waiting_area: generator
        the event of sailing towards the vessel's first to be encountered waiting area of the lock complex
    """

    def __init__(self, *args, **kwargs):
        """
        Initialization
        """
        super().__init__(*args, **kwargs)

        # Add attributes to the vessels movable functions
        self.on_pass_node_functions.append(self.register_to_lock_master)
        self.on_pass_edge_functions.append(self.sail_to_waiting_area)

        # Save speeds that are calculated by vessel_traffic_service
        self.overruled_speed = pd.DataFrame(
            data=[], columns=["Speed"], index=pd.MultiIndex.from_arrays([[], []], names=("node_start", "node_stop"))
        )

    def _find_route_to_lock(self, lock):
        """Determines the route of a vessel to the lock

        Parameters
        ----------
        lock : object
            the lock chamber object generated with IsLockChamber

        Returns
        -------
        route_to_lock : list or str
            list of the node names that make up the route to the lock
        """
        route_to_come = self.route_ahead
        index = 0
        for index, edge in enumerate(zip(route_to_come[:-1], route_to_come[1:])):
            if edge == lock.edge or edge == lock.edge[::-1]:
                index += 1
                break
        route_to_lock = route_to_come[:(index+1)]
        return route_to_lock

    def _find_upcoming_lock_registration_nodes(self):
        """
        Find the upcoming locks that use long-term planning by looping over the vessel's route

        Returns
        -------
        upcoming_locks : dict
            dictionary of lock objects that are to be encountered on the vessel's route
            mapping from node (key) to lock object (value)
        """
        # initiate empty lists
        upcoming_locks = {}

        # loop over all nodes on the route ahead.
        route_to_come = self.route_ahead
        for node in route_to_come:
            node_info = self.multidigraph.nodes[node]

            # check if the node has a registration node
            if ("Lock_registration_node" not in node_info.keys()):
                continue

            # unpack the lock complex information using the lock_edge stored in the registration node
            lock_edge = node_info["Lock_registration_node"]
            lock = self.multidigraph.edges[lock_edge]["Lock"][0]  # TODO: write test to prevent that multiple lock complexes are located at the same registration node, also: maybe we need to change "Lock" to "Lock complex"

            # check if lock is already stored
            if lock in upcoming_locks.values():
                continue
            # store the lock object in the list of locks with long_term_planning enabled
            upcoming_locks[node] = lock
        return upcoming_locks

    def _find_upcoming_locks(self):
        """
        Find the upcoming locks that use long-term planning by looping over the vessel's route

        Parameters
        ----------

        Returns
        -------
        upcoming_locks : dict
            dictionary of lock objects that are to be encountered on the vessel's route
            mapping from node (key) to lock object (value)
        """
        # initiate empty lists
        upcoming_locks = {}

        # loop over all edges on the route ahead.
        route_to_come = self.route_ahead
        for node_start, node_stop in zip(route_to_come[:-1], route_to_come[1:]):
            k = sorted(self.multidigraph[node_start][node_stop],key=lambda x: get_length_of_edge(self.multidigraph,(node_start, node_stop, x)))[0] #TODO: k-berekening in een functie zetten (nu bepaald op minste lengte, maar sluismeester moet/kan dit bepalen).
            lock_edge = (node_start,node_stop,k)
            if "Lock" not in self.multidigraph.edges[lock_edge].keys():
                continue
            lock = self.multidigraph.edges[lock_edge]["Lock"][0]

            # check if lock is already stored
            if lock in upcoming_locks.values():
                continue

            # store the lock object in the list of locks with long_term_planning enabled
            upcoming_locks[node_start] = lock

        return upcoming_locks

    def register_to_lock_master(self, origin):
        """
        Request lock master to register when vessel reaches a registration node of a lock complex object

        Parameters
        ----------
        origin : str
            node name (that has to be in the graph) on which the vessel is currently starting to navigate an edge

        Yields
        ------
        Request to the lock complex master to register the vessel
        """

        # find the lock complex object that is associated with the registration node
        lock = _get_lock_object_on_registration_node(self.multidigraph, origin)
        if not lock:
            return

        upcoming_locks = self._find_upcoming_locks()
        for _,upcoming_lock in upcoming_locks.items():
            if lock == upcoming_lock:
                # if a lock complex object is found, request registration to the lock master of the lock complex
                yield from lock.register_vessel(self)
                break

    def sail_to_waiting_area(self, origin, destination):
        """
        Vessel sails to the waiting area

        Parameters
        ----------
        origin : str
            node name (that has to be in the graph) on which the vessel is currently sailing, to navigate an edge
        destination : str
            node name (that has to be in the graph) on which the vessel is currently sailing to, to navigate an edge (should form an edge with the origin)

        Yields
        ------

        """

        # determine which part of the route we still need to consider: if the route does not pass the lock complex, then skip function (vessel should not interact with the lock complex)
        route_to_come = self.route_ahead
        if len(route_to_come) <= 1:
            return

        # TODO: misschien losse functie maken hier
        # find the lock the vessel has been assigned to TODO: this should be faster, so that if the vessel has not been assigned to a lock, it does not check the entire route
        # TODO: @Floor. Ziet eruit alsof dit de laatste lock is die op de route ligt. Ik den kdat we juist de eerste willen hebben toch?
        # TODO: @Floor: in register_to_lock_master zoeken we gewoon de lock die aan de origin-node grenst. Kunnen we dat hier niet ook doen?
        locks = self._find_upcoming_locks()

        # if no lock is found, stop function
        if not bool(locks):
            return

        # determine the waiting area based on the direction of the vessel
        for lock_start_node,lock in locks.items():
            if lock_start_node == lock.start_node:
                direction = 0
                waiting_area = lock.waiting_area_A
            else:
                direction = 1
                waiting_area = lock.waiting_area_B

            # if the origin of the vessel has not reached the waiting area edge, then skip this function
            if origin != waiting_area.edge[0]:
                return

            # unpack the vessel and lock operation planning of the lock
            operation_planning = lock.lock_master.operation_planning
            vessel_planning = lock.lock_master.vessel_planning

            # determine the vessel index and operation index
            vessel_planning_index = vessel_planning[vessel_planning.id == self.id].iloc[-1].name
            operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']

            # calculate the sailing duration left to the waiting area
            sailing_time_to_waiting_area, sailing_distance_to_waiting_area, vessel_speed = lock.calculate_sailing_time_to_waiting_area(self, direction, overwrite=False)
            sailing_time_to_waiting_area = sailing_time_to_waiting_area.total_seconds()

            # if there is still sailing time left to the waiting area then continue sailing and log this process (here the locking module takes over the function of the movable)
            if sailing_time_to_waiting_area:
                self.log_entry_v0("Sailing to waiting area start", self.env.now, self.output.copy(),self.logbook[-1]['Geometry'],)

            # the sailing process can be interrupted, as vessel can be subject to changes in its speed, then the remaining sailing time is determined and continued with the changed speed -> when sailing to the waiting area has been completed: log the process
            while sailing_time_to_waiting_area:
                start_sailing = self.env.now
                try:
                    yield self.env.timeout(sailing_time_to_waiting_area)
                    sailing_time_to_waiting_area = 0.
                except simpy.Interrupt as e:
                    sailing_time_to_waiting_area -= self.env.now - start_sailing
                    remaining_sailing_distance = vessel_speed * sailing_time_to_waiting_area
                    sailing_time_to_waiting_area = remaining_sailing_distance / self.current_speed
                self.log_entry_v0("Sailing to waiting area stop", self.env.now, self.output.copy(),waiting_area.location,)

            # let vessel wait in the waiting area TODO: can we decouple this?
            yield from self.wait_in_waiting_area(waiting_area=waiting_area)

            # if done waiting -> release vessel from waiting area and let vessel continue
            yield waiting_area.resource.release(self.waiting_area_request)

            # vessel is now allowed to continue passing the lock -> create vessel specific functions and add those function to the functions that communicate with the move function
            allow_vessel_to_sail_into_lock = functools.partial(lock.allow_vessel_to_sail_into_lock, vessel=self)
            initiate_levelling = functools.partial(lock.initiate_levelling, vessel=self)
            allow_vessel_to_sail_out_of_lock = functools.partial(lock.allow_vessel_to_sail_out_of_lock, vessel=self)
            self.on_pass_edge_functions.append(allow_vessel_to_sail_into_lock)
            self.on_pass_edge_functions.append(initiate_levelling)
            self.on_pass_edge_functions.append(allow_vessel_to_sail_out_of_lock)

            # correct distance left on edge with the already covered distance through this function (to communicate with the move function)
            self.distance_left_on_edge -= sailing_distance_to_waiting_area

            # on continuing sailing to the lock complex, determine the current time and whether the vessel is the first vessel or will arrive after another vessel
            current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
            first_in_lock = operation_planning.loc[operation_index].vessels[0] == self
            between_arrivals = False
            if not first_in_lock:
                between_arrivals = True

            # determine if the door is closed, and when the doors are required to be open, and how long this will take (given the lock master's policy)
            door_is_closed, doors_required_to_be_open, operation_time = lock.determine_if_door_is_closed(
                self, operation_index, direction, first_in_lock=first_in_lock, between_arrivals=between_arrivals
            )
            # if door is open, then the vessel can continue normally
            if not door_is_closed:
                return

            # if not, and if the time that the doors will be open lies ahead of the current time -> create a door open request with a delay so that the doors are open at the right moment (according to the lock master's policy)
            if (doors_required_to_be_open - operation_time) > current_time:
                delay = ((doors_required_to_be_open - operation_time) - current_time).total_seconds()
                self.door_open_request = self.env.process(lock.open_door(to_level=lock_start_node, delay=delay, vessel=self))
                return

            # if it is already too late, the doors should open immediately -> determine the time that the doors are required to be opened again (this can include a new levelling process in case of tidal water levels)
            levelling_required = False
            if operation_time > pd.Timedelta(seconds=lock.doors_closing_time):
                levelling_required = True

            # log the door open process and the lock levelling process if this is required TODO: this should preferably also be requested from the lock master elsewhere (especially the levelling process)
            if levelling_required:
                lock.log_entry_v0("Lock chamber converting start", doors_required_to_be_open.round('s').to_pydatetime().timestamp() - operation_time.total_seconds(), self.output.copy(),lock_start_node, )
                lock.log_entry_v0("Lock chamber converting stop", doors_required_to_be_open.round('s').to_pydatetime().timestamp() - lock.doors_opening_time, self.output.copy(),lock_end_node, )
            lock.log_entry_v0("Lock doors opening start", doors_required_to_be_open.round('s').to_pydatetime().timestamp() - lock.doors_opening_time, self.output.copy(),lock_end_node, )
            lock.log_entry_v0("Lock doors opening stop",doors_required_to_be_open.round('s').to_pydatetime().timestamp(),self.output.copy(), lock_end_node, )

            # set the new side to which the lock has been opened
            lock.node_open = lock._directional_edge(direction)[0]

            # set the new water level for the lock if there is hydrodynamic data included in the simulation TODO: also this should preferably be included elsewhere and not here
            if self.env.vessel_traffic_service.hydrodynamic_information_path:
                hydromanager = HydrodynamicDataManager()
                time_index = np.absolute(
                    hydromanager.hydrodynamic_times
                    - np.datetime64(doors_required_to_be_open)
                    - np.timedelta64(int(lock.doors_opening_time), "s")
                ).argmin()
                station_index = np.where(np.array(list((hydromanager.hydrodynamic_data["STATION"]))) == lock.node_open)[0]
                lock.water_level[time_index:] = hydromanager.hydrodynamic_data["Water level"][station_index, time_index:]

    def wait_in_waiting_area(self, waiting_area):
        """
        Let the vessel wait in the waiting area

        Parameters
        ----------
        waiting_area : class
            the waiting area of the lock chamber (IsLockWaitingArea-class)

        Yields
        ------
            waiting time in the waiting area: (1) for another vessel and (2) for the start of the assigned lock operation
        """

        # unpack the lock complex of which the waiting area is part of
        lock = waiting_area.lock

        # determine the direction of the vessel with respect to the lock complex: coming from node A (direction = 0), or from node B (direction = 1)
        if waiting_area.name == 'waiting_area_A':
            direction = 0
            distance_left_on_edge = lock.distance_waiting_area_A_to_end_edge_waiting_area_A
        else:
            direction = 1
            distance_left_on_edge = lock.distance_waiting_area_B_to_end_edge_waiting_area_B

        # unpacks the lock complex master's vessel and lock planning
        vessel_planning = lock.lock_complex.vessel_planning
        operation_planning = lock.lock_complex.operation_planning

        # determines the vessel index and lock operation index to which the vessel is assigned -> determine how many vessels are assigned to this operation and at which time the vessel starts entering the lock
        vessel_planning_index = vessel_planning[vessel_planning.id == self.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']
        vessels_in_operation = operation_planning.loc[operation_index, 'vessels']
        start_time_entering_lock = vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start']

        # determines the sailing time to reach the approach point of the lock complex
        sailing_to_approach = lock.calculate_sailing_time_to_approach_point(self, direction, from_waiting_area=True,overwrite=False)# - lock.calculate_sailing_time_to_waiting_area(self, direction, overwrite=False)[0]

        # set the moment in time that the waiting in the waiting area has started
        waiting_start = self.env.now
        # check if vessel has to wait for other vessels (if there is a policy that a minimum number of vessels have go with each lock operation, and this criteria has yet not been matched)
        if len(vessels_in_operation) < lock.min_vessels_in_operation:
            # log the waiting event
            self.log_entry_v0("Waiting for other vessel in lock operation start", waiting_start, self.output.copy(), self.logbook[-1]['Geometry'],)

            # create a request to wait for another vessel (this is a request for a filter store: only if there are enough vessels the operation will be assigned to the store and all vessels will continue to the lock chamber)
            request = lock.wait_for_other_vessel_to_arrive.get(lambda operation: operation.operation_index == operation_index)
            # waiting in the waiting area, if request is interrupted, the vessel keeps waiting TODO: Dit stuk code hoort eigenlijk bij lockmaster.
            while len(operation_planning.loc[operation_index,'vessels']) < lock.min_vessels_in_operation:
                try:
                    yield request
                except simpy.Interrupt as e:
                    pass

            # determine the moment in time that the waiting has stopped
            waiting_stop = self.env.now

            # if the moment of the vessel starting to enter the lock has shifted, then update the vessel planning and the operation planning if it is the first assigned vessel to the lock
            if pd.Timestamp(datetime.datetime.fromtimestamp(waiting_stop)) + sailing_to_approach > start_time_entering_lock:
                # TODO functie in lock_master met input vessel.
                vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
                vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
                vessel_planning.loc[vessel_planning_index, 'time_lock_entry_stop'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
                vessel_index = vessels_in_operation.index(self)
                if vessel_index == 0:
                    operation_planning.loc[operation_index, 'time_entry_start'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
                vessel_planning.loc[vessel_planning_index, 'time_arrival_at_lineup_area'] += pd.Timedelta(seconds=waiting_stop - waiting_start)

            # log that the waiting has stopped
            self.log_entry_v0("Waiting for other vessel in lock operation stop", self.env.now, self.output.copy(),self.logbook[-1]['Geometry'],)

        # determine the current time (after waiting for another vessel, or not) and the time that the vessel will be at the approach point if it will continue and what was planned before
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(lock.env.now))
        time_at_approach = current_time + sailing_to_approach
        planned_start_time_entering_lock = vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start']

        # determine (additional) waiting time for the vessel
        waiting_time = planned_start_time_entering_lock-time_at_approach

        # determine the waiting time that a vessel can do by decreasing it sailing speed and the waiting time that the vessel has to wait stationary in the waiting area (due to a minimum required speed for safe manoeuvrability)
        # remaining_static_waiting_time, waiting_time_while_sailing = lock.determine_waiting_time_while_sailing_to_lock(self,direction,waiting_time.total_seconds()) TODO: kijken waarom deze uitgecommand is, en of we deze toch wel willen gebruiken
        remaining_static_waiting_time = waiting_time.total_seconds()
        waiting_time_while_sailing = 0.

        # if there is stationary waiting time -> let vessel wait (longer) in the waiting area
        if remaining_static_waiting_time > 0.:
            # log the start of the waiting process
            self.log_entry_v0("Waiting for lock operation start", self.env.now, self.output.copy(), self.logbook[-1]['Geometry'], )
            # waiting in the waiting area, if request is interrupted, the vessel keeps waiting but time that vessel already has waited is subtracted
            while remaining_static_waiting_time > 0.:
                try:
                    yield lock.env.timeout(remaining_static_waiting_time)
                    time_at_approach += pd.Timedelta(seconds=remaining_static_waiting_time)
                    remaining_static_waiting_time = 0.
                    time_operation_start = vessel_planning.loc[vessel_planning_index,'time_lock_passing_start']
                    remaining_static_waiting_time = (time_operation_start-time_at_approach).total_seconds()
                except simpy.Interrupt as e:
                    remaining_static_waiting_time -= lock.env.now - waiting_start

            # log the stop of the waiting process
            self.log_entry_v0("Waiting for lock operation stop", self.env.now, self.output.copy(), self.logbook[-1]['Geometry'], )

        # if there is waiting time that can be performed while sailing, adjust sailing speed
        if waiting_time_while_sailing:
            lock.overrule_vessel_speed(self,lock_end_node,waiting_time=waiting_time_while_sailing)
            self.process.interrupt()

        self.overruled_speed.loc[waiting_area.edge, 'Speed'] = lock.vessel_sailing_in_speed(self, direction)
        self.distance_left_on_edge = distance_left_on_edge


@inherit_docstring
class IsLockWaitingArea(HasResource, Identifiable, Log, HasOutput, HasMultiDiGraph):
    """Mixin class: lock complex has waiting area object:

    creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity

    Attributes
    ----------
    waiting_area : simpy.PriorityResource
        the waiting area resource with a certain capacity
    location : Location
        the location of the waiting area on the edge

    """

    def __init__(
        self, edge, lock, distance_from_edge_start, *args, **kwargs  # a string which indicates the location of the start of the waiting area
    ):
        node = edge[0]
        self.node = node
        self.edge = edge
        self.lock = lock
        self.distance_from_edge_start = distance_from_edge_start
        super().__init__(*args, **kwargs, nr_resources=1000000)
        """Initialization"""

        self.waiting_area = simpy.PriorityResource(self.env, capacity=1000000)
        self.location = self.env.vessel_traffic_service.provide_location_over_edges(edge[0],edge[1],distance_from_edge_start)
        # TODO: gebruik self.resource vanuit hasresource in plaats van self.waiting_area
        # TODO: checken of deze parents allemaal nodig zijn.
        # TODO: locatable mixin gebruiken in plaats van self.location


class IsLockMaster(SimpyObject, HasLockPlanning):
    """Mixin class: lock complex has a lock master:

    Creates a lock master that schedules the vessels into lock operations

    Attributes
    ----------
    create_operational_hours :
        creates an DataFrame with the operational hours of the lock complex
    register_vessel :
        registers a vessel to the lock operation and vessel planning
    calculate_sailing_information_on_route_to_lock_complex :
        calculates the sailing information (i.e., duration, distance, and speed) of the vessel per edge of its route between its current location and the lock doors
    overrule_vessel_speed :
        overrules the speed of an vessel based on the additional waiting time
    initiate_levelling :

    allow_vessel_to_sail_out_of_lock :

    allow_vessel_to_sail_into_lock :

    add_vessel_to_vessel_planning :
        adds vessel to the vessel planning of the lock complex upon request
    add_empty_lock_operation_to_planning :
        adds an empty lock operation to the operation planning
    determine_route_to_waiting_area_from_node :

    calculate_sailing_time_to_waiting_area :
        calculates the sailing time of a vessel from its location to the waiting area
    calculate_sailing_time_to_lineup_area :
        calculates the sailing time of a vessel from its location to the line-up area
    calculate_sailing_time_to_approach_point :
        calculates the sailing time of a vessel from its location to the approach point
    calculate_sailing_time_to_lock_door :
        calculates the sailing time of a vessel from its location to the first lock doors that it will encounter
    calculate_sailing_time_in_lock :
        calculates the time duration that a vessel needs to enter the lock until laying still
    calculate_sailing_in_time_delay :
        calculates the minimum required time gap between two entering vessels for safety, resulting in a delay
    calculate_vessel_entry_duration :
        calculates the moment in time that a vessel starts entering the lock
    calculate_vessel_passing_start_time :
        calculates the start time that a vessel can start its manoeuvre of entering the lock
    calculate_lock_operation_start_time :
        calculates the new earliest possible start time of a lock operation
    calculate_lock_door_opening_time :
        .
    calculate_lock_entry_start_time :
        .
    calculate_vessel_entry_stop_time :
        calculates the moment in time that a vessel finished its lock entry process
    calculate_lock_entry_stop_time :
        calculates the moment in time that a lock operation entry process of all the assigned vessels is finished (all vessels are in lock chamber)
    calculate_lock_operation_times :
        calculates the moments in time of the start and stop of the operation steps of the lock: (1) door closing, (2) levelling, (3) door opening
    calculate_vessel_departure_start_time :
        .
    calculate_lock_departure_start_time :
        .
    calculate_vessel_sailing_time_out_of_lock :
        .
    calculate_vessel_departure_stop_time :
        .
    calculate_lock_departure_stop_time :
        .
    calculate_vessel_passing_stop_time :
        .
    calculate_lock_operation_stop_time :
        .
    minimum_delay_to_close_doors :
        calculates the time delay between when the last vessel has entered the lock and when the lock doors can be closed
    minimum_advance_to_open_doors :
        determines the minimum time in advance that a lock door should be opened
    calculate_lock_door_closing_time :

    determine_first_vessel_of_lock_operation :
        determines the first vessel that was assigned to the lock operation
    determine_last_vessel_of_lock_operation:
        determines the last vessel that was assigned to the lock operation
    calculate_delay_to_open_doors :
        .
    determine_if_door_can_be_closed :
        .
    determine_if_door_is_closed :
        .
    determine_time_to_open_door :
        .
    determine_water_levels_before_and_after_levelling :
        determines the water level at both sides of the lock
    get_vessel_from_planned_operation :
        gets the vessels that are assigned to a certain lock operation in the operation planning of the lock master
    update_operation_planning :
        updates the lock master's lock operation planning
    add_vessel_to_new_lock_operation :
        adds a vessel to a newly to be planned lock operation
    add_vessel_to_planned_lock_operation :
        add vessel to a planned lock operation
    assign_vessel_to_lock_operation :
        adds a vessel to the lock operation planning
    convert_chamber :
        converts the lock chamber and logs this event
    close_door :
        .
    level_lock :
        .
    open_door :
        .

    """

    def __init__(
        self,
        lock_complex,
        min_vessels_in_operation=0,
        max_vessels_in_operation=100,
        clustering_time=0.5 * 60 * 60,
        water_level_difference_limit_to_open_doors=0.05,
        minimize_door_open_times=False,
        closing_doors_in_between_operations=False,
        closing_doors_in_between_arrivals=False,
        close_doors_before_vessel_is_laying_still=False,
        operational_hour_start_times=None,
        operational_hour_stop_times=None,
        *args,
        **kwargs,
    ):
        """Initialization"""
        super().__init__(*args, **kwargs)
        self.lock_complex = lock_complex
        self.min_vessels_in_operation = min_vessels_in_operation
        self.max_vessels_in_operation = max_vessels_in_operation
        self.clustering_time = clustering_time
        self.minimize_door_open_times = minimize_door_open_times
        self.closing_doors_in_between_operations = closing_doors_in_between_operations
        self.closing_doors_in_between_arrivals = closing_doors_in_between_arrivals
        self.close_doors_before_vessel_is_laying_still = close_doors_before_vessel_is_laying_still
        self.water_level_difference_limit_to_open_doors = water_level_difference_limit_to_open_doors

        if operational_hour_start_times is not None and operational_hour_stop_times is not None:
            operational_hours = self.create_operational_hours(operational_hour_start_times,operational_hour_stop_times)
        else:
            operational_hours = self.create_operational_hours([datetime.datetime.min], [datetime.datetime.max])
        self.operational_hours = operational_hours

    def create_operational_hours(self,start_times,stop_times):
        """
        Creates an DataFrame with the operational hours of the lock complex

        Parameters
        ---------
        start_times: list of pd.Timestamp
            the time at which the operation of the lock starts
        stop_times: list of pd.Timestamp
            the time at which the operation of the lock stops (after the start times)
        Returns
        -------
        operational_hours : pd.DataFrame
            a dataframe with the windows of operation for the lock complex

        """
        # TODO: this is more an utility function as it does not include the lock master (self)
        # creates default dataframe
        operational_hours = pd.DataFrame(columns=['start_time', 'stop_time'])

        # includes the start and stop times of the operation windows in the dataframe
        for start_time,stop_time in zip(start_times,stop_times):
            operational_hours.loc[len(operational_hours),:] = [start_time,stop_time]

        return operational_hours

    def register_vessel(self, vessel):
        """
        Registers a vessel to the lock operation and vessel planning

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        """

        # TODO: van vessel_planning en operation_planning properties  maken? #Antwoord Floor: ja, lijkt me een goed plan

        # unpacks the lock complex master's vessel and lock operation planning
        vessel_planning = self.lock_complex.vessel_planning
        operation_planning = self.lock_complex.operation_planning

        # determine the orientation of the vessel to unpack the lock complex infrastructure at the correct side of the lock chamber
        # TODO hier een property van maken?
        # TODO Floor: De direction wordt hier bepaald met
        #   - vessel.current node == self.lock_complex.registration_nodes[0]. #Comment Floor. Ja, dit lijkt me goed, maar we moeten hiermee oppassen. to_level, waiting_area.name en self.node_open zijn andere zaken. Lock_edge[0] en self.lock_complex.registration_nodes[0] kunnen we gladstrijken.
        # In andere formules staat
        #   - if current_node == lock.start_node:
        #   - if to_level == self.start_node:
        #   - if lock_edge[0] == lock.start_node:
        #   - if waiting_area.name == 'waiting_area_A':
        #   - if self.node_open == self.start_node:
        # komen al deze formules op hetzelfde neer? Kan er een algemene formule worden geschreven voor de direction, lock_end_node en waiting area die in alle berekeningen werkt?
        # en zijn deze attributes dan eigenschappen van de lockmaster, van de lockcomplex of van de lockchamber?
        if vessel.current_node == self.lock_complex.registration_nodes[0]:
            direction = 0
            lock_end_node = self.lock_complex.end_node
            waiting_area = self.waiting_area_A
        else:
            direction = 1
            lock_end_node = self.lock_complex.start_node
            waiting_area = self.waiting_area_B

        # add vessel to vessel planning (already done when lock master planned for the long-term) and extract the index of this vessel in this planning
        self.add_vessel_to_vessel_planning(vessel, direction)
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

        # add vessel to lock operation planning (already done when lock master planned for the long-term), else subtract operation_index from pre-assignment
        operation_index, add_operation, available_operations = self.assign_vessel_to_lock_operation(vessel, direction)
        yield from self.update_operation_planning(vessel, direction, operation_index, add_operation)

        # request access to the waiting area
        vessel.waiting_area_request = waiting_area.resource.request()
        yield vessel.waiting_area_request

        # unpack its assigned operation to determine if there are other vessels in the lock operation that need to wait for the vessel (skip rest of function if this is not the case or if there is no policy of minimizing the door open times)
        assigned_operation = operation_planning.loc[operation_index]
        if not self.minimize_door_open_times or len(assigned_operation.vessels) == 1:
            return

        # determine the extra waiting time of the previous vessel in the lock by the difference between the sailing in times of the registered vessel and its predecessor with the goal of optimizing this time: just in time ahead of the newly registered vessel with enough safety (to reduce the door open time, hence saltwater intrusion, without causing extra delay)
        other_vessel = assigned_operation.vessels[-2]
        other_vessel_planning_index = vessel_planning[vessel_planning.id == other_vessel.id].iloc[-1].name
        registered_vessel_time_lock_entry_start = vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start']
        other_vessel_time_lock_entry_start = vessel_planning.loc[other_vessel_planning_index, 'time_lock_entry_start']
        minimum_sailing_in_time_gap_through_doors = datetime.timedelta(seconds=self.sailing_in_time_gap_through_doors)
        preceding_vessel_waiting_time_to_shorten_door_open_time = registered_vessel_time_lock_entry_start - minimum_sailing_in_time_gap_through_doors - other_vessel_time_lock_entry_start
        sailing_information_other_vessel = self.calculate_sailing_information_on_route_to_lock_complex(other_vessel, lock_end_node)

        # if there is no sailing information available or when there is no extra waiting time for the previously registered vessel in the lock operation -> then skip rest of function (nothing to optimise here)
        if sailing_information_other_vessel.empty or preceding_vessel_waiting_time_to_shorten_door_open_time.total_seconds() <= 0.0:
            return

        # determine the total distance and sailing time to the lock
        total_time_to_lock_other_vessel = sailing_information_other_vessel.Time.sum()
        total_distance_to_lock_other_vessel = sailing_information_other_vessel.Distance.sum()

        # if there is no more sailing distance left to the lock doors for the previous vessel -> then skip rest of function (nothing to optimise here)
        if total_time_to_lock_other_vessel <= 0.0:
            return

        # determine the optimum speed of this preceding vessel to delay its entering time into the lock, but that its sailing at a safe speed
        average_speed = total_distance_to_lock_other_vessel / total_time_to_lock_other_vessel
        overruled_speed = np.max([self.minimum_manoeuvrability_speed, total_distance_to_lock_other_vessel / (preceding_vessel_waiting_time_to_shorten_door_open_time.total_seconds() + total_time_to_lock_other_vessel)])

        # determine whether the full amount of the optimal reduction in extra waiting time in the lock chamber for the preceding vessel has been achieved, or whether there is a rest term
        delay = total_distance_to_lock_other_vessel / overruled_speed - total_distance_to_lock_other_vessel / average_speed
        difference_waiting_time = preceding_vessel_waiting_time_to_shorten_door_open_time.total_seconds() - delay

        # determine the newly planned arrival time for the preceding vessel, and whether this difference is greater than before
        planned_arrival_time_other_vessel = vessel_planning.loc[other_vessel_planning_index, 'time_lock_entry_start'] + preceding_vessel_waiting_time_to_shorten_door_open_time
        planned_arrival_time_other_vessel = planned_arrival_time_other_vessel - pd.Timedelta(seconds=difference_waiting_time)
        arrival_time_difference = registered_vessel_time_lock_entry_start - planned_arrival_time_other_vessel

        # if there was no optimisation possible, or the arrival time difference is still greater than the closing and opening the doors in between, or the other vessel is not sailing at this moment or did not request the door yet -> then do nothing
        if arrival_time_difference > pd.Timedelta(seconds=self.doors_closing_time + self.doors_opening_time) or delay <= 0 or 'process' not in dir(other_vessel) or 'door_open_request' not in dir(other_vessel):
            return

        # update the vessel and operation plannings, overrule the other vessels speed by interrupting its sailing process TODO: this communication of interrupting should be checked
        vessel_planning.loc[other_vessel_planning_index, 'time_lock_entry_start'] += datetime.timedelta(seconds=delay)
        self.overrule_vessel_speed(other_vessel, lock_end_node, waiting_time=delay)
        other_vessel.process.interrupt()
        operation_planning.loc[operation_index, 'time_entry_start'] += datetime.timedelta(seconds=delay)
        operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop'] += datetime.timedelta(seconds=delay)
        vessel_planning.loc[other_vessel_planning_index, 'time_potential_lock_door_opening_stop'] += datetime.timedelta(seconds=delay)
        vessel_planning.loc[other_vessel_planning_index, 'time_lock_entry_stop'] += datetime.timedelta(seconds=delay)
        other_vessel.door_open_request.interrupt(str(delay)) #

    def calculate_sailing_information_on_route_to_lock_complex(self, vessel, lock_end_node):
        """
        Calculates the sailing information (i.e., duration, distance, and speed) of the vessel per edge of its route between its current location and the lock doors

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        lock_end_node : str
            the node name that forms the end node of the lock complex given the direction of the vessel

        Returns
        -------
        sailing_time : pd.DataFrame
            sailing information (i.e., duration, distance, and speed) per edge of the route of the vessel between its current location and the lock doors
        """

        # unpacks the logbook of the vessel
        vessel_df = pd.DataFrame(vessel.logbook)
        if vessel_df.empty:
            return pd.DataFrame()

        # determine the sailing time already based on the current edge (if registration node is not coupled to node, but instead is somewhere along the edge: not sure if this is already implemented)
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
        reversed_vessel_df = vessel_df.iloc[::-1]
        for index,message in reversed_vessel_df.iterrows():
            if 'node' in message.Message:
                break
        passed_time = (current_time - message.Timestamp).total_seconds()

        # determines the distance from the node of the edge to the lock doors (depending on the direction of the vessel)
        distance = self.distance_from_start_node_to_lock_doors_A
        if lock_end_node != self.end_node:
            distance = self.distance_from_end_node_to_lock_doors_B

        # determine the sailing time from its current node to the end of the lock complex (depending on the direction of the vessel)
        route_vessel = vessel.route_ahead
        route_index_current_node = route_vessel.index(vessel.current_node)
        route_index_end_of_lock_complex = route_vessel.index(lock_end_node)
        route_vessel_to_pass_lock_complex = route_vessel[route_index_current_node:route_index_end_of_lock_complex]
        sailing_information = self.env.vessel_traffic_service.provide_sailing_time(vessel, route_vessel_to_pass_lock_complex) #TODO: maybe rename this function in the VTS, because it provides a dataframe of the sailing information (i.e., time, speed, and distance) per edge over the route of the vessel

        # correct the sailing time at the lock complex edge to the distance on that edge from the node to the lock doors (depending on the direction of the vessel)
        last_sailing_index = sailing_information.iloc[-1].index
        sailing_information.loc[last_sailing_index, 'Distance'] = distance
        sailing_information.loc[last_sailing_index, 'Time'] = distance / sailing_information.loc[last_sailing_index, 'Speed']

        # if there are overruled speeds implemented, correct the above speeds and sailing times
        if not vessel.overruled_speed.empty:
            for edge, overruled_speed in vessel.overruled_speed.iterrows():
                edge_index_mask = sailing_information.index == edge
                sailing_information.loc[edge_index_mask, 'Speed'] = overruled_speed.Speed
                sailing_information.loc[edge_index_mask, 'Time'] = sailing_information.loc[edge_index_mask, 'Distance'] / sailing_information.loc[edge_index_mask, 'Speed']

        # determine the index of the first edge in the sailing time dataframe to correct the sailing distance and sailing time of this edge with the already passed time and passed distance by this ship over this edge
        index_sailing_on_first_edge = (sailing_information[sailing_information.index.isin([(vessel.current_node, route_vessel_to_pass_lock_complex[1], 0)])].iloc[0].name)
        index_mask = sailing_information.index == index_sailing_on_first_edge
        interpolation = 1 - passed_time / sailing_information.loc[index_mask].Time
        sailing_information.loc[sailing_information[index_mask].index, 'Distance'] = sailing_information.loc[sailing_information[index_mask].index, 'Distance'] * interpolation
        sailing_information.loc[sailing_information[index_mask].index, 'Time'] = sailing_information.loc[sailing_information[index_mask].index, 'Time'] * interpolation
        sailing_information['Speed'] = sailing_information['Speed'].astype(float)

        return sailing_information

    def overrule_vessel_speed(self, vessel, lock_end_node, waiting_time=0.):
        """
        Overrules the speed of an vessel based on the additional waiting time

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        lock_end_node : str
            the node name that forms the end node of the lock complex given the direction of the vessel
        waiting_time : float
            waiting duration in seconds
        """

        # determines the sailing information of the vessel (i.e., speed, distance, time) over the edges from its current location to its first encountered lock doors
        sailing_information = self.calculate_sailing_information_on_route_to_lock_complex(vessel, lock_end_node)

        # skip function if no sailing information is available
        if sailing_information.empty:
            return

        # determines the average speed of the vessel over its route and calculate the overruled speed of the vessel based on the waiting time
        average_speed = sailing_information.loc[:, 'Distance'].sum()/sailing_information.loc[:, 'Time'].sum()
        overruled_speed = np.max([self.minimum_manoeuvrability_speed, sailing_information.loc[:, 'Distance'].sum()/(sailing_information.loc[:, 'Time'].sum() + waiting_time)])
        reversed_sailing_information = sailing_information.iloc[::-1]

        # TODO: Dit lijkt me een goed algoritme om los te koppelen.
        # TODO Floor: Wil je de naam van het algoritme in de documentatie zetten als die bestaat? #Comment Floor: we moeten hier samen even naar kijken

        # loops over the sailing information of the edges to adhere to the overruled speed (averaged over the route), the stops if too much iterations are required or when the difference between the new average speed and the overruled speed are sufficiently close to each other or when there are no speeds to be reduced
        iteration = 0
        speed_mask = reversed_sailing_information.Speed < self.minimum_manoeuvrability_speed
        while not np.abs(average_speed-overruled_speed) <= 0.01 and not reversed_sailing_information[speed_mask].empty:
            if iteration == 100:
                break

            # the difference in new average speed and overrulled speed
            speed_difference = average_speed - overruled_speed

            # identifies all speeds that are still greater than the minimum required speed for manoevrability (safety), so that these speeds can be reduced -> adjust the speed and time
            speed_mask = reversed_sailing_information.Speed > self.minimum_manoeuvrability_speed
            reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'Speed'] -= speed_difference
            reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'Time'] = reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'Distance'] / \
                                                                                                       reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'Speed']

            # if in the previous steps speeds have been reduced to less than the minimum manoevrability speed, then change these speeds to this minimum -> adjust again the speed and time
            speed_mask = reversed_sailing_information.Speed < self.minimum_manoeuvrability_speed
            reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'Speed'] = self.minimum_manoeuvrability_speed
            reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'Time'] = reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'Distance'] / \
                                                                                                       reversed_sailing_information.loc[reversed_sailing_information[speed_mask].index, 'Speed']

            # calculate the new average speed and increase the iteration number by one
            average_speed = reversed_sailing_information.Distance.sum()/reversed_sailing_information.Time.sum()
            iteration += 1

        # store the new sailing information info in an overruled speed dataframe object for the vessel
        for edge, reversed_sailing_information_info in reversed_sailing_information.iterrows():
            vessel.overruled_speed.loc[edge] = reversed_sailing_information_info.Speed

    def add_vessel_to_vessel_planning(self, vessel, direction, time_of_registration=None):
        """
        Adds vessel to the vessel planning of the lock complex upon request

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        time_of_registration : pd.Timestamp
            the time that the vessel registers to the lock master
        """

        node_from, node_to = _get_lock_operation_to_and_from_node(self, direction)

        # determining current time
        if time_of_registration is None:
            time_of_registration = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))

        # unpacks the vessel planning
        vessel_planning = self.lock_complex.vessel_planning

        # add vessel to the vessel planning dataframe with its information
        vessel_planning_index = len(vessel_planning)
        vessel_planning.loc[vessel_planning_index, 'id'] = vessel.id
        vessel_planning.loc[vessel_planning_index, 'time_of_registration'] = time_of_registration
        vessel_planning.loc[vessel_planning_index, 'time_of_acceptance'] = time_of_registration
        vessel_planning.loc[vessel_planning_index, 'node_from'] = node_from
        vessel_planning.loc[vessel_planning_index, 'node_to'] = node_to
        vessel_planning.loc[vessel_planning_index, 'direction'] = direction
        vessel_planning.loc[vessel_planning_index, 'L'] = vessel.L
        vessel_planning.loc[vessel_planning_index, 'B'] = vessel.B
        vessel_planning.loc[vessel_planning_index, 'T'] = vessel.T

        # adds to the vessel planning the arrival time at each of the infrastructures of the lock complex
        _ = self.calculate_sailing_time_to_waiting_area(vessel, direction)
        if (not direction and self.has_lineup_area_A) or (direction and self.has_lineup_area_B): #if lock has a lineup area
            self.calculate_sailing_time_to_lineup_area(vessel, direction)
        _ = self.calculate_sailing_time_to_approach_point(vessel, direction)
        _ = self.calculate_sailing_time_to_lock_door(vessel, direction)

    def add_empty_lock_operation_to_planning(self, operation_index, direction):
        """
        Adds an empty lock operation to the operation planning
        
        Parameters
        ----------
        operation_index : int
            index of the lock operation to which the vessel can be added (can either be an existing or a new lock operation)
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        """
        # unpack the lock master's lock operation planning
        operation_planning = self.lock_complex.operation_planning

        node_from, node_to = _get_lock_operation_to_and_from_node(self, direction)

        # determine the start time of this empty lock operation
        preceding_operations = operation_planning[operation_planning.index < operation_index]
        if not preceding_operations.empty:
            preceding_operation = operation_planning.loc[operation_index-1]
            first_empty_lock_operation_start = preceding_operation.time_potential_lock_door_closure_start
        else:
            first_empty_lock_operation_start = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))

        # add empty lock operation to operation planning of the lock master, including deriving the lock operation information (i.e., start and stop times of individual events, water levels, and status)
        operation_planning.loc[operation_index, 'node_from'] = node_from
        operation_planning.loc[operation_index, 'node_to'] = node_to
        operation_planning.loc[operation_index, 'direction'] = direction
        operation_planning.loc[operation_index, "lock_chamber"] = self.lock_complex.name
        operation_planning.loc[operation_index, 'vessels'] = []
        operation_planning.loc[operation_index, 'capacity_L'] = self.lock_complex.lock_length
        operation_planning.loc[operation_index, 'capacity_B'] = self.lock_complex.lock_width

        levelling_information = self.calculate_lock_operation_times(operation_index=operation_index,
                                                                    last_entering_time=first_empty_lock_operation_start,
                                                                    start_time=first_empty_lock_operation_start,
                                                                    direction=direction)

        wlev_A, wlev_B = levelling_information["wlev_A"], levelling_information["wlev_B"]
        operation_planning.loc[operation_index, 'time_operation_start'] = first_empty_lock_operation_start
        operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop'] = first_empty_lock_operation_start
        operation_planning.loc[operation_index, 'time_entry_start'] = first_empty_lock_operation_start
        operation_planning.loc[operation_index, 'time_entry_stop'] = first_empty_lock_operation_start
        operation_planning.loc[operation_index, 'time_door_closing_start'] = levelling_information["time_door_closing_start"]
        operation_planning.loc[operation_index, 'time_door_closing_stop'] = levelling_information["time_door_closing_stop"]
        operation_planning.loc[operation_index, 'time_levelling_start'] = levelling_information["time_levelling_start"]
        operation_planning.loc[operation_index, 'time_levelling_stop'] = levelling_information["time_levelling_stop"]
        operation_planning.loc[operation_index, 'time_door_opening_start'] = levelling_information["time_levelling_stop"]
        operation_planning.loc[operation_index, 'time_door_opening_stop'] = levelling_information["time_door_opening_stop"]
        operation_planning.loc[operation_index, 'time_departure_start'] = levelling_information["time_door_opening_stop"]
        operation_planning.loc[operation_index, 'time_departure_stop'] = levelling_information["time_door_opening_stop"]
        operation_planning.loc[operation_index, 'time_potential_lock_door_closure_start'] = levelling_information["time_door_opening_stop"]
        operation_planning.loc[operation_index, 'time_operation_stop'] = levelling_information["time_door_opening_stop"]
        operation_planning.loc[operation_index, 'wlev_A'] = wlev_A
        operation_planning.loc[operation_index, 'wlev_B'] = wlev_B
        operation_planning.loc[operation_index, 'maximum_individual_delay'] = pd.Timedelta(seconds=0)
        operation_planning.loc[operation_index, 'total_delay'] = pd.Timedelta(seconds=0)
        operation_planning.loc[operation_index, 'status'] = 'available'

    def calculate_sailing_time_to_waiting_area(self, vessel, direction, current_node=None, prognosis=False, overwrite=True):
        """TODO: note that this function looks a lot like other 'calculate_sailing_time_to'-functions below, so maybe we can investigate to combine the functions
        Calculates the sailing time of a vessel from its location to the waiting area

        Parameters
        -------------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        current_node : str
            the node name (which has to be in the graph) at which the vessel is currently sailing
        prognosis: bool
            .
        overwrite: bool
            .

        Returns
        -------
        sailing_to_waiting_area_time: pd.Timedelta
            sailing time to the waiting area in [s]
        sailing_distance: float
            sailing distance to the waiting area in [m]
        average_sailing_speed: float
            average sailing speed to the lock chambers's waiting area in [m/s]

        """

        # determine route to the start node of the edge at which the waiting area is located
        route_to_waiting_area = determine_route_to_closest_waiting_area(
            vessel=vessel, waiting_area_A=self.lock_complex.waiting_area_A, waiting_area_B=self.lock_complex.waiting_area_B
        )

        # unpack vessel planning
        vessel_planning = self.lock_complex.vessel_planning

        # unpack first encountered waiting area
        waiting_area_approach = self._get_appropriate_waiting_area(direction)

        # unpack the function that calculates sailing time from distance on edge to distance on another edge
        calculate_sailing_time = self.env.vessel_traffic_service.provide_sailing_time_distance_on_edge_to_distance_on_another_edge

        # determine the distance that the vessel has to sail on the edge at which the waiting area is located (from the start node of the edge)
        distance_to_waiting_area_on_last_edge = waiting_area_approach.distance_from_edge_start

        # calculation of the sailing information (time, distance, speed) per edge on route to the waiting area
        sailing_to_waiting_area = calculate_sailing_time(vessel, route=route_to_waiting_area, distance_sailed_on_last_edge=distance_to_waiting_area_on_last_edge)

        # calculation of the sailing time, distance, and average speed to the waiting area
        sailing_to_waiting_area_time = pd.Timedelta(seconds=sailing_to_waiting_area['Time'].sum())
        sailing_distance = sailing_to_waiting_area['Distance'].sum()
        average_sailing_speed = sailing_to_waiting_area['Speed']
        if sailing_to_waiting_area_time.total_seconds():
            average_sailing_speed = sailing_distance / sailing_to_waiting_area['Time'].sum()

        # calculate arrival time of vessel at the waiting area and add to the vessel planning of the lock complex master
        if not prognosis and overwrite:
            current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            vessel_planning.loc[vessel_planning_index, 'time_arrival_at_waiting_area'] = current_time + sailing_to_waiting_area_time

        return sailing_to_waiting_area_time, sailing_distance, average_sailing_speed

    def _get_appropriate_waiting_area(self, direction):
        """
        Returns the appropriate waiting area based on the direction of the vessel

        Parameters
        ----------
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Returns
        -------
        waiting_area : WaitingArea
            the appropriate waiting area object based on the direction of the vessel
        """
        if not direction:
            waiting_area = self.lock_complex.waiting_area_A
        else:
            waiting_area = self.lock_complex.waiting_area_B

        return waiting_area

    def _get_appropriate_lineup_area(self, direction):
        """
        Returns the appropriate line-up area based on the direction of the vessel

        Parameters
        ----------
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Returns
        -------
        lineup_area : LineupArea
            the appropriate line-up area object based on the direction of the vessel
        """
        if not direction:
            lineup_area = self.lock_complex.lineup_area_A
        else:
            lineup_area = self.lock_complex.lineup_area_B

        return lineup_area

    def calculate_sailing_time_to_lineup_area(self, vessel, direction, current_node=None, prognosis=False, overwrite=True):
        """
        Calculates the sailing time of a vessel from its location to the line-up area

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        current_node : str
            the node name (which has to be in the graph) at which the vessel is currently sailing
        prognosis :
            .
        overwrite :
            .

        Returns
        -------
        sailing_to_lineup_area_time : pd.Timedelta
            sailing time to the lock chambers's line-up area in [s]

        """
        # determine the current node of the vessel
        if current_node is None:
            current_node = vessel.current_node

        # unpack first encountered line-up area
        lineup_area_approach = self._get_appropriate_lineup_area(direction)

        # determine the route of the vessel to the line-up area edge
        route_to_lineup_area = nx.dijkstra_path(self.env.graph, current_node, lineup_area_approach.end_node)

        # unpack vessel planning
        vessel_planning = self.lock_complex.vessel_planning

        # unpack the function that calculates sailing time from distance on edge to distance on another edge
        calculate_sailing_time = self.env.vessel_traffic_service.provide_sailing_time_distance_on_edge_to_distance_on_another_edge

        # determine the distance that the vessel has to sail on the edge at which the line-up area is located (from the start node of the edge)
        distance_to_lineup_area_from_last_node = lineup_area_approach.distance_from_start_edge

        # calculation of the sailing information (time, distance, speed) per edge on route to the line-up area
        sailing_to_lineup_area = calculate_sailing_time(vessel, route=route_to_lineup_area,
                                                        distance_sailed_on_last_edge=distance_to_lineup_area_from_last_node)

        # calculation of the sailing time to the line-up area
        sailing_to_lineup_area_time = pd.Timedelta(seconds=sailing_to_lineup_area['Time'].sum())

        # calculate arrival time of vessel at the line-up area and add to the vessel planning of the lock complex master
        if not prognosis and overwrite:
            current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            vessel_planning.loc[vessel_planning_index, 'time_arrival_at_lineup_area'] = current_time + sailing_to_lineup_area_time

        return sailing_to_lineup_area_time

    def calculate_sailing_time_to_approach_point(
        self, vessel, direction, from_waiting_area=False, operation_index=None, prognosis=False, overwrite=True
    ):
        """
        Calculates the sailing time of a vessel from its location to the approach point

        The approach point is the closest location in front of the lock doors where the outdirection vessel(s) can pass the indirection vessel waiting to enter the lock.
        The point is located in between the line-up area and the lock doors.

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        current_node : str
            the node name (which has to be in the graph) at which the vessel is currently sailing
        prognosis :
            .
        overwrite :
            .

        Returns
        -------
        sailing_to_lineup_area_time : pd.Timedelta
            sailing time to the lock chambers's line-up area in [s]

        """
        # unpack vessel planning
        vessel_planning = self.lock_complex.vessel_planning

        # unpack sailing distance from crossing point to lock doors
        sailing_distance_from_entry = self.sailing_distance_to_crossing_point

        # determine the time of entering the lock
        sailing_speed_during_entry = self.vessel_sailing_in_speed(vessel, direction)
        sailing_time_entry = pd.Timedelta(seconds=sailing_distance_from_entry / sailing_speed_during_entry)

        # determine the time of the vessel to its first encountered waiting area and lock_door TODO: in the 'add_vessel_to_planning'-function these functions has already been done, so doing these again can be computational intensive and should be prevented. Can we include tests that before this function is ran, these following functions have already been ran? How can we extract the earlier output?
        sailing_time_to_waiting_area = pd.Timedelta(seconds=0)
        if from_waiting_area:
            sailing_time_to_waiting_area = self.calculate_sailing_time_to_waiting_area(vessel, direction, overwrite=overwrite)[0]
        sailing_time_to_lock_door = self.calculate_sailing_time_to_lock_door(vessel, direction, overwrite=overwrite)

        # determine the sailing time to the approach point
        sailing_time_to_start_approach = sailing_time_to_lock_door - sailing_time_entry - sailing_time_to_waiting_area #- sailing_time_to_waiting_area TODO: later check if we indeed can get rid of the sailing time to waiting area

        # calculate arrival time of vessel at the approach point and add to the vessel planning of the lock complex master
        if not prognosis and overwrite:
            current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start'] = current_time + sailing_time_to_start_approach
            if operation_index is not None:
                passing_start_time = self.calculate_vessel_passing_start_time(
                    vessel, operation_index, direction, prognosis=prognosis, overwrite=overwrite
                )
                vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start'] = passing_start_time

        return sailing_time_to_start_approach

    def calculate_sailing_time_to_lock_door(self, vessel, direction, prognosis=False, overwrite=True):
        """
        Calculates the sailing time of a vessel from its location to the first lock doors that it will encounter

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        current_node : str
            the node name (which has to be in the graph) at which the vessel is currently sailing
        prognosis :
            .
        overwrite :
            .

        Returns
        -------
        sailing_to_lineup_area_time : pd.Timedelta
            sailing time to the lock chambers's line-up area in [s]

        """
        # unpack vessel planning
        vessel_planning = self.lock_complex.vessel_planning

        # determine the end node of the lock complex from the perspective of the vessel and the distance from the start node of the lock complex to the lock doors
        distance_to_lock = self._distance_to_lock(direction)

        # determine the route of the vessel to the end node of the lock complex from the perspective of the vessel
        route_to_lock_chamber = vessel._find_route_to_lock(lock=self)

        # unpack the function that calculates sailing time from distance on edge to distance on another edge
        calculate_sailing_time = self.env.vessel_traffic_service.provide_sailing_time_distance_on_edge_to_distance_on_another_edge

        # calculate sailing time to the start node of the edge of lock complex from the perspective of the vessel
        sailing_to_lock_chamber = calculate_sailing_time(vessel, route=route_to_lock_chamber)
        sailing_to_lock_chamber_distance = sailing_to_lock_chamber['Distance'].sum()
        sailing_to_lock_chamber_time = sailing_to_lock_chamber['Time'].sum()

        # add sailing distance and time to the lock doors on the edge of the lock complex to sailing information to the start node of this edge
        sailing_to_lock_chamber_distance += distance_to_lock
        sailing_to_lock_chamber_time += distance_to_lock / self.vessel_sailing_in_speed(vessel, direction)
        sailing_to_lock_chamber_time = pd.Timedelta(seconds=sailing_to_lock_chamber_time)

        # calculate arrival time of vessel at the first to be encountered lock doors and add to the vessel planning of the lock complex master
        if not prognosis and overwrite:
            current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start'] = current_time + sailing_to_lock_chamber_time

        return sailing_to_lock_chamber_time

    def _distance_to_lock(self, direction):
        """get the distance from the start node of the lock to the lock doors from the perspective of the vessel

        Parameters
        ----------
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        """
        if not direction:
            return self.distance_from_start_node_to_lock_doors_A
        else:
            return self.distance_from_end_node_to_lock_doors_B

    def calculate_sailing_time_in_lock(self, vessel, operation_index, prognosis=False):
        """
        Calculates the time duration that a vessel needs to enter the lock until laying still

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        prognosis : bool
            .

        Returns
        -------
        sailing_time_into_lock : pd.Timedelta
            the time duration of the process of sail in the lock [s]

        """
        # determine the vessels assigned to the lock operation (that are already in the lock)
        vessels = self.get_vessel_from_planned_operation(
            operation_index=operation_index,
        )

        # determine the sailing distance from the lock door to the position assigned to the vessel
        if not prognosis:
            # TODO: @Floor: In principe werkt de eerste formule altijd toch? Stel er zitten 2 vessels in de lock, dan wil je dat de afstand verschilt per vessel toch?
            vessel_index = vessels.index(vessel)
            sailing_distance_from_lock_doors = (self.lock_length - np.sum([vessel.L for vessel in vessels[:vessel_index]])) - 0.5 * vessel.L
        else:
            sailing_distance_from_lock_doors = (self.lock_length - np.sum([vessel.L for vessel in vessels]) - 0.5 * vessel.L)

        # determine the sailing speed of the vessel in the lock
        sailing_speed_into_lock = self.vessel_sailing_speed_in_lock(vessel)

        # calculate the time required to complete the process of sailing from the lock doors to laying still in the lock chamber on the assigned longitudinal coordinate (x)
        sailing_time_into_lock = pd.Timedelta(seconds=sailing_distance_from_lock_doors / sailing_speed_into_lock)

        return sailing_time_into_lock

    def calculate_sailing_in_time_delay(
        self, vessel, operation_index, direction, minimum_difference_with_previous_vessel=False, prognosis=False, overwrite=True
    ):
        """
        Calculates the minimum required time gap between two entering vessels for safety, resulting in a delay

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            the index of the lock operation in the operation planning dataframe
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        minimum_difference_with_previous_vessel : bool
            .
        prognosis : bool
            .
        overwrite : bool
            .

        Returns
        -------
        sailing_in_time_delay : pd.Timedelta
            time delay because of waiting for the vessel to sail entering the lock [s]

        """

        # determine current time and set default sailing in time delay
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        sailing_in_time_delay = pd.Timedelta(seconds=0)

        # unpack the vessel planning of the lock complex master
        vessel_planning = self.lock_complex.vessel_planning

        # unpack the vessels from the lock operations
        vessels = self.get_vessel_from_planned_operation(
            operation_index=operation_index,
        )

        # determine the first vessel of the lock operation
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index)

        # determine the index of the vessel in the vessel planning of the lock complex master
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

        # determine the sailing time to the lock door to determine the vessel entry start time (if this changed over the route of the vessel) TODO: is this required or can we extract this from the vessel planning?
        sailing_time_to_lock = self.calculate_sailing_time_to_lock_door(vessel, direction, prognosis=prognosis, overwrite=overwrite)
        vessel_entry_start_timestamp = np.max([current_time + sailing_time_to_lock, vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start']])

        # determine the vessel index in the vessels assigned to the lock operation
        if not prognosis:
            vessel_index = vessels.index(vessel)
        else:
            vessel_index = -1

        # determine the previously assigned vessel to the lock operation
        previous_vessel = None
        if not prognosis and vessel != first_vessel:
            previous_vessel = vessels[vessel_index - 1]
        elif prognosis and len(vessels):
            previous_vessel = vessels[-1]

        # if the assigned vessel is the first one (there is no assigned vessel), there is no delay
        if previous_vessel is None:
            return sailing_in_time_delay

        # if there is a previous vessel: determine its entry start and stop times
        previous_vessel_planning_index = vessel_planning[vessel_planning.id == previous_vessel.id].iloc[-1].name
        previous_vessel_entry_start_timestamp = vessel_planning.loc[previous_vessel_planning_index,'time_lock_entry_start']
        previous_vessel_laying_still_time = vessel_planning.loc[previous_vessel_planning_index, 'time_lock_entry_stop']

        # if there is a rule set that requires a certain minimum time gap of the vessel with respect to the previous vessel, than use the previous vessel's entry time
        if minimum_difference_with_previous_vessel:
            vessel_entry_start_timestamp = previous_vessel_entry_start_timestamp

        # determine the difference between the entry start times of the vessel and the previous vessel, and also the entry stop times
        difference_entry_start_timestamp = vessel_entry_start_timestamp - previous_vessel_entry_start_timestamp
        difference_berthing_time_previous_vessel_and_vessel_sailing_in_time = (vessel_entry_start_timestamp - previous_vessel_laying_still_time)

        # calculate sailing in time delay if the difference between these entry start times is too small given the rule set for the time gap of the vessels sailing through the lock doors
        if difference_entry_start_timestamp < pd.Timedelta(seconds=self.sailing_in_time_gap_through_doors):
            sailing_in_time_delay = pd.Timedelta(seconds=self.sailing_in_time_gap_through_doors)-difference_entry_start_timestamp

        # calculate sailing in time delay if the difference between these entry stop times is too small given the rule set for the time gap of the vessels between berthing in the lock
        if difference_berthing_time_previous_vessel_and_vessel_sailing_in_time < pd.Timedelta(seconds=self.sailing_in_time_gap_after_berthing_previous_vessel):
            sailing_in_time_delay = np.max([(previous_vessel_laying_still_time+pd.Timedelta(seconds=self.sailing_in_time_gap_after_berthing_previous_vessel))-vessel_entry_start_timestamp,sailing_in_time_delay])

        return sailing_in_time_delay

    def calculate_vessel_entry_duration(self, vessel, direction):
        """
        Calculates the time duration required for a vessel starts entering the lock (from approach point to first encountered lock doors)

        Parameters
        ----------
        vessel : type [optional]
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Returns
        -------
        sailing_time_entry : pd.Timedelta
            the time duration required for a vessel starts entering the lock [s]

        """
        # determine the distance from the lock doors to the approach point
        sailing_distance_from_entry = self.sailing_distance_to_crossing_point

        # determine the vessel speed when entering the lock
        sailing_speed_during_entry = self.vessel_sailing_in_speed(vessel, direction)

        # determine the time of the process of entering
        sailing_time_entry = pd.Timedelta(seconds=sailing_distance_from_entry / sailing_speed_during_entry)

        return sailing_time_entry

    def calculate_vessel_passing_start_time(self, vessel, operation_index, direction, prognosis=False, overwrite=True):
        """
        Calculates the start time that a vessel can start its manoeuvre of entering the lock

        Parameters
        ----------
        vessel : type [optional]
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            the index of the lock operation in the operation planning of the lock complex master
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)
        overwrite : bool
            overwrites the vessel planning: True (yes) or False (no)

        Returns
        -------
        vessel_passing_start_timestamp : pd.Timestamp
            the moment in time that a vessel starts entering the lock from the approach point

        """
        # determines the current time
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))

        # calculate the sailing time durations to the lock door, the approach point and if there is any form of delay for this
        sailing_time_to_lock = self.calculate_sailing_time_to_lock_door(vessel, direction, prognosis=prognosis, overwrite=overwrite)
        sailing_time_entry = self.calculate_vessel_entry_duration(vessel, direction)
        sailing_in_delay = self.calculate_sailing_in_time_delay(vessel, operation_index, direction, prognosis=prognosis, overwrite=overwrite)

        # calculate time that the vessel can start passing the lock
        vessel_passing_start_timestamp = current_time + (sailing_time_to_lock - sailing_time_entry) + sailing_in_delay

        return vessel_passing_start_timestamp

    def calculate_lock_operation_start_time(self, vessel, operation_index, direction, prognosis=False, overwrite=True):
        """
        Calculates the new earliest possible start time of a lock operation

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            the index of the lock operation in the operation planning of the lock complex master
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)
        overwrite : bool
            overwrites the vessel planning: True (yes) or False (no)

        Returns
        -------
        lock_operation_start_time : pd.Timestamp
            the moment in time of the start of the lock operation

        """
        # unpacks the lock complex master's operation planning
        operation_planning = self.lock_complex.operation_planning

        # determines the lock operation start time based on the first vessel that was assigned to this lock operation
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index)
        lock_operation_start_time = self.calculate_vessel_passing_start_time(first_vessel, operation_index, direction, prognosis, overwrite=overwrite)

        # determines the lock_operation_start_time based on whether it fits given the previous lock operations (should not be overlapping)
        previous_operations = operation_planning[operation_planning.index < operation_index]
        if not previous_operations.empty:
            previous_operation = previous_operations.iloc[-1]
            previous_lock_operation_stop_time = previous_operation.time_operation_stop
            if lock_operation_start_time < previous_lock_operation_stop_time:
                lock_operation_start_time = previous_lock_operation_stop_time

        return lock_operation_start_time

    def calculate_lock_door_opening_time(self, vessel, operation_index, direction, operation_start_time):
        """
        Calculates the time at which the lock doors can open before an vessel arrival

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            the index of the lock operation in the operation planning of the lock complex master
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Returns
        -------
        lock_entry_start_time : pd.Timestamp
            the time at which the lock doors can open before an vessel arrival
        """
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index)
        lock_entry_start_duration = self.calculate_vessel_entry_duration(first_vessel, direction)
        lock_entry_start_duration -= self.lock_chamber.minimum_advance_to_open_doors()
        lock_entry_start_time = lock_entry_start_duration + operation_start_time
        return lock_entry_start_time

    def calculate_lock_entry_start_time(self, vessel, operation_index, direction, operation_start_time):
        """
        Calculates the time at which the vessel can start sailing into to lock

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            the index of the lock operation in the operation planning of the lock complex master
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Returns
        -------
        lock_entry_start_time : pd.Timestamp
            the time at which the vessel can start sailing into to lock
        """
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index)
        lock_entry_start_duration = self.calculate_vessel_entry_duration(first_vessel, direction)
        lock_entry_start_time = lock_entry_start_duration + operation_start_time
        return lock_entry_start_time

    def calculate_vessel_entry_stop_time(self, vessel, operation_index, direction, prognosis=False):
        """
        Calculates the moment in time that a vessel finished its lock entry process

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)

        Returns
        -------
        vessel_entry_stop_time : pd.Timestamp
             the moment in time that the vessel stops entering the lock
        """

        # determine the moment in time that the vessel starts to enter the lock
        vessel_entry_start_time = self.calculate_vessel_entry_duration(vessel, direction)

        # determine the time duration of the vessel in the lock
        sailing_time_in_lock = self.calculate_sailing_time_in_lock(vessel, operation_index, prognosis)

        # calculate the moment in time that the vessel stops entering the lock
        vessel_entry_stop_time = vessel_entry_start_time + sailing_time_in_lock

        return vessel_entry_stop_time

    def calculate_lock_entry_stop_time(self, vessel, operation_index, direction, lock_entry_start_time, prognosis=False):
        """
        Calculates the moment in time that a lock operation entry process of a lock operation is finished (all vessels are in lock chamber)

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)

        Returns
        -------
        lock_entry_stop_time : pd.Timestamp
            the time that a lock operation entry process is finished
        """

        # determine the last assigned vessel of the lock operation to determine the lock entry stop time
        last_vessel = self.determine_last_vessel_of_lock_operation(vessel, operation_index, prognosis)
        lock_entry_stop_duration = self.calculate_vessel_entry_stop_time(last_vessel, operation_index, direction, prognosis)
        lock_entry_stop_time = lock_entry_stop_duration + lock_entry_start_time
        return lock_entry_stop_time

    def calculate_lock_operation_times(self, operation_index, last_entering_time, start_time, vessel = None, direction=None):
        """
        Calculates the moments in time of the start and stop of the operation steps of the lock: (1) door closing, (2) levelling, (3) door opening

        Parameters
        ----------
        operation_index : int
            the index of the lock operation in the operation planning of the lock complex master
        last_entering_time : pd.Timestamp
            the time that the last vessel entered the lock
        start_time : pd.Timestamp
            the start time of the lock operation (i.e., for the doors to close)
        vessel : type [optional]
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int [optional]
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

        Returns
        -------
        levelling_information : dict
            including:
                time_door_closing_start : pd.Timestamp
                    the time that the lock doors are planned to start closing
                time_door_closing_stop : pd.Timestamp
                    the time that the lock doors are planned to stop closing
                time_levelling_start : pd.Timestamp
                    the time that the lock chamber is planned to start levelling
                time_levelling_stop : pd.Timestamp
                    the time that the lock chamber is planned to stop levelling
                time_door_opening_start : pd.Timestamp
                    the time that the lock doors are planned to start opening
                time_door_opening_stop : pd.Timestamp
                    the time that the lock doors are planned to stop opening
        """
        # unpack the lock complex master's vessel and operation plannings
        vessel_planning = self.lock_complex.vessel_planning
        operation_planning = self.lock_complex.operation_planning

        try:
            vessel_goes_with_previous_operation = start_time < operation_planning.loc[operation_index, "time_door_closing_start"]
        except:
            vessel_goes_with_previous_operation = False

        if vessel_goes_with_previous_operation:
            time_door_closing_start = operation_planning.loc[operation_index, "time_door_closing_start"]
            time_door_closing_stop = operation_planning.loc[operation_index, "time_door_closing_stop"]
            time_levelling_start = operation_planning.loc[operation_index, "time_levelling_start"]
            time_levelling_stop = operation_planning.loc[operation_index, "time_levelling_stop"]
            time_door_opening_start = operation_planning.loc[operation_index, "time_door_opening_start"]
            time_door_opening_stop = operation_planning.loc[operation_index, "time_door_opening_stop"]

        else:
            # set default time door closing start as start time
            time_door_closing_start = start_time

            # overwrite the time door closing start if there is a rule that the doors can close before a vessel is laying still and there are vessels in the lock
            if self.close_doors_before_vessel_is_laying_still and vessel is not None:
                time_door_closing_start = last_entering_time + self.lock_chamber.minimum_delay_to_close_doors()

            # determine the new closing stop times of the doors and the time that the levelling can hence start
            time_door_closing_stop = time_door_closing_start + pd.Timedelta(seconds=self.lock_complex.doors_closing_time)
            time_levelling_start = time_door_closing_stop

            # overwrite the time of levelling start if there is a rule that the doors can close before a vessel is laying still and there are vessels in the lock (the vessel always has to lay still before levelling can start)
            if self.close_doors_before_vessel_is_laying_still and vessel is not None:
                vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
                if not isinstance(vessel_planning.loc[vessel_planning_index,'time_lock_entry_stop'],float):
                    time_levelling_start = np.max([vessel_planning.loc[vessel_planning_index,'time_lock_entry_stop'],time_levelling_start])
                else:
                    time_levelling_start = time_levelling_start

            # determine levelling stop time and door opening start and stop times
            time_levelling_stop,_,_ = self.lock_complex.determine_levelling_time(t_start=time_levelling_start, operation_index=operation_index, direction=direction, prediction=True)
            time_levelling_stop = time_levelling_start + pd.Timedelta(seconds=time_levelling_stop)
            time_door_opening_start = time_levelling_stop
            time_door_opening_stop = time_levelling_stop + pd.Timedelta(seconds=self.lock_complex.doors_opening_time)

        wlev_A, wlev_B = self.lock_chamber.determine_water_levels_before_and_after_levelling(
            time_levelling_start, time_levelling_stop, direction
        )

        levelling_information = {"time_door_closing_start":time_door_closing_start,
                                 "time_door_closing_stop":time_door_closing_stop,
                                 "time_levelling_start":time_levelling_start,
                                 "time_levelling_stop":time_levelling_stop,
                                 "time_door_opening_start":time_door_opening_start,
                                 "time_door_opening_stop":time_door_opening_stop,
                                 "wlev_A":wlev_A,
                                 "wlev_B":wlev_B}
        return levelling_information

    def calculate_vessel_departure_start_delay(self, vessel, operation_index, direction, prognosis=False):
        """
        Calculates the delay for a vessel to start leaving the lock

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)

        Returns
        -------
        delay_to_departure : pd.Timestamp
            the delay of a vessel to start its departure process out of the lock
        """
        vessels = self.get_vessel_from_planned_operation(operation_index=operation_index)

        if not prognosis:
            vessel_index = vessels.index(vessel)
            number_of_previous_vessels = vessel_index
        else:
            vessel_index = -1
            number_of_previous_vessels = len(vessels)

        delay_to_departure = pd.Timedelta(seconds=0)
        if number_of_previous_vessels:
            previous_vessel = vessels[vessel_index - 1]
            vessel_sailing_out_time = self.calculate_vessel_sailing_time_out_of_lock(vessel,
                                                                                     operation_index,
                                                                                     prognosis=prognosis)
            previous_vessel_sailing_out_time = self.calculate_vessel_sailing_time_out_of_lock(previous_vessel,
                                                                                              operation_index,
                                                                                              prognosis=prognosis)
            sailing_out_time_gap_through_doors = (vessel_sailing_out_time - previous_vessel_sailing_out_time)
            if sailing_out_time_gap_through_doors < pd.Timedelta(seconds=self.sailing_out_time_gap_through_doors):
                delay_to_departure += number_of_previous_vessels * \
                                      pd.Timedelta(seconds=self.sailing_out_time_gap_through_doors) - \
                                      sailing_out_time_gap_through_doors

            if self.sailing_out_time_gap_after_berthing_previous_vessel is not None and delay_to_departure < pd.Timedelta(seconds=self.sailing_out_time_gap_after_berthing_previous_vessel) * number_of_previous_vessels:
                delay_to_departure = pd.Timedelta(seconds=self.sailing_out_time_gap_after_berthing_previous_vessel) * number_of_previous_vessels

        delay_to_departure += pd.Timedelta(seconds=self.start_sailing_out_time_after_doors_have_been_opened)
        return delay_to_departure

    def calculate_vessel_departure_start_time(self, vessel, operation_index, direction, operation_stop_time, prognosis=False):
        """
        Calculates the moment in time that a vessel can start leaving the lock

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        operation_stop_time : pd.Timestamp
            the time that the lock operation has stopped (i.e., doors have been opened again)
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)

        Returns
        -------
        departure_start_time : pd.Timestamp
            the time that a vessel's departure process out of the lock can start
        """
        delay_to_departure = self.calculate_vessel_departure_start_delay(vessel, operation_index, direction, prognosis)
        departure_start_time = operation_stop_time + delay_to_departure
        return departure_start_time

    def calculate_lock_departure_start_time(self, vessel, operation_index, direction, operation_stop_time, prognosis=False):
        """
        Calculates the moment in time the departure can start of a lock operation

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        operation_stop_time : pd.Timestamp
            the time that the lock operation has stopped (i.e., doors have been opened again)
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)

        Returns
        -------
        departure_start_time : pd.Timestamp
            the time that a lock operation's departure process can start
        """
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index)
        time_departure_start = self.calculate_vessel_departure_start_time(first_vessel, operation_index, direction, operation_stop_time, prognosis)
        return time_departure_start

    def calculate_vessel_sailing_time_out_of_lock(self, vessel, operation_index, prognosis=False):
        """
        Calculates the sailing time for a vessel to sail from its position in the lock to the lock doors that have to be passed to sail out of the lock

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)

        Returns
        -------
        departure_start_time : pd.Timestamp
            the time that a lock operation's departure process can start
        """
        vessels = self.get_vessel_from_planned_operation(operation_index=operation_index,)
        if not prognosis:
            vessel_index = vessels.index(vessel)
            distance_to_lock = np.sum([vessel.L for vessel in vessels[:vessel_index]]) + 0.5 * vessel.L
        else:
            distance_to_lock = np.sum([vessel.L for vessel in vessels]) + 0.5 * vessel.L

        vessel_speed = self.vessel_sailing_speed_out_lock(vessel)
        sailing_out_time = pd.Timedelta(seconds=distance_to_lock / vessel_speed)
        return sailing_out_time

    def calculate_vessel_departure_stop_time(self, vessel, operation_index, direction, operation_stop_time, prognosis=False):
        """
        Calculates the moment in time the departure process of a vessel stops

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        operation_stop_time : pd.Timestamp
            the time that the lock operation has stopped (i.e., doors have been opened again)
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)

        Returns
        -------
        time_departure_stop : pd.Timestamp
            the moment in time that a vessel's departure process stops
        """
        time_departure_start = self.calculate_vessel_departure_start_time(vessel, operation_index, direction, operation_stop_time, prognosis)
        sailing_out_time = self.calculate_vessel_sailing_time_out_of_lock(vessel, operation_index, prognosis)
        time_departure_stop = time_departure_start + sailing_out_time
        return time_departure_stop

    def calculate_lock_departure_stop_time(self, vessel, operation_index, direction, operation_stop_time, prognosis=False):
        """
        Calculates the moment in time the departure process of a lock operation stops

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        operation_stop_time : pd.Timestamp
            the time that the lock operation has stopped (i.e., doors have been opened again)
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)

        Returns
        -------
        time_departure_stop : pd.Timestamp
            the moment in time that a lock operation's departure process stops
        """
        last_vessel = self.determine_last_vessel_of_lock_operation(vessel, operation_index, prognosis)
        time_departure_stop = self.calculate_vessel_departure_stop_time(last_vessel, operation_index, direction, operation_stop_time, prognosis)
        return time_departure_stop

    def calculate_vessel_passing_stop_time(self, vessel, operation_index, direction, operation_stop_time, prognosis=False):
        """
        Calculates the moment in time the vessel has reached the approach point at the other side of the lock (while sailing away from the lock)

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        operation_stop_time : pd.Timestamp
            the time that the lock operation has stopped (i.e., doors have been opened again)
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)

        Returns
        -------
        time_departure_stop : pd.Timestamp
            the moment in time the vessel has reached the approach point at the other side of the lock
        """
        time_departure_stop = self.calculate_vessel_departure_stop_time(vessel, operation_index, direction, operation_stop_time, prognosis)
        vessel_speed = self.vessel_sailing_out_speed(vessel, direction, until_crossing_point=True)
        time_departure_stop += pd.Timedelta(seconds = self.sailing_distance_to_crossing_point/vessel_speed)
        return time_departure_stop

    def calculate_lock_operation_stop_time(self, vessel, operation_index, direction, operation_stop_time, prognosis=False):
        """
        Calculates the moment in time a new lock operation can start

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        operation_stop_time : pd.Timestamp
            the time that the lock operation has stopped (i.e., doors have been opened again)
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)

        Returns
        -------
        time_operation_stop : pd.Timestamp
            the moment in time a new lock operation can start
        """
        last_vessel = self.determine_last_vessel_of_lock_operation(vessel, operation_index, prognosis)
        time_operation_stop = self.calculate_vessel_passing_stop_time(last_vessel, operation_index, direction, operation_stop_time, prognosis)
        return time_operation_stop

    def calculate_lock_door_closing_time(self, vessel, operation_index, direction, operation_stop_time, prognosis=False):
        """
        Calculates the moment in time a new lock operation can start

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        operation_stop_time : pd.Timestamp
            the time that the lock operation has stopped (i.e., doors have been opened again)
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)

        Returns
        -------
        time_operation_stop : pd.Timestamp
            the moment in time a new lock operation can start
        """
        lock_doors_closing_time = self.calculate_lock_departure_stop_time(vessel, operation_index, direction, operation_stop_time, prognosis)
        lock_doors_closing_time += self.lock_chamber.minimum_delay_to_close_doors()
        return lock_doors_closing_time

    def determine_first_vessel_of_lock_operation(self, vessel, operation_index):
        """
        Determines the first vessel that was assigned to the lock operation

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            the index of the lock operation in the operation planning of the lock complex master
        """

        # set vessel as first vessel if no vessel has been assigned to the lock operation
        first_vessel = vessel

        # unpack the vessels of the specified lock operation
        vessels = self.get_vessel_from_planned_operation(operation_index=operation_index,)

        # determine the first vessel if vessels are already assigned to the lock operation
        if len(vessels):
            first_vessel = vessels[0]

        return first_vessel

    def determine_last_vessel_of_lock_operation(self, vessel, operation_index, prognosis=False):
        """
        Determines the last vessel that was assigned to the lock operation

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        prognosis : bool
            is for planning purposes or for actual events: True (yes) or False (no)

        Returns
        -------
        last_vessel : type
            the last assigned vessel of the lock operation (the one that will enter and leave the lock chamber last)
        """
        # identify the vessels assigned the lock operation
        vessels = self.get_vessel_from_planned_operation(operation_index=operation_index,)

        # determine the last vessel
        last_vessel = vessel
        if not prognosis:
            last_vessel = vessels[-1]

        return last_vessel


@inherit_docstring
class IsLockComplex(IsLockMaster):
    """Mixin-class: a lock complex object

    TODO: I would like the lock complex to be decoupled from its infrastructure, so that you can add multiple lock chambers, line-up areas and waiting areas


    Attributes:
    -----------
    _verify_node_AB :
        .
    create_time_distance_plot :
        .

    """

    def __init__(self,
                 node_A,                                        # a string with the node at which side A of the lock complex is located
                 node_B,                                        # a string with the node at which side B of the lock complex is located
                 edge_waiting_area_A = None,                    # a tuple with str that is the edge at which waiting area A is located
                 edge_waiting_area_B = None,                    # a tuple with str that is the edge at which waiting area B is located
                 distance_lock_doors_A_to_waiting_area_A=0.,    # a float that is the distance from lock doors A to waiting area A [m]
                 distance_lock_doors_B_to_waiting_area_B=0.,    # a float that is the distance from lock doors B to waiting area B [m]
                 lineup_area_A_length=None,                     # a float that is the actual length of line-up area A [m]
                 lineup_area_B_length=None,                     # a float that is the actual length of line-up area B [m]
                 distance_lock_doors_A_to_lineup_area_A=None,   # a float that is the distance from lock doors A to line-up area A [m]
                 distance_lock_doors_B_to_lineup_area_B=None,   # a float that is the distance from lock doors B to line-up area B [m]
                 effective_lineup_area_A_length=None,           # a float that is the effective length of line-up area A that can be requested by a vessel [m]
                 effective_lineup_area_B_length=None,           # a float that is the effective length of line-up area B that can be requested by a vessel [m]
                 passing_allowed_in_lineup_area_A=False,        # a bool to indicate that ... ?
                 passing_allowed_in_lineup_area_B=False,        # a bool to indicate that ... ?
                 speed_reduction_factor_lineup_area_A=0.75,     # a float that is the reduction factor for the vessel speed from its original speed when sailing towards the lock chamber from line-up area A
                 speed_reduction_factor_lineup_area_B=0.75,     # a float that is the reduction factor for the vessel speed from its original speed when sailing towards the lock chamber from line-up area B
                 P_used_to_break_before_lock=None,              # a float that is the breaking power used by the vessel to gradually decelerate in front of the lock [kW]
                 P_used_to_break_in_lock=None,                  # a float that is the breaking power used by the vessel to gradually decelerate inside the lock chamber [kW]
                 P_used_to_accelerate_in_lock=None,             # a float that is the acceleration power used by the vessel to gradually accelerate inside the lock chamber [kW]
                 P_used_to_accelerate_after_lock=None,          # a float that is the acceleration power used by the vessel to gradually accelerate to sail way from the lock chamber [kW]
                 k = 0,                                         # a int that is the identifier of the edge between two nodes at which the lock complex is located on the multidigraph network
                 *args,
                 **kwargs):
        """Initialization"""
        # TODO: we need to make an algorithm/utility that sets the infrastructure at the correct distances at the edge
        # set nodes
        self.node_A = node_A
        self.node_B = node_B
        self.start_node = node_A
        self.end_node = node_B
        self.edge = (node_A, node_B)

        # initialization
        super().__init__(lock_complex=self, *args, **kwargs)
        self.lock_chamber = IsLockChamber(lock_master=self, start_node=self.start_node, end_node=self.end_node, *args, **kwargs)

        # verify if nodes A and B are part of the graph, and have an edge between them
        self._verify_node_AB()

        # set distances between waiting area and lock doors
        self.distance_lock_doors_A_to_waiting_area_A = distance_lock_doors_A_to_waiting_area_A
        self.distance_lock_doors_B_to_waiting_area_B = distance_lock_doors_B_to_waiting_area_B

        # set power used to pass lock TODO: should maybe be added to the vessels
        self.P_used_to_break_before_lock = P_used_to_break_before_lock
        self.P_used_to_break_in_lock = P_used_to_break_in_lock
        self.P_used_to_accelerate_in_lock = P_used_to_accelerate_in_lock
        self.P_used_to_accelerate_after_lock = P_used_to_accelerate_after_lock
        self.k = k

        # create the waiting area objects
        if edge_waiting_area_A is None:
            edge_waiting_area_A = (self.start_node, self.end_node)

        self.distance_waiting_area_A_from_edge_start_waiting_area_A = self.distance_from_start_node_to_lock_doors_A - self.distance_lock_doors_A_to_waiting_area_A
        if edge_waiting_area_A != (node_A, node_B):
            geometry_edge_start_waiting_area_A_to_lock_node_A = self.env.vessel_traffic_service.provide_trajectory(edge_waiting_area_A[0], node_A)
            geometry_edge_start_waiting_area_A_to_lock_node_A_m = self.env.vessel_traffic_service.transform_geometry(geometry_edge_start_waiting_area_A_to_lock_node_A)
            self.distance_waiting_area_A_from_edge_start_waiting_area_A = geometry_edge_start_waiting_area_A_to_lock_node_A_m.length - self.distance_lock_doors_A_to_waiting_area_A

        self.waiting_area_A = IsLockWaitingArea(env=self.env,
                                                name="waiting_area_A",
                                                lock=self,
                                                edge=edge_waiting_area_A,
                                                distance_from_edge_start=self.distance_waiting_area_A_from_edge_start_waiting_area_A)
        self.distance_waiting_area_A_to_end_edge_waiting_area_A = get_length_of_edge(self.env.graph, edge_waiting_area_A)
        self.distance_waiting_area_A_to_end_edge_waiting_area_A -= self.distance_waiting_area_A_from_edge_start_waiting_area_A

        if edge_waiting_area_B is None:
            edge_waiting_area_B = (self.end_node, self.start_node)

        self.distance_waiting_area_B_from_start_edge_waiting_area_B = self.distance_from_end_node_to_lock_doors_B - self.distance_lock_doors_B_to_waiting_area_B
        if edge_waiting_area_B !=(node_B, node_A):
            geometry_edge_start_waiting_area_B_to_lock_node_B = self.env.vessel_traffic_service.provide_trajectory(edge_waiting_area_B[0],node_B)
            geometry_edge_start_waiting_area_B_to_lock_node_B_m = self.env.vessel_traffic_service.transform_geometry(geometry_edge_start_waiting_area_B_to_lock_node_B)
            self.distance_waiting_area_B_from_start_edge_waiting_area_B = geometry_edge_start_waiting_area_B_to_lock_node_B_m.length - self.distance_lock_doors_B_to_waiting_area_B

        self.waiting_area_B = IsLockWaitingArea(env=self.env,
                                                name="waiting_area_B",
                                                lock=self,
                                                edge=edge_waiting_area_B,
                                                distance_from_edge_start=self.distance_waiting_area_B_from_start_edge_waiting_area_B)
        self.distance_waiting_area_B_to_end_edge_waiting_area_B = get_length_of_edge(self.env.graph, edge_waiting_area_B)
        self.distance_waiting_area_B_to_end_edge_waiting_area_B -= self.distance_waiting_area_B_from_start_edge_waiting_area_B

        # create the line-up area at side A if there is a line-up area at side A (lineup_area_A_length is not None)
        self.has_lineup_area_A = False
        if lineup_area_A_length is not None:
            self.has_lineup_area_A = True
            self.lineup_area_A_length = lineup_area_A_length
            self.effective_lineup_area_A_length = effective_lineup_area_A_length
            self.passing_allowed_in_lineup_area_A = passing_allowed_in_lineup_area_A
            self.speed_reduction_factor_lineup_area_A = speed_reduction_factor_lineup_area_A

            # the effective line-up length should at least be equal to the lock length TODO: set warning?
            if lineup_area_A_length < self.lock_length and not effective_lineup_area_A_length:
                self.effective_lineup_area_A_length = self.lock_length

            self.distance_lock_doors_A_to_lineup_area_A = distance_lock_doors_A_to_lineup_area_A

            # get the edge at which the line-up area is located TODO: can a lock complex be located along multiple edges?
            distance_from_start_node_to_lineup_A = self.distance_lock_doors_A_to_lineup_area_A - self.distance_from_start_node_to_lock_doors_A
            edge_lineup_area_A = self.env.vessel_traffic_service.provide_edge_by_distance_from_node(self.env,
                                                                                                    self.start_node,
                                                                                                    self.node_A,
                                                                                                    distance_from_start_node_to_lineup_A)

            route_to_lineup_area_A = nx.dijkstra_path(self.env.graph, self.start_node, edge_lineup_area_A[1]) # TODO: can a lock complex be located along multiple edges?
            distance_start_node_to_node_waiting_area_A = self.env.vessel_traffic_service.provide_sailing_distance_over_route(route_to_lineup_area_A)["Distance"].sum()
            self.distance_lineup_area_A_from_edge_lineup_area_A_start = distance_start_node_to_node_waiting_area_A - (self.distance_lock_doors_A_to_lineup_area_A - self.distance_from_start_node_to_lock_doors_A)

            # create lineup area A object
            self.lineup_area_A = IsLockLineUpArea(env=self.env,
                                                  name=self.name,
                                                  start_node=edge_lineup_area_A[1],
                                                  end_node=edge_lineup_area_A[0],
                                                  lineup_area_length=self.lineup_area_A_length,
                                                  distance_from_start_edge=self.distance_lineup_area_A_from_edge_lineup_area_A_start,
                                                  effective_lineup_area_length=self.effective_lineup_area_A_length,
                                                  passing_allowed=self.passing_allowed_in_lineup_area_A,
                                                  speed_reduction_factor=self.speed_reduction_factor_lineup_area_A)

        # create the line-up area at side B if there is a line-up area at side B (lineup_area_B_length is not None)
        self.has_lineup_area_B = False
        if lineup_area_B_length is not None:
            self.has_lineup_area_B = True
            self.lineup_area_B_length = lineup_area_B_length
            self.effective_lineup_area_B_length = effective_lineup_area_B_length
            self.passing_allowed_in_lineup_area_B = passing_allowed_in_lineup_area_B
            self.speed_reduction_factor_lineup_area_B = speed_reduction_factor_lineup_area_B

            # the effective line-up length should at least be equal to the lock length TODO: set warning?
            if lineup_area_B_length < self.lock_length and not effective_lineup_area_B_length:
                self.effective_lineup_area_B_length = self.lock_length

            self.distance_lock_doors_B_to_lineup_area_B = distance_lock_doors_B_to_lineup_area_B

            # get the edge at which the line-up area is located TODO: can a lock complex be located along multiple edges?
            distance_from_end_node_to_lineup_B = self.distance_lock_doors_B_to_lineup_area_B - self.distance_from_end_node_to_lock_doors_B
            edge_lineup_area_B = self.env.vessel_traffic_service.provide_edge_by_distance_from_node(self.env,
                                                                                                    self.end_node,
                                                                                                    self.node_B,
                                                                                                    distance_from_end_node_to_lineup_B)

            route_to_lineup_area_B = nx.dijkstra_path(self.env.graph, self.end_node, edge_lineup_area_B[1]) #TODO: can a lock complex be located along multiple edges?
            distance_end_node_to_node_waiting_area_B = self.env.vessel_traffic_service.provide_sailing_distance_over_route(route_to_lineup_area_B)["Distance"].sum()
            self.distance_lineup_area_B_from_edge_lineup_area_B_start = distance_end_node_to_node_waiting_area_B - (self.distance_lock_doors_B_to_lineup_area_B - self.distance_from_end_node_to_lock_doors_B)

            # create lineup area B object
            self.lineup_area_B = IsLockLineUpArea(env=self.env,
                                                  name=self.name,
                                                  start_node=edge_lineup_area_B[1],
                                                  end_node=edge_lineup_area_B[0],
                                                  distance_from_start_edge=self.distance_lineup_area_B_from_edge_lineup_area_B_start,
                                                  lineup_area_length=self.lineup_area_B_length,
                                                  effective_lineup_area_length=self.effective_lineup_area_B_length,
                                                  passing_allowed=self.passing_allowed_in_lineup_area_B,
                                                  speed_reduction_factor=self.speed_reduction_factor_lineup_area_B)

    # properties from lock chamber. Should be adapted to functions based on which lock we take
    @property
    def distance_from_start_node_to_lock_doors_A(self):
        """
        Get distance from start node to lock doors A through lock chamber.
        """
        return self.lock_chamber.distance_from_start_node_to_lock_doors_A

    @property
    def distance_from_end_node_to_lock_doors_B(self):
        """
        Get distance from end node to lock doors B through lock chamber.
        """
        return self.lock_chamber.distance_from_end_node_to_lock_doors_B

    @property
    def registration_nodes(self):
        """
        Get registration nodes of the lock chamber.
        """
        return self.lock_chamber.registration_nodes

    @property
    def sailing_distance_to_crossing_point(self):
        """
        Get sailing distance to crossing point through lock chamber.
        """
        return self.lock_chamber.sailing_distance_to_crossing_point

    def vessel_sailing_in_speed(self, vessel, direction):
        """
        Get vessel sailing in speed through lock chamber.

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex,  Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        """
        return self.lock_chamber.vessel_sailing_in_speed(vessel, direction)

    @property
    def location_lock_doors_A(self):
        """
        Get location of lock doors A through lock chamber.
        """
        return self.lock_chamber.location_lock_doors_A

    @property
    def location_lock_doors_B(self):
        """
        Get location of lock doors B through lock chamber.
        """
        return self.lock_chamber.location_lock_doors_B

    @property
    def node_open(self):
        """
        Get node open status through lock chamber.
        """
        return self.lock_chamber.node_open

    @property
    def name(self):
        """
        Get name of the lock chamber.
        """
        return self.lock_chamber.name

    @property
    def lock_length(self):
        """
        Get length of the lock chamber.
        """
        return self.lock_chamber.lock_length

    @property
    def lock_width(self):
        """
        Get width of the lock chamber.
        """
        return self.lock_chamber.lock_width

    @property
    def doors_closing_time(self):
        """
        Get doors closing time through lock chamber.
        """
        return self.lock_chamber.doors_closing_time

    @property
    def doors_opening_time(self):
        """
        Get doors opening time through lock chamber.
        """
        return self.lock_chamber.doors_opening_time

    @property
    def start_sailing_out_time_after_doors_have_been_opened(self):
        """
        Get start sailing out time after doors have been opened through lock chamber.
        """
        return self.lock_chamber.start_sailing_out_time_after_doors_have_been_opened

    @property
    def sailing_in_time_gap_through_doors(self):
        """
        Get sailing in time gap through doors.
        """
        return self.lock_chamber.sailing_in_time_gap_through_doors

    @property
    def sailing_out_time_gap_through_doors(self):
        """
        Get sailing out time gap through doors.
        """
        return self.lock_chamber.sailing_out_time_gap_through_doors

    @property
    def sailing_in_time_gap_after_berthing_previous_vessel(self):
        """
        Get sailing in time gap after berthing previous vessel through lock chamber.
        """
        return self.lock_chamber.sailing_in_time_gap_after_berthing_previous_vessel

    @property
    def sailing_out_time_gap_after_berthing_previous_vessel(self):
        """
        Get sailing out time gap after berthing previous vessel through lock chamber.
        """
        return self.lock_chamber.sailing_out_time_gap_after_berthing_previous_vessel

    def vessel_sailing_out_speed(self, vessel, direction, P_used=None, h0=17, until_crossing_point=False):
        """
        Get vessel sailing out speed through lock chamber.
        """
        return self.lock_chamber.vessel_sailing_out_speed(vessel, direction, P_used, h0, until_crossing_point)

    def vessel_sailing_speed_out_lock(self, vessel):
        """
        Get vessel sailing speed out of lock through lock chamber.
        """
        return self.lock_chamber.vessel_sailing_speed_out_lock(vessel)

    def vessel_sailing_speed_in_lock(self, vessel):
        """
        Get vessel sailing speed in lock through lock chamber.
        """
        return self.lock_chamber.vessel_sailing_speed_in_lock(vessel)

    def determine_levelling_time(self, t_start, direction, wlev_init=None, operation_index=0, prediction=False):
        """
        Determine levelling time through lock chamber.
        """
        return self.lock_chamber.determine_levelling_time(t_start, direction, wlev_init, operation_index, prediction)

    def _verify_node_AB(self):
        """Function to verify if nodes A and B are part of the graph, and have an edge between them."""
        if self.node_A not in self.env.graph.nodes or self.node_B not in self.env.graph.nodes:
            raise ValueError(
                f"LockComplex {self.name} has invalid node_A {self.node_A} or node_B {self.node_B} which are not part of the graph."
            )
        if not self.env.graph.has_edge(self.node_A, self.node_B):
            raise ValueError(
                f"LockComplex {self.name} does not have an edge between node A {self.node_A} and node B {self.node_B}."
            )

    def create_time_distance_plot(self, vessels, xlimmin=None, xlimmax=None, ylimmin=None, ylimmax=None, method = 'Matplotlib'):
        """Create a time-distance plot of vessels passing a lock complex

        Parameters
        ----------
        vessels: list of vessel type objects
            the vessels that have been simulated (a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput)
        xlimmin : float
            minimum x coordinate as distance front the lock complex (should be negative) [m]
        xlimmax : float
            maximum x coordinate as distance front the lock complex (should be positive) [m]
        ylimmin : pd.Timestamp
            minimum time (should be equal or greater that the simulation start time)
        ylimmax : pd.Timestamp
            maximum time (should be equal or smaller that the simulation stop time)

        Returns
        -------
        nothing, but creates a plot

        """

        # create lock edge geometry in [m]
        route_between_nodes_of_registration = nx.dijkstra_path(self.env.graph, self.registration_nodes[0], self.registration_nodes[1])
        lock_edge_geometry = self.env.vessel_traffic_service.provide_trajectory(route_between_nodes_of_registration[0],route_between_nodes_of_registration[-1])
        lock_edge_geometry_m = self.env.vessel_traffic_service.transform_geometry(lock_edge_geometry)

        # plot the lock geometry over time
        location_lock_doors_A_m = self.env.vessel_traffic_service.transform_geometry(self.location_lock_doors_A)
        location_lock_doors_B_m = self.env.vessel_traffic_service.transform_geometry(self.location_lock_doors_B)
        x_lock_doorsA = (lock_edge_geometry_m.line_locate_point(location_lock_doors_A_m))
        x_lock_doorsB = (lock_edge_geometry_m.line_locate_point(location_lock_doors_B_m))
        x_correction_indirection = x_lock_doorsA + self.lock_length/2
        x_correction_outdirection = x_lock_doorsB - self.lock_length / 2

        # determine the accepted messages for plotting
        accepted_messages = []
        for node_start, node_end in zip(route_between_nodes_of_registration[:-1],route_between_nodes_of_registration[1:]):
            accepted_messages.extend([f"Sailing from node {node_start} to node {node_end} start",
                                      f"Sailing from node {node_end} to node {node_start} start",
                                      f"Sailing from node {node_start} to node {node_end} stop",
                                      f"Sailing from node {node_end} to node {node_start} stop"])

        accepted_messages.extend(["Waiting for other vessel in lock operation start",
                                  "Waiting for other vessel in lock operation stop",
                                  "Waiting for lock operation start",
                                  "Waiting for lock operation stop",
                                  "Sailing to first lock doors start",
                                  "Sailing to first lock doors stop",
                                  "Sailing to position in lock start",
                                  "Sailing to position in lock stop",
                                  "Levelling start",
                                  "Levelling stop",
                                  "Sailing to second lock doors start",
                                  "Sailing to second lock doors stop",
                                  "Sailing to lock complex exit start",
                                  "Sailing to lock complex exit stop"])

        # loop over vessels to extract time and distance from lock passage messages and store them in a list
        all_times = []
        all_distances = []
        traces = []
        for vessel in vessels:
            times = []
            distances = []
            vessel_df = pd.DataFrame(vessel.logbook)
            vessel_df["Geometry"] = vessel_df["Geometry"].apply(lambda x: self.env.vessel_traffic_service.transform_geometry(x))
            x_correction = 0.0
            for index, message_info in vessel_df.iterrows():
                time = message_info.Timestamp
                distance = lock_edge_geometry_m.line_locate_point(message_info.Geometry)
                route = vessel.route
                if self.start_node not in route or self.end_node not in route:
                    continue

                if message_info.Message in accepted_messages:
                    if message_info.Message == f"Sailing from node {self.start_node} to node {self.end_node} start":
                        x_correction = x_correction_indirection
                    elif message_info.Message == f"Sailing from node {self.end_node} to node {self.start_node} start":
                        x_correction = x_correction_outdirection
                    times.append(time)
                    distances.append(distance)

            distances = np.array(distances) - x_correction
            all_times.append(times)
            all_distances.append(distances)

            # Add vessel trace with vessel.name in legend
            if method == 'Plotly':
                traces.append(go.Scatter(x=distances, y=times, mode='lines', name=vessel.name))

        if method == 'Matplotlib':
            fig, ax = plt.subplots()
            for distances, times in zip(all_distances, all_times):
                ax.plot(distances, times)
        elif method == 'Plotly':
            fig = go.Figure(data=traces)

        # Determine y-axis limits
        all_y_values = [t for sublist in all_times for t in sublist]
        if all_y_values:
            if ylimmin is None:
                ylimmin = min(all_y_values)
            if ylimmax is None:
                ylimmax = max(all_y_values)

        # Determine x-axis limits
        sailing_distance_to_crossing_point = self.sailing_distance_to_crossing_point + self.lock_length / 2
        if xlimmin is None:
            xlimmin = -2 * sailing_distance_to_crossing_point
        if xlimmax is None:
            xlimmax = 2 * sailing_distance_to_crossing_point

        if method == 'Matplotlib':
            lock_extend_x = np.array([x_lock_doorsA, x_lock_doorsA, x_lock_doorsB, x_lock_doorsB]) - x_correction_indirection
            ax.fill(lock_extend_x, [ylimmin, ylimmax, ylimmax, ylimmin], color="lightgrey", zorder=0)
        elif method == 'Plotly':
            fig.add_shape(type="rect",
                          x0=x_lock_doorsA - x_correction_indirection, x1=x_lock_doorsB - x_correction_indirection,
                          y0=ylimmin, y1=ylimmax,
                          fillcolor="lightgrey", opacity=0.5,
                          layer="below", line_width=0,
                          name="Lock Geometry")

        # plot the lock phases
        lock_df = pd.DataFrame(self.lock_chamber.logbook)
        for index, message_info in lock_df.iterrows():
            message_found = False
            if message_info.Message == "Lock doors opening stop" and index != 0:
                time_start = lock_df.loc[index - 1, "Timestamp"]
                time_stop = message_info.Timestamp
                color = "darkgrey"
                name = "Lock doors opening"
                message_found = True
            if message_info.Message == "Lock doors closing stop" and index != 0:
                time_start = lock_df.loc[index - 1, "Timestamp"]
                time_stop = message_info.Timestamp
                color = "darkgrey"
                name = "Lock doors closing"
                message_found = True
            if message_info.Message == "Lock chamber converting stop" and index != 0:
                time_start = lock_df.loc[index - 1, "Timestamp"]
                time_stop = message_info.Timestamp
                color = "grey"
                name = "Lock chamber converting"
                message_found = True

            if method == 'Matplotlib' and message_found:
                ax.fill(lock_extend_x, [time_start, time_stop, time_stop, time_start], color=color, zorder=0)
            elif method == 'Plotly' and message_found:
                fig.add_shape(type="rect",
                              x0=x_lock_doorsA - x_correction_indirection, x1=x_lock_doorsB - x_correction_indirection,
                              y0=time_start, y1=time_stop,
                              fillcolor=color, opacity=0.5,
                              layer="below", line_width=0,
                              name=name)

        # plot the approach points
        sailing_distance_to_crossing_point = self.sailing_distance_to_crossing_point + self.lock_length / 2
        xlabel = "Distance from Lock Complex [m]"
        ylabel = "Timestamp"
        title = "Time-Distance Plot of Vessel Movements"
        if method == 'Matplotlib':
            ax.axvline(-sailing_distance_to_crossing_point, color="lightgrey", zorder=0)
            ax.axvline(sailing_distance_to_crossing_point, color="lightgrey", zorder=0)
            ax.set_xlim([xlimmin,xlimmax])
            ax.set_ylim([ylimmin,ylimmax])
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        elif method == 'Plotly':
            fig.add_vline(x=-sailing_distance_to_crossing_point, line=dict(color="lightgrey"))
            fig.add_vline(x=sailing_distance_to_crossing_point, line=dict(color="lightgrey"))
            fig.update_layout(title=title,
                              xaxis_title=xlabel,
                              yaxis_title=ylabel,
                              xaxis_range=[xlimmin, xlimmax],
                              yaxis_range=[ylimmin, ylimmax],
                              showlegend=True)

        return fig
