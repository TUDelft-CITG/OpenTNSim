def add_vessel_to_planned_lock_operation(self, vessel, operation_index, direction):
    """
    Add vessel to a planned lock operation

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex
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
    vessel_planning = self.vessel_planning
    operation_planning = self.operation_planning

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
        # operation_planning.loc[operation_index, "vessels"] = (vessels_in_operation)
        # calculate_sailing_time_to_approach_point(self, vessel, direction, operation_index=operation_index)  # TODO: can this be removed?

        # if there is a rule that prescribes a minimum amount of vessels in the lock operation and this condition is satisfied, put an operation-object in the FilterStore to communicate that the earlier waiting vessels do not have to wait any longer
        # if self.min_vessels_in_operation and len(vessels_in_operation) == self.min_vessels_in_operation:
        #     Operation = namedtuple("Operation", "operation_index")
        #     operation = Operation(operation_index)
        #     yield self.wait_for_other_vessel_to_arrive.put(operation)
        #
        #     # calculate the required sailing in time delay
        #     sailing_in_gap = calculate_sailing_in_time_delay(self, vessel, operation_index, direction, prognosis=False, overwrite=False)

    # calculate the new arrival time at the lock entry
    time_arrival_time_at_lock_entry = vessel_planning.loc[
                                          vessel_planning_index, "time_lock_passing_start"] + sailing_in_gap

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
    if time_first_vessel_required_to_be_at_lock_approach > operation_planning.loc[
        operation_index, "time_operation_start"] and not len(other_vessels_in_operation):
        time_lock_operation_start = time_first_vessel_required_to_be_at_lock_approach

    # add to vessel entry delay if the time of starting the approach lies ahead of the operation start time
    elif time_first_vessel_required_to_be_at_lock_approach < operation_planning.loc[
        operation_index, "time_operation_start"]:
        vessel_entry_delay += (operation_planning.loc[
                                   operation_index, "time_operation_start"] - time_first_vessel_required_to_be_at_lock_approach)

    if not len(other_vessels_in_operation) and time_lock_operation_start < operation_planning.loc[
        operation_index - 1, "time_operation_stop"]:
        vessel_entry_delay += operation_planning.loc[
                                  operation_index - 1, "time_operation_stop"] - time_lock_operation_start

    # add the delay to the expected time of lock entry to the vessel
    if vessel_entry_delay > pd.Timedelta(seconds=0):
        time_arrival_time_at_lock_entry += vessel_entry_delay

    # update the vessel planning based on the above delays
    time_vessel_entry_start = calculate_vessel_entry_duration(self, vessel, direction) + time_arrival_time_at_lock_entry
    time_lock_entry_stop = calculate_lock_entry_stop_time(self, vessel, operation_index, direction,
                                                          time_arrival_time_at_lock_entry)
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
        potential_lock_door_opening_stop = operation_planning.loc[
            operation_index, "time_potential_lock_door_opening_stop"]
        time_lock_operation_start = operation_planning.loc[operation_index, "time_operation_start"]
        time_entry_start += operation_start_delay

    # if there is a delay in the start op the operation: update the vessel planning of the previous arriving vessels of this operation
    self._process_delay_in_vessel_planning(operation_start_delay, other_vessels_in_operation)

    # determine the times of door closing, levelling and door opening: if lock entry stop time or extract them when the new lock entry stop time is ahead of the door closing start time TODO: check if this is correct
    levelling_information = calculate_lock_operation_times(self.lock_complex.lock_chamber,
                                                           operation_index=operation_index,
                                                           last_entering_time=time_vessel_entry_start,
                                                           start_time=time_lock_entry_stop,
                                                           vessel=vessel,
                                                           direction=direction, )

    # determine water levels to be included in the planning
    wlev_A, wlev_B = levelling_information["wlev_A"], levelling_information["wlev_B"]

    # if there is a delay in the departure of the vessels, also include that in the planning
    additional_sailing_out_delay = levelling_information["time_door_opening_stop"] - operation_planning.loc[
        operation_index, "time_door_opening_stop"]
    if additional_sailing_out_delay > pd.Timedelta(seconds=0):
        for other_vessel in other_vessels_in_operation:
            other_vessel_planning_index = vessel_planning[vessel_planning.id == other_vessel.id].iloc[-1].name
            vessel_planning.loc[
                other_vessel_planning_index, "time_lock_departure_start"] += additional_sailing_out_delay
            vessel_planning.loc[other_vessel_planning_index, "time_lock_departure_stop"] += additional_sailing_out_delay
            vessel_planning.loc[other_vessel_planning_index, "time_lock_passing_stop"] += additional_sailing_out_delay
            vessel_planning.loc[other_vessel_planning_index, "delay"] += additional_sailing_out_delay

    # update the operation planning with the above information
    operation_planning.loc[operation_index, "time_potential_lock_door_opening_stop"] = potential_lock_door_opening_stop
    operation_planning.loc[operation_index, "time_operation_start"] = time_lock_operation_start
    operation_planning.loc[operation_index, "time_entry_start"] = time_entry_start
    operation_planning.loc[operation_index, "time_entry_stop"] = time_lock_entry_stop
    operation_planning.loc[operation_index, "time_door_closing_start"] = levelling_information[
        "time_door_closing_start"]
    operation_planning.loc[operation_index, "time_door_closing_stop"] = levelling_information["time_door_closing_stop"]
    operation_planning.loc[operation_index, "time_levelling_start"] = levelling_information["time_levelling_start"]
    operation_planning.loc[operation_index, "time_levelling_stop"] = levelling_information["time_levelling_stop"]
    operation_planning.loc[operation_index, "time_door_opening_start"] = levelling_information[
        "time_door_opening_start"]
    operation_planning.loc[operation_index, "time_door_opening_stop"] = levelling_information["time_door_opening_stop"]
    operation_planning.loc[operation_index, "maximum_individual_delay"] = np.max(
        vessel_planning[vessel_planning.operation_index == operation_index].delay)
    operation_planning.loc[operation_index, "total_delay"] = np.sum(
        vessel_planning[vessel_planning.operation_index == operation_index].delay)

    # determine the new departure and operation start and stop times
    lock_departure_information = calculate_lock_departure_information(self, vessel, operation_index, direction,
                                                                      levelling_information)
    if self.close_doors_before_vessel_is_laying_still:
        time_potential_lock_door_closure_start = time_vessel_entry_start + self.lock_chamber.minimum_delay_to_close_doors()
    else:
        time_potential_lock_door_closure_start = levelling_information["time_door_closing_start"]

    # update vessel and operation plannings accordingly
    lock_operation_information = {"time_departure_start": lock_departure_information["time_lock_departure_start"],
                                  "time_departure_stop": lock_departure_information["time_lock_departure_stop"],
                                  "time_operation_stop": lock_departure_information["time_lock_operation_stop"],
                                  "time_potential_lock_door_closure_start": lock_departure_information[
                                      "time_lock_door_closing_start"],
                                  "wlev_A": wlev_A,
                                  "wlev_B": wlev_B}
    _update_lock_operation_planning(self, operation_index, lock_operation_information)

    vessel_passage_information = {
        "time_potential_lock_door_closure_start": time_potential_lock_door_closure_start,
        "time_potential_lock_door_opening_stop": (
                    time_vessel_entry_start - self.lock_chamber.minimum_advance_to_open_doors()),
        "time_lock_departure_start": lock_departure_information["time_vessel_departure_start"],
        "time_lock_departure_stop": lock_departure_information["time_vessel_departure_stop"],
        "time_lock_passing_stop": lock_departure_information["time_vessel_passing_stop"],
    }
    _update_lock_vessel_planning(self, vessel_planning_index, vessel_passage_information)

    # update the next lock operations if the previous lock operation caused a delay
    self._update_future_lock_operations_by_lock_delay_previous_operation(operation_index, lock_departure_information)
    return operation_planning


def add_vessel_to_new_lock_operation(self, vessel, operation_index, direction):
    """
    Adds a vessel to a newly to be planned lock operation

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex
    operation_index : int
        index of the lock operation
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    """
    # unpack the lock master's vessel and lock operation plannings
    vessel_planning = self.vessel_planning
    operation_planning = self.operation_planning
    node_of_approach, to_node = _get_lock_operation_to_and_from_node(self, direction)

    # determine if the new lock operation should follow a empty lock operation (when the new lock operation has the same direction as the previous lock operation)
    operation_index, empty_lock_operation_to_be_requested = self.check_if_empty_lock_operation_is_required(
        operation_index, direction)
    if empty_lock_operation_to_be_requested:
        self.request_empty_levelling(direction)

    # determine the index of the vessel in the lock master's vessel planning
    vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

    # add operation to the planning with information
    vessel_passage_information = {"operation_index": operation_index}
    _update_lock_vessel_planning(self, vessel_planning_index, vessel_passage_information)

    lock_operation_information = {"node_from": node_of_approach,
                                  "node_to": to_node,
                                  "direction": direction,
                                  "lock_chamber": self.lock_complex.name,
                                  "vessels": [],  # leave vessels empty for now
                                  "capacity_L": self.lock_complex.lock_length - vessel.L,
                                  "capacity_B": self.lock_complex.lock_width - vessel.B, }
    _update_lock_operation_planning(self, operation_index, lock_operation_information)

    (time_lock_operation_start,
     time_lock_entry_start,
     minimum_advance_to_open_doors,
     time_potential_lock_door_opening_stop,
     time_lock_entry_stop,
     time_lock_door_opening_stop,
     vessel_entry_delay) = calculate_lock_operation_start_information(self, vessel, operation_index, direction)

    # determine the moments in time of the lock operation process steps starts and stops
    levelling_information = calculate_lock_operation_times(self.lock_complex.lock_chamber,
                                                           operation_index=operation_index,
                                                           last_entering_time=time_lock_entry_start,
                                                           start_time=time_lock_entry_stop,
                                                           vessel=vessel,
                                                           direction=direction)

    # determine the water levels and set the list of vessels
    wlev_A, wlev_B = levelling_information["wlev_A"], levelling_information["wlev_B"]

    # determine the moments in time of the vessel's departure from the lock (steps starts and stops) and the time the operation has stopped and the doors can close again
    time_lock_departure_start = calculate_lock_departure_start_time(self, vessel, operation_index, direction,
                                                                    levelling_information["time_door_opening_stop"],
                                                                    prognosis=True)
    time_lock_departure_stop = calculate_lock_departure_stop_time(self, vessel, operation_index, direction,
                                                                  levelling_information["time_door_opening_stop"],
                                                                  prognosis=True)
    time_lock_operation_stop = calculate_lock_operation_stop_time(self, vessel, operation_index, direction,
                                                                  levelling_information["time_door_opening_stop"],
                                                                  prognosis=True)
    time_lock_door_closing_start = calculate_lock_door_closing_time(self, vessel, operation_index, direction,
                                                                    levelling_information["time_door_opening_stop"],
                                                                    prognosis=True)

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
    vessel_passage_information = {"time_potential_lock_door_opening_stop": time_potential_lock_door_opening_stop,
                                  "time_lock_passing_start": time_lock_operation_start,
                                  "time_lock_entry_start": time_lock_entry_start,
                                  "time_lock_entry_stop": time_lock_entry_stop,
                                  "time_potential_lock_door_closure_start": time_potential_lock_door_closure_start,
                                  "time_lock_departure_start": time_lock_departure_start,
                                  "time_lock_departure_stop": time_lock_departure_stop,
                                  "time_lock_passing_stop": time_lock_operation_stop,
                                  "delay": delay}
    _update_lock_vessel_planning(self, vessel_planning_index, vessel_passage_information)

    lock_operation_information = {"time_operation_start": time_lock_operation_start,
                                  "time_potential_lock_door_opening_stop": time_lock_door_opening_stop,
                                  "time_entry_start": time_lock_entry_start,
                                  "time_entry_stop": time_lock_entry_stop,
                                  "vessels": vessels,
                                  "time_door_closing_start": levelling_information["time_door_closing_start"],
                                  "time_door_closing_stop": levelling_information["time_door_closing_stop"],
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


def add_vessel_to_lock_operation(self, vessel, direction):
    """
    Adds the vessel the lock master's operation planning

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex
    direction : int
        the direction of the lock operation: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)

    Yields
    -------
    Adds vessel to new or planned lock operation

    """
    # unpack the lock master's vessel and lock operation plannings
    operation_planning = self.lock_complex.operation_planning
    vessel_planning = self.lock_complex.vessel_planning

    # add vessel to vessel planning and operation planning
    vessel_planning_index = self.add_vessel_to_vessel_planning(vessel, direction)
    operation_index, add_operation, available_operations = self.find_available_lock_operation(vessel, direction)

    # add vessel to a new lock operation or to a planned one
    if operation_planning.empty or add_operation:
        yield from self.add_vessel_to_new_lock_operation(vessel, operation_index, direction)
    else:
        yield from self.add_vessel_to_planned_lock_operation(vessel, operation_index, direction)

    operation_index = vessel_planning.loc[vessel_planning_index, "operation_index"]
    operation_planning.loc[operation_index, "maximum_individual_delay"] = np.max(vessel_planning[vessel_planning.operation_index == operation_index].delay)
    operation_planning.loc[operation_index, "total_delay"] = np.sum(vessel_planning[vessel_planning.operation_index == operation_index].delay)
    return operation_index