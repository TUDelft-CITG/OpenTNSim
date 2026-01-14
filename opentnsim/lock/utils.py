"""This module contains utility functions for lock operations in the OpenTNSim simulation environment."""
import pandas as pd
import datetime
import networkx as nx
import numpy as np
from numpy.testing import assert_almost_equal
from opentnsim.graph.utils import get_length_of_edge, get_edge, check_graph_is_multidigraph_type, get_sailing_information_on_edge_to_distance_on_another_edge
from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from IPython.display import display

def _get_lock_operation_to_and_from_node(lock, direction):
    """Get the nodes from and to which the lock operation is directed based on the direction convention

    Convention: direction = 0 (when A -> B), direction = 1 (when B -> A)
        with A = lock.start_node and B = lock.end_node

    Parameters
    ----------
    lock : object
        the lock object generated with IsLockComplex
    direction : int
        the direction of the lock operation: 0 or 1 (see above convention)

    Returns
    -------
    node_of_approach : str
        the name of the node from which the lock operation is directed
    to_node : str
        the name of the node to which the lock operation is directed
    """
    node_of_approach = lock.end_node
    to_node = lock.start_node
    if not direction:
        node_of_approach = lock.start_node
        to_node = lock.end_node
    return node_of_approach, to_node


def _get_lock_object_on_registration_node(graph, registration_node):
    """Get the lock complex object that is associated with a registration node node

    Parameters
    ----------
    multidigraph : nx.MultiDiGraph
        the graph of the simulation as MultiDiGraph-version (to allow for parallel locks between the same node pair)
    registration_node : str
        node name (that has to be in the graph) on which the vessel is currently starting to navigate an edge

    Returns
    -------
    lock : Union(class, None)
        the lock complex object that is associated with the registration node, or None if no lock complex is associated with the registration node
    """
    # check if node is a registration node
    if "Lock_registration_node" not in graph.nodes[registration_node].keys():
        return []

    lock_complexes = graph.nodes[registration_node]["Lock_registration_node"]
    return lock_complexes


def _get_vessels_from_planned_operation(lock_complex, operation_index = None):
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
    if operation_index is None:
        return vessels

    # determines the vessels in the lock operation
    selected_operation = lock_complex.operation_planning[lock_complex.operation_planning.index == operation_index]
    if not selected_operation.empty:
        vessels = selected_operation.loc[operation_index, "vessels"].copy()
    return vessels


def _update_lock_operation_planning(lock, operation_index, operation_information):
    """Updates the lock operation planning

    Parameters
    ----------
    lock : object
        the lock object generated with IsLockComplex
    operation_index : int
            index of the lock operation
    operation_information : dict
        information to be added to the dataframe with keys as column names and values
    """
    for key, value in operation_information.items():
        if key not in lock.operation_planning.columns:
            #warnings.warn(f"Column name ({key}) not in the operation planning dataframe -> skipped.")
            continue
        lock.operation_planning.loc[int(operation_index), key] = value


def _update_lock_vessel_planning(lock, vessel_index, passage_information):
    """Updates the lock vessel planning

    Parameters
    ----------
    lock : object
        the lock object generated with IsLockComplex
    operation_index : int
            index of the lock operation
    passage_information : dict
        information to be added to the dataframe with keys as column names and values
    """
    for key, value in passage_information.items():
        if key not in lock.vessel_planning.columns:
            #warnings.warn(f"Column name ({key}) not in the vessel planning dataframe -> skipped.")
            continue
        lock.vessel_planning.loc[vessel_index, key] = value


def _find_available_waiting_area(vessel, lock_chamber, direction):
    lock_end_node = lock_chamber.end_node
    distance_to_lock_on_edge = lock_chamber.distance_from_start_node_to_lock_gate_A
    if direction:
        lock_end_node = lock_chamber.start_node
        distance_to_lock_on_edge = lock_chamber.distance_from_end_node_to_lock_gate_B
    routes = nx.all_simple_paths(vessel.env.graph, vessel.current_node, lock_end_node)
    suitable_waiting_areas = pd.DataFrame(columns=['sailing_time_waiting_area_to_lock','available'])
    for route in routes:
        for edge in zip(route[:-1],route[1:]):
            if 'Waiting area' not in vessel.env.graph.edges[edge].keys():
                continue
            waiting_areas = vessel.env.graph.edges[edge]['Waiting area']
            for waiting_area in waiting_areas:
                distance_to_waiting_area_on_edge = waiting_area.distance_from_edge_start
                get_sailing_info = get_sailing_information_on_edge_to_distance_on_another_edge
                route_to_lock = nx.dijkstra_path(vessel.env.graph, waiting_area.edge[0], lock_end_node)
                sailing_info = get_sailing_info(vessel, route_to_lock, distance_to_waiting_area_on_edge, distance_to_lock_on_edge)
                sailing_time = pd.Timedelta(seconds=sailing_info.time.sum())
                available = waiting_area.resource.capacity > len(waiting_area.resource.users)
                suitable_waiting_areas.loc[waiting_area.name,:] = [sailing_time, available]

    available_waiting_areas = suitable_waiting_areas[suitable_waiting_areas.available]
    print(available_waiting_areas)
    waiting_area = None
    if not available_waiting_areas.empty:
        waiting_area = available_waiting_areas.sort_values('sailing_time_waiting_area_to_lock').iloc[0].name

    if waiting_area is None:
        raise ValueError(f"No route found to waiting area")

    return waiting_area


def _get_lock_operation_direction(lock, to_node):
    """Get the direction of the lock based on the node to which the lock operation is directed

    Convention: direction = 0 (when A -> B), direction = 1 (when B -> A)
        with A = lock.start_node and B = lock.end_node

    Parameters
    ----------
    lock : object
        the lock object generated with IsLockComplex
    to_node : str
        the name of the node to which the lock operation is directed

    Returns
    -------
    direction: int
        the direction of the lock operation: 0 or 1 (see above convention)
    """
    direction = 0
    if to_node == lock.start_node:
        direction = 1

    return direction


def _get_previous_assigned_vessel(lock_complex, operation_index):
    operation_planning = lock_complex.operation_planning
    assigned_operation = operation_planning.loc[operation_index]
    if len(assigned_operation.vessels) == 1:
        return None
    previous_vessel = assigned_operation.vessels[-2]
    return previous_vessel


def _get_waiting_area(lock_complex, direction):
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
        waiting_area = lock_complex.waiting_area_A
    else:
        waiting_area = lock_complex.waiting_area_B

    return waiting_area


def _get_lineup_area(lock_complex, direction):
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
        try:
            lineup_area = lock_complex.lineup_area_A
        except:
            lineup_area = None
    else:
        try:
            lineup_area = lock_complex.lineup_area_B
        except:
            lineup_area = None

    return lineup_area


def _get_distance_to_lock(lock, direction):
    """get the distance from the start node of the lock to the lock gate from the perspective of the vessel

    Parameters
    ----------
    direction : int
        the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    """
    if not direction:
        return lock.distance_from_start_node_to_lock_gate_A
    else:
        return lock.distance_from_end_node_to_lock_gate_B
    
    
def _get_first_vessel_of_lock_operation(lock_complex, vessel, operation_index):
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
    vessels = _get_vessels_from_planned_operation(lock_complex, operation_index=operation_index,)

    # determine the first vessel if vessels are already assigned to the lock operation
    if len(vessels):
        first_vessel = vessels[0]

    return first_vessel

def _get_last_vessel_of_lock_operation(lock_complex, operation_index):
    """
    Determines the last vessel that was assigned to the lock operation

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
    operation_index : int
        index of the lock operation

    Returns
    -------
    last_vessel : type
        the last assigned vessel of the lock operation (the one that will enter and leave the lock chamber last)
    """
    # identify the vessels assigned the lock operation
    vessels = _get_vessels_from_planned_operation(lock_complex, operation_index=operation_index,)

    # determine the last vessel
    last_vessel = vessels[-1]

    return last_vessel

def _get_route_to_lock(vessel, lock, last_node_included = False):
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
    route_to_come = vessel.route_ahead
    index = 0
    for index, edge in enumerate(zip(route_to_come[:-1], route_to_come[1:])):
        if edge == lock.edge or edge == lock.edge[::-1]:
            index += 1
            break
    if last_node_included:
        index += 1
    route_to_lock = route_to_come[:(index)]
    return route_to_lock


def _get_information_for_lock_operation(lock_chamber, operation_index, direction):
    node_of_approach, to_node = _get_lock_operation_to_and_from_node(lock_chamber, direction)
    vessels = _get_vessels_from_planned_operation(lock_chamber.lock_complex, operation_index)
    capacity_L = lock_chamber.lock_length
    capacity_B = lock_chamber.lock_width
    for vessel in vessels:
        capacity_L -= vessel.L
    lock_operation_information = {"node_from": node_of_approach,
                                  "node_to": to_node,
                                  "direction": direction,
                                  "lock_chamber": lock_chamber.name,
                                  "vessels": vessels,
                                  "capacity_L": capacity_L,
                                  "capacity_B": capacity_B}
    return lock_operation_information


def _get_upcoming_lock_registration_nodes(lock_complex):
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
    route_to_come = lock_complex.route_ahead
    for node in route_to_come:
        node_info = lock_complex.multidigraph.nodes[node]

        # check if the node has a registration node
        if ("Lock_registration_node" not in node_info.keys()):
            continue

        # unpack the lock complex information using the lock_edge stored in the registration node
        lock_edge = node_info["Lock_registration_node"]
        lock = lock_complex.multidigraph.edges[lock_edge]["Lock"][0]  # TODO: write test to prevent that multiple lock complexes are located at the same registration node, also: maybe we need to change "Lock" to "Lock complex"

        # check if lock is already stored
        if lock in upcoming_locks.values():
            continue
        # store the lock object in the list of locks with long_term_planning enabled
        upcoming_locks[node] = lock
    return upcoming_locks

def _get_upcoming_locks(vessel, object = 'Lock chamber'):
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
    route_to_come = vessel.route_ahead
    for edge in zip(route_to_come[:-1], route_to_come[1:]):
        is_multidigraph = check_graph_is_multidigraph_type(vessel.env.graph)
        edge = get_edge(vessel.env.graph, edge, is_multidigraph)
        if "Lock chamber" not in vessel.env.graph.edges[edge].keys():
            continue
        lock = vessel.env.graph.edges[edge]["Lock chamber"][0]

        # check if lock is already stored
        if lock in upcoming_locks.values():
            continue

        # store the lock object in the list of locks with long_term_planning enabled
        if object == "Lock chamber":
            object = lock
        elif object == "Lock complex":
            object = lock.lock_complex

        upcoming_locks[edge[0]] = object

    return upcoming_locks


def _get_upcoming_lock_complexes(vessel):
    upcoming_lock_complexes = _get_upcoming_locks(vessel, object = 'Lock complex')
    return upcoming_lock_complexes


def _create_operational_hours(start_times,stop_times):
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
    # TODO: this is more an utility function as it does not include the lock master (lock_complex)
    # creates default dataframe
    operational_hours = pd.DataFrame(columns=['start_time', 'stop_time'])

    # includes the start and stop times of the operation windows in the dataframe
    for start_time,stop_time in zip(start_times,stop_times):
        operational_hours.loc[len(operational_hours),:] = [start_time,stop_time]

    return operational_hours


def _get_water_levels_before_and_after_levelling(lock, levelling_start, levelling_stop, direction):
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
        wlev_A = hydromanager._get_hydrodynamic_data_value(t_start, lock.start_node, "Water level")
        wlev_B = hydromanager._get_hydrodynamic_data_value(t_stop, lock.end_node, "Water level")
    else:
        wlev_A = hydromanager._get_hydrodynamic_data_value(t_stop, lock.start_node, "Water level")
        wlev_B = hydromanager._get_hydrodynamic_data_value(t_start, lock.end_node, "Water level")

    return wlev_A, wlev_B

def _check_if_empty_lock_operation_is_required(lock_chamber, operation_index, direction):
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
    node_of_approach, to_node = _get_lock_operation_to_and_from_node(lock_chamber, direction)
    lock_complex = lock_chamber.lock_complex
    previous_planned_operations = lock_complex.operation_planning[lock_complex.operation_planning.index < operation_index]
    empty_lock_operation_to_be_requested = False
    if not previous_planned_operations.empty:
        previous_planned_operation = previous_planned_operations.iloc[-1]
        if previous_planned_operation.direction == direction:
            operation_index += 1  # the new operation index lies now one ahead
    elif lock_chamber.gate_open != node_of_approach:
        empty_lock_operation_to_be_requested = True
        operation_index += 1
    return operation_index, empty_lock_operation_to_be_requested

def _update_vessel_planning_for_delayed_arrival(lock_complex, vessel, delay):
    vessel_planning = lock_complex.vessel_planning
    vessel_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
    vessel_planning.loc[vessel_index, 'time_lock_entry_start'] += datetime.timedelta(seconds=delay)
    vessel_planning.loc[vessel_index, 'time_potential_lock_gate_opening_stop'] += datetime.timedelta(seconds=delay)
    vessel_planning.loc[vessel_index, 'time_lock_entry_stop'] += datetime.timedelta(seconds=delay)

def _update_operation_planning_for_delayed_arrival(lock_complex, vessel, operation_index, delay):
    first_vessel = _get_first_vessel_of_lock_operation(lock_complex, vessel, operation_index)
    if first_vessel.id != vessel.id:
        return
    operation_planning = lock_complex.operation_planning
    operation_planning.loc[operation_index, 'time_lock_entry_start'] += datetime.timedelta(seconds=delay)
    operation_planning.loc[operation_index, 'time_potential_lock_gate_opening_stop'] += datetime.timedelta(
        seconds=delay)

    
def _find_available_lock_operation(lock_complex, vessel, direction):
    """
    Function that adds a vessel to the lock operation planning

    Parameters
    ----------
    vessel : type
        a type including the following parent-classes: PassesLockComplex
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
    operation_planning = lock_complex.operation_planning

    # determine the index of the vessel in the vessel planning to determine when the vessel is estimated to pass the approach point and enters the lock
    if not vessel.registered_to_lock:
        raise ValueError("Vessel has not been registered to the lock master")

    suitable_lock_chambers = []
    for lock_chamber in lock_complex.lock_chambers.values():
        if vessel.T < lock_chamber.lock_depth and vessel.L < lock_chamber.lock_length and vessel.B < lock_chamber.lock_width:
            suitable_lock_chambers.append(lock_chamber)

    if not len(suitable_lock_chambers):
        raise ValueError("Vessel cannot pass lock complex")

    most_suitable_lock_chamber = pd.DataFrame(columns=['time_lock_operation_start','operation_index','new_lock_operation'])
    for lock_chamber in suitable_lock_chambers:
        route_to_lock_chamber = _get_route_to_lock(vessel, lock_chamber, last_node_included=True)
        lock_distance_last_edge = lock_chamber.distance_from_start_node_to_lock_gate_A
        if direction:
            lock_distance_last_edge = lock_chamber.distance_from_end_node_to_lock_gate_B
        sailing_time_to_lock_df = get_sailing_information_on_edge_to_distance_on_another_edge(vessel, route_to_lock_chamber, distance_to_be_sailed_on_last_edge = lock_distance_last_edge) #TODO: include registration node on edge (distance_sailed_on_first_edge)
        sailing_time_to_lock = pd.Timedelta(seconds=sailing_time_to_lock_df['time'].sum())
        time_lock_entry_start = datetime.datetime.fromtimestamp(vessel.env.now) + sailing_time_to_lock

        # determine the maximum delay of an individual vessel in all the planned lock operation if the vessel is assigned to that operation
        operation_planning_lock = operation_planning[operation_planning.lock_chamber == lock_chamber.name]
        maximum_individual_delay = operation_planning_lock.maximum_individual_delay + (time_lock_entry_start - operation_planning_lock.time_lock_entry_stop)

        # filter the planned lock operations based on the following criteria to select available operations to which the vessel can be assigned
        mask_direction = operation_planning_lock.direction == direction  # lock operations in the same direction as the vessel
        mask_available = operation_planning_lock.status == "available"  # lock operations that are not unavailable
        mask_capacity_L = (operation_planning_lock.capacity_L >= vessel.L)  # lock operations that have a capacity in which the vessel fits longitudinally (based on the vessel's length)
        mask_max_waiting_time = maximum_individual_delay < pd.Timedelta(seconds=lock_chamber.clustering_time)  # lock operations that will not exceed the maximum set waiting time for individual vessels
        mask_empty_lock = operation_planning_lock.vessels.apply(len) == 0  # lock operations that are still empty

        # max vessels mask: lock operations that do not exceed a maximum number of vessels
        mask_max_vessels = mask_available
        if lock_chamber.max_vessels_in_operation:
            mask_max_vessels = operation_planning_lock.vessels.apply(len) < lock_chamber.max_vessels_in_operation

        # future operations mask: lock operations that still have to take place
        mask_future_operations = operation_planning_lock.time_levelling_start >= time_lock_entry_start

        # combinations of the masks
        mask_max_waiting_time = (mask_max_waiting_time & ~mask_empty_lock)  # non-empty lock operations with non-exceedance of the maximum waiting time
        if lock_chamber.min_vessels_in_operation:
            mask_min_vessels = operation_planning_lock.vessels.apply(len) < lock_chamber.min_vessels_in_operation
        else:
            mask_min_vessels = operation_planning_lock.vessels.apply(len) >= lock_chamber.min_vessels_in_operation

        mask_empty_available_lock = mask_empty_lock & mask_future_operations

        # select available operations
        available_operations = operation_planning_lock[
            mask_available
            & mask_direction
            & mask_min_vessels
            & mask_max_vessels
            & mask_capacity_L
            & (mask_future_operations | mask_max_waiting_time | mask_empty_available_lock)
        ].copy()
        # TODO: include mask_capacity_B for 2D implementation
        # TODO: create a selection method that can pick the lock operation based on minimizing expected delay or freshwater loss/saltwater intrusion

        display(operation_planning_lock)
        if available_operations.empty:
            new_operation = True
            if not operation_planning_lock.empty:
                last_operation = operation_planning_lock.iloc[-1]
                operation_index = last_operation.name + 1
                time_lock_operation_start = last_operation.time_lock_operation_start + sailing_time_to_lock
            else:
                operation_index = 0
                time_lock_operation_start = sailing_time_to_lock
        else:
            new_operation = False
            operation_index = available_operations.iloc[0].operation_index
            time_lock_operation_start = available_operations.iloc[0].time_lock_operation_start + sailing_time_to_lock

        most_suitable_lock_chamber.loc[lock_chamber.name] = [time_lock_operation_start, operation_index, new_operation]

    add_to_existing_operation_df = most_suitable_lock_chamber[most_suitable_lock_chamber.new_lock_operation == False]
    if not add_to_existing_operation_df.empty:
        most_efficient_operation = add_to_existing_operation_df[add_to_existing_operation_df.time_lock_operation_start == add_to_existing_operation_df.time_lock_operation_start.min()].iloc[0]
        lock_chamber = most_efficient_operation.name
        operation_index = most_efficient_operation.operation_index
        new_operation = most_efficient_operation.new_lock_operation
    else:
        most_efficient_operation = most_suitable_lock_chamber[most_suitable_lock_chamber.time_lock_operation_start == most_suitable_lock_chamber.time_lock_operation_start.min()].iloc[0]
        lock_chamber = most_efficient_operation.name
        operation_index = most_efficient_operation.operation_index
        new_operation = most_efficient_operation.new_lock_operation

    return lock_chamber, operation_index, new_operation


def _correct_lock_operation_start_time_if_outside_of_operational_hours(lock_chamber, time_lock_operation_start):
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
    operational_hours = lock_chamber.operational_hours
    within_operation_hours = operational_hours[(time_lock_operation_start >= operational_hours.start_time) & (time_lock_operation_start <= operational_hours.stop_time)]
    if within_operation_hours.empty:
        first_available_hour = operational_hours[operational_hours.start_time >= time_lock_operation_start].iloc[0]
        time_lock_operation_start = first_available_hour.start_time
    return time_lock_operation_start


def _update_future_lock_operations_by_lock_delay_previous_operation(lock_complex, operation_index, lock_departure_information):
    """Updates the lock operation and vessel plannings based on a delay in a previous planned operation

    Parameters
    ----------
    operation_index : int
        index of the lock operation
    lock_departure_information : dict
        information with start and stop times of events that make up the departure of vessels from the lock operation
        required keys: "time_lock_gate_closing_start", "time_lock_operation_stop"
    """
    operation_planning = lock_complex.operation_planning
    vessel_planning = lock_complex.vessel_planning

    # update the next lock operations if the previous lock operation caused a delay
    next_planned_operations = operation_planning[operation_planning.index > operation_index]
    for next_operation_index, next_operation_info in next_planned_operations.iterrows():

        # determine time delay of the process of sailing into the lock if the next operation in the planning confict with the delayed operation
        sailing_in_delay = pd.Timedelta(seconds=0)
        if not len(next_operation_info) and lock_departure_information["time_lock_gate_closing_start"] > next_operation_info.time_potential_lock_gate_opening_stop:
            sailing_in_delay = lock_departure_information["time_lock_gate_closing_start"] - next_operation_info.time_potential_lock_gate_opening_stop
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
        operation_planning.loc[next_operation_index, "time_potential_lock_gate_opening_stop"] += sailing_in_delay
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
            vessel_planning.loc[next_vessel_planning_index, "time_potential_lock_gate_opening_stop"] += sailing_in_delay
            vessel_planning.loc[next_vessel_planning_index, "time_potential_lock_gate_closure_start"] += sailing_in_delay
            vessel_planning.loc[next_vessel_planning_index, "time_arrival_at_lineup_area"] += sailing_in_delay
            vessel_planning.loc[next_vessel_planning_index, "time_lock_operation_start"] += sailing_in_delay
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
                    extra_delay = calculate_sailing_in_time_delay(lock_complex.lock_complex, next_next_vessel, next_operation_index,
                                                                       next_direction,
                                                                       minimum_difference_with_previous_vessel=True,
                                                                       overwrite=False)
                    sailing_in_delay += extra_delay

        # determine the new start and stop times of the lock operation (i.e., gate-closing, levelling, gate-opening) as it can be that the levelling time is now changed due to the shift of this operation in time (i.e., due to tides)
        time_gate_closing = operation_planning.loc[next_operation_index, "time_entry_stop"]
        levelling_information = calculate_lock_operation_times(lock_complex.lock_complex.lock_chamber,
                                                               operation_index=next_operation_index,
                                                               last_entering_time=last_vessel_entering_time,
                                                               start_time=time_gate_closing,
                                                               vessel=next_vessel,
                                                               direction=next_direction,)
        # update the operation planning accordingly
        operation_planning.loc[next_operation_index, "time_gate_closing_start"] = levelling_information["time_gate_closing_start"]
        operation_planning.loc[next_operation_index, "time_gate_closing_stop"] = levelling_information["time_gate_closing_stop"]
        operation_planning.loc[next_operation_index, "time_levelling_start"] = levelling_information["time_levelling_start"]
        delay_after_levelling = levelling_information["time_levelling_stop"] - operation_planning.loc[next_operation_index, "time_levelling_stop"]
        operation_planning.loc[next_operation_index, "time_levelling_stop"] = levelling_information["time_levelling_stop"]
        operation_planning.loc[next_operation_index, "time_gate_opening_start"] = levelling_information["time_gate_opening_start"]
        operation_planning.loc[next_operation_index, "time_gate_opening_stop"] = levelling_information["time_gate_opening_stop"]
        if delay_after_levelling > pd.Timedelta(seconds=0):
            operation_planning.loc[next_operation_index, "time_departure_start"] += delay_after_levelling
            operation_planning.loc[next_operation_index, "time_departure_stop"] += delay_after_levelling
            operation_planning.loc[next_operation_index, "time_operation_stop"] += delay_after_levelling
            operation_planning.loc[next_operation_index, "time_potential_lock_gate_closure_start"] += delay_after_levelling
            operation_planning.loc[next_operation_index, "total_delay"] += delay_after_levelling * len(next_vessels)
            operation_planning.loc[next_operation_index, "maximum_individual_delay"] += delay_after_levelling

        # update also the departure information of the affected vessels
        for vessel_index, next_vessel in enumerate(next_vessels):
            next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name
            vessel_planning.loc[next_vessel_planning_index, "time_lock_departure_start"] += delay_after_levelling
            vessel_planning.loc[next_vessel_planning_index, "time_lock_departure_stop"] += delay_after_levelling
            vessel_planning.loc[next_vessel_planning_index, "time_lock_operation_stop"] += delay_after_levelling
            vessel_planning.loc[next_vessel_planning_index, "delay"] += delay_after_levelling


def check_lock_distances_to_nodes_of_edge(lock_chamber):
    edge_length = get_length_of_edge(lock_chamber.env.graph, lock_chamber.edge)
    lock_edge_length = lock_chamber.distance_from_start_node_to_lock_gate_A + \
                       lock_chamber.lock_length + \
                       lock_chamber.distance_from_end_node_to_lock_gate_B
    try:
        assert_almost_equal(edge_length / 100, lock_edge_length / 100, decimal=1)
    except AssertionError as e:
        raise ValueError(f'Invalid lock dimensions: geometrical edge length ({edge_length} m) is not equal to '
                         f'the lock length and distances to the nodes of the lock edge ({lock_edge_length} m). '
                         f'The distance from node A to lock gate A is {lock_chamber.distance_from_start_node_to_lock_gate_A} m, '
                         f'the lock chamber length between the lock gates is  {lock_chamber.lock_length} m, and '
                         f'the distance from node B to lock gate B is {lock_chamber.distance_from_end_node_to_lock_gate_B} m') from None


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


def add_lock_to_graph(lock_chamber):
    # Add the lock to the edge or append it to the existing list
    if "Lock chamber" not in lock_chamber.env.graph.edges[lock_chamber.edge].keys():
        lock_chamber.env.graph.edges[lock_chamber.edge]["Lock chamber"] = [lock_chamber]
    elif lock_chamber not in lock_chamber.env.graph.edges[lock_chamber.edge]["Lock chamber"]:
        lock_chamber.env.graph.edges[lock_chamber.edge]["Lock chamber"].append(lock_chamber)

    lock_chamber.edge_reversed = (lock_chamber.edge[1], lock_chamber.edge[0]) + lock_chamber.edge[2:]
    if "Lock chamber" not in lock_chamber.env.graph.edges[lock_chamber.edge_reversed].keys():
        lock_chamber.env.graph.edges[lock_chamber.edge_reversed]["Lock chamber"] = [lock_chamber]
    elif lock_chamber not in lock_chamber.env.graph.edges[lock_chamber.edge_reversed]["Lock chamber"]:
        lock_chamber.env.graph.edges[lock_chamber.edge_reversed]["Lock chamber"].append(lock_chamber)