"""This module contains utility functions for lock operations in the OpenTNSim simulation environment."""
from collections import deque
import datetime
from itertools import permutations
import math
import networkx as nx
import numpy as np
from operator import itemgetter
import pandas as pd
from numpy.testing import assert_almost_equal
from opentnsim.graph.utils import get_length_of_edge, get_edge, get_sailing_information_on_edge_to_distance_on_another_edge, expand_path_edges, node_path_to_edge_path
from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from IPython.display import display
import warnings


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
    m_find_available_lock_operationultidigraph : nx.MultiDiGraph
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


def _get_operation_info(lock_chamber, operation_index):
    try:
        operation_planning = lock_chamber.lock_complex.operation_planning
        operation_planning_lock = operation_planning[operation_planning.lock_chamber == lock_chamber.name]
        operation_info = operation_planning_lock[operation_planning_lock.operation_index == operation_index].iloc[-1]
    except:
        operation_info = pd.Series()
    return operation_info


def _get_previous_operations(lock_chamber, operation_index):
    try:
        operation_planning = lock_chamber.lock_complex.operation_planning
        operation_planning_lock = operation_planning[operation_planning.lock_chamber == lock_chamber.name]
        previous_operations = operation_planning_lock[operation_planning_lock.operation_index < operation_index]
    except:
        previous_operations = pd.DataFrame()
    return previous_operations


def _get_next_operations(lock_chamber, operation_index):
    try:
        operation_planning = lock_chamber.lock_complex.operation_planning
        operation_planning_lock = operation_planning[operation_planning.lock_chamber == lock_chamber.name]
        next_operations = operation_planning_lock[operation_planning_lock.operation_index > operation_index]
    except:
        next_operations = pd.DataFrame()
    return next_operations


def _get_vessels_from_planned_operation(lock_chamber, operation_index = None):
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
    operation_info = _get_operation_info(lock_chamber, operation_index)
    if not operation_info.empty and isinstance(operation_info['vessels'], list):
        vessels = operation_info['vessels'].copy()
    return vessels


def _update_lock_operation_planning(lock_chamber, operation_index, operation_information):
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
    operation_planning = lock_chamber.lock_complex.operation_planning
    for key, value in operation_information.items():
        if key not in operation_planning.columns:
            continue
        operation_info = _get_operation_info(lock_chamber, operation_index)
        operation_planning.at[operation_info.name, key] = value


def _update_lock_vessel_planning(lock_chamber, vessel_index, passage_information):
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
    vessel_planning = lock_chamber.lock_complex.vessel_planning
    for key, value in passage_information.items():
        if key not in vessel_planning.columns:
            continue

        if isinstance(value, pd.Timedelta):
            value = value.round('us')

        vessel_planning.at[int(vessel_index), key] = value


def _find_available_waiting_area(vessel, lock_chamber, direction):
    distance_to_lock_on_edge = lock_chamber.distance_from_start_node_to_lock_gate_A
    if direction:
        distance_to_lock_on_edge = lock_chamber.distance_from_end_node_to_lock_gate_B
    suitable_waiting_areas = pd.DataFrame(columns=['sailing_time_waiting_area_to_lock','available'])
    for edge in vessel.edge_route:
        if 'Waiting area' not in vessel.env.graph.edges[edge].keys():
            continue
        waiting_areas = vessel.env.graph.edges[edge]['Waiting area']
        for waiting_area in waiting_areas:
            distance_to_waiting_area_on_edge = waiting_area.distance_from_edge_start
            get_sailing_info = get_sailing_information_on_edge_to_distance_on_another_edge
            edge_route_to_waiting_area = _get_edge_route_to_waiting_area(vessel, waiting_area, last_node_included=True)
            if vessel.current_edge != waiting_area.edge:
                distance_to_waiting_area_on_edge = 0.
            sailing_info = get_sailing_info(
                vessel, edge_route_to_waiting_area, distance_to_waiting_area_on_edge, distance_to_lock_on_edge)
            sailing_time = pd.Timedelta(seconds=sailing_info.time.sum())
            available = waiting_area.resource.capacity > len(waiting_area.resource.users)
            suitable_waiting_areas.loc[waiting_area.name,:] = [sailing_time, available]

    print(vessel.route, direction)
    display(suitable_waiting_areas)
    available_waiting_areas = suitable_waiting_areas[suitable_waiting_areas.available]
    waiting_area_name = None
    if not available_waiting_areas.empty:
        waiting_area_name = available_waiting_areas.sort_values('sailing_time_waiting_area_to_lock').iloc[0].name

    if waiting_area_name is None:
        raise ValueError(f"No route found to waiting area")

    return waiting_area_name


def _get_lock_operation_direction(lock_chamber, to_node):
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
    if to_node == lock_chamber.start_node:
        direction = 1

    return direction


def _get_previous_assigned_vessel(lock_chamber, operation_index):
    operation_info = _get_operation_info(lock_chamber, operation_index)
    if operation_info.empty or len(operation_info["vessels"]) == 1:
        return None
    previous_vessel = operation_info.vessels[-2]
    return previous_vessel


def _get_distance_to_lock(lock_chamber, direction):
    """get the distance from the start node of the lock to the lock gate from the perspective of the vessel

    Parameters
    ----------
    direction : int
        the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
    """
    if not direction:
        return lock_chamber.distance_from_start_node_to_lock_gate_A
    else:
        return lock_chamber.distance_from_end_node_to_lock_gate_B


def _check_if_vessel_is_first_vessel(lock_chamber, vessel, operation_index):
    is_first_vessel = False
    first_vessel = _get_first_vessel_of_lock_operation(lock_chamber, vessel, operation_index)
    if vessel == first_vessel:
        is_first_vessel = True
    return is_first_vessel


def _check_if_vessel_is_last_vessel(lock_chamber, vessel, operation_index):
    is_last_vessel = False
    last_vessel = _get_last_vessel_of_lock_operation(lock_chamber, operation_index)
    if vessel == last_vessel:
        is_last_vessel = True
    return is_last_vessel


def _get_first_vessel_of_lock_operation(lock_chamber, vessel, operation_index):
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
    vessels = _get_vessels_from_planned_operation(lock_chamber, operation_index=operation_index,)

    # determine the first vessel if vessels are already assigned to the lock operation
    if len(vessels):
        first_vessel = vessels[0]

    return first_vessel

def _get_last_vessel_of_lock_operation(lock_chamber, operation_index):
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
    vessels = _get_vessels_from_planned_operation(lock_chamber, operation_index=operation_index,)

    # determine the last vessel
    last_vessel = None
    if len(vessels):
        last_vessel = vessels[-1]

    return last_vessel

def _get_edge_route_to_lock(vessel, lock_chamber, last_node_included = False):
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
    edge_route_to_come = vessel.edge_route_ahead
    index = 0
    for index, edge in enumerate(edge_route_to_come):
        edge_rev = (edge[1], edge[0]) + edge[2:]
        if edge == lock_chamber.edge or edge_rev == lock_chamber.edge:
            break
        index += 1
    if last_node_included:
        index += 1
    edge_route_to_lock = edge_route_to_come[:(index)]
    return edge_route_to_lock


def _get_edge_route_to_waiting_area(vessel, waiting_area, last_node_included = False):
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
    edge_route_to_come = vessel.edge_route_ahead
    index = 0
    for index, edge in enumerate(edge_route_to_come):
        edge_rev = (edge[1], edge[0]) + edge[2:]
        if edge == waiting_area.edge or edge_rev == waiting_area.edge:
            break
        index += 1
    if last_node_included:
        index += 1
    edge_route_to_lock = edge_route_to_come[:(index)]
    return edge_route_to_lock


def _get_information_for_lock_operation(lock_chamber, operation_index, direction):
    node_of_approach, to_node = _get_lock_operation_to_and_from_node(lock_chamber, direction)
    vessels = _get_vessels_from_planned_operation(lock_chamber, operation_index)
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

def _get_upcoming_lock_complexes(vessel):
    """
    Find the upcoming locks that use long-term planning by looping over the vessel's route

    Parameters
    ----------

    Returns
    -------
    upcoming_lock_complexes : dict
        dictionary of lock objects that are to be encountered on the vessel's route
        mapping from node (key) to lock object (value)
    """
    # initiate empty lists
    upcoming_lock_complexes = {}

    # loop over all edges on the route ahead.
    edge_route_to_come = vessel.edge_route_ahead
    for edge in edge_route_to_come:
        if "Lock chamber" not in vessel.env.graph.edges[edge].keys():
            continue
        lock_chamber = vessel.env.graph.edges[edge]["Lock chamber"][0]
        lock_complex = lock_chamber.lock_complex

        # check if lock is already stored
        if lock_complex in upcoming_lock_complexes.values():
            continue

        upcoming_lock_complexes[edge[0]] = lock_complex

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
    previous_planned_operations = _get_previous_operations(lock_chamber, operation_index)
    current_time = datetime.datetime.fromtimestamp(lock_chamber.env.now)
    operations_yet_to_be_processed = previous_planned_operations[
        previous_planned_operations['time_lock_operation_stop'] > current_time
    ]
    empty_lock_operation_to_be_requested = False
    lock_operation_to_be_executed = False
    if not previous_planned_operations.empty:
        previous_planned_operation = previous_planned_operations.iloc[-1]
        if previous_planned_operation.direction == direction:
            empty_lock_operation_to_be_requested = True
            operation_index += 1
            if operations_yet_to_be_processed.empty:
                lock_operation_to_be_executed = True
    elif lock_chamber.gate_open_at_node != node_of_approach:
        lock_operation_to_be_executed = True
        empty_lock_operation_to_be_requested = True
        operation_index += 1
    return operation_index, empty_lock_operation_to_be_requested, lock_operation_to_be_executed

def _update_vessel_planning_for_delayed_arrival(lock_complex, vessel, delay):
    vessel_planning = lock_complex.vessel_planning
    vessel_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
    delay = datetime.timedelta(seconds=delay).round("us")
    vessel_planning.loc[vessel_index, 'time_arrival_at_approach_point'] += delay
    vessel_planning.loc[vessel_index, 'time_lock_entry_start'] += delay
    vessel_planning.loc[vessel_index, 'time_potential_lock_gate_opening_stop'] += delay
    vessel_planning.loc[vessel_index, 'time_lock_entry_stop'] += delay

def _update_vessel_planning_for_delayed_deparature(lock_complex, vessel, delay):
    vessel_planning = lock_complex.vessel_planning
    vessel_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
    delay = datetime.timedelta(seconds=delay).round("us")
    vessel_planning.loc[vessel_index, 'time_lock_departure_start'] += delay
    vessel_planning.loc[vessel_index, 'time_lock_departure_stop'] += delay
    vessel_planning.loc[vessel_index, 'time_lock_operation_stop'] += delay
    vessel_planning.loc[vessel_index, 'time_potential_lock_gate_closure_start'] += delay


def _update_operation_planning_for_delayed_arrival(lock_chamber, vessel, operation_index, delay):
    first_vessel = _get_first_vessel_of_lock_operation(lock_chamber, vessel, operation_index)
    if first_vessel.id != vessel.id:
        return
    delay = datetime.timedelta(seconds=delay).round("us")
    operation_planning = lock_chamber.lock_complex.operation_planning
    index = _get_operation_info(lock_chamber, operation_index)
    operation_planning.loc[index, 'time_lock_entry_start'] += delay
    operation_planning.loc[index, 'time_potential_lock_gate_opening_stop'] += delay

    
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
        lock_edge = lock_chamber.edge
        if direction:
            lock_edge = (lock_chamber.edge[1], lock_chamber.edge[0]) + lock_chamber.edge[2:]
        route_to_lock_chamber = nx.dijkstra_path(vessel.env.graph, vessel.current_node, lock_edge[0])
        edge_route_to_lock_chamber = node_path_to_edge_path(vessel.env.graph, route_to_lock_chamber)
        lock_distance_last_edge = lock_chamber.distance_from_start_node_to_lock_gate_A
        if direction:
            lock_distance_last_edge = lock_chamber.distance_from_end_node_to_lock_gate_B
        sailing_time_to_lock_df = get_sailing_information_on_edge_to_distance_on_another_edge(
            vessel, edge_route_to_lock_chamber, distance_to_be_sailed_on_last_edge = lock_distance_last_edge)
        sailing_time_to_lock = pd.Timedelta(seconds=sailing_time_to_lock_df['time'].sum())
        time_lock_entry_start = datetime.datetime.fromtimestamp(vessel.env.now) + sailing_time_to_lock

        # determine the maximum delay of an individual vessel in all the planned lock operation if the vessel is assigned to that operation
        operation_planning_lock = operation_planning[operation_planning.lock_chamber == lock_chamber.name]
        maximum_individual_delay = operation_planning_lock.maximum_individual_delay + (time_lock_entry_start - operation_planning_lock.time_lock_entry_stop)

        # filter the planned lock operations based on the following criteria to select available operations to which the vessel can be assigned
        mask_direction = operation_planning_lock.direction == direction  # lock operations in the same direction as the vessel
        mask_capacity_L = (operation_planning_lock.capacity_L >= vessel.L)  # lock operations that have a capacity in which the vessel fits longitudinally (based on the vessel's length)
        mask_max_waiting_time = maximum_individual_delay < pd.Timedelta(seconds=lock_chamber.clustering_time)  # lock operations that will not exceed the maximum set waiting time for individual vessels
        mask_empty_lock = operation_planning_lock.vessels.apply(len) == 0  # lock operations that are still empty

        # max vessels mask: lock operations that do not exceed a maximum number of vessels
        mask_max_vessels = mask_direction
        if lock_chamber.max_vessels_in_operation:
            mask_max_vessels = operation_planning_lock.vessels.apply(len) < lock_chamber.max_vessels_in_operation

        # future operations mask: lock operations that still have to take place
        mask_future_operations = operation_planning_lock.time_gate_closing_start >= time_lock_entry_start

        # combinations of the masks
        mask_max_waiting_time = (mask_max_waiting_time & ~mask_empty_lock)  # non-empty lock operations with non-exceedance of the maximum waiting time
        if lock_chamber.min_vessels_in_operation:
            mask_min_vessels = operation_planning_lock.vessels.apply(len) < lock_chamber.min_vessels_in_operation
        else:
            mask_min_vessels = operation_planning_lock.vessels.apply(len) >= lock_chamber.min_vessels_in_operation

        mask_empty_available_lock = mask_empty_lock & mask_future_operations
        # select available operations
        available_operations = operation_planning_lock[
            mask_direction
            & mask_min_vessels
            & mask_max_vessels
            & mask_capacity_L
            & (mask_future_operations | mask_max_waiting_time | mask_empty_available_lock)
        ].copy()
        # TODO: include mask_capacity_B for 2D implementation
        # TODO: create a selection method that can pick the lock operation based on minimizing expected delay or freshwater loss/saltwater intrusion

        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        if available_operations.empty:
            new_operation = True
            if not operation_planning_lock.empty:
                last_operation = operation_planning_lock.iloc[-1]
                operation_index = len(operation_planning[operation_planning.lock_chamber == lock_chamber.name])
                time_lock_operation_start = (last_operation.time_lock_operation_start - current_time) + sailing_time_to_lock
            else:
                operation_index = 0
                time_lock_operation_start = sailing_time_to_lock
        else:
            new_operation = False
            operation_index = available_operations.iloc[0].operation_index
            time_lock_operation_start = (available_operations.iloc[0].time_lock_operation_start - current_time) + sailing_time_to_lock

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


def _update_future_lock_operations_by_lock_delay_previous_operation(lock_chamber, operation_index, lock_departure_information):
    """Updates the lock operation and vessel plannings based on a delay in a previous planned operation

    Parameters
    ----------
    operation_index : int
        index of the lock operation
    lock_departure_information : dict
        information with start and stop times of events that make up the departure of vessels from the lock operation
        required keys: "time_lock_gate_closing_start", "time_lock_operation_stop"
    """
    from opentnsim.lock.calculations import calculate_sailing_in_time_delay, calculate_lock_operation_times

    lock_complex = lock_chamber.lock_complex
    operation_planning = lock_complex.operation_planning
    vessel_planning = lock_complex.vessel_planning

    # update the next lock operations if the previous lock operation caused a delay
    next_planned_operations = _get_next_operations(lock_chamber, operation_index)
    for next_operation_index, next_operation_info in next_planned_operations.iterrows():
        next_operation_planning_index = next_operation_info.name

        # determine time delay of the process of sailing into the lock if the next operation in the planning confict with the delayed operation
        sailing_in_delay = pd.Timedelta(seconds=0)
        if not len(next_operation_info) and lock_departure_information["time_lock_gate_closing_start"] > next_operation_info.time_potential_lock_gate_opening_stop:
            sailing_in_delay = lock_departure_information["time_lock_gate_closing_start"] - next_operation_info.time_potential_lock_gate_opening_stop
        elif len(next_operation_info) and lock_departure_information["time_lock_operation_stop"] > next_operation_info.time_lock_operation_start:
            sailing_in_delay = lock_departure_information["time_lock_operation_stop"] - next_operation_info.time_lock_operation_start

        # determine the new start time of the next operation (dependening on whether it will fall withing the operation hours)
        sailing_in_delay = sailing_in_delay.round("us")
        new_operation_start = operation_planning.loc[next_operation_planning_index, "time_lock_operation_start"] + sailing_in_delay
        operational_hours = lock_chamber.operational_hours
        within_operation_hours = operational_hours[(new_operation_start >= operational_hours.start_time) &
                                                   (new_operation_start <= operational_hours.stop_time)]
        if within_operation_hours.empty:
            first_available_hour = operational_hours[operational_hours.start_time >= new_operation_start].iloc[0]
            sailing_in_delay += first_available_hour.start_time - new_operation_start

        # break loop if there is no delay (next operations will then also not experience a delay)
        if not sailing_in_delay.total_seconds() > 0:
            break

        # update the operation planning if there is a delay
        operation_planning.loc[next_operation_planning_index, "time_potential_lock_gate_opening_stop"] += sailing_in_delay
        operation_planning.loc[next_operation_planning_index, "time_lock_operation_start"] += sailing_in_delay
        operation_planning.loc[next_operation_planning_index, "time_lock_entry_start"] += sailing_in_delay
        operation_planning.loc[next_operation_planning_index, "time_lock_entry_stop"] += sailing_in_delay

        # update the vessel planning
        next_vessel = None
        next_vessels = next_operation_info.vessels
        next_direction = next_operation_info.direction
        for next_vessel_index, next_vessel in enumerate(next_vessels):
            next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name
            vessel_planning.loc[next_vessel_planning_index, "time_potential_lock_gate_opening_stop"] += sailing_in_delay
            vessel_planning.loc[next_vessel_planning_index, "time_potential_lock_gate_closure_start"] += sailing_in_delay
            vessel_planning.loc[next_vessel_planning_index, "time_arrival_at_approach_point"] += sailing_in_delay
            vessel_planning.loc[next_vessel_planning_index, "time_arrival_at_lineup_area"] += sailing_in_delay
            vessel_planning.loc[next_vessel_planning_index, "time_lock_operation_start"] += sailing_in_delay
            vessel_planning.loc[next_vessel_planning_index, "time_lock_entry_start"] += sailing_in_delay
            vessel_planning.loc[next_vessel_planning_index, "time_lock_entry_stop"] += sailing_in_delay

        # determine the new start and stop times of the lock operation (i.e., gate-closing, levelling, gate-opening) as it can be that the levelling time is now changed due to the shift of this operation in time (i.e., due to tides)
        time_gate_closing = operation_planning.loc[next_operation_planning_index, "time_lock_entry_stop"]
        levelling_information = calculate_lock_operation_times(lock_chamber,
                                                               operation_index=next_operation_index,
                                                               start_time=time_gate_closing,
                                                               vessel=next_vessel,
                                                               direction=next_direction,)
        # update the operation planning accordingly
        operation_planning.loc[next_operation_planning_index, "time_gate_closing_start"] = levelling_information["time_gate_closing_start"]
        operation_planning.loc[next_operation_planning_index, "time_gate_closing_stop"] = levelling_information["time_gate_closing_stop"]
        operation_planning.loc[next_operation_planning_index, "time_levelling_start"] = levelling_information["time_levelling_start"]
        delay_after_levelling = levelling_information["time_levelling_stop"] - operation_planning.loc[next_operation_planning_index, "time_levelling_stop"]
        operation_planning.loc[next_operation_planning_index, "time_levelling_stop"] = levelling_information["time_levelling_stop"]
        operation_planning.loc[next_operation_planning_index, "time_gate_opening_start"] = levelling_information["time_gate_opening_start"]
        operation_planning.loc[next_operation_planning_index, "time_gate_opening_stop"] = levelling_information["time_gate_opening_stop"]
        if delay_after_levelling > pd.Timedelta(seconds=0):
            delay_after_levelling = delay_after_levelling.round("us")
            operation_planning.loc[next_operation_planning_index, "time_lock_departure_start"] += delay_after_levelling
            operation_planning.loc[next_operation_planning_index, "time_lock_departure_stop"] += delay_after_levelling
            operation_planning.loc[next_operation_planning_index, "time_lock_operation_stop"] += delay_after_levelling
            operation_planning.loc[next_operation_planning_index, "time_potential_lock_gate_closure_start"] += delay_after_levelling
            operation_planning.loc[next_operation_planning_index, "total_delay"] += delay_after_levelling * len(next_vessels)
            operation_planning.loc[next_operation_planning_index, "maximum_individual_delay"] += delay_after_levelling

        # update also the departure information of the affected vessels
        for vessel_index, next_vessel in enumerate(next_vessels):
            next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name
            delay_after_levelling = delay_after_levelling.round("us")
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


def _check_if_lock_chamber_is_next_lock_complex_object(lock_chamber, edge):
    if not 'Lock chamber' in lock_chamber.env.graph.edges[edge].keys():
        return

    lock_chambers_on_edge = lock_chamber.env.graph.edges[edge]['Lock chamber']
    lock_chamber_is_next_lock_complex_object = False
    for lock_chamber_on_edge in lock_chambers_on_edge:
        if lock_chamber == lock_chamber_on_edge:
            lock_chamber_is_next_lock_complex_object = True
            break
    return lock_chamber_is_next_lock_complex_object


def _get_lock_gate_position(lock_chamber, direction):
    if not direction:
        first_lock_gate_position = lock_chamber.gate_A.geometry
    else:
        first_lock_gate_position = lock_chamber.gate_B.geometry

    return first_lock_gate_position


def determine_if_gate_can_be_closed(lock_chamber, vessel, direction, operation_index, between_arrivals=False):
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
    from opentnsim.lock.calculations import calculate_time_to_open_gate
    operation_planning = lock_chamber.lock_complex.operation_planning
    vessel_planning = lock_chamber.lock_complex.vessel_planning

    this_operation = _get_operation_info(lock_chamber, operation_index)
    vessels_in_operation = this_operation.vessels
    last_vessel_to_enter_lock = vessels_in_operation[-1] == vessel

    gate_can_be_closed = False
    if not between_arrivals and not lock_chamber.closing_gate_in_between_operations:
        return gate_can_be_closed
    if between_arrivals and (not lock_chamber.closing_gate_in_between_arrivals or not last_vessel_to_enter_lock):
        return gate_can_be_closed
    gate_can_be_closed = True

    if not between_arrivals:
        last_time_gate_closed = this_operation.time_potential_lock_gate_closure_start
    else:
        last_time_gate_closed = pd.Timestamp(datetime.datetime.fromtimestamp(lock_chamber.env.now))
    gate_closing_time = pd.Timedelta(seconds=lock_chamber.gate_closing_time).round("us")
    last_time_gate_closed += gate_closing_time

    next_operations = operation_planning[(operation_planning.lock_chamber == lock_chamber.name) &
                                         (operation_planning.index > operation_index)]
    vessels_in_operation = this_operation["vessels"]
    vessel_index = vessels_in_operation.index(vessel)

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

    gate_opening_time = calculate_time_to_open_gate(lock_chamber, operation_index + operation_step,
                                                    direction, gate_required_to_be_open)

    if (
        gate_required_to_be_open - gate_opening_time < last_time_gate_closed
        or gate_required_to_be_open - last_time_gate_closed
        < lock_chamber.minimum_time_between_operations
    ):
        gate_can_be_closed = False
    return gate_can_be_closed


def determine_if_gate_is_closed(lock_chamber, operation_index, direction, vessel = None, first_in_lock=False, between_arrivals=False):
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
    from opentnsim.lock.calculations import calculate_time_to_open_gate
    operation_planning = lock_chamber.lock_complex.operation_planning
    vessel_planning = lock_chamber.lock_complex.vessel_planning
    this_operation = _get_operation_info(lock_chamber, operation_index)
    vessels = this_operation["vessels"]
    vessel_index = 0
    if vessel is not None:
        vessel_index = vessels.index(vessel)
    else:
        first_in_lock = True

    if between_arrivals and not lock_chamber.closing_gate_in_between_arrivals:
        return False, None, None

    if not between_arrivals and not lock_chamber.closing_gate_in_between_operations:
        return False, None, None

    last_lockage_was_empty = False
    if operation_index - 2 in operation_planning.index:

        last_lockage_was_empty = len(operation_planning.loc[operation_index - 1, "vessels"]) == 0
    if last_lockage_was_empty:
        return False, None, None

    if not first_in_lock and vessel_index:
        previous_vessel_id = this_operation["vessels"][vessel_index - 1].id
        previous_vessel_planning_index = vessel_planning[vessel_planning.id == previous_vessel_id].iloc[-1].name
        last_time_gate_closed = vessel_planning.loc[previous_vessel_planning_index,
                                                    "time_potential_lock_gate_closure_start"] + \
                                pd.Timedelta(seconds=lock_chamber.gate_closing_time)
    elif operation_index == 0:
        last_time_gate_closed = datetime.datetime.fromtimestamp(lock_chamber.env.now)
    else:
        previous_operations = _get_previous_operations(lock_chamber, operation_index)
        if len(previous_operations):
            previous_operation = previous_operations.iloc[-1]
            last_time_gate_closed = previous_operation.time_potential_lock_gate_closure_start + \
                                    pd.Timedelta(seconds=lock_chamber.gate_closing_time)
        elif not lock_chamber.closing_gate_in_between_operations:
            last_time_gate_closed = datetime.datetime.fromtimestamp(lock_chamber.env.now)
        else:
            last_time_gate_closed = lock_chamber.env.simulation_start

    if first_in_lock:
        gate_required_to_be_open = this_operation.time_potential_lock_gate_opening_stop
    else:
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        gate_required_to_be_open = vessel_planning.loc[vessel_planning_index, "time_potential_lock_gate_opening_stop"]

    operation_time = calculate_time_to_open_gate(lock_chamber, operation_index, direction, gate_required_to_be_open)
    gate_are_closed = False
    gates_are_closed1 = gate_required_to_be_open - operation_time > last_time_gate_closed
    gates_are_closed2 = gate_required_to_be_open - last_time_gate_closed  > lock_chamber.minimum_time_between_operations
    if gates_are_closed1 and gates_are_closed2:
        gate_are_closed = True

    return gate_are_closed, gate_required_to_be_open, operation_time


def check_lock_complex_geometry(lock_complex):
    for node_start, node_stop in permutations(lock_complex.registration_nodes, 2):
        if node_start == node_stop:
            continue

        locks_found = {}
        for lock_chamber in lock_complex.lock_chambers.values():
            locks_found[lock_chamber.name] = False

        route = nx.dijkstra_path(lock_chamber.env.graph, node_start, node_stop)
        edge_routes = expand_path_edges(lock_chamber.env.graph, route)
        for edge_route in edge_routes:
            lock_found = False
            waiting_area_before_lock_chamber = False
            distance_waiting_area_from_edge_start = math.inf
            for edge in edge_route:
                edge_info = lock_complex.env.graph.edges[edge]
                waiting_areas = []
                locks = []
                if not waiting_area_before_lock_chamber and 'Waiting area' in edge_info.keys():
                    waiting_areas = lock_complex.env.graph.edges[edge]['Waiting area']

                if not len(waiting_areas):
                    continue

                distance_from_edge_start = math.inf
                for waiting_area in waiting_areas:
                    if waiting_area.lock_complex.name == lock_complex.name:
                        waiting_area_before_lock_chamber = True
                    if not waiting_area_before_lock_chamber:
                        continue
                    if waiting_area.distance_from_edge_start < distance_from_edge_start:
                        distance_waiting_area_from_edge_start = waiting_area.distance_from_edge_start

                lock_chambers = []
                if waiting_area_before_lock_chamber and 'Lock chamber' in edge_info.keys():
                    lock_chambers = lock_complex.env.graph.edges[edge]['Lock chamber']

                if not len(locks):
                    continue

                for lock_chamber in lock_chambers:
                    if lock_chamber.lock_complex.name == lock_complex.name:
                        lock_found = True
                    if not lock_found:
                        continue

                    distance_lock_gate_to_start_node = lock_chamber.distance_from_start_node_to_lock_gate_A
                    if edge == lock_chamber.edge[::-1]:
                        distance_lock_gate_to_start_node = lock_chamber.distance_from_end_node_to_lock_gate_B

                    if distance_lock_gate_to_start_node > distance_waiting_area_from_edge_start:
                        locks_found[lock_chamber.name] = True

            if lock_found:
                for lock_chamber in lock_complex.lock_chambers.values():
                    if not locks_found[lock_chamber.name]:
                        raise ValueError('Setup of lock complex is not correct')

def check_all_paths_through_registration(lock_complex):
    """
    Check that all paths in each direction from a lock edge pass at least one registration node.

    Returns True if valid, False if there exists a path leaving the lock without registration.
    """
    graph = lock_complex.env.graph
    for lock_chamber in lock_complex.lock_chambers.values():
        u, v = lock_chamber.edge[:2]

        # For undirected graphs, we need to traverse away from u and v separately
        def bfs_check(start, blocked):
            """
            BFS from start, avoiding going back to blocked node.
            """
            visited = set()
            queue = deque([(start, start in lock_complex.registration_nodes)])

            while queue:
                node, passed_reg = queue.popleft()
                if (node, passed_reg) in visited:
                    continue
                visited.add((node, passed_reg))

                # If this path reaches a leaf without registration
                neighbors = set(graph.neighbors(node)) - {blocked}
                if not neighbors and not passed_reg:
                    raise ValueError('Setup of lock complex is not correct')

                for nbr in neighbors:
                    queue.append((nbr, passed_reg or nbr in lock_complex.registration_nodes))

        bfs_check(u, blocked=v)
        bfs_check(v, blocked=u)


def _get_directional_edge(lock_chamber, direction):
    """get the edge of the lock chamber in the correct direction"""
    if not direction:
        edge = (lock_chamber.start_node, lock_chamber.end_node, lock_chamber.k)
    else:
        edge = (lock_chamber.end_node, lock_chamber.start_node, lock_chamber.k)

    if not (isinstance(lock_chamber.env.graph, nx.MultiGraph) or isinstance(lock_chamber.env.graph, nx.MultiDiGraph)):
        edge = edge[:2]

    return edge


def _get_vessel_sailing_speed_in_lock(lock_chamber, vessel):
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
        the average speed in the lock from the lock gate to the location of berthing

    """
    # TODO: sailing_in_speed_B zou A of B moeten zijn. Checken of deze eigenschap vaker voorkomt.
    speed = lock_chamber.sailing_in_speed_B
    if vessel.bound == 'inbound':
        speed = lock_chamber.sailing_in_speed_A

    return speed

def _get_vessel_sailing_speed_out_lock(lock_chamber, vessel):
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
        the average speed in the lock from the lock gate to the location of berthing

    """
    speed = lock_chamber.sailing_out_speed_A
    if vessel.bound == 'inbound':
        speed = lock_chamber.sailing_out_speed_B

    return speed

def _get_vessel_sailing_in_speed(lock_chamber, vessel, direction):
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
        the average speed in the lock from the lock gate to the location of berthing

    """
    # determine the edge on which the vessel is sailing and the distance to the lock gate
    if vessel is None:
        return 0

    edge = _get_directional_edge(lock_chamber, direction)

    # determine the speed of the vessel over the edge
    speed = vessel._compute_velocity_on_edge(edge)

    # if there is an overruled speed on the edge, use this speed
    if "overruled_speed" in dir(vessel) and edge in vessel.overruled_speed.index:
        speed = vessel.overruled_speed.loc[edge, "speed"]
    return speed


def _get_vessel_sailing_out_speed(lock_chamber, vessel, direction):
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
        the average speed in the lock from the lock gate to the location of berthing

    """
    if vessel is None:
        return 0

    # determine the edge on which the vessel is sailing and the distance to the lock gate
    edge = _get_directional_edge(lock_chamber, direction)

    # determine the speed of the vessel over the edge
    speed = vessel._compute_velocity_on_edge(edge)

    # if there is an overruled speed on the edge, use this speed
    if 'overruled_speed' in dir(vessel) and edge in vessel.overruled_speed.index:
        speed = vessel.overruled_speed.loc[edge, 'speed']

    return speed


def _get_vessel_departure_start_delay(lock_chamber, vessel, operation_index):
    delay_to_departure = pd.Timedelta(seconds=0)
    if vessel is None:
        return delay_to_departure

    lock_complex = lock_chamber.lock_complex
    vessel_planning = lock_complex.vessel_planning
    vessels = _get_vessels_from_planned_operation(lock_chamber, operation_index=operation_index)
    vessel_index = vessels.index(vessel)

    departure_start_delay = pd.Timedelta(seconds = 0)
    previous_vessels = vessels[:(vessel_index)]
    if not len(previous_vessels):
        return departure_start_delay

    index_vessel = vessel_planning[vessel_planning.id == vessel.id].iloc[0].name
    current_time = datetime.datetime.fromtimestamp(vessel.env.now)
    sailing_out_start_v1 = vessel_planning.loc[index_vessel, 'time_lock_departure_start']
    delay_sailing_through_gate = sailing_out_start_v1 - current_time
    return delay_sailing_through_gate


def _get_vessels_that_passed_the_lock_chamber(lock_chamber):
    lock_complex = lock_chamber.lock_complex
    vessel_ids = lock_complex.vessel_planning[lock_complex.vessel_planning.lock_chamber == lock_chamber.name].id
    if not len(vessel_ids):
        return []
    vessels = np.array([itemgetter(*vessel_ids)(lock_complex.env.vessels)]).flatten()
    return vessels

