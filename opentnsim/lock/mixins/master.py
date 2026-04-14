# package(s) related to the simulation
import functools
import numpy as np
import pandas as pd
import networkx as nx
import simpy

from opentnsim.lock.calculations import (
    calculate_delay_previous_vessel_to_optimize_sailing_in_process,
    calculate_lock_operation_information_and_update_planning,
    calculate_vessel_approach_information,
    calculate_empty_lock_operation_information_and_update_planning,
    calculate_optimal_approach_speed_information
)
from opentnsim.lock.utils import (
    _check_if_empty_lock_operation_is_required,
    _update_lock_vessel_planning,
    _get_lock_operation_to_and_from_node,
    _find_available_waiting_area,
    _get_previous_assigned_vessel,
    _update_vessel_planning_for_delayed_arrival,
    _update_operation_planning_for_delayed_arrival,
    _find_available_lock_operation,
    _update_future_lock_operations_by_lock_delay_previous_operation,
)
from opentnsim.graph.utils import node_path_to_edge_path, expand_path_edges

class IsLockMaster:
    """Mixin class: lock complex has a lock master:

    Creates a lock master that schedules the vessels into lock operations

    Attributes
    ----------
    register_vessel :
        registers a vessel to the lock operation and vessel planning
    add_vessel_to_lock_operation :
        adds the vessel the lock master's operation planning
    communicate_vessel_to_sail_to_waiting_area :
        communicates to the vessel to continue sailing to the waiting area
    optimize_arrival_time_previous_vessel :
        optimizes the approach speed of the previous vessel to reduce the gate-open time of the lock
    update_vessel_planning_for_delayed_arrival :
    update_operation_planning_for_delayed_arrival :
    overrule_vessel_speed :

    """

    def __init__(
            self,
            lock_complex,
            *args,
            **kwargs,
    ):
        """Initialization"""
        self.lock_complex = lock_complex
        super().__init__(*args, **kwargs)

        self.vessel_planning = pd.DataFrame(
            index=pd.Index([]),
            columns=[
                "id",
                "node_from",
                "node_to",
                "direction",
                "waiting_area",
                "lineup_area",
                "lock_chamber",
                "L",
                "B",
                "T",
                "operation_index",
                "time_of_registration",
                "time_of_acceptance",
                "time_potential_lock_gate_opening_stop",
                "time_arrival_at_waiting_area",
                "time_arrival_at_lineup_area",
                "time_arrival_at_approach_point",
                "time_lock_operation_start",
                "time_lock_entry_start",
                "time_lock_entry_stop",
                "time_lock_departure_start",
                "time_lock_departure_stop",
                "time_lock_operation_stop",
                "time_potential_lock_gate_closure_start",
                "time_to_traverse_waterway_without_lock",
                "delay",
            ],
        )

        self.operation_planning = pd.DataFrame(
            index=pd.Index([]),
            columns=[
                "node_from",
                "node_to",
                "direction",
                "lock_chamber",
                "operation_index",
                "vessels",
                "capacity_L",
                "capacity_B",
                "time_potential_lock_gate_opening_stop",
                "time_lock_operation_start",  # See comments below
                "time_lock_entry_start",  # See comments below
                "time_lock_entry_stop",
                "time_gate_closing_start",
                "time_gate_closing_stop",
                "time_levelling_start",
                "time_levelling_stop",
                "time_gate_opening_start",
                "time_gate_opening_stop",
                "time_lock_departure_start",
                "time_lock_departure_stop",  # Note that start and stop times of different operations can overlap, but entry start and departure stop can not
                "time_lock_operation_stop",  # Operation start and stop times are solely required when leaving and entering vessels need to pass each other at the safe crossing point
                "time_potential_lock_gate_closure_start",
                "wlev_A",
                "wlev_B",
                "maximum_individual_delay",
                "total_delay",
                "status",
            ],
        )

    def register_vessel(self, vessel):
        """
        Registers a vessel to the lock operation and vessel planning

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex
        """
        # determine the orientation of the vessel to unpack the lock complex infrastructure at the correct side of the lock chamber
        if vessel.current_node == self.lock_complex.registration_nodes[0]:
            direction = 0
        else:
            direction = 1

        if not vessel.has_registered:
            operation_index = self.add_vessel_to_lock_operation_planning(vessel, direction)
        else:
            operation_index = self.vessel_planning[self.vessel_planning.id == vessel.id].iloc[-1].operation_index
        vessel_info = self.vessel_planning[self.vessel_planning.id == vessel.id].iloc[-1]
        waiting_area = self.waiting_areas[vessel_info.waiting_area]
        lock_chamber = self.lock_chambers[vessel_info.lock_chamber]
        yield from self.communicate_vessel_to_proceed_to_lock(vessel, waiting_area, lock_chamber)
        yield from self.optimize_arrival_time_previous_vessel(vessel, operation_index, lock_chamber)


    def add_vessel_to_vessel_planning(self, vessel, direction):
        """
        Adds vessel to the vessel planning of the lock complex upon request

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex
        direction : int
            the direction of the vessel: 0 (direction from node_A to node_B) or 1 (direction from node_B to node_A)
        time_of_registration : pd.Timestamp
            the time that the vessel registers to the lock master
        """

        # add vessel to the vessel planning dataframe with its information
        vessel_planning_index = len(self.vessel_planning)
        self.vessel_planning.loc[vessel_planning_index,"id"] = vessel.id
        self.vessel_planning.loc[vessel_planning_index,"node_from"] = ''
        self.vessel_planning.loc[vessel_planning_index,"node_to"] = ''
        self.vessel_planning.loc[vessel_planning_index, "direction"] = direction
        self.vessel_planning.loc[vessel_planning_index, "waiting_area"] = None
        self.vessel_planning.loc[vessel_planning_index, "lineup_area"] = None
        self.vessel_planning.loc[vessel_planning_index, "lock_chamber"] = None
        self.vessel_planning.loc[vessel_planning_index, "L"] = vessel.L
        self.vessel_planning.loc[vessel_planning_index, "B"] = vessel.B
        self.vessel_planning.loc[vessel_planning_index, "T"] = vessel.T
        self.vessel_planning.loc[vessel_planning_index, "operation_index"] = np.nan
        self.vessel_planning.loc[vessel_planning_index, "time_of_registration"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_of_acceptance"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_potential_lock_gate_opening_stop"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_arrival_at_waiting_area"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_arrival_at_lineup_area"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_arrival_at_approach_point"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_lock_operation_start"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_lock_entry_start"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_lock_entry_stop"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_lock_departure_start"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_lock_departure_stop"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_lock_operation_stop"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_potential_lock_gate_closure_start"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "time_to_traverse_waterway_without_lock"] = pd.Timestamp('NaT')
        self.vessel_planning.loc[vessel_planning_index, "delay"] = pd.Timedelta('NaT')
        vessel.registered_to_lock = True
        return vessel_planning_index


    def add_vessel_to_lock_operation_planning(self, vessel, direction):
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
        # add vessel to vessel planning and operation planning
        vessel_planning_index = self.add_vessel_to_vessel_planning(vessel, direction)
        lock_chamber_name, operation_index, new_operation = _find_available_lock_operation(self, vessel, direction)
        lock_chamber = self.lock_complex.lock_chambers[lock_chamber_name]
        waiting_area_name = _find_available_waiting_area(vessel, lock_chamber, direction)
        self.vessel_planning.loc[vessel_planning_index, 'waiting_area'] = waiting_area_name
        self.vessel_planning.loc[vessel_planning_index, 'lock_chamber'] = lock_chamber_name

        route = vessel.route
        route_goes_through_lock = False
        for edge in vessel.edge_route:
            edge_rev = (edge[1], edge[0]) + edge[2:]
            if edge == lock_chamber.edge or edge_rev == lock_chamber.edge:
                route_goes_through_lock = True
                break

        if not route_goes_through_lock:
            new_route = []
            nodes = [vessel.current_node, lock_chamber.edge[0], route[-1]]
            for i in range(len(nodes) - 1):
                segment = nx.dijkstra_path(lock_chamber.env.graph, nodes[i], nodes[i + 1])
                if i > 0:
                    segment = segment[1:]
                new_route.extend(segment)

            vessel.route = new_route
            vessel.position_on_route = 0
            vessel.edge_route = node_path_to_edge_path(vessel.env.graph, new_route)
            expanded_routes = expand_path_edges(vessel.env.graph, new_route)
            for expanded_route in expanded_routes:
                for edge in expanded_route:
                    if 'Lock chamber' not in vessel.env.graph.edges[edge].keys():
                        continue

                    lock_chambers = vessel.env.graph.edges[edge]['Lock chamber']
                    for lock_chamber_found in lock_chambers:
                        if lock_chamber_found.name == lock_chamber.name:
                            break

                    for index, routed_edge in enumerate(vessel.edge_route):
                        if routed_edge[:2] == edge[:2]:
                            vessel.edge_route[index] = edge
                            break
            waiting_area_name = _find_available_waiting_area(vessel, lock_chamber, direction)
            self.vessel_planning.loc[vessel_planning_index, 'waiting_area'] = waiting_area_name
            self.vessel_planning.loc[vessel_planning_index, 'lock_chamber'] = lock_chamber_name

        vessel_information = calculate_vessel_approach_information(self, vessel, direction)
        _update_lock_vessel_planning(self, vessel_planning_index, vessel_information)
        if new_operation:
            new_lockage_info = _check_if_empty_lock_operation_is_required(lock_chamber, operation_index + 1, direction)
            _, empty_lock_operation_to_be_requested, lock_operation_to_be_executed = new_lockage_info
            if empty_lock_operation_to_be_requested:
                _ = calculate_empty_lock_operation_information_and_update_planning(lock_chamber, operation_index - 1,
                                                                                   1 - direction)
                if lock_operation_to_be_executed:
                    self.request_empty_levelling(lock_chamber, direction)

        # information of lock chamber
        lock_operation_information = calculate_lock_operation_information_and_update_planning(lock_chamber,
                                                                                              vessel,
                                                                                              operation_index,
                                                                                              direction)
        # update the next lock operations if the previous lock operation caused a delay
        _update_future_lock_operations_by_lock_delay_previous_operation(lock_chamber, operation_index,
                                                                        lock_operation_information)

        if not route_goes_through_lock:
            vessel.has_registered = True
            raise simpy.exceptions.Interrupt('Route of vessel has changed.')

        return operation_index


    def communicate_vessel_to_proceed_to_lock(self, vessel, waiting_area, lock_chamber):
        vessel.waiting_area_request = waiting_area.resource.request()
        yield vessel.waiting_area_request

        sail_to_waiting_area = functools.partial(vessel.sail_to_waiting_area, waiting_area = waiting_area, lock_chamber = lock_chamber)
        vessel.on_pass_edge_functions.append(sail_to_waiting_area)


    def optimize_arrival_time_previous_vessel(self, vessel, operation_index, lock_chamber):
        """
        Checks the possibility of reducing the approach speed of the previous vessel to reduce the gate-open time of the lock (and applies it)
        
        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex
        operation_index : int
            index of the lock operation
        """

        # get previous vessel
        previous_vessel = _get_previous_assigned_vessel(self, operation_index)
        if not lock_chamber.minimize_gate_open_times or previous_vessel is None:
            return

        # update the vessel and operation plannings
        delay_previous_vessel = calculate_delay_previous_vessel_to_optimize_sailing_in_process(lock_chamber, vessel, previous_vessel)
        _update_operation_planning_for_delayed_arrival(self, previous_vessel, operation_index, delay_previous_vessel)
        _update_vessel_planning_for_delayed_arrival(self, previous_vessel, delay_previous_vessel)
        
        # overrule the other vessels speed by interrupting its sailing process
        yield from self.overrule_vessel_speed(previous_vessel, lock_chamber.end_node, delay_previous_vessel)

    
    def overrule_vessel_speed(self, vessel, lock_end_node, waiting_time=0., delay=0.):
        """
        Overrules the speed of an vessel based on the additional waiting time

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex
        lock_end_node : str
            the node name that forms the end node of the lock complex given the direction of the vessel
        waiting_time : float
            waiting duration in seconds
        """

        new_approach_speed_information = calculate_optimal_approach_speed_information(self, vessel, lock_end_node, waiting_time)
        if new_approach_speed_information.empty:
            return

        # store the new sailing information info in an overruled speed dataframe object for the vessel
        for edge, reversed_sailing_information_info in new_approach_speed_information.iterrows():
            vessel.overruled_speed.loc[edge] = reversed_sailing_information_info.speed

        # TODO: this communication of interrupting should be checked
        vessel.process.interrupt()
        vessel.gate_open_request.interrupt(str(delay))
        yield from []


    def request_empty_levelling(self, lock_chamber, direction):
        # TODO: check if this can be done differently
        node_of_approach, _ = _get_lock_operation_to_and_from_node(lock_chamber, direction)
        empty_levelling_process = lock_chamber.convert_chamber(new_level=node_of_approach, direction=1 - direction)
        self.env.process(empty_levelling_process)

