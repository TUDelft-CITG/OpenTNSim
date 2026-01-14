"""This is the lock module as part of the OpenTNSim package. See the locking examples in the book for detailed descriptions."""

import datetime
import math
import networkx as nx
import numpy as np
import pandas as pd
import simpy

from opentnsim.core import HasResource, Identifiable, Log, Movable, ExtraMetadata, SimpyObject, Locatable
from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from opentnsim.graph.calculations import calculate_location_over_edges, transform_geometry
from opentnsim.graph.mixins import HasMultiDiGraph, OnEdge
from opentnsim.graph.utils import (
    get_length_of_edge,
    get_trajectory,
    get_edge,
    check_graph_is_multidigraph_type,
    get_edge_at_distance_from_node,)
from opentnsim.lock.calculations import calculate_sailing_time_to_waiting_area, calculate_sailing_time_to_approach_point
from opentnsim.lock.mixins.chamber import IsLockChamber
from opentnsim.lock.mixins.master import IsLockMaster
from opentnsim.lock.utils import (
    _get_lock_object_on_registration_node,
    _get_upcoming_lock_complexes,
    _get_upcoming_locks,
)
from opentnsim.lock.visualizations import create_time_distance_plot
from opentnsim.output import HasOutput
from opentnsim.utils import inherit_docstring
from IPython.display import display

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
        self.registered_to_lock = False
        self.overruled_speed = pd.DataFrame(
            data=[], columns=["speed"], index=pd.MultiIndex.from_arrays([[], []], names=("node_start", "node_stop"))
        )

        # TODO: should not be here but in a "Vessel"-module
        if not hasattr(self.env,'vessels'):
            self.env.vessels = {}
        self.env.vessels[self.id] = self


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
        lock_complexes = _get_lock_object_on_registration_node(self.env.graph, origin)
        upcoming_lock_complexes = _get_upcoming_lock_complexes(self)
        for lock_complex in lock_complexes:
            for upcoming_lock_complex in upcoming_lock_complexes.values():
                if upcoming_lock_complex == lock_complex:
                    yield from lock_complex.register_vessel(self)


    def sail_to_waiting_area(self, origin, destination, waiting_area, lock_chamber):
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

        # if the origin of the vessel has not reached the waiting area edge, then skip this function
        edge = (origin, destination)
        if 'Waiting area' not in self.env.graph.edges[edge].keys():
            return

        waiting_areas_on_edge = self.env.graph.edges[edge]['Waiting area']
        waiting_area_found = False
        for waiting_area_on_edge in waiting_areas_on_edge:
            if waiting_area == waiting_area_on_edge:
                waiting_area_found = True
                break

        if not waiting_area_found:
            return

        # unpack the vessel and lock operation planning of the lock
        operation_planning = lock_chamber.lock_complex.operation_planning
        vessel_planning = lock_chamber.lock_complex.vessel_planning

        # determine the vessel index and operation index
        vessel_planning_index = vessel_planning[vessel_planning.id == self.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']

        # calculate the sailing duration left to the waiting area
        sailing_time_to_waiting_area, sailing_distance_to_waiting_area, vessel_speed = calculate_sailing_time_to_waiting_area(waiting_area, self)
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
        yield from self.wait_in_waiting_area(lock_chamber = lock_chamber, waiting_area=waiting_area)

        # if done waiting -> release vessel from waiting area and let vessel continue
        yield waiting_area.resource.release(self.waiting_area_request)

        # correct distance left on edge with the already covered distance through this function (to communicate with the move function)
        self.distance_left_on_edge -= sailing_distance_to_waiting_area

        # on continuing sailing to the lock complex, determine the current time and whether the vessel is the first vessel or will arrive after another vessel
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        first_in_lock = operation_planning.loc[operation_index].vessels[0] == self
        between_arrivals = False
        if not first_in_lock:
            between_arrivals = True

        # determine if the gate is closed, and when the gate are required to be open, and how long this will take (given the lock master's policy)
        direction = 0
        if lock_chamber.start_node != origin:
            direction = 1
        gate_is_closed, gate_required_to_be_open, operation_time = lock_chamber.determine_if_gate_is_closed(self, operation_index, direction, first_in_lock=first_in_lock, between_arrivals=between_arrivals)
        # if gate is open, then the vessel can continue normally
        if not gate_is_closed:
            return

        # if not, and if the time that the gate will be open lies ahead of the current time -> create a gate open request with a delay so that the gate are open at the right moment (according to the lock master's policy)
        if (gate_required_to_be_open - operation_time) > current_time:
            delay = ((gate_required_to_be_open - operation_time) - current_time).total_seconds()
            self.gate_open_request = self.env.process(lock_chamber.open_gate(to_level=lock_start_node, delay=delay, vessel=self))
            return

        # if it is already too late, the gate should open immediately -> determine the time that the gate are required to be opened again (this can include a new levelling process in case of tidal water levels)
        levelling_required = False
        if operation_time > pd.Timedelta(seconds=lock_chamber.gate_closing_time):
            levelling_required = True

        # log the gate open process and the lock levelling process if this is required TODO: this should preferably also be requested from the lock master elsewhere (especially the levelling process)
        if levelling_required:
            lock_chamber.log_entry_v0("Lock chamber converting start", gate_required_to_be_open.round('s').to_pydatetime().timestamp() - operation_time.total_seconds(), self.output.copy(),lock_start_node, )
            lock_chamber.log_entry_v0("Lock chamber converting stop", gate_required_to_be_open.round('s').to_pydatetime().timestamp() - lock_chamber.gate_opening_time, self.output.copy(),lock_end_node, )
        lock_chamber.log_entry_v0("Lock gate opening start", gate_required_to_be_open.round('s').to_pydatetime().timestamp() - lock_chamber.gate_opening_time, self.output.copy(),lock_end_node, )
        lock_chamber.log_entry_v0("Lock gate opening stop",gate_required_to_be_open.round('s').to_pydatetime().timestamp(),self.output.copy(), lock_end_node, )

        # set the new side to which the lock has been opened
        lock_chamber.gate_open = lock_chamber._directional_edge(direction)[0]

        # set the new water level for the lock if there is hydrodynamic data included in the simulation TODO: also this should preferably be included elsewhere and not here
        hydromanager = HydrodynamicDataManager()
        time_of_gate_opening = np.datetime64(gate_required_to_be_open) - np.timedelta64(int(lock_chamber.gate_opening_time))
        time_index = hydromanager._get_time_index_of_hydrodynamic_data(time_of_gate_opening)
        water_level = hydromanager._get_hydrodynamic_data_series(time_of_gate_opening, lock_chamber.gate_open, "Water level")
        if len(water_level):
            lock_chamber.water_level[time_index:] = water_level

    def wait_in_waiting_area(self, lock_chamber, waiting_area):
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

        # determine the direction of the vessel with respect to the lock complex: coming from node A (direction = 0), or from node B (direction = 1)
        if waiting_area.name == 'waiting_area_A':
            direction = 0
        else:
            direction = 1
        distance_left_on_edge = waiting_area.distance_waiting_area_to_end_edge

        # unpacks the lock complex master's vessel and lock planning
        vessel_planning = lock_chamber.lock_complex.vessel_planning
        operation_planning = lock_chamber.lock_complex.operation_planning

        # determines the vessel index and lock operation index to which the vessel is assigned -> determine how many vessels are assigned to this operation and at which time the vessel starts entering the lock
        vessel_planning_index = vessel_planning[vessel_planning.id == self.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index, 'operation_index']
        vessels_in_operation = operation_planning.loc[operation_index, 'vessels']
        start_time_entering_lock = vessel_planning.loc[vessel_planning_index, 'time_lock_operation_start']

        # determines the sailing time to reach the approach point of the lock complex
        sailing_to_approach = calculate_sailing_time_to_approach_point(lock_chamber, self, direction)

        # set the moment in time that the waiting in the waiting area has started
        waiting_start = self.env.now
        # check if vessel has to wait for other vessels (if there is a policy that a minimum number of vessels have go with each lock operation, and this criteria has yet not been matched)
        if len(vessels_in_operation) < lock_chamber.min_vessels_in_operation:
            # log the waiting event
            self.log_entry_v0("Waiting for other vessel in lock operation start", waiting_start, self.output.copy(), self.logbook[-1]['Geometry'],)

            # create a request to wait for another vessel (this is a request for a filter store: only if there are enough vessels the operation will be assigned to the store and all vessels will continue to the lock chamber)
            request = lock_chamber.wait_for_other_vessel_to_arrive.get(lambda operation: operation.operation_index == operation_index)
            # waiting in the waiting area, if request is interrupted, the vessel keeps waiting TODO: Dit stuk code hoort eigenlijk bij lockmaster.
            while len(operation_planning.loc[operation_index,'vessels']) < lock_chamber.min_vessels_in_operation:
                try:
                    yield request
                except simpy.Interrupt as e:
                    pass

            # determine the moment in time that the waiting has stopped
            waiting_stop = self.env.now

            # if the moment of the vessel starting to enter the lock has shifted, then update the vessel planning and the operation planning if it is the first assigned vessel to the lock
            if pd.Timestamp(datetime.datetime.fromtimestamp(waiting_stop)) + sailing_to_approach > start_time_entering_lock:
                # TODO functie in lock_master met input vessel.
                vessel_planning.loc[vessel_planning_index, 'time_lock_operation_start'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
                vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
                vessel_planning.loc[vessel_planning_index, 'time_lock_entry_stop'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
                vessel_index = vessels_in_operation.index(self)
                if vessel_index == 0:
                    operation_planning.loc[operation_index, 'time_entry_start'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
                vessel_planning.loc[vessel_planning_index, 'time_arrival_at_lineup_area'] += pd.Timedelta(seconds=waiting_stop - waiting_start)

            # log that the waiting has stopped
            self.log_entry_v0("Waiting for other vessel in lock operation stop", self.env.now, self.output.copy(),self.logbook[-1]['Geometry'],)

        # determine the current time (after waiting for another vessel, or not) and the time that the vessel will be at the approach point if it will continue and what was planned before
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(lock_chamber.env.now))
        time_at_approach = current_time + sailing_to_approach
        planned_start_time_entering_lock = vessel_planning.loc[vessel_planning_index, 'time_lock_operation_start']

        # determine (additional) waiting time for the vessel
        waiting_time = planned_start_time_entering_lock-time_at_approach

        # determine the waiting time that a vessel can do by decreasing it sailing speed and the waiting time that the vessel has to wait stationary in the waiting area (due to a minimum required speed for safe manoeuvrability)
        # remaining_static_waiting_time, waiting_time_while_sailing = lock_chamber.determine_waiting_time_while_sailing_to_lock(self,direction,waiting_time.total_seconds()) TODO: kijken waarom deze uitgecommand is, en of we deze toch wel willen gebruiken
        remaining_static_waiting_time = waiting_time.total_seconds()
        waiting_time_while_sailing = 0.

        # if there is stationary waiting time -> let vessel wait (longer) in the waiting area
        if remaining_static_waiting_time > 0.:
            # log the start of the waiting process
            self.log_entry_v0("Waiting for lock operation start", self.env.now, self.output.copy(), self.logbook[-1]['Geometry'], )
            # waiting in the waiting area, if request is interrupted, the vessel keeps waiting but time that vessel already has waited is subtracted
            while remaining_static_waiting_time > 0.:
                try:
                    yield lock_chamber.env.timeout(remaining_static_waiting_time)
                    time_at_approach += pd.Timedelta(seconds=remaining_static_waiting_time)
                    remaining_static_waiting_time = 0.
                    time_operation_start = vessel_planning.loc[vessel_planning_index,'time_lock_operation_start']
                    remaining_static_waiting_time = (time_operation_start-time_at_approach).total_seconds()
                except simpy.Interrupt as e:
                    remaining_static_waiting_time -= lock_chamber.env.now - waiting_start

            # log the stop of the waiting process
            self.log_entry_v0("Waiting for lock operation stop", self.env.now, self.output.copy(), self.logbook[-1]['Geometry'], )

        # if there is waiting time that can be performed while sailing, adjust sailing speed
        if waiting_time_while_sailing:
            lock_chamber.overrule_vessel_speed(self,lock_end_node,waiting_time=waiting_time_while_sailing)
            self.process.interrupt()

        self.overruled_speed.loc[waiting_area.edge, 'speed'] = lock_chamber.vessel_sailing_in_speed(self, direction)
        self.distance_left_on_edge = distance_left_on_edge


@inherit_docstring
class IsLockWaitingArea(HasResource, OnEdge, Locatable, Identifiable, Log):
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
            self,
            distance_from_edge_start,
            geometry = None,
            capacity = math.inf,
            crs_m = 'EPSG:4087',
            *args,
            **kwargs,
    ):
        """Initialization"""
        self.distance_from_edge_start = distance_from_edge_start
        super().__init__(geometry = geometry, nr_resources=capacity, *args, **kwargs)
        if geometry is None:
            self.geometry = calculate_location_over_edges(self.env.graph, self.edge, self.distance_from_edge_start, crs_m=crs_m)
        if 'Waiting area' not in self.env.graph.edges[self.edge].keys():
            self.env.graph.edges[self.edge]['Waiting area'] = [self]
        elif self not in self.env.graph.edges[self.edge]['Waiting area']:
            self.env.graph.edges[self.edge]['Waiting area'].append(self)
        self.distance_waiting_area_to_end_edge = self.env.graph.edges[self.edge]['length_m'] - distance_from_edge_start

import math
from itertools import permutations
def check_lock_complex_geometry(lock_complex):
    for node_start, node_stop in permutations(lock_complex.registration_nodes, 2):
        if node_start == node_stop:
            continue

        locks_found = {}
        for lock_chamber in lock_complex.lock_chambers.values():
            locks_found[lock_chamber.name] = False

        routes = nx.all_simple_paths(lock_complex.env.graph, node_start, node_stop)
        for path in routes:
            lock_found = False
            waiting_area_before_lock_chamber = False
            distance_waiting_area_from_edge_start = math.inf
            for edge in zip(path[:-1], path[1:]):
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

import networkx as nx
from collections import deque
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


@inherit_docstring
class IsLockComplex(SimpyObject, Identifiable, IsLockMaster):
    """Mixin-class: a lock complex object"""

    def __init__(self,
                 lock_chambers,                                 #
                 waiting_areas,                                 #
                 registration_nodes,                            #
                 lineup_areas = [],                             #
                 *args,
                 **kwargs):

        """Initialization"""
        self.registration_nodes = registration_nodes
        self.lock_chambers = {}
        for lock_chamber in lock_chambers:
            self.lock_chambers[lock_chamber.name] = lock_chamber
            lock_chamber.lock_complex = self
        self.waiting_areas = {}
        for waiting_area in waiting_areas:
            self.waiting_areas[waiting_area.name] = waiting_area
            waiting_area.lock_complex = self
        self.lineup_areas = {}
        for lineup_area in lineup_areas:
            self.lineup_areas[lineup_area.name] = lineup_area
            lineup_area.lock_complex = self
        super().__init__(lock_complex=self, *args, **kwargs)

        for registration_node in self.registration_nodes:
            if 'Lock_registration_node' not in self.env.graph.nodes[registration_node]:
                self.env.graph.nodes[registration_node]['Lock_registration_node'] = [self]
            elif self not in self.env.graph.nodes[registration_node]['Lock_registration_node']:
                self.env.graph.nodes[registration_node]['Lock_registration_node'].append(self)

        # checks
        check_lock_complex_geometry(self)
        check_all_paths_through_registration(self)

    def plot(self, lock_chamber, xlimmin=None, xlimmax=None, ylimmin=None, ylimmax=None, method = 'Matplotlib'):
        fig = create_time_distance_plot(lock_chamber, xlimmin=xlimmin, xlimmax=xlimmax, ylimmin=ylimmin, ylimmax=ylimmax, method = method)
        return fig

        # # set power used to pass lock
        # # TODO: should maybe be added to the vessels
        # self.P_used_to_break_before_lock = None
        # self.P_used_to_break_in_lock = None
        # self.P_used_to_accelerate_in_lock = None
        # self.P_used_to_accelerate_after_lock = None
        # self.crs_m = crs_m
        #
        #
        #
        # self.distance_waiting_area_A_from_edge_start_waiting_area_A = self.distance_from_start_node_to_lock_gate_A - self.distance_lock_gate_A_to_waiting_area_A
        # if edge_waiting_area_A != (node_A, node_B):
        #     geometry_edge_start_waiting_area_A_to_lock_node_A = get_trajectory(self.env.graph,
        #                                                                        edge_waiting_area_A[0], node_A)
        #     geometry_edge_start_waiting_area_A_to_lock_node_A_m = transform_geometry(geometry_edge_start_waiting_area_A_to_lock_node_A, epsg_out=crs_m)
        #     self.distance_waiting_area_A_from_edge_start_waiting_area_A = geometry_edge_start_waiting_area_A_to_lock_node_A_m.length - self.distance_lock_gate_A_to_waiting_area_A
        #
        # self.waiting_area_A = IsLockWaitingArea(env=self.env,
        #                                         name="waiting_area_A",
        #                                         lock=self,
        #                                         edge=edge_waiting_area_A,
        #                                         distance_from_edge_start=self.distance_waiting_area_A_from_edge_start_waiting_area_A,
        #                                         crs_m = self.crs_m)
        # self.distance_waiting_area_A_to_end_edge_waiting_area_A = get_length_of_edge(self.env.graph, edge_waiting_area_A)
        # self.distance_waiting_area_A_to_end_edge_waiting_area_A -= self.distance_waiting_area_A_from_edge_start_waiting_area_A
        #
        # if edge_waiting_area_B is None:
        #     edge_waiting_area_B = (self.end_node, self.start_node)
        #
        # self.distance_waiting_area_B_from_start_edge_waiting_area_B = self.distance_from_end_node_to_lock_gate_B - self.distance_lock_gate_B_to_waiting_area_B
        # if edge_waiting_area_B !=(node_B, node_A):
        #     geometry_edge_start_waiting_area_B_to_lock_node_B = get_trajectory(self.env.graph,
        #                                                                        edge_waiting_area_B[0],node_B)
        #     geometry_edge_start_waiting_area_B_to_lock_node_B_m = transform_geometry(geometry_edge_start_waiting_area_B_to_lock_node_B, epsg_out=crs_m)
        #     self.distance_waiting_area_B_from_start_edge_waiting_area_B = geometry_edge_start_waiting_area_B_to_lock_node_B_m.length - self.distance_lock_gate_B_to_waiting_area_B
        #
        # self.waiting_area_B = IsLockWaitingArea(env=self.env,
        #                                         name="waiting_area_B",
        #                                         lock=self,
        #                                         edge=edge_waiting_area_B,
        #                                         distance_from_edge_start=self.distance_waiting_area_B_from_start_edge_waiting_area_B,
        #                                         crs_m = self.crs_m)
        # self.distance_waiting_area_B_to_end_edge_waiting_area_B = get_length_of_edge(self.env.graph, edge_waiting_area_B)
        # self.distance_waiting_area_B_to_end_edge_waiting_area_B -= self.distance_waiting_area_B_from_start_edge_waiting_area_B
        #
        #
        #
        # # create the line-up area at side A if there is a line-up area at side A (lineup_area_A_length is not None)
        # self.has_lineup_area_A = False
        # if lineup_area_A_length is not None:
        #     self.has_lineup_area_A = True
        #     self.lineup_area_A_length = lineup_area_A_length
        #     self.effective_lineup_area_A_length = effective_lineup_area_A_length
        #     self.passing_allowed_in_lineup_area_A = passing_allowed_in_lineup_area_A
        #     self.speed_reduction_factor_lineup_area_A = speed_reduction_factor_lineup_area_A
        #
        #     # the effective line-up length should at least be equal to the lock length TODO: set warning?
        #     if lineup_area_A_length < self.lock_length and not effective_lineup_area_A_length:
        #         self.effective_lineup_area_A_length = self.lock_length
        #
        #     self.distance_lock_gate_A_to_lineup_area_A = distance_lock_gate_A_to_lineup_area_A
        #
        #     # get the edge at which the line-up area is located TODO: can a lock complex be located along multiple edges?
        #     distance_from_start_node_to_lineup_A = self.distance_lock_gate_A_to_lineup_area_A - self.distance_from_start_node_to_lock_gate_A
        #     edge_lineup_area_A = get_edge_at_distance_from_node(self.env, self.start_node, self.node_A,
        #                                                         distance_from_start_node_to_lineup_A)
        #     route_to_lineup_area_A = nx.dijkstra_path(self.env.graph, self.start_node, edge_lineup_area_A[1]) # TODO: can a lock complex be located along multiple edges?
        #     distance_start_node_to_node_waiting_area_A, _ = get_sailing_distance(self.env.graph, route_to_lineup_area_A)
        #     self.distance_lineup_area_A_from_edge_lineup_area_A_start = distance_start_node_to_node_waiting_area_A - (self.distance_lock_gate_A_to_lineup_area_A - self.distance_from_start_node_to_lock_gate_A)
        #
        #     # create lineup area A object
        #     self.lineup_area_A = IsLockLineUpArea(env=self.env,
        #                                           name=self.name,
        #                                           start_node=edge_lineup_area_A[1],
        #                                           end_node=edge_lineup_area_A[0],
        #                                           lineup_area_length=self.lineup_area_A_length,
        #                                           distance_from_start_edge=self.distance_lineup_area_A_from_edge_lineup_area_A_start,
        #                                           effective_lineup_area_length=self.effective_lineup_area_A_length,
        #                                           passing_allowed=False,
        #                                           speed_reduction_factor=0.75)
        #
        # # create the line-up area at side B if there is a line-up area at side B (lineup_area_B_length is not None)
        # self.has_lineup_area_B = False
        # if lineup_area_B_length is not None:
        #     self.has_lineup_area_B = True
        #     self.lineup_area_B_length = lineup_area_B_length
        #     self.effective_lineup_area_B_length = effective_lineup_area_B_length
        #     self.passing_allowed_in_lineup_area_B = passing_allowed_in_lineup_area_B
        #     self.speed_reduction_factor_lineup_area_B = speed_reduction_factor_lineup_area_B
        #
        #     # the effective line-up length should at least be equal to the lock length TODO: set warning?
        #     if lineup_area_B_length < self.lock_length and not effective_lineup_area_B_length:
        #         self.effective_lineup_area_B_length = self.lock_length
        #
        #     self.distance_lock_gate_B_to_lineup_area_B = distance_lock_gate_B_to_lineup_area_B
        #
        #     # get the edge at which the line-up area is located TODO: can a lock complex be located along multiple edges?
        #     distance_from_end_node_to_lineup_B = self.distance_lock_gate_B_to_lineup_area_B - self.distance_from_end_node_to_lock_gate_B
        #     edge_lineup_area_B = get_edge_at_distance_from_node(self.env, self.end_node, self.node_B,
        #                                                         distance_from_end_node_to_lineup_B)
        #
        #     route_to_lineup_area_B = nx.dijkstra_path(self.env.graph, self.end_node, edge_lineup_area_B[1]) #TODO: can a lock complex be located along multiple edges?
        #     distance_end_node_to_node_waiting_area_B = provide_sailing_distance_over_route(route_to_lineup_area_B)["Distance"].sum()
        #     self.distance_lineup_area_B_from_edge_lineup_area_B_start = distance_end_node_to_node_waiting_area_B - (self.distance_lock_gate_B_to_lineup_area_B - self.distance_from_end_node_to_lock_gate_B)
        #
        #     # create lineup area B object
        #     self.lineup_area_B = IsLockLineUpArea(env=self.env,
        #                                           name=self.name,
        #                                           start_node=edge_lineup_area_B[1],
        #                                           end_node=edge_lineup_area_B[0],
        #                                           distance_from_start_edge=self.distance_lineup_area_B_from_edge_lineup_area_B_start,
        #                                           lineup_area_length=0.,
        #                                           effective_lineup_area_length=0.,
        #                                           passing_allowed=False,
        #                                           speed_reduction_factor=0.75)

    def plot(self, lock_chamber, xlimmin=None, xlimmax=None, ylimmin=None, ylimmax=None, method = 'Matplotlib'):
        fig = create_time_distance_plot(lock_chamber, xlimmin=xlimmin, xlimmax=xlimmax, ylimmin=ylimmin, ylimmax=ylimmax, method = method)
        return fig
