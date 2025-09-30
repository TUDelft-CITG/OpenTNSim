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

# spatial libraries
from collections import namedtuple
import simpy
import xarray as xr
# from netCDF4 import Dataset
from IPython.display import display

from opentnsim import output, graph
from opentnsim import core
from opentnsim.core import HasResource, Identifiable, Log, Movable, HasLength, SimpyObject, ExtraMetadata
from opentnsim.graph import HasMultiDiGraph
from opentnsim.output import HasOutput

# Constants
knots_to_ms = knots = 0.514444444
gravitational_acceleration = 9.81


def _get_lock_on_node(multidigraph, registration_node):
    """Get the lock complex object that is associated with a registration node node

    Parameters
    ----------
    registration_node : str
        node name (that has to be in the graph) on which the vessel is currently starting to navigate an edge

    Returns
    -------
    lock : Union(class, None)
        the lock complex object that is associated with the registration node, or None if no lock complex is associated with the registration node
    """
    # check if node is a registration node
    if "Lock_registration_node" not in multidigraph.nodes[registration_node].keys():
        return None

    edge = multidigraph.nodes[registration_node]["Lock_registration_node"]
    # return lock if it exists on the edge
    if "Lock" in multidigraph.edges[edge].keys():
        lock = multidigraph.edges[edge]["Lock"][0]
        return lock
    # Return None if no lock exists on the edge
    else:
        return None


class HasLockPlanning:
    """This class keeps track of the lock-planning of a lock-master."""

    def __init__(self, *args, **kwargs):
        self.vessel_planning = pd.DataFrame(
            index=pd.Index([]),
            columns=[
                "id",
                "bound",
                "L",
                "B",
                "T",
                "operation_index",
                "time_of_registration",
                "time_of_acceptance",
                "time_arrival_at_waiting_area",
                "time_arrival_at_lineup_area",
                "time_lock_passing_start",
                "time_lock_entry_start",
                "time_lock_entry_stop",
                "time_lock_departure_start",
                "time_lock_departure_stop",
                "time_lock_passing_stop",
            ],
        )
        self.operation_planning = pd.DataFrame(
            index=pd.Index([], name="lock_operation"),
            columns=[
                "bound",
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
        print(f"output: {vessels}")
        return vessels


class PassesLockComplex(Movable, HasMultiDiGraph):
    """Mixin class: Something that passes a lock complex (i.e., can be added to a vessel-object)

    Parent classes
    --------------
    Movable :
        to be able to pass edges and nodes of the graph
    HasMultiDiGraph :
        a networkx.MultiDiGraph is constructed where edges are constructed with a start_node, end_node, and an
        identifier (k) to be able to construct multiple edges between the same node pair (i.e., parallel lock chambers)


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
            data=[], columns=["Speed"], index=pd.MultiIndex.from_arrays([[], [], []], names=("node_start", "node_stop", "k"))
        )

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
            lock = self.multidigraph.edges[lock_edge]["Lock"][
                0
            ]  # TODO: write test to prevent that multiple lock complexes are located at the same registration node, also: maybe we need to change "Lock" to "Lock complex"

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
            k = sorted(self.multidigraph[node_start][node_stop],key=lambda x: self.multidigraph[node_start][node_stop][x]['geometry'].length)[0] #TODO: k-berekening in een functie zetten (nu bepaald op minste lengte, maar sluismeester moet/kan dit bepalen).
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

        TODO: origin hoeft geen input te zijn. Kan ook met self.current_node (comment: dit nagaan, want deze generator wordt toegevoegd aan lijst 'on_pass_node_functions', welke in de movable met input = origin wordt gevoed)
        """

        # find the lock complex object that is associated with the registration node
        lock = _get_lock_on_node(self.multidigraph, origin)
        # if a lock complex object is found, request registration to the lock master of the lock complex
        if lock:
            yield from lock.register_vessel(self)

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
        # k = sorted(self.multidigraph[node_start][node_stop],key=lambda x: self.multidigraph[node_start][node_stop][x]['geometry'].length)[0] #TODO: k-berekening in een functie zetten (nu bepaald op minste lengte, maar sluismeester moet/kan dit bepalen).
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
            operation_planning = lock.lock_complex.operation_planning
            vessel_planning = lock.lock_complex.vessel_planning

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

            # let vessel wait in the waiting area
            yield from self.wait_in_waiting_area(waiting_area=waiting_area)

            # if done waiting -> release vessel from waiting area and let vessel continue
            yield waiting_area.waiting_area.release(self.waiting_area_request)

            # vessel is now allowed to continue passing the lock -> create vessel specific functions and add those function to the functions that communicate with the move function
            allow_vessel_to_sail_in_lock = functools.partial(lock.allow_vessel_to_sail_in_lock, vessel=self)
            initiate_levelling = functools.partial(lock.initiate_levelling, vessel=self)
            allow_vessel_to_sail_out_of_lock = functools.partial(lock.allow_vessel_to_sail_out_of_lock, vessel=self)
            self.on_pass_edge_functions.append(allow_vessel_to_sail_in_lock)
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
            door_is_closed, doors_required_to_be_open, operation_time = lock.determine_if_door_is_closed(self,
                                                                                                         operation_index,
                                                                                                         direction,
                                                                                                         first_in_lock=first_in_lock,
                                                                                                         between_arrivals=between_arrivals)
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
                lock.log_entry_v0("Lock chamber converting stop", doors_required_to_be_open.round('s').to_pydatetime().timestamp() - lock.doors_opening_time, self.output.copy(),lock_start_node, )
            lock.log_entry_v0("Lock doors opening start", doors_required_to_be_open.round('s').to_pydatetime().timestamp() - lock.doors_opening_time, self.output.copy(),lock_start_node, )
            lock.log_entry_v0("Lock doors opening stop",doors_required_to_be_open.round('s').to_pydatetime().timestamp(),self.output.copy(), lock_start_node, )

            # set the new side to which the lock has been opened
            if not direction:
                lock.node_open = lock.start_node
            else:
                lock.node_open = lock.end_node

            # set the new water level for the lock if there is hydrodynamic data included in the simulation TODO: also this should preferably be included elsewhere and not here
            if self.env.vessel_traffic_service.hydrodynamic_information_path:
                time_index = np.absolute(hydrodynamic_times - np.datetime64(doors_required_to_be_open) - np.timedelta64(int(lock.doors_opening_time), 's')).argmin()
                station_index = np.where(np.array(list((hydrodynamic_data['STATION']))) == lock.node_open)[0]
                lock.water_level[time_index:] = hydrodynamic_data['Water level'][station_index, time_index:]

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
        start_node = waiting_area.edge[0]
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
        sailing_to_approach = lock.calculate_sailing_time_to_approach_point(self, direction, current_node=start_node,overwrite=False)# - lock.calculate_sailing_time_to_waiting_area(self, direction, overwrite=False)[0]

        # set the moment in time that the waiting in the waiting area has started
        waiting_start = self.env.now

        # check if vessel has to wait for other vessels (if there is a policy that a minimum number of vessels have go with each lock operation, and this criteria has yet not been matched)
        if len(vessels_in_operation) < lock.min_vessels_in_operation:
            # log the waiting event
            self.log_entry_v0("Waiting for other vessel in lock operation start", waiting_start, self.output.copy(), self.logbook[-1]['Geometry'],)

            # create a request to wait for another vessel (this is a request for a filter store: only if there are enough vessels the operation will be assigned to the store and all vessels will continue to the lock chamber)
            request = lock.wait_for_other_vessel_to_arrive.get(lambda operation: operation.operation_index == operation_index)
            # waiting in the waiting area, if request is interrupted, the vessel keeps waiting TODO: Dit stuk code hoort eigenlijk bij lockmaster.
            while operation_planning.loc[operation_index,'status'] == 'waiting for vessel':
                try:
                    yield request
                    operation_planning.loc[operation_index,'status'] = 'ready'
                except simpy.Interrupt as e:
                    operation_planning.loc[operation_index,'status'] = 'waiting for vessel'

            # determine the moment in time that the waiting has stopped
            waiting_stop = self.env.now

            # empty the overruled speed dataframe, as vessel will sail to the lock with the correct speed to minimize the door-open time of the lock
            self.overruled_speed = self.overruled_speed.iloc[0:0]

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

        # set that the lock operation is now ready to be operated
        operation_planning.loc[operation_index, 'status'] = 'ready'

        self.distance_left_on_edge = distance_left_on_edge


class IsLockWaitingArea(HasResource, Identifiable, Log, HasOutput, HasMultiDiGraph):
    """Mixin class: lock complex has waiting area object:

    creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity

    Parent classes
    --------------
    HasResource :
        to be able to pass edges and nodes of the graph
    Identifiable :
        to be identifiable (id)
    Log :
        to maintain a logbook
    HasOutput :
        to keep track of specific output
    HasMultiDiGraph :
        a networkx.MultiDiGraph is constructed where edges are constructed with a start_node, end_node, and an
        identifier (k) to be able to construct multiple edges between the same node pair (i.e., parallel lock chambers)

    Attributes
    ----------
    none

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


class IsLockChamber(HasResource, HasLength, Identifiable, Log, HasOutput, HasMultiDiGraph, ExtraMetadata):
    """Mixin class: lock complex has a lock chamber:

    creates a lock chamber with a resource which is requested when a vessels wants to enter the area with limited capacity

    Parent classes
    --------------
    HasResource :
        to be able to pass edges and nodes of the graph
    HasLength :
        to have a length that can be requested by a vessel
    Identifiable :
        to be identifiable (id)
    Log :
        to maintain a logbook
    HasOutput :
        to keep track of specific output
    HasMultiDiGraph :
        a networkx.MultiDiGraph is constructed where edges are constructed with a start_node, end_node, and an
        identifier (k) to be able to construct multiple edges between the same node pair (i.e., parallel lock chambers)
    ExtraMetadata :
        to have extra parameters

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
        # start_node,  # a string which indicates the location of the first pair of lock doors
        # end_node,  # a string which indicates the location of the second pair of lock doors
        # lock_length,  # a float which contains the length of the lock chamber
        # lock_width,  # a float which contains the width of the lock chamber
        # lock_depth,  # a float which contains the depth of the lock chamber
        # k=0,  # a int which is the identifier of the edge between two nodes in a multidigraph network
        # distance_from_start_node_to_lock_doors_A=0.0,  # a float that is the distance between the start_node of the edge and the lock doors A [m]
        # distance_from_end_node_to_lock_doors_B=0.0,  # a float that is the distance between the end_node of the edge and the lock doors B [m]
        # registration_nodes=[],  # a list of str with the node names at which the vessels request registration to the lock complex master
        # doors_opening_time=300.0,  # a float which contains the time it takes to open the doors [s]
        # doors_closing_time=300.0,  # a float which contains the time it takes to close the doors [s]
        # disch_coeff=0.0,  # a float which contains the discharge coefficient of filling system
        # opening_area=0.0,  # a float which contains the cross-sectional area of filling system [m^2]
        # opening_depth=0.0,  # a float which contains the depth at which filling system is located [m^2]
        # speed_reduction_factor_lock_chamber=0.3,  # a float that is the reduction factor for the vessel speed from its original speed when entering the lock
        # start_sailing_out_time_after_doors_have_been_opened=0.0,  # a float that is the time that the vessel wait to start sailing out of the lock after the doors have been opened after levelling [s]
        # sailing_time_before_opening_lock_doors=600.0,  # a float that is the time that the doors are opened before a vessel arrives at the doors [s]
        # sailing_time_before_closing_lock_doors=120.0,  # a float that is the time that the doors are closed after a vessel has sailed through the doors [s]
        start_node,                                                         # a string which indicates the location of the first pair of lock doors
        end_node,                                                           # a string which indicates the location of the second pair of lock doors
        lock_length,                                                        # a float which contains the length of the lock chamber
        lock_width,                                                         # a float which contains the width of the lock chamber
        lock_depth,                                                         # a float which contains the depth of the lock chamber
        k=0,                                                                # a int which is the identifier of the edge between two nodes in a multidigraph network
        distance_from_start_node_to_lock_doors_A=0.0,                       # a float that is the distance between the start_node of the edge and the lock doors A [m]
        distance_from_end_node_to_lock_doors_B=0.0,                         # a float that is the distance between the end_node of the edge and the lock doors B [m]
        registration_nodes=[],                                                  # a list of str with the node names at which the vessels request registration to the lock complex master
        doors_opening_time=300.0,                                           # a float which contains the time it takes to open the doors [s]
        doors_closing_time=300.0,                                           # a float which contains the time it takes to close the doors [s]
        disch_coeff=0.4,                                                    # a float which contains the discharge coefficient of filling system
        opening_area=12.0,                                                  # a float which contains the cross-sectional area of filling system [m^2]
        opening_depth=None,                                                 # a float which contains the depth at which filling system is located [m^2]
        speed_reduction_factor_lock_chamber=0.3,                            # a float that is the reduction factor for the vessel speed from its original speed when entering the lock
        start_sailing_out_time_after_doors_have_been_opened=0.0,            # a float that is the time that the vessel wait to start sailing out of the lock after the doors have been opened after levelling [s]
        sailing_time_before_opening_lock_doors=600.,                        # a float that is the time that the doors are opened before a vessel arrives at the doors [s]
        sailing_time_before_closing_lock_doors=120.,                        # a float that is the time that the doors are closed after a vessel has sailed through the doors [s]
        minimum_time_between_operations_for_intermediate_door_closure=0.0,  # a float that is the minimum required time between lock operations that the lock doors can be both closed (to reduce salt intrusion) [s]
        sailing_distance_to_crossing_point=500.0,  # a float that is the distance at which vessels can safely pass each other in front of the lock (last vessel that sails out and first vessel that sails in) [m]
        passage_time_door=300.0,  # a float [s] ?
        sailing_in_time_gap_through_doors=180.0,  # a float that is the time gap after which the next vessel can sail into the lock through the lock doors (after another vessel has sailed through to enter the lock) [s]
        sailing_out_time_gap_through_doors=180.0,  # a float that is the time gap after which the next vessel can sail out of the lock through the lock doors (after another vessel has sailed through to leave the lock)[s]
        sailing_in_time_gap_after_berthing_previous_vessel=0.0,  # a float that is the time gap after which the next vessel can sail into the lock (after another vessel has berthed) [s]
        sailing_out_time_gap_after_berthing_previous_vessel=0.0,  # a float that is the time gap after which the next vessel can sail out of the lock (after another vessel has deberthed) [s]

        sailing_in_speed_A=2 * knots,  # a float that is the speed at which the vessel sails into the lock to the sea side [m/s]
        sailing_out_speed_A=2
        * knots,  # a float that is the speed at which the vessel sails out of the lock to the sea side [m/s]
        sailing_in_speed_B=2
        * knots,  # a float that is the speed at which the vessel sails into the lock to the canal side [m/s]
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
        super().__init__(lock_complex=self, capacity=100, length=lock_length, remaining_length=lock_length, *args, **kwargs)
        if self.env.vessel_traffic_service.hydrodynamic_information_path is not None:
            global hydrodynamic_data
            if isinstance(self.env.vessel_traffic_service.hydrodynamic_information_path,str):
                hydrodynamic_data = Dataset(self.env.vessel_traffic_service.hydrodynamic_information_path)
            else:
                hydrodynamic_data = self.env.vessel_traffic_service.hydrodynamic_information
            global hydrodynamic_times
            if isinstance(self.env.vessel_traffic_service.hydrodynamic_information_path, str):
                hydrodynamic_times = hydrodynamic_data['TIME'][:].data.astype("timedelta64[m]") + self.env.vessel_traffic_service.hydrodynamic_start_time
            else:
                hydrodynamic_times = hydrodynamic_data['TIME'][:]

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
        length_edge = edge_info['length_m']
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

        if self.env.vessel_traffic_service.hydrodynamic_information_path is not None:
            if isinstance(hydrodynamic_data,xr.Dataset):
                station_index = np.where(np.array(list((hydrodynamic_data['STATION'].values))) == self.node_open)[0][0]
                water_level = hydrodynamic_data['Water level'][station_index][0].values
                self.water_level = np.ones(len(hydrodynamic_data['Water level'][station_index].values)) * water_level
            else:
                station_index = np.where(np.array(list((hydrodynamic_data['STATION']))) == self.node_open)[0]
                water_level = hydrodynamic_data['Water level'][station_index][0][0]
                self.water_level = np.ones(len(hydrodynamic_data['Water level'][station_index,:][0]))*water_level

        # TODO: in functie zetten.
        # TODO: In de documentatie zetten dat detecotr nodes op volgorde moeten komen. En ook een assert maken.
        for registration_node, lock_edge in zip(self.registration_nodes,[(self.start_node,self.end_node,self.k),(self.end_node,self.start_node,self.k)]):
            if 'Lock_registration_node' not in self.multidigraph.nodes[registration_node]:
                self.multidigraph.nodes[registration_node]['Lock_registration_node'] = lock_edge

        # Add to the graph:
        # TODO: In losse functie (add_lock_to_graph)
        if "graph" in dir(self.env):
            k = sorted(self.multidigraph[self.start_node][self.end_node],
                       key=lambda x: self.multidigraph[self.start_node][self.end_node][x]['geometry'].length)[0]
            # Add the lock to the edge or append it to the existing list
            if "Lock" not in self.multidigraph.edges[self.start_node, self.end_node, k].keys():
                self.multidigraph.edges[self.start_node, self.end_node, k]["Lock"] = [self]
                self.multidigraph.edges[self.end_node, self.start_node, k]["Lock"] = [self]
            else:
                self.multidigraph.edges[self.start_node, self.end_node, k]["Lock"].append(self)
                self.multidigraph.edges[self.end_node, self.start_node, k]["Lock"].append(self)

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
        if not direction:
            edge = (self.start_node,self.end_node,self.k)
            distance_to_lock_doors = self.distance_from_start_node_to_lock_doors_A
        elif direction:
            edge = (self.end_node,self.start_node,self.k)
            distance_to_lock_doors = self.distance_from_end_node_to_lock_doors_B

        # determine the speed of the vessel over the edge
        speed = self.env.vessel_traffic_service.provide_speed_over_edge(vessel, edge)

        # if there is an overruled speed on the edge, use this speed TODO: checken of dit nu goed gaat. De indentatie is veranderd.
        if "overruled_speed" in dir(vessel) and not vessel.overruled_speed.empty:
            if edge in vessel.overruled_speed.index:
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
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
        P_used : float
            the breaking power used by the vessel to gradually decelerate [kW]
        until_crossing_point : bool


        Returns
        -------
        speed : float
            the average speed in the lock from the lock doors to the location of berthing

        """
        # determine the edge on which the vessel is sailing and the distance to the lock doors
        edge = None
        distance_to_exit = 0.
        if not direction:
            edge = (self.start_node, self.end_node, self.k)
            distance_to_exit = self.distance_from_end_node_to_lock_doors_B
        elif direction:
            edge = (self.end_node, self.start_node, self.k)
            distance_to_exit = self.distance_from_start_node_to_lock_doors_A

        # determine the speed of the vessel over the edge
        speed = speed_edge = self.env.vessel_traffic_service.provide_speed_over_edge(vessel, edge)

        # if there is an overruled speed on the edge, use this speed TODO: checken of dit nu goed gaat. De indentatie is veranderd.
        if 'overruled_speed' in dir(vessel) and not vessel.overruled_speed.empty:
            if edge in vessel.overruled_speed.index:
                speed = vessel.overruled_speed.loc[edge, 'Speed']

        return speed

    def determine_levelling_time(self, t_start, direction=None, wlev_init=None, same_direction=False, prediction=False):
        """
        Calculates the levelling time of a lock operation

        Parameters
        ----------
        t_start :
            the start time of the levelling process
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
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

        # set default time and water level difference series
        dt = self.time_step
        t_final = 3600 # maximum levelling time has been set to an hour
        t = np.arange(0, t_final + float(dt), float(dt))
        z = np.zeros_like(t)

        # if there is no hydrodynamic data included in the run, use the constant levelling time included in the lock object
        if self.env.vessel_traffic_service.hydrodynamic_information_path is None:
            levelling_time = self.levelling_time
            return levelling_time, t, z

        # convert given t_start into np.datetime64 (this is required to communicate with the hydrodynamic data via the NetCDF package)
        if isinstance(t_start,float):
            t_start = np.datetime64(datetime.datetime.fromtimestamp(t_start))
        elif isinstance(t_start,datetime.datetime):
            t_start = np.datetime64(t_start)
        elif isinstance(t_start,pd.Timestamp):
            t_start = np.array([t_start], dtype=np.datetime64)[0]

        # if there is hydrodynamic data, unpack the water levels at the nodes of the lock complex
        if self.env.vessel_traffic_service.hydrodynamic_information_path is not None:
            if isinstance(hydrodynamic_data, xr.Dataset):
                stationA_index = np.where(np.array(list((hydrodynamic_data['STATION'].values))) == self.start_node)[0][0]
                stationB_index = np.where(np.array(list((hydrodynamic_data['STATION'].values))) == self.end_node)[0][0]
                H_A = hydrodynamic_data["Water level"][stationA_index].values.copy()
                H_B = hydrodynamic_data["Water level"][stationB_index].values.copy()
            else:
                stationA_index = np.where(np.array(list((hydrodynamic_data['STATION']))) == self.start_node)[0][0]
                stationB_index = np.where(np.array(list((hydrodynamic_data['STATION']))) == self.end_node)[0][0]
                H_A = hydrodynamic_data["Water level"][stationA_index].copy()
                H_B = hydrodynamic_data["Water level"][stationB_index].copy()

        # determine the direction
        if direction is None:
            if self.node_open == self.start_node:
                direction = 0
            else:
                direction = 1

        # if the lock operation is in the same direction (?), swap the direction
        if same_direction:
            direction = 1 - direction

        # determine the actual water levels
        time_index = np.absolute(hydrodynamic_times - t_start).argmin()
        H_A_init = H_A[time_index]
        H_B_init = H_B[time_index]

        if wlev_init is None:
            wlev_init = self.water_level[time_index]

        if not direction:
            z[0] = H_B_init - wlev_init

        else:
            z[0] = H_A_init - wlev_init

        # if a function has been included to predict the levelling time based on the water level difference: calculate the levelling time based on the initial water level difference
        if callable(self.levelling_time):
            levelling_time = self.levelling_time(z[0])
            return levelling_time, t, z

        # if no function has been included: compute the levelling time based on Eq. 4.64 of Ports and Waterways Open Textbook (https://books.open.tudelft.nl/home/catalog/book/204)
        A_ch = self.lock_length * self.lock_width # surface area of the lock chamber [m^2] (constant over time)
        m = self.disch_coeff # discharge coefficient [-] (constant over time)
        g = gravitational_acceleration # gravitational acceleration [m/(s^2)] (constant over time)
        T1 = self.gate_opening_time # time to open the gate [s] (constant over time)
        A_s = np.linspace(0, self.opening_area, int(T1 / float(dt))) # sluice opening area over time when opening [m^2] (time-dependent)
        A_s = np.append(A_s, [self.opening_area] * (len(z) - len(A_s))) # sluice opening over full levelling process [m^2] (time-dependent)
        H_time = hydrodynamic_times.astype(float) # time series of the hydrodynamic data [s]

        # time-integration by (self-coded) Euler's method TODO Checken of we een standaard solver kunnen gebruiken. En of we dit algoritme los kunnen maken van de klasse.
        for i in range(len(t) - 1):
            H_Ai = np.interp((np.timedelta64(int(i * float(dt) * 10 ** 6), 'us') + t_start - np.datetime64('1970-01-01')) / np.timedelta64(1, 'us'), H_time, H_A) # water level at side A at time = i
            H_Aii = np.interp((np.timedelta64(int((i + 1) * float(dt) * 10 ** 6), 'us') + t_start - np.datetime64('1970-01-01')) / np.timedelta64(1, 'us'), H_time, H_A) # water level at side A at time = i + 1
            H_Bi = np.interp((np.timedelta64(int(i * float(dt) * 10 ** 6), 'us') + t_start - np.datetime64('1970-01-01')) / np.timedelta64(1, 'us'), H_time, H_B) # water level at side B at time = i
            H_Bii = np.interp((np.timedelta64(int((i + 1) * float(dt) * 10 ** 6), 'us') + t_start - np.datetime64('1970-01-01')) / np.timedelta64(1, 'us'), H_time, H_B) # water level at side B at time = i + 1
            deltaH_A = H_Aii - H_Ai # water level difference at side A between time = i and time = i + 1
            deltaH_B = H_Bii - H_Bi # water level difference at side B between time = i and time = i + 1

            # determine the contribution to the change in water level difference outside of the lock (i.e., due to tides) in the water level difference at time = i + 1
            if not direction:
                to_wlev_change = - deltaH_B
            else:
                to_wlev_change = - deltaH_A

            # calculate change in water level difference between time = i and time = i + 1
            z_i = abs(z[i])  # absolute water level difference at time = i

            dz_dt = -m * A_s[i] * np.sqrt(2 * g * np.max([0, z_i])) / A_ch # change in water level difference over time [m/s]
            if z[i] < 0: # correct if water level difference is negative
                dz_dt = -dz_dt
            dz = dz_dt * float(dt) + to_wlev_change

            # calculate the new water level difference at time = i + 1
            z[i + 1] = z[i] + dz
            if np.sign(z[i + 1]) != np.sign(z[i]): # prevents overshooting of the water level difference
                z[i + 1] = 0

            if np.abs(z[i + 1]) <= self.water_level_difference_limit_to_open_doors: # breaks the integration if the water level difference is smaller than a default 5 cm (the last 5 cm of water level difference takes long to overcome, so lock master opens doors)
                z[(i + 1):] = np.nan # set all next values of the water level series to nan
                break

        # determining levelling time based on the first nan of the series TODO: Class-functie maken _determine_levelling_time()
        if len(np.argwhere(np.isnan(z))):
            levelling_time = t[np.argwhere(np.isnan(z))[0]][0]
        else:
            levelling_time = t[-1]

        # if this function was not ran as a prediction, but rather as the actual levelling event: update the water level time series of the lock chamber
        if not prediction:
            # TODO: de self.water_level wordt niet gebruikt, maar is wel leuk om als logging terug te zien na een berekening. Nadenken of we dat zo willen laten, of anders willen bijhouden.
            if isinstance(hydrodynamic_data, xr.Dataset):
                t_index_final = np.absolute(hydrodynamic_times - (t_start + np.timedelta64(int(levelling_time), 's'))).argmin().values
            else:
                t_index_final = np.absolute(hydrodynamic_times - (t_start+np.timedelta64(int(levelling_time),'s'))).argmin()
            if not direction:
                self.water_level[t_index_final:] = H_B[t_index_final:].copy()
            else:
                self.water_level[t_index_final:] = H_A[t_index_final:].copy()

        return levelling_time, t, z


class IsLockMaster(SimpyObject, HasLockPlanning):
    """Mixin class: lock complex has a lock master:

    Creates a lock master that schedules the vessels into lock operations

    Parent classes
    --------------
    SimpyObject :
        to be able to pass edges and nodes of the graph

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

    allow_vessel_to_sail_in_lock :

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
    calculate_vessel_entry_start_time :
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

        Yields
        ------
        nothing

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
        add_operation = False
        available_operations = pd.DataFrame()
        operation_index, add_operation, available_operations = self.assign_vessel_to_lock_operation(vessel, direction)

        # if available operation have been identified, add vessel to one of these lock operations (already done when lock master planned for the long-term: available_operations will be empty)
        if not available_operations.empty:
            operation_index = available_operations.iloc[0].name
            copy_operation_planning = operation_planning.copy()
            copy_vessel_planning = vessel_planning.copy()
            yield from self.add_vessel_to_planned_lock_operation(vessel, operation_index, direction,vessel_planning=copy_vessel_planning,operation_planning=copy_operation_planning)
            if copy_operation_planning[copy_operation_planning.index >= operation_index].maximum_individual_delay.max() > pd.Timedelta(seconds=self.clustering_time):
                operation_index = len(operation_planning)
                add_operation = True

        # update lock operation planning based on this assignment (already done when lock master planned for the long-term)
        yield from self.update_operation_planning(vessel, direction, operation_index, add_operation)
        operation_index = vessel_planning.loc[vessel_planning_index, "operation_index"]

        # request access to the waiting area
        vessel.waiting_area_request = waiting_area.waiting_area.request()
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
        lock_end_node :


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

        waiting_time : float


        Returns
        -------
        nothing

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

    def initiate_levelling(self, origin, destination, vessel=None, k=0, *args, **kwargs):
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
        if 'Lock' not in vessel.multidigraph.edges[origin, destination, k].keys():
            return

        # get the lock complex object
        lock = vessel.multidigraph.edges[origin, destination, k]['Lock'][0]

        # unpack the lock complex master's vessel and lock operation plannings
        vessel_planning = lock.vessel_planning
        operation_planning = lock.operation_planning

        # determine the index of the vessel and the lock operation to which it is assigned to and the index of this operation
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']
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
            delay = operation_planning.loc[operation_index].time_door_closing_start.round('s').to_pydatetime().timestamp() - lock.env.now
            if delay > 0:
                yield lock.env.timeout(delay)

            # Convert lock chamber
            close_doors = True
            if lock.close_doors_before_vessel_is_laying_still and this_operation.time_door_closing_start < vessel_planning.loc[vessel_planning_index,'time_lock_entry_stop']:
                close_doors = False

            yield from lock.convert_chamber(next_node, vessel, close_doors, direction=direction)

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
            while waiting_for_other_vessels:
                try:
                    yield lock.wait_for_other_vessels.get(filter=(lambda request: request.id == vessel.id))
                    waiting_for_other_vessels = False
                except simpy.Interrupt as e:
                    waiting_for_other_vessels = True

            # Follow the converting lock chamber
            vessel.log_entry_v0("Levelling start", vessel.env.now, vessel.output.copy(), vessel.position_in_lock,)
            waiting_for_levelling = True
            while waiting_for_levelling:
                try:
                    yield lock.wait_for_levelling.get(filter=(lambda request: request.id == vessel.id))
                    waiting_for_levelling = False
                except simpy.Interrupt as e:
                    waiting_for_levelling = True
            vessel.log_entry_v0("Levelling stop", vessel.env.now, vessel.output.copy(), vessel.position_in_lock,)

        # Determine and yield sailing out delay
        sailing_out_delay = lock.calculate_vessel_departure_start_time(vessel, operation_index, direction).total_seconds()
        delay_start = vessel.env.now
        while sailing_out_delay:
            try:
                yield vessel.env.timeout(sailing_out_delay)
                sailing_out_delay = 0
            except simpy.Interrupt as e:
                sailing_out_delay -= vessel.env.now - delay_start

    def allow_vessel_to_sail_out_of_lock(self, origin, destination, vessel=None, k=0, *args, **kwargs):
        """
        DOCUMENTATION HERE

        :param origin:
        :param destination:
        :param vessel:
        :param k:
        :param args:
        :param kwargs:
        :return:
        """
        if 'Lock' in vessel.multidigraph.edges[origin, destination, k].keys():
            lock = vessel.multidigraph.edges[origin, destination, k]['Lock'][0]
            vessel_planning = lock.lock_complex.vessel_planning
            operation_planning = lock.lock_complex.operation_planning

            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            direction = vessel_planning.loc[vessel_planning_index,'bound']
            vessel_operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']
            distance_in_lock_from_position = lock.lock_length - vessel.distance_position_from_first_lock_doors

            # Sail to lock
            if not direction:
                second_lock_doors_position = lock.location_lock_doors_B
                distance_from_lock_position = distance_in_lock_from_position
                remaining_distance = lock.distance_from_end_node_to_lock_doors_B
                exit_geom = vessel.env.graph.nodes[lock.end_node]["geometry"]
                next_level_in_case_of_following_empty_lockage = lock.start_node
            else:
                second_lock_doors_position = lock.location_lock_doors_A
                distance_from_lock_position = distance_in_lock_from_position
                remaining_distance = lock.distance_from_start_node_to_lock_doors_A
                exit_geom = vessel.env.graph.nodes[lock.start_node]["geometry"]
                next_level_in_case_of_following_empty_lockage = lock.end_node

            release_lock_access = False
            while not release_lock_access:
                try:
                    yield lock.length.put(vessel.L)
                    release_lock_access = True
                except simpy.Interrupt as e:
                    release_lock_access = True

            waiting_to_sail_out_time = (vessel_planning.loc[vessel_planning_index,'time_lock_departure_start']-pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))).total_seconds()
            waiting_to_sail_out_time_start = vessel.env.now
            while waiting_to_sail_out_time > 0:
                try:
                    yield vessel.env.timeout(waiting_to_sail_out_time)
                    waiting_to_sail_out_time = 0
                except simpy.Interrupt as e:
                    waiting_to_sail_out_time -= vessel.env.now - waiting_to_sail_out_time_start

            vessel.log_entry_v0("Sailing to second lock doors start", vessel.env.now, vessel.output.copy(),vessel.position_in_lock, )
            vessel_speed = lock.vessel_sailing_speed_out_lock(vessel)
            sailing_out_time = distance_from_lock_position/vessel_speed
            sailing_out_start = vessel.env.now
            while sailing_out_time:
                try:
                    yield vessel.env.timeout(sailing_out_time)
                    sailing_out_time = 0
                except simpy.Interrupt as e:
                    sailing_out_time -= vessel.env.now - sailing_out_start
            vessel.log_entry_v0("Sailing to second lock doors stop", vessel.env.now, vessel.output.copy(),second_lock_doors_position, )

            # remove functions specific to this lock from vessel.
            remove_functions = [lock.allow_vessel_to_sail_in_lock, lock.initiate_levelling, lock.allow_vessel_to_sail_out_of_lock]
            for function in vessel.on_pass_edge_functions:
                if isinstance(function, functools.partial):
                    if function.func in remove_functions:
                        vessel.on_pass_edge_functions.remove(function)
                elif function in remove_functions:
                    vessel.on_pass_edge_functions.remove(function)

            made_operation = operation_planning.loc[vessel_operation_index]
            vessels = made_operation.vessels
            is_last_vessel_sailing_out = vessels[-1] == vessel

            doors_can_be_closed = lock.determine_if_door_can_be_closed(vessel, direction, vessel_operation_index)

            next_operations = operation_planning[operation_planning.index >= vessel_operation_index+1]
            next_lockage_is_empty = False
            if not next_operations.empty:
                next_operation = next_operations.iloc[0]
                if not len(next_operation.vessels):
                    next_lockage_is_empty = True

            current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
            if is_last_vessel_sailing_out:
                if next_lockage_is_empty:
                    next_operation = next_operations.iloc[0]
                    door_closing_start = next_operation.time_door_closing_start
                    closing_delay = np.max([self.sailing_time_before_closing_lock_doors,(door_closing_start - current_time).total_seconds()])
                    if lock.closing_doors_in_between_operations:
                        next_next_operation = next_operations.iloc[1]
                        door_opening_start = next_next_operation.time_potential_lock_door_opening_stop
                        operation_time = self.determine_time_to_open_door(operation_index = vessel_operation_index+1,
                                                                          direction = 1 - direction,
                                                                          last_time_doors_closed = door_closing_start,
                                                                          doors_required_to_be_open = door_opening_start,
                                                                          same_direction=False)
                        opening_delay = np.max([0, (door_opening_start - current_time).total_seconds()]) - operation_time.total_seconds()
                        if opening_delay > (closing_delay+self.lock_complex.doors_closing_time):
                            vessel.env.process(lock.close_door(delay=closing_delay))
                            vessel.env.process(lock.convert_chamber(new_level=next_level_in_case_of_following_empty_lockage, vessel=None,close_doors=False, delay=opening_delay, direction=1 - direction))
                        else:
                            vessel.env.process(lock.convert_chamber(new_level=next_level_in_case_of_following_empty_lockage,vessel=None, close_doors=True, delay=closing_delay,direction=1 - direction))
                    else:
                        vessel.env.process(lock.convert_chamber(new_level=next_level_in_case_of_following_empty_lockage, vessel=None,close_doors=True, delay=closing_delay, direction=1 - direction))
                elif doors_can_be_closed and lock.closing_doors_in_between_operations:
                    door_closing_time = made_operation.time_potential_lock_door_closure_start
                    delay = np.max([self.sailing_time_before_closing_lock_doors,(door_closing_time-current_time).total_seconds()])
                    vessel.env.process(lock.close_door(delay=delay))

            vessel.log_entry_v0("Sailing to lock complex exit start", vessel.env.now, vessel.output.copy(),second_lock_doors_position, )
            vessel_speed = lock.vessel_sailing_out_speed(vessel, direction)
            sailing_out_time = remaining_distance / vessel_speed
            sailing_out_start = vessel.env.now
            while sailing_out_time:
                try:
                    yield vessel.env.timeout(sailing_out_time)
                    sailing_out_time = 0
                except simpy.Interrupt as e:
                    sailing_out_time -= (vessel.env.now - sailing_out_start)
                    remaining_sailing_distance = vessel_speed * sailing_out_time
                    sailing_out_time = remaining_sailing_distance / vessel.current_speed
            vessel.log_entry_v0("Sailing to lock complex exit stop", vessel.env.now, vessel.output.copy(), exit_geom, )
            vessel.distance_left_on_edge = 0

    def allow_vessel_to_sail_in_lock(self, origin, destination, vessel=None, k=0, *args, **kwargs):
        """
        DOCUMENTATION HERE

        :param origin:
        :param destination:
        :param vessel:
        :param k:
        :param args:
        :param kwargs:
        :return:
        """
        if 'Lock' in vessel.multidigraph.edges[origin,destination,k].keys():
            lock = vessel.multidigraph.edges[origin,destination,k]['Lock'][0]
            vessel_planning = lock.lock_complex.vessel_planning
            operation_planning = lock.lock_complex.operation_planning

            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            direction = vessel_planning.loc[vessel_planning_index,'bound']

            # Sail to lock
            if not direction:
                lock_start_node = lock.start_node
                lock_end_node = lock.end_node
                waiting_area = lock.waiting_area_A
                distance_to_lock_position = lock.distance_from_start_node_to_lock_doors_A
                first_lock_door_position = lock.location_lock_doors_A
            else:
                lock_start_node = lock.end_node
                lock_end_node = lock.start_node
                waiting_area = lock.waiting_area_B
                distance_to_lock_position = lock.distance_from_end_node_to_lock_doors_B
                first_lock_door_position = lock.location_lock_doors_B
            if (lock_start_node, lock_end_node, lock.k) == waiting_area.edge:
                distance_to_lock_position -= waiting_area.distance_from_edge_start
            vessel.log_entry_v0("Sailing to first lock doors start", vessel.env.now, vessel.output.copy(),vessel.logbook[-1]['Geometry'],)
            start_sailing = vessel.env.now
            vessel_speed = lock.vessel_sailing_in_speed(vessel, direction)
            remaining_sailing_time = distance_to_lock_position / vessel_speed
            while remaining_sailing_time > 0:
                try:
                    yield vessel.env.timeout(remaining_sailing_time)
                    remaining_sailing_time = 0
                except simpy.Interrupt as e:
                    remaining_sailing_time -= (vessel.env.now - start_sailing)
                    remaining_sailing_distance = vessel_speed*remaining_sailing_time
                    remaining_sailing_time = remaining_sailing_distance/vessel.current_speed
                    if vessel_speed != vessel.current_speed:
                        distance = distance_to_lock_position-remaining_sailing_distance + waiting_area.distance_from_edge_start
                        geometry = vessel.env.vessel_traffic_service.provide_location_over_edges(lock_start_node,lock_end_node,distance)
                        vessel.log_entry_v0("Sailing speed changed", vessel.env.now, vessel.output.copy(),geometry,)
                    # TODO for later research: the speed changes should be checked if they are realistic by combining it with a smoothly decreasing velocity (P_used)

            lock_accessed = False
            remaining_lock_length = lock.length.level
            vessel.overruled_speed = vessel.overruled_speed.iloc[0:0]

            yield lock.length.get(vessel.L)

            vessel.log_entry_v0("Sailing to first lock doors stop", vessel.env.now, vessel.output.copy(),first_lock_door_position, )

            # Checks if door should be closed intermediately
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']
            this_operation = operation_planning.loc[operation_index]
            vessels = this_operation.vessels
            current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
            delay_to_close_doors = vessel_planning.loc[vessel_planning_index, 'time_potential_lock_door_closure_start'] - current_time

            last_vessel_to_enter_lock = vessels[-1] == vessel
            doors_can_be_closed_between_vessel_arrivals = lock.determine_if_door_can_be_closed(vessel, direction, operation_index, between_arrivals=True)
            if lock.close_doors_before_vessel_is_laying_still and ((lock.closing_doors_in_between_arrivals and doors_can_be_closed_between_vessel_arrivals) or (last_vessel_to_enter_lock and this_operation.time_door_closing_start < vessel_planning.loc[vessel_planning_index,'time_lock_entry_stop'])):
                vessel.env.process(lock.close_door(delay=delay_to_close_doors.total_seconds()))

            vessel.log_entry_v0("Sailing to position in lock start", vessel.env.now, vessel.output.copy(),first_lock_door_position, )
            vessel.distance_position_from_first_lock_doors = remaining_lock_length - 0.5*vessel.L

            if not direction: ###hereh
                vessel.position_in_lock = vessel.env.vessel_traffic_service.provide_location_over_edges(lock.start_node,lock.end_node,lock.distance_from_start_node_to_lock_doors_A + vessel.distance_position_from_first_lock_doors)
            elif direction:
                vessel.position_in_lock = vessel.env.vessel_traffic_service.provide_location_over_edges(lock.end_node,lock.start_node,lock.distance_from_end_node_to_lock_doors_B + vessel.distance_position_from_first_lock_doors)

            vessel_speed = lock.vessel_sailing_speed_in_lock(vessel)
            remaining_sailing_time = vessel.distance_position_from_first_lock_doors / vessel_speed
            while remaining_sailing_time > 0:
                try:
                    yield vessel.env.timeout(remaining_sailing_time)
                    remaining_sailing_time = 0
                except simpy.Interrupt as e:
                    remaining_sailing_time -= vessel.env.now - start_sailing
            vessel.log_entry_v0("Sailing to position in lock stop", vessel.env.now, vessel.output.copy(),vessel.position_in_lock,)

            doors_can_be_closed_between_vessel_arrivals = lock.determine_if_door_can_be_closed(vessel, direction, operation_index, between_arrivals=True)
            if not lock.close_doors_before_vessel_is_laying_still and not last_vessel_to_enter_lock and lock.closing_doors_in_between_arrivals and doors_can_be_closed_between_vessel_arrivals:
                vessel.env.process(lock.close_door())

    def add_vessel_to_vessel_planning(self, vessel, direction, time_of_registration=None):
        """
        Adds vessel to the vessel planning of the lock complex upon request

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
        time_of_registration : pd.Timestamp
            the time that the vessel registers to the lock master

        Returns
        -------
        nothing

        """

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
        vessel_planning.loc[vessel_planning_index, 'bound'] = direction
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
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)

        Returns
        -------
        nothing
        """
        # unpack the lock master's lock operation planning
        operation_planning = self.lock_complex.operation_planning

        # determine the start time of this empty lock operation
        preceding_operations = operation_planning[operation_planning.index < operation_index]
        if not preceding_operations.empty:
            preceding_operation = operation_planning.loc[operation_index-1]
            first_empty_lock_operation_start = preceding_operation.time_potential_lock_door_closure_start
        else:
            first_empty_lock_operation_start = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))

        # add empty lock operation to operation planning of the lock master, including deriving the lock operation information (i.e., start and stop times of individual events, water levels, and status)
        operation_planning.loc[operation_index, 'bound'] = direction
        operation_planning.loc[operation_index, 'vessels'] = []
        operation_planning.loc[operation_index, 'capacity_L'] = self.lock_complex.lock_length
        operation_planning.loc[operation_index, 'capacity_B'] = self.lock_complex.lock_width
        time_door_closing_start, time_door_closing_stop, time_levelling_start, time_levelling_stop, time_door_opening_start, time_door_opening_stop = self.calculate_lock_operation_times(operation_index=operation_index,
                                                                                                                                                                                          last_entering_time=first_empty_lock_operation_start,
                                                                                                                                                                                          start_time=first_empty_lock_operation_start,
                                                                                                                                                                                          direction=direction)
        wlev_A, wlev_B = self.determine_water_levels_before_and_after_levelling(time_levelling_start,time_levelling_stop, direction)
        operation_planning.loc[operation_index, 'time_operation_start'] = first_empty_lock_operation_start
        operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop'] = first_empty_lock_operation_start
        operation_planning.loc[operation_index, 'time_entry_start'] = first_empty_lock_operation_start
        operation_planning.loc[operation_index, 'time_entry_stop'] = first_empty_lock_operation_start
        operation_planning.loc[operation_index, 'time_door_closing_start'] = time_door_closing_start
        operation_planning.loc[operation_index, 'time_door_closing_stop'] = time_door_closing_stop
        operation_planning.loc[operation_index, 'time_levelling_start'] = time_levelling_start
        operation_planning.loc[operation_index, 'time_levelling_stop'] = time_levelling_stop
        operation_planning.loc[operation_index, 'time_door_opening_start'] = time_levelling_stop
        operation_planning.loc[operation_index, 'time_door_opening_stop'] = time_door_opening_stop
        operation_planning.loc[operation_index, 'time_departure_start'] = time_door_opening_stop
        operation_planning.loc[operation_index, 'time_departure_stop'] = time_door_opening_stop
        operation_planning.loc[operation_index, 'time_potential_lock_door_closure_start'] = time_door_opening_stop
        operation_planning.loc[operation_index, 'time_operation_stop'] = time_door_opening_stop
        operation_planning.loc[operation_index, 'wlev_A'] = wlev_A
        operation_planning.loc[operation_index, 'wlev_B'] = wlev_B
        operation_planning.loc[operation_index, 'maximum_individual_delay'] = pd.Timedelta(seconds=0)
        operation_planning.loc[operation_index, 'total_delay'] = pd.Timedelta(seconds=0)
        operation_planning.loc[operation_index, 'status'] = ''

    def determine_route_to_waiting_area_from_node(self, node, vessel):
        """
        DOCUMENTATION HERE

        :param node:
        :param vessel:
        :return:
        """
        remaining_route = nx.dijkstra_path(self.env.graph, node, vessel.route[-1])
        for origin in remaining_route:
            if origin == self.lock_complex.waiting_area_A.edge[0]:
                waiting_area_node = self.lock_complex.waiting_area_A.edge[1]
                break
            elif origin == self.lock_complex.waiting_area_B.edge[0]:
                waiting_area_node = self.lock_complex.waiting_area_B.edge[1]
                break
        route_to_waiting_area = nx.dijkstra_path(self.env.graph, vessel.current_node, waiting_area_node)
        return route_to_waiting_area

    def calculate_sailing_time_to_waiting_area(self, vessel, direction, current_node=None, prognosis=False, overwrite=True):
        """TODO: note that this function looks a lot like other 'calculate_sailing_time_to'-functions below, so maybe we can investigate to combine the functions
        Calculates the sailing time of a vessel from its location to the waiting area

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
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

        # determine the current node of the vessel
        if current_node is None:
            current_node = vessel.current_node

        # determine route to the start node of the edge at which the waiting area is located
        route_to_waiting_area = self.determine_route_to_waiting_area_from_node(node=current_node, vessel=vessel)

        # unpack vessel planning
        vessel_planning = self.lock_complex.vessel_planning

        # unpack first encountered waiting area
        if not direction:
            waiting_area_approach = self.lock_complex.waiting_area_A
        else:
            waiting_area_approach = self.lock_complex.waiting_area_B

        # unpack the function that calculates sailing time from distance on edge to distance on another edge
        calculate_sailing_time = self.env.vessel_traffic_service.provide_sailing_time_distance_on_edge_to_distance_on_another_edge

        # determine the distance that the vessel has to sail on the edge at which the waiting area is located (from the start node of the edge)
        distance_to_waiting_area_on_last_edge = waiting_area_approach.distance_from_edge_start

        # calculation of the sailing information (time, distance, speed) per edge on route to the waiting area
        sailing_to_waiting_area = calculate_sailing_time(vessel, route=route_to_waiting_area,
                                                         distance_sailed_on_last_edge=distance_to_waiting_area_on_last_edge)

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

    def calculate_sailing_time_to_lineup_area(self, vessel, direction, current_node=None, prognosis=False, overwrite=True):
        """
        Calculates the sailing time of a vessel from its location to the line-up area

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
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
        if not direction:
            lineup_area_approach = self.lock_complex.lineup_area_A
        else:
            lineup_area_approach = self.lock_complex.lineup_area_B

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
        self, vessel, direction, current_node=None, operation_index=None, prognosis=False, overwrite=True
    ):
        """
        Calculates the sailing time of a vessel from its location to the approach point

        The approach point is the closest location in front of the lock doors where the outbound vessel(s) can pass the inbound vessel waiting to enter the lock.
        The point is located in between the line-up area and the lock doors.

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
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

        # unpack vessel planning
        vessel_planning = self.lock_complex.vessel_planning

        # unpack sailing distance from crossing point to lock doors
        sailing_distance_from_entry = self.sailing_distance_to_crossing_point

        # determine the time of entering the lock
        sailing_speed_during_entry = self.vessel_sailing_in_speed(vessel, direction)
        sailing_time_entry = pd.Timedelta(seconds=sailing_distance_from_entry / sailing_speed_during_entry)

        # determine the time of the vessel to its first encountered waiting area and lock_door TODO: in the 'add_vessel_to_planning'-function these functions has already been done, so doing these again can be computational intensive and should be prevented. Can we include tests that before this function is ran, these following functions have already been ran? How can we extract the earlier output?
        # sailing_time_to_waiting_area = self.calculate_sailing_time_to_waiting_area(vessel, direction, current_node = current_node,overwrite=overwrite)[0]
        sailing_time_to_lock_door = self.calculate_sailing_time_to_lock_door(
            vessel, direction, current_node=current_node, overwrite=overwrite
        )

        # determine the sailing time to the approach point
        sailing_time_to_start_approach = sailing_time_to_lock_door - sailing_time_entry #- sailing_time_to_waiting_area TODO: later check if we indeed can get rid of the sailing time to waiting area

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

    def calculate_sailing_time_to_lock_door(self, vessel, direction, current_node=None, prognosis=False, overwrite=True):
        """
        Calculates the sailing time of a vessel from its location to the first lock doors that it will encounter

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
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

        # unpack vessel planning
        vessel_planning = self.lock_complex.vessel_planning

        # determine the end node of the lock complex from the perspective of the vessel and the distance from the start node of the lock complex to the lock doors
        lock_end_node = self._lock_end_node(direction)
        distance_to_lock = self._distance_to_lock(direction)

        # determine the route of the vessel to the end node of the lock complex from the perspective of the vessel
        # TODO: @Floor: kan dit ook worden worden (vessel.route_to_come tot de lock complex)?
        route_to_lock_chamber = nx.dijkstra_path(self.env.graph, current_node, lock_end_node)

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

    def _lock_end_node(self, direction):
        """get the end node of the lock from the perspective of the vessel

        Parameters
        ----------
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
        """
        if not direction:
            return self.lock_complex.end_node
        else:
            return self.lock_complex.start_node

    def _distance_to_lock(self, direction):
        """get the distance from the start node of the lock to the lock doors from the perspective of the vessel

        Parameters
        ----------
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
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
            the direction of the lock operation: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
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
            print(vessel.name)
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
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
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

    def calculate_vessel_entry_start_time(self, vessel, direction):
        """
        Calculates the moment in time that a vessel starts entering the lock

        Parameters
        ----------
        vessel : type [optional]
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)

        Returns
        -------
        sailing_time_entry : pd.Timedelta
            the time duration of the process of a vessel entering the lock [s]

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
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
        prognosis :
            .
        overwrite :
            .

        Returns
        -------
        vessel_passing_start_timestamp : pd.Timestamp
            the moment in time that a vessel starts entering the lock from the approach point

        """
        # unpack the lock complex master's vessel planning
        vessel_planning = self.lock_complex.vessel_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

        # determines the current time
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))

        # calculate the sailing time durations to the lock door, the approach point and if there is any form of delay for this
        sailing_time_to_lock = self.calculate_sailing_time_to_lock_door(vessel, direction, prognosis=prognosis, overwrite=overwrite)
        sailing_time_entry = self.calculate_vessel_entry_start_time(vessel, direction)
        sailing_in_delay = self.calculate_sailing_in_time_delay(
            vessel, operation_index, direction, prognosis=prognosis, overwrite=overwrite
        )

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
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
        prognosis :
            .
        overwrite :
            .

        Returns
        -------
        lock_operation_start_time : pd.Timestamp
            the moment in time of the start of the lock operation

        """
        # unpacks the lock complex master's operation planning
        operation_planning = self.lock_complex.operation_planning

        # determines the lock operation start time based on the first vessel that was assigned to this lock operation
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index)
        lock_operation_start_time = self.calculate_vessel_passing_start_time(
            first_vessel, operation_index, direction, prognosis, overwrite=overwrite
        )

        # determines the lock_operation_start_time based on whether it fits given the previous lock operations (should not be overlapping)
        previous_operations = operation_planning[operation_planning.index < operation_index]
        if not previous_operations.empty:
            previous_operation = previous_operations.iloc[-1]
            previous_lock_operation_stop_time = previous_operation.time_operation_stop
            if lock_operation_start_time < previous_lock_operation_stop_time:
                lock_operation_start_time = previous_lock_operation_stop_time

        return lock_operation_start_time

    def calculate_lock_door_opening_time(self, vessel, operation_index, direction, prognosis=False, overwrite=True):
        """
        DOCUMENTATION HERE

        :param vessel:
        :param operation_index:
        :param direction:
        :param prognosis:
        :param overwrite:
        :return:
        """
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index)
        lock_entry_start_time = self.calculate_vessel_entry_start_time(first_vessel, direction)
        lock_entry_start_time -= self.minimum_advance_to_open_doors(vessel, direction)
        return lock_entry_start_time

    def calculate_lock_entry_start_time(self, vessel, operation_index, direction, prognosis=False, overwrite=True):
        """
        DOCUMENTATION HERE

        :param vessel:
        :param operation_index:
        :param direction:
        :param prognosis:
        :param overwrite:
        :return:
        """
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index)
        lock_entry_start_time = self.calculate_vessel_entry_start_time(first_vessel, direction)
        return lock_entry_start_time

    def calculate_vessel_entry_stop_time(self, vessel, operation_index, direction, prognosis=False, overwrite=True):
        """
        Calculates the moment in time that a vessel finished its lock entry process

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
        prognosis : bool
            .
        overwrite : bool
            .

        Returns
        -------
        vessel_entry_stop_time : pd.Timestamp
             the moment in time that the vessel stops entering the lock

        """

        # determine the moment in time that the vessel starts to enter the lock
        vessel_entry_start_time = self.calculate_vessel_entry_start_time(vessel, direction)

        # determine the time duration of the vessel in the lock
        sailing_time_in_lock = self.calculate_sailing_time_in_lock(vessel, operation_index, prognosis)

        # calculate the moment in time that the vessel stops entering the lock
        vessel_entry_stop_time = vessel_entry_start_time + sailing_time_in_lock

        return vessel_entry_stop_time

    def calculate_lock_entry_stop_time(self, vessel, operation_index, direction, prognosis=False, overwrite=True):
        """
        Calculates the moment in time that a lock operation entry process of all the assigned vessels is finished (all vessels are in lock chamber)

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            index of the lock operation
        direction : int
            the direction of the lock operation: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
        prognosis : bool
            .
        overwrite : bool
            .

        Returns
        -------
        lock_entry_stop_time : pd.Timestamp
            .
        """

        # determine the last assigned vessel of the lock operation to determine the lock entry stop time
        last_vessel = self.determine_last_vessel_of_lock_operation(vessel, operation_index, prognosis)
        lock_entry_stop_time = self.calculate_vessel_entry_stop_time(
            last_vessel, operation_index, direction, prognosis, overwrite=overwrite
        )

        return lock_entry_stop_time

    def calculate_lock_operation_times(self, operation_index, last_entering_time, start_time, vessel = None, direction=None, same_direction = False):
        """
        Calculates the moments in time of the start and stop of the operation steps of the lock: (1) door closing, (2) levelling, (3) door opening

        Parameters
        ----------
        operation_index : int
            the index of the lock operation in the operation planning of the lock complex master
        last_entering_time : pd.Timestamp
            .
        start_time : pd.Timestamp
            .
        vessel : type [optional]
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int [optional]
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
        same_direction : bool
            .

        Returns
        -------
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
        operation_planning = self.lock_complex.operation_planning
        vessel_planning = self.lock_complex.vessel_planning

        # determine the longitudinal location (x) of the last vessel that will/has enter(ed)
        x_location_lock = 0.
        if vessel is not None:
            vessels_in_lock = operation_planning.loc[operation_index].vessels
            if vessels_in_lock == []:
                vessels_in_lock = [vessel]
            x_location_lock = np.sum([v.L for v in vessels_in_lock[:-1]])+0.5*vessel.L

        # set default time door closing start as start time
        time_door_closing_start = start_time

        # overwrite the time door closing start if there is a rule that the doors can close before a vessel is laying still and there are vessels in the lock
        if self.close_doors_before_vessel_is_laying_still and vessel is not None:
            time_door_closing_start = last_entering_time + self.minimum_delay_to_close_doors(vessel, direction, after_lock_entry=True, x_location_lock=x_location_lock)

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
        time_levelling_stop,_,_ = self.lock_complex.determine_levelling_time(t_start=time_levelling_start, direction=direction, prediction=True, same_direction = same_direction)
        time_levelling_stop = time_levelling_start + pd.Timedelta(seconds=time_levelling_stop)
        time_door_opening_start = time_levelling_stop
        time_door_opening_stop = time_levelling_stop + pd.Timedelta(seconds=self.lock_complex.doors_opening_time)

        return time_door_closing_start, time_door_closing_stop, time_levelling_start, time_levelling_stop, time_door_opening_start, time_door_opening_stop

    def calculate_vessel_departure_start_time(self, vessel, operation_index, direction, prognosis=False):
        """
        DOCUMENTATION HERE

        :param vessel:
        :param operation_index:
        :param direction:
        :param prognosis:
        :return:
        """
        vessels = self.get_vessel_from_planned_operation(
            operation_index=operation_index,
        )
        if not prognosis:
            vessel_index = vessels.index(vessel)
            number_of_previous_vessels = vessel_index
        else:
            vessel_index = -1
            number_of_previous_vessels = len(vessels)

        delay = pd.Timedelta(seconds=0)
        if number_of_previous_vessels:
            previous_vessel = vessels[vessel_index-1]
            vessel_sailing_out_time = self.calculate_vessel_sailing_time_out_of_lock(
                vessel, operation_index, direction, prognosis=prognosis
            )
            previous_vessel_sailing_out_time = self.calculate_vessel_sailing_time_out_of_lock(
                previous_vessel, operation_index, direction, prognosis=prognosis
            )
            sailing_out_time_gap_through_doors = (vessel_sailing_out_time - previous_vessel_sailing_out_time)
            if sailing_out_time_gap_through_doors < pd.Timedelta(seconds=self.sailing_out_time_gap_through_doors):
                delay += number_of_previous_vessels*pd.Timedelta(seconds=self.sailing_out_time_gap_through_doors)-sailing_out_time_gap_through_doors

            if self.sailing_out_time_gap_after_berthing_previous_vessel is not None and delay < pd.Timedelta(seconds=self.sailing_out_time_gap_after_berthing_previous_vessel)*number_of_previous_vessels:
                delay = pd.Timedelta(seconds=self.sailing_out_time_gap_after_berthing_previous_vessel)*number_of_previous_vessels

        delay += pd.Timedelta(seconds=self.start_sailing_out_time_after_doors_have_been_opened)
        return delay

    def calculate_lock_departure_start_time(self, vessel, operation_index, direction, prognosis=False, first_vessel=False):
        """
        DOCUMENTATION HERE

        :param vessel:
        :param operation_index:
        :param direction:
        :param prognosis:
        :param first_vessel:
        :return:
        """
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index)
        time_departure_start = self.calculate_vessel_departure_start_time(first_vessel, operation_index, direction, prognosis)
        return time_departure_start

    def calculate_vessel_sailing_time_out_of_lock(self, vessel, operation_index, direction, prognosis=False):
        """
        DOCUMENTATION HERE

        :param vessel:
        :param operation_index:
        :param direction:
        :param prognosis:
        :return:
        """
        vessels = self.get_vessel_from_planned_operation(
            operation_index=operation_index,
        )
        # Time to sail out
        if not prognosis:
            vessel_index = vessels.index(vessel)
            distance_to_lock = np.sum([vessel.L for vessel in vessels[:vessel_index]]) + 0.5 * vessel.L
        else:
            distance_to_lock = np.sum([vessel.L for vessel in vessels]) + 0.5 * vessel.L
        vessel_speed = self.vessel_sailing_speed_out_lock(vessel)
        sailing_out_time = pd.Timedelta(seconds=distance_to_lock / vessel_speed)
        return sailing_out_time

    def calculate_vessel_departure_stop_time(self, vessel, operation_index, direction, prognosis=False):
        """
        DOCUMENTATION HERE

        :param vessel:
        :param operation_index:
        :param direction:
        :param prognosis:
        :return:
        """
        time_departure_start = self.calculate_vessel_departure_start_time(vessel, operation_index, direction, prognosis)
        sailing_out_time = self.calculate_vessel_sailing_time_out_of_lock(vessel, operation_index, direction, prognosis)
        time_departure_stop = time_departure_start + sailing_out_time
        return time_departure_stop

    def calculate_lock_departure_stop_time(self, vessel, operation_index, direction, prognosis=False):
        """
        DOCUMENTATION HERE

        :param vessel:
        :param operation_index:
        :param direction:
        :param prognosis:
        :return:
        """
        last_vessel = self.determine_last_vessel_of_lock_operation(vessel, operation_index, prognosis)
        time_departure_stop = self.calculate_vessel_departure_stop_time(last_vessel, operation_index, direction, prognosis)
        return time_departure_stop

    def calculate_vessel_passing_stop_time(self, vessel, operation_index, direction, prognosis=False):
        """
        DOCUMENTATION HERE

        :param vessel:
        :param operation_index:
        :param direction:
        :param prognosis:
        :return:
        """
        time_departure_stop = self.calculate_vessel_departure_stop_time(vessel, operation_index, direction, prognosis)
        vessel_speed = self.vessel_sailing_out_speed(vessel, direction, until_crossing_point=True)
        time_departure_stop += pd.Timedelta(seconds = self.sailing_distance_to_crossing_point/vessel_speed)
        return time_departure_stop

    def calculate_lock_operation_stop_time(self, vessel, operation_index, direction, prognosis=False):
        """
        DOCUMENTATION HERE

        :param vessel:
        :param operation_index:
        :param direction:
        :param prognosis:
        :return:
        """
        last_vessel = self.determine_last_vessel_of_lock_operation(vessel, operation_index, prognosis)
        time_operation_stop = self.calculate_vessel_passing_stop_time(last_vessel, operation_index, direction, prognosis)
        return time_operation_stop

    def minimum_delay_to_close_doors(self, vessel, direction, after_lock_entry=False, x_location_lock=0.):
        """
        Calculates the time delay between when the last vessel has entered the lock and when the lock doors can be closed

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
        after_lock_entry : bool
            .
        x_location_lock : float
            longitudinal coordinate at which the vessel is located

        Returns
        -------
        minimum_delay_to_close_doors : pd.Timedelta
            the minimum time delay that the lock doors can be closed after a vessel has entered the lock
        """
        minimum_delay_to_close_doors = pd.Timedelta(seconds=self.sailing_time_before_closing_lock_doors)
        # if not after_lock_entry:
        #     minimum_delay_to_close_doors += pd.Timedelta(seconds=0.5*vessel.L/self.vessel_sailing_out_speed(vessel,direction))
        # else:
        #     minimum_delay_to_close_doors += pd.Timedelta(seconds=0.5*vessel.L/self.vessel_sailing_speed_in_lock(vessel)
        # TODO: take into account the vessels' bows and sterns to determine the time before and after which the door should be respectively opened and closed
        return minimum_delay_to_close_doors

    def minimum_advance_to_open_doors(self, vessel, direction):
        """
        Determines the minimum time in advance that a lock door should be opened

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)


        Returns
        -------
        minimum_advance_to_open_doors : pd.Timedelta
            the minimum time in advance that a lock door should be opened [s]

        """
        minimum_advance_to_open_doors = pd.Timedelta(seconds=self.sailing_time_before_opening_lock_doors)
        # minimum_advance_to_open_doors += pd.Timedelta(seconds=vessel.L/self.vessel_sailing_in_speed(vessel,direction))
        # TODO: take into account the vessels' bows and sterns to determine the time before and after which the door should be respectively opened and closed
        return minimum_advance_to_open_doors

    def calculate_lock_door_closing_time(self, vessel, operation_index, direction, prognosis=False):
        """
        DOCUMENTATION HERE

        :param vessel:
        :param operation_index:
        :param direction:
        :param prognosis:
        :return:
        """
        lock_doors_closing_time = self.calculate_lock_departure_stop_time(vessel, operation_index, direction, prognosis)
        lock_doors_closing_time += self.minimum_delay_to_close_doors(vessel, direction)
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
        vessels = self.get_vessel_from_planned_operation(
            operation_index=operation_index,
        )

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
            .

        Returns
        -------
        last_vessel : type
            the last assigned vessel of the lock operation (the one that will enter and leave the lock chamber last)
        """
        # identify the vessels assigned the lock operation
        vessels = self.get_vessel_from_planned_operation(
            operation_index=operation_index,
        )

        # TODO @Floor: moeten we hier vessel opgeven als input? of kan die ook worden bepaald uit de berekende vessels als prognosis=True?
        # determine the last vessel
        last_vessel = vessel
        if not prognosis:
            last_vessel = vessels[-1]

        return last_vessel

    def calculate_delay_to_open_doors(self, vessel):
        """
        DOCUMENTATION HERE

        :param vessel:
        :return:
        """
        vessel_planning = self.lock_complex.vessel_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        arrival_time_at_lock = vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start']
        time_index = np.absolute(hydrodynamic_times - np.datetime64(arrival_time_at_lock) - np.timedelta64(int(self.doors_opening_time),'s')).argmin()
        expected_levelling_time,_,_ = self.determine_levelling_time(t_start=hydrodynamic_times[time_index - 1], same_direction=True, prediction=True)
        delay = self.sailing_time_before_opening_lock_doors + expected_levelling_time + self.doors_opening_time
        return delay

    def determine_if_door_can_be_closed(self, vessel, direction, operation_index, between_arrivals=False):
        """
        DOCUMENTATION HERE

        :param vessel:
        :param direction:
        :param operation_index:
        :param between_arrivals:
        :return:
        """
        doors_can_be_closed = False
        if not between_arrivals and not self.closing_doors_in_between_operations:
            return doors_can_be_closed
        if between_arrivals and not self.closing_doors_in_between_arrivals:
            return doors_can_be_closed
        doors_can_be_closed = True

        operation_planning = self.lock_complex.operation_planning
        vessel_planning = self.lock_complex.vessel_planning

        if not between_arrivals:
            last_time_doors_closed = operation_planning.loc[
                operation_index, "time_potential_lock_door_closure_start"
            ] + pd.Timedelta(seconds=self.doors_closing_time)
        else:
            last_time_doors_closed = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now)) + pd.Timedelta(seconds=self.doors_closing_time)

        next_operations = operation_planning[operation_planning.index > operation_index]

        vessel_index = operation_planning.loc[operation_index, 'vessels'].index(vessel)
        vessels_in_operation = operation_planning.loc[operation_index, 'vessels']

        operation_step = 1
        if between_arrivals and vessel_index != len(vessels_in_operation)-1:
            next_vessel = vessels_in_operation[vessel_index+1]
            next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name
            doors_required_to_be_open = vessel_planning.loc[next_vessel_planning_index,'time_potential_lock_door_opening_stop']
            same_direction = True
        elif not next_operations.empty:
            next_operation = next_operations.iloc[0]
            if not len(next_operation.vessels):
                next_operation = next_operations.iloc[1]
                operation_step += 1
            doors_required_to_be_open = next_operation.time_potential_lock_door_opening_stop
            same_direction = direction != next_operation.bound
        else:
            return doors_can_be_closed

        if same_direction:
            direction = 1 - direction
        door_opening_time = self.determine_time_to_open_door(operation_index+operation_step, direction, last_time_doors_closed,doors_required_to_be_open, same_direction)

        if doors_required_to_be_open-door_opening_time < last_time_doors_closed or doors_required_to_be_open-last_time_doors_closed < self.minimum_time_between_operations_for_intermediate_door_closure:
            doors_can_be_closed = False
        return doors_can_be_closed

    def determine_if_door_is_closed(self, vessel, operation_index, direction, first_in_lock=False, between_arrivals=False):
        """
        DOCUMENTATION HERE

        :param vessel:
        :param operation_index:
        :param direction:
        :param first_in_lock:
        :param between_arrivals:
        :return:
        """
        operation_planning = self.lock_complex.operation_planning
        vessel_planning = self.lock_complex.vessel_planning
        vessels = operation_planning.loc[operation_index, 'vessels']
        vessel_index = vessels.index(vessel)

        if between_arrivals and not self.closing_doors_in_between_arrivals:
            return False, None, None

        if not between_arrivals and not self.closing_doors_in_between_operations:
            return False, None, None

        last_lockage_was_empty = False
        if operation_index - 2 in operation_planning.index:
            last_lockage_was_empty = len(operation_planning.loc[operation_index - 1, 'vessels']) == 0
        if last_lockage_was_empty:
            return False, None, None

        if not first_in_lock and vessel_index:
            previous_vessel_planning_index = vessel_planning[vessel_planning.id == operation_planning.loc[operation_index, 'vessels'][vessel_index-1].id].iloc[-1].name
            last_time_doors_closed = vessel_planning.loc[previous_vessel_planning_index,'time_potential_lock_door_closure_start'] + pd.Timedelta(seconds=self.doors_closing_time)
        elif operation_index == 0:
            last_time_doors_closed = datetime.datetime.fromtimestamp(self.env.now)
        else:
            last_time_doors_closed = operation_planning.loc[operation_index - 1].time_potential_lock_door_closure_start + pd.Timedelta(seconds=self.doors_closing_time)

        same_direction = False
        if first_in_lock:
            doors_required_to_be_open = operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop']
        else:
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            doors_required_to_be_open = vessel_planning.loc[vessel_planning_index, 'time_potential_lock_door_opening_stop']
            same_direction = True

        if (not direction and self.node_open == self.start_node) or (direction and self.node_open == self.end_node):
            same_direction = True

        operation_time = self.determine_time_to_open_door(operation_index, direction, last_time_doors_closed, doors_required_to_be_open, same_direction)
        doors_is_closed = False

        if doors_required_to_be_open - operation_time > last_time_doors_closed and doors_required_to_be_open-last_time_doors_closed > self.minimum_time_between_operations_for_intermediate_door_closure:
            doors_is_closed = True

        return doors_is_closed, doors_required_to_be_open, operation_time

    def determine_time_to_open_door(self, operation_index, direction, last_time_doors_closed, doors_required_to_be_open, same_direction):
        """
        DOCUMENTATION HERE

        :param operation_index:
        :param direction:
        :param last_time_doors_closed:
        :param doors_required_to_be_open:
        :param same_direction:
        :return:
        """
        _, _, time_levelling_start, time_levelling_stop, _, _ = self.calculate_lock_operation_times(operation_index=operation_index,
                                                                                                    last_entering_time=doors_required_to_be_open - pd.Timedelta(seconds=self.doors_opening_time),
                                                                                                    start_time=doors_required_to_be_open - pd.Timedelta(seconds=self.doors_opening_time),
                                                                                                    direction=direction,
                                                                                                    same_direction=same_direction)
        levelling_time = time_levelling_stop - time_levelling_start
        wlev_before, wlev_after = self.determine_water_levels_before_and_after_levelling(last_time_doors_closed + pd.Timedelta(seconds=self.doors_closing_time),
                                                                                         doors_required_to_be_open - pd.Timedelta(seconds=self.doors_opening_time) - levelling_time,
                                                                                         direction,
                                                                                         same_direction=same_direction)

        levelling_required = True
        if abs(wlev_after - wlev_before) < 0.1:
            levelling_required = False

        if not levelling_required:
            levelling_time = pd.Timedelta(seconds=0.)

        operation_time = levelling_time + pd.Timedelta(seconds=self.doors_opening_time)
        return operation_time

    def determine_water_levels_before_and_after_levelling(self,levelling_start,levelling_stop,direction,same_direction=False):
        """
        Determines the water level at both sides of the lock

        Parameters
        ----------
        levelling_start : pd.Timestamp
            the start time of the levelling process
        levelling_stop : pd.Timestamp
            the stop time of the levelling process
        direction : int
            the direction of the lock operation: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
        same_direction : bool
            .

        Returns
        ------
        wlev_A :
            the water level at side A [m] before or after the levelling process (depending on the direction of the operation)
        wlev_B :
            the actual water level at side B [m] before or after the levelling process (depending on the direction of the operation)

        """
        # set default output
        wlev_A = np.nan
        wlev_B = np.nan

        # return function if there is no hydrodynamic data
        if not self.env.vessel_traffic_service.hydrodynamic_information_path:
            return wlev_A, wlev_B

        # determine the station indexes of the start node and end node in the hydrodynamic data
        station_index_start_node = np.where(np.array(list((hydrodynamic_data['STATION']))) == self.start_node)[0]
        station_index_end_node = np.where(np.array(list((hydrodynamic_data['STATION']))) == self.end_node)[0]

        # determine the start and stop time indexes of the levelling process
        time_index_start = np.absolute(hydrodynamic_times - np.datetime64(levelling_start)).argmin()
        time_index_stop = np.absolute(hydrodynamic_times - np.datetime64(levelling_stop)).argmin()

        # determine the water levels before and after the levelling processes based on the direction of the lock operation
        if not direction:
            wlev_A = hydrodynamic_data["Water level"][station_index_start_node][time_index_stop]
            wlev_B = self.water_level[time_index_start] # take the value from the lock chamber
            if same_direction:
                wlev_A = self.water_level[time_index_start] # take the value from the lock chamber
                wlev_B = hydrodynamic_data["Water level"][station_index_start_node][time_index_stop]
        elif direction:
            wlev_B = self.water_level[time_index_start] # take the value from the lock chamber
            wlev_A = hydrodynamic_data["Water level"][station_index_start_node][time_index_stop]
            if same_direction:
                wlev_A = self.water_level[time_index_start] # take the value from the lock chamber
                wlev_B = hydrodynamic_data["Water level"][station_index_end_node][time_index_stop]

        return wlev_A, wlev_B

    def update_operation_planning(self, vessel, direction, operation_index, add_operation):
        """
        Updates the lock master's lock operation planning

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the lock operation: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
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
        operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']
        operation_planning.loc[operation_index, 'maximum_individual_delay'] = np.max(vessel_planning[vessel_planning.operation_index == operation_index].delay)
        operation_planning.loc[operation_index, 'total_delay'] = np.sum(vessel_planning[vessel_planning.operation_index == operation_index].delay)

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
            the direction of the lock operation: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)

        Yields
        -------
        nothing

        """
        # unpack the lock master's vessel and lock operation plannings
        vessel_planning = self.lock_complex.vessel_planning
        operation_planning = self.lock_complex.operation_planning

        node_of_approach = self.end_node
        to_node = self.start_node
        if not direction:
            node_of_approach = self.start_node
            to_node = self.end_node

        # determine if the new lock operation should follow a empty lock operation (when the new lock operation has the same direction as the previous lock operation)
        previous_planned_operations = operation_planning[operation_planning.index <= operation_index]
        if not previous_planned_operations.empty:
            previous_planned_operation = previous_planned_operations.iloc[-1]
            if previous_planned_operation.bound == direction:
                self.add_empty_lock_operation_to_planning(operation_index, 1 - direction)
                operation_index += 1 # the new operation index lies now one ahead
        elif self.node_open != node_of_approach:
            self.add_empty_lock_operation_to_planning(operation_index, 1 - direction)
            self.env.process(self.convert_chamber(new_level = to_node))
            operation_index += 1  # the new operation index lies now one ahead

        # get the new previous planned operations (including the empty one)
        previous_planned_operations = operation_planning[operation_planning.index <= operation_index]

        # determine the index of the vessel in the lock master's vessel planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

        # determine the earlier possible arrival time of the vessel (vessel perspective)
        earliest_possible_time_lock_entry_start = vessel_planning.loc[vessel_planning_index,'time_lock_entry_start']

        # determine the time that the lock operation can start (operation perspective)
        time_lock_operation_start = self.calculate_lock_operation_start_time(vessel, operation_index, direction, prognosis=True)

        # correct the start time of the lock operation if it will fall outside of the operation hours of the lock complex
        operational_hours = self.operational_hours
        within_operation_hours = operational_hours[(time_lock_operation_start >= operational_hours.start_time) & (time_lock_operation_start <= operational_hours.stop_time)]
        if within_operation_hours.empty:
            first_available_hour = operational_hours[operational_hours.start_time >= time_lock_operation_start].iloc[0]
            time_lock_operation_start = first_available_hour.start_time

        # determine the time that vessel can start entering the lock
        time_lock_entry_start = (
            self.calculate_lock_entry_start_time(vessel, operation_index, direction, prognosis=True) + time_lock_operation_start
        )

        # add operation to the planning with information
        operation_planning.loc[operation_index, 'bound'] = direction
        operation_planning.loc[operation_index, 'vessels'] = [] # leave vessels empty for now
        operation_planning.loc[operation_index, 'capacity_L'] = self.lock_complex.lock_length - vessel.L
        operation_planning.loc[operation_index, 'capacity_B'] = self.lock_complex.lock_width - vessel.B

        # determine the minimum time that doors should be opened in advance of a vessel arrival and add this to the vessel planning
        minimum_advance_to_open_doors = self.minimum_advance_to_open_doors(vessel, direction)
        time_potential_lock_door_opening_stop = time_lock_entry_start - minimum_advance_to_open_doors
        vessel_planning.loc[vessel_planning_index, 'time_potential_lock_door_opening_stop'] = time_potential_lock_door_opening_stop
        if not previous_planned_operations.empty:
            previous_operation = previous_planned_operations.iloc[-1]
            if not len(previous_operation.vessels):
                if time_potential_lock_door_opening_stop < previous_operation.time_operation_stop:
                    operation_delay = previous_operation.time_operation_stop - time_potential_lock_door_opening_stop
                    time_lock_operation_start += operation_delay
                    time_lock_entry_start += operation_delay
                    vessel_planning.loc[vessel_planning_index, 'time_potential_lock_door_opening_stop'] += operation_delay

        # determine the lock entry stop and door opening stop time
        time_lock_entry_stop = (
            self.calculate_lock_entry_stop_time(vessel, operation_index, direction, prognosis=True) + time_lock_operation_start
        )
        time_lock_door_opening_stop = (
            self.calculate_lock_door_opening_time(vessel, operation_index, direction, prognosis=True) + time_lock_operation_start
        )

        # update the vessel and operation plannings with the above information
        vessel_planning.loc[vessel_planning_index, 'operation_index'] = operation_index
        vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start'] = time_lock_operation_start
        vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start'] = time_lock_entry_start
        vessel_planning.loc[vessel_planning_index, 'time_lock_entry_stop'] = time_lock_entry_stop

        operation_planning.loc[operation_index, 'time_operation_start'] = time_lock_operation_start
        operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop'] = time_lock_door_opening_stop
        operation_planning.loc[operation_index, 'time_entry_start'] = time_lock_entry_start
        operation_planning.loc[operation_index, 'time_entry_stop'] = time_lock_entry_stop

        # determine the delay time for the vessel to enter the lock
        vessel_entry_delay = time_lock_entry_start - earliest_possible_time_lock_entry_start

        # determine the time that the doors can start closing after the vessel has entered the lock (depending on whether the doors can close before the vessel has berthed), and add this to vessel planning
        if self.close_doors_before_vessel_is_laying_still:
            x_location_lock = operation_planning.loc[operation_index, 'capacity_L'] + 0.5 * vessel.L # determine the longitudinal location coordinate (x) of the vessel to calculate the time that the lock door closing process can start
            vessel_planning.loc[vessel_planning_index, 'time_potential_lock_door_closure_start'] = time_lock_entry_start + self.minimum_delay_to_close_doors(vessel, direction, after_lock_entry = True, x_location_lock = x_location_lock)
        else:
            vessel_planning.loc[vessel_planning_index, 'time_potential_lock_door_closure_start'] = time_lock_entry_stop

        # determine the moments in time of the lock operation process steps starts and stops
        time_door_closing_start, time_door_closing_stop, time_levelling_start, time_levelling_stop, time_door_opening_start, time_door_opening_stop = self.calculate_lock_operation_times(operation_index=operation_index,
                                                                                                                                                                                          last_entering_time=time_lock_entry_start,
                                                                                                                                                                                          start_time=time_lock_entry_stop,
                                                                                                                                                                                          vessel=vessel,
                                                                                                                                                                                          direction=direction)
        # determine the moments in time of the vessel's departure from the lock (steps starts and stops) and the time the operation has stopped and the doors can close again
        time_lock_departure_start = (
            self.calculate_lock_departure_start_time(vessel, operation_index, direction, prognosis=True) + time_door_opening_stop
        )
        time_lock_departure_stop = (
            self.calculate_lock_departure_stop_time(vessel, operation_index, direction, prognosis=True) + time_door_opening_stop
        )
        time_lock_operation_stop = (
            self.calculate_lock_operation_stop_time(vessel, operation_index, direction, prognosis=True) + time_door_opening_stop
        )
        time_lock_door_closing_start = (
            self.calculate_lock_door_closing_time(vessel, operation_index, direction, prognosis=True) + time_door_opening_stop
        )

        # determine the water levels and set the list of vessels
        wlev_A, wlev_B = self.determine_water_levels_before_and_after_levelling(time_levelling_start,time_levelling_stop, direction)
        vessels = [vessel]

        # add above information to the operation and vessel plannings
        operation_planning.loc[operation_index, 'vessels'] = vessels
        operation_planning.loc[operation_index, 'time_door_closing_start'] = time_door_closing_start
        operation_planning.loc[operation_index, 'time_door_closing_stop'] = time_door_closing_stop
        operation_planning.loc[operation_index, 'time_levelling_start'] = time_levelling_start
        operation_planning.loc[operation_index, 'time_levelling_stop'] = time_levelling_stop
        operation_planning.loc[operation_index, 'time_door_opening_start'] = time_door_opening_start
        operation_planning.loc[operation_index, 'time_door_opening_stop'] = time_door_opening_stop
        operation_planning.loc[operation_index, 'time_departure_start'] = time_lock_departure_start
        operation_planning.loc[operation_index, 'time_departure_stop'] = time_lock_departure_stop
        operation_planning.loc[operation_index, 'time_operation_stop'] = time_lock_operation_stop
        operation_planning.loc[operation_index, 'time_potential_lock_door_closure_start'] = time_lock_door_closing_start
        operation_planning.loc[operation_index, 'wlev_A'] = wlev_A
        operation_planning.loc[operation_index, 'wlev_B'] = wlev_B
        vessel_planning.loc[vessel_planning_index, 'time_lock_departure_start'] = time_lock_departure_start
        vessel_planning.loc[vessel_planning_index, 'time_lock_departure_stop'] = time_lock_departure_stop
        vessel_planning.loc[vessel_planning_index, 'time_lock_passing_stop'] = time_lock_operation_stop
        vessel_planning.loc[vessel_planning_index,'delay'] += vessel_entry_delay

        # include the update of the lock operation, if there is a rule of a required minumum number of vessels, then wait, otherwise the lock operation is ready
        if len(vessels) < self.min_vessels_in_operation:
            operation_planning.loc[operation_index, 'status'] = 'waiting for vessel'
        else:
            operation_planning.loc[operation_index, 'status'] = 'ready'

        # if there is another lock operation is planned after this newly planned operation, check if an additional empty lock operation is required (not if there is a policy that both lock doors are closed in between operations)
        later_planned_operations = operation_planning[operation_planning.index > operation_index]
        if not later_planned_operations.empty and not self.closing_doors_in_between_operations:
            next_operation = later_planned_operations.iloc[0]
            if direction == next_operation['bound']:
                self.add_empty_lock_operation_to_planning(operation_index, 1-direction)

        yield from []

    def add_vessel_to_planned_lock_operation(
        self, vessel, operation_index, direction, prognosis=True, vessel_planning=None, operation_planning=None
    ):
        """
        Add vessel to a planned lock operation

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        operation_index : int
            the index of the already planned lock operation to which the vessel is added to
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
        prognosis : bool
            .
        vessel_planning : pd.DataFrame [default = none]
            the vessel planning of the lock complex master
        operation_planning : pd.DataFrame [default = none]
            the operation planning of the lock complex master

        Returns
        -------
        operation_planning : pd.DataFrame
            the lock complex master's new planning of lock operations

        """
        # TODO: this is a very long and a bit of a chaotic function where a lot is going, we need to split this function up
        if operation_planning is None and vessel_planning is None:
            prognosis = False

        # unpack the lock complex' vessel and operations planning
        if vessel_planning is None:
            vessel_planning = self.lock_complex.vessel_planning

        if operation_planning is None:
            operation_planning = self.lock_complex.operation_planning

        # determine the vessel index in the lock complex master's vessel planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

        # set sailing in time gap and vessel entry delay time
        sailing_in_gap = pd.Timedelta(seconds=0)
        vessel_entry_delay = pd.Timedelta(seconds=0)

        # determine the number of vessels that are already assigned to the lock operation to which the vessels is/will be added
        vessels_in_operation = operation_planning.loc[operation_index, 'vessels']

        # add vessel to the operation if it is not yet part of it
        if vessel not in vessels_in_operation:
            vessels_in_operation.append(vessel)
            operation_planning.loc[operation_index, 'vessels'] = vessels_in_operation #TODO: is they redundant? or do we need to overwrite the information in the operation planning dataframe again
            self.calculate_sailing_time_to_approach_point(
                vessel, direction, operation_index=operation_index
            )  # TODO: can this be removed?

            # if there is a rule that prescribes a minimum amount of vessels in the lock operation and this condition is satisfied, put an operation-object in the FilterStore to communicate that the earlier waiting vessels do not have to wait any longer
            if self.min_vessels_in_operation and len(vessels_in_operation) == self.min_vessels_in_operation and not prognosis:
                Operation = namedtuple('Operation', 'operation_index')
                operation = Operation(operation_index)
                yield self.wait_for_other_vessel_to_arrive.put(operation)
                yield self.env.timeout(0.) #required to update the vessel_planning TODO: we may want to try to remove this

                # calculate the required sailing in time delay
                sailing_in_gap = self.calculate_sailing_in_time_delay(
                    vessel, operation_index, direction, prognosis=False, overwrite=False
                )

        # calculate the new arrival time at the lock entry
        time_arrival_time_at_lock_entry = vessel_planning.loc[vessel_planning_index,'time_lock_passing_start'] + sailing_in_gap

        # if the condition of minimum amount of vessels in the lock operation is satisfied, change status of lock operation to ready
        if len(vessels_in_operation) >= self.min_vessels_in_operation:
            operation_planning.loc[operation_index, 'status'] = 'ready'

        # update capacity parameters
        operation_planning.loc[operation_index, 'capacity_L'] -= vessel.L
        operation_planning.loc[operation_index, 'capacity_B'] -= vessel.B

        # determine the other vessels in the lock and the planned times to start the operation and the time that the lock door has been opened
        other_vessels_in_operation = operation_planning.loc[operation_index, 'vessels'][:-1]
        time_lock_operation_start = operation_planning.loc[operation_index, 'time_operation_start']
        potential_lock_door_opening_stop = operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop']

        # determine if the arrival time is now outside of the lock's operational hours, determine the additional delay TODO: does this overcomplicate things? never will a newly arriving vessel delay the whole lock planning because it will fall outside the operational hours, probably the
        operational_hours = self.operational_hours
        within_operation_hours = operational_hours[(time_arrival_time_at_lock_entry >= operational_hours.start_time) & (time_arrival_time_at_lock_entry <= operational_hours.stop_time)]
        if within_operation_hours.empty:
            first_available_hour = operational_hours[operational_hours.start_time >= (time_arrival_time_at_lock_entry + vessel_entry_delay)].iloc[0]
            vessel_entry_delay += first_available_hour.start_time - (time_arrival_time_at_lock_entry + vessel_entry_delay)
            if time_lock_operation_start < first_available_hour.start_time:
                time_lock_operation_start = first_available_hour.start_time

        # determine the time that the vessel has to be at the approach point
        time_first_vessel_required_to_be_at_lock_approach = (time_arrival_time_at_lock_entry + vessel_entry_delay)

        # correct start time of lock operation if there are no other vessels scheduled in the lock and the approach start time lies behond than the earlier estimated operation start time
        if time_first_vessel_required_to_be_at_lock_approach > operation_planning.loc[operation_index, 'time_operation_start'] and not len(other_vessels_in_operation):
            time_lock_operation_start = time_first_vessel_required_to_be_at_lock_approach

        # add to vessel entry delay if the time of starting the approach lies ahead of the operation start time
        elif time_first_vessel_required_to_be_at_lock_approach < operation_planning.loc[operation_index, 'time_operation_start']:
            vessel_entry_delay += operation_planning.loc[operation_index, 'time_operation_start']-time_first_vessel_required_to_be_at_lock_approach

        # add the delay to the expected time of lock entry to the vessel
        if vessel_entry_delay > pd.Timedelta(seconds=0):
            time_arrival_time_at_lock_entry += vessel_entry_delay

        # update the vessel planning based on the above delays
        time_vessel_entry_start = self.calculate_vessel_entry_start_time(vessel,direction) + time_arrival_time_at_lock_entry
        time_lock_entry_stop = (
            self.calculate_lock_entry_stop_time(vessel, operation_index, direction) + time_arrival_time_at_lock_entry
        )
        vessel_planning.loc[vessel_planning_index, 'operation_index'] = operation_index
        vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start'] = time_arrival_time_at_lock_entry
        vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start'] = time_vessel_entry_start
        vessel_planning.loc[vessel_planning_index, 'time_lock_entry_stop'] = time_lock_entry_stop

        # determine the operation start delay
        operation_start_delay = time_lock_operation_start - operation_planning.loc[operation_index, 'time_operation_start']

        # determine the times of door closing, levelling and door opening: if lock entry stop time or extract them when the new lock entry stop time is ahead of the door closing start time TODO: check if this is correct
        if time_lock_entry_stop < operation_planning.loc[operation_index, 'time_door_closing_start']:
            time_door_closing_start = operation_planning.loc[operation_index, 'time_door_closing_start']
            time_door_closing_stop = operation_planning.loc[operation_index, 'time_door_closing_stop']
            time_levelling_start = operation_planning.loc[operation_index, 'time_levelling_start']
            time_levelling_stop = operation_planning.loc[operation_index, 'time_levelling_stop']
            time_door_opening_start = operation_planning.loc[operation_index, 'time_door_opening_start']
            time_door_opening_stop = operation_planning.loc[operation_index, 'time_door_opening_stop']
        else:
            time_door_closing_start, time_door_closing_stop, time_levelling_start, time_levelling_stop, time_door_opening_start, time_door_opening_stop = self.calculate_lock_operation_times(operation_index=operation_index,
                                                                                                                                                                                              last_entering_time=time_vessel_entry_start,
                                                                                                                                                                                              start_time=time_lock_entry_stop,
                                                                                                                                                                                              vessel=vessel,
                                                                                                                                                                                              direction=direction)

        # update the lock master's vessel and lock operation planning by adding the operation start and vessel entry delay
        operation_planning.loc[operation_index, 'time_operation_start'] += operation_start_delay
        if vessel_entry_delay > pd.Timedelta(seconds=0):
            vessel_planning.loc[vessel_planning_index,'delay'] += vessel_entry_delay
        operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop'] += operation_start_delay

        # if there is a delay in the start op the operation: update the vessel planning of the previous arriving vessels of this operation
        if operation_start_delay > pd.Timedelta(seconds=0):
            for vessel_index,other_vessel in enumerate(other_vessels_in_operation):
                other_vessel_planning_index = vessel_planning[vessel_planning.id == other_vessel.id].iloc[-1].name
                vessel_planning.loc[other_vessel_planning_index, 'time_potential_lock_door_opening_stop'] += operation_start_delay
                vessel_planning.loc[other_vessel_planning_index, 'time_potential_lock_door_closure_start'] += operation_start_delay
                vessel_planning.loc[other_vessel_planning_index, 'time_arrival_at_waiting_area'] += operation_start_delay
                vessel_planning.loc[other_vessel_planning_index, 'time_arrival_at_lineup_area'] += operation_start_delay
                vessel_planning.loc[other_vessel_planning_index, 'time_lock_passing_start'] += operation_start_delay
                vessel_planning.loc[other_vessel_planning_index, 'time_lock_entry_start'] += operation_start_delay
                vessel_planning.loc[other_vessel_planning_index, 'time_lock_entry_stop'] += operation_start_delay
                vessel_planning.loc[other_vessel_planning_index, 'delay'] += operation_start_delay
                if vessel_index < len(other_vessels_in_operation)-1:
                    next_vessel = other_vessels_in_operation[vessel_index+1]
                    next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name

                    # if there is slack in planning, plan two subsequent entering vessels closer to each other by adjusting the 'operation start' delay
                    operation_start_delay = (vessel_planning.loc[other_vessel_planning_index, 'time_lock_entry_start'] - vessel_planning.loc[next_vessel_planning_index, 'time_lock_entry_start'])

        # if there is a delay in the departure of the vessels, also include that in the planning
        additional_sailing_out_delay = time_door_opening_stop - operation_planning.loc[operation_index, 'time_door_opening_stop']
        if additional_sailing_out_delay > pd.Timedelta(seconds=0):
            for other_vessel in other_vessels_in_operation:
                other_vessel_planning_index = vessel_planning[vessel_planning.id == other_vessel.id].iloc[-1].name
                vessel_planning.loc[other_vessel_planning_index, 'time_lock_departure_start'] += additional_sailing_out_delay
                vessel_planning.loc[other_vessel_planning_index, 'time_lock_departure_stop'] += additional_sailing_out_delay
                vessel_planning.loc[other_vessel_planning_index, 'time_lock_passing_stop'] += additional_sailing_out_delay
                vessel_planning.loc[other_vessel_planning_index, 'delay'] += additional_sailing_out_delay

        # determine water levels to be included in the planning
        wlev_A, wlev_B = self.determine_water_levels_before_and_after_levelling(time_levelling_start, time_levelling_stop, direction)

        # update the values of the entry start, and (if there are no other vessels) overwrite the operation start
        if not len(other_vessels_in_operation):
            operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop'] = potential_lock_door_opening_stop
            operation_planning.loc[operation_index, 'time_operation_start'] = time_lock_operation_start
            operation_planning.loc[operation_index, 'time_entry_start'] = time_vessel_entry_start
        else:
            operation_planning.loc[operation_index, 'time_entry_start'] += operation_start_delay

        # update the operation planning with the above information
        operation_planning.loc[operation_index, 'time_entry_stop'] = time_lock_entry_stop
        operation_planning.loc[operation_index, 'time_door_closing_start'] = time_door_closing_start
        operation_planning.loc[operation_index, 'time_door_closing_stop'] = time_door_closing_stop
        operation_planning.loc[operation_index, 'time_levelling_start'] = time_levelling_start
        operation_planning.loc[operation_index, 'time_levelling_stop'] = time_levelling_stop
        operation_planning.loc[operation_index, 'time_door_opening_start'] = time_door_opening_start
        operation_planning.loc[operation_index, 'time_door_opening_stop'] = time_door_opening_stop
        operation_planning.loc[operation_index, 'maximum_individual_delay'] = np.max(vessel_planning[vessel_planning.operation_index == operation_index].delay)
        operation_planning.loc[operation_index, 'total_delay'] = np.sum(vessel_planning[vessel_planning.operation_index == operation_index].delay)

        # determine the new departure and operation start and stop times
        time_lock_departure_start = (
            self.calculate_lock_departure_start_time(vessel, operation_index, direction) + time_door_opening_stop
        )
        time_vessel_departure_start = (
            self.calculate_vessel_departure_start_time(vessel, operation_index, direction) + time_door_opening_stop
        )
        time_lock_departure_stop = (
            self.calculate_lock_departure_stop_time(vessel, operation_index, direction) + time_door_opening_stop
        )
        time_vessel_departure_stop = (
            self.calculate_vessel_departure_stop_time(vessel, operation_index, direction) + time_door_opening_stop
        )
        time_lock_operation_stop = (
            self.calculate_lock_operation_stop_time(vessel, operation_index, direction) + time_door_opening_stop
        )
        time_vessel_passing_stop = (
            self.calculate_vessel_passing_stop_time(vessel, operation_index, direction) + time_door_opening_stop
        )
        time_lock_door_closing_start = (
            self.calculate_lock_door_closing_time(vessel, operation_index, direction) + time_door_opening_stop
        )

        # update vessel and operation plannings accordingly
        operation_planning.loc[operation_index, 'time_departure_start'] = time_lock_departure_start
        operation_planning.loc[operation_index, 'time_departure_stop'] = time_lock_departure_stop
        operation_planning.loc[operation_index, 'time_operation_stop'] = time_lock_operation_stop
        operation_planning.loc[operation_index, 'time_potential_lock_door_closure_start'] = time_lock_door_closing_start
        operation_planning.loc[operation_index, 'wlev_A'] = wlev_A
        operation_planning.loc[operation_index, 'wlev_B'] = wlev_B
        vessel_planning.loc[vessel_planning_index, 'time_potential_lock_door_opening_stop'] = time_vessel_entry_start - self.minimum_advance_to_open_doors(vessel, direction)
        if self.close_doors_before_vessel_is_laying_still:
            vessel_planning.loc[vessel_planning_index, 'time_potential_lock_door_closure_start'] = time_vessel_entry_start + self.minimum_delay_to_close_doors(vessel, direction)
        else:
            vessel_planning.loc[vessel_planning_index, 'time_potential_lock_door_closure_start'] = time_door_closing_start

        vessel_planning.loc[vessel_planning_index, 'time_lock_departure_start'] = time_vessel_departure_start
        vessel_planning.loc[vessel_planning_index, 'time_lock_departure_stop'] = time_vessel_departure_stop
        vessel_planning.loc[vessel_planning_index, 'time_lock_passing_stop'] = time_vessel_passing_stop

        # update previous lock operations TODO: provide posibility to move lockages ahead of earlier delayed ones, if they can start earlier than these lockages
        previous_planned_operations = operation_planning[operation_planning.index < operation_index]
        if not previous_planned_operations.empty:
            if previous_planned_operations.iloc[-1].time_potential_lock_door_closure_start < operation_planning.loc[operation_index,'time_potential_lock_door_opening_stop']:
                pass

        # update the next lock operations if the previous lock operation caused a delay
        next_planned_operations = operation_planning[operation_planning.index > operation_index]
        for next_operation_index, next_operation_info in next_planned_operations.iterrows():

            # determine time delay of the process of sailing into the lock if the next operation in the planning confict with the delayed operation
            sailing_in_delay = pd.Timedelta(seconds=0)
            if not len(next_operation_info) and time_lock_door_closing_start > next_operation_info.time_potential_lock_door_opening_stop:
                sailing_in_delay = time_lock_door_closing_start - next_operation_info.time_potential_lock_door_opening_stop
            elif len(next_operation_info) and time_lock_operation_stop > next_operation_info.time_operation_start:
                sailing_in_delay = time_lock_operation_stop - next_operation_info.time_operation_start

            # determine the new start time of the next operation (dependening on whether it will fall withing the operation hours)
            new_operation_start = operation_planning.loc[next_operation_index, 'time_operation_start'] + sailing_in_delay
            within_operation_hours = operational_hours[(new_operation_start >= operational_hours.start_time)&(new_operation_start <= operational_hours.stop_time)]
            if within_operation_hours.empty:
                first_available_hour = operational_hours[operational_hours.start_time >= new_operation_start].iloc[0]
                sailing_in_delay += first_available_hour.start_time - new_operation_start

            # break loop if there is no delay (next operations will then also not experience a delay)
            if not sailing_in_delay.total_seconds() > 0:
                break

            # update the operation planning if there is a delay
            operation_planning.loc[next_operation_index, 'time_potential_lock_door_opening_stop'] += sailing_in_delay
            operation_planning.loc[next_operation_index, 'time_operation_start'] += sailing_in_delay
            operation_planning.loc[next_operation_index, 'time_entry_start'] += sailing_in_delay
            operation_planning.loc[next_operation_index, 'time_entry_stop'] += sailing_in_delay

            # update the vessel planning
            next_vessels = next_operation_info.vessels
            next_direction = next_operation_info.bound
            last_vessel_entering_time = operation_planning.loc[next_operation_index, 'time_entry_start']
            for next_vessel_index,next_vessel in enumerate(next_vessels):
                next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name
                vessel_planning.loc[next_vessel_planning_index, 'time_potential_lock_door_opening_stop'] += sailing_in_delay
                vessel_planning.loc[next_vessel_planning_index, 'time_potential_lock_door_closure_start'] += sailing_in_delay
                vessel_planning.loc[next_vessel_planning_index, 'time_arrival_at_lineup_area'] += sailing_in_delay
                vessel_planning.loc[next_vessel_planning_index, 'time_lock_passing_start'] += sailing_in_delay
                vessel_planning.loc[next_vessel_planning_index, 'time_lock_entry_start'] += sailing_in_delay
                vessel_planning.loc[next_vessel_planning_index, 'time_lock_entry_stop'] += sailing_in_delay
                last_vessel_entering_time = vessel_planning.loc[next_vessel_planning_index, 'time_lock_entry_start']
                if next_vessel_index != len(next_vessels)-1:
                    next_next_vessel = next_vessels[next_vessel_index + 1]
                    next_next_vessel_planning_index = vessel_planning[vessel_planning.id == next_next_vessel.id].iloc[-1].name

                    # determine sailing in delay for next vessel (it can be that there is some slack time between two vessel arrivals)
                    sailing_in_delay = pd.Timedelta(seconds=0)
                    if vessel_planning.loc[next_next_vessel_planning_index, 'time_lock_entry_start'] < vessel_planning.loc[next_vessel_planning_index, 'time_lock_entry_start']:
                        sailing_in_delay += self.calculate_sailing_in_time_delay(
                            next_next_vessel,
                            next_operation_index,
                            next_direction,
                            minimum_difference_with_previous_vessel=True,
                            overwrite=False,
                        )

            # determine the new start and stop times of the lock operation (i.e., door-closing, levelling, door-opening) as it can be that the levelling time is now changed due to the shift of this operation in time (i.e., due to tides)
            time_doors_closing = operation_planning.loc[next_operation_index, 'time_entry_stop']
            time_door_closing_start, time_door_closing_stop, time_levelling_start, time_levelling_stop, time_door_opening_start, time_door_opening_stop = self.calculate_lock_operation_times(operation_index=next_operation_index,
                                                                                                                                                                                              last_entering_time=last_vessel_entering_time,
                                                                                                                                                                                              start_time=time_doors_closing,
                                                                                                                                                                                              vessel=next_vessel,
                                                                                                                                                                                              direction=direction)
            # update the operation planning accordingly
            operation_planning.loc[next_operation_index, 'time_door_closing_start'] = time_door_closing_start
            operation_planning.loc[next_operation_index, 'time_door_closing_stop'] = time_door_closing_stop
            operation_planning.loc[next_operation_index, 'time_levelling_start'] = time_levelling_start
            delay_after_levelling = time_levelling_stop - operation_planning.loc[next_operation_index, 'time_levelling_stop']
            operation_planning.loc[next_operation_index, 'time_levelling_stop'] = time_levelling_stop
            operation_planning.loc[next_operation_index, 'time_door_opening_start'] = time_door_opening_start
            operation_planning.loc[next_operation_index, 'time_door_opening_stop'] = time_door_opening_stop
            if delay_after_levelling > pd.Timedelta(seconds=0):
                operation_planning.loc[next_operation_index, 'time_departure_start'] += delay_after_levelling
                operation_planning.loc[next_operation_index, 'time_departure_stop'] += delay_after_levelling
                operation_planning.loc[next_operation_index, 'time_operation_stop'] += delay_after_levelling
                operation_planning.loc[next_operation_index, 'time_potential_lock_door_closure_start'] += delay_after_levelling
                operation_planning.loc[next_operation_index, 'total_delay'] += delay_after_levelling*len(next_vessels)
                operation_planning.loc[next_operation_index, 'maximum_individual_delay'] += delay_after_levelling

            # update also the departure information of the affected vessels
            for vessel_index,next_vessel in enumerate(next_vessels):
                next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name
                vessel_planning.loc[next_vessel_planning_index, 'time_lock_departure_start'] += delay_after_levelling
                vessel_planning.loc[next_vessel_planning_index, 'time_lock_departure_stop'] += delay_after_levelling
                vessel_planning.loc[next_vessel_planning_index, 'time_lock_passing_stop'] += delay_after_levelling
                vessel_planning.loc[next_vessel_planning_index, 'delay'] += delay_after_levelling
            time_lock_operation_stop = operation_planning.loc[next_operation_index, 'time_operation_stop']
            time_lock_door_closing_start = operation_planning.loc[next_operation_index, 'time_potential_lock_door_closure_start']

        return operation_planning

    def assign_vessel_to_lock_operation(self, vessel, direction):
        """
        Function that adds a vessel to the lock operation planning

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)

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
        time_lock_passing_start = vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start']
        time_lock_entry_start = vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start']

        # add to the vessel planning that the vessel has a delay (which is still 0 [s])
        vessel_planning.loc[vessel_planning_index, 'delay'] = pd.Timedelta(seconds=0)

        # determine the current time
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))

        # determine whether the planned approach fits within the operation hours of the lock and add a delay to the planned approach of the vessel when it is outside of the operational hours
        operational_hours = self.operational_hours
        within_operation_hours = operational_hours[(time_lock_passing_start >= operational_hours.start_time)&(time_lock_passing_start <= operational_hours.stop_time)]
        if within_operation_hours.empty:
            first_available_hour = operational_hours[operational_hours.start_time >= time_lock_passing_start].iloc[0]
            delay = first_available_hour.start_time - time_lock_passing_start
            time_lock_entry_start += delay
            vessel_planning.loc[vessel_planning_index, 'time_arrival_at_waiting_area'] += delay
            vessel_planning.loc[vessel_planning_index, 'time_arrival_at_lineup_area'] += delay
            vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start'] += delay
            vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start'] += delay
            vessel_planning.loc[vessel_planning_index, 'time_of_acceptance'] += delay
            vessel_planning.loc[vessel_planning_index, 'delay'] += delay

        # determine the maximum delay of an individual vessel in all the planned lock operation if the vessel is assigned to that operation
        maximum_individual_delay = operation_planning.maximum_individual_delay + (time_lock_entry_start - operation_planning.time_entry_stop)

        # filter the planned lock operations based on the following criteria to select available operations to which the vessel can be assigned
        mask_bound = operation_planning.bound == direction # lock operations in the same direction as the vessel
        mask_status = operation_planning.status == 'waiting for vessel' # lock operations that are still on hold (waiting for another vessel)
        mask_available = operation_planning.status != 'not available' # lock operations that are not unavailable
        mask_capacity_L = operation_planning.capacity_L >= vessel.L # lock operations that have a capacity in which the vessel fits longitudinally (based on the vessel's length)
        mask_capacity_B = operation_planning.capacity_B >= vessel.B # lock operations that have a capacity in which the vessel fits laterally (based on the vessel's beam) TODO: implement this later
        mask_max_waiting_time = maximum_individual_delay < pd.Timedelta(seconds=self.lock_complex.clustering_time) # lock operations that will not exceed the maximum set waiting time for individual vessels
        mask_empty_lock = operation_planning.vessels.apply(len) == 0 # lock operations that are still empty

        # max vessels mask: lock operations that do not exceed a maximum number of vessels
        mask_max_vessels = mask_available
        if self.max_vessels_in_operation:
            mask_max_vessels = operation_planning.vessels.apply(len) < self.max_vessels_in_operation

        # future operations mask: lock operations that still have to take place
        mask_future_operations = operation_planning.time_levelling_start >= current_time

        # combinations of the masks TODO: this part of the code should be improved in clarity
        mask_empty_future_lockages = mask_empty_lock&mask_future_operations # empty future lock operations
        mask_max_waiting_time = mask_max_waiting_time&~mask_empty_lock # non-empty lock operations with non-exceedance of the maximum waiting time
        mask_min_vessels = mask_future_operations # future operations that do not exceed a minimum required number of vessels
        if self.min_vessels_in_operation > 1:
            mask_min_vessels = operation_planning.vessels.apply(len) < self.min_vessels_in_operation
        mask_future_operations = (mask_empty_future_lockages&mask_max_waiting_time)|(mask_min_vessels&mask_future_operations)

        # select available operations TODO: this part of the code should be improved in clarity and readability
        available_operations = operation_planning[mask_available&mask_bound&mask_max_vessels&mask_capacity_L&mask_future_operations&(mask_min_vessels|mask_status|mask_empty_future_lockages|mask_max_waiting_time)].copy()
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

    def convert_chamber(self, new_level, vessel=None, close_doors=True, delay=0., direction = None):
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
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)

        Yields
        ------
        The conversion of the lock chamber
        """

        # if there is a delay -> yield time out
        start_delay = self.env.now
        while delay > 0:
            try:
                yield self.env.timeout(delay)
                delay = 0
            except simpy.Interrupt as e:
                delay -= (self.env.now - start_delay)

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
        yield from self.level_lock(new_level, vessel=vessel, direction = direction)
        yield from self.open_door()

    def close_door(self, delay=0.):
        """
        Lock operator closes the lock doors TODO: attribute for lock operator

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
                delay -= (self.env.now - start_delay)

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
                remaining_doors_closing_time -= (self.env.now - start_time_closing)

        # set water level to the side at which the door has been closed
        if self.env.vessel_traffic_service.hydrodynamic_information_path is not None:
            if isinstance(hydrodynamic_data,xr.Dataset):
                time_index = np.absolute(hydrodynamic_times - np.datetime64(datetime.datetime.fromtimestamp(self.env.now))).argmin().values
            else:
                time_index = np.absolute(hydrodynamic_times - np.datetime64(datetime.datetime.fromtimestamp(self.env.now))).argmin()

            if self.node_open == self.start_node:
                if isinstance(hydrodynamic_data,xr.Dataset):
                    station_index = np.where(np.array(list((hydrodynamic_data['STATION'].values))) == self.start_node)[0][0]
                    self.water_level[time_index:] = hydrodynamic_data["Water level"][station_index][time_index].values.copy()
                else:
                    station_index = np.where(np.array(list((hydrodynamic_data['STATION']))) == self.start_node)[0]
                    self.water_level[time_index:] = hydrodynamic_data["Water level"][station_index][time_index].copy()
            else:
                if isinstance(hydrodynamic_data,xr.Dataset):
                    station_index = np.where(np.array(list((hydrodynamic_data['STATION'].values))) == self.end_node)[0][0]
                    self.water_level[time_index:] = hydrodynamic_data["Water level"][station_index][time_index].values.copy()
                else:
                    station_index = np.where(np.array(list((hydrodynamic_data['STATION']))) == self.end_node)[0]
                    self.water_level[time_index:] = hydrodynamic_data["Water level"][station_index][time_index].copy()

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

    def level_lock(self, new_level, vessel=None, direction=None, same_direction=False):
        """
        Lock operator levels the water level of the lock chamber to the harbour side of the direction of the lock operation TODO: attribute for lock operator

        new_level : str
            node of the edge of lock complex to which the lock chamber is levelling
        vessel : type
            a type including the following parent-classes: PassesLockComplex, Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        direction : int
            the direction of the vessel: 0 (bound from node_A to node_B) or 1 (bound from node_B to node_A)
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
        levelling_time,_,_ = self.determine_levelling_time(t_start=self.env.now,direction=direction,same_direction=same_direction)

        # log the start of the event
        if vessel is not None:
            vessel.log_entry_v0("Levelling start", vessel.env.now, vessel.output.copy(), vessel.position_in_lock, )
        self.log_entry_v0("Lock chamber converting start", self.env.now, self.output.copy(), self.node_open, )

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
        self.log_entry_v0("Lock chamber converting stop", self.env.now, self.output.copy(), self.node_open, )
        if vessel is not None:
            vessel.log_entry_v0("Levelling stop", vessel.env.now, vessel.output.copy(), vessel.position_in_lock, )

        # release all lock elements that were requested, so the next process can start
        self.door_A.release(hold_door_A)
        self.levelling.release(hold_levelling)
        self.door_B.release(hold_door_B)

    def open_door(self, to_level=None, vessel=None, delay=0.):
        """
        Lock operator opens the lock doors TODO: attribute for lock operator

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
            except simpy.Interrupt as e: # if there is a delay -> yield time out with new delay (remaining delay added with a delay equal to the exception)
                delay -= self.env.now - start_delay
                if vessel is not None:
                    if e.cause is not None:
                        delay += float(e.cause)

        # delete attribute as form of communication of the vessel TODO: a bit complex, better do it in another way
        if vessel is not None:
            delattr(vessel,'door_open_request')

        # determine the water level in the lock chamber
        wlev_chamber = 0.
        if self.env.vessel_traffic_service.hydrodynamic_information_path is not None:
            if isinstance(hydrodynamic_data, xr.Dataset):
                time_index = np.absolute(hydrodynamic_times - np.datetime64(datetime.datetime.fromtimestamp(self.env.now))).argmin().values
            else:
                time_index = np.absolute(hydrodynamic_times - np.datetime64(datetime.datetime.fromtimestamp(self.env.now))).argmin()
            wlev_chamber = self.water_level[time_index]

        # determine to_level
        if to_level is None:
            to_level = self.node_open

        # determine the water level in the harbour
        wlev_harbour = 0.
        if self.env.vessel_traffic_service.hydrodynamic_information_path is not None:
            if isinstance(hydrodynamic_data, xr.Dataset):
                time_index = np.absolute(hydrodynamic_times - np.datetime64(datetime.datetime.fromtimestamp(self.env.now))).argmin().values
            else:
                time_index = np.absolute(hydrodynamic_times - np.datetime64(datetime.datetime.fromtimestamp(self.env.now))).argmin()
            if to_level == self.start_node:
                wlev_harbour = hydrodynamic_data["Water level"][0][time_index]
            else:
                wlev_harbour = hydrodynamic_data["Water level"][1][time_index]

        # determine the direction to which the vessels are sailing out
        if to_level == self.start_node:
            direction = 1
        else:
            direction = 0

        # ?
        same_direction = False
        if to_level == self.node_open:
            same_direction = True

        # if the water levels in the chamber and harbour are not aligned -> level lock again
        if np.abs(wlev_chamber - wlev_harbour) >= 0.1:
            if same_direction:
                direction = 1-direction
            yield from self.level_lock(to_level,direction=direction,same_direction=same_direction)
        else:
            self.node_open = to_level

        # adjust water level of the lock chamber to the harbour side
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        if self.env.vessel_traffic_service.hydrodynamic_information_path is not None:
            if isinstance(hydrodynamic_data, xr.Dataset):
                time_index = np.absolute(hydrodynamic_times - np.datetime64(current_time)).argmin().values + 1
                station_index = np.where(np.array(list((hydrodynamic_data['STATION'].values))) == self.node_open)[0][0]
            else:
                time_index = np.absolute(hydrodynamic_times - np.datetime64(current_time)).argmin() + 1
                station_index = np.where(np.array(list((hydrodynamic_data['STATION']))) == self.node_open)[0]
            self.water_level[time_index:] = hydrodynamic_data["Water level"][station_index,time_index:].copy()

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
                remaining_doors_opening_time -= (self.env.now - start_time_opening)

        # log the process stop
        self.log_entry_v0("Lock doors opening stop", self.env.now, self.output.copy(), self.node_open, )

        # determine which side the door is open to
        if self.node_open == self.start_node:
            self.door_A_open = True
        else:
            self.door_B_open = True

        # release all lock elements that were requested, so the next process can start
        self.door_A.release(hold_door_A)
        self.levelling.release(hold_levelling)
        self.door_B.release(hold_door_B)


class IsLockComplex(IsLockChamber,IsLockMaster):
    """Mixin-class: a lock complex object

    TODO: I would like the lock complex to be decoupled from its infrastructure, so that you can add multiple lock chambers, line-up areas and waiting areas
    Parent classes
    --------------
    IsLockChamber :
        lock complex has a lock chamber
    IsLockMaster :
        lock complex has a lock master

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

        # initialization
        super().__init__(start_node=self.node_A, end_node=self.node_B, *args, **kwargs)

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
        self.distance_waiting_area_A_to_end_edge_waiting_area_A = self.env.graph.edges[edge_waiting_area_A]["length_m"]
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
        self.distance_waiting_area_B_to_end_edge_waiting_area_B = self.env.graph.edges[edge_waiting_area_B]["length_m"]
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
        x_correction_inbound = x_lock_doorsA + self.lock_length/2
        x_correction_outbound = x_lock_doorsB - self.lock_length / 2

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
                        x_correction = x_correction_inbound
                    elif message_info.Message == f"Sailing from node {self.end_node} to node {self.start_node} start":
                        x_correction = x_correction_outbound
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
            lock_extend_x = np.array([x_lock_doorsA, x_lock_doorsA, x_lock_doorsB, x_lock_doorsB]) - x_correction_inbound
            ax.fill(lock_extend_x, [ylimmin, ylimmax, ylimmax, ylimmin], color="lightgrey", zorder=0)
        elif method == 'Plotly':
            fig.add_shape(type="rect",
                          x0=x_lock_doorsA - x_correction_inbound, x1=x_lock_doorsB - x_correction_inbound,
                          y0=ylimmin, y1=ylimmax,
                          fillcolor="lightgrey", opacity=0.5,
                          layer="below", line_width=0,
                          name="Lock Geometry")

        # plot the lock phases
        lock_df = pd.DataFrame(self.logbook)
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
                              x0=x_lock_doorsA - x_correction_inbound, x1=x_lock_doorsB - x_correction_inbound,
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
