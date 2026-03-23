"""This is the lock module as part of the OpenTNSim package. See the locking examples in the book for detailed descriptions."""

import datetime
import functools
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
    _get_vessel_sailing_speed_in_lock,
    _get_vessel_sailing_speed_out_lock,
    _get_vessel_sailing_in_speed,
    _get_vessel_sailing_out_speed,
    _check_if_vessel_is_first_vessel,
    _get_lock_gate_position,
    _get_lock_operation_to_and_from_node,
    _get_distance_to_lock,
    _get_lock_object_on_registration_node,
    _get_upcoming_lock_complexes,
    _get_upcoming_locks,
    determine_if_gate_is_closed,
    check_all_paths_through_registration,
    check_lock_complex_geometry,
)
from opentnsim.output import HasOutput
from opentnsim.utils import inherit_docstring
from IPython.display import display

@inherit_docstring
class LockComplexTraversable(Movable, HasMultiDiGraph):
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
        self.has_registered = False


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
            self.log_entry_v0("Sailing to waiting area stop", self.env.now, self.output.copy(), waiting_area.geometry)

        yield from self.request_to_pass_waiting_area(lock_chamber, waiting_area)


    def request_to_pass_waiting_area(self, lock_chamber, waiting_area):
        yield from lock_chamber.allow_vessel_to_pass_waiting_area(self, waiting_area)


    def sail_to_lock_chamber(self, lock_chamber, waiting_area, direction):
        # determines the geometry objects of the lock based on the direction of the vessel TODO: function?
        distance_to_lock = _get_distance_to_lock(lock_chamber, direction)

        lock_gate_position = _get_lock_gate_position(lock_chamber, direction)

        # correct the distance to the lock gate if the vessel is in the waiting area, located at the same edge of the lock
        lock_start_node, lock_end_node = _get_lock_operation_to_and_from_node(lock_chamber, direction)
        if (lock_start_node, lock_end_node) == waiting_area.edge:
            distance_to_lock -= waiting_area.distance_from_edge_start

        # log the start of sailing to the lock gate
        last_position_vessel = self.logbook[-1]["Geometry"]
        self.log_entry_v0("Sailing to first lock gate start", self.env.now, self.output.copy(), last_position_vessel,)

        # let vessel sail to the lock gate
        current_time = self.env.now
        vessel_speed = _get_vessel_sailing_in_speed(lock_chamber, self, direction)
        remaining_sailing_time = distance_to_lock / vessel_speed
        while remaining_sailing_time > 0:
            try:
                yield self.env.timeout(remaining_sailing_time)
                remaining_sailing_time = 0
            except simpy.Interrupt as e:
                remaining_sailing_time -= self.env.now - current_time
                remaining_sailing_distance = vessel_speed * remaining_sailing_time
                remaining_sailing_time = remaining_sailing_distance / self.current_speed

        # vessel entering now the lock -> delete the overruled speeds imposed on the vessel
        self.overruled_speed = self.overruled_speed.iloc[0:0]

        # log the stop of sailing to the lock gate
        self.log_entry_v0("Sailing to first lock gate stop", self.env.now, self.output.copy(), lock_gate_position, )


    def sail_to_position_in_lock_chamber(self, lock_chamber, direction):
        lock_gate_position = _get_lock_gate_position(lock_chamber, direction)
        edge = lock_chamber.edge
        if direction:
            edge = lock_chamber.edge_reversed

        # log the start of sailing to the position within the lock chamber
        self.log_entry_v0("Sailing to position in lock start", self.env.now, self.output.copy(), lock_gate_position, )

        # determine position in the lock chamber and distance to sail to this location
        self.distance_position_from_first_lock_gate = lock_chamber.length.level + 0.5 * self.L
        if not direction:
            distance_to_position_in_lock = lock_chamber.distance_from_start_node_to_lock_gate_A + \
                                           self.distance_position_from_first_lock_gate
        else:
            distance_to_position_in_lock = lock_chamber.distance_from_end_node_to_lock_gate_B + \
                                           self.distance_position_from_first_lock_gate

        self.position_in_lock = calculate_location_over_edges(self.env.graph, edge,
                                                              distance_to_position_in_lock, crs_m=lock_chamber.crs_m)

        # let vessel sail to the assigned location in the lock chamber
        vessel_speed = _get_vessel_sailing_speed_in_lock(lock_chamber, self)
        remaining_sailing_time = self.distance_position_from_first_lock_gate / vessel_speed
        while remaining_sailing_time > 0:
            try:
                yield self.env.timeout(remaining_sailing_time)
                remaining_sailing_time = 0
            except simpy.Interrupt as e:
                remaining_sailing_time -= self.env.now - start_sailing

        # log the stop of the sailing event to the assigned locaiton in the lock chamber
        self.log_entry_v0("Sailing to position in lock stop", self.env.now, self.output.copy(), self.position_in_lock, )


    def sail_out_of_lock_chamber(self, lock_chamber, direction):
        # log that the vessel can start sailing out of the lock (up to the lock gate)
        self.log_entry_v0("Sailing to second lock gate start", self.env.now, self.output.copy(), self.position_in_lock,)

        # determines the distance from the vessel to the lock gate that have to be passed
        lock_gate_position = _get_lock_gate_position(lock_chamber, 1 - direction)
        distance_in_lock_from_position = lock_chamber.lock_length - self.distance_position_from_first_lock_gate

        # determine the process of sailing to the lock gate that have to be passed (distance to these gate divided by the sailing out speed of the vessel)
        vessel_speed = _get_vessel_sailing_speed_out_lock(lock_chamber, self)
        sailing_out_time = distance_in_lock_from_position / vessel_speed
        sailing_out_start = self.env.now
        while sailing_out_time:
            try:
                yield self.env.timeout(sailing_out_time)
                sailing_out_time = 0
            except simpy.Interrupt as e:
                sailing_out_time -= self.env.now - sailing_out_start

        # log that the vessel can stops sailing out of the lock (up to the lock gate)
        self.log_entry_v0("Sailing to second lock gate stop", self.env.now, self.output.copy(), lock_gate_position,)


    def leave_lock_complex(self, lock_chamber, direction):
        # determines the geometry objects of the lock based on the direction of the vessel TODO: function?
        if not direction:
            lock_gate_position = lock_chamber.gate_B.geometry
            remaining_distance = lock_chamber.distance_from_end_node_to_lock_gate_B
            exit_geom = self.env.graph.nodes[lock_chamber.end_node]["geometry"]
        else:
            lock_gate_position = lock_chamber.gate_A.geometry
            remaining_distance = lock_chamber.distance_from_start_node_to_lock_gate_A
            exit_geom = self.env.graph.nodes[lock_chamber.start_node]["geometry"]

        # log that sailing out of the lock complex is starting
        self.log_entry_v0("Sailing to lock complex exit start", self.env.now, self.output.copy(), lock_gate_position)

        # let the vessel sail to the end of the lock complex
        vessel_speed = _get_vessel_sailing_out_speed(lock_chamber, self, direction)
        sailing_out_time = remaining_distance / vessel_speed
        sailing_out_start = self.env.now
        while sailing_out_time:
            try:
                yield self.env.timeout(sailing_out_time)
                sailing_out_time = 0
            except simpy.Interrupt as e:
                sailing_out_time -= self.env.now - sailing_out_start
                remaining_sailing_distance = vessel_speed * sailing_out_time
                sailing_out_time = remaining_sailing_distance / self.current_speed

        # log that sailing out of the lock complex is stopping and set that no distance has to be sailed along the edge (vessel is at end of lock complex)
        self.log_entry_v0("Sailing to lock complex exit stop", self.env.now, self.output.copy(), exit_geom, )
        self.distance_left_on_edge = 0


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
