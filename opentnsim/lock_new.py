"""This is the lock module as part of the OpenTNSim package. See the locking examples in the book for detailed descriptions."""

# package(s) related to the simulation
import bisect
import datetime
import math
import re
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# spatial libraries
from collections import namedtuple
import pyproj
import pytz
import shapely.geometry
import simpy
import xarray as xr
from netCDF4 import Dataset
from IPython.display import display

from opentnsim import core, output, graph

# Constants
knots_to_ms = knots = 0.514444444

class PassesLockComplex(core.Movable, graph.HasMultiDiGraph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_look_ahead_to_node_functions.append(self.pre_register_to_lock_master)
        self.on_pass_node_functions.append(self.register_to_lock_master)
        self.on_pass_edge_functions.append(self.sail_to_waiting_area)


    def pre_register_to_lock_master(self,origin):


        route = self.route
        origin_index = route.index(origin)
        lock_edge = []
        detector_node = []
        next_route = route[origin_index:]
        if len(next_route) <= 1:
            return

        lock = None
        for node in next_route:
            node_info = self.multidigraph.nodes[node]
            if 'Detector' in node_info.keys():
                lock_edge = node_info['Detector']
                detector_node = node
            if node in lock_edge and 'Lock' in self.multidigraph.edges[lock_edge].keys():
                lock = self.multidigraph.edges[lock_edge]['Lock'][0]
                break

        if lock is not None and lock.predictive:
            operation_planning = lock.operation_pre_planning
            vessel_planning = lock.vessel_pre_planning
            arrival_time = np.max([self.metadata['arrival_time'],pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))])
            for origin, destination in zip(route[:-1], route[1:]):
                k = sorted(self.multidigraph[origin][destination],key=lambda x: self.multidigraph[origin][destination][x]['geometry'].length)[0]
                edge = self.multidigraph[origin][destination][k]
                if origin == detector_node:
                    break
                arrival_time += pd.Timedelta(seconds=edge['length'] / self.env.vessel_traffic_service.provide_speed_over_edge(self,(origin, destination, k)))

            direction = 1
            if lock_edge[0] == lock.start_node:
                direction = 0

            lock.add_vessel_to_vessel_planning(self, direction, time_of_registration=arrival_time, pre_planning=True)
            operation_index, add_operation, available_operations = lock.assign_vessel_to_lock_operation(self, direction, pre_planning=True)
            if not available_operations.empty:
                operation_index = available_operations.iloc[0].name
                copy_operation_planning = operation_planning.copy()
                copy_vessel_planning = vessel_planning.copy()
                yield from lock.add_vessel_to_planned_lock_operation(self, operation_index, direction,vessel_planning=copy_vessel_planning,operation_planning=copy_operation_planning, pre_planning=True)
                if copy_operation_planning[copy_operation_planning.index >= operation_index].maximum_individual_delay.max() > pd.Timedelta(seconds=lock.clustering_time):
                    operation_index = len(operation_planning)
                    add_operation = True

            yield from lock.update_operation_planning(self, direction, operation_index, add_operation, pre_planning=True)


    def register_to_lock_master(self,origin):
        if 'Detector' in self.multidigraph.nodes[origin].keys():
            edge = self.multidigraph.nodes[origin]['Detector']
            if 'Lock' in self.multidigraph.edges[edge].keys():
                lock = self.multidigraph.edges[edge]['Lock'][0]
                yield from lock.register_vessel(self)


    def sail_to_waiting_area(self, origin, destination, *args, **kwargs):
        route = self.route
        origin_index = route.index(origin)
        next_route = route[origin_index:]
        if len(next_route) <= 1:
            return

        lock = None
        for node_start,node_stop in zip(next_route[:-1],next_route[1:]):
            k = sorted(self.multidigraph[node_start][node_stop],key=lambda x: self.multidigraph[node_start][node_stop][x]['geometry'].length)[0]
            lock_edge = (node_start,node_stop,k)
            if 'Lock' in self.multidigraph.edges[lock_edge].keys():
                lock = self.multidigraph.edges[lock_edge]['Lock'][0]
                vessel_planning = lock.vessel_planning
                if lock.predictive:
                    vessel_planning = lock.vessel_pre_planning

                planned_vessel_lock_passages = vessel_planning[vessel_planning.id == self.id]
                if not planned_vessel_lock_passages.empty:
                    last_planned_vessel_lock_passage = planned_vessel_lock_passages.iloc[-1]
                    if last_planned_vessel_lock_passage.time_lock_entry_start > pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now)):
                        break
                    else:
                        lock = None

        if lock is None:
            return

        vessel_planning = lock.vessel_planning
        operation_planning = lock.operation_planning
        if lock.predictive:
            vessel_planning = lock.vessel_pre_planning
            operation_planning = lock.operation_pre_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == self.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']

        if node_start == lock.start_node:
            direction = 0
            lock_start_node = lock.start_node
            waiting_area = lock.waiting_area_A
        else:
            direction = 1
            lock_start_node = lock.end_node
            waiting_area = lock.waiting_area_B

        if origin != waiting_area.edge[0]:
            return

        sailing_time_to_waiting_area, sailing_distance_to_waiting_area, vessel_speed = lock.calculate_sailing_time_to_waiting_area(self, direction, overwrite=False)
        sailing_time_to_waiting_area = sailing_time_to_waiting_area.total_seconds()
        if sailing_time_to_waiting_area:
            self.log_entry_v0("Sailing to waiting area start", self.env.now, self.output.copy(),self.logbook[-1]['Geometry'],)
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

        # Let vessel to wait in the waiting area
        yield from self.wait_in_waiting_area(waiting_area=waiting_area)

        # Release vessel from waiting area and let vessel continue
        yield waiting_area.waiting_area.release(self.waiting_area_request)

        #Vessel allowed to pass lock
        self.on_pass_edge_functions.append(lock.allow_vessel_to_sail_in_lock)
        self.on_pass_edge_functions.append(lock.initiate_levelling)
        self.on_pass_edge_functions.append(lock.allow_vessel_to_sail_out_of_lock)

        # Check if doors should be opened
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        first_in_lock = operation_planning.loc[operation_index].vessels[0] == self
        door_is_closed, doors_required_to_be_open, operation_time = lock.determine_if_door_is_closed(self,
                                                                                                     operation_index,
                                                                                                     direction,
                                                                                                     first_in_lock=first_in_lock)

        if door_is_closed and lock.closing_doors_in_between_operations:
            levelling_required = False
            if operation_time > pd.Timedelta(seconds=lock.doors_closing_time):
                levelling_required = True

            if (doors_required_to_be_open - operation_time) > current_time:
                delay = ((doors_required_to_be_open - operation_time) - current_time).total_seconds()
                self.door_open_request = self.env.process(lock.open_door(to_level=lock_start_node, delay=delay, vessel=self))
            else:
                if levelling_required:
                    lock.log_entry_v0("Lock chamber converting start", doors_required_to_be_open.round('s').to_pydatetime().timestamp() - operation_time.total_seconds(), self.output.copy(),lock_start_node, )
                    lock.log_entry_v0("Lock chamber converting stop", doors_required_to_be_open.round('s').to_pydatetime().timestamp() - lock.doors_opening_time, self.output.copy(),lock_start_node, )
                lock.log_entry_v0("Lock doors opening start", doors_required_to_be_open.round('s').to_pydatetime().timestamp() - lock.doors_opening_time, self.output.copy(),lock_start_node, )
                lock.log_entry_v0("Lock doors opening stop",doors_required_to_be_open.round('s').to_pydatetime().timestamp(),self.output.copy(), lock_start_node, )
                if not direction:
                    lock.node_open = lock.start_node
                else:
                    lock.node_open = lock.end_node

                if self.env.vessel_traffic_service.hydrodynamic_information_path:
                    time_index = np.absolute(hydrodynamic_times - np.datetime64(doors_required_to_be_open) - np.timedelta64(int(lock.doors_opening_time), 's')).argmin()
                    station_index = np.where(np.array(list((hydrodynamic_data['STATION']))) == lock.node_open)[0]
                    lock.water_level[time_index:] = hydrodynamic_data['Water level'][station_index, time_index:]
        self.distance -= sailing_distance_to_waiting_area


    def wait_in_waiting_area(self, waiting_area):
        lock = waiting_area.lock
        start_node = waiting_area.edge[0]
        if waiting_area.name == 'waiting_area_A':
            direction = 0
        else:
            direction = 1

        # Check if vessel has to wait for other vessels (rule-based clustering policy)
        vessel_planning = lock.lock_complex.vessel_planning
        operation_planning = lock.lock_complex.operation_planning
        if lock.predictive:
            vessel_planning = lock.lock_complex.vessel_pre_planning
            operation_planning = lock.lock_complex.operation_pre_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == self.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']

        sailing_to_approach = lock.calculate_sailing_time_to_approach(self, direction, start_node=start_node,overwrite=False)# - lock.calculate_sailing_time_to_waiting_area(self, direction, overwrite=False)[0]
        vessels_in_operation = operation_planning.loc[operation_index, 'vessels']
        waiting_start = lock.env.now
        if len(vessels_in_operation) < lock.min_vessels_in_operation:
            waiting_start = self.env.now
            self.log_entry_v0("Waiting for other vessel start", waiting_start, self.output.copy(), self.logbook[-1]['Geometry'],)
            request = lock.wait_for_other_vessel_to_arrive.get(lambda operation: operation.operation_index == operation_index)
            while operation_planning.loc[operation_index,'status'] == 'waiting for vessel':
                try:
                    yield request
                    operation_planning.loc[operation_index,'status'] = 'ready'
                except simpy.Interrupt as e:
                    operation_planning.loc[operation_index,'status'] = 'waiting for vessel'
            self.overruled_speed = self.overruled_speed.iloc[0:0]
            waiting_stop = self.env.now
            start_time_lock_operation = operation_planning.loc[operation_index, 'time_entry_start']
            if pd.Timestamp(datetime.datetime.fromtimestamp(waiting_stop)) + sailing_to_approach > start_time_lock_operation:
                vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
                vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
                vessel_planning.loc[vessel_planning_index, 'time_lock_entry_stop'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
                vessel_index = vessels_in_operation.index(self)
                if vessel_index == 0:
                    operation_planning.loc[operation_index, 'time_entry_start'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
                vessel_planning.loc[vessel_planning_index, 'time_arrival_at_lineup_area'] += pd.Timedelta(seconds=waiting_stop - waiting_start)
            self.log_entry_v0("Waiting for other vessel stop", self.env.now, self.output.copy(),self.logbook[-1]['Geometry'],)

        time_operation_start = vessel_planning.loc[vessel_planning_index,'time_lock_passing_start']
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(lock.env.now))
        time_at_approach = current_time + sailing_to_approach
        waiting_time = time_operation_start-time_at_approach
        operation_planning.loc[operation_index, 'status'] = 'ready'

        #remaining_static_waiting_time, waiting_time_while_sailing = lock.determine_waiting_time_while_sailing_to_lock(self,direction,waiting_time.total_seconds())
        remaining_static_waiting_time = waiting_time.total_seconds()
        waiting_time_while_sailing = 0.
        if remaining_static_waiting_time > 0.:
            self.log_entry_v0("Waiting start", self.env.now, self.output.copy(), self.logbook[-1]['Geometry'], )
            while remaining_static_waiting_time > 0.:
                try:
                    yield lock.env.timeout(remaining_static_waiting_time)
                    time_at_approach += pd.Timedelta(seconds=remaining_static_waiting_time)
                    remaining_static_waiting_time = 0.
                    time_operation_start = vessel_planning.loc[vessel_planning_index,'time_lock_passing_start']
                    remaining_static_waiting_time = (time_operation_start-time_at_approach).total_seconds()
                except simpy.Interrupt as e:
                    remaining_static_waiting_time -= lock.env.now - waiting_start
            self.log_entry_v0("Waiting stop", self.env.now, self.output.copy(), self.logbook[-1]['Geometry'], )

        if waiting_time_while_sailing:
            lock.overrule_vessel_speed(self,lock_end_node,waiting_time=waiting_time_while_sailing)
            self.process.interrupt()


class IsLockWaitingArea(core.HasResource, core.Identifiable, core.Log, output.HasOutput, graph.HasMultiDiGraph):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
    creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity
    """

    def __init__(
        self, edge, lock, distance_from_node, *args, **kwargs  # a string which indicates the location of the start of the waiting area
    ):
        node = edge[0]
        self.node = node
        self.edge = edge
        self.lock = lock
        self.distance_from_node = distance_from_node
        super().__init__(*args, **kwargs)
        """Initialization"""

        self.waiting_area = simpy.PriorityResource(self.env, capacity=1000000)
        self.location = self.env.vessel_traffic_service.provide_location_over_edges(edge[0],edge[1],distance_from_node)


class IsLockLineUpArea(core.HasResource, core.HasLength, core.Identifiable, core.Log, output.HasOutput, graph.HasMultiDiGraph):
    """Mixin class: Something has line-up area object properties as part of the lock complex [in SI-units]:
            creates a line-up area with the following resources:
                - enter_line_up_area: resource used when entering the line-up area (assures one-by-one entry of the line-up area by vessels)
                - line_up_area: resource with unlimited capacity used to formally request access to the line-up area
                - converting_while_in_line_up_area: resource used when requesting for an empty conversion of the lock chamber
                - pass_line_up_area: resource used to pass the second encountered line-up area"""

    def __init__(
        self,
        start_node, #a string which indicates the location of the start of the line-up area
        end_node, #a string which indicates the location of the start of the line-up area
        lineup_area_length, #a float which contains the length of the line-up area
        distance_from_start_edge,
        effective_lineup_area_length = None,
        passing_allowed = False,
        speed_reduction_factor = 0.75,
        k=0,
        *args,
        **kwargs):

        self.start_node = start_node
        self.end_node = end_node
        self.k = k
        self.lineup_area_length = self.effective_lineup_area_length = lineup_area_length
        self.distance_from_start_edge = distance_from_start_edge
        if effective_lineup_area_length:
            self.effective_lineup_area_length = effective_lineup_area_length
        self.passing_allowed = passing_allowed
        self.speed_reduction_factor = speed_reduction_factor
        super().__init__(length = self.effective_lineup_area_length, init = self.effective_lineup_area_length, *args, **kwargs)

        """Initialization"""
        # Geometry
        self.start_location = self.env.vessel_traffic_service.provide_location_over_edges(start_node, end_node, distance_from_start_edge)
        self.end_location = self.env.vessel_traffic_service.provide_location_over_edges(start_node, end_node, distance_from_start_edge+lineup_area_length)

        # Lay-Out
        self.enter_line_up_area = {start_node: simpy.PriorityResource(self.env, capacity=1),}  # used to regulate one by one entering of line-up area, so capacity must be 1
        self.line_up_area = {start_node: simpy.PriorityResource(self.env, capacity=100),}  # line-up area itself, infinite capacity, as this is regulated by the HasLength, so capacity = inf
        self.converting_while_in_line_up_area = {start_node: simpy.PriorityResource(self.env, capacity=1),}  # used to minimize the number of empty convertion requests by one by multiple waiting vessels, so capacity must be 1
        self.pass_line_up_area = {start_node: simpy.PriorityResource(self.env, capacity=1),}  # used to prevent vessel from entering the lock before all previously locked vessels have passed the line-up area one by one, so capacity must be 1


class IsLockChamber(core.HasResource, core.HasLength, core.Identifiable, core.Log, output.HasOutput, graph.HasMultiDiGraph):
    """Mixin class: Something which has lock chamber object properties as part of a lock complex [in SI-units] """

    def __init__(
        self,
        start_node, #a string which indicates the location of the first pair of lock doors
        end_node, #a string which indicates the location of the second pair of lock doors
        lock_length=200., #a float which contains the length of the lock chamber
        lock_width=30., #a float which contains the width of the lock chamber
        lock_depth=15., #a float which contains the depth of the lock chamber
        k = 0,
        distance_from_start_node_to_lock_doors_A = 0.,
        distance_from_end_node_to_lock_doors_B = 0.,
        detector_nodes = [],
        doors_opening_time = 300.,  # a float which contains the time it takes to open the doors
        doors_closing_time = 300.,  # a float which contains the time it takes to close the doors
        disch_coeff = 0.,  # a float which contains the discharge coefficient of filling system
        opening_area = 0.,  # a float which contains the cross-sectional area of filling system
        opening_depth = 0.,  # a float which contains the depth at which filling system is located
        speed_reduction_factor_lock_chamber = 0.3,
        start_sailing_out_time_after_doors_have_been_opened = 0.,
        sailing_time_before_opening_lock_doors = 180,
        sailing_time_before_closing_lock_doors = 60,
        sailing_distance_to_crossing_point = 500,
        passage_time_door = 300.,
        sailing_in_time_gap_through_doors = 180.,
        sailing_out_time_gap_through_doors = 180.,
        sailing_in_time_gap_after_berthing_previous_vessel=None,
        sailing_out_time_gap_after_berthing_previous_vessel=None,
        sailing_in_speed_sea = 2*knots,
        sailing_out_speed_sea = 2*knots,
        sailing_in_speed_canal= 2 * knots,
        sailing_out_speed_canal= 2 * knots,
        minimum_manoeuvrability_speed = 2*knots,
        levelling_time = 600.,
        grav_acc = 9.81, #a float which contains the gravitational acceleration
        time_step = 10.,
        gate_opening_time = 60.,
        node_open = None,
        conditions = None,
        priority_rules = None,
        used_as_one_way_traffic_regulation = False,
        seed_nr = None,
        *args,
        **kwargs):

        """Initialization"""
        # Properties
        if disch_coeff != 0 and opening_area != 0:
            levelling_time = 0.
        self.lock_length = lock_length
        self.lock_width = lock_width
        self.lock_depth = lock_depth
        self.disch_coeff = disch_coeff
        self.grav_acc = grav_acc
        self.opening_area = opening_area
        self.opening_depth = opening_depth
        self.levelling_time = levelling_time
        self.start_sailing_out_time_after_doors_have_been_opened = start_sailing_out_time_after_doors_have_been_opened
        self.sailing_time_before_opening_lock_doors = sailing_time_before_opening_lock_doors
        self.sailing_time_before_closing_lock_doors = sailing_time_before_closing_lock_doors
        self.sailing_in_time_gap_after_berthing_previous_vessel = sailing_in_time_gap_after_berthing_previous_vessel
        self.sailing_out_time_gap_after_berthing_previous_vessel = sailing_out_time_gap_after_berthing_previous_vessel
        self.sailing_in_speed_sea = sailing_in_speed_sea
        self.sailing_out_speed_sea = sailing_out_speed_sea
        self.sailing_in_speed_canal = sailing_in_speed_canal
        self.sailing_out_speed_canal = sailing_out_speed_canal
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
        self.detector_nodes = detector_nodes
        self.gate_opening_time = gate_opening_time
        self.door_A_open = True
        self.door_B_open = True
        if not detector_nodes:
            self.detector_nodes = [start_node,end_node]
        self.distance_from_start_node_to_lock_doors_A = distance_from_start_node_to_lock_doors_A
        self.distance_from_end_node_to_lock_doors_B = distance_from_end_node_to_lock_doors_B
        self.used_as_one_way_traffic_regulation = used_as_one_way_traffic_regulation
        self.converting_chamber = False
        self.vessel_planning = pd.DataFrame(index=pd.Index([]),columns=['id',
                                                                        'bound',
                                                                        'L',
                                                                        'B',
                                                                        'T',
                                                                        'operation_index',
                                                                        'time_of_registration',
                                                                        'time_of_acceptance',
                                                                        'time_arrival_at_waiting_area',
                                                                        'time_arrival_at_lineup_area',
                                                                        'time_lock_passing_start',
                                                                        'time_lock_entry_start',
                                                                        'time_lock_entry_stop',
                                                                        'time_lock_departure_start',
                                                                        'time_lock_departure_stop',
                                                                        'time_lock_passing_stop'])
        self.vessel_pre_planning = self.vessel_planning.copy()
        self.operation_planning = pd.DataFrame(index=pd.Index([],name='lock_operation'),columns=['bound',
                                                                                                 'vessels',
                                                                                                 'capacity_L',
                                                                                                 'capacity_B',
                                                                                                 'time_potential_lock_door_opening_stop',
                                                                                                 'time_operation_start',#See comments below
                                                                                                 'time_entry_start',    #See comments below
                                                                                                 'time_entry_stop',
                                                                                                 'time_door_closing_start',
                                                                                                 'time_door_closing_stop',
                                                                                                 'time_levelling_start',
                                                                                                 'time_levelling_stop',
                                                                                                 'time_door_opening_start',
                                                                                                 'time_door_opening_stop',
                                                                                                 'time_departure_start',
                                                                                                 'time_departure_stop', # Note that start and stop times of different operations can overlap, but entry start and departure stop can not
                                                                                                 'time_operation_stop', # Operation start and stop times are solely required when leaving and entering vessels need to pass each other at the safe crossing point
                                                                                                 'time_potential_lock_door_closure_start',
                                                                                                 'wlev_A',
                                                                                                 'wlev_B',
                                                                                                 'maximum_individual_delay',
                                                                                                 'total_delay',
                                                                                                 'status'])
        self.operation_pre_planning = self.operation_planning.copy()
        if seed_nr is not None:
            np.random.seed(seed_nr)

        super().__init__(lock_complex=self,capacity=100,length = lock_length, init = lock_length, *args, **kwargs)
        if self.env.vessel_traffic_service.hydrodynamic_information_path:
            global hydrodynamic_data
            hydrodynamic_data = Dataset(self.env.vessel_traffic_service.hydrodynamic_information_path)
            global hydrodynamic_times
            hydrodynamic_times = hydrodynamic_data['TIME'][:].data.astype("timedelta64[m]") + self.env.vessel_traffic_service.hydrodynamic_start_time

        if self.closing_doors_in_between_operations:
            self.door_A_open = False
            self.door_B_open = False
        else:
            if self.node_open == self.start_node:
                self.door_B_open = False
            else:
                self.door_A_open = False

        #Geometry on edge
        edge = self.env.FG.edges[start_node,end_node,0]
        length_edge = edge['length']
        if distance_from_start_node_to_lock_doors_A == 0 and distance_from_end_node_to_lock_doors_B == 0:
            self.distance_from_start_node_to_lock_doors_A = self.distance_from_end_node_to_lock_doors_B = length_edge / 2 - lock_length / 2
            self.location_lock_doors_A = self.env.vessel_traffic_service.provide_location_over_edges(start_node, end_node, self.distance_from_start_node_to_lock_doors_A)
            self.location_lock_doors_B = self.env.vessel_traffic_service.provide_location_over_edges(end_node, start_node, self.distance_from_end_node_to_lock_doors_B)
        if distance_from_start_node_to_lock_doors_A != 0:
            self.distance_from_end_node_to_lock_doors_B = length_edge - (distance_from_start_node_to_lock_doors_A+lock_length)
            self.location_lock_doors_A = self.env.vessel_traffic_service.provide_location_over_edges(start_node, end_node, self.distance_from_start_node_to_lock_doors_A)
            self.location_lock_doors_B = self.env.vessel_traffic_service.provide_location_over_edges(end_node, start_node, self.distance_from_end_node_to_lock_doors_B)

        self.lock_pos_length = simpy.Container(self.env, capacity=lock_length, init=lock_length)
        self.door_A= simpy.PriorityResource(self.env, capacity = 1)
        self.levelling = simpy.Resource(self.env, capacity=1)
        self.door_B = simpy.PriorityResource(self.env, capacity = 1)
        self.wait_for_other_vessel_to_arrive = simpy.FilterStore(self.env,capacity=100000000)
        self.wait_for_levelling = simpy.FilterStore(self.env,capacity=100000000)
        self.wait_for_other_vessels = simpy.FilterStore(self.env,capacity=100000000)

        # Operating
        self.doors_opening_time = doors_opening_time
        self.doors_closing_time = doors_closing_time

        # Water level
        assert start_node != end_node

        if self.node_open is None:
            self.node_open = np.random.choice([start_node, end_node])

        if self.env.vessel_traffic_service.hydrodynamic_information_path:
            station_index = np.where(np.array(list((hydrodynamic_data['STATION']))) == self.node_open)[0]
            water_level = hydrodynamic_data['Water level'][station_index][0][0]
            self.water_level = np.ones(len(hydrodynamic_data['Water level'][station_index,:][0]))*water_level

        for detector_node, lock_edge in zip(self.detector_nodes,[(self.start_node,self.end_node,self.k),(self.end_node,self.start_node,self.k)]):
            if 'Detector' not in self.multidigraph.nodes[detector_node]:
                self.multidigraph.nodes[detector_node]['Detector'] = lock_edge

        # Add to the graph:
        if "FG" in dir(self.env):
            k = sorted(self.multidigraph[self.start_node][self.end_node],
                       key=lambda x: self.multidigraph[self.start_node][self.end_node][x]['geometry'].length)[0]
            # Add the lock to the edge or append it to the existing list
            if "Lock" not in self.multidigraph.edges[self.start_node, self.end_node, k].keys():
                self.multidigraph.edges[self.start_node, self.end_node, k]["Lock"] = [self]
                self.multidigraph.edges[self.end_node, self.start_node, k]["Lock"] = [self]
            else:
                self.multidigraph.edges[self.start_node, self.end_node, k]["Lock"].append(self)
                self.multidigraph.edges[self.end_node, self.start_node, k]["Lock"].append(self)


    def vessel_sailing_speed_in_lock(self, vessel, x_location_lock, P_used=None):
        h0 = self.lock_depth
        speed = self.sailing_in_speed_canal
        if vessel.bound == 'inbound':
            speed = self.sailing_in_speed_sea

        if P_used is None:
            P_used = self.P_used_to_break_in_lock
        if P_used is not None:
            deceleration_stats = vessel.distance_to_desired_speed(v_target=0.1, P_used=P_used, h0=h0,v0=speed)
            length_decelerating = deceleration_stats['sailed_distance']
            length_to_lock_doors_steady = x_location_lock - length_decelerating
            sailing_time_steady = length_to_lock_doors_steady / speed
            sailing_time_decelerating = deceleration_stats['sailing_time']
            speed = x_location_lock / (sailing_time_steady + sailing_time_decelerating)
        return speed


    def vessel_sailing_speed_out_lock(self, vessel, x_location_lock, P_used=None):
        h0 = self.lock_depth
        speed = self.sailing_out_speed_sea
        if vessel.bound == 'inbound':
            speed = self.sailing_out_speed_canal
        if P_used is None:
            P_used = self.P_used_to_accelerate_in_lock
        if P_used is not None:
            acceleration_stats = vessel.distance_to_desired_speed(v_target=speed, P_used=P_used, h0=h0, v0=0.1)
            length_accelerating = acceleration_stats['sailed_distance']
            length_to_lock_doors_steady = np.max([x_location_lock - length_accelerating,0.])
            sailing_time_steady = length_to_lock_doors_steady / speed
            sailing_time_accelerating = acceleration_stats['sailing_time']
            speed = x_location_lock / (sailing_time_steady + sailing_time_accelerating)
        return speed


    def vessel_sailing_in_speed(self, vessel, direction, P_used=None, h0=17, from_crossing_point=False):
        edge = None
        distance_to_lock_doors = 0.
        if not direction:
            edge = (self.start_node,self.end_node,self.k)
            distance_to_lock_doors = self.distance_from_start_node_to_lock_doors_A
        elif direction:
            edge = (self.end_node,self.start_node,self.k)
            distance_to_lock_doors = self.distance_from_end_node_to_lock_doors_B

        speed = speed_edge = self.env.vessel_traffic_service.provide_speed_over_edge(vessel, edge)
        if P_used is None:
            P_used = self.P_used_to_break_before_lock

        if P_used is not None:
            speed = self.sailing_in_speed_canal
            if vessel.bound == 'inbound':
                speed = self.sailing_in_speed_sea
            deceleration_stats = vessel.distance_to_desired_speed(v_target=speed,P_used=P_used,h0=h0,v0=speed)
            length_decelerating = deceleration_stats['sailed_distance']
            dt = deceleration_stats['time'][-1] - deceleration_stats['time'][-2]
            length_to_lock_doors_steady = distance_to_lock_doors - length_decelerating
            sailing_time_steady = length_to_lock_doors_steady/speed
            sailing_time_decelerating = deceleration_stats['sailing_time']
            speed = distance_to_lock_doors/(sailing_time_steady+sailing_time_decelerating)

            if 'overruled_speed' in dir(vessel) and not vessel.overruled_speed.empty:
                if edge in vessel.overruled_speed.index:
                    speed = vessel.overruled_speed.loc[edge,'Speed']

            if self.sailing_distance_to_crossing_point and from_crossing_point:
                if deceleration_stats['sailed_distance'] >= self.sailing_distance_to_crossing_point:
                    decelerating_distances_rev = -1*(np.array(list(reversed(deceleration_stats['distance']))) - deceleration_stats['distance'][-1])
                    sailing_time = dt * (np.absolute(decelerating_distances_rev - self.sailing_distance_to_crossing_point).argmin())
                else:
                    sailing_time = deceleration_stats['sailing_time']
                    remaining_distance = self.sailing_distance_to_crossing_point - deceleration_stats['sailed_distance']
                    sailing_time += remaining_distance/speed_edge
                speed = self.sailing_distance_to_crossing_point / sailing_time
        return speed


    def vessel_sailing_out_speed(self, vessel, direction, P_used=None, h0=17, until_crossing_point=False):
        edge = None
        distance_to_exit = 0.
        if not direction:
            edge = (self.start_node, self.end_node, self.k)
            distance_to_exit = self.distance_from_end_node_to_lock_doors_B
        elif direction:
            edge = (self.end_node, self.start_node, self.k)
            distance_to_exit = self.distance_from_start_node_to_lock_doors_A

        speed = speed_edge = self.env.vessel_traffic_service.provide_speed_over_edge(vessel, edge)
        if P_used is None:
            P_used = self.P_used_to_accelerate_after_lock
        if P_used is not None:
            speed = self.sailing_out_speed_sea
            if vessel.bound == 'inbound':
                speed = self.sailing_out_speed_canal
            acceleration_stats = vessel.distance_to_desired_speed(v_target=speed, P_used=P_used, h0=h0, v0=speed)
            dt = acceleration_stats['time'][-1]-acceleration_stats['time'][-2]
            length_accelerating = acceleration_stats['sailed_distance']
            length_exit_steady = distance_to_exit - length_accelerating
            sailing_time_steady = length_exit_steady / speed
            sailing_time_accelerating = acceleration_stats['sailing_time']
            speed = distance_to_exit / (sailing_time_steady + sailing_time_accelerating)

            if 'overruled_speed' in dir(vessel) and not vessel.overruled_speed.empty:
                if edge in vessel.overruled_speed.index:
                    speed = vessel.overruled_speed.loc[edge, 'Speed']

            if self.sailing_distance_to_crossing_point and until_crossing_point:
                if acceleration_stats['sailed_distance'] <= self.sailing_distance_to_crossing_point:
                    sailing_time = acceleration_stats['sailing_time']
                    remaining_distance_to_crossing_point = self.sailing_distance_to_crossing_point - acceleration_stats['sailed_distance']
                    sailing_time += remaining_distance_to_crossing_point/speed_edge
                else:
                    sailing_time = dt*(np.absolute(np.array(acceleration_stats['distance'])-self.sailing_distance_to_crossing_point).argmin())
                speed = self.sailing_distance_to_crossing_point/sailing_time

        return speed


    def determine_levelling_time(self, t_start, direction=None, wlev_init=None, same_direction=False, prediction=False):
        if isinstance(t_start,float):
            t_start = np.datetime64(datetime.datetime.fromtimestamp(t_start))
        elif isinstance(t_start,datetime.datetime):
            t_start = np.datetime64(t_start)
        elif isinstance(t_start,pd.Timestamp):
            t_start = np.array([t_start], dtype=np.datetime64)[0]

        if self.env.vessel_traffic_service.hydrodynamic_information_path or hydrodynamic_data is not None:
            stationA_index = np.where(np.array(list((hydrodynamic_data['STATION']))) == self.start_node)[0][0]
            stationB_index = np.where(np.array(list((hydrodynamic_data['STATION']))) == self.end_node)[0][0]
            H_A = hydrodynamic_data["Water level"][stationA_index].copy()
            H_B = hydrodynamic_data["Water level"][stationB_index].copy()

        if direction is None:
            if self.node_open == self.start_node:
                direction = 0
            else:
                direction = 1

        if same_direction:
            direction = 1 - direction

        if self.levelling_time or (not self.env.vessel_traffic_service.hydrodynamic_information_path and hydrodynamic_data is None):
            if not prediction:
                if self.env.vessel_traffic_service.hydrodynamic_information_path:
                    t_index_final = np.absolute(hydrodynamic_times - (t_start + np.timedelta64(int(self.levelling_time), 's'))).argmin()
                    if not direction:
                        self.water_level[t_index_final:] = H_B[t_index_final:].copy()
                    else:
                        self.water_level[t_index_final:] = H_A[t_index_final:].copy()
            return self.levelling_time, [], []

        # Initialize time and water level arrays
        A_ch = self.lock_length * self.lock_width
        m = self.disch_coeff
        g = self.grav_acc
        dt = self.time_step
        T1 = self.gate_opening_time
        t_final = 3600*12.5
        t = np.arange(0, t_final + float(dt), float(dt))
        z = np.zeros_like(t)
        A_s = np.linspace(0, self.opening_area, int(T1 / float(dt)))
        A_s = np.append(A_s, [self.opening_area] * (len(z) - len(A_s)))
        time_index = np.absolute(hydrodynamic_times - t_start).argmin()
        H_time = hydrodynamic_times.astype(float)
        H_A_init = H_A[time_index]
        H_B_init = H_B[time_index]

        if wlev_init is None:
            wlev_init = self.water_level[time_index]

        if not direction:
            z[0] = H_B_init - wlev_init

        else:
            z[0] = H_A_init - wlev_init
        # Euler's method
        for i in range(len(t) - 1):
            H_Ai = np.interp((np.timedelta64(int(i * float(dt) * 10 ** 6), 'us') + t_start - np.datetime64('1970-01-01')) / np.timedelta64(1, 'us'), H_time, H_A)
            H_Aii = np.interp((np.timedelta64(int((i + 1) * float(dt) * 10 ** 6), 'us') + t_start - np.datetime64('1970-01-01')) / np.timedelta64(1, 'us'), H_time, H_A)
            H_Bi = np.interp((np.timedelta64(int(i * float(dt) * 10 ** 6), 'us') + t_start - np.datetime64('1970-01-01')) / np.timedelta64(1, 'us'), H_time, H_B)
            H_Bii = np.interp((np.timedelta64(int((i + 1) * float(dt) * 10 ** 6), 'us') + t_start - np.datetime64('1970-01-01')) / np.timedelta64(1, 'us'), H_time, H_B)
            deltaH_A = H_Aii - H_Ai
            deltaH_B = H_Bii - H_Bi
            z_i = abs(z[i])
            if not direction:
                to_wlev_change = - deltaH_B
            else:
                to_wlev_change = - deltaH_A

            dz_dt = -m * A_s[i] * np.sqrt(2 * g * np.max([0, z_i])) / A_ch
            if z[i] < 0:
                dz_dt = -dz_dt

            dz = dz_dt * float(dt) + to_wlev_change
            z[i + 1] = z[i] + dz
            if np.sign(z[i + 1]) != np.sign(z[i]):
                z[i + 1] = 0

            if np.abs(z[i + 1]) <= 0.05:
                z[(i + 1):] = np.nan
                break

        if len(np.argwhere(np.isnan(z))):
            levelling_time = t[np.argwhere(np.isnan(z))[0]][0]

        if not prediction:
            t_index_final = np.absolute(hydrodynamic_times - (t_start+np.timedelta64(int(levelling_time),'s'))).argmin()
            if not direction:
                self.water_level[t_index_final:] = H_B[t_index_final:].copy()
            else:
                self.water_level[t_index_final:] = H_A[t_index_final:].copy()
        return levelling_time, t, z


class IsLockMaster(core.SimpyObject):
    def __init__(self,
                 lock_complex,
                 min_vessels_in_operation = 0,
                 max_vessels_in_operation = 100,
                 clustering_time = 0.5*60*60,
                 minimize_door_open_times=False,
                 closing_doors_in_between_operations = False,
                 close_doors_before_vessel_is_laying_still = False,
                 predictive=False,
                 operational_hour_start_times=None,
                 operational_hour_stop_times=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock_complex = lock_complex
        self.min_vessels_in_operation = min_vessels_in_operation
        self.max_vessels_in_operation = max_vessels_in_operation
        self.clustering_time = clustering_time
        self.minimize_door_open_times = minimize_door_open_times
        self.closing_doors_in_between_operations = closing_doors_in_between_operations
        self.close_doors_before_vessel_is_laying_still = close_doors_before_vessel_is_laying_still
        self.predictive = predictive

        if operational_hour_start_times is not None and operational_hour_stop_times is not None:
            operational_hours = self.create_operational_hours(operational_hour_start_times,operational_hour_stop_times)
        else:
            operational_hours = self.create_operational_hours([self.env.simulation_start],[self.env.simulation_stop])
        self.operational_hours = operational_hours


    def create_operational_hours(self,start_times,stop_times):
        operational_hours = pd.DataFrame(columns=['start_time', 'stop_time'])
        for start_time,stop_time in zip(start_times,stop_times):
            operational_hours.loc[len(operational_hours),:] = [start_time,stop_time]
        return operational_hours


    def register_vessel(self, vessel):
        vessel_planning = self.vessel_planning
        operation_planning = self.operation_planning
        if self.predictive:
            vessel_planning = self.vessel_pre_planning
            operation_planning = self.operation_pre_planning

        # Determine the orientation of the vessel
        if vessel.origin == self.lock_complex.detector_nodes[0]:
            direction = 0
            lock_end_node = self.lock_complex.end_node
            waiting_area = self.waiting_area_A
        else:
            direction = 1
            lock_end_node = self.lock_complex.start_node
            waiting_area = self.waiting_area_B

        # Add vessel to vessel planning (already done when lock master is all-mighty predictive)
        if not self.predictive:
            self.add_vessel_to_vessel_planning(vessel, direction)
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

        # Add vessel to lock operation planning (already done when lock master is all-mighty predictive), else subtract operation_index from pre-assignment
        add_operation = False
        available_operations = pd.DataFrame()
        if not self.predictive:
            operation_index, add_operation,available_operations = self.assign_vessel_to_lock_operation(vessel, direction)
        else:
            operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']

        if not available_operations.empty:
            operation_index = available_operations.iloc[0].name
            copy_operation_planning = operation_planning.copy()
            copy_vessel_planning = vessel_planning.copy()
            yield from self.add_vessel_to_planned_lock_operation(vessel, operation_index, direction,vessel_planning=copy_vessel_planning,operation_planning=copy_operation_planning)
            if copy_operation_planning[copy_operation_planning.index >= operation_index].maximum_individual_delay.max() > pd.Timedelta(seconds=self.clustering_time):
                operation_index = len(operation_planning)
                add_operation = True


        # Update lock operation planning based on lockage assignment (already done when lock master is all-mighty predictive)
        if not self.predictive:
            yield from self.update_operation_planning(vessel, direction, operation_index, add_operation)
            operation_index = vessel_planning.loc[vessel_planning_index, 'operation_index']

        # Request waiting area:
        vessel.waiting_area_request = waiting_area.waiting_area.request()
        yield vessel.waiting_area_request

        # Check if vessel speed should be changed
        next_operation = operation_planning.loc[operation_index]
        vessel_time_lock_entry_start = vessel_planning.loc[vessel_planning_index,'time_lock_entry_start']
        if len(next_operation.vessels) > 1 and self.minimize_door_open_times:
            other_vessel = next_operation.vessels[-2]
            other_vessel_planning_index = vessel_planning[vessel_planning.id == other_vessel.id].iloc[-1].name
            preceding_vessel_waiting_time_to_shorten_door_open_time = vessel_time_lock_entry_start - datetime.timedelta(seconds=self.sailing_in_time_gap_through_doors) - vessel_planning.loc[other_vessel_planning_index, 'time_lock_entry_start']
            sailing_time = self.calculate_sailing_time_and_distance_to_lock(other_vessel, lock_end_node)
            if sailing_time is not None:
                total_distance_to_lock = sailing_time.Distance.sum()
                total_time_to_lock = sailing_time.Time.sum()
                planned_arrival_time = vessel_planning.loc[other_vessel_planning_index, 'time_lock_entry_start'] + preceding_vessel_waiting_time_to_shorten_door_open_time
                if preceding_vessel_waiting_time_to_shorten_door_open_time.total_seconds() > 0 and total_time_to_lock > 0:
                    average_speed = total_distance_to_lock / total_time_to_lock
                    overruled_speed = np.max([self.minimum_manoeuvrability_speed, total_distance_to_lock / (preceding_vessel_waiting_time_to_shorten_door_open_time.total_seconds() + total_time_to_lock)])
                    delay = total_distance_to_lock / overruled_speed - total_distance_to_lock / average_speed
                    difference_waiting_time = preceding_vessel_waiting_time_to_shorten_door_open_time.total_seconds() - delay
                    planned_arrival_time = planned_arrival_time - pd.Timedelta(seconds=difference_waiting_time)
                    arrival_time_difference = vessel_time_lock_entry_start - planned_arrival_time
                    if arrival_time_difference <= pd.Timedelta(seconds=self.doors_closing_time + self.doors_opening_time) and delay > 0:
                        vessel_planning.loc[other_vessel_planning_index, 'time_lock_entry_start'] += datetime.timedelta(seconds=delay)
                        if 'process' in dir(other_vessel):
                            if 'door_open_request' in dir(other_vessel):
                                self.overrule_vessel_speed(other_vessel, lock_end_node, waiting_time=delay)
                                other_vessel.process.interrupt()
                                operation_planning.loc[operation_index, 'time_entry_start'] += datetime.timedelta(seconds=delay)
                                operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop'] += datetime.timedelta(seconds=delay)
                                vessel_planning.loc[other_vessel_planning_index, 'time_potential_lock_door_opening_stop'] += datetime.timedelta(seconds=delay)
                                vessel_planning.loc[other_vessel_planning_index, 'time_lock_entry_stop'] += datetime.timedelta(seconds=delay)
                                other_vessel.door_open_request.interrupt(str(delay))


    def determine_waiting_time_while_sailing_to_lock(self, vessel, direction, waiting_time):
        if not direction:
            lock_start_node = self.start_node
            waiting_area = self.waiting_area_A
            distance_to_doors = self.distance_from_start_node_to_lock_doors_A-waiting_area.distance_from_node
            distance_after_doors = self.lock_complex.lock_length + self.distance_from_end_node_to_lock_doors_B
        else:
            lock_start_node = self.end_node
            waiting_area = self.waiting_area_B
            distance_to_doors = self.distance_from_end_node_to_lock_doors_B-waiting_area.distance_from_node
            distance_after_doors = self.lock_complex.lock_length + self.distance_from_start_node_to_lock_doors_A

        if vessel.origin != lock_start_node:
            route_vessel = nx.dijkstra_path(self.env.FG, vessel.origin, lock_start_node)
        else:
            route_vessel = nx.dijkstra_path(self.env.FG, vessel.origin, vessel.destination)

        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
        last_event = pd.DataFrame(vessel.logbook).iloc[-1]
        passed_time = 0
        if 'Sailing' in last_event.Message:
            passed_time = (last_event.Timestamp - current_time).total_seconds()

        sailing_time = self.env.vessel_traffic_service.provide_sailing_time(vessel, route_vessel)
        index_sailing_on_first_edge = sailing_time.iloc[0].name
        index_mask = sailing_time.index == index_sailing_on_first_edge
        interpolation = 1 - passed_time / sailing_time.iloc[0].Time
        sailing_time.loc[sailing_time[index_mask].index, 'Distance'] = sailing_time.loc[sailing_time[index_mask].index, 'Distance'] * interpolation
        sailing_time.loc[sailing_time[index_mask].index, 'Time'] = sailing_time.loc[sailing_time[index_mask].index, 'Time'] * interpolation
        total_distance_to_lock = sailing_time.loc[:, 'Distance'].sum()
        total_sailing_time = sailing_time.loc[:, 'Time'].sum()
        if vessel.origin != lock_start_node:
            fraction = (total_distance_to_lock-distance_after_doors)/total_distance_to_lock
            total_distance_to_lock = total_distance_to_lock*fraction
            total_sailing_time = total_sailing_time*fraction
        else:
            last_sailing_speed = sailing_time.iloc[-1]['Distance']/sailing_time.iloc[-1]['Time']
            total_distance_to_lock += distance_to_doors
            total_sailing_time +=  distance_to_doors/(last_sailing_speed*self.lock_complex.speed_reduction_factor)

        average_speed = total_distance_to_lock / total_sailing_time
        total_waiting_time_while_sailing = total_distance_to_lock / (average_speed - self.minimum_manoeuvrability_speed)
        remaining_static_waiting_time = np.max([0,waiting_time - total_waiting_time_while_sailing])
        waiting_time_while_sailing = waiting_time - remaining_static_waiting_time
        return remaining_static_waiting_time,waiting_time_while_sailing


    def calculate_sailing_time_and_distance_to_lock(self, vessel, lock_end_node):
        vessel_df = pd.DataFrame(vessel.logbook)
        if vessel_df.empty:
            return None

        reversed_vessel_df = vessel_df.iloc[::-1]
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
        for index,message in reversed_vessel_df.iterrows():
            if 'node' in message.Message:
                break
        passed_time = (current_time - message.Timestamp).total_seconds()

        distance = self.distance_from_start_node_to_lock_doors_A
        if lock_end_node != self.end_node:
            distance = self.distance_from_end_node_to_lock_doors_B

        route_other_vessel = nx.dijkstra_path(self.env.FG, vessel.origin, lock_end_node)
        sailing_time = self.env.vessel_traffic_service.provide_sailing_time(vessel, route_other_vessel)
        column_names = list(sailing_time.columns)
        distance_column_iloc = column_names.index('Distance')
        time_column_iloc = column_names.index('Time')
        speed_column_iloc = column_names.index('Speed')
        sailing_time.iloc[-1, distance_column_iloc] = distance
        sailing_time.iloc[-1, time_column_iloc] = distance / sailing_time.iloc[-1, speed_column_iloc]

        if not vessel.overruled_speed.empty:
            for edge, overruled_speed in vessel.overruled_speed.iterrows():
                edge_index_mask = sailing_time.index == edge
                sailing_time.loc[edge_index_mask, 'Speed'] = overruled_speed.Speed
                sailing_time.loc[edge_index_mask, 'Time'] = sailing_time.loc[edge_index_mask, 'Distance'] / sailing_time.loc[edge_index_mask, 'Speed']

        index_sailing_on_first_edge = sailing_time[sailing_time.index.isin([(vessel.origin,route_other_vessel[1],0)])].iloc[0].name
        index_mask = sailing_time.index == index_sailing_on_first_edge
        interpolation = 1 - passed_time / sailing_time.loc[index_mask].Time
        sailing_time.loc[sailing_time[index_mask].index, 'Distance'] = sailing_time.loc[sailing_time[index_mask].index, 'Distance'] * interpolation
        sailing_time.loc[sailing_time[index_mask].index, 'Time'] = sailing_time.loc[sailing_time[index_mask].index, 'Time'] * interpolation
        sailing_time['Speed'] = sailing_time['Speed'].astype(float)
        return sailing_time


    def overrule_vessel_speed(self, vessel, lock_end_node, waiting_time=0):
        sailing_time = self.calculate_sailing_time_and_distance_to_lock(vessel, lock_end_node)
        if sailing_time is not None:
            average_speed = sailing_time.loc[:, 'Distance'].sum()/sailing_time.loc[:, 'Time'].sum()
            overruled_speed = np.max([self.minimum_manoeuvrability_speed, sailing_time.loc[:, 'Distance'].sum()/(sailing_time.loc[:, 'Time'].sum() + waiting_time)])
            sailing_time = sailing_time.iloc[::-1]
            iteration = 0
            while not np.abs(average_speed-overruled_speed) <= 0.00001:
                if iteration == 100:
                    break
                speed_difference = average_speed - overruled_speed
                speed_mask = sailing_time.Speed > self.minimum_manoeuvrability_speed
                sailing_time.loc[sailing_time[speed_mask].index, 'Speed'] -= speed_difference
                sailing_time.loc[sailing_time[speed_mask].index, 'Time'] = sailing_time.loc[sailing_time[speed_mask].index, 'Distance'] / sailing_time.loc[sailing_time[speed_mask].index, 'Speed']
                speed_mask = sailing_time.Speed < self.minimum_manoeuvrability_speed
                sailing_time.loc[sailing_time[speed_mask].index, 'Speed'] = self.minimum_manoeuvrability_speed
                sailing_time.loc[sailing_time[speed_mask].index, 'Time'] = sailing_time.loc[sailing_time[speed_mask].index, 'Distance'] / sailing_time.loc[sailing_time[speed_mask].index, 'Speed']
                average_speed = sailing_time.Distance.sum()/sailing_time.Time.sum()
                iteration += 1
            for edge,sailing_time_info in sailing_time.iterrows():
                vessel.overruled_speed.loc[edge] = sailing_time_info.Speed


    def initiate_levelling(self, origin, destination, vessel=None, k=0, *args, **kwargs):
        if 'Lock' in vessel.multidigraph.edges[origin, destination, k].keys():
            lock = vessel.multidigraph.edges[origin, destination, k]['Lock'][0]
            vessel_planning = lock.vessel_planning
            operation_planning = lock.operation_planning
            if lock.predictive:
                vessel_planning = lock.vessel_pre_planning
                operation_planning = lock.operation_pre_planning

            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']
            this_operation = operation_planning.loc[operation_index]
            current_node = lock.node_open
            if current_node == lock.start_node:
                direction = 0
                next_node = lock.end_node
            else:
                direction = 1
                next_node = lock.start_node

            vessels = this_operation.vessels

            #If vessel is the last assigned vessel in the lock
            if vessel == vessels[-1]:
                #Liberate waiting vessels in lock chamber
                for other_vessel in vessels[:-1]:
                    terminate_waiting_time_for_other_vessel = False
                    while not terminate_waiting_time_for_other_vessel:
                        try:
                            yield lock.wait_for_other_vessels.put(other_vessel)
                            terminate_waiting_time_for_other_vessel = True
                        except simpy.Interrupt as e:
                            terminate_waiting_time_for_other_vessel = False

                #Wait for other vessels to lay still
                delay = operation_planning.loc[operation_index].time_door_closing_start.round('s').to_pydatetime().timestamp() - lock.env.now
                if delay > 0:
                    yield lock.env.timeout(delay)

                #Convert lock chamber
                close_doors = True
                if lock.close_doors_before_vessel_is_laying_still and this_operation.time_door_closing_start < vessel_planning.loc[vessel_planning_index,'time_lock_entry_stop']:
                    close_doors = False

                yield from lock.convert_chamber(next_node, vessel, close_doors, direction=direction)

                #Liberate waiting vessels in lock chamber
                for other_vessel in vessels[:-1]:
                    terminate_levelling_for_other_vessel = False
                    while not terminate_levelling_for_other_vessel:
                        try:
                            yield lock.wait_for_levelling.put(other_vessel)
                            terminate_levelling_for_other_vessel = True
                        except simpy.Interrupt as e:
                            terminate_levelling_for_other_vessel = False

            #If vessel is not the last assigned vessel
            else:
                #Wait for last assigned vessel of lock operation
                waiting_for_other_vessels = True
                while waiting_for_other_vessels:
                    try:
                        yield lock.wait_for_other_vessels.get(filter=(lambda request: request.id == vessel.id))
                        waiting_for_other_vessels = False
                    except simpy.Interrupt as e:
                        waiting_for_other_vessels = True

                #Follow the converting lock chamber
                vessel.log_entry_v0("Levelling start", vessel.env.now, vessel.output.copy(), vessel.position_in_lock,)
                waiting_for_levelling = True
                while waiting_for_levelling:
                    try:
                        yield lock.wait_for_levelling.get(filter=(lambda request: request.id == vessel.id))
                        waiting_for_levelling = False
                    except simpy.Interrupt as e:
                        waiting_for_levelling = True
                vessel.log_entry_v0("Levelling stop", vessel.env.now, vessel.output.copy(), vessel.position_in_lock,)

            #Determine and yield sailing out delay
            sailing_out_delay = lock.calculate_vessel_departure_start_time(vessel, operation_index, direction, pre_planning=lock.predictive).total_seconds()
            delay_start = vessel.env.now
            while sailing_out_delay:
                try:
                    yield vessel.env.timeout(sailing_out_delay)
                    sailing_out_delay = 0
                except simpy.Interrupt as e:
                    sailing_out_delay -= vessel.env.now - delay_start


    def allow_vessel_to_sail_out_of_lock(self, origin, destination, vessel=None, k=0, *args, **kwargs):
        if 'Lock' in vessel.multidigraph.edges[origin, destination, k].keys():
            lock = vessel.multidigraph.edges[origin, destination, k]['Lock'][0]
            vessel_planning = lock.vessel_planning
            operation_planning = lock.operation_planning
            if lock.predictive:
                vessel_planning = lock.vessel_pre_planning
                operation_planning = lock.operation_pre_planning

            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            direction = vessel_planning.loc[vessel_planning_index,'bound']
            vessel_operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']
            distance_in_lock_from_position = lock.lock_length - vessel.distance_position_from_first_lock_doors

            # Sail to lock
            if not direction:
                second_lock_doors_position = lock.location_lock_doors_B
                distance_from_lock_position = distance_in_lock_from_position
                remaining_distance = lock.distance_from_end_node_to_lock_doors_B
                exit_geom = vessel.env.FG.nodes[lock.end_node]['geometry']
                next_level_in_case_of_following_empty_lockage = lock.start_node
            else:
                second_lock_doors_position = lock.location_lock_doors_A
                distance_from_lock_position = distance_in_lock_from_position
                remaining_distance = lock.distance_from_start_node_to_lock_doors_A
                exit_geom = vessel.env.FG.nodes[lock.start_node]['geometry']
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
            vessel_speed = lock.vessel_sailing_speed_out_lock(vessel, distance_from_lock_position)
            sailing_out_time = distance_from_lock_position/vessel_speed
            sailing_out_start = vessel.env.now
            while sailing_out_time:
                try:
                    yield vessel.env.timeout(sailing_out_time)
                    sailing_out_time = 0
                except simpy.Interrupt as e:
                    sailing_out_time -= vessel.env.now - sailing_out_start
            vessel.log_entry_v0("Sailing to second lock doors stop", vessel.env.now, vessel.output.copy(),second_lock_doors_position, )

            vessel.on_pass_edge_functions.remove(lock.allow_vessel_to_sail_in_lock)
            vessel.on_pass_edge_functions.remove(lock.initiate_levelling)
            vessel.on_pass_edge_functions.remove(lock.allow_vessel_to_sail_out_of_lock)

            made_operation = operation_planning.loc[vessel_operation_index]
            vessels = made_operation.vessels
            is_last_vessel_sailing_out = vessels[-1] == vessel

            doors_can_be_closed = lock.determine_if_door_can_be_closed(vessel, direction, vessel_operation_index)

            next_operations = operation_planning[operation_planning.index == vessel_operation_index+1]
            next_lockage_is_empty = False
            if not next_operations.empty:
                next_operation = next_operations.iloc[0]
                if not len(next_operation.vessels):
                    next_lockage_is_empty = True
                    doors_can_be_closed = True
            if lock.closing_doors_in_between_operations:
                next_lockage_is_empty = False

            current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
            if is_last_vessel_sailing_out and doors_can_be_closed:
                if next_lockage_is_empty:
                    next_operation = next_operations.iloc[0]
                    door_closing_start = next_operation.time_door_closing_start
                    closing_delay = np.max([0,(door_closing_start - current_time).total_seconds()])
                    vessel.env.process(lock.convert_chamber(new_level=next_level_in_case_of_following_empty_lockage,vessel=None,close_doors=True,delay=closing_delay,direction=1-direction))
                elif lock.closing_doors_in_between_operations:
                    door_closing_time = made_operation.time_potential_lock_door_closure_start
                    delay = (door_closing_time-current_time).total_seconds()
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
            vessel.distance = 0


    def allow_vessel_to_sail_in_lock(self, origin, destination, vessel=None, k=0, *args, **kwargs):
        if 'Lock' in vessel.multidigraph.edges[origin,destination,k].keys():
            lock = vessel.multidigraph.edges[origin,destination,k]['Lock'][0]
            vessel_planning = lock.vessel_planning
            operation_planning = lock.operation_planning
            if lock.predictive:
                vessel_planning = lock.vessel_pre_planning
                operation_planning = lock.operation_pre_planning

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
                distance_to_lock_position -= waiting_area.distance_from_node


            vessel.log_entry_v0("Sailing to first lock doors start", vessel.env.now, vessel.output.copy(),vessel.logbook[-1]['Geometry'],)
            start_sailing = vessel.env.now
            vessel_speed = lock.vessel_sailing_in_speed(vessel, direction)
            remaining_sailing_time = distance_to_lock_position/vessel_speed
            while remaining_sailing_time:
                try:
                    yield vessel.env.timeout(remaining_sailing_time)
                    remaining_sailing_time = 0
                except simpy.Interrupt as e:
                    remaining_sailing_time -= (vessel.env.now - start_sailing)
                    remaining_sailing_distance = vessel_speed*remaining_sailing_time
                    remaining_sailing_time = remaining_sailing_distance/vessel.current_speed
                    if vessel_speed != vessel.current_speed:
                        distance = distance_to_lock_position-remaining_sailing_distance + waiting_area.distance_from_node
                        geometry = vessel.env.vessel_traffic_service.provide_location_over_edges(lock_start_node,lock_end_node,distance)
                        vessel.log_entry_v0("Sailing speed changed", vessel.env.now, vessel.output.copy(),geometry,)
                    #TODO for later research: the speed changes should be checked if they are realistic by combining it with a smoothly decreasing velocity (P_used)

            lock_accessed = False
            remaining_lock_length = lock.length.level
            vessel.overruled_speed = vessel.overruled_speed.iloc[0:0]

            while not lock_accessed:
                try:
                    yield lock.length.get(vessel.L)
                    lock_accessed = True
                except simpy.Interrupt as e:
                    lock_accessed = False

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
            if lock.close_doors_before_vessel_is_laying_still and ((lock.closing_doors_in_between_operations and doors_can_be_closed_between_vessel_arrivals) or (last_vessel_to_enter_lock and this_operation.time_door_closing_start < vessel_planning.loc[vessel_planning_index,'time_lock_entry_stop'])):
                vessel.env.process(lock.close_door(delay=delay_to_close_doors.total_seconds()))

            vessel.log_entry_v0("Sailing to position in lock start", vessel.env.now, vessel.output.copy(),first_lock_door_position, )
            vessel.distance_position_from_first_lock_doors = remaining_lock_length - 0.5*vessel.L
            if not direction:
                vessel.position_in_lock = vessel.env.vessel_traffic_service.provide_location_over_edges(lock.start_node,lock.end_node,lock.distance_from_start_node_to_lock_doors_A + vessel.distance_position_from_first_lock_doors)
            elif direction:
                vessel.position_in_lock = vessel.env.vessel_traffic_service.provide_location_over_edges(lock.end_node,lock.start_node,lock.distance_from_end_node_to_lock_doors_B + vessel.distance_position_from_first_lock_doors)

            vessel_speed = lock.vessel_sailing_speed_in_lock(vessel, vessel.distance_position_from_first_lock_doors)
            remaining_sailing_time = vessel.distance_position_from_first_lock_doors / vessel_speed
            while remaining_sailing_time > 0:
                try:
                    yield vessel.env.timeout(remaining_sailing_time)
                    remaining_sailing_time = 0
                except simpy.Interrupt as e:
                    remaining_sailing_time -= vessel.env.now - start_sailing
            vessel.log_entry_v0("Sailing to position in lock stop", vessel.env.now, vessel.output.copy(),vessel.position_in_lock,)

            doors_can_be_closed_between_vessel_arrivals = lock.determine_if_door_can_be_closed(vessel, direction, operation_index, between_arrivals=True)
            if not lock.close_doors_before_vessel_is_laying_still and not last_vessel_to_enter_lock and lock.closing_doors_in_between_operations and doors_can_be_closed_between_vessel_arrivals:
                vessel.env.process(lock.close_door())


    def add_vessel_to_vessel_planning(self,vessel,direction, time_of_registration=None, pre_planning=False):
        if time_of_registration is None:
            time_of_registration = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        vessel_planning = self.vessel_planning
        if self.predictive:
            vessel_planning = self.vessel_pre_planning
        vessel_planning_index = len(vessel_planning)
        vessel_planning.loc[vessel_planning_index, 'id'] = vessel.id
        vessel_planning.loc[vessel_planning_index, 'time_of_registration'] = time_of_registration
        vessel_planning.loc[vessel_planning_index, 'time_of_acceptance'] = time_of_registration
        vessel_planning.loc[vessel_planning_index, 'bound'] = direction
        vessel_planning.loc[vessel_planning_index, 'L'] = vessel.L
        vessel_planning.loc[vessel_planning_index, 'B'] = vessel.B
        vessel_planning.loc[vessel_planning_index, 'T'] = vessel.T
        _ = self.calculate_sailing_time_to_waiting_area(vessel, direction, pre_planning=pre_planning)
        if (not direction and self.has_lineup_area_A) or (direction and self.has_lineup_area_B):
            self.calculate_sailing_time_to_lineup_area(vessel, direction, pre_planning=pre_planning)
        _ = self.calculate_sailing_time_to_approach(vessel, direction, pre_planning=pre_planning)
        _ = self.calculate_sailing_time_to_lock_door(vessel, direction, pre_planning=pre_planning)


    def add_empty_lock_operation_to_planning(self, operation_index, direction, pre_planning=False):
        operation_planning = self.operation_planning
        if self.predictive:
            operation_planning = self.operation_pre_planning

        preceding_operations = operation_planning[operation_planning.index < operation_index]
        if not preceding_operations.empty:
            preceding_operation = operation_planning.loc[operation_index-1]
            first_empty_lock_operation_start = preceding_operation.time_potential_lock_door_closure_start
        else:
            first_empty_lock_operation_start = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
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
        remaining_route = nx.dijkstra_path(self.env.FG,node,vessel.route[-1])
        for origin in remaining_route:
            if origin == self.lock_complex.waiting_area_A.edge[0]:
                waiting_area_node = self.lock_complex.waiting_area_A.edge[1]
                break
            elif origin == self.lock_complex.waiting_area_B.edge[0]:
                waiting_area_node = self.lock_complex.waiting_area_B.edge[1]
                break
        route_to_waiting_area = nx.dijkstra_path(self.env.FG,vessel.origin,waiting_area_node)
        return route_to_waiting_area


    def calculate_sailing_time_to_waiting_area(self, vessel, direction, node=None , prognosis=False, pre_planning = False, overwrite=True):
        vessel_planning = self.vessel_planning
        if self.predictive:
            vessel_planning = self.vessel_pre_planning

        if not direction:
            waiting_area_approach = self.lock_complex.waiting_area_A
        else:
            waiting_area_approach = self.lock_complex.waiting_area_B

        # Calculate sailing time function
        vessel_traffic_service = self.env.vessel_traffic_service
        calculate_sailing_time = vessel_traffic_service.provide_sailing_time_distance_on_edge_to_distance_on_another_edge

        # Calculate sailing time to waiting area
        distance_to_waiting_area_from_last_node = waiting_area_approach.distance_from_node
        if node is None:
            node = vessel.origin
        route_to_waiting_area = self.determine_route_to_waiting_area_from_node(node=node,vessel=vessel)
        if not direction:
            sailing_to_waiting_area = calculate_sailing_time(vessel,
                                                             route=route_to_waiting_area,
                                                             distance_sailed_on_last_edge=distance_to_waiting_area_from_last_node)
            sailing_to_waiting_area_time = pd.Timedelta(seconds=sailing_to_waiting_area['Time'].sum())
        else:
            sailing_to_waiting_area = calculate_sailing_time(vessel,
                                                             route=route_to_waiting_area,
                                                             distance_sailed_on_last_edge=distance_to_waiting_area_from_last_node)
            sailing_to_waiting_area_time = pd.Timedelta(seconds=sailing_to_waiting_area['Time'].sum())

        if not prognosis and overwrite:
            current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            if pre_planning:
                current_time = vessel_planning.loc[vessel_planning_index, 'time_of_acceptance']
            vessel_planning.loc[vessel_planning_index, 'time_arrival_at_waiting_area'] = current_time + sailing_to_waiting_area_time

        sailing_distance = sailing_to_waiting_area['Distance'].sum()
        sailing_time = sailing_to_waiting_area['Time'].sum()
        average_sailing_speed = sailing_to_waiting_area['Speed']
        if sailing_time:
            average_sailing_speed = sailing_distance/sailing_to_waiting_area['Time'].sum()
        return sailing_to_waiting_area_time, sailing_distance, average_sailing_speed


    def calculate_sailing_time_to_lineup_area(self, vessel, direction, prognosis=False, pre_planning=False, overwrite=True):
        vessel_planning = self.vessel_planning
        if self.predictive:
            vessel_planning = self.vessel_pre_planning

        if not direction:
            lineup_area_approach = self.lock_complex.lineup_area_A
        else:
            lineup_area_approach = self.lock_complex.lineup_area_B

        # Calculate sailing time function
        vessel_traffic_service = self.env.vessel_traffic_service
        calculate_sailing_time = vessel_traffic_service.provide_sailing_time_distance_on_edge_to_distance_on_another_edge

        # Calculate sailing time to lineup area
        distance_to_lineup_area_from_last_node = lineup_area_approach.distance_from_start_edge
        route_to_lineup_area = nx.dijkstra_path(self.env.FG,vessel.origin,lineup_area_approach.end_node)
        sailing_to_lineup_area = calculate_sailing_time(vessel,
                                                        route=route_to_lineup_area,
                                                        distance_sailed_on_last_edge=distance_to_lineup_area_from_last_node)
        sailing_to_lineup_area_time = pd.Timedelta(seconds=sailing_to_lineup_area['Time'].sum())
        if not prognosis and overwrite:
            current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            if pre_planning:
                current_time = vessel_planning.loc[vessel_planning_index, 'time_of_acceptance']
            vessel_planning.loc[vessel_planning_index, 'time_arrival_at_lineup_area'] = current_time + sailing_to_lineup_area_time

        return sailing_to_lineup_area_time


    def calculate_sailing_time_to_approach(self, vessel, direction, start_node=None, operation_index=None,prognosis=False, pre_planning=False, overwrite=True):
        vessel_planning = self.vessel_planning
        if self.predictive:
            vessel_planning = self.vessel_pre_planning

        sailing_time_to_waiting_area = self.calculate_sailing_time_to_waiting_area(vessel, direction, node = start_node, pre_planning=pre_planning,overwrite=overwrite)[0]
        sailing_time_to_lock_door = self.calculate_sailing_time_to_lock_door(vessel, direction, start_node = start_node, pre_planning=pre_planning, overwrite=overwrite)
        sailing_distance_from_entry = self.sailing_distance_to_crossing_point
        sailing_speed_during_entry = self.vessel_sailing_in_speed(vessel, direction,from_crossing_point=True)
        sailing_time_entry = pd.Timedelta(seconds=sailing_distance_from_entry / sailing_speed_during_entry)
        sailing_time_to_start_approach = sailing_time_to_lock_door - sailing_time_entry - sailing_time_to_waiting_area

        if not prognosis and overwrite:
            current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            if pre_planning:
                current_time = vessel_planning.loc[vessel_planning_index, 'time_of_acceptance']

            vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start'] = current_time + sailing_time_to_start_approach
            if operation_index is not None:
                passing_start_time = self.calculate_vessel_passing_start_time(vessel, operation_index, direction, prognosis=prognosis,pre_planning=pre_planning,overwrite=overwrite)
                vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start'] = passing_start_time

        return sailing_time_to_start_approach


    def calculate_sailing_time_to_lock_door(self, vessel, direction, start_node=None,prognosis=False, pre_planning=False, overwrite=True):
        vessel_planning = self.vessel_planning
        if self.predictive:
            vessel_planning = self.vessel_pre_planning

        if not direction:
            lock_end_node = self.lock_complex.end_node
        else:
            lock_end_node = self.lock_complex.start_node

        # Calculate sailing time function
        vessel_traffic_service = self.env.vessel_traffic_service
        calculate_sailing_time = vessel_traffic_service.provide_sailing_time_distance_on_edge_to_distance_on_another_edge

        # Calculate sailing time to lock chamber
        if start_node is None:
            start_node = vessel.origin
        route_to_lock_chamber = nx.dijkstra_path(self.env.FG,start_node,lock_end_node)
        sailing_to_lock_chamber = calculate_sailing_time(vessel, route=route_to_lock_chamber)
        sailing_to_lock_chamber_distance = sailing_to_lock_chamber['Distance'].sum()
        sailing_to_lock_chamber_time = sailing_to_lock_chamber['Time'].sum()
        if not direction:
            distance_to_lock = self.distance_from_start_node_to_lock_doors_A
        else:
            distance_to_lock = self.distance_from_end_node_to_lock_doors_B
        sailing_to_lock_chamber_distance += distance_to_lock
        sailing_to_lock_chamber_time += distance_to_lock / self.vessel_sailing_in_speed(vessel, direction)
        sailing_to_lock_chamber_time = pd.Timedelta(seconds=sailing_to_lock_chamber_time)
        if not prognosis and overwrite:
            current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            if pre_planning:
                current_time = vessel_planning.loc[vessel_planning_index, 'time_of_acceptance']
            vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start'] = current_time + sailing_to_lock_chamber_time
        return sailing_to_lock_chamber_time


    def calculate_sailing_time_in_lock(self, vessel, operation_index, direction, prognosis=False, pre_planning=False):
        vessels = self.subtract_vessels_from_lock_operation(operation_index, pre_planning=pre_planning)
        if not prognosis:
            vessel_index = vessels.index(vessel)
            sailing_distance_from_lock_doors = (self.lock_length - np.sum([vessel.L for vessel in vessels[:vessel_index]])) - 0.5 * vessel.L
        else:
            sailing_distance_from_lock_doors = (self.lock_length - np.sum([vessel.L for vessel in vessels]) - 0.5 * vessel.L)
        sailing_speed_into_lock = self.vessel_sailing_speed_in_lock(vessel, sailing_distance_from_lock_doors)
        sailing_time_into_lock = pd.Timedelta(seconds=sailing_distance_from_lock_doors / sailing_speed_into_lock)
        return sailing_time_into_lock


    def calculate_sailing_in_time_delay(self, vessel, operation_index, direction, minimum_difference_with_previous_vessel=False, prognosis=False, pre_planning=False, overwrite=True):
        vessels = self.subtract_vessels_from_lock_operation(operation_index, pre_planning=pre_planning)
        vessel_planning = self.vessel_planning
        if self.predictive:
            vessel_planning = self.vessel_pre_planning

        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index, pre_planning=pre_planning)
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        sailing_time_to_lock = self.calculate_sailing_time_to_lock_door(vessel, direction, prognosis=prognosis,pre_planning=pre_planning, overwrite=overwrite)
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        vessel_entry_start_timestamp = np.max([current_time + sailing_time_to_lock,vessel_planning.loc[vessel_planning_index,'time_lock_entry_start']])

        if not prognosis:
            vessel_index = vessels.index(vessel)
        else:
            vessel_index = -1

        previous_vessel = None
        if not prognosis and vessel != first_vessel:
            previous_vessel = vessels[vessel_index - 1]
        elif prognosis and len(vessels):
            previous_vessel = vessels[-1]

        sailing_in_time_delay = pd.Timedelta(seconds=0)
        if previous_vessel is not None:
            previous_vessel_planning_index = vessel_planning[vessel_planning.id == previous_vessel.id].iloc[-1].name
            previous_vessel_entry_start_timestamp = vessel_planning.loc[previous_vessel_planning_index,'time_lock_entry_start']
            if minimum_difference_with_previous_vessel:
                vessel_entry_start_timestamp = previous_vessel_entry_start_timestamp
            difference_entry_start_timestamp = vessel_entry_start_timestamp - previous_vessel_entry_start_timestamp
            if difference_entry_start_timestamp < pd.Timedelta(seconds=self.sailing_in_time_gap_through_doors):
                sailing_in_time_delay = pd.Timedelta(seconds=self.sailing_in_time_gap_through_doors)-difference_entry_start_timestamp
            previous_vessel_laying_still_time = vessel_planning.loc[previous_vessel_planning_index,'time_lock_entry_stop']
            difference_berthing_time_previous_vessel_and_vessel_sailing_in_time = (vessel_entry_start_timestamp - previous_vessel_laying_still_time)
            if self.sailing_in_time_gap_after_berthing_previous_vessel is not None and difference_berthing_time_previous_vessel_and_vessel_sailing_in_time < pd.Timedelta(seconds=self.sailing_in_time_gap_after_berthing_previous_vessel):
                sailing_in_time_delay = np.max([(previous_vessel_laying_still_time+pd.Timedelta(seconds=self.sailing_in_time_gap_after_berthing_previous_vessel))-vessel_entry_start_timestamp,sailing_in_time_delay])
        return sailing_in_time_delay


    def calculate_vessel_entry_start_time(self, vessel, direction):
        sailing_distance_from_entry = self.sailing_distance_to_crossing_point
        sailing_speed_during_entry = self.vessel_sailing_in_speed(vessel, direction,from_crossing_point=True)
        sailing_time_entry = pd.Timedelta(seconds=sailing_distance_from_entry / sailing_speed_during_entry)
        return sailing_time_entry


    def calculate_vessel_passing_start_time(self, vessel, operation_index, direction, prognosis=False, pre_planning=False, overwrite=True):
        vessel_planning = self.vessel_planning
        if self.predictive:
            vessel_planning = self.vessel_pre_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        if pre_planning:
            current_time = vessel_planning.loc[vessel_planning_index,'time_of_acceptance']
        sailing_time_to_lock = self.calculate_sailing_time_to_lock_door(vessel, direction, prognosis=prognosis,pre_planning=pre_planning,overwrite=overwrite)
        sailing_time_entry = self.calculate_vessel_entry_start_time(vessel, direction)
        sailing_in_delay = self.calculate_sailing_in_time_delay(vessel, operation_index, direction, prognosis=prognosis, pre_planning=pre_planning,overwrite=overwrite)
        vessel_passing_start_timestamp = current_time + (sailing_time_to_lock - sailing_time_entry)
        vessel_passing_start_timestamp += sailing_in_delay
        return vessel_passing_start_timestamp


    def calculate_lock_operation_start_time(self, vessel, operation_index, direction, prognosis=False,pre_planning=False, overwrite=True):
        operation_planning = self.operation_planning
        if pre_planning:
            operation_planning = self.operation_pre_planning

        previous_operations = operation_planning[operation_planning.index < operation_index]
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index)
        lock_operation_start_time = self.calculate_vessel_passing_start_time(first_vessel, operation_index, direction, prognosis,pre_planning=pre_planning,overwrite=overwrite)
        if not previous_operations.empty:
            previous_operation = previous_operations.iloc[-1]
            previous_lock_operation_stop_time = previous_operation.time_operation_stop
            if lock_operation_start_time < previous_lock_operation_stop_time:
                lock_operation_start_time = previous_lock_operation_stop_time
        return lock_operation_start_time


    def calculate_lock_door_opening_time(self, vessel, operation_index, direction, prognosis=False, pre_planning=False, overwrite=True):
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index)
        lock_entry_start_time = self.calculate_vessel_entry_start_time(first_vessel, direction)
        lock_entry_start_time -= self.minimum_advance_to_open_doors(vessel, direction)
        return lock_entry_start_time


    def calculate_lock_entry_start_time(self, vessel, operation_index, direction, prognosis=False, pre_planning=False, overwrite=True):
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index, pre_planning=pre_planning)
        lock_entry_start_time = self.calculate_vessel_entry_start_time(first_vessel, direction)
        return lock_entry_start_time


    def calculate_vessel_entry_stop_time(self, vessel, operation_index, direction, prognosis=False, pre_planning=False, overwrite=True):
        vessel_entry_start_time = self.calculate_vessel_entry_start_time(vessel, direction)
        sailing_time_in_lock = self.calculate_sailing_time_in_lock(vessel, operation_index, direction, prognosis, pre_planning=pre_planning)
        vessel_entry_stop_time = vessel_entry_start_time + sailing_time_in_lock
        return vessel_entry_stop_time


    def calculate_lock_entry_stop_time(self, vessel, operation_index, direction, prognosis=False, pre_planning=False, overwrite=True):
        last_vessel = self.determine_last_vessel_of_lock_operation(vessel, operation_index, prognosis, pre_planning=pre_planning)
        lock_entry_stop_time = self.calculate_vessel_entry_stop_time(last_vessel, operation_index, direction, prognosis, pre_planning=pre_planning,overwrite=overwrite)
        return lock_entry_stop_time


    def calculate_lock_operation_times(self, operation_index, last_entering_time, start_time, vessel = None, direction=None, same_direction = False):
        operation_planning = self.operation_planning
        vessel_planning = self.vessel_planning
        if self.predictive:
            operation_planning = self.operation_pre_planning
            vessel_planning = self.vessel_pre_planning

        x_location_lock = 0.
        if vessel is not None:
            vessels_in_lock = operation_planning.loc[operation_index].vessels
            if vessels_in_lock == []:
                vessels_in_lock = [vessel]
            x_location_lock = np.sum([v.L for v in vessels_in_lock[:-1]])+0.5*vessel.L

        time_door_closing_start = start_time
        if self.close_doors_before_vessel_is_laying_still and vessel is not None:
            time_door_closing_start = last_entering_time + self.minimum_delay_to_close_doors(vessel, direction, after_lock_entry=True, x_location_lock=x_location_lock)
        time_door_closing_stop = time_door_closing_start + pd.Timedelta(seconds=self.lock_complex.doors_closing_time)
        time_levelling_start = time_door_closing_stop
        if self.close_doors_before_vessel_is_laying_still and vessel is not None:
            vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
            if not isinstance(vessel_planning.loc[vessel_planning_index,'time_lock_entry_stop'],float):
                time_levelling_start = np.max([vessel_planning.loc[vessel_planning_index,'time_lock_entry_stop'],time_levelling_start])
            else:
                time_levelling_start = time_levelling_start
        time_levelling_stop,_,_ = self.lock_complex.determine_levelling_time(t_start=time_levelling_start, direction=direction, prediction=True, same_direction = same_direction)
        time_levelling_stop = time_levelling_start + pd.Timedelta(seconds=time_levelling_stop)
        time_door_opening_start = time_levelling_stop
        time_door_opening_stop = time_levelling_stop + pd.Timedelta(seconds=self.lock_complex.doors_opening_time)
        return time_door_closing_start, time_door_closing_stop, time_levelling_start, time_levelling_stop, time_door_opening_start, time_door_opening_stop


    def calculate_vessel_departure_start_time(self, vessel, operation_index, direction, prognosis=False, pre_planning=False):
        vessels = self.subtract_vessels_from_lock_operation(operation_index, pre_planning=pre_planning)
        if not prognosis:
            vessel_index = vessels.index(vessel)
            number_of_previous_vessels = vessel_index
        else:
            vessel_index = -1
            number_of_previous_vessels = len(vessels)

        delay = pd.Timedelta(seconds=0)
        if number_of_previous_vessels:
            previous_vessel = vessels[vessel_index-1]
            vessel_sailing_out_time = self.calculate_vessel_sailing_time_out_of_lock(vessel, operation_index, direction,prognosis=prognosis,pre_planning=pre_planning)
            previous_vessel_sailing_out_time = self.calculate_vessel_sailing_time_out_of_lock(previous_vessel, operation_index, direction, prognosis=prognosis,pre_planning=pre_planning)
            sailing_out_time_gap_through_doors = (vessel_sailing_out_time - previous_vessel_sailing_out_time)
            if sailing_out_time_gap_through_doors < pd.Timedelta(seconds=self.sailing_out_time_gap_through_doors):
                delay += pd.Timedelta(seconds=self.sailing_out_time_gap_through_doors)-sailing_out_time_gap_through_doors

            if self.sailing_out_time_gap_after_berthing_previous_vessel is not None and delay < pd.Timedelta(seconds=self.sailing_out_time_gap_after_berthing_previous_vessel)*number_of_previous_vessels:
                delay = pd.Timedelta(seconds=self.sailing_out_time_gap_after_berthing_previous_vessel)*number_of_previous_vessels

        delay += pd.Timedelta(seconds=self.start_sailing_out_time_after_doors_have_been_opened)
        return delay


    def calculate_lock_departure_start_time(self, vessel, operation_index, direction, prognosis=False, pre_planning=False):
        first_vessel = self.determine_first_vessel_of_lock_operation(vessel, operation_index, pre_planning=pre_planning)
        time_departure_start = self.calculate_vessel_departure_start_time(first_vessel, operation_index, direction, prognosis, pre_planning=pre_planning)
        return time_departure_start


    def calculate_vessel_sailing_time_out_of_lock(self, vessel, operation_index, direction, prognosis=False, pre_planning=False):
        vessels = self.subtract_vessels_from_lock_operation(operation_index,pre_planning=pre_planning)
        # Time to sail out
        if not prognosis:
            vessel_index = vessels.index(vessel)
            distance_to_lock = np.sum([vessel.L for vessel in vessels[:vessel_index]]) + 0.5 * vessel.L
        else:
            distance_to_lock = np.sum([vessel.L for vessel in vessels]) + 0.5 * vessel.L
        vessel_speed = self.vessel_sailing_speed_out_lock(vessel, distance_to_lock)
        sailing_out_time = pd.Timedelta(seconds=distance_to_lock / vessel_speed)
        return sailing_out_time


    def calculate_vessel_departure_stop_time(self, vessel, operation_index, direction, prognosis=False, pre_planning=False):
        time_departure_start = self.calculate_vessel_departure_start_time(vessel, operation_index, direction, prognosis, pre_planning=pre_planning)
        sailing_out_time = self.calculate_vessel_sailing_time_out_of_lock(vessel, operation_index, direction, prognosis, pre_planning=pre_planning)
        time_departure_stop = time_departure_start + sailing_out_time
        return time_departure_stop


    def calculate_lock_departure_stop_time(self, vessel, operation_index, direction, prognosis=False, pre_planning=False):
        last_vessel = self.determine_last_vessel_of_lock_operation(vessel, operation_index, prognosis, pre_planning=pre_planning)
        time_departure_stop = self.calculate_vessel_departure_stop_time(last_vessel, operation_index, direction, prognosis, pre_planning=pre_planning)
        return time_departure_stop


    def calculate_vessel_passing_stop_time(self, vessel, operation_index, direction, prognosis=False, pre_planning=False):
        time_departure_stop = self.calculate_vessel_departure_stop_time(vessel, operation_index, direction, prognosis, pre_planning=pre_planning)
        vessel_speed = self.vessel_sailing_out_speed(vessel, direction, until_crossing_point=True)
        time_departure_stop += pd.Timedelta(seconds = self.sailing_distance_to_crossing_point/vessel_speed)
        return time_departure_stop


    def calculate_lock_operation_stop_time(self, vessel, operation_index, direction, prognosis=False, pre_planning=False):
        last_vessel = self.determine_last_vessel_of_lock_operation(vessel, operation_index, prognosis, pre_planning=pre_planning)
        time_operation_stop = self.calculate_vessel_passing_stop_time(last_vessel, operation_index, direction, prognosis, pre_planning=pre_planning)
        return time_operation_stop


    def minimum_delay_to_close_doors(self, vessel, direction, after_lock_entry=False, x_location_lock=0):
        minimum_delay_to_close_doors = pd.Timedelta(seconds=self.sailing_time_before_closing_lock_doors)
        # if not after_lock_entry:
        #     minimum_delay_to_close_doors += pd.Timedelta(seconds=0.5*vessel.L/self.vessel_sailing_out_speed(vessel,direction))
        # else:
        #     minimum_delay_to_close_doors += pd.Timedelta(seconds=0.5*vessel.L/self.vessel_sailing_speed_in_lock(vessel, x_location_lock = x_location_lock, distance = 0.5*vessel.L))
        # TODO: take into account the vessels' bows and sterns to determine the time before and after which the door should be respectively opened and closed
        return minimum_delay_to_close_doors


    def minimum_advance_to_open_doors(self, vessel, direction):
        minimum_advance_to_open_doors = pd.Timedelta(seconds=self.sailing_time_before_opening_lock_doors)
        #minimum_advance_to_open_doors += pd.Timedelta(seconds=vessel.L/self.vessel_sailing_in_speed(vessel,direction))
        # TODO: take into account the vessels' bows and sterns to determine the time before and after which the door should be respectively opened and closed
        return minimum_advance_to_open_doors


    def calculate_lock_door_closing_time(self, vessel, operation_index, direction, prognosis=False, pre_planning=False):
        lock_doors_closing_time = self.calculate_lock_departure_stop_time(vessel, operation_index, direction,prognosis, pre_planning=pre_planning)
        lock_doors_closing_time += self.minimum_delay_to_close_doors(vessel, direction)
        return lock_doors_closing_time


    def determine_first_vessel_of_lock_operation(self, vessel, operation_index, pre_planning=False):
        vessels = self.subtract_vessels_from_lock_operation(operation_index,pre_planning=pre_planning)
        first_vessel = vessel
        if len(vessels):
            first_vessel = vessels[0]
        return first_vessel


    def determine_last_vessel_of_lock_operation(self, vessel, operation_index, prognosis=False, pre_planning=False):
        vessels = self.subtract_vessels_from_lock_operation(operation_index,pre_planning=pre_planning)
        last_vessel = vessel
        if not prognosis:
            last_vessel = vessels[-1]
        return last_vessel


    def calculate_delay_to_open_doors(self, vessel):
        vessel_planning = self.vessel_planning
        if self.predictive:
            vessel_planning = self.vessel_pre_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        arrival_time_at_lock = vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start']
        time_index = np.absolute(hydrodynamic_times - np.datetime64(arrival_time_at_lock) - np.timedelta64(int(self.doors_opening_time),'s')).argmin()
        expected_levelling_time,_,_ = self.determine_levelling_time(t_start=hydrodynamic_times[time_index - 1], same_direction=True, prediction=True)
        delay = self.sailing_time_before_opening_lock_doors + expected_levelling_time + self.doors_opening_time
        return delay


    def determine_if_door_can_be_closed(self, vessel, direction, operation_index, between_arrivals=False):
        doors_can_be_closed = False
        if not self.closing_doors_in_between_operations:
            return doors_can_be_closed
        doors_can_be_closed = True

        operation_planning = self.operation_planning
        vessel_planning = self.vessel_planning
        if self.predictive:
            operation_planning = self.operation_pre_planning
            vessel_planning = self.vessel_pre_planning

        if not between_arrivals:
            last_time_doors_closed = operation_planning.loc[operation_index,'time_potential_lock_door_closure_start']+pd.Timedelta(seconds=self.doors_closing_time)
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

        if doors_required_to_be_open-door_opening_time < last_time_doors_closed:
            doors_can_be_closed = False
        return doors_can_be_closed


    def determine_if_door_is_closed(self, vessel, operation_index, direction, first_in_lock=False):
        operation_planning = self.operation_planning
        vessel_planning = self.vessel_planning
        if self.predictive:
            operation_planning = self.operation_pre_planning
            vessel_planning = self.vessel_pre_planning
        vessels = operation_planning.loc[operation_index, 'vessels']
        vessel_index = vessels.index(vessel)

        if not first_in_lock and vessel_index:
            previous_vessel_planning_index = vessel_planning[vessel_planning.id == operation_planning.loc[operation_index, 'vessels'][vessel_index-1].id].iloc[-1].name
            last_time_doors_closed = vessel_planning.loc[previous_vessel_planning_index,'time_potential_lock_door_closure_start'] + pd.Timedelta(seconds=self.doors_closing_time)
        elif operation_index == 0:
            last_time_doors_closed = self.env.simulation_start
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
        if doors_required_to_be_open - operation_time > last_time_doors_closed:
            doors_is_closed = True

        return doors_is_closed, doors_required_to_be_open, operation_time


    def determine_time_to_open_door(self, operation_index, direction, last_time_doors_closed, doors_required_to_be_open, same_direction):
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
        index_start_node = 0  # list(self.multidigraph.nodes).index(self.start_node)
        index_end_node = 1  # list(self.multidigraph.nodes).index(self.end_node)
        wlev_A = np.nan
        wlev_B = np.nan
        if self.env.vessel_traffic_service.hydrodynamic_information_path:
            time_index_start = np.absolute(hydrodynamic_times - np.datetime64(levelling_start)).argmin()
            time_index_stop = np.absolute(hydrodynamic_times - np.datetime64(levelling_stop)).argmin()
            if not direction:
                wlev_A = hydrodynamic_data["Water level"][index_start_node][time_index_start]
                wlev_B = hydrodynamic_data["Water level"][index_end_node][time_index_stop]
                if same_direction:
                    wlev_A = self.water_level[time_index_start]
                    wlev_B = hydrodynamic_data["Water level"][index_start_node][time_index_stop]
            elif direction:
                wlev_B = self.water_level[time_index_start]
                wlev_A = hydrodynamic_data["Water level"][index_start_node][time_index_stop]
                if same_direction:
                    wlev_A = self.water_level[time_index_start]
                    wlev_B = hydrodynamic_data["Water level"][index_end_node][time_index_stop]
        return wlev_A, wlev_B


    def subtract_vessels_from_lock_operation(self, operation_index, pre_planning=False):
        vessels = []
        operation_planning = self.operation_planning
        if self.predictive:
            operation_planning = self.operation_pre_planning

        selected_operation = operation_planning[operation_planning.index == operation_index]
        if not selected_operation.empty:
            vessels = operation_planning.loc[operation_index, 'vessels'].copy()
        return vessels


    def update_operation_planning(self,vessel,direction,operation_index,add_operation,pre_planning=False):
        operation_planning = self.lock_complex.operation_planning
        vessel_planning = self.lock_complex.vessel_planning
        if self.predictive:
            operation_planning = self.lock_complex.operation_pre_planning
            vessel_planning = self.lock_complex.vessel_pre_planning

        if operation_planning.empty or add_operation:
            self.add_vessel_to_new_lock_operation(vessel, operation_index, direction, pre_planning=pre_planning)
        else:
            yield from self.add_vessel_to_planned_lock_operation(vessel, operation_index, direction, pre_planning=pre_planning)

        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        operation_index = vessel_planning.loc[vessel_planning_index,'operation_index']
        operation_planning.loc[operation_index, 'maximum_individual_delay'] = np.max(vessel_planning[vessel_planning.operation_index == operation_index].delay)
        operation_planning.loc[operation_index, 'total_delay'] = np.sum(vessel_planning[vessel_planning.operation_index == operation_index].delay)


    def add_vessel_to_new_lock_operation(self, vessel, operation_index, direction, add_vessel_to_planning=True,pre_planning=False):
        vessel_planning = self.lock_complex.vessel_planning
        operation_planning = self.lock_complex.operation_planning
        if self.predictive:
            vessel_planning = self.lock_complex.vessel_pre_planning
            operation_planning = self.lock_complex.operation_pre_planning

        previous_planned_operations = operation_planning[operation_planning.index <= operation_index]
        if not previous_planned_operations.empty:
            previous_planned_operation = previous_planned_operations.iloc[-1]
            if previous_planned_operation.bound == direction:
                self.add_empty_lock_operation_to_planning(operation_index, 1 - direction, pre_planning=pre_planning)
                operation_index += 1

        previous_planned_operations = operation_planning[operation_planning.index <= operation_index]
        later_planned_operations = operation_planning[operation_planning.index >= operation_index]
        if not add_vessel_to_planning:
            operation_planning = operation_planning.copy()

        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        earliest_possible_time_lock_entry_start = vessel_planning.loc[vessel_planning_index,'time_lock_entry_start']
        time_lock_operation_start = self.calculate_lock_operation_start_time(vessel, operation_index, direction,prognosis=True, pre_planning=pre_planning)
        operational_hours = self.operational_hours
        within_operation_hours = operational_hours[(time_lock_operation_start >= operational_hours.start_time) & (time_lock_operation_start <= operational_hours.stop_time)]
        if within_operation_hours.empty:
            first_available_hour = operational_hours[operational_hours.start_time >= time_lock_operation_start].iloc[0]
            time_lock_operation_start = first_available_hour.start_time

        time_lock_entry_start = self.calculate_lock_entry_start_time(vessel, operation_index, direction, prognosis=True,pre_planning=pre_planning) + time_lock_operation_start
        vessels = [vessel]
        operation_planning.loc[operation_index, 'bound'] = direction
        operation_planning.loc[operation_index, 'vessels'] = []
        operation_planning.loc[operation_index, 'capacity_L'] = self.lock_complex.lock_length - vessel.L
        operation_planning.loc[operation_index, 'capacity_B'] = self.lock_complex.lock_width - vessel.B

        x_location_lock = operation_planning.loc[operation_index, 'capacity_L'] + 0.5*vessel.L
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

        time_lock_entry_stop = self.calculate_lock_entry_stop_time(vessel, operation_index, direction, prognosis=True,pre_planning=pre_planning) + time_lock_operation_start
        time_lock_door_opening_stop = self.calculate_lock_door_opening_time(vessel, operation_index, direction,prognosis=True,pre_planning=pre_planning) + time_lock_operation_start
        vessel_entry_delay = time_lock_entry_start - earliest_possible_time_lock_entry_start
        if self.close_doors_before_vessel_is_laying_still:
            vessel_planning.loc[vessel_planning_index, 'time_potential_lock_door_closure_start'] = time_lock_entry_start + self.minimum_delay_to_close_doors(vessel, direction, after_lock_entry = True, x_location_lock = x_location_lock)
        else:
            vessel_planning.loc[vessel_planning_index, 'time_potential_lock_door_closure_start'] = time_lock_entry_stop
        vessel_planning.loc[vessel_planning_index, 'operation_index'] = operation_index
        vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start'] = time_lock_operation_start
        vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start'] = time_lock_entry_start
        vessel_planning.loc[vessel_planning_index, 'time_lock_entry_stop'] = time_lock_entry_stop


        operation_planning.loc[operation_index, 'time_operation_start'] = time_lock_operation_start
        operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop'] = time_lock_door_opening_stop
        operation_planning.loc[operation_index, 'time_entry_start'] = time_lock_entry_start
        operation_planning.loc[operation_index, 'time_entry_stop'] = time_lock_entry_stop

        time_door_closing_start, time_door_closing_stop, time_levelling_start, time_levelling_stop, time_door_opening_start, time_door_opening_stop = self.calculate_lock_operation_times(operation_index=operation_index,
                                                                                                                                                                                          last_entering_time=time_lock_entry_start,
                                                                                                                                                                                          start_time=time_lock_entry_stop,
                                                                                                                                                                                          vessel=vessel,
                                                                                                                                                                                          direction=direction)
        wlev_A, wlev_B = self.determine_water_levels_before_and_after_levelling(time_levelling_start,time_levelling_stop, direction)
        time_lock_departure_start = self.calculate_lock_departure_start_time(vessel, operation_index, direction, prognosis=True, pre_planning=pre_planning) + time_door_opening_stop
        time_lock_departure_stop = self.calculate_lock_departure_stop_time(vessel, operation_index, direction, prognosis=True, pre_planning=pre_planning) + time_door_opening_stop
        time_lock_operation_stop = self.calculate_lock_operation_stop_time(vessel, operation_index, direction, prognosis=True, pre_planning=pre_planning) + time_door_opening_stop
        time_lock_door_closing_start = self.calculate_lock_door_closing_time(vessel, operation_index, direction, prognosis=True, pre_planning=pre_planning) + time_door_opening_stop
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
        if len(vessels) < self.min_vessels_in_operation:
            operation_planning.loc[operation_index, 'status'] = 'waiting for vessel'
        else:
            operation_planning.loc[operation_index, 'status'] = 'ready'

        vessel_planning.loc[vessel_planning_index, 'time_lock_departure_start'] = time_lock_departure_start
        vessel_planning.loc[vessel_planning_index, 'time_lock_departure_stop'] = time_lock_departure_stop
        vessel_planning.loc[vessel_planning_index, 'time_lock_passing_stop'] = time_lock_operation_stop
        if vessel_entry_delay > pd.Timedelta(seconds=0):
            vessel_planning.loc[vessel_planning_index,'delay'] += vessel_entry_delay

        if not later_planned_operations.empty and not self.closing_doors_in_between_operations:
            self.add_empty_lock_operation_to_planning(operation_index, 1-direction, pre_planning=pre_planning)


    def add_vessel_to_planned_lock_operation(self, vessel, operation_index, direction, prognosis=True, vessel_planning=None, operation_planning=None, pre_planning=False):
        if operation_planning is None and vessel_planning is None:
            prognosis = False

        if operation_planning is None:
            operation_planning = self.operation_planning
            if self.predictive:
                operation_planning = self.operation_pre_planning

        if vessel_planning is None:
            vessel_planning = self.vessel_planning
            if self.predictive:
                vessel_planning = self.vessel_pre_planning

        # Determine capacity parameters
        vessels_in_operation = operation_planning.loc[operation_index, 'vessels']
        sailing_in_gap = pd.Timedelta(seconds=0)
        if vessel not in operation_planning.loc[operation_index, 'vessels']:
            operation_planning.loc[operation_index, 'vessels'].append(vessel)
            vessels_in_operation = operation_planning.loc[operation_index, 'vessels']
            self.calculate_sailing_time_to_approach(vessel, direction, operation_index=operation_index,pre_planning=pre_planning)
            if self.min_vessels_in_operation and len(vessels_in_operation) == self.min_vessels_in_operation:
                if not pre_planning and not prognosis:
                    Operation = namedtuple('Operation', 'operation_index')
                    operation = Operation(operation_index)
                    yield self.wait_for_other_vessel_to_arrive.put(operation)
                    yield self.env.timeout(0.) #required to update the vessel_planning
                    sailing_in_gap = self.calculate_sailing_in_time_delay(vessel, operation_index, direction, prognosis=False,pre_planning=self.predictive, overwrite=False)

        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name
        time_arrival_time_at_lock_entry = vessel_planning.loc[vessel_planning_index,'time_lock_passing_start'] + sailing_in_gap
        if len(vessels_in_operation) >= self.min_vessels_in_operation:
            operation_planning.loc[operation_index, 'status'] = 'ready'

        operation_planning.loc[operation_index, 'capacity_L'] -= vessel.L
        operation_planning.loc[operation_index, 'capacity_B'] -= vessel.B

        # Determine new entry times
        operational_hours = self.operational_hours
        other_vessels_in_operation = operation_planning.loc[operation_index, 'vessels'][:-1]
        time_lock_operation_start = operation_planning.loc[operation_index, 'time_operation_start']
        vessel_entry_delay = pd.Timedelta(seconds=0)
        potential_lock_door_opening_stop = operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop']

        within_operation_hours = operational_hours[(time_arrival_time_at_lock_entry + vessel_entry_delay >= operational_hours.start_time) & (time_arrival_time_at_lock_entry + vessel_entry_delay <= operational_hours.stop_time)]
        if within_operation_hours.empty:
            first_available_hour = operational_hours[operational_hours.start_time >= (time_arrival_time_at_lock_entry + vessel_entry_delay)].iloc[0]
            vessel_entry_delay += first_available_hour.start_time - (time_arrival_time_at_lock_entry + vessel_entry_delay)
            if time_lock_operation_start < first_available_hour.start_time:
                time_lock_operation_start = first_available_hour.start_time

        time_first_vessel_required_to_be_at_lock_approach = (time_arrival_time_at_lock_entry + vessel_entry_delay)
        if time_first_vessel_required_to_be_at_lock_approach > operation_planning.loc[operation_index, 'time_operation_start'] and not len(other_vessels_in_operation):
            time_lock_operation_start = time_first_vessel_required_to_be_at_lock_approach
        elif time_first_vessel_required_to_be_at_lock_approach < operation_planning.loc[operation_index, 'time_operation_start']:
            vessel_entry_delay += operation_planning.loc[operation_index, 'time_operation_start']-time_first_vessel_required_to_be_at_lock_approach

        if vessel_entry_delay > pd.Timedelta(seconds=0):
            time_arrival_time_at_lock_entry += vessel_entry_delay

        # Update plannings start
        time_vessel_entry_start = self.calculate_vessel_entry_start_time(vessel,direction) + time_arrival_time_at_lock_entry
        time_lock_entry_stop = self.calculate_lock_entry_stop_time(vessel, operation_index, direction,pre_planning=pre_planning) + time_arrival_time_at_lock_entry
        vessel_planning.loc[vessel_planning_index, 'operation_index'] = operation_index
        vessel_planning.loc[vessel_planning_index, 'time_lock_passing_start'] = time_arrival_time_at_lock_entry
        vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start'] = time_vessel_entry_start
        vessel_planning.loc[vessel_planning_index, 'time_lock_entry_stop'] = time_lock_entry_stop
        operation_start_delay = time_lock_operation_start - operation_planning.loc[operation_index, 'time_operation_start']

        #If predictive, previous arrivals can be planned later
        if self.predictive and self.minimize_door_open_times and len(other_vessels_in_operation):
            latter_vessel = vessel
            for index,previous_vessel in enumerate(list(reversed(other_vessels_in_operation))):
                latter_vessel_planning_index = vessel_planning[vessel_planning.id == latter_vessel.id].iloc[-1].name
                previous_vessel_planning_index = vessel_planning[vessel_planning.id == previous_vessel.id].iloc[-1].name
                sailing_in_gap = self.calculate_sailing_in_time_delay(latter_vessel, operation_index, direction, minimum_difference_with_previous_vessel=True, prognosis=False,pre_planning=self.predictive, overwrite=False)
                sailing_in_delay = (vessel_planning.loc[latter_vessel_planning_index, 'time_lock_entry_start'] - vessel_planning.loc[previous_vessel_planning_index, 'time_lock_entry_start']) - sailing_in_gap
                vessel_planning.loc[previous_vessel_planning_index, 'time_lock_passing_start'] += sailing_in_delay
                vessel_planning.loc[previous_vessel_planning_index, 'time_lock_entry_start'] += sailing_in_delay
                vessel_planning.loc[previous_vessel_planning_index, 'time_lock_entry_stop'] += sailing_in_delay
                vessel_planning.loc[previous_vessel_planning_index, 'time_potential_lock_door_opening_stop'] += sailing_in_delay
                vessel_planning.loc[previous_vessel_planning_index, 'time_potential_lock_door_closure_start'] += sailing_in_delay
                vessel_planning.loc[previous_vessel_planning_index, 'time_arrival_at_waiting_area'] += sailing_in_delay
                if index == len(other_vessels_in_operation)-1:
                    operation_planning.loc[operation_index, 'time_operation_start'] = vessel_planning.loc[previous_vessel_planning_index, 'time_lock_passing_start']
                    operation_planning.loc[operation_index, 'time_entry_start'] = vessel_planning.loc[previous_vessel_planning_index, 'time_lock_entry_start']
                    operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop'] = vessel_planning.loc[previous_vessel_planning_index, 'time_potential_lock_door_opening_stop']
                else:
                    latter_vessel = previous_vessel
                # TODO for further research: speed should be changed before approaching the waiting area to arrive just in time so that a vessel can continue sailing instead of waiting in the waiting area

        # Determine new lock door closing and opening and levelling times
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
        operation_planning.loc[operation_index, 'time_operation_start'] += operation_start_delay
        if vessel_entry_delay > pd.Timedelta(seconds=0):
            vessel_planning.loc[vessel_planning_index,'delay'] += vessel_entry_delay
        operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop'] += operation_start_delay

        # Update previous arriving vessels in same lockage
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
                    operation_start_delay = (vessel_planning.loc[other_vessel_planning_index, 'time_lock_entry_start'] - vessel_planning.loc[next_vessel_planning_index, 'time_lock_entry_start'])

        additional_sailing_out_delay = time_door_opening_stop - operation_planning.loc[operation_index, 'time_door_opening_stop']
        if additional_sailing_out_delay > pd.Timedelta(seconds=0):
            for other_vessel in other_vessels_in_operation:
                other_vessel_planning_index = vessel_planning[vessel_planning.id == other_vessel.id].iloc[-1].name
                vessel_planning.loc[other_vessel_planning_index, 'time_lock_departure_start'] += additional_sailing_out_delay
                vessel_planning.loc[other_vessel_planning_index, 'time_lock_departure_stop'] += additional_sailing_out_delay
                vessel_planning.loc[other_vessel_planning_index, 'time_lock_passing_stop'] += additional_sailing_out_delay
                vessel_planning.loc[other_vessel_planning_index, 'delay'] += additional_sailing_out_delay

        # Determine water levels
        wlev_A, wlev_B = self.determine_water_levels_before_and_after_levelling(time_levelling_start, time_levelling_stop, direction)

        if not len(other_vessels_in_operation):
            operation_planning.loc[operation_index, 'time_potential_lock_door_opening_stop'] = potential_lock_door_opening_stop
            operation_planning.loc[operation_index, 'time_operation_start'] = time_lock_operation_start
            operation_planning.loc[operation_index, 'time_entry_start'] = time_vessel_entry_start
        else:
            operation_planning.loc[operation_index, 'time_entry_start'] += operation_start_delay

        operation_planning.loc[operation_index, 'time_entry_stop'] = time_lock_entry_stop
        operation_planning.loc[operation_index, 'time_door_closing_start'] = time_door_closing_start
        operation_planning.loc[operation_index, 'time_door_closing_stop'] = time_door_closing_stop
        operation_planning.loc[operation_index, 'time_levelling_start'] = time_levelling_start
        operation_planning.loc[operation_index, 'time_levelling_stop'] = time_levelling_stop
        operation_planning.loc[operation_index, 'time_door_opening_start'] = time_door_opening_start
        operation_planning.loc[operation_index, 'time_door_opening_stop'] = time_door_opening_stop
        operation_planning.loc[operation_index, 'maximum_individual_delay'] = np.max(vessel_planning[vessel_planning.operation_index == operation_index].delay)
        operation_planning.loc[operation_index, 'total_delay'] = np.sum(vessel_planning[vessel_planning.operation_index == operation_index].delay)

        # Determine new departure and operation stop times
        time_lock_departure_start = self.calculate_lock_departure_start_time(vessel, operation_index,direction,pre_planning=pre_planning) + time_door_opening_stop
        time_vessel_departure_start = self.calculate_vessel_departure_start_time(vessel, operation_index, direction,pre_planning=pre_planning) + time_door_opening_stop
        time_lock_departure_stop = self.calculate_lock_departure_stop_time(vessel, operation_index,direction,pre_planning=pre_planning) + time_door_opening_stop
        time_vessel_departure_stop = self.calculate_vessel_departure_stop_time(vessel, operation_index,direction,pre_planning=pre_planning) + time_door_opening_stop
        time_lock_operation_stop = self.calculate_lock_operation_stop_time(vessel, operation_index,direction,pre_planning=pre_planning) + time_door_opening_stop
        time_vessel_passing_stop = self.calculate_vessel_passing_stop_time(vessel, operation_index,direction,pre_planning=pre_planning) + time_door_opening_stop
        time_lock_door_closing_start = self.calculate_lock_door_closing_time(vessel, operation_index, direction,pre_planning=pre_planning) + time_door_opening_stop

        # Update plannings end
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

        # Update previous lock operations:
        previous_planned_operations = operation_planning[operation_planning.index < operation_index]
        if not previous_planned_operations.empty:
            if previous_planned_operations.iloc[-1].time_potential_lock_door_closure_start < operation_planning.loc[operation_index,'time_potential_lock_door_opening_stop']:
                pass
                #TODO: move lockages ahead of earlier delayed ones, if they can start earlier than these lockages

        # Update next lock operations
        next_planned_operations = operation_planning[operation_planning.index > operation_index]
        for next_operation_index, next_operation_info in next_planned_operations.iterrows():
            sailing_in_delay = pd.Timedelta(seconds=0)
            if not len(next_operation_info) and time_lock_door_closing_start > next_operation_info.time_potential_lock_door_opening_stop:
                sailing_in_delay = time_lock_door_closing_start - next_operation_info.time_potential_lock_door_opening_stop
            elif len(next_operation_info) and time_lock_operation_stop > next_operation_info.time_operation_start:
                sailing_in_delay = time_lock_operation_stop - next_operation_info.time_operation_start
            new_operation_start = operation_planning.loc[next_operation_index, 'time_operation_start'] + sailing_in_delay
            within_operation_hours = operational_hours[(new_operation_start >= operational_hours.start_time)&(new_operation_start <= operational_hours.stop_time)]
            if within_operation_hours.empty:
                first_available_hour = operational_hours[operational_hours.start_time >= new_operation_start].iloc[0]
                sailing_in_delay += first_available_hour.start_time - new_operation_start

            if sailing_in_delay.total_seconds() > 0:
                operation_planning.loc[next_operation_index, 'time_potential_lock_door_opening_stop'] += sailing_in_delay
                operation_planning.loc[next_operation_index, 'time_operation_start'] += sailing_in_delay
                operation_planning.loc[next_operation_index, 'time_entry_start'] += sailing_in_delay
                operation_planning.loc[next_operation_index, 'time_entry_stop'] += sailing_in_delay
                next_vessels = next_operation_info.vessels
                last_vessel_entering_time = operation_planning.loc[next_operation_index, 'time_entry_start']
                next_vessel = None
                if len(next_vessels):
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
                            sailing_in_delay = pd.Timedelta(seconds=0)
                            if vessel_planning.loc[next_next_vessel_planning_index, 'time_lock_entry_start'] < vessel_planning.loc[next_vessel_planning_index, 'time_lock_entry_start']:
                                sailing_in_delay = (vessel_planning.loc[next_vessel_planning_index, 'time_lock_entry_start'] - vessel_planning.loc[next_next_vessel_planning_index, 'time_lock_entry_start'])
                                sailing_in_delay += self.calculate_sailing_in_time_delay(next_next_vessel, next_operation_index, direction, minimum_difference_with_previous_vessel=True, pre_planning=self.predictive, overwrite=False)



                time_doors_closing = operation_planning.loc[next_operation_index, 'time_entry_stop']
                time_door_closing_start, time_door_closing_stop, time_levelling_start, time_levelling_stop, time_door_opening_start, time_door_opening_stop = self.calculate_lock_operation_times(operation_index=next_operation_index,
                                                                                                                                                                                                  last_entering_time=last_vessel_entering_time,
                                                                                                                                                                                                  start_time=time_doors_closing,
                                                                                                                                                                                                  vessel=next_vessel,
                                                                                                                                                                                                  direction=direction)

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

                if len(next_vessels):
                    for vessel_index,next_vessel in enumerate(next_vessels):
                        next_vessel_planning_index = vessel_planning[vessel_planning.id == next_vessel.id].iloc[-1].name
                        vessel_planning.loc[next_vessel_planning_index, 'time_lock_departure_start'] += delay_after_levelling
                        vessel_planning.loc[next_vessel_planning_index, 'time_lock_departure_stop'] += delay_after_levelling
                        vessel_planning.loc[next_vessel_planning_index, 'time_lock_passing_stop'] += delay_after_levelling
                        vessel_planning.loc[next_vessel_planning_index, 'delay'] += delay_after_levelling
                time_lock_operation_stop = operation_planning.loc[next_operation_index, 'time_operation_stop']
                time_lock_door_closing_start = operation_planning.loc[next_operation_index, 'time_potential_lock_door_closure_start']

        return operation_planning


    def assign_vessel_to_lock_operation(self,vessel,direction,pre_planning=False):
        # Definitions
        operation_planning = self.lock_complex.operation_planning
        vessel_planning = self.lock_complex.vessel_planning
        if pre_planning:
            operation_planning = self.lock_complex.operation_pre_planning
            vessel_planning = self.lock_complex.vessel_pre_planning
        vessel_planning_index = vessel_planning[vessel_planning.id == vessel.id].iloc[-1].name

        # Parameters
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        if pre_planning:
            current_time = vessel_planning.loc[vessel_planning_index,'time_of_acceptance']

        time_lock_entry_start = vessel_planning.loc[vessel_planning_index,'time_lock_entry_start']
        time_lock_passing_start = vessel_planning.loc[vessel_planning_index, 'time_lock_entry_start']
        operational_hours = self.operational_hours
        within_operation_hours = operational_hours[(time_lock_passing_start >= operational_hours.start_time)&(time_lock_passing_start <= operational_hours.stop_time)]
        vessel_planning.loc[vessel_planning_index, 'delay'] = pd.Timedelta(seconds = 0)
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

        maximum_individual_delay = operation_planning.maximum_individual_delay + (time_lock_entry_start - operation_planning.time_entry_stop)
        maximum_additional_waiting_time = time_lock_entry_start - operation_planning.time_entry_stop

        # Filter operations to select available operation
        mask_bound = operation_planning.bound == direction # locks in the same direction mask_bound: operation_planning.bound == direction # locks in the same direction
        mask_status = operation_planning.status == 'waiting for vessel'
        mask_available = operation_planning.status != 'not available'
        mask_capacity_L = operation_planning.capacity_L >= vessel.L # vessel length fits in the lock
        mask_capacity_B = operation_planning.capacity_B >= vessel.B # vessel width fits in the lock
        mask_max_waiting_time = maximum_individual_delay < pd.Timedelta(seconds=self.lock_complex.clustering_time)
        mask_empty_lock = operation_planning.vessels.apply(len) == 0 # if lock is still empty (overrules max_waiting_time which becomes irrelevant)
        mask_max_vessels = mask_available
        if self.max_vessels_in_operation:
            mask_max_vessels = operation_planning.vessels.apply(len) < self.max_vessels_in_operation
        if not self.predictive:
            mask_future_operations = operation_planning.time_levelling_start >= current_time
        else:
            mask_future_operations = operation_planning.time_levelling_start >= self.env.simulation_start

        mask_empty_future_lockages = mask_empty_lock&mask_future_operations
        mask_max_waiting_time = mask_max_waiting_time&~mask_empty_lock
        mask_min_vessels = (mask_future_operations&mask_max_waiting_time)
        if self.min_vessels_in_operation > 1:
            mask_min_vessels = operation_planning.vessels.apply(len) < self.min_vessels_in_operation
        mask_future_operations = (mask_future_operations&mask_max_waiting_time)|mask_min_vessels

        available_operations = operation_planning[mask_available&mask_bound&mask_max_vessels&mask_capacity_L&mask_future_operations&(mask_min_vessels|mask_status|mask_empty_future_lockages|mask_max_waiting_time)].copy()
        # TODO: include mask_capacity_B for 2D implementation
        # TODO: create a selection method that can pick the lock operation based on minimizing expected delay or freshwater loss/saltwater intrusion

        add_operation = False
        if not available_operations.empty:
            operation_index = available_operations.iloc[0].name

        else:
            operation_index = len(operation_planning)
            add_operation = True

        return operation_index, add_operation, available_operations


    def convert_chamber(self, new_level, vessel=None, close_doors=True, delay=0, direction = None):
        """Function which converts the lock chamber and logs this event.

        Input:
            - environment: see init function
            - new_level: a string which represents the node and indicates the side at which the lock is currently levelled
            - number_of_vessels: the total number of vessels which are levelled simultaneously"""

        # Close the doors
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
        yield from self.level_lock(new_level, vessel=vessel, direction = direction)
        yield from self.open_door()


    def close_door(self, delay=0):
        # Close the doors
        start_delay = self.env.now
        while delay > 0:
            try:
                yield self.env.timeout(delay)
                delay = 0
            except simpy.Interrupt as e:
                delay -= (self.env.now - start_delay)

        hold_door_A = self.door_A.request()
        hold_levelling = self.levelling.request()
        hold_door_B = self.door_B.request()
        yield hold_door_A
        yield hold_levelling
        yield hold_door_B

        self.log_entry_v0("Lock doors closing start", self.env.now, self.output.copy(), self.node_open)
        remaining_doors_closing_time = self.doors_closing_time
        start_time_closing = self.env.now
        while remaining_doors_closing_time:
            try:
                yield self.env.timeout(remaining_doors_closing_time)
                remaining_doors_closing_time = 0
            except simpy.Interrupt as e:
                remaining_doors_closing_time -= (self.env.now - start_time_closing)

        if self.env.vessel_traffic_service.hydrodynamic_information_path:
            time_index = np.absolute(hydrodynamic_times - np.datetime64(datetime.datetime.fromtimestamp(self.env.now))).argmin()
            if self.node_open == self.start_node:
                self.water_level[time_index:] = hydrodynamic_data["Water level"][0][time_index].copy()
            else:
                self.water_level[time_index:] = hydrodynamic_data["Water level"][1][time_index].copy()

        self.log_entry_v0("Lock doors closing stop", self.env.now, self.output.copy(), self.node_open)
        if self.node_open == self.start_node:
            self.door_A_open = False
        else:
            self.door_B_open = False

        self.door_A.release(hold_door_A)
        self.levelling.release(hold_levelling)
        self.door_B.release(hold_door_B)


    def level_lock(self, new_level, vessel=None, direction=None, same_direction=False):
        # Water level will shift
        hold_door_A = self.door_A.request()
        hold_levelling = self.levelling.request()
        hold_door_B = self.door_B.request()
        yield hold_door_A
        yield hold_levelling
        yield hold_door_B

        levelling_time,_,_ = self.determine_levelling_time(t_start=self.env.now,direction=direction,same_direction=same_direction)
        if vessel is not None:
            vessel.log_entry_v0("Levelling start", vessel.env.now, vessel.output.copy(), vessel.position_in_lock, )
        self.log_entry_v0("Lock chamber converting start", self.env.now, self.output.copy(), self.node_open, )
        self.node_open = new_level
        remaining_levelling_time = levelling_time
        start_levelling = self.env.now
        while remaining_levelling_time:
            try:
                yield self.env.timeout(remaining_levelling_time)
                remaining_levelling_time = 0
            except simpy.Interrupt as e:
                remaining_levelling_time -= self.env.now - start_levelling

        self.log_entry_v0("Lock chamber converting stop", self.env.now, self.output.copy(), self.node_open, )
        if vessel is not None:
            vessel.log_entry_v0("Levelling stop", vessel.env.now, vessel.output.copy(), vessel.position_in_lock, )

        self.door_A.release(hold_door_A)
        self.levelling.release(hold_levelling)
        self.door_B.release(hold_door_B)


    def open_door(self, to_level=None, vessel=None, delay=0):
        start_delay = self.env.now
        while delay > 0:
            try:
                yield self.env.timeout(delay)
                delay = 0
            except simpy.Interrupt as e:
                delay -= self.env.now - start_delay
                if vessel is not None:
                    if e.cause is not None:
                        delay += float(e.cause)
        if vessel is not None:
            delattr(vessel,'door_open_request')
        wlev_chamber = 0.
        if self.env.vessel_traffic_service.hydrodynamic_information_path:
            time_index = np.absolute(hydrodynamic_times - np.datetime64(datetime.datetime.fromtimestamp(self.env.now))).argmin()
            wlev_chamber = self.water_level[time_index]

        if to_level is None:
            to_level = self.node_open

        wlev_harbour = 0.
        if self.env.vessel_traffic_service.hydrodynamic_information_path:
            time_index = np.absolute(hydrodynamic_times - np.datetime64(datetime.datetime.fromtimestamp(self.env.now))).argmin()
            if to_level == self.start_node:
                wlev_harbour = hydrodynamic_data["Water level"][0][time_index]
            else:
                wlev_harbour = hydrodynamic_data["Water level"][1][time_index]

        if to_level == self.start_node:
            direction = 1
        else:
            direction = 0

        same_direction = False
        if to_level == self.node_open:
            same_direction = True

        if np.abs(wlev_chamber - wlev_harbour) >= 0.1:
            if same_direction:
                direction = 1-direction
            yield from self.level_lock(to_level,direction=direction,same_direction=same_direction)
        else:
            self.node_open = to_level

        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now))
        if self.env.vessel_traffic_service.hydrodynamic_information_path:
            time_index = np.absolute(hydrodynamic_times - np.datetime64(current_time)).argmin()+1
            station_index = np.where(np.array(list((hydrodynamic_data['STATION']))) == self.node_open)[0]
            self.water_level[time_index:] = hydrodynamic_data["Water level"][station_index,time_index:].copy()

        # Open the doors
        hold_door_A = self.door_A.request()
        hold_levelling = self.levelling.request()
        hold_door_B = self.door_B.request()
        yield hold_door_A
        yield hold_levelling
        yield hold_door_B

        self.log_entry_v0("Lock doors opening start", self.env.now, self.output.copy(), self.node_open)
        remaining_doors_opening_time = self.doors_opening_time
        start_time_opening = self.env.now
        while remaining_doors_opening_time:
            try:
                yield self.env.timeout(remaining_doors_opening_time)
                remaining_doors_opening_time = 0
            except simpy.Interrupt as e:
                remaining_doors_opening_time -= (self.env.now - start_time_opening)
        self.log_entry_v0("Lock doors opening stop", self.env.now, self.output.copy(), self.node_open, )
        if self.node_open == self.start_node:
            self.door_A_open = True
        else:
            self.door_B_open = True

        self.door_A.release(hold_door_A)
        self.levelling.release(hold_levelling)
        self.door_B.release(hold_door_B)


class IsLockComplex(IsLockChamber,IsLockMaster):

    def __init__(self,
                 node_A,
                 node_B,
                 distance_lock_doors_A_to_waiting_area_A=0,
                 distance_lock_doors_B_to_waiting_area_B=0,
                 lineup_area_A_length=None,
                 lineup_area_B_length=None,
                 distance_lock_doors_A_to_lineup_area_A=None,
                 distance_lock_doors_B_to_lineup_area_B=None,
                 effective_lineup_area_A_length=None,
                 effective_lineup_area_B_length=None,
                 passing_allowed_in_lineup_area_A=False,
                 passing_allowed_in_lineup_area_B=False,
                 speed_reduction_factor_lineup_area_A=0.75,
                 speed_reduction_factor_lineup_area_B=0.75,
                 P_used_to_break_before_lock=None,
                 P_used_to_break_in_lock=None,
                 P_used_to_accelerate_in_lock=None,
                 P_used_to_accelerate_after_lock=None,
                 k = 0,
                 *args,
                 **kwargs):

        self.node_A = node_A
        self.node_B = node_B
        if 'start_node' not in kwargs:
            start_node = self.node_A
        else:
            start_node = kwargs['start_node']
            kwargs.pop('start_node')
        if 'end_node' not in kwargs:
            end_node = self.node_B
        else:
            end_node = kwargs['end_node']
            kwargs.pop('end_node')
        super().__init__(start_node=start_node, end_node=end_node, *args, **kwargs)
        self.distance_lock_doors_A_to_waiting_area_A = distance_lock_doors_A_to_waiting_area_A
        self.distance_lock_doors_B_to_waiting_area_B = distance_lock_doors_B_to_waiting_area_B
        self.P_used_to_break_before_lock = P_used_to_break_before_lock
        self.P_used_to_break_in_lock = P_used_to_break_in_lock
        self.P_used_to_accelerate_in_lock = P_used_to_accelerate_in_lock
        self.P_used_to_accelerate_after_lock = P_used_to_accelerate_after_lock
        self.k = k

        # Waiting area (mandatory)
        edge_waiting_area_A = (self.end_node, self.start_node, self.k)
        if self.start_node != self.node_A:
            edge_waiting_area_A = self.env.vessel_traffic_service.provide_edge_by_distance_from_node(self.env, self.start_node, self.node_A, self.distance_lock_doors_A_to_waiting_area_A-self.distance_from_start_node_to_lock_doors_A)
            distance_start_node_to_node_waiting_area_A = self.env.vessel_traffic_service.provide_sailing_distance_over_route(nx.dijkstra_path(self.env.FG, self.start_node, edge_waiting_area_A[1]))['Distance'].sum()
            self.distance_waiting_area_A_from_node_waiting_area_A = distance_start_node_to_node_waiting_area_A - (self.distance_lock_doors_A_to_waiting_area_A - self.distance_from_start_node_to_lock_doors_A)
        else:
            self.distance_waiting_area_A_from_node_waiting_area_A = self.distance_from_start_node_to_lock_doors_A - self.distance_lock_doors_A_to_waiting_area_A

        edge_waiting_area_B = (self.start_node, self.end_node, self.k)
        if self.end_node != self.node_B:
            edge_waiting_area_B = self.env.vessel_traffic_service.provide_edge_by_distance_from_node(self.env, self.end_node, self.node_B, self.distance_lock_doors_B_to_waiting_area_B-self.distance_from_end_node_to_lock_doors_B)
            distance_end_node_to_node_waiting_area_B = self.env.vessel_traffic_service.provide_sailing_distance_over_route(nx.dijkstra_path(self.env.FG,self.end_node, edge_waiting_area_B[1]))['Distance'].sum()
            self.distance_waiting_area_B_from_node_waiting_area_B = distance_end_node_to_node_waiting_area_B-(self.distance_lock_doors_B_to_waiting_area_B-self.distance_from_end_node_to_lock_doors_B)
        else:
            self.distance_waiting_area_B_from_node_waiting_area_B = self.distance_from_end_node_to_lock_doors_B - self.distance_lock_doors_B_to_waiting_area_B

        # Waiting area objects
        self.waiting_area_A = IsLockWaitingArea(env=self.env,
                                                name="waiting_area_A",
                                                lock=self,
                                                edge=(edge_waiting_area_A[1],edge_waiting_area_A[0],edge_waiting_area_A[2]),
                                                distance_from_node=self.distance_waiting_area_A_from_node_waiting_area_A)

        self.waiting_area_B = IsLockWaitingArea(env=self.env,
                                                name="waiting_area_B",
                                                lock=self,
                                                edge=(edge_waiting_area_B[1],edge_waiting_area_B[0],edge_waiting_area_B[2]),
                                                distance_from_node=self.distance_waiting_area_B_from_node_waiting_area_B)

        # If there is a line-up area at side A (optional)
        self.has_lineup_area_A = False
        if lineup_area_A_length is not None:
            self.has_lineup_area_A = True
            self.lineup_area_A_length = lineup_area_A_length
            self.effective_lineup_area_A_length = effective_lineup_area_A_length
            self.passing_allowed_in_lineup_area_A = passing_allowed_in_lineup_area_A
            self.speed_reduction_factor_lineup_area_A = speed_reduction_factor_lineup_area_A
            if lineup_area_A_length < self.lock_length and not effective_lineup_area_A_length:
                self.effective_lineup_area_A_length = self.lock_length
            self.distance_lock_doors_A_to_lineup_area_A = distance_lock_doors_A_to_lineup_area_A

            # Lineup area geometry
            edge_lineup_area_A = self.env.vessel_traffic_service.provide_edge_by_distance_from_node(self.env,
                                                                                                    self.start_node,
                                                                                                    self.node_A,
                                                                                                    self.distance_lock_doors_A_to_lineup_area_A - self.distance_from_start_node_to_lock_doors_A)

            distance_start_node_to_node_waiting_area_A = self.env.vessel_traffic_service.provide_sailing_distance_over_route(nx.dijkstra_path(self.env.FG, self.start_node, edge_lineup_area_A[1]))['Distance'].sum()
            self.distance_lineup_area_A_from_edge_lineup_area_A_start = distance_start_node_to_node_waiting_area_A - (self.distance_lock_doors_A_to_lineup_area_A - self.distance_from_start_node_to_lock_doors_A)

            # Lineup area object
            self.lineup_area_A = IsLockLineUpArea(env=self.env,
                                                  name=self.name,
                                                  start_node=edge_lineup_area_A[1],
                                                  end_node=edge_lineup_area_A[0],
                                                  lineup_area_length=self.lineup_area_A_length,
                                                  distance_from_start_edge=self.distance_lineup_area_A_from_edge_lineup_area_A_start,
                                                  effective_lineup_area_length=self.effective_lineup_area_A_length,
                                                  passing_allowed=self.passing_allowed_in_lineup_area_A,
                                                  speed_reduction_factor=self.speed_reduction_factor_lineup_area_A)

        # If there is a line-up area at side B (optional)
        self.has_lineup_area_B = False
        if lineup_area_B_length is not None:
            self.has_lineup_area_B = True
            self.lineup_area_B_length = lineup_area_B_length
            self.effective_lineup_area_B_length = effective_lineup_area_B_length
            self.passing_allowed_in_lineup_area_B = passing_allowed_in_lineup_area_B
            self.speed_reduction_factor_lineup_area_B = speed_reduction_factor_lineup_area_B
            if lineup_area_B_length < self.lock_length and not effective_lineup_area_B_length:
                self.effective_lineup_area_B_length = self.lock_length
            self.distance_lock_doors_B_to_lineup_area_B = distance_lock_doors_B_to_lineup_area_B


            edge_lineup_area_B = self.env.vessel_traffic_service.provide_edge_by_distance_from_node(self.env,
                                                                                                    self.end_node,
                                                                                                    self.node_B,
                                                                                                    self.distance_lock_doors_B_to_lineup_area_B - self.distance_from_end_node_to_lock_doors_B)

            distance_end_node_to_node_waiting_area_B = self.env.vessel_traffic_service.provide_sailing_distance_over_route(nx.dijkstra_path(self.env.FG, self.end_node, edge_lineup_area_B[1]))['Distance'].sum()
            self.distance_lineup_area_B_from_edge_lineup_area_B_start = distance_end_node_to_node_waiting_area_B - (self.distance_lock_doors_B_to_lineup_area_B - self.distance_from_end_node_to_lock_doors_B)

            # Lineup area object
            self.lineup_area_B = IsLockLineUpArea(env=self.env,
                                                  name=self.name,
                                                  start_node=edge_lineup_area_B[1],
                                                  end_node=edge_lineup_area_B[0],
                                                  distance_from_start_edge=self.distance_lineup_area_B_from_edge_lineup_area_B_start,
                                                  lineup_area_length=self.lineup_area_B_length,
                                                  effective_lineup_area_length=self.effective_lineup_area_B_length,
                                                  passing_allowed=self.passing_allowed_in_lineup_area_B,
                                                  speed_reduction_factor=self.speed_reduction_factor_lineup_area_B)
