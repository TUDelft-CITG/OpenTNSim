"""This is the lock module as part of the OpenTNSim package. See the locking examples in the book for detailed descriptions."""

# package(s) related to the simulation
import bisect
import datetime
import math
import random
import time
import time as timepy

import networkx as nx
import numpy as np
import pandas as pd

# spatial libraries
import pyproj
import pytz
import shapely.geometry
import simpy
import xarray as xr

from opentnsim import core, output


class CustomLog(core.Log):
    """this module stores log information slightly different"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_entry_extra(self, t, geometry_log, log, output):
        """Log an entry with a dictionary as output"""
        entry = {
            "Message": log,
            "Timestamp": datetime.datetime.fromtimestamp(t),
            "Geometry": geometry_log,
            "ObjectState": output,
        }
        self.logbook.append(entry)


class IsDetectorNode:
    def __init__(self, infrastructure, *args, **kwargs):
        self.infrastructure = infrastructure
        super().__init__(*args, **kwargs)


class HasMultiDiGraph(core.SimpyObject):
    """This locking module uses a MultiDiGraph to represent the network. This converts other graphs to a MultiDiGraph."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def multidigraph(self):
        # create a multidigraph copy of graph if it was not done before

        if not hasattr(self.env, "_multidigraph"):
            self.env._multidigraph = self.copy()
        return self.env._multidigraph

    def copy(self):
        multidigraph = self.env.FG
        if not isinstance(self.env.FG, nx.MultiDiGraph):
            multidigraph = nx.MultiDiGraph(multidigraph)
        return multidigraph


class HasLockInformation:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HasLock(core.Movable, HasMultiDiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node_functions.append(self.register_vessel)
        self.on_pass_edge_functions.append(self.leave_lock_chamber)

    def register_vessel(self, origin):
        if "Detector" in self.multidigraph.nodes[origin].keys():
            yield self.env.timeout(0)
            lock = []
            for node1, node2 in zip(self.route[self.route.index(origin) : -1], self.route[self.route.index(origin) + 1 :]):
                k = sorted(self.multidigraph[node1][node2], key=lambda x: self.multidigraph[node1][node2][x]["geometry"].length)[0]
                if "Lock" in self.multidigraph.edges[node1, node2, k].keys():
                    lock = self.multidigraph.edges[node1, node2, k]["Lock"][0]
                    node_doors1 = lock.node_doors1
                    node_doors2 = lock.node_doors2
                    doors2 = lock.doors_2[node_doors2]
                    if node_doors1 != node1:
                        node_doors1 = lock.node_doors2
                        node_doors2 = lock.node_doors1
                        doors2 = lock.doors_1[node_doors2]

            if lock and "lock_information" not in dir(self):
                self.lock_information = {}
            if lock and lock.name not in self.lock_information.keys():
                self.lock_information[lock.name] = HasLockInformation()

            if lock:
                self.lock_information[lock.name].in_lock = False
                if origin == lock.detector_nodes[0]:
                    direction = 1
                else:
                    direction = 0
                if not lock.next_lockage[node_doors2].users:
                    self.lock_information[lock.name].next_lockage = lock.next_lockage[node_doors2].request()
                    self.lock_information[lock.name].next_lockage.obj = self
                    self.lock_information[lock.name].next_lockage_length = lock.next_lockage_length[node_doors2].get(self.L)
                    self.lock_information[lock.name].next_lockage_length.obj = self
                    self.lock_information[lock.name].converting = False
                    self.lock_information[lock.name].lock_dist = lock.length.capacity - 0.5 * self.L
                    if not lock.next_lockage[node_doors1].users:
                        self.lock_information[lock.name].access_lock_door2 = doors2.request(priority=-1)
                        self.lock_information[lock.name].access_lock_door2.obj = self
                        if lock.node_open != node_doors1:
                            lineup_area = lock.find_previous_lineup_area(self, direction)
                            yield from lock.request_empty_lock_conversion(lineup_area, self, node_doors1, timeout_required=False)
                else:
                    self.lock_information[lock.name].next_lockage = lock.next_lockage[node_doors2].request()
                    self.lock_information[lock.name].next_lockage.obj = self
                    self.lock_information[lock.name].next_lockage_length = lock.next_lockage_length[node_doors2].get(self.L)
                    self.lock_information[lock.name].next_lockage_length.obj = self
                    self.lock_information[lock.name].converting = False

    def leave_lock_chamber(self, origin, destination):
        k = sorted(
            self.multidigraph[origin][destination], key=lambda x: self.multidigraph[origin][destination][x]["geometry"].length
        )[0]
        if "Lock" in self.multidigraph.edges[origin, destination, k].keys():
            locks = self.multidigraph.edges[origin, destination, k]["Lock"]
            if origin == locks[0].node_doors1:
                direction = 1
            else:
                direction = 0

            yield from PassLock.leave_lock(self, origin, destination, direction)
            self.v = self.v_before_lock


class HasWaitingArea(core.Movable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node_functions.append(self.leave_waiting_area)

    def leave_waiting_area(self, origin):
        if "Waiting area" in self.multidigraph.nodes[origin].keys():  # if vessel is in waiting area
            yield from PassLock.leave_waiting_area(self, origin)


class HasLineUpArea(core.Movable, HasMultiDiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_edge_functions.append(self.approach_lineup_area)  # on_look_ahead_to_node
        self.on_pass_edge_functions.append(self.leave_lineup_area)

    def approach_lineup_area(self, origin, destination):
        k1 = sorted(
            self.multidigraph[origin][destination], key=lambda x: self.multidigraph[origin][destination][x]["geometry"].length
        )[0]
        if "Line-up area" in self.multidigraph.edges[origin, destination, k1].keys():  # if vessel is approaching the line-up area
            yield from PassLock.approach_lineup_area(self, origin, destination)

    def leave_lineup_area(self, origin, destination):
        k1 = sorted(
            self.multidigraph[origin][destination], key=lambda x: self.multidigraph[origin][destination][x]["geometry"].length
        )[0]
        k2 = sorted(
            self.multidigraph[destination][origin], key=lambda x: self.multidigraph[destination][origin][x]["geometry"].length
        )[0]
        if "Line-up area" in self.multidigraph.edges[origin, destination, k1].keys():  # if vessel is located in the line-up
            yield from PassLock.leave_lineup_area(self, origin, destination)
        elif "Line-up area" in self.multidigraph.edges[destination, origin, k2].keys():
            yield from PassLock.leave_lineup_area(self, destination, origin)


class IsLockWaitingArea(core.HasResource, core.Identifiable, core.Log, output.HasOutput, HasMultiDiGraph):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
    creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity
    """

    def __init__(
        self, node, distance_from_node, *args, **kwargs  # a string which indicates the location of the start of the waiting area
    ):
        self.node = node
        self.distance_from_node = distance_from_node
        super().__init__(*args, **kwargs)
        """Initialization"""

        waiting_area_resources = 100
        self.waiting_area = {
            node: simpy.PriorityResource(self.env, capacity=waiting_area_resources),
        }

        # Add to the graph:
        if "FG" in dir(self.env):
            if "Waiting area" not in self.multidigraph.nodes[node].keys():
                self.multidigraph.nodes[node]["Waiting area"] = [self]
            else:
                self.multidigraph.nodes[node]["Waiting area"].append(self)

    def find_lineup_areas(self, vessel, index_node_waiting_area):
        lineup_areas = []
        for node1, node2 in zip(vessel.route[index_node_waiting_area:-1], vessel.route[index_node_waiting_area + 1 :]):
            k = sorted(vessel.multidigraph[node1][node2], key=lambda x: vessel.multidigraph[node1][node2][x]["geometry"].length)[0]
            if "Line-up area" in vessel.multidigraph.edges[node1, node2, k].keys():
                lineup_areas = vessel.multidigraph.edges[node1, node2, k]["Line-up area"]
            else:
                continue

        return lineup_areas

    def find_locks(self, vessel, index_node_waiting_area):
        locks = []
        for approach_node, departure_node in zip(
            vessel.route[index_node_waiting_area:-1], vessel.route[index_node_waiting_area + 1 :]
        ):
            k = sorted(
                vessel.multidigraph[approach_node][departure_node],
                key=lambda x: vessel.multidigraph[approach_node][departure_node][x]["geometry"].length,
            )[0]
            if "Lock" in vessel.multidigraph.edges[approach_node, departure_node, k].keys():
                locks.extend(vessel.multidigraph.edges[approach_node, departure_node, k]["Lock"])
            else:
                continue

        directions = []
        for lock in locks:
            if approach_node == lock.node_doors1:
                directions.append(1)
            else:
                directions.append(0)

        return locks, directions


class IsLockLineUpArea(core.HasResource, core.HasLength, core.Identifiable, core.Log, output.HasOutput, HasMultiDiGraph):
    """Mixin class: Something has line-up area object properties as part of the lock complex [in SI-units]:
    creates a line-up area with the following resources:
        - enter_line_up_area: resource used when entering the line-up area (assures one-by-one entry of the line-up area by vessels)
        - line_up_area: resource with unlimited capacity used to formally request access to the line-up area
        - converting_while_in_line_up_area: resource used when requesting for an empty conversion of the lock chamber
        - pass_line_up_area: resource used to pass the second encountered line-up area"""

    def __init__(
        self,
        start_node,  # a string which indicates the location of the start of the line-up area
        end_node,  # a string which indicates the location of the start of the line-up area
        lineup_length,  # a float which contains the length of the line-up area
        distance_to_lock_doors,
        effective_lineup_length=None,
        passing_allowed=False,
        speed_reduction_factor=0.75,
        k_edge=0,
        *args,
        **kwargs
    ):
        self.start_node = start_node
        self.end_node = end_node
        self.lineup_length = self.effective_lineup_length = lineup_length
        if effective_lineup_length:
            self.effective_lineup_length = effective_lineup_length
        self.distance_to_lock_doors = distance_to_lock_doors
        self.passing_allowed = passing_allowed
        self.speed_reduction_factor = speed_reduction_factor
        super().__init__(length=self.effective_lineup_length, remaining_length=self.effective_lineup_length, *args, **kwargs)

        """Initialization"""
        # Lay-Out
        self.enter_line_up_area = {
            start_node: simpy.PriorityResource(self.env, capacity=1),
        }  # used to regulate one by one entering of line-up area, so capacity must be 1
        self.line_up_area = {
            start_node: simpy.PriorityResource(self.env, capacity=100),
        }  # line-up area itself, infinite capacity, as this is regulated by the HasLength, so capacity = inf
        self.converting_while_in_line_up_area = {
            start_node: simpy.PriorityResource(self.env, capacity=1),
        }  # used to minimize the number of empty convertion requests by one by multiple waiting vessels, so capacity must be 1
        self.pass_line_up_area = {
            start_node: simpy.PriorityResource(self.env, capacity=1),
        }  # used to prevent vessel from entering the lock before all previously locked vessels have passed the line-up area one by one, so capacity must be 1

        # Add to the graph:
        if "FG" in dir(self.env):
            if "Line-up area" not in self.multidigraph.edges[self.start_node, self.end_node, k_edge].keys():
                self.multidigraph.edges[self.start_node, self.end_node, k_edge]["Line-up area"] = [self]
            else:
                self.multidigraph.edges[self.start_node, self.end_node, k_edge]["Line-up area"].append(self)

    def find_lock(self, vessel, start_node, end_node, direction=0):
        lock = None
        if direction:
            route = list(reversed(vessel.route))
        else:
            route = vessel.route

        index_node_start_lineup_area = route.index(start_node)
        index_node_end_lineup_area = route.index(end_node)

        if direction:
            loop_route_1 = route[index_node_end_lineup_area:]
            loop_route_2 = route[index_node_start_lineup_area:-1]
        else:
            loop_route_1 = route[index_node_start_lineup_area:-1]
            loop_route_2 = route[index_node_end_lineup_area:]

        for approach_node, departure_node in zip(loop_route_1, loop_route_2):
            k = sorted(
                vessel.multidigraph[approach_node][departure_node],
                key=lambda x: vessel.multidigraph[approach_node][departure_node][x]["geometry"].length,
            )[0]
            if "Lock" in vessel.multidigraph.edges[approach_node, departure_node, k].keys():
                locks = vessel.multidigraph.edges[approach_node, departure_node, k]["Lock"]
                if approach_node == locks[0].node_doors1:
                    direction = 1
                else:
                    direction = 0
            else:
                continue

            for lock in locks:
                if lock.name is self.name:
                    break
            if lock.name is self.name:
                break

        return lock, direction


class IsLock(core.HasResource, core.HasLength, core.Identifiable, CustomLog, output.HasOutput, HasMultiDiGraph):
    """Mixin class: Something which has lock chamber object properties as part of a lock complex [in SI-units]"""

    def __init__(
        self,
        node_doors1,  # a string which indicates the location of the first pair of lock doors
        node_doors2,  # a string which indicates the location of the second pair of lock doors
        lock_length,  # a float which contains the length of the lock chamber
        lock_width,  # a float which contains the width of the lock chamber
        lock_depth,  # a float which contains the depth of the lock chamber
        distance_doors1_from_first_waiting_area,
        distance_doors2_from_second_waiting_area,
        detector_nodes=None,
        doors_open=0,  # a float which contains the time it takes to open the doors
        doors_close=0,  # a float which contains the time it takes to close the doors
        disch_coeff=0,  # a float which contains the discharge coefficient of filling system
        opening_area=0,  # a float which contains the cross-sectional area of filling system
        opening_depth=0,  # a float which contains the depth at which filling system is located
        speed_reduction_factor=0.3,
        k_edge=0,
        levelling_time=0,
        grav_acc=9.81,  # a float which contains the gravitational acceleration
        time_step=10,
        node_open=None,
        conditions=None,
        priority_rules=None,
        mandatory_time_gap_between_entering_vessels=None,
        used_as_one_way_traffic_regulation=False,
        *args,
        **kwargs
    ):
        """Initialization"""
        # Properties
        self.lock_length = lock_length
        self.lock_width = lock_width
        self.lock_depth = lock_depth
        self.disch_coeff = disch_coeff
        self.grav_acc = grav_acc
        self.opening_area = opening_area
        self.opening_depth = opening_depth
        self.levelling_time = levelling_time
        self.speed_reduction_factor = speed_reduction_factor
        self.node_open = node_open
        self.conditions = conditions
        self.time_step = time_step
        self.priority_rules = priority_rules
        if detector_nodes is None:
            detector_nodes = []
        self.detector_nodes = detector_nodes
        self.distance_doors1_from_first_waiting_area = distance_doors1_from_first_waiting_area
        self.distance_doors2_from_second_waiting_area = distance_doors2_from_second_waiting_area
        self.mandatory_time_gap_between_entering_vessels = mandatory_time_gap_between_entering_vessels
        self.used_as_one_way_traffic_regulation = used_as_one_way_traffic_regulation
        self.converting_chamber = False
        self.schedule = pd.DataFrame(
            index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=["Name", "VesselName"]),
            columns=["direction", "ETA", "ETD", "Length", "Beam", "Draught", "VesselType", "Priority"],
        )

        # 100 spaces in the lock by default
        capacity = 100

        super().__init__(nr_resources=capacity, length=lock_length, remaining_length=lock_length, *args, **kwargs)
        self.simulation_start = self.env.simulation_start.timestamp()
        self.next_lockage = {
            node_doors1: simpy.PriorityResource(self.env, capacity=capacity),
            node_doors2: simpy.PriorityResource(self.env, capacity=capacity),
        }
        self.next_lockage_length = {
            node_doors1: simpy.Container(self.env, capacity=lock_length, init=lock_length),
            node_doors2: simpy.Container(self.env, capacity=lock_length, init=lock_length),
        }
        self.in_next_lockage = {
            node_doors1: simpy.PriorityResource(self.env, capacity=capacity),
            node_doors2: simpy.PriorityResource(self.env, capacity=capacity),
        }
        self.doors_1 = {
            node_doors1: simpy.PriorityResource(self.env, capacity=1),
        }  # Only one ship can pass at a time: capacity = 1, request can have priority
        self.doors_2 = {
            node_doors2: simpy.PriorityResource(self.env, capacity=1),
        }  # Only one ship can pass at a time: capacity = 1, request can have priority

        # Operating
        self.doors_open = doors_open
        self.doors_close = doors_close

        # Water level
        assert node_doors1 != node_doors2

        self.node_doors1 = node_doors1
        self.node_doors2 = node_doors2
        if not self.node_open:
            self.node_open = random.choice([node_doors1, node_doors2])
        index_node_open = list(self.multidigraph.nodes).index(self.node_open)

        if "hydrodynamic_information" in dir(self.env.vessel_traffic_service):
            iter_data = self.env.vessel_traffic_service.hydrodynamic_information.sel(STATIONS=index_node_open).interp(
                TIME=np.arange(self.env.simulation_start, self.env.simulation_stop, datetime.timedelta(seconds=self.time_step))
            )
            self.water_level = iter_data["Water level"].copy()
            self.salinity = iter_data["Salinity"].copy()
            self.discharge_res = np.abs(self.water_level.rename("Residual discharge").copy()) * 0
            self.discharge_saline = np.abs(self.water_level.rename("Saline discharge").copy()) * 0
            self.discharge_fresh = np.abs(self.water_level.rename("Fresh discharge").copy()) * 0

        for detector_node in self.detector_nodes:
            if "Detector" not in self.multidigraph.nodes[detector_node]:
                self.multidigraph.nodes[detector_node]["Detector"] = {}

            route1 = nx.dijkstra_path(self.multidigraph, detector_node, self.node_doors1)
            route2 = nx.dijkstra_path(self.multidigraph, detector_node, self.node_doors2)
            for route in [route1, route2]:
                if len(route) > 1 and (
                    [self.node_doors1, self.node_doors2] == [route[-2], route[-1]]
                    or [self.node_doors1, self.node_doors2] == [route[-1], route[-2]]
                ):
                    self.multidigraph.nodes[detector_node]["Detector"][route[-1]] = IsDetectorNode(self)
                    break

        # Add to the graph:
        if "FG" in dir(self.env):
            # Add the lock to the edge or append it to the existing list
            if "Lock" not in self.multidigraph.edges[self.node_doors1, self.node_doors2, k_edge].keys():
                self.multidigraph.edges[self.node_doors1, self.node_doors2, k_edge]["Lock"] = [self]
                self.multidigraph.edges[self.node_doors2, self.node_doors1, k_edge]["Lock"] = [self]
            else:
                self.multidigraph.edges[self.node_doors1, self.node_doors2, k_edge]["Lock"].append(self)
                self.multidigraph.edges[self.node_doors2, self.node_doors1, k_edge]["Lock"].append(self)

    def check_priority(self, vessel, direction):
        # we do not have priority
        condition = False
        waiting_area = self.find_previous_waiting_area(vessel, direction)
        if self.priority_rules is not None:
            condition = self.priority_rules.evaluate(waiting_area.node).iloc[0]
        if condition:
            priority = -1
        else:
            priority = 0
        return priority

    def request_empty_lock_conversion(self, lineup_area, vessel, approach_node, hold_request=False, timeout_required=True):
        """Vessel will request the lock chamber to be converted without vessels to his side of the lock chamber. This
            is programmed by requesting the converting_while_in_line_up_area resource of the line-up area the vessels is
            currently located in. If there was already a request by another vessel waiting in the same line-up area, this
            original request can be holded.

        Input:
            - hold_request: a boolean which defines where an earlier request for the same empty lock conversion which
                            was made by another ship should be holded"""

        vessel.lock_information[self.name].waiting_during_converting = lineup_area.converting_while_in_line_up_area[
            lineup_area.start_node
        ].request()
        vessel.lock_information[self.name].waiting_during_converting.obj = vessel
        vessel.lock_information[self.name].empty_converting = True
        if timeout_required:
            yield vessel.lock_information[self.name].waiting_during_converting
        if not hold_request:
            if timeout_required:
                yield from self.convert_chamber(vessel.env, approach_node, 0, vessel)
                delattr(vessel.lock_information[self.name], "empty_converting")
            else:
                yield from self.convert_chamber(vessel.env, approach_node, 0, vessel, timeout_required=False)
        lineup_area.converting_while_in_line_up_area[lineup_area.start_node].release(
            vessel.lock_information[self.name].waiting_during_converting
        )

    def find_previous_lineup_area(self, vessel, direction):
        lineup_area = None
        if direction:
            index_node_doors2 = vessel.route.index(self.node_doors2)
        else:
            index_node_doors2 = vessel.route.index(self.node_doors1)

        for approach_node, departure_node in zip(
            list(reversed(vessel.route[:index_node_doors2])), list(reversed(vessel.route[1 : index_node_doors2 + 1]))
        ):
            k = sorted(
                vessel.multidigraph[approach_node][departure_node],
                key=lambda x: vessel.multidigraph[approach_node][departure_node][x]["geometry"].length,
            )[0]
            if "Line-up area" in vessel.multidigraph.edges[approach_node, departure_node, k].keys():
                lineup_areas = vessel.multidigraph.edges[approach_node, departure_node, k]["Line-up area"]
            else:
                continue

            for lineup_area in lineup_areas:
                if lineup_area.name is self.name:
                    break
            if lineup_area.name is self.name:
                break

        return lineup_area

    def find_next_lineup_area(self, vessel, direction):
        lineup_area = None
        if direction:
            index_node_doors1 = vessel.route.index(self.node_doors1)
            index_node_doors2 = vessel.route.index(self.node_doors2)
        else:
            index_node_doors1 = vessel.route.index(self.node_doors2)
            index_node_doors2 = vessel.route.index(self.node_doors1)

        for departure_node, approach_node in zip(vessel.route[index_node_doors2:], vessel.route[index_node_doors1:-1]):
            k = sorted(
                vessel.multidigraph[departure_node][approach_node],
                key=lambda x: vessel.multidigraph[departure_node][approach_node][x]["geometry"].length,
            )[0]
            if "Line-up area" in vessel.multidigraph.edges[departure_node, approach_node, k].keys():
                lineup_areas = vessel.multidigraph.edges[departure_node, approach_node, k]["Line-up area"]
            else:
                continue

            for lineup_area in lineup_areas:
                if lineup_area.name is self.name:
                    break
            if lineup_area.name is self.name:
                break

        return lineup_area

    def find_previous_waiting_area(self, vessel, direction):
        waiting_area = None
        if direction:
            index_node_doors2 = vessel.route.index(self.node_doors2)
        else:
            index_node_doors2 = vessel.route.index(self.node_doors1)

        for previous_node in list(reversed(vessel.route[:index_node_doors2])):
            if "Waiting area" in vessel.multidigraph.nodes[previous_node].keys():
                waiting_areas = vessel.multidigraph.nodes[previous_node]["Waiting area"]
            else:
                continue

            for waiting_area in waiting_areas:
                if waiting_area.name is self.name:
                    break
            if waiting_area.name is self.name:
                break

        return waiting_area

    def find_next_waiting_area(self, vessel, direction):
        waiting_area = None
        if direction:
            index_node_doors1 = vessel.route.index(self.node_doors1)
        else:
            index_node_doors1 = vessel.route.index(self.node_doors2)

        for next_node in vessel.route[index_node_doors1:]:
            if "Waiting area" in vessel.multidigraph.nodes[next_node].keys():
                waiting_areas = vessel.multidigraph.nodes[next_node]["Waiting area"]
            else:
                continue

            for waiting_area in waiting_areas:
                if waiting_area.name is self.name:
                    break
            if waiting_area.name is self.name:
                break

        return waiting_area

    def exchange_flux_time_series_calculator(self, T_door_open, time_index):
        index_node_doors1 = list(self.multidigraph.nodes).index(self.node_doors1)
        index_node_doors2 = list(self.multidigraph.nodes).index(self.node_doors2)
        index_node_open = list(self.multidigraph.nodes).index(self.node_open)
        time_value = self.salinity.TIME[time_index].values
        S_lock = self.salinity[time_index]
        S_lock_harbour = (
            self.env.vessel_traffic_service.hydrodynamic_information["Salinity"][index_node_open].interp(TIME=time_value).values
        )
        S_lock_average = (
            self.env.vessel_traffic_service.hydrodynamic_information["Salinity"][index_node_doors1].interp(TIME=time_value).values
            + self.env.vessel_traffic_service.hydrodynamic_information["Salinity"][index_node_doors2].interp(TIME=time_value).values
        ) / 2
        Wlev_lock = (
            self.env.vessel_traffic_service.hydrodynamic_information["Water level"][index_node_open].interp(TIME=time_value).values
        )
        V_lock = self.lock_length * self.lock_width * (Wlev_lock + self.lock_depth)
        v_exch = 0.5 * np.sqrt(
            self.grav_acc * (0.8 * abs(S_lock_harbour - S_lock) / (1000 + 0.8 * S_lock_average)) * (Wlev_lock + self.lock_depth)
        )
        if v_exch != 0:
            T_LE = (2 * self.lock_length) / v_exch
        else:
            T_LE = 0

        time = np.arange(0, np.min([T_door_open, 2 * 3600]), self.time_step)
        Q = []
        V_tot = 0
        for t in enumerate(time):
            if t[0] == 0:
                continue
            delta_t = t[1] - time[t[0] - 1]
            if T_LE != 0:
                delta_V = V_lock * (np.tanh(t[1] / T_LE) - np.tanh(time[t[0] - 1] / T_LE))
            else:
                delta_V = 0
            V_tot += delta_V
            V_tot += delta_V
            Q.append(delta_V / delta_t)
            M = (S_lock_harbour - self.salinity[time_index].values) * delta_V
            S_lock = (S_lock * V_lock + M) / V_lock
            self.salinity[time_index + t[0]] = S_lock.values
            if not self.discharge_saline[time_index + t[0]].values:
                self.discharge_saline[time_index + t[0]] += delta_V / delta_t
            if not self.discharge_fresh[time_index + t[0]].values:
                self.discharge_fresh[time_index + t[0]] += -delta_V / delta_t
            if np.abs(S_lock.values - S_lock_harbour) < 0.25:
                outer_salinity = self.env.vessel_traffic_service.hydrodynamic_information["Salinity"][index_node_open].interp(
                    TIME=np.arange(self.env.simulation_start, self.env.simulation_stop, datetime.timedelta(seconds=self.time_step))
                )
                self.salinity[time_index + t[0] :] = outer_salinity[time_index + t[0] :]
                break

    def levelling_to_harbour(self, V_ship, levelling_time, side, delay=0):
        index_node_doors1 = list(self.multidigraph.nodes).index(self.node_doors1)
        index_node_doors2 = list(self.multidigraph.nodes).index(self.node_doors2)
        time_index = np.absolute(
            self.env.vessel_traffic_service.hydrodynamic_information.TIME.values
            - np.datetime64(datetime.datetime.fromtimestamp(self.env.now + delay))
        ).argmin()
        time_index_start = np.absolute(
            self.salinity.TIME.values - np.datetime64(datetime.datetime.fromtimestamp(self.env.now + delay))
        ).argmin()
        time_value_start = self.salinity.TIME.values[time_index_start]
        time_index_stop = np.absolute(
            self.salinity.TIME.values - np.datetime64(datetime.datetime.fromtimestamp(self.env.now + levelling_time + delay))
        ).argmin()
        time = np.arange(self.env.now + delay, self.env.now + levelling_time + delay, self.time_step)
        S_lock_start = self.salinity.values[time_index_start]
        WLev_lock_inner = self.env.vessel_traffic_service.hydrodynamic_information["Water level"][index_node_doors2].interp(
            TIME=time_value_start
        )
        V_lock_inner = self.lock_length * self.lock_width * (WLev_lock_inner.values + self.lock_depth)
        WLev_lock_outer = self.env.vessel_traffic_service.hydrodynamic_information["Water level"][index_node_doors1].interp(
            TIME=time_value_start
        )
        V_lock_outer = self.lock_length * self.lock_width * (WLev_lock_outer.values + self.lock_depth)
        if side == self.node_doors1:
            WLev_to_side = WLev_lock_inner
            WLev_from_side = WLev_lock_outer
            V_to_side = V_lock_inner
            V_from_side = V_lock_outer
            if WLev_from_side < WLev_to_side:
                filling = True
                S_to_side = self.env.vessel_traffic_service.hydrodynamic_information["Salinity"][index_node_doors2].values[
                    time_index
                ]
            else:
                filling = False
        else:
            WLev_to_side = WLev_lock_outer
            WLev_from_side = WLev_lock_inner
            V_to_side = V_lock_outer
            V_from_side = V_lock_inner
            if WLev_from_side < WLev_to_side:
                filling = True
                S_to_side = self.env.vessel_traffic_service.hydrodynamic_information["Salinity"][index_node_doors1].values[
                    time_index
                ]
            else:
                filling = False

        if filling:
            V_levelling = self.lock_length * self.lock_width * (WLev_to_side - WLev_from_side)
            S_lock_final = S_lock = (S_lock_start * (V_from_side - V_ship) + V_levelling * S_to_side) / (V_to_side - V_ship)
            dt = (self.water_level.TIME.values[1] - self.water_level.TIME.values[0]) / np.timedelta64(1, "s")
            for t in enumerate(time):
                sum_disch = abs((self.discharge_res[time_index_start : (time_index_start + t[0])]).sum())
                S_lock = (S_lock_start * (V_from_side - V_ship) + sum_disch * dt * S_to_side) / (
                    (V_from_side + sum_disch * dt) - V_ship
                )
                self.salinity[time_index_start + t[0]] = S_lock
            V_loss_lev = 0

        else:
            V_levelling = self.lock_length * self.lock_width * (WLev_from_side - WLev_to_side)
            S_lock_final = S_lock = S_lock_start
            V_loss_lev = V_levelling

        self.salinity[time_index_stop:] = S_lock
        return V_levelling, S_lock_final, V_loss_lev

    def sailing_out_to_harbour(self, V_ship, time_index):
        index_node_open = list(self.multidigraph.nodes).index(self.node_open)
        S_lock = self.salinity[time_index]
        time_value = self.salinity.TIME.values[time_index]
        WLev_lock_harbour = (
            self.env.vessel_traffic_service.hydrodynamic_information["Water level"][index_node_open].interp(TIME=time_value).values
        )
        S_lock_harbour = (
            self.env.vessel_traffic_service.hydrodynamic_information["Salinity"][index_node_open].interp(TIME=time_value).values
        )
        V_lock_harbour = self.lock_length * self.lock_width * (WLev_lock_harbour + self.lock_depth)
        S_lock = (S_lock * (V_lock_harbour - V_ship) + V_ship * S_lock_harbour) / V_lock_harbour
        start_time_passing_door = self.env.now
        end_time_passing_door = self.env.now + 0.5 * self.doors_open
        passing_time_door = np.arange(start_time_passing_door, end_time_passing_door, self.time_step)
        if self.node_open == self.node_doors1:
            for t in enumerate(passing_time_door):
                if time_index + t[0] < len(self.discharge_fresh):
                    self.discharge_saline[time_index + t[0]] += V_ship / (end_time_passing_door - start_time_passing_door)
                    self.discharge_res[time_index + t[0]] += V_ship / (end_time_passing_door - start_time_passing_door)
        else:
            for t in enumerate(passing_time_door):
                if time_index + t[0] < len(self.discharge_fresh):
                    self.discharge_fresh[time_index + t[0]] += -V_ship / (end_time_passing_door - start_time_passing_door)
                    self.discharge_res[time_index + t[0]] += -V_ship / (end_time_passing_door - start_time_passing_door)
        return S_lock

    def door_open_harbour(self, T_door_open, time_index_start):
        index_node_doors1 = list(self.multidigraph.nodes).index(self.node_doors1)
        index_node_doors2 = list(self.multidigraph.nodes).index(self.node_doors2)
        index_node_open = list(self.multidigraph.nodes).index(self.node_open)
        time_value = self.salinity.TIME[time_index_start].values
        S_lock = self.salinity[time_index_start]
        S_lock_harbour = (
            self.env.vessel_traffic_service.hydrodynamic_information["Salinity"][index_node_open].interp(TIME=time_value).values
        )
        S_lock_average = (
            self.env.vessel_traffic_service.hydrodynamic_information["Salinity"][index_node_doors1].interp(TIME=time_value).values
            + self.env.vessel_traffic_service.hydrodynamic_information["Salinity"][index_node_doors2].interp(TIME=time_value).values
        ) / 2
        WLev_lock_harbour = (
            self.env.vessel_traffic_service.hydrodynamic_information["Water level"][index_node_open].interp(TIME=time_value).values
        )
        V_lock_harbour = self.lock_length * self.lock_width * (WLev_lock_harbour + self.lock_depth)
        self.exchange_flux_time_series_calculator(T_door_open, time_index_start)

        # A loop that breaks at a certain moment assigning discharges to the sluice? (discharge should be separated for positive negative (and maybe sluice gates))
        if S_lock_harbour != S_lock:
            T_exch = self.lock_length / (
                0.5
                * np.sqrt(
                    self.grav_acc
                    * (0.8 * abs(S_lock_harbour - S_lock) / (1000 + 0.8 * S_lock_average))
                    * (WLev_lock_harbour + self.lock_depth)
                )
            )
            V_exch = V_lock_harbour * np.tanh(T_door_open / (2 * T_exch))
        else:
            V_exch = 0
        M_exch = (S_lock_harbour - S_lock) * V_exch
        S_lock = (S_lock * V_lock_harbour + M_exch) / V_lock_harbour
        return S_lock

    def sailing_in_from_harbour(self, V_ship, time_index):
        S_lock = self.salinity[time_index]
        start_time_passing_door = self.env.now
        end_time_passing_door = start_time_passing_door + 0.5 * self.doors_close
        passing_time_door = np.arange(start_time_passing_door, end_time_passing_door, self.time_step)
        if self.node_open == self.node_doors2:
            for t in enumerate(passing_time_door):
                if time_index + t[0] < len(self.discharge_saline) - 1:
                    self.discharge_saline[time_index + t[0]] = V_ship / (end_time_passing_door - start_time_passing_door)
                    self.discharge_res[time_index + t[0]] = V_ship / (end_time_passing_door - start_time_passing_door)
        if self.node_open == self.node_doors1:
            for t in enumerate(passing_time_door):
                if time_index + t[0] < len(self.discharge_fresh) - 1:
                    self.discharge_fresh[time_index + t[0]] = -V_ship / (end_time_passing_door - start_time_passing_door)
                    self.discharge_res[time_index + t[0]] = -V_ship / (end_time_passing_door - start_time_passing_door)
        return S_lock

    def total_ship_volume_in_lock(self):
        volume = 0
        for user in self.resource.users:
            volume += user.obj.B * user.obj.L * user.obj.T
        return volume

    def determine_levelling_time(self, new_level, delay=0):
        """Function which calculates the operation time: based on the constant or nearest in the signal of the water level difference

        Input:
            - environment: see init function"""

        def calculate_discharge(lock, z, to_wlev, from_wlev, time_index, time_step):
            two_gz = np.sqrt(2 * 9.81 * z)
            if to_wlev[time_step] <= from_wlev[time_step]:
                lock.water_level[time_index + time_step] = z + to_wlev[time_step]
                if z != 0:
                    if lock.node_open == lock.node_doors2:
                        lock.discharge_res[time_step + time_index] += -1 * lock.disch_coeff * lock.opening_area * two_gz
                        lock.discharge_fresh[time_step + time_index] += -1 * lock.disch_coeff * lock.opening_area * two_gz
                    else:
                        lock.discharge_res[time_step + time_index] += lock.disch_coeff * lock.opening_area * two_gz
                        lock.discharge_saline[time_step + time_index] += lock.disch_coeff * lock.opening_area * two_gz
                else:
                    lock.discharge_res[time_step + time_index] = 0
                    lock.discharge_fresh[time_step + time_index] = 0
            else:
                lock.water_level[time_index + time_step] = to_wlev[time_step] - z
                if z != 0:
                    if lock.node_open == lock.node_doors2:
                        lock.discharge_res[time_step + time_index] += lock.disch_coeff * lock.opening_area * two_gz
                        lock.discharge_saline[time_step + time_index] += lock.disch_coeff * lock.opening_area * two_gz
                    else:
                        lock.discharge_res[time_step + time_index] += -1 * lock.disch_coeff * lock.opening_area * two_gz
                        lock.discharge_fresh[time_step + time_index] += -1 * lock.disch_coeff * lock.opening_area * two_gz
                else:
                    lock.discharge_res[time_step + time_index] += 0
                    lock.discharge_fresh[time_step + time_index] += 0

        index_node_doors1 = list(self.multidigraph.nodes).index(self.node_doors1)
        index_node_doors2 = list(self.multidigraph.nodes).index(self.node_doors2)
        if "hydrodynamic_information" in dir(self.env.vessel_traffic_service):
            time = np.arange(
                0,
                2 * 3600,
                (
                    self.env.vessel_traffic_service.hydrodynamic_information["TIME"].values[1]
                    - self.env.vessel_traffic_service.hydrodynamic_information["TIME"].values[0]
                )
                / np.timedelta64(1, "s"),
            )
            time_index = np.absolute(
                self.env.vessel_traffic_service.hydrodynamic_information.TIME.values
                - np.datetime64(datetime.datetime.fromtimestamp(self.env.now + delay))
            ).argmin()
            time_value = self.env.vessel_traffic_service.hydrodynamic_information.TIME.values[time_index]
            wlev_outer = self.env.vessel_traffic_service.hydrodynamic_information["Water level"][index_node_doors1][
                time_index : (time_index + len(time))
            ].interp(TIME=np.arange(time_value, time_value + np.timedelta64(2, "h"), datetime.timedelta(seconds=self.time_step)))
            wlev_inner = self.env.vessel_traffic_service.hydrodynamic_information["Water level"][index_node_doors2][
                time_index : (time_index + len(time))
            ].interp(TIME=np.arange(time_value, time_value + np.timedelta64(2, "h"), datetime.timedelta(seconds=self.time_step)))
            H = abs(wlev_outer[0] - wlev_inner[0]).values
            z = abs(wlev_outer[0] - wlev_inner[0]).values
            if self.node_open == self.node_doors2:
                from_wlev = wlev_inner
                to_wlev = wlev_outer
                index_node_doors = index_node_doors1
            else:
                from_wlev = wlev_outer
                to_wlev = wlev_inner
                index_node_doors = index_node_doors2
            time_index = np.absolute(
                self.discharge_res.TIME.values
                - self.env.vessel_traffic_service.hydrodynamic_information["Water level"][index_node_doors1].TIME[time_index].values
            ).argmin()
            calculate_discharge(self, z, to_wlev, from_wlev, time_index, 0)
            time_series = np.arange(0, 2 * 3600, self.time_step)

            for t in enumerate(time_series):
                if t[0] == 0:
                    continue
                if t[0] + time_index >= len(self.water_level) - 1:
                    levelling_time = t[1]
                    break

                dz = (
                    np.sqrt(H)
                    - ((self.disch_coeff * self.opening_area * np.sqrt(2 * 9.81)) / (2 * (self.lock_length * self.lock_width)))
                    * (t[1] + self.time_step / 2)
                ) ** 2 - (
                    np.sqrt(H)
                    - ((self.disch_coeff * self.opening_area * np.sqrt(2 * 9.81)) / (2 * (self.lock_length * self.lock_width)))
                    * (t[1] - self.time_step / 2)
                ) ** 2
                z += dz
                calculate_discharge(self, z, to_wlev, from_wlev, time_index, t[0])
                if z <= 0.05:
                    levelling_time = t[1]
                    outer_water_level = self.env.vessel_traffic_service.hydrodynamic_information["Water level"][
                        index_node_doors
                    ].interp(
                        TIME=np.arange(
                            self.env.simulation_start, self.env.simulation_stop, datetime.timedelta(seconds=self.time_step)
                        )
                    )
                    self.water_level[(t[0] + time_index + 1) :] = outer_water_level[(t[0] + time_index + 1) :]
                    break
            else:
                levelling_time = 0

        else:
            levelling_time = self.levelling_time

        # V_levelling, S_lock, V_loss_lev are not used
        _ = self.levelling_to_harbour(self.total_ship_volume_in_lock(), levelling_time, self.node_open, delay)

        return levelling_time

    def convert_chamber(self, environment, new_level, number_of_vessels, vessel, timeout_required=True):
        """Function which converts the lock chamber and logs this event.

        Input:
            - environment: see init function
            - new_level: a string which represents the node and indicates the side at which the lock is currently levelled
            - number_of_vessels: the total number of vessels which are levelled simultaneously"""

        self.converting_chamber = True

        def door_open():
            if len(self.logbook) > 4:
                t_doors_open = time.mktime(pd.Timestamp(self.logbook[-4]["Timestamp"]).timetuple())
            else:
                t_doors_open = time.mktime(self.env.simulation_start.timetuple())

            if timeout_required:
                T_door_open = self.env.now - t_doors_open - 0.5 * self.doors_open
            else:
                T_door_open = self.env.now + self.doors_close - t_doors_open - 0.5 * self.doors_open

            if "hydrodynamic_information" in dir(self.env.vessel_traffic_service):
                time_index_start = np.absolute(
                    self.water_level.TIME.values - np.datetime64(datetime.datetime.fromtimestamp(self.env.now - T_door_open))
                ).argmin()
                time_index_stop = np.absolute(
                    self.water_level.TIME.values
                    - np.datetime64(datetime.datetime.fromtimestamp(self.env.now - 0.5 * self.doors_close))
                ).argmin()
                S_lock = self.door_open_harbour(T_door_open - 0.5 * self.doors_close, time_index_start)
                time_series = self.env.vessel_traffic_service.hydrodynamic_information["TIME"].values
                if time_index_stop != len(time_series) - 2:
                    self.salinity[time_index_stop:] = S_lock

        # Close the doors
        self.log_entry_extra(self.env.now, self.node_open, "Lock doors closing start", self.output.copy())
        if "hydrodynamic_information" in dir(self.env.vessel_traffic_service):
            time_index = np.max(
                [
                    0,
                    np.absolute(
                        self.salinity.TIME.values
                        - np.datetime64(
                            datetime.datetime.fromtimestamp(
                                timepy.mktime(pd.Timestamp(self.logbook[-1]["Timestamp"]).timetuple()) + 0.5 * self.doors_close
                            )
                        )
                    ).argmin(),
                ]
            )
            if len(self.resource.users):
                volume = 0
                for user in self.resource.users:
                    volume += user.obj.L * user.obj.B * user.obj.T
                S_lock = self.sailing_in_from_harbour(volume, time_index)
                self.salinity[time_index:] = S_lock

        if timeout_required:
            yield environment.timeout(self.doors_close)
            self.log_entry_extra(environment.now, self.node_open, "Lock doors closing stop", self.output.copy())
            door_open()
            vessel.levelling_time = self.determine_levelling_time(new_level)
        else:
            self.log_entry_extra(environment.now + self.doors_close, self.node_open, "Lock doors closing stop", self.output.copy())
            door_open()
            vessel.levelling_time = self.determine_levelling_time(new_level, delay=self.doors_close)

        # Convert the chamber
        if timeout_required:
            self.log_entry_extra(self.env.now, self.node_open, "Lock chamber converting start", self.output.copy())
        else:
            self.log_entry_extra(
                self.env.now + self.doors_close, self.node_open, "Lock chamber converting start", self.output.copy()
            )

        # Water level will shift
        self.change_water_level(new_level)
        if timeout_required:
            yield environment.timeout(vessel.levelling_time)
            self.log_entry_extra(self.env.now, self.node_open, "Lock chamber converting stop", self.output.copy())
        else:
            self.log_entry_extra(
                self.env.now + self.doors_close + vessel.levelling_time,
                self.node_open,
                "Lock chamber converting stop",
                self.output.copy(),
            )

        # Open the doors
        if timeout_required:
            self.log_entry_extra(self.env.now, self.node_open, "Lock doors opening start", self.output.copy())
        else:
            self.log_entry_extra(
                self.env.now + self.doors_close + vessel.levelling_time,
                self.node_open,
                "Lock doors opening start",
                self.output.copy(),
            )

        if "hydrodynamic_information" in dir(vessel.env.vessel_traffic_service):
            time_index = np.max(
                [
                    0,
                    np.absolute(
                        self.salinity.TIME.values
                        - np.datetime64(
                            datetime.datetime.fromtimestamp(timepy.mktime(pd.Timestamp(self.logbook[-1]["Timestamp"]).timetuple()))
                        )
                    ).argmin(),
                ]
            )
            if len(self.resource.users):
                volume = 0
                for user in self.resource.users:
                    volume += user.obj.L * user.obj.B * user.obj.T
                S_lock = self.sailing_out_to_harbour(volume, time_index)
                self.salinity[(time_index + int(0.5 * self.doors_open / self.time_step)) :] = S_lock

        if timeout_required:
            yield environment.timeout(self.doors_open)
            self.log_entry_extra(self.env.now, self.node_open, "Lock doors opening stop", self.output.copy())
        else:
            self.log_entry_extra(
                self.env.now + self.doors_close + vessel.levelling_time + self.doors_open,
                self.node_open,
                "Lock doors opening stop",
                self.output.copy(),
            )
        self.converting_chamber = False

    def change_water_level(self, side):
        """Function which changes the water level in the lock chamber and priorities in queue"""

        self.node_open = side
        for request in self.resource.queue:
            request.priority = -1 if request.priority == 0 else 0

            if request.priority == -1:
                self.resource.queue.insert(0, self.resource.queue.pop(self.resource.queue.index(request)))
            else:
                self.resource.queue.insert(-1, self.resource.queue.pop(self.resource.queue.index(request)))


class PassLock:
    """Mixin class: a collection of functions which are used to pass a lock complex consisting of a waiting area, line-up areas, and lock chambers"""

    @staticmethod
    def leave_waiting_area(vessel, node_waiting_area):
        """Processes vessels which are waiting in the waiting area of the lock complex and requesting access to preceding the line-up area:
            if there area multiple parallel lock chambers, the chamber with the least expected total waiting time is chosen,
            after which access is requested to enter the line-up area corresponding with the assigned lock chain series.

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
            - node_waiting_area: a string which includes the name of the node the waiting area is located in the network"""

        def choose_lock_chamber(vessel, lock, lock_position, series_number, lineup_areas, lock_queue_length):
            """Assigns the lock chamber with the least expected total waiting time to the vessel in case of parallell lock chambers. The
                expected total waiting time is calculated through quantifying the total length of the queued vessels. If a vessel does
                not fit in a lockage, it will create a new lockage cycle by requesting the full length capacity of the line-up area. When
                this request is granted, the vessel will immediately release the obsolete length, such that more vessels can go with the
                next lockage.
            This function is evaluated in the leave_waiting_area function.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - lock: an object within the network which is generated with the IsLock mixin class
                - lock_position: a string which includes the name of the node the lock chamber is located in the network
                - series_number: an integer number which indicates which lock series is evaluated (should not be greater than
                                 the maximum number of parallel lock chambers)
                - lineup_areas: the collection of line-up areas in the node at which the line-up areas are located (the total
                                amount of line-up areas should not exceed the amount of lock series)
                - lock_queue_length: an empty list at which the resistance in total queue length (~ total waiting time) is
                                     appended."""

            # Imports the properties of the evaluated line-up area
            lineup_area = lineup_areas[series_number]

            # Assesses the total queue length within this lock series
            # - if the queue for the line-up area is empty, a name is set if the vessel fits in the lock chamber and line-up right away, otherwise the queue is calculated
            if lineup_area.length.get_queue == []:
                if (
                    vessel.L <= lock.length.level
                    and vessel.L <= lineup_area.length.level
                    and lock.node_open == vessel.route[vessel.route.index(lock_position)]
                ):
                    if "lock_information" in dir(vessel) and lock.name not in vessel.lock_information.keys():
                        vessel.lock_information[lock.name] = HasLockInformation()
                elif vessel.L <= lineup_area.length.level:
                    lock_queue_length.append(lineup_area.length.level)
                else:
                    lock_queue_length.append(lineup_area.length.capacity)

            # - else, if the vessel does not fit in the line-up area, the total length of the queued is calculated added with the full length capacity of the line-up area
            else:
                line_up_queue_length = lineup_area.length.capacity
                for queued_vessel in lineup_area.length.get_queue:
                    line_up_queue_length += queued_vessel.amount
                lock_queue_length.append(line_up_queue_length)

        def access_lineup_area(vessel, lineup_area):
            """Processes the request of vessels to access the line-up area by claiming a position (which equals the length of
                the vessel) along its jetty.
            This function is evaluated in the leave_waiting_area function

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                               assigned to the vessel as the lock series with the least expected total waiting time"""

            def create_new_lock_cycle_and_request_access(vessel, lineup_area):
                """Creates a new locking cycle (wave of vessels assigned to same lockage) by requesting the full length
                    capacity of the line-up area, assigning the vessel to this request and returning the obsolete length
                    when the request is granted.
                This function is used in the access_lineup_area function within the leave_waiting_area function

                No input required"""

                vessel.lock_information[lineup_area.name].access_lineup_length = lineup_area.length.get(lineup_area.length.capacity)
                vessel.lock_information[lineup_area.name].access_lineup_length.obj = vessel
                yield vessel.lock_information[lineup_area.name].access_lineup_length
                lineup_area.length.put(lineup_area.length.capacity - vessel.L)

            def request_access_lock_cycle(vessel, lineup_area, total_length_waiting_vessels=0):
                """Processes the request of a vessel to enter a lock cycle (wave of vessels assigned to same lockage), depending
                    on the governing conditions regarding the current situation in the line-up area.
                This function is used in the access_lineup_area function within the leave_waiting_area function

                No input required"""

                # - If the line-up area has no queue, the vessel will access the lock cycle
                if lineup_area.length.get_queue == []:
                    vessel.lock_information[lineup_area.name].access_lineup_length = lineup_area.length.get(vessel.L)
                    vessel.lock_information[lineup_area.name].access_lineup_length.obj = vessel

                # Else, if there are already preceding vessels waiting in a queue, the vessel will request access to a lock cycle
                else:
                    # Calculates the total length of vessels assigned to this lock cycle
                    for queued_vessel in reversed(lineup_area.length.get_queue):
                        total_length_waiting_vessels += queued_vessel.obj.L

                    # If the vessels does not fit in this lock cycle, it will start a new lock cycle
                    if vessel.L > lineup_area.length.capacity - total_length_waiting_vessels:
                        yield from create_new_lock_cycle_and_request_access(vessel, lineup_area)

                    # Else, if the vessel does fit in this last lock cycle, it will request a place in this cycle
                    else:
                        vessel.lock_information[lineup_area.name].access_lineup_length = lineup_area.length.get(vessel.L)
                        vessel.lock_information[lineup_area.name].access_lineup_length.obj = vessel
                        yield vessel.lock_information[lineup_area.name].access_lineup_length

            # Requesting procedure for access to line-up area
            # - If there area vessels in the line-up area
            if lineup_area.line_up_area[lineup_area.start_node].users != []:
                # - If the vessels fits in the lock cycle right away
                if vessel.L <= (
                    lineup_area.line_up_area[lineup_area.start_node].users[-1].obj.lock_information[lineup_area.name].lineup_dist
                    - 0.5 * lineup_area.line_up_area[lineup_area.start_node].users[-1].obj.L
                ):
                    yield from request_access_lock_cycle(vessel, lineup_area)

                # - Else, if the vessel does not fit in the lock cyle right away
                else:
                    if lineup_area.length.get_queue == []:
                        yield from create_new_lock_cycle_and_request_access(vessel, lineup_area)
                    else:
                        yield from request_access_lock_cycle(vessel, lineup_area)

            # - Else, if there are no vessels yet in the line-up area
            else:
                # - If the vessels fits in the lock cycle right away
                if vessel.L <= lineup_area.length.level:
                    yield from request_access_lock_cycle(vessel, lineup_area)

                # - Else, if the vessel does not fit in the lock cyle right away
                else:
                    if lineup_area.length.get_queue == []:
                        yield from create_new_lock_cycle_and_request_access(vessel, lineup_area)
                    else:
                        yield from request_access_lock_cycle(vessel, lineup_area)

        # Imports the properties of the waiting area
        for waiting_area in vessel.multidigraph.nodes[node_waiting_area]["Waiting area"]:
            if "lock_information" not in dir(vessel):
                vessel.lock_information = {}

            vessel.waiting_area_access = waiting_area.waiting_area[node_waiting_area].request()
            vessel.waiting_area_access.obj = vessel
            # Identifies the index of the node of the waiting area within the route of the vessel
            index_node_waiting_area = vessel.route.index(node_waiting_area)

            # Checks whether the waiting area is the first encountered waiting area of the lock complex

            lineup_areas = waiting_area.find_lineup_areas(vessel, index_node_waiting_area)

            # Imports the location of the lock chamber of the lock complex
            locks, directions = waiting_area.find_locks(vessel, index_node_waiting_area)

            # Determines the current time
            for lineup_area in lineup_areas:
                if waiting_area.name.split("_")[0] != lineup_area.name.split("_")[0]:
                    continue

                wait_for_lineup_area = vessel.env.now
                vessel.v_before_lock = vessel.v

                # Assigning the lock chain series with least expected waiting time to the vessel
                lock_queue_length = []
                for count, (lock, direction) in enumerate(zip(locks, directions)):
                    if waiting_area.name.split("_")[0] == lock.name.split("_")[0]:
                        if direction:
                            approach_node = lock.node_doors1
                        else:
                            approach_node = lock.node_doors2
                        if lock.used_as_one_way_traffic_regulation:
                            if "lock_information" in dir(vessel) and lock.name not in vessel.lock_information.keys():
                                vessel.lock_information[lock.name] = HasLockInformation()
                        else:
                            choose_lock_chamber(vessel, lock, approach_node, count, lineup_areas, lock_queue_length)
                        break

                # If the function did not yet assign a lock chain series
                if (
                    lineup_area.name not in vessel.lock_information.keys()
                    and "lock_information" in dir(vessel)
                    and lineup_areas[lock_queue_length.index(min(lock_queue_length))].name not in vessel.lock_information.keys()
                ):
                    vessel.lock_information[
                        lineup_areas[lock_queue_length.index(min(lock_queue_length))].name
                    ] = HasLockInformation()

                yield from access_lineup_area(vessel, lineup_area)

                # Calculation of location in line-up area as a distance in [m] from start line-up jetty
                # - If the line-up area is not empty
                if len(lineup_area.line_up_area[lineup_area.start_node].users) != 0:
                    vessel.lock_information[lineup_area.name].lineup_dist = (
                        lineup_area.line_up_area[lineup_area.start_node]
                        .users[-1]
                        .obj.lock_information[lineup_area.name]
                        .lineup_dist
                        - 0.5 * lineup_area.line_up_area[lineup_area.start_node].users[-1].obj.L
                        - 0.5 * vessel.L
                    )
                # - Else, if the line-up area is empty
                else:
                    vessel.lock_information[lineup_area.name].lineup_dist = lineup_area.length.capacity - 0.5 * vessel.L
                # Calculation of the (lat,lon)-coordinates of the assigned position in the line-up area
                if direction:
                    k = sorted(
                        lineup_area.multidigraph[lineup_area.start_node][lineup_area.end_node],
                        key=lambda x: lineup_area.multidigraph[lineup_area.start_node][lineup_area.end_node][x]["geometry"].length,
                    )[0]
                    vessel.lock_information[lineup_area.name].lineup_position = lineup_area.multidigraph.edges[
                        lineup_area.start_node, lineup_area.end_node, k
                    ]["geometry"].interpolate(
                        lock.distance_doors1_from_first_waiting_area
                        - lineup_area.distance_to_lock_doors
                        - (lineup_area.lineup_length - vessel.lock_information[lineup_area.name].lineup_dist)
                    )

                elif not direction:
                    k = sorted(
                        lineup_area.multidigraph[lineup_area.start_node][lineup_area.end_node],
                        key=lambda x: lineup_area.multidigraph[lineup_area.start_node][lineup_area.end_node][x]["geometry"].length,
                    )[0]
                    vessel.lock_information[lineup_area.name].lineup_position = lineup_area.multidigraph.edges[
                        lineup_area.start_node, lineup_area.end_node, k
                    ]["geometry"].interpolate(
                        lock.distance_doors2_from_second_waiting_area
                        - lineup_area.distance_to_lock_doors
                        - (lineup_area.lineup_length - vessel.lock_information[lineup_area.name].lineup_dist)
                    )

                # Formal request of the vessel to access the line-up area assigned to the vessel (always granted)
                vessel.lock_information[lineup_area.name].access_lineup_area = lineup_area.line_up_area[
                    lineup_area.start_node
                ].request()
                vessel.lock_information[lineup_area.name].access_lineup_area.obj = vessel
                vessel.lock_information[lineup_area.name].access_lineup_area.obj.lock_information[lineup_area.name].n = len(
                    lineup_area.line_up_area[lineup_area.start_node].users
                )

                # Request of entering the line-up area to assure that vessels will enter the line-up area one-by-one
                vessel.lock_information[lineup_area.name].enter_lineup_length = lineup_area.enter_line_up_area[
                    lineup_area.start_node
                ].request()
                vessel.lock_information[lineup_area.name].enter_lineup_length.obj = vessel
                yield vessel.lock_information[lineup_area.name].enter_lineup_length

                # Speed reduction in the approach to the line-up area
                vessel.lock_information[lineup_area.name].wait_for_next_cycle = False
                vessel.lock_information[lineup_area.name].waited_in_waiting_area = False

                # Calculates and reports the total waiting time in the waiting area
                if wait_for_lineup_area != vessel.env.now:
                    vessel.log_entry_extra(
                        wait_for_lineup_area,
                        nx.get_node_attributes(vessel.multidigraph, "geometry")[node_waiting_area],
                        "Waiting in waiting area start",
                        vessel.output.copy(),
                    )
                    vessel.log_entry_extra(
                        vessel.env.now,
                        nx.get_node_attributes(vessel.multidigraph, "geometry")[node_waiting_area],
                        "Waiting in waiting area stop",
                        vessel.output.copy(),
                    )
                    # Speed reduction in the approach to the line-up area, as the vessel had to lay still in the waiting area
                    vessel.v = lineup_area.speed_reduction_factor * lock.speed_reduction_factor * vessel.v_before_lock

                    # Changes boolean of the vessel which indicates that it had to wait in the waiting area
                    for line_up_user in lineup_area.line_up_area[lineup_area.start_node].users:
                        if line_up_user.obj.id == vessel.id:
                            line_up_user.obj.lock_information[lineup_area.name].waited_in_waiting_area = True
                            break

                vessel.update_waiting_area_status_report(waiting_area, node_waiting_area)
                waiting_area.waiting_area[node_waiting_area].release(vessel.waiting_area_access)
                waiting_area.waiting_area[node_waiting_area].release(vessel.waiting_area_access)
                break

    @staticmethod
    def approach_lineup_area(vessel, start_node, end_node):
        """Processes vessels which are approaching the line-up area of the lock complex:
            determines whether the assigned position in the line-up area (distance in [m]) should be changed as the preceding vessel(s),
            which was/were waiting in the line-up area, has/have of is/are already accessed/accessing the lock.

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
            - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network"""

        # Imports the properties of the line-up area the vessel is assigned to
        k = sorted(
            vessel.multidigraph[start_node][end_node], key=lambda x: vessel.multidigraph[start_node][end_node][x]["geometry"].length
        )[0]
        lineup_areas = vessel.multidigraph.edges[start_node, end_node, k]["Line-up area"]

        for lineup_area in lineup_areas:
            if lineup_area.name in vessel.lock_information.keys():
                break

        lock, direction = lineup_area.find_lock(vessel, start_node, end_node)
        if isinstance(lock, type(None)):
            return

        if direction:
            distance_to_lineup_area = (
                lock.distance_doors1_from_first_waiting_area - lineup_area.distance_to_lock_doors - lineup_area.lineup_length
            )
        else:
            distance_to_lineup_area = (
                lock.distance_doors2_from_second_waiting_area - lineup_area.distance_to_lock_doors - lineup_area.lineup_length
            )
        location_of_start_lineup_area = vessel.env.vessel_traffic_service.provide_location_over_edges(
            vessel, start_node, end_node, distance_to_lineup_area
        )

        if distance_to_lineup_area:
            vessel.log_entry_extra(
                vessel.env.now,
                vessel.multidigraph.nodes[start_node]["geometry"],
                "Sailing to start of line-up area start",
                vessel.output.copy(),
            )
            yield vessel.env.timeout(distance_to_lineup_area / vessel.v)
            vessel.log_entry_extra(
                vessel.env.now, location_of_start_lineup_area, "Sailing to start of line-up area stop", vessel.output.copy()
            )

        def change_lineup_dist(vessel, lock, lineup_area, lineup_dist, lineup_area_user):
            """Determines whether the assigned position in the line-up area (distance in [m]) should be changed as the preceding vessel(s),
                which was/were waiting in the line-up area, has/have of is/are already accessed/accessing the lock.
            This function is used in the approach_lineup_area function.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                               assigned to the vessel as the lock series with the least expected total waiting time
                - lock: an object within the network which is generated with the IsLock mixin class and
                        assigned to the vessel as the lock series with the least expected total waiting time
                - lineup_dist: the initial position of the vessel in the line-up area as the distance from the origin of the jetty in [m]
                - q: an integer number which represents the assigned position of the vessel in the line-up area, only the vessel which is
                     the new first in line (q=0) will be processed"""

            if lineup_area_user[0] == 0 and lineup_area_user[1].obj.lock_information[lock.name].n != (
                lineup_area_user[1].obj.lock_information[lock.name].n - len(lock.resource.users)
            ):
                lineup_dist = lineup_area.length.capacity - 0.5 * vessel.L
            return lineup_dist

        # Checks the need to change the position of the vessel within the line-up area
        for vessel_index, lineup_area_user in enumerate(lineup_area.line_up_area[start_node].users):
            if lineup_area_user.obj.id == vessel.id:
                # Imports information about the current lock cycle
                lock_door_1_user_priority = 0
                lock_door_2_user_priority = 0
                lock_door_1_users = lock.doors_1[lock.node_doors1].users
                lock_door_2_users = lock.doors_2[lock.node_doors2].users

                if direction and lock_door_2_users != []:
                    lock_door_2_user_priority = lock.doors_2[lock.node_doors2].users[0].priority

                elif not direction and lock_door_1_users != []:
                    lock_door_1_user_priority = lock.doors_1[lock.node_doors1].users[0].priority

                # Decision if position should be changed
                if direction and lock_door_2_user_priority == -1:
                    vessel.lock_information[lock.name].lineup_dist = change_lineup_dist(
                        vessel, lock, lineup_area, vessel.lock_information[lock.name].lineup_dist, (vessel_index, lineup_area_user)
                    )
                    k = sorted(
                        lineup_area.multidigraph[lineup_area.start_node][lineup_area.end_node],
                        key=lambda x: lineup_area.multidigraph[lineup_area.start_node][lineup_area.end_node][x]["geometry"].length,
                    )[0]
                    vessel.lock_information[lock.name].lineup_position = lineup_area.multidigraph.edges[
                        lineup_area.start_node, lineup_area.end_node, k
                    ]["geometry"].interpolate(
                        lock.distance_doors1_from_first_waiting_area
                        - lineup_area.distance_to_lock_doors
                        - lineup_area.lineup_length
                        + vessel.lock_information[lock.name].lineup_dist
                    )

                elif not direction and lock_door_1_user_priority == -1:
                    vessel.lock_information[lock.name].lineup_dist = change_lineup_dist(
                        vessel, lock, lineup_area, vessel.lock_information[lock.name].lineup_dist, (vessel_index, lineup_area_user)
                    )
                    k = sorted(
                        lineup_area.multidigraph[lineup_area.start_node][lineup_area.end_node],
                        key=lambda x: lineup_area.multidigraph[lineup_area.start_node][lineup_area.end_node][x]["geometry"].length,
                    )[0]
                    vessel.lock_information[lock.name].lineup_position = lineup_area.multidigraph.edges[
                        lineup_area.start_node, lineup_area.end_node, k
                    ]["geometry"].interpolate(
                        lock.distance_doors2_from_second_waiting_area
                        - lineup_area.distance_to_lock_doors
                        - lineup_area.lineup_length
                        + vessel.lock_information[lock.name].lineup_dist
                    )
                break
            break

        if vessel.v == vessel.v_before_lock:
            vessel.v = lineup_area.speed_reduction_factor * vessel.v_before_lock

        # Sail to the assigned position in the line-up area
        if lineup_area.lineup_length:
            vessel.log_entry_extra(
                vessel.env.now, location_of_start_lineup_area, "Sailing to position in line-up area start", vessel.output.copy()
            )
            yield vessel.env.timeout(vessel.lock_information[lock.name].lineup_dist / vessel.v)
            vessel.log_entry_extra(
                vessel.env.now,
                vessel.lock_information[lock.name].lineup_position,
                "Sailing to position in line-up area stop",
                vessel.output.copy(),
            )
            vessel.distance = 0

    @staticmethod
    def leave_lineup_area(vessel, start_node, end_node):
        """Processes vessels which are waiting in the line-up area of the lock complex:
            requesting access to the lock chamber given the governing phase in the lock cycle of the lock chamber and calculates the
            position within the lock chamber.

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
            - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network"""

        # Imports the properties of the line-up area the vessel is assigned to
        k_lineup = sorted(
            vessel.multidigraph[start_node][end_node], key=lambda x: vessel.multidigraph[start_node][end_node][x]["geometry"].length
        )[0]
        lineup_areas = vessel.multidigraph.edges[start_node, end_node, k_lineup]["Line-up area"]
        distance_to_lineup_areas = [lineup_area.distance_to_lock_doors for lineup_area in lineup_areas]
        lineup_areas = [x for _, x in sorted(zip(distance_to_lineup_areas, lineup_areas))]
        total_waiting_time = True
        while total_waiting_time:
            total_waiting_time = 0
            for lineup_area in lineup_areas:
                if lineup_area.name not in vessel.lock_information.keys():
                    continue

                # Determines lock
                lock, direction = lineup_area.find_lock(vessel, start_node, end_node)
                if isinstance(lock, type(None)):
                    return

                # If lock is used as a one-way traffic regulation
                waiting_required = True
                if lock.used_as_one_way_traffic_regulation:
                    waiting_required = False
                    vessel.lock_information[lock.name].distance_to_lock_doors1 = (
                        lineup_area.distance_to_lock_doors + vessel.lock_information[lock.name].lineup_dist
                    )
                    vessel.lock_information[lock.name].distance_to_lock_doors2 = (
                        vessel.lock_information[lock.name].distance_to_lock_doors1 + lock.lock_length
                    )
                    vessel.lock_information[lock.name].ETA = (
                        vessel.env.now + vessel.lock_information[lock.name].distance_to_lock_doors1 / vessel.v
                    )
                    vessel.lock_information[lock.name].ETD = (
                        vessel.env.now + vessel.lock_information[lock.name].distance_to_lock_doors2 / vessel.v
                    )

                    no_waiting_time = False
                    if lock.conditions and "Draught" in lock.conditions.keys():
                        no_waiting_time = vessel.T < lock.conditions["Draught"]

                    if not no_waiting_time:
                        waiting_time = True
                        while waiting_time:
                            waiting_time = 0
                            mask = lock.schedule.direction != direction
                            if lock.conditions:
                                for condition, value in lock.conditions.items():
                                    mask = mask & (lock.schedule[condition] >= value)

                            for loc, user in lock.schedule[mask].iterrows():
                                loc = lock.schedule[mask].index.get_loc(loc)
                                if not loc and vessel.lock_information[lock.name].ETD < user.ETA:
                                    break
                                elif loc < len(lock.schedule[mask]) - 1:
                                    if (
                                        vessel.lock_information[lock.name].ETA > user.ETD
                                        and vessel.lock_information[lock.name].ETD < lock.schedule[mask].iloc[loc + 1].ETA
                                    ):
                                        break
                                    elif vessel.lock_information[lock.name].ETA < user.ETD:
                                        waiting_time = user.ETD - vessel.lock_information[lock.name].ETA
                                        vessel.lock_information[lock.name].ETA += waiting_time
                                        vessel.lock_information[lock.name].ETD += waiting_time
                                else:
                                    if (
                                        vessel.lock_information[lock.name].ETA > user.ETD
                                        or vessel.lock_information[lock.name].ETD < user.ETA
                                    ):
                                        break
                                    else:
                                        waiting_time = user.ETD - vessel.lock_information[lock.name].ETA
                                        vessel.lock_information[lock.name].ETA += waiting_time
                                        vessel.lock_information[lock.name].ETD += waiting_time

                            if waiting_time:
                                total_waiting_time += waiting_time
                                if vessel.logbook[-1]["Message"] != "Waiting in line-up area stop":
                                    vessel.lock_information[lineup_area.name].vessel_waited_in_lineup_area = True
                                    vessel.log_entry_extra(
                                        vessel.env.now,
                                        vessel.logbook[-1]["Geometry"],
                                        "Waiting in line-up area start",
                                        vessel.output.copy(),
                                    )
                                yield vessel.env.timeout(waiting_time)

                if vessel.logbook[-1]["Message"] == "Waiting in line-up area stop":
                    vessel.logbook[-1]["Timestamp"] = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now)).to_datetime64()
                elif vessel.logbook[-1]["Message"] == "Waiting in line-up area start":
                    vessel.log_entry_extra(
                        vessel.env.now, vessel.logbook[-1]["Geometry"], "Waiting in line-up area stop", vessel.output.copy()
                    )

        for lineup_area in lineup_areas:
            if lineup_area.name not in vessel.lock_information.keys():
                continue

            # Determines lock
            lock, direction = lineup_area.find_lock(vessel, start_node, end_node)
            if isinstance(lock, type(None)):
                return

            # Vessel releases its request to enter the line-up area, made in the waiting area
            lineup_area.enter_line_up_area[start_node].release(vessel.lock_information[lock.name].enter_lineup_length)

            def access_lock_chamber(
                vessel,
                direction,
                lineup_area,
                lock,
                approach_node,
                door1,
                door2,
                node_door1,
                node_door2,
                waiting_required,
            ):
                """Processes vessels which are waiting in the line-up area of the lock complex:
                    determines the current phase within the lock cycle and adjusts the request of the vessel accordingly
                This function is used in the leave_lineup_area function.

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                                   assigned to the vessel as the lock series with the least expected total waiting time
                    - start_node: a string which includes the name of the node at which the line-up area is located in the network
                    - lock: an object within the network which is generated with the IsLock mixin class and
                            assigned to the vessel as the lock series with the least expected total waiting time
                    - approach_node: a string which includes the name of the node at which the lock chamber is located in the network
                    - door1: an object created in the IsLock class which resembles the set of lock doors which is first encountered by
                             the vessel, which should be supscripted to, using a string which includes the name of the node at which this
                             lock door is located in the network and was specified as input in the IsLock class
                    - door2: an object created in the IsLock class which resembles the set of lock doors which is last encountered by
                             the vessel, which should be supscripted to, using a string which includes the name of the node at which this
                             lock door is located in the network and was specified as input in the IsLock class"""

                def request_approach_lock_chamber(vessel, direction, lock, door1, timeout_required=True, priority=0):
                    """Vessel will request if it can enter the lock by requesting access to the first set of lock doors. This
                        request always has priority = 0, as vessels can only pass these doors when the doors are open (not
                        claimed by a priority = -1 request). The capacity of the doors equals one, to prevent ships from
                        entering simultaneously. The function yields a timeout. This can be switched off if it is assured
                        the vessel can approach immediately.

                    Input:
                        - timeout_required: a boolean which defines whether the requesting vessel receives a timeout."""

                    vessel.lock_information[lock.name].access_lock_door1 = door1.request(priority=priority)
                    vessel.lock_information[lock.name].access_lock_door1.obj = vessel

                    if timeout_required:
                        yield vessel.lock_information[lock.name].access_lock_door1

                    priority = lock.check_priority(vessel, direction)
                    lock.schedule.loc[(lock.name, vessel.name), :] = [
                        direction,
                        vessel.env.now + vessel.lock_information[lock.name].distance_to_lock_doors1 / vessel.v,
                        vessel.env.now + (vessel.lock_information[lock.name].distance_to_lock_doors1 + lock.lock_length) / vessel.v,
                        vessel.L,
                        vessel.B,
                        vessel.T,
                        vessel.type,
                        priority,
                    ]
                    lock.schedule = lock.schedule.sort_values("ETA")

                def secure_lock_cycle(vessel, direction, lock, door2, hold_request=False, timeout_required=True, priority=-1):
                    """Vessel will indicate the direction of the next lock cycle by requesting access to the second pair of lock
                        doors. Therefore, this request by default has priority = -1, which indicates the direction of the lockage
                        as requests to access the same doors by vessels on the opposite side of the lock will be queued. A timeout
                        is yielded. This can be switched off if it is assured the vessel can approach immediately. Furthermore, if
                        there was already a request by another vessel waiting in the same line-up area, this original request can be
                        holded. Lastly, the function is also used with priority = 0 in order to let vessels wait for the next lockage.

                    Input:
                        - hold_request: a boolean which defines where an earlier request for the same empty lock conversion which
                                        was made by another ship should be holded
                        - timeout_required: a boolean which defines whether the requesting vessel receives a timeout.
                        - priority: an integer [-1,0] which indicates the priority of the request: either with (-1) or without (0)
                                    priority."""

                    if "access_lock_door2" not in dir(vessel.lock_information[lock.name]):
                        vessel.lock_information[lock.name].access_lock_door2 = door2.request(priority=priority)
                        vessel.lock_information[lock.name].access_lock_door2.obj = vessel
                        if hold_request:
                            door2.release(door2.users[0])
                        if timeout_required:
                            yield vessel.lock_information[lock.name].access_lock_door2

                        priority = lock.check_priority(vessel, direction)
                        lock.schedule.loc[(lock.name, vessel.name), :] = [
                            direction,
                            vessel.env.now
                            + (vessel.lock_information[lock.name].distance_to_lock_doors2 - lock.lock_length) / vessel.v,
                            vessel.env.now + vessel.lock_information[lock.name].distance_to_lock_doors2 / vessel.v,
                            vessel.L,
                            vessel.B,
                            vessel.T,
                            vessel.type,
                            priority,
                        ]

                        lock.schedule = lock.schedule.sort_values("ETA")

                def wait_for_next_lockage(vessel, direction, lock, door2, timeout_required=True, priority=0):
                    """Vessels will wait for the next lockage by requesting access to the second pair of lock doors without priority. If
                        granted, the request will immediately be released.

                    No input required."""

                    yield from secure_lock_cycle(
                        vessel, direction, lock, door2, priority=priority, timeout_required=timeout_required
                    )
                    door2.release(vessel.lock_information[lock.name].access_lock_door2)
                    delattr(vessel.lock_information[lock.name], "access_lock_door2")

                # If the vessel was not yet assigned to the next lockage:
                vessel.lock_information[lock.name].in_lock = False

                # If vessel does not fit in the next lockage, it has to wait first
                yield vessel.lock_information[lock.name].next_lockage_length
                vessel.lock_information[lock.name].in_next_lockage = lock.in_next_lockage[node_door2].request()
                vessel.lock_information[lock.name].in_next_lockage.obj = vessel

                # Request procedure of the lock doors, which is dependent on the current moment within the lock cycle:
                # If lock is at same side:
                if approach_node == lock.node_open:
                    # Request to start the lock cycle
                    if lock.resource.users:
                        if door1.users != [] and door1.users[0].priority == -1:
                            yield from request_approach_lock_chamber(vessel, direction, lock, door1, timeout_required=True)
                            yield from secure_lock_cycle(vessel, direction, lock, door2, timeout_required=False)
                        else:
                            yield from request_approach_lock_chamber(vessel, direction, lock, door1, timeout_required=True)
                            yield from secure_lock_cycle(vessel, direction, lock, door2, hold_request=True, timeout_required=False)
                    else:
                        for user in lineup_area.line_up_area[lineup_area.start_node].users:
                            if "empty_converting" in dir(user.obj.lock_information[lock.name]):
                                if time.mktime(pd.Timestamp(lock.logbook[-1]["Timestamp"]).timetuple()) - vessel.env.now > 0:
                                    yield vessel.env.timeout(
                                        time.mktime(pd.Timestamp(lock.logbook[-1]["Timestamp"]).timetuple()) - vessel.env.now
                                    )

                        if (door2.users != [] and door2.users[0].priority == -1) or (
                            door1.queue != [] and door1.queue[0].priority == 0
                        ):
                            yield from request_approach_lock_chamber(vessel, direction, lock, door1, timeout_required=True)
                            yield from secure_lock_cycle(vessel, direction, lock, door2, hold_request=True, timeout_required=False)
                        elif door1.users != [] and door1.users[0].priority == -1:
                            yield from wait_for_next_lockage(vessel, direction, lock, door2)
                            yield from request_approach_lock_chamber(vessel, direction, lock, door1, timeout_required=True)
                            yield from secure_lock_cycle(vessel, direction, lock, door2, timeout_required=False)
                        else:
                            yield from request_approach_lock_chamber(vessel, direction, lock, door1, timeout_required=False)
                            yield from secure_lock_cycle(vessel, direction, lock, door2, timeout_required=False)

                # If lock is at other side
                else:
                    # If there are no lock users or vessels waiting at the other side of the lock to enter:
                    if not lock.in_next_lockage[node_door1].users and not lock.resource.users:
                        if door2.users and door2.users[0].priority:
                            yield from request_approach_lock_chamber(
                                vessel, direction, lock, door1, timeout_required=waiting_required
                            )
                            yield from secure_lock_cycle(
                                vessel, direction, lock, door2, hold_request=True, timeout_required=waiting_required
                            )
                        else:
                            yield from request_approach_lock_chamber(vessel, direction, lock, door1, timeout_required=False)
                            yield from secure_lock_cycle(vessel, direction, lock, door2, timeout_required=False)

                    # - Else, if no users on other side:
                    else:
                        yield from wait_for_next_lockage(vessel, direction, lock, door2)
                        if approach_node != lock.node_open:
                            yield from wait_for_next_lockage(vessel, direction, lock, door2)
                        if door2.users != [] and door2.users[0].priority == -1:
                            yield from request_approach_lock_chamber(
                                vessel, direction, lock, door1, timeout_required=waiting_required
                            )
                            yield from secure_lock_cycle(
                                vessel, direction, lock, door2, hold_request=True, timeout_required=waiting_required
                            )
                        else:
                            yield from request_approach_lock_chamber(
                                vessel, direction, lock, door1, timeout_required=waiting_required
                            )
                            yield from secure_lock_cycle(vessel, direction, lock, door2, timeout_required=False)

                if approach_node != lock.node_open:
                    if any(
                        [
                            "empty_converting" in dir(user.obj.lock_information[lock.name])
                            for user in lock.in_next_lockage[node_door2].users
                        ]
                    ):
                        yield from lock.request_empty_lock_conversion(
                            lineup_area, vessel, approach_node, hold_request=True, timeout_required=waiting_required
                        )
                    else:
                        yield from lock.request_empty_lock_conversion(
                            lineup_area, vessel, approach_node, timeout_required=waiting_required
                        )

            # Determines start time of potential waiting time
            vessel.lock_information[lock.name].distance_to_lock_doors1 = (
                lineup_area.distance_to_lock_doors + vessel.lock_information[lock.name].lineup_dist
            )
            vessel.lock_information[lock.name].distance_to_lock_doors2 = (
                vessel.lock_information[lock.name].distance_to_lock_doors1 + lock.lock_length
            )
            start_waiting_time_in_lineup_area = vessel.env.now
            priority = lock.check_priority(vessel, direction)
            lock.schedule.loc[(lock.name, vessel.name), :] = [
                direction,
                vessel.env.now + (vessel.lock_information[lock.name].distance_to_lock_doors2 - lock.lock_length) / vessel.v,
                vessel.env.now + vessel.lock_information[lock.name].distance_to_lock_doors2 / vessel.v,
                vessel.L,
                vessel.B,
                vessel.T,
                vessel.type,
                priority,
            ]

            lock.schedule = lock.schedule.sort_values("ETA")
            # Request access to lock chamber
            if direction:
                yield from access_lock_chamber(
                    vessel,
                    direction,
                    lineup_area,
                    lock,
                    lock.node_doors1,
                    lock.doors_1[lock.node_doors1],
                    lock.doors_2[lock.node_doors2],
                    lock.node_doors1,
                    lock.node_doors2,
                    waiting_required,
                )
                vessel.location_end_of_lineup_area = vessel.env.vessel_traffic_service.provide_location_over_edges(
                    vessel,
                    start_node,
                    lock.node_doors2,
                    lock.distance_doors1_from_first_waiting_area - lineup_area.distance_to_lock_doors,
                )

            if not direction:
                yield from access_lock_chamber(
                    vessel,
                    direction,
                    lineup_area,
                    lock,
                    lock.node_doors2,
                    lock.doors_2[lock.node_doors2],
                    lock.doors_1[lock.node_doors1],
                    lock.node_doors2,
                    lock.node_doors1,
                    waiting_required,
                )
                vessel.location_end_of_lineup_area = vessel.env.vessel_traffic_service.provide_location_over_edges(
                    vessel,
                    start_node,
                    lock.node_doors1,
                    lock.distance_doors2_from_second_waiting_area - lineup_area.distance_to_lock_doors,
                )

            priority = lock.check_priority(vessel, direction)
            lock.schedule.loc[(lock.name, vessel.name), :] = [
                direction,
                vessel.env.now + (vessel.lock_information[lock.name].distance_to_lock_doors2 - lock.lock_length) / vessel.v,
                vessel.env.now + vessel.lock_information[lock.name].distance_to_lock_doors2 / vessel.v,
                vessel.L,
                vessel.B,
                vessel.T,
                vessel.type,
                priority,
            ]

            if vessel.env.now != start_waiting_time_in_lineup_area:
                vessel.log_entry_extra(
                    start_waiting_time_in_lineup_area,
                    vessel.lock_information[lock.name].lineup_position,
                    "Waiting in line-up area start",
                    vessel.output.copy(),
                )
                vessel.log_entry_extra(
                    vessel.env.now,
                    vessel.lock_information[lock.name].lineup_position,
                    "Waiting in line-up area stop",
                    vessel.output.copy(),
                )

            vessel.update_lineup_area_status_report(lineup_area, lineup_area.start_node)

            # Releases the vessel's formal request to access the line-up area and releases its occupied length of the line-up area
            lineup_area.line_up_area[lineup_area.start_node].release(vessel.lock_information[lock.name].access_lineup_area)
            lineup_area.length.put(vessel.L)

            vessel.v = vessel.v_before_lock * lock.speed_reduction_factor * lineup_area.speed_reduction_factor

            # Start sailing event to end of line-up area
            waiting_area = lock.find_previous_waiting_area(vessel, direction)

            if lineup_area.lineup_length:
                vessel.log_entry_extra(
                    vessel.env.now,
                    vessel.lock_information[lock.name].lineup_position,
                    "Sailing to end of line-up area start",
                    vessel.output.copy(),
                )
                yield vessel.env.timeout(
                    np.max([0, (lineup_area.lineup_length - vessel.lock_information[lock.name].lineup_dist)]) / vessel.v
                )
                if direction:
                    distance_to_end_lineup = lock.distance_doors1_from_first_waiting_area - lineup_area.distance_to_lock_doors
                else:
                    distance_to_end_lineup = lock.distance_doors2_from_second_waiting_area - lineup_area.distance_to_lock_doors
                location_of_end_of_lineup_area = vessel.env.vessel_traffic_service.provide_location_over_edges(
                    vessel, waiting_area.node, end_node, distance_to_end_lineup
                )
                vessel.log_entry_extra(
                    vessel.env.now, location_of_end_of_lineup_area, "Sailing to end of line-up area stop", vessel.output.copy()
                )
                vessel.distance = (
                    vessel.env.vessel_traffic_service.provide_trajectory(vessel.multidigraph, end_node, waiting_area.node).length
                    - distance_to_end_lineup
                )

    @staticmethod
    def leave_lock(vessel, node_doors1, node_doors2, direction):
        """Processes vessels which are waiting in the lock chamber to be levelled and after levelling:
            checks if vessels which have entered the lock chamber have to wait for the other vessels to enter the lock chamber and
            requests conversion of the lock chamber.

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
            - node_lock: a string which includes the name of the node at which the lock chamber is located in the network"""

        # Imports the properties of the lock chamber the vessel is assigned to
        k = sorted(
            vessel.multidigraph[node_doors1][node_doors2],
            key=lambda x: vessel.multidigraph[node_doors1][node_doors2][x]["geometry"].length,
        )[0]
        locks = vessel.multidigraph.edges[node_doors1, node_doors2, k]["Lock"]
        for lock in locks:
            if lock.name not in vessel.lock_information.keys():
                continue

            previous_waiting_area = lock.find_previous_waiting_area(vessel, direction)
            lineup_area = lock.find_previous_lineup_area(vessel, direction)
            opposing_lineup_area = lock.find_next_lineup_area(vessel, direction)

            # Sail to position in the lock
            distance_from_start_edge_to_lock_doors = 0
            vessel.lock_information[lock.name].lock_dist = lock.length.level - 0.5 * vessel.L
            if direction:
                if "Line-up area" not in vessel.multidigraph.edges[node_doors1, node_doors2, k]:
                    distance_from_start_edge_to_lock_doors = (
                        lock.distance_doors1_from_first_waiting_area
                        - vessel.env.vessel_traffic_service.provide_trajectory(
                            vessel.multidigraph, previous_waiting_area.node, lock.node_doors1
                        ).length
                    )
                else:
                    distance_from_start_edge_to_lock_doors = lineup_area.distance_to_lock_doors

                k = sorted(
                    lock.multidigraph[lock.node_doors1][lock.node_doors2],
                    key=lambda x: lock.multidigraph[lock.node_doors1][lock.node_doors2][x]["geometry"].length,
                )[0]
                location_first_set_of_lock_doors = vessel.env.vessel_traffic_service.provide_location_over_edges(
                    vessel, previous_waiting_area.node, lock.node_doors2, lock.distance_doors1_from_first_waiting_area
                )
                vessel.lock_information[lock.name].lock_position = lock.multidigraph.edges[lock.node_doors1, lock.node_doors2, k][
                    "geometry"
                ].interpolate(lock.distance_doors1_from_first_waiting_area + vessel.lock_information[lock.name].lock_dist)

            if not direction:
                if "Line-up area" not in vessel.multidigraph.edges[node_doors1, node_doors2, k]:
                    distance_from_start_edge_to_lock_doors = (
                        lock.distance_doors2_from_second_waiting_area
                        - vessel.env.vessel_traffic_service.provide_trajectory(
                            vessel.multidigraph, previous_waiting_area.node, lock.node_doors2
                        ).length
                    )
                else:
                    distance_from_start_edge_to_lock_doors = lineup_area.distance_to_lock_doors
                k = sorted(
                    lock.multidigraph[lock.node_doors2][lock.node_doors1],
                    key=lambda x: lock.multidigraph[lock.node_doors2][lock.node_doors1][x]["geometry"].length,
                )[0]
                location_first_set_of_lock_doors = vessel.env.vessel_traffic_service.provide_location_over_edges(
                    vessel, previous_waiting_area.node, lock.node_doors1, lock.distance_doors2_from_second_waiting_area
                )
                vessel.lock_information[lock.name].lock_position = lock.multidigraph.edges[lock.node_doors2, lock.node_doors1, k][
                    "geometry"
                ].interpolate(lock.distance_doors2_from_second_waiting_area + vessel.lock_information[lock.name].lock_dist)

            vessel.log_entry_extra(
                vessel.env.now, vessel.logbook[-1]["Geometry"], "Sailing to first set of lock doors start", vessel.output.copy()
            )
            yield vessel.env.timeout(np.max([0, distance_from_start_edge_to_lock_doors]) / vessel.v)
            vessel.log_entry_extra(
                vessel.env.now, location_first_set_of_lock_doors, "Sailing to first set of lock doors stop", vessel.output.copy()
            )

            vessel.log_entry_extra(
                vessel.env.now, location_first_set_of_lock_doors, "Sailing to assigned location in lock start", vessel.output.copy()
            )
            yield vessel.env.timeout(np.max([0, (vessel.lock_information[lock.name].lock_dist)]) / vessel.v)
            vessel.log_entry_extra(
                vessel.env.now,
                vessel.lock_information[lock.name].lock_position,
                "Sailing to assigned location in lock stop",
                vessel.output.copy(),
            )

            if lock.mandatory_time_gap_between_entering_vessels is None:
                if direction:
                    node_doors2 = lock.node_doors2
                    yield lock.doors_1[lock.node_doors1].release(vessel.lock_information[lock.name].access_lock_door1)
                    vessel.distance = vessel.lock_information[lock.name].lock_dist + lock.distance_doors2_from_second_waiting_area

                elif not direction:
                    node_doors2 = lock.node_doors1
                    yield lock.doors_2[lock.node_doors2].release(vessel.lock_information[lock.name].access_lock_door1)
                    vessel.distance = vessel.lock_information[lock.name].lock_dist + lock.distance_doors1_from_first_waiting_area

            # Determines current time and reports this to vessel's log as start time of lock passage
            vessel.log_entry_extra(
                vessel.env.now, vessel.lock_information[lock.name].lock_position, "Passing lock start", vessel.output.copy()
            )

            # Request access to pass the next line-up area after the lock chamber has levelled, so that vessels will leave the lock chamber one-by-one
            if not opposing_lineup_area.passing_allowed:
                vessel.lock_information[lock.name].departure_lock = opposing_lineup_area.pass_line_up_area[
                    opposing_lineup_area.start_node
                ].request(priority=-1)
                vessel.lock_information[lock.name].departure_lock.obj = vessel

            # Formal request access to lock chamber and calculate position within the lock chamber
            for user in lock.in_next_lockage[node_doors2].users:
                if user.obj.id != vessel.id:
                    yield lock.length.put(lock.length.capacity - lock.length.level)
                else:
                    break

            lock.length.get(vessel.L)
            vessel.lock_information[lock.name].access_lock = lock.resource.request()
            vessel.lock_information[lock.name].access_lock.obj = vessel
            user_found = False
            remaining_lock_capacity = lock.length.level
            vessels_not_in_lock = []
            for user in lock.next_lockage[node_doors2].users:
                if user_found:
                    if user.obj.L < remaining_lock_capacity:
                        vessels_not_in_lock.append(user.obj)
                        remaining_lock_capacity -= user.obj.L
                        yield lock.length.get(lock.length.capacity)
                        lock.length.put(lock.length.capacity)
                        lock.length.get(vessel.L)
                        break
                    else:
                        break
                if user.obj.id == vessel.id:
                    user_found = True
            lock.in_next_lockage[node_doors2].release(vessel.lock_information[lock.name].in_next_lockage)

            if vessels_not_in_lock and not lock.used_as_one_way_traffic_regulation:
                vessel.lock_information[lock.name].in_lock = True
                yield lock.length.get(lock.length.capacity)
                lock.length.put(lock.length.capacity)
                for lock_user in lock.resource.users:
                    if lock_user.obj.id != vessel.id:
                        continue
                    lock_user.obj.lock_information[lock.name].converting = True
                    if not opposing_lineup_area.passing_allowed:
                        yield vessel.lock_information[lock.name].departure_lock
                if lock.resource.users[-1].obj.id == vessel.id:
                    lock.length.put(lock.length.capacity - lock.length.level)

            # Determines if the vessel explicitly has to request the conversion of the lock chamber (only the last entered vessel) or can go with a previously made request
            else:
                vessel.lock_information[lock.name].in_lock = True
                vessel.lock_information[lock.name].converting = True
                number_of_vessels = len(lock.resource.users)
                if lock.next_lockage_length[node_doors2].level and not lock.next_lockage_length[node_doors2].get_queue:
                    lock.next_lockage_length[node_doors2].get(lock.next_lockage_length[node_doors2].level)
                vessel.update_lock_status_report(lock)
                yield from lock.convert_chamber(vessel.env, node_doors2, number_of_vessels, vessel)
                old_level = lock.length.level
                yield lock.length.put(lock.length.capacity - lock.length.level)
                lock.length.get(lock.length.capacity - old_level)
                if not opposing_lineup_area.passing_allowed:
                    yield vessel.lock_information[lock.name].departure_lock

            # Calculates and reports the total locking time
            vessel.log_entry_extra(
                vessel.env.now, vessel.lock_information[lock.name].lock_position, "Passing lock stop", vessel.output.copy()
            )
            vessel.lock_information[lock.name].lineup_position = vessel.multidigraph.nodes[opposing_lineup_area.end_node][
                "geometry"
            ]

            # Sails to the lock doors
            distance_to_second_pair_of_lock_doors = lock.lock_length - vessel.lock_information[lock.name].lock_dist
            if direction:
                k = sorted(
                    lock.multidigraph[lock.node_doors1][lock.node_doors2],
                    key=lambda x: lock.multidigraph[lock.node_doors1][lock.node_doors2][x]["geometry"].length,
                )[0]
                vessel.lock_information[lock.name].position_second_pair_of_lock_doors = lock.multidigraph.edges[
                    lock.node_doors1, lock.node_doors2, k
                ]["geometry"].interpolate(lock.distance_doors2_from_second_waiting_area + lock.lock_length)

            if not direction:
                k = sorted(
                    lock.multidigraph[lock.node_doors2][lock.node_doors1],
                    key=lambda x: lock.multidigraph[lock.node_doors2][lock.node_doors1][x]["geometry"].length,
                )[0]
                vessel.lock_information[lock.name].position_second_pair_of_lock_doors = lock.multidigraph.edges[
                    lock.node_doors2, lock.node_doors1, k
                ]["geometry"].interpolate(lock.distance_doors1_from_first_waiting_area + lock.lock_length)

            vessel.log_entry_extra(
                vessel.env.now,
                vessel.lock_information[lock.name].lock_position,
                "Sailing to second set of lock doors start",
                vessel.output.copy(),
            )
            yield vessel.env.timeout(np.max([0, (distance_to_second_pair_of_lock_doors)]) / vessel.v)
            vessel.log_entry_extra(
                vessel.env.now,
                vessel.lock_information[lock.name].position_second_pair_of_lock_doors,
                "Sailing to second set of lock doors stop",
                vessel.output.copy(),
            )

            vessel.v = vessel.v_before_lock
            if opposing_lineup_area.passing_allowed:
                if direction and lock.doors_2[lock.node_doors2].users[0].obj.id == vessel.id:
                    lock.doors_2[lock.node_doors2].release(vessel.lock_information[lock.name].access_lock_door2)

                if not direction and lock.doors_1[lock.node_doors1].users[0].obj.id == vessel.id:
                    lock.doors_1[lock.node_doors1].release(vessel.lock_information[lock.name].access_lock_door2)

                lock.resource.release(vessel.lock_information[lock.name].access_lock)
                departure_lock_length = lock.length.put(vessel.L)  # put length back in lock
                yield departure_lock_length

            # Releases the vessel's formal request of the lock chamber and returns its occupied length in the lock chamber
            lock.resource.release(vessel.lock_information[lock.name].access_lock)
            lock.length.put(vessel.L)

            if (
                not lock.resource.users
                and lock.next_lockage_length[node_doors2].capacity - lock.next_lockage_length[node_doors2].level
            ):
                lock.next_lockage_length[node_doors2].put(
                    lock.next_lockage_length[node_doors2].capacity - lock.next_lockage_length[node_doors2].level
                )

            lock.next_lockage[node_doors2].release(vessel.lock_information[lock.name].next_lockage)

            if opposing_lineup_area.start_node == node_doors2 and not opposing_lineup_area.passing_allowed:
                yield from PassLock.leave_opposite_lineup_area(vessel, node_doors2, node_doors1, direction)

        if direction:
            vessel.distance = vessel.env.vessel_traffic_service.provide_distance_to_node(
                vessel, lock.node_doors1, lock.node_doors2, vessel.logbook[-1]["Geometry"]
            )

        if not direction:
            vessel.distance = vessel.env.vessel_traffic_service.provide_distance_to_node(
                vessel, lock.node_doors2, lock.node_doors1, vessel.logbook[-1]["Geometry"]
            )

    @staticmethod
    def leave_opposite_lineup_area(vessel, start_node, end_node, direction):
        """Processes vessels which have left the lock chamber after levelling and are now in the next line-up area in order to leave the lock complex through the next waiting area:
            release of their requests for accessing their second encountered line-up area and lock chamber.

        Input:
            - vessel: an identity which is Identifiable, Movable,and Routable, and has VesselProperties
            - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network"""

        # Imports the properties of the line-up area the vessel is assigned to
        k = sorted(
            vessel.multidigraph[start_node][end_node], key=lambda x: vessel.multidigraph[start_node][end_node][x]["geometry"].length
        )[0]
        lineup_areas = vessel.multidigraph.edges[start_node, end_node, k]["Line-up area"]

        for lineup_area in lineup_areas:
            if lineup_area.name not in vessel.lock_information.keys():
                continue

            lock, direction = lineup_area.find_lock(vessel, start_node, end_node, direction=direction)

            if direction:
                distance_from_lock_doors = lock.distance_doors2_from_second_waiting_area
            else:
                distance_from_lock_doors = lock.distance_doors1_from_first_waiting_area
            distance_to_lineup_end = lineup_area.distance_to_lock_doors
            location_end_lineup_area = vessel.env.vessel_traffic_service.provide_location_over_edges(
                vessel, start_node, end_node, distance_from_lock_doors - distance_to_lineup_end
            )
            distance_to_lineup_start = distance_to_lineup_end + lineup_area.lineup_length
            location_start_lineup_area = vessel.env.vessel_traffic_service.provide_location_over_edges(
                vessel, start_node, end_node, distance_from_lock_doors - distance_to_lineup_start
            )

            vessel.lock_information[lock.name].in_lock = False
            if distance_to_lineup_end:
                vessel.log_entry_extra(
                    vessel.env.now, vessel.logbook[-1]["Geometry"], "Sailing to line-up area start", vessel.output.copy()
                )
                yield vessel.env.timeout(np.max([0, distance_to_lineup_end]) / vessel.v)
                vessel.log_entry_extra(
                    vessel.env.now, location_end_lineup_area, "Sailing to line-up area stop", vessel.output.copy()
                )

            vessel.log_entry_extra(vessel.env.now, location_end_lineup_area, "Passing line-up area start", vessel.output.copy())
            yield vessel.env.timeout(lineup_area.lineup_length / vessel.v)
            vessel.log_entry_extra(vessel.env.now, location_start_lineup_area, "Passing line-up area stop", vessel.output.copy())
            vessel.distance = vessel.env.vessel_traffic_service.provide_distance_to_node(
                vessel, start_node, end_node, vessel.logbook[-1]["Geometry"]
            )

            if not lineup_area.passing_allowed:
                # Releases the vessel's request of their second encountered set of lock doors
                if (
                    direction
                    and lock.doors_2[lock.node_doors2].users
                    and lock.doors_2[lock.node_doors2].users[0].obj.id == vessel.id
                ):
                    lock.doors_2[lock.node_doors2].release(vessel.lock_information[lock.name].access_lock_door2)
                if (
                    not direction
                    and lock.doors_1[lock.node_doors1].users
                    and lock.doors_1[lock.node_doors1].users[0].obj.id == vessel.id
                ):
                    lock.doors_1[lock.node_doors1].release(vessel.lock_information[lock.name].access_lock_door2)
                # Releases the vessel's request to enter the second line-up area
                lineup_area.pass_line_up_area[lineup_area.start_node].release(vessel.lock_information[lock.name].departure_lock)
