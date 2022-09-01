import random
import math

import numpy as np
import simpy
import shapely
import networkx as nx
import pyproj

#
from opentnsim.core import HasResource, Identifiable, Log, HasLength, SimpyObject
import opentnsim.core


class IsLockWaitingArea(HasResource, Identifiable, Log):
    """Mixin class: Something has lock object properties

    - properties in meters
    - operation in seconds
    """

    def __init__(self, node, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization
        """

        waiting_area_resources = 100
        self.waiting_area = {
            node: simpy.PriorityResource(self.env, capacity=waiting_area_resources),
        }

        # departure_resources = 4
        # self.departure = {
        #    node: simpy.PriorityResource(self.env, capacity=departure_resources),
        # }


class IsLockLineUpArea(HasResource, HasLength, Identifiable, Log):
    """Mixin class: Something has lock object properties
    - properties in meters
    - operation in seconds
    """

    def __init__(self, node, lineup_length, *args, **kwargs):
        super().__init__(length=lineup_length, remaining_length=lineup_length, *args, **kwargs)
        """Initialization"""

        self.lock_queue_length = 0

        # Lay-Out
        self.enter_line_up_area = {
            node: simpy.PriorityResource(self.env, capacity=1),
        }

        self.line_up_area = {
            node: simpy.PriorityResource(self.env, capacity=100),
        }

        self.converting_while_in_line_up_area = {
            node: simpy.PriorityResource(self.env, capacity=1),
        }

        self.pass_line_up_area = {
            node: simpy.PriorityResource(self.env, capacity=1),
        }


class HasLockDoors(SimpyObject):
    def __init__(self, node_1, node_3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization
        """

        self.doors_1 = {
            node_1: simpy.PriorityResource(self.env, capacity=1),
        }
        self.doors_2 = {
            node_3: simpy.PriorityResource(self.env, capacity=1),
        }


class IsLock(HasResource, HasLength, HasLockDoors, Identifiable, Log):
    """Mixin class: Something has lock object properties
    - properties in meters
    - operation in seconds
    """

    def __init__(
        self,
        node_1,
        node_2,
        node_3,
        lock_length,
        lock_width,
        lock_depth,
        doors_open,
        doors_close,
        wlev_dif,
        disch_coeff,
        grav_acc,
        opening_area,
        opening_depth,
        simulation_start,
        operating_time,
        *args,
        **kwargs,
    ):

        """Initialization"""

        # Properties
        self.lock_length = lock_length
        self.lock_width = lock_width
        self.lock_depth = lock_depth
        self.wlev_dif = wlev_dif
        self.disch_coeff = disch_coeff
        self.grav_acc = grav_acc
        self.opening_area = opening_area
        self.opening_depth = opening_depth
        self.simulation_start = simulation_start.timestamp()
        self.operating_time = operating_time

        # Operating
        self.doors_open = doors_open
        self.doors_close = doors_close

        # Water level
        assert node_1 != node_3

        self.node_1 = node_1
        self.node_3 = node_3
        self.water_level = random.choice([node_1, node_3])

        super().__init__(
            length=lock_length,
            remaining_length=lock_length,
            node_1=node_1,
            node_3=node_3,
            *args,
            **kwargs,
        )

    def operation_time(self, environment):
        if type(self.wlev_dif) == list:
            operating_time = (
                2
                * self.lock_width
                * self.lock_length
                * abs(self.wlev_dif[1][np.abs(self.wlev_dif[0] - (environment.now - self.simulation_start)).argmin()])
            ) / (self.disch_coeff * self.opening_area * math.sqrt(2 * self.grav_acc * self.opening_depth))

        elif type(self.wlev_dif) == float or type(self.wlev_dif) == int:
            operating_time = (2 * self.lock_width * self.lock_length * abs(self.wlev_dif)) / (
                self.disch_coeff * self.opening_area * math.sqrt(2 * self.grav_acc * self.opening_depth)
            )
        assert not isinstance(operating_time, complex), f"operating_time number should not be complex: {operating_time}"

        return operating_time

    def convert_chamber(self, environment, new_level, number_of_vessels):
        """Convert the water level"""

        # Close the doors
        self.log_entry(
            "Lock doors closing start",
            environment.now,
            number_of_vessels,
            self.water_level,
        )
        yield environment.timeout(self.doors_close)
        self.log_entry(
            "Lock doors closing stop",
            environment.now,
            number_of_vessels,
            self.water_level,
        )

        # Convert the chamber
        self.log_entry(
            "Lock chamber converting start",
            environment.now,
            number_of_vessels,
            self.water_level,
        )

        # Water level will shift
        self.change_water_level(new_level)
        yield environment.timeout(self.operation_time(environment))
        self.log_entry(
            "Lock chamber converting stop",
            environment.now,
            number_of_vessels,
            self.water_level,
        )
        # Open the doors
        self.log_entry(
            "Lock doors opening start",
            environment.now,
            number_of_vessels,
            self.water_level,
        )
        yield environment.timeout(self.doors_open)
        self.log_entry(
            "Lock doors opening stop",
            environment.now,
            number_of_vessels,
            self.water_level,
        )

    def change_water_level(self, side):
        """Change water level and priorities in queue"""

        self.water_level = side

        for request in self.resource.queue:
            request.priority = -1 if request.priority == 0 else 0

            if request.priority == -1:
                self.resource.queue.insert(0, self.resource.queue.pop(self.resource.queue.index(request)))
            else:
                self.resource.queue.insert(-1, self.resource.queue.pop(self.resource.queue.index(request)))


####
####
class CanPassLock(opentnsim.core.Movable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization
        """
        self.on_pass_edge_functions.append(self.pass_edge)

        self.lineup_pos_lat = None
        self.lineup_pos_lon = None

    def get_node_idx(self, node):
        node_idx = self.route.index(node)
        return node_idx

    def pass_edge(self, origin, destination):
        speed = self.v
        node_idx = self.route.index(destination)

        departure_lock = None

        if "Lock" in self.env.FG.nodes[origin].keys():
            # TODO: Shapely expects coordinates in x,y order
            # https://shapely.readthedocs.io/en/stable/manual.html#Point
            orig = shapely.geometry.Point(self.lock_pos_lat, self.lock_pos_lon)

        if "Lock" in self.env.FG.nodes[destination].keys():
            dest = shapely.geometry.Point(self.lock_pos_lat, self.lock_pos_lon)

        if "Line-up area" in self.env.FG.nodes[origin].keys():
            orig = shapely.geometry.Point(self.lineup_pos_lat, self.lineup_pos_lon)

        if "Line-up area" in self.env.FG.nodes[destination].keys():
            dest = shapely.geometry.Point(self.lineup_pos_lat, self.lineup_pos_lon)

        if "Waiting area" in self.env.FG.nodes[destination].keys():
            locks = self.env.FG.nodes[destination]["Waiting area"]
            for lock in locks:
                for r in self.route[node_idx:]:
                    if "Line-up area" in self.env.FG.nodes[r].keys():
                        wait_for_waiting_area = self.env.now
                        access_waiting_area = lock.waiting_area[destination].request()
                        yield access_waiting_area

                        if wait_for_waiting_area != self.env.now:
                            waiting = self.env.now - wait_for_waiting_area
                            self.log_entry(
                                "Waiting to enter waiting area start",
                                wait_for_waiting_area,
                                0,
                                nx.get_node_attributes(self.env.FG, "geometry")[origin],
                            )
                            self.log_entry(
                                "Waiting to enter waiting area stop",
                                self.env.now,
                                waiting,
                                nx.get_node_attributes(self.env.FG, "geometry")[origin],
                            )

        if "Waiting area" in self.env.FG.nodes[origin].keys():
            locks = self.env.FG.nodes[origin]["Waiting area"]
            for lock in locks:
                for r in self.route[node_idx]:
                    if "Line-up area" in self.env.FG.nodes[r].keys():
                        locks2 = self.env.FG.nodes[r]["Line-up area"]
                        for r2 in self.route[node_idx:]:
                            if "Lock" in self.env.FG.nodes[r2].keys():
                                locks3 = self.env.FG.nodes[r2]["Lock"]
                                break

                        self.lock_name = []
                        for lock3 in locks3:
                            if lock3.water_level == self.route[self.route.index(r2) - 1]:
                                for lock2 in locks2:
                                    if lock2.name == lock3.name:
                                        if lock2.lock_queue_length == 0:
                                            self.lock_name = lock3.name
                                    break

                        lock_queue_length = []
                        if self.lock_name == []:
                            for lock2 in locks2:
                                lock_queue_length.append(lock2.lock_queue_length)

                            self.lock_name = locks2[lock_queue_length.index(min(lock_queue_length))].name

                        for lock2 in locks2:
                            if lock2.name == self.lock_name:
                                lock2.lock_queue_length += 1

                        for lock2 in locks2:
                            if lock2.name == self.lock_name:
                                self.v = 0.5 * speed
                                break

                        wait_for_lineup_area = self.env.now
                        lock.waiting_area[origin].release(access_waiting_area)

                        if self.route[self.route.index(r2) - 1] == lock3.node_1:
                            if lock3.doors_2[lock3.node_3].users != [] and lock3.doors_2[lock3.node_3].users[0].priority == -1:
                                if self.L < lock2.length.level + lock3.length.level:
                                    access_lineup_length = lock2.length.get(self.L)
                                elif self.L < lock2.length.level:
                                    if lock2.length.level == lock2.length.capacity:
                                        access_lineup_length = lock2.length.get(self.L)
                                    elif (
                                        lock2.line_up_area[r].users != []
                                        and lock3.length.level < lock2.line_up_area[r].users[0].length
                                    ):
                                        access_lineup_length = lock2.length.get(self.L)
                                    else:
                                        if lock2.length.get_queue == []:
                                            access_lineup_length = lock2.length.get(lock2.length.capacity)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(lock2.length.capacity - self.L)
                                        else:
                                            total_length_waiting_vessels = 0
                                            for q in reversed(range(len(lock2.length.get_queue))):
                                                if lock2.length.get_queue[q].amount == lock2.length.capacity:
                                                    break
                                            for q2 in range(q, len(lock2.length.get_queue)):
                                                total_length_waiting_vessels += lock2.length.get_queue[q2].length

                                            if self.L > lock2.length.capacity - total_length_waiting_vessels:
                                                access_lineup_length = lock2.length.get(lock2.length.capacity)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length
                                                correct_lineup_length = lock2.length.put(lock2.length.capacity - self.L)
                                            else:
                                                access_lineup_length = lock2.length.get(self.L)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length
                                else:
                                    if lock2.length.get_queue == []:
                                        access_lineup_length = lock2.length.get(lock2.length.capacity)
                                        lock2.length.get_queue[-1].length = self.L
                                        yield access_lineup_length
                                        correct_lineup_length = lock2.length.put(lock2.length.capacity - self.L)
                                    else:
                                        total_length_waiting_vessels = 0
                                        for q in reversed(range(len(lock2.length.get_queue))):
                                            if lock2.length.get_queue[q].amount == lock2.length.capacity:
                                                break
                                        for q2 in range(q, len(lock2.length.get_queue)):
                                            total_length_waiting_vessels += lock2.length.get_queue[q2].length

                                        if self.L > lock2.length.capacity - total_length_waiting_vessels:
                                            access_lineup_length = lock2.length.get(lock2.length.capacity)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(lock2.length.capacity - self.L)
                                        else:
                                            access_lineup_length = lock2.length.get(self.L)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length

                            else:
                                if lock2.length.level == lock2.length.capacity:
                                    access_lineup_length = lock2.length.get(self.L)
                                elif (
                                    lock2.line_up_area[r].users != []
                                    and self.L
                                    < lock2.line_up_area[r].users[-1].lineup_dist - 0.5 * lock2.line_up_area[r].users[-1].length
                                ):
                                    access_lineup_length = lock2.length.get(self.L)
                                else:
                                    if lock2.length.get_queue == []:
                                        access_lineup_length = lock2.length.get(lock2.length.capacity)
                                        lock2.length.get_queue[-1].length = self.L
                                        yield access_lineup_length
                                        correct_lineup_length = lock2.length.put(lock2.length.capacity - self.L)
                                    else:
                                        total_length_waiting_vessels = 0
                                        for q in reversed(range(len(lock2.length.get_queue))):
                                            if lock2.length.get_queue[q].amount == lock2.length.capacity:
                                                break
                                        for q2 in range(q, len(lock2.length.get_queue)):
                                            total_length_waiting_vessels += lock2.length.get_queue[q2].length

                                        if self.L > lock2.length.capacity - total_length_waiting_vessels:
                                            access_lineup_length = lock2.length.get(lock2.length.capacity)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(lock2.length.capacity - self.L)
                                        else:
                                            access_lineup_length = lock2.length.get(self.L)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length

                        elif self.route[self.route.index(r2) - 1] == lock3.node_3:
                            if lock3.doors_1[lock3.node_1].users != [] and lock3.doors_1[lock3.node_1].users[0].priority == -1:
                                if self.L < lock2.length.level + lock3.length.level:
                                    access_lineup_length = lock2.length.get(self.L)
                                elif self.L < lock2.length.level:
                                    if lock2.length.level == lock2.length.capacity:
                                        access_lineup_length = lock2.length.get(self.L)
                                    elif (
                                        lock2.line_up_area[r].users != []
                                        and lock3.length.level < lock2.line_up_area[r].users[0].length
                                    ):
                                        access_lineup_length = lock2.length.get(self.L)
                                    else:
                                        if lock2.length.get_queue == []:
                                            access_lineup_length = lock2.length.get(lock2.length.capacity)
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(lock2.length.capacity - self.L)
                                            yield correct_lineup_length
                                        else:
                                            total_length_waiting_vessels = 0
                                            for q in reversed(range(len(lock2.length.get_queue))):
                                                if lock2.length.get_queue[q].amount == lock2.length.capacity:
                                                    break
                                            for q2 in range(q, len(lock2.length.get_queue)):
                                                total_length_waiting_vessels += lock2.length.get_queue[q2].length

                                            if self.L > lock2.length.capacity - total_length_waiting_vessels:
                                                access_lineup_length = lock2.length.get(lock2.length.capacity)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length
                                                correct_lineup_length = lock2.length.put(lock2.length.capacity - self.L)
                                            else:
                                                access_lineup_length = lock2.length.get(self.L)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length
                                else:
                                    if lock2.length.get_queue == []:
                                        access_lineup_length = lock2.length.get(lock2.length.capacity)
                                        lock2.length.get_queue[-1].length = self.L
                                        yield access_lineup_length
                                        correct_lineup_length = lock2.length.put(lock2.length.capacity - self.L)
                                    else:
                                        total_length_waiting_vessels = 0
                                        for q in reversed(range(len(lock2.length.get_queue))):
                                            if lock2.length.get_queue[q].amount == lock2.length.capacity:
                                                break
                                        for q2 in range(q, len(lock2.length.get_queue)):
                                            total_length_waiting_vessels += lock2.length.get_queue[q2].length

                                        if self.L > lock2.length.capacity - total_length_waiting_vessels:
                                            access_lineup_length = lock2.length.get(lock2.length.capacity)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(lock2.length.capacity - self.L)
                                        else:
                                            access_lineup_length = lock2.length.get(self.L)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                            else:
                                if lock2.length.level == lock2.length.capacity:
                                    access_lineup_length = lock2.length.get(self.L)
                                elif (
                                    lock2.line_up_area[r].users != []
                                    and self.L
                                    < lock2.line_up_area[r].users[-1].lineup_dist - 0.5 * lock2.line_up_area[r].users[-1].length
                                ):
                                    access_lineup_length = lock2.length.get(self.L)
                                else:
                                    if lock2.length.get_queue == []:
                                        access_lineup_length = lock2.length.get(lock2.length.capacity)
                                        lock2.length.get_queue[-1].length = self.L
                                        yield access_lineup_length
                                        correct_lineup_length = lock2.length.put(lock2.length.capacity - self.L)
                                    else:
                                        total_length_waiting_vessels = 0
                                        for q in reversed(range(len(lock2.length.get_queue))):
                                            if lock2.length.get_queue[q].amount == lock2.length.capacity:
                                                break
                                        for q2 in range(q, len(lock2.length.get_queue)):
                                            total_length_waiting_vessels += lock2.length.get_queue[q2].length

                                        if self.L > lock2.length.capacity - total_length_waiting_vessels:
                                            access_lineup_length = lock2.length.get(lock2.length.capacity)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(lock2.length.capacity - self.L)
                                        else:
                                            access_lineup_length = lock2.length.get(self.L)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length

                        if len(lock2.line_up_area[r].users) != 0:
                            self.lineup_dist = (
                                lock2.line_up_area[r].users[-1].lineup_dist
                                - 0.5 * lock2.line_up_area[r].users[-1].length
                                - 0.5 * self.L
                            )
                        else:
                            self.lineup_dist = lock2.length.capacity - 0.5 * self.L

                        self.wgs84 = pyproj.Geod(ellps="WGS84")
                        [lineup_area_start_lat, lineup_area_start_lon, lineup_area_stop_lat, lineup_area_stop_lon] = [
                            self.env.FG.nodes[self.route[self.route.index(r)]]["geometry"].x,
                            self.env.FG.nodes[self.route[self.route.index(r)]]["geometry"].y,
                            self.env.FG.nodes[self.route[self.route.index(r) + 1]]["geometry"].x,
                            self.env.FG.nodes[self.route[self.route.index(r) + 1]]["geometry"].y,
                        ]
                        fwd_azimuth, _, _ = self.wgs84.inv(
                            lineup_area_start_lat,
                            lineup_area_start_lon,
                            lineup_area_stop_lat,
                            lineup_area_stop_lon,
                        )
                        [self.lineup_pos_lat, self.lineup_pos_lon, _] = self.wgs84.fwd(
                            self.env.FG.nodes[self.route[self.route.index(r)]]["geometry"].x,
                            self.env.FG.nodes[self.route[self.route.index(r)]]["geometry"].y,
                            fwd_azimuth,
                            self.lineup_dist,
                        )

                        access_lineup_area = lock2.line_up_area[r].request()
                        lock2.line_up_area[r].users[-1].length = self.L
                        lock2.line_up_area[r].users[-1].id = self.id
                        lock2.line_up_area[r].users[-1].lineup_pos_lat = self.lineup_pos_lat
                        lock2.line_up_area[r].users[-1].lineup_pos_lon = self.lineup_pos_lon
                        lock2.line_up_area[r].users[-1].lineup_dist = self.lineup_dist
                        lock2.line_up_area[r].users[-1].n = len(lock2.line_up_area[r].users)
                        lock2.line_up_area[r].users[-1].v = 0.25 * speed
                        lock2.line_up_area[r].users[-1].wait_for_next_cycle = False
                        yield access_lineup_area

                        enter_lineup_length = lock2.enter_line_up_area[r].request()
                        yield enter_lineup_length
                        lock2.enter_line_up_area[r].users[0].id = self.id

                        if wait_for_lineup_area != self.env.now:
                            self.v = 0.25 * speed
                            waiting = self.env.now - wait_for_lineup_area
                            self.log_entry(
                                "Waiting in waiting area start",
                                wait_for_lineup_area,
                                0,
                                nx.get_node_attributes(self.env.FG, "geometry")[origin],
                            )
                            self.log_entry(
                                "Waiting in waiting area stop",
                                self.env.now,
                                waiting,
                                nx.get_node_attributes(self.env.FG, "geometry")[origin],
                            )
                        break

        if "Line-up area" in self.env.FG.nodes[destination].keys():
            locks = self.env.FG.nodes[destination]["Line-up area"]
            for lock in locks:
                if lock.name == self.lock_name:
                    orig = shapely.geometry.Point(self.lineup_pos_lat, self.lineup_pos_lon)
                    for r in self.route[node_idx:]:
                        if "Lock" in self.env.FG.nodes[r].keys():
                            locks = self.env.FG.nodes[r]["Lock"]
                            for lock2 in locks:
                                for q in range(len(lock.line_up_area[destination].users)):
                                    if lock.line_up_area[destination].users[q].id == self.id:
                                        if self.route[self.route.index(r) - 1] == lock2.node_1:
                                            if (
                                                lock2.doors_2[lock2.node_3].users != []
                                                and lock2.doors_2[lock2.node_3].users[0].priority == -1
                                            ):
                                                if q <= 1 and lock.line_up_area[destination].users[q].n != lock.line_up_area[
                                                    destination
                                                ].users[q].n - len(lock2.resource.users):
                                                    self.lineup_dist = lock.length.capacity - 0.5 * self.L
                                        elif self.route[self.route.index(r) - 1] == lock2.node_3:
                                            if (
                                                lock2.doors_1[lock2.node_1].users != []
                                                and lock2.doors_1[lock2.node_1].users[0].priority == -1
                                            ):
                                                if q <= 1 and lock.line_up_area[destination].users[q].n != lock.line_up_area[
                                                    destination
                                                ].users[q].n - len(lock2.resource.users):
                                                    self.lineup_dist = lock.length.capacity - 0.5 * self.L
                                        [self.lineup_pos_lat, self.lineup_pos_lon, _] = self.wgs84.fwd(
                                            self.env.FG.nodes[self.route[self.route.index(destination)]]["geometry"].x,
                                            self.env.FG.nodes[self.route[self.route.index(destination)]]["geometry"].y,
                                            fwd_azimuth,
                                            self.lineup_dist,
                                        )
                                        lock.line_up_area[destination].users[q].lineup_pos_lat = self.lineup_pos_lat
                                        lock.line_up_area[destination].users[q].lineup_pos_lon = self.lineup_pos_lon
                                        lock.line_up_area[destination].users[q].lineup_dist = self.lineup_dist
                                        break

        if "Line-up area" in self.env.FG.nodes[origin].keys():
            locks = self.env.FG.nodes[origin]["Line-up area"]
            for lock in locks:
                if lock.name == self.lock_name:
                    orig = shapely.geometry.Point(self.lineup_pos_lat, self.lineup_pos_lon)
                    for r in self.route[node_idx:]:
                        if "Lock" in self.env.FG.nodes[r].keys():
                            locks = self.env.FG.nodes[r]["Lock"]
                            lock.enter_line_up_area[origin].release(enter_lineup_length)
                            for q in range(len(lock.line_up_area[origin].users)):
                                if lock.line_up_area[origin].users[q].id == self.id:
                                    if q > 0:
                                        _, _, distance = self.wgs84.inv(
                                            orig.x,
                                            orig.y,
                                            lock.line_up_area[origin].users[0].lineup_pos_lat,
                                            lock.line_up_area[origin].users[0].lineup_pos_lon,
                                        )
                                        yield self.env.timeout(distance / self.v)
                                        break

                            for lock2 in locks:
                                if lock2.name == self.lock_name:
                                    self.v = 0.25 * speed
                                    wait_for_lock_entry = self.env.now

                                    for r2 in self.route[(node_idx + 1) :]:
                                        if "Line-up area" in self.env.FG.nodes[r2].keys():
                                            locks = self.env.FG.nodes[r2]["Line-up area"]
                                            for lock3 in locks:
                                                if lock3.name == self.lock_name:
                                                    break
                                            break

                                    if self.route[self.route.index(r) - 1] == lock2.node_1:
                                        if len(lock2.doors_2[lock2.node_3].users) != 0:
                                            if lock2.doors_2[lock2.node_3].users[0].priority == -1:
                                                if (
                                                    self.L
                                                    > (lock2.resource.users[-1].lock_dist - 0.5 * lock2.resource.users[-1].length)
                                                    or lock2.resource.users[-1].converting
                                                ):
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority=-1)
                                                    yield access_lock_door2
                                                    lock2.doors_2[lock2.node_3].release(access_lock_door2)

                                                    wait_for_next_cycle = lock3.pass_line_up_area[r2].request()
                                                    yield wait_for_next_cycle
                                                    lock3.pass_line_up_area[r2].release(wait_for_next_cycle)

                                                if lock.converting_while_in_line_up_area[origin].users != []:
                                                    waiting_during_converting = lock.converting_while_in_line_up_area[
                                                        origin
                                                    ].request(priority=-1)
                                                    yield waiting_during_converting
                                                    lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)

                                                elif (
                                                    len(lock2.doors_1[lock2.node_1].users) == 0
                                                    or (
                                                        len(lock2.doors_1[lock2.node_1].users) != 0
                                                        and lock2.doors_1[lock2.node_1].users[0].priority != -1
                                                    )
                                                ) and self.route[self.route.index(r) - 1] != lock2.water_level:
                                                    waiting_during_converting = lock.converting_while_in_line_up_area[
                                                        origin
                                                    ].request()
                                                    yield waiting_during_converting
                                                    yield from lock2.convert_chamber(
                                                        self.env,
                                                        self.route[self.route.index(r) - 1],
                                                        0,
                                                    )
                                                    lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)

                                                access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                yield access_lock_door1

                                                if (
                                                    lock2.doors_2[lock2.node_3].users != []
                                                    and lock2.doors_2[lock2.node_3].users[0].priority == -1
                                                ):
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority=-1)
                                                    lock2.doors_2[lock2.node_3].release(lock2.doors_2[lock2.node_3].users[0])
                                                    yield access_lock_door2
                                                    lock2.doors_2[lock2.node_3].users[0].id = self.id
                                                else:
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority=-1)
                                                    yield access_lock_door2
                                                    lock2.doors_2[lock2.node_3].users[0].id = self.id

                                            else:
                                                if lock3.converting_while_in_line_up_area[r2].users != []:
                                                    waiting_during_converting = lock3.converting_while_in_line_up_area[r2].request()
                                                    yield waiting_during_converting
                                                    lock3.converting_while_in_line_up_area[r2].release(waiting_during_converting)

                                                access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                yield access_lock_door1

                                                if (
                                                    lock2.doors_2[lock2.node_3].users != []
                                                    and lock2.doors_2[lock2.node_3].users[0].priority == -1
                                                ):
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority=-1)
                                                    lock2.doors_2[lock2.node_3].release(lock2.doors_2[lock2.node_3].users[0])
                                                    yield access_lock_door2
                                                    lock2.doors_2[lock2.node_3].users[0].id = self.id
                                                else:
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority=-1)
                                                    yield access_lock_door2
                                                    lock2.doors_2[lock2.node_3].users[0].id = self.id
                                        else:
                                            if (
                                                lock2.doors_2[lock2.node_3].users != []
                                                and lock2.doors_2[lock2.node_3].users[0].priority == -1
                                            ):
                                                access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                yield access_lock_door1
                                                access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority=-1)
                                                lock2.doors_2[lock2.node_3].release(lock2.doors_2[lock2.node_3].users[0])
                                                yield access_lock_door2
                                                lock2.doors_2[lock2.node_3].users[0].id = self.id

                                            elif (
                                                lock2.doors_2[lock2.node_3].users != []
                                                and lock2.doors_2[lock2.node_3].users[0].priority == 0
                                            ):
                                                access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                yield access_lock_door1
                                                access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority=-1)
                                                yield access_lock_door2
                                                lock2.doors_2[lock2.node_3].users[0].id = self.id

                                            else:
                                                if lock.converting_while_in_line_up_area[origin].users != []:
                                                    waiting_during_converting = lock.converting_while_in_line_up_area[
                                                        origin
                                                    ].request(priority=-1)
                                                    yield waiting_during_converting
                                                    lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                    yield access_lock_door1

                                                elif (
                                                    len(lock2.doors_1[lock2.node_1].users) == 0
                                                    or (
                                                        len(lock2.doors_1[lock2.node_1].users) != 0
                                                        and lock2.doors_1[lock2.node_1].users[0].priority != -1
                                                    )
                                                ) and self.route[self.route.index(r) - 1] != lock2.water_level:
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                    waiting_during_converting = lock.converting_while_in_line_up_area[
                                                        origin
                                                    ].request()
                                                    yield waiting_during_converting
                                                    yield from lock2.convert_chamber(
                                                        self.env,
                                                        self.route[self.route.index(r) - 1],
                                                        0,
                                                    )
                                                    lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)

                                                elif (
                                                    len(lock2.doors_1[lock2.node_1].users) != 0
                                                    and lock2.doors_1[lock2.node_1].users[0].priority == -1
                                                ):
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                    yield access_lock_door1

                                                else:
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request()

                                                if (
                                                    lock2.doors_2[lock2.node_3].users != []
                                                    and lock2.doors_2[lock2.node_3].users[0].priority == -1
                                                ):
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority=-1)
                                                    lock2.doors_2[lock2.node_3].release(lock2.doors_2[lock2.node_3].users[0])
                                                    yield access_lock_door2
                                                    lock2.doors_2[lock2.node_3].users[0].id = self.id
                                                else:
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority=-1)
                                                    yield access_lock_door2
                                                    lock2.doors_2[lock2.node_3].users[0].id = self.id

                                    elif self.route[self.route.index(r) - 1] == lock2.node_3:
                                        if len(lock2.doors_1[lock2.node_1].users) != 0:
                                            if lock2.doors_1[lock2.node_1].users[0].priority == -1:
                                                if (
                                                    self.L
                                                    > (lock2.resource.users[-1].lock_dist - 0.5 * lock2.resource.users[-1].length)
                                                    or lock2.resource.users[-1].converting
                                                ):
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority=-1)
                                                    yield access_lock_door1
                                                    lock2.doors_1[lock2.node_1].release(access_lock_door1)

                                                    wait_for_next_cycle = lock3.pass_line_up_area[r2].request()
                                                    yield wait_for_next_cycle
                                                    lock3.pass_line_up_area[r2].release(wait_for_next_cycle)

                                                if lock.converting_while_in_line_up_area[origin].users != []:
                                                    waiting_during_converting = lock.converting_while_in_line_up_area[
                                                        origin
                                                    ].request(priority=-1)
                                                    yield waiting_during_converting
                                                    lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)

                                                elif (
                                                    len(lock2.doors_2[lock2.node_3].users) == 0
                                                    or (
                                                        len(lock2.doors_2[lock2.node_3].users) != 0
                                                        and lock2.doors_2[lock2.node_3].users[0].priority != -1
                                                    )
                                                ) and self.route[self.route.index(r) - 1] != lock2.water_level:
                                                    waiting_during_converting = lock.converting_while_in_line_up_area[
                                                        origin
                                                    ].request()
                                                    yield waiting_during_converting
                                                    yield from lock2.convert_chamber(
                                                        self.env,
                                                        self.route[self.route.index(r) - 1],
                                                        0,
                                                    )
                                                    lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)

                                                access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                yield access_lock_door2

                                                if (
                                                    lock2.doors_1[lock2.node_1].users != []
                                                    and lock2.doors_1[lock2.node_1].users[0].priority == -1
                                                ):
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority=-1)
                                                    lock2.doors_1[lock2.node_1].release(lock2.doors_1[lock2.node_1].users[0])
                                                    yield access_lock_door1
                                                    lock2.doors_1[lock2.node_1].users[0].id = self.id
                                                else:
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority=-1)
                                                    yield access_lock_door1
                                                    lock2.doors_1[lock2.node_1].users[0].id = self.id

                                            else:
                                                if lock3.converting_while_in_line_up_area[r2].users != []:
                                                    waiting_during_converting = lock3.converting_while_in_line_up_area[r2].request()
                                                    yield waiting_during_converting
                                                    lock3.converting_while_in_line_up_area[r2].release(waiting_during_converting)

                                                access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                yield access_lock_door2

                                                if (
                                                    lock2.doors_1[lock2.node_1].users != []
                                                    and lock2.doors_1[lock2.node_1].users[0].priority == -1
                                                ):
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority=-1)
                                                    lock2.doors_1[lock2.node_1].release(lock2.doors_1[lock2.node_1].users[0])
                                                    yield access_lock_door1
                                                    lock2.doors_1[lock2.node_1].users[0].id = self.id
                                                else:
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority=-1)
                                                    yield access_lock_door1
                                                    lock2.doors_1[lock2.node_1].users[0].id = self.id
                                        else:
                                            if (
                                                lock2.doors_1[lock2.node_1].users != []
                                                and lock2.doors_1[lock2.node_1].users[0].priority == -1
                                            ):
                                                access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                yield access_lock_door2
                                                access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority=-1)
                                                lock2.doors_1[lock2.node_1].release(lock2.doors_1[lock2.node_1].users[0])
                                                yield access_lock_door1
                                                lock2.doors_1[lock2.node_1].users[0].id = self.id

                                            elif (
                                                lock2.doors_1[lock2.node_1].users != []
                                                and lock2.doors_1[lock2.node_1].users[0].priority == 0
                                            ):
                                                access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                yield access_lock_door2
                                                access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority=-1)
                                                yield access_lock_door1
                                                lock2.doors_1[lock2.node_1].users[0].id = self.id

                                            else:
                                                if lock.converting_while_in_line_up_area[origin].users != []:
                                                    waiting_during_converting = lock.converting_while_in_line_up_area[
                                                        origin
                                                    ].request(priority=-1)
                                                    yield waiting_during_converting
                                                    lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                    yield access_lock_door2

                                                elif (
                                                    len(lock2.doors_2[lock2.node_3].users) == 0
                                                    or (
                                                        len(lock2.doors_2[lock2.node_3].users) != 0
                                                        and lock2.doors_2[lock2.node_3].users[0].priority != -1
                                                    )
                                                ) and self.route[self.route.index(r) - 1] != lock2.water_level:
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                    waiting_during_converting = lock.converting_while_in_line_up_area[
                                                        origin
                                                    ].request()
                                                    yield waiting_during_converting
                                                    yield from lock2.convert_chamber(
                                                        self.env,
                                                        self.route[self.route.index(r) - 1],
                                                        0,
                                                    )
                                                    lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)

                                                elif (
                                                    len(lock2.doors_2[lock2.node_3].users) != 0
                                                    and lock2.doors_2[lock2.node_3].users[0].priority == -1
                                                ):
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                    yield access_lock_door2

                                                else:
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request()

                                                if (
                                                    lock2.doors_1[lock2.node_1].users != []
                                                    and lock2.doors_1[lock2.node_1].users[0].priority == -1
                                                ):
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority=-1)
                                                    lock2.doors_1[lock2.node_1].release(lock2.doors_1[lock2.node_1].users[0])
                                                    yield access_lock_door1
                                                    lock2.doors_1[lock2.node_1].users[0].id = self.id
                                                else:
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority=-1)
                                                    yield access_lock_door1
                                                    lock2.doors_1[lock2.node_1].users[0].id = self.id

                                    access_lock_length = lock2.length.get(self.L)
                                    access_lock = lock2.resource.request()

                                    access_lock_pos_length = lock2.pos_length.get(self.L)
                                    self.lock_dist = lock2.pos_length.level + 0.5 * self.L
                                    yield access_lock_pos_length

                                    lock2.resource.users[-1].id = self.id
                                    lock2.resource.users[-1].length = self.L
                                    lock2.resource.users[-1].lock_dist = self.lock_dist
                                    lock2.resource.users[-1].converting = False
                                    if self.route[self.route.index(r) - 1] == lock2.node_1:
                                        lock2.resource.users[-1].dir = 1.0
                                    else:
                                        lock2.resource.users[-1].dir = 2.0

                                    if wait_for_lock_entry != self.env.now:
                                        waiting = self.env.now - wait_for_lock_entry
                                        self.log_entry(
                                            "Waiting in line-up area start",
                                            wait_for_lock_entry,
                                            0,
                                            orig,
                                        )
                                        self.log_entry(
                                            "Waiting in line-up area stop",
                                            self.env.now,
                                            waiting,
                                            orig,
                                        )

                                    self.wgs84 = pyproj.Geod(ellps="WGS84")
                                    [doors_origin_lat, doors_origin_lon, doors_destination_lat, doors_destination_lon] = [
                                        self.env.FG.nodes[self.route[self.route.index(r) - 1]]["geometry"].x,
                                        self.env.FG.nodes[self.route[self.route.index(r) - 1]]["geometry"].y,
                                        self.env.FG.nodes[self.route[self.route.index(r) + 1]]["geometry"].x,
                                        self.env.FG.nodes[self.route[self.route.index(r) + 1]]["geometry"].y,
                                    ]
                                    fwd_azimuth, _, distance = self.wgs84.inv(
                                        doors_origin_lat,
                                        doors_origin_lon,
                                        doors_destination_lat,
                                        doors_destination_lon,
                                    )
                                    [self.lock_pos_lat, self.lock_pos_lon, _] = self.wgs84.fwd(
                                        self.env.FG.nodes[self.route[self.route.index(r) - 1]]["geometry"].x,
                                        self.env.FG.nodes[self.route[self.route.index(r) - 1]]["geometry"].y,
                                        fwd_azimuth,
                                        self.lock_dist,
                                    )

                                    for r4 in reversed(self.route[: (node_idx - 1)]):
                                        if "Line-up area" in self.env.FG.nodes[r4].keys():
                                            locks = self.env.FG.nodes[r4]["Line-up area"]
                                            for lock4 in locks:
                                                if lock4.name == self.lock_name:
                                                    lock4.lock_queue_length -= 1
                            break

                        elif "Waiting area" in self.env.FG.nodes[r].keys():
                            for r2 in reversed(self.route[: (node_idx - 1)]):
                                if "Lock" in self.env.FG.nodes[r2].keys():
                                    locks = self.env.FG.nodes[r2]["Lock"]
                                    for lock2 in locks:
                                        if lock2.name == self.lock_name:
                                            if (
                                                self.route[self.route.index(r2) + 1] == lock2.node_3
                                                and len(lock2.doors_2[lock2.node_3].users) != 0
                                                and lock2.doors_2[lock2.node_3].users[0].id == self.id
                                            ):
                                                lock2.doors_2[lock2.node_3].release(access_lock_door2)
                                            elif (
                                                self.route[self.route.index(r2) + 1] == lock2.node_1
                                                and len(lock2.doors_1[lock2.node_1].users) != 0
                                                and lock2.doors_1[lock2.node_1].users[0].id == self.id
                                            ):
                                                lock2.doors_1[lock2.node_1].release(access_lock_door1)

                                            if departure_lock is not None:
                                                lock.pass_line_up_area[origin].release(departure_lock)
                                            lock2.resource.release(access_lock)
                                            departure_lock_length = lock2.length.put(self.L)
                                            departure_lock_pos_length = lock2.pos_length.put(self.L)
                                            yield departure_lock_length
                                            yield departure_lock_pos_length
                                    break

        node_idx = self.get_node_idx(origin)
        if "Line-up area" in self.env.FG.nodes[self.route[node_idx - 1]].keys():
            locks = self.env.FG.nodes[self.route[node_idx - 1]]["Line-up area"]
            for lock in locks:
                if lock.name == self.lock_name:
                    for r in self.route[node_idx:]:
                        if "Lock" in self.env.FG.nodes[r].keys():
                            locks = self.env.FG.nodes[r]["Lock"]
                            lock.line_up_area[self.route[node_idx - 1]].release(access_lineup_area)
                            departure_lineup_length = lock.length.put(self.L)
                            yield departure_lineup_length

        if "Lock" in self.env.FG.nodes[origin].keys():
            locks = self.env.FG.nodes[origin]["Lock"]
            for lock in locks:
                if lock.name == self.lock_name:
                    if self.route[node_idx - 1] == lock.node_1:
                        lock.doors_1[lock.node_1].release(access_lock_door1)
                    elif self.route[node_idx - 1] == lock.node_3:
                        lock.doors_2[lock.node_3].release(access_lock_door2)
                    orig = shapely.geometry.Point(self.lock_pos_lat, self.lock_pos_lon)
                    for r2 in reversed(self.route[node_idx:]):
                        if "Line-up area" in self.env.FG.nodes[r2].keys():
                            locks = self.env.FG.nodes[r2]["Line-up area"]
                            for lock3 in locks:
                                if lock3.name == self.lock_name:
                                    departure_lock = lock3.pass_line_up_area[r2].request(priority=-1)
                                    break
                            break

                    for r in reversed(self.route[: (node_idx - 1)]):
                        if "Line-up area" in self.env.FG.nodes[r].keys():
                            locks = self.env.FG.nodes[r]["Line-up area"]
                            for lock2 in locks:
                                if lock2.name == self.lock_name:
                                    for q2 in range(0, len(lock.resource.users)):
                                        if lock.resource.users[q2].id == self.id:
                                            break

                                    start_time_in_lock = self.env.now
                                    self.log_entry("Passing lock start", self.env.now, 0, orig)

                                    if (
                                        len(lock2.line_up_area[r].users) != 0
                                        and lock2.line_up_area[r].users[0].length < lock.length.level
                                    ):
                                        if self.route[node_idx - 1] == lock.node_1:
                                            access_line_up_area = lock2.enter_line_up_area[r].request()
                                            yield access_line_up_area
                                            lock2.enter_line_up_area[r].release(access_line_up_area)
                                            access_lock_door1 = lock.doors_1[lock.node_1].request()
                                            yield access_lock_door1
                                            lock.doors_1[lock.node_1].release(access_lock_door1)

                                        elif self.route[node_idx - 1] == lock.node_3:
                                            access_line_up_area = lock2.enter_line_up_area[r].request()
                                            yield access_line_up_area
                                            lock2.enter_line_up_area[r].release(access_line_up_area)
                                            access_lock_door2 = lock.doors_2[lock.node_3].request()
                                            yield access_lock_door2
                                            lock.doors_2[lock.node_3].release(access_lock_door2)

                                    if lock.resource.users[0].id == self.id:
                                        lock.resource.users[0].converting = True
                                        number_of_vessels = len(lock.resource.users)
                                        yield from lock.convert_chamber(self.env, destination, number_of_vessels)
                                    else:
                                        for u in range(len(lock.resource.users)):
                                            if lock.resource.users[u].id == self.id:
                                                lock.resource.users[u].converting = True
                                                yield self.env.timeout(
                                                    lock.doors_close + lock.operation_time(self.env) + lock.doors_open
                                                )
                                                break

                    yield departure_lock

                    self.log_entry(
                        "Passing lock stop",
                        self.env.now,
                        self.env.now - start_time_in_lock,
                        orig,
                    )
                    [self.lineup_pos_lat, self.lineup_pos_lon] = [
                        self.env.FG.nodes[self.route[self.route.index(r2)]]["geometry"].x,
                        self.env.FG.nodes[self.route[self.route.index(r2)]]["geometry"].y,
                    ]
                    yield from self.pass_edge(origin, destination)
                    self.v = speed
