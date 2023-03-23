# package(s) related to the simulation
import networkx as nx

from opentnsim import core

# spatial libraries
import pyproj
import shapely.geometry
import simpy
import random
import numpy as np
import networkx as nx
import math
import bisect
import datetime
import time as timepy

class HasLock(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.register_vessel)
        self.on_pass_edge.append(self.leave_lock_chamber)

    def register_vessel(self,origin):

        if 'Detector' in self.env.FG.nodes[origin].keys():
            yield self.env.timeout(0)
            lock = []
            for node1,node2 in zip(self.route[self.route.index(origin):-1],self.route[self.route.index(origin)+1:]):
                if "Lock" in self.env.FG.edges[node1, node2].keys():
                    lock = self.env.FG.edges[node1, node2]['Lock'][0]
                    node_doors1 = lock.node_doors1
                    node_doors2 = lock.node_doors2
                    doors2 = lock.doors_2[node_doors2]

                elif "Lock" in self.env.FG.edges[node2, node1].keys():
                    lock = self.env.FG.edges[node2, node1]['Lock'][0]
                    node_doors1 = lock.node_doors2
                    node_doors2 = lock.node_doors1
                    doors2 = lock.doors_1[node_doors2]


            if lock and lock.next_lockage_length[node_doors2].level - self.L >= 0:
                if not lock.next_lockage_order[node_doors2].users:
                    self.first_in_next_lockage = lock.next_lockage_order[node_doors2].request()
                    lock.next_lockage_length[node_doors2].get(self.L)
                    self.lock_dist = lock.next_lockage_length[node_doors2].level + 0.5 * self.L  # (distance from first set of doors in [m])
                    self.first_in_next_lockage.obj = self
                    if not lock.next_lockage_order[node_doors1].users:
                        self.access_lock_door2 = doors2.request(priority=-1)
                        self.access_lock_door2.obj = self
                        if lock.node_open != node_doors1:
                            yield from lock.convert_chamber(self.env, node_doors1, 0, self, False)

                elif ('converting' in dir(lock.next_lockage_order[node_doors2].users[-1].obj) and not lock.next_lockage_order[node_doors2].users[-1].obj.converting) and not lock.next_lockage_order[node_doors1].users:
                    self.in_next_lockage = lock.next_lockage_order[node_doors2].request()
                    lock.next_lockage_length[node_doors2].get(self.L)
                    self.lock_dist = lock.next_lockage_length[node_doors2].level + 0.5 * self.L  # (distance from first set of doors in [m])
                    self.in_next_lockage.obj = self

    def leave_lock_chamber(self,origin,destination):
        if "Lock" in self.env.FG.edges[origin,destination].keys():
            yield from PassLock.leave_lock(self, origin, destination, direction=1)
            self.v = self.v_before_lock

        elif "Lock" in self.env.FG.edges[destination,origin].keys():
            yield from PassLock.leave_lock(self, destination, origin, direction=0)
            self.v = self.v_before_lock

class HasWaitingArea(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.leave_waiting_area)

    def leave_waiting_area(self, origin):
        if "Waiting area" in self.env.FG.nodes[origin].keys():  # if vessel is in waiting area
            yield from PassLock.leave_waiting_area(self, origin)

class HasLineUpArea(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_look_ahead_to_node.append(self.approach_lineup_area)
        self.on_pass_edge.append(self.leave_lineup_area)

    def approach_lineup_area(self,destination):
        if destination != self.route[-1]:
            next_node = self.route[self.route.index(destination)+1]
            if "Line-up area" in self.env.FG.edges[destination,next_node].keys():  # if vessel is approaching the line-up area
                yield from PassLock.approach_lineup_area(self, destination, next_node)
            elif "Line-up area" in self.env.FG.edges[next_node,destination].keys():
                yield from PassLock.approach_lineup_area(self, next_node, destination)

    def leave_lineup_area(self,origin,destination):
        if "Line-up area" in self.env.FG.edges[origin,destination].keys():  # if vessel is located in the line-up
            yield from PassLock.leave_lineup_area(self, origin, destination)
        elif "Line-up area" in self.env.FG.edges[destination, origin].keys():
            yield from PassLock.leave_lineup_area(self, destination, origin)

class IsLockWaitingArea(core.HasResource, core.Identifiable, core.Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
        self,
        node, #a string which indicates the location of the start of the waiting area
        *args,
        **kwargs):

        self.node = node
        super().__init__(*args, **kwargs)
        """Initialization"""

        waiting_area_resources = 100
        self.waiting_area = {node: simpy.PriorityResource(self.env, capacity=waiting_area_resources),}

class IsLockLineUpArea(core.HasResource, core.HasLength, core.Identifiable, core.Log):
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
        lineup_length, #a float which contains the length of the line-up area
        *args,
        **kwargs):

        self.start_node = start_node
        self.end_node = end_node
        super().__init__(length = lineup_length, remaining_length = lineup_length, *args, **kwargs)

        """Initialization"""
        # Lay-Out
        self.enter_line_up_area = {start_node: simpy.PriorityResource(self.env, capacity=1),} #used to regulate one by one entering of line-up area, so capacity must be 1
        self.line_up_area = {start_node: simpy.PriorityResource(self.env, capacity=100),} #line-up area itself, infinite capacity, as this is regulated by the HasLength, so capacity = inf
        self.converting_while_in_line_up_area = {start_node: simpy.PriorityResource(self.env, capacity=1),} #used to minimize the number of empty convertion requests by one by multiple waiting vessels, so capacity must be 1
        self.pass_line_up_area = {start_node: simpy.PriorityResource(self.env, capacity=1),} #used to prevent vessel from entering the lock before all previously locked vessels have passed the line-up area one by one, so capacity must be 1

class IsLock(core.HasResource, core.HasLength, core.Identifiable, core.Log):
    """Mixin class: Something which has lock chamber object properties as part of a lock complex [in SI-units] """

    def __init__(
        self,
        capacity,
        node_doors1, #a string which indicates the location of the first pair of lock doors
        node_doors2, #a string which indicates the location of the second pair of lock doors
        lock_length, #a float which contains the length of the lock chamber
        lock_width, #a float which contains the width of the lock chamber
        lock_depth, #a float which contains the depth of the lock chamber
        doors_open, #a float which contains the time it takes to open the doors
        doors_close, #a float which contains the time it takes to close the doors
        disch_coeff, #a float which contains the discharge coefficient of filling system
        opening_area, #a float which contains the cross-sectional area of filling system
        opening_depth, #a float which contains the depth at which filling system is located
        simulation_start, #a datetime which contains the simulation start time
        detector_nodes,
        levelling = 'Calculated', #possibility to set a constant levelling_time (change to float)
        wlev_dif = 'Calculated', #possibility to set a constant water level difference (change to float)
        grav_acc = 9.81, #a float which contains the gravitational acceleration
        *args,
        **kwargs):

        """Initialization"""
        # Properties
        self.lock_length = lock_length
        self.lock_width = lock_width
        self.lock_depth = lock_depth
        self.disch_coeff = disch_coeff
        self.grav_acc = grav_acc
        self.opening_area = opening_area
        self.opening_depth = opening_depth
        self.levelling = levelling
        self.wlev_dif = wlev_dif
        self.simulation_start = simulation_start.timestamp()

        super().__init__(capacity = capacity, length = lock_length, remaining_length = lock_length, *args, **kwargs)
        self.next_lockage_length = {node_doors1: simpy.Container(self.env, capacity = lock_length, init=lock_length),
                                    node_doors2: simpy.Container(self.env, capacity = lock_length, init=lock_length),}
        self.next_lockage_order = {node_doors1: simpy.PriorityResource(self.env, capacity=100),
                                   node_doors2: simpy.PriorityResource(self.env, capacity=100),}
        self.doors_1 = {node_doors1: simpy.PriorityResource(self.env, capacity = 1),} #Only one ship can pass at a time: capacity = 1, request can have priority
        self.doors_2 = {node_doors2: simpy.PriorityResource(self.env, capacity = 1),} #Only one ship can pass at a time: capacity = 1, request can have priority

        # Operating
        self.doors_open = doors_open
        self.doors_close = doors_close

        # Water level
        assert node_doors1 != node_doors2

        self.node_doors1 = node_doors1
        self.node_doors2 = node_doors2
        self.node_open = random.choice([node_doors1, node_doors2])
        index_node_open = list(self.env.FG.nodes).index(self.node_open)
        self.water_level = self.env.vessel_traffic_service.hydrodynamic_information['Water level'][index_node_open].copy()
        self.salinity = self.env.vessel_traffic_service.hydrodynamic_information['Water level'][index_node_open].copy()
        self.discharge_res = np.zeros(len(self.water_level))
        self.discharge_saline = np.zeros(len(self.water_level))
        self.discharge_fresh = np.zeros(len(self.water_level))

        for detector_node in detector_nodes:
            if 'Detector' not in self.env.FG.nodes[detector_node]:
                self.env.FG.nodes[detector_node]['Detector'] = {}

            route1 = nx.dijkstra_path(self.env.FG,detector_node,self.node_doors1)
            route2 = nx.dijkstra_path(self.env.FG,detector_node,self.node_doors2)
            for route in [route1,route2]:
                if [self.node_doors1,self.node_doors2] == [route[-2],route[-1]] or [self.node_doors1,self.node_doors2] == [route[-1],route[-2]]:
                    self.env.FG.nodes[detector_node]['Detector'][route[-1]] = core.IsDetectorNode(self)
                    break

    def exchange_flux_time_series_calculator(self,T_door_open,time_index):
        index_node_doors1 = list(self.env.FG.nodes).index(self.node_doors1)
        index_node_doors2 = list(self.env.FG.nodes).index(self.node_doors2)
        index_node_open = list(self.env.FG.nodes).index(self.node_open)
        S_lock = self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_open].values[time_index]
        S_lock_harbour = self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_open].values[time_index]
        S_lock_average = (self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_doors1].values[time_index] +
                          self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_doors2].values[time_index]) / 2
        Wlev_lock = self.env.vessel_traffic_service.hydrodynamic_information['Water level'][index_node_open].values[time_index]
        V_lock = self.lock_length * self.lock_width * (Wlev_lock + self.lock_depth)
        v_exch = (0.5 * np.sqrt(self.grav_acc * (0.8 * abs(S_lock_harbour - S_lock) / (1000 + 0.8 * S_lock_average)) * (Wlev_lock + self.lock_depth)))
        if v_exch != 0:
            T_LE = (2 * self.lock_length) / v_exch
        else:
            T_LE = 0
        time = np.arange(0, T_door_open, self.env.vessel_traffic_service.hydrodynamic_information['Times'].values[1]-self.env.vessel_traffic_service.hydrodynamic_information['Times'].values[0])
        Q = []
        V_tot = 0
        salinity = list(self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_open].values)
        for t in enumerate(time):
            if t[0] == 0:
                continue
            delta_t = (t[1] - time[t[0] - 1])
            if T_LE != 0:
                delta_V = V_lock * (np.tanh(t[1] / T_LE) - np.tanh(time[t[0] - 1] / T_LE))
            else:
                delta_V = 0
            V_tot += delta_V
            V_tot += delta_V
            Q.append(delta_V / delta_t)
            M = (S_lock_harbour - salinity[time_index]) * delta_V
            S_lock = (S_lock * V_lock + M) / V_lock
            salinity[time_index+t[0]] = S_lock
            self.discharge_saline[time_index+t[0]] += delta_V / delta_t
            self.discharge_fresh[time_index+t[0]] += -delta_V / delta_t
        self.salinity = salinity
        return

    def levelling_to_harbour(self,V_ship,levelling_time,time_index,side):
        index_node_doors1 = list(self.env.FG.nodes).index(self.node_doors1)
        index_node_doors2 = list(self.env.FG.nodes).index(self.node_doors2)
        index_node_open = list(self.env.FG.nodes).index(self.node_open)
        time_index_stop = bisect.bisect_right(self.env.vessel_traffic_service.hydrodynamic_information['Times'].values, self.env.now+levelling_time)-1
        time = np.arange(time_index,time_index_stop+1,1)
        S_lock_start = self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_open].values[time_index]
        WLev_lock_inner = self.env.vessel_traffic_service.hydrodynamic_information['Water level'][index_node_doors2].values[time_index]
        V_lock_inner = self.lock_length * self.lock_width * (WLev_lock_inner + self.lock_depth)
        WLev_lock_outer = self.env.vessel_traffic_service.hydrodynamic_information['Water level'][index_node_doors1].values[time_index]
        V_lock_outer = self.lock_length * self.lock_width * (WLev_lock_outer + self.lock_depth)
        if side == self.node_doors1:
            WLev_to_side = WLev_lock_inner
            WLev_from_side = WLev_lock_outer
            V_to_side = V_lock_inner
            V_from_side = V_lock_outer
            if WLev_from_side < WLev_to_side:
                filling = True
                S_to_side = self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_doors2].values[time_index]
            else:
                filling = False
        else:
            WLev_to_side = WLev_lock_outer
            WLev_from_side = WLev_lock_inner
            V_to_side = V_lock_outer
            V_from_side = V_lock_inner
            if WLev_from_side < WLev_to_side:
                filling = True
                S_to_side = self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_doors1].values[time_index]
            else:
                filling = False

        if filling:
            V_levelling = self.lock_length * self.lock_width * (WLev_to_side - WLev_from_side)
            S_lock_final = (S_lock_start * (V_from_side - V_ship) + V_levelling * S_to_side) / (V_to_side - V_ship)
            dt = self.env.vessel_traffic_service.hydrodynamic_information['Times'].values[1] - self.env.vessel_traffic_service.hydrodynamic_information['Times'].values[0]
            salinity = list(self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_open].values)
            for t in time:
                S_lock = (S_lock_start * (V_from_side - V_ship) + abs(sum(self.discharge_res[time[0]:t])) * dt * S_to_side) / ((V_from_side+abs(sum(self.discharge_res[time[0]:t])) * dt) - V_ship)
                salinity[t] = S_lock
            self.salinity = salinity
            V_loss_lev = 0

        else:
            V_levelling = self.lock_length * self.lock_width * (WLev_from_side - WLev_to_side)
            S_lock_final = S_lock_start
            V_loss_lev = V_levelling

        salinity = list(self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_open].values)
        for index in enumerate(salinity[time_index_stop:]):
            salinity[time_index_stop+index[0]] = S_lock_final
        self.salinity = salinity

        return V_levelling, S_lock_final, V_loss_lev

    def sailing_out_to_harbour(self, vessel, time_index):
        index_node_open = list(self.env.FG.nodes).index(self.node_open)
        V_ship = vessel.L*vessel.B*vessel.T_f
        S_lock = self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_open].values[time_index]
        WLev_lock_harbour = self.env.vessel_traffic_service.hydrodynamic_information['Water level'][index_node_open].values[time_index]
        S_lock_harbour = self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_open].values[time_index]
        V_lock_harbour = self.lock_length * self.lock_width * (WLev_lock_harbour + self.lock_depth)
        S_lock = (S_lock * (V_lock_harbour - V_ship) + V_ship * S_lock_harbour) / V_lock_harbour
        start_distance = self.lock_length - vessel.lock_dist - 0.5 * vessel.L
        start_time_passing_door = start_distance/vessel.v
        time_index = bisect.bisect_right(self.env.vessel_traffic_service.hydrodynamic_information['Times'].values, self.env.now+start_time_passing_door)-1
        end_time_passing_door = vessel.L/vessel.v + start_time_passing_door
        passing_time_door = np.arange(start_time_passing_door,end_time_passing_door,self.env.vessel_traffic_service.hydrodynamic_information['Times'].values[1]-self.env.vessel_traffic_service.hydrodynamic_information['Times'].values[0])
        if self.node_open == self.node_doors1:
            for t in enumerate(passing_time_door):
                if time_index + t[0] < len(self.discharge_fresh):
                    self.discharge_saline[time_index+t[0]] += V_ship / (end_time_passing_door-start_time_passing_door)
                    self.discharge_res[time_index + t[0]] += V_ship / (end_time_passing_door - start_time_passing_door)
        else:
            for t in enumerate(passing_time_door):
                if time_index+t[0] < len(self.discharge_fresh):
                    self.discharge_fresh[time_index+t[0]] += -V_ship / (end_time_passing_door-start_time_passing_door)
                    self.discharge_res[time_index + t[0]] += -V_ship / (end_time_passing_door - start_time_passing_door)
        return S_lock

    def door_open_harbour(self, T_door_open, time_index):
        index_node_doors1 = list(self.env.FG.nodes).index(self.node_doors1)
        index_node_doors2 = list(self.env.FG.nodes).index(self.node_doors2)
        index_node_open = list(self.env.FG.nodes).index(self.node_open)
        S_lock = self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_open].values[time_index]
        S_lock_harbour = self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_open].values[time_index]
        S_lock_average = (self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_doors1].values[time_index] + self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_doors2].values[time_index]) / 2
        WLev_lock_harbour = self.env.vessel_traffic_service.hydrodynamic_information['Water level'][index_node_open].values[time_index]
        V_lock_harbour = self.lock_length * self.lock_width * (WLev_lock_harbour + self.lock_depth)
        self.exchange_flux_time_series_calculator(T_door_open,time_index)
        # A loop that breaks at a certain moment assigning discharges to the sluice? (discharge should be separated for positive negative (and maybe sluice gates))
        if S_lock_harbour != S_lock:
            T_exch = self.lock_length / (0.5 * np.sqrt(self.grav_acc * (0.8 * abs(S_lock_harbour - S_lock) / (1000 + 0.8 * S_lock_average)) * (WLev_lock_harbour + self.lock_depth)))
            V_exch = V_lock_harbour * np.tanh(T_door_open /(2 * T_exch))
        else:
            V_exch = 0
        M_exch = (S_lock_harbour - S_lock) * V_exch
        S_lock = (S_lock * V_lock_harbour + M_exch) / V_lock_harbour
        return S_lock

    def sailing_in_from_harbour(self, vessel, distance_from_lineup_area, time_index):
        index_node_open = list(self.env.FG.nodes).index(self.node_open)
        V_ship = vessel.L * vessel.B * vessel.T_f
        S_lock = self.env.vessel_traffic_service.hydrodynamic_information['Salinity'][index_node_open].values[time_index]
        start_distance = distance_from_lineup_area-vessel.lineup_dist-0.5 * vessel.L
        start_time_passing_door = start_distance / vessel.v
        time_index = bisect.bisect_right(self.env.vessel_traffic_service.hydrodynamic_information['Times'].values,self.env.now + start_time_passing_door) - 1
        end_time_passing_door = vessel.L / vessel.v + start_time_passing_door
        passing_time_door = np.arange(start_time_passing_door, end_time_passing_door,self.env.vessel_traffic_service.hydrodynamic_information['Times'].values[1] - self.env.vessel_traffic_service.hydrodynamic_information['Times'].values[0])
        if self.node_open == self.node_doors2:
            for t in enumerate(passing_time_door):
                if time_index + t[0] < len(self.discharge_saline)-1:
                    self.discharge_saline[time_index + t[0]] += V_ship / (end_time_passing_door - start_time_passing_door)
                    self.discharge_res[time_index + t[0]] += V_ship / (end_time_passing_door - start_time_passing_door)
        if self.node_open == self.node_doors1:
            for t in enumerate(passing_time_door):
                if time_index + t[0] < len(self.discharge_fresh) - 1:
                    self.discharge_fresh[time_index + t[0]] += -V_ship / (end_time_passing_door - start_time_passing_door)
                    self.discharge_res[time_index + t[0]] += -V_ship / (end_time_passing_door - start_time_passing_door)
        # Q_avg = (V_ship_upstr + V_exch_inner) / T_door_open
        # V_loss_exch = V_ship_upstr - V_ship_downstr
        # S_avg = -(M_inner_a + M_inner_b + M_inner_c - (V_ship_upstr + V_exch_inner) * S_lock_inner) / (V_ship_downstr + V_exch_inner)
        return S_lock

    def total_ship_volume_in_lock(self):
        volume = 0
        for vessel in self.resource.users:
            volume += vessel.obj.B*vessel.obj.L*vessel.obj.T_f
        return volume

    def determine_levelling_time(self, delay):
        """ Function which calculates the operation time: based on the constant or nearest in the signal of the water level difference

            Input:
                - environment: see init function"""

        def calculate_discharge(lock,z,to_wlev,from_wlev,time_index,t,water_level):
            if to_wlev[t[0] + time_index] <= from_wlev[t[0] + time_index]:
                water_level[time_index + t[0]] = z + to_wlev[t[0] + time_index]
                if z != 0:
                    if lock.node_open == lock.node_doors2:
                        lock.discharge_res[t[0] + time_index] = -1 * lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                        lock.discharge_fresh[t[0] + time_index] = -1 * lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                    else:
                        lock.discharge_res[t[0] + time_index] = lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                        lock.discharge_saline[t[0] + time_index] = lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                else:
                    lock.discharge_res[t[0] + time_index] = 0
                    lock.discharge_fresh[t[0] + time_index] = 0
            else:
                water_level[time_index + t[0]] = to_wlev[t[0] + time_index] - z
                if z != 0:
                    if lock.node_open == lock.node_doors2:
                        lock.discharge_res[t[0] + time_index] = lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                        lock.discharge_saline[t[0] + time_index] = lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                    else:
                        lock.discharge_res[t[0] + time_index] = -1 * lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                        lock.discharge_fresh[t[0] + time_index] = -1 * lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                else:
                    lock.discharge_res[t[0] + time_index] = 0
                    lock.discharge_fresh[t[0] + time_index] = 0
            return

        index_node_doors1 = list(self.env.FG.nodes).index(self.node_doors1)
        index_node_doors2 = list(self.env.FG.nodes).index(self.node_doors2)
        if self.levelling == 'Calculated':
            if self.wlev_dif == 'Calculated':
                times = np.arange(0, 2*3600, self.env.vessel_traffic_service.hydrodynamic_information['Times'].values[1]-self.env.vessel_traffic_service.hydrodynamic_information['Times'].values[0])
                wlev_outer = self.env.vessel_traffic_service.hydrodynamic_information['Water level'][index_node_doors1].values
                wlev_inner = self.env.vessel_traffic_service.hydrodynamic_information['Water level'][index_node_doors2].values
                time_index = bisect.bisect_right(self.env.vessel_traffic_service.hydrodynamic_information['Times'].values, self.env.now+delay)-1
                if abs(wlev_outer[time_index]-wlev_inner[time_index])> 0.05:
                    z = abs(wlev_outer[time_index] - wlev_inner[time_index])
                    if self.node_open == self.node_doors2:
                        from_wlev = wlev_inner
                        to_wlev = wlev_outer
                    else:
                        from_wlev = wlev_outer
                        to_wlev = wlev_inner
                    calculate_discharge(self,z,to_wlev,from_wlev,time_index,[0,self.env.now],self.water_level)

                    time_series = list(self.env.vessel_traffic_service.hydrodynamic_information['Times'].values)
                    for t in enumerate(times):
                        if t[0] == 0:
                            continue
                        if t[0] + time_index >= len(to_wlev) - 1:
                            levelling_time = t[1]
                            break

                        dz = -self.disch_coeff * self.opening_area / (self.lock_length * self.lock_width) * np.sqrt(2 * 9.81 * z) * (t[1] - times[t[0] - 1])
                        if self.node_open == self.node_doors2:
                            dh = to_wlev[t[0] + time_index] - to_wlev[t[0] - 1 + time_index]
                        else:
                            dh = 0
                        z = z + dz - dh

                        calculate_discharge(self, z, to_wlev, from_wlev, time_index, t, self.water_level)
                        if z <= 0.05:
                            levelling_time = t[1]
                            delta_disch = self.discharge_res[t[0] + time_index]-self.discharge_res[t[0] + time_index-1]
                            for ta in enumerate(time_series[(t[0]+time_index):-1]):
                                new_disch = self.discharge_res[t[0] + time_index + ta[0]] + delta_disch
                                if new_disch != 0 and np.sign(new_disch) == np.sign(self.discharge_res[t[0] + time_index + ta[0]]):
                                    self.discharge_res[t[0] + time_index + ta[0]+1] = new_disch
                                    levelling_time = t[1]+ta[0]*(time_series[1]-time_series[0])
                                self.water_level[ta[0]+(t[0]+1)+time_index] = to_wlev[ta[0]+(t[0]+1)+time_index]
                            break
                else:
                    levelling_time = 0

            else:
                levelling_time = (2*self.lock_width * self.lock_length*math.sqrt(self.wlev_dif)) / (self.disch_coeff * self.opening_area * math.sqrt(2 * self.grav_acc))
        else:
            levelling_time = self.levelling

        return levelling_time

    def convert_chamber(self, environment, new_level, number_of_vessels, vessel, timeout_required = True):
        """ Function which converts the lock chamber and logs this event.

            Input:
                - environment: see init function
                - new_level: a string which represents the node and indicates the side at which the lock is currently levelled
                - number_of_vessels: the total number of vessels which are levelled simultaneously"""

        # Close the doors
        vessel.levelling_time = self.determine_levelling_time(delay=self.doors_close)
        self.log_entry("Lock doors closing start", environment.now, number_of_vessels, self.node_open)
        if timeout_required:
            yield environment.timeout(self.doors_close)
        self.log_entry("Lock doors closing stop", environment.now, number_of_vessels, self.node_open)

        if len(self.log['Message']) != 2:
            T_door_open = (self.env.now - self.log['Timestamp'][-3].timestamp())
        else:
            T_door_open = self.env.now-self.simulation_start

        time_index = bisect.bisect_right(self.env.vessel_traffic_service.hydrodynamic_information['Times'].values, self.env.now-T_door_open) - 1
        S_lock = self.door_open_harbour(T_door_open, time_index)
        time_index = bisect.bisect_right(self.env.vessel_traffic_service.hydrodynamic_information['Times'].values, self.env.now) - 1
        time_series = list(self.env.vessel_traffic_service.hydrodynamic_information['Times'].values)
        if time_index != len(time_series)-2:
            for index in range(len(self.water_level[time_index:])):
                self.salinity[index + time_index] = S_lock

        # Convert the chamber
        self.log_entry("Lock chamber converting start", environment.now, number_of_vessels, self.node_open)

        # Water level will shift
        if timeout_required:
            yield environment.timeout(vessel.levelling_time)
        self.change_water_level(new_level)
        self.log_entry("Lock chamber converting stop", environment.now, number_of_vessels, self.node_open)
        # Open the doors
        self.log_entry("Lock doors opening start", environment.now, number_of_vessels, self.node_open)
        if timeout_required:
            yield environment.timeout(self.doors_open)
        self.log_entry("Lock doors opening stop", environment.now, number_of_vessels, self.node_open)

    def change_water_level(self, side):
        """ Function which changes the water level in the lock chamber and priorities in queue """

        self.node_open = side

        for request in self.resource.queue:
            request.priority = -1 if request.priority == 0 else 0

            if request.priority == -1:
                self.resource.queue.insert(0, self.resource.queue.pop(self.resource.queue.index(request)))
            else:
                self.resource.queue.insert(-1, self.resource.queue.pop(self.resource.queue.index(request)))

class PassLock():
    """ Mixin class: a collection of functions which are used to pass a lock complex consisting of a waiting area, line-up areas, and lock chambers"""
    @staticmethod
    def leave_waiting_area(vessel,node_waiting_area):
        """ Processes vessels which are waiting in the waiting area of the lock complex and requesting access to preceding the line-up area:
                if there area multiple parallel lock chambers, the chamber with the least expected total waiting time is chosen,
                after which access is requested to enter the line-up area corresponding with the assigned lock chain series.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_waiting_area: a string which includes the name of the node the waiting area is located in the network """

        #Imports the properties of the waiting area
        waiting_area = vessel.env.FG.nodes[node_waiting_area]["Waiting area"][0]

        #Identifies the index of the node of the waiting area within the route of the vessel
        index_node_waiting_area = vessel.route.index(node_waiting_area)
        #Checks whether the waiting area is the first encountered waiting area of the lock complex
        for node1,node2 in zip(vessel.route[index_node_waiting_area:-1],vessel.route[index_node_waiting_area+1:]):
            if 'Line-up area' in vessel.env.FG.edges[node1,node2].keys():
                lineup_areas = vessel.env.FG.edges[node1, node2]["Line-up area"]
            elif 'Line-up area' in vessel.env.FG.edges[node2,node1].keys():
                lineup_areas = vessel.env.FG.edges[node2,node1]["Line-up area"]
            else:
                continue

            for lineup_area in lineup_areas:
                if waiting_area.name.split('_')[0] != lineup_area.name.split('_')[0]:
                    continue

                #Imports the location of the lock chamber of the lock complex
                for approach_node,departure_node in zip(vessel.route[index_node_waiting_area:-1],vessel.route[index_node_waiting_area+1:]):
                    if 'Lock' in vessel.env.FG.edges[approach_node,departure_node].keys():
                        locks = vessel.env.FG.edges[approach_node,departure_node]["Lock"]
                        break
                    elif 'Lock' in vessel.env.FG.edges[departure_node,approach_node].keys():
                        locks = vessel.env.FG.edges[departure_node,approach_node]["Lock"]
                        break

                def choose_lock_chamber(vessel,lock,lock_position,series_number,lineup_areas,lock_queue_length):
                    """ Assigns the lock chamber with the least expected total waiting time to the vessel in case of parallell lock chambers. The
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
                                                 appended. """

                    #Imports the properties of the evaluated line-up area
                    lineup_area = lineup_areas[series_number]

                    #Assesses the total queue length within this lock series
                    #- if the queue for the line-up area is empty, a name is set if the vessel fits in the lock chamber and line-up right away, otherwise the queue is calculated
                    if lineup_area.length.get_queue == []:
                        if (vessel.L <= lock.length.level and vessel.L <= lineup_area.length.level and lock.node_open == vessel.route[vessel.route.index(lock_position)-1]):
                            vessel.lock_name = lock.name
                        elif vessel.L <= lineup_area.length.level:
                            lock_queue_length.append(lineup_area.length.level)
                        else:
                            lock_queue_length.append(lineup_area.length.capacity)

                    #- else, if the vessel does not fit in the line-up area, the total length of the queued is calculated added with the full length capacity of the line-up area
                    else:
                        line_up_queue_length = lineup_area.length.capacity
                        for queued_vessel in lineup_area.length.get_queue:
                            line_up_queue_length += queued_vessel.amount
                        lock_queue_length.append(line_up_queue_length)

                def access_lineup_area(vessel,lineup_area):
                    """ Processes the request of vessels to access the line-up area by claiming a position (which equals the length of
                            the vessel) along its jetty.
                        This function is evaluated in the leave_waiting_area function

                        Input:
                            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                            - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                                           assigned to the vessel as the lock series with the least expected total waiting time """

                    def create_new_lock_cycle_and_request_access():
                        """ Creates a new locking cycle (wave of vessels assigned to same lockage) by requesting the full length
                                capacity of the line-up area, assigning the vessel to this request and returning the obsolete length
                                when the request is granted.
                            This function is used in the access_lineup_area function within the leave_waiting_area function

                            No input required """

                        vessel.access_lineup_length = lineup_area.length.get(lineup_area.length.capacity)
                        vessel.access_lineup_length.obj = vessel
                        yield vessel.access_lineup_length
                        lineup_area.length.put(lineup_area.length.capacity-vessel.L)

                    def request_access_lock_cycle(total_length_waiting_vessels = 0):
                        """ Processes the request of a vessel to enter a lock cycle (wave of vessels assigned to same lockage), depending
                                on the governing conditions regarding the current situation in the line-up area.
                            This function is used in the access_lineup_area function within the leave_waiting_area function

                            No input required """

                        #- If the line-up area has no queue, the vessel will access the lock cycle
                        if lineup_area.length.get_queue == []:
                            vessel.access_lineup_length = lineup_area.length.get(vessel.L)
                            vessel.access_lineup_length.obj = vessel

                        #Else, if there are already preceding vessels waiting in a queue, the vessel will request access to a lock cycle
                        else:
                            #Calculates the total length of vessels assigned to this lock cycle
                            for queued_vessel in reversed(lineup_area.length.get_queue):
                                total_length_waiting_vessels += queued_vessel.obj.L

                            #If the vessels does not fit in this lock cycle, it will start a new lock cycle
                            if vessel.L > lineup_area.length.capacity - total_length_waiting_vessels:
                                yield from create_new_lock_cycle_and_request_access()

                            #Else, if the vessel does fit in this last lock cycle, it will request a place in this cycle
                            else:
                                vessel.access_lineup_length = lineup_area.length.get(vessel.L)
                                vessel.access_lineup_length.obj = vessel
                                yield vessel.access_lineup_length

                    #Requesting procedure for access to line-up area
                    #- If there area vessels in the line-up area
                    if lineup_area.line_up_area[lineup_area.start_node].users != []:
                        # - If the vessels fits in the lock cycle right away
                        if vessel.L <= (lineup_area.line_up_area[lineup_area.start_node].users[-1].obj.lineup_dist-0.5*lineup_area.line_up_area[lineup_area.start_node].users[-1].obj.L):
                            yield from request_access_lock_cycle()

                        #- Else, if the vessel does not fit in the lock cyle right away
                        else:
                            if lineup_area.length.get_queue == []:
                                yield from create_new_lock_cycle_and_request_access()
                            else:
                                yield from request_access_lock_cycle()

                    #- Else, if there are no vessels yet in the line-up area
                    else:
                        # - If the vessels fits in the lock cycle right away
                        if vessel.L <= lineup_area.length.level:
                            yield from request_access_lock_cycle()

                        # - Else, if the vessel does not fit in the lock cyle right away
                        else:
                            if lineup_area.length.get_queue == []:
                                yield from create_new_lock_cycle_and_request_access()
                            else:
                                yield from request_access_lock_cycle()

                #Determines the current time
                wait_for_lineup_area = vessel.env.now
                vessel.v_before_lock = vessel.v
                vessel.id = vessel.id + '_' + str(vessel.metadata['start_time'])

                #Assigning the lock chain series with least expected waiting time to the vessel
                vessel.lock_name = []
                lock_queue_length = []
                for count,lock in enumerate(locks):
                    choose_lock_chamber(vessel,lock,approach_node,count,lineup_areas,lock_queue_length)

                #If the function did not yet assign a lock chain series
                if vessel.lock_name == []:
                    vessel.lock_name = lineup_areas[lock_queue_length.index(min(lock_queue_length))].name

                #Request access line-up area which is assigned to the vessel
                if lineup_area.name != vessel.lock_name:
                    continue

                yield from access_lineup_area(vessel,lineup_area)

                #Calculation of location in line-up area as a distance in [m] from start line-up jetty
                #- If the line-up area is not empty
                if len(lineup_area.line_up_area[lineup_area.start_node].users) != 0:
                    vessel.lineup_dist = (lineup_area.line_up_area[lineup_area.start_node].users[-1].obj.lineup_dist -0.5*lineup_area.line_up_area[lineup_area.start_node].users[-1].obj.L -0.5*vessel.L)
                #- Else, if the line-up area is empty
                else:
                    vessel.lineup_dist = lineup_area.length.capacity - 0.5*vessel.L

                #Calculation of the (lat,lon)-coordinates of the assigned position in the line-up area
                vessel.wgs84 = pyproj.Geod(ellps="WGS84")
                [lineup_area_start_lat,
                 lineup_area_start_lon,
                 lineup_area_stop_lat,
                 lineup_area_stop_lon] = [vessel.env.FG.nodes[lineup_area.start_node]['geometry'].x,
                                          vessel.env.FG.nodes[lineup_area.start_node]['geometry'].y,
                                          vessel.env.FG.nodes[lineup_area.end_node]['geometry'].x,
                                          vessel.env.FG.nodes[lineup_area.end_node]['geometry'].y]
                fwd_azimuth,_,_ = vessel.wgs84.inv(lineup_area_start_lat, lineup_area_start_lon, lineup_area_stop_lat, lineup_area_stop_lon)
                [vessel.lineup_pos_lat,
                 vessel.lineup_pos_lon,_] = pyproj.Geod(ellps="WGS84").fwd(vessel.env.FG.nodes[lineup_area.start_node]['geometry'].x,
                                                                           vessel.env.FG.nodes[lineup_area.start_node]['geometry'].y,
                                                                           fwd_azimuth,vessel.lineup_dist)

                #Formal request of the vessel to access the line-up area assigned to the vessel (always granted)
                vessel.access_lineup_area = lineup_area.line_up_area[lineup_area.start_node].request()
                vessel.access_lineup_area.obj = vessel
                vessel.access_lineup_area.obj.n = len(lineup_area.line_up_area[lineup_area.start_node].users)

                #make

                #Request of entering the line-up area to assure that vessels will enter the line-up area one-by-one
                vessel.enter_lineup_length = lineup_area.enter_line_up_area[lineup_area.start_node].request()
                vessel.enter_lineup_length.obj = vessel
                yield vessel.enter_lineup_length

                #Speed reduction in the approach to the line-up area
                vessel.wait_for_next_cycle = False
                vessel.waited_in_waiting_area = False
                vessel.v = 0.5*vessel.v

                #Calculates and reports the total waiting time in the waiting area
                if wait_for_lineup_area != vessel.env.now:
                    waiting = vessel.env.now - wait_for_lineup_area
                    vessel.log_entry("Waiting in waiting area start", wait_for_lineup_area, 0,
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[node_waiting_area])
                    vessel.log_entry("Waiting in waiting area stop", vessel.env.now, waiting,
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[node_waiting_area])

                    #Speed reduction in the approach to the line-up area, as the vessel had to lay still in the waiting area
                    vessel.v = 0.5*vessel.v

                    #Changes boolean of the vessel which indicates that it had to wait in the waiting area
                    for line_up_user in lineup_area.line_up_area[lineup_area.start_node].users:
                        if line_up_user.obj.id == vessel.id:
                            line_up_user.obj.waited_in_waiting_area = True
                            break
                break
            break

    def approach_lineup_area(vessel,start_node, end_node):
        """ Processes vessels which are approaching the line-up area of the lock complex:
                determines whether the assigned position in the line-up area (distance in [m]) should be changed as the preceding vessel(s),
                which was/were waiting in the line-up area, has/have of is/are already accessed/accessing the lock.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """

        #Imports the properties of the line-up area the vessel is assigned to
        lineup_areas = vessel.env.FG.edges[start_node, end_node]["Line-up area"]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name:
                continue

            #Identifies the index of the node of the waiting area within the route of the vessel
            index_node_lineup_area = vessel.route.index(end_node)
            yield vessel.env.timeout(0) #to create a generator out of this function

            #Determines corresponding lock
            locks = []
            for node1,node2 in zip(vessel.route[index_node_lineup_area:-1],vessel.route[index_node_lineup_area+1:]):
                if 'Lock' in vessel.env.FG.edges[node1,node2].keys():
                    direction = 1
                    locks = vessel.env.FG.edges[node1,node2]["Lock"]
                    break
                elif 'Lock' in vessel.env.FG.edges[node2,node1].keys():
                    direction = 0
                    locks = vessel.env.FG.edges[node2,node1]["Lock"]
                    break
                else:
                    continue

            for lock in locks:
                if lock.name != vessel.lock_name:
                    continue

                # Alters vessel's approach speed to lock chamber, if vessel didn't have to wait in the waiting area
                for line_up_user in lineup_area.line_up_area[start_node].users:
                    if (line_up_user.obj.id == vessel.id and not line_up_user.obj.waited_in_waiting_area):
                        vessel.v = 0.5 * vessel.v
                        break

                def change_lineup_dist(vessel, lock, lineup_dist, lineup_area_user):
                    """ Determines whether the assigned position in the line-up area (distance in [m]) should be changed as the preceding vessel(s),
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

                    if lineup_area_user[0] == 0 and lineup_area_user[1].obj.n != (lineup_area_user[1].obj.n-len(lock.resource.users)):
                        lineup_dist = lock.length.capacity - 0.5*vessel.L
                    return lineup_dist

                #Checks the need to change the position of the vessel within the line-up area
                for vessel_index,lineup_area_user in enumerate(lineup_area.line_up_area[start_node].users):
                    if lineup_area_user.obj.id == vessel.id:

                        #Imports information about the current lock cycle
                        lock_door_1_user_priority = 0
                        lock_door_2_user_priority = 0
                        lock_door_1_users = lock.doors_1[lock.node_doors1].users
                        lock_door_2_users = lock.doors_2[lock.node_doors2].users

                        if direction and lock_door_2_users != []:
                            lock_door_2_user_priority = lock.doors_2[lock.node_doors2].users[0].priority

                        elif not direction and lock_door_1_users != []:
                            lock_door_1_user_priority = lock.doors_1[lock.node_doors1].users[0].priority

                        #Decision if position should be changed
                        if direction and lock_door_2_user_priority == -1:
                            vessel.lineup_dist = change_lineup_dist(vessel, lock, vessel.lineup_dist, (vessel_index,lineup_area_user))

                        elif not direction and lock_door_1_user_priority == -1:
                            vessel.lineup_dist = change_lineup_dist(vessel, lock, vessel.lineup_dist, (vessel_index,lineup_area_user))

                        #Calculation of (lat,lon)-coordinates based on (newly) assigned position (line-up distance in [m])
                        [lineup_area_start_lat,
                         lineup_area_start_lon,
                         lineup_area_stop_lat,
                         lineup_area_stop_lon] = [vessel.env.FG.nodes[start_node]['geometry'].x,
                                                  vessel.env.FG.nodes[start_node]['geometry'].y,
                                                  vessel.env.FG.nodes[end_node]['geometry'].x,
                                                  vessel.env.FG.nodes[end_node]['geometry'].y]

                        fwd_azimuth,_,_ = pyproj.Geod(ellps="WGS84").inv(lineup_area_start_lat,
                                                                         lineup_area_start_lon,
                                                                         lineup_area_stop_lat,
                                                                         lineup_area_stop_lon)

                        [vessel.lineup_pos_lat,
                         vessel.lineup_pos_lon,_] = pyproj.Geod(ellps="WGS84").fwd(vessel.env.FG.nodes[vessel.route[vessel.route.index(start_node)]]['geometry'].x,
                                                                                   vessel.env.FG.nodes[vessel.route[vessel.route.index(start_node)]]['geometry'].y,
                                                                                   fwd_azimuth,vessel.lineup_dist)

                        break
                    break
                break
            break

    def leave_lineup_area(vessel,start_node,end_node):
        """ Processes vessels which are waiting in the line-up area of the lock complex:
                requesting access to the lock chamber given the governing phase in the lock cycle of the lock chamber and calculates the
                position within the lock chamber.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """

        #Imports the properties of the line-up area the vessel is assigned to
        lineup_areas = vessel.env.FG.edges[start_node,end_node]["Line-up area"]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name:
                continue

            #Identifies the index of the node of the line-up area within the route of the vessel
            index_node_lineup_area = vessel.route.index(end_node)

            #Determines lock
            locks = []
            for approach_node,departure_node in zip(vessel.route[index_node_lineup_area:-1],vessel.route[index_node_lineup_area+1:]):
                if 'Lock' in vessel.env.FG.edges[approach_node,departure_node].keys():
                    direction = 1
                    locks = vessel.env.FG.edges[approach_node,departure_node]["Lock"]
                    break
                elif 'Lock' in vessel.env.FG.edges[departure_node,approach_node].keys():
                    direction = 0
                    locks = vessel.env.FG.edges[departure_node,approach_node]["Lock"]
                    break
                else:
                    if departure_node != vessel.route[-1]:
                        continue
                    yield from PassLock.leave_opposite_lineup_area(vessel,start_node,end_node)

            for lock in locks:
                if lock.name != vessel.lock_name:
                    continue

                #Imports position of the vessel in the line-up area
                position_in_lineup_area = shapely.geometry.Point(vessel.lineup_pos_lat,vessel.lineup_pos_lon)

                #Vessel releases its request to enter the line-up area, made in the waiting area
                lineup_area.enter_line_up_area[start_node].release(vessel.enter_lineup_length)

                #Determines current time
                wait_for_lock_entry = vessel.env.now

                #Checks whether the lock chamber is the first encountered lock chamber of the lock complex
                for node1,node2 in zip(vessel.route[index_node_lineup_area:-1],vessel.route[index_node_lineup_area+1:]):
                    if 'Line-up area' in vessel.env.FG.edges[node1,node2].keys():
                        opposing_lineup_areas = vessel.env.FG.edges[node1, node2]["Line-up area"]
                    elif 'Line-up area' in vessel.env.FG.edges[node2,node1].keys():
                        opposing_lineup_areas = vessel.env.FG.edges[node2,node1]["Line-up area"]
                    else:
                        continue

                    #Imports the properties of the opposing line-up area the vessel is assigned to
                    for opposing_lineup_area in opposing_lineup_areas:
                        if opposing_lineup_area.name == vessel.lock_name:
                            break
                    break

                def access_lock_chamber(vessel,lineup_area,lock,approach_node,opposing_lineup_area,door1,door2,node_door1,node_door2):
                    """ Processes vessels which are waiting in the line-up area of the lock complex:
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
                            - opposing_lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                                                    assigned to the vessel to be entered when leaving the lock chamber
                            - node_opposing_lineup_area: a string which includes the name of the node at which the line-up area is located on the
                                                         opposite side of the lock chamber in the network
                            - door1: an object created in the IsLock class which resembles the set of lock doors which is first encountered by
                                     the vessel, which should be supscripted to, using a string which includes the name of the node at which this
                                     lock door is located in the network and was specified as input in the IsLock class
                            - door2: an object created in the IsLock class which resembles the set of lock doors which is last encountered by
                                     the vessel, which should be supscripted to, using a string which includes the name of the node at which this
                                     lock door is located in the network and was specified as input in the IsLock class """

                    def request_approach_lock_chamber(timeout_required=True,priority=0):
                        """ Vessel will request if it can enter the lock by requesting access to the first set of lock doors. This
                                request always has priority = 0, as vessels can only pass these doors when the doors are open (not
                                claimed by a priority = -1 request). The capacity of the doors equals one, to prevent ships from
                                entering simultaneously. The function yields a timeout. This can be switched off if it is assured
                                the vessel can approach immediately.

                            Input:
                                - timeout_required: a boolean which defines whether the requesting vessel receives a timeout. """

                        vessel.access_lock_door1 = door1.request(priority=priority)
                        vessel.access_lock_door1.obj = vessel
                        if timeout_required:
                            yield vessel.access_lock_door1

                    def request_empty_lock_conversion(lock,hold_request = False):
                        """ Vessel will request the lock chamber to be converted without vessels to his side of the lock chamber. This
                                is programmed by requesting the converting_while_in_line_up_area resource of the line-up area the vessels is
                                currently located in. If there was already a request by another vessel waiting in the same line-up area, this
                                original request can be holded.

                            Input:
                                - hold_request: a boolean which defines where an earlier request for the same empty lock conversion which
                                                was made by another ship should be holded"""

                        vessel.waiting_during_converting = lineup_area.converting_while_in_line_up_area[lineup_area.start_node].request()
                        vessel.waiting_during_converting.obj = vessel
                        yield vessel.waiting_during_converting
                        if not hold_request:
                            yield from lock.convert_chamber(vessel.env, approach_node, 0, vessel)
                        lineup_area.converting_while_in_line_up_area[lineup_area.start_node].release(vessel.waiting_during_converting)

                    def secure_lock_cycle(hold_request = False, timeout_required = True, priority = -1):
                        """ Vessel will indicate the direction of the next lock cycle by requesting access to the second pair of lock
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
                                            priority. """

                        if 'access_lock_door2' not in dir(vessel):
                            vessel.access_lock_door2 = door2.request(priority = priority)
                            vessel.access_lock_door2.obj = vessel
                            if hold_request:
                                door2.release(door2.users[0])
                            if timeout_required:
                                yield vessel.access_lock_door2

                    def wait_for_next_lockage():
                        """ Vessels will wait for the next lockage by requesting access to the second pair of lock doors without priority. If
                                granted, the request will immediately be released.

                            No input required. """

                        yield from secure_lock_cycle(priority = 1)
                        door2.release(vessel.access_lock_door2)
                        delattr(vessel, 'access_lock_door2')

                    def request_place_in_next_lockage():
                        if 'in_next_lockage' not in dir(vessel) and 'first_in_next_lockage' not in dir(vessel):
                            vessel.in_lockage = lock.next_lockage_order[node_door2].request()
                            vessel.in_lockage.obj = vessel
                            lock.next_lockage_length[node_door2].get(vessel.L)
                            vessel.lock_dist = lock.next_lockage_length[node_door2].level + 0.5 * vessel.L  # (distance from first set of doors in [m])

                            order_index_last_vessel = len(lock.next_lockage_order[node_door2].users)-1
                            order_of_users = lock.next_lockage_order[node_door2].users
                            for order_index,vessel_in_lockage in enumerate(order_of_users):
                                if 'in_next_lockage' in dir(vessel_in_lockage.obj):
                                    order_index_last_vessel = order_index

                                if vessel_in_lockage.obj.id == vessel.id:
                                    if order_index > order_index_last_vessel:
                                        order_of_users[order_index], order_of_users[order_index_last_vessel] = order_of_users[order_index_last_vessel], order_of_users[order_index]
                                        order_of_users[order_index].obj.lock_dist += order_of_users[order_index_last_vessel].obj.L
                                        order_of_users[order_index_last_vessel].obj.lock_dist -= order_of_users[order_index].obj.L
                                    break

                    #Determines current moment within the lock cycle
                    lock_door_2_user_priority = 0
                    if door2.users != []:
                        lock_door_2_user_priority = door2.users[0].priority

                    #Request procedure of the lock doors, which is dependent on the current moment within the lock cycle:
                    #- If there is a lock cycle being prepared or going on in the same direction of the vessel
                    if lock_door_2_user_priority == -1:
                        #If vessel does not fit in next lock cycle or locking has already started
                        if lock.resource.users != [] and (vessel.L > (lock.resource.users[-1].obj.lock_dist-0.5*lock.resource.users[-1].obj.L) or lock.resource.users[-1].obj.converting):
                            yield from wait_for_next_lockage()

                        request_place_in_next_lockage()

                        # Request to start the lock cycle
                        if lock.resource.users and ('converting' in dir(lock.resource.users[-1].obj) and not lock.resource.users[-1].obj.converting):
                            yield from request_approach_lock_chamber()
                        else:
                            yield from request_approach_lock_chamber(priority=-1)

                        if door2.users != [] and door2.users[0].priority == -1:
                            yield from secure_lock_cycle(hold_request=True)
                        else:
                            yield from secure_lock_cycle(timeout_required=False)

                            # Determines if an empty lockage is required
                            if approach_node != lock.node_open:
                                yield from request_empty_lock_conversion(lock)

                    #- If there is a lock cycle being prepared or going on to the direction of the vessel or if the lock chamber is empty
                    else:
                        request_place_in_next_lockage()

                        #Determining (new) situation
                        #- If the lock chamber is empty
                        if not lock.resource.users and not lock.next_lockage_order[node_door1].users:
                            if door2.users and door2.users[0].priority:
                                yield from request_approach_lock_chamber()
                                yield from secure_lock_cycle(hold_request=True)
                            else:
                                yield from request_approach_lock_chamber(timeout_required=False)
                                yield from secure_lock_cycle(timeout_required=False)

                            #Determines if an empty lockage is required
                            if approach_node != lock.node_open:
                                yield from request_empty_lock_conversion(lock)

                        #- Else, if the lock chamber is occupied or claimed:
                        else:
                            #Request to start the lock cycle
                            yield from request_approach_lock_chamber()

                            if door2.users != [] and door2.users[0].priority == -1:
                                yield from secure_lock_cycle(hold_request=True)
                            else:
                                yield from secure_lock_cycle(timeout_required=False)

                    #Formal request access to lock chamber and calculate position within the lock chamber
                    lock.length.get(vessel.L)
                    vessel.access_lock = lock.resource.request()
                    vessel.access_lock.obj = vessel
                    vessel.converting = False

                #Request access to lock chamber
                if direction:
                    yield from access_lock_chamber(vessel,lineup_area,lock,approach_node,opposing_lineup_area,lock.doors_1[lock.node_doors1],lock.doors_2[lock.node_doors2],lock.node_doors1,lock.node_doors2)
                if not direction:
                    yield from access_lock_chamber(vessel,lineup_area,lock,approach_node,opposing_lineup_area,lock.doors_2[lock.node_doors2],lock.doors_1[lock.node_doors1],lock.node_doors2,lock.node_doors1)

                #Releases the vessel's formal request to access the line-up area and releases its occupied length of the line-up area
                lineup_area.line_up_area[lineup_area.start_node].release(vessel.access_lineup_area)
                lineup_area.length.put(vessel.L)

                #Calculates and reports the total waiting time in the line-up area
                if wait_for_lock_entry != vessel.env.now:
                    waiting = vessel.env.now - wait_for_lock_entry
                    vessel.log_entry("Waiting in line-up area start", wait_for_lock_entry, 0, position_in_lineup_area)
                    vessel.log_entry("Waiting in line-up area stop", vessel.env.now, waiting, position_in_lineup_area)

                #Calculation of (lat,lon)-coordinates of assigned position in lock chamber
                vessel.wgs84 = pyproj.Geod(ellps="WGS84")
                [node_start_lat,
                 node_start_lon,
                 node_end_lat,
                 node_end_lon] = [vessel.env.FG.nodes[lineup_area.start_node]['geometry'].x,
                                  vessel.env.FG.nodes[lineup_area.start_node]['geometry'].y,
                                  vessel.env.FG.nodes[lineup_area.end_node]['geometry'].x,
                                  vessel.env.FG.nodes[lineup_area.end_node]['geometry'].y]
                _, _, distance = vessel.wgs84.inv(node_start_lat,node_start_lon,node_end_lat,node_end_lon)

                [doors1_lat,
                 doors1_lon,
                 doors2_lat,
                 doors2_lon] = [vessel.env.FG.nodes[approach_node]['geometry'].x,
                                vessel.env.FG.nodes[approach_node]['geometry'].y,
                                vessel.env.FG.nodes[departure_node]['geometry'].x,
                                vessel.env.FG.nodes[departure_node]['geometry'].y]

                fwd_azimuth,_,_ = vessel.wgs84.inv(doors1_lat, doors1_lon, doors2_lat, doors2_lon)

                [vessel.lock_pos_lat,vessel.lock_pos_lon,_] = vessel.wgs84.fwd(vessel.env.FG.nodes[vessel.route[vessel.route.index(approach_node)]]['geometry'].x,
                                                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(approach_node)]]['geometry'].y,
                                                                               fwd_azimuth,vessel.lock_dist)

                time_index = bisect.bisect_right(vessel.env.vessel_traffic_service.hydrodynamic_information['Times'].values,vessel.env.now) - 1
                S_lock = lock.sailing_in_from_harbour(vessel, distance, time_index)
                for index in range(len(lock.water_level[time_index:])):
                    lock.salinity[index + time_index] = S_lock
                break
            break

    def leave_lock(vessel,node_doors1,node_doors2,direction):
        """ Processes vessels which are waiting in the lock chamber to be levelled and after levelling:
                checks if vessels which have entered the lock chamber have to wait for the other vessels to enter the lock chamber and
                requests conversion of the lock chamber.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lock: a string which includes the name of the node at which the lock chamber is located in the network """

        #Imports the properties of the lock chamber the vessel is assigned to
        locks = vessel.env.FG.edges[node_doors1,node_doors2]["Lock"]
        for lock in locks:
            if lock.name != vessel.lock_name:
                continue

            position_in_lock = shapely.geometry.Point(vessel.lock_pos_lat,vessel.lock_pos_lon) #alters node_lock in accordance with given position in lock

            #Identifies the index of the node of the lock chamber within the route of the vessel
            index_node_doors1 = vessel.route.index(node_doors1)
            index_node_doors2 = vessel.route.index(node_doors2)

            #Checks whether the lock chamber is the first encountered lock chamber of the lock complex
            for node1, node2 in zip(vessel.route[index_node_doors2:-1],vessel.route[index_node_doors2+1:]):
                if "Line-up area" in vessel.env.FG.edges[node1,node2].keys():
                    opposing_lineup_areas = vessel.env.FG.edges[node1,node2]["Line-up area"]
                elif "Line-up area" in vessel.env.FG.edges[node2,node1].keys():
                    opposing_lineup_areas = vessel.env.FG.edges[node2, node1]["Line-up area"]
                else:
                    continue

                for opposing_lineup_area in opposing_lineup_areas:
                    if opposing_lineup_area.name != vessel.lock_name:
                        continue
                    break
                break

            #Releases the vessel's request of their first encountered set of lock doors
            if direction:
                node_doors2 = lock.node_doors2
                yield lock.doors_1[lock.node_doors1].release(vessel.access_lock_door1)
            elif not direction:
                node_doors2 = lock.node_doors1
                yield lock.doors_2[lock.node_doors2].release(vessel.access_lock_door1)

            #Imports the properties of the line-up area the vessel was assigned to
            for node1,node2 in zip(reversed(vessel.route[:index_node_doors1-1]),reversed(vessel.route[1:index_node_doors1])):
                if "Line-up area" in vessel.env.FG.edges[node1,node2].keys():
                    lineup_areas = vessel.env.FG.edges[node1,node2]["Line-up area"]
                elif "Line-up area" in vessel.env.FG.edges[node2,node1].keys():
                    lineup_areas = vessel.env.FG.edges[node2,node1]["Line-up area"]
                else:
                    continue

                for lineup_area in lineup_areas:
                    if lineup_area.name != vessel.lock_name:
                        continue

                    #Determines current time and reports this to vessel's log as start time of lock passage
                    start_time_in_lock = vessel.env.now
                    vessel.log_entry("Passing lock start", vessel.env.now, 0, position_in_lock)

                    #Determines if accessed vessel has to wait on accessing vessels
                    if lock.next_lockage_order[node_doors2].users[-1].obj.id != vessel.id:
                        yield lock.next_lockage_length[node_doors2].get(lock.length.capacity)
                        lock.next_lockage_length[node_doors2].put(lock.length.capacity)

                    # Request access to pass the next line-up area after the lock chamber has levelled, so that vessels will leave the lock chamber one-by-one
                    vessel.departure_lock = opposing_lineup_area.pass_line_up_area[opposing_lineup_area.start_node].request(priority=-1)
                    vessel.departure_lock.obj = vessel

                    #Determines if the vessel explicitely has to request the conversion of the lock chamber (only the last entered vessel) or can go with a previously made request
                    if lock.next_lockage_order[node_doors2].users[-1].obj.id == vessel.id:
                        yield vessel.departure_lock
                        yield lock.next_lockage_length[node_doors2].put(lock.next_lockage_length[node_doors2].capacity-lock.next_lockage_length[node_doors2].level)
                        yield lock.next_lockage_length[node_doors2].get(vessel.L)
                        vessel.converting = True
                        number_of_vessels = len(lock.resource.users)
                        yield from lock.convert_chamber(vessel.env, node_doors2,number_of_vessels,vessel)
                        opposing_lineup_area.pass_line_up_area[opposing_lineup_area.start_node].release(vessel.departure_lock)
                        vessel.departure_lock = opposing_lineup_area.pass_line_up_area[opposing_lineup_area.start_node].request(priority=-1)
                        yield vessel.departure_lock
                    else:
                        for lock_user in lock.resource.users:
                            if lock_user.obj.id != vessel.id:
                                continue
                            lock_user.obj.converting = True
                            yield vessel.departure_lock
                            break

                if lock.next_lockage_order[node_doors2].users[-1].obj.id == vessel.id:
                    lock.next_lockage_length[node_doors2].put(lock.next_lockage_length[node_doors2].capacity-lock.next_lockage_length[node_doors2].level)

                if 'in_lockage' in dir(vessel):
                    lock.next_lockage_order[node_doors2].release(vessel.in_lockage)
                elif 'in_next_lockage' in dir(vessel):
                    lock.next_lockage_order[node_doors2].release(vessel.in_next_lockage)
                elif 'first_in_next_lockage' in dir(vessel):
                    lock.next_lockage_order[node_doors2].release(vessel.first_in_next_lockage)

                time_index = bisect.bisect_right(vessel.env.vessel_traffic_service.hydrodynamic_information['Times'].values,vessel.env.now) - 1
                S_lock = lock.sailing_out_to_harbour(vessel, time_index)
                for index in range(len(lock.water_level[time_index:])):
                    lock.salinity[index + time_index] = S_lock

                #Calculates and reports the total locking time
                vessel.log_entry("Passing lock stop", vessel.env.now, vessel.env.now-start_time_in_lock, position_in_lock,)
                vessel.lineup_pos_lat = vessel.env.FG.nodes[opposing_lineup_area.end_node]['geometry'].x
                vessel.lineup_pos_lon = vessel.env.FG.nodes[opposing_lineup_area.end_node]['geometry'].y
                break
            break

    def leave_opposite_lineup_area(vessel,start_node,end_node):
        """ Processes vessels which have left the lock chamber after levelling and are now in the next line-up area in order to leave the lock complex through the next waiting area:
                release of their requests for accessing their second encountered line-up area and lock chamber.

            Input:
                - vessel: an identity which is Identifiable, Movable,and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """

        #Imports the properties of the line-up area the vessel is assigned to
        lineup_areas = vessel.env.FG.edges[start_node,end_node]["Line-up area"]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name: #assure lock chain = assigned chain
                continue
            index_node_lineup_area = vessel.route.index(start_node)

            #Checks whether the line-up area is the second encountered line-up area of the lock complex
            locks = []
            for approach_node, departure_node in zip(reversed(vessel.route[:index_node_lineup_area-1]),reversed(vessel.route[1:index_node_lineup_area])):
                if 'Lock' in vessel.env.FG.edges[approach_node, departure_node].keys():
                    direction = 1
                    locks = vessel.env.FG.edges[approach_node, departure_node]["Lock"]
                    break
                elif 'Lock' in vessel.env.FG.edges[departure_node, approach_node].keys():
                    direction = 0
                    locks = vessel.env.FG.edges[departure_node, approach_node]["Lock"]
                    break
                else:
                    continue

            #Imports the properties of the lock chamber the vessel was assigned to
            for lock in locks:
                if lock.name != vessel.lock_name:
                    continue

                #Releases the vessel's request of their second encountered set of lock doors
                if direction and lock.doors_2[lock.node_doors2].users[0].obj.id == vessel.id:
                    lock.doors_2[lock.node_doors2].release(vessel.access_lock_door2)

                if not direction and lock.doors_1[lock.node_doors1].users[0].obj.id == vessel.id:
                    lock.doors_1[lock.node_doors1].release(vessel.access_lock_door2)

                #Releases the vessel's request to enter the second line-up area
                lineup_area.pass_line_up_area[lineup_area.start_node].release(vessel.departure_lock)

                #Releases the vessel's formal request of the lock chamber and returns its occupied length in the lock chamber
                lock.resource.release(vessel.access_lock)
                departure_lock_length = lock.length.put(vessel.L) #put length back in lock
                yield departure_lock_length
                break
            break