# package(s) related to the simulation
import networkx as nx

from opentnsim import core

# spatial libraries
import pyproj
import shapely.geometry
import simpy
import random
import numpy as np
import math
import bisect
import datetime

class HasLock(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.leave_lock_chamber)

    def leave_lock_chamber(self,origin):
        if "Lock" in self.env.FG.nodes[origin].keys():  # if vessel in lock
            yield from PassLock.leave_lock(self, origin)
            self.v = 4 * self.v

class HasWaitingArea(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.leave_waiting_area)
        self.on_look_ahead_to_node.append(self.approach_waiting_area)

    def approach_waiting_area(self,destination):
        if "Waiting area" in self.env.FG.nodes[destination].keys():  # if waiting area is located at next node
            yield from PassLock.approach_waiting_area(self, destination)

    def leave_waiting_area(self, origin):
        if "Waiting area" in self.env.FG.nodes[origin].keys():  # if vessel is in waiting area
            yield from PassLock.leave_waiting_area(self, origin)

class HasLineUpArea(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.leave_lineup_area)
        self.on_look_ahead_to_node.append(self.approach_lineup_area)

    def approach_lineup_area(self,destination):
        if "Line-up area" in self.env.FG.nodes[destination].keys():  # if vessel is approaching the line-up area
            yield from PassLock.approach_lineup_area(self, destination)

    def leave_lineup_area(self,origin):
        if "Line-up area" in self.env.FG.nodes[origin].keys():  # if vessel is located in the line-up
            lineup_areas = self.env.FG.nodes[origin]["Line-up area"]
            for lineup_area in lineup_areas:
                if lineup_area.name != self.lock_name:  # picks the assigned parallel lock chain
                    continue

                index_node_lineup_area = self.route.index(origin)
                for node_lock in self.route[index_node_lineup_area:]:
                    if 'Lock' in self.env.FG.nodes[node_lock].keys():
                        yield from PassLock.leave_lineup_area(self, origin)
                        break

                    elif 'Waiting area' in self.env.FG.nodes[node_lock].keys():  # if vessel is leaving the lock complex
                        yield from PassLock.leave_opposite_lineup_area(self, origin)
                        break

class IsLockWaitingArea(core.HasResource, core.Identifiable, core.Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
        self,
        node, #a string which indicates the location of the start of the waiting area
        *args,
        **kwargs):

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
        node, #a string which indicates the location of the start of the line-up area
        lineup_length, #a float which contains the length of the line-up area
        *args,
        **kwargs):

        super().__init__(length = lineup_length, remaining_length = lineup_length, *args, **kwargs)

        """Initialization"""
        # Lay-Out
        self.enter_line_up_area = {node: simpy.PriorityResource(self.env, capacity=1),} #used to regulate one by one entering of line-up area, so capacity must be 1
        self.line_up_area = {node: simpy.PriorityResource(self.env, capacity=100),} #line-up area itself, infinite capacity, as this is regulated by the HasLength, so capacity = inf
        self.converting_while_in_line_up_area = {node: simpy.PriorityResource(self.env, capacity=1),} #used to minimize the number of empty convertion requests by one by multiple waiting vessels, so capacity must be 1
        self.pass_line_up_area = {node: simpy.PriorityResource(self.env, capacity=1),} #used to prevent vessel from entering the lock before all previously locked vessels have passed the line-up area one by one, so capacity must be 1

class IsLock(core.HasResource, core.HasLength, core.Identifiable, core.Log):
    """Mixin class: Something which has lock chamber object properties as part of a lock complex [in SI-units] """

    def __init__(
        self,
        node_1, #a string which indicates the location of the first pair of lock doors
        node_2, #a string which indicates the center of the lock chamber
        node_3, #a string which indicates the location of the second pair of lock doors
        lock_length, #a float which contains the length of the lock chamber
        lock_width, #a float which contains the width of the lock chamber
        lock_depth, #a float which contains the depth of the lock chamber
        doors_open, #a float which contains the time it takes to open the doors
        doors_close, #a float which contains the time it takes to close the doors
        disch_coeff, #a float which contains the discharge coefficient of filling system
        opening_area, #a float which contains the cross-sectional area of filling system
        opening_depth, #a float which contains the depth at which filling system is located
        simulation_start, #a datetime which contains the simulation start time
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

        super().__init__(length = lock_length, remaining_length = lock_length, *args, **kwargs)

        self.doors_1 = {node_1: simpy.PriorityResource(self.env, capacity = 1),} #Only one ship can pass at a time: capacity = 1, request can have priority
        self.doors_2 = {node_3: simpy.PriorityResource(self.env, capacity = 1),} #Only one ship can pass at a time: capacity = 1, request can have priority

        # Operating
        self.doors_open = doors_open
        self.doors_close = doors_close

        # Water level
        assert node_1 != node_3

        self.node_1 = node_1
        self.node_2 = node_2
        self.node_3 = node_3
        self.water_level = random.choice([node_1, node_3])
        self.discharge_res = np.zeros(len(self.env.FG.nodes[self.node_2]['Info']['Water level']))
        self.discharge_saline = np.zeros(len(self.env.FG.nodes[self.node_2]['Info']['Water level']))
        self.discharge_fresh = np.zeros(len(self.env.FG.nodes[self.node_2]['Info']['Water level']))
        self.env.FG.nodes[self.node_2]['Info']['Water level'] = [self.env.FG.nodes[self.water_level]['Info']['Water level'][index] for index in range(len(self.env.FG.nodes[self.node_2]['Info']['Water level']))]

    def exchange_flux_time_series_calculator(self,T_door_open,time_index):
        S_lock = self.env.FG.nodes[self.node_2]['Info']['Salinity'][time_index]
        S_lock_harbour = self.env.FG.nodes[self.water_level]['Info']['Salinity'][time_index]
        S_lock_average = (self.env.FG.nodes[self.node_1]['Info']['Salinity'][time_index] +
                          self.env.FG.nodes[self.node_3]['Info']['Salinity'][time_index]) / 2
        Wlev_lock = self.env.FG.nodes[self.water_level]['Info']['Water level'][time_index]
        V_lock = self.lock_length * self.lock_width * (Wlev_lock + self.lock_depth)
        v_exch = (0.5 * np.sqrt(self.grav_acc * (0.8 * abs(S_lock_harbour - S_lock) / (1000 + 0.8 * S_lock_average)) * (Wlev_lock + self.lock_depth)))
        T_LE = (2 * self.lock_length) / v_exch
        time = np.arange(0, T_door_open, self.env.FG.nodes[self.node_1]['Info']['Times'][1]-self.env.FG.nodes[self.node_1]['Info']['Times'][0])
        Q = []
        V_tot = 0
        for t in enumerate(time):
            if t[0] == 0:
                continue
            delta_t = (t[1] - time[t[0] - 1])
            delta_V = V_lock * (np.tanh(t[1] / T_LE) - np.tanh(time[t[0] - 1] / T_LE))
            V_tot += delta_V
            V_tot += delta_V
            Q.append(delta_V / delta_t)
            M = (S_lock_harbour - self.env.FG.nodes[self.node_2]['Info']['Salinity'][time_index]) * delta_V
            S_lock = (S_lock * V_lock + M) / V_lock
            self.env.FG.nodes[self.node_2]['Info']['Salinity'][time_index+t[0]] = S_lock
            self.discharge_saline[time_index+t[0]] += delta_V / delta_t
            self.discharge_fresh[time_index+t[0]] += -delta_V / delta_t
        return

    def levelling_to_harbour(self,V_ship,levelling_time,time_index,side):
        time_index_stop = bisect.bisect_right(self.env.FG.nodes[self.node_2]['Info']['Times'], self.env.now+levelling_time)-1
        time = np.arange(time_index,time_index_stop+1,1)
        S_lock_start = S_lock = self.env.FG.nodes[self.node_2]['Info']['Salinity'][time_index]
        WLev_lock_inner = self.env.FG.nodes[self.node_3]['Info']['Water level'][time_index]
        V_lock_inner = self.lock_length * self.lock_width * (WLev_lock_inner + self.lock_depth)
        WLev_lock_outer = self.env.FG.nodes[self.node_1]['Info']['Water level'][time_index]
        V_lock_outer = self.lock_length * self.lock_width * (WLev_lock_outer + self.lock_depth)
        if side == self.node_1:
            WLev_to_side = WLev_lock_inner
            WLev_from_side = WLev_lock_outer
            V_to_side = V_lock_inner
            V_from_side = V_lock_outer
            if WLev_from_side < WLev_to_side:
                filling = True
                S_to_side = self.env.FG.nodes[self.node_3]['Info']['Salinity'][time_index]
            else:
                filling = False
        else:
            WLev_to_side = WLev_lock_outer
            WLev_from_side = WLev_lock_inner
            V_to_side = V_lock_outer
            V_from_side = V_lock_inner
            if WLev_from_side < WLev_to_side:
                filling = True
                S_to_side = self.env.FG.nodes[self.node_1]['Info']['Salinity'][time_index]
            else:
                filling = False

        if filling:
            V_levelling = self.lock_length * self.lock_width * (WLev_to_side - WLev_from_side)
            levelling_time = self.env.FG.nodes[self.node_2]['Info']['Times'][time_index_stop]-self.env.FG.nodes[self.node_2]['Info']['Times'][time_index]
            S_lock_final = (S_lock_start * (V_from_side - V_ship) + V_levelling * S_to_side) / (V_to_side - V_ship)
            dt = self.env.FG.nodes[self.node_2]['Info']['Times'][time[1]] - self.env.FG.nodes[self.node_2]['Info']['Times'][time[0]]
            for t in time:
                S_lock = (S_lock_start * (V_from_side - V_ship) + abs(sum(self.discharge_res[time[0]:t])) * dt * S_to_side) / ((V_from_side+abs(sum(self.discharge_res[time[0]:t])) * dt) - V_ship)
                self.env.FG.nodes[self.node_2]['Info']['Salinity'][t] = S_lock
            V_loss_lev = 0

        else:
            V_levelling = self.lock_length * self.lock_width * (WLev_from_side - WLev_to_side)
            S_lock_final = S_lock_start
            V_loss_lev = V_levelling
        for index in enumerate(self.env.FG.nodes[self.node_2]['Info']['Times'][time_index_stop:]):
            self.env.FG.nodes[self.node_2]['Info']['Salinity'][time_index_stop+index[0]] = S_lock_final
        return V_levelling, S_lock_final, V_loss_lev

    def sailing_out_to_harbour(self, vessel, time_index):
        V_ship = vessel.L*vessel.B*vessel.T_f
        S_lock = self.env.FG.nodes[self.node_2]['Info']['Salinity'][time_index]
        WLev_lock_harbour = self.env.FG.nodes[self.water_level]['Info']['Water level'][time_index]
        S_lock_harbour = self.env.FG.nodes[self.water_level]['Info']['Salinity'][time_index]
        V_lock_harbour = self.lock_length * self.lock_width * (WLev_lock_harbour + self.lock_depth)
        S_lock = (S_lock * (V_lock_harbour - V_ship) + V_ship * S_lock_harbour) / V_lock_harbour
        start_distance = self.lock_length - vessel.lock_dist - 0.5 * vessel.L
        start_time_passing_door = start_distance/vessel.v
        time_index = time_index = bisect.bisect_right(self.env.FG.nodes[self.node_2]['Info']['Times'], self.env.now+start_time_passing_door)-1
        end_time_passing_door = vessel.L/vessel.v + start_time_passing_door
        passing_time_door = np.arange(start_time_passing_door,end_time_passing_door,self.env.FG.nodes[self.node_1]['Info']['Times'][1]-self.env.FG.nodes[self.node_1]['Info']['Times'][0])
        if self.water_level == self.node_1:
            for t in enumerate(passing_time_door):
                self.discharge_saline[time_index+t[0]] += V_ship / (end_time_passing_door-start_time_passing_door)
                self.discharge_res[time_index + t[0]] += V_ship / (end_time_passing_door - start_time_passing_door)
        else:
            for t in enumerate(passing_time_door):
                self.discharge_fresh[time_index+t[0]] += -V_ship / (end_time_passing_door-start_time_passing_door)
                self.discharge_res[time_index + t[0]] += -V_ship / (end_time_passing_door - start_time_passing_door)
        return S_lock

    def door_open_harbour(self, T_door_open, time_index):
        S_lock = self.env.FG.nodes[self.node_2]['Info']['Salinity'][time_index]
        S_lock_harbour = self.env.FG.nodes[self.water_level]['Info']['Salinity'][time_index]
        S_lock_average = (self.env.FG.nodes[self.node_1]['Info']['Salinity'][time_index] + self.env.FG.nodes[self.node_3]['Info']['Salinity'][time_index]) / 2
        WLev_lock_harbour = self.env.FG.nodes[self.water_level]['Info']['Water level'][time_index]
        V_lock_harbour = self.lock_length * self.lock_width * (WLev_lock_harbour + self.lock_depth)
        self.exchange_flux_time_series_calculator(T_door_open,time_index)
        # A loop that breaks at a certain moment assigning discharges to the sluice? (discharge should be separated for positive negative (and maybe sluice gates))
        T_exch = self.lock_length / (0.5 * np.sqrt(self.grav_acc * (0.8 * abs(S_lock_harbour - S_lock) / (1000 + 0.8 * S_lock_average)) * (WLev_lock_harbour + self.lock_depth)))
        V_exch = V_lock_harbour * np.tanh(T_door_open /(2 * T_exch))
        M_exch = (S_lock_harbour - S_lock) * V_exch
        S_lock = (S_lock * V_lock_harbour + M_exch) / V_lock_harbour
        return S_lock

    def sailing_in_from_harbour(self, vessel, distance_from_lineup_area, time_index):
        V_ship = vessel.L * vessel.B * vessel.T_f
        S_lock = self.env.FG.nodes[self.node_2]['Info']['Salinity'][time_index]
        WLev_lock_harbour = self.env.FG.nodes[self.water_level]['Info']['Water level'][time_index]
        V_lock_harbour = self.lock_length * self.lock_width * (WLev_lock_harbour + self.lock_depth)
        S_lock = S_lock
        start_distance = distance_from_lineup_area-vessel.lineup_dist-0.5 * vessel.L
        start_time_passing_door = start_distance / vessel.v
        time_index = time_index = bisect.bisect_right(self.env.FG.nodes[self.node_2]['Info']['Times'],self.env.now + start_time_passing_door) - 1
        end_time_passing_door = vessel.L / vessel.v + start_time_passing_door
        passing_time_door = np.arange(start_time_passing_door, end_time_passing_door,self.env.FG.nodes[self.node_1]['Info']['Times'][1] - self.env.FG.nodes[self.node_1]['Info']['Times'][0])
        if self.water_level == self.node_3:
            for t in enumerate(passing_time_door):
                self.discharge_saline[time_index + t[0]] += V_ship / (end_time_passing_door - start_time_passing_door)
                self.discharge_res[time_index + t[0]] += V_ship / (end_time_passing_door - start_time_passing_door)
        if self.water_level == self.node_1:
            for t in enumerate(passing_time_door):
                self.discharge_fresh[time_index + t[0]] += -V_ship / (end_time_passing_door - start_time_passing_door)
                self.discharge_res[time_index + t[0]] += -V_ship / (end_time_passing_door - start_time_passing_door)
        # Q_avg = (V_ship_upstr + V_exch_inner) / T_door_open
        # V_loss_exch = V_ship_upstr - V_ship_downstr
        # S_avg = -(M_inner_a + M_inner_b + M_inner_c - (V_ship_upstr + V_exch_inner) * S_lock_inner) / (V_ship_downstr + V_exch_inner)
        return S_lock  # Assign S_lock to 'Salinity' of Lock for remaining time steps and store metadata in log?

    def total_ship_volume_in_lock(self):
        volume = 0
        for vessel in self.resource.users:
            volume += vessel.width*vessel.length*vessel.draught
        return volume

    def levelling_time(self, environment, levelling):
        """ Function which calculates the operation time: based on the constant or nearest in the signal of the water level difference

            Input:
                - environment: see init function"""

        def calculate_discharge(lock,z,to_wlev,from_wlev,time_index,t):
            if to_wlev[t[0] + time_index] <= from_wlev[t[0] + time_index]:
                lock.env.FG.nodes[lock.node_2]['Info']['Water level'][time_index + t[0]] = z + to_wlev[t[0] + time_index]
                if lock.water_level == lock.node_3:
                    lock.discharge_res[t[0] + time_index] = -1 * lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                    lock.discharge_fresh[t[0] + time_index] = -1 * lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                else:
                    lock.discharge_res[t[0] + time_index] = lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                    lock.discharge_saline[t[0] + time_index] = lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
            else:
                lock.env.FG.nodes[lock.node_2]['Info']['Water level'][time_index + t[0]] = to_wlev[t[0] + time_index] - z
                if lock.water_level == lock.node_3:
                    lock.discharge_res[t[0] + time_index] = lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                    lock.discharge_saline[t[0] + time_index] = lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                else:
                    lock.discharge_res[t[0] + time_index] = -1 * lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
                    lock.discharge_fresh[t[0] + time_index] = -1 * lock.disch_coeff * lock.opening_area * np.sqrt(2 * 9.81 * z)
            return

        if self.levelling == 'Calculated':
            if self.wlev_dif == 'Calculated':
                times = np.arange(0, 2*3600, self.env.FG.nodes[self.node_1]['Info']['Times'][1]-self.env.FG.nodes[self.node_1]['Info']['Times'][0])
                wlev_outer = self.env.FG.nodes[self.node_1]['Info']['Water level']
                wlev_inner = self.env.FG.nodes[self.node_3]['Info']['Water level']
                time_index = bisect.bisect_right(self.env.FG.nodes[self.node_2]['Info']['Times'], self.env.now)-1
                z = abs(wlev_outer[time_index] - wlev_inner[time_index])
                if self.water_level == self.node_3:
                    from_wlev = wlev_inner
                    to_wlev = wlev_outer
                else:
                    from_wlev = wlev_outer
                    to_wlev = wlev_inner
                calculate_discharge(self,z,to_wlev,from_wlev,time_index,[0,self.env.now])

                for t in enumerate(times):
                    if t[0] == 0:
                        continue
                    if t[0] + time_index >= len(to_wlev) - 1:
                        break

                    dz = -self.disch_coeff * self.opening_area / (self.lock_length * self.lock_width) * np.sqrt(2 * 9.81 * z) * (t[1] - times[t[0] - 1])
                    if self.water_level == self.node_3:
                        dh = to_wlev[t[0] + time_index] - to_wlev[t[0] - 1 + time_index]
                    else:
                        dh = 0
                    z = z + dz - dh
                    calculate_discharge(self, z, to_wlev, from_wlev, time_index, t)
                    if z <= 0.05:
                        levelling_time = t[1]
                        delta_disch = self.discharge_res[t[0] + time_index]-self.discharge_res[t[0] + time_index-1]
                        for ta in enumerate(self.env.FG.nodes[self.node_2]['Info']['Times'][(t[0]+time_index):-1]):
                            new_disch = self.discharge_res[t[0] + time_index + ta[0]] + delta_disch
                            if new_disch != 0 and np.sign(new_disch) == np.sign(self.discharge_res[t[0] + time_index + ta[0]]):
                                self.discharge_res[t[0] + time_index + ta[0]+1] = new_disch
                                levelling_time = t[1]+ta[0]*(self.env.FG.nodes[self.node_1]['Info']['Times'][1]-self.env.FG.nodes[self.node_1]['Info']['Times'][0])
                            self.env.FG.nodes[self.node_2]['Info']['Water level'][ta[0]+(t[0]+1)+time_index] = to_wlev[ta[0]+(t[0]+1)+time_index]
                        break
            else:
                levelling_time = (2*self.lock_width * self.lock_length*math.sqrt(self.wlev_dif)) / (self.disch_coeff * self.opening_area * math.sqrt(2 * self.grav_acc))
        else:
            levelling_time = self.levelling

        time_index = bisect.bisect_right(self.env.FG.nodes[self.node_2]['Info']['Times'], self.env.now) - 1
        V_levelling, S_lock, V_loss_lev = self.levelling_to_harbour(self.total_ship_volume_in_lock(),levelling_time,time_index,self.water_level)
        return levelling_time

    def convert_chamber(self, environment, new_level, number_of_vessels):
        """ Function which converts the lock chamber and logs this event.

            Input:
                - environment: see init function
                - new_level: a string which represents the node and indicates the side at which the lock is currently levelled
                - number_of_vessels: the total number of vessels which are levelled simultaneously"""

        # Close the doors
        self.log_entry("Lock doors closing start", environment.now, number_of_vessels, self.water_level)
        yield environment.timeout(self.doors_close)
        self.log_entry("Lock doors closing stop", environment.now, number_of_vessels, self.water_level)

        if len(self.log['Message']) != 2:
            T_door_open = (self.env.now - self.log['Timestamp'][-3].timestamp())
        else:
            T_door_open = self.env.now-self.simulation_start

        time_index = bisect.bisect_right(self.env.FG.nodes[self.node_2]['Info']['Times'], self.env.now-T_door_open) - 1
        S_lock = self.door_open_harbour(T_door_open, time_index)
        time_index = bisect.bisect_right(self.env.FG.nodes[self.node_2]['Info']['Times'], self.env.now) - 1
        if time_index != len(self.env.FG.nodes[self.node_2]['Info']['Times'])-2:
            for index in range(len(self.env.FG.nodes[self.node_2]['Info']['Water level'][time_index:])):
                self.env.FG.nodes[self.node_2]['Info']['Salinity'][index + time_index] = S_lock

        # Convert the chamber
        self.log_entry("Lock chamber converting start", environment.now, number_of_vessels, self.water_level)

        # Water level will shift
        yield environment.timeout(self.levelling_time(environment,self.levelling))
        self.change_water_level(new_level)
        self.log_entry("Lock chamber converting stop", environment.now, number_of_vessels, self.water_level)
        # Open the doors
        self.log_entry("Lock doors opening start", environment.now, number_of_vessels, self.water_level)
        yield environment.timeout(self.doors_open)
        self.log_entry("Lock doors opening stop", environment.now, number_of_vessels, self.water_level)

    def change_water_level(self, side):
        """ Function which changes the water level in the lock chamber and priorities in queue """

        self.water_level = side

        for request in self.resource.queue:
            request.priority = -1 if request.priority == 0 else 0

            if request.priority == -1:
                self.resource.queue.insert(0, self.resource.queue.pop(self.resource.queue.index(request)))
            else:
                self.resource.queue.insert(-1, self.resource.queue.pop(self.resource.queue.index(request)))

class PassLock():
    """ Mixin class: a collection of functions which are used to pass a lock complex consisting of a waiting area, line-up areas, and lock chambers"""
    @staticmethod
    def approach_waiting_area(vessel,node_waiting_area):
        """ Processes vessels which are approaching a waiting area of a lock complex: 
                if the waiting area is full, vessels will be waiting outside the waiting area for a spot, otherwise if the vessel
                fits within the waiting area the vessels will proceed to the waiting area. 

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_waiting_area: a string which includes the name of the node the waiting area is located in the network """

        #Imports the properties of the waiting area
        waiting_area = vessel.env.FG.nodes[node_waiting_area]["Waiting area"][0]

        #Identifies the index of the node of the waiting area within the route of the vessel
        index_node_waiting_area = vessel.route.index(node_waiting_area)

        #Checks whether the waiting area is the first encountered waiting area of the lock complex
        for node_lineup_area in vessel.route[index_node_waiting_area:]:
            if 'Line-up area' not in vessel.env.FG.nodes[node_lineup_area].keys():
                continue

            #Imports the properties of the line-up areas
            lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
            for lineup_area in lineup_areas:
                if waiting_area.name.split('_')[0] != lineup_area.name.split('_')[0]:
                    continue

                #Determines the current time
                wait_for_waiting_area = vessel.env.now

                #Requests access to the waiting area
                vessel.access_waiting_area = waiting_area.waiting_area[node_waiting_area].request()
                yield vessel.access_waiting_area

                #Calculates and reports the waiting time for entering the waiting area
                if wait_for_waiting_area != vessel.env.now:
                    waiting = vessel.env.now - wait_for_waiting_area
                    vessel.log_entry("Waiting to enter waiting area start", wait_for_waiting_area, 0,
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node_waiting_area)-1]],)
                    vessel.log_entry("Waiting to enter waiting area stop", vessel.env.now, waiting,
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node_waiting_area)-1]],)
                break
            break

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
        for node_lineup_area in vessel.route[index_node_waiting_area:]:
            if 'Line-up area' not in vessel.env.FG.nodes[node_lineup_area].keys():
                continue

            #Imports the properties of the line-up areas
            lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
            for lineup_area in lineup_areas:
                if waiting_area.name.split('_')[0] != lineup_area.name.split('_')[0]:
                    continue

                #Imports the location of the lock chamber of the lock complex
                for node_lock in vessel.route[index_node_waiting_area:]:
                    if 'Lock' not in vessel.env.FG.nodes[node_lock].keys():
                        continue
                    locks = vessel.env.FG.nodes[node_lock]["Lock"]
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
                        if (vessel.L < lock.length.level and vessel.L < lineup_area.length.level and lock.water_level == vessel.route[vessel.route.index(lock_position)-1]):
                            vessel.lock_name = lock.name
                        elif vessel.L < lineup_area.length.level:
                            lock_queue_length.append(lineup_area.length.level)
                        else:
                            lock_queue_length.append(lineup_area.length.capacity)

                    #- else, if the vessel does not fit in the line-up area, the total length of the queued is calculated added with the full length capacity of the line-up area
                    else:
                        line_up_queue_length = lineup_area.length.capacity
                        for q in range(len(lineup_area.length.get_queue)):
                            line_up_queue_length += lineup_area.length.get_queue[q].amount
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

                        access_lineup_length = lineup_area.length.get(lineup_area.length.capacity)
                        lineup_area.length.get_queue[-1].length = vessel.L
                        yield access_lineup_length
                        lineup_area.length.put(lineup_area.length.capacity-vessel.L)

                    def request_access_lock_cycle(total_length_waiting_vessels = 0):
                        """ Processes the request of a vessel to enter a lock cycle (wave of vessels assigned to same lockage), depending
                                on the governing conditions regarding the current situation in the line-up area.
                            This function is used in the access_lineup_area function within the leave_waiting_area function

                            No input required """

                        #- If the line-up area has no queue, the vessel will access the lock cycle
                        if lineup_area.length.get_queue == []:
                            access_lineup_length = lineup_area.length.get(vessel.L)

                        #Else, if there are already preceding vessels waiting in a queue, the vessel will request access to a lock cycle
                        else:
                            #Determines the vessel which has started the last lock cycle
                            for q in reversed(range(len(lineup_area.length.get_queue))):
                                if lineup_area.length.get_queue[q].amount == lineup_area.length.capacity:
                                    break

                            #Calculates the total length of vessels assigned to this lock cycle
                            for q2 in range(q,len(lineup_area.length.get_queue)):
                                total_length_waiting_vessels += lineup_area.length.get_queue[q2].length

                            #If the vessels does not fit in this lock cycle, it will start a new lock cycle
                            if vessel.L > lineup_area.length.capacity - total_length_waiting_vessels:
                                yield from create_new_lock_cycle_and_request_access()

                            #Else, if the vessel does fit in this last lock cycle, it will request a place in this cycle
                            else:
                                access_lineup_length = lineup_area.length.get(vessel.L)

                                #Assigns the length of the vessel to this request
                                lineup_area.length.get_queue[-1].length = vessel.L

                                yield access_lineup_length

                    #Requesting procedure for access to line-up area
                    #- If there area vessels in the line-up area
                    if lineup_area.line_up_area[node_lineup_area].users != []:
                        # - If the vessels fits in the lock cycle right away
                        if vessel.L < (lineup_area.line_up_area[node_lineup_area].users[-1].lineup_dist-0.5*lineup_area.line_up_area[node_lineup_area].users[-1].length):
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
                        if vessel.L < lineup_area.length.level:
                            yield from request_access_lock_cycle()

                        # - Else, if the vessel does not fit in the lock cyle right away
                        else:
                            if lineup_area.length.get_queue == []:
                                yield from create_new_lock_cycle_and_request_access()
                            else:
                                yield from request_access_lock_cycle()

                #Determines the current time
                wait_for_lineup_area = vessel.env.now

                #Assigning the lock chain series with least expected waiting time to the vessel
                vessel.lock_name = []
                lock_queue_length = []
                for count,lock in enumerate(locks):
                    choose_lock_chamber(vessel,lock,node_lock,count,lineup_areas,lock_queue_length)

                #If the function did not yet assign a lock chain series
                if vessel.lock_name == []:
                    vessel.lock_name = lineup_areas[lock_queue_length.index(min(lock_queue_length))].name

                #Request access line-up area which is assigned to the vessel
                if lineup_area.name != vessel.lock_name:
                    continue

                yield from access_lineup_area(vessel,lineup_area)

                #Release of vessel's occupation of the waiting area
                waiting_area.waiting_area[node_waiting_area].release(vessel.access_waiting_area)

                #Calculation of location in line-up area as a distance in [m] from start line-up jetty
                #- If the line-up area is not empty
                if len(lineup_area.line_up_area[node_lineup_area].users) != 0:
                    vessel.lineup_dist = (lineup_area.line_up_area[node_lineup_area].users[-1].lineup_dist -
                                          0.5*lineup_area.line_up_area[node_lineup_area].users[-1].length -
                                          0.5*vessel.L)
                #- Else, if the line-up area is empty
                else:
                    vessel.lineup_dist = lineup_area.length.capacity - 0.5*vessel.L

                #Calculation of the (lat,lon)-coordinates of the assigned position in the line-up area
                vessel.wgs84 = pyproj.Geod(ellps="WGS84")
                [lineup_area_start_lat,
                 lineup_area_start_lon,
                 lineup_area_stop_lat,
                 lineup_area_stop_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x,
                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)+1]]['geometry'].x,
                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)+1]]['geometry'].y]
                fwd_azimuth,_,_ = vessel.wgs84.inv(lineup_area_start_lat, lineup_area_start_lon, lineup_area_stop_lat, lineup_area_stop_lon)
                [vessel.lineup_pos_lat,
                 vessel.lineup_pos_lon,_] = pyproj.Geod(ellps="WGS84").fwd(vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x,
                                                                           vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                                                           fwd_azimuth,vessel.lineup_dist)

                #Formal request of the vessel to access the line-up area assigned to the vessel (always granted)
                vessel.access_lineup_area = lineup_area.line_up_area[node_lineup_area].request()

                #Some attributes are assigned to the vessel's formal request to access the line-up area
                lineup_area.line_up_area[node_lineup_area].users[-1].length = vessel.L
                lineup_area.line_up_area[node_lineup_area].users[-1].id = vessel.id
                [lineup_area.line_up_area[node_lineup_area].users[-1].lineup_pos_lat,
                 lineup_area.line_up_area[node_lineup_area].users[-1].lineup_pos_lon] = [vessel.lineup_pos_lat, vessel.lineup_pos_lon]
                lineup_area.line_up_area[node_lineup_area].users[-1].lineup_dist = vessel.lineup_dist
                lineup_area.line_up_area[node_lineup_area].users[-1].n = len(lineup_area.line_up_area[node_lineup_area].users) #(the number of vessels in the line-up area at the moment)
                lineup_area.line_up_area[node_lineup_area].users[-1].v = 0.5*vessel.v
                lineup_area.line_up_area[node_lineup_area].users[-1].wait_for_next_cycle = False #(a boolean which indicates if the vessel has to wait for a next lock cycle)
                lineup_area.line_up_area[node_lineup_area].users[-1].waited_in_waiting_area = False #(a boolean which indicates if the vessel had to wait in the waiting area)

                #Request of entering the line-up area to assure that vessels will enter the line-up area one-by-one
                vessel.enter_lineup_length = lineup_area.enter_line_up_area[node_lineup_area].request()
                yield vessel.enter_lineup_length

                #Speed reduction in the approach to the line-up area
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
                    for line_up_user in range(len(lineup_area.line_up_area[node_lineup_area].users)):
                        if lineup_area.line_up_area[node_lineup_area].users[line_up_user].id == vessel.id:
                            lineup_area.line_up_area[node_lineup_area].users[line_up_user].waited_in_waiting_area = True
                            break
                break
            break

    def approach_lineup_area(vessel,node_lineup_area):
        """ Processes vessels which are approaching the line-up area of the lock complex:
                determines whether the assigned position in the line-up area (distance in [m]) should be changed as the preceding vessel(s),
                which was/were waiting in the line-up area, has/have of is/are already accessed/accessing the lock.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """

        #Imports the properties of the line-up area the vessel is assigned to
        lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
        [vessel.lineup_pos_lat,vessel.lineup_pos_lon] = [vessel.env.FG.nodes[node_lineup_area]['geometry'].x,vessel.env.FG.nodes[node_lineup_area]['geometry'].y]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name:
                continue

            #Identifies the index of the node of the waiting area within the route of the vessel
            index_node_lineup_area = vessel.route.index(node_lineup_area)
            yield vessel.env.timeout(0)

            #Checks whether the line-up area is the first encountered line-up area of the lock complex
            for node_lock in vessel.route[index_node_lineup_area:]:
                if 'Lock' in vessel.env.FG.nodes[node_lock].keys():
                    locks = vessel.env.FG.nodes[node_lock]["Lock"]

                    for lock in locks:
                        if lock.name != vessel.lock_name:
                            continue

                        def change_lineup_dist(vessel, lineup_area, lock, lineup_dist, q):
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

                            if q == 0 and lineup_area.line_up_area[node_lineup_area].users[q].n != (lineup_area.line_up_area[node_lineup_area].users[q].n-len(lock.resource.users)):
                                lineup_dist = lock.length.capacity - 0.5*vessel.L
                            return lineup_dist

                        #Checks the need to change the position of the vessel within the line-up area
                        for q in range(len(lineup_area.line_up_area[node_lineup_area].users)):
                            if lineup_area.line_up_area[node_lineup_area].users[q].id == vessel.id:

                                #Imports information about the current lock cycle
                                direction = vessel.route[vessel.route.index(node_lock)-1] == lock.node_1
                                lock_door_1_user_priority = 0
                                lock_door_2_user_priority = 0
                                lock_door_1_users = lock.doors_1[lock.node_1].users
                                lock_door_2_users = lock.doors_2[lock.node_3].users

                                if direction and lock_door_2_users != []:
                                    lock_door_2_user_priority = lock.doors_2[lock.node_3].users[0].priority

                                elif not direction and lock_door_1_users != []:
                                    lock_door_1_user_priority = lock.doors_1[lock.node_1].users[0].priority

                                #Decision if position should be changed
                                if direction and lock_door_2_user_priority == -1:
                                    vessel.lineup_dist = change_lineup_dist(vessel, lineup_area, lock, vessel.lineup_dist, q)

                                elif not direction and lock_door_1_user_priority == -1:
                                    vessel.lineup_dist = change_lineup_dist(vessel, lineup_area, lock, vessel.lineup_dist, q)

                                #Calculation of (lat,lon)-coordinates based on (newly) assigned position (line-up distance in [m])
                                [lineup_area_start_lat,
                                 lineup_area_start_lon,
                                 lineup_area_stop_lat,
                                 lineup_area_stop_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x,
                                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)+1]]['geometry'].x,
                                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)+1]]['geometry'].y]

                                fwd_azimuth,_,_ = pyproj.Geod(ellps="WGS84").inv(lineup_area_start_lat,
                                                                                 lineup_area_start_lon,
                                                                                 lineup_area_stop_lat,
                                                                                 lineup_area_stop_lon)
                                [vessel.lineup_pos_lat,
                                 vessel.lineup_pos_lon,_] = pyproj.Geod(ellps="WGS84").fwd(vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x,
                                                                                           vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                                                                           fwd_azimuth,vessel.lineup_dist)

                                #Changes the positional attributes of the vessel as user of the line-up area accordingly
                                lineup_area.line_up_area[node_lineup_area].users[q].lineup_pos_lat = vessel.lineup_pos_lat
                                lineup_area.line_up_area[node_lineup_area].users[q].lineup_pos_lon = vessel.lineup_pos_lon
                                lineup_area.line_up_area[node_lineup_area].users[q].lineup_dist = vessel.lineup_dist
                                break
                        break
                    break
                else:
                    continue
            break

    def leave_lineup_area(vessel,node_lineup_area):
        """ Processes vessels which are waiting in the line-up area of the lock complex:
                requesting access to the lock chamber given the governing phase in the lock cycle of the lock chamber and calculates the
                position within the lock chamber.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """

        #Imports the properties of the line-up area the vessel is assigned to
        lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name:
                continue

            #Identifies the index of the node of the line-up area within the route of the vessel
            index_node_lineup_area = vessel.route.index(node_lineup_area)

            #Checks whether the line-up area is the first encountered line-up area of the lock complex
            for node_lock in vessel.route[index_node_lineup_area:]: #(loops over line-up area)
                if 'Lock' not in vessel.env.FG.nodes[node_lock].keys():
                    continue

                #Imports the properties of the lock chamber the vessel is assigned to
                locks = vessel.env.FG.nodes[node_lock]["Lock"]
                for lock in locks:
                    if lock.name != vessel.lock_name:
                        continue

                    #Alters vessel's approach speed to lock chamber, if vessel didn't have to wait in the waiting area
                    for line_up_user in range(len(lineup_area.line_up_area[node_lineup_area].users)):
                        if (lineup_area.line_up_area[node_lineup_area].users[line_up_user].id == vessel.id and not
                            lineup_area.line_up_area[node_lineup_area].users[line_up_user].waited_in_waiting_area):
                            vessel.v = 0.5*vessel.v
                            break

                    #Imports position of the vessel in the line-up area
                    position_in_lineup_area = shapely.geometry.Point(vessel.lineup_pos_lat,vessel.lineup_pos_lon)

                    #Vessel releases its request to enter the line-up area, made in the waiting area
                    lineup_area.enter_line_up_area[node_lineup_area].release(vessel.enter_lineup_length)

                    #Determines current time
                    wait_for_lock_entry = vessel.env.now

                    #Checks whether the lock chamber is the first encountered lock chamber of the lock complex
                    for node_opposing_lineup_area in vessel.route[(index_node_lineup_area+1):]:
                        if 'Line-up area' not in vessel.env.FG.nodes[node_opposing_lineup_area].keys():
                            continue

                        #Imports the properties of the opposing line-up area the vessel is assigned to
                        opposing_lineup_areas = vessel.env.FG.nodes[node_opposing_lineup_area]["Line-up area"]
                        for opposing_lineup_area in opposing_lineup_areas:
                            if opposing_lineup_area.name == vessel.lock_name:
                                break
                        break

                    def access_lock_chamber(vessel,lineup_area,node_lineup_area,lock,node_lock,opposing_lineup_area,node_opposing_lineup_area,door1,door2):
                        """ Processes vessels which are waiting in the line-up area of the lock complex:
                                determines the current phase within the lock cycle and adjusts the request of the vessel accordingly
                            This function is used in the leave_lineup_area function.

                            Input:
                                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                                - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                                               assigned to the vessel as the lock series with the least expected total waiting time
                                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network
                                - lock: an object within the network which is generated with the IsLock mixin class and
                                        assigned to the vessel as the lock series with the least expected total waiting time
                                - node_lock: a string which includes the name of the node at which the lock chamber is located in the network
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

                        def wait_for_requested_empty_conversion_by_oppositely_directed_vessel():
                            """ Vessel will wait for the lock chamber to be converted without vessels, as it was requested by
                                    the vessel(s) waiting on the other side of the lock chamber. This is programmed by a vessel's
                                    request of the converting while_in_line_up_area-resource of the opposing line-up area with
                                    capacity = 1, yielding a timeout, and immediately releasing the request.

                                No input required. """

                            vessel.waiting_during_converting = opposing_lineup_area.converting_while_in_line_up_area[node_opposing_lineup_area].request()
                            yield vessel.waiting_during_converting
                            opposing_lineup_area.converting_while_in_line_up_area[node_opposing_lineup_area].release(vessel.waiting_during_converting)

                        def request_approach_lock_chamber(timeout_required=True,priority=0):
                            """ Vessel will request if it can enter the lock by requesting access to the first set of lock doors. This
                                    request always has priority = 0, as vessels can only pass these doors when the doors are open (not
                                    claimed by a priority = -1 request). The capacity of the doors equals one, to prevent ships from
                                    entering simultaneously. The function yields a timeout. This can be switched off if it is assured
                                    the vessel can approach immediately.

                                Input:
                                    - timeout_required: a boolean which defines whether the requesting vessel receives a timeout. """

                            vessel.access_lock_door1 = door1.request(priority=priority)
                            if timeout_required:
                                yield vessel.access_lock_door1

                        def request_empty_lock_conversion(hold_request = False):
                            """ Vessel will request the lock chamber to be converted without vessels to his side of the lock chamber. This
                                    is programmed by requesting the converting_while_in_line_up_area resource of the line-up area the vessels is
                                    currently located in. If there was already a request by another vessel waiting in the same line-up area, this
                                    original request can be holded.

                                Input:
                                    - hold_request: a boolean which defines where an earlier request for the same empty lock conversion which
                                                    was made by another ship should be holded"""

                            vessel.waiting_during_converting = lineup_area.converting_while_in_line_up_area[node_lineup_area].request()
                            yield vessel.waiting_during_converting
                            if not hold_request:
                                yield from lock.convert_chamber(vessel.env, vessel.route[vessel.route.index(node_lock)-1], 0)
                            lineup_area.converting_while_in_line_up_area[node_lineup_area].release(vessel.waiting_during_converting)

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

                            vessel.access_lock_door2 = door2.request(priority = priority)
                            if hold_request:
                                door2.release(door2.users[0])
                            if timeout_required:
                                yield vessel.access_lock_door2
                            door2.users[0].id = vessel.id

                        def wait_for_next_lockage():
                            """ Vessels will wait for the next lockage by requesting access to the second pair of lock doors without priority. If
                                    granted, the request will immediately be released.

                                No input required. """

                            yield from secure_lock_cycle(priority = 0)
                            door2.release(vessel.access_lock_door2)

                        #Determines current moment within the lock cycle
                        lock_door_2_user_priority = 0
                        if door2.users != []:
                            lock_door_2_user_priority = door2.users[0].priority

                        #Request procedure of the lock doors, which is dependent on the current moment within the lock cycle:
                        #- If there is a lock cycle being prepared or going on in the same direction of the vessel
                        if lock_door_2_user_priority == -1:
                            #If vessel does not fit in next lock cycle or locking has already started
                            if lock.resource.users != [] and (vessel.L > (lock.resource.users[-1].lock_dist-0.5*lock.resource.users[-1].length) or lock.resource.users[-1].converting):
                                yield from wait_for_next_lockage()

                            #Determines whether an empty conversion is needed or already requested by another vessel going the same way
                            if lineup_area.converting_while_in_line_up_area[node_lineup_area].users != []:
                                yield from request_empty_lock_conversion(hold_request = True)

                            elif len(door2.users) == 0 and len(door1.users) == 0 and vessel.route[vessel.route.index(node_lock)-1] != lock.water_level:
                                yield from request_empty_lock_conversion()

                            # Request to start the lock cycle
                            if not lock.resource.users[-1].converting:
                                yield from request_approach_lock_chamber()
                            else:
                                yield from request_approach_lock_chamber(priority=-1)

                            if door2.users != [] and door2.users[0].priority == -1:
                                yield from secure_lock_cycle(hold_request=True)
                            else:
                                yield from secure_lock_cycle(timeout_required=False)

                        #- If there is a lock cycle being prepared or going on to the direction of the vessel or if the lock chamber is empty
                        else:
                            #If the lock is already converting empty to the other side as requested by a vessel on the opposite side of the lock chamber
                            if opposing_lineup_area.converting_while_in_line_up_area[node_opposing_lineup_area].users != []:
                                yield from wait_for_requested_empty_conversion_by_oppositely_directed_vessel()

                            elif lineup_area.converting_while_in_line_up_area[node_lineup_area].users != []:
                                yield from request_empty_lock_conversion(hold_request=True)

                            #Determining (new) situation
                            #- If the lock chamber is empty
                            if lock.resource.users == []:
                                if door2.users != [] and door2.users[0].priority == -1:
                                    yield from request_approach_lock_chamber()
                                    yield from secure_lock_cycle(hold_request=True)
                                else:
                                    yield from request_approach_lock_chamber(timeout_required=False)
                                    yield from secure_lock_cycle(timeout_required=False)

                                #Determines if an empty lockage is required
                                if vessel.route[vessel.route.index(node_lock)-1] != lock.water_level:
                                    yield from request_empty_lock_conversion()

                            #- Else, if the lock chamber is occupied
                            else:
                                #Request to start the lock cycle
                                if not lock.resource.users[-1].converting:
                                    yield from request_approach_lock_chamber()
                                else:
                                    yield from request_approach_lock_chamber(priority=-1)

                                if door2.users != [] and door2.users[0].priority == -1:
                                    yield from secure_lock_cycle(hold_request=True)
                                else:
                                    yield from secure_lock_cycle(timeout_required=False)

                        #Formal request access to lock chamber and calculate position within the lock chamber
                        lock.length.get(vessel.L)
                        vessel.access_lock = lock.resource.request()

                        lock.pos_length.get(vessel.L)
                        vessel.lock_dist = lock.pos_length.level + 0.5*vessel.L #(distance from first set of doors in [m])

                        #Assign attributes to granted request
                        lock.resource.users[-1].id = vessel.id + '_' + str(vessel.lock_dist)
                        lock.resource.users[-1].name = vessel.type
                        lock.resource.users[-1].length = vessel.L
                        lock.resource.users[-1].width = vessel.B
                        lock.resource.users[-1].draught = vessel.T_f
                        lock.resource.users[-1].lock_dist = vessel.lock_dist
                        lock.resource.users[-1].converting = False #(boolean which indicates if the lock is already converting)

                    #Request access to lock chamber
                    direction = vessel.route[vessel.route.index(node_lock)-1] == lock.node_1 #(determines vessel's direction)
                    if direction:
                        yield from access_lock_chamber(vessel,lineup_area,node_lineup_area,lock,node_lock,opposing_lineup_area,node_opposing_lineup_area,
                                                       lock.doors_1[lock.node_1],lock.doors_2[lock.node_3])
                    if not direction:
                        yield from access_lock_chamber(vessel,lineup_area,node_lineup_area,lock,node_lock,opposing_lineup_area,node_opposing_lineup_area,
                                                       lock.doors_2[lock.node_3],lock.doors_1[lock.node_1])

                    #Releases the vessel's formal request to access the line-up area and releases its occupied length of the line-up area
                    lineup_area.line_up_area[node_lineup_area].release(vessel.access_lineup_area)
                    lineup_area.length.put(vessel.L)

                    #Calculates and reports the total waiting time in the line-up area
                    if wait_for_lock_entry != vessel.env.now:
                        waiting = vessel.env.now - wait_for_lock_entry
                        vessel.log_entry("Waiting in line-up area start", wait_for_lock_entry, 0, position_in_lineup_area)
                        vessel.log_entry("Waiting in line-up area stop", vessel.env.now, waiting, position_in_lineup_area)

                    #Calculation of (lat,lon)-coordinates of assigned position in lock chamber
                    vessel.wgs84 = pyproj.Geod(ellps="WGS84")
                    [node_lineup_area_lat,
                     node_lineup_area_lon,
                     doors_lat,
                     doors_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x,
                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock) - 1]]['geometry'].x,
                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock) - 1]]['geometry'].y]
                    _, _, distance = vessel.wgs84.inv(node_lineup_area_lat,node_lineup_area_lon,doors_lat,doors_lon)
                    [doors_node_lineup_area_lat,
                     doors_node_lineup_area_lon,
                     doors_destination_lat,
                     doors_destination_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)-1]]['geometry'].x,
                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)-1]]['geometry'].y,
                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)+1]]['geometry'].x,
                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)+1]]['geometry'].y]
                    fwd_azimuth,_,_ = vessel.wgs84.inv(doors_node_lineup_area_lat, doors_node_lineup_area_lon, doors_destination_lat, doors_destination_lon)
                    [vessel.lock_pos_lat,vessel.lock_pos_lon,_] = vessel.wgs84.fwd(vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)-1]]['geometry'].x,
                                                                                   vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)-1]]['geometry'].y,
                                                                                   fwd_azimuth,vessel.lock_dist)

                    time_index = bisect.bisect_right(vessel.env.FG.nodes[lock.node_2]['Info']['Times'],vessel.env.now) - 1
                    S_lock = lock.sailing_in_from_harbour(vessel, distance, time_index)
                    for index in range(len(vessel.env.FG.nodes[lock.node_2]['Info']['Water level'][time_index:])):
                        vessel.env.FG.nodes[lock.node_2]['Info']['Salinity'][index + time_index] = S_lock

                    break
                break
            break

    def leave_lock(vessel,node_lock):
        """ Processes vessels which are waiting in the lock chamber to be levelled and after levelling:
                checks if vessels which have entered the lock chamber have to wait for the other vessels to enter the lock chamber and
                requests conversion of the lock chamber.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lock: a string which includes the name of the node at which the lock chamber is located in the network """

        #Imports the properties of the lock chamber the vessel is assigned to
        locks = vessel.env.FG.nodes[node_lock]["Lock"]
        for lock in locks:
            if lock.name != vessel.lock_name:
                continue

            position_in_lock = shapely.geometry.Point(vessel.lock_pos_lat,vessel.lock_pos_lon) #alters node_lock in accordance with given position in lock
            first_user_in_lineup_length = lock.length.capacity

            #Identifies the index of the node of the lock chamber within the route of the vessel
            index_node_lock = vessel.route.index(node_lock)

            #Checks whether the lock chamber is the first encountered lock chamber of the lock complex
            for node_opposing_lineup_area in vessel.route[index_node_lock:]:
                if "Line-up area" not in vessel.env.FG.nodes[node_opposing_lineup_area].keys():
                    continue

                #Imports the properties of the opposing line-up area the vessel is assigned to
                opposing_lineup_areas = vessel.env.FG.nodes[node_opposing_lineup_area]["Line-up area"]
                for opposing_lineup_area in opposing_lineup_areas:
                    if opposing_lineup_area.name != vessel.lock_name:
                        continue
                    break
                break

            #Request access to pass the next line-up area after the lock chamber has levelled, so that vessels will leave the lock chamber one-by-one
            vessel.departure_lock = opposing_lineup_area.pass_line_up_area[node_opposing_lineup_area].request(priority = -1)

            #Releases the vessel's request of their first encountered set of lock doors
            direction = vessel.route[vessel.route.index(node_lock)-1] == lock.node_1
            if direction:
                lock.doors_1[lock.node_1].release(vessel.access_lock_door1)
            elif not direction:
                lock.doors_2[lock.node_3].release(vessel.access_lock_door1)

            #Imports the properties of the line-up area the vessel was assigned to
            for node_lineup_area in reversed(vessel.route[:(index_node_lock-1)]):
                if "Line-up area" not in vessel.env.FG.nodes[node_lineup_area].keys():
                    continue
                lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
                for lineup_area in lineup_areas:
                    if lineup_area.name != vessel.lock_name:
                        continue
                    lineup_users = lineup_area.line_up_area[node_lineup_area].users
                    if lineup_users != []:
                        first_user_in_lineup_length = lineup_area.line_up_area[node_lineup_area].users[0].length

                    #Determines current time and reports this to vessel's log as start time of lock passage
                    start_time_in_lock = vessel.env.now
                    vessel.log_entry("Passing lock start", vessel.env.now, 0, position_in_lock)

                    def waiting_for_other_lock_users(vessel,lock,node_lock,lineup_area,node_lineup_area,door1):
                        """ Function which yields a timeout to vessels as they have to wait for the other vessels to enter the lock.
                                the timeout is calculated by subsequently requesting and releasing the previously passed line-up area
                                and lock doors again without priority, such that the all the vessels within the line-up area can enter
                                the lock before the levelling will start.
                            This function is used in the leave_lock function.

                            Input:
                                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                                - lock: an object within the network which is generated with the IsLock mixin class and is asigned to the
                                        vessel as the lock chamber in the lock chain series with the least expected total waiting time
                                - node_lock: a string which includes the name of the node at which the lock chamber is located in the network
                                - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and is assigned
                                               to the vessel as the line-up area in the lock chain series with the least expected total waiting time
                                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network
                                - door1: an object created in the IsLock class which resembles the set of lock doors which is first encountered by
                                         the vessel, which should be supscripted to, using a string which includes the name of the node at which this
                                         lock door is located in the network and was specified as input in the IsLock class """

                        vessel.access_line_up_area = lineup_area.enter_line_up_area[node_lineup_area].request()
                        yield vessel.access_line_up_area
                        lineup_area.enter_line_up_area[node_lineup_area].release(vessel.access_line_up_area)
                        vessel.access_lock_door1 = door1.request() #therefore it requests first the entering of line-up area and then the lock doors again
                        yield vessel.access_lock_door1
                        door1.release(vessel.access_lock_door1)

                    #Determines if accessed vessel has to wait on accessing vessels
                    if direction and first_user_in_lineup_length < lock.length.level:
                        yield from waiting_for_other_lock_users(vessel,lock,node_lock,lineup_area,node_lineup_area,lock.doors_1[lock.node_1])
                    if not direction and first_user_in_lineup_length < lock.length.level:
                        yield from waiting_for_other_lock_users(vessel,lock,node_lock,lineup_area,node_lineup_area,lock.doors_2[lock.node_3])

                    #Determines if the vessel explicitely has to request the conversion of the lock chamber (only the last entered vessel) or can go with a previously made request
                    if lock.resource.users[-1].id == vessel.id + '_' + str(vessel.lock_dist):
                        lock.resource.users[-1].converting = True
                        number_of_vessels = len(lock.resource.users)
                        yield from lock.convert_chamber(vessel.env, vessel.route[vessel.route.index(node_lock)+1],number_of_vessels)
                    else:
                        for lock_user in range(len(lock.resource.users)):
                            if lock.resource.users[lock_user].id != vessel.id + '_' + str(vessel.lock_dist):
                                continue
                            lock.resource.users[lock_user].converting = True
                            yield vessel.env.timeout(lock.doors_close + lock.levelling_time(vessel.env,lock.levelling) + lock.doors_open)
                            break

                #Yield request to leave the lock chamber
                yield vessel.departure_lock

                time_index = bisect.bisect_right(vessel.env.FG.nodes[lock.node_2]['Info']['Times'],vessel.env.now) - 1
                S_lock = lock.sailing_out_to_harbour(vessel, time_index)
                for index in range(len(vessel.env.FG.nodes[lock.node_2]['Info']['Water level'][time_index:])):
                    vessel.env.FG.nodes[lock.node_2]['Info']['Salinity'][index + time_index] = S_lock

                #Calculates and reports the total locking time
                vessel.log_entry("Passing lock stop", vessel.env.now, vessel.env.now-start_time_in_lock, position_in_lock,)

                #Adjusts position of the vessel in line-up area, apart from the first line-up area also used to position the vessel in the next line-up area, to the origin of the next line-up area
                [vessel.lineup_pos_lat,vessel.lineup_pos_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_opposing_lineup_area)]]['geometry'].x,
                                                                 vessel.env.FG.nodes[vessel.route[vessel.route.index(node_opposing_lineup_area)]]['geometry'].y]
                break
            break

    def leave_opposite_lineup_area(vessel,node_lineup_area):
        """ Processes vessels which have left the lock chamber after levelling and are now in the next line-up area in order to leave the lock complex through the next waiting area:
                release of their requests for accessing their second encountered line-up area and lock chamber.

            Input:
                - vessel: an identity which is Identifiable, Movable,and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """

        #Imports the properties of the line-up area the vessel is assigned to
        lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name: #assure lock chain = assigned chain
                continue
            index_node_lineup_area = vessel.route.index(node_lineup_area)

            #Checks whether the line-up area is the second encountered line-up area of the lock complex
            for node_lock in reversed(vessel.route[:(index_node_lineup_area-1)]):
                if 'Lock' not in vessel.env.FG.nodes[node_lock].keys():
                    continue

                #Imports the properties of the lock chamber the vessel was assigned to
                locks = vessel.env.FG.nodes[node_lock]["Lock"]
                for lock in locks:
                    if lock.name != vessel.lock_name:
                        continue

                    #Releases the vessel's request of their second encountered set of lock doors
                    direction = vessel.route[vessel.route.index(node_lock)+1] == lock.node_3
                    if direction and lock.doors_2[lock.node_3].users[0].id == vessel.id:
                        lock.doors_2[lock.node_3].release(vessel.access_lock_door2)
                    if not direction and lock.doors_1[lock.node_1].users[0].id == vessel.id:
                        lock.doors_1[lock.node_1].release(vessel.access_lock_door2)

                    #Releases the vessel's request to enter the second line-up area
                    opposing_lineup_area = lineup_area
                    node_opposing_lineup_area = node_lineup_area
                    opposing_lineup_area.pass_line_up_area[node_opposing_lineup_area].release(vessel.departure_lock)

                    #Releases the vessel's formal request of the lock chamber and returns its occupied length in the lock chamber
                    lock.resource.release(vessel.access_lock)
                    departure_lock_length = lock.length.put(vessel.L) #put length back in lock
                    departure_lock_pos_length = lock.pos_length.put(vessel.L) #put position length back in lock
                    yield departure_lock_length
                    yield departure_lock_pos_length
                    break
                break
            break