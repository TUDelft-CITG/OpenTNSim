# package(s) related to the simulation
import simpy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from copy import deepcopy
import datetime
import pytz

# OpenTNSim
from opentnsim import core
#from opentnsim import waterway
from opentnsim import output

# spatial libraries
import pyproj
import shapely.geometry

class IsJetty():
    def __init__(self,name,length,depth):
        self.name = name
        self.length = length
        self.depth = depth

class HasTerminal(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_edge_functions.append(self.pass_terminal)

    def find_berths_in_terminal(self,edge):
        no_terminal = True
        terminal_index = 0
        for terminal_name,terminals in self.env.FG.edges[edge]['Terminal'].items():
            if self.metadata['terminal_of_call'][0] == terminal_name:
                for terminal_index,terminal in enumerate(terminals):
                    berths = self.env.FG.edges[edge]['Terminal'][terminal_name][terminal_index].berths
                    if self.metadata['berth_of_call'][0] in berths:
                        no_terminal = False
                        break

        return no_terminal, terminal_index

    def pass_terminal(self,origin,destination):
        # Terminal
        k = sorted(self.env.FG[origin][destination], key=lambda x: self.env.FG[origin][destination][x]['geometry'].length)[0]
        if 'Terminal' not in self.env.FG.edges[origin, destination, k].keys():
            return

        if not len(self.metadata['terminal_of_call']):
            return

        no_terminal, terminal_index = self.find_berths_in_terminal((origin,destination,k))
        if no_terminal:
            return

        terminal = self.env.FG.edges[origin, destination, k]['Terminal'][self.metadata['terminal_of_call'][0]][terminal_index]
        if self.metadata['berth_of_call'][0] in terminal.berths:
            yield from PassTerminal.pass_terminal(self, [origin, destination,k])
            raise simpy.exceptions.Interrupt('New route determined')

class HasPortAccess(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node_functions.append(self.request_terminal_access)

    def request_terminal_access(self, origin):
        # Request for a terminal
        if 'port_accessible' not in dir(self):
            self.bound = 'inbound'
            u,v = self.route[-2], self.route[-1]
            k = sorted(self.env.FG[u][v], key=lambda x: self.env.FG[u][v][x]['geometry'].length)[0]
            yield from PassTerminal.request_terminal_access(self, [u,v,k], origin)
            if self.move_to_anchorage or not self.port_accessible:
                raise simpy.exceptions.Interrupt('New route determined')
            elif self.route[0] == '8866969':
                self.update_waiting_status(waiting_stop=True)
            elif self.route[0] != '8866969':
                self.update_waiting_status(terminal=True)
                self.update_waiting_status(waiting_stop=True,terminal=True)

class HasAnchorage(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_complete_pass_edge_functions.append(self.pass_anchorage_area)

    def pass_anchorage_area(self,destination):
        # Anchorage
        if 'Anchorage' in self.env.FG.nodes[destination].keys():
            yield from PassTerminal.pass_anchorage(self, destination)
            raise simpy.exceptions.Interrupt('New route determined')

class HasTurningBasin(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node_functions.append(self.enter_turning_basin)
        self.on_look_ahead_to_node_functions.append(self.request_turning_basin)
        self.metadata['turning_basin_of_turn'] = []

    def enter_turning_basin(self, origin):
        if 'Turning basin' in self.env.FG.nodes[origin].keys():
            turning_basins = self.env.FG.nodes[origin]['Turning basin']
            for loc,(name, turning_basin) in enumerate(turning_basins.items()):
                if self.bound == 'outbound' and turning_basin.length >= self.L:
                    self.metadata['turning_basin_of_turn'].append(name)
                    self.update_turing_basin_status_report(turning_basin)
                    self.log_entry_v0("Turning start", self.env.now, deepcopy(self.output), self.env.FG.nodes[origin]['geometry'])
                    turning_basin.log_entry_v0("Turning start", self.env.now, deepcopy(turning_basin.output), self.env.FG.nodes[origin]['geometry'])
                    yield self.env.timeout(self.metadata['t_turning'][0])
                    self.bound = 'inbound'
                    self.update_turing_basin_status_report(turning_basin, turning_stop = True)
                    self.log_entry_v0("Turning stop", self.env.now,  deepcopy(self.output), self.env.FG.nodes[origin]['geometry'])
                    turning_basin.log_entry_v0("Turning stop", self.env.now, deepcopy(turning_basin.output), self.env.FG.nodes[origin]['geometry'])
                    break
                elif loc == len(turning_basins)-1:
                    self.log_entry_v0("Passing Turning Basin", self.env.now, deepcopy(self.output), self.env.FG.nodes[origin]['geometry'])
                    turning_basin.log_entry_v0("Vessel Passing", self.env.now, deepcopy(turning_basin.output), self.env.FG.nodes[origin]['geometry'])
                    break

            if 'request_access_turning_basin' in dir(self):
                turning_basins = self.env.FG.nodes[origin]['Turning basin']
                for name,turning_basin in turning_basins.items():
                    if turning_basin.resource == self.request_access_turning_basin.resource:
                        break

                turning_basin.resource.release(self.request_access_turning_basin)
                del(self.request_access_turning_basin)

    def request_turning_basin(self, destination):
        if 'Turning basin' in self.env.FG.nodes[destination].keys() and self.route.index(destination) != len(self.route)-1:
            turning_basins = self.env.FG.nodes[destination]['Turning basin']
            for name, turning_basin in turning_basins.items():
                if turning_basin.length >= self.L:
                    if self.bound == 'outbound':
                        self.request_access_turning_basin = turning_basin.resource.request()
                    else:
                        self.request_access_turning_basin = turning_basin.resource.request(priority=-1)
                    self.request_access_turning_basin.vessel = self
                    yield self.request_access_turning_basin
                    break

class IsAnchorage(core.HasResource,core.Identifiable, core.Log, output.HasOutput):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(self,capacity,*args,**kwargs):
        super().__init__(capacity = capacity,*args, **kwargs)

class IsTurningBasin(core.HasResource, core.Identifiable, core.Log, output.HasOutput):
    """Mixin class: Something which has a turning basin object properties as part of a lock complex [in SI-units] """

    def __init__(self,information,*args,**kwargs):
        super().__init__(capacity=1,*args, **kwargs)
        self.length = information['Length'].iloc[0]

class IsQuayTerminal(core.Identifiable, core.Log, output.HasOutput):

    def __init__(self,env,name,information,*args,**kwargs):
        self.occupying_vessels = simpy.Resource(env=env, capacity=1000)
        self.quays = {}
        self.name = name
        self.berths = information.index.to_list()
        for berth_name,info in information.iterrows():
            self.quays[berth_name] = core.HasLength(env=env,length=info.Length,init=info.Length)
            self.quays[berth_name].available_quay_lengths = [[0, 0], [0, info.Length]]
        super().__init__(env=env,name=name,*args, **kwargs)

    def release_terminal_access(self,vessel,berth_name=None,delay=0):
        if not berth_name:
            berth_name = vessel.metadata['berth_of_call'][0]
        if vessel.port_accessible:
            # Determine the old maximum available length of the quay
            old_level = self.quays[berth_name].length.level

            #print('releasing', vessel.name, berth_name, self.quays[berth_name].available_quay_lengths,old_level,self.quays[berth_name].length.level)
            # Readjust the available quay lengths as the vessel is leaving the terminal
            self.quays[berth_name].available_quay_lengths = self.readjust_available_quay_lengths(vessel,berth_name=berth_name)
            # Calculate the new maximum available quay length
            new_level = self.calculate_quay_length_level(berth_name)
            # If this length does not equal the current maximum available quay length (is smaller), then put this length back to the quay
            if old_level < new_level:
                yield self.quays[berth_name].length.put(new_level - old_level)
            elif new_level < old_level:
                yield self.quays[berth_name].length.get(old_level - new_level)
            else:
                yield self.env.timeout(0)
            #print('released', vessel.name, berth_name, self.quays[berth_name].available_quay_lengths,new_level,old_level,self.quays[berth_name].length.level)
            if self.quays[berth_name].length.level != new_level:
                pass
            # Give vessel length back to keep track of the total claimed vessel length and log this value and the departure event in the logfile of the terminal, and release the request of the vessel to access the terminal
        self.occupying_vessels.release(vessel.accessed_terminal)

    def request_quay_position(self, vessel):
        """ Function that claims a length along the quay equal to the length of the vessel itself and calculates the relative position of the vessel along the quay. If there are multiple
            relative positions possible, the vessel claims the first position. If there is no suitable position availalble (vessel does not fit), then it returns the action
            of moving to the anchorage area.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class

        """
        # Set some default parameters
        berth_name = vessel.metadata['berth_of_call'][0]
        available_quay_lengths = []
        aql = self.quays[berth_name].available_quay_lengths #the current configuration of vessels located at the quay
        index_quay_position = 0
        move_to_anchorage = False
        #print(vessel.name, 'requesting', self.name, berth_name, aql)
        # Loop over the locations of the current configuration of vessels located at the quay
        for index in range(len(aql)):
            # If the index of the locaton is 0, or if the previous location is the same as the current location (and hence the index of the location is not 0) or if the location is not available (value = 1):
            if not index%2:
                continue

            if aql[index][0] != 0:
                # Continue, else if its the last index and there is not yet a suitable index found for an available location: return that vessel has to move to anchorage
                available_quay_lengths.append(0)
                if index == len(aql) - 1 and not index_quay_position:
                    move_to_anchorage = True
                continue

            # If there is an available location: append indexes to list
            available_quay_lengths.append(aql[index][1] - aql[index - 1][1])
            # Loop over the list:
            for jndex in range(len(available_quay_lengths)):
                # If there is the available location is suitable (available length of that location is greater than the vessel length): return index and break loop
                if vessel.L <= available_quay_lengths[jndex]:
                    index_quay_position = jndex
                    break

                # Else: if there were not available locations found: return that vessel has to move to anchorage
                elif jndex == len(available_quay_lengths) - 1 and not index_quay_position:
                    move_to_anchorage = True

            # The index can only still be default if the vessel has to move to the anchorage area: so break the loop then
            if index_quay_position != 0:
                break

        if len(self.quays[berth_name].length.get_queue):
            move_to_anchorage = True

        return index_quay_position, move_to_anchorage

    def calculate_quay_length_level(self, berth_name):
        """ Function that keeps track of the maximum length that is available at the quay

            Input:
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class

        """
        # Set default parameters
        aql = self.quays[berth_name].available_quay_lengths
        new_level = self.quays[berth_name].length.level
        available_quay_lengths = []

        # Loop over the position indexes
        for index in range(len(aql)):
            # If the index of the locaton is 0, or if the previous location is the same as the current location (and hence the index of the location is not 0) or if the location is not available (value = 1):
            if not index%2:
                continue

            if aql[index][0] != 0:
                # Continue, else if its the last index and there is not yet a suitable index found for an available location: return that vessel has to move to anchorage
                available_quay_lengths.append(0)
                continue

            available_quay_lengths.append(aql[index][1] - aql[index - 1][1])
            new_level = np.max(available_quay_lengths)

        return new_level


    def adjust_available_quay_lengths(self, vessel, berth_name=None):
        """ Function that adjusts the available quay lenghts and positions given a honored request of a vessel at a given position

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class
                - index_quay_position: quay position index at which the vessel is located at the quay with respect to the other vessels

        """
        if not berth_name:
            berth_name = vessel.metadata['berth_of_call'][0]

        # Import the locations of the current configuration of vessels located at the quay
        aql = self.quays[berth_name].available_quay_lengths
        # Determine the current maximum available length of the terminal
        old_level = self.calculate_quay_length_level(berth_name)
        index_quay_position = 2*vessel.index_quay_position
        # If the value of the position index before the honered quay position (start of the available position) is still available (=0), change it to 1
        if aql[index_quay_position][0] == 0:
            aql[index_quay_position][0] = vessel.id

        # If the value of the honered quay position (end of the available position) is still available (=0) and the end of this position equals the start of the position added with the vessel length, change it to 1
        if aql[index_quay_position+1][0] == 0 and aql[index_quay_position + 1][1] == aql[index_quay_position][1] + vessel.L:
            aql[index_quay_position+1][0] = vessel.id

        # Else insert a new stopping location in the locations of the current configuration of vessels located at the quay by twice adding the vessel length to the start position of the location, once with a occupied value (=1), followed by a available value (=0)
        else:
            aql.insert(index_quay_position + 1, [vessel.id, vessel.L + aql[index_quay_position][1]])
            aql.insert(index_quay_position + 2, [0, vessel.L + aql[index_quay_position][1]])

        # Replace the list of the locations of the current configuration of vessels located at the quay of the terminal
        self.quays[berth_name].available_quay_lengths = aql
        # Calculate the quay position and append to the vessel (mid-length of the vessel + starting length of the position)
        vessel.quay_position = 0.5 * vessel.L + aql[index_quay_position - 1][1]
        # Determine the new current maximum available length of the terminal
        new_level = self.calculate_quay_length_level(berth_name)
        #print('requested', vessel.name, berth_name, aql, new_level, old_level)
        return old_level,new_level

    def readjust_available_quay_lengths(self, vessel, berth_name=None, copy=False):
        """ Function that readjusts the available quay lenghts and positions given a release of a request of a vessel at a given position

            Input:
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class
                - position: quay position index at which the vessel is located at the quay with respect to the other vessels

        """
        if not berth_name:
            berth_name = vessel.metadata['berth_of_call'][0]
        position = vessel.index_quay_position + vessel.index_quay_position%2
        # Import the locations of the current configuration of vessels located at the quay
        if not copy:
            aql = self.quays[berth_name].available_quay_lengths
        else:
            aq1 = self.quays[berth_name].available_quay_lengths.copy()
        # Set both values of these position bounds to zero (available again)
        for index,(vessel_id,_) in enumerate(aql):
            if vessel_id == vessel.id:
                aql[index][0] = 0

        # Set a default list of redundant indexes to be removed
        to_remove = []
        # Nested loop over the position indexes
        for index in enumerate(aql):
            for jndex in enumerate(aql):
                # If the two indexes are not equal and the value at position index 1 and index 2 are both zero (available) and the locations of the two indexes are equal: remove the first positional index
                if index[0] != jndex[0] and index[1][0] == 0 and jndex[1][0] == 0 and index[1][1] == jndex[1][1]:
                    to_remove.append(index[0])

        # If there are indexes to be removed, loop over these indexes and remove them
        for index in list(reversed(to_remove)):
            aql.pop(index)

        # Return the locations of the new configuration of vessels located at the quay
        return aql

class PriorityFilterStore(simpy.FilterStore):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_with_priority(self, vessel, filter, priority=0):
        vessels_in_waiting_area_old = self.get_queue.copy()
        request = self.get(filter)
        request.priority = priority
        request.obj = vessel
        if priority and vessels_in_waiting_area_old:
            for number_in_line,waiting_vessels in enumerate(vessels_in_waiting_area_old):
                if not waiting_vessels.priority:
                    break
            self.get_queue.insert(number_in_line, self.get_queue.pop())

        return request

# class IsTerminal(core.Identifiable, core.Log, output.HasOutput)
#
#     def __init__(self, env, name, jetty_berths=[], quay_berths=[], *args, **kwargs):
#         self.terminals = np.append(jetty_berths, quay_berths)
#
#     def find_first_available_berth(vessel):
#         first_available_berth = {}
#         for terminal in self.terminals:
#             if isinstance(terminal,IsJettyTerminal):
#                 for jetty in terminal:
#                     first_available_berth[jetty.name] = jetty.occupying_vessels.resource.users[0].etd
#             elif isinstance(terminal,IsQuayTerminal):
#                 for quay in terminal:
#                     etd = datetime.datetime.min
#                     available_quay_length = self.calculate_quay_length_level(quay.name)
#                     if available_quay_length < vessel.L:
#                         users = quay.occupying_vessels.resource.users
#                         etd_users = [user.etd for user in users]
#                         sorted_users = [x for _, x in sorted(zip(etd_users, users))]
#                         for user in sorted_users:
#                             aql = quay.readjust_available_quay_lengths(user, berth_name=quay.name, copy=True)
#                             available_quay_length = self.calculate_quay_length_level(berth_name)
#                             if available_quay_length > vessel.L:
#                                 etd = user.etd
#                                 break
#                     first_available_berth[quay.name] = user.etd

class IsJettyTerminal(core.Identifiable, core.Log, output.HasOutput):

    def __init__(self,env,name,information,*args,**kwargs):
        self.resource = PriorityFilterStore(env)
        self.occupying_vessels = simpy.Resource(env, capacity=1000)
        self.name = name
        self.berths = information.index.to_list()
        for berth_name,berth_info in information.iterrows():
            self.resource.put(IsJetty(berth_name,berth_info.Length,berth_info.MBL))
        super().__init__(env=env, name=name, *args, **kwargs)

    def request_terminal(self,vessel):
        waiting_in_anchorage = False
        kicked_vessel = False
        vessels_in_waiting_area_old = self.resource.get_queue.copy()
        if 'berth_of_call' in vessel.metadata.keys() and vessel.metadata['berth_of_call'][0]:
            #print(vessel.name,vessel.metadata['berth_of_call'][0])
            request = self.resource.get_with_priority(vessel,(lambda request: request.name == vessel.metadata['berth_of_call'][0]),
                                                      priority=vessel.metadata['priority'])
        else:
            request = self.resource.get_with_priority(vessel,(lambda request: request.depth > vessel.T) and (lambda request: request.length > vessel.L),
                                                      priority=vessel.metadata['priority'])

        vessels_in_waiting_area_new = self.resource.get_queue
        request.vessel = vessel
        if vessels_in_waiting_area_new != vessels_in_waiting_area_old:
            vessel.waiting_for_available_berth = request
            waiting_in_anchorage = True
        if len(self.occupying_vessels.users) and 'accessed_terminal' in dir(self.occupying_vessels.users[-1].vessel):
            if 'waiting_for_available_berth' in dir(vessel):
                del (vessel.waiting_for_available_berth)
            waiting_in_anchorage = False
            vessel.terminal_released = True

        return request, waiting_in_anchorage

    def release_terminal(self, jetty):
        yield self.resource.put(jetty)


class PassTerminal:
    """Mixin class: Collection of interacting functions that handle the vessels that call at a terminal and take the correct measures"""

    def move_to_anchorage(vessel,node):
        """ Function: moves a vessel to the anchorage area instead of continuing its route to the terminal if a vessel is required to wait in the anchorage area

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node: a string that contains the node of the route that the vessel is currently on

        """

        # Set some default parameters:
        node_anchorage = vessel.env.vessel_traffic_service.provide_nearest_anchorage_area(vessel,vessel.route[node])
        # If there is not an available anchorage area: leave the port after entering the anchorage area
        if not node_anchorage:
            vessel.port_accessible = False
            return

        vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route[-1], vessel.route[0])
        yield vessel.env.timeout(0)
        vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[node], node_anchorage)
        vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[node]]
        vessel.env.process(vessel.move())

    def pass_anchorage(vessel, node):
        """ Function: function that handles a vessel waiting in an anchorage area

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node: string that contains the node of the route that the vessel is moving to (the anchorage area)

        """

        # Set default parameter and extract information of anchorage area
        node_index = vessel.route.index(node)
        arrival_time = vessel.env.now
        vessel.arrival_time_in_anchorage = vessel.env.now
        if 'Anchorage' in vessel.env.FG.nodes[node]:
            anchorage = vessel.env.FG.nodes[node]['Anchorage'][0]
        else:
            anchorage = None

        if not vessel.port_accessible:
            return

        if 'route_after_anchorage' not in dir(vessel):
            vessel.route_after_anchorage = nx.dijkstra_path(vessel.env.FG,node,vessel.route_after_terminal[0])
            # Request access to the anchorage area and log this to the anchorage area log and vessel log (including the calculated value for the net ukc)
            if isinstance(anchorage, IsAnchorage):
                vessel.anchorage_access = anchorage.resource.request()
                yield vessel.anchorage_access
                vessel.update_anchorage_status_report(anchorage)
                anchorage.log_entry_v0("Vessel arrival", vessel.env.now, deepcopy(anchorage.output), nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]])
                vessel.log_entry_v0("Waiting in anchorage start", vessel.env.now, deepcopy(vessel.output), nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]])
        _, terminal_index = vessel.find_berths_in_terminal((vessel.route_after_anchorage[-2], vessel.route_after_anchorage[-1], 0))
        terminal = vessel.env.FG.edges[vessel.route_after_anchorage[-2], vessel.route_after_anchorage[-1], 0]['Terminal'][vessel.metadata['terminal_of_call'][0]][terminal_index]

        if 'waiting_for_available_berth' in dir(vessel):
            vessel.log_entry_v0("Waiting for available berth start", vessel.env.now, deepcopy(vessel.output), nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]])
            # if isinstance(terminal,IsJettyTerminal):
            #     for user in terminal.occupying_vessels.users:
            #         if user.vessel.berth.name == vessel.metadata['berth_of_call'][0]:
            #             yield vessel.env.timeout(np.max([0,(user.vessel.etd-vessel.env.now-vessel.env.vessel_traffic_service.provide_sailing_time(vessel,vessel.route_after_anchorage[:-1])['Time'].sum())]))
            #             # if 'accessed_terminal' in dir(user.vessel):
            #             #     terminal.release_terminal(user.vessel.accessed_terminal.value)
            #             #     del(user.vessel.accessed_terminal)
            yield vessel.waiting_for_available_berth
            if isinstance(terminal, IsQuayTerminal):
                vessel.index_quay_position,_ = terminal.request_quay_position(vessel)
                berth_name = vessel.metadata['berth_of_call'][0]
                _,new_level = terminal.adjust_available_quay_lengths(vessel)
                existing_level = terminal.quays[berth_name].length.level
                if new_level > existing_level:
                    yield terminal.quays[berth_name].length.put(new_level-existing_level)
                elif new_level < existing_level:
                    yield terminal.quays[berth_name].length.get(existing_level-new_level)


            vessel.etd = vessel.env.now + vessel.metadata['t_turning'][0] + vessel.metadata['t_(un)loading'][0] + vessel.metadata['t_berthing'][0] + vessel.metadata['t_unberthing'][0] + vessel.env.vessel_traffic_service.provide_sailing_time(vessel, vessel.route_after_anchorage[:-1])['Time'].sum()
            vessel.update_waiting_status(availability=True,waiting_stop=True)
            vessel.log_entry_v0("Waiting for available berth stop", vessel.env.now, deepcopy(vessel.output), nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]])

        if vessel.port_accessible:
            calculate_tidal_window = True
            if 'expected_departure_time_from_anchorage' in dir(vessel):
                calculate_tidal_window = False
                vessel.waiting_for_inbound_tidal_window = np.max([0.0,vessel.expected_departure_time_from_anchorage-arrival_time])
                if not vessel.waiting_for_inbound_tidal_window:
                    calculate_tidal_window = True

            if calculate_tidal_window:
                vessel.waiting_for_inbound_tidal_window = vessel.env.vessel_traffic_service.provide_waiting_time_for_inbound_tidal_window(vessel, route=vessel.route_after_anchorage, delay=0, plot=False)

            if vessel.waiting_for_inbound_tidal_window:
                vessel.log_entry_v0("Waiting for tidal window in anchorage start", vessel.env.now, deepcopy(vessel.output), nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]])
                timeout = np.min([vessel.waiting_for_inbound_tidal_window,vessel.metadata['max_waiting_time']-(vessel.arrival_time_in_anchorage-vessel.env.now)])
                yield vessel.env.timeout(timeout)
                vessel.etd += vessel.waiting_for_inbound_tidal_window
                vessel.update_waiting_status(tidal_window=True, waiting_stop=True)
                vessel.log_entry_v0("Waiting for tidal window in anchorage stop", vessel.env.now, deepcopy(vessel.output), nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]])

        if vessel.env.now - vessel.arrival_time_in_anchorage > vessel.metadata['max_waiting_time']:
            vessel.port_accessible = False

        if not vessel.port_accessible:
            vessel.accessed_terminal = vessel.access_terminal
            del(vessel.access_terminal)
            if isinstance(vessel.terminal,IsJettyTerminal):
                yield from vessel.terminal.release_terminal(vessel.accessed_terminal.value)
            elif isinstance(vessel.terminal,IsQuayTerminal):
                yield from vessel.terminal.release_terminal_access(vessel, delay=0)
            vessel.route.reverse()
        else:
            if isinstance(vessel.terminal, IsJettyTerminal):
                vessel.berth = vessel.access_terminal.value
            vessel.reservation = terminal.occupying_vessels.request()
            vessel.reservation.vessel = vessel
            yield vessel.reservation

        if isinstance(anchorage, IsAnchorage):
            vessel.route = vessel.route_after_anchorage
            vessel.update_anchorage_status_report(anchorage,departure=True)
            vessel.log_entry_v0("Waiting in anchorage stop", vessel.env.now, deepcopy(vessel.output), nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]])
            #waterway.PassWaterway.release_access_previous_section(vessel, vessel.route[0])
            #yield from waterway.PassWaterway.request_access_next_section(vessel, vessel.route[0])
            vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
            vessel.env.process(vessel.move())
            anchorage.resource.release(vessel.anchorage_access)
            anchorage.log_entry_v0("Vessel departure", vessel.env.now, deepcopy(anchorage.output), nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]])

        if not vessel.port_accessible:
            vessel.metadata['terminal_of_call'] = []
            vessel.metadata['berth_of_call'] = []
            yield from PassTerminal.move_to_anchorage(vessel, 0)

    def request_terminal_access(vessel, edge, node, visited_terminal=None):
        """ Function: function that handles the request of a vessel to access the terminal of call: it lets the vessel move to the correct terminal (quay position and jetty) or moves it to the
            anchorage area to wait on either the terminal (quay or jetty) availability or tidal window

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - edge: list of two strings that form the edge at which the terminal is located
                - node: string that contains the node of the route that the vessel is currently on (either the origin or anchorage area)

        """

        # Set some default parameters
        node_index = vessel.route.index(node)
        _, terminal_index = vessel.find_berths_in_terminal(tuple(edge))
        terminal = vessel.env.FG.edges[edge]["Terminal"][vessel.metadata['terminal_of_call'][0]][terminal_index]
        vessel.terminal = terminal
        vessel.port_accessible = True
        vessel.move_to_anchorage = False
        vessel.waiting_for_availability_terminal = False
        # Loop over the nodes of the route and determine whether there is turning basin that suits the vessel (dependent on a vessel length restriction)
        k = sorted(vessel.env.FG[vessel.route[0]][vessel.route[1]],key=lambda x: vessel.env.FG[vessel.route[0]][vessel.route[1]][x]['geometry'].length)[0]

        #TODO: check if vessel needs turning
        # suitable_turning_basin = False
        # for basin in vessel.route:
        #     if 'Turning basin' not in vessel.env.FG.nodes[basin].keys():
        #         continue
        #
        #     turning_basins = vessel.env.FG.nodes[basin]['Turning basin']
        #     for name,turning_basin in turning_basins.items():
        #         if turning_basin.length >= vessel.L:
        #             suitable_turning_basin = True
        #             break
        #
        # if not suitable_turning_basin:
        #     vessel.port_accessible = False
        #     vessel.move_to_anchorage = True

        # Request terminal access and check tidal window
        if vessel.port_accessible:
            if isinstance(terminal,IsJettyTerminal):
                vessel.access_terminal, vessel.move_to_anchorage = terminal.request_terminal(vessel)
                if 'terminal_released' in dir(vessel):
                    if 'accessed_terminal' in dir(vessel) and visited_terminal is not None:
                        print(vessel.name,vessel.accessed_terminal.value)
                        yield from visited_terminal.release_terminal(vessel.accessed_terminal.value)
                        del (vessel.accessed_terminal)
                    del (vessel.terminal_released)
                if not vessel.move_to_anchorage:
                    yield vessel.access_terminal
                vessel.access_terminal.obj = vessel

            elif isinstance(terminal,IsQuayTerminal):
                # If the queue of vessels waiting for an available quay length is still empty: request quay position
                ##TODO: quays can also be subdivided in areas with different allowable vessel dimensions, this needs to be added in a later stage
                ##TODO: include priority in a later stage
                berth_name = vessel.metadata['berth_of_call'][0]
                vessel.index_quay_position, vessel.move_to_anchorage = terminal.request_quay_position(vessel)
                if not vessel.move_to_anchorage:
                    old_level = terminal.quays[berth_name].length.level
                    _,new_level = terminal.adjust_available_quay_lengths(vessel)
                    # If the old level does not equal (is greater than) the new level and the vessel does not have to wait in the anchorage first: then claim the difference between these lengths
                    if new_level < old_level:
                        vessel.access_terminal = terminal.quays[berth_name].length.get(old_level - new_level)
                    elif old_level < new_level:
                        vessel.access_terminal = terminal.quays[berth_name].length.put(new_level - old_level)
                    else:
                        vessel.access_terminal = vessel.env.timeout(0)
                    yield vessel.access_terminal
                else:
                    vessel.waiting_for_available_berth = vessel.access_terminal = terminal.quays[berth_name].length.get(vessel.L)

            #Vessel should use the terminal as anchorage area
            if vessel.move_to_anchorage and vessel.route[0] != '8866969':
                #vessel.output['waiting_time'] = {'Priority': pd.Timedelta(0, 's'),'Availability': pd.Timedelta(0, 's'),'Tidal window': pd.Timedelta(0, 's')}
                vessel.update_waiting_status(waiting_stop=False, terminal=True)
                vessel.log_entry_v0("Waiting in terminal for available berth start", vessel.env.now,
                                    deepcopy(vessel.output),nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]])
                if isinstance(vessel.terminal, IsJettyTerminal):
                    try:
                        vessel.access_terminal.value
                    except:
                        # if isinstance(terminal,IsJettyTerminal):
                        #     for user in terminal.occupying_vessels.users:
                        #         if user.vessel.berth.name == vessel.metadata['berth_of_call'][0]:
                        #             yield vessel.env.timeout(np.max([0,(user.vessel.etd-vessel.env.now-vessel.env.vessel_traffic_service.provide_sailing_time(vessel,vessel.route[:-1])['Time'].sum())]))
                        #             if 'accessed_terminal' in dir(user.vessel):
                        #                 terminal.release_terminal(user.vessel.accessed_terminal.value)
                        #                 del (user.vessel.accessed_terminal)
                        yield vessel.access_terminal
                    else:
                        yield vessel.access_terminal

                elif isinstance(vessel.terminal, IsQuayTerminal):
                    yield vessel.access_terminal
                    vessel.index_quay_position, _ = terminal.request_quay_position(vessel)
                    terminal.adjust_available_quay_lengths(vessel)

                vessel.update_waiting_status(availability=True, waiting_stop=True, terminal=True)
                vessel.log_entry_v0("Waiting in terminal for available berth stop", vessel.env.now,
                                    deepcopy(vessel.output),nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]])

                if isinstance(vessel.terminal, IsJettyTerminal):
                    vessel.berth = vessel.access_terminal.value
                vessel.waiting_time_in_anchorage = vessel.env.vessel_traffic_service.provide_waiting_time_for_inbound_tidal_window(vessel, route=vessel.route, delay=0, plot=False)
                vessel.etd = vessel.env.now + vessel.metadata['t_turning'][0] + vessel.metadata['t_(un)loading'][0] + vessel.metadata['t_berthing'][0] + vessel.metadata['t_unberthing'][0] + vessel.env.vessel_traffic_service.provide_sailing_time(vessel, vessel.route[:-1])['Time'].sum()+vessel.waiting_time_in_anchorage
                if vessel.waiting_time_in_anchorage:
                    vessel.log_entry_v0("Waiting in terminal for tidal window start", vessel.env.now, deepcopy(vessel.output), nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]])
                    yield vessel.env.timeout(vessel.waiting_time_in_anchorage)
                    vessel.update_waiting_status(tidal_window=True, waiting_stop=True, terminal=True)
                    vessel.log_entry_v0("Waiting in terminal for tidal window stop", vessel.env.now, deepcopy(vessel.output), nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]])
                vessel.move_to_anchorage = False

            #Vessel should use the anchorage area offshore
            if vessel.route[0] == '8866969':
                vessel.update_waiting_status(waiting_stop=False)
                if not vessel.move_to_anchorage:
                    vessel.waiting_time_in_anchorage = vessel.env.vessel_traffic_service.provide_waiting_time_for_inbound_tidal_window(vessel, route=vessel.route, delay=0, plot=False)
                    vessel.etd = vessel.env.now + vessel.metadata['t_turning'][0] + vessel.metadata['t_(un)loading'][0] + vessel.metadata['t_berthing'][0] + vessel.metadata['t_unberthing'][0] + vessel.env.vessel_traffic_service.provide_sailing_time(vessel, vessel.route[:-1])['Time'].sum() + vessel.waiting_time_in_anchorage
                    if vessel.waiting_time_in_anchorage >= vessel.metadata['max_waiting_time']:
                        vessel.port_accessible = False
                        vessel.move_to_anchorage = True
                    elif vessel.waiting_time_in_anchorage:
                        vessel.move_to_anchorage = True
                        vessel.expected_departure_time_from_anchorage = vessel.env.now+vessel.waiting_time_in_anchorage
                    else:
                        if isinstance(vessel.terminal, IsJettyTerminal):
                            vessel.berth = vessel.access_terminal.value

            if not vessel.move_to_anchorage:
                vessel.reservation = terminal.occupying_vessels.request()
                vessel.reservation.vessel = vessel
                yield vessel.reservation

            else:
                yield from PassTerminal.move_to_anchorage(vessel,node_index)

        if not vessel.move_to_anchorage:
            vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route[-1], vessel.route[0])

    def pass_terminal(vessel,edge):
        """ Function: function that handles the vessel at the terminal

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - edge: list of two strings that form the edge at which the terminal is located

        """

        # Import information about the terminal and the corresponding index of the start of the edge at which the terminal is located
        reservation = vessel.reservation
        if 'request_access_turning_basin' in dir(vessel):
            turning_basins = vessel.env.FG.nodes[origin]['Turning basin']
            for name, turning_basin in turning_basins.items():
                if turning_basin.resource == vessel.request_access_turning_basin.resource:
                    break
            turning_basin.resource.release(vessel.request_access_turning_basin)
            del (vessel.request_access_turning_basin)

        _, terminal_index = vessel.find_berths_in_terminal(tuple(edge))
        terminal = vessel.env.FG.edges[edge]["Terminal"][vessel.metadata['terminal_of_call'][0]][terminal_index]
        if isinstance(vessel.terminal, IsJettyTerminal):
            vessel.berth = vessel.access_terminal.value

        index = vessel.route[vessel.route.index(edge[1]) - 1]
        vessel.berth_released = False

        ##TODO: create method to determine last sailing distance (start edge to nearest point on edge to berth)
        vessel.wgs84 = pyproj.Geod(ellps="WGS84")

        [edge_start_lat,
         edge_start_lon,
         edge_end_lat,
         edge_end_lon] = [vessel.env.FG.nodes[edge[0]]['geometry'].x,
                          vessel.env.FG.nodes[edge[0]]['geometry'].y,
                          vessel.env.FG.nodes[edge[1]]['geometry'].x,
                          vessel.env.FG.nodes[edge[1]]['geometry'].y]
        edge_fwd_azimuth,_,_ = vessel.wgs84.inv(edge_start_lat,edge_start_lon,edge_end_lat,edge_end_lon)

        [vessel.terminal_pos_lat, vessel.terminal_pos_lon] = [vessel.env.FG.nodes[vessel.route[-1]]['geometry'].x,
                                                              vessel.env.FG.nodes[vessel.route[-1]]['geometry'].y]

        # If the terminal is of type 'quay': log in logfile of terminal keeping track of the available length (by getting the so-called position length)
        vessel.update_terminal_berth_status_report(terminal)
        terminal.log_entry_v0("Arrival of vessel", vessel.env.now, deepcopy(terminal.output), nx.get_node_attributes(vessel.env.FG, "geometry")[index])
        route_to_nearest_turning_basin = []
        for node in vessel.route_after_terminal:
            route_to_nearest_turning_basin.append(node)
            if 'Turning basin' in vessel.env.FG.nodes[node].keys():
                turning_basins = vessel.env.FG.nodes[node]['Turning basin']
                for name, turning_basin in turning_basins.items():
                    if turning_basin.length >= vessel.L:
                        suitable_turning_basin = True
                        break

        time_of_arrival = vessel.env.now

        # Berthing: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to berth
        vessel.log_entry_v0("Berthing start", vessel.env.now, deepcopy(vessel.output), shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon))
        yield vessel.env.timeout(vessel.metadata['t_berthing'][0])
        vessel.log_entry_v0("Berthing stop", vessel.env.now, deepcopy(vessel.output), shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon))

        # If terminal is part of a junction: release request of this section (vessel is berthed and not in channel/basin)
        #if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
        #    waterway.PassWaterway.release_access_previous_section(vessel, edge[1])

        # Determine the new route of the vessel (depending on whether the vessel came from the anchorage area or sailed to the terminal directly) and changing the direction of the vessel
        if vessel.metadata['next_destination'][0]:
            route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.destination, vessel.metadata['next_destination'][0])
        else:
            route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.destination, vessel.origin)

        if len(route_after_terminal) < 2:
            route_after_terminal = [edge[0], edge[1]]
        k = sorted(vessel.env.FG[route_after_terminal[-1]][route_after_terminal[-2]],key=lambda x: vessel.env.FG[route_after_terminal[-1]][route_after_terminal[-2]][x]['geometry'].length)[0]
        if "Terminal" in vessel.env.FG.edges[route_after_terminal[-1], route_after_terminal[-2], k] and len(vessel.metadata['berth_of_call'])>1:
            del (vessel.port_accessible)
            vessel.metadata['priority'] = -1

        # (Un)loading: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to unload
        vessel.log_entry_v0("(Un)loading start", vessel.env.now, deepcopy(vessel.output), shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon))
        yield vessel.env.timeout(vessel.metadata['t_(un)loading'][0])
        vessel._T += vessel.metadata['loading'][0]
        vessel.log_entry_v0("(Un)loading stop", vessel.env.now, deepcopy(vessel.output), shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon))
        vessel.route = route_after_terminal
        visited_terminal_name = vessel.metadata['terminal_of_call'].copy()[0] + ' ' + vessel.metadata['berth_of_call'].copy()[0]
        visited_berth_name = vessel.metadata['berth_of_call'].copy()[0]
        vessel.metadata['berth_of_call'] = np.delete(vessel.metadata['berth_of_call'], 0, 0)
        vessel.metadata['terminal_of_call'] = np.delete(vessel.metadata['terminal_of_call'], 0, 0)
        vessel.destination = vessel.metadata['next_destination'][0]
        vessel.metadata['next_destination'] = np.delete(vessel.metadata['next_destination'], 0, 0)
        vessel.metadata['loading'] = np.delete(vessel.metadata['loading'], 0, 0)
        vessel.metadata['t_(un)loading'] = np.delete(vessel.metadata['t_(un)loading'], 0, 0)
        vessel.accessed_terminal = vessel.access_terminal
        del (vessel.access_terminal)

        # If there is waiting time: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the waiting time
        vessel.update_waiting_status(terminal=True)
        if "Terminal" in vessel.env.FG.edges[route_after_terminal[-1], route_after_terminal[-2], k] and len(vessel.metadata['berth_of_call']):
            yield from PassTerminal.request_terminal_access(vessel, [route_after_terminal[-1],route_after_terminal[-2],k], route_after_terminal[-1], visited_terminal = terminal)
        vessel.update_waiting_status(availability=True, waiting_stop=True, terminal=True)

        #Deberthing: if the terminal is part of an section, request access to this section first
        #if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
        #    yield from waterway.PassWaterway.request_access_next_section(vessel, edge[0])
        vessel.waiting_time = vessel.env.vessel_traffic_service.provide_waiting_time_for_outbound_tidal_window(vessel, route=vessel.route, delay=vessel.metadata['t_unberthing'][-1], plot=False)
        yield vessel.env.timeout(vessel.waiting_time)
        vessel.update_waiting_status(tidal_window=True, waiting_stop=True, terminal=True)

        vessel.request_turning_basin(vessel.route[0])

        # Deberthing: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to deberth
        vessel.log_entry_v0("Deberthing start", vessel.env.now, deepcopy(vessel.output), shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon))
        yield vessel.env.timeout(vessel.metadata['t_unberthing'][0])
        vessel.update_terminal_berth_status_report(terminal, berth_name=visited_terminal_name,departure=True)
        vessel.log_entry_v0("Deberthing stop", vessel.env.now, deepcopy(vessel.output), shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon))
        terminal.log_entry_v0("Departure of vessel", vessel.env.now, deepcopy(terminal.output), nx.get_node_attributes(vessel.env.FG, "geometry")[index])
        vessel.metadata['t_berthing'] = np.delete(vessel.metadata['t_berthing'], 0, 0)
        vessel.metadata['t_unberthing'] = np.delete(vessel.metadata['t_unberthing'], 0, 0)
        vessel.bound = 'outbound'

        if 'accessed_terminal' in dir(vessel):
            if isinstance(terminal,IsJettyTerminal):
                print(vessel.name,vessel.accessed_terminal.value)
                yield from terminal.release_terminal(vessel.accessed_terminal.value)
                del(vessel.accessed_terminal)
            elif isinstance(terminal,IsQuayTerminal):
                yield from terminal.release_terminal_access(vessel,berth_name=visited_berth_name)

        terminal.occupying_vessels.release(reservation)

        # Move vessel to start node of the terminal, release request of this section, and change vessel route by removing the first node of the route (as vessel will already be located in the second node of the route after the move event)
        #waterway.PassWaterway.release_access_previous_section(vessel, edge[0])

        # Initiate move of vessel back to sea, setting a bool of leaving port to true
        if vessel.route:
            vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
            vessel.env.process(vessel.move())
            vessel.leaving_port = True
