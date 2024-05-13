# package(s) related to the simulation
import simpy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time as timepy
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

    def pass_terminal(self,origin,destination):
        # Terminal
        k = sorted(self.env.FG[origin][destination], key=lambda x: self.env.FG[origin][destination][x]['geometry'].length)[0]
        if 'Terminal' in self.env.FG.edges[origin, destination, k].keys() and self.route[-1] == destination:
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
            if self.move_to_anchorage:
                raise simpy.exceptions.Interrupt('New route determined')

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

    def enter_turning_basin(self, origin):
        if 'Turning Basin' in self.env.FG.nodes[origin].keys():
            turning_basin = self.env.FG.nodes[origin]['Turning Basin'][0]
            if self.bound == 'inbound' and turning_basin.length >= self.L:
                self.update_turing_basin_status_report(turning_basin)
                self.log_entry(self.env.now, self.env.FG.nodes[origin]['geometry'], "Turning start", deepcopy(self.output))
                turning_basin.log_entry(self.env.now, self.env.FG.nodes[origin]['geometry'], "Turning start", deepcopy(turning_basin.output))
                yield self.env.timeout(self.metadata['t_turning'][0])
                self.update_turing_basin_status_report(turning_basin, turning_stop = True)
                self.log_entry(self.env.now, self.env.FG.nodes[origin]['geometry'], "Turning stop", deepcopy(self.output))
                turning_basin.log_entry(self.env.now, self.env.FG.nodes[origin]['geometry'], "Turning stop", deepcopy(turning_basin.output))
            else:
                self.log_entry(self.env.now, self.env.FG.nodes[origin]['geometry'], "Passing Turning Basin", deepcopy(self.output))
                turning_basin.log_entry(self.env.now, self.env.FG.nodes[origin]['geometry'], "Vessel Passing", deepcopy(turning_basin.output))
            if 'request_access_turning_basin' in dir(self):
                turning_basin.resource.release(self.request_access_turning_basin)
                del(self.request_access_turning_basin)

    def request_turning_basin(self, destination):
        if 'Turning Basin' in self.env.FG.nodes[destination].keys():
            turning_basin = self.env.FG.nodes[destination]['Turning Basin'][0]
            if turning_basin.length >= self.L:
                if self.bound == 'inbound':
                    self.request_access_turning_basin = turning_basin.resource.request()
                else:
                    self.request_access_turning_basin = turning_basin.resource.request(priority=-1)
                yield self.request_access_turning_basin

class IsAnchorage(core.HasResource,core.Identifiable, core.Log, output.HasOutput):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(self,capacity,*args,**kwargs):
        super().__init__(capacity = capacity,*args, **kwargs)

class IsTurningBasin(core.HasResource, core.Identifiable, core.Log, output.HasOutput):
    """Mixin class: Something which has a turning basin object properties as part of a lock complex [in SI-units] """

    def __init__(self,information,*args,**kwargs):
        super().__init__(capacity=1,*args, **kwargs)
        self.length = information['Length']

class IsTerminal(core.HasType, core.HasLength, core.HasResource, core.Identifiable, core.Log, output.HasOutput):

    def __init__(self,type,information,*args,**kwargs):

        if type == 'quay':
            self.length = information['Length']
            self.depth = information['Depth']
            self.available_quay_lengths = [[0,0],[0,length]]
            capacity = 100

        elif type == 'jetty':
            self.jetty_depths = information['Depth']
            self.jetty_lengths = information['Length']
            self.length = np.sum(self.jetty_lengths)
            capacity = len(self.jetty_lengths)

        super().__init__(type=type,capacity=capacity,length=self.length, *args, **kwargs)

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

class IsJettyTerminal(core.HasType, core.Identifiable, core.Log, output.HasOutput):

    def __init__(self,env,name,type,information,*args,**kwargs):
        self.resource = PriorityFilterStore(env)
        self.occupying_vessels = simpy.Resource(env, capacity=len(information))
        for berth_name,berth_info in information.iterrows():
            self.resource.put(IsJetty(berth_name,berth_info.Length,berth_info.MBL))
        super().__init__(env=env, name=name, type=type, *args, **kwargs)

    def request_terminal(self,vessel):
        waiting_in_anchorage = False
        kicked_vessel = False
        vessels_in_waiting_area_old = self.resource.get_queue.copy()
        if 'berth_of_call' in vessel.metadata.keys():
            request = self.resource.get_with_priority(vessel,(lambda request: request.name == vessel.metadata['berth_of_call'][0]),priority=vessel.metadata['priority'])
        else:
            request = self.resource.get_with_priority(vessel,(lambda request: request.depth > vessel.T) and (lambda request: request.length > vessel.L),priority=vessel.metadata['priority'])

        vessels_in_waiting_area_new = self.resource.get_queue
        if vessels_in_waiting_area_new != vessels_in_waiting_area_old:
            vessel.waiting_for_available_berth = request
            waiting_in_anchorage = True

        for user in self.occupying_vessels.users:
            if user.vessel.berth.name == vessel.metadata['berth_of_call'][0]:
                if not np.max([0, (user.vessel.etd - vessel.env.now - vessel.env.vessel_traffic_service.provide_sailing_time(vessel, vessel.route[:-1])['Time'].sum())]):
                    if 'accessed_terminal' in dir(user.vessel):
                        self.release_terminal(user.vessel.accessed_terminal.value)
                        del (user.vessel.accessed_terminal)
                        waiting_in_anchorage = False
                        kicked_vessel = True

        return request, waiting_in_anchorage, kicked_vessel

    def release_terminal(self, jetty):
        self.resource.put(jetty)
        return

class PassTerminal:
    """Mixin class: Collection of interacting functions that handle the vessels that call at a terminal and take the correct measures"""

    def move_to_anchorage(vessel,node):
        """ Function: moves a vessel to the anchorage area instead of continuing its route to the terminal if a vessel is required to wait in the anchorage area

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node: a string that contains the node of the route that the vessel is currently on

        """

        # Set some default parameters:
        #vessel.waiting_in_anchorage = True
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
        return

    def pass_anchorage(vessel, node):
        """ Function: function that handles a vessel waiting in an anchorage area

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node: string that contains the node of the route that the vessel is moving to (the anchorage area)

        """
        # Set default parameter and extract information of anchorage area
        anchorage = vessel.env.FG.nodes[node]['Anchorage'][0]
        if 'route_after_anchorage' not in dir(vessel):
            vessel.route_after_anchorage = nx.dijkstra_path(vessel.env.FG,node,vessel.route_after_terminal[0])
            # Request access to the anchorage area and log this to the anchorage area log and vessel log (including the calculated value for the net ukc)
            vessel.anchorage_access = anchorage.resource.request()
            yield vessel.anchorage_access
            vessel.update_waiting_status()
            vessel.update_anchorage_status_report(anchorage)
            anchorage.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], "Vessel arrival", deepcopy(anchorage.output))
            vessel.log_entry(vessel.env.now,nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], "Waiting in anchorage start", deepcopy(vessel.output))
            vessel.arrival_time_in_anchorage = vessel.env.now
        terminal = vessel.env.FG.edges[vessel.route_after_anchorage[-2], vessel.route_after_anchorage[-1], 0]['Terminal'][vessel.metadata['terminal_of_call'][0]]

        if 'additional_waiting_time' in vessel.metadata.keys() and vessel.metadata['additional_waiting_time']:
            vessel.update_waiting_status(priority=True)
            vessel.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]],"Waiting in anchorage for arrival prioritized vessel start", deepcopy(vessel.output))
            yield vessel.env.timeout(vessel.metadata['additional_waiting_time'])
            vessel.update_waiting_status(priority=True, waiting_stop=True)
            vessel.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]],"Waiting in anchorage for arrival prioritized vessel stop", deepcopy(vessel.output))
            vessel.arrival_time_in_anchorage = vessel.env.now
            vessel.metadata['additional_waiting_time'] = 0.
            vessel.route = vessel.route_after_anchorage
            del(vessel.port_accessible)
            vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
            vessel.env.process(vessel.move())
            return

        if 'waiting_for_available_berth' in dir(vessel):
            vessel.update_waiting_status(availability=True)
            vessel.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]],"Waiting for available berth in anchorage stop", deepcopy(vessel.output))
            for user in terminal.occupying_vessels.users:
                if user.vessel.berth.name == vessel.metadata['berth_of_call'][0]:
                    yield vessel.env.timeout(np.max([0,(user.vessel.etd-vessel.env.now-vessel.env.vessel_traffic_service.provide_sailing_time(vessel,vessel.route_after_anchorage[:-1])['Time'].sum())]))
                    if 'accessed_terminal' in dir(user.vessel):
                        terminal.release_terminal(user.vessel.accessed_terminal.value)
                        del(user.vessel.accessed_terminal)
            yield vessel.waiting_for_available_berth
            vessel.etd = vessel.env.now + vessel.metadata['t_turning'][0] + vessel.metadata['t_(un)loading'][0] + 2 * vessel.metadata['t_berthing'] + vessel.env.vessel_traffic_service.provide_sailing_time(vessel, vessel.route_after_anchorage[:-1])['Time'].sum()
            vessel.update_waiting_status(availability=True,waiting_stop=True)
            vessel.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]],"Waiting for available berth in anchorage stop", deepcopy(vessel.output))

        vessel.waiting_for_inbound_tidal_window = vessel.env.vessel_traffic_service.provide_waiting_time_for_inbound_tidal_window(vessel, route=vessel.route_after_anchorage[:-1], delay=0, plot=False)
        if vessel.waiting_for_inbound_tidal_window:
            vessel.update_waiting_status(tidal_window=True)
            vessel.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]],"Waiting for tidal window in anchorage start", deepcopy(vessel.output))
            yield vessel.env.timeout(vessel.waiting_for_inbound_tidal_window) | vessel.env.timeout(vessel.metadata['max_waiting_time']-(vessel.arrival_time_in_anchorage-vessel.env.now))
            vessel.etd += vessel.waiting_for_inbound_tidal_window
            vessel.update_waiting_status(tidal_window=True, waiting_stop=True)
            vessel.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]],"Waiting for tidal window in anchorage stop", deepcopy(vessel.output))

        if vessel.env.now - vessel.arrival_time_in_anchorage > vessel.metadata['max_waiting_time']:
            vessel.port_accessible = False

        # if vessel.port_accessible and 'waiting_for_outbound_tidal_window' not in dir(vessel):
        #     vessel.waiting_for_outbound_tidal_window = vessel.env.vessel_traffic_service.provide_waiting_time_for_outbound_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel.etd, plot=False)
        #     vessel.etd += vessel.waiting_for_outbound_tidal_window
        #     if vessel.waiting_for_outbound_tidal_window >= vessel.metadata['max_waiting_time']:
        #         vessel.port_accessible = False

        if not vessel.port_accessible:
            if vessel.terminal.type == 'jetty':
                vessel.terminal.release_terminal(vessel.access_terminal.value)
            else:
                PassTerminal.release_terminal_access(vessel, vessel.terminal, vessel.route_after_terminal[0], delay=0)
            vessel.route.reverse()
        else:
            vessel.route = vessel.route_after_anchorage
            vessel.berth = vessel.access_terminal.value
            vessel.reservation = terminal.occupying_vessels.request()
            vessel.reservation.vessel = vessel
            yield vessel.reservation

        vessel.update_anchorage_status_report(anchorage,departure=True)
        vessel.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]], "Waiting in anchorage stop", deepcopy(vessel.output))
        #waterway.PassWaterway.release_access_previous_section(vessel, vessel.route[0])
        #yield from waterway.PassWaterway.request_access_next_section(vessel, vessel.route[0])
        vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
        vessel.env.process(vessel.move())
        anchorage.resource.release(vessel.anchorage_access)
        anchorage.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], "Vessel departure", deepcopy(anchorage.output))

    def request_quay_position(vessel, terminal):
        """ Function that claims a length along the quay equal to the length of the vessel itself and calculates the relative position of the vessel along the quay. If there are multiple
            relative positions possible, the vessel claims the first position. If there is no suitable position availalble (vessel does not fit), then it returns the action
            of moving to the anchorage area.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class

        """

        # Set some default parameters
        available_quay_lengths = [0]
        aql = terminal.available_quay_lengths #the current configuration of vessels located at the quay
        index_quay_position = 0
        move_to_anchorage = False

        # Loop over the locations of the current configuration of vessels located at the quay
        for index in range(len(aql)):
            # If the index of the locaton is 0, or if the previous location is the same as the current location (and hence the index of the location is not 0) or if the location is not available (value = 1):
            if index == 0 or aql[index][1] == aql[index - 1][1] or aql[index][0] == 1:
                # Continue, else if its the last index and there is not yet a suitable index found for an available location: return that vessel has to move to anchorage
                if index == len(aql) - 1 and not index_quay_position:
                    move_to_anchorage = True
                continue

            # If there is an available location: append indexes to list
            available_quay_lengths.append(aql[index][1] - aql[index - 1][1])

            # Loop over the list:
            for jndex in range(len(available_quay_lengths)):
                # If there is the available location is suitable (available length of that location is greater than the vessel length): return index and break loop
                if vessel.L <= available_quay_lengths[jndex]:
                    index_quay_position = index
                    break

                # Else: if there were not available locations found: return that vessel has to move to anchorage
                elif jndex == len(available_quay_lengths) - 1 and not index_quay_position:
                    move_to_anchorage = True

            # The index can only still be default if the vessel has to move to the anchorage area: so break the loop then
            if index_quay_position != 0:
                break

        return index_quay_position, move_to_anchorage

    def calculate_quay_length_level(terminal):
        """ Function that keeps track of the maximum length that is available at the quay

            Input:
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class

        """

        # Set default parameters
        aql = terminal.available_quay_lengths
        new_level = np.max(aql)
        available_quay_lengths = [0]

        # Loop over the position indexes
        for index in range(len(aql)):
            # If the index of the locaton is 0, or if the previous location is the same as the current location (and hence the index of the location is not 0) or if the location is not available (value = 1):
            if index == 0 or aql[index][1] == aql[index - 1][1] or aql[index][0] == 1:
                # Continue, else if its the last index and there is not yet a suitable index found for an available location: return that available length is the last one in the list (=0)
                if index == len(aql) - 1:
                    new_level = available_quay_lengths[-1]
                continue

            # If there is an available location: append length to list and return the maximum of the list
            available_quay_lengths.append(aql[index][1] - aql[index - 1][1])
            new_level = np.max(available_quay_lengths)
        return new_level

    def adjust_available_quay_lengths(vessel, terminal, index_quay_position):
        """ Function that adjusts the available quay lenghts and positions given a honored request of a vessel at a given position

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class
                - index_quay_position: quay position index at which the vessel is located at the quay with respect to the other vessels

        """

        # Import the locations of the current configuration of vessels located at the quay
        aql = terminal.available_quay_lengths

        # Determine the current maximum available length of the terminal
        old_level = PassTerminal.calculate_quay_length_level(terminal)

        # If the value of the position index before the honered quay position (start of the available position) is still available (=0), change it to 1
        if aql[index_quay_position - 1][0] == 0:
            aql[index_quay_position - 1][0] = 1

        # If the value of the honered quay position (end of the available position) is still available (=0) and the end of this position equals the start of the position added with the vessel length, change it to 1
        if aql[index_quay_position][0] == 0 and aql[index_quay_position][1] == aql[index_quay_position - 1][1] + vessel.L:
            aql[index_quay_position][0] = 1

        # Else insert a new stopping location in the locations of the current configuration of vessels located at the quay by twice adding the vessel length to the start position of the location, once with a occupied value (=1), followed by a available value (=0)
        else:
            aql.insert(index_quay_position, [1, vessel.L + aql[index_quay_position - 1][1]])
            aql.insert(index_quay_position + 1, [0, vessel.L + aql[index_quay_position - 1][1]])

        # Replace the list of the locations of the current configuration of vessels located at the quay of the terminal
        terminal.available_quay_lengths = aql
        # Calculate the quay position and append to the vessel (mid-length of the vessel + starting length of the position)
        vessel.quay_position = 0.5 * vessel.L + aql[index_quay_position - 1][1]
        # Determine the new current maximum available length of the terminal
        new_level = PassTerminal.calculate_quay_length_level(terminal)
        # If the old level does not equal (is greater than) the new level and the vessel does not have to wait in the anchorage first: then claim the difference between these lengths
        if old_level != new_level and vessel.waiting_in_anchorage != True:
            terminal.length.get(old_level - new_level)
        # Else if the vessel has to wait in the anchorage first: calculate the difference between the lengths corrected by the vessel length to be claimed by the vessel (account for this vessel, so that it has priority over new vessels)
        elif vessel.waiting_in_anchorage == True:
            new_level = old_level-vessel.L-new_level
            # If this difference is negative: give absolute length back to terminal
            if new_level < 0:
                terminal.length.put(-new_level)
            # Else if this difference is positive: claim this length of the terminal
            elif new_level > 0:
                terminal.length.get(new_level)
        return

    def readjust_available_quay_lengths(terminal, position):
        """ Function that readjusts the available quay lenghts and positions given a release of a request of a vessel at a given position

            Input:
                - terminal: the terminal of call of the vessel, created with the IsTerminal-class
                - position: quay position index at which the vessel is located at the quay with respect to the other vessels

        """

        # Import the locations of the current configuration of vessels located at the quay
        aql = terminal.available_quay_lengths
        # Loop over the position indexes
        for index in range(len(aql)):
            # Skip the first position index
            if index == 0:
                continue
            # If the position of the vessel falls within the position bounds in the current configuration: break loop (save index)
            if aql[index - 1][1] < position and aql[index][1] > position:
                break

        # Set both values of these position bounds to zero (available again)
        aql[index - 1][0] = 0
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

    def release_terminal_access(vessel,terminal,index,delay):
        ##TODO: keep track of total length instead of maximum length
        if terminal.type == 'quay':
            if vessel.port_accessible:
                # Determine the old maximum available length of the quay
                old_level = PassTerminal.calculate_quay_length_level(terminal)

                # Readjust the available quay lengths as the vessel is leaving the terminal
                terminal.available_quay_lengths = PassTerminal.readjust_available_quay_lengths(terminal,vessel.quay_position)
                # Calculate the new maximum available quay length
                new_level = PassTerminal.calculate_quay_length_level(terminal)
                # If this length does not equal the current maximum available quay length (is smaller), then put this length back to the quay
                if old_level != new_level:
                    terminal.length.put(new_level - old_level)
                # Give vessel length back to keep track of the total claimed vessel length and log this value and the departure event in the logfile of the terminal, and release the request of the vessel to access the terminal
                terminal.log_entry(vessel.env.now+delay, nx.get_node_attributes(vessel.env.FG, "geometry")[index], "Departure of vessel", deepcopy(terminal.output)[berth.name])
            terminal.resource.release(vessel.accessed_terminal)

        # Else if the terminal is of type 'jetty': adjust number of vessels occupying a jetty, log the departure of this vessel and this number, and release the request of the vessel for the specific jetty
        elif terminal.type == 'jetty':
            if vessel.port_accessible:
                terminal.log_entry(vessel.env.now+delay, nx.get_node_attributes(vessel.env.FG, "geometry")[index], "Departure of vessel", deepcopy(terminal.output)[berth.name])
            terminal.resource.release(vessel.accessed_terminal)

    def request_terminal_access(vessel, edge, node):
        """ Function: function that handles the request of a vessel to access the terminal of call: it lets the vessel move to the correct terminal (quay position and jetty) or moves it to the
            anchorage area to wait on either the terminal (quay or jetty) availability or tidal window

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - edge: list of two strings that form the edge at which the terminal is located
                - node: string that contains the node of the route that the vessel is currently on (either the origin or anchorage area)

        """
        # Set some default parameters
        node_index = vessel.route.index(node)
        terminal = vessel.env.FG.edges[edge]["Terminal"][vessel.metadata['terminal_of_call'][0]]
        vessel.terminal = terminal
        vessel.port_accessible = True
        vessel.move_to_anchorage = False
        vessel.waiting_for_availability_terminal = False
        vessel.bound = 'inbound'

        # Loop over the nodes of the route and determine whether there is turning basin that suits the vessel (dependent on a vessel length restriction)
        k = sorted(vessel.env.FG[vessel.route[0]][vessel.route[1]],key=lambda x: vessel.env.FG[vessel.route[0]][vessel.route[1]][x]['geometry'].length)[0]
        if "Terminal" not in vessel.env.FG.edges[vessel.route[0],vessel.route[1],k]:
            suitable_turning_basin = False
            for basin in vessel.route:
                if 'Turning Basin' not in vessel.env.FG.nodes[basin].keys():
                    continue

                turning_basin = vessel.env.FG.nodes[basin]['Turning Basin'][0]
                if turning_basin.length >= vessel.L:
                    suitable_turning_basin = True
                    break

            if not suitable_turning_basin:
                vessel.port_accessible = False

        # Request terminal access and check tidal window
        if vessel.port_accessible:
            if terminal.type == 'jetty':
                if 'additional_waiting_time' in vessel.metadata.keys() and vessel.metadata['additional_waiting_time']:
                    if ("Terminal" in vessel.env.FG.edges[vessel.route[0], vessel.route[1], k] or vessel.route[0] != '8866969'):
                        vessel.update_waiting_status(priority=True)
                        vessel.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]],"Waiting in terminal for arrival prioritized vessel start", deepcopy(vessel.output))
                        yield vessel.env.timeout(vessel.metadata['additional_waiting_time'])
                        vessel.update_waiting_status(priority=True, waiting_stop=True)
                        vessel.log_entry(vessel.env.now,nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]],"Waiting in terminal for arrival prioritized vessel stop", deepcopy(vessel.output))
                        vessel.metadata['additional_waiting_time'] = 0.
                    else:
                        vessel.move_to_anchorage = True
                        yield from PassTerminal.move_to_anchorage(vessel, node_index)
                        return

                vessel.access_terminal, vessel.waiting_for_availability_terminal, kicked_vessel = terminal.request_terminal(vessel)
                if kicked_vessel:
                    yield vessel.access_terminal
                if ("Terminal" in vessel.env.FG.edges[vessel.route[0], vessel.route[1],k] or vessel.route[0] not in ['8866969','anchorage']) and vessel.waiting_for_availability_terminal:
                    vessel.move_to_anchorage = False
                elif vessel.waiting_for_availability_terminal:
                    vessel.move_to_anchorage = True
                vessel.access_terminal.obj = vessel

            # elif terminal.type == 'quay':
            #     # If the queue of vessels waiting for an available quay length is still empty: request quay position
            #     ##TODO: quays can also be subdivided in areas with different allowable vessel dimensions, this needs to be added
            #     ##TODO: include priority
            #     if terminal.length.get_queue == []:
            #         vessel.index_quay_position,vessel.move_to_anchorage = PassTerminal.request_quay_position(vessel, terminal)
            #         vessel.access_terminal = terminal.length.get(vessel.L)
            #         vessel.status = 'moving to terminal'
            #     # Else if this queue is not empty: vessel has to move to anchorage area (according to FCFS-policy)
            #     else:
            #         vessel.move_to_anchorage = True
            #         vessel.waiting_for_availability_terminal = True
            #         vessel.waiting_time_in_anchorage = vessel.access_terminal = terminal.length.get(vessel.L)
            #
            #     PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)

            # calculate waiting time due to tidal window
            vessel.update_waiting_status()
            if vessel.waiting_for_availability_terminal and ("Terminal" in vessel.env.FG.edges[vessel.route[0], vessel.route[1],k] or vessel.route[0] not in ['8866969','anchorage']):
                #vessel.output['waiting_time'] = {'Priority': pd.Timedelta(0, 's'),'Availability': pd.Timedelta(0, 's'),'Tidal window': pd.Timedelta(0, 's')}
                try:
                    vessel.access_terminal.value
                except:
                    vessel.update_waiting_status(availability=True, waiting_stop=False)
                    vessel.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]],"Waiting in terminal for available berth start", deepcopy(vessel.output))
                    for user in terminal.occupying_vessels.users:
                        if user.vessel.berth.name == vessel.metadata['berth_of_call'][0]:
                            yield vessel.env.timeout(np.max([0,(user.vessel.etd-vessel.env.now-vessel.env.vessel_traffic_service.provide_sailing_time(vessel,vessel.route[:-1])['Time'].sum())]))
                            if 'accessed_terminal' in dir(user.vessel):
                                terminal.release_terminal(user.vessel.accessed_terminal.value)
                                del (user.vessel.accessed_terminal)
                    yield vessel.access_terminal
                    vessel.update_waiting_status(availability=True, waiting_stop=True)
                    vessel.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]],"Waiting in terminal for available berth stop", deepcopy(vessel.output))
                else:
                    yield vessel.access_terminal
                vessel.berth = vessel.access_terminal.value
                vessel.reservation = terminal.occupying_vessels.request()
                vessel.reservation.vessel = vessel
                vessel.waiting_time_in_anchorage = vessel.env.vessel_traffic_service.provide_waiting_time_for_inbound_tidal_window(vessel, route=vessel.route, delay=0, plot=False)
                vessel.etd = vessel.env.now + vessel.metadata['t_turning'][0] + vessel.metadata['t_(un)loading'][0] + 2 * vessel.metadata['t_berthing'] + vessel.env.vessel_traffic_service.provide_sailing_time(vessel, vessel.route[:-1])['Time'].sum()+vessel.waiting_time_in_anchorage
                yield vessel.reservation
                if vessel.waiting_time_in_anchorage:
                    vessel.update_waiting_status(tidal_window=True, waiting_stop=False)
                    vessel.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]], "Waiting in terminal for tidal window start", deepcopy(vessel.output))
                    yield vessel.env.timeout(vessel.waiting_time_in_anchorage)
                    vessel.update_waiting_status(tidal_window=True, waiting_stop=True)
                    vessel.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]], "Waiting in terminal for tidal window stop", deepcopy(vessel.output))

            if not vessel.waiting_for_availability_terminal:
                vessel.waiting_time_in_anchorage = vessel.env.vessel_traffic_service.provide_waiting_time_for_inbound_tidal_window(vessel, route=vessel.route, delay=0, plot=False)
                vessel.etd = vessel.env.now + vessel.metadata['t_turning'][0] + vessel.metadata['t_(un)loading'][0] + 2 * vessel.metadata['t_berthing'] + vessel.env.vessel_traffic_service.provide_sailing_time(vessel, vessel.route[:-1])['Time'].sum() + vessel.waiting_time_in_anchorage
                if vessel.waiting_time_in_anchorage >= vessel.metadata['max_waiting_time']:
                    vessel.port_accessible = False
                    vessel.move_to_anchorage = True
                elif vessel.waiting_time_in_anchorage:
                    vessel.move_to_anchorage = True
                else:
                    vessel.berth = vessel.access_terminal.value
                    vessel.reservation = terminal.occupying_vessels.request()
                    vessel.reservation.vessel = vessel
                    yield vessel.reservation

            if vessel.move_to_anchorage and ("Terminal" in vessel.env.FG.edges[vessel.route[0], vessel.route[1],k] or vessel.route[0] not in ['8866969','anchorage']):
                yield vessel.env.timeout(vessel.waiting_time_in_anchorage)
                vessel.move_to_anchorage = False
                vessel.berth = vessel.access_terminal.value
                vessel.reservation = terminal.occupying_vessels.request()
                vessel.reservation.vessel = vessel
                yield vessel.reservation

            elif vessel.move_to_anchorage:
                if 'Anchorage' not in vessel.env.FG.nodes[vessel.route[0]].keys():
                    yield from PassTerminal.move_to_anchorage(vessel,node_index)
                else:
                    yield from PassTerminal.pass_anchorage(vessel,vessel.route[0])

        if not vessel.move_to_anchorage:
            vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route[-1], vessel.route[0])
            # vessel.waiting_for_outbound_tidal_window = vessel.env.vessel_traffic_service.provide_waiting_time_for_outbound_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel.etd, plot=False)
            # vessel.etd += vessel.waiting_for_outbound_tidal_window
            # if vessel.waiting_for_outbound_tidal_window >= vessel.metadata['max_waiting_time'] and not "Terminal" in vessel.env.FG.edges[vessel.route[0], vessel.route[1],k] and not vessel.route[0] != '8866969':
            #     vessel.port_accessible = False
            #     yield from PassTerminal.move_to_anchorage(vessel, node_index)
            #     terminal.occupying_vessels.release(vessel.reservation)

    def pass_terminal(vessel,edge):
        """ Function: function that handles the vessel at the terminal

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - edge: list of two strings that form the edge at which the terminal is located

        """

        # Import information about the terminal and the corresponding index of the start of the edge at which the terminal is located
        terminal = vessel.env.FG.edges[edge]["Terminal"][vessel.metadata['terminal_of_call'][0]]
        berth = vessel.access_terminal.value
        vessel.accessed_terminal = vessel.access_terminal
        del(vessel.access_terminal)
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
        vessel.update_terminal_berth_status_report(terminal,berth)
        terminal.log_entry(vessel.env.now,nx.get_node_attributes(vessel.env.FG, "geometry")[index], "Arrival of vessel", deepcopy(terminal.output)[berth.name])

        route_to_nearest_turning_basin = []
        for node in vessel.route_after_terminal:
            route_to_nearest_turning_basin.append(node)
            if 'Turning Basin' in vessel.env.FG.nodes[node].keys():
                turning_basin = vessel.env.FG.nodes[node]['Turning Basin'][0]
                if turning_basin.length >= vessel.L:
                    break

        time_of_arrival = vessel.env.now

        # Berthing: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to berth
        vessel.log_entry(vessel.env.now, shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon), "Berthing start", deepcopy(vessel.output))
        yield vessel.env.timeout(vessel.metadata['t_berthing'])
        vessel.log_entry(vessel.env.now, shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon), "Berthing stop", deepcopy(vessel.output))

        # If terminal is part of a junction: release request of this section (vessel is berthed and not in channel/basin)
        #if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
        #    waterway.PassWaterway.release_access_previous_section(vessel, edge[1])

        # Determine the new route of the vessel (depending on whether the vessel came from the anchorage area or sailed to the terminal directly) and changing the direction of the vessel
        if vessel.next_destination[0]:
            route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.destination[0], vessel.next_destination[0])
        else:
            route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.destination[0], vessel.origin)

        if len(route_after_terminal) < 2:
            route_after_terminal = [edge[0], edge[1]]
        k = sorted(vessel.env.FG[route_after_terminal[-1]][route_after_terminal[-2]],key=lambda x: vessel.env.FG[route_after_terminal[-1]][route_after_terminal[-2]][x]['geometry'].length)[0]
        if "Terminal" in vessel.env.FG.edges[route_after_terminal[-1], route_after_terminal[-2], k] and len(vessel.metadata['berth_of_call'])>1:
            del (vessel.port_accessible)
            vessel.metadata['priority'] = -1

        # (Un)loading: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to unload
        vessel.log_entry(vessel.env.now, shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),"(Un)loading start", deepcopy(vessel.output))
        yield vessel.env.timeout(vessel.metadata['t_(un)loading'][0])
        vessel._T -= vessel.metadata['(un)loading'][0]
        vessel.log_entry(vessel.env.now, shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon), "(Un)loading stop", deepcopy(vessel.output))
        vessel.route = route_after_terminal
        vessel.bound = 'outbound'
        vessel.metadata['berth_of_call'] = np.delete(vessel.metadata['berth_of_call'], 0, 0)
        vessel.metadata['terminal_of_call'] = np.delete(vessel.metadata['terminal_of_call'], 0, 0)
        vessel.destination = np.delete(vessel.destination, 0, 0)
        vessel.next_destination = np.delete(vessel.next_destination, 0, 0)
        vessel.metadata['(un)loading'] = np.delete(vessel.metadata['(un)loading'], 0, 0)
        vessel.metadata['t_(un)loading'] = np.delete(vessel.metadata['t_(un)loading'], 0, 0)
        if vessel.route:
            if not vessel.metadata['berth_of_call'].size:
                vessel.route = vessel.route[1:]

        # If there is waiting time: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the waiting time
        if "Terminal" in vessel.env.FG.edges[route_after_terminal[-1], route_after_terminal[-2], k] and len(vessel.metadata['berth_of_call'])>1:
            yield from PassTerminal.request_terminal_access(vessel, [route_after_terminal[-1],route_after_terminal[-2],k], route_after_terminal[-1])

        # elif vessel.waiting_for_outbound_tidal_window:
        #     vessel.log_entry(vessel.env.now, shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon), "Waiting for tidal window start", deepcopy(vessel.output))
        #     yield vessel.env.timeout(np.max([0, vessel.waiting_for_outbound_tidal_window]))
        #     vessel.log_entry(vessel.env.now, shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon), "Waiting for tidal window stop", deepcopy(vessel.output))

        # Deberthing: if the terminal is part of an section, request access to this section first
        #if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
        #    yield from waterway.PassWaterway.request_access_next_section(vessel, edge[0])

        # Deberthing: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to deberth
        vessel.log_entry(vessel.env.now, shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon), "Deberthing start", deepcopy(vessel.output))
        yield vessel.env.timeout(vessel.metadata['t_berthing'])
        vessel.update_terminal_berth_status_report(terminal, berth, departure=True)
        vessel.log_entry(vessel.env.now, shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon), "Deberthing stop", deepcopy(vessel.output))
        terminal.log_entry(vessel.env.now, nx.get_node_attributes(vessel.env.FG, "geometry")[index],"Departure of vessel", deepcopy(terminal.output)[berth.name])

        if 'accessed_terminal' in dir(vessel):
            terminal.release_terminal(vessel.accessed_terminal.value)
            del(vessel.accessed_terminal)
        terminal.occupying_vessels.release(vessel.reservation)

        #terminal.occupying_vessels.release(vessel.reservation)
        # Move vessel to start node of the terminal, release request of this section, and change vessel route by removing the first node of the route (as vessel will already be located in the second node of the route after the move event)
        #waterway.PassWaterway.release_access_previous_section(vessel, edge[0])

        # Initiate move of vessel back to sea, setting a bool of leaving port to true
        if vessel.route:
            vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
            vessel.env.process(vessel.move())
            vessel.leaving_port = True