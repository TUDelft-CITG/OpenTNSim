# package(s) related to the simulation
import simpy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time as timepy

# OpenTNSim
from opentnsim import core
from opentnsim import waterway

# spatial libraries
import pyproj
import shapely.geometry

class IsJetty():
    def __init__(self,length,depth):
        self.length = length
        self.depth = depth

class HasTerminal(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_edge.append(self.pass_terminal)

    def pass_terminal(self,origin,destination):
        # Terminal
        k = sorted(self.env.FG[origin][destination], key=lambda x: self.env.FG[origin][destination][x]['geometry'].length)[0]
        if 'Terminal' in self.env.FG.edges[origin, destination, k].keys() and self.route[-1] == destination:
            yield from PassTerminal.pass_terminal(self, [origin, destination,k])
            raise simpy.exceptions.Interrupt('New route determined')

class HasPortAccess(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.request_terminal_access)

    def request_terminal_access(self, origin):
        # Request for a terminal
        if "Port Entrance" in self.env.FG.nodes[origin] and 'waiting_time_in_anchorage' not in dir(self) and 'accessibility' not in dir(self):
            self.bound = 'inbound'
            self.terminal_accessed = False
            u,v = self.route[-2], self.route[-1]
            k = sorted(self.env.FG[u][v], key=lambda x: self.env.FG[u][v][x]['geometry'].length)[0]
            yield from PassTerminal.request_terminal_access(self, [u,v,k], origin)
            if self.move_to_anchorage:
                raise simpy.exceptions.Interrupt('New route determined')

class HasAnchorage(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_complete_pass_edge.append(self.pass_anchorage_area)

    def pass_anchorage_area(self,destination):
        # Anchorage
        if 'Anchorage' in self.env.FG.nodes[destination].keys():
            yield from PassTerminal.pass_anchorage(self, destination)
            raise simpy.exceptions.Interrupt('New route determined')

class HasTurningBasin(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.enter_turning_basin)
        self.on_look_ahead_to_node.append(self.request_turning_basin)

    def enter_turning_basin(self, origin):
        if 'Turning Basin' in self.env.FG.nodes[origin].keys():
            turning_basin = self.env.FG.nodes[origin]['Turning Basin'][0]
            ukc = self.env.vessel_traffic_service.provide_ukc_clearance(self, origin)
            if self.bound == 'outbound' and turning_basin.length >= self.L:
                self.log_entry("Vessel Turning Start", self.env.now, ukc, self.env.FG.nodes[origin]['geometry'])
                turning_basin.log_entry("Vessel Turning Start", self.env.now, 0, self.env.FG.nodes[origin]['geometry'])
                yield self.env.timeout(self.metadata['t_t'])
                ukc = self.env.vessel_traffic_service.provide_ukc_clearance(self, origin)
                turning_basin.log_entry("Vessel Turning Stop", self.env.now, self.metadata['t_t'],self.env.FG.nodes[origin]['geometry'])
                self.log_entry("Vessel Turning Stop", self.env.now, ukc, self.env.FG.nodes[origin]['geometry'])
            else:
                self.log_entry("Passing Turning Basin", self.env.now, ukc, self.env.FG.nodes[origin]['geometry'])
                turning_basin.log_entry("Vessel Passing", self.env.now, 0, self.env.FG.nodes[origin]['geometry'])
            turning_basin.resource.release(self.request_access_turning_basin)

    def request_turning_basin(self, destination):
        if 'Turning Basin' in self.env.FG.nodes[destination].keys():
            turning_basin = self.env.FG.nodes[destination]['Turning Basin'][0]
            if turning_basin.length >= self.L:
                if self.bound == 'outbound':
                    self.request_access_turning_basin = turning_basin.resource.request()
                else:
                    self.request_access_turning_basin = turning_basin.resource.request(priority=-1)
                yield self.request_access_turning_basin

class IsPortEntrance(core.SimpyObject,core.Identifiable, core.Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

class IsAnchorage(core.HasResource,core.Identifiable, core.Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
            self,
            capacity,
            *args,
            **kwargs
    ):
        super().__init__(capacity = capacity,*args, **kwargs)

class IsTurningBasin(core.HasResource, core.Identifiable, core.Log):
    """Mixin class: Something which has a turning basin object properties as part of a lock complex [in SI-units] """

    def __init__(
        self,
        information,
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)
        """Initialization"""
        self.length = information['Length']

class IsTerminal(core.HasType, core.HasLength, core.HasResource, core.Identifiable, core.Log):

    def __init__(
        self,
        type,
        information,
        *args,
        **kwargs
    ):

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

        "Initialization"
        super().__init__(type=type,capacity=capacity,length=self.length, *args, **kwargs)

class IsJettyTerminal(core.SimpyObject,core.HasType, core.Identifiable, core.Log):

    def __init__(
            self,
            env,
            name,
            type,
            information,
            *args,
            **kwargs
    ):
        super().__init__(env=env,name=name,type=type,*args,*kwargs)
        self.jetty_depths = information['Depth']
        self.jetty_lengths = information['Length']
        capacity = len(self.jetty_lengths)
        self.resource = simpy.FilterStore(env,capacity)
        for length, depth in zip(self.jetty_lengths,self.jetty_depths):
            self.resource.put(IsJetty(length,depth))

    def request_terminal(self,vessel):
        waiting_in_anchorage = False
        vessels_in_waiting_area_old = self.resource.get_queue
        if vessel.T_f <= 11.85:
            request = self.resource.get((lambda request: request.depth > vessel.T_f) and (lambda request: request.depth <= 11.85) and (lambda request: request.length > vessel.L))
        else:
            request = self.resource.get((lambda request: request.depth > vessel.T_f) and (lambda request: request.length > vessel.L))

        vessels_in_waiting_area_new = self.resource.get_queue
        if vessels_in_waiting_area_new != vessels_in_waiting_area_old:
            waiting_in_anchorage = True

        return request, waiting_in_anchorage

    def release_terminal(self, jetty):
        self.resource.put(jetty)
        return

class PassTerminal:
    """Mixin class: Collection of interacting functions that handle the vessels that call at a terminal and take the correct measures"""

    def waiting_time_for_tidal_window(vessel,route,delay=0,plot=False):
        """ Function: calculates the time that a vessel has to wait depending on the available tidal windows

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routeable, and has VesselProperties
                - route: a list of strings that resemble the route of the vessel (can be different than the vessel.route)
                - delay: a delay that can be included to calculate a future situation
                - plot: bool that specifies if a plot is requested or not

        """

        #Create sub-routes based on anchorage areas on the route
        anchorage_node_indexes = []
        for node in route:
            if 'Anchorage' in vessel.env.FG.nodes[node].keys():
                anchorage_node_indexes.append(route.index(node))

        sub_routes = []
        for index,node_index in enumerate(anchorage_node_indexes):
            if index == 0:
                sub_routes.append(route[0:(node_index+1)])
            else:
                sub_routes.append(route[anchorage_node_indexes[node_index-1]:(node_index + 1)])

        if len(anchorage_node_indexes):
            sub_routes.append(route[anchorage_node_indexes[index]:])
        else:
            sub_routes = [route]

        waiting_times = {}
        waiting_time = 0
        cumulative_waiting_time = 0
        for sub_route in sub_routes:
            route_to_start_sub_route = nx.dijkstra_path(vessel.env.FG,route[0],sub_route[0])
            sailing_time_to_node = vessel.env.vessel_traffic_service.provide_sailing_time(vessel,route_to_start_sub_route)
            cumulative_waiting_time += waiting_time
            new_delay = delay + sailing_time_to_node + cumulative_waiting_time
            accessible_windows, _ = vessel.env.vessel_traffic_service.provide_sail_in_times_tidal_window(vessel,route=sub_route,delay=new_delay,plot=plot)

            # Set default parameters
            waiting_time = 0
            current_time = vessel.env.now+new_delay

            # Loop over the available sail-in (or sail-out) times:
            for t in range(len(accessible_windows)):
                # If the next sail-in time contains a starting condition for a restriction: if it is the last time, then let the vessel wait wait for this time, else continue the loop
                if accessible_windows[t][1] == 'Start':
                    if t == len(accessible_windows)-1:
                        waiting_time = accessible_windows[t][0] - current_time
                        break
                    else:
                        continue
                # If the current time of the vessel is greater or equal to the next sail-in time containing a stopping condition for a restriction and is smaller than the next starting time of a restriction: waiting time = 0
                if current_time >= accessible_windows[t][0]:
                    waiting_time = 0
                    if t == len(accessible_windows)-1 or current_time < accessible_windows[t+1][0]:
                        break
                # Else if the current time of the vessel is smaller or equal to the next sail-in time containing a stopping condition for a restriction
                elif current_time <= accessible_windows[t][0]:
                    # And is smaller than the previous starting time of a restriction: waiting time = 0
                    if current_time < accessible_windows[t-1][0]:
                        waiting_time = 0
                    # Waiting time = next stopping time - current time
                    else:
                        waiting_time = accessible_windows[t][0] - current_time
                    break
                # Else if it is the last time, then let the vessel wait wait for this time
                elif t == len(accessible_windows) - 1:
                    waiting_time = accessible_windows[t][0] - current_time
                    break
                # Else if none of the above conditions hold: continue the loop
                else:
                    continue

            k = sorted(vessel.env.FG[sub_route[0]][sub_route[1]],key=lambda x: vessel.env.FG[sub_route[0]][sub_route[1]][x]['geometry'].length)[0]
            if 'Terminal' in vessel.env.FG.edges[sub_route[0],sub_route[1],k].keys() or 'Anchorage' in vessel.env.FG.nodes[sub_route[0]].keys():
                starting_node = sub_route[0]
            else:
                starting_node = vessel.env.vessel_traffic_service.provide_nearest_anchorage_area(vessel, sub_route[0])

            waiting_times[starting_node] = waiting_time
        # Return the vessel waiting time. If the vessel is not able to return within the maximum waiting time at the terminal: the waiting time is unacceptable and the vessel will return
        return waiting_times

    def move_to_anchorage(vessel,node):
        """ Function: moves a vessel to the anchorage area instead of continuing its route to the terminal if a vessel is required to wait in the anchorage area

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node: a string that contains the node of the route that the vessel is currently on

        """

        # Set some default parameters:
        vessel.waiting_in_anchorage = True
        node_anchorage = vessel.env.vessel_traffic_service.provide_nearest_anchorage_area(vessel,vessel.route[node])

        # If there is not an available anchorage area: leave the port after entering the anchorage area
        if not node_anchorage:
            vessel.accessibility = False
            return

        # Set the route that the vessel will take after calling at the terminal (back to the origin) and after waiting in the anchorage area
        vessel.route_after_anchorage = nx.dijkstra_path(vessel.env.FG, node_anchorage, vessel.route[-1])
        vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route[-1], vessel.origin)
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

        # Set route after anchorage if not yet set
        if 'route_after_anchorage' not in dir(vessel) or (vessel.bound == 'inbound' and (vessel.route_after_anchorage[0] != node or vessel.destination != vessel.route_after_anchorage[-1])):
            vessel.route_after_anchorage = nx.dijkstra_path(vessel.env.FG,node,vessel.route[-1])

        elif vessel.bound == 'outbound' and (vessel.route_after_anchorage[0] != node or vessel.origin != vessel.route_after_anchorage[-1]):
            vessel.route_after_anchorage = nx.dijkstra_path(vessel.env.FG, node, vessel.route[-1])

        # Moves the vessel to the node of the anchorage area
        if node != vessel.route[0]:
            #yield from core.Movable.pass_edge(vessel, vessel.route[vessel.route.index(node) - 1],vessel.route[vessel.route.index(node)])
            # Request access to the anchorage area and log this to the anchorage area log and vessel log (including the calculated value for the net ukc)
            vessel.anchorage_access = anchorage.resource.request()
            yield vessel.anchorage_access
            anchorage.log_entry("Vessel arrival", vessel.env.now, len(anchorage.resource.users),nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
            ukc = vessel.env.vessel_traffic_service.provide_ukc_clearance(vessel,node)
            vessel.log_entry("Waiting in anchorage start", vessel.env.now, ukc,nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )

        if not isinstance(vessel.waiting_time_in_anchorage, dict):
            arrival_time = vessel.env.now
            yield vessel.waiting_time_in_anchorage | vessel.env.timeout(vessel.metadata['max_waiting_time'])
            vessel.bound = 'inbound'

            if vessel.env.now - arrival_time < vessel.metadata['max_waiting_time']:
                vessel.status = 'moving to terminal'
                vessel.waiting_time_in_anchorage = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_anchorage, delay=0, plot=True)
                vessel.etd = vessel.waiting_time_in_anchorage[node] + vessel.env.now + vessel.sailing_time_to_terminal + vessel.metadata['t_l'] + 2 * vessel.metadata['t_b']
                if vessel.waiting_time_in_anchorage and vessel.waiting_time_in_anchorage[node]:
                    vessel.status = 'waiting in anchorage'
                    yield vessel.env.timeout(vessel.waiting_time_in_anchorage[node])
                    vessel.status = 'moving to terminal'
            else:
                vessel.accessibility = False

        else:
            yield vessel.env.timeout(vessel.waiting_time_in_anchorage[node]) | vessel.env.timeout(vessel.metadata['max_waiting_time'])
            vessel.status = 'moving to terminal'

        if vessel.accessibility and 'waiting_time_after_terminal' not in dir(vessel):
            vessel.sailing_time_to_terminal = vessel.env.vessel_traffic_service.provide_sailing_time(vessel,vessel.route_after_anchorage[:-1])
            vessel.bound = 'outbound'
            vessel.waiting_time_after_terminal = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel.metadata['t_t']+vessel.metadata['t_l'] + 2 * vessel.metadata['t_b'] + vessel.sailing_time_to_terminal, plot=True)
            vessel.total_waiting_time_after_terminal = np.sum(list(vessel.waiting_time_after_terminal.values()))
            vessel.bound = 'inbound'

            if vessel.total_waiting_time_after_terminal >= vessel.metadata['max_waiting_time']:
                vessel.accessibility = False

        if not vessel.accessibility:
            if vessel.terminal.type == 'jetty':
                vessel.terminal.release_terminal(vessel.jetty)
            else:
                PassTerminal.release_terminal_access(vessel, vessel.terminal, vessel.route_after_terminal[0], delay=0)
            vessel.route.reverse()
        else:
            vessel.route = vessel.route_after_anchorage

        ukc = vessel.env.vessel_traffic_service.provide_ukc_clearance(vessel, node)
        vessel.log_entry("Waiting in anchorage stop", vessel.env.now, ukc, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]], )
        #waterway.PassWaterway.release_access_previous_section(vessel, vessel.route[0])
        #yield from waterway.PassWaterway.request_access_next_section(vessel, vessel.route[0])
        vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
        vessel.env.process(vessel.move())
        anchorage.resource.release(vessel.anchorage_access)
        anchorage.log_entry("Vessel departure", vessel.env.now, len(anchorage.resource.users),nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]],)

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
            if vessel.accessibility:
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
                terminal.log_entry("Departure of vessel", vessel.env.now+delay, terminal.length.level, nx.get_node_attributes(vessel.env.FG, "geometry")[index], )
            terminal.resource.release(vessel.access_terminal)

        # Else if the terminal is of type 'jetty': adjust number of vessels occupying a jetty, log the departure of this vessel and this number, and release the request of the vessel for the specific jetty
        elif terminal.type == 'jetty':
            if vessel.accessibility:
                terminal.log_entry("Departure of vessel", vessel.env.now+delay, 0,nx.get_node_attributes(vessel.env.FG, "geometry")[index],)
            terminal.resource.release(vessel.access_terminal)

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
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        vessel.terminal = terminal
        vessel.move_to_anchorage = False
        vessel.terminal_requested = False
        vessel.waiting_for_availability_terminal = False
        vessel.terminal_request_accepted = False
        vessel.return_to_sea = False
        vessel.origin = vessel.route[0]
        vessel.destination = vessel.route[-1]

        # Loop over the nodes of the route and determine whether there is turning basin that suits the vessel (dependent on a vessel length restriction)
        vessel.accessibility = True

        suitable_turning_basin = False
        for basin in vessel.route:
            if 'Turning Basin' not in vessel.env.FG.nodes[basin].keys():
                continue

            turning_basin = vessel.env.FG.nodes[basin]['Turning Basin'][0]
            if turning_basin.length >= vessel.L:
                suitable_turning_basin = True
                break

        if not suitable_turning_basin:
            vessel.accessibility = False

        # Calculate the distance to the terminal
        vessel.sailing_time_to_terminal = vessel.env.vessel_traffic_service.provide_sailing_time(vessel,vessel.route[:-1])
        vessel.etd = vessel.env.now + vessel.sailing_time_to_terminal + vessel.metadata['t_l'] + 2 * vessel.metadata['t_b']
        node_anchorage = vessel.env.vessel_traffic_service.provide_nearest_anchorage_area(vessel, node)

        # Request terminal access and check tidal window
        if vessel.accessibility:
            vessel.berth = terminal
            if terminal.type == 'jetty':
                ##TODO: include priority
                vessel.access_terminal,vessel.move_to_anchorage = terminal.request_terminal(vessel)
                vessel.jetty = vessel.access_terminal.value
                vessel.access_terminal.obj = vessel
                vessel.etd = vessel.env.now+vessel.sailing_time_to_terminal+vessel.metadata['t_l']+2*vessel.metadata['t_b']
                vessel.terminal_requested = True
                vessel.status = 'moving to terminal'
                vessel.terminal_request_accepted = True

            elif terminal.type == 'quay':
                # If the queue of vessels waiting for an available quay length is still empty: request quay position
                ##TODO: quays can also be subdivided in areas with different allowable vessel dimensions, this needs to be added
                ##TODO: include priority
                if terminal.length.get_queue == []:
                    vessel.index_quay_position,vessel.move_to_anchorage = PassTerminal.request_quay_position(vessel, terminal)
                    vessel.access_terminal = terminal.length.get(vessel.L)
                    vessel.status = 'moving to terminal'
                    vessel.terminal_requested = True
                    vessel.terminal_request_accepted =  True
                # Else if this queue is not empty: vessel has to move to anchorage area (according to FCFS-policy)
                else:
                    vessel.move_to_anchorage = True
                    vessel.waiting_for_availability_terminal = True
                    vessel.waiting_time_in_anchorage = vessel.access_terminal = terminal.length.get(vessel.L)
                    vessel.terminal_requested = True

                PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)

            # calculate waiting time due to tidal window
            if not vessel.waiting_for_availability_terminal:
                vessel.bound = 'inbound'
                vessel.waiting_time_in_anchorage = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route, delay=0, plot=True)
                vessel.total_waiting_time = np.sum(list(vessel.waiting_time_in_anchorage.values()))
                if terminal.type == 'jetty':
                    vessel.etd += vessel.total_waiting_time
                elif terminal.type == 'quay':
                    vessel.etd += vessel.total_waiting_time

                if vessel.total_waiting_time >= vessel.metadata['max_waiting_time']:
                    vessel.accessibility = False
                    vessel.waiting_time_in_anchorage[node_anchorage] = 0
                    vessel.move_to_anchorage = True

                elif vessel.waiting_time_in_anchorage and vessel.waiting_time_in_anchorage[node_anchorage]:
                    vessel.move_to_anchorage = True

            if vessel.move_to_anchorage:
                vessel.status = 'waiting in anchorage'
                yield from PassTerminal.move_to_anchorage(vessel,node_index)

        # If the vessel does not have to return to sea
        if vessel.accessibility:
            if vessel.move_to_anchorage:
                vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route_after_anchorage[-1],vessel.origin)
            else:
                vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route[-1], vessel.route[0])

        if not vessel.move_to_anchorage:
            vessel.bound = 'outbound'
            vessel.waiting_time_after_terminal = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel.metadata['t_t']+vessel.metadata['t_l'] + 2 * vessel.metadata['t_b'] + vessel.sailing_time_to_terminal, plot=True)
            vessel.total_waiting_time_after_terminal = np.sum(list(vessel.waiting_time_after_terminal.values()))
            vessel.bound = 'inbound'
            if vessel.total_waiting_time_after_terminal >= vessel.metadata['max_waiting_time']:
                vessel.accessibility = False
                yield from PassTerminal.move_to_anchorage(vessel, node_index)


    def pass_terminal(vessel,edge):
        """ Function: function that handles the vessel at the terminal

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - edge: list of two strings that form the edge at which the terminal is located

        """

        # Import information about the terminal and the corresponding index of the start of the edge at which the terminal is located
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        index = vessel.route[vessel.route.index(edge[1]) - 1]
        vessel.etd = vessel.env.now + vessel.metadata['t_l'] + 2 * vessel.metadata['t_b']
        vessel.berth_released = False

        # Calculate the location of the assigned position along the edge at which the terminal is located and finish the pass_edge move
        vessel.terminal_accessed = True

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

        orig = nx.get_node_attributes(vessel.env.FG, "geometry")[edge[0]]
        dest = shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon)
        distance = vessel.wgs84.inv(shapely.geometry.asShape(orig).x,
                                    shapely.geometry.asShape(orig).y,
                                    shapely.geometry.asShape(dest).x,
                                    shapely.geometry.asShape(dest).y, )[2]

        ukc = vessel.env.vessel_traffic_service.provide_ukc_clearance(vessel, edge[0])
        vessel.log_entry("Sailing from node {} to node {} start".format(edge[0], edge[1]), vessel.env.now, ukc, orig, )
        yield vessel.env.timeout(distance / vessel.current_speed)
        ukc = vessel.env.vessel_traffic_service.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Sailing from node {} to node {} stop".format(edge[0], edge[1]), vessel.env.now, ukc, dest, )

        # If the terminal is of type 'quay': log in logfile of terminal keeping track of the available length (by getting the so-called position length)
        if terminal.type == 'quay':
            terminal.log_entry("Arrival of vessel", vessel.env.now, terminal.length.level,nx.get_node_attributes(vessel.env.FG, "geometry")[index], )
        # Else if terminal is of type 'jetty': log in logfile of terminal calculating keeping track of number of jetties to be occupied
        elif terminal.type == 'jetty':
            terminal.log_entry("Arrival of vessel", vessel.env.now, 0,nx.get_node_attributes(vessel.env.FG, "geometry")[index], )

        # Calculate if the vessel has to wait due to be ready for departure beyond an available tidal window
        vessel.waiting_time = 0
        if vessel.waiting_time_after_terminal:
            vessel.waiting_time = vessel.waiting_time_after_terminal[edge[1]]
        vessel.etd += vessel.waiting_time

        route_to_nearest_turning_basin = []
        for node in vessel.route_after_terminal:
            route_to_nearest_turning_basin.append(node)
            if 'Turning Basin' in vessel.env.FG.nodes[node].keys():
                turning_basin = vessel.env.FG.nodes[node]['Turning Basin'][0]
                if turning_basin.length >= vessel.L:
                    break

        sailing_times_to_anchorages = []
        for node_anchorage in vessel.env.FG.nodes:
            if 'Anchorage' in vessel.env.FG.nodes[node_anchorage]:
                #Extract information over the individual anchorage areas: capacity, users, and the sailing distance to the anchorage area from the designated terminal the vessel is planning to call
                route_from_anchorage = nx.dijkstra_path(vessel.env.FG, node_anchorage, vessel.route[-1])
                sailing_time_to_anchorage = vessel.env.vessel_traffic_service.provide_sailing_time(vessel,route_from_anchorage)
                sailing_times_to_anchorages.append(sailing_time_to_anchorage)

        sailing_time_to_nearest_turning_basin = vessel.env.vessel_traffic_service.provide_sailing_time(vessel,route_to_nearest_turning_basin)

        if vessel.berth.type == 'jetty':
            if vessel.berth.resource.get_queue:
                queued_vessel = vessel.berth.resource.queue[0]
                total_time = vessel.waiting_time + vessel.metadata['t_l'] + 2 * vessel.metadata['t_b'] + vessel.metadata['t_t'] + sailing_time_to_nearest_turning_basin - queued_vessel.obj.sailing_time_to_terminal
            else:
                total_time = vessel.waiting_time + vessel.metadata['t_l'] + 2 * vessel.metadata['t_b'] + vessel.metadata['t_t'] + sailing_time_to_nearest_turning_basin - np.min(sailing_times_to_anchorages)
        elif vessel.berth.type == 'terminal':
            if vessel.berth.length.get_queue:
                queued_vessel = vessel.berth.length.get_queue[0]
                total_time = vessel.waiting_time + vessel.metadata['t_l'] + 2 * vessel.metadata['t_b'] + vessel.metadata['t_t'] + sailing_time_to_nearest_turning_basin - queued_vessel.obj.sailing_time_to_terminal
            else:
                total_time = vessel.waiting_time + vessel.metadata['t_l'] + 2 * vessel.metadata['t_b'] + vessel.metadata['t_t'] + sailing_time_to_nearest_turning_basin - np.min(sailing_times_to_anchorages)

        time_of_arrival = vessel.env.now

        # Berthing: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to berth
        ukc = vessel.env.vessel_traffic_service.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Berthing start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_b'])
        ukc = vessel.env.vessel_traffic_service.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Berthing stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # If terminal is part of a junction: release request of this section (vessel is berthed and not in channel/basin)
        #if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
        #    waterway.PassWaterway.release_access_previous_section(vessel, edge[1])

        # (Un)loading: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to unload
        ukc = vessel.env.vessel_traffic_service.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("(Un)loading start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        yield vessel.env.timeout(vessel.metadata['t_l']) | vessel.env.timeout(total_time-vessel.metadata['t_b'])
        if vessel.env.now == time_of_arrival + total_time:
            remaining_waiting_time = vessel.metadata['t_l']-(total_time-vessel.metadata['t_b'])
            if terminal.type == 'jetty':
                terminal.release_terminal(vessel.jetty)
            else:
                PassTerminal.release_terminal_access(vessel, terminal, index, remaining_waiting_time)
            yield vessel.env.timeout(remaining_waiting_time)
            vessel.berth_released = True
        ukc = vessel.env.vessel_traffic_service.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("(Un)loading stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Determine the new route of the vessel (depending on whether the vessel came from the anchorage area or sailed to the terminal directly) and changing the direction of the vessel
        vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[vessel.route.index(edge[1])], vessel.origin)

        # If there is waiting time: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the waiting time
        if vessel.waiting_time:
            if vessel.waiting_time-vessel.metadata['t_b'] > 0:
                ukc = vessel.env.vessel_traffic_service.provide_ukc_clearance(vessel, edge[1])
                vessel.log_entry("Waiting for tidal window start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
                if not vessel.berth_released:
                    yield vessel.env.timeout(np.max([0,vessel.waiting_time])) | vessel.env.timeout(total_time-vessel.metadata['t_b']-vessel.metadata['t_l'])
                    if vessel.env.now == time_of_arrival + total_time:
                        remaining_waiting_time = np.max([0,vessel.waiting_time]) - (total_time-vessel.metadata['t_b']-vessel.metadata['t_l'])
                        if terminal.type == 'jetty':
                            terminal.release_terminal(vessel.jetty)
                        else:
                            PassTerminal.release_terminal_access(vessel, terminal, index, remaining_waiting_time)
                        yield vessel.env.timeout(remaining_waiting_time)
                        vessel.berth_released = True
                else:
                    yield vessel.env.timeout(np.max([0, vessel.waiting_time]))
                ukc = vessel.env.vessel_traffic_service.provide_ukc_clearance(vessel, edge[1])
                vessel.log_entry("Waiting for tidal window stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Deberthing: if the terminal is part of an section, request access to this section first
        #if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
        #    yield from waterway.PassWaterway.request_access_next_section(vessel, edge[0])

        # Deberthing: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to deberth
        ukc = vessel.env.vessel_traffic_service.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Deberthing start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_b'])
        if vessel.env.now == time_of_arrival + total_time:
            remaining_waiting_time = vessel.metadata['t_b'] - (total_time - vessel.metadata['t_b'] - vessel.metadata['t_l'] - np.max([0,vessel.waiting_time-vessel.metadata['t_b']]))
            if terminal.type == 'jetty':
                terminal.release_terminal(vessel.jetty)
            else:
                PassTerminal.release_terminal_access(vessel, terminal, index, remaining_waiting_time)
            yield vessel.env.timeout(remaining_waiting_time)
        ukc = vessel.env.vessel_traffic_service.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Deberthing stop", vessel.env.now, ukc, shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Move vessel to start node of the terminal, release request of this section, and change vessel route by removing the first node of the route (as vessel will already be located in the second node of the route after the move event)
        #waterway.PassWaterway.release_access_previous_section(vessel, edge[0])

        # Initiate move of vessel back to sea, setting a bool of leaving port to true
        vessel.bound = 'outbound'
        vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
        vessel.env.process(vessel.move())
        vessel.leaving_port = True