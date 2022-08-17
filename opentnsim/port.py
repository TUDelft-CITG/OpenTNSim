# package(s) related to the simulation
import simpy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# OpenTNSim
from opentnsim import core
from opentnsim import vessel_traffic_service
from opentnsim import waterway

# spatial libraries
import pyproj
import shapely.geometry

class HasTerminal(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_edge.append(self.pass_terminal)

    def pass_terminal(self,origin,destination):
        # Terminal
        if 'Terminal' in self.env.FG.edges[origin, destination].keys() and self.route[-1] == destination:
            yield from PassTerminal.pass_terminal(self, [origin, destination])
            raise simpy.exceptions.Interrupt('New route determined')

class HasPortAccess(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node.append(self.request_terminal_access)

    def request_terminal_access(self, origin):
        # Request for a terminal
        if "Port Entrance" in self.env.FG.nodes[origin] and 'leaving_port' not in dir(self):
            self.bound = 'inbound'
            self.terminal_accessed = False
            yield from PassTerminal.request_terminal_access(self, [self.route[-2], self.route[-1]], origin)
            if self.waiting_in_anchorage:
                raise simpy.exceptions.Interrupt('New route determined')

class HasAnchorage(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_look_ahead_to_node.append(self.pass_anchorage_area)

    def pass_anchorage_area(self,destination):
        # Anchorage
        if 'Anchorage' in self.env.FG.nodes[destination].keys() and self.route[-1] == destination:
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
            ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(self, origin)
            if self.bound == 'outbound' and turning_basin.length >= self.L:
                self.log_entry("Vessel Turning Start", self.env.now, ukc, self.env.FG.nodes[origin]['geometry'])
                turning_basin.log_entry("Vessel Turning Start", self.env.now, 0, self.env.FG.nodes[origin]['geometry'])
                yield self.env.timeout(10 * 60)
                ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(self, origin)
                turning_basin.log_entry("Vessel Turning Stop", self.env.now, 10 * 60,
                                        self.env.FG.nodes[origin]['geometry'])
                self.log_entry("Vessel Turning Stop", self.env.now, ukc, self.env.FG.nodes[origin]['geometry'])
            else:
                self.log_entry("Passing Turning Basin", self.env.now, ukc, self.env.FG.nodes[origin]['geometry'])
                turning_basin.log_entry("Vessel Passing", self.env.now, 0, self.env.FG.nodes[origin]['geometry'])
            turning_basin.resource.release(self.request_access_turning_basin)

    def request_turning_basin(self, destination):
        if 'Turning Basin' in self.env.FG.nodes[destination].keys():
            turning_basin = self.env.FG.nodes[destination]['Turning Basin'][0]
            if turning_basin.length >= self.L:
                self.request_access_turning_basin = turning_basin.resource.request()
                yield self.request_access_turning_basin

class IsPortEntrance(core.Identifiable, core.Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

class IsAnchorage(core.HasResource,core.Identifiable, core.HasType, core.Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
            self,
            node,
            max_capacity,
            *args,
            **kwargs
    ):
        super().__init__(nr_resources = max_capacity,*args, **kwargs)

        #self.anchorage_area = {node: simpy.PriorityResource(self.env, capacity=max_capacity),}

class IsTurningBasin(core.HasResource, core.Identifiable, core.Log):
    """Mixin class: Something which has a turning basin object properties as part of a lock complex [in SI-units] """

    def __init__(
        self,
        node,
        length,
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)
        """Initialization"""
        self.length = length

class IsTerminal(core.HasType, core.HasLength, core.HasResource, core.Identifiable, core.Log):

    def __init__(
        self,
        node_start,
        node_end,
        length,
        type,
        jetty_locations = [],
        jetty_lengths=[],
        *args,
        **kwargs
    ):

        "Initialization"
        super().__init__(type=type,typ=type,length=length, remaining_length=length, number_of_independent_resources=len(jetty_locations), *args, **kwargs)

        if self.type == 'quay':
            self.available_quay_lengths = [[0,0],[0,length]]

        elif self.type == 'jetty':
            self.terminal = []
            self.jetties_occupied = 0
            self.jetty_locations = jetty_locations
            self.jetty_lengths = jetty_lengths

class PassTerminal:
    """Mixin class: Collection of interacting functions that handle the vessels that call at a terminal and take the correct measures"""

    def waiting_time_for_tidal_window(vessel,route,delay=0,plot=False):
        """ Function: calulates the time that a vessel has to wait depending on the available tidal windows

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - route: a list of strings that resemble the route of the vessel (can be different than the vessel.route)
                - delay: a delay that can be included to calculate a future situation
                - plot: bool that specifies if a plot is requested or not

        """

        # If the sail-in times of the vessel is not calculated before: request VesselTrafficService what are the available sail-in times given the tidal window policy
        if 'sail_in_times' not in dir(vessel):
            vessel.sail_in_times = vessel_traffic_service.VesselTrafficService.provide_sail_in_times_tidal_window(vessel,route=route,plot=plot)
        # If a vessel is bound for offshore (meaning that sail-out times have already been calculated): store its sail-in times in a parameter and use the sail-out times as input (sail-in times)
        if vessel.bound == 'outbound':
            sail_in_times = vessel.sail_in_times
            vessel.sail_in_times = vessel.sail_out_times

        # Set default parameters
        waiting_time = 0
        current_time = vessel.env.now+delay

        # Loop over the available sail-in (or sail-out) times:
        for t in range(len(vessel.sail_in_times)):
            # If the next sail-in time contains a starting condition for a restriction: if it is the last time, then let the vessel wait wait for this time, else continue the loop
            if vessel.sail_in_times[t][1] == 'Start':
                if t == len(vessel.sail_in_times)-1:
                    waiting_time = vessel.sail_in_times[t][0] - current_time
                    break
                else:
                    continue
            # If the current time of the vessel is greater or equal to the next sail-in time containing a stopping condition for a restriction and is smaller than the next starting time of a restriction: waiting time = 0
            if current_time >= vessel.sail_in_times[t][0]:
                waiting_time = 0
                if t == len(vessel.sail_in_times)-1 or current_time < vessel.sail_in_times[t+1][0]:
                    break
            # Else if the current time of the vessel is smaller or equal to the next sail-in time containing a stopping condition for a restriction
            elif current_time <= vessel.sail_in_times[t][0]:
                # And is smaller than the previous starting time of a restriction: waiting time = 0
                if current_time < vessel.sail_in_times[t-1][0]:
                    waiting_time = 0
                # Waiting time = next stopping time - current time
                else:
                    waiting_time = vessel.sail_in_times[t][0] - current_time
                break
            # Else if it is the last time, then let the vessel wait wait for this time
            elif t == len(vessel.sail_in_times) - 1:
                waiting_time = vessel.sail_in_times[t][0] - current_time
            # Else if none of the above conditions hold: continue the loop
            else:
                continue

        # If vessel is bound for offshore: reset the sail-in times to their original
        if vessel.bound == 'outbound':
            vessel.sail_in_times = sail_in_times

        # Else if the vessel is bound for the terminal: request the sail-out times by reversing the route
        elif vessel.bound == 'inbound':
            network = vessel.env.FG
            distance_to_node = 0
            route.reverse()
            # Calculate the total time that a vessel will spend in the port before returning: sailing time of the vessel + 2 times (de)berthing time + (un)loading time
            for n in range(len(route)):
                if n == 0:
                    continue
                elif n == len(route)-1:
                    break
                distance_to_node += pyproj.Geod(ellps='WGS84').inv(network.nodes[route[n - 1]]['geometry'].x,
                                                                   network.nodes[route[n - 1]]['geometry'].y,
                                                                   network.nodes[route[n]]['geometry'].x,
                                                                   network.nodes[route[n]]['geometry'].y)[2]

            sailing_time = distance_to_node / vessel.v
            # Calculate delay and include in current time
            delay = sailing_time + 2 * vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60 + waiting_time
            current_time = vessel.env.now + delay
            # Request the sail-out times by temporarily setting the vessel bound as outbound
            vessel.bound = 'outbound'
            vessel.sail_out_times = vessel_traffic_service.VesselTrafficService.provide_sail_in_times_tidal_window(vessel,route=route,plot=plot)
            vessel.bound = 'inbound'

            # Loop over the provided sail-out times: if there is no suitable tidal window or if the waiting time is too large: set waiting time to maximum waiting time of vessel (it will return without entering the port)
            for t in range(len(vessel.sail_out_times)):
                # If the next sail-in time contains a starting condition for a restriction: continue the loop
                if vessel.sail_out_times[t][1] == 'Start':
                    continue
                # If the current time of the vessel is greater or equal to the next sail-in time containing a stopping condition for a restriction and is smaller than the next starting time of a restriction: break loop
                if current_time >= vessel.sail_out_times[t][0]:
                    if t == len(vessel.sail_out_times)-1 or current_time < vessel.sail_out_times[t+1][0]:
                        break
                # Else if the current time of the vessel is smaller or equal to the next sail-in time containing a stopping condition for a restriction
                elif current_time <= vessel.sail_out_times[t][0]:
                    # Determine if the waiting time is allowed or not: if not revise waiting time to maximum waiting time, else break the loop
                    if vessel.sail_out_times[t][0]-current_time >= vessel.metadata['max_waiting_time']:
                        waiting_time = vessel.metadata['max_waiting_time']
                    else:
                        break
                # Else if it is the last time, then revise waiting time to maximum waiting time
                elif t == len(vessel.sail_out_times)-1:
                    waiting_time = vessel.metadata['max_waiting_time']
                # Else if none of the conditions holds: continue the loop
                else:
                    continue
            # Reset the route
            route.reverse()

        # Return the vessel waiting time. If the vessel is not able to return within the maximum waiting time at the terminal: the waiting time is unacceptable and the vessel will return
        return waiting_time

    def move_to_anchorage(vessel,node):
        """ Function: moves a vessel to the anchorage area instead of continuing its route to the terminal if a vessel is required to wait in the anchorage area

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node: a string that contains the node of the route that the vessel is currently on

        """

        # Set some default parameters:
        network = vessel.env.FG
        vessel.waiting_in_anchorage = True
        nodes_of_anchorages = []
        capacity_of_anchorages = []
        users_of_anchorages = []
        sailing_distances_from_anchorages = []

        # Loop over the nodes of the network and identify all the anchorage areas:
        for node_anchorage in network.nodes:
            if 'Anchorage' in network.nodes[node_anchorage]:
                #Extract information over the individual anchorage areas: capacity, users, and the sailing distance to the anchorage area from the designated terminal the vessel is planning to call
                nodes_of_anchorages.append(node_anchorage)
                capacity_of_anchorages.append(vessel.env.FG.nodes[node_anchorage]['Anchorage'][0].resource.capacity)
                users_of_anchorages.append(len(vessel.env.FG.nodes[node_anchorage]['Anchorage'][0].resource.users))
                route_from_anchorage = nx.dijkstra_path(vessel.env.FG, node_anchorage, vessel.route[-1])
                sailing_distance_from_anchorage = 0
                for route_node in enumerate(route_from_anchorage):
                    if route_node[0] == 0:
                        continue
                    _, _, distance = vessel.wgs84.inv(vessel.env.FG.nodes[route_from_anchorage[route_node[0]-1]]['geometry'].x,
                                                      vessel.env.FG.nodes[route_from_anchorage[route_node[0]-1]]['geometry'].y,
                                                      vessel.env.FG.nodes[route_from_anchorage[route_node[0]]]['geometry'].x,
                                                      vessel.env.FG.nodes[route_from_anchorage[route_node[0]]]['geometry'].y)
                    sailing_distance_from_anchorage += distance
                sailing_distances_from_anchorages.append(sailing_distance_from_anchorage)

        # Sort the lists based on the sailing distance to the anchorage area from the designated terminal the vessel is planning to call
        sorted_nodes_anchorages = [nodes for (distances,nodes) in sorted(zip(sailing_distances_from_anchorages, nodes_of_anchorages))]
        sorted_users_of_anchorages = [nodes for (distances,nodes) in sorted(zip(sailing_distances_from_anchorages, users_of_anchorages))]
        sorted_capacity_of_anchorages = [nodes for (distances,nodes) in sorted(zip(sailing_distances_from_anchorages, capacity_of_anchorages))]

        # Take the anchorage area that is closest to the designated terminal the vessel is planning to call if there is sufficient capacity:
        node_anchorage = sorted_nodes_anchorages[np.argmin(sailing_distances_from_anchorages)]
        for node_anchorage_area in enumerate(sorted_nodes_anchorages):
            if sorted_users_of_anchorages[node_anchorage_area[0]] < sorted_capacity_of_anchorages[node_anchorage_area[0]]:
                node_anchorage = node_anchorage_area[1]
                break

        # If there is not an available anchorage area: leave the port after entering the anchorage area
        if node_anchorage != node_anchorage_area[1]:
           vessel.return_to_sea = True
           vessel.waiting_time = vessel.metadata['max_waiting_time']
        # Set the route that the vessel will take after calling at the terminal (back to the origin) and after waiting in the anchorage area
        anchorage = network.nodes[node_anchorage]['Anchorage'][0]
        vessel.route_after_anchorage = []
        vessel.true_origin = vessel.route[0]
        current_time = vessel.env.now
        vessel.route_after_anchorage = nx.dijkstra_path(vessel.env.FG, node_anchorage, vessel.route[-1])
        vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route[-1], vessel.route[0])
        # Start the moving process by setting the route of the vessel from current node to chosen anchorage area
        yield from core.Movable.pass_edge(vessel,vessel.route[node], vessel.route[node+1])
        vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[node+1], node_anchorage)
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
        network = vessel.env.FG
        anchorage = network.nodes[node]['Anchorage'][0]

        # Moves the vessel to the node of the anchorage area
        #yield from Movable.pass_edge(vessel, vessel.route[vessel.route.index(node) - 1],vessel.route[vessel.route.index(node)])

        # Request access to the anchorage area and log this to the anchorage area log and vessel log (including the calculated value for the net ukc)
        vessel.anchorage_access = anchorage.resource.request()
        yield vessel.anchorage_access
        anchorage.log_entry("Vessel arrival", vessel.env.now, len(anchorage.resource.users),nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
        current_time = vessel.env.now
        ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel,node)
        vessel.log_entry("Waiting in anchorage start", vessel.env.now, ukc,nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )

        # Determine the sailing distance to the destined terminal (important for later in the calculation)
        edge = vessel.route_after_anchorage[-2], vessel.route_after_anchorage[-1]
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        sailing_distance = 0
        for nodes in enumerate(vessel.route_after_anchorage[:-1]):
            _, _, distance = vessel.wgs84.inv(vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0]]]['geometry'].x,
                                              vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0]]]['geometry'].y,
                                              vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0] + 1]]['geometry'].x,
                                              vessel.env.FG.nodes[vessel.route_after_anchorage[nodes[0] + 1]]['geometry'].y)
            sailing_distance += distance

        # If the vessel is allowed in (so has not to return to sea) and should wait in the anchorage area for an available berth
        if vessel.return_to_sea == False and vessel.waiting_for_availability_terminal:
            # Vessel waits for the available berth, or if it takes longer than the permitted maximum waiting time: the vessel will wait until that specific waiting time
            yield vessel.waiting_time_in_anchorage | vessel.env.timeout(vessel.metadata['max_waiting_time'])
            new_current_time = vessel.env.now
            # If waiting time is greater or equal than the maximum waiting time: vessel returns to sea and releases its request for the berth at the terminal of call
            if new_current_time - current_time >= vessel.metadata['max_waiting_time']:
                vessel.return_to_sea = True
                vessel.waiting_time = 0
                if terminal.type == 'quay': terminal = terminal.resource
                elif terminal.type == 'jetty': terminal = terminal.resource[vessel.index_jetty_position]
                terminal.release(vessel.waiting_time_in_anchorage)
            # Else:
            else:
                # If terminal of call is of type 'quay': determine the quay position, request the quay length and adjust the available quay length
                if terminal.type == 'quay':
                    vessel.index_quay_position, _ = PassTerminal.request_quay_position(vessel, terminal)
                    PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)
                    terminal = terminal.resource
                # Else if terminal of call is of type 'jetty': request terminal again (with priority) before releasing the initial request (based on which the waiting time was calculated)
                elif terminal.type == 'jetty':
                    terminal = terminal.resource[vessel.index_jetty_position]
                    vessel.access_terminal = terminal.request(priority=-1)
                    terminal.release(vessel.waiting_time_in_anchorage)
                    yield vessel.access_terminal

                # Determine the new sail-in times of the vessel (since the vessel had to wait in the anchorage area, the available sail-in times are now different)
                vessel.sail_in_times = vessel_traffic_service.VesselTrafficService.provide_sail_in_times_tidal_window(vessel, vessel.route_after_anchorage)
                # Loop over the sail-in times to determine if the vessel should wait or not, or even has to return to sea (due to non-allowable waiting time by exceeding the set maximum)
                for t in range(len(vessel.sail_in_times)):
                    # If the next sail-in time contains a starting condition for a restriction: if it is the last time, then let the vessel wait wait for this time (if permitted, else return to sea), else continue the loop
                    if vessel.sail_in_times[t][1] == 'Start':
                        if t == len(vessel.sail_in_times) - 1:
                            waiting_time = vessel.sail_in_times[t][0] - current_time
                            if waiting_time >= vessel.metadata['max_waiting_time']:
                                vessel.return_to_sea = True
                                vessel.waiting_time = 0
                                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                    terminal.release(vessel.access_terminal)
                            else:
                                # Yield waiting time and if terminal of call is of type 'jetty': calculate the estimated time of arrival and departure (important for the queuing procedure)
                                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                    terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.sail_in_times[t][0] - new_current_time
                                    vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                    vessel.bound = 'outbound'
                                    waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel,route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                                    vessel.bound = 'inbound'
                                    terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                    terminal.users[-1].type = vessel.type
                                yield vessel.env.timeout(vessel.sail_in_times[t][0] - new_current_time)
                            break
                        else:
                            continue
                    # If the current time of the vessel is greater or equal to the next sail-in time containing a stopping condition for a restriction and is smaller than the next starting time of a restriction
                    if new_current_time >= vessel.sail_in_times[t][0]:
                        if t == len(vessel.sail_in_times) - 1 or new_current_time < vessel.sail_in_times[t + 1][0]:
                            # No waiting time and if terminal of call is of type 'jetty': calculate the estimated time of arrival and departure (important for the queuing procedure)
                            if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
                                vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                vessel.bound = 'outbound'
                                waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                                vessel.bound = 'inbound'
                                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                terminal.users[-1].type = vessel.type
                            break
                    # Else if the current time of the vessel is smaller or equal to the next sail-in time containing a stopping condition for a restriction
                    elif new_current_time <= vessel.sail_in_times[t][0]:
                        # If the current time of the vessel is smaller than the next starting time of a restriction:
                        if new_current_time < vessel.sail_in_times[t - 1][0]:
                            # No waiting time and if terminal of call is of type 'jetty': calculate the estimated time of arrival and departure (important for the queuing procedure)
                            if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
                                vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                vessel.bound = 'outbound'
                                waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                                vessel.bound = 'inbound'
                                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                terminal.users[-1].type = vessel.type
                        # Else: there is waiting time: determine if allowed or not (exceeds the maximum permitted or not)
                        else:
                            waiting_time = vessel.sail_in_times[t][0] - current_time
                            if waiting_time >= vessel.metadata['max_waiting_time']:
                                vessel.return_to_sea = True
                                vessel.waiting_time = 0
                                terminal.release(vessel.access_terminal)
                            else:
                                # Yield waiting time and if terminal of call is of type 'jetty': calculate the estimated time of arrival and departure (important for the queuing procedure)
                                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                    terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.sail_in_times[t][0] - new_current_time
                                    vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                    vessel.bound = 'outbound'
                                    waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                                    vessel.bound = 'inbound'
                                    terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                    terminal.users[-1].type = vessel.type
                                yield vessel.env.timeout(vessel.sail_in_times[t][0] - new_current_time)
                        break
                    # Else if it is the last time, then there is waiting time, so determine if allowed or not (exceeds the maximum permitted or not)
                    elif t == len(vessel.sail_in_times) - 1:
                        waiting_time = vessel.sail_in_times[t][0] - current_time
                        if waiting_time >= vessel.metadata['max_waiting_time']:
                            vessel.return_to_sea = True
                            vessel.waiting_time = 0
                            terminal.release(vessel.access_terminal)
                        else:
                            # Yield waiting time and if terminal of call is of type 'jetty': calculate the estimated time of arrival and departure (important for the queuing procedure)
                            if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                                terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.sail_in_times[t][0] - new_current_time
                                vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                                vessel.bound = 'outbound'
                                waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                                vessel.bound = 'inbound'
                                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                                terminal.users[-1].type = vessel.type
                            yield vessel.env.timeout(vessel.sail_in_times[t][0] - new_current_time)
                    # Else if none of the conditions holds: continue the loop
                    else:
                        continue

        # Else if the vessel (so has not to return to sea) and should wait in the anchorage area for an available tidal window:
        elif vessel.return_to_sea == False and vessel.waiting_time:
            # If terminal of call is of type 'jetty': calculate the estimated time of arrival and departure (important for the queuing procedure) including the waiting time for the tidal window
            if terminal.type == 'jetty':
                terminal = terminal.resource[vessel.index_jetty_position]
                terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v + vessel.waiting_time
                vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                vessel.bound = 'outbound'
                waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel, route=vessel.route_after_terminal,delay=vessel_etd - vessel.env.now,plot=False)
                vessel.bound = 'inbound'
                terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                terminal.users[-1].type = vessel.type
            # Else if terminal of call is of type 'quay': vessel will pick predetermined quay position, get this length, so adjust available quay lengths
            elif terminal.type == 'quay':
                PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)
            # Determine the new time: if the maximum allowabel waiting time will be or has been exceeded: return to sea, otherwise yield waiting time
            new_current_time = vessel.env.now + vessel.waiting_time
            if new_current_time - current_time >= vessel.metadata['max_waiting_time']:
                vessel.return_to_sea = True
                vessel.waiting_time = 0
            yield vessel.env.timeout(vessel.waiting_time)

        # If vessel does not has to return to sea: log this in the anchorage log and vessel log, set the route from anchorage to terminal, release the access to the section to the anchorage area, and request access to the first section on its route and initate the move
        if vessel.return_to_sea == False:
            ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel,node)
            vessel.log_entry("Waiting in anchorage stop", vessel.env.now, ukc, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
            vessel.route = vessel.route_after_anchorage
        # Else if the vessel has to return to sea: log this as well in the log files of the anchorage and vessel, set route back to origin, release the access to the section to the anchorage area, and request access to the first section on its route and initate the move
        else:
            if 'waiting_time_in_anchorage' in dir(vessel):
                vessel.waiting_time_in_anchorage.cancel()
            yield vessel.env.timeout(vessel.waiting_time)
            ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel,node)
            vessel.log_entry("Waiting in anchorage stop", vessel.env.now, ukc, nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node)]], )
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[vessel.route.index(node)], vessel.true_origin)
        waterway.PassSection.release_access_previous_section(vessel, vessel.route[0])
        yield from waterway.PassSection.request_access_next_section(vessel, vessel.route[0], vessel.route[1])
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

    def request_terminal_access(vessel, edge, node):
        """ Function: function that handles the request of a vessel to access the terminal of call: it lets the vessel move to the correct terminal (quay position and jetty) or moves it to the
            anchorage area to wait on either the terminal (quay or jetty) availability or tidal window

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - edge: list of two strings that form the edge at which the terminal is located
                - node: string that contains the node of the route that the vessel is currently on (either the origin or anchorage area)

        """
        # Set some default parameters
        node = vessel.route.index(node)
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        vessel.move_to_anchorage = False
        vessel.waiting_in_anchorage = False
        vessel.waiting_for_availability_terminal = False

        # Function that is used in the request procedure
        def checks_waiting_time_due_to_tidal_window(vessel, route, node, maximum_waiting_time = False):
            """ Function: function that checks if the vessel arrives beyond a tidal window and calculates the waiting time if that is the case (or return no waiting time otherwise)

                Input:
                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                    - route: list of strings that resemble the route of the vessel (can be different than vessel.route)
                    - node: string that contains the node of the route that the vessel is currently on (origin)
                    - maximum_waiting_time: bool that specifies if there is a maximum to the waiting time of the vessel

            """

            # Calculate the waiting time using the waiting_time_for_tidal_window-function
            vessel.waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel,route=route,delay=0,plot=False)
            # If there is a maximum waiting time and this is exceeded: vessel will have to return to sea with zero waiting time
            if vessel.waiting_time >= vessel.metadata['max_waiting_time'] and maximum_waiting_time:
                vessel.return_to_sea = True
                vessel.waiting_time = 0
            # Else: vessel does not have to return to sea
            else:
                vessel.return_to_sea = False

        # Calculate the waiting time and whether this waiting time is acceptable
        checks_waiting_time_due_to_tidal_window(vessel, route = vessel.route, node = node, maximum_waiting_time=True)

        # Set default parameter
        available_turning_basin = False

        # Loop over the nodes of the route and determine whether there is turning basin that suits the vessel (dependent on a vessel length restriction)
        for basin in vessel.route:
            if 'Turning Basin' in vessel.env.FG.nodes[basin].keys():
                turning_basin = vessel.env.FG.nodes[basin]['Turning Basin'][0]
                if turning_basin.length >= vessel.L:
                    available_turning_basin = True
                    break

        # If there is no available turning basin: vessel will have to return to sea without waiting in the anchorage area
        if available_turning_basin == False:
            vessel.return_to_sea = True
            vessel.waiting_time = 0

        # If vessel does not have to return to sea:
        if not vessel.return_to_sea:
            # If the terminal is of type 'jetty:
            if terminal.type == 'jetty':
                # Set empty default lists
                minimum_waiting_time = []
                vessels_waiting = []

                # Loop over the jetties of the terminal
                for jetty in enumerate(terminal.resource):
                    # If the length of the vessel is greater than the maximum allowable length of the jetty: vessel has to move to the anchorage area (for now), but continue loop
                    if vessel.L > terminal.jetty_lengths[jetty[0]]:
                        vessel.move_to_anchorage = True
                        continue
                    # Else if the length of the vessel is less or equal than the maximum allowable length of the jetty and there are no vessels at this jetty or waiting for this jetty: vessel does not have to move to the anchorage area, break loop
                    if jetty[1].users == [] and jetty[1].queue == []: #jetty[1][[edge[0]]
                        vessel.index_jetty_position = jetty[0]
                        vessel.move_to_anchorage = False
                        break
                    # Else if the length of the vessel is less or equal than the maximum allowable length of the jetty but there are vessel at this jetty already or waiting for this jetty:
                    else:
                        # If the queue is still empty: calculate the estimated time of departure of the currently (un)loading vessel and append to list
                        if jetty[1].queue == []:
                            minimum_waiting_time.append(jetty[1].users[-1].etd)
                        # Else append zero to the list
                        else:
                            minimum_waiting_time.append(0)
                        # Append the number of vessels waiting in the queue for the specific jetty to the list
                        vessels_waiting.append(len(jetty[1].queue))

                        # Continue loop if the jetty is not the last jetty in the list
                        if jetty[0] != len(terminal.resource)-1:
                            continue

                    # If loop is not broken, then vessel has to move to the anchorage area
                    vessel.move_to_anchorage = True

            # Else if the terminal is of type 'quay:
            elif terminal.type == 'quay':
                # Import the locations of the current configuration of vessels located at the quay
                aql = terminal.available_quay_lengths
                # If the queue of vessels waiting for an available quay length is still empty: request quay position
                if terminal.length.get_queue == []:
                    vessel.index_quay_position,vessel.move_to_anchorage = PassTerminal.request_quay_position(vessel, terminal)
                # Else if this queue is not empty: vessel has to move to anchorage area (according to FCFS-policy)
                else:
                    vessel.move_to_anchorage = True

            # If the vessel has some waiting time due to the fact that it has arrived beyond a tidal window (move to anchorage still set to False)
            if vessel.waiting_time and not vessel.move_to_anchorage:
                yield from PassTerminal.move_to_anchorage(vessel, node)

            # Else if the vessel has to wait because there is no available spot at the terminal (move to anchorage was set to True)
            elif vessel.move_to_anchorage:
                # Set bool that says that vessel has to wait in the anchorage area because its waiting for an available spot at the terminal
                vessel.waiting_for_availability_terminal = True

                # If the terminal is of type 'quay:
                if terminal.type == 'quay':
                    # Make a request by getting the length of the terminal equal to the length of the vessel (which was not available), which functions as a yield timeout event equal to the waiting time for availability terminal
                    vessel.waiting_time_in_anchorage = terminal.length.get(vessel.L)

                # Else if the terminal is of type 'jetty:
                elif terminal.type == 'jetty':
                    # Set defual position index of the jetty
                    vessel.index_jetty_position = []

                    # Determine which indexes have an empty queue (jetties which have no vessels waiting)
                    indices_empty_queue_for_jetty = [waiting_vessels[0] for waiting_vessels in enumerate(vessels_waiting) if waiting_vessels[1] == 0]
                    # If this list of indexes is not empty:
                    if indices_empty_queue_for_jetty != []:
                        # Set default minimum waiting time (equal to the first jetty without queue)
                        min_minimum_waiting_time = minimum_waiting_time[indices_empty_queue_for_jetty[0]]
                        # Loop over the jetty indexes which do not have a queue:
                        for index in indices_empty_queue_for_jetty:
                            # If the minimum waiting time of the jetty is equal or less than the minimum waiting time for a jetty that was found up to now and the length of the vessel fits the jetty length: overwrite minimum waiting time with the minimum waiting time of that jetty and (temporarily) set the index of this jetty as the jetty index the vessel will be waiting for
                            if minimum_waiting_time[index] <= min_minimum_waiting_time and vessel.L <= terminal.jetty_lengths[index]:
                                min_minimum_waiting_time = minimum_waiting_time[index]
                                vessel.index_jetty_position = index
                    # Else if the list was empty and there is still no chosen jetty index: pick the jetty with the least number of vessels waiting in the queue
                    if vessel.index_jetty_position == []:
                        # Set defaul indexes list:
                        indexes = []
                        # Loop over the lenghts of the jetties:
                        for length in enumerate(terminal.jetty_lengths):
                            #If the length of the vessel fits in the jetty: append index to list
                            if vessel.L <= length[1]:
                                indexes.append(length[0])
                        # If the list of indexes are not empty: pick the jetty with the least number of vessels in the queue waiting for this particular jetty
                        if indexes != []:
                            vessel.index_jetty_position = np.min([y[0] for y in enumerate(vessels_waiting) if y[0] in indexes])

                    # If the jetty has been chosen: request that jetty which will function as the timeout event equalling the time that the jetty will be available
                    if vessel.index_jetty_position != []:
                        vessel.waiting_time_in_anchorage = terminal.resource[vessel.index_jetty_position].request()
                    # Else if there is no suitable jetty: vessel will return to sea without waiting in the anchorage area
                    else:
                        vessel.return_to_sea = True
                        vessel.waiting_time = 0

                # Move vessel to the anchorage area
                yield from PassTerminal.move_to_anchorage(vessel, node)

            # Else if the vessel does not have to wait for either an available terminal or tidal window
            else:
                if terminal.type == 'quay':
                    PassTerminal.adjust_available_quay_lengths(vessel, terminal, vessel.index_quay_position)

            # If the vessel does not have to return to sea
            if vessel.return_to_sea == False:
                # Import information about the terminal, based on the terminal type (and if type 'jetty': based on the picked jetty position).
                if terminal.type == 'quay':
                    terminal = terminal.resource
                elif terminal.type == 'jetty':
                    terminal = terminal.resource[vessel.index_jetty_position]

                # If the terminal type is 'jetty' and the vessel has to wait for an available jetty: pass; else: continue code
                if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty' and vessel.waiting_for_availability_terminal == True:
                    pass
                else:
                    # Request access to the terminal
                    vessel.access_terminal = terminal.request()
                    yield vessel.access_terminal

                    # If terminal type is 'quay': revise the locations of the current configuration of vessels located at the quay with the new configuration
                    if vessel.env.FG.edges[edge]["Terminal"][0].type == 'quay':
                        vessel.env.FG.edges[edge]["Terminal"][0].available_quay_lengths = aql

                    # Set route after the terminal
                    if vessel.waiting_time:
                        vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route_after_anchorage[-1], vessel.true_origin)
                    else:
                        vessel.route_after_terminal = nx.dijkstra_path(vessel.env.FG, vessel.route[-1], vessel.route[0])

                    # If terminal is of type 'jetty': calculate the estimated time of departure consisting of the estimated time of arrival (sailing distance + berthing time + current time) + waiting time + (un)loading time + berthing time again
                    if vessel.env.FG.edges[edge]["Terminal"][0].type == 'jetty':
                        # Take the route when the vessel is leaving the terminal and reverse this route:
                        route = vessel.route_after_terminal
                        route.reverse()
                        sailing_distance = 0
                        # Loop over the route to calculate the total sailing distance and time
                        for nodes in enumerate(route[:-1]):
                            _, _, distance = vessel.wgs84.inv(vessel.env.FG.nodes[route[nodes[0]]]['geometry'].x,
                                                              vessel.env.FG.nodes[route[nodes[0]]]['geometry'].y,
                                                              vessel.env.FG.nodes[route[nodes[0] + 1]]['geometry'].x,
                                                              vessel.env.FG.nodes[route[nodes[0] + 1]]['geometry'].y)
                            sailing_distance += distance
                        terminal.users[-1].eta = vessel.env.now + sailing_distance / vessel.v
                        vessel_etd = terminal.users[-1].eta + vessel.metadata['t_b'] * 60 + vessel.metadata['t_l'] * 60
                        waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel,route = vessel.route_after_terminal, delay=vessel_etd-vessel.env.now,plot=False)
                        terminal.users[-1].etd = vessel_etd + vessel.metadata['t_b'] * 60 + np.max([0, waiting_time - vessel.metadata['t_b'] * 60])
                        terminal.users[-1].type = vessel.type

        # Else if vessel has to return to sea: move vessel to anchorage first
        else:
            yield from PassTerminal.move_to_anchorage(vessel, node)

    def pass_terminal(vessel,edge):
        """ Function: function that handles the vessel at the terminal

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - edge: list of two strings that form the edge at which the terminal is located

        """

        # Import information about the terminal and the corresponding index of the start of the edge at which the terminal is located
        terminal = vessel.env.FG.edges[edge]["Terminal"][0]
        index = vessel.route[vessel.route.index(edge[1]) - 1]

        # Calculate the location of the assigned position along the edge at which the terminal is located and finish the pass_edge move
        vessel.terminal_accessed = True
        vessel.wgs84 = pyproj.Geod(ellps="WGS84")
        [origin_lat,
         origin_lon,
         destination_lat,
         destination_lon] = [vessel.env.FG.nodes[edge[0]]['geometry'].x,
                             vessel.env.FG.nodes[edge[0]]['geometry'].y,
                             vessel.env.FG.nodes[edge[1]]['geometry'].x,
                             vessel.env.FG.nodes[edge[1]]['geometry'].y]
        fwd_azimuth, _, _ = vessel.wgs84.inv(origin_lat, origin_lon, destination_lat, destination_lon)

        if terminal.type == 'quay':
            position = vessel.quay_position

        elif terminal.type == 'jetty':
            position = terminal.jetty_locations[vessel.index_jetty_position]

        [vessel.terminal_pos_lat, vessel.terminal_pos_lon, _] = vessel.wgs84.fwd(vessel.env.FG.nodes[edge[0]]['geometry'].x,
                                                                                 vessel.env.FG.nodes[edge[0]]['geometry'].y,
                                                                                 fwd_azimuth, position)

        orig = nx.get_node_attributes(vessel.env.FG, "geometry")[edge[0]]
        dest = shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon)
        distance = vessel.wgs84.inv(shapely.geometry.asShape(orig).x,
                                    shapely.geometry.asShape(orig).y,
                                    shapely.geometry.asShape(dest).x,
                                    shapely.geometry.asShape(dest).y, )[2]

        ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel, edge[0])
        vessel.log_entry("Sailing from node {} to node {} start".format(edge[0], edge[1]), vessel.env.now, ukc, orig, )
        yield vessel.env.timeout(distance / vessel.current_speed)
        ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Sailing from node {} to node {} stop".format(edge[0], edge[1]), vessel.env.now, ukc, dest, )

        # If the terminal is of type 'quay': log in logfile of terminal keeping track of the available length (by getting the so-called position length)
        if terminal.type == 'quay':
            terminal.pos_length.get(vessel.L)
            terminal.log_entry("Arrival of vessel", vessel.env.now, terminal.length.capacity-terminal.pos_length.level,nx.get_node_attributes(vessel.env.FG, "geometry")[index], )
        # Else if terminal is of type 'jetty': log in logfile of terminal calculating keeping track of number of jetties to be occupied
        elif terminal.type == 'jetty':
            terminal.jetties_occupied += 1
            terminal.log_entry("Arrival of vessel", vessel.env.now, terminal.jetties_occupied,nx.get_node_attributes(vessel.env.FG, "geometry")[index], )

        # Berthing: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to berth
        ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Berthing start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_b']*60)
        ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Berthing stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # If terminal is part of a junction: release request of this section (vessel is berthed and not in channel/basin)
        if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
            waterway.PassSection.release_access_previous_section(vessel, edge[1])

        # Unloading: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to unload
        ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Unloading start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_l']*60/2)
        ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Unloading stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Loading: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to load
        ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Loading start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_l']*60/2)
        ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Loading stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Determine the new route of the vessel (depending on whether the vessel came from the anchorage area or sailed to the terminal directly) and changing the direction of the vessel
        vessel.bound = 'outbound'  # to be removed later
        if 'true_origin' in dir(vessel):
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[vessel.route.index(edge[1])], vessel.true_origin)
        else:
            vessel.route = nx.dijkstra_path(vessel.env.FG, vessel.route[vessel.route.index(edge[1])], vessel.route[0])
        if edge == [vessel.route[1],vessel.route[0]]: vessel.bound = 'outbound'

        # Calculate if the vessel has to wait due to be ready for departure beyond an available tidal window
        vessel.waiting_time = PassTerminal.waiting_time_for_tidal_window(vessel,route=vessel.route,delay=0,plot=False)
        # If there is waiting time: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the waiting time
        if vessel.waiting_time:
            ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
            vessel.log_entry("Waiting for tidal window start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
            yield vessel.env.timeout(np.max([0,vessel.waiting_time-vessel.metadata['t_b']*60]))
            ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
            vessel.log_entry("Waiting for tidal window stop", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Deberthing: if the terminal is part of an section, request access to this section first
        if 'Junction' in vessel.env.FG.nodes[edge[1]].keys():
            yield from waterway.PassSection.request_access_next_section(vessel, edge[1], edge[0])

        # Deberthing: if the terminal is attached to a turning basin: check whether the vessel can turn here (if the length of the vessel allows to turn the vessel in this basin)
        if 'Turning Basin' in vessel.env.FG.nodes[edge[0]].keys():
            turning_basin = vessel.env.FG.nodes[edge[0]]['Turning Basin'][0]
            if turning_basin.length >= vessel.L:
                vessel.request_access_turning_basin = turning_basin.resource.request()
                yield vessel.request_access_turning_basin

        # Deberthing: log the start and stop of this event in log file of vessel including the calculated net ukc, and yield timeout equal to the time it takes to deberth
        ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Deberthing start", vessel.env.now, ukc,shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)
        yield vessel.env.timeout(vessel.metadata['t_b']*60)
        ukc = vessel_traffic_service.VesselTrafficService.provide_ukc_clearance(vessel, edge[1])
        vessel.log_entry("Deberthing stop", vessel.env.now, ukc, shapely.geometry.Point(vessel.terminal_pos_lat, vessel.terminal_pos_lon),)

        # Move vessel to start node of the terminal, release request of this section, and change vessel route by removing the first node of the route (as vessel will already be located in the second node of the route after the move event)
        waterway.PassSection.release_access_previous_section(vessel, edge[0])

        # If the terminal is of type 'quay'
        if terminal.type == 'quay':
            # Determine the old maximum available length of the quay
            old_level = PassTerminal.calculate_quay_length_level(terminal)

            # Function that is used to readjust the available quay lengths:
            def readjust_available_quay_lengths(terminal,position):
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

            # Readjust the available quay lengths as the vessel is leaving the terminal
            terminal.available_quay_lengths = readjust_available_quay_lengths(terminal,vessel.quay_position)
            # Calculate the new maximum available quay length
            new_level = PassTerminal.calculate_quay_length_level(terminal)
            # If this length does not equal the current maximum available quay length (is smaller), then put this length back to the quay
            if old_level != new_level:
                terminal.length.put(new_level - old_level)
            # Give vessel length back to keep track of the total claimed vessel length and log this value and the departure event in the logfile of the terminal, and release the request of the vessel to access the terminal
            terminal.pos_length.put(vessel.L)
            terminal.log_entry("Departure of vessel", vessel.env.now, terminal.length.capacity-terminal.pos_length.level,nx.get_node_attributes(vessel.env.FG, "geometry")[index], )
            terminal.resource.release(vessel.access_terminal)

        # Else if the terminal is of type 'jetty': adjust number of vessels occupying a jetty, log the departure of this vessel and this number, and release the request of the vessel for the specific jetty
        elif terminal.type == 'jetty':
            terminal.jetties_occupied -= 1
            terminal.log_entry("Departure of vessel", vessel.env.now, terminal.jetties_occupied,nx.get_node_attributes(vessel.env.FG, "geometry")[index],)
            terminal.resource[vessel.index_jetty_position].release(vessel.access_terminal)

        # Initiate move of vessel back to sea, setting a bool of leaving port to true
        vessel.geometry = nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[0]]
        vessel.env.process(vessel.move())
        vessel.leaving_port = True