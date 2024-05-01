# package(s) related to the simulation
import simpy
import numpy as np
from itertools import cycle
import pandas as pd

# OpenTNSim
from opentnsim import core
from opentnsim import vessel_traffic_service

class HasWaterway(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_edge.append(self.request_access_new_section)
        self.on_complete_pass_edge.append(self.release_last_section)

    def request_access_new_section(self,origin,destination):
        # Leave and access waterway section
        if 'Junction' in self.env.FG.nodes[origin].keys():
            yield from PassWaterway.release_access_previous_section(self, origin)

        if 'Detector' in self.env.FG.nodes[origin].keys():
            if destination != self.route[-1]:
                yield from PassWaterway.request_access_next_section(self, origin)

    def release_last_section(self,destination):
        if 'Junction' in self.env.FG.nodes[destination].keys() and destination == self.route[-1]:
            yield from PassWaterway.release_access_previous_section(self, destination)

    def open_waterway(vessel,node):
        detector = vessel.env.FG.nodes[node]['Detector']
        infrastructures = []
        ahead_nodes = []
        next_nodes = []
        for ahead_node in vessel.route[vessel.route.index(node):]:
            if ahead_node not in detector.keys():
                continue

            infrastructure = detector[ahead_node].infrastructure

            if isinstance(infrastructure, IsWaterwayJunction):
                for next_node in vessel.route[vessel.route.index(ahead_node):]:
                    if [ahead_node, next_node] in infrastructure.sections:
                        infrastructures.append(infrastructure)
                        ahead_nodes.append(ahead_node)
                        next_nodes.append(next_node)
                        break

        return infrastructures, ahead_nodes, next_nodes

class IsWaterwayJunction(core.HasResource,core.SimpyObject,core.Identifiable, core.Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
        self,
        sections,
        detector_nodes,
        encounter_restrictions = {},
        overtaking_restrictions = {},
        *args,
        **kwargs
    ):
        self.sections = sections
        self.encounter_restrictions = encounter_restrictions
        self.overtaking_restrictions = overtaking_restrictions
        self.detector_nodes = detector_nodes
        independent_resources = {section[-1]: 1 for section in self.sections}
        super().__init__(independent_resources=independent_resources,*args, **kwargs)
        "Initialization"

        for section,detector_node in zip(self.sections,self.detector_nodes):
            if 'Detector' not in self.env.FG.nodes[detector_node]:
                self.env.FG.nodes[detector_node]['Detector'] = {}
            self.env.FG.nodes[detector_node]['Detector'][section[0]] = core.IsDetectorNode(self)

class PassWaterway:
    """Mixin class: Collection of functions that release and request sections. Important to obey the traffic regulations (safety distance and one-way-traffic) """

    # Functions used to calculate the sail-in-times for a specific vessel
    def rule_determinator(vessel,infrastructure, restriction_type, node):
        restriction = getattr(infrastructure, restriction_type)[node]
        number_of_restrictions = len(restriction.conditions) - 1
        previous_operator = None

        boolean = True
        restriction_applies = False
        for index, (condition, limit, operator) in enumerate(zip(restriction.conditions,
                                                                 restriction.values,
                                                                 cycle(restriction.operators))):
            last_condition = index == number_of_restrictions

            if not boolean and last_condition and previous_operator == 'and':
                previous_operator = operator
                continue

            if not boolean and not last_condition and operator == 'and':
                previous_operator = operator
                continue

            if not boolean and not last_condition and operator == 'or':
                previous_operator = operator
                boolean = True
                continue

            condition_type, condition_operator = condition.value

            if condition_type.find('Length') != -1: value = getattr(vessel, 'L')
            if condition_type.find('Draught') != -1: value = getattr(vessel, 'T_f')
            if condition_type.find('Beam') != -1: value = getattr(vessel, 'B')
            if condition_type.find('UKC') != -1: value = 0
            if condition_type.find('Type') != -1: value = getattr(vessel, 'type')

            df = pd.DataFrame({'Value': [value], 'Restriction': [limit]})
            boolean = df.eval('Value' + condition_operator + 'Restriction')[0]

            # - if condition is not met: continue loop
            if not boolean and not last_condition:
                previous_operator = operator
                continue

            # - if one of the conditions is met and the restriction contains an 'OR' boolean statement:
            if boolean and not last_condition and operator == 'or':
                restriction_applies = True
                break

            # - elif all the conditions are met:
            if boolean and last_condition:
                restriction_applies = True
                break

            previous_operator = operator

        return restriction_applies

    def release_access_previous_section(vessel, origin):
        """ Function: when a vessel sails out of a section, it releases the request of the previous section

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - origin: string that contains the node that the vessel is currently on

        """
        # If a junction is encountered on that node: extract information of the previous junction
        if 'Junction' in vessel.env.FG.nodes[origin] and origin != vessel.route[0]:
            for previous_node in reversed(vessel.route[:vessel.route.index(origin)]):
                if 'Junction' not in vessel.env.FG.nodes[previous_node]:
                    continue

                previous_junction = vessel.env.FG.nodes[previous_node]['Junction']

                if origin not in vessel.request_access_waterway:
                    continue

                previous_junction.resource[origin].release(vessel.request_access_waterway[origin])
                if 'request_one_way_access_waterway' in dir(vessel):
                    junction = vessel.env.FG.nodes[origin]['Junction']
                    yield junction.resource[previous_node].release(vessel.request_one_way_access_waterway)
                    delattr(vessel, 'request_one_way_access_waterway')
                break

        return

    def request_access_next_section(vessel, origin):
        """ Function: when a vessel sails out of a section, it releases the request of the previous section

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - origin: string that contains the node of the route that the vessel is currently on
                - destination: string that contains the node of the route that the vessel is heading to

        """

        def check_if_request_is_in_same_direction(vessel,request,ahead_node):
            boolean = False
            if not request.obj.route.index(ahead_node) == len(request.obj.route)-1 and request.obj.route[request.obj.route.index(ahead_node) + 1] == vessel.route[vessel.route.index(ahead_node) + 1]:
                boolean = True
            elif request.obj.route[request.obj.route.index(ahead_node) - 1] == vessel.route[vessel.route.index(ahead_node) - 1]:
                boolean = True

            return boolean

        if 'request_access_waterway' not in dir(vessel):
            vessel.request_access_waterway = {}

        if 'request_one_way_access_waterway' not in dir(vessel):
            vessel.request_one_way_access_waterway = {}

        waiting_time = 0
        junctions, ahead_nodes, next_nodes = HasWaterway.open_waterway(vessel,origin)
        for junction, ahead_node, next_node in zip(junctions, ahead_nodes, next_nodes):
            print(vessel.name,ahead_node,next_node,waiting_time)
            sailing_time_to_ahead_junction = vessel_traffic_service.VesselTrafficService.provide_sailing_time(vessel,vessel,vessel.route[vessel.route.index(origin):(vessel.route.index(ahead_node) + 1)])
            sailing_time_to_next_junction = vessel_traffic_service.VesselTrafficService.provide_sailing_time(vessel,vessel,vessel.route[vessel.route.index(origin):(vessel.route.index(next_node)+1)])
            vessel.estimated_time_of_leaving_waterway = vessel.env.now + sailing_time_to_next_junction

            next_junction = vessel.env.FG.nodes[next_node]['Junction']
            encounter_restriction = False
            if junction.encounter_restrictions and next_node in junction.encounter_restrictions.keys():
                if junction.encounter_restrictions[next_node]:
                    encounter_restriction = PassWaterway.rule_determinator(vessel,junction,'encounter_restrictions',next_node)
                if encounter_restriction:
                    priority = 0
                    if next_junction.resource[ahead_node].users and not next_junction.resource[ahead_node].queue:
                        priority = next_junction.resource[ahead_node].users[0].priority - 1
                        user = next_junction.resource[ahead_node].users[0]
                        if 'request_one_way_access_waterway' in dir(user) and check_if_request_is_in_same_direction(vessel, user, ahead_node):
                            priority = 0

                    vessel.request_one_way_access_waterway[ahead_node] = next_junction.resource[ahead_node].request(priority=priority)
                    vessel.request_one_way_access_waterway[ahead_node].obj = vessel
                    vessel.request_one_way_access_waterway[ahead_node].eta = vessel.env.now + sailing_time_to_ahead_junction
                    vessel.request_one_way_access_waterway[ahead_node].etd = vessel.request_one_way_access_waterway[ahead_node].eta + (sailing_time_to_next_junction-sailing_time_to_ahead_junction)

            priority = 0
            if junction.resource[next_node].users and not junction.resource[next_node].queue:
                priority = junction.resource[next_node].users[0].priority - 1
                user = junction.resource[next_node].users[0]
                if not check_if_request_is_in_same_direction(vessel, user, ahead_node):
                    priority = 0

            vessel.request_access_waterway[next_node] = junction.resource[next_node].request(priority=priority)
            vessel.request_access_waterway[next_node].obj = vessel
            vessel.request_access_waterway[next_node].eta = vessel.env.now + sailing_time_to_ahead_junction
            scheduled_eta = new_eta = vessel.request_access_waterway[next_node].eta + waiting_time
            vessel.request_access_waterway[next_node].etd = vessel.request_access_waterway[next_node].eta + (sailing_time_to_next_junction-sailing_time_to_ahead_junction)

            if encounter_restriction:
                if next_junction.resource[ahead_node].queue:
                    if len(next_junction.resource[ahead_node].queue) == 1:
                        queued_request = next_junction.resource[ahead_node].users[0]
                        if check_if_request_is_in_same_direction(vessel, queued_request, ahead_node):
                            queue_index = -1
                        else:
                            queue_index = 0
                    else:
                        queue_index = len(next_junction.resource[ahead_node].queue[:-1])
                        for index,queued_request in enumerate(reversed(next_junction.resource[ahead_node].queue[:-1])):
                            if check_if_request_is_in_same_direction(vessel, queued_request, ahead_node):
                                if not encounter_restriction or queued_request.eta >= vessel.env.now+sailing_time_to_ahead_junction+waiting_time:
                                    queue_index = next_junction.resource[ahead_node].queue.index(queued_request)
                                    break
                                if index == len(next_junction.resource[ahead_node].queue[:-1])-1:
                                    queued_request = next_junction.resource[ahead_node].queue[-2]
                                    queue_index = next_junction.resource[ahead_node].queue.index(queued_request)

                    print('one-way',vessel.name, vessel.env.now, next_junction.resource[ahead_node].users[0].obj.name,queue_index,queued_request.obj.name,queued_request.eta,sailing_time_to_ahead_junction)
                    if queue_index+1:
                        if check_if_request_is_in_same_direction(vessel, queued_request, ahead_node):
                            new_eta = queued_request.eta
                            queue = next_junction.resource[ahead_node].queue
                            queue.insert(queue_index+1, vessel.request_one_way_access_waterway[ahead_node])
                            queue.pop(-1)
                        else:
                            new_eta = queued_request.etd

                    waiting_time += np.max([0,new_eta-scheduled_eta])

            print(waiting_time)
            scheduled_eta = new_eta = vessel.env.now + sailing_time_to_ahead_junction + waiting_time
            if junction.resource[next_node].queue:
                if len(junction.resource[next_node].queue) == 1:
                    queued_request = junction.resource[next_node].users[0]
                    if check_if_request_is_in_same_direction(vessel, queued_request, ahead_node):
                        queue_index = -1
                    else:
                        queue_index = 0
                else:
                    queue_index = len(junction.resource[next_node].queue[:-1])
                    for index,queued_request in enumerate(reversed(junction.resource[next_node].queue[:-1])):
                        if check_if_request_is_in_same_direction(vessel, queued_request, ahead_node):
                            if not encounter_restriction or queued_request.eta >= vessel.env.now+sailing_time_to_ahead_junction+waiting_time:
                                queue_index = junction.resource[next_node].queue.index(queued_request)
                                break
                            if index == len(next_junction.resource[ahead_node].queue[:-1])-1:
                                queued_request = junction.resource[next_node].queue[-2]
                                queue_index = junction.resource[next_node].queue.index(queued_request)

                print('two-way',queue_index,queued_request.obj.name,queued_request.eta,scheduled_eta)
                if queue_index+1:
                    if check_if_request_is_in_same_direction(vessel, queued_request, ahead_node):
                        print('hi')
                        new_eta = queued_request.eta
                        queue = junction.resource[next_node].queue
                        queue.insert(queue_index+1, vessel.request_access_waterway[next_node])
                        queue.pop(-1)
                    else:
                        new_eta = queued_request.etd

                waiting_time += np.max([0,new_eta-scheduled_eta])
                print(waiting_time)

        for junction, ahead_node, next_node in zip(junctions, ahead_nodes, next_nodes):
            encounter_restriction = False
            if junction.encounter_restrictions and next_node in junction.encounter_restrictions.keys():
                if junction.encounter_restrictions[next_node]:
                    encounter_restriction = PassWaterway.rule_determinator(vessel, junction, 'encounter_restrictions',next_node)
                if encounter_restriction:
                    vessel.request_one_way_access_waterway[ahead_node].eta += waiting_time
                    vessel.request_one_way_access_waterway[ahead_node].etd += waiting_time

            vessel.request_access_waterway[next_node].eta += waiting_time
            vessel.request_access_waterway[next_node].etd += waiting_time
            print(next_node,vessel.request_access_waterway[next_node].eta)

        print(waiting_time)
        yield vessel.env.timeout(waiting_time)
        return