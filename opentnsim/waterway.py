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
            PassWaterway.release_access_previous_section(self, origin)

        if 'Detector' in self.env.FG.nodes[origin].keys():
            if destination != self.route[-1]:
                yield from PassWaterway.request_access_next_section(self, origin)

    def release_last_section(self,destination):
        if 'Junction' in self.env.FG.nodes[destination].keys() and destination == self.route[-1]:
            PassWaterway.release_access_previous_section(self, destination)

    def open_waterway(vessel,node):
        detector = vessel.env.FG.nodes[node]['Detector']
        infrastructures = []
        ahead_nodes = []
        next_nodes = []
        for ahead_node in vessel.route[vessel.route.index(node):]:
            if ahead_node not in detector.keys():
                continue

            infrastructure = detector[ahead_node].infrastructure
            infrastructures.append(infrastructure)

            if isinstance(infrastructure, IsWaterwayJunction):
                for next_node in vessel.route[vessel.route.index(ahead_node):]:
                    if [ahead_node, next_node] in infrastructure.sections:
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
                    junction.resource[previous_node].release(vessel.request_one_way_access_waterway)
                break

        return

    def request_access_next_section(vessel, origin):
        """ Function: when a vessel sails out of a section, it releases the request of the previous section

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - origin: string that contains the node of the route that the vessel is currently on
                - destination: string that contains the node of the route that the vessel is heading to

        """

        # Loop over the nodes of the route of the vessel (from the node that it is heading to onwards)

        if 'request_access_waterway' not in dir(vessel):
            vessel.request_access_waterway = {}
        junctions, ahead_nodes, next_nodes = HasWaterway.open_waterway(vessel,origin)
        for junction, ahead_node, next_node in zip(junctions, ahead_nodes, next_nodes):
            sailing_time_to_next_junction = vessel_traffic_service.VesselTrafficService.provide_sailing_time(vessel,vessel,vessel.route[vessel.route.index(origin):(vessel.route.index(next_node)+1)])
            vessel.estimated_time_of_leaving_waterway = vessel.env.now + sailing_time_to_next_junction
            if junction.encounter_restrictions and next_node in junction.encounter_restrictions.keys():
                encounter_restriction = PassWaterway.rule_determinator(vessel,junction,'encounter_restrictions',next_node)
                if encounter_restriction:
                    next_junction = vessel.env.FG.nodes[next_node]['Junction']
                    vessel.request_one_way_access_waterway = next_junction.resource[ahead_node].request(priority=0,) #preempt=True
                    vessel.request_one_way_access_waterway.obj = vessel
                    yield vessel.request_one_way_access_waterway

            if junction.overtaking_restrictions and next_node in junction.overtaking_restrictions.keys():
                overtaking_restriction = PassWaterway.rule_determinator(vessel, junction, 'overtaking_restrictions',next_node)
                if overtaking_restriction and junction.resource[next_node].users:
                    estimated_time_of_leaving_waterway_previous_vessel = junction.resource[next_node].users[0].obj.estimated_time_of_leaving_waterway
                    safety_time = (10 * vessel.L) / vessel.v
                    if vessel.estimated_time_of_leaving_waterway < (estimated_time_of_leaving_waterway_previous_vessel - safety_time):
                        yield vessel.env.timeout((estimated_time_of_leaving_waterway_previous_vessel - safety_time)-vessel.estimated_time_of_leaving_waterway)

            if junction.resource[next_node].users:
                vessel.request_access_waterway[next_node] = junction.resource[next_node].request(priority=0,) #preempt=True
            else:
                vessel.request_access_waterway[next_node] = junction.resource[next_node].request(priority=0,) #preempt=False

            vessel.request_access_waterway[next_node].obj = vessel

            try:
                yield vessel.request_access_waterway[next_node]
            except simpy.Interrupt as e:
                continue

        return