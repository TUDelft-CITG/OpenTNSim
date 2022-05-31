# package(s) related to the simulation
import simpy
import numpy as np

# OpenTNSim
from opentnsim import core

class HasSection(core.Movable):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_edge.append(self.request_access_new_section)

    def request_access_new_section(self,origin,destination):
        # Leave and access waterway section
        if 'Junction' in self.env.FG.nodes[origin].keys():
            if 'Anchorage' not in self.env.FG.nodes[origin].keys():
                waterway_module.PassSection.release_access_previous_section(self, origin)
                yield from waterway_module.PassSection.request_access_next_section(self, origin, destination)

class IsJunction(core.Identifiable,core.HasType, core.Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
        self,
        sections,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        "Initialization"

        self.sections = sections
        self.section = []

        # Loop over the sections which are connected to the junction
        for edge in enumerate(self.sections):
            # If section is type 'one-way-traffic'
            if self.type[edge[0]] == 'one-way_traffic':
                #Set default direction parameter
                direction = 0
                #If longitude of first node of the edge is smaller than the second node and strictly not equal to each other: set direction = 1
                if self.env.FG.nodes[edge[1][0]]['geometry'].x != self.env.FG.nodes[edge[1][1]]['geometry'].x:
                    if self.env.FG.nodes[edge[1][0]]['geometry'].x < self.env.FG.nodes[edge[1][1]]['geometry'].x:
                        direction = 1
                # Else is longitudes are equal: if latitude of first node of the edge is smaller than the second node: set direction = 1
                elif self.env.FG.nodes[edge[1][0]]['geometry'].y < self.env.FG.nodes[edge[1][1]]['geometry'].y:
                    direction = 1

                # If direction: append two access resources to that node
                if direction:
                    if 'access1' not in dir(self):
                        self.access1 = []
                        self.access2 = []

                    self.access1.append({edge[1][0]: simpy.PriorityResource(self.env, capacity=1),})
                    self.access2.append({edge[1][1]: simpy.PriorityResource(self.env, capacity=1),})

            # Append a resource to the section with 'infinite' capacity
            self.section.append({edge[1][1]: simpy.PriorityResource(self.env, capacity=10000),})

class PassSection:
    """Mixin class: Collection of functions that release and request sections. Important to obey the traffic regulations (safety distance and one-way-traffic) """

    def release_access_previous_section(vessel, origin):
        """ Function: when a vessel sails out of a section, it releases the request of the previous section

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - origin: string that contains the node that the vessel is currently on

        """

        # Reversely loop over the nodes of the route of the vessel (from the node that it is currently on backwards)
        for n in reversed(vessel.route[:vessel.route.index(origin)]):
            # If a junction is encountered on that node: extract information of the previous junction
            if 'Junction' in vessel.env.FG.nodes[n]:
                junction = vessel.env.FG.nodes[n]['Junction'][0]
                # Loop over the section of that junction
                for section in enumerate(junction.section):
                    # Pick the correct section by checking which section contains the current node of the vessel:
                    if origin not in list(section[1].keys()):
                        continue
                    # Release the request of that section made previously by the vessel
                    section[1][origin].release(vessel.request_access_section)

                    # If the section is of type 'one-way traffic':
                    if junction.type[section[0]] == 'one-way_traffic':
                        # If the entrance/exit resources are not in the previous junction: find the current junction and release the specific requests at that junction
                        if 'access1' not in dir(junction):
                            junction = vessel.env.FG.nodes[origin]['Junction'][0]
                            for section in enumerate(junction.section):
                                if n not in list(section[1].keys()):
                                    continue
                                junction.access2[0][n].release(vessel.request_access_entrance_section) #section[0]
                                junction.access1[0][origin].release(vessel.request_access_exit_section) #section[0]
                        # Else: release the specific requests of the entrance/exit resources at the previous junction
                        else:
                            junction.access1[0][n].release(vessel.request_access_entrance_section) #section[0]
                            junction.access2[0][origin].release(vessel.request_access_exit_section) #section[0]
                        break
                break
        return

    def request_access_next_section(vessel, origin, destination):
        """ Function: when a vessel sails out of a section, it releases the request of the previous section

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - origin: string that contains the node of the route that the vessel is currently on
                - destination: string that contains the node of the route that the vessel is heading to

        """

        # Loop over the nodes of the route of the vessel (from the node that it is heading to onwards)
        for n in vessel.route[vessel.route.index(destination):]:
            # If a junction is encountered on that node: extract information of the current junction
            if 'Junction' in vessel.env.FG.nodes[n]:
                junction = vessel.env.FG.nodes[origin]['Junction'][0]
                # Loop over the section of that junction
                for section in enumerate(junction.section):
                    # Pick the correct section by checking which section contains the current node of the vessel:
                    if n not in list(section[1].keys()):
                        continue
                    # Setting the stopping distance and stopping time
                    vessel.stopping_distance = 15 * vessel.L
                    vessel.stopping_time = vessel.stopping_distance / vessel.v
                    # If there is already a vessel present in the section and the time the current vessel arrives within the safety distance: request access and yield timeout until safety distance is complied to
                    if section[1][n].users != [] and (section[1][n].users[-1].ta + vessel.stopping_time) > vessel.env.now:
                        vessel.request_access_section = section[1][n].request()
                        section[1][n].users[-1].id = vessel.id
                        section[1][n].users[-1].ta = (section[1][n].users[-2].ta + vessel.stopping_time)
                        yield vessel.env.timeout((section[1][n].users[-2].ta + vessel.stopping_time) - vessel.env.now)
                    # Else if there are no other vessels present in the section: request access
                    else:
                        vessel.request_access_section = section[1][n].request()
                        section[1][n].users[-1].ta = vessel.env.now
                        section[1][n].users[-1].id = vessel.id

                    # If the section is of type 'one-way traffic':
                    if junction.type[section[0]] == 'one-way_traffic':
                        # If the entrance/exit resources are not in the current junction: find the next junction and request the specific requests at that junction
                        if 'access1' not in dir(junction):
                            junction = vessel.env.FG.nodes[n]['Junction'][0]
                            for section in enumerate(junction.section):
                                if origin not in list(section[1].keys()):
                                    continue

                                vessel.request_access_entrance_section = junction.access2[0][origin].request() #section[0]
                                junction.access2[0][origin].users[-1].id = vessel.id #section[0]
                                vessel.request_access_exit_section = junction.access1[0][n].request() #section[0]
                                junction.access1[0][n].users[-1].id = vessel.id #section[0]

                        # Else: request the specific requests for the entrance/exit resources at the current junction
                        else:
                            vessel.request_access_entrance_section = junction.access1[0][origin].request() #section[0]
                            junction.access1[0][origin].users[-1].id = vessel.id #section[0]
                            vessel.request_access_exit_section = junction.access2[0][n].request() #section[0]
                            junction.access2[0][n].users[-1].id = vessel.id #section[0]
                        break
                break
        return