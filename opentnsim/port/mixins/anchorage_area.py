from opentnsim.core import Movable, Identifiable, VesselProperties
from opentnsim.waiting_area import IsWaitingArea
from opentnsim.port.mixins.port import IsPortComponent
from opentnsim.port.utils import determine_nearest_anchorage_area

import simpy
import pandas as pd
import networkx as nx

class PassesAnchorage(Movable, Identifiable, VesselProperties):

    def __init__(self,*args,**kwargs):
        self.anchorage_areas = []
        super().__init__(*args, **kwargs)


    def sail_to_anchorage(self, node):
        """ Function: moves a vessel to the anchorage area instead of continuing its route to the terminal if a vessel is required to wait in the anchorage area

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node: a string that contains the node of the route that the vessel is currently on

        """

        # Set some default parameters:
        anchorage_area = determine_nearest_anchorage_area(self, node, route = self.route_ahead)
        yield from anchorage_area.request_anchorage_area_access(vessel=self)
        self.route_to_anchorage_area = nx.dijkstra_path(self.env.graph, node, anchorage_area.node)
        self.route_after_anchorage_area = nx.dijkstra_path(self.env.graph, anchorage_area.node, self.route[-1])
        if len(self.route_to_anchorage_area) > 1:
            self.routes_sailed.append(self.route_to_anchorage_area)
            self.route = self.route_to_anchorage_area
            try:
                yield self.env.process(self.move())
            except simpy.exceptions.Interrupt as e:
                raise e
        self.on_pass_node_functions.append(self.pass_anchorage)


    def pass_anchorage(self, origin):
        if 'Anchorage' not in self.env.graph.nodes[origin].keys():
            return

        if 'route_to_anchorage_area' not in dir(self):
            return

        yield from []

        self.route = self.route_after_anchorage_area
        delattr(self,'route_to_anchorage_area')
        delattr(self,'route_after_anchorage_area')
        self.env.process(self.move())
        #raise simpy.exceptions.Interrupt('Route of vessel has changed.')


class IsAnchorage(IsWaitingArea, IsPortComponent):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(self, depth, *args, **kwargs):
        self.depth = depth
        super().__init__(*args, **kwargs)

        self.register_waiting_area()

        self.port.anchorage_areas[self.name] = self
        if 'Anchorage' not in self.env.graph.nodes[self.node].keys():
            self.env.graph.nodes[self.node]['Anchorage'] = []
        self.env.graph.nodes[self.node]['Anchorage'].append(self)

    def request_anchorage_area_access(self, vessel):
        vessel.anchorage_area_request = self.resource.request()
        yield vessel.anchorage_area_request

    def release_anchorage_area_access(self, vessel):
        yield self.resource.release(vessel.anchorage_area_request)
