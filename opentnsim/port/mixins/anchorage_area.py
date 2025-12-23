from opentnsim.core import HasResource, Identifiable, Log, Locatable, Movable
from opentnsim.output import HasOutput
from opentnsim.waiting_area import IsWaitingArea
from opentnsim.port.mixins.port import IsPartofPort

import simpy
import pandas as pd
import networkx as nx

class PassesAnchorage(Movable):

    def __init__(self,*args,**kwargs):
        self.anchorage_areas = []
        super().__init__(*args, **kwargs)


    def determine_sailing_time_to_anchorage_area(self, route_to_anchorage_area):
        sailing_time_to_anchorage_area = self.env.vessel_traffic_service.provide_sailing_time(self,route_to_anchorage_area)["Time"].sum()
        return sailing_time_to_anchorage_area


    def sail_to_anchorage(self, node):
        """ Function: moves a vessel to the anchorage area instead of continuing its route to the terminal if a vessel is required to wait in the anchorage area

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node: a string that contains the node of the route that the vessel is currently on

        """

        # Set some default parameters:
        anchorage_area = self.find_nearest_anchorage_area(node)
        yield from anchorage_area.request_anchorage_area_access(vessel=self)
        self.route_to_anchorage_area = nx.dijkstra_path(self.env.graph, node, anchorage_area.node)
        self.route_after_anchorage_area = nx.dijkstra_path(self.env.graph, anchorage_area.node, self.route[-1])
        if len(self.route_to_anchorage_area) > 1:
            self.routes_sailed.append(self.route_to_anchorage_area)
            self.route = self.route_to_anchorage_area
            self.env.process(self.move())
            raise simpy.exceptions.Interrupt('Route of vessel has changed.')
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
        raise simpy.exceptions.Interrupt('Route of vessel has changed.')


    def find_nearest_anchorage_area(self, node):
        provide_sailing_time = self.env.vessel_traffic_service.provide_sailing_time

        # Loop over the nodes of the network and identify all the anchorage areas:
        sailing_time_to_anchorages = []
        capacity_of_anchorages = []
        for anchorage_area in self.terminal.port.anchorage_areas:
            # Determine if the anchorage area can be reached
            route_to_anchorage = nx.dijkstra_path(self.env.graph, node, anchorage_area.node)
            sailing_time_to_anchorage = provide_sailing_time(self, route_to_anchorage)["Time"].sum()
            sailing_time_to_anchorages.append(sailing_time_to_anchorage)
            capacity_of_anchorages.append(anchorage_area.resource.capacity > 0)

        anchorage_selection_df = pd.DataFrame({'Sailing time':sailing_time_to_anchorages,
                                               'Capacity':capacity_of_anchorages})

        suitable_anchorage_areas = anchorage_selection_df[anchorage_selection_df.Capacity]
        suitable_anchorage_areas = suitable_anchorage_areas.sort_values('Sailing time')
        anchorage_area = self.terminal.port.anchorage_areas[suitable_anchorage_areas.iloc[0].name]
        return anchorage_area


class IsAnchorage(IsWaitingArea, IsPartofPort, HasOutput):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(self,depth,*args,**kwargs):
        self.depth = depth
        super().__init__(*args, **kwargs)
        self.port.anchorage_areas.append(self)
        self.env.graph.nodes[self.node]['Anchorage'] = self

    def request_anchorage_area_access(self, vessel):
        vessel.anchorage_area_request = self.resource.request()
        yield vessel.anchorage_area_request

    def release_anchorage_area_access(self, vessel):
        yield self.resource.release(vessel.anchorage_area_request)
