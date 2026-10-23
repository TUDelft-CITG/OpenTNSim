from opentnsim.core import Movable, Identifiable, VesselProperties, HasResource, Log, Locatable
from opentnsim.port.mixins.port import IsPortComponent
from opentnsim.port.utils import determine_nearest_anchorage_area
from opentnsim.graph.mixins import OnNode

import simpy
import networkx as nx

class CanWaitAtAnchorage(Movable, Identifiable, VesselProperties):
    """Mixin class: Object that can wait at an IsAnchorage object
        
    Required for an object to wait at an anchorage area of a port

    Attributes
    ----------
    anchorage_areas: list of IsAnchorageArea objects
        stores the anchorage areas the vessel has visited

    Methods
    -------
    sail_to_anchorage:
        requesting access to a waterway
    pass_anchorage: 
        releasing access to a waterway
    """
    def __init__(self,*args,**kwargs):
        """Initialization"""
        self.anchorage_areas = []
        super().__init__(*args, **kwargs)

    def sail_to_anchorage(self, origin):
        """Vessel is able to find the nearest anchorage area and sail to it

        Parameters:
        ----------
        origin: str
            the name of a node in env.graph.nodes the vessel is currently at

        """
        # find nearest available anchorage area
        anchorage_area = determine_nearest_anchorage_area(self, origin, route = self.route_ahead)

        # request access to this anchorage area
        yield from self.request_anchorage_area_access(anchorage_area)

        # determine routes to and from anchorage areas
        self.route_to_anchorage_area = nx.dijkstra_path(self.env.graph, origin, anchorage_area.node)
        self.route_after_anchorage_area = nx.dijkstra_path(self.env.graph, anchorage_area.node, self.route[-1])

        # if the vessel is not yet at the anchorage area -> move to the anchorage area
        if len(self.route_to_anchorage_area) > 1:
            self.routes_sailed.append(self.route_to_anchorage_area)
            self.route = self.route_to_anchorage_area
            try:
                yield self.env.process(self.move())
            except simpy.exceptions.Interrupt as e:
                raise e

        # add process that allows vessel to continue trip after waiting at anchorage area TODO: check and aggregate with port.py -> create wait process here???
        self.on_pass_node_functions.append(self.pass_anchorage)

    def pass_anchorage(self, origin):
        """Vessel is able to continue its trip to its berth of destination (after waiting)
        
        Parameters:
        ----------
        origin: str
            the name of a node in env.graph.nodes the vessel is currently at
        """
        # if no anchorage on origin -> do nothing (return)
        if 'Anchorage' not in self.env.graph.nodes[origin].keys():
            return

        # get anchorage areas at origin node
        anchorage_areas = self.env.graph.nodes[origin]['Anchorage']

        # if vessel passes an anchorage but was not destined to an anchorage area
        if not hasattr(self, 'anchorage_area_request'):
            return

        # determine if one of anchorages at node is the anchorage the vessel is destined to
        anchorage_at_origin = False
        for anchorage_area in anchorage_areas:
            if anchorage_area == self.anchorage_area_request.anchorage_area:
                anchorage_at_origin = True
                break

        # if destined anchorage not at origin -> do nothing (return)
        if not anchorage_at_origin:
            return

        # create a generator so that the function can run as a simpy process
        yield from self.release_anchorage_area_access(anchorage_area)

        # set route to terminal from anchorage as new route of vessel
        self.route = self.route_after_anchorage_area

        # create new move process for vessel to sail to the terminal
        self.env.process(self.move())
        raise simpy.exceptions.Interrupt('Route of vessel has changed.')

    def request_anchorage_area_access(self, anchorage_area):
        """Requests access to an anchorage area
        
        Parameters
        ----------
        anchorage_area: obj
            the IsAnchorage object the vessel wants to access
        """
        # create request to access anchorage area
        self.anchorage_area_request = anchorage_area.resource.request()
        self.anchorage_area_request.anchorage_area = anchorage_area

        # yield request
        yield self.anchorage_area_request

    def release_anchorage_area_access(self, anchorage_area):
        """Releases request to access the anchorage area
        
        Parameters
        ----------
        anchorage_area: obj
            the IsAnchorage object the vessel wants to leave from
        """
        # yield release of request to access the anchorage area
        yield anchorage_area.resource.release(self.anchorage_area_request)

        # delete attributes to make future anchorage area interactions possible
        delattr(self,'anchorage_area_request')
        delattr(self,'route_to_anchorage_area')
        delattr(self,'route_after_anchorage_area')


class IsAnchorage(IsPortComponent, OnNode, HasResource, Log, Identifiable, Locatable):
    """Mixin class: An IsAnchorage object
            
    Used to create an anchorage area at a node of the env.graph

    Parameters
    ----------
    depth: float
        the depth in meters of the anchorage area
    capacity: int
        the capacity in number of vessels that the anchorage can handle at the same time

    Attributes
    ----------
    depth: float
        the depth in meters of the anchorage area
    """
    def __init__(self, depth, capacity, *args,**kwargs):
        """Initialization"""
        self.depth = depth
        super().__init__(nr_resources=capacity, *args, **kwargs)
        
        # adds the anchorage area to the collection of anchorage areas of the port
        self.port.anchorage_areas[self.name] = self
        
        # adds the anchorage area as waiting area to the infrastructure at the node
        if 'Waiting Area' not in self.env.graph.nodes[self.node].keys():
            self.env.graph.nodes[self.node]['Waiting Area'] = []
        self.env.graph.nodes[self.node]['Waiting Area'].append(self)

        # adds the anchorage area as anchorage area to the infrastructure at the node
        if 'Anchorage' not in self.env.graph.nodes[self.node].keys():
            self.env.graph.nodes[self.node]['Anchorage'] = []
        self.env.graph.nodes[self.node]['Anchorage'].append(self)
        