import simpy
import math
from opentnsim.core.movable import Interrupted

class HasDraughtRestrictions:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node_functions.append(self.wait_for_tidal_window)
        self.bound = 'inbound'

    def check_if_route_contains_restrictions(self):
        contains_restriction = False
        for node in self.route:
            if 'Vertical tidal restriction' in self.env.graph.nodes[node].keys():
                contains_restriction = True
                break
        return contains_restriction


    def find_route_with_restrictions(self):
        routes_with_restrictions = []
        route_with_restriction = []
        for node_start,node_stop in zip(self.route[:-1],self.route[1:]):
            if 'Vertical tidal restriction' in self.env.graph.nodes[node_start].keys():
                route_with_restriction.append(node_start)
            if 'Vertical tidal restriction' not in self.env.graph.nodes[node_stop].keys(): #create new route
                routes_with_restrictions.append(route_with_restriction)
                route_with_restriction = []
            elif node_stop == self.route[-1]: #if last node
                route_with_restriction.append(node_stop)
                routes_with_restrictions.append(route_with_restriction)
        return routes_with_restrictions


    def wait_for_tidal_window(self, origin):
        contains_restriction = self.check_if_route_contains_restrictions()
        if not contains_restriction:
            return

        routes_with_restrictions = self.find_route_with_restrictions()
        waiting_time = 0.

        if origin != routes_with_restrictions[0][0]:
            return

        for route in routes_with_restrictions:
            waiting_time = self.env.vessel_traffic_service.provide_waiting_time_for_inbound_tidal_window(vessel=self, route=route, delay=0, plot=True)

        if math.isnan(waiting_time):
            raise Interrupted('Port not accessible for vessel.')

        if not waiting_time:
            return

        self.log_entry_v0("Waiting for tidal window start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        yield self.env.timeout(waiting_time)
        self.log_entry_v0("Waiting for tidal window stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
