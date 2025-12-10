import simpy
import math


def check_if_route_contains_restrictions(vessel):
    contains_restriction = False
    for node in vessel.route:
        if 'Vertical tidal restriction' in vessel.env.graph.nodes[node].keys():
            contains_restriction = True
            break
    return contains_restriction


def find_route_with_restrictions(vessel):
    routes_with_restrictions = []
    route_with_restriction = []
    for node_start, node_stop in zip(vessel.route[:-1], vessel.route[1:]):
        if 'Vertical tidal restriction' in vessel.env.graph.nodes[node_start].keys():
            route_with_restriction.append(node_start)
        if 'Vertical tidal restriction' not in vessel.env.graph.nodes[node_stop].keys():  # create new route
            routes_with_restrictions.append(route_with_restriction)
            route_with_restriction = []
        elif node_stop == vessel.route[-1]:  # if last node
            route_with_restriction.append(node_stop)
            routes_with_restrictions.append(route_with_restriction)
    return routes_with_restrictions


class HasDraughtRestrictions:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.on_pass_node_functions.append(self.wait_for_tidal_window)
        self.bound = 'inbound'

    # def request_tidal_window(self, origin):
    #     infrastructure_suitable_to_wait_for_tidal_window = ['Waiting Area', 'Terminal']
    #     suitable_location_to_wait_for_tidal_window = False
    #     for infrastructure in infrastructure_suitable_to_wait_for_tidal_window:
    #         if infrastructure in self.env.graph.nodes[origin].keys():
    #             suitable_location_to_wait_for_tidal_window = True
    #             break
    #
    #     if not suitable_location_to_wait_for_tidal_window:
    #         return self.tidal_waiting_time
    #
    #     contains_restriction = check_if_route_contains_restrictions(self)
    #     if not contains_restriction:
    #         return self.tidal_waiting_time
    #
    #     routes_with_restrictions = find_route_with_restrictions(self)
    #     self.tidal_waiting_time = 0.
    #
    #     if origin != routes_with_restrictions[0][0]:
    #         return self.tidal_waiting_time
    #
    #     for route in routes_with_restrictions:
    #         self.tidal_waiting_time = self.env.vessel_traffic_service.provide_waiting_time_for_inbound_tidal_window(vessel=self, route=route, delay=0, plot=True)
    #
    #     if math.isnan(self.tidal_waiting_time):
    #         raise simpy.exceptions.Interrupt('Port not accessible for vessel.')
    #
    #     return self.tidal_waiting_time


    def wait_for_tidal_window(self, origin, waiting_time = None):
        if waiting_time is not None:
            self.tidal_waiting_time = waiting_time
        self.log_entry_v0("Waiting for tidal window start",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        yield self.env.timeout(self.tidal_waiting_time)
        self.log_entry_v0("Waiting for tidal window stop",
                          self.env.now,
                          self.distance,
                          self.env.graph.nodes[origin]["geometry"])
        self.tidal_waiting_time = 0.