import pandas as pd
import networkx as nx
import numpy as np
import datetime
import math

from opentnsim.core import Identifiable, Movable, HasResource, VesselProperties
from opentnsim.graph.utils import get_sailing_distance, get_sailing_time, node_path_to_edge_path, get_previous_and_next_edge_of_vessel
from opentnsim.port.utils import get_vessel_direction_with_waterway, get_waterway, get_oriented_waterway_route
from opentnsim.port.mixins.port import IsPortComponent


class WaterwayTraversable(Movable, Identifiable, VesselProperties):
    """Mixin class: Something can traverse an IsWaterway object
        
    Required for object that needs to pass a waterway (IsWaterway object)

    Methods
    -------
    request_waterway_access:
        requesting access to a waterway
    release_waterway_access: 
        releasing access to a waterway
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_edge_functions.append(self.request_waterway_access)
        self.on_complete_pass_edge_functions.append(self.release_waterway_access)

    def request_waterway_access(self, edge):
        """Request waterway access
        
        Parameters
        ----------
        edge: tuple of str
            resembling an edge in self.env.graph.edges
        """
        # set origin and destination
        origin, destination = edge

        # if no waterway in either origin or destination -> do nothing (return)
        if 'Waterway' not in self.env.graph.nodes[origin] or 'Waterway' not in self.env.graph.nodes[destination]:
            return

        # get waterway to pass
        waterway = get_waterway(self, edge)

        # if no waterway found -> do nothing (return)
        if waterway is None:
            return

        # if vessel has made a request but not for the found waterway -> error (should not happen)
        if hasattr(self, 'waterway_request') and self.waterway_request.waterway == waterway:
            raise ValueError(
                f"Vessel ({self.id}) did not request access to the correct waterway while accessing waterway ({waterway.name})."
            )
        # if vessel has already made a request for a waterway
        if hasattr(self, 'waterway_request'):
            # if this is the waterway in question -> do nothing (return)
            if self.waterway_request.waterway == waterway:
                return
            # else it is another waterway -> error (should not happen)
            else:
                raise ValueError(
                    f"Vessel ({self.id}) did not request access for waterway ({waterway.name}) but another waterway."
                )

        # create waterway request
        self.waterway_request = waterway.resource.request()
        self.waterway_request.trip_id = self.trip_id
        self.waterway_request.waterway = waterway

        # yield waterway request (formality -> instant: the waterway capacity is unlimited in ports)
        yield self.waterway_request

        # adapt status of passage_planning
        if self.trip_id in waterway.passage_planning.index:
            waterway.passage_planning.at[self.trip_id, 'is_passing'] = True

    def release_waterway_access(self, edge):
        """Releases waterway access

        Parameters
        ----------
        edge: tuple of str
            resembling an edge in self.env.graph.edges
        """
        # set origin and destination
        origin, destination = edge

        # if no waterway has been requested -> do nothing (return)
        if not hasattr(self, 'waterway_request'):
            return

        # if no waterway in either origin or destination -> do nothing (return)
        if 'Waterway' not in self.env.graph.nodes[origin] or 'Waterway' not in self.env.graph.nodes[destination]:
            return

        # get waterway to pass
        waterway = get_waterway(self, edge)
        waterway_route = get_oriented_waterway_route(waterway, self)

        # if no waterway found -> do nothing (return)
        if waterway is None:
            return

        # check if vessel leaves the waterway
        leaving_waterway = False
        if destination == waterway_route[-1]:
            leaving_waterway = True
        elif destination == self.route[-1]:
            leaving_waterway = True

        # if vessel does not leave the waterway -> do nothing (return)
        if not leaving_waterway:
            return

        # vessel request not in users -> error (should not be possible)
        if self.waterway_request not in waterway.resource.users:
            raise ValueError(
                f"Vessel ({self.id}) did not request access to waterway ({waterway.name}) while leaving it."
            )


        yield waterway.resource.release(self.waterway_request)

        # adapt status of passage_planning
        if self.trip_id in waterway.passage_planning.index:
            waterway.passage_planning.at[self.trip_id, 'is_passing'] = False

        # delete vessel's waterway_request-attribute -> opens possibility to request next waterway
        delattr(self, 'waterway_request')

class IsWaterway(Identifiable, HasResource, IsPortComponent):
    """Mixin class: An IsWaterway object
        
    Used to manage traffic on a route

    Parameters
    ----------
    node_start:

    node_stop:

    width:

    safety_margin:

    priority_rules:


    Attributes
    ----------
    node_start:

    node_stop:

    passage_planning:

    passing_vessels_per_edge:

    queue:

    safety_margin:

    priority_rules:


    Methods
    -------
    request_waterway_access:
        requesting access to a waterway
    release_waterway_access: 
        releasing access to a waterway
    """
    def __init__(
            self, 
            node_start, 
            node_stop, 
            safety_margin = pd.Timedelta(minutes=0), 
            priority_rules = None,
            *args, 
            **kwargs
        ):
        """Initialization"""
        self.node_start = node_start
        self.node_stop = node_stop
        self.passage_planning = pd.DataFrame(
            columns=[
                'vessel_id', 
                'vessel_name', 
                'vessel_L',
                'vessel_B',
                'vessel_T',
                'origin',
                'destination',
                'time_of_registration',
                'time_passage_start',
                'time_passage_stop',
                'direction',
                'priority',
                'is_passing',
            ],
        )
        self.passing_vessels_per_edge = pd.DataFrame(
            columns=[
                'vessel_id',
                'edge',
                'time_start',
                'time_stop',
                'direction',
                ],
            )
        self.queue = self.passage_planning.copy()
        self.safety_margin = safety_margin
        self.priority_rules = priority_rules
        super().__init__(nr_resources = math.inf, *args, **kwargs)

        self.port.waterways[self.name] = self
        self.graph = self.env.graph

        # Compute route (shortest path by default)
        self.route = nx.dijkstra_path(self.graph, self.node_start, self.node_stop)
        self.route_reversed = list(reversed(self.route))

        # Annotate nodes in the graph
        for node in self.route:
            if "Waterway" not in self.graph.nodes[node]:
                self.graph.nodes[node]["Waterway"] = [self]
            else:
                self.graph.nodes[node]["Waterway"].append(self)

        #Determine node distances
        edge_route = list(zip(self.route[:-1],self.route[1:]))
        _, edge_distance_df = get_sailing_distance(self.env.graph, edge_route)
        edge_distance_df['distance_stop'] = edge_distance_df['distance'].cumsum()
        edge_distance_df['distance_start'] = edge_distance_df['distance_stop'] - edge_distance_df['distance']
        
        start_nodes = edge_distance_df[
            ["node_start", "distance_start"]
        ].rename(columns={
            "node_start": "node",
            "distance_start": "distance"
        })

        stop_nodes = edge_distance_df[
            ["node_stop", "distance_stop"]
        ].rename(columns={
            "node_stop": "node",
            "distance_stop": "distance"
        })

        node_distance_df = pd.concat([start_nodes, stop_nodes])
        node_distance_df = (
            node_distance_df
            .drop_duplicates(subset="node")
            .sort_values("distance")
            .reset_index(drop=True)
        )
        
        self.node_distances = node_distance_df
        self.node_distance = node_distance_df.set_index("node")["distance"]
        self.edge_distances = edge_distance_df


    def add_vessel_to_passage_planning(self, vessel, origin, destination, delay = 0.):
        df = self.passage_planning
        passing_df = self.passing_vessels_per_edge.copy()
        added_to_planning = False
        if vessel.id in df.index:
            added_to_planning = True
        route_to_node_start = nx.dijkstra_path(self.graph, origin, self.node_start)
        route_to_node_stop = nx.dijkstra_path(self.graph, origin, self.node_stop)
        route_to_waterway = route_to_node_start
        route_over_waterway = self.route
        direction = 0
        if len(route_to_node_start) > len(route_to_node_stop):
            route_to_waterway = route_to_node_stop
            route_over_waterway = self.route_reversed
            direction = 1

        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))
        edge_route_to_waterway = node_path_to_edge_path(vessel.env.graph, route_to_waterway)
        edge_route_over_waterway = node_path_to_edge_path(vessel.env.graph, route_over_waterway)
        sailing_time_to_waterway, _ = get_sailing_time(vessel,edge_route_to_waterway)
        sailing_time_to_waterway = pd.Timedelta(seconds=sailing_time_to_waterway)
        last_message = pd.DataFrame(vessel.logbook).iloc[-1] if len(vessel.logbook) > 0 else None
        if last_message is not None and 'Sailing' in last_message.Message:
            start_time_sailing_on_current_node = last_message.timestamp
            sailing_time_on_current_edge = current_time - start_time_sailing_on_current_node
            sailing_time_to_waterway -= sailing_time_on_current_edge
        sailing_time_over_waterway, sailing_time_over_waterway_df = get_sailing_time(vessel,edge_route_over_waterway)
        time_passage_start = current_time + sailing_time_to_waterway + pd.Timedelta(seconds=delay)
        time_passage_stop = time_passage_start + pd.Timedelta(seconds=sailing_time_over_waterway)
        time_start = time_passage_start
        for _, sailing_info_edge in sailing_time_over_waterway_df.iterrows():
            edge = (sailing_info_edge.node_start, sailing_info_edge.node_stop)
            if direction:
                edge = (sailing_info_edge.node_stop, sailing_info_edge.node_start)
                
            if not added_to_planning:
                loc = len(self.passing_vessels_per_edge)
            else:
                try:
                    loc = passing_df[(passing_df.Edge == edge)&(passing_df.vessel_id == vessel.id)].index[0]
                except:
                    loc = len(self.passing_vessels_per_edge)
            
            time_stop = time_start + pd.Timedelta(seconds=sailing_info_edge.time)
            time_start, time_stop = time_start.round('s'), time_stop.round('s')
            self.passing_vessels_per_edge.at[loc, 'vessel_id'] = vessel.id
            self.passing_vessels_per_edge.at[loc, 'edge'] = edge
            self.passing_vessels_per_edge.at[loc, 'time_start'] = time_start
            self.passing_vessels_per_edge.at[loc, 'time_stop'] = time_stop
            self.passing_vessels_per_edge.at[loc, 'direction'] = direction
            time_start = time_stop
        self.passing_vessels_per_edge = self.passing_vessels_per_edge.sort_values("time_start").reset_index(drop=True)
        
        priority = 0
        priority = self.priority_rules(vessel) if self.priority_rules else priority
        vessel.priority = priority
        time_passage_start, time_passage_stop = time_passage_start.round('s'), time_passage_stop.round('s')
        if not added_to_planning:
            df.loc[vessel.trip_id] = [
                vessel.id, 
                vessel.name,
                vessel.L, 
                vessel.B, 
                vessel.T, 
                origin, 
                destination, 
                current_time, 
                time_passage_start, 
                time_passage_stop, 
                direction, 
                priority, 
                False
            ]
            df['time_of_registration'] = df['time_of_registration'].astype('datetime64[ns]')
            df['time_passage_start'] = df['time_passage_start'].astype('datetime64[ns]')
            df['time_passage_stop'] = df['time_passage_stop'].astype('datetime64[ns]')
            df.sort_values(by=['time_of_registration'], ascending=[True],inplace=True)
        else:
            registration_time = df.loc[vessel.id, 'time_of_registration']
            df.loc[vessel.trip_id] = [
                vessel.id, 
                vessel.name,
                vessel.L, 
                vessel.B, 
                vessel.T, 
                origin, 
                destination, 
                registration_time, 
                time_passage_start, 
                time_passage_stop, 
                direction, 
                priority, 
                False
            ]
            df.sort_values(by=['priority', 'time_passage_start'], ascending=[False, True],inplace=True)


    def check_if_vessels_can_pass_each_other_at_node(
            self, previous_vessel, next_vessel, node, route_previous_vessel = None, route_next_vessel = None
            ):
        node_info = self.env.graph.nodes[node]
        if 'Waterway' not in node_info:
            return True

        # passage possible if area is wide enough
        passage_possible = 'Anchorage' in node_info or 'Turning Basin' in node_info
        if passage_possible:
            return passage_possible

        # passage possible if waterway restriction ends and graph continues
        waterways = node_info['Waterway']
        if len(waterways) == 1:
            waterway = waterways[0]
            is_waterway_end = node == waterway.node_start or node == waterway.node_stop
            is_graph_end = len(self.env.graph[node]) == 1
            if is_waterway_end and not is_graph_end:
                return True

        # get orientation of vessels
        previous_edge_previous_vessel, next_edge_previous_vessel = get_previous_and_next_edge_of_vessel(
            previous_vessel, route = route_previous_vessel, node = node)
        previous_edge_next_vessel, next_edge_next_vessel = get_previous_and_next_edge_of_vessel(
            next_vessel, route = route_next_vessel, node = node)
        
        # get if vessels perform an overtaking or encountering manoeuver
        overtaking = False
        next_edge_is_same = next_edge_previous_vessel and next_edge_previous_vessel == next_edge_next_vessel
        previous_edge_is_same = previous_edge_previous_vessel and previous_edge_previous_vessel == previous_edge_next_vessel
        if next_edge_is_same or previous_edge_is_same:
            overtaking = True

        # get waterways of vessels
        next_waterway_previous_vessel = get_waterway(previous_vessel, next_edge_previous_vessel)
        previous_waterway_next_vessel = get_waterway(next_vessel, previous_edge_next_vessel)

        # get waterways in question
        waterways_of_passage = [self]
        if next_waterway_previous_vessel is not None:
            waterways_of_passage.append(next_waterway_previous_vessel)
        if previous_waterway_next_vessel is not None:
            waterways_of_passage.append(next_waterway_previous_vessel)
        waterways_of_passage = list(set(waterways_of_passage))

        # vessels move into or came from different waterways -> passage possible
        if len(waterways_of_passage) > 2:
            return True

        # determine if vessels are conflicting
        vessels = [previous_vessel, next_vessel]
        restriction = 1
        for waterway in waterways_of_passage:
            node_index = waterway.route.index(node)
            edges = []
            if node_index:
                previous_node = waterway.route[node_index - 1]
                edges.append([previous_node, node])
            if node_index != len(waterway.route) - 1:
                next_node = waterway.route[node_index - 1]
                edges.append([node, next_node])

            if overtaking:
                restriction = 1
                for edge in edges:
                    restriction, _, _ = self.check_for_overtaking_conflicts(edge, vessels)
                    if not restriction:
                        break
            else:
                restriction = 1
                for edge in edges:
                    restriction, _, _ = self.check_for_encountering_conflicts(edge, vessels)
                    if not restriction:
                        break

        return False if restriction else True


    def check_for_encountering_conflicts(self, edge, vessels):
        restriction = 0
        rules = []
        exceptions = []
        try:
            restrictions = self.env.graph.edges[edge]["Traffic_encountering_restriction"].evaluate(vessels)
            rules = [rule for rule, value in restrictions.items() if value == 0]
            restriction = 1 if rules else 0

            reservation_v1 = reservation_v2 = 0
            if "Traffic_reservation" in self.env.graph.edges[edge].keys():
                reservation_v1 = next(iter(self.env.graph.edges[edge]["Traffic_reservation"].evaluate(vessels[0]).values()))
                reservation_v2 = next(iter(self.env.graph.edges[edge]["Traffic_reservation"].evaluate(vessels[1]).values()))
            if reservation_v1 or reservation_v2:
                restriction = 1
                rules.extend(['reservation'])
            elif restriction and "Traffic_encountering_exception" in self.env.graph.edges[edge].keys():
                restrictions = self.env.graph.edges[edge]["Traffic_encountering_exception"].evaluate(vessels)
                exceptions = [rule for rule, value in restrictions.items() if value == 0]
                restriction = next(iter(restrictions.values()))
                if not restriction:
                    rules = []
        except:
            pass
        return restriction, rules, exceptions


    def check_for_overtaking_conflicts(self, edge, vessels):
        restriction = 0
        rules = []
        exceptions = []
        try:
            restrictions = self.env.graph.edges[edge]["Traffic_overtaking_restriction"].evaluate(vessels)
            rules = [rule for rule, value in restrictions.items() if value == 1]
            restriction = 1 if rules else 0
            if restriction and "Traffic_overtaking_exception" in self.env.graph.edges[edge].keys():
                restrictions = next(iter(self.env.graph.edges[edge]["Traffic_overtaking_exception"].evaluate(vessels).values()))
                exceptions = [rule for rule, value in restrictions.items() if value == 0]
                restriction = next(iter(restrictions.values()))
                if not restriction:
                    rules = []
        except:
            pass
        return restriction, rules, exceptions


    def check_conflicts_for_new_vessel(self, new_vessel):
        overtaking_conflicts = []
        encountering_conflicts = []
        current_time = datetime.datetime.fromtimestamp(new_vessel.env.now)
        new_vessel_direction = get_vessel_direction_with_waterway(self.route,new_vessel.route)
        start_node = self.node_distance.index[0]
        end_node = self.node_distance.index[-1]
        if new_vessel_direction:
            start_node, end_node = end_node, start_node
        new_vessel_priority = self.priority_rules(new_vessel) if self.priority_rules else 0
        passing_vessels_per_edge_df = self.passing_vessels_per_edge.copy()
        passing_vessels_per_edge_df = passing_vessels_per_edge_df[
            (passing_vessels_per_edge_df.vessel_id != new_vessel.id)&
            (passing_vessels_per_edge_df.time_stop >= current_time)
        ]

        for edge, group in passing_vessels_per_edge_df.groupby('edge'):
            if ("Traffic_encountering_restriction" not in self.env.graph.edges[edge].keys() 
                and "Traffic_overtaking_restriction" not in self.env.graph.edges[edge].keys()):
                continue

            for _, vessel_event in group.iterrows():
                vessel = self.env.vessels[vessel_event['vessel_id']]
                vessel_direction = vessel_event['direction']
                vessel_priority = self.passage_planning.at[vessel.trip_id, 'priority']
                vessel_on_waterway = self.passage_planning.at[vessel.trip_id, 'is_passing']
                if new_vessel_priority > vessel_priority and not vessel_on_waterway:
                    continue
                vessels = [vessel, new_vessel]
                if vessel_direction == new_vessel_direction:
                    # Overtaking
                    restriction, rules, exceptions = self.check_for_overtaking_conflicts(edge, vessels)
                    if restriction:
                        for time in vessel_event[['time_start', 'time_stop']]:
                            t_start = time - self.safety_margin
                            t_stop = time + self.safety_margin
                            overtaking_conflicts.append((edge, t_start, t_stop, vessels[0], vessels[1], rules))
                else:
                    # Encountering
                    restriction, rules, exceptions = self.check_for_encountering_conflicts(edge, vessels)
                    if restriction:
                        t_start = vessel_event['time_start']
                        t_stop = vessel_event['time_stop']
                        encountering_conflicts.append((edge, t_start, t_stop, vessels[0], vessels[1], rules))
        return encountering_conflicts, overtaking_conflicts
    

    def get_waterway_passage_information_for_vessel(self, vessel, origin):
        route_to_node_start = nx.dijkstra_path(self.graph, origin, self.node_start)
        route_to_node_stop = nx.dijkstra_path(self.graph, origin, self.node_stop)
        route_over_waterway = self.route
        direction = 0
        if len(route_to_node_start) > len(route_to_node_stop):
            route_over_waterway = self.route_reversed
            route_to_node_start = route_to_node_stop
            direction = 1
        edge_route_to_waterway = node_path_to_edge_path(vessel.env.graph, route_to_node_start)
        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        total_sailing_time_to_waterway = pd.Timedelta(seconds=0)
        total_sailing_time_to_waterway, _ = get_sailing_time(vessel, edge_route_to_waterway)
        total_sailing_time_to_waterway = pd.Timedelta(seconds=total_sailing_time_to_waterway)   
        last_message = pd.DataFrame(vessel.logbook).iloc[-1] if len(vessel.logbook) > 0 else None
        if last_message is not None and 'Sailing' in last_message.Message:
            start_time_sailing_on_current_node = last_message.timestamp
            sailing_time_on_current_edge = current_time - start_time_sailing_on_current_node
            total_sailing_time_to_waterway -= sailing_time_on_current_edge
        
        edge_route_over_waterway = node_path_to_edge_path(vessel.env.graph, route_over_waterway)
        _, sailing_time_over_waterway_df = get_sailing_time(vessel,edge_route_over_waterway)
        time_passage_start = current_time + total_sailing_time_to_waterway
        sailing_time_over_waterway_df['time_start'] = sailing_time_over_waterway_df['time'].shift(1).cumsum().astype('timedelta64[s]')
        sailing_time_over_waterway_df.loc[0,'time_start'] = np.timedelta64(0,'s')
        sailing_time_over_waterway_df['time_stop'] = sailing_time_over_waterway_df['time'].cumsum().astype('timedelta64[s]')
        sailing_time_over_waterway_df['time_start'] += time_passage_start
        sailing_time_over_waterway_df['time_stop'] += time_passage_start
        edges = list(zip(sailing_time_over_waterway_df["node_start"], sailing_time_over_waterway_df["node_stop"]))
        if direction:
            edges = list(zip(sailing_time_over_waterway_df["node_stop"], sailing_time_over_waterway_df["node_start"]))
        sailing_time_over_waterway_df['edge'] = edges
        return sailing_time_over_waterway_df, time_passage_start
    
    
    def get_waterway_availability_for_vessel(
            self, 
            encountering_conflicts, 
            overtaking_conflicts,
            vessel, 
            origin, 
            delay = 0.
            ):
        
        sailing_time_over_waterway_df, time_passage_start = self.get_waterway_passage_information_for_vessel(
            vessel, 
            origin)

        records = []

        route_to_node_start = nx.dijkstra_path(self.graph, origin, self.node_start)
        route_to_node_stop = nx.dijkstra_path(self.graph, origin, self.node_stop)
        if len(route_to_node_start) > len(route_to_node_stop):
            route_to_node_start = route_to_node_stop
        edge_route_to_waterway = node_path_to_edge_path(vessel.env.graph, route_to_node_start)
        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        total_sailing_time_to_waterway = pd.Timedelta(seconds=0)
        total_sailing_time_to_waterway, _ = get_sailing_time(vessel, edge_route_to_waterway)
        total_sailing_time_to_waterway = pd.Timedelta(seconds=total_sailing_time_to_waterway)   
        last_message = pd.DataFrame(vessel.logbook).iloc[-1] if len(vessel.logbook) > 0 else None
        if last_message is not None and 'Sailing' in last_message.Message:
            start_time_sailing_on_current_node = last_message.timestamp
            sailing_time_on_current_edge = current_time - start_time_sailing_on_current_node
            total_sailing_time_to_waterway -= sailing_time_on_current_edge

        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        time_passage_start = current_time + total_sailing_time_to_waterway + pd.Timedelta(seconds=delay)
        event_map = (sailing_time_over_waterway_df.set_index("edge")[["time_start", "time_stop"]])
        event_map.index = pd.MultiIndex.from_tuples(event_map.index)

        index = 0
        for conflict_index, conflicts in enumerate([encountering_conflicts, overtaking_conflicts]):
            conflict_type = "encountering"
            if conflict_index:
                conflict_type = "overtaking"
                conflict_index = index
                if conflict_index%2:
                    conflict_index += 1

            for index, (edge, time_start0, time_stop0, vessel1, vessel2, rules) in enumerate(conflicts):
                if vessel1.id == vessel2.id:
                    continue
                index += conflict_index
                event = event_map.loc[edge]

                time_stop = event.time_stop
                time_start = event.time_start
                if conflict_type == "overtaking":
                    if index%2:
                        time_start = time_stop
                    else:
                        time_stop = time_start
              
                time_start_not_available = (
                    time_passage_start - (time_stop - time_start0)
                )

                time_stop_not_available = (
                    time_passage_start - (time_start - time_stop0)
                )
                if time_stop_not_available < time_start_not_available:
                    time_start_not_available, time_stop_not_available = time_stop_not_available, time_start_not_available

                other_vessel = vessel1
                if other_vessel.id == vessel.id:
                    other_vessel = vessel2

                for rule in rules:
                    record = [
                        {
                            "time_start": time_start_not_available,
                            "time_stop": time_stop_not_available,
                            "edge": edge,
                            "type": conflict_type,
                            "vessel": other_vessel,
                            "rule": rule,
                        },
                    ]
                    records.extend(record)

        # aggregate results
        conflict_df = pd.DataFrame(records)
        time_start = sailing_time_over_waterway_df.time_start.min()
        time_stop = sailing_time_over_waterway_df.time_stop.min()
        waterway_availability_df = pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
        if conflict_df.empty:
            times = pd.DatetimeIndex(
                [pd.Timestamp(time_start), pd.Timestamp(time_stop)],
                name="time",
            )

            waterway_availability_df = pd.DataFrame(
                {"Traffic": [True, True]},
                index=times,
            )
            return waterway_availability_df, conflict_df
        
        conflict_df = conflict_df.sort_values("time_start").reset_index(drop=True)

        times = sorted(
            set(conflict_df["time_start"]) | set(conflict_df["time_stop"])
        )

        times = (
            [times[0] - pd.Timedelta(microseconds=1)]
            + times
            + [times[-1] + pd.Timedelta(microseconds=1)]
        )

        waterway_availability_df = pd.DataFrame(
            index=pd.DatetimeIndex(times, name="time")
        )

        waterway_availability_df["Traffic"] = [
            not (
                (conflict_df["time_start"] <= time)
                & (conflict_df["time_stop"] >= time)
            ).any()
            for time in times
        ]
        return waterway_availability_df, conflict_df
    

    def check_waterway_availability_info(self, vessel, origin, delay=0.):
        encountering_conflicts, overtaking_conflicts = (
            self.check_conflicts_for_new_vessel(vessel)
        )

        waterway_availability_df, conflict_df = (
            self.get_waterway_availability_for_vessel(
                encountering_conflicts,
                overtaking_conflicts,
                vessel,
                origin,
                delay,
            )
        )

        return waterway_availability_df, conflict_df

