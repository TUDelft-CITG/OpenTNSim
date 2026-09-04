import pandas as pd
import networkx as nx
import numpy as np
import datetime
import math

from opentnsim.core import SimpyObject, Identifiable, Movable, VesselProperties

#Imports from the port-module
from opentnsim.graph.utils import get_sailing_distance, get_sailing_time, node_path_to_edge_path 
from opentnsim.port.utils import get_vessel_direction_with_waterway
from opentnsim.port.mixins.port import IsPortComponent


class WaterwayTraversable(Movable, Identifiable, VesselProperties):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_edge_functions.append(self.access_waterway)
        self.on_complete_pass_edge_functions.append(self.leave_waterway)


    def access_waterway(self, edge):
        origin = edge[0]
        if 'Waterway' not in self.env.graph.nodes[origin]:
            return
        
        waterway = self.env.graph.nodes[origin]['Waterway']
        waterway_route = waterway.route
        if origin in waterway_route and self.id in waterway.passing_vessels.index:
            waterway.passing_vessels.at[self.id, 'Passing'] = True
        yield from []


    def leave_waterway(self, destination):
        if 'Waterway' not in self.env.graph.nodes[destination]:
            return
        
        waterway = self.env.graph.nodes[destination]['Waterway']
        waterway_route = waterway.route
        # if destination == waterway_route[0] or destination == waterway_route[-1]:
        #     waterway.passing_vessels.drop(self.id, inplace=True, errors='ignore')
        yield from []


class IsWaterway(SimpyObject, Identifiable, IsPortComponent):
    def __init__(
            self, 
            node_start, 
            node_stop, 
            width = 200., 
            safety_margin = pd.Timedelta(minutes=15), 
            priority_rules = None,
            *args, 
            **kwargs
            ):
        self.node_start = node_start
        self.node_stop = node_stop
        self.width = width
        self.passing_vessels = pd.DataFrame(columns=['Vessel_length','Vessel_beam','Vessel_draught','Time_of_registration','Time_passage_start','Time_passage_stop','Direction','Priority', 'Passing'])
        self.passing_vessels_per_edge = pd.DataFrame(columns=['Vessel_id','Edge','Time_start','Time_stop','Direction'])
        self.queue = self.passing_vessels.copy()
        self.safety_margin = safety_margin
        self.priority_rules = priority_rules
        super().__init__(*args, **kwargs)
        self.port.waterways[self.name] = self
        self.graph = self.env.graph
        self.variables = {"waterway_width": self.width}

        # Compute route (shortest path by default)
        self.route = nx.dijkstra_path(self.graph, self.node_start, self.node_stop)
        self.route_reversed = list(reversed(self.route))

        # Annotate nodes in the graph
        for node in self.route:
            if "Waterway" not in self.graph.nodes[node]:
                self.graph.nodes[node]["Waterway"] = self

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
        super().__init__(*args, **kwargs)


    def add_vessel_to_passing_vessels(self, vessel, origin, delay = 0.):
        df = self.passing_vessels
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

        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        edge_route_to_waterway = node_path_to_edge_path(vessel.env.graph, route_to_waterway)
        edge_route_over_waterway = node_path_to_edge_path(vessel.env.graph, route_over_waterway)
        sailing_time_to_waterway, _ = get_sailing_time(vessel,edge_route_to_waterway)
        sailing_time_to_waterway = pd.Timedelta(seconds=sailing_time_to_waterway)
        last_message = pd.DataFrame(vessel.logbook).iloc[-1] if len(vessel.logbook) > 0 else None
        if last_message is not None and 'Sailing' in last_message.Message:
            start_time_sailing_on_current_node = last_message.Timestamp
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
                    loc = passing_df[(passing_df.Edge == edge)&(passing_df.Vessel_id == vessel.id)].index[0]
                except:
                    loc = len(self.passing_vessels_per_edge)
            
            time_stop = time_start + pd.Timedelta(seconds=sailing_info_edge.time)
            self.passing_vessels_per_edge.at[loc, 'Vessel_id'] = vessel.id
            self.passing_vessels_per_edge.at[loc, 'Edge'] = edge
            self.passing_vessels_per_edge.at[loc, 'Time_start'] = time_start
            self.passing_vessels_per_edge.at[loc, 'Time_stop'] = time_stop
            self.passing_vessels_per_edge.at[loc, 'Direction'] = direction
            time_start = time_stop
        self.passing_vessels_per_edge = self.passing_vessels_per_edge.sort_values("Time_start").reset_index(drop=True)
        
        priority = 0
        priority = self.priority_rules(vessel) if self.priority_rules else priority
        vessel.priority = priority
        if not added_to_planning:
            df.loc[vessel.id] = [
                vessel.L, vessel.B, vessel.T, current_time, time_passage_start, time_passage_stop, direction, priority, False]
            df['Time_of_registration'] = df['Time_of_registration'].astype('datetime64[ns]')
            df['Time_passage_start'] = df['Time_passage_start'].astype('datetime64[ns]')
            df['Time_passage_stop'] = df['Time_passage_stop'].astype('datetime64[ns]')
            df.sort_values(by=['Time_of_registration'], ascending=[True],inplace=True)
        else:
            registration_time = df.loc[vessel.id, 'Time_of_registration']
            df.loc[vessel.id] = [
                vessel.L, vessel.B, vessel.T, registration_time, time_passage_start, time_passage_stop, direction, priority, False]
            df.sort_values(by=['Priority', 'Time_passage_start'], ascending=[False, True],inplace=True)
        

    def update_passing_vessels_planning(
            self, 
            last_added_vessel,
            port_availability_df, 
            waiting_events, 
            total_waiting_time,
            traffic_conflicts_edge,
            traffic_conflicts_type,
            traffic_conflicts_vessels,
            traffic_conflict_rules,
            traffic_conflict_downtimes,):
        df = self.passing_vessels
        vessel_ids_to_pass_waterway = df[~df.Passing].index.to_list()

        if last_added_vessel.id not in vessel_ids_to_pass_waterway:
            return (port_availability_df, 
                waiting_events, 
                total_waiting_time, 
                traffic_conflicts_edge, 
                traffic_conflicts_type,
                traffic_conflicts_vessels,
                traffic_conflict_rules,
                traffic_conflict_downtimes,)
        
        vessel_ids_to_pass_waterway.remove(last_added_vessel.id)
        if not len(vessel_ids_to_pass_waterway):
            return (port_availability_df, 
                waiting_events, 
                total_waiting_time, 
                traffic_conflicts_edge, 
                traffic_conflicts_type,
                traffic_conflicts_vessels,
                traffic_conflict_rules,
                traffic_conflict_downtimes,)
        
        priority_last_added_vessel = self.priority_rules(last_added_vessel) if self.priority_rules else 0
        last_added_vessel_pos = df.index.get_loc(last_added_vessel.id)
        prior_df = df.iloc[:last_added_vessel_pos]
        prior_vessel_priorities_to_pass_waterway = [math.inf]
        if not prior_df.empty:
            prior_vessel_priorities_to_pass_waterway = prior_df['Priority'].to_list()
        if np.min(prior_vessel_priorities_to_pass_waterway) >= priority_last_added_vessel:
            return (port_availability_df, 
                waiting_events, 
                total_waiting_time, 
                traffic_conflicts_edge, 
                traffic_conflicts_type,
                traffic_conflicts_vessels,
                traffic_conflict_rules,
                traffic_conflict_downtimes,)
        
        start_time_last_added_vessel = df.loc[last_added_vessel.id, 'Time_passage_start']
        vessel_stop_times_to_pass_waterway = df.loc[vessel_ids_to_pass_waterway, 'Time_passage_stop'].to_list()
        if np.min(vessel_stop_times_to_pass_waterway) <= start_time_last_added_vessel:
            return (port_availability_df, 
                waiting_events, 
                total_waiting_time, 
                traffic_conflicts_edge, 
                traffic_conflicts_type,
                traffic_conflicts_vessels,
                traffic_conflict_rules,
                traffic_conflict_downtimes,)
        
        vessel_ids_to_pass_waterway = df[~df.Passing].index.to_list()
        modifiable_planning_df = df.loc[vessel_ids_to_pass_waterway,:]
        last_added_vessel_pos = modifiable_planning_df.index.get_loc(last_added_vessel.id)
        vessel_selection_df = modifiable_planning_df.iloc[:last_added_vessel_pos].loc[lambda x: x["Priority"] < priority_last_added_vessel]
        modifiable_planning_df = pd.concat([vessel_selection_df, modifiable_planning_df.loc[[last_added_vessel.id]]])
        is_first = modifiable_planning_df.index[0] == last_added_vessel.id
        if is_first:
            return (port_availability_df, 
                waiting_events, 
                total_waiting_time, 
                traffic_conflicts_edge, 
                traffic_conflicts_type,
                traffic_conflicts_vessels,
                traffic_conflict_rules,
                traffic_conflict_downtimes,)

        modifiable_planning_df = modifiable_planning_df.sort_values(by=["Priority", "Time_passage_start"], ascending=[False, True])
        self.passing_vessels_per_edge = self.passing_vessels_per_edge[(~self.passing_vessels_per_edge.Vessel_id.isin(modifiable_planning_df.index))].sort_values('Time_start').reset_index(drop=True)
        for vessel_id, _ in modifiable_planning_df.iterrows():
            vessel = self.env.vessels[vessel_id]
            if vessel_id != last_added_vessel.id:   
                if not vessel.waiting and "communicate_vessel_to_wait" in repr(vessel.on_pass_node_functions): 
                    vessel.on_pass_node_functions = [cb for cb in vessel.on_pass_node_functions if "communicate_vessel_to_wait" not in repr(cb)]
                vessel.port.communicate_port_accessibility_info(vessel, vessel.current_node)
                if vessel.waiting:
                    vessel.mission.interrupt('Changed waiting event')
            else:
                (
                    port_availability_df_per_waterway, 
                    waiting_events_per_waterway, 
                    total_waiting_time_per_waterway,
                    traffic_conflicts_edge_per_waterway,
                    traffic_conflicts_type_per_waterway,
                    traffic_conflicts_vessels_per_waterway,
                    traffic_conflict_rules_per_waterway,
                    traffic_conflict_downtimes_per_waterway
                    ) = vessel.port.replan_vessel_trip(vessel, vessel.current_node)
                port_availability_df = port_availability_df_per_waterway[self.name]
                waiting_events = waiting_events_per_waterway[self.name]
                total_waiting_time = total_waiting_time_per_waterway[self.name]
                traffic_conflicts_edge = traffic_conflicts_edge_per_waterway[self.name]
                traffic_conflicts_type = traffic_conflicts_type_per_waterway[self.name]
                traffic_conflicts_vessels = traffic_conflicts_vessels_per_waterway[self.name]
                traffic_conflict_rules = traffic_conflict_rules_per_waterway[self.name]
                traffic_conflict_downtimes = traffic_conflict_downtimes_per_waterway[self.name]
                self.passing_vessels.sort_values('Priority', ascending=False, inplace=True)

        self.passing_vessels.sort_values(by=['Time_of_registration','Priority', 'Time_passage_start'], ascending=[True, False, True], inplace=True)
        return (port_availability_df, 
                waiting_events, 
                total_waiting_time, 
                traffic_conflicts_edge, 
                traffic_conflicts_type,
                traffic_conflicts_vessels,
                traffic_conflict_rules,
                traffic_conflict_downtimes)

    def check_for_encountering_conflicts(self, edge, vessels):
        restriction = 0
        rules = []
        exceptions = []
        try:
            restrictions = self.env.graph.edges[edge]["Traffic_encountering_restriction"].evaluate(vessels)
            rules = [rule for rule, value in restrictions.items() if value == 1]
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
            (passing_vessels_per_edge_df.Vessel_id != new_vessel.id)&
            (passing_vessels_per_edge_df.Time_stop >= current_time)
        ]
        for edge, group in passing_vessels_per_edge_df.groupby('Edge'):
            if ("Traffic_encountering_restriction" not in self.env.graph.edges[edge].keys() 
                and "Traffic_overtaking_restriction" not in self.env.graph.edges[edge].keys()):
                continue

            for _, vessel_event in group.iterrows():
                vessel = self.env.vessels[vessel_event['Vessel_id']]
                vessel_direction = vessel_event['Direction']
                vessel_priority = self.passing_vessels.at[vessel.id, 'Priority']
                vessel_on_waterway = self.passing_vessels.at[vessel.id, 'Passing']
                if new_vessel_priority > vessel_priority and not vessel_on_waterway:
                    continue
                vessels = [vessel, new_vessel]
                if vessel_direction == new_vessel_direction:
                    # Overtaking
                    restriction, rules, exceptions = self.check_for_overtaking_conflicts(edge, vessels)
                    if restriction:
                        for time in vessel_event[['Time_start', 'Time_stop']]:
                            t_start = time - self.safety_margin
                            t_stop = time + self.safety_margin
                            overtaking_conflicts.append((edge, t_start, t_stop, vessels[0], vessels[1], rules))
                else:
                    # Encountering
                    restriction, rules, exceptions = self.check_for_encountering_conflicts(edge, vessels)
                    if restriction:
                        t_start = vessel_event['Time_start']
                        t_stop = vessel_event['Time_stop']
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
            start_time_sailing_on_current_node = last_message.Timestamp
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
            start_time_sailing_on_current_node = last_message.Timestamp
            sailing_time_on_current_edge = current_time - start_time_sailing_on_current_node
            total_sailing_time_to_waterway -= sailing_time_on_current_edge

        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        time_passage_start = current_time + total_sailing_time_to_waterway
        event_map = (sailing_time_over_waterway_df.set_index("edge")[["time_start", "time_stop"]])
        event_map.index = pd.MultiIndex.from_tuples(event_map.index)
        eps = pd.Timedelta(milliseconds=1)

        index = 0
        for conflict_index, conflicts in enumerate([encountering_conflicts, overtaking_conflicts]):
            conflict_type = "encountering"
            if conflict_index:
                conflict_type = "overtaking"
                conflict_index = index
                if conflict_index%2:
                    conflict_index += 1

            for index, (edge, time_start0, time_stop0, vessel1, vessel2, rules) in enumerate(conflicts):
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

                record = [
                    {   "index": index,
                        "time": time_start_not_available - eps,
                        "edge": edge,
                        "conflict_type": conflict_type,
                        "vessels": (vessel1, vessel2),
                        "rule": rules,
                        "value": True,
                    },
                    {
                        "index": index,
                        "time": time_start_not_available,
                        "edge": edge,
                        "conflict_type": conflict_type,
                        "vessels": (vessel1, vessel2),
                        "rule": rules,
                        "value": False,
                    },
                    {
                        "index": index,
                        "time": time_stop_not_available,
                        "edge": edge,
                        "conflict_type": conflict_type,
                        "vessels": (vessel1, vessel2),
                        "rule": rules,
                        "value": False,
                    },
                    {
                        "index": index,
                        "time": time_stop_not_available + eps,
                        "edge": edge,
                        "conflict_type": conflict_type,
                        "vessels": (vessel1, vessel2),
                        "rule": rules,
                        "value": True,
                    },
                ]
                records.extend(record)

        if len(encountering_conflicts) or len(overtaking_conflicts):
            df = pd.DataFrame(records)
            df = df.sort_values("time").reset_index(drop=True)

            edge_dfs = {}
            for edge, group in df.groupby("edge"):
                edge_dfs[edge] = (
                    group.pivot(
                        index="time",
                        columns="index",
                        values="value"
                    )
                    .ffill()
                    .bfill()
                )

                # Keep only the actual rule columns
                rule_cols = edge_dfs[edge].columns.tolist()

                # Which rules are False?
                edge_dfs[edge]["rule_index"] = edge_dfs[edge][rule_cols].apply(
                    lambda row: [col for col in rule_cols if not row[col]],
                    axis=1
                )

                # Available if ALL rules are True
                edge_dfs[edge]["availability"] = edge_dfs[edge][rule_cols].all(axis=1)


            combined_availability_df = pd.concat(
                {
                    edge: edge_df[["rule_index", "availability"]]
                    for edge, edge_df in edge_dfs.items()
                },
                axis=1
            ).ffill().bfill()

            # Select only the availability columns
            rule_index_cols = combined_availability_df.iloc[:, 0::2]
            availability_cols = combined_availability_df.iloc[:, 1::2]

            # Route availability: ALL edges must be available
            combined_availability_df["availability"] = availability_cols.all(axis=1)

            # Route rule_index: combine all lists
            combined_availability_df["rule_index"] = rule_index_cols.apply(
                lambda row: [
                    rule
                    for rules in row
                    for rule in rules
                ],
                axis=1
            )

            new_waterway_availability_df = combined_availability_df[['availability']].copy().rename(columns = {'availability':'Traffic'})
            new_conflict_df = combined_availability_df[['rule_index']].copy()

            lookup = (
                df
                .drop_duplicates("index")
                .set_index("index")[["edge", "conflict_type", "vessels", "rule"]]
            )

            def get_conflicts(indexes):
                if not indexes:
                    return {
                        "edges": [],
                        "conflict_type": [],
                        "vessels_in_conflict": [],
                        "rules": []
                    }

                rows = lookup.loc[indexes]

                return {
                    "edges": rows["edge"].tolist(),
                    "conflict_type": rows["conflict_type"].tolist(),
                    "vessels_in_conflict": rows["vessels"].tolist(),
                    "rules": rows["rule"].tolist()
                }

            conflicts = new_conflict_df["rule_index"].apply(get_conflicts)

            # Expand the dictionaries into columns
            result = conflicts.apply(pd.Series)
            result.index.name = "time"
            result['Traffic'] = new_waterway_availability_df['Traffic']
            result['Rule_index'] =  new_conflict_df["rule_index"]
            waterway_availability_df = result.copy()
        else:
            time_start = sailing_time_over_waterway_df.time_start.min()
            time_stop = sailing_time_over_waterway_df.time_stop.min()

            waterway_availability_df = pd.DataFrame(
                columns=[
                    "edges",
                    "conflict_type",
                    "vessels_in_conflict",
                    "rules",
                    "Traffic",
                    "Rule_index"
                ]
            )

            waterway_availability_df.index.name = "time"

            waterway_availability_df.loc[time_start] = [[], [], [], [], True, []]
            waterway_availability_df.loc[time_stop] = [[], [], [], [], True, []]

        return waterway_availability_df
    

    def check_waterway_availability_info(self, vessel, origin, delay=0.):

        encountering_conflicts, overtaking_conflicts = (
            self.check_conflicts_for_new_vessel(vessel)
        )

        waterway_availability_df = (
            self.get_waterway_availability_for_vessel(
                encountering_conflicts,
                overtaking_conflicts,
                vessel,
                origin,
                delay,
            )
        )

        df = waterway_availability_df.copy()
        df["downtime"] = None

        active_rules = {}
        rule_downtime = {}

        previous_rules = set()

        for timestamp, row in df.iterrows():

            current_rules = set(row["Rule_index"])

            # New rules
            for r in current_rules - previous_rules:
                active_rules[r] = timestamp

            # Rules that disappear
            for r in previous_rules - current_rules:
                rule_downtime[r] = timestamp - active_rules.pop(r)

            previous_rules = current_rules

        # Close any remaining rules
        last_time = df.index[-1]

        for r, start in active_rules.items():
            rule_downtime[r] = last_time - start

        df["_block"] = (
            df["Traffic"]
            .ne(df["Traffic"].shift())
            .cumsum()
        )

        output = []

        groups = list(df.groupby("_block", sort=False))

        first_block = groups[0][0]
        last_block = groups[-1][0]

        for block_number, group in groups:

            traffic = group["Traffic"].iloc[0]

            if traffic:

                is_first_block = block_number == first_block
                is_last_block = block_number == last_block
                has_two_or_more_rows = len(group) >= 2

                if (
                    is_first_block
                    or is_last_block
                    or has_two_or_more_rows
                ):
                    output.append(group.copy())

                continue

            conflicts = {}

            for timestamp, row in group.iterrows():

                for edge, conflict_type, vessels, rule, rule_idx in zip(
                    row["edges"],
                    row["conflict_type"],
                    row["vessels_in_conflict"],
                    row["rules"],
                    row["Rule_index"],
                ):

                    vessel_ids = tuple(sorted(v.id for v in vessels))

                    rule_key = tuple(rule) if isinstance(rule, list) else rule

                    key = (
                        edge,
                        rule_key,
                        vessel_ids,
                    )

                    if key not in conflicts:

                        conflicts[key] = {
                            "edge": edge,
                            "conflict_type": conflict_type,
                            "vessels": vessels,
                            "rule": rule,
                            "rule_idx": rule_idx,
                            "downtime": rule_downtime[rule_idx],
                        }

            conflict_list = list(conflicts.values())

            start = group.iloc[[0]].copy()
            stop = group.iloc[[-1]].copy()

            edges = [c["edge"] for c in conflict_list]
            conflict_types = [c["conflict_type"] for c in conflict_list]
            vessels = [c["vessels"] for c in conflict_list]
            rules = [c["rule"] for c in conflict_list]
            downtimes = [c["downtime"] for c in conflict_list]

            for row in (start, stop):

                idx = row.index[0]

                row.at[idx, "edges"] = edges
                row.at[idx, "conflict_type"] = conflict_types
                row.at[idx, "vessels_in_conflict"] = vessels
                row.at[idx, "rules"] = rules
                row.at[idx, "downtime"] = downtimes

            output.append(start)
            output.append(stop)

        result = pd.concat(output).sort_index()

        self.waterway_conflict_information = result.copy()
        return self.waterway_conflict_information

