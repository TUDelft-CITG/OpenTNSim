from turtle import delay

import pandas as pd
import networkx as nx
import numpy as np
import datetime
import simpy

from sphinx.addnodes import index
from opentnsim.core import SimpyObject, Identifiable, Movable, VesselProperties

#Imports from the port-module
from opentnsim.graph.utils import get_sailing_distance, get_sailing_time, node_path_to_edge_path 
from opentnsim.port.utils import get_vessel_direction_with_waterway


class WaterwayTraversable(Movable, Identifiable, VesselProperties):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_pass_node_functions.append(self.access_waterway)
        self.on_complete_pass_edge_functions.append(self.leave_waterway)


    def access_waterway(self, origin):
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


class IsWaterway(SimpyObject, Identifiable):
    def __init__(
            self, 
            node_start, 
            node_stop, 
            width, 
            safety_margin = pd.Timedelta(minutes=10), 
            priority_rules = None,
            *args, 
            **kwargs
            ):
        self.node_start = node_start
        self.node_stop = node_stop
        self.width = width
        self.passing_vessels = pd.DataFrame(columns=['Vessel_length','Vessel_beam','Vessel_draught','Time_passage_start','Time_passage_stop','Direction','Priority', 'Passing'])
        self.passing_vessels_per_edge = pd.DataFrame(columns=['Vessel_id','Edge','Time_start','Time_stop','Direction'])
        self.queue = self.passing_vessels.copy()
        self.safety_margin = safety_margin
        self.priority_rules = priority_rules
        super().__init__(*args, **kwargs)
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
        self.edge_distances = edge_distance_df
        super().__init__(*args, **kwargs)


    def add_vessel_to_passing_vessels(self, vessel, origin, delay = 0.):
        df = self.passing_vessels
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

        current_time = datetime.datetime.fromtimestamp(vessel.env.now) + pd.Timedelta(seconds=delay)
        edge_route_to_waterway = node_path_to_edge_path(vessel.env.graph, route_to_waterway)
        edge_route_over_waterway = node_path_to_edge_path(vessel.env.graph, route_over_waterway)
        sailing_time_to_waterway, _ = get_sailing_time(vessel,edge_route_to_waterway)
        sailing_time_over_waterway, sailing_time_over_waterway_df = get_sailing_time(vessel,edge_route_over_waterway)
        time_passage_start = current_time + pd.Timedelta(seconds=sailing_time_to_waterway)
        time_passage_stop = time_passage_start + pd.Timedelta(seconds=sailing_time_over_waterway)
        time_start = time_passage_start
        for _, sailing_info_edge in sailing_time_over_waterway_df.iterrows():
            edge = (sailing_info_edge.node_start, sailing_info_edge.node_stop)
            if direction:
                edge = (sailing_info_edge.node_stop, sailing_info_edge.node_start)
                
            if not added_to_planning:
                loc = len(self.passing_vessels_per_edge)
            else:
                passing_df = self.passing_vessels_per_edge.copy()
                loc = passing_df[(passing_df.Edge == edge)&(passing_df.Vessel_id == vessel.id)].index[0]
            
            time_stop = time_start + pd.Timedelta(seconds=sailing_info_edge.time)
            self.passing_vessels_per_edge.at[loc, 'Vessel_id'] = vessel.id
            self.passing_vessels_per_edge.at[loc, 'Edge'] = edge
            self.passing_vessels_per_edge.at[loc, 'Time_start'] = time_start
            self.passing_vessels_per_edge.at[loc, 'Time_stop'] = time_stop
            self.passing_vessels_per_edge.at[loc, 'Direction'] = direction
            time_start = time_stop
        self.passing_vessels_per_edge = self.passing_vessels_per_edge.sort_values("Time_start")
        
        priority = 0
        priority = self.priority_rules(vessel) if self.priority_rules else priority
        if not added_to_planning:
            df.loc[vessel.id] = [vessel.L, vessel.B, vessel.T, time_passage_start, time_passage_stop, direction, priority, False]
            df['Time_passage_start'] = df['Time_passage_start'].astype('datetime64[ns]')
            df['Time_passage_stop'] = df['Time_passage_stop'].astype('datetime64[ns]')
            df.sort_values("Time_passage_start",inplace=True)
        else:
            df.loc[vessel.id] = [vessel.L, vessel.B, vessel.T, time_passage_start, time_passage_stop, direction, priority, False]
        


    def update_passing_vessels_planning(
            self, 
            last_added_vessel,
            port_availability_df, 
            waiting_events, 
            total_waiting_time):
        df = self.passing_vessels

        vessel_ids_to_pass_waterway = df[~df.Passing].index.to_list()
        if last_added_vessel.id not in vessel_ids_to_pass_waterway:
            return port_availability_df, waiting_events, total_waiting_time
        
        vessel_ids_to_pass_waterway.remove(last_added_vessel.id)
        if not len(vessel_ids_to_pass_waterway):
            return port_availability_df, waiting_events, total_waiting_time
        
        priority_last_added_vessel = self.priority_rules(last_added_vessel) if self.priority_rules else 0
        vessel_priorities_to_pass_waterway = df.loc[vessel_ids_to_pass_waterway, 'Priority'].to_list()
        if np.max(vessel_priorities_to_pass_waterway) >= priority_last_added_vessel:
            return port_availability_df, waiting_events, total_waiting_time
        
        
        start_time_last_added_vessel = df.loc[last_added_vessel.id, 'Time_passage_start']
        vessel_stop_times_to_pass_waterway = df.loc[vessel_ids_to_pass_waterway, 'Time_passage_stop'].to_list()
        if np.min(vessel_stop_times_to_pass_waterway) <= start_time_last_added_vessel:
            return port_availability_df, waiting_events, total_waiting_time
        
        vessel_ids_to_pass_waterway = df[~df.Passing].index.to_list()
        new_planning_df = df.loc[vessel_ids_to_pass_waterway,:]
        is_first = new_planning_df.index[0] == last_added_vessel.id

        if is_first:
            return port_availability_df, waiting_events, total_waiting_time

        new_planning_df = new_planning_df.sort_values(by=["Priority", "Time_passage_start"], ascending=[False, True])
        static_df = self.passing_vessels[self.passing_vessels.Passing].copy()
        dynamic_df = self.passing_vessels[~self.passing_vessels.Passing].copy()
        dynamic_df = dynamic_df.sort_values(by=['Priority','Time_passage_start'], ascending=[False, True])
        self.passing_vessels = pd.concat([static_df,dynamic_df])
        for vessel_id, _ in new_planning_df.iterrows():
            vessel = self.env.vessels[vessel_id]
            if vessel_id != last_added_vessel.id:
                try:
                    vessel.mission.interrupt("Replanning vessel (complete pass edge, move to anchorage)")
                except:
                    pass
            else:
                port_availability_df, waiting_events, total_waiting_time = vessel.port.replan_vessel_trip(vessel, vessel.current_node)

        return port_availability_df, waiting_events, total_waiting_time

    def check_for_encountering_conflicts(self, edge, vessels):
        restriction = self.env.graph.edges[edge]["Traffic_encountering_restriction"].evaluate(vessels)
        reservation_v1 = reservation_v2 = 0
        if "Traffic_reservation" in self.env.graph.edges[edge].keys():
            reservation_v1 = self.env.graph.edges[edge]["Traffic_reservation"].evaluate(vessels[0])
            reservation_v2 = self.env.graph.edges[edge]["Traffic_reservation"].evaluate(vessels[1])
        if reservation_v1 or reservation_v2:
            restriction = 1
        elif restriction and "Traffic_encountering_exception" in self.env.graph.edges[edge].keys():
            restriction = self.env.graph.edges[edge]["Traffic_encountering_exception"].evaluate(vessels)
        return restriction


    def check_for_overtaking_conflicts(self, edge, vessels):
        restriction = self.env.graph.edges[edge]["Traffic_overtaking_restriction"].evaluate(vessels)
        if restriction and "Traffic_overtaking_exception" in self.env.graph.edges[edge].keys():
            restriction = self.env.graph.edges[edge]["Traffic_overtaking_exception"].evaluate(vessels)
        return restriction


    def check_conflicts_for_new_vessel(self, new_vessel):
        overtaking_conflicts = []
        encountering_conflicts = []
        new_vessel_direction = get_vessel_direction_with_waterway(self.route,new_vessel.route)
        new_vessel_priority = self.priority_rules(new_vessel) if self.priority_rules else 0
        passing_vessels_per_edge_df = self.passing_vessels_per_edge.copy()
        passing_vessels_per_edge_df = passing_vessels_per_edge_df[passing_vessels_per_edge_df.Vessel_id != new_vessel.id]
        
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
                    restriction = self.check_for_overtaking_conflicts(edge, vessels)
                    if restriction:
                        t_start = vessel_event['Time_start']
                        t_stop = vessel_event['Time_start'] + self.safety_margin
                        overtaking_conflicts.append((edge, t_start, t_stop))
                else:
                    # Encountering
                    restriction = self.check_for_encountering_conflicts(edge, vessels)
                    if restriction:
                        t_start = vessel_event['Time_start']
                        t_stop = vessel_event['Time_stop']
                        encountering_conflicts.append((edge, t_start, t_stop))
        return encountering_conflicts, overtaking_conflicts
    

    def get_waterway_passage_information_for_vessel(self, vessel, origin, delay = 0.):
        route_to_node_start = nx.dijkstra_path(self.graph, origin, self.node_start)
        route_to_node_stop = nx.dijkstra_path(self.graph, origin, self.node_stop)
        route_to_waterway = route_to_node_start
        route_over_waterway = self.route
        direction = 0
        if len(route_to_node_start) > len(route_to_node_stop):
            route_to_waterway = route_to_node_stop
            route_over_waterway = self.route_reversed
            direction = 1

        current_time = datetime.datetime.fromtimestamp(vessel.env.now) + pd.Timedelta(seconds=delay)
        edge_route_to_waterway = node_path_to_edge_path(vessel.env.graph, route_to_waterway)
        edge_route_over_waterway = node_path_to_edge_path(vessel.env.graph, route_over_waterway)
        sailing_time_to_waterway, _ = get_sailing_time(vessel,edge_route_to_waterway)
        _, sailing_time_over_waterway_df = get_sailing_time(vessel,edge_route_over_waterway)
        time_passage_start = current_time + pd.Timedelta(seconds=sailing_time_to_waterway)
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
            origin, 
            delay)
                
        records = []
        current_time = datetime.datetime.fromtimestamp(vessel.env.now) + pd.Timedelta(seconds=delay)
        event_map = (sailing_time_over_waterway_df.set_index("edge")[["time_start", "time_stop"]])
        event_map.index = pd.MultiIndex.from_tuples(event_map.index)
        eps = pd.Timedelta(milliseconds=10)
        for index, conflicts in enumerate([encountering_conflicts, overtaking_conflicts]):
            conflict_type = "encountering"
            if index:
                conflict_type = "overtaking"

            for edge, time_start0, time_stop0 in conflicts:
                event = event_map.loc[edge]

                time_start_not_available = (
                    time_start0 - (event.time_stop - current_time)
                )

                time_stop_not_available = (
                    time_stop0 - (event.time_start - current_time)
                )
                edge = edge + (conflict_type,)
                records.extend([
                    {
                        "time": time_start_not_available - eps,
                        "edge": edge,
                        "value": True,
                    },
                    {
                        "time": time_start_not_available,
                        "edge": edge,
                        "value": False,
                    },
                    {
                        "time": time_stop_not_available,
                        "edge": edge,
                        "value": False,
                    },
                    {
                        "time": time_stop_not_available + eps,
                        "edge": edge,
                        "value": True,
                    },
                ])

        if len(encountering_conflicts) or len(overtaking_conflicts):
            # Build once
            tmp = pd.DataFrame(records)
            waterway_availability_df = (
                tmp.pivot(index="time", columns="edge", values="value")
                .sort_index()
            )
        else:
            waterway_availability_df = pd.DataFrame({
                edge: [True]
                for edge in sailing_time_over_waterway_df['edge']
            }, index=[time_passage_start])

        waterway_availability_df = waterway_availability_df.ffill().bfill()
        false_mask = waterway_availability_df.eq(False)

        blocking_edges = false_mask.apply(
            lambda row: list(row.index[row.values]),
            axis=1
        )

        waterway_availability = ~false_mask.any(axis=1)
        waterway_availability_df = waterway_availability.to_frame(name="Traffic")
        waterway_availability_df["Cause"] = blocking_edges
        return waterway_availability_df
    
    
    def check_waterway_availability_info(self, vessel, origin, delay = 0.):
        encountering_conflicts, overtaking_conflicts = self.check_conflicts_for_new_vessel(vessel)
        waterway_availability_df = self.get_waterway_availability_for_vessel(
            encountering_conflicts, overtaking_conflicts, vessel, origin, delay)
        waterway_availability_df = waterway_availability_df[['Traffic']]
        df = waterway_availability_df.copy()
        
        # simplifying availability df
        while True:
            prev_len = len(df)

            # more than two consecutive True or False values -> keep only first and last instance
            s = df["Traffic"]
            grp = s.ne(s.shift()).cumsum()
            run_sizes = s.groupby(grp).transform("size")
            pos_in_group = s.groupby(grp).cumcount()
            keep_run_edges = (
                (run_sizes < 3)
                | pos_in_group.eq(0)
                | pos_in_group.eq(run_sizes - 1)
            )

            df = df.loc[keep_run_edges].copy()

            # isolated True values -> remove
            s = df["Traffic"]
            isolated_values = (
                s
                & ~s.shift(1, fill_value=False)
                & ~s.shift(-1, fill_value=False)
            )

            # prevent first and last values from being removed
            if len(df) > 0:
                isolated_values.iloc[0] = False
                isolated_values.iloc[-1] = False

            df = df.loc[~isolated_values].copy()

            if len(df) == prev_len:
                break

        waterway_availability_df = df
        return waterway_availability_df

