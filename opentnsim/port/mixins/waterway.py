import pandas as pd
import networkx as nx
import numpy as np
import datetime
from opentnsim.core import SimpyObject, Identifiable
from IPython.display import display

#Imports from the port-module
from opentnsim.port.mixins.rules import Expr, AggregateExpr, ComparisonExpr
from opentnsim.port.utils import get_vessel_from_id
from opentnsim.graph.utils import get_sailing_time

class PassesWaterway:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class IsWaterway(SimpyObject, Identifiable):
    def __init__(self, node_start, node_stop, width, *args, **kwargs):
        self.node_start = node_start
        self.node_stop = node_stop
        self.width = width
        self.passing_vessels = pd.DataFrame(columns=['Vessel_length','Vessel_beam','Vessel_draught','Time_passage_start','Time_passage_stop','Direction','Priority'])
        self.queue = self.passing_vessels.copy()
        super().__init__(*args, **kwargs)
        self.graph = self.env.graph
        self.rules = []
        self.variables = {"waterway_width": self.width}

        # Compute route (shortest path by default)
        self.route = nx.dijkstra_path(self.graph, self.node_start, self.node_stop)
        self.route_reversed = list(reversed(self.route))
        # Annotate nodes in the graph
        for node in self.route:
            if "Waterway" not in self.graph.nodes[node]:
                self.graph.nodes[node]["Waterway"] = self

        super().__init__(*args, **kwargs)  # in case of multiple inheritance


    def add_vessel_to_passing_vessels(self, vessel, origin, df = None, delay = 0., priority = 0):
        overwrite = False
        if df is None:
            overwrite = True
            df = self.passing_vessels
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
        sailing_time_to_waterway, _ = get_sailing_time(vessel,route_to_waterway)
        sailing_time_over_waterway, _ = get_sailing_time(vessel,route_over_waterway)
        time_passage_start = current_time + pd.Timedelta(seconds=sailing_time_to_waterway)
        time_passage_stop = time_passage_start + pd.Timedelta(seconds=sailing_time_over_waterway)
        waterway_available = True
        priority_passage = False
        other_vessels = []
        if overwrite:
            delay += (time_passage_start - current_time).total_seconds()
            waterway_available, priority_passage, other_vessels = self.check_waterway_availability(vessel, origin, priority=priority, delay=delay)

        df.loc[vessel.id] = [vessel.L, vessel.B, vessel.T, time_passage_start, time_passage_stop, direction, priority]
        if overwrite and not waterway_available and priority_passage:
            for other_vessel in other_vessels:
                other_vessel.waiting_event.interrupt()

        df['Time_passage_start'] = df['Time_passage_start'].astype('datetime64[ns]')
        df['Time_passage_stop'] = df['Time_passage_stop'].astype('datetime64[ns]')
        df.sort_values("Time_passage_start",inplace=True)


    def add_traffic_rule(self, rule: Expr):
        self.rules.append(rule)


    def find_vessel_passage_conflicts(self, df=None):
        conflicts = []
        overtaking_conflicts_list = []
        active_vessels = []

        if df is None:
            df = self.passing_vessels

        for id, vessel in df.iterrows():
            # Remove vessels that have already finished
            active_vessels = [other_vessel for other_vessel in active_vessels
                              if other_vessel['Time_passage_stop'] > vessel['Time_passage_start']]

            # Check encountering conflicts (opposite direction)
            encountering_group = [id]
            # Check overtaking conflicts (same direction)
            overtaking_group = [id]

            for other_vessel in active_vessels:
                if other_vessel['Direction'] != vessel['Direction']:
                    encountering_group.append(other_vessel.name)
                else:
                    overtaking_group.append(other_vessel.name)

            if len(encountering_group) > 1:
                conflicts.append(encountering_group)

            if len(overtaking_group) > 1:
                overtaking_conflicts_list.append(overtaking_group)

            # Add current vessel to active list
            active_vessels.append(vessel)

        # Remove duplicates using frozenset for both types
        def deduplicate_conflicts(conflict_list):
            unique_conflicts = {}
            seen_sets = []
            conflict_id = 1
            for group in conflict_list:
                group_set = frozenset(group)
                if group_set not in seen_sets:
                    unique_conflicts[f"conflict_{conflict_id}"] = np.array(list(group_set))
                    seen_sets.append(group_set)
                    conflict_id += 1
            return unique_conflicts

        encountering_conflicts = deduplicate_conflicts(conflicts)
        overtaking_conflicts = deduplicate_conflicts(overtaking_conflicts_list)
        return encountering_conflicts, overtaking_conflicts


    def extract_column_variable(self, rule):
        property = None
        rule = rule.render()
        for variable in self.passing_vessels.columns:
            if variable in rule:
                property = variable
        return property


    def extract_waterway_variables(self, rule):
        property = None
        rule = rule.render()
        for variable in self.variables:
            if variable in rule:
                property = variable
        return property


    def check_waterway_availability(self, vessel, origin, priority = 0, delay = 0., df = None):
        if df is None:
            df = self.passing_vessels
        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        df_copy = df.copy()
        df_copy = df_copy[df_copy.Time_passage_start > current_time]
        self.add_vessel_to_passing_vessels(vessel, origin, df=df_copy, delay=delay, priority=priority)
        df_copy = df_copy.sort_values("Time_passage_start")
        conflicts, priority_passage, other_vessels = self.check_for_conflicts(vessel, df=df_copy)
        available = True
        if conflicts:
            available = False
        return available, priority_passage, other_vessels


    def get_waterway_availability_info(self, vessel, origin, priority = 0, df = None):
        current_time = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now))
        if df is None:
            df = self.passing_vessels
        unique_times = pd.unique(df[["Time_passage_start", "Time_passage_stop"]].values.flatten()).astype("datetime64[ns]")
        unique_times = unique_times[unique_times >= current_time]
        availability_df = pd.DataFrame(columns=['Traffic'])
        for time in unique_times:
            delay = (time - current_time) / np.timedelta64(1, 's')
            available, priority_passage, _ = self.check_waterway_availability(vessel, origin, priority = priority, delay = delay, df = df)
            if priority_passage:
                available = True
            availability_df.loc[time] = available
        return availability_df


    def check_for_conflicts(self, vessel, df=None):
        if df is None:
            df = self.passing_vessels
        encountering_conflicts, overtaking_conflicts = self.find_vessel_passage_conflicts(df)
        conflicts = False
        rules_satisfied = True
        priority_passage = False
        other_vessels = []
        for encountering_conflict in encountering_conflicts.values():
            other_vessels_id = encountering_conflict[encountering_conflict != vessel.id]
            other_vessels = get_vessel_from_id(self.env,other_vessels_id)
            max_priority_other_vessels = np.max([other_vessel.priority for other_vessel in other_vessels])
            if vessel.priority > max_priority_other_vessels:
                rules_satisfied = False
                priority_passage = True
                continue
            for rule in self.rules:
                vessel_variable = self.extract_column_variable(rule)
                operator = rule.op.render()
                waterway_variable = self.extract_waterway_variables(rule)
                conflict_df = df.copy()
                conflict_df = conflict_df.loc[encountering_conflict]
                waterway_value = self.variables[waterway_variable]
                if isinstance(rule, AggregateExpr):
                    vessel_value = np.sum(conflict_df[vessel_variable])
                    rules_satisfied = pd.eval(f"{vessel_value}" + operator + f"{waterway_value}")
                elif isinstance(rule, ComparisonExpr):
                    rules_satisfied_df = pd.eval(f"conflict_df.{vessel_variable}" + operator + f"{waterway_value}")
                    rules_satisfied = True
                    if False in rules_satisfied_df.values:
                        rules_satisfied = False
                priority_passage = False
                other_vessels = []
        if not rules_satisfied:
            conflicts = True
        return conflicts, priority_passage, other_vessels

