import pandas as pd
import datetime
import math
import networkx as nx

from opentnsim.core import Identifiable, Movable, HasResource, VesselProperties
from opentnsim.graph.utils import get_sailing_distance, node_path_to_edge_path
from opentnsim.port.utils import (
    get_waterway, 
    get_waterway_orientation, 
    check_conflicts_for_vessel,
    get_waterway_passage_information_for_vessel,
)
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
        origin, destination = edge[:2]

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
        if self.trip_id in waterway.vessel_passages.index:
            waterway.vessel_passages.at[self.trip_id, 'is_passing'] = True

    def release_waterway_access(self, edge):
        """Releases waterway access

        Parameters
        ----------
        edge: tuple of str
            resembling an edge in self.env.graph.edges
        """
        # set origin and destination
        origin, destination = edge[:2]

        # if no waterway has been requested -> do nothing (return)
        if not hasattr(self, 'waterway_request'):
            return

        # if no waterway in either origin or destination -> do nothing (return)
        if 'Waterway' not in self.env.graph.nodes[origin] or 'Waterway' not in self.env.graph.nodes[destination]:
            return

        # get waterway to pass
        waterway = get_waterway(self, edge)
        _, waterway_route, _ = get_waterway_orientation(waterway, self)

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
        if self.trip_id in waterway.vessel_passages.index:
            waterway.vessel_passages.at[self.trip_id, 'is_passing'] = False

        # delete vessel's waterway_request-attribute -> opens possibility to request next waterway
        delattr(self, 'waterway_request')

class IsWaterway(Identifiable, HasResource, IsPortComponent):
    """Mixin class: An IsWaterway object
        
    Used to manage traffic on a route

    Parameters
    ----------
    node_start: str
        the node name at which the waterway starts as in env.graph.nodes
    node_stop: str
        the node name at which the waterway stops as in env.graph.nodes
    passage_planning: pandas.DataFrame
        stores passage information of vessels passing the waterway
    passing_vessels_per_edge: pandas.DataFrame
        stores sailing information of each vessel for each waterway edge specifically
    safety_margin: pandas.Timedelta, optional
        safety margin between two vessels in the same direction that cannot overtake each other
    priority_rules: callable(vessel) -> int, optional
        a function that imports a vessel and determines a priority to pass the waterway (higher integer is higher priority)

    Attributes
    ----------
    node_start: str
        the node name at which the waterway starts as in env.graph.nodes
    node_stop: str
        the node name at which the waterway stops as in env.graph.nodes
    passage_planning: pandas.DataFrame
        stores passage information of vessels passing the waterway
    passing_vessels_per_edge: pandas.DataFrame
        stores sailing information of each vessel for each waterway edge specifically
    safety_margin: pandas.Timedelta, optional
        safety margin between two vessels in the same direction that cannot overtake each other
    priority_rules: callable(vessel) -> int, optional
        a function that imports a vessel and determines a priority to pass the waterway (higher integer is higher priority)
    route: list of str
        the series of node names comprising the waterway in direction from start to stop node (direction = 0)
    route_reversed: list of str
        the series of node names comprising the waterway in opposite direction (direction = 1)
    edge_route: list of tuple of str
        the series of edge names comprising the waterway in direction from start to stop node (direction = 0)
    route: list of str
        the series of edge names comprising the waterway in opposite direction (direction = 1)
    edge_distances: pandas.DataFrame
        stored information of all the distances of the edge's nodes with respect to the start node
    node_distances: pandas.DataFrame
        stored information of all the distances of the nodes with respect to the start node

    Methods
    -------
    add_vessel_to_passage_planning:
        requesting access to a waterway
    check_waterway_availability_info: 
        releasing access to a waterway
    """
    def __init__(
            self, 
            node_start, 
            node_stop, 
            safety_margin = pd.Timedelta(minutes=10), 
            priority_rules = None,
            *args, 
            **kwargs
        ):
        """Initialization"""
        self.node_start = node_start
        self.node_stop = node_stop
        self.safety_margin = safety_margin
        self.priority_rules = priority_rules
        super().__init__(nr_resources = math.inf, *args, **kwargs)

        # creating dataframes for vessel passages of the waterway and per edge
        self.vessel_passages = pd.DataFrame(
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
        self.vessel_edge_passages = pd.DataFrame(
            columns=[
                'vessel_id',
                'trip_id',
                'edge',
                'time_start',
                'time_stop',
                'direction',
                ],
            )
        
        # add waterway to the dictionary of port waterways
        self.port.waterways[self.name] = self

        # compute and add node and edge routes in both directions
        self.route = nx.dijkstra_path(self.env.graph, self.node_start, self.node_stop, weight='length_m')
        self.route_reversed = list(reversed(self.route))
        self.edge_route = node_path_to_edge_path(self.env.graph, self.route)
        self.edge_route_reversed = node_path_to_edge_path(self.env.graph, self.route_reversed)

        # add waterway to edges in route TODO: remove this -> only on edges
        for node in self.route:
            if "Waterway" not in self.env.graph.nodes[node]:
                self.env.graph.nodes[node]["Waterway"] = [self]
            else:
                self.env.graph.nodes[node]["Waterway"].append(self)

        for edge in self.edge_route:
            if "Waterway" not in self.env.graph.edges[edge]:
                self.env.graph.edges[edge]["Waterway"] = [self]
            else:
                self.env.graph.edges[edge]["Waterway"].append(self)

        # calculate edge distances
        _, edge_distance_df = get_sailing_distance(self.env.graph, self.edge_route)
        edge_distance_df['distance_stop'] = edge_distance_df['distance'].cumsum()
        edge_distance_df['distance_start'] = edge_distance_df['distance_stop'] - edge_distance_df['distance']
        self.edge_distances = edge_distance_df

        # calculate node distances
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
        self.node_distance = node_distance_df.set_index("node")["distance"]
        
    def add_vessel_to_passage_planning(self, vessel, origin, delay = pd.Timedelta(seconds=0.)):
        """Adds vessel to the vessel passage planning and vessel edge passage times dataframe
        
        Parameters
        ----------
        vessel: obj
            the vessel object that should be WaterwayTransversable
        origin: str
            the origin node name of the vessel in env.graph.nodes
        destination: str
            the destination node name of the vessel in env.graph.nodes
        delay: float
            the delay of the vessel
        """
        df = self.vessel_passages
        passing_df = self.vessel_edge_passages.copy()

        # set current time
        current_time = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now))

        # determine if vessel has been added to the vessel passage planning
        added_to_planning = False
        registration_time = current_time
        if vessel.trip_id in df.index:
            added_to_planning = True
            registration_time = df.loc[vessel.trip_id, 'time_of_registration']
        
        # determine node and edge route to waterway and over waterway
        sailing_time_over_waterway_df, direction = get_waterway_passage_information_for_vessel(self, vessel, origin, delay)
        waterway_start_node = sailing_time_over_waterway_df['edge'].iloc[0][0]
        waterway_stop_node = sailing_time_over_waterway_df['edge'].iloc[-1][1]
        time_passage_start = sailing_time_over_waterway_df['time_start'].iloc[0]
        time_passage_stop = sailing_time_over_waterway_df['time_stop'].iloc[-1]
        time_passage_start, time_passage_stop = time_passage_start.round('s'), time_passage_stop.round('s')

        # determine vessel priority
        priority = self.priority_rules(vessel) if self.priority_rules else 0
        
        # add vessel to the vessel passage planning
        df.loc[vessel.trip_id] = [
            vessel.id, 
            vessel.name,
            vessel.L, 
            vessel.B, 
            vessel.T, 
            waterway_start_node, 
            waterway_stop_node, 
            registration_time, 
            time_passage_start, 
            time_passage_stop, 
            direction, 
            priority, 
            False
        ]

        # sort vessel passage planning
        df.sort_values(by=['priority', 'time_passage_start'], ascending=[False, True],inplace=True)

        # add vessel to the vessel passage planning per edge (by looping over edgewise sailing time over waterway dataframe)
        time_start = time_passage_start
        for _, sailing_info_edge in sailing_time_over_waterway_df.iterrows():
            # determine the directed edge
            edge = sailing_info_edge.edge
            if direction:
                edge = (edge[1], edge[0]) + edge[2:]

            # determine index
            index = len(self.vessel_edge_passages)
            if added_to_planning:
                index = passing_df[(passing_df.edge == edge)&(passing_df.trip_id == vessel.trip_id)].index[0]

            # determine time start and stop over edge
            time_stop = time_start + pd.Timedelta(seconds=sailing_info_edge.time)
            time_start, time_stop = time_start.round('s'), time_stop.round('s')

            # add to planning
            self.vessel_edge_passages.loc[index, :] = [
                vessel.id,
                vessel.trip_id,
                edge,
                time_start,
                time_stop,
                direction,
            ]

            # determine new start time of sailing over next edge
            time_start = time_stop

        # sort vessel edge passage dataframe
        self.vessel_edge_passages = self.vessel_edge_passages.sort_values("time_start").reset_index(drop=True)
    
    def check_waterway_availability_info(
            self, 
            vessel, 
            origin, 
            delay = pd.Timedelta(seconds=0.)
            ):
        """Gets the waterway availability of a vessel

        Parameters
        ----------
        vessel: obj
            the vessel object that is WaterwayTraversable
        origin: str
            the node name of the origin of the vessel in env.graph.nodes
        delay: pandas.Timedelta
            the delay of the vessel

        Returns
        -------
        waterway_availability_df: pandas.DataFrame
            availability of the waterway as a single-column dataframe ('Traffic') with True (available) and False (unavailable) values or time (datetime index)
        conflict_df: pandas.DataFrame
            a collection of conflicts with other vessels that the vessel is facing, including information on: time_start, time_stop, edge, type of conflict, vessel in conflict, specific rule that causes the conflict
        """
        # get the sailing information of the vessel over the waterway and get passage start time
        sailing_time_over_waterway_df, direction = get_waterway_passage_information_for_vessel(self, vessel, origin, delay)

        time_passage_start = sailing_time_over_waterway_df['time_start'].iloc[0]
        time_passage_stop = sailing_time_over_waterway_df['time_stop'].iloc[-1]

        # find the ecountering and overtaking conflicts for the vessel given the prior planned vessels
        encountering_conflicts, overtaking_conflicts = check_conflicts_for_vessel(self, vessel, direction)

        # set empty list for events that waterway is not available for the vessel in question
        events_waterway_not_available = []

        # construct event map (sailing information per edge)
        sailing_event_map = (sailing_time_over_waterway_df.set_index("edge")[["time_start", "time_stop"]])
        sailing_event_map.index = pd.MultiIndex.from_tuples(sailing_event_map.index)

        # loop over conficts
        for conflict_type, conflicts in {
             "encountering": encountering_conflicts, 
             "overtaking": overtaking_conflicts,
        }.items():
            for edge, time_conflict_start, time_conflict_stop, prior_planned_vessel, vessel, rules in conflicts:
                if prior_planned_vessel.id == vessel.id:
                    continue
                # get sailing event information of vessel on edge
                sailing_event = sailing_event_map.loc[edge]

                # determine start and stop time when edge is not available
                time_start_not_available = (
                    time_passage_start - (sailing_event.time_stop - time_conflict_start)
                )
                time_stop_not_available = (
                    time_passage_start - (sailing_event.time_start - time_conflict_stop)
                )

                # per rule: 
                for rule in rules:
                    event_waterway_not_available = [
                        {
                            "time_start": time_start_not_available,
                            "time_stop": time_stop_not_available,
                            "edge": edge,
                            "type": conflict_type,
                            "vessel": prior_planned_vessel,
                            "rule": rule,
                        },
                    ]
                    events_waterway_not_available.extend(event_waterway_not_available)
        
        # create conflict-dataframe by aggregating events that waterway is not available
        conflict_df = pd.DataFrame(events_waterway_not_available)

        # if there are no conflicts -> create waterway avilability dataframe of Trues
        if conflict_df.empty:
            time_index = pd.DatetimeIndex(
                [pd.Timestamp(time_passage_start), pd.Timestamp(time_passage_stop)],
                name="time",
            )
            waterway_availability_df = pd.DataFrame(
                {"Traffic": [True, True]},
                index=time_index,
            )
            return waterway_availability_df, conflict_df

        # else, if there are conflicts -> sort conflict dataframe
        conflict_df = conflict_df.sort_values("time_start").reset_index(drop=True)

        # create time index
        times = sorted(
            set(conflict_df["time_start"]) | set(conflict_df["time_stop"])
        )
        times = (
            [times[0] - pd.Timedelta(microseconds=1)]
            + times
            + [times[-1] + pd.Timedelta(microseconds=1)]
        )

        # crete waterway_availability_df
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
