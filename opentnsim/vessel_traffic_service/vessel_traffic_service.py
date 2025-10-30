# package(s) related to time, space and id
import bisect
import scipy as sc
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge
import shapely
import pandas as pd
from matplotlib import pyplot as plt, dates
import math
import networkx as nx
import datetime
import pytz
import pyproj

# mathematical packages
import bisect
import math
import scipy as sc

# packages for data handling
import numpy as np
import pandas as pd
import xarray as xr
import pickle
# import netCDF4
from shapely import reverse
from shapely.ops import transform
from opentnsim import core
from opentnsim import model
from opentnsim.graph.mixins import get_length_of_edge, get_geometry_of_edge, determine_length_of_edge_geometry
from opentnsim.graph import mixins
from opentnsim.vessel_traffic_service.hydrodanamic_data_manager import HydrodynamicDataManager

# spatial libraries
import networkx as nx
import shapely
import shapely.ops
from shapely.geometry import MultiPolygon, Point, Polygon


class VesselTrafficService(mixins.HasMultiDiGraph):
    """Class: a collection of functions that processes requests of vessels regarding the nautical processes on ow to enter the port safely"""

    def __init__(
        self,
        graph,
        crs_m = "EPSG:4087",
        hydrodynamic_information_path=None,
        vessel_speed_information_path=None,
        hydrodynamic_information=None,
        vessel_speed_information=None,
        hydrodynamic_start_time=pd.Timedelta(seconds=0),
        *args,
        **kwargs,
    ):
        self.crs_m = crs_m
        self.hydrodynamic_information = hydrodynamic_information
        self.hydrodynamic_information_path = hydrodynamic_information_path
        self.graph = graph

        if isinstance(hydrodynamic_information, xr.Dataset):
            self.hydrodynamic_information_path = False
        if isinstance(vessel_speed_information, xr.Dataset):
            self.vessel_speeds = vessel_speed_information

        global vertical_tidal_restrictions_condition_df
        self.vertical_tidal_restrictions_condition_df = pd.DataFrame()

        global horizontal_tidal_restrictions_condition_df
        self.horizontal_tidal_restrictions_condition_df = pd.DataFrame()

        global restricted_vessel_speeds
        self.restricted_vessel_speeds = pd.DataFrame()

        global edges_info
        self.edges_info = self.get_edges_info()

        for edge in self.graph.edges:
            length = determine_length_of_edge_geometry(graph, edge, crs_meter = self.crs_m)
            self.graph.edges[edge]["length_m"] = length

        for node in graph.nodes:
            node_info = graph.nodes[node]
            if 'Horizontal tidal restriction' in node_info.keys():
                specification_df = graph.nodes[node]['Horizontal tidal restriction']['Specifications']
                specification_df['Node'] = node
                self.horizontal_tidal_restrictions_condition_df = pd.concat(
                    [self.horizontal_tidal_restrictions_condition_df, specification_df]
                )
            if 'Vertical tidal restriction' in node_info.keys():
                specification_df = graph.nodes[node]['Vertical tidal restriction']['Specifications']
                specification_df['Node'] = node
                self.vertical_tidal_restrictions_condition_df = pd.concat(
                    [self.vertical_tidal_restrictions_condition_df, specification_df]
                )

        self.horizontal_tidal_restrictions_condition_df = self.horizontal_tidal_restrictions_condition_df.reset_index(drop=True)
        self.vertical_tidal_restrictions_condition_df = self.vertical_tidal_restrictions_condition_df.reset_index(drop=True)

        if self.hydrodynamic_information_path is not None:
            hydro_manager = HydrodynamicDataManager()
            if isinstance(hydrodynamic_information_path,str):
                hydro_manager.hydrodynamic_data = netCDF4.Dataset(self.hydrodynamic_information_path)
            else:
                hydro_manager.hydrodynamic_data = hydrodynamic_information

            self.hydrodynamic_start_time = hydrodynamic_start_time
            if isinstance(hydrodynamic_information_path, str):
                self.hydrodynamic_times = hydro_manager.hydrodynamic_times = (
                    hydro_manager.hydrodynamic_data["TIME"][:].data.astype("timedelta64[m]") + hydrodynamic_start_time
                )
            else:
                self.hydrodynamic_times = hydro_manager.hydrodynamic_data["TIME"][:]

        if isinstance(vessel_speed_information_path, str):
            with open(vessel_speed_information_path, "rb") as file:
                self.restricted_vessel_speeds = pickle.load(file)

        super().__init__(*args, **kwargs)

    def get_edges_info(self):
        graph = self.multidigraph
        edges_info = pd.DataFrame(columns=["Edge", "Distance", "MBL"])
        for edge in graph.edges:
            edge_info = graph.edges[edge]
            index = len(edges_info)
            edges_info.loc[index, "Edge"] = edge
            edges_info.loc[index, "Distance"] = get_length_of_edge(graph, edge)
            if "MBL" in edge_info.keys():
                edges_info.loc[index, "MBL"] = edge_info["MBL"]
            else:
                edges_info.loc[index, "MBL"] = 999.0
        return edges_info.set_index("Edge")

    def read_tidal_periods(self,hydrodynamic_data,tidal_period_type,station_index):
        if 'tidal_period_type' not in hydrodynamic_data.variables:
            return None
        data = hydrodynamic_data[tidal_period_type][station_index, :, :]
        for tide_index, (time_start, tide) in enumerate(data):
            if time_start == 'nan':
                new_time = 'NaT'
            else:
                new_time = time_start
            data[tide_index] = (np.datetime64(new_time), tide)
        return data

    def provide_waiting_time_for_inbound_tidal_window(self, vessel, route, time_start=None, time_stop=None, delay=0, plot=False):
        """Function: calculates the time that a vessel has to wait depending on the available tidal windows

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routeable, and has VesselProperties
            - route: a list of strings that resemble the route of the vessel (can be different than the vessel.route)
            - delay: a delay that can be included to calculate a future situation
            - plot: bool that specifies if a plot is requested or not

        """

        # Create sub-routes based on anchorage areas on the route
        if not time_start:
            time_start = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now)).to_datetime64()
        if not time_stop:
            time_stop = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now + vessel.metadata['max_waiting_time'])).to_datetime64()
        _,tidal_windows = self.provide_tidal_windows(vessel,route,time_start,time_stop,delay,plot=plot)

        waiting_time = pd.Timedelta(vessel.metadata["max_waiting_time"], "s")
        for window in tidal_windows:
            if time_start > window[1]:
                continue
            if time_start >= window[0]:
                waiting_time = pd.Timedelta(0, "s")
            else:
                waiting_time = window[0] - time_start
            break

        waiting_time = waiting_time.total_seconds()
        return waiting_time

    def provide_waiting_time_for_outbound_tidal_window(self,vessel,route,delay=0,plot=False):
        vessel.bound = 'outbound'
        vessel._T += vessel.metadata['loading'][0]
        waiting_time = self.provide_waiting_time_for_inbound_tidal_window(vessel,route=route,delay=delay, plot=plot)
        vessel._T -= vessel.metadata['loading'][0]
        vessel.bound = 'inbound'
        return waiting_time

    def provide_speed_over_edge(self,vessel,edge):
        v = vessel.v
        restricted_vessel_speeds_edge = self.restricted_vessel_speeds[self.restricted_vessel_speeds.index.isin([edge])]
        if not restricted_vessel_speeds_edge.empty:
            v = restricted_vessel_speeds_edge.Speed.iloc[0]
        if math.isnan(v):
            v = vessel.v
        if 'restricted_speed' in dir(vessel):
            v = vessel.restricted_speed
        return v

    def provide_speed_over_route(self,vessel,route,edges=[]):
        """
        Provides the speed along a vessel's route

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        route : list of str
            str resemble node names that have to be in the graph
        edges : list of tuple
            tuples resemble edges with: a start_node [u] as str, end_node (v) as str, and identifier (k) as int

        Returns
        vessel_speed_over_route : pd.DataFrame
            vessel speed per edge of the route
        -------
        """

        # get edges of route
        if not edges:
            for idx, (u, v) in enumerate(zip(route[:-1], route[1:])):
                k = sorted(self.multidigraph[u][v], key=lambda x: get_length_of_edge(self.multidigraph,(u, v, x)))[0]
                edges.append((u,v,k))

        # construct dataframe of speed information per edge
        vessel_speed_over_route = pd.DataFrame(columns=['Speed'],index=edges)

        # predefined speed restrictions should be added to the dataframe
        if not self.restricted_vessel_speeds.empty:
            vessel_speed_over_route = self.restricted_vessel_speeds[self.restricted_vessel_speeds.index.isin(edges)]
        vessel_speed_over_route = vessel_speed_over_route.reindex(edges)

        # missing speeds should be replaced with the vessel target speed
        vessel_speed_over_route[vessel_speed_over_route.Speed.isna() | (vessel_speed_over_route.Speed == 0)] = vessel.v

        # speeds overruled by lock masters should be obeyed to
        if 'overruled_speed' in dir(vessel):
            for edge,overruled_speed_limit in vessel.overruled_speed.iterrows():
                vessel_speed_over_route.loc[edge,'Speed'] = overruled_speed_limit

        return vessel_speed_over_route

    def provide_heading(self, vessel, edge):
        distance = []
        origin_location = vessel.multidigraph.nodes[edge[0]]["geometry"]
        k = sorted(vessel.multidigraph[edge[0]][edge[1]], key=lambda x: get_length_of_edge(self.multidigraph,(edge[0], edge[1], x)))[0]
        edge_geometry = vessel.multidigraph.edges[edge[0], edge[1], k]["Info"]["geometry"]
        for coord in edge_geometry.coords:
            distance.append(origin_location.distance(Point(coord)))
        if np.argmin(distance):
            edge_geometry = shapely.ops.transform(self.reverse_geometry, edge_geometry)
        heading = np.degrees(
            math.atan2(
                edge_geometry.coords[0][0] - edge_geometry.coords[-1][0], edge_geometry.coords[0][1] - edge_geometry.coords[-1][1]
            )
        )
        return heading

    def reverse_geometry(self, geometry):
        reversed_geometry = reverse(geometry)
        return reversed_geometry

    def provide_trajectory(self,node_1,node_2):
        geometry = None
        route = nx.dijkstra_path(self.multidigraph, node_1, node_2)
        for node_I, node_II in zip(route[:-1], route[1:]):
            k = sorted(self.multidigraph[node_I][node_II], key=lambda x: get_length_of_edge(self.multidigraph,(node_I, node_II, x)))[0]
            edge = (node_I, node_II, k)
            edge_geometry = self.multidigraph.edges[edge]['geometry']
            aligned = self.check_if_geometry_is_aligned_with_edge(edge)
            if not aligned:
                edge_geometry = self.reverse_geometry(edge_geometry)

            if geometry:
                geometry = shapely.ops.linemerge(MultiLineString([geometry, edge_geometry]))
            else:
                geometry = edge_geometry

        return geometry

    def provide_distance_over_network_to_location(self,node_1,node_2,location,tolerance=0.0001):
        geod = pyproj.Geod(ellps="WGS84")
        geometry = self.provide_trajectory(node_1,node_2)
        geometries = shapely.ops.split(shapely.ops.snap(geometry, location, tolerance=tolerance), location).geoms
        distance_sailed = 0
        distance_to_go = 0
        if len(geometries) < 2:
            if self.multidigraph.nodes[node_1]['geometry'] == location:
                distance_to_go = geod.geometry_length(geometries[0])
            elif self.multidigraph.nodes[node_2]['geometry'] == location:
                distance_sailed = geod.geometry_length(geometries[0])
            elif self.multidigraph.nodes[node_1]['geometry'].distance(location) > self.multidigraph.nodes[node_2]['geometry'].distance(location):
                distance_sailed = geod.geometry_length(geometries[0])
            else:
                distance_to_go = geod.geometry_length(geometries[0])
        else:
            distance_sailed = geod.geometry_length(geometries[0])
            distance_to_go = geod.geometry_length(geometries[1])
        return distance_sailed,distance_to_go

    def check_if_geometry_is_aligned_with_edge(self, edge):
        node_start = edge[0]
        node_stop = edge[1]
        edge_geometry = get_geometry_of_edge(self.multidigraph, edge)
        first_point = Point(edge_geometry.coords[0])
        distance_to_edge_nodes = {}
        for node in [node_start, node_stop]:
            node_geometry = self.multidigraph.nodes[node]["geometry"]
            distance_to_edge_nodes[node] = first_point.distance(node_geometry)
        closest_node = min(distance_to_edge_nodes, key=distance_to_edge_nodes.get)
        aligned = closest_node == node_start
        return aligned

    def get_closest_location_on_edge_to_point(self, graph, edge, point):
        edge_geometry = graph.edges[edge]["geometry"]
        point_on_edge = edge_geometry.interpolate(edge_geometry.project(point))
        return point_on_edge

    def transform_geometry(self, geometry, epsg_in="EPSG:4326", epsg_out=None):
        if epsg_out is None:
            epsg_out = self.crs_m
        crs_in = pyproj.CRS(epsg_in)
        crs_out = pyproj.CRS(epsg_out)
        crs_in_to_crs_out = pyproj.transformer.Transformer.from_crs(crs_in, crs_out, always_xy=True).transform
        geometry_transformed = transform(crs_in_to_crs_out, geometry)
        return geometry_transformed

    def provide_location_over_edges(self,node_1,node_2,interpolation_length):
        geometry = self.provide_trajectory(node_1, node_2)
        if geometry is None or geometry.is_empty:
            return None
        geometry_m = self.transform_geometry(geometry, epsg_out = self.crs_m)
        interpolation_point_m = geometry_m.line_interpolate_point(interpolation_length)
        interpolation_point = self.transform_geometry(interpolation_point_m, epsg_in = self.crs_m, epsg_out = "EPSG:4326")
        return interpolation_point

    def provide_distance_from_location_over_edge(self,edge,location,tolerance=0.0001):
        geod = pyproj.Geod(ellps="WGS84")
        if len(edge) == 2:
            k = sorted(self.multidigraph[edge[0]][edge[1]],key=lambda x: get_length_of_edge(self.multidigraph,(edge[0], edge[1], x)))[0]
            edge = (edge[0],edge[1],k)
        geometry = self.multidigraph.edges[(edge[0],edge[1],edge[2])]['geometry']

        distance_sailed = 0
        distance_to_go = 0
        if geometry.coords[0] == location.coords[0]:
            distance_to_go = self.multidigraph.edges[(edge[0],edge[1],edge[2])]['length']
        elif geometry.coords[-1] == location.coords[0]:
            distance_sailed = self.multidigraph.edges[(edge[0],edge[1],edge[2])]['length']
        else:
            lines = shapely.ops.split(shapely.ops.snap(geometry, location, tolerance), location).geoms
            for index, line in enumerate(lines):
                distance = 0
                for point_I, point_II in zip(line.coords[:-1], line.coords[1:]):
                    sub_edge_geometry = LineString([Point(point_I), Point(point_II)])
                    distance += geod.geometry_length(sub_edge_geometry)
                if not index:
                    distance_sailed = distance
                else:
                    distance_to_go = distance
        return distance_sailed, distance_to_go

    def provide_edge_by_distance_from_node(self,env,node_1,node_2,distance):
        route = nx.dijkstra_path(self.multidigraph, node_1, node_2)
        total_length = 0
        for node_I, node_II in zip(route[:-1], route[1:]):
            k = sorted(self.multidigraph[node_I][node_II],key=lambda x: get_length_of_edge(self.multidigraph,(node_I,node_II,x)))[0]
            edge_length = self.multidigraph.edges[node_I,node_II,k]['length']
            total_length += edge_length
            if total_length < distance:
                continue
            break
        return (node_I,node_II,k)

    def provide_sailing_distance(self,edge,k=None):
        """
        Calculates sailing distance of edge

        Parameters
        ----------
        edge : tuple
            tuple resembles an edge with: a start_node [u] as str, end_node (v) as str
        k : int [optional]
            identifier of the edge (used in networkx MultiDiGraph to distinguish between edges between the same pair of nodes)

        Returns
        -------
        sailing_distance : float
            sailing distance along the edge in [m]
        """
        if k is None:
            k = sorted(self.env.multidigraph[edge[0]][edge[1]], key=lambda x: get_length_of_edge(self.multidigraph,(edge[0],edge[1],x)))[0]

        sailing_distance = self.env.multidigraph.edges[edge[0], edge[1], k]['length']

        return sailing_distance

    def provide_sailing_distance_over_route(self, route, edges=None):
        """
        Calculates sailing distance of a route

        Parameters
        ----------
        vessel :
            a type including the following parent-classes: Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        edge : tuple
            tuple resembles an edge with: a start_node [u] as str, end_node (v) as str

        Returns
        -------
        sailing_distance_over_route : float
            sailing distance along the route in [m]
        """

        # get edge info
        edges_info = self.edges_info

        # get edges of route
        if not edges:
            edges = []
            for idx, (u, v) in enumerate(zip(route[:-1], route[1:])):
                k = sorted(self.multidigraph[u][v], key=lambda x: get_length_of_edge(self.multidigraph,(u, v, x)))[0]
                edges.append((u,v,k))

        # calculate sailing distance along route
        sailing_distance_over_route = edges_info[edges_info.index.isin(edges)]

        return sailing_distance_over_route

    def provide_sailing_time(self, vessel, route, edges=None):
        """
        Calculates sailing time of vessel

        Parameters
        ----------
        vessel :
            a type including the following parent-classes: Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        route : list of str
            str resemble node names that have to be in the graph
        edges : list of tuples
            tuples resemble edges with: a start_node [u] as str, end_node (v) as str, and identifier (k) as int

        Returns
        -------
        sailing_time_over_route : pd.DataFrame
            dataframe with edges as (multi)index and the following column-information: Speed, Distance, Time

        """

        # determine list of edges that the vessel passes given its route
        if not edges:
            edges = []
            for idx, (u, v) in enumerate(zip(route[:-1], route[1:])):
                k = sorted(self.multidigraph[u][v], key=lambda x: get_length_of_edge(self.multidigraph,(u, v, x)))[0]
                edges.append((u,v,k))

        # calculate sailing distance over route
        sailing_distance_over_route_df = self.provide_sailing_distance_over_route(route, edges)

        # obtain dataframe with information of sailing speed along route
        sailing_information_df = self.provide_speed_over_route(vessel, route, edges)

        # add distance and duraction information to the sailing information dataframe
        sailing_information_df['Distance'] = sailing_distance_over_route_df['Distance']
        sailing_information_df['Time'] = sailing_information_df['Distance']/sailing_information_df['Speed']

        return sailing_information_df

    def provide_sailing_time_distance_on_edge_to_distance_on_another_edge(self, vessel, route, distance_sailed_on_first_edge=0., distance_sailed_on_last_edge=0., edges=None):
        """
        Calculates the distance from a location along an edge A to another location along an edge B

        Parameters
        ----------
        vessel : type
            a type including the following parent-classes: Identifiable, Movable, VesselProperties, ExtraMetadata, HasMultiDiGraph, HasOutput
        route : list of str
            str resemble node names that have to be in the graph
        distance_sailed_on_first_edge : float
            distance that is already covered on the edge at which the vessel is currently sailing
        distance_sailed_on_last_edge : float
            distance on the last edge that the vessel has to sail to reach its location of interest
        edges : list of tuples
            tuples resemble edges with: a start_node [u] as str, end_node (v) as str, and identifier (k) as int

        Returns
        -------
        sailing_information_df : pd.DataFrame
            dataframe with edges as (multi)index and the following column-information: Speed, Distance, Time

        """

        # obtain dataframe with information of sailing speed, distance and time along route
        sailing_information_df = self.provide_sailing_time(vessel=vessel, route=route, edges=edges)

        # determine indexes of first and last edges
        index_first_edge = pd.Index([sailing_information_df.iloc[0].name])
        index_last_edge = pd.Index([sailing_information_df.iloc[-1].name])

        # determine distance that must still be sailed on the current edge of the vessel
        distance_to_sail_on_first_edge = (sailing_information_df.loc[index_first_edge, 'Distance']-distance_sailed_on_first_edge)

        # adjust information of the sailing distance and sailing time on the first and last edges
        sailing_information_df.loc[index_first_edge, 'Time'] = sailing_information_df.loc[index_first_edge, 'Time']*(distance_to_sail_on_first_edge/sailing_information_df.loc[index_first_edge, 'Distance'])
        sailing_information_df.loc[index_first_edge, 'Distance'] = distance_to_sail_on_first_edge
        sailing_information_df.loc[index_last_edge, 'Time'] = sailing_information_df.loc[index_last_edge, 'Time']*(distance_sailed_on_last_edge/sailing_information_df.loc[index_last_edge, 'Distance'])
        sailing_information_df.loc[index_last_edge, 'Distance'] = distance_sailed_on_last_edge

        return sailing_information_df

    def provide_nearest_anchorage_area(self, vessel, node):
        nodes_of_anchorages = []
        capacity_of_anchorages = []
        users_of_anchorages = []
        sailing_times_to_anchorages = []
        # Loop over the nodes of the network and identify all the anchorage areas:
        for node_anchorage in vessel.multidigraph.nodes:
            if "Anchorage" in vessel.multidigraph.nodes[node_anchorage]:
                # Determine if the anchorage area can be reached
                anchorage_reachable = True
                route_to_anchorage = nx.dijkstra_path(vessel.multidigraph, node, node_anchorage)
                for node_on_route in route_to_anchorage:
                    station_index = list(HydrodynamicDataManager().hydrodynamic_data["STATION"][:]).index(node_on_route)
                    min_water_level = np.min(HydrodynamicDataManager().hydrodynamic_data["Water level"][:, station_index].data)
                    _, _, _, required_water_depth, _, MBL = self.provide_ukc_clearance(vessel,node)
                    if min_water_level - MBL < required_water_depth:
                        anchorage_reachable = False
                        break

                if not anchorage_reachable:
                    continue

                # Extract information over the individual anchorage areas:
                # capacity, users, and the sailing distance to the anchorage area
                # from the designated terminal the vessel is planning to call
                nodes_of_anchorages.append(node_anchorage)
                capacity_of_anchorages.append(vessel.multidigraph.nodes[node_anchorage]["Anchorage"][0].resource.capacity)
                users_of_anchorages.append(len(vessel.multidigraph.nodes[node_anchorage]["Anchorage"][0].resource.users))
                route_from_anchorage = nx.dijkstra_path(vessel.multidigraph, node_anchorage, vessel.route[-1])
                sailing_time_to_anchorage = vessel.env.vessel_traffic_service.provide_sailing_time(vessel, route_from_anchorage)[
                    "Time"
                ].sum()
                sailing_times_to_anchorages.append(sailing_time_to_anchorage)

        # Sort the lists based on the sailing distance to the anchorage area from the designated terminal
        #  the vessel is planning to call
        sorted_nodes_anchorages = [nodes for (distances, nodes) in sorted(zip(sailing_times_to_anchorages, nodes_of_anchorages))]
        sorted_users_of_anchorages = [nodes for (distances, nodes) in sorted(zip(sailing_times_to_anchorages, users_of_anchorages))]
        sorted_capacity_of_anchorages = [
            nodes for (distances, nodes) in sorted(zip(sailing_times_to_anchorages, capacity_of_anchorages))
        ]

        # Take the anchorage area that is closest to the designated terminal the vessel is planning to call if there
        # is sufficient capacity:
        node_anchorage = 0
        for anchorage_index, node_anchorage in enumerate(sorted_nodes_anchorages):
            if sorted_users_of_anchorages[anchorage_index] < sorted_capacity_of_anchorages[anchorage_index]:
                # node anchorage is found
                break
        return node_anchorage

    def provide_governing_current_velocity(self,vessel,node,time_start_index,time_end_index):
        hydromanager = HydrodynamicDataManager()
        station_index = list(hydromanager.hydrodynamic_data["STATION"][:]).index(node)
        times = hydromanager.hydrodynamic_times[time_start_index:time_end_index]
        start_time = times[0]
        end_time = times[-1]
        time_step = times[1]-times[0]
        relative_layer_height = hydromanager.hydrodynamic_data["LAYER"][:].data
        current_velocity = hydromanager.hydrodynamic_data["Primary current velocity"][
            time_start_index:time_end_index, station_index
        ].data

        def depth_averaged_current_velocity(interpolation_depth, times, relative_layer_height, current_velocity, station_index):
            layer_boundaries = []
            average_current_velocity = []
            number_of_layers = len(relative_layer_height)
            water_depth = (
                -1 * hydromanager.hydrodynamic_data["MBL"][:, station_index].data
                + hydromanager.hydrodynamic_data["Water level"][:, station_index].data
            )
            relative_water_depth = np.outer(water_depth, relative_layer_height)
            cumulative_water_depth = np.cumsum(relative_water_depth, axis=1)

            for ti in range(len(times)):
                layer_boundaries.append(
                    np.interp(interpolation_depth, cumulative_water_depth[ti], np.arange(0, number_of_layers, 1))
                )

            layer_boundary = np.floor(layer_boundaries)
            relative_boundary_layer_thickness = layer_boundaries - layer_boundary

            for ti in range(len(times)):
                if int(layer_boundary[ti]) + 2 < len(relative_layer_height):
                    rel_layer_heights = relative_layer_height[0 : int(layer_boundary[ti]) + 2].copy()
                    rel_layer_heights[-1] = rel_layer_heights[-1] * relative_boundary_layer_thickness[ti]
                    average_current_velocity.append(
                        np.average(current_velocity[ti][0 : int(layer_boundary[ti]) + 2], weights=rel_layer_heights)
                    )
                elif int(layer_boundary[ti]) == 0:
                    average_current_velocity = current_velocity[ti]
                else:
                    average_current_velocity.append(np.average(current_velocity[ti], weights=relative_layer_height))

            return average_current_velocity

        if "LAYER" in list(hydromanager.hydrodynamic_data["Current velocity"].dimensions):
            if vessel._T <= 5:
                current_velocity = depth_averaged_current_velocity(5, times, relative_layer_height, current_velocity, station_index)
            elif vessel._T <= 15:
                current_velocity = depth_averaged_current_velocity(
                    vessel._T, times, relative_layer_height, current_velocity, station_index
                )
            else:
                current_velocity = [np.average(current_velocity[t], weights=relative_layer_height) for t in range(len(times))]

        if len(current_velocity) > 2:
            current_governing_current_velocity = current_velocity[2]
        else:
            current_governing_current_velocity = current_velocity[-1]

        return current_velocity, current_governing_current_velocity

    # Functions used to calculate the sail-in-times for a specific vessel

    def provide_tidal_window_restriction(self, vessel, route, node, delay, restriction_type='Vertical'):
        """Function: determines which tidal window restriction applies to the vessel at the specific node

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
            - route: a list of strings of node names that resemble the route that the vessel is planning
            to sail (can be different than vessel.route)
            - specifications: the specific data regarding the properties for which the restriction holds
            - node: a string that defines the node of the tidal window restriction

        """

        # Predefined bool
        restriction = []
        boolean = True
        no_tidal_window = True
        node_index = route.index(node)

        # Vessel parameters
        previous_terminal_of_call = np.nan
        terminal_of_call = np.nan
        if len(dict.get(vessel.output, 'visited_terminals')):
            previous_terminal_of_call = dict.get(vessel.output, 'visited_terminals')[-1]
        if len(dict.get(vessel.metadata, 'terminal_of_call')):
            terminal_of_call = dict.get(vessel.metadata, 'terminal_of_call')[0]
        route_before_node = route[:node_index]
        route_after_node = route[node_index:]
        ukc = 0.
        if restriction_type == 'Horizontal':
            restriction_condition_df = self.horizontal_tidal_restrictions_condition_df
            ukc, _, _, _, _, _ = self.provide_ukc_clearance(vessel, node, delay)
        elif restriction_type == 'Vertical':
            restriction_condition_df = self.vertical_tidal_restrictions_condition_df

        node_mask = restriction_condition_df.Node == node
        length_mask = ((restriction_condition_df.min_ge_Length <= vessel.L)&
                       (restriction_condition_df.min_gt_Length < vessel.L)&
                       (restriction_condition_df.max_lt_Length > vessel.L) &
                       (restriction_condition_df.max_le_Length >= vessel.L))
        draught_mask = ((restriction_condition_df.min_ge_Draught <= vessel.T)&
                        (restriction_condition_df.min_gt_Draught < vessel.T)&
                        (restriction_condition_df.max_lt_Draught > vessel.T)&
                        (restriction_condition_df.max_le_Draught >= vessel.T))
        ukc_mask = ((restriction_condition_df.min_ge_UKC <= ukc)&
                    (restriction_condition_df.min_gt_UKC < ukc)&
                    (restriction_condition_df.max_lt_UKC > ukc)&
                    (restriction_condition_df.max_le_UKC >= ukc))
        from_node_mask = (restriction_condition_df.bound_from.isin(route_before_node)|
                          (restriction_condition_df.bound_from.isna()))
        to_node_mask = (restriction_condition_df.bound_to.isin(route_before_node)|
                        (restriction_condition_df.bound_to.isna()))
        terminal_mask = ((restriction_condition_df.terminal == terminal_of_call)|
                         (restriction_condition_df.terminal.isna()))
        previous_terminal_mask = ((restriction_condition_df.visited_terminal == previous_terminal_of_call)|
                                  (restriction_condition_df.visited_terminal.isna()))
        restriction_mask = node_mask&from_node_mask&to_node_mask&length_mask&draught_mask&terminal_mask&ukc_mask&terminal_mask&previous_terminal_mask
        restriction_condition_df = restriction_condition_df[restriction_mask]
        if restriction_condition_df.empty:
            return restriction, no_tidal_window

        no_tidal_window = False
        # conditions_df = restriction_condition_df[restriction_mask]
        restriction = restriction_condition_df.iloc[0]

        return restriction, no_tidal_window

    def provide_water_depth(self,vessel,node,delay=0):
        hydromanager = HydrodynamicDataManager()
        node_index = list(hydromanager.hydrodynamic_data["STATION"][:]).index(node)
        time_index = np.absolute(
            hydromanager.hydrodynamic_times - pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now + delay)).to_datetime64()
        ).argmin()
        water_level = hydromanager.hydrodynamic_data["Water level"][time_index, node_index].data
        MBL = hydromanager.hydrodynamic_data["MBL"][time_index, node_index].data
        available_water_depth = water_level - MBL
        return MBL,water_level,available_water_depth

    def provide_ukc_clearance(self,vessel,node,delay=0):
        """Function: calculates the sail-in-times for a specific vssel with certain properties and a pre-determined route and provides this information to the vessel

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
            - node:
            - components_calc:

        """
        MBL,water_level,available_water_depth = self.provide_water_depth(vessel,node,delay)
        ukc_s, ukc_p, ukc_r, fwa = np.zeros(4)
        ship_related_factors = {'ukc_s': ukc_s, 'ukc_p': ukc_p, 'ukc_r': ukc_r, 'fwa': fwa,'extra_ukc': vessel.metadata['ukc']}
        if 'Vertical tidal restriction' in vessel.multidigraph.nodes[node].keys():
            restrictions, _ = self.provide_tidal_window_restriction(vessel, [node], node, delay)
            if restrictions.empty:
                return [], [], available_water_depth, 0., ship_related_factors, MBL
            specifications = self.graph.nodes['A']['Vertical tidal restriction']['Specifications']
            specifications = self.graph.nodes["A"]["Vertical tidal restriction"]["Specifications"]
            ukcs_s = []
            ukcs_p = []
            ukcs_r = []
            fwas = []
            for _,specs in specifications.iterrows():
                ukcs_s.append(specs.Restriction.ukc_s)
                ukcs_p.append(specs.Restriction.ukc_p)
                ukcs_r.append(specs.Restriction.ukc_r)
                fwas.append(specs.Restriction.fwa)

            # Determine which restriction applies to vessel
            restriction_index = 0

            # Calculate ukc policy based on the applied restriction
            ukc_s = ukcs_s[restriction_index]
            ukc_p = ukcs_p[restriction_index] * vessel.T
            ukc_r = ukcs_r[restriction_index][0] * (vessel.T - ukcs_r[restriction_index][1])
            fwa = fwas[restriction_index] * vessel.T
        ship_related_factors = {"ukc_s": ukc_s, "ukc_p": ukc_p, "ukc_r": ukc_r, "fwa": fwa, "extra_ukc": vessel.metadata["ukc"]}
        required_water_depth = vessel.T + sum(ship_related_factors.values())
        net_ukc = available_water_depth - required_water_depth
        gross_ukc = available_water_depth - vessel.T
        return net_ukc, gross_ukc, available_water_depth, required_water_depth, ship_related_factors, MBL

    def provide_minimum_available_water_depth_along_route(self,vessel, route, time_start, time_end, delay=0):
        """Function: calculates the minimum available water depth (predicted/modelled/measured water level minus the local maintained bed level) along the route over time,
                      subtracted with the difference between the gross ukc and net ukc (hence: subtracted with additional safety margins consisting of vessel-related factors
                      and water level factors). The bottom-related factors are already accounted for in the use of the MBL instead of the actual depth.

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
            - route: a list of strings of node names that resemble the route that the vessel is planning
            to sail (can be different than vessel.route)
            - delay:
        """
        hydromanager = HydrodynamicDataManager()
        time_start_index = np.max(
            [0, np.absolute(hydromanager.hydrodynamic_times - (time_start + np.timedelta64(int(delay), "s"))).argmin() - 2]
        )
        time_end_index = np.absolute(hydromanager.hydrodynamic_times - (time_end + np.timedelta64(int(delay), "s"))).argmin()
        net_ukc = pd.DataFrame()
        times = hydromanager.hydrodynamic_times[time_start_index:time_end_index]
        start_time = times[0]
        end_time = times[-1]
        t_step = times[1] - times[0]
        time_range = np.arange(start_time, end_time + t_step, t_step)
        nodes_of_interest = route.copy()
        for node in route:
            if self.graph.nodes[node]['LAT']-self.graph.nodes[node]['MBL'] >= vessel.T+np.max([1.0,vessel.T*1.125]):
                nodes_of_interest.remove(node)

        if not nodes_of_interest:
            net_ukc['min_net_ukc'] = 0.0
            net_ukc['station'] = ''
            net_ukc.loc[time_start,:] = [1.0,route[0]]
            net_ukc.loc[time_end,:] = [1.0, route[-1]]
            return net_ukc

        # Start of calculation by looping over the nodes of the route
        for route_index, node_name in enumerate(nodes_of_interest):
            station_index = list(hydromanager.hydrodynamic_data["STATION"][:]).index(node_name)
            sailing_time_to_next_node = vessel.env.vessel_traffic_service.provide_sailing_time(vessel,route[:(route_index+1)])
            time_correction_index = int(np.round(sailing_time_to_next_node['Time'].sum() / (t_step/np.timedelta64(1, 's'))))
            water_level = hydromanager.hydrodynamic_data["Water level"][time_start_index:time_end_index, station_index].data
            MBL = hydromanager.hydrodynamic_data["MBL"][time_start_index:time_end_index, station_index].data
            _, _, _, required_water_depth, _, _ = self.provide_ukc_clearance(vessel,node_name,delay)
            water_depth = water_level - MBL
            net_ukc = pd.concat(
                [
                    net_ukc,
                    pd.DataFrame(
                        [available_water_depth - required_water_depth for available_water_depth in water_depth],
                        index=[t - time_correction_index * t_step for t in times],
                        columns=[node_name],
                    ),
                ],
                axis=1,
            )

        # Pick the minimum of the water depths for each time and each node
        net_ukc = net_ukc.dropna(axis=0)
        net_ukc["min_net_ukc"] = net_ukc.min(axis=1)
        return net_ukc

    def provide_vertical_tidal_windows(self, vessel, route, time_start, time_end, delay=0, plot=False):
        """Function: calculates the windows available to sail-in and -out of the port given the
          vertical tidal restrictions according to the tidal window policy.

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
            - route: a list of strings of node names that resemble the route that the vessel is planning
              to sail (can be different than vessel.route)
            - sailing_time_correction: a bool that indicates whether the calculation should correct for
              sailing_speed (dynamic calculation) or not (static calculation)

        """
        hydromanager = HydrodynamicDataManager()
        vertical_tidal_accessibility = pd.DataFrame(columns=['Tidal period', 'Period number', 'Limit', 'Accessibility'])
        time_start_index = np.max(
            [0, np.absolute(hydromanager.hydrodynamic_times - (time_start + np.timedelta64(int(delay), "s"))).argmin() - 2]
        )
        time_end_index = np.absolute(hydromanager.hydrodynamic_times - (time_end + np.timedelta64(int(delay), "s"))).argmin()

        net_ukcs = self.provide_minimum_available_water_depth_along_route(vessel, route, time_start, time_end, delay)
        new_net_ukcs = pd.DataFrame()
        for station in list(dict.fromkeys(net_ukcs['station'])):
            station_index = list(hydromanager.hydrodynamic_data["STATION"][:]).index(station)
            tidal_periods = self.read_tidal_periods(hydromanager.hydrodynamic_data, "Vertical tidal periods", station_index)
            if tidal_periods == None:
                tidal_periods = [
                    [hydromanager.hydrodynamic_times[0], "Flood start"],
                    [hydromanager.hydrodynamic_times[-1], "Flood stop"],
                ]
            tidal_periods = pd.DataFrame(tidal_periods, columns=['Period start', 'Tidal period'])
            tidal_periods = tidal_periods.reset_index(names='Period number')
            tidal_periods = tidal_periods.set_index('Period start')
            net_ukc = net_ukcs[net_ukcs.station == station]
            data = pd.concat([net_ukc, tidal_periods])
            data = data.sort_index()
            data[['Period number','Tidal period']] = data[['Period number','Tidal period']].ffill()
            data[['Period number','Tidal period']] = data[['Period number','Tidal period']].bfill()
            data = data.dropna()
            new_net_ukcs = pd.concat([new_net_ukcs,data])

        net_ukcs = new_net_ukcs.sort_index()
        net_ukcs['Period number'] = net_ukcs['Period number'].astype(int)
        net_ukcs['Tidal period'] = [tidal_period.split(' ')[0] for tidal_period in net_ukcs['Tidal period']]
        net_ukcs = net_ukcs[['Tidal period', 'Period number', 'min_net_ukc']]

        # Determine zero crossings
        zero_crossings = np.where(np.diff(np.sign(net_ukcs['min_net_ukc'])))[0]

        for iloc in zero_crossings:
            net_ukc = net_ukcs.iloc[iloc]
            if net_ukc['min_net_ukc'] > 0:
                vertical_tidal_accessibility.loc[net_ukc.name] = [net_ukc['Tidal period'], net_ukc['Period number'], 0, 'Inaccessible']
            else:
                vertical_tidal_accessibility.loc[net_ukc.name] = [net_ukc['Tidal period'], net_ukc['Period number'], 0, 'Accessible']

        # Default values
        if net_ukcs.iloc[0]['min_net_ukc'] < 0:
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None))] = [np.nan, np.nan, 0,'Inaccessible']
        else:
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None))] = [np.nan, np.nan, 0, 'Accessible']

        if vertical_tidal_accessibility.iloc[-1,-1] == 'Accessible':
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None))] = [np.nan, np.nan, 0,'Inaccessible']
        else:
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None))] = [np.nan, np.nan, 0,'Accessible']

        vertical_tidal_accessibility = vertical_tidal_accessibility.sort_index()
        vertical_tidal_accessibility[['Period number', 'Tidal period']] = vertical_tidal_accessibility[['Period number', 'Tidal period']].ffill()
        vertical_tidal_accessibility[['Period number', 'Tidal period']] = vertical_tidal_accessibility[['Period number', 'Tidal period']].bfill()

        # Return the sail-in or -out-times given the vertical tidal restrictions over the route of the vessel
        vertical_tidal_accessibility['Condition'] = 'Water level'
        vertical_tidal_accessibility = vertical_tidal_accessibility[~(vertical_tidal_accessibility['Accessibility'] == vertical_tidal_accessibility['Accessibility'].shift(1))]
        vertical_tidal_windows = [
            [window_start[0], window_end[0]]
            for window_start, window_end in zip(
                vertical_tidal_accessibility.iloc[:-1].iterrows(), vertical_tidal_accessibility.iloc[1:].iterrows()
            )
            if window_start[1].Accessibility == "Accessible"
        ]

        if plot:
            # Create figure
            fig, ax = plt.subplots(figsize=[16 * 2 / 3, 6])

            # Plot vertical tidal windows
            for window in vertical_tidal_windows:
                (vertical_tidal_window,) = ax.fill(
                    [window[0], window[0], window[1], window[1]],
                    [-1.5, 1.5, 1.5, -1.5],
                    facecolor="C0",
                    alpha=0.25,
                    edgecolor="none",
                )

            # Plot net UKC
            # net_ukc = self.provide_minimum_available_water_depth_along_route(vessel,route,time_start, time_end, delay)
            (net_UKC,) = ax.plot(net_ukcs["min_net_ukc"], color="C0", linewidth=2)
            ax.axhline(0, color="k", linewidth=0.5)

            # Figure bounds
            ax.set_xlim(hydromanager.hydrodynamic_times[time_start_index], hydromanager.hydrodynamic_times[time_end_index - 36])
            ax.set_ylim(-1.5, 1.5)

            # Figure ticks
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d %H:%M"))

            # Figure axes
            ax.set_xlabel("Date")
            ax.set_ylabel("Net UKC [m]")

            # Legend
            ax.legend(
                [net_UKC, vertical_tidal_window],
                ["Net UKC", "Vertical tidal windows"],
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.0, 1.0),
            )
            fig.tight_layout()

            plt.show()

        return vertical_tidal_accessibility, vertical_tidal_windows, net_ukcs

    def provide_horizontal_tidal_windows(self, vessel, route, time_start, time_end, delay=0, plot=False):
        hydromanager = HydrodynamicDataManager()
        def calculate_horizontal_tidal_window(vessel, time_start_index, time_end_index, current_velocity_station, critical_limits, restriction_type, flood=True, ebb=True,decreasing=False):
            station_index = list(hydromanager.hydrodynamic_data["STATION"][:]).index(current_velocity_station)
            time_step = hydromanager.hydrodynamic_times[1] - hydromanager.hydrodynamic_times[0]
            time_start_index = np.max([0,time_start_index-int(np.timedelta64(12,'h')/time_step)])
            currents_time = hydromanager.hydrodynamic_times[time_start_index:time_end_index]
            currents_data,_ = self.provide_governing_current_velocity(vessel,current_velocity_station,time_start_index,time_end_index)
            tidal_periods = self.read_tidal_periods(hydromanager.hydrodynamic_data, "Horizontal tidal periods", station_index)
            tidal_periods = [condition for condition in tidal_periods if condition[0] <= currents_time[-1] and condition[0] >= currents_time[0]]
            currents = pd.DataFrame({'Current velocity': currents_data}, index=currents_time)
            currents = abs(currents)
            tidal_periods = pd.DataFrame(tidal_periods,columns=['Period start','Tidal period'])
            tidal_periods = tidal_periods.reset_index(names='Period number')
            tidal_periods = tidal_periods.set_index('Period start')
            currents = pd.concat([currents,tidal_periods])
            currents = currents.sort_index()
            for column_name in ['Tidal period','Period number']:
                currents[column_name] = currents[column_name].fillna(method='ffill')
                currents[column_name] = currents[column_name].fillna(method='bfill')
            currents['Tidal period'] = [tidal_period.split(' ')[0] for tidal_period in currents['Tidal period']]
            currents['Current velocity'] = currents['Current velocity'].interpolate()
            currents["Period number"] = currents["Period number"].astype(int)

            roots_cv = [
                root for root in roots if root >= currents_time[0].astype(float) and root <= currents_time[-1].astype(float)
            ]
            times_horizontal_tidal_period = []
            for root in roots_cv:
                root = pd.Timestamp(root).to_datetime64()
                index_current_root = bisect.bisect_right(currents_time, root) - 2
                if index_current_root == -1:
                    index_current_root = index_current_root + 1
                if len(currents_data[index_prev_root:index_current_root]) == 0:
                    continue
                cvel_diff_cross = currents_data[index_current_root + 1] - currents_data[index_current_root - 1]
                if cvel_diff_cross < 0:
                    times_horizontal_tidal_period.append([root, "Ebb Start"])
                    index_prev_root = index_current_root
                elif cvel_diff_cross > 0:
                    times_horizontal_tidal_period.append([root, "Flood Start"])
                    index_prev_root = index_current_root

            tidal_periods = [condition for condition in times_horizontal_tidal_period if condition[0] <= currents_time[-1]]
            currents_time = np.append(
                currents_time, np.array([tide[0] for tide in tidal_periods if tide[0] not in currents_time], dtype="datetime64[ns]")
            )
            currents_data = [abs(value) for value in currents_data]
            currents_data = np.append(currents_data, -999 * np.ones(len(tidal_periods)))
            currents_time, currents_data = [np.array(data) for data in zip(*sorted(zip(currents_time, currents_data)))]
            # Find the intersection points with critical current velocity
            current_intersections = []
            if isinstance(cross_current_limit_dataframe, pd.DataFrame) and not cross_current_limit_dataframe.empty:
                critical_limit = np.interp(
                    currents_time.astype("float"),
                    cross_current_limit_dataframe.index.to_numpy().astype("float"),
                    cross_current_limit_dataframe.Limit.to_numpy(),
                )
                idx = np.argwhere(np.diff(np.sign(critical_limit - currents_data))).flatten()
                roots = currents_time[idx]
                current_intersections.extend(roots.astype(dtype="datetime64[ns]"))
                critical_current_velocity = np.interp(
                    np.array(current_intersections).astype("float"), currents_time.astype("float"), critical_limit
                )
                horizontal_tidal_accessibility = pd.DataFrame(
                    data=critical_current_velocity, columns=["Limit"], index=current_intersections
                )
            elif isinstance(critical_limits, list):
                critical_current_velocity = []
                for critical_limit in critical_limits:
                    idx = np.argwhere(
                        np.diff(np.sign([current_velocity - critical_limit for current_velocity in currents_data]))
                    ).flatten()
                    roots = currents_time[idx]
                    current_intersections.extend(roots.astype(dtype="datetime64[ns]"))
                    critical_current_velocity.extend(np.ones(len(roots)) * critical_limit)
                horizontal_tidal_accessibility = pd.DataFrame(
                    data=critical_current_velocity, columns=["Limit"], index=current_intersections
                )
            horizontal_tidal_accessibility = horizontal_tidal_accessibility.sort_index()

            # Determine the tidal period of the found interpolation points
            horizontal_tidal_accessibility["Period"] = ""
            horizontal_tidal_accessibility["Period_nr"] = -999
            for period_nr, (tidal_period_start, tidal_period_end) in enumerate(zip(tidal_periods[:-1], tidal_periods[1:])):
                tidal_period = tidal_period_start[1].split(" ")[0]
                if tidal_period == "Rising":
                    tidal_period = "Flood"
                if tidal_period == "Falling":
                    tidal_period = "Ebb"
                tidal_period_start = tidal_period_start[0]
                tidal_period_end = tidal_period_end[0]
                horizontal_tidal_accessibility.loc[
                    horizontal_tidal_accessibility[
                        (horizontal_tidal_accessibility.index >= tidal_period_start)
                        & (horizontal_tidal_accessibility.index <= tidal_period_end)
                    ].index,
                    "Period",
                ] = tidal_period
                horizontal_tidal_accessibility.loc[
                    horizontal_tidal_accessibility[
                        (horizontal_tidal_accessibility.index >= tidal_period_start)
                        & (horizontal_tidal_accessibility.index <= tidal_period_end)
                    ].index,
                    "Period_nr",
                ] = period_nr
            horizontal_tidal_accessibility["Period_nr"] = horizontal_tidal_accessibility["Period_nr"].astype(int)

            # Filter the found interpolation points: remove from errors
            # (multiple interpolated numbers and flood/ebb values if not required)
            if decreasing:
                selected_horizontal_tidal_accessibility = pd.DataFrame(columns=horizontal_tidal_accessibility.columns)
                if flood:
                    selected_horizontal_tidal_accessibility = pd.concat(
                        [
                            selected_horizontal_tidal_accessibility,
                            horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period == "Flood"],
                        ]
                    )
                if ebb:
                    selected_horizontal_tidal_accessibility = pd.concat(
                        [
                            selected_horizontal_tidal_accessibility,
                            horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period == "Ebb"],
                        ]
                    )
                horizontal_tidal_accessibility = selected_horizontal_tidal_accessibility.sort_index()
                horizontal_tidal_accessibility = horizontal_tidal_accessibility.loc[
                    horizontal_tidal_accessibility.index.drop_duplicates(keep=False)
                ]

            # Correct the found interpolation points
            if decreasing:
                tide_number = horizontal_tidal_accessibility.iloc[0]["Period_nr"]
                number_of_tidal_periods = horizontal_tidal_accessibility.iloc[-1]["Period_nr"]
                end_time_windows = []
                for period_nr in [
                    idx for idx, count in horizontal_tidal_accessibility.value_counts("Period_nr").items() if count == 2
                ]:
                    sub_df = horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period_nr == period_nr]
                    end_time_windows.append(tidal_periods[period_nr + 1][0] - sub_df.iloc[-1].name)
                mean_end_time_window = np.mean(end_time_windows)
                missing_tides = set(list(np.arange(tide_number, number_of_tidal_periods, 2))) - set(
                    list(dict.fromkeys(horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period == "Flood"].Period_nr))
                )
                for tide_index in missing_tides:
                    starting_time = np.datetime64(tidal_periods[tide_index][0])
                    closing_time = np.datetime64(tidal_periods[tide_index + 1][0] - mean_end_time_window)
                    next_index = bisect.bisect_right(currents_time, closing_time)
                    previous_index = next_index - 1
                    current_velocity = np.interp(
                        closing_time,
                        [currents_time[previous_index], currents_time[next_index]],
                        [currents_data[previous_index], currents_data[next_index]],
                    )
                    horizontal_tidal_accessibility.loc[starting_time, :] = [0, "Flood", tide_index]
                    horizontal_tidal_accessibility.loc[closing_time, :] = [current_velocity, "Flood", tide_index]

                for period_nr, count in [
                    (idx, count) for idx, count in horizontal_tidal_accessibility.value_counts("Period_nr").items() if count != 2
                ]:
                    if count == 1:
                        for iloc, (loc, info) in enumerate(
                            horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period_nr == period_nr].iterrows()
                        ):
                            if not iloc % 2 and info.Limit != np.max(critical_limits):
                                starting_time = np.datetime64(tidal_periods[info.Period_nr][0])
                                horizontal_tidal_accessibility.loc[starting_time] = info
                                horizontal_tidal_accessibility.loc[starting_time, "Limit"] = np.max(critical_limits)
                                break
                    else:
                        horizontal_tidal_accessibility = horizontal_tidal_accessibility.drop(
                            horizontal_tidal_accessibility[
                                (horizontal_tidal_accessibility.Period_nr == period_nr)
                                & (horizontal_tidal_accessibility.Limit == np.max(critical_limits))
                            ].index[:-1]
                        )
                        horizontal_tidal_accessibility = horizontal_tidal_accessibility.drop(
                            horizontal_tidal_accessibility[
                                (horizontal_tidal_accessibility.Period_nr == period_nr)
                                & (horizontal_tidal_accessibility.Limit == np.min(critical_limits))
                            ].index[:-1]
                        )
                        if len(horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period_nr == period_nr]) < 2:
                            for iloc, (loc, info) in enumerate(
                                horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period_nr == period_nr].iterrows()
                            ):
                                if not iloc % 2 and info.Limit != np.max(critical_limits):
                                    starting_time = np.datetime64(tidal_periods[info.Period_nr][0])
                                    horizontal_tidal_accessibility.loc[starting_time] = info
                                    horizontal_tidal_accessibility.loc[starting_time, "Limit"] = np.max(critical_limits)
                                    break

            else:
                for period_nr in [
                    idx for idx, count in horizontal_tidal_accessibility.value_counts("Period_nr").items() if count % 2
                ]:
                    for loc, info in horizontal_tidal_accessibility[
                        horizontal_tidal_accessibility.Period_nr == period_nr
                    ].iterrows():
                        if (
                            currents_data[list(currents_time).index(loc) - 1] < info.Limit
                            and currents_data[list(currents_time).index(loc) + 1] < info.Limit
                        ):
                            horizontal_tidal_accessibility[loc + np.timedelta64(1, "ns")] = info
                            break

            # Add fully (in)accessible tides
            for tide in tides:
                critical_limit = critical_limits[tide]
                if critical_limit == tidal_window_constructor.accessibility.accessible.value:
                    for period_time,period_info in tidal_periods.iterrows():
                        if period_info['Tidal period'].find(tide)+1:
                            horizontal_tidal_accessibility.loc[period_time] = [critical_limit, 'Accessible', 'Current velocity', tide, period_info['Period number']]
                elif critical_limit == tidal_window_constructor.accessibility.inaccessible.value:
                    for period_time,period_info in tidal_periods.iterrows():
                        if period_info['Tidal period'].find(tide)+1:
                            horizontal_tidal_accessibility.loc[period_time] = [critical_limit, 'Inaccessible', 'Current velocity', tide, period_info['Period number']]
            last_tidal_period = tidal_periods.iloc[-1]
            horizontal_tidal_accessibility.loc[last_tidal_period.name] = [0.0, 'Inaccessible', 'Current velocity', last_tidal_period['Tidal period'], last_tidal_period['Period number']]
            return horizontal_tidal_accessibility

        # Start calculation
        horizontal_tidal_restriction_nodes = []
        horizontal_tidal_restriction_stations = []
        restrictions = pd.DataFrame()
        horizontal_tidal_accessibility = pd.DataFrame(columns=['Limit', 'Condition', 'Accessibility','Period_nr'])
        horizontal_tidal_window = False
        time_start_index = np.max([0, np.absolute(hydromanager.hydrodynamic_times - (time_start)).argmin() - 2])
        time_end_index = np.absolute(hydromanager.hydrodynamic_times - (time_end)).argmin()
        for route_index, node_name in enumerate(route):
            if 'Horizontal tidal restriction' in vessel.multidigraph.nodes[node_name].keys():
                sailing_time_to_next_node = self.provide_sailing_time(vessel, route[:(route_index + 1)])
                restriction, no_tidal_window = self.provide_tidal_window_restriction(vessel, route, node_name,sailing_time_to_next_node.Time.sum(),restriction_type='Horizontal')
                if no_tidal_window:
                    continue

                restrictions = pd.concat([restrictions,pd.DataFrame([restriction])])
                time_start_index = np.max(
                    [0, np.absolute(hydromanager.hydrodynamic_times - (time_start + np.timedelta64(int(delay), "s"))).argmin() - 2]
                )
                time_end_index = np.absolute(
                    hydromanager.hydrodynamic_times - (time_end + np.timedelta64(int(delay), "s"))
                ).argmin()

                horizontal_tidal_window = True
                current_velocity_station = restriction.Data
                cross_current_limit = {}
                cross_current_limit['Flood'] = restriction.Restriction.current_velocity_values['Flood']
                cross_current_limit['Ebb'] = restriction.Restriction.current_velocity_values['Ebb']
                if restriction.Restriction.window_method == 'Maximum':
                    next_horizontal_tidal_accessibility = calculate_horizontal_tidal_window(vessel,time_start_index,time_end_index,current_velocity_station,cross_current_limit,restriction.Restriction)
                if restriction.Restriction.window_method == 'Point-based':
                    if isinstance(restriction.Restriction.current_velocity_values['Flood'], list) and restriction.Restriction.current_velocity_values['Ebb'] == -999:
                        next_horizontal_tidal_accessibility = calculate_horizontal_tidal_window(vessel,time_start_index,time_end_index,current_velocity_station,cross_current_limit,restriction.Restriction,ebb=False,decreasing=True)
                    elif isinstance(restriction.Restriction.current_velocity_values['Ebb'], list) and restriction.Restriction.current_velocity_values['Flood'] == -999:
                        next_horizontal_tidal_accessibility = calculate_horizontal_tidal_window(vessel,time_start_index,time_end_index,current_velocity_station,cross_current_limit,restriction.Restriction,flood=False,decreasing=True)
                    else:
                        next_horizontal_tidal_accessibility = calculate_horizontal_tidal_window(vessel,time_start_index,time_end_index,current_velocity_station,cross_current_limit,restriction.Restriction,decreasing=True)

                horizontal_tidal_restriction_nodes.append(node_name)
                horizontal_tidal_restriction_stations.append(current_velocity_station)
                next_horizontal_tidal_accessibility_time_correction = np.timedelta64(int(sailing_time_to_next_node['Time'].sum()), 's')
                next_horizontal_tidal_accessibility.index -= next_horizontal_tidal_accessibility_time_correction
                next_horizontal_tidal_accessibility = next_horizontal_tidal_accessibility.sort_index()
                next_horizontal_tidal_accessibility = next_horizontal_tidal_accessibility[~(next_horizontal_tidal_accessibility['Accessibility'] == next_horizontal_tidal_accessibility['Accessibility'].shift(1))]
                if horizontal_tidal_accessibility.empty:
                    horizontal_tidal_accessibility = next_horizontal_tidal_accessibility
                else:
                    horizontal_tidal_accessibility = self.combine_tidal_windows(
                        horizontal_tidal_accessibility, next_horizontal_tidal_accessibility, current_velocity_windows=True
                    )

                horizontal_tidal_restriction_nodes.append(node_name)
                horizontal_tidal_restriction_stations.append(station)
                next_horizontal_tidal_accessibility_time_correction = np.timedelta64(
                    int(sailing_time_to_next_node["Time"].sum()), "s"
                )
                next_horizontal_tidal_accessibility.index -= next_horizontal_tidal_accessibility_time_correction
                if horizontal_tidal_accessibility.empty:
                    horizontal_tidal_accessibility = next_horizontal_tidal_accessibility
                else:
                    horizontal_tidal_accessibility = self.combine_tidal_windows(
                        horizontal_tidal_accessibility, next_horizontal_tidal_accessibility
                    )

        if horizontal_tidal_accessibility.empty or not horizontal_tidal_window:
            horizontal_tidal_accessibility = pd.DataFrame(columns=["Limit", "Condition", "Accessibility"])
            horizontal_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [
                0,
                "Current velocity",
                "Accessible",
            ]
            horizontal_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [
                0,
                "Current velocity",
                "Inaccessible",
            ]
        else:
            if horizontal_tidal_accessibility.iloc[0].Accessibility == "Inaccessible":
                horizontal_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [
                    0,
                    "Current velocity",
                    "Accessible",
                ]
            else:
                horizontal_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [
                    0,
                    "Current velocity",
                    "Inaccessible",
                ]
            if horizontal_tidal_accessibility.iloc[-1].Accessibility == "Inaccessible":
                horizontal_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [
                    0,
                    "Current velocity",
                    "Accessible",
                ]
            else:
                horizontal_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [
                    0,
                    "Current velocity",
                    "Inaccessible",
                ]

        horizontal_tidal_accessibility = horizontal_tidal_accessibility.sort_index()
        horizontal_tidal_windows = [
            [window_start[0], window_end[0]]
            for window_start, window_end in zip(
                horizontal_tidal_accessibility.iloc[:-1].iterrows(), horizontal_tidal_accessibility.iloc[1:].iterrows()
            )
            if window_start[1].Accessibility == "Accessible"
        ]

        if plot:
            # Create figure
            fig, ax = plt.subplots(figsize=[16 * 2 / 3, 6])

            # Plot vertical tidal windows
            for window in horizontal_tidal_windows:
                (horizontal_tidal_window,) = ax.fill(
                    [window[0], window[0], window[1], window[1]],
                    [-1.5, 1.5, 1.5, -1.5],
                    facecolor="firebrick",
                    alpha=0.25,
                    edgecolor="none",
                )

            # Plot governing current velocity
            for node,station in zip(horizontal_tidal_restriction_nodes,horizontal_tidal_restriction_stations):
                governing_current_velocity,_ = self.provide_governing_current_velocity(vessel,station,time_start_index,time_end_index)
                horizontal_tidal_accessibility_time_correction = np.timedelta64(int(self.provide_sailing_time(vessel, route[:(route.index(node) + 1)])['Time'].sum() + delay),'s')
                (current_velocity,) = ax.plot(
                    [
                        time - horizontal_tidal_accessibility_time_correction
                        for time in hydromanager.hydrodynamic_times[time_start_index:time_end_index]
                    ],
                    governing_current_velocity,
                    color="firebrick",
                    linewidth=2,
                )

            ax.axhline(0, color='k', linewidth=1)

            # Figure bounds
            ax.set_xlim(hydromanager.hydrodynamic_times[time_start_index], hydromanager.hydrodynamic_times[time_end_index - 36])
            ax.set_ylim(-1.5, 1.5)

            # Figure ticks
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d %H:%M"))

            # Figure axes
            ax.set_xlabel("Date")
            ax.set_ylabel("Current velocity [m/s]")

            # Legend
            ax.legend(
                [current_velocity, horizontal_tidal_window],
                ["Current velocity", "Horizontal tidal windows"],
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.0, 1.0),
            )
            fig.tight_layout()
            plt.show()

        return horizontal_tidal_accessibility, horizontal_tidal_windows, restrictions

    def combine_tidal_windows(self, tidal_window_1, tidal_window_2):
        accessibility_1 = tidal_window_1.iloc[
            [bisect.bisect_right(tidal_window_1.index, index) - 1 for index in tidal_window_2.index]
        ].Accessibility.to_numpy()
        accessibility_2 = tidal_window_2.iloc[
            [bisect.bisect_right(tidal_window_2.index, index) - 1 for index in tidal_window_1.index]
        ].Accessibility.to_numpy()
        accessibility_1_interpolated = tidal_window_1.copy()
        accessibility_2_interpolated = tidal_window_2.copy()
        for key, value in zip(tidal_window_2.index, accessibility_1):
            accessibility_1_interpolated.loc[key, "Accessibility"] = value
        for key, value in zip(tidal_window_1.index, accessibility_2):
            accessibility_2_interpolated.loc[key, "Accessibility"] = value
        accessibility_1_interpolated = accessibility_1_interpolated.sort_index()
        accessibility_2_interpolated = accessibility_2_interpolated.sort_index()
        tidal_accessibility = pd.concat([accessibility_1_interpolated, accessibility_2_interpolated], axis=1)
        tidal_accessibility = tidal_accessibility.sort_index()
        tidal_accessibility_limit = [
            limit_1 if not math.isnan(limit_1) else limit_2 for limit_1, limit_2 in tidal_accessibility.Limit.to_numpy()
        ]
        tidal_accessibility_condition = [
            condition_1 if isinstance(condition_1, str) else condition_2
            for condition_1, condition_2 in tidal_accessibility.Condition.to_numpy()
        ]
        tidal_accessibility_accessibility = [
            "Accessible" if accessibility_1 == accessibility_2 and accessibility_1 == "Accessible" else "Inaccessible"
            for accessibility_1, accessibility_2 in tidal_accessibility.Accessibility.to_numpy()
        ]
        tidal_accessibility = tidal_accessibility.drop(["Limit", "Condition", "Accessibility"], axis=1)
        tidal_accessibility["Limit"] = tidal_accessibility_limit
        tidal_accessibility["Condition"] = tidal_accessibility_condition
        tidal_accessibility["Accessibility"] = tidal_accessibility_accessibility
        accessible_indexes = [
            idx for idx, accessibility in enumerate((tidal_accessibility.Accessibility == "Accessible").to_numpy()) if accessibility
        ]
        inaccessible_indexes = [
            idx
            for idx, inaccessibility in enumerate((tidal_accessibility.Accessibility == "Inaccessible").to_numpy())
            if inaccessibility
        ]
        accessible_indexes = np.array(
            [indexes[-1] for indexes in np.split(accessible_indexes, np.nonzero(np.diff(accessible_indexes) != 1) + 1)], dtype=int
        )
        inaccessible_indexes = np.array(
            [indexes[0] for indexes in np.split(inaccessible_indexes, np.nonzero(np.diff(inaccessible_indexes) != 1) + 1)],
            dtype=int,
        )
        tidal_window_indexes = np.sort(np.append(accessible_indexes, inaccessible_indexes))
        tidal_accessibility = tidal_accessibility.iloc[tidal_window_indexes]
        return tidal_accessibility

    def provide_tidal_windows(self,vessel,route,time_start,time_end,ax_left=None,ax_right=None,delay=0,plot=False):
        hydromanager = HydrodynamicDataManager()
        time_start_index = np.max(
            [0, np.absolute(hydromanager.hydrodynamic_times - (time_start + np.timedelta64(int(delay), "s"))).argmin() - 2]
        )
        time_end_index = np.absolute(hydromanager.hydrodynamic_times - (time_end + np.timedelta64(int(delay), "s"))).argmin()
        vertical_tidal_accessibility,vertical_tidal_windows,net_ukcs = self.provide_vertical_tidal_windows(vessel, route, time_start, time_end, delay)
        horizontal_tidal_accessibility,horizontal_tidal_windows,horizontal_tidal_restrictions = self.provide_horizontal_tidal_windows(vessel, route, time_start, time_end, delay)
        if not horizontal_tidal_accessibility.empty:
            tidal_accessibility = self.combine_tidal_windows(vertical_tidal_accessibility,horizontal_tidal_accessibility)
        else:
            tidal_accessibility = vertical_tidal_accessibility

        tidal_windows = [[window_start[0], window_end[0]] for window_start, window_end in zip(tidal_accessibility.iloc[:-1].iterrows(), tidal_accessibility.iloc[1:].iterrows()) if window_start[1].Accessibility == 'Accessible']

        # Plot
        if plot:
            # Create figure
            if not ax_left:
                _, ax_left = plt.subplots(figsize=[16 * 2 / 3, 6])
                ax_right = ax_left.twinx()

            # Plot net UKC
            # net_ukc = self.provide_minimum_available_water_depth_along_route(vessel, route, time_start, time_end, delay)
            (net_UKC,) = ax_left.plot(net_ukcs["min_net_ukc"], color="C0", linewidth=2, zorder=1)
            minimum_required_net_ukc = ax_left.axhline(0, color="C0", linestyle="--", linewidth=2)

            # Plot governing current velocity
            horizontal_restriction_type = None
            for index,restriction_info in horizontal_tidal_restrictions.iterrows():
                horizontal_restriction_type = restriction_info.Restriction.window_method
                governing_current_velocity,_ = self.provide_governing_current_velocity(vessel,restriction_info.Data,time_start_index, time_end_index)
                horizontal_tidal_accessibility_time_correction = np.timedelta64(int(self.provide_sailing_time(vessel, route[:(route.index(restriction_info.Node) + 1)])['Time'].sum()), 's')
                (current_velocity,) = ax_right.plot(
                    [
                        time - horizontal_tidal_accessibility_time_correction
                        for time in hydromanager.hydrodynamic_times[time_start_index:time_end_index]
                    ],
                    governing_current_velocity,
                    color="firebrick",
                    linewidth=2,
                    zorder=1,
                )
                if horizontal_restriction_type == 'Maximum':
                    critical_current_velocity = ax_right.axhline(restriction_info.Restriction.current_velocity_values['Flood'], color='firebrick',linestyle='--', linewidth=2)
                    ax_right.axhline(-1 * restriction_info.Restriction.current_velocity_values['Ebb'], color='firebrick',linestyle='--', linewidth=2)
                ax_right.set_ylim(np.floor(np.min(governing_current_velocity)),np.ceil(np.max(governing_current_velocity)))

            # Figure bounds
            ax_left.set_xlim(
                hydromanager.hydrodynamic_times[time_start_index], hydromanager.hydrodynamic_times[time_end_index - 36]
            )
            ax_left.set_ylim(
                np.min([np.floor(np.min(net_ukcs["min_net_ukc"].to_numpy())), -1.0]),
                np.max([np.ceil(np.max(net_ukcs["min_net_ukc"])), 1.0]),
            )

            # Calculate vertical and horizontal tidal windows
            vertical_tidal_window_polygons = []
            for window in vertical_tidal_windows:
                vertical_tidal_window_polygons.append(
                    Polygon(
                        [
                            Point((window[0] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[0]),
                            Point((window[0] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[1]),
                            Point((window[1] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[1]),
                            Point((window[1] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[0]),
                        ]
                    )
                )
            horizontal_tidal_window_polygons = []
            for window in horizontal_tidal_windows:
                horizontal_tidal_window_polygons.append(
                    Polygon(
                        [
                            Point((window[0] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[0]),
                            Point((window[0] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[1]),
                            Point((window[1] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[1]),
                            Point((window[1] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[0]),
                        ]
                    )
                )
            tidal_window_polygons = []
            for window in tidal_windows:
                tidal_window_polygons.append(
                    Polygon(
                        [
                            Point((window[0] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[0]),
                            Point((window[0] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[1]),
                            Point((window[1] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[1]),
                            Point((window[1] - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"), ax_left.get_ylim()[0]),
                        ]
                    )
                )

            if not isinstance(horizontal_tidal_window_polygons, Polygon):
                horizontal_tidal_window_polygons = MultiPolygon(horizontal_tidal_window_polygons)
            vertical_tidal_window_polygons = MultiPolygon(vertical_tidal_window_polygons).difference(
                horizontal_tidal_window_polygons
            )
            vertical_tidal_window_polygons = vertical_tidal_window_polygons.difference(MultiPolygon(tidal_window_polygons))
            if not isinstance(vertical_tidal_window_polygons, Polygon):
                vertical_tidal_window_polygons = MultiPolygon(vertical_tidal_window_polygons)
            horizontal_tidal_window_polygons = MultiPolygon(horizontal_tidal_window_polygons).difference(
                vertical_tidal_window_polygons
            )
            horizontal_tidal_window_polygons = horizontal_tidal_window_polygons.difference(MultiPolygon(tidal_window_polygons))

            # Plot vertical tidal windows
            if not isinstance(vertical_tidal_window_polygons, Polygon):
                for polygon in vertical_tidal_window_polygons.geoms:
                    polygon_x = []
                    for timestamp in polygon.exterior.xy[0]:
                        polygon_x.append(datetime.datetime.fromtimestamp(timestamp,tz=pytz.utc))
                    polygon_y = list(polygon.exterior.xy[1])

                    (vertical_tidal_window,) = ax_left.fill(
                        polygon_x, polygon_y, facecolor="C0", alpha=0.25, edgecolor="none", zorder=0
                    )
            elif isinstance(vertical_tidal_window_polygons, Polygon):
                polygon = vertical_tidal_window_polygons
                polygon_x = []
                for timestamp in polygon.exterior.xy[0]:
                    polygon_x.append(datetime.datetime.fromtimestamp(timestamp,tz=pytz.utc))
                polygon_y = list(polygon.exterior.xy[1])
                (vertical_tidal_window,) = ax_left.fill(
                    polygon_x, polygon_y, facecolor="C0", alpha=0.25, edgecolor="none", zorder=0
                )

            # Plot horizontal tidal windows
            if not isinstance(horizontal_tidal_window_polygons, Polygon):
                for polygon in horizontal_tidal_window_polygons.geoms:
                    polygon_x = []
                    for timestamp in polygon.exterior.xy[0]:
                        polygon_x.append(datetime.datetime.fromtimestamp(timestamp,tz=pytz.utc))
                    polygon_y = list(polygon.exterior.xy[1])
                    (horizontal_tidal_window,) = ax_left.fill(
                        polygon_x, polygon_y, facecolor="firebrick", alpha=0.25, edgecolor="none", zorder=0
                    )
            elif isinstance(horizontal_tidal_window_polygons, Polygon):
                polygon = horizontal_tidal_window_polygons
                polygon_x = []
                for timestamp in polygon.exterior.xy[0]:
                    polygon_x.append(datetime.datetime.fromtimestamp(timestamp,tz=pytz.utc))
                polygon_y = list(polygon.exterior.xy[1])
                (horizontal_tidal_window,) = ax_left.fill(
                    polygon_x, polygon_y, facecolor="firebrick", alpha=0.25, edgecolor="none", zorder=0
                )

            # Plot tidal windows
            for window in tidal_windows:
                (tidal_window,) = ax_left.fill(
                    [window[0], window[0], window[1], window[1]],
                    [ax_left.get_ylim()[0], ax_left.get_ylim()[1], ax_left.get_ylim()[1], ax_left.get_ylim()[0]],
                    facecolor="limegreen",
                    alpha=0.25,
                    edgecolor="none",
                    zorder=0,
                )
            if not tidal_windows:
                tidal_window = ax_left.fill(
                    [0, 0, 0, 0],
                    [ax_left.get_ylim()[0], ax_left.get_ylim()[1], ax_left.get_ylim()[1], ax_left.get_ylim()[0]],
                    facecolor="limegreen",
                    alpha=0.25,
                    edgecolor="none",
                    zorder=0,
                )

            # Figure ticks
            ax_left.set_xticks(ax_left.get_xticks())
            ax_left.set_xticklabels(ax_left.get_xticklabels(), rotation=45, ha="right")
            ax_left.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d %H:%M"))

            # Figure axes
            ax_left.set_xlabel("Date")
            ax_left.set_ylabel("Net UKC [m]")
            ax_right.set_ylabel("Current velocity [m/s]")
            # Legend and title
            if horizontal_restriction_type:
                if horizontal_restriction_type == "Maximum":
                    ax_left.legend(
                        [
                            net_UKC,
                            minimum_required_net_ukc,
                            current_velocity,
                            critical_current_velocity,
                            vertical_tidal_window,
                            horizontal_tidal_window,
                            tidal_window,
                        ],
                        [
                            "Net UKC",
                            "Required net UKC",
                            "Current velocity",
                            "Vertical tidal windows",
                            "Horizontal tidal windows",
                            "Resulting tidal windows",
                        ],
                        frameon=False,
                        loc="upper left",
                        bbox_to_anchor=(1.1, 1.0),
                    )
                else:
                    ax_left.legend(
                        [
                            net_UKC,
                            minimum_required_net_ukc,
                            current_velocity,
                            vertical_tidal_window,
                            horizontal_tidal_window,
                            tidal_window,
                        ],
                        [
                            "Net UKC",
                            "Required net UKC",
                            "Current velocity",
                            "Vertical tidal windows",
                            "Horizontal tidal windows",
                            "Resulting tidal windows",
                        ],
                        frameon=False,
                        loc="upper left",
                        bbox_to_anchor=(1.1, 1.0),
                    )
            else:
                ax_left.legend(
                    [net_UKC, minimum_required_net_ukc, vertical_tidal_window, tidal_window],
                    ["Net UKC", "Required net UKC", "Vertical tidal windows", "Resulting tidal windows"],
                    frameon=False,
                    loc="upper left",
                    bbox_to_anchor=(1.05, 1.0),
                )
            ax_left.set_title(
                f"Accessibility of {vessel.type}-class vessel with name {vessel.name} "
                f"and {np.round(vessel.T, 2)}m draught and a length of {np.round(vessel.L)}m, sailing {vessel.bound}"
            )
            plt.show()
            return tidal_accessibility, tidal_windows, ax_left, ax_right

        return tidal_accessibility, tidal_windows
