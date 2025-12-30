# package(s) related to time, space and id
import bisect
import datetime
import pytz

# mathematical packages
import bisect
import math
import scipy as sc

# packages for data handling
import numpy as np
import pandas as pd
import xarray as xr

# spatial libraries
import networkx as nx
import shapely
import shapely.ops
from shapely.geometry import MultiPolygon, Point, Polygon
from IPython.display import display

class VesselTrafficService:
    """Class: a collection of functions that processes requests of vessels regarding
    the nautical processes on ow to enter the port safely"""

    def __init__(self, hydrodynamic_data=None, vessel_speed_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(hydrodynamic_data, xr.Dataset):
            self.hydrodynamic_information = hydrodynamic_data
        if isinstance(vessel_speed_data, xr.Dataset):
            self.vessel_speeds = vessel_speed_data

    def provide_speed(self, vessel, edge):
        if "vessel_speeds" in dir(self) and edge in list(self.vessel_speeds.edge.values):
            vessel_speed = float(self.vessel_speeds.sel({"edge": edge}).average_speed.values)
        else:
            vessel_speed = vessel.v
        return vessel_speed

    def provide_speed_over_route(self, vessel, route):
        vessel_speed_over_route = pd.DataFrame(columns=["edge", "Speed"])
        for idx, (u, v) in enumerate(zip(route[:-1], route[1:])):
            vessel_speed_over_route.loc[idx, :] = [(u, v), self.provide_speed(vessel, (u, v))]
        vessel_speed_over_route = vessel_speed_over_route.set_index("edge")
        return vessel_speed_over_route

    def provide_heading(self, vessel, edge):
        def reverse_geometry(x, y):
            return x[::-1], y[::-1]

        distance = []
        origin_location = vessel.multidigraph.nodes[edge[0]]["geometry"]
        k = sorted(
            vessel.multidigraph[edge[0]][edge[1]], key=lambda x: vessel.multidigraph[edge[0]][edge[1]][x]["geometry"].length
        )[0]
        edge_geometry = vessel.multidigraph.edges[edge[0], edge[1], k]["geometry"]
        for coord in edge_geometry.coords:
            distance.append(origin_location.distance(Point(coord)))
        if np.argmin(distance):
            edge_geometry = shapely.ops.transform(reverse_geometry, edge_geometry)
        heading = np.degrees(
            math.atan2(
                edge_geometry.coords[0][0] - edge_geometry.coords[-1][0], edge_geometry.coords[0][1] - edge_geometry.coords[-1][1]
            )
        )
        return heading

    def provide_distance_to_node(self, vessel, node_1, node_2, location):
        geometry = self.provide_trajectory(vessel.multidigraph, node_1, node_2)
        geometries = shapely.ops.split(shapely.ops.snap(geometry, location, tolerance=100), location).geoms
        distance = np.round(geometries[-1].length, 1)
        return distance

    def provide_location_over_edges(self, vessel, node_1, node_2, distance):
        location = self.provide_trajectory(vessel.multidigraph, node_1, node_2).interpolate(distance)
        return location

    def provide_sailing_distance(self, vessel, edge):
        k = sorted(
            vessel.multidigraph[edge[0]][edge[1]], key=lambda x: vessel.multidigraph[edge[0]][edge[1]][x]["geometry"].length
        )[0]
        sailing_distance = [edge, vessel.multidigraph.edges[edge[0], edge[1], k]["length_m"]]
        return sailing_distance

    def provide_sailing_distance_over_route(self, vessel, route):
        sailing_distance_over_route = pd.DataFrame(columns=["edge", "Distance"])
        for idx, (u, v) in enumerate(zip(route[:-1], route[1:])):
            sailing_distance_over_route.loc[idx, :] = self.provide_sailing_distance(vessel, (u, v))
        sailing_distance_over_route = sailing_distance_over_route.set_index("edge")
        return sailing_distance_over_route

    def provide_sailing_time(self, vessel, route, distance=None):
        if distance is None:
            sailing_distance_over_route = self.provide_sailing_distance_over_route(vessel, route)
        else:
            sailing_distance_over_route = pd.DataFrame(columns=["edge", "distance"])
            sailing_distance_over_route.loc[0, "edge"] = (route[0], route[1])
            sailing_distance_over_route.loc[0, "Distance"] = distance
            sailing_distance_over_route = sailing_distance_over_route.set_index("edge")
        vessel_speed_over_route = self.provide_speed_over_route(vessel, route)
        sailing_time_over_route = pd.concat([sailing_distance_over_route, vessel_speed_over_route], axis=1)
        sailing_time_over_route["Time"] = sailing_time_over_route["Distance"] / sailing_time_over_route["Speed"]
        return sailing_time_over_route

    def provide_governing_current_velocity(self, vessel, node, time_start_index, time_end_index):
        station_index = list(self.hydrodynamic_information["STATION"].values).index(node)
        times = self.hydrodynamic_information["TIME"].values[time_start_index:time_end_index]
        relative_layer_height = self.hydrodynamic_information["LAYER"].values
        current_velocity = (
            self.hydrodynamic_information["Primary current velocity"][station_index]
            .transpose()
            .values[time_start_index:time_end_index]
        )

        def depth_averaged_current_velocity(interpolation_depth, times, relative_layer_height, current_velocity, station_index):
            layer_boundaries = []
            average_current_velocity = []
            number_of_layers = len(relative_layer_height)
            water_depth = (
                self.hydrodynamic_information["MBL"][station_index] + self.hydrodynamic_information["Water level"][station_index]
            )
            relative_water_depth = water_depth * self.hydrodynamic_information["LAYER"]
            cumulative_water_depth = relative_water_depth.cumsum("LAYER").values

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

        if "LAYER" in list(self.hydrodynamic_information["Current velocity"].dims):
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

    def provide_water_depth(self, vessel, node, delay=0):
        node_index = list(self.hydrodynamic_information["STATION"].values).index(node)
        time_index = np.absolute(
            self.hydrodynamic_information.TIME.values
            - pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now + delay, tz=pytz.utc)).to_datetime64()
        ).argmin()
        water_level = self.hydrodynamic_information["Water level"][node_index].values[time_index]
        MBL = self.hydrodynamic_information["MBL"][node_index].values[time_index]
        available_water_depth = water_level + MBL
        return MBL, water_level, available_water_depth


