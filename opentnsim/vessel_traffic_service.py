# package(s) related to the simulation
import numpy as np
import bisect
import scipy as sc
from shapely.geometry import Point, Polygon, MultiPolygon
import shapely
import pandas as pd
from matplotlib import pyplot as plt, dates
import math
import networkx as nx
import time as timepy
import datetime
import pytz
import xarray as xr

from shapely.ops import transform

# spatial libraries
import pyproj


class VesselTrafficService:
    """Class: a collection of functions that processes requests of vessels regarding the nautical processes on ow to enter the port safely"""

    def __init__(self, hydrodynamic_data=None, vessel_speed_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(hydrodynamic_data, xr.Dataset):
            self.hydrodynamic_information = hydrodynamic_data
        if isinstance(vessel_speed_data, xr.Dataset):
            self.vessel_speeds = vessel_speed_data

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
            time_start = pd.Timestamp(datetime.datetime.fromtimestamp(vessel.env.now, tz=pytz.utc)).to_datetime64()
        if not time_stop:
            time_stop = pd.Timestamp(
                datetime.datetime.fromtimestamp(vessel.env.now + vessel.metadata["max_waiting_time"], tz=pytz.utc)
            ).to_datetime64()
        _, tidal_windows = self.provide_tidal_windows(vessel, route, time_start, time_stop, delay, plot=plot)
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

    def provide_waiting_time_for_outbound_tidal_window(self, vessel, route, delay=0, plot=False):
        vessel.bound = "outbound"
        vessel._T -= vessel.metadata["(un)loading"][0]
        waiting_time = self.provide_waiting_time_for_inbound_tidal_window(vessel, route=route, delay=delay, plot=plot)
        vessel._T += vessel.metadata["(un)loading"][0]
        vessel.bound = "inbound"
        return waiting_time

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
        edge_geometry = vessel.multidigraph.edges[edge[0], edge[1], k]["Info"]["geometry"]
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

    @staticmethod
    def provide_trajectory(multidigraph, node_1, node_2):
        nodes = nx.dijkstra_path(multidigraph, node_1, node_2)
        for loc, edge in enumerate(zip(nodes[:-1], nodes[1:])):
            k = sorted(multidigraph[edge[0]][edge[1]], key=lambda x: multidigraph[edge[0]][edge[1]][x]["geometry"].length)[0]
            geom = multidigraph.edges[edge[0], edge[1], k]["geometry"]
            if loc:
                multi_line = shapely.geometry.MultiLineString([final_geometry, geom])
                final_geometry = shapely.ops.linemerge(multi_line)
            else:
                final_geometry = geom
        return final_geometry

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
        sailing_distance = [edge, vessel.multidigraph.edges[edge[0], edge[1], k]["Info"]["length"]]
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
                    station_index = list(self.hydrodynamic_information["STATION"]).index(node_on_route)
                    min_water_level = np.min(self.hydrodynamic_information["Water level"][station_index].values)
                    _, _, _, required_water_depth, _, MBL = self.provide_ukc_clearance(vessel, node)
                    if min_water_level + MBL < required_water_depth:
                        anchorage_reachable = False
                        break

                if not anchorage_reachable:
                    continue

                # Extract information over the individual anchorage areas: capacity, users, and the sailing distance to the anchorage area from the designated terminal the vessel is planning to call
                nodes_of_anchorages.append(node_anchorage)
                capacity_of_anchorages.append(vessel.multidigraph.nodes[node_anchorage]["Anchorage"][0].resource.capacity)
                users_of_anchorages.append(len(vessel.multidigraph.nodes[node_anchorage]["Anchorage"][0].resource.users))
                route_from_anchorage = nx.dijkstra_path(vessel.multidigraph, node_anchorage, vessel.route[-1])
                sailing_time_to_anchorage = vessel.env.vessel_traffic_service.provide_sailing_time(vessel, route_from_anchorage)[
                    "Time"
                ].sum()
                sailing_times_to_anchorages.append(sailing_time_to_anchorage)

        # Sort the lists based on the sailing distance to the anchorage area from the designated terminal the vessel is planning to call
        sorted_nodes_anchorages = [nodes for (distances, nodes) in sorted(zip(sailing_times_to_anchorages, nodes_of_anchorages))]
        sorted_users_of_anchorages = [nodes for (distances, nodes) in sorted(zip(sailing_times_to_anchorages, users_of_anchorages))]
        sorted_capacity_of_anchorages = [
            nodes for (distances, nodes) in sorted(zip(sailing_times_to_anchorages, capacity_of_anchorages))
        ]

        # Take the anchorage area that is closest to the designated terminal the vessel is planning to call if there is sufficient capacity:
        node_anchorage = 0
        for anchorage_index, node_anchorage in enumerate(sorted_nodes_anchorages):
            if sorted_users_of_anchorages[anchorage_index] < sorted_capacity_of_anchorages[anchorage_index]:
                # node anchorage is found
                break

        return node_anchorage

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

    # Functions used to calculate the sail-in-times for a specific vessel
    def provide_tidal_window_restriction(self, vessel, route, specifications, node):
        """Function: determines which tidal window restriction applies to the vessel at the specific node

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
            - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
            - specifications: the specific data regarding the properties for which the restriction holds
            - node: a string that defines the node of the tidal window restriction

        """

        # Predefined bool
        boolean = True
        no_tidal_window = True
        # Determining if and which restriction applies for the vessel by looping over the restriction class
        for restriction_class in enumerate(specifications[0]):
            # - if restriction does not apply to vessel because it is for vessels sailing in the opposite direction: continue loop
            if vessel.bound != specifications[2][restriction_class[0]]:
                continue
            if specifications[5][restriction_class[0]] != [] and (
                specifications[5][restriction_class[0]][0] not in route or specifications[5][restriction_class[0]][1] not in route
            ):
                continue
            # - else: looping over the restriction criteria
            for restriction_type in enumerate(restriction_class[1]):
                # - if previous condition is not met and there are no more restriction criteria and the previous condition has an 'AND' boolean statement: continue loop

                if (
                    not boolean
                    and restriction_type[0] == len(restriction_class[1]) - 1
                    and specifications[4][restriction_class[0]][restriction_type[0] - 1] == "and"
                ):
                    continue
                # - if previous condition is not met and there are more restriction criteria and the next condition has an 'AND' boolean statement: continue loop
                if (
                    not boolean
                    and restriction_type[0] != len(restriction_class[1]) - 1
                    and specifications[4][restriction_class[0]][restriction_type[0]] == "and"
                ):
                    continue
                # - if previous condition is not met and the next condition has an 'OR' boolean statement: continue loop with predefined boolean
                if (
                    not boolean
                    and restriction_type[0] != len(restriction_class[1]) - 1
                    and specifications[4][restriction_class[0]][restriction_type[0]] == "or"
                ):
                    boolean = True
                    continue

                # Extracting the correct vessel property for the restriction type
                if restriction_type[1].find("Length") != -1:
                    value = getattr(vessel, "L")
                if restriction_type[1].find("Draught") != -1:
                    value = getattr(vessel, "T")
                if restriction_type[1].find("Beam") != -1:
                    value = getattr(vessel, "B")
                if restriction_type[1].find("UKC") != -1:
                    value, _, _, _, _, _ = self.provide_ukc_clearance(vessel, node, delay)
                if restriction_type[1].find("Type") != -1:
                    value = getattr(vessel, "type")
                # Determine if the value for the property satisfies the condition of the restriction type
                df = pd.DataFrame({"Value": [value], "Restriction": [specifications[1][restriction_class[0]][restriction_type[0]]]})
                boolean = df.eval("Value" + specifications[3][restriction_class[0]][restriction_type[0]] + "Restriction")[0]

                # - if condition is not met: continue loop
                if not boolean and restriction_type[0] != len(restriction_class[1]) - 1:
                    continue

                # - if one of the conditions is met and the restriction contains an 'OR' boolean statement:
                if (
                    boolean
                    and restriction_type[0] != len(restriction_class[1]) - 1
                    and specifications[-1][restriction_class[0]][restriction_type[0]] == "or"
                ):
                    no_tidal_window = False
                    break
                # - elif all the conditions are met and the restriction contains solely 'AND' boolean statements:
                elif boolean and restriction_type[0] == len(restriction_class[1]) - 1:
                    no_tidal_window = False
                    break

            # - if condition is met: break the loop
            if boolean == True:
                break

            # - else: restart the loop with predefined bool
            else:
                boolean = True

        return restriction_class[0], no_tidal_window

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

    def provide_ukc_clearance(self, vessel, node, delay=0):
        """Function: calculates the sail-in-times for a specific vssel with certain properties and a pre-determined route and provides this information to the vessel

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
            - node:
            - components_calc:

        """
        # ignore water_level
        MBL, _, available_water_depth = self.provide_water_depth(vessel, node, delay)

        ukc_s, ukc_p, ukc_r, fwa = np.zeros(4)
        if "Vertical tidal restriction" in vessel.multidigraph.nodes[node]["Info"].keys():
            ukcs_s, ukcs_p, ukcs_r, fwas = vessel.multidigraph.nodes[node]["Info"]["Vertical tidal restriction"]["Type"]
            specifications = vessel.multidigraph.nodes[node]["Info"]["Vertical tidal restriction"]["Specification"]

            # Determine which restriction applies to vessel
            restriction_index, _ = self.provide_tidal_window_restriction(vessel, [node], specifications, node)

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

    def provide_minimum_available_water_depth_along_route(self, vessel, route, time_start, time_end, delay=0):
        """Function: calculates the minimum available water depth (predicted/modelled/measured water level minus the local maintained bed level) along the route over time,
                  subtracted with the difference between the gross ukc and net ukc (hence: subtracted with additional safety margins consisting of vessel-related factors
                  and water level factors). The bottom-related factors are already accounted for in the use of the MBL instead of the actual depth.

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
            - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
            - delay:
        """
        time_start_index = np.max(
            [
                0,
                np.absolute(self.hydrodynamic_information.TIME.values - (time_start + np.timedelta64(int(delay), "s"))).argmin()
                - 2,
            ]
        )
        time_end_index = np.absolute(
            self.hydrodynamic_information.TIME.values - (time_end + np.timedelta64(int(delay), "s"))
        ).argmin()
        net_ukc = pd.DataFrame()
        times = self.hydrodynamic_information["TIME"].values[time_start_index:time_end_index]
        t_step = times[1] - times[0]

        # Start of calculation by looping over the nodes of the route
        for route_index, node_name in enumerate(route):
            node_index = list(self.hydrodynamic_information["STATION"].values).index(node_name)
            sailing_time_to_next_node = vessel.env.vessel_traffic_service.provide_sailing_time(vessel, route[: (route_index + 1)])
            time_correction_index = int(np.round(sailing_time_to_next_node["Time"].sum() / (t_step / np.timedelta64(1, "s"))))
            water_level = self.hydrodynamic_information["Water level"][node_index].values[time_start_index:time_end_index]
            _, _, _, required_water_depth, _, _ = self.provide_ukc_clearance(vessel, node_name, delay)
            MBL = self.hydrodynamic_information["MBL"][node_index].values[time_start_index:time_end_index]
            water_depth = water_level + MBL
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
        """Function: calculates the windows available to sail-in and -out of the port given the vertical tidal restrictions according to the tidal window policy.

        Input:
            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
            - route: a list of strings of node names that resemble the route that the vessel is planning to sail (can be different than vessel.route)
            - sailing_time_correction: a bool that indicates whether the calculation should correct for sailing_speed (dynamic calculation) or not (static calculation)

        """
        vertical_tidal_accessibility = pd.DataFrame(columns=["Limit", "Accessibility"])
        net_ukcs = self.provide_minimum_available_water_depth_along_route(vessel, route, time_start, time_end, delay)
        time_start_index = np.max(
            [
                0,
                np.absolute(self.hydrodynamic_information.TIME.values - (time_start + np.timedelta64(int(delay), "s"))).argmin()
                - 2,
            ]
        )
        time_end_index = np.absolute(
            self.hydrodynamic_information.TIME.values - (time_end + np.timedelta64(int(delay), "s"))
        ).argmin()
        time_net_ukc = net_ukcs.index.to_numpy()
        net_ukc = net_ukcs["min_net_ukc"].to_numpy()

        # Determine zero crossinsg
        zero_crossings = np.nonzero(np.diff(np.sign(net_ukc)))
        for root in [time_net_ukc[i] for i in zero_crossings]:
            # -if the net ukc a moment later than the root is higher than the required water depth:
            if net_ukc[bisect.bisect_right(time_net_ukc, root)] > 0:
                vertical_tidal_accessibility.loc[root, :] = [0, "Accessible"]
            elif net_ukc[bisect.bisect_right(time_net_ukc, root)] < 0:
                vertical_tidal_accessibility.loc[root, :] = [0, "Inaccessible"]

        # Default values
        if vertical_tidal_accessibility.empty:
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [0, "Accessible"]
            vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [
                0,
                "Inaccessible",
            ]
        else:
            if vertical_tidal_accessibility.iloc[0].Accessibility == "Inaccessible":
                vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [
                    0,
                    "Accessible",
                ]
            else:
                vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_start.replace(tzinfo=None)), :] = [
                    0,
                    "Inaccessible",
                ]
            if vertical_tidal_accessibility.iloc[-1].Accessibility == "Inaccessible":
                vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [
                    0,
                    "Accessible",
                ]
            else:
                vertical_tidal_accessibility.loc[np.datetime64(vessel.env.simulation_stop.replace(tzinfo=None)), :] = [
                    0,
                    "Inaccessible",
                ]

        # Return the sail-in or -out-times given the vertical tidal restrictions over the route of the vessel
        vertical_tidal_accessibility = vertical_tidal_accessibility.sort_index()
        vertical_tidal_accessibility["Condition"] = "Water level"
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
            ax.set_xlim(
                self.hydrodynamic_information.TIME.values[time_start_index],
                self.hydrodynamic_information.TIME.values[time_end_index - 36],
            )
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
        def calculate_horizontal_tidal_window(
            vessel,
            time_start_index,
            time_end_index,
            hydrodynamic_data,
            critical_limits=[],
            cross_current_limit_dataframe=pd.DataFrame(),
            flood=True,
            ebb=True,
            decreasing=False,
        ):
            station = hydrodynamic_data.STATION.values
            time_start_index = np.max(
                [
                    0,
                    time_start_index
                    - int(np.timedelta64(12, "h") / (hydrodynamic_data.TIME.values[1] - hydrodynamic_data.TIME.values[0])),
                ]
            )
            currents_time = hydrodynamic_data.TIME.values[time_start_index:time_end_index]
            currents_data, _ = self.provide_governing_current_velocity(vessel, station, time_start_index, time_end_index)
            index_prev_root = 0
            roots = sc.interpolate.CubicSpline(currents_time, currents_data).roots()

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

            # Filter the found interpolation points: remove from errors (multiple interpolated numbers and flood/ebb values if not required)
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

            horizontal_tidal_accessibility = horizontal_tidal_accessibility.sort_index()
            horizontal_tidal_accessibility = horizontal_tidal_accessibility[horizontal_tidal_accessibility.Period_nr != -999]
            # Add accessibility information to intersection points
            if not horizontal_tidal_accessibility.empty:
                if not decreasing:
                    horizontal_tidal_accessibility.loc[
                        list(horizontal_tidal_accessibility.index)[1::2], "Accessibility"
                    ] = "Accessible"
                    horizontal_tidal_accessibility.loc[
                        list(horizontal_tidal_accessibility.index)[::2], "Accessibility"
                    ] = "Inaccessible"
                    limiting_currents = np.interp(
                        horizontal_tidal_accessibility.index.to_numpy().astype(float),
                        cross_current_limit_dataframe.index.to_numpy().astype(float),
                        cross_current_limit_dataframe.Limit.to_numpy(),
                    )
                    horizontal_tidal_accessibility["Accessibility"] = [
                        "Accessible" if interpcur < limit else accessibility
                        for accessibility, interpcur, limit in zip(
                            horizontal_tidal_accessibility.Accessibility.to_numpy(),
                            horizontal_tidal_accessibility.Limit.to_numpy(),
                            limiting_currents,
                        )
                    ]
                    horizontal_tidal_accessibility["Limit"] = [
                        0 if interpcur < limit else limit
                        for interpcur, limit in zip(horizontal_tidal_accessibility.Limit.to_numpy(), limiting_currents)
                    ]
                    horizontal_tidal_accessibility = horizontal_tidal_accessibility[horizontal_tidal_accessibility.Limit != -1.0]
                else:
                    horizontal_tidal_accessibility.loc[
                        list(horizontal_tidal_accessibility.index)[1::2], "Accessibility"
                    ] = "Inaccessible"
                    horizontal_tidal_accessibility.loc[
                        list(horizontal_tidal_accessibility.index)[::2], "Accessibility"
                    ] = "Accessible"
                horizontal_tidal_accessibility["Condition"] = "Current velocity"
                horizontal_tidal_accessibility = horizontal_tidal_accessibility[["Limit", "Condition", "Accessibility"]]
            return horizontal_tidal_accessibility, station

        # Start calculation
        horizontal_tidal_restriction_nodes = []
        horizontal_tidal_restriction_stations = []
        window_specifications = []
        horizontal_tidal_accessibility = pd.DataFrame(columns=["Limit", "Condition", "Accessibility"])
        horizontal_tidal_window = False
        for route_index, node_name in enumerate(route):
            if "Horizontal tidal restriction" in vessel.multidigraph.nodes[node_name]["Info"].keys():
                horizontal_tidal_window = True
                sailing_time_to_next_node = self.provide_sailing_time(vessel, route[: (route_index + 1)])
                specifications = vessel.multidigraph.nodes[node_name]["Info"]["Horizontal tidal restriction"]["Specification"]
                restriction_index, no_tidal_window = self.provide_tidal_window_restriction(vessel, route, specifications, node_name)
                if no_tidal_window:
                    continue
                hydrodynamic_data = vessel.multidigraph.nodes[node_name]["Info"]["Horizontal tidal restriction"]["Data"][
                    restriction_index
                ]
                cross_current_limit = vessel.multidigraph.nodes[node_name]["Info"]["Horizontal tidal restriction"]["Limit"][
                    restriction_index
                ]
                window_specifications = vessel.multidigraph.nodes[node_name]["Info"]["Horizontal tidal restriction"]["Type"][
                    restriction_index
                ]
                time_start_index = np.max(
                    [
                        0,
                        np.absolute(
                            self.hydrodynamic_information.TIME.values - (time_start + np.timedelta64(int(delay), "s"))
                        ).argmin()
                        - 2,
                    ]
                )
                time_end_index = np.absolute(hydrodynamic_data.TIME.values - (time_end + np.timedelta64(int(delay), "s"))).argmin()
                if window_specifications.window_method == "Maximum":
                    next_horizontal_tidal_accessibility, station = calculate_horizontal_tidal_window(
                        vessel,
                        time_start_index,
                        time_end_index,
                        hydrodynamic_data,
                        cross_current_limit_dataframe=cross_current_limit,
                    )
                if window_specifications.window_method == "Point-based":
                    if isinstance(window_specifications.current_velocity_values["Flood"], list) and not isinstance(
                        window_specifications.current_velocity_values["Ebb"], list
                    ):
                        next_horizontal_tidal_accessibility, station = calculate_horizontal_tidal_window(
                            vessel,
                            time_start_index,
                            time_end_index,
                            hydrodynamic_data,
                            critical_limits=cross_current_limit,
                            ebb=False,
                            decreasing=True,
                        )
                    elif isinstance(window_specifications.current_velocity_values["Ebb"], list) and not isinstance(
                        window_specifications.current_velocity_values["Flood"], list
                    ):
                        next_horizontal_tidal_accessibility, station = calculate_horizontal_tidal_window(
                            vessel,
                            time_start_index,
                            time_end_index,
                            hydrodynamic_data,
                            critical_limits=cross_current_limit,
                            flood=False,
                            decreasing=True,
                        )
                    else:
                        next_horizontal_tidal_accessibility, station = calculate_horizontal_tidal_window(
                            vessel,
                            time_start_index,
                            time_end_index,
                            hydrodynamic_data,
                            critical_limits=cross_current_limit,
                            decreasing=True,
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
            for node, station in zip(horizontal_tidal_restriction_nodes, horizontal_tidal_restriction_stations):
                governing_current_velocity, _ = self.provide_governing_current_velocity(
                    vessel, station, time_start_index, time_end_index
                )
                horizontal_tidal_accessibility_time_correction = np.timedelta64(
                    int(self.provide_sailing_time(vessel, route[: (route.index(node) + 1)])["Time"].sum() + delay), "s"
                )
                (current_velocity,) = ax.plot(
                    [
                        time - horizontal_tidal_accessibility_time_correction
                        for time in self.hydrodynamic_information.TIME.values[time_start_index:time_end_index]
                    ],
                    governing_current_velocity,
                    color="firebrick",
                    linewidth=2,
                )
            ax.axhline(0, color="k", linewidth=1)

            # Figure bounds
            ax.set_xlim(
                self.hydrodynamic_information.TIME.values[time_start_index],
                self.hydrodynamic_information.TIME.values[time_end_index - 36],
            )
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

        return (
            horizontal_tidal_accessibility,
            horizontal_tidal_windows,
            horizontal_tidal_restriction_nodes,
            horizontal_tidal_restriction_stations,
            window_specifications,
        )

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

    def provide_tidal_windows(self, vessel, route, time_start, time_end, ax_left=None, ax_right=None, delay=0, plot=False):
        time_start_index = np.max(
            [
                0,
                np.absolute(self.hydrodynamic_information.TIME.values - (time_start + np.timedelta64(int(delay), "s"))).argmin()
                - 2,
            ]
        )
        time_end_index = np.absolute(
            self.hydrodynamic_information.TIME.values - (time_end + np.timedelta64(int(delay), "s"))
        ).argmin()
        vertical_tidal_accessibility, vertical_tidal_windows, net_ukcs = self.provide_vertical_tidal_windows(
            vessel, route, time_start, time_end, delay
        )
        (
            horizontal_tidal_accessibility,
            horizontal_tidal_windows,
            horizontal_tidal_restriction_nodes,
            horizontal_tidal_restriction_stations,
            window_specifications,
        ) = self.provide_horizontal_tidal_windows(vessel, route, time_start, time_end, delay)
        tidal_accessibility = self.combine_tidal_windows(vertical_tidal_accessibility, horizontal_tidal_accessibility)
        tidal_windows = [
            [window_start[0], window_end[0]]
            for window_start, window_end in zip(tidal_accessibility.iloc[:-1].iterrows(), tidal_accessibility.iloc[1:].iterrows())
            if window_start[1].Accessibility == "Accessible"
        ]
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
            for node, station in zip(horizontal_tidal_restriction_nodes, horizontal_tidal_restriction_stations):
                governing_current_velocity, _ = self.provide_governing_current_velocity(
                    vessel, station, time_start_index, time_end_index
                )
                horizontal_tidal_accessibility_time_correction = np.timedelta64(
                    int(self.provide_sailing_time(vessel, route[: (route.index(node) + 1)])["Time"].sum()), "s"
                ) + np.timedelta64(15, "m")
                (current_velocity,) = ax_right.plot(
                    [
                        time - horizontal_tidal_accessibility_time_correction
                        for time in self.hydrodynamic_information.TIME.values[time_start_index:time_end_index]
                    ],
                    governing_current_velocity,
                    color="firebrick",
                    linewidth=2,
                    zorder=1,
                )
                if window_specifications.window_method == "Maximum":
                    critical_current_velocity = ax_right.axhline(
                        window_specifications.current_velocity_values["Flood"], color="firebrick", linestyle="--", linewidth=2
                    )
                    ax_right.axhline(
                        -1 * window_specifications.current_velocity_values["Ebb"], color="firebrick", linestyle="--", linewidth=2
                    )
                ax_right.set_ylim(np.floor(np.min(governing_current_velocity)), np.ceil(np.max(governing_current_velocity)))

            # Figure bounds
            ax_left.set_xlim(
                self.hydrodynamic_information.TIME.values[time_start_index],
                self.hydrodynamic_information.TIME.values[time_end_index - 36],
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
                        polygon_x.append(pd.Timestamp(datetime.datetime.fromtimestamp(timestamp, tz=pytz.utc)))
                    polygon_y = list(polygon.exterior.xy[1])
                    (vertical_tidal_window,) = ax_left.fill(
                        polygon_x, polygon_y, facecolor="C0", alpha=0.25, edgecolor="none", zorder=0
                    )
            elif isinstance(vertical_tidal_window_polygons, Polygon):
                polygon = vertical_tidal_window_polygons
                polygon_x = []
                for timestamp in polygon.exterior.xy[0]:
                    polygon_x.append(pd.Timestamp(datetime.datetime.fromtimestamp(timestamp, tz=pytz.utc)))
                polygon_y = list(polygon.exterior.xy[1])
                (vertical_tidal_window,) = ax_left.fill(
                    polygon_x, polygon_y, facecolor="C0", alpha=0.25, edgecolor="none", zorder=0
                )

            # Plot horizontal tidal windows
            if not isinstance(horizontal_tidal_window_polygons, Polygon):
                for polygon in horizontal_tidal_window_polygons.geoms:
                    polygon_x = []
                    for timestamp in polygon.exterior.xy[0]:
                        polygon_x.append(pd.Timestamp(datetime.datetime.fromtimestamp(timestamp, tz=pytz.utc)))
                    polygon_y = list(polygon.exterior.xy[1])
                    (horizontal_tidal_window,) = ax_left.fill(
                        polygon_x, polygon_y, facecolor="firebrick", alpha=0.25, edgecolor="none", zorder=0
                    )
            elif isinstance(horizontal_tidal_window_polygons, Polygon):
                polygon = horizontal_tidal_window_polygons
                polygon_x = []
                for timestamp in polygon.exterior.xy[0]:
                    polygon_x.append(pd.Timestamp(datetime.datetime.fromtimestamp(timestamp, tz=pytz.utc)))
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
                    [ax_left.get_ylim()[0], ax_left.get_ylim()[1], ax.get_ylim()[1], ax_left.get_ylim()[0]],
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
            if window_specifications:
                if window_specifications.window_method == "Maximum":
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
                        bbox_to_anchor=(1.05, 1.0),
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
                        bbox_to_anchor=(1.05, 1.0),
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
                f"Accessibility of {vessel.type}-class vessel with name {vessel.name} and {np.round(vessel.T,2)}m draught and a length of {np.round(vessel.L)}m, sailing {vessel.bound}"
            )
            plt.show()
            return tidal_accessibility, tidal_windows, ax_left, ax_right

        return tidal_accessibility, tidal_windows
