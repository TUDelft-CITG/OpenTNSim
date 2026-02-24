"""Contains the mixin for lock chambers. Also contains the parent class LockChamberOperatior."""
import datetime
import math
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import simpy
import networkx as nx

from opentnsim.constants import knots
from opentnsim.core import HasResource, Identifiable, Log, HasLength, ExtraMetadata, SimpyObject, Locatable
from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from opentnsim.graph.calculations import calculate_location_over_edges
from opentnsim.graph.mixins import HasMultiDiGraph, OnEdge
from opentnsim.graph.utils import (
    check_if_geometry_is_aligned_with_edge,
    _get_edges_from_geometry,
)
from opentnsim.lock.calculations import (
    calculate_lock_dimensions_from_geometry,
    calculate_and_check_lock_dimensions,
    calculate_lock_distances_to_nodes_of_edge_from_geometry,
)
from opentnsim.lock.mixins.operator import IsLockChamberOperator
from opentnsim.lock.utils import (
    _get_directional_edge,
    _get_lock_operation_to_and_from_node,
    _get_waiting_area,
    _get_distance_to_lock,
    add_lock_to_graph,
    _create_operational_hours,
    _verify_node_AB,
    check_lock_distances_to_nodes_of_edge,
)
from opentnsim.lock.visualizations import create_time_distance_plot
from opentnsim.output import HasOutput


class IsLockGate(HasResource, Locatable, Identifiable):
    def __init__(self, *args, **kwargs):
        super().__init__(nr_resources=1, *args, **kwargs)


class IsLockChamber(IsLockChamberOperator, OnEdge, HasResource, HasLength, Identifiable, Log, HasOutput, HasMultiDiGraph, ExtraMetadata):
    """Mixin class: lock complex has a lock chamber:

    creates a lock chamber with a resource which is requested when a vessels wants to enter the area with limited capacity

    """

    def __init__(
        self,
        env,
        edge = None,
        lock_length=0.0, # a float which contains the length of the lock chamber
        lock_width=0.0, # a float which contains the width of the lock chamber
        lock_depth=0.0, # a float which contains the depth of the lock chamber
        geometry=None,
        geometry_m=None,
        distance_from_start_node_to_lock_gate_A=0.0,  # a float that is the distance between the start_node of the edge and the lock gate A [m]
        distance_from_end_node_to_lock_gate_B=0.0,  # a float that is the distance between the end_node of the edge and the lock gate B [m]
        disch_coeff=0.4,  # a float which contains the discharge coefficient of filling system
        opening_area=12.0,  # a float which contains the cross-sectional area of filling system [m^2]
        opening_depth=None,  # a float which contains the depth at which filling system is located [m^2]
        levelling_time=600.0,  # a float that fixates the levelling time [s]
        time_step=10.0,  # a float that is the integration time step to determine the levelling time [s]
        valve_opening_time=60.0,  # a float that is the time it takes for the levelling gate to open [s]
        gate_opening_time=300.0,  # a float which contains the time it takes to open the gate [s]
        gate_closing_time=300.0,  # a float which contains the time it takes to close the gate [s]
        speed_reduction_factor_lock_chamber=0.3,  # a float that is the reduction factor for the vessel speed from its original speed when entering the lock
        sailing_distance_to_crossing_point=370.0,  # a float that is the distance at which vessels can safely pass each other in front of the lock (last vessel that sails out and first vessel that sails in) [m]
        sailing_in_speed_A=2 * knots,  # a float that is the speed at which the vessel sails into the lock to the sea side [m/s]
        sailing_out_speed_A=2 * knots,  # a float that is the speed at which the vessel sails out of the lock to the sea side [m/s]
        sailing_in_speed_B=2 * knots,  # a float that is the speed at which the vessel sails into the lock to the canal side [m/s]
        sailing_out_speed_B=2 * knots,  # a float that is the speed at which the vessel sails out of the lock to the canal side [m/s]
        minimum_manoeuvrability_speed=2 * knots,  # a float that is the minimum speed at which the vessel is still safely manoeuvrable [m/s]
        gate_open=None,  # a string that is the node name to which the lock was last levelled to at the initial time of simulation (either start_node or end_node)
        operational_hour_start_times=None,
        operational_hour_stop_times=None,
        crs_m = "EPSG:4087",
        *args,
        **kwargs,
    ):
        """Initialization"""
        # geometrical information (before initialization)
        self.lock_length = lock_length
        self.lock_width = lock_width
        self.lock_depth = lock_depth
        self.geometry = geometry
        self.geometry_m = geometry_m
        self.crs_m = crs_m
        calculate_and_check_lock_dimensions(self)
        if edge is None and geometry is None and geometry_m is None:
            raise ValueError("User did not specify an edge for the lock complex, and could not be computed based on a geometry")
        elif edge is None:
            if geometry_m is not None:
                edges = _get_edges_from_geometry(env.graph, geometry_m, crs_m, m=True)
            else:
                edges = _get_edges_from_geometry(env.graph, geometry, crs_m)

        allowed_nr_edges = 1
        if env.graph.is_directed():
            allowed_nr_edges = 2
        if len(edges) != allowed_nr_edges:
            raise ValueError(f"Lock geometry does not cover a single geometry, but {len(edges)} edges.")
        edge = edges[0]

        # initialization
        super().__init__(env=env,
                         edge=edge,
                         capacity=math.inf,
                         length=self.lock_length,
                         remaining_length=self.lock_length,
                         *args, **kwargs)

        # more geometrical information (after initialization)
        m = None
        if self.geometry is not None:
            m = False
        elif self.geometry_m is not None:
            m = True
        if m is not None:
            (distance_from_start_node_to_lock_gate_A,
             distance_from_end_node_to_lock_gate_B) = calculate_lock_distances_to_nodes_of_edge_from_geometry(self, m=m)
        self.distance_from_start_node_to_lock_gate_A = distance_from_start_node_to_lock_gate_A
        self.distance_from_end_node_to_lock_gate_B = distance_from_end_node_to_lock_gate_B
        self.start_node = self.edge[0]
        self.end_node = self.edge[1]
        self.k = 0
        self.edge_reversed = (self.end_node, self.start_node)
        if isinstance(self.env.graph, nx.MultiDiGraph):
            self.k = self.edge[2]
            self.edge_reversed = (self.end_node, self.start_node, self.k)

        # gate information
        geometry_gate_A = calculate_location_over_edges(self.env.graph, self.edge, distance_from_start_node_to_lock_gate_A, crs_m = self.crs_m)
        self.gate_A = IsLockGate(env = self.env, name = 'Gate A', geometry = geometry_gate_A)
        self.levelling = HasResource(env = self.env, nr_resources = 1)
        geometry_gate_B = calculate_location_over_edges(self.env.graph, self.edge, distance_from_start_node_to_lock_gate_A + self.lock_length, crs_m = self.crs_m)
        self.gate_B = IsLockGate(env = self.env, name = 'Gate A', geometry = geometry_gate_B)
        self.gate_opening_time = gate_opening_time
        self.gate_closing_time = gate_closing_time
        self.gate_A_open = True
        self.gate_B_open = True
        self.gate_open = gate_open
        if self.gate_open is None:
            self.gate_open = self.start_node
        if self.gate_open == self.start_node:
            self.gate_B_open = False
        else:
            self.gate_A_open = False

        # valve and hydrodynamic information
        self.disch_coeff = disch_coeff
        self.opening_area = opening_area
        if opening_depth is None:
            opening_depth = lock_depth / 2
        self.opening_depth = opening_depth
        self.levelling_time = levelling_time
        self.time_step = time_step
        self.valve_opening_time = valve_opening_time
        time = np.datetime64(datetime.datetime.fromtimestamp(self.env.now))
        hydromanager = HydrodynamicDataManager()
        wlev_series = hydromanager._get_hydrodynamic_data_series(time, self.gate_open, "Water level")
        self.water_level = wlev_series

        # operational information
        self.minimum_manoeuvrability_speed = minimum_manoeuvrability_speed
        self.sailing_in_speed_A = sailing_in_speed_A
        self.sailing_out_speed_A = sailing_out_speed_A
        self.sailing_in_speed_B = sailing_in_speed_B
        self.sailing_out_speed_B = sailing_out_speed_B
        self.sailing_distance_to_crossing_point = sailing_distance_to_crossing_point
        self.speed_reduction_factor = speed_reduction_factor_lock_chamber
        self.converting_chamber = False

        if operational_hour_start_times is not None and operational_hour_stop_times is not None:
            operational_hours = _create_operational_hours(operational_hour_start_times, operational_hour_stop_times)
        else:
            operational_hours = _create_operational_hours([datetime.datetime.min], [datetime.datetime.max])
        self.operational_hours = operational_hours

        # checks
        check_lock_distances_to_nodes_of_edge(self)
        check_if_geometry_is_aligned_with_edge(self.env.graph, self.edge)
        _verify_node_AB(self)

        # Add to the graph:
        add_lock_to_graph(self)


    def plot(self, xlimmin=None, xlimmax=None, ylimmin=None, ylimmax=None, method = 'Matplotlib'):
        fig = create_time_distance_plot(self, xlimmin=xlimmin, xlimmax=xlimmax, ylimmin=ylimmin, ylimmax=ylimmax, method = method)
        return fig
