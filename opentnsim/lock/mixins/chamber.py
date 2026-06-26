"""Contains the mixin for lock chambers. Also contains the parent class LockChamberOperatior."""
import datetime
import math
import numpy as np
import pandas as pd
import networkx as nx
from shapely.geometry import LineString

from opentnsim.constants import knots
from opentnsim.core import HasResource, Identifiable, Log, HasLength, ExtraMetadata, Locatable
from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from opentnsim.graph.calculations import calculate_location_over_edges, transform_geometry
from opentnsim.graph.mixins import OnEdge
from opentnsim.graph.utils import (
    check_if_geometry_is_aligned_with_edge,
    _get_edges_from_geometry,
)
from opentnsim.lock.calculations import (
    calculate_ic_ratio,
    calculate_and_check_lock_dimensions,
    calculate_lock_distances_to_nodes_of_edge_from_geometry,
    calculate_lock_occupancy,
    calculate_cycle_event_durations,
    calculate_saltwater_intrusion
)
from opentnsim.lock.mixins.operator import IsLockChamberOperator
from opentnsim.lock.logutils import get_vessel_delays, calculate_cycle_information
from opentnsim.lock.utils import (
    add_lock_to_graph,
    _create_operational_hours,
    _verify_node_AB,
    check_lock_distances_to_nodes_of_edge,
)
from opentnsim.lock.visualizations import create_time_distance_plot, show_results


class IsLockGate(HasResource, Log, Locatable, Identifiable):
    def __init__(self, *args, **kwargs):
        super().__init__(nr_resources=1, *args, **kwargs)


class IsLockChamber(IsLockChamberOperator, OnEdge, HasResource, HasLength, Identifiable, Log, ExtraMetadata):
    """Mixin class: lock complex has a lock chamber:

    creates a lock chamber with a resource which is requested when a vessels wants to enter the area with limited capacity

    """

    def __init__(
        self,
        env,
        lock_depth, # a float which contains the depth of the lock chamber
        lock_length = None,  # a float which contains the length of the lock chamber
        lock_width = None,  # a float which contains the width of the lock chamber
        edge=None,
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
        sailing_distance_to_crossing_point_A = None,
        sailing_distance_to_crossing_point_B = None,
        sailing_in_speed_A=2 * knots,  # a float that is the speed at which the vessel sails into the lock to the sea side [m/s]
        sailing_out_speed_A=2 * knots,  # a float that is the speed at which the vessel sails out of the lock to the sea side [m/s]
        sailing_in_speed_B=2 * knots,  # a float that is the speed at which the vessel sails into the lock to the canal side [m/s]
        sailing_out_speed_B=2 * knots,  # a float that is the speed at which the vessel sails out of the lock to the canal side [m/s]
        minimum_manoeuvrability_speed=2 * knots,  # a float that is the minimum speed at which the vessel is still safely manoeuvrable [m/s]
        gate_open_at_node=None,  # a string that is the node name to which the lock was last levelled to at the initial time of simulation (either start_node or end_node)
        water_level_init = None,
        salinity_init = None,
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

        if edge is None:
            if len(edges) != allowed_nr_edges:
                raise ValueError(f"Lock geometry does not cover a single geometry, but {len(edges)} edges.")
            else:
                edge = edges[0]

        # initialization
        super().__init__(env=env,
                         edge=edge,
                         nr_resources=math.inf,
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
        
        edge_length = self.env.graph.edges[self.edge]['length_m']
        calculated_edge_length = distance_from_start_node_to_lock_gate_A + self.lock_length + distance_from_end_node_to_lock_gate_B 
        if not self.geometry and not self.geometry_m and calculated_edge_length != edge_length:
            distance_from_start_node_to_lock_gate_A = (edge_length - self.lock_length) / 2
            distance_from_end_node_to_lock_gate_B = (edge_length - self.lock_length) / 2
        
        self.start_node = self.edge[0]
        self.end_node = self.edge[1]
        self.k = 0
        self.edge_reversed = (self.end_node, self.start_node)
        if isinstance(self.env.graph, nx.MultiDiGraph):
            self.k = self.edge[2]
            self.edge_reversed = (self.end_node, self.start_node, self.k)

        # gate information
        self.distance_from_start_node_to_lock_gate_A = distance_from_start_node_to_lock_gate_A
        self.distance_from_end_node_to_lock_gate_B = distance_from_end_node_to_lock_gate_B
        geometry_gate_A = calculate_location_over_edges(self.env.graph, self.edge, distance_from_start_node_to_lock_gate_A, crs_m = self.crs_m)
        self.gate_A = IsLockGate(env = self.env, name = self.start_node, geometry = geometry_gate_A)
        self.levelling = HasResource(env = self.env, nr_resources = 1)
        geometry_gate_B = calculate_location_over_edges(self.env.graph, self.edge, distance_from_start_node_to_lock_gate_A + self.lock_length, crs_m = self.crs_m)
        self.gate_B = IsLockGate(env = self.env, name = self.end_node, geometry = geometry_gate_B)
        self.gate_opening_time = gate_opening_time
        self.gate_closing_time = gate_closing_time
        self.gate_A_open = True
        self.gate_B_open = True
        self.gate_open_at_node = gate_open_at_node
        if self.gate_open_at_node is None:
            self.gate_open_at_node = self.start_node
        if self.gate_open_at_node == self.start_node:
            self.gate_B_open = False
        else:
            self.gate_A_open = False
        self.gate_open_at_node_init = self.gate_open_at_node

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
        try:
            self.has_water_level = True
            wlev_init = hydromanager._get_hydrodynamic_data_value(time, self.gate_open_at_node, "Water level")
            if pd.isna(wlev_init):
                self.has_water_level = False
        except:
            self.has_water_level = False
        try:
            self.has_salinity = True
            sal_init = hydromanager._get_hydrodynamic_data_value(time, self.gate_open_at_node, "Salinity")
            if pd.isna(sal_init):
                self.has_salinity = False
        except:
            self.has_salinity = False

        if self.has_water_level:
            time_series = pd.date_range(time, self.env.simulation_stop, freq=pd.Timedelta(seconds=self.time_step))        
            wlev_series = hydromanager._get_interpolated_hydrodynamic_series(time_series,self.gate_open_at_node, "Water level",)
            if self.closing_gate_in_between_operations:
                if water_level_init is None:
                    water_level_init = wlev_series[0][0]
                wlev_series = water_level_init * np.ones(len(time_series))
            self.time = time_series
            self.water_level = wlev_series
        if self.has_salinity:
            time_series = pd.date_range(time, self.env.simulation_stop, freq=pd.Timedelta(seconds=self.time_step))
            sal_series = hydromanager._get_interpolated_hydrodynamic_series(time_series, self.gate_open_at_node,"Salinity", )
            if self.closing_gate_in_between_operations and salinity_init is not None:
                if salinity_init is None:
                    salinity_init = sal_series[0][0]
                sal_series = salinity_init * np.ones(len(time_series))
            self.time = time_series
            self.salinity = sal_series
            self.saltmass = (self.water_level + self.lock_depth)*self.lock_width*self.lock_length*sal_series
            self.node_sea = self.start_node
            self.node_lake = self.end_node
            salt_start = hydromanager._get_hydrodynamic_data_series(time, self.start_node, "Salinity").mean()
            salt_end = hydromanager._get_hydrodynamic_data_series(time, self.end_node, "Salinity").mean()
            if salt_start < salt_end:
                self.node_sea = self.end_node
                self.node_lake = self.start_node

        # operational information
        self.minimum_manoeuvrability_speed = minimum_manoeuvrability_speed
        self.sailing_in_speed_A = sailing_in_speed_A
        self.sailing_out_speed_A = sailing_out_speed_A
        self.sailing_in_speed_B = sailing_in_speed_B
        self.sailing_out_speed_B = sailing_out_speed_B
        if sailing_distance_to_crossing_point_A is None or sailing_distance_to_crossing_point_B is None:
            self.sailing_distance_to_crossing_point_A = sailing_distance_to_crossing_point
            self.sailing_distance_to_crossing_point_B = sailing_distance_to_crossing_point
        else:
            self.sailing_distance_to_crossing_point_A = sailing_distance_to_crossing_point_A
            self.sailing_distance_to_crossing_point_B = sailing_distance_to_crossing_point_B
        self.speed_reduction_factor = speed_reduction_factor_lock_chamber
        self.converting_chamber = False

        if operational_hour_start_times is not None and operational_hour_stop_times is not None:
            operational_hours = _create_operational_hours(operational_hour_start_times, operational_hour_stop_times)
        else:
            operational_hours = _create_operational_hours([datetime.datetime.min], [datetime.datetime.max])
        self.operational_hours = operational_hours

        if self.geometry_m is not None and self.geometry is None:
            self.geometry = transform_geometry(self.geometry_m, self.crs_m, "EPSG:4326")
        elif self.geometry is not None and self.geometry_m is None:
            self.geometry_m = transform_geometry(self.geometry, "EPSG:4326", self.crs_m)
        elif self.geometry is None and self.geometry_m is None:
            line = LineString([geometry_gate_A, geometry_gate_B])
            line_m = transform_geometry(line, "EPSG:4326", self.crs_m)
            self.geometry_m = line_m.buffer(self.lock_width / 2, cap_style=2) 
            self.geometry = transform_geometry(self.geometry_m, self.crs_m, "EPSG:4326")

        # checks
        check_lock_distances_to_nodes_of_edge(self)
        check_if_geometry_is_aligned_with_edge(self.env.graph, self.edge)
        _verify_node_AB(self)

        # Add to the graph:
        add_lock_to_graph(self)


    def get_performance(self):
        Tc_df = calculate_cycle_information(self)
        if Tc_df.empty:
            return pd.Series()
        ic, capacity = calculate_ic_ratio(self, Tc_df)
        occupancy, _ = calculate_lock_occupancy(self)
        event_durations, event_durations_summary = calculate_cycle_event_durations(Tc_df)
        _, vessel_delays, vessel_delays_causes = get_vessel_delays(self)
        vessel_delays_locations = vessel_delays[['waiting_area (%)',
                                                 'sailing_to_lock (%)',
                                                 'in_lock (%)',
                                                 'sailing_from_lock (%)']]
        vessel_delays_causes = vessel_delays_causes[['congestion (%)',
                                                     'obstruction (%)',
                                                     'traffic (%)',
                                                     'operation of lock (%)']]
        results = {'Minimum individual vessel delay': vessel_delays['min_vessel_delay'],
                   'Average individual vessel delay': vessel_delays['average_vessel_delay'],
                   'Maximum individual vessel delay': vessel_delays['max_vessel_delay'],
                   'Total vessel delay': vessel_delays['total_delay'],
                   'Delay areas': dict(sorted(vessel_delays_locations.to_dict().items(), key=lambda x: x[1], reverse=True)),
                   'Delay causes': dict(sorted(vessel_delays_causes.to_dict().items(), key=lambda x: x[1], reverse=True)),
                   'Cycle-averaged occupancy (%)': occupancy,
                   'Average cycle time': event_durations['Average cycle time'],
                   'Cycle event composition': event_durations_summary.to_dict(),
                   'Lock capacity': np.round(capacity,2),
                   'Cycle-averaged I/C-ratio': np.round(ic,2)}

        if self.has_salinity:
            saltwater_intrusion_results, saltwater_intrusion_causes = calculate_saltwater_intrusion(self)
            results['Water volume lost [m3]'] = np.round(saltwater_intrusion_results['water_volume_lost'], 1)
            results['Average outflow of water [m3/s]'] = np.round(saltwater_intrusion_results['water_outflow'], 1)
            results['Saltwater intrusion [kg]'] = np.round(saltwater_intrusion_results['saltwater_intrusion'], 1)
            results['Average saltwater intrusion flux [kg/s]'] = np.round(
                saltwater_intrusion_results['saltwater_intrusion_flux'], 1
            )
            results['Saltwater intrusion causes'] = saltwater_intrusion_causes

        summary = pd.Series(results)
        show_results(summary)
        return summary


    def get_aggregated_cycle_information(self):
        Tc_df = calculate_cycle_information(self)
        event_durations, _ = calculate_cycle_event_durations(Tc_df)
        return event_durations


    def plot(self, xlimmin, xlimmax, ylimmin, ylimmax, offset_x = 0., method = 'Matplotlib', boundary_nodes = None, fig=None, ax=None, legend=True):
        fig = create_time_distance_plot(self, xlimmin=xlimmin, xlimmax=xlimmax, ylimmin=ylimmin, ylimmax=ylimmax, offset_x = offset_x,
                                        method = method, boundary_nodes = boundary_nodes, fig=fig, ax=ax, legend = legend)
        return fig
