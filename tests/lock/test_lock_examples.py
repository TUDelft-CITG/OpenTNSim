# package(s) used for creating and geo-locating the graph
import networkx as nx
import pyproj
from shapely.geometry import Point, LineString
from shapely.ops import transform

# package(s) related to the simulation (creating the vessel, running the simulation)
import datetime
import simpy
import opentnsim
from opentnsim.core.logutils import logbook2eventtable
from opentnsim.core.plotutils import generate_vessel_gantt_chart

# import of modules important for locking
from opentnsim.lock import lock as lock_module
from opentnsim.vessel_traffic_service import vessel_traffic_service as vessel_traffic_service_module

# package(s) needed for inspecting the output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import pytest

def generate_vessel(
    env,
    name,
    start_node,
    end_node,
    arrival_time,
    vessel_speed=4,
    vessel_length=100,
    vessel_beam=20,
    vessel_draft=10,
    vessel_type="tanker",
    vessel_class=None
    ):
    """
    Creates and returns a Vessel object with a computed route through the environment graph.

    Parameters:
    ----------
    env : Environment
        The simulation environment containing the graph and other context.
    name : str
        Human readabile identifier for the vessel.
    start_node : str or int
        The starting node in the graph (converted to string).
    end_node : str or int
        The destination node in the graph (converted to string).
    arrival_time : pd.Timestamp
        The scheduled arrival time of the vessel at the start node.
    vessel_speed : float, optional
        Speed of the vessel in knots or simulation units (default is 4).
    vessel_length : float, optional
        Length of the vessel in meters (default is 100).
    vessel_beam : float, optional
        Beam (width) of the vessel in meters (default is 20).
    vessel_draft : float, optional
        Draught (depth below waterline) of the vessel in meters (default is 10).
    vessel_type : str, optional
        Type of vessel (e.g., "tanker", "cargo", "container") (default is "tanker").

    Returns:
    -------
    Vessel or None
        A Vessel object initialized with the given parameters and route.
        Returns None if no valid path exists between start_node and end_node.
    """
    
    # Ensure nodes are strings
    start_node = str(start_node)
    end_node = str(end_node)

    try:
        route = nx.dijkstra_path(env.graph, start_node, end_node)
    except nx.NetworkXNoPath:
        print(f"⚠️ No path from {start_node} to {end_node}. Vessel {name} not created.")
        return None

    geometry = env.graph.nodes[start_node]['geometry']

    data_vessel = {
        "env": env,
        "name": name,
        "geometry": geometry,
        "route": route,
        "v": vessel_speed,
        "L": vessel_length,
        "B": vessel_beam,
        "T": vessel_draft,
        "type": vessel_type,
        "arrival_time": arrival_time,
    }

    vessel = vessel_class(**data_vessel)

    return vessel

def generate_vessels_with_distributions(
    env,
    num_vessels,
    start_time,
    arrival_dist_up=None,
    arrival_dist_down=None,
    seed_up=None,
    seed_down=None,
    vessel_class=None
    ):
    """
    Generates a list of vessels with interarrival times drawn from specified distributions
    for upward and downward directions. Supports independent seeding for reproducibility.

    Parameters
    ----------
    env : Environment
        The simulation environment containing the graph and vessel context.
    num_vessels : int
        Total number of vessels to generate. Vessels alternate between up and down directions.
    start_time : pd.Timestamp
        The initial timestamp from which vessel arrivals begin.
    arrival_dist_up : callable, optional
        A function returning interarrival times (in minutes) for upward-moving vessels.
        If None, defaults to an exponential distribution with mean 20 minutes.
    arrival_dist_down : callable, optional
        A function returning interarrival times (in minutes) for downward-moving vessels.
        If None, defaults to an exponential distribution with mean 20 minutes.
    seed_up : int or None, optional
        Seed for the random number generator used in upward direction.
    seed_down : int or None, optional
        Seed for the random number generator used in downward direction.

    Returns
    -------
    list of Vessel
        A list of Vessel objects with assigned routes and arrival times.
        Vessels for which no valid path exists are skipped.
    """

    vessels = []

    # Create independent random generators
    rng_up = np.random.default_rng(seed_up)
    rng_down = np.random.default_rng(seed_down)

    # Default to exponential distribution with mean 20 minutes
    if arrival_dist_up is None:
        arrival_dist_up = lambda: rng_up.exponential(scale=20)
    if arrival_dist_down is None:
        arrival_dist_down = lambda: rng_down.exponential(scale=20)

    up_time = start_time
    down_time = start_time

    for i in range(num_vessels):
        if i % 2 == 0:
            # Upward direction: -1 → +1
            start_node, end_node = "-1", "+1"
            delta_minutes = arrival_dist_up()
            arrival_time = up_time + pd.Timedelta(minutes=delta_minutes)
            up_time = arrival_time
        else:
            # Downward direction: +1 → -1
            start_node, end_node = "+1", "-1"
            delta_minutes = arrival_dist_down()
            arrival_time = down_time + pd.Timedelta(minutes=delta_minutes)
            down_time = arrival_time

        vessel = generate_vessel(
            env=env,
            name=f"Vessel {i + 1}",
            start_node=start_node,
            end_node=end_node,
            arrival_time=arrival_time,
            vessel_class=vessel_class
        )

        if vessel:
            vessels.append(vessel)

    return vessels

@pytest.fixture
def Vessel():
    Vessel = type(
        "Vessel", 
        (
            lock_module.PassesLockComplex,             # allows to interact with a lock
            opentnsim.core.Identifiable,               # allows to give the object a name and a random ID,
            opentnsim.core.Movable,                    # allows the object to move, with a fixed speed, while logging this activity
            opentnsim.core.VesselProperties,           # allows vessel to have dimensions, namely a length (L), width (B), and draught (T)
            opentnsim.core.ExtraMetadata,              # allow additional information, such as an arrival time (required for passing a lock)
            opentnsim.graph.HasMultiDiGraph,           # allow to operate on a graph that can include parallel edges from and to the same nodes
            opentnsim.output.HasOutput,                # allow additional output to be stored
        ), 
        {}
    )
    return Vessel

@pytest.fixture
def wgs84eqd_to_wgs84rad():    
    """transformer function from equidistant WGS84 to radial WGS84"""
    # define reference systems
    wgs84eqd = pyproj.CRS('4087')
    wgs84rad = pyproj.CRS('4326')
    # define transformer functions
    wgs84eqd_to_wgs84rad = pyproj.transformer.Transformer.from_crs(wgs84eqd,wgs84rad,always_xy=True).transform #equidistant wgs84 to radial wgs84
    return wgs84eqd_to_wgs84rad

@pytest.fixture
def two_node_graph(wgs84eqd_to_wgs84rad):
    # create a directed graph
    graph = nx.DiGraph()

    # add nodes
    graph.add_node('0',geometry=transform(wgs84eqd_to_wgs84rad, Point(-5000,0)))
    graph.add_node('1',geometry=transform(wgs84eqd_to_wgs84rad, Point(5000,0)))

    # add edges
    graph.add_edge('0','1', geometry = transform(wgs84eqd_to_wgs84rad, LineString([Point(-5000, 0),Point(5000, 0)])), weight=1)
    graph.add_edge('1','0', weight=1); #it is not required to have a geometry
    return graph

@pytest.fixture
def three_node_graph(two_node_graph, wgs84eqd_to_wgs84rad):
    three_node_graph = two_node_graph.copy()

    # add nodes
    three_node_graph.add_node('-1',geometry=transform(wgs84eqd_to_wgs84rad,Point(-350600,0)))
    three_node_graph.add_node('+1',geometry=transform(wgs84eqd_to_wgs84rad,Point(350600,0)))

    # add edges
    three_node_graph.add_edge('-1','0', weight=1)
    three_node_graph.add_edge('0','-1', weight=1)
    three_node_graph.add_edge('0','1', weight=1)
    three_node_graph.add_edge('1','0', weight=1)
    three_node_graph.add_edge('1','+1', weight=1)
    three_node_graph.add_edge('+1','1', weight=1)
    return three_node_graph


@pytest.fixture
def five_node_graph(wgs84eqd_to_wgs84rad):
    graph = nx.DiGraph()

    # add nodes
    graph.add_node("-2", geometry=transform(wgs84eqd_to_wgs84rad, Point(-350600, 0)))
    graph.add_node("-1", geometry=transform(wgs84eqd_to_wgs84rad, Point(-15000, 0)))
    graph.add_node("0", geometry=transform(wgs84eqd_to_wgs84rad, Point(-5000, 0)))
    graph.add_node("1", geometry=transform(wgs84eqd_to_wgs84rad, Point(5000, 0)))
    graph.add_node("+1", geometry=transform(wgs84eqd_to_wgs84rad, Point(15000, 0)))
    graph.add_node("+2", geometry=transform(wgs84eqd_to_wgs84rad, Point(350600, 0)))

    # add edges
    graph.add_edge("-2", "-1", weight=1)
    graph.add_edge("-1", "-2", weight=1)

    graph.add_edge("-1", "0", weight=1)
    graph.add_edge("0", "-1", weight=1)

    graph.add_edge("0", "1", weight=1)
    graph.add_edge("1", "0", weight=1)

    graph.add_edge("1", "+1", weight=1)
    graph.add_edge("+1", "1", weight=1)

    graph.add_edge("+2", "+1", weight=1)
    graph.add_edge("+1", "+2", weight=1)
    return graph


def mission(env, vessel):
    """
    Method that defines the mission of the vessel.
    
    In this case: 
        keep moving along the path until its end point is reached
    """
    while True:
        yield from vessel.move()
        
        if vessel.geometry == nx.get_node_attributes(env.graph, "geometry")[vessel.route[-1]]:
            break


@pytest.fixture
def two_node_env(two_node_graph):# start simpy environment
    simulation_start = datetime.datetime(2025, 1, 1, 0, 0, 0)
    env = simpy.Environment(initial_time=simulation_start.timestamp())
    env.epoch = simulation_start
    # add graph to environment
    env.graph = two_node_graph
    # add components important for locking to the environment
    env.vessel_traffic_service = vessel_traffic_service_module.VesselTrafficService(graph=two_node_graph, crs_m="4087")

    return env

@pytest.fixture
def three_node_env(three_node_graph):# start simpy environment
    simulation_start = datetime.datetime(2025, 1, 1, 0, 0, 0)
    env = simpy.Environment(initial_time=simulation_start.timestamp())
    env.epoch = simulation_start
    # add graph to environment
    env.graph = three_node_graph
    # add components important for locking to the environment
    env.vessel_traffic_service = vessel_traffic_service_module.VesselTrafficService(graph=three_node_graph, crs_m="4087")
    return env


@pytest.fixture
def five_node_env(five_node_graph):  # start simpy environment
    # start simpy environment
    simulation_start = datetime.datetime(2025, 1, 1, 0, 0, 0)
    env = simpy.Environment(initial_time=simulation_start.timestamp())
    env.epoch = simulation_start

    # add graph to environment
    env.graph = five_node_graph

    # add components important for locking to the environment
    env.vessel_traffic_service = vessel_traffic_service_module.VesselTrafficService(graph=five_node_graph)
    return env


@pytest.fixture
def vessel_1(two_node_env, Vessel):
    data_vessel_1 = {
    "env": two_node_env,                                          # needed for simpy simulation
    "name": "Vessel 1",                                  # required by Identifiable
    "geometry": two_node_env.graph.nodes['0']['geometry'],        # required by Locatable
    "route": nx.dijkstra_path(two_node_env.graph, "0", "1"),      # required by Routeable
    "v": 4,                                              # required by Movable, 4 m/s to check if the distance is covered in the expected time
    "L": 100,                                            # required by VesselProperties, interacts with the lock capacity
    "B": 20,                                             # required by VesselProperties
    "T": 10,                                             # required by VesselProperties
    "type": 'tanker',                                    # required by VesselProperties
    "arrival_time": pd.Timestamp('2025-01-01 00:00:00')  # required by PassesLockComplex
    }  
    vessel_1 = Vessel(**data_vessel_1)
    vessel_1.name = 'Vessel 1'
    return vessel_1

@pytest.fixture
def vessel_2(two_node_env, Vessel):
    data_vessel_2 = {
        "env": two_node_env,                                          # needed for simpy simulation
        "name": "Vessel 2",                                  # required by Identifiable
        "geometry": two_node_env.graph.nodes['1']['geometry'],        # required by Locatable
        "route": nx.dijkstra_path(two_node_env.graph, "1", "0"),      # required by Routeable
        "v": 4,                                              # required by Movable, 4 m/s to check if the distance is covered in the expected time
        "L": 100,                                            # required by VesselProperties, interacts with the lock capacity
        "B": 20,                                             # required by VesselProperties
        "T": 10,                                             # required by VesselProperties
        "type": 'tanker',                                    # required by VesselProperties
        "arrival_time": pd.Timestamp('2025-01-01 00:05:00')  # required by PassesLockComplex
    }  
    vessel_2 = Vessel(**data_vessel_2)
    vessel_2.name = 'Vessel 2'
    return vessel_2


@pytest.fixture
def vessel_3(five_node_env, Vessel):
    # create vessels from dict
    data_vessel_1 = {
        "env": five_node_env,  # needed for simpy simulation
        "name": "Vessel 1",  # required by Identifiable
        "geometry": five_node_env.graph.nodes["-2"]["geometry"],  # required by Locatable
        "route": nx.dijkstra_path(five_node_env.graph, "-2", "+2"),  # required by Routeable
        "v": 4,  # required by Movable, 4 m/s to check if the distance is covered in the expected time
        "L": 100,  # required by VesselProperties, interacts with the lock capacity
        "B": 20,  # required by VesselProperties
        "T": 10,  # required by VesselProperties
        "type": "tanker",  # required by VesselProperties
        "arrival_time": pd.Timestamp("2025-01-01 00:00:00"),  # required by PassesLockComplex
    }
    vessel_1 = Vessel(**data_vessel_1)
    vessel_1.name = "Vessel 1"
    return vessel_1


@pytest.fixture
def vessel_4(five_node_env, Vessel):

    data_vessel_2 = {
        "env": five_node_env,  # needed for simpy simulation
        "name": "Vessel 2",  # required by Identifiable
        "geometry": five_node_env.graph.nodes["+2"]["geometry"],  # required by Locatable
        "route": nx.dijkstra_path(five_node_env.graph, "+2", "-2"),  # required by Routeable
        "v": 4,  # required by Movable, 4 m/s to check if the distance is covered in the expected time
        "L": 100,  # required by VesselProperties, interacts with the lock capacity
        "B": 20,  # required by VesselProperties
        "T": 10,  # required by VesselProperties
        "type": "tanker",  # required by VesselProperties
        "arrival_time": pd.Timestamp("2025-01-01 00:05:00"),  # required by PassesLockComplex
    }
    vessel_2 = Vessel(**data_vessel_2)
    vessel_2.name = "Vessel 2"
    return vessel_2


@pytest.fixture
def two_node_lock(two_node_env):
    lock = lock_module.IsLockComplex(
        env=two_node_env,
        name='Lock',
        node_open='0',
        node_A = '0',
        node_B = '1',
        distance_lock_doors_A_to_waiting_area_A = 4800,
        distance_lock_doors_B_to_waiting_area_B = 4800,
        distance_from_start_node_to_lock_doors_A = 4800,
        distance_from_end_node_to_lock_doors_B = 4800,
        lock_length = 400,
        lock_width = 50,
        lock_depth = 15,
        levelling_time = 300,
        sailing_distance_to_crossing_point = 1800,
        doors_opening_time= 300,
        doors_closing_time= 300,
        speed_reduction_factor_lock_chamber=0.5,
        sailing_in_time_gap_through_doors = 300,
        sailing_in_speed_sea = 1.5,
        sailing_in_speed_canal = 1.5,
        sailing_out_time_gap_through_doors = 120,
        sailing_time_before_opening_lock_doors = 600,
        sailing_time_before_closing_lock_doors = 120,
        registration_nodes = ['0','1'],
        predictive=False
    )
    return lock

@pytest.fixture
def three_node_lock(three_node_env):
    three_node_lock = lock_module.IsLockComplex(
        env=three_node_env,
        name='Lock',
        node_open='0',
        node_A = '0',
        node_B = '1',
        distance_lock_doors_A_to_waiting_area_A = 4800,
        distance_lock_doors_B_to_waiting_area_B = 4800,
        distance_from_start_node_to_lock_doors_A = 4800,
        distance_from_end_node_to_lock_doors_B = 4800,
        lock_length = 400,
        lock_width = 50,
        lock_depth = 15,
        levelling_time = 300,
        sailing_distance_to_crossing_point = 1800,
        doors_opening_time= 300,
        doors_closing_time= 300,
        speed_reduction_factor_lock_chamber=0.5,
        sailing_in_time_gap_through_doors = 300,
        sailing_in_speed_sea = 1.5,
        sailing_in_speed_canal = 1.5,
        sailing_out_time_gap_through_doors = 120,
        sailing_time_before_opening_lock_doors = 600,
        sailing_time_before_closing_lock_doors = 120,
        registration_nodes = ['-1','+1'],
    )
    return three_node_lock


@pytest.fixture
def five_node_lock_1(five_node_env):
    lock_1 = lock_module.IsLockComplex(
        env=five_node_env,
        name="Lock_1",
        node_open="-1",
        node_A="-1",
        node_B="0",
        distance_lock_doors_A_to_waiting_area_A=4800,
        distance_lock_doors_B_to_waiting_area_B=4800,
        distance_from_start_node_to_lock_doors_A=4800,
        distance_from_end_node_to_lock_doors_B=4800,
        lock_length=400,
        lock_width=50,
        lock_depth=15,
        levelling_time=300,
        sailing_distance_to_crossing_point=1800,
        doors_opening_time=300,
        doors_closing_time=300,
        speed_reduction_factor_lock_chamber=0.5,
        sailing_in_time_gap_through_doors=300,
        sailing_in_speed_sea=1.5,
        sailing_in_speed_canal=1.5,
        sailing_out_time_gap_through_doors=120,
        sailing_time_before_opening_lock_doors=600,
        sailing_time_before_closing_lock_doors=120,
        registration_nodes=["-2", "1"],
        predictive=False,
    )
    return lock_1


@pytest.fixture
def five_node_lock_2(five_node_env):
    lock_2 = lock_module.IsLockComplex(
        env=five_node_env,
        name="Lock_2",
        node_open="1",
        node_A="1",
        node_B="+1",
        distance_lock_doors_A_to_waiting_area_A=4800,
        distance_lock_doors_B_to_waiting_area_B=4800,
        distance_from_start_node_to_lock_doors_A=4800,
        distance_from_end_node_to_lock_doors_B=4800,
        lock_length=400,
        lock_width=50,
        lock_depth=15,
        levelling_time=300,
        sailing_distance_to_crossing_point=1800,
        doors_opening_time=300,
        doors_closing_time=300,
        speed_reduction_factor_lock_chamber=0.5,
        sailing_in_time_gap_through_doors=300,
        sailing_in_speed_sea=1.5,
        sailing_in_speed_canal=1.5,
        sailing_out_time_gap_through_doors=120,
        sailing_time_before_opening_lock_doors=600,
        sailing_time_before_closing_lock_doors=120,
        registration_nodes=["0", "+2"],
        predictive=False,
    )
    return lock_2


def test_notebook_0201(vessel_1, vessel_2, two_node_env, two_node_lock):

    # start the simulation
    two_node_env = two_node_env
    two_node_env.process(mission(two_node_env, vessel_1))
    two_node_env.process(mission(two_node_env, vessel_2))
    two_node_env.run()

    # check vessel planning and operation planning shapes
    assert two_node_lock.vessel_planning.shape == (2,22), "The lock planning shape does not have the expected dimensions"
    assert two_node_lock.operation_planning.shape == (2,26), "The lock operation planning shape does not have the expected dimensions"

    #check logbook shapes
    df_vessel_1 = pd.DataFrame.from_dict(vessel_1.logbook)
    df_vessel_2 = pd.DataFrame.from_dict(vessel_2.logbook)
    lock_df = pd.DataFrame.from_dict(two_node_lock.lock_chamber.logbook)
    assert df_vessel_1.shape == (12,4), "The vessel 1 logbook does not have the expected dimensions"
    assert df_vessel_2.shape == (14,4), "The vessel 2 logbook does not have the expected dimensions"
    assert lock_df.shape == (12,4), "The lock logbook does not have the expected dimensions"

    #check if waiting time is as expected
    vessel_2_stop_waiting= df_vessel_2[df_vessel_2.Message == 'Waiting for lock operation stop']['Timestamp'].iloc[0]
    assert abs(vessel_2_stop_waiting - pd.Timestamp("2025-01-01 00:36:28")) <= pd.Timedelta(minutes=1), f"The vessel 2 stop waiting time is not as expected, namely {vessel_2_stop_waiting}"

    # check if levelling times of lock and vessels correspond
    vessel_2_levelling_start= df_vessel_2[df_vessel_2.Message == 'Levelling start']['Timestamp'].iloc[0]
    vessel_2_levelling_stop = df_vessel_2[df_vessel_2.Message == 'Levelling stop']['Timestamp'].iloc[0]
    vessel_1_levelling_start= df_vessel_1[df_vessel_1.Message == 'Levelling start']['Timestamp'].iloc[0]
    vessel_1_levelling_stop = df_vessel_1[df_vessel_1.Message == 'Levelling stop']['Timestamp'].iloc[0]
    lock_levelling_start_v1 = lock_df[lock_df.Message == "Lock chamber converting start"]['Timestamp'].iloc[0]
    lock_levelling_stop_v1 = lock_df[lock_df.Message == "Lock chamber converting stop"]['Timestamp'].iloc[0]
    lock_levelling_start_v2 = lock_df[lock_df.Message == "Lock chamber converting start"]['Timestamp'].iloc[1]
    lock_levelling_stop_v2 = lock_df[lock_df.Message == "Lock chamber converting stop"]['Timestamp'].iloc[1]
    assert vessel_1_levelling_start == lock_levelling_start_v1, "The vessel 1 levelling start time does not correspond to the lock levelling start time"
    assert vessel_1_levelling_stop == lock_levelling_stop_v1, "The vessel 1 levelling stop time does not correspond to the lock levelling stop time"
    assert vessel_2_levelling_start == lock_levelling_start_v2, "The vessel 2 levelling start time does not correspond to the lock levelling start time"
    assert vessel_2_levelling_stop == lock_levelling_stop_v2, "The vessel 2 levelling stop time does not correspond to the lock levelling stop time"

    # generate gantt chart
    df_eventtable = opentnsim.core.logutils.logbook2eventtable([vessel_1, vessel_2, two_node_lock.lock_chamber])
    generate_vessel_gantt_chart(df_eventtable)

def test_notebook_0202(Vessel, three_node_env, three_node_lock):

    start_time = pd.Timestamp('2025-01-01 00:00:00')

    # Example using numpy
    # Create independent random generators
    rng_up = np.random.default_rng(123)
    rng_down = np.random.default_rng(456)

    # Exponential distributions with different means (scale = mean)
    arrival_dist_up = lambda: rng_up.exponential(scale=5)   # mean 30 minutes
    arrival_dist_down = lambda: rng_down.exponential(scale=5)  # mean 15 minutes

    vessels = generate_vessels_with_distributions(
        env=three_node_env,
        num_vessels=11, 
        start_time=start_time,
        arrival_dist_up=arrival_dist_up,
        arrival_dist_down=arrival_dist_down,
        seed_up=123,
        seed_down=456,
        vessel_class=Vessel
    )

    for vessel in vessels:
        three_node_env.process(mission(three_node_env, vessel))

    three_node_env.run()

    # check vessel planning and operation planning shapes
    assert three_node_lock.vessel_planning.shape == (11,22), "The lock planning shape does not have the expected dimensions"
    assert three_node_lock.operation_planning.shape == (5,26), "The lock operation planning shape does not have the expected dimensions"

    # check logbook shapes
    for vessel in vessels:
        df_vessel = pd.DataFrame.from_dict(vessel.logbook)
        assert df_vessel.shape[0] in [
            18,
            20,
        ], f"The vessel {vessel.name} logbook does not have the expected number of entries, namely {df_vessel.shape[0]}"
        assert df_vessel.shape[1]==4, f"The vessel {vessel.name} logbook does not have the expected number of columns, namely {df_vessel.shape[1]}"

    # check if we can create gantt chart
    df_eventtable = opentnsim.core.logutils.logbook2eventtable([*vessels, three_node_lock.lock_chamber])
    fig = generate_vessel_gantt_chart(df_eventtable)

    # check if we can create time distance plot
    # We can plot the time-distance diagram
    fig = three_node_lock.create_time_distance_plot(vessels = vessels, 
                                     xlimmin = -6050, 
                                     xlimmax = 6050,
                                     ylimmin = pd.Timestamp('2025-01-01 22:00:00'),
                                     ylimmax = pd.Timestamp('2025-01-02 09:00:00'),
                                     method='Plotly')


def test_notebook_0203(Vessel, three_node_env):

    # generate synthetic hydrodynamic data
    hydrodynamic_data = xr.Dataset()

    simulation_start = three_node_env.epoch
    time = pd.date_range(start=simulation_start, end=simulation_start + pd.Timedelta(days=7), freq=pd.Timedelta(minutes=5))

    static_water_level = np.zeros(len(time))
    tidal_amplitude = 1.0
    tidal_period = 12.5  # hours
    tidal_water_level = [
        tidal_amplitude * np.sin(2 * np.pi * (t - simulation_start).total_seconds() / (tidal_period * 3600)) for t in time
    ]

    stations = ["0", "1"]
    water_level_data = xr.DataArray(data=[static_water_level, tidal_water_level], coords={"STATION": stations, "TIME": time})

    hydrodynamic_data["Water level"] = water_level_data

    # add hydrodynamic data to the environment
    # add components important for locking to the environment
    three_node_env.vessel_traffic_service = vessel_traffic_service_module.VesselTrafficService(
        graph=three_node_env.graph, crs_m="4087", hydrodynamic_information=hydrodynamic_data
    )
    three_node_lock = lock_module.IsLockComplex(
        env=three_node_env,
        name="Lock",
        node_open="0",
        node_A="0",
        node_B="1",
        distance_lock_doors_A_to_waiting_area_A=4800,
        distance_lock_doors_B_to_waiting_area_B=4800,
        distance_from_start_node_to_lock_doors_A=4800,
        distance_from_end_node_to_lock_doors_B=4800,
        lock_length=400,
        lock_width=50,
        lock_depth=15,
        levelling_time=300,
        sailing_distance_to_crossing_point=1800,
        doors_opening_time=300,
        doors_closing_time=300,
        speed_reduction_factor_lock_chamber=0.5,
        sailing_in_time_gap_through_doors=300,
        sailing_in_speed_sea=1.5,
        sailing_in_speed_canal=1.5,
        sailing_out_time_gap_through_doors=120,
        sailing_time_before_opening_lock_doors=600,
        sailing_time_before_closing_lock_doors=120,
        registration_nodes=["-1", "+1"],
    )
    # create vessels
    rng_up = np.random.default_rng(123)
    rng_down = np.random.default_rng(456)

    # Exponential distributions with different means (scale = mean)
    arrival_dist_up = lambda: rng_up.exponential(scale=30)  # mean 30 minutes
    arrival_dist_down = lambda: rng_down.exponential(scale=15)  # mean 15 minutes

    vessels = generate_vessels_with_distributions(
        env=three_node_env,
        num_vessels=9,
        start_time=simulation_start,
        arrival_dist_up=arrival_dist_up,
        arrival_dist_down=arrival_dist_down,
        seed_up=123,
        seed_down=456,
        vessel_class=Vessel,
    )

    # simulate missions
    for vessel in vessels:
        three_node_env.process(mission(three_node_env, vessel))
    three_node_env.run()

    # check vessel planning and operation planning shapes
    assert three_node_lock.vessel_planning.shape == (9, 22), "The lock planning shape does not have the expected dimensions"
    assert three_node_lock.operation_planning.shape == (
        5,
        26,
    ), "The lock operation planning shape does not have the expected dimensions"

    # check logbook shapes
    for vessel in vessels:
        df_vessel = pd.DataFrame.from_dict(vessel.logbook)
        assert df_vessel.shape[0] in (
            16,
            18,
            20,
        ), f"The vessel {vessel.name} logbook does not have the expected number of entries, namely {df_vessel.shape[0]}"
        assert (
            df_vessel.shape[1] == 4
        ), f"The vessel {vessel.name} logbook does not have the expected number of columns, namely {df_vessel.shape[1]}"

    # check if hydrodynamic data is in half of the time similar to lock water level
    # leveling starts 01-02 om 02 uur en stops at 01-02 om 4 uur. Selecteer indexes tijdens en na levelling
    indexes_during_levelling = np.where(
        (hydrodynamic_data.TIME >= pd.Timestamp("2025-01-02 02:30")) & (hydrodynamic_data.TIME <= pd.Timestamp("2025-01-02 03:30"))
    )
    similar_during_levelling = (
        hydrodynamic_data.sel({"STATION": "1"})["Water level"][indexes_during_levelling]
        == three_node_lock.lock_chamber.water_level[indexes_during_levelling]
    )

    indexes_after_levelling = np.where(hydrodynamic_data.TIME > pd.Timestamp("2025-01-02 04:00"))
    similar_after_levelling = (
        hydrodynamic_data.sel({"STATION": "1"})["Water level"][indexes_after_levelling]
        == three_node_lock.lock_chamber.water_level[indexes_after_levelling]
    )

    assert (
        similar_during_levelling.sum() / len(similar_during_levelling) < 0.5
    ), "During levelling, the hydrodynamic water level corresponds too much to the lock water level"
    assert (
        similar_after_levelling.all()
    ), "After levelling, the hydrodynamic water level does not correspond to the lock water level"


def test_notebook_0204(five_node_env, vessel_3, vessel_4, five_node_lock_1, five_node_lock_2):
    # start the simulation
    five_node_env.process(mission(five_node_env, vessel_3))
    five_node_env.process(mission(five_node_env, vessel_4))

    five_node_env.run()

    # check logbook shapes
    df_vessel_3 = pd.DataFrame.from_dict(vessel_3.logbook)
    df_vessel_4 = pd.DataFrame.from_dict(vessel_4.logbook)
    assert df_vessel_3.shape == (30, 4), "The vessel 3 logbook does not have the expected dimensions"
    assert df_vessel_4.shape == (30, 4), "The vessel 4 logbook does not have the expected dimensions"

    df_lock_1 = pd.DataFrame.from_dict(five_node_lock_1.lock_chamber.logbook)
    df_lock_2 = pd.DataFrame.from_dict(five_node_lock_2.lock_chamber.logbook)
    assert df_lock_1.shape == (12, 4), "The lock 1 logbook does not have the expected dimensions"
    assert df_lock_2.shape == (18, 4), "The lock 2 logbook does not have the expected dimensions"

    # check if we can create gantt chart
    df_eventtable = opentnsim.core.logutils.logbook2eventtable(
        [vessel_3, vessel_4, five_node_lock_1.lock_chamber, five_node_lock_2.lock_chamber]
    )
    fig = generate_vessel_gantt_chart(df_eventtable)
