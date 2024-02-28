import datetime
import pytest
import shapely
import simpy
import numpy as np
import xarray as xr

import opentnsim.core as core
import opentnsim.output as output
import opentnsim.vessel_traffic_service

import opentnsim.lock
import networkx as nx


@pytest.fixture
def graph():
    FG = nx.DiGraph()
    Node = type("Site", (core.Identifiable, core.Log, core.Locatable, core.HasResource), {})
    coordinates = [(0, 0), (1200, 0)]
    for index, coord in enumerate(coordinates):
        x, y = coord
        geometry = shapely.geometry.Point(x, y)
        node = Node(**{"env": [], "name": index, "geometry": geometry})
        FG.add_node(node.name, geometry=node.geometry, name=index, Info={"geometry": geometry})

    edges = [(FG.nodes[0], FG.nodes[1]), (FG.nodes[1], FG.nodes[0])]
    for node_start, node_stop in edges:
        geometry = shapely.LineString([node_start["geometry"], node_stop["geometry"]])

        FG.add_edge(
            node_start["name"],
            node_stop["name"],
            weight=1,
            geometry=geometry,
            Info={"geometry": geometry, "length": geometry.length},
        )

    # run for a day
    return FG


@pytest.fixture
def env(graph):
    t_start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    t_stop = datetime.datetime(2024, 2, 1, 0, 0, 0)
    env = simpy.Environment(initial_time=t_start.timestamp())

    env.epoch = t_start
    env.simulation_start = t_start
    env.simulation_stop = t_stop
    # TODO: switch to env.graph
    env.FG = graph
    # run for a day
    return env


@pytest.fixture
def hydrodynamics_env(env):
    t_end = env.simulation_stop
    stations = list(env.FG.nodes)
    times = np.arange(env.epoch, t_end, datetime.timedelta(seconds=600))
    water_depth = [np.linspace(10, 10, len(times)), np.linspace(10, 10, len(times))]
    water_level = [np.linspace(0, 0, len(times)), np.linspace(1, 1, len(times))]
    salinity = [np.linspace(0, 0, len(times)), np.linspace(0, 0, len(times))]
    station_data = xr.DataArray(data=stations, dims=["STATIONS"])
    time_data = xr.DataArray(data=times, dims=["TIME"])
    depth_data = xr.DataArray(data=water_depth, dims=["STATIONS", "TIME"])
    water_level_data = xr.DataArray(data=[wlev for wlev in water_level], dims=["STATIONS", "TIME"])
    salinity_data = xr.DataArray(data=[sal for sal in salinity], dims=["STATIONS", "TIME"])
    hydrodynamic_data = xr.Dataset(
        {"TIME": time_data, "Stations": station_data, "MBL": depth_data, "Water level": water_level_data, "Salinity": salinity_data}
    )
    env.vessel_traffic_service = opentnsim.vessel_traffic_service.VesselTrafficService(hydrodynamic_data)
    return env


def create_vessel(env, name, origin, destination, vessel_type, L, B, T, v, arrival_time):
    Vessel = type(
        "Vessel",
        (
            opentnsim.lock.HasLock,
            opentnsim.lock.HasLineUpArea,
            opentnsim.lock.HasWaitingArea,
            core.Movable,
            core.VesselProperties,
            output.HasOutput,
            opentnsim.lock.CustomLog,
            core.Identifiable,
            core.SimpyObject,
            core.ExtraMetadata,
        ),
        {},
    )

    node = env.FG.nodes[origin]
    geometry = node["geometry"]
    vessel = Vessel(
        **{
            "env": env,
            "name": name,
            "origin": origin,
            "destination": destination,
            "geometry": geometry,
            "node": origin,
            "route": nx.dijkstra_path(env.FG, origin, destination),
            "type": vessel_type,
            "L": L,
            "B": B,
            "T": T,
            "v": v,
            "arrival_time": arrival_time,
        }
    )

    env.process(vessel.move())
    return vessel


@pytest.fixture
def lock_env(hydrodynamics_env):
    env = hydrodynamics_env
    lock = opentnsim.lock.IsLock(
        env=env,
        name="Lock",
        lock_length=400,
        lock_width=40,
        lock_depth=10,
        node_doors1=0,
        node_doors2=1,
        distance_doors1_from_first_waiting_area=400,
        distance_doors2_from_second_waiting_area=400,
        speed_reduction_factor=1,
        detector_nodes=[0, 1],
    )
    opentnsim.lock.IsLockLineUpArea(
        env=env, name="Lock", distance_to_lock_doors=0, start_node=0, end_node=1, lineup_length=400, speed_reduction_factor=1
    )
    opentnsim.lock.IsLockLineUpArea(
        env=env, name="Lock", distance_to_lock_doors=0, start_node=1, end_node=0, lineup_length=400, speed_reduction_factor=1
    )
    opentnsim.lock.IsLockWaitingArea(env=env, name="Lock", distance_from_node=0, node=0)
    opentnsim.lock.IsLockWaitingArea(env=env, name="Lock", distance_from_node=0, node=1)

    env.lock = lock
    return env


def test_is_lock(lock_env):
    env = lock_env
    lock = env.lock
    assert hasattr(lock, "next_lockage_length"), "lock should have next_lockage_length"


def test_sail_through_lock(lock_env):
    env = lock_env

    vessel = create_vessel(env, 0, 0, 1, None, 200, 30, 8, v=4, arrival_time=datetime.datetime(2024, 1, 1, 0, 0, 0))

    env.run()

    assert len(vessel.logbook) > 2
