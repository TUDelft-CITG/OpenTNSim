"""This file contains tests for the lock module."""

import pytest
import networkx as nx
from shapely.geometry import Point, LineString

import simpy
import datetime as dt
import pyproj
from shapely.ops import transform

from opentnsim.lock import (
    LockComplexTraversable,
    IsLockChamber,
    IsLockMaster,
    IsLockComplex,
    IsLockWaitingArea,
)


@pytest.fixture
def env():
    tstart = dt.datetime(2025, 1, 1, 0, 0, 0)
    env = simpy.Environment(initial_time=tstart.timestamp())
    env.simulation_start = tstart
    env.simulation_stop = dt.datetime(2026, 1, 1, 0, 0, 0)
    env.epoch = tstart
    return env


@pytest.fixture
def graph():
    wgs84eqd = pyproj.CRS("4087")
    wgs84rad = pyproj.CRS("4326")
    wgs84eqd_to_wgs84rad = pyproj.transformer.Transformer.from_crs(wgs84eqd, wgs84rad, always_xy=True).transform

    graph = nx.DiGraph()
    graph.add_node("A", geometry=transform(wgs84eqd_to_wgs84rad, Point(-5000, 0)))
    graph.add_node("B", geometry=transform(wgs84eqd_to_wgs84rad, Point(5000, 0)))
    graph.add_edge("A", "B", geometry=transform(wgs84eqd_to_wgs84rad, LineString([Point(-5000, 0), Point(5000, 0)])), length_m=10000)
    graph.add_edge("B", "A", geometry=transform(wgs84eqd_to_wgs84rad, LineString([Point(5000, 0), Point(-5000, 0)])), length_m=10000)
    return graph


def test_islockwaitingarea_init(env, graph):
    env.graph = graph
    waiting_area = IsLockWaitingArea(name="test lock waiting area A", edge=("A", "B"), orientation=0, distance_from_edge_start=100, env=env, capacity=1000000)
    assert waiting_area.name == "test lock waiting area A"
    assert waiting_area.edge == ("A", "B")
    assert waiting_area.orientation == 0
    assert waiting_area.distance_from_edge_start == 100
    assert waiting_area.resource.capacity == 1000000


@pytest.mark.skip(reason="Does not exist yet")
def test_islocklineuparea_init(env, graph):
    env.graph = graph
    lineup_area = IsLockLineUpArea(
        name="lineup", start_node="A", end_node="B", lineup_area_length=100, distance_from_start_edge=10, env=env
    )
    assert lineup_area.name == "lineup"


def test_islockchamber_init(env, graph):
    env.graph = graph

    lock_chamber = IsLockChamber(name="test lock chamber", 
                                 edge=("A", "B"), 
                                 env=env, 
                                 lock_length=200, 
                                 lock_width=20, 
                                 lock_depth=5,
                                 distance_from_start_node_to_lock_gate_A=4900,
                                 distance_from_end_node_to_lock_gate_B=4900)
    
    assert lock_chamber.name == "test lock chamber"


def test_islockcomplex_init(env, graph):
    """Test the initialization of IsLockComplex."""
    env.graph = graph

    lock_chamber = IsLockChamber(name="test lock chamber", 
                                 edge=("A", "B"), 
                                 env=env, 
                                 lock_length=200, 
                                 lock_width=20, 
                                 lock_depth=5,
                                 distance_from_start_node_to_lock_gate_A=4900,
                                 distance_from_end_node_to_lock_gate_B=4900)
    waiting_area_A = IsLockWaitingArea(name="test lock waiting area A", edge=("A", "B"), orientation=0, distance_from_edge_start=100, env=env)
    waiting_area_B = IsLockWaitingArea(name="test lock waiting area B", edge=("B", "A"), orientation=1, distance_from_edge_start=100, env=env)
    lock = IsLockComplex(lock_chambers=[lock_chamber], 
                         waiting_areas=[waiting_area_A, waiting_area_B], 
                         name="test lock complex", 
                         env=env,
                         registration_nodes=["A", "B"])

    # check attributes
    assert isinstance(lock.waiting_areas["test lock waiting area A"], IsLockWaitingArea)
    assert isinstance(lock.waiting_areas["test lock waiting area B"], IsLockWaitingArea)


def test_islockmaster_init(env, graph):
    env.graph = graph

    lock_chamber = IsLockChamber(name="test lock chamber", 
                                 edge=("A", "B"), 
                                 env=env, 
                                 lock_length=200, 
                                 lock_width=20, 
                                 lock_depth=5,
                                 distance_from_start_node_to_lock_gate_A=4900,
                                 distance_from_end_node_to_lock_gate_B=4900)
    waiting_area_A = IsLockWaitingArea(name="test lock waiting area A", edge=("A", "B"), orientation=0, distance_from_edge_start=100, env=env)
    waiting_area_B = IsLockWaitingArea(name="test lock waiting area B", edge=("B", "A"), orientation=1, distance_from_edge_start=100, env=env)
    lock_complex = IsLockComplex(lock_chambers=[lock_chamber], 
                         waiting_areas=[waiting_area_A, waiting_area_B], 
                         name="test lock complex", 
                         env=env,
                         registration_nodes=["A", "B"])
    
    master = IsLockMaster(lock_complex=lock_complex)
    assert master.lock_complex == lock_complex


def test_passeslockcomplex_init(env, graph):
    """Test the initialization of LockComplexTraversable."""
    env.graph = graph
    vessel = LockComplexTraversable(v=4, geometry=graph.nodes["A"]["geometry"], route=["A", "B"], env=env, name = 'test vessel', B = 10, L = 100, T = 2, type="barge")
    assert len(vessel.on_pass_node_functions) == 1


def test_find_upcoming_locks(env, graph):
    env.graph = graph
    vessel = LockComplexTraversable(v=4, geometry=graph.nodes["A"]["geometry"], route=["A", "B"], env=env, name = 'test vessel', B = 10, L = 100, T = 2, type="barge")
    vessel.position_on_route = 0

    assert vessel.route_ahead == ["A", "B"]
