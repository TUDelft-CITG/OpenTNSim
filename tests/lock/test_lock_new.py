"""This file contains tests for the lock_new module."""

import pytest
import networkx as nx
from shapely.geometry import Point, LineString, Polygon

import simpy
import xarray as xr

from opentnsim.lock import lock_new as lock_module
from opentnsim.core import vessel_properties as vessel_module
from opentnsim import vessel_traffic_service as vessel_traffic_service_module
import datetime as dt
import pyproj
from shapely.ops import transform

from opentnsim.lock.lock_new import (
    PassesLockComplex,
    IsLockChamber,
    IsLockMaster,
    IsLockComplex,
    IsLockWaitingArea,
    # IsLockLineUpArea, nog niet af...
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
    graph.add_edge("A", "B", geometry=transform(wgs84eqd_to_wgs84rad, LineString([Point(-5000, 0), Point(5000, 0)])), length=10000)
    graph.add_edge("B", "A", geometry=transform(wgs84eqd_to_wgs84rad, LineString([Point(5000, 0), Point(-5000, 0)])), length=10000)
    return graph


@pytest.mark.skip(reason="module must be edited")
def test_islockwaitingarea_init(env, graph):
    env.graph = graph
    env.vessel_traffic_service = vessel_traffic_service_module.VesselTrafficService(graph=graph)
    waiting_area = IsLockWaitingArea(name="queue", edge=("A", "B", 0), lock=None, distance_from_node=100, env=env)
    assert waiting_area.name == "queue"
    assert waiting_area.node == "A"
    assert waiting_area.waiting_area.capacity == 1000000  # een van beide moet weg denk ik.
    assert waiting_area.resource.capacity == 1000000  # een van beide moet weg denk ik.


@pytest.mark.skip(reason="Does not exist yet")
def test_islocklineuparea_init(env, graph):
    env.graph = graph
    env.vessel_traffic_service = vessel_traffic_service_module.VesselTrafficService(graph=graph)
    lineup_area = IsLockLineUpArea(
        name="lineup", start_node="A", end_node="B", lineup_area_length=100, distance_from_start_edge=10, env=env
    )
    assert lineup_area.name == "lineup"
    assert lineup_area.node == "A"


@pytest.mark.skip(reason="make independent of IsLockMaster")
def test_islockchamber_init(env, graph):
    env.graph = graph
    env.vessel_traffic_service = vessel_traffic_service_module.VesselTrafficService(graph=graph)

    chamber = IsLockChamber(
        name="test chamber", start_node="A", end_node="B", env=env, lock_length=200, lock_width=20, lock_depth=5
    )
    assert chamber.name == "test chamber"


def test_islockcomplex_init(env, graph):
    """Test the initialization of IsLockComplex."""
    env.graph = graph
    env.vessel_traffic_service = vessel_traffic_service_module.VesselTrafficService(graph=graph)

    lock = IsLockComplex(node_A="A", node_B="B", name="test lock", env=env, lock_length=200, lock_width=20, lock_depth=5)

    # check attributes
    assert isinstance(lock.waiting_area_A, IsLockWaitingArea)
    assert isinstance(lock.waiting_area_B, IsLockWaitingArea)


def test_islockmaster_init(env, graph):
    env.graph = graph
    env.vessel_traffic_service = vessel_traffic_service_module.VesselTrafficService(graph=graph)

    lock_complex = IsLockComplex(node_A="A", node_B="B", name="test lock", env=env, lock_length=200, lock_width=20, lock_depth=5)
    master = IsLockMaster(lock_complex=lock_complex, env=env)
    assert master.lock_complex == lock_complex


def test_islockcomplex_invalid_nodes(env, graph):
    """Test that IsLockComplex raises ValueError for invalid nodes."""
    env.graph = graph
    env.vessel_traffic_service = vessel_traffic_service_module.VesselTrafficService(graph=graph)

    with pytest.raises(
        ValueError, match="Lock chamber test lock has invalid node_A X or node_B B which are not part of the graph."
    ):
        lock = IsLockComplex(node_A="X", node_B="B", name="test lock", env=env, lock_length=200, lock_width=20, lock_depth=5)

    with pytest.raises(
        ValueError, match="Lock chamber test lock has invalid node_A A or node_B Y which are not part of the graph."
    ):
        lock = IsLockComplex(node_A="A", node_B="Y", name="test lock", env=env, lock_length=200, lock_width=20, lock_depth=5)

    with pytest.raises(ValueError, match="does not have an edge"):
        lock = IsLockComplex(node_A="A", node_B="A", name="test lock", env=env, lock_length=200, lock_width=20, lock_depth=5)


def test_passeslockcomplex_init(env, graph):
    """Test the initialization of PassesLockComplex."""
    env.graph = graph
    vessel = PassesLockComplex(v=4, geometry=graph.nodes["A"]["geometry"], route=["A", "B"], env=env)
    assert len(vessel.on_pass_node_functions) == 1
    assert len(vessel.on_pass_edge_functions) == 1


def test_find_upcoming_locks(env, graph):
    env.graph = graph
    vessel = PassesLockComplex(v=4, geometry=graph.nodes["A"]["geometry"], route=["A", "B"], env=env)
    vessel.position_on_route = 0

    assert vessel.route_ahead == ["A", "B"]
