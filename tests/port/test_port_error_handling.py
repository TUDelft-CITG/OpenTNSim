"""Tests of the situations that were handled by a bare `except:` in the port module (issue #153)."""

import types

import networkx as nx
import pandas as pd
import pytest

from opentnsim.port import utils
from opentnsim.port.utils import determine_nearest_anchorage_area


def _anchorage(node):
    return types.SimpleNamespace(node=node, resource=types.SimpleNamespace(capacity=2, users=[]))


@pytest.fixture
def scenario(monkeypatch):
    """A vessel at node 'origin' with a route to 'destination' and two anchorage areas; 'far' is not reachable."""
    port = types.SimpleNamespace(anchorage_areas={"far": _anchorage("far"), "near": _anchorage("near")})
    vessel = types.SimpleNamespace(name="vessel", env=types.SimpleNamespace(graph=nx.DiGraph()), route=["origin", "destination"])

    def get_shortest_route(graph, origin, destination, vessel=None):
        if "far" in (origin, destination):
            raise nx.NetworkXNoPath(f"No path between {origin} and {destination}.")
        return [origin, destination]

    monkeypatch.setattr(utils, "get_shortest_route", get_shortest_route)
    monkeypatch.setattr(utils, "node_path_to_edge_path", lambda graph, route: list(zip(route[:-1], route[1:])))
    monkeypatch.setattr(utils, "get_sailing_time", lambda vessel, edge_route: (600.0, None))
    return vessel, port


def test_unreachable_anchorage_areas_are_skipped(scenario):
    vessel, port = scenario
    anchorage, info = determine_nearest_anchorage_area(vessel, "origin", port)
    assert anchorage is port.anchorage_areas["near"]
    assert info.total_sailing_time == pd.Timedelta(seconds=1200)


def test_no_reachable_anchorage_area(scenario):
    vessel, port = scenario
    del port.anchorage_areas["near"]
    with pytest.raises(ValueError, match="no available anchorage areas"):
        determine_nearest_anchorage_area(vessel, "origin", port)


def test_unexpected_errors_in_the_route_calculation_are_not_hidden(scenario, monkeypatch):
    vessel, port = scenario

    def broken(*args, **kwargs):
        raise RuntimeError("bug in the route calculation")

    monkeypatch.setattr(utils, "get_shortest_route", broken)
    with pytest.raises(RuntimeError):
        determine_nearest_anchorage_area(vessel, "origin", port)
