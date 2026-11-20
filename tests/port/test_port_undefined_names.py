"""Regression tests for names that were used in the port module without being defined or imported (issue #145)."""

import types

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from opentnsim.port import visualizations
from opentnsim.port.calculations import calculate_inerpolated_water_levels_over_network
from opentnsim.port.mixins.rules import RuleEngine, TrafficRules
from opentnsim.port.utils import render_rule_expression


# %% calculate_inerpolated_water_levels_over_network (used pyproj without importing it)
@pytest.fixture
def water_levels():
    """Water levels at two stations, 0 m and 1 m at all times."""
    time = pd.date_range("2025-01-01", periods=4, freq="h")
    data = xr.Dataset(
        {"Water level": (("TIME", "STATION"), np.array([[0.0, 1.0]] * 4))},
        coords={"TIME": time, "STATION": ["west", "east"]},
    )
    # the geometry of the measurement stations is stored in the attributes (in the order of the stations)
    data.attrs = {"west": Point(4.0, 52.0), "east": Point(4.2, 52.0)}
    return data


@pytest.fixture
def network():
    graph = nx.Graph()
    graph.add_node("near_west", geometry=Point(4.01, 52.0))
    graph.add_node("near_east", geometry=Point(4.19, 52.0))
    return graph


def test_interpolated_water_levels_nearest(network, water_levels):
    result = calculate_inerpolated_water_levels_over_network(network, water_levels, method="nearest")
    levels = result["Water level"]
    assert levels.dims == ("TIME", "STATION")
    # each node gets the water level of its nearest station
    assert list(levels.sel(TIME=levels.TIME[0]).values) == [0.0, 1.0]


def test_interpolated_water_levels_weighted(network, water_levels):
    result = calculate_inerpolated_water_levels_over_network(network, water_levels, method="weighted")
    levels = result["Water level"].sel(STATION="near_west").values
    # close to the west station (0 m), so far below the 0.5 m average of the two stations
    assert (levels < 0.1).all()
    levels = result["Water level"].sel(STATION="near_east").values
    assert (levels > 0.9).all()


# %% RuleEngine.overview (used a parser that no longer exists)
def condition_long_vessel(vessel):
    return vessel.length > 100 and vessel.draught > 5


def policy_closed(vessel):
    return 1


def test_rule_overview_shows_name_condition_and_policy(capsys):
    engine = RuleEngine(default=TrafficRules.allowed)
    engine.add_rule(condition_long_vessel, policy_closed, "long_and_deep_vessels")
    engine.overview()
    output = capsys.readouterr().out
    assert "long_and_deep_vessels" in output
    assert "vessel.length > 100 and vessel.draught > 5" in output
    assert "policy: 1" in output


def test_rule_overview_without_rules(capsys):
    RuleEngine(default=TrafficRules.allowed).overview()
    assert capsys.readouterr().out.strip() == ""


def test_render_rule_expression_falls_back_to_name_without_source():
    # the source of a builtin cannot be retrieved
    assert render_rule_expression(len) == "len"


# %% create_plot_horizontal_tidal_window (used an undefined `delay` and shifted the hydrodynamic time axis in place)
def test_horizontal_tidal_window_plot(monkeypatch):
    time = pd.date_range("2025-01-01", periods=200, freq="h").values
    hydrodynamic_data = xr.Dataset(coords={"TIME": time})
    time_before = hydrodynamic_data.TIME.values.copy()
    monkeypatch.setattr(
        visualizations, "HydrodynamicDataManager", lambda: types.SimpleNamespace(hydrodynamic_data=hydrodynamic_data)
    )

    # a route over two edges with a restriction at the last node
    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B", length_m=1000.0)
    graph.add_edge("B", "C", length_m=1000.0)

    sailing_calls = []

    def fake_get_sailing_time(vessel, edge_route):
        sailing_calls.append(list(edge_route))
        return 1800.0, None

    monkeypatch.setattr(visualizations, "get_sailing_time", fake_get_sailing_time)
    monkeypatch.setattr(visualizations, "get_governing_current_velocity", lambda vessel, station, start, stop: (np.zeros(stop - start), None))

    window = (pd.Timestamp("2025-01-02 00:00"), pd.Timestamp("2025-01-02 03:00"))
    results = pd.DataFrame(
        {
            "time_start_index": [0],
            "time_end_index": [100],
            "route": [["A", "B", "C"]],
            "bound": ["inbound"],
            "draught": [5.0],
            "horizontal_tidal_windows": [[window]],
            "horizontal_tidal_restriction_nodes": [["C"]],
            "horizontal_tidal_restriction_stations": [["station"]],
        },
        index=["waterway"],
    )
    vessel = types.SimpleNamespace(
        env=types.SimpleNamespace(graph=graph),
        trips=pd.DataFrame({"trip_id": ["trip"]}),
        tidal_window_calculations={"trip": results},
    )

    fig, handles, labels = visualizations.create_plot_horizontal_tidal_window(vessel)

    assert labels == ["Current velocity", "Horizontal tidal windows"]
    # the sailing time is determined over the edges up to the restriction node (like in the tidal window calculation)
    assert sailing_calls == [[("A", "B", 0), ("B", "C", 0)]]
    # plotting must not shift the hydrodynamic data
    assert np.array_equal(hydrodynamic_data.TIME.values, time_before)


# %% provide_trajectory (final_geometry was flagged as undefined name for routes without edges)
def test_provide_trajectory_merges_the_geometry_of_the_edges(monkeypatch):
    from shapely.geometry import LineString

    from opentnsim.port import utils

    graph = nx.Graph()
    graph.add_edge("A", "B", geometry=LineString([(0, 0), (1, 0)]))
    graph.add_edge("B", "C", geometry=LineString([(1, 0), (2, 0)]))
    vessel = types.SimpleNamespace(env=types.SimpleNamespace(graph=graph))

    monkeypatch.setattr(utils, "get_shortest_route", lambda graph, node_1, node_2, vessel: [node_1, "B", node_2])
    assert utils.provide_trajectory(vessel, "A", "C").equals(LineString([(0, 0), (1, 0), (2, 0)]))

    # a route without edges has no trajectory (used to raise UnboundLocalError)
    monkeypatch.setattr(utils, "get_shortest_route", lambda graph, node_1, node_2, vessel: [node_1])
    assert utils.provide_trajectory(vessel, "A", "A") is None
