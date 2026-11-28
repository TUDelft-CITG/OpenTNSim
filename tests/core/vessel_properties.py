"""Tests for the vessel_properties module."""

import pytest
import simpy
import shapely
import networkx as nx

from opentnsim.core.vessel_properties import VesselProperties
from opentnsim.core.movable import Movable

@pytest.fixture
def vessel():
    environment = simpy.Environment(initial_time=0)
    environment.graph = nx.DiGraph()
    environment.graph.add_node("A", geometry=shapely.geometry.Point(0, 0))
    environment.graph.add_node("B", geometry=shapely.geometry.Point(1, 0))
    environment.graph.add_node("C", geometry=shapely.geometry.Point(1, 1))

    # we can go from A to B to C
    environment.graph.add_edge("A", "B")
    environment.graph.add_edge("B", "C")

    #or from A directly to C
    environment.graph.add_edge("A", "C")

    Vessel = type("Vessel", (Movable, VesselProperties), {})
    vessel = Vessel(
        type="TestType",
        B=10,
        L=100,
        v=10,
        env=environment, 
        geometry = shapely.geometry.Point(0, 0),
        route = []
    )
    return vessel

def test_vessel_properties_initialization():
    vessel = VesselProperties(
        type="TestType",
        B=10,
        L=100
    )
    assert vessel.type == "TestType"
    assert vessel.B == 10
    assert vessel.L == 100

def test_get_route_no_restrictions(vessel):
    route = vessel.get_route(origin="A", destination="C")
    #should give the direct route from A to C
    assert route == ['A', 'C']

def test_get_route_with_all_restrictions(vessel):
    # Add restrictions to the graph edges
    vessel.graph["A"]["B"]["Width"] = 15
    vessel.graph["A"]["B"]["Height"] = 5
    vessel.graph["A"]["B"]["Depth"] = 3

    vessel.graph["B"]["C"]["Width"] = 15
    vessel.graph["B"]["C"]["Height"] = 5
    vessel.graph["B"]["C"]["Depth"] = 3

    # is not wide enough:
    vessel.graph["A"]["C"]["Width"] = 5
    vessel.graph["A"]["C"]["Height"] = 5
    vessel.graph["A"]["C"]["Depth"] = 3

    # vessel needs attributes for H and T to use all restrictions
    vessel.H = 2
    vessel._T = 2

    #calculate the route
    route = vessel.get_route(origin="A", destination="C")

    # The direct route from A to C is restricted, so it should go through B
    assert route == ['A', 'B', 'C']

def test_get_route_with_only_width_restriction(vessel):
    # Add restrictions to the graph edges
    vessel.graph["A"]["B"]["Width"] = 15

    vessel.graph["B"]["C"]["Width"] = 15

    # is not wide enough:
    vessel.graph["A"]["C"]["Width"] = 5

    #calculate the route 
    route = vessel.get_route(origin="A", destination="C")

    # The direct route from A to C is not restricted, so it should go directly
    assert route == ['A', 'B', 'C']

def test_get_route_with_only_height_restriction(vessel):
    # Add restrictions to the graph edges
    vessel.graph["A"]["B"]["Height"] = 3
    vessel.graph["B"]["C"]["Height"] = 3

    # is not high enough:
    vessel.graph["A"]["C"]["Height"] = 1

    #calculate the route 
    route = vessel.get_route(origin="A", destination="C")

    # The height of the vessel is not known, so the height restriction is not applied
    assert route == ['A', 'C']

    # Now the height restriction is being applied, so it should go through B
    vessel.H = 2
    route = vessel.get_route(origin="A", destination="C")
    assert route == ['A', 'B', 'C']

def test_get_route_with_restrictions_and_edge_weights(vessel):

    #route without weights goes directly from A to C
    route = vessel.get_route(origin="A", destination="C")
    assert route == ['A', 'C']

    # the route from abc has lower weight than the direct route from A to C
    vessel.graph["A"]["B"]["weight"] = 1
    vessel.graph["B"]["C"]["weight"] = 1
    vessel.graph["A"]["C"]["weight"] = 5

    #calculate the route (without restrictions)
    route = vessel.get_route(origin="A", destination="C")

    assert route == ['A', 'B', 'C']

    #now add restrictions that are high enough, and see if route stays the same
    vessel.graph["A"]["B"]["Width"] = 15
    vessel.graph["B"]["C"]["Width"] = 15
    vessel.graph["A"]["C"]["Width"] = 15

    route = vessel.get_route(origin="A", destination="C")
    assert route == ['A', 'B', 'C']