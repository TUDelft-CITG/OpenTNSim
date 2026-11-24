import pytest 
import shapely.geometry
import simpy
import networkx as nx
import pandas as pd

from opentnsim.plot import vessel_planning
from opentnsim.core import Movable, Identifiable


@pytest.fixture
def environment():
    environment = simpy.Environment(initial_time=0)
    environment.graph = nx.DiGraph()
    environment.graph.add_node("A", geometry=shapely.geometry.Point(0, 0))
    environment.graph.add_node("B", geometry=shapely.geometry.Point(1, 0))
    environment.graph.add_edge("A", "B")
    environment.graph.add_edge("B", "A")
    return environment

@pytest.fixture
def vessels(environment):
    Vessel = type("Vessel", (Movable, Identifiable), {})
    vessel1 = Vessel(name = 'vessel1', env = environment, v=1, geometry = shapely.geometry.Point(0, 0), route = ["A","B"])
    vessel2 = Vessel(name = 'vessel2', env = environment, v=1, geometry = shapely.geometry.Point(0, 0), route = ["A","B", "A"])
    vessels = [vessel1, vessel2]
    return vessels


def test_vessel_planning_two_vessels(environment, vessels):

    environment.process(vessels[0].move())
    environment.process(vessels[1].move())
    environment.run()

    activities = ["Sailing from node A to node B", "Sailing from node B to node A"]
    fig = vessel_planning(vessels= vessels, activities = activities, colors = ["blue", "red"])

    #check namen van de activiteiten in de figuren
    assert fig.data[0].name == activities[0]
    assert fig.data[1].name == activities[1]

    #check of vessels in juiste scatters voorkomen
    assert "vessel1" in fig.data[0].y
    assert "vessel2" in fig.data[0].y
    assert "vessel2" in fig.data[1].y

    # check of vessel 1 alleen voorkomt op 'none' tijd in tweede scatter (vessel 1 gaat niet van B naar A)
    keys_vessel_1 = [i for i, x in enumerate(fig.data[1].y) if x == "vessel1"]
    assert all(fig.data[1].x[i] is None for i in keys_vessel_1)


def test_vessel_planning_one_empty_logbook(environment, vessels):

    environment.process(vessels[0].move())
    environment.run()

    assert vessels[0].geometry == shapely.geometry.Point(1, 0)
    fig = vessel_planning(vessels= vessels, activities = ["Sailing from node A to node B"], colors = ["blue"])

    #check of vessel 2 wel in de scatter voorkomt, maar zonder tijd (vessel 2 heeft nog niet gevaren)
    keys_vessel_2 = [i for i, x in enumerate(fig.data[0].y) if x == "vessel2"]
    assert all(fig.data[0].x[i] is None for i in keys_vessel_2)