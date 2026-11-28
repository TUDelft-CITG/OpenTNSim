import pytest 
import shapely.geometry
import simpy
import networkx as nx
import pandas as pd

from opentnsim.core import Movable, Identifiable

from opentnsim.core.logutils import logbook2eventtable

@pytest.fixture
def vessels():
    #environment
    environment = simpy.Environment(initial_time=0)
    environment.graph = nx.DiGraph()
    environment.graph.add_node("A", geometry=shapely.geometry.Point(0, 0))
    environment.graph.add_node("B", geometry=shapely.geometry.Point(1, 0))
    environment.graph.add_edge("A", "B")
    environment.graph.add_edge("B", "A")

    # two vessels
    Vessel = type("Vessel", (Movable, Identifiable), {})
    vessel1 = Vessel(name = 'vessel1', env = environment, v=1, geometry = shapely.geometry.Point(0, 0), route = ["A","B"])
    vessel2 = Vessel(name = 'vessel2', env = environment, v=1, geometry = shapely.geometry.Point(0, 0), route = ["A","B", "A"])
    vessels = [vessel1, vessel2]

    #add vessel movement to processes
    environment.process(vessel1.move())
    environment.process(vessel2.move())
    return vessels

def test_logbook2eventtable_normal(vessels):
    #run the environment to generate logbook entries
    vessels[0].env.run()

    #convert logbook to event table
    event_table = logbook2eventtable(vessels)

    #event table should have three events
    assert len(event_table) == 3

    # start locations should be (0, 0), (0, 0) and (1, 0)
    start_locations = [tuple(event_table.iloc[i]["start location"].coords[0]) for i in range(len(event_table))]
    assert start_locations == [(0, 0), (0, 0), (1, 0)]

    # stop locations should be (1, 0), (1, 0) and (0, 0)
    stop_locations = [tuple(event_table.iloc[i]["stop location"].coords[0]) for i in range(len(event_table))]
    assert stop_locations == [(1, 0), (1, 0), (0, 0)]

    # third start time should be same as second stop time
    start_times = [event_table.iloc[i]["start time"] for i in range(len(event_table))]
    stop_times = [event_table.iloc[i]["stop time"] for i in range(len(event_table))]
    assert start_times[2] == stop_times[1]


def test_logbook2eventtable_missing_stop_time(vessels):
    #run the environment to generate logbook entries
    # at time 200000 the second vessel will be on its way from B to A
    vessels[0].env.run(until=200000)

    #convert logbook to event table
    event_table = logbook2eventtable(vessels)

    #event table should have three events
    assert len(event_table) == 3

    # start locations should be (0, 0), (0, 0) and (1, 0)
    start_locations = [tuple(event_table.iloc[i]["start location"].coords[0]) for i in range(len(event_table))]
    assert start_locations == [(0, 0), (0, 0), (1, 0)]

    # stop locations should be (1, 0), (1, 0) and None
    stop_locations = [event_table.iloc[i]["stop location"] for i in range(len(event_table))]
    stop_locations = [tuple(loc.coords[0]) if loc is not None else None for loc in stop_locations]
    assert stop_locations == [(1, 0), (1, 0), None]

    # third start time should be same as second stop time
    start_times = [event_table.iloc[i]["start time"] for i in range(len(event_table))]
    stop_times = [event_table.iloc[i]["stop time"] for i in range(len(event_table))]
    assert start_times[2] == stop_times[1]

    # third stop times should be None
    stop_time_types = [type(time) for time in stop_times]
    assert stop_time_types == [type(stop_times[0]), type(stop_times[1]), type(pd.NaT)]
