# %% IMPORT DEPENDENCIES
import pytest
import simpy
import datetime
import networkx as nx
from shapely.geometry import Point
from pyproj import Geod

from opentnsim.core.movable import Routable, Routeable, Movable, ContainerDependentMovable
from opentnsim.graph import calculate_distance_along_path

# %% FIXTURES
@pytest.fixture
def graph():
    """Fixture for Movable object."""
    # create a directed graph with distance of 100000 meters between two points
    graph = nx.DiGraph()
    graph.add_node(0, geometry=Point(0, 0))
    lon1, lat1, _ = Geod(ellps="WGS84").fwd(0, 0, 90, 100000)  # East from Point 0

    graph.add_node(1, geometry=Point(lon1, lat1))
    graph.add_edge(0, 1, weight=1)
    return graph

@pytest.fixture
def env(graph):
    """Fixture for simpy environment with graph."""
    simulation_start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    env = simpy.Environment(initial_time=simulation_start.timestamp())
    env.epoch = simulation_start
    env.graph = graph
    return env

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


# %% TESTING Routable
def test_routable_initialization(env, graph):
    routable = Routable(route=[1, 2, 3], env=env)
    
    assert routable.route == [1, 2, 3]
    assert routable.env == env
    assert routable.graph == graph

def test_routable_no_env():
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'env'"):
        Routable(route=[1, 2, 3])   

def test_routable_env_has_no_graph():
    simulation_start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    env = simpy.Environment(initial_time=simulation_start.timestamp())
    with pytest.raises(AssertionError, match="Routable expects `.graph`"):
        routable = Routable(route=[1, 2, 3], env=env)

def test_routable_env_with_fg(graph):
    simulation_start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    env = simpy.Environment(initial_time=simulation_start.timestamp())
    env.FG = graph
    with pytest.warns():
        routable = Routable(route=[1, 2, 3], env=env)
    assert routable.env.graph == graph
    assert routable.graph == graph

def test_routeable_warning(env):
    with pytest.warns(DeprecationWarning, match=".Use Routable instead of Routeable"):
        Routeable(route=[1, 2, 3], env=env)

# %% TESTING Movable
def test_movable_initialization(env):
    path = nx.dijkstra_path(env.graph, 0,1)
    geometry = env.graph.nodes[path[0]]['geometry']
    movable = Movable(route=path, env=env, v=1.0, geometry=geometry)

    assert movable.route == path
    assert movable.env == env
    assert movable.graph == env.graph
    assert movable.v == 1.0
    assert movable.on_pass_edge_functions == []

@pytest.mark.timeout(0.5)
def test_movable_move_simple(env):
    path = nx.dijkstra_path(env.graph, 0,1)
    geometry = env.graph.nodes[0]['geometry']

    movable = Movable(route=path, env=env, v=1.0, geometry=geometry)

    assert movable.geometry == geometry
    env.process(mission(env, movable))
    env.run()

    assert movable.geometry == env.graph.nodes[1]['geometry'], "Movable should have moved to the end of the path"

    trip_distance = calculate_distance_along_path(env.graph, movable.route)
    assert pytest.approx(trip_distance, 1e-5) == 100000, "Expected trip distance of 100000, now {}".format(trip_distance)
    assert pytest.approx(movable.distance, 1e-5) == 100000, "Movable should have traveled 100000 meters, now {}".format(
        movable.distance
    )

    traveltime = datetime.timedelta.total_seconds(movable.logbook[-1]["Timestamp"] - movable.logbook[0]["Timestamp"])
    assert pytest.approx(traveltime, 1e-5) == 100000, "Expected travel time of 100000 seconds, now {}".format(traveltime)


@pytest.mark.timeout(0.5)
def test_move_to_start(env):
    path = nx.dijkstra_path(env.graph, 0, 1)
    # start at the end of the path. From here we will move to start point in straight line
    geometry = env.graph.nodes[1]["geometry"]
    movable = Movable(route=path, env=env, v=1.0, geometry=geometry)

    assert movable.geometry == env.graph.nodes[1]["geometry"], "Movable should start at 1,0"

    def mission_move_to_start(env, vessel):
        yield from vessel._move_to_start()

    env.process(mission_move_to_start(env, movable))
    env.run()

    assert movable.geometry == env.graph.nodes[path[0]]["geometry"], "Movable should be at (0,0), not {}".format(movable.geometry)
    assert pytest.approx(movable.distance, 1e-5) == 100000, "Expected distance of 100000, now {}".format(movable.distance)
