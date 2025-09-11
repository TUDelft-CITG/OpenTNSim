# %% IMPORT DEPENDENCIES
import pytest
import simpy
import datetime
import networkx as nx
from shapely.geometry import Point
from pyproj import Geod

from opentnsim.core.movable import Routable, Routeable, Movable, ContainerDependentMovable
from opentnsim.graph import calculate_distance_along_path
from opentnsim.energy.mixins import ConsumesEnergy
from opentnsim.core.vessel_properties import VesselProperties

# %% FIXTURES
@pytest.fixture
def graph():
    """Fixture for Movable object."""
    # create a directed graph with distance of 100000 meters between two points
    graph = nx.Graph()
    graph.add_node(0, geometry=Point(0, 0))
    lon1, lat1, _ = Geod(ellps="WGS84").fwd(0, 0, 90, 100000)  # East from Point 0

    graph.add_node(1, geometry=Point(lon1, lat1))
    graph.add_edge(0, 1, weight=1)
    return graph


@pytest.fixture
def digraph():
    """Fixture for Movable object."""
    # create a directed graph with distance of 100000 meters between two points
    graph = nx.DiGraph()
    graph.add_node(0, geometry=Point(0, 0))
    lon1, lat1, _ = Geod(ellps="WGS84").fwd(0, 0, 90, 100000)  # East from Point 0

    graph.add_node(1, geometry=Point(lon1, lat1))
    graph.add_edge(0, 1, weight=1)
    graph.add_edge(1, 0, weight=1)
    return graph


@pytest.fixture
def digraph_movable(digraph):
    simulation_start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    env = simpy.Environment(initial_time=simulation_start.timestamp())
    env.epoch = simulation_start
    env.graph = digraph

    movable = Movable(route=[0, 1], env=env, v=1.0, geometry=digraph.nodes[0]["geometry"])
    return movable


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

        if vessel.position_on_route == len(vessel.route) - 1:
            break


# %% TESTING Routable
def test_routable_initialization(env, graph):
    routable = Routable(route=[0, 1], env=env)

    assert routable.route == [0, 1]
    assert routable.env == env
    assert routable.graph == graph

def test_routable_no_env():
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'env'"):
        Routable(route=[0, 1])

def test_routable_env_has_no_graph():
    simulation_start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    env = simpy.Environment(initial_time=simulation_start.timestamp())
    with pytest.raises(AssertionError, match="Routable expects `.graph`"):
        routable = Routable(route=[0, 1], env=env)

def test_routable_env_with_fg(graph):
    simulation_start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    env = simpy.Environment(initial_time=simulation_start.timestamp())
    env.FG = graph
    with pytest.warns():
        routable = Routable(route=[0, 1], env=env)
    assert routable.env.graph == graph
    assert routable.graph == graph


def test_routable_wrong_route(env):
    with pytest.raises(ValueError, match="Routable route must be on the graph"):
        Routable(route=[0, 1, 2], env=env)


def test_routeable_warning(env):
    with pytest.warns(DeprecationWarning, match=".Use Routable instead of Routeable"):
        Routeable(route=[0, 1], env=env)

# %% TESTING Movable init
def test_movable_initialization(env):
    path = [0, 1]
    geometry = env.graph.nodes[path[0]]['geometry']
    movable = Movable(route=path, env=env, v=1.0, geometry=geometry)

    assert movable.route == path
    assert movable.env == env
    assert movable.graph == env.graph
    assert movable.v == 1.0
    assert movable.on_pass_edge_functions == []

def test_movable_no_geom():
    graph = nx.DiGraph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, weight=1)
    env = simpy.Environment()
    env.graph = graph
    path = nx.dijkstra_path(graph, 0, 1)
    with pytest.raises(ValueError, match="Nodes on route must have a geometry attribute."):
        Movable(route=path, env=env, v=1.0, geometry=Point(0, 0))


@pytest.mark.skip(reason="functionality not implemented yet")
def test_movable_no_generaldepth(env):
    path = [0, 1]

    class P_tot_given_mixin:
        def __init__(self):
            self.P_tot_given = 1

    MixinNew = type("mixinNew", (Movable, P_tot_given_mixin), {})
    with pytest.raises(ValueError, match="Nodes on route must have a 'GeneralDepth' attribute in their 'info'."):
        MixinNew(route=path, env=env, v=1.0, geometry=Point(0, 0))


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


# %% TESTING  _move_to_start method

@pytest.mark.timeout(0.5)
def test_move_to_start(env):
    path = [0, 1]
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
    assert len(movable.logbook) == 2, "Logbook should have 2 entries, now {}".format(len(movable.logbook))
    assert (
        movable.logbook[0]["Message"] == "Sailing to start start"
    ), "First log entry should be 'Sailing to start start', now {}".format(movable.logbook[0]["Message"])
    traveltime = datetime.timedelta.total_seconds(movable.logbook[-1]["Timestamp"] - movable.logbook[0]["Timestamp"])
    assert pytest.approx(traveltime, 1e-5) == 100000, "Expected travel time of 100000 seconds, now {}".format(traveltime)


@pytest.mark.timeout(0.5)
def test_move_to_start_already_there(env):
    path = [0, 1]
    # start at the end of the path. From here we will move to start point in straight line
    geometry = env.graph.nodes[0]["geometry"]
    movable = Movable(route=path, env=env, v=1.0, geometry=geometry)
    starttime = env.now

    assert movable.geometry == env.graph.nodes[0]["geometry"], "Movable should start at 0,0"

    def mission_move_to_start(env, vessel):
        yield from vessel._move_to_start()

    env.process(mission_move_to_start(env, movable))
    env.run()

    assert movable.geometry == env.graph.nodes[path[0]]["geometry"], "Movable should be at (0,0), not {}".format(movable.geometry)
    assert pytest.approx(movable.distance, 1e-5) == 0, "Expected distance of 100000, now {}".format(movable.distance)
    assert len(movable.logbook) == 0, "Logbook should have 2 entries, now {}".format(len(movable.logbook))
    assert env.now == starttime, "Environment time should not have changed, now {}".format(env.now)


# %% TESTING pass_nodes function

def test_movable_pass_node_easy(env):
    path = [0, 1]
    geometry = env.graph.nodes[0]["geometry"]
    movable = Movable(route=path, env=env, v=1.0, geometry=geometry)
    starttime = env.now

    def mission_pass_node(env, vessel):
        yield from vessel.pass_node(1)

    env.process(mission_pass_node(env, movable))
    env.run()

    assert movable.geometry == env.graph.nodes[0]["geometry"], "Movable should still be at node 0"
    assert env.now == starttime, "Environment time should be 0 seconds later, now {}".format(env.now)


def test_movable_pass_node_with_functions(env):
    path = [0, 1]
    geometry = env.graph.nodes[0]["geometry"]
    movable = Movable(route=path, env=env, v=1.0, geometry=geometry)
    starttime = env.now

    def on_pass_node_wait_1(node):
        yield movable.env.timeout(1)

    def on_pass_node_wait_2(node):
        yield movable.env.timeout(2)

    movable.on_pass_node_functions.append(on_pass_node_wait_1)
    movable.on_pass_node_functions.append(on_pass_node_wait_2)

    def mission_pass_node(env, vessel):
        yield from vessel.pass_node(1)

    env.process(mission_pass_node(env, movable))
    env.run()

    assert movable.geometry == env.graph.nodes[0]["geometry"], "Movable should still be at node 0"
    assert env.now == starttime + 3, "Environment time should be 3 seconds later, now {}".format(env.now)


# %% test for simple pass edge

def test_movable_pass_edge(env):
    path = [0, 1]
    geometry = env.graph.nodes[0]["geometry"]
    movable = Movable(route=path, env=env, v=1.0, geometry=geometry)
    movable.update_position(0)

    starttime = env.now

    def mission_pass_edge(env, vessel):
        yield from vessel.pass_edge(origin=0, destination=1)

    env.process(mission_pass_edge(env, movable))
    env.run()

    assert movable.geometry == env.graph.nodes[1]["geometry"], "Movable should be at node 1"
    assert pytest.approx(env.now) == starttime + 100000, "Environment time should be 100000 seconds later, now {}".format(env.now)
    assert pytest.approx(movable.distance, 1e-5) == 100000, "Expected distance of 100000, now {}".format(movable.distance)

# %% tests for movable with energy consumption
# skip if ConsumesEnergy tests fails
# @pytest.mark.skipif()
def test_movable_pass_edge_with_energy_error(env):
    path = [0, 1]
    geometry = env.graph.nodes[0]["geometry"]
    starttime = env.now

    Vessel = type("vessel", (Movable, ConsumesEnergy, VesselProperties), {})
    data_vessel = {
        "env": env,
        "type": "Va/M9 - Verl. Groot Rijnschip",  # This indicates the vessel class. This info is mainly informative.
        "L": 135,  # m
        "B": 11.45,  # m
        "v": 5,  # m/s If None: this value is calculated based on P_tot_given
        "h_squat": False,  # if the ship should squat while moving, set to True, otherwise set to False
        "P_installed": 1750.0,  # kW
        "P_tot_given": 10,  # kW If None: this value is calculated value based on speed
        "bulbous_bow": False,  # if a vessel has no bulbous_bow, set to False; otherwise set to True.
        "karpov_correction": False,  # if False, don't apply the karpov correction, if True, apply the karpov correction
        "P_hotel_perc": 0.05,  # 0: all power goes to propulsion
        "P_hotel": None,  # None: calculate P_hotel from percentage
        "L_w": 3.0,
        "C_B": 0.85,  # block coefficient
        "C_year": 1990,  # engine build year
        "geometry": env.graph.nodes[path[0]]["geometry"],
        "route": path,  # the route to sail
    }  #

    # create an instance of the Vessel class using the input dict data_vessel
    movable = Vessel(**data_vessel)
    movable.update_position(0)

    def mission_pass_edge(env, vessel):
        yield from vessel.pass_edge(origin=0, destination=1)

    env.process(mission_pass_edge(env, movable))
    with pytest.raises(ValueError, match="has no GeneralDepth in Info"):
        env.run()


def test_movable_pass_edge_with_energy(env):
    env.graph.edges[0, 1]["Info"] = {"GeneralDepth": 5}  # from '0' to '1' you sail against the current
    path = [0, 1]
    geometry = env.graph.nodes[0]["geometry"]
    starttime = env.now

    Vessel = type("vessel", (Movable, ConsumesEnergy, VesselProperties), {})
    data_vessel = {
        "env": env,
        "route": path,
        "geometry": geometry,
        "v": None,  # m/s
        "type": None,
        "B": 11.4,
        "L": 110,
        "T": 1,
        "P_installed": 1750.0,  # kW
        "P_tot_given": 333,  # kW
        "L_w": 3.0,
        "C_year": 1990,
    }

    # create an instance of the Vessel class using the input dict data_vessel
    movable = Vessel(**data_vessel)

    movable.update_position(0)

    def mission_pass_edge(env, vessel):
        yield from vessel.pass_edge(origin=0, destination=1)

    env.process(mission_pass_edge(env, movable))
    env.run()
    assert movable.geometry == env.graph.nodes[1]["geometry"], "Movable should be at node 1"
    assert movable.v == pytest.approx(1.65, 0.01), "Expected speed of 3.5 m/s, now {}".format(movable.v)
    assert pytest.approx(env.now, abs=1) == starttime + 60528, "Environment time should be 100000 seconds later, now {}".format(
        env.now - starttime
    )


# %% Tests for _get_current method
def test_movable_get_current_no_info(digraph_movable):
    current = digraph_movable._get_current(origin=0, destination=1)
    assert current == 0.0, "Expected current to be 0.0, now {}".format(current)


def test_movable_get_current_no_current(digraph_movable):
    digraph_movable.env.graph.edges[0, 1]["Info"] = {}  # 0.5 current
    current = digraph_movable._get_current(origin=0, destination=1)
    assert current == 0.0, "Expected current to be 0.0, now {}".format(current)


def test_movable_get_current(digraph_movable):
    digraph_movable.env.graph.edges[0, 1]["Info"] = {"Current": 0.5}  # 0.5 current
    current = digraph_movable._get_current(origin=0, destination=1)
    assert current == 0.5, "Expected current to be 0.5, now {}".format(current)


def test_movable_get_current_negative(digraph_movable):
    digraph_movable.env.graph.edges[0, 1]["Info"] = {"Current": -0.5}  # -0.5 current
    current = digraph_movable._get_current(origin=0, destination=1)
    assert current == -0.5, "Expected current to be -0.5, now {}".format(current)


def test_movable_get_current_too_high(digraph_movable):
    digraph_movable.env.graph.edges[0, 1]["Info"] = {"Current": -1.5}  # 1.5 current
    with pytest.raises(ValueError, match="Current -1.5 m/s is larger than current speed 1.0 m/s"):
        digraph_movable._get_current(origin=0, destination=1)


def test_movable_get_current_no_digraph(digraph_movable, graph):
    graph.edges[0, 1]["Info"] = {"Current": 0.5}  # 0.5 current
    digraph_movable.env.graph = graph  # change to a non-DiGraph
    with pytest.raises(
        TypeError, match="Current is only available on a DiGraph. Use a Digraph to use current in your calculations."
    ):
        digraph_movable._get_current(origin=0, destination=1)


def test_movable_pass_edge_with_current(digraph_movable):
    env = digraph_movable.env
    # add current to the edge
    env.graph.edges[0, 1]["Info"] = {"Current": 0.5}  # 0.5 current
    env.graph.edges[1, 0]["Info"] = {"Current": -0.5}  # -0.5 current in the opposite direction
    digraph_movable.update_position(0)

    starttime = env.now

    def mission_pass_edge(env, vessel):
        yield from vessel.pass_edge(origin=0, destination=1)

    env.process(mission_pass_edge(env, digraph_movable))
    env.run()

    assert digraph_movable.geometry == env.graph.nodes[1]["geometry"], "Movable should be at node 1"
    assert pytest.approx(env.now, abs=1) == starttime + 66666, "Environment time should be 100000 seconds later, now {}".format(
        env.now
    )
    assert pytest.approx(digraph_movable.distance, 1e-5) == 100000, "Expected distance of 100000, now {}".format(movable.distance)


# %% Tests for Movable with resource restrictions
def test_movable_resource_restriction_one_edge(env):
    path = [0, 1]
    geometry = env.graph.nodes[0]["geometry"]

    # create two similar movables
    movable1 = Movable(route=path, env=env, v=1.0, geometry=geometry)
    movable1.current_node, movable1.next_node = 0, 1

    movable2 = Movable(route=path, env=env, v=1.0, geometry=geometry)
    movable2.current_node, movable2.next_node = 0, 1

    # Assign a resource to the edge
    resource = simpy.Resource(env, 1)
    env.graph.edges[0, 1]["Resources"] = resource

    # run simulation
    starttime = env.now

    def mission_pass_edge(env, vessel):
        yield from vessel.pass_edge(origin=0, destination=1)

    env.process(mission_pass_edge(env, movable1))
    env.process(mission_pass_edge(env, movable2))
    env.run()

    # check outputs
    assert movable1.geometry == env.graph.nodes[1]["geometry"], "Movable1 should be at node 1"
    assert movable2.geometry == env.graph.nodes[1]["geometry"], "Movable2 should be at node 1"

    # check if the time is correct
    assert len(movable1.logbook) == 2, "Movable1 logbook should have 2 entries, now {}".format(len(movable1.logbook))
    assert len(movable2.logbook) == 4, "Movable2 logbook should have 4 entries, now {}".format(len(movable2.logbook))
    assert movable1.logbook[1]["Timestamp"] - movable1.logbook[0]["Timestamp"] == datetime.timedelta(seconds=100000)
    assert movable2.logbook[3]["Timestamp"] - movable2.logbook[0]["Timestamp"] == datetime.timedelta(seconds=200000)
    assert movable2.logbook[2]["Timestamp"] - movable2.logbook[0]["Timestamp"] == datetime.timedelta(seconds=100000)

    # check if log messages are correct
    assert (
        movable1.logbook[0]["Message"] == "Sailing from node 0 to node 1 start"
    ), "Movable1 first log entry should be 'Sailing from node 0 to node 1 start', now {}".format(movable1.logbook[0]["Message"])
    assert (
        movable1.logbook[1]["Message"] == "Sailing from node 0 to node 1 stop"
    ), "Movable1 second log entry should be 'Sailing from node 0 to node 1 stop', now {}".format(movable1.logbook[1]["Message"])
    assert (
        movable2.logbook[0]["Message"] == "Waiting to pass edge 0 - 1 start"
    ), "Movable2 first log entry should be 'Waiting to pass edge 0 - 1 start', now {}".format(movable2.logbook[0]["Message"])
    assert (
        movable2.logbook[1]["Message"] == "Waiting to pass edge 0 - 1 stop"
    ), "Movable2 second log entry should be 'Waiting to pass edge 0 - 1 stop', now {}".format(movable2.logbook[1]["Message"])
    assert (
        movable2.logbook[2]["Message"] == "Sailing from node 0 to node 1 start"
    ), "Movable2 third log entry should be 'Sailing from node 0 to node 1 start', now {}".format(movable2.logbook[2]["Message"])


def test_movable_resource_restriction_two_edges_and_node(env):
    path = [0, 1, 0]
    geometry = env.graph.nodes[0]["geometry"]

    # create two similar movables
    movable1 = Movable(route=path, env=env, v=1.0, geometry=geometry)
    movable2 = Movable(route=path, env=env, v=1.0, geometry=geometry)

    # Assign a resource to the edge and the node
    resource = simpy.Resource(env, 1)
    env.graph.edges[0, 1]["Resources"] = resource
    env.graph.nodes[1]["Resources"] = resource

    # run simulation
    starttime = env.now

    env.process(mission(env, movable1))
    env.process(mission(env, movable2))
    env.run()

    # check outputs
    assert movable1.geometry == env.graph.nodes[0]["geometry"], "Movable1 should be at node 0"
    assert movable2.geometry == env.graph.nodes[0]["geometry"], "Movable2 should be at node 0"

    # check if the time is correct
    assert len(movable1.logbook) == 4, "Movable1 logbook should have 4 entries (4 for sailing), now {}".format(
        len(movable1.logbook)
    )
    assert len(movable2.logbook) == 6, "Movable2 logbook should have 6 entries(2 for waiting, 4 for sailing), now {}".format(
        len(movable2.logbook)
    )
    assert movable1.logbook[1]["Timestamp"] - movable1.logbook[0]["Timestamp"] == datetime.timedelta(
        seconds=100000
    ), "Movable1 should have sailed for 100000 seconds, now {}".format(
        movable1.logbook[1]["Timestamp"] - movable1.logbook[0]["Timestamp"]
    )
    assert movable1.logbook[3]["Timestamp"] - movable1.logbook[0]["Timestamp"] == datetime.timedelta(
        seconds=200000
    ), "Movable1 should have sailed for 200000 seconds, now {}".format(
        movable1.logbook[3]["Timestamp"] - movable1.logbook[0]["Timestamp"]
    )
    assert movable2.logbook[1]["Timestamp"] - movable2.logbook[0]["Timestamp"] == datetime.timedelta(
        seconds=200000
    ), "Movable2 should have waited for 200000 seconds, now {}".format(
        movable2.logbook[1]["Timestamp"] - movable2.logbook[0]["Timestamp"]
    )
    assert movable2.logbook[5]["Timestamp"] - movable2.logbook[0]["Timestamp"] == datetime.timedelta(
        seconds=400000
    ), "Movable2 should have sailed for 400000 seconds, now {}".format(
        movable2.logbook[5]["Timestamp"] - movable2.logbook[0]["Timestamp"]
    )

    # check if log messages are correct
    assert (
        movable1.logbook[0]["Message"] == "Sailing from node 0 to node 1 start"
    ), "Movable1 first log entry should be 'Sailing from node 0 to node 1 start', now {}".format(movable1.logbook[0]["Message"])
    assert (
        movable1.logbook[1]["Message"] == "Sailing from node 0 to node 1 stop"
    ), "Movable1 second log entry should be 'Sailing from node 0 to node 1 stop', now {}".format(movable1.logbook[1]["Message"])
    assert (
        movable2.logbook[0]["Message"] == "Waiting to pass edge 0 - 1 start"
    ), "Movable2 first log entry should be 'Waiting to pass edge 0 - 1 start', now {}".format(movable2.logbook[0]["Message"])
    assert (
        movable2.logbook[1]["Message"] == "Waiting to pass edge 0 - 1 stop"
    ), "Movable2 second log entry should be 'Waiting to pass edge 0 - 1 stop', now {}".format(movable2.logbook[1]["Message"])
    assert (
        movable2.logbook[2]["Message"] == "Sailing from node 0 to node 1 start"
    ), "Movable2 third log entry should be 'Sailing from node 0 to node 1 start', now {}".format(movable2.logbook[2]["Message"])


def test_movable_resource_restriction_two_directions(env):
    # create two similar movables
    movable1 = Movable(route=[0, 1], env=env, v=1.0, geometry=env.graph.nodes[0]["geometry"])
    movable2 = Movable(route=[1, 0], env=env, v=1.0, geometry=env.graph.nodes[1]["geometry"])

    # Assign a resource to the edge and the node
    resource = simpy.Resource(env, 1)
    env.graph.edges[0, 1]["Resources"] = resource
    env.graph.nodes[1]["Resources"] = resource

    # run simulation
    starttime = env.now

    env.process(mission(env, movable1))
    env.process(mission(env, movable2))
    env.run()

    # check outputs
    assert movable1.geometry == env.graph.nodes[1]["geometry"], "Movable1 should be at node 1"
    assert movable2.geometry == env.graph.nodes[0]["geometry"], "Movable2 should be at node 0"

    # check if the time is correct
    assert len(movable1.logbook) == 2, "Movable1 logbook should have 2 entries (2 for sailing), now {}".format(
        len(movable1.logbook)
    )
    assert len(movable2.logbook) == 4, "Movable2 logbook should have 4 entries(2 for waiting, 2 for sailing), now {}".format(
        len(movable2.logbook)
    )
    assert movable1.logbook[1]["Timestamp"] - movable1.logbook[0]["Timestamp"] == datetime.timedelta(
        seconds=100000
    ), "Movable1 should have sailed for 100000 seconds, now {}".format(
        movable1.logbook[1]["Timestamp"] - movable1.logbook[0]["Timestamp"]
    )
    assert movable2.logbook[1]["Timestamp"] - movable2.logbook[0]["Timestamp"] == datetime.timedelta(
        seconds=100000
    ), "Movable2 should have waited for 100000 seconds, now {}".format(
        movable2.logbook[1]["Timestamp"] - movable2.logbook[0]["Timestamp"]
    )
    assert movable2.logbook[3]["Timestamp"] - movable2.logbook[0]["Timestamp"] == datetime.timedelta(
        seconds=200000
    ), "Movable2 should have sailed for 200000 seconds, now {}".format(
        movable2.logbook[3]["Timestamp"] - movable2.logbook[0]["Timestamp"]
    )

    # check if log messages are correct
    assert (
        movable1.logbook[0]["Message"] == "Sailing from node 0 to node 1 start"
    ), "Movable1 first log entry should be 'Sailing from node 0 to node 1 start', now {}".format(movable1.logbook[0]["Message"])
    assert (
        movable1.logbook[1]["Message"] == "Sailing from node 0 to node 1 stop"
    ), "Movable1 second log entry should be 'Sailing from node 0 to node 1 stop', now {}".format(movable1.logbook[1]["Message"])
    assert (
        movable2.logbook[0]["Message"] == "Waiting to pass node 1 start"
    ), "Movable2 first log entry should be 'Waiting to pass node 1 start', now {}".format(movable2.logbook[0]["Message"])
    assert (
        movable2.logbook[1]["Message"] == "Waiting to pass node 1 stop"
    ), "Movable2 second log entry should be 'Waiting to pass node 1 stop', now {}".format(movable2.logbook[1]["Message"])
    assert (
        movable2.logbook[2]["Message"] == "Sailing from node 1 to node 0 start"
    ), "Movable2 third log entry should be 'Sailing from node 1 to node 0 start', now {}".format(movable2.logbook[2]["Message"])
