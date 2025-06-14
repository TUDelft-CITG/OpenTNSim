# IMPORTS
# package(s) related to time, space and id
import datetime
import time

# package(s) related to the simulation
import simpy
import networkx as nx
import numpy as np
import math

# OpenTNSim
from opentnsim import core
import opentnsim.lock

# spatial libraries
import shapely.geometry
import shapely.wkt
import pyproj
import shapely.geometry

import pytest

# define the coorinate system
geod = pyproj.Geod(ellps="WGS84")


@pytest.mark.skip(reason="broken lock module")
def test_basic_simulation():
    # CREATION OF GRAPH
    Node = type("Site", (core.Identifiable, core.Log, core.Locatable, core.HasResource), {})
    nodes = []
    path = []

    distances = [550, 500, 300, 500, 150, 150, 500, 300, 500, 550]
    coords = []
    coords.append([0, 0])

    for d in range(len(distances)):
        coords.append([pyproj.Geod(ellps="WGS84").fwd(coords[d][0], coords[d][1], 90, distances[d])[0], 0])

    for d in range(len(coords)):
        data_node = {"env": [], "name": "Node " + str(d + 1), "geometry": shapely.geometry.Point(coords[d][0], coords[d][1])}
        node = Node(**data_node)
        nodes.append(node)

    for i in range(2):
        if i == 0:
            for j in range(len(nodes) - 1):
                path.append([nodes[j], nodes[j + 1]])
        if i == 1:
            for j in range(len(nodes) - 1):
                path.append([nodes[j + 1], nodes[j]])

    FG = nx.DiGraph()

    positions = {}
    for node in nodes:
        positions[node.name] = (node.geometry.x, node.geometry.y)
        FG.add_node(node.name, geometry=node.geometry)

    for edge in path:
        FG.add_edge(edge[0].name, edge[1].name, weight=1)

    # SIMULATION SET-UP
    simulation_start = datetime.datetime.now()
    env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))

    # CREATION OF VESSELS
    Vessel = type(
        "Vessel", (core.Identifiable, core.Movable, core.HasContainer, core.HasResource, core.Routeable, core.VesselProperties), {}
    )

    start_point = "Node 11"
    end_point = "Node 1"

    data_vessel_one = {
        "env": env,
        "name": "Vessel",
        "route": nx.dijkstra_path(FG, start_point, end_point, weight="length"),
        "geometry": FG.nodes[start_point]["geometry"],
        "capacity": 1_000,
        "v": 4,
        "type": "CEMT - Va",
        "B": 10,
        "L": 155.0,
    }

    env.FG = FG
    vessels = []

    for v in range(2):
        vessel = Vessel(**data_vessel_one)
        vessels.append(vessel)

    # SYSTEM PARAMETERS
    # water level difference
    wlev_dif = [np.linspace(0, 45000, 1000), np.zeros(1000)]
    for i in range(len(wlev_dif[0])):
        wlev_dif[1][i] = 2

    # lock area parameters
    waiting_area_1 = opentnsim.lock.IsLockWaitingArea(
        env=env, nr_resources=1, priority=True, name="Volkeraksluizen_1", node="Node 2"
    )

    lineup_area_1 = opentnsim.lock.IsLockLineUpArea(
        env=env, nr_resources=1, priority=True, name="Volkeraksluizen_1", node="Node 3", lineup_length=300
    )

    lock_1 = opentnsim.lock.IsLock(
        env=env,
        nr_resources=100,
        priority=True,
        name="Volkeraksluizen_1",
        node_1="Node 5",
        node_2="Node 6",
        node_3="Node 7",
        lock_length=300,
        lock_width=24,
        lock_depth=4.5,
        doors_open=10 * 60,
        doors_close=10 * 60,
        wlev_dif=wlev_dif,
        disch_coeff=0.8,
        grav_acc=9.81,
        opening_area=4.0,
        opening_depth=5.0,
        simulation_start=simulation_start,
        operating_time=25 * 60,
    )

    waiting_area_2 = opentnsim.lock.IsLockWaitingArea(
        env=env, nr_resources=1, priority=True, name="Volkeraksluizen_1", node="Node 10"
    )

    lineup_area_2 = opentnsim.lock.IsLockLineUpArea(
        env=env, nr_resources=1, priority=True, name="Volkeraksluizen_1", node="Node 9", lineup_length=300
    )

    lock_1.water_level = "Node 5"

    # location of lock areas in graph
    FG.nodes["Node 6"]["Lock"] = [lock_1]

    FG.nodes["Node 2"]["Waiting area"] = [waiting_area_1]
    FG.nodes["Node 3"]["Line-up area"] = [lineup_area_1]

    FG.nodes["Node 10"]["Waiting area"] = [waiting_area_2]
    FG.nodes["Node 9"]["Line-up area"] = [lineup_area_2]

    # INITIATE VESSELS
    for vessel in vessels:
        vessel.env = env
        env.process(vessel.move())

    # RUN MODEL
    env.FG = FG
    env.run()

    # OUTPUT ANALYSIS
    lock_cycle_start = np.zeros(len(vessels))
    lock_cycle_duration = np.zeros(len(vessels))
    waiting_in_lineup_start = np.zeros(len(vessels))
    waiting_in_lineup_duration = np.zeros(len(vessels))
    waiting_in_waiting_start = np.zeros(len(vessels))
    waiting_in_waiting_duration = np.zeros(len(vessels))

    for v in range(len(vessels)):
        for t in range(0, len(vessels[v].log["Message"]) - 1):
            if vessels[v].log["Message"][t] == "Passing lock start":
                lock_cycle_start[v] = vessels[v].log["Timestamp"][t].timestamp() - simulation_start.timestamp()
                lock_cycle_duration[v] = vessels[v].log["Value"][t + 1]
                break

        for t in range(0, len(vessels[v].log["Message"]) - 1):
            if vessels[v].log["Message"][t] == "Waiting in line-up area start":
                waiting_in_lineup_start[v] = vessels[v].log["Timestamp"][t].timestamp() - simulation_start.timestamp()
                waiting_in_lineup_duration[v] = vessels[v].log["Value"][t + 1]
                break

        for t in range(0, len(vessels[v].log["Message"]) - 1):
            if vessels[v].log["Message"][t] == "Waiting in waiting area start":
                waiting_in_waiting_start[v] = vessels[v].log["Timestamp"][t].timestamp() - simulation_start.timestamp()
                waiting_in_waiting_duration[v] = vessels[v].log["Value"][t + 1]
                break

    # TESTS
    # start times vessel 1
    np.testing.assert_almost_equal(
        lock_1.doors_open
        + lock_1.doors_close
        + 2
        * lock_1.lock_width
        * lock_1.lock_length
        * wlev_dif[1][0]
        / (lock_1.disch_coeff * lock_1.opening_area * math.sqrt(2 * lock_1.grav_acc * lock_1.opening_depth))
        + 550 / 4
        + 500 / 2
        + (300 - 0.5 * 155) / 2
        + (0.5 * 155 + 500) / 1
        + (300 - 0.5 * 155) / 1,
        lock_cycle_start[0],
        decimal=0,
        err_msg="",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        550 / 4 + 500 / 2 + (300 - 0.5 * 155) / 2, waiting_in_lineup_start[0], decimal=0, err_msg="", verbose=True
    )

    np.testing.assert_almost_equal(0, waiting_in_waiting_start[0], decimal=0, err_msg="", verbose=True)

    # durations vessel 1
    np.testing.assert_almost_equal(
        lock_1.doors_open
        + lock_1.doors_close
        + 2
        * lock_1.lock_width
        * lock_1.lock_length
        * wlev_dif[1][0]
        / (lock_1.disch_coeff * lock_1.opening_area * math.sqrt(2 * lock_1.grav_acc * lock_1.opening_depth)),
        lock_cycle_duration[0],
        decimal=0,
        err_msg="",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        lock_1.doors_open
        + lock_1.doors_close
        + 2
        * lock_1.lock_width
        * lock_1.lock_length
        * wlev_dif[1][0]
        / (lock_1.disch_coeff * lock_1.opening_area * math.sqrt(2 * lock_1.grav_acc * lock_1.opening_depth)),
        waiting_in_lineup_duration[0],
        decimal=0,
        err_msg="",
        verbose=True,
    )

    np.testing.assert_almost_equal(0, waiting_in_waiting_duration[0], decimal=0, err_msg="", verbose=True)

    # start times vessel 2
    np.testing.assert_almost_equal(
        3
        * (
            lock_1.doors_open
            + lock_1.doors_close
            + 2
            * lock_1.lock_width
            * lock_1.lock_length
            * wlev_dif[1][0]
            / (lock_1.disch_coeff * lock_1.opening_area * math.sqrt(2 * lock_1.grav_acc * lock_1.opening_depth))
        )
        + 2 * (0.5 * 155 + (500 + 300 - 0.5 * 155))
        + 550 / 4
        + 0.5 * 155
        + 800 / 4
        + (500 + 300 - 0.5 * 155) / 2,
        lock_cycle_start[1],
        decimal=0,
        err_msg="",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        (
            lock_1.doors_open
            + lock_1.doors_close
            + 2
            * lock_1.lock_width
            * lock_1.lock_length
            * wlev_dif[1][0]
            / (lock_1.disch_coeff * lock_1.opening_area * math.sqrt(2 * lock_1.grav_acc * lock_1.opening_depth))
        )
        + 550 / 4
        + (500 + 300 - 0.5 * 155) / 2
        + 500
        + 0.5 * 155
        + 300
        - 0.5 * 155,
        waiting_in_lineup_start[1],
        decimal=0,
        err_msg="",
        verbose=True,
    )

    np.testing.assert_almost_equal(550 / 4, waiting_in_waiting_start[1], decimal=0, err_msg="", verbose=True)

    # durations vessel 2
    np.testing.assert_almost_equal(
        lock_1.doors_open
        + lock_1.doors_close
        + 2
        * lock_1.lock_width
        * lock_1.lock_length
        * wlev_dif[1][0]
        / (lock_1.disch_coeff * lock_1.opening_area * math.sqrt(2 * lock_1.grav_acc * lock_1.opening_depth)),
        lock_cycle_duration[1],
        decimal=0,
        err_msg="",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        2
        * (
            lock_1.doors_open
            + lock_1.doors_close
            + 2
            * lock_1.lock_width
            * lock_1.lock_length
            * wlev_dif[1][0]
            / (lock_1.disch_coeff * lock_1.opening_area * math.sqrt(2 * lock_1.grav_acc * lock_1.opening_depth))
        )
        + 0.5 * 155
        + 500 / 4
        + 300 / 4,
        waiting_in_lineup_duration[1],
        decimal=0,
        err_msg="",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        lock_1.doors_open
        + lock_1.doors_close
        + 2
        * lock_1.lock_width
        * lock_1.lock_length
        * wlev_dif[1][0]
        / (lock_1.disch_coeff * lock_1.opening_area * math.sqrt(2 * lock_1.grav_acc * lock_1.opening_depth))
        + (300 - 0.5 * 155) / 2
        + 500 / 2
        + 0.5 * 155,
        waiting_in_waiting_duration[1],
        decimal=0,
        err_msg="",
        verbose=True,
    )
