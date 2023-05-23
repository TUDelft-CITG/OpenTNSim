## IMPORTS
# package(s) related to time, space and id
import datetime, time

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
    ## CREATION OF GRAPH
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

    ## SIMULATION SET-UP
    simulation_start = datetime.datetime.now()
    env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))

    ## CREATION OF VESSELS
    Vessel = type(
        "Vessel",
        (
            core.Identifiable,
            core.HasContainer,
            core.HasResource,
            core.VesselProperties,
            opentnsim.lock.CanPassLock,
        ),
        {},
    )
    start_point = "Node 1"
    end_point = "Node 11"

    data_vessel_one = {
        "env": env,
        "name": "Vessel",
        "route": nx.dijkstra_path(FG, end_point, start_point, weight="length"),
        "geometry": FG.nodes[end_point]["geometry"],
        "capacity": 1_000,
        "v": 4,
        "type": "CEMT - Va",
        "B": 10,
        "L": 135.0,
    }

    env.FG = FG
    vessels = []

    vessel = Vessel(**data_vessel_one)
    vessels.append(vessel)

    ## SYSTEM PARAMETERS
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

    ## INITIATE VESSELS
    for vessel in vessels:
        vessel.env = env
        env.process(vessel.move())

    ## RUN MODEL
    env.FG = FG
    env.run()

    ## OUTPUT ANALYSIS
    def calculate_distance(orig, dest):
        wgs84 = pyproj.Geod(ellps="WGS84")

        distance = wgs84.inv(orig[0], orig[1], dest[0], dest[1])[2]
        return distance

    lock_cycle_start = np.zeros(len(vessels))
    lock_cycle_duration = np.zeros(len(vessels))
    waiting_in_lineup_start = np.zeros(len(vessels))
    waiting_in_lineup_duration = np.zeros(len(vessels))
    waiting_in_waiting_start = np.zeros(len(vessels))
    waiting_in_waiting_duration = np.zeros(len(vessels))
    vessel_speed = []
    vessel_speeds = []

    for vessel in vessels:
        for t, row in enumerate(vessel.logbook):
            if t == 0:
                continue
            if t == len(vessel.logbook) - 1:
                continue
            prev_row = vessel.logbook[t - 1]
            next_row = vessel.logbook[t + 1]
            if "node" in row["Message"] and "stop" in row["Message"]:
                vessel_speed.append(
                    calculate_distance(
                        [row["Geometry"].x, row["Geometry"].y],
                        [prev_row["Geometry"].x, prev_row["Geometry"].y],
                    )
                    / (row["Timestamp"].timestamp() - prev_row["Timestamp"].timestamp())
                )

            if row["Message"] == "Passing lock start":
                lock_cycle_start[v] = row["Timestamp"].timestamp() - simulation_start.timestamp()
                lock_cycle_duration[v] = next_row["Value"]

            if row["Message"] == "Waiting in line-up area start":
                waiting_in_lineup_start[v] = row["Timestamp"].timestamp() - simulation_start.timestamp()
                waiting_in_lineup_duration[v] = next_row["Value"]

            if row["Message"] == "Waiting in waiting area start":
                waiting_in_waiting_start[v] = row["Timestamp"].timestamp() - simulation_start.timestamp()
                waiting_in_waiting_duration[v] = next_row["Value"]
        vessel_speeds.append(vessel_speed)

    ## TESTS
    # start times
    np.testing.assert_almost_equal(
        lock_1.doors_open
        + lock_1.doors_close
        + 2
        * lock_1.lock_width
        * lock_1.lock_length
        * wlev_dif[1][0]
        / (lock_1.disch_coeff * lock_1.opening_area * math.sqrt(2 * lock_1.grav_acc * lock_1.opening_depth)),
        lock_cycle_duration,
        decimal=0,
        err_msg="",
        verbose=True,
    )

    np.testing.assert_almost_equal(lock_cycle_duration, waiting_in_lineup_duration, decimal=0, err_msg="", verbose=True)

    np.testing.assert_almost_equal(0, waiting_in_waiting_duration, decimal=0, err_msg="", verbose=True)

    # durations
    np.testing.assert_almost_equal(
        lock_1.doors_open
        + lock_1.doors_close
        + 2
        * lock_1.lock_width
        * lock_1.lock_length
        * wlev_dif[1][0]
        / (lock_1.disch_coeff * lock_1.opening_area * math.sqrt(2 * lock_1.grav_acc * lock_1.opening_depth))
        + distances[0] / vessel_speeds[v][0]
        + distances[1] / vessel_speeds[v][1]
        + (distances[2] - 0.5 * vessels[v].L) / vessel_speeds[v][1]
        + (0.5 * vessels[v].L + distances[3]) / vessel_speeds[v][3]
        + ((distances[4] + distances[5]) - 0.5 * vessels[v].L) / vessel_speeds[v][4],
        lock_cycle_start,
        decimal=0,
        err_msg="",
        verbose=True,
    )

    np.testing.assert_almost_equal(
        distances[0] / vessel_speeds[v][0]
        + distances[1] / vessel_speeds[v][1]
        + (distances[2] - 0.5 * vessels[v].L) / vessel_speeds[v][1],
        waiting_in_lineup_start,
        decimal=0,
        err_msg="",
        verbose=True,
    )

    np.testing.assert_almost_equal(0, waiting_in_waiting_start, decimal=0, err_msg="", verbose=True)
