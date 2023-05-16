# Importing libraries

# package(s) related to time, space and id
import datetime

# Used for mathematical functions
import math
import platform
import time

import matplotlib.pyplot as plt

# Used for making the graph to visualize our problem
import networkx as nx

# package(s) for data handling
import numpy as np

# OpenTNSim
import opentnsim
import pandas as pd

# spatial libraries
import pyproj
import shapely.geometry

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy
from simplekml import Kml, Style

# Creating the test objects


# Actual testing starts here
def test_basic_simulation():
    # 1. create graph
    FG = nx.DiGraph()

    Node = type("Site", (opentnsim.core.Identifiable, opentnsim.core.Locatable), {})

    data_node_1 = {"name": "Node 1", "geometry": shapely.geometry.Point(0, 0)}
    data_node_2 = {"name": "Node 2", "geometry": shapely.geometry.Point(0.1, 0)}
    data_node_3 = {"name": "Node 3", "geometry": shapely.geometry.Point(0.2, 0)}
    data_node_4 = {"name": "Node 4", "geometry": shapely.geometry.Point(0.3, 0)}

    node_1 = Node(**data_node_1)
    node_2 = Node(**data_node_2)
    node_3 = Node(**data_node_3)
    node_4 = Node(**data_node_4)

    nodes = [node_1, node_2, node_3, node_4]

    positions = {}
    for node in nodes:
        positions[node.name] = (node.geometry.x, node.geometry.y)
        FG.add_node(node.name, geometry=node.geometry)

    path = [
        [node_1, node_2],  # From node 1 to node 2
        [node_2, node_3],  # From node 2 to node 3
        [node_3, node_4],  # From node 3 to node 4
        [node_4, node_3],  # From node 4 to node 3
        [node_3, node_2],  # From node 3 to node 2
        [node_2, node_1],
    ]  # From node 2 to node 1

    for edge in path:
        FG.add_edge(edge[0].name, edge[1].name, weight=1)

    # 2. Create vessel
    # Make your preferred class out of available mix-ins.
    TransportResource = type(
        "Vessel",
        (opentnsim.core.Identifiable, opentnsim.core.Movable, opentnsim.core.Routeable),
        {},
    )

    # Create a dict with all important settings
    data_vessel = {
        "env": None,
        "name": None,
        "route": None,
        "geometry": None,
        "v": 4,
    }  # constant speed of the vessel

    vessel = TransportResource(**data_vessel)

    # 3. Define path
    path = nx.dijkstra_path(FG, node_1.name, node_4.name)

    # 4. Run simulation
    # Start simpy environment
    simulation_start = datetime.datetime.now()
    env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))
    env.epoch = time.mktime(simulation_start.timetuple())

    # Add graph to environment
    env.FG = FG

    # Add environment and path to the vessel
    vessel.env = env  # the created environment
    vessel.name = "Vessel No.1"
    vessel.route = path  # the route (the sequence of nodes, as stored as the second column in the path)
    vessel.geometry = env.FG.nodes[path[0]][
        "geometry"
    ]  # a shapely.geometry.Point(lon,lat) (here taken as the starting node of the vessel)

    # Start the simulation
    env.process(vessel.move())
    env.run()

    # 5. Test output
    def calculate_distance(orig, dest):
        wgs84 = pyproj.Geod(ellps="WGS84")

        distance = wgs84.inv(orig[0], orig[1], dest[0], dest[1])[2]

        return distance

    vessel_path_x = []
    vessel_path_t = []

    vessel_path_x0 = vessel.logbook[0]["Geometry"].x, vessel.logbook[0]["Geometry"].y
    vessel_path_t0 = vessel.logbook[0]["Timestamp"].timestamp() / 60
    for row in vessel.logbook:
        vessel_path_x.append(
            calculate_distance(
                vessel_path_x0,
                (row["Geometry"].x, row["Geometry"].y),
            )
        )
        vessel_path_t.append(row["Timestamp"].timestamp() / 60 - vessel_path_t0)

    # Test if max distance / speed roughly is equal to simulation time (NB: should be exactly the same (bug))
    np.testing.assert_almost_equal(
        (vessel_path_x[-1] / vessel.v) / 60,
        vessel_path_t[-1],
        decimal=7,
        err_msg="",
        verbose=True,
    )


if __name__ == "__main__":
    test_basic_simulation()
