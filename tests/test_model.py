"""test all modules in the model.py script"""

from datetime import datetime
import pandas as pd
import time
import shapely
import networkx as nx
import simpy

from opentnsim.model import VesselGenerator
from opentnsim.core import Movable, Identifiable

def test_vessel_generator_initialization():

    vessel_database = pd.DataFrame({
        "vessel_id": [1],
        "v": 10
    })
    vg = VesselGenerator(vessel_type=Movable, vessel_database=vessel_database)
    assert vg.vessel_type is Movable
    assert vg.vessel_database is vessel_database
    assert vg.loaded is None

def test_vessel_generator_generate():
    """test the generate method of VesselGenerator in normal use"""
    vessel_database = pd.DataFrame({
        "vessel_id": [1],
        "v": 10
    })

    # vessel type is mixin van movable en identifyable
    vessel_type = type("Vessel", (Movable, Identifiable), {})
    vg = VesselGenerator(vessel_type=vessel_type, vessel_database=vessel_database)

    # Start simpy environment
    simulation_start = datetime.now()
    env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))

    # Initialize the graph for the environment with two nodes and one edge
    # 1. create graph
    env.graph = nx.DiGraph()
    env.graph.add_node("A", geometry=shapely.geometry.Point(0, 0))
    env.graph.add_node("B", geometry=shapely.geometry.Point(0.1, 0))
    env.graph.add_edge("A", "B")

    #generate
    vessel = vg.generate(env, "Vessel", route=["A", "B"])
    assert vessel is not None
    assert vessel.name == "Vessel"
    assert vessel.route == ["A", "B"]

def test_vessel_generator_generate_with_empty_route():
    """test the generate method of VesselGenerator with an empty route"""
    vessel_database = pd.DataFrame({
        "vessel_id": [1],
        "v": 10
    })

    vessel_type = type("Vessel", (Movable, Identifiable), {})
    vg = VesselGenerator(vessel_type=vessel_type, vessel_database=vessel_database)

    simulation_start = datetime.now()
    env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))
    env.graph = nx.DiGraph()

    vessel = vg.generate(env, "Vessel", route=[])
    assert vessel is not None
    assert vessel.name == "Vessel"
    assert vessel.route == []

def test_vessel_generator_arrival_process():
    """test the arrival_process method of VesselGenerator"""
    #initiate vessel generator
    vessel_database = pd.DataFrame({
        "vessel_id": [1],
        "v": 10
    })
    vessel_type = type("Vessel", (Movable, Identifiable), {})
    vg = VesselGenerator(vessel_type=vessel_type, vessel_database=vessel_database)

    # Start simpy environment
    env = simpy.Environment(initial_time=0)

    # Initialize the graph for the environment with two nodes and one edge
    # 1. create graph
    env.graph = nx.DiGraph()
    env.graph.add_node("A", geometry=shapely.geometry.Point(0, 0))
    env.graph.add_node("B", geometry=shapely.geometry.Point(0.1, 0))
    env.graph.add_edge("A", "B")

    #add arrival process to the environment
    env.process(vg.arrival_process(
        environment=env,
        origin="A",
        destination="B",
        arrival_distribution=1000,
        scenario=None,
        arrival_process="Markovian",
        fleet_distribution=None
    ))

    # Run the simulation for a short time to allow vessel generation
    env.run(until=10)

    # Check if a vessel has been added to the environment
    assert len(env.vessels) > 0

    # Check if the vessel has the correct route and geometry
    vessel = env.vessels[0]
    assert vessel.route == ["A", "B"]
    assert vessel.geometry == shapely.geometry.Point(0, 0)