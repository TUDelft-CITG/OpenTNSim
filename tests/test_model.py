"""test all modules in the model.py script"""

from datetime import datetime
import pandas as pd
import time
import pytest
import shapely
import networkx as nx
import simpy

from opentnsim.model import VesselGenerator
from opentnsim.core import Movable, Identifiable, ExtraMetadata
from tests.energy.test_basic_energy import vessel


@pytest.fixture
def environment():
    environment = simpy.Environment(initial_time=0)
    environment.graph = nx.DiGraph()
    environment.graph.add_node("A", geometry=shapely.geometry.Point(0, 0))
    environment.graph.add_node("B", geometry=shapely.geometry.Point(0.1, 0))
    environment.graph.add_edge("A", "B")
    return environment

def test_vessel_generator_initialization():

    vessel_database = pd.DataFrame({
        "vessel_id": [1],
        "v": 10
    })
    vg = VesselGenerator(vessel_type=Movable, vessel_database=vessel_database)
    assert vg.vessel_type is Movable
    assert vg.vessel_database is vessel_database
    assert vg.loaded is None

def test_vessel_generator_generate(environment):
    """test the generate method of VesselGenerator in normal use"""
    vessel_database = pd.DataFrame({
        "vessel_id": [1],
        "v": 10
    })

    # vessel type is mixin van movable en identifyable
    vessel_type = type("Vessel", (Movable, Identifiable), {})
    vg = VesselGenerator(vessel_type=vessel_type, vessel_database=vessel_database)

    # generate
    vessel = vg.generate(environment, "Vessel", route=["A", "B"])
    assert vessel is not None
    assert vessel.name == "Vessel"
    assert vessel.route == ["A", "B"]

def test_vessel_generator_generate_with_mixed_scenario(environment):
    """test the generate method of VesselGenerator with a mixed database"""
    # create database
    vessel_database = pd.DataFrame(
        {"vessel_id": [1, 2, 2], "v": [10, 20, 15], "scenario": ["A", "B", "A"]}
    )

    vessel_type = type("Vessel", (Movable, Identifiable, ExtraMetadata), {})
    vg = VesselGenerator(vessel_type=vessel_type, vessel_database=vessel_database)

    # when no scenario is specified, vessel 1 is 33% of the time, vessel 2 is 66% of the time
    vessels = []
    vessel_ids = []
    for i in range(100):
        vessels.append(vg.generate(environment, "Vessel", route=["A", "B"]))
        vessel_ids.append(vessels[-1].id)

    assert 23 < vessel_ids.count(1) < 43, "Vessel 1 count is out of expected range"
    assert 56 < vessel_ids.count(2) < 76, "Vessel 2 count is out of expected range"

    # when scenario "A" is specified, we have 50% vessel 1 and 50% vessel 2
    vessels = []
    vessel_ids = []
    for i in range(100):
        vessels.append(vg.generate(environment, "Vessel", route=["A", "B"], scenario="A"))
        vessel_ids.append(vessels[-1].id)

    assert 40 < vessel_ids.count(1) < 60, "Vessel 1 count in scenario A is out of expected range"
    assert 40 < vessel_ids.count(2) < 60, "Vessel 2 count in scenario A is out of expected range"

    # when scenario "B" is specified, we only have vessel 2
    vessels = []
    vessel_ids = []
    for i in range(20):
        vessels.append(vg.generate(environment, "Vessel", route=["A", "B"], scenario="B"))
        vessel_ids.append(vessels[-1].id)

    assert vessel_ids.count(1) == 0, "Vessel 1 count in scenario B should be 0"
    assert vessel_ids.count(2) == 20, "Vessel 2 count in scenario B is out of expected range"

    # when scenario C is specified, we have an error
    with pytest.raises(ValueError, match="No vessels available for scenario C"):
        vg.generate(environment, "Vessel", route=["A", "B"], scenario="C")


def test_vessel_generator_generate_with_empty_route(environment):
    """test the generate method of VesselGenerator with an empty route"""
    vessel_database = pd.DataFrame({
        "vessel_id": [1],
        "v": 10
    })

    vessel_type = type("Vessel", (Movable, Identifiable), {})
    vg = VesselGenerator(vessel_type=vessel_type, vessel_database=vessel_database)

    vessel = vg.generate(environment, "Vessel", route=[])
    assert vessel is not None
    assert vessel.name == "Vessel"
    assert vessel.route == []

def test_vessel_generator_arrival_process(environment):
    """test the arrival_process method of VesselGenerator"""
    #initiate vessel generator
    vessel_database = pd.DataFrame({
        "vessel_id": [1],
        "v": 10
    })
    vessel_type = type("Vessel", (Movable, Identifiable), {})
    vg = VesselGenerator(vessel_type=vessel_type, vessel_database=vessel_database)

    #add arrival process to the environment
    environment.process(
        vg.arrival_process(
            environment=environment,
            origin="A",
            destination="B",
            arrival_distribution=2,
            scenario=None,
            arrival_process="Uniform",
            fleet_distribution=None,
        )
    )

    # Run the simulation for half an hour to allow vessel generation but no movement
    environment.run(until=30 * 60 + 1)
    # Check if the vessel has the correct route and geometry
    vessel = environment.vessels[0]
    assert vessel.route == ["A", "B"]
    assert vessel.geometry == shapely.geometry.Point(0, 0)

    # Run simulation for a day to check the arrival of subsequent vessels
    environment.run(until=60 * 60 + 1)  # run for 1 day

    # Check if a vessel has been added to the environment
    assert len(environment.vessels) == 2

    # check arrival times (exactly every half hour)
    assert environment.vessels[0].logbook[0]["Timestamp"] == datetime(1970, 1, 1, 1, 30)
    assert environment.vessels[1].logbook[0]["Timestamp"] == datetime(1970, 1, 1, 2, 0)


def test_vessel_generator_arrival_process_with_zero_arrival_rate(environment):
    vessel_database = pd.DataFrame({"vessel_id": [1], "v": 10})
    vessel_type = type("Vessel", (Movable, Identifiable), {})
    vg = VesselGenerator(vessel_type=vessel_type, vessel_database=vessel_database)

    # add arrival process to the environment
    environment.process(
        vg.arrival_process(
            environment=environment,
            origin="A",
            destination="B",
            arrival_distribution=[5] * 2 + [0] * 20 + [5] * 2,
            scenario=None,
            arrival_process="Markovian",
            fleet_distribution=None,
        )
    )

    # Run the simulation for a short time to allow vessel generation
    environment.run(until=24 * 60 * 60)  # run 1 dag

    # check arrival times of each vessel:
    arrival_times = [v.logbook[0]["Timestamp"] for v in environment.vessels]
    arrival_hours = [t.hour for t in arrival_times]

    # expected arrival in hour 0, 1, hour 22, hour 23. Eentje kan nog in uur twee komen omdat die al in uur 1 was gepland
    assert set(arrival_hours).issubset({0, 1, 2, 22, 23})
    assert arrival_hours.count(2) <= 1
