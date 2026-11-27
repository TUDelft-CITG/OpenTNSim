"""
Test integration of OpenTNSim with OpenCLSim.
"""

# %% DEPENDENCIES
import simpy
import pandas as pd
import networkx as nx
from openclsim.plot import get_log_dataframe
from shapely.geometry import Point

import openclsim.core as clcore
import openclsim.model as clmodel
import opentnsim.core as tncore
import opentnsim.model as tnmodel


# %% ACTUAL TESTING
def test_integration():
    """Test simple integration of OpenTNSim and OpenCLSim."""
    # simulation objects
    Site = type(
        "Site",
        (
            clcore.Identifiable,
            clcore.Log,
            clcore.Locatable,
            clcore.HasResource,
            clcore.HasContainer,
            clcore.Processor,
        ),
        {},
    )
    Mover = type(
        "Mover",
        (
            clcore.Identifiable,
            clcore.HasResource,
            clcore.HasContainer,
            tncore.Movable,
        ),
        {},
    )

    # the network
    nodes = [
        ("quay", Point(0, 0)),
        ("wp1", Point(0, 1)),
        ("wp2", Point(0, 2)),
        ("wp3", Point(0, 3)),
        ("yard", Point(0, 4)),
    ]
    edges = [
        ("quay", "wp1"),
        ("wp1", "wp2"),
        ("wp2", "wp3"),
        ("wp3", "yard"),
    ]
    G = nx.Graph()
    for node_name, node_geometry in nodes:
        G.add_node(
            node_name,
            pos=(node_geometry.x, node_geometry.y),
            geometry=node_geometry,
        )
    G.add_edges_from(edges)

    # initialize the environment
    env = simpy.Environment()
    env.graph = G
    registry = {}

    # objects
    quay = Site(
        env=env,
        name="quay",
        geometry=G.nodes["quay"]["geometry"],
        capacity=10,
        level=10,
    )
    yard = Site(
        env=env,
        name="yard",
        geometry=G.nodes["yard"]["geometry"],
        capacity=9999,
        level=0,
    )
    mover = Mover(
        env=env,
        name="mover",
        geometry=G.nodes["quay"]["geometry"],
        capacity=1,
        v=10,
        route=[],
    )

    # activity
    main_activity = clmodel.WhileActivity(
        env=env,
        name=f"{mover.name}_while",
        registry=registry,
        sub_processes=[
            clmodel.ShiftAmountActivity(
                env=env,
                registry=registry,
                name=f"{mover.name}_loading",
                processor=quay,
                origin=quay,
                destination=mover,
                amount=1,
                duration=1,
            ),
            tnmodel.NetworkMoveActivity(
                env=env,
                registry=registry,
                name=f"{mover.name}_move_to_yard",
                mover=mover,
                route=["quay", "wp1", "wp2", "wp3", "yard"],
            ),
            clmodel.ShiftAmountActivity(
                env=env,
                registry=registry,
                name=f"{mover.name}_unloading",
                processor=yard,
                origin=mover,
                destination=yard,
                amount=1,
                duration=1,
            ),
            tnmodel.NetworkMoveActivity(
                env=env,
                registry=registry,
                name=f"{mover.name}_move_to_quay",
                mover=mover,
                route=["yard", "wp3", "wp2", "wp1", "quay"],
            ),
        ],
        condition_event=[{"type": "container", "concept": quay, "state": "empty"}],
    )

    # run simulation
    clmodel.register_processes([main_activity])
    env.run()

    # checks on the outcome
    assert quay.container.get_level() == 0
    assert yard.container.get_level() == 10

    df_log = get_log_dataframe(mover)
    for _, node_geometry in nodes:
        assert node_geometry in df_log["geometry"].values, f"{node_geometry} not visited"
