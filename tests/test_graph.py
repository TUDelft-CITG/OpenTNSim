"""testing the graph module of OpenTNSim"""

from shapely.geometry import Point
import pytest

import opentnsim.fis
from opentnsim.graph.graph import calculate_distance, calculate_depth


# %% TESTING calculate_distance
def test_calculate_distance():
    """Test the calculate_distance function."""

    #  Define points
    point1 = Point(6.162318, 52.255165)
    point2 = Point(6.143504, 52.261667)
    # Calculate distance
    distance = calculate_distance(point1, point2)
    assert pytest.approx(distance, abs=1) == 1474.4, f"Expected distance 1474.4, but got {distance}"


def test_calculate_depth():
    """Test the calculate_depth function.
    TODO Loading graph takes long. put in conftest file?"""
    geom_start = Point(4.95297433281719, 52.3765489377136)
    geom_end = Point(4.95421136373392, 52.3744310413328)
    # edge ('8867414', '8865307') has general depth of 4.0
    graph = opentnsim.fis.load_network(version="0.3")
    depth = calculate_depth(geom_start, geom_end, graph)
    assert depth == 4.0, f"Expected depth 4.0, but got {depth}"
