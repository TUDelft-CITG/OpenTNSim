"""
Test the displacement module
"""
import pytest

import opentnsim.displacement


@pytest.fixture
def displacement_ship():
    ship = opentnsim.displacement.Ship.create_new_ship(110, 11.45, "motorship", "container", "double")
    return ship


def test_displacement_ship(displacement_ship):
    displacement_ship.print_properties()
    displacement_ship.draught_displacement_table()
