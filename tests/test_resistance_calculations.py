# Import libraries
# Import pytest library for testing
import pytest

# import pandas for data handling
import pandas as pd

# Import the resistance formulations

import opentnsim.resistance as resistance

# Import numpy for math
import numpy as np

"""
Testing the EnergyConsumption class of resistance.py
"""


@pytest.fixture()
def vessel_database():
	return pd.read_csv("tests/vessels/vessels.csv")

