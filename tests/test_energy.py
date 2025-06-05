"""Tests related to the energy module."""

# %% IMPORT DEPENDENCIES
import pytest

from opentnsim.energy import calculate_engine_age


# %% FIXTURES


# %% TESTING calculate_engine_age
@pytest.mark.parametrize("L_w", [1, 2, 3])
def test_calculate_engine_age(L_w):
    """Test the calculate_engine_age function."""
    age = calculate_engine_age(L_w)
    assert isinstance(age, int), "Engine age should be a int."


@pytest.mark.parametrize("L_w", [None, 0, -1, 4, 100, "str", [], {}])
def test_calculate_engine_age_infeasible_input(L_w):
    """Test the calculate_engine_age function with infeasible inputs."""
    with pytest.raises(ValueError):
        _ = calculate_engine_age(L_w)
