"""Tests related to the energy module."""

# %% IMPORT DEPENDENCIES
import pytest

from opentnsim.vessel import VesselProperties
from opentnsim.energy import ConsumesEnergy
from opentnsim.energy import sample_engine_age, calculate_max_sinkage


# %% FIXTURES


# %% TESTING sample_engine_age
@pytest.mark.parametrize("L_w", [1, 2, 3])
def test_sample_engine_age(L_w):
    """Test the sample_engine_age function."""
    age = sample_engine_age(L_w)
    assert isinstance(age, int), "Engine age should be a int."


@pytest.mark.parametrize("L_w", [None, 0, -1, 4, 100, "str", [], {}])
def test_sample_engine_age_infeasible_input(L_w):
    """Test the behavior sample_engine_age function with infeasible inputs."""
    with pytest.raises(ValueError):
        _ = sample_engine_age(L_w)


# %% TESTING calculate_max_sinkage
@pytest.mark.parametrize(
    "v,h_0,T,B,C_B,width,outcome",
    [
        (0, 2, 3, 4, 5, 6, 0.0),
        (1, 2, 3, 4, 5, 6, 0.99),
    ],
)
def test_calculate_max_sinkage(v, h_0, T, B, C_B, width, outcome):
    """Regression test for the calculate_max_sinkage function."""
    r = calculate_max_sinkage(v=v, h_0=h_0, T=T, B=B, C_B=C_B, width=width)
    if not r == pytest.approx(outcome, abs=1e-2):
        raise AssertionError(
            f"Expected {outcome}, but got {r} for v={v}, h_0={h_0}, T={T}, B={B}, C_B={C_B}, width={width}"
        )


@pytest.mark.parametrize(
    "v,h_0,T,B,C_B,width",
    [
        (-99, 1, 1, 1, 1, 1),
        (1, 0, 1, 1, 1, 1),
        (1, -99, 1, 1, 1, 1),
        (1, 1, 0, 1, 1, 1),
        (1, 1, -99, 1, 1, 1),
        (1, 1, 1, 0, 1, 1),
        (1, 1, 1, -99, 1, 1),
        (1, 1, 1, 1, 0, 1),
        (1, 1, 1, 1, -99, 1),
        (1, 1, 1, 1, 1, 0),
        (1, 1, 1, 1, 1, -99),
        (1, 1, 1, 10, 1, 1),  # width larger than B
    ],
)
def test_calculate_max_sinkage_wrong_input(v, h_0, T, B, C_B, width):
    """
    Test the behavior of calculate_max_sinkage if individual parameters receive
    wrong input.
    """
    with pytest.raises(ValueError):
        _ = calculate_max_sinkage(v=v, h_0=h_0, T=T, B=B, C_B=C_B, width=width)


# %% TESTING
