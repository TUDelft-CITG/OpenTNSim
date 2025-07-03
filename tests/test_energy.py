"""Tests related to the energy module."""

# %% IMPORT DEPENDENCIES
import pytest

from opentnsim.vessel import VesselProperties
from opentnsim.energy import ConsumesEnergy
from opentnsim.energy import (
    sample_engine_age,
    calculate_max_sinkage,
    calculate_properties,
)


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


# %% TESTING calculate_properties
def test_calculate_properties_bulbous_bow():
    """Test energy.calculate_properties."""
    # make a calculation with some values
    C_M, C_WP, C_P, delta, lcb, L_R, A_T, A_BT, S, S_APP, S_B, T_F, h_B = (
        calculate_properties(C_B=0.8, L=20, B=5, T=2, bulbous_bow=True, C_BB=0.7)
    )

    # check the outcome
    C_M == pytest.approx(0.99, abs=1e-2)
    C_WP == pytest.approx(0.86, abs=1e-2)
    C_P == pytest.approx(0.80, abs=1e-2)
    delta == pytest.approx(160, abs=1e-2)
    lcb == pytest.approx(2.12, abs=1e-2)
    L_R == pytest.approx(5.85, abs=1e-2)
    A_T == pytest.approx(1.0, abs=1e-2)
    A_BT == pytest.approx(6.96, abs=1e-2)
    S == pytest.approx(170.38, abs=1e-2)
    S_APP == pytest.approx(8.52, abs=1e-2)
    S_B == pytest.approx(100, abs=1e-2)
    T_F == pytest.approx(2.0, abs=1e-2)
    h_B == pytest.approx(0.4, abs=1e-2)


def test_calculate_properties_negative_block_coefficient():
    """Test calculate_properties with negative block coefficient."""
    with pytest.raises(Exception):
        _ = calculate_properties(C_B=-0.1, L=20, B=5, T=2, bulbous_bow=True, C_BB=0.7)


def test_calculate_properties_no_bulbous_bow():
    """Test calculate_properties without bulbous bow."""
    # make a calculation with some values
    C_M, C_WP, C_P, delta, lcb, L_R, A_T, A_BT, S, S_APP, S_B, T_F, h_B = (
        calculate_properties(C_B=0.8, L=20, B=5, T=2, bulbous_bow=False, C_BB=0.7)
    )

    # check the outcome
    C_M == pytest.approx(0.99, abs=1e-2)
    C_WP == pytest.approx(0.86, abs=1e-2)
    C_P == pytest.approx(0.80, abs=1e-2)
    delta == pytest.approx(160, abs=1e-2)
    lcb == pytest.approx(2.12, abs=1e-2)
    L_R == pytest.approx(5.85, abs=1e-2)
    A_T == pytest.approx(1.0, abs=1e-2)
    A_BT == pytest.approx(6.96, abs=1e-2)
    S == pytest.approx(149.69, abs=1e-2)
    S_APP == pytest.approx(7.48, abs=1e-2)
    S_B == pytest.approx(100, abs=1e-2)
    T_F == pytest.approx(2.0, abs=1e-2)
    h_B == pytest.approx(0.4, abs=1e-2)
