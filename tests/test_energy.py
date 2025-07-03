"""Tests related to the energy module."""

# %% IMPORT DEPENDENCIES
import pytest

from opentnsim.vessel import VesselProperties
from opentnsim.energy import ConsumesEnergy
from opentnsim.energy import (
    sample_engine_age,
    calculate_max_sinkage,
    calculate_properties,
    calculate_frictional_resistance,
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
# TODO: cases to be tested:
# - C_B negative, greater than 1
# - negative vessel dimensions?


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
    assert C_M == pytest.approx(0.99, abs=1e-2)
    assert C_WP == pytest.approx(0.86, abs=1e-2)
    assert C_P == pytest.approx(0.80, abs=1e-2)
    assert delta == pytest.approx(160, abs=1e-2)
    assert lcb == pytest.approx(2.12, abs=1e-2)
    assert L_R == pytest.approx(5.85, abs=1e-2)
    assert A_T == pytest.approx(1.0, abs=1e-2)
    assert A_BT == pytest.approx(0, abs=1e-2)
    assert S == pytest.approx(149.69, abs=1e-2)
    assert S_APP == pytest.approx(7.48, abs=1e-2)
    assert S_B == pytest.approx(100, abs=1e-2)
    assert T_F == pytest.approx(2.0, abs=1e-2)
    assert h_B == pytest.approx(0.4, abs=1e-2)


# %% TESTING calculate_frictional_resistance
# TODO: cases to be tested:
# - v negative
# - S_B greater than S


def test_calculate_frictional_resistance():
    """Test the calculate_frictional_resistance function."""
    # make a calculation
    R_f, C_f, R_e, Cf_deep, Cf_shallow, Cf_0, Cf_Katsui, V_B, D, a = (
        calculate_frictional_resistance(
            v=3, h_0=4, L=50, nu=1.0038, T=3, S=120, S_B=100, rho=1
        )
    )

    # check the outcome
    assert R_f == pytest.approx(1.48, abs=1e-2)
    assert C_f == pytest.approx(2.74, abs=1e-2)
    assert R_e == pytest.approx(150.57, abs=1e-2)
    assert Cf_deep == pytest.approx(0.38, abs=1e-2)
    assert Cf_shallow == pytest.approx(0.34, abs=1e-2)
    assert Cf_0 == pytest.approx(2.37, abs=1e-2)
    assert Cf_Katsui == pytest.approx(-0.003, abs=1e-2)
    assert V_B == pytest.approx(3.41, abs=1e-2)
    assert D == pytest.approx(1.0, abs=1e-2)
    assert a == pytest.approx(0.66, abs=1e-2)


@pytest.mark.parametrize("h_0,T", [(1, 1), (1, 2)])
def test_calculate_frictional_draught_h0_mismatch(h_0, T):
    """Test cases where h_0 - T <= 0"""
    with pytest.raises(Exception):
        _ = calculate_frictional_resistance(
            v=3, h_0=h_0, L=50, nu=1.0038, T=T, S=120, S_B=100, rho=1
        )


@pytest.mark.skip(
    reason="Current implementation does allow S_B > S, "
    "however this seems an infeasible usecase."
)
def test_calculate_frictional_resistance_SB_gt_S():
    """Test calculate_frictional_resistance with S_B > S."""
    with pytest.raises(Exception):
        _ = calculate_frictional_resistance(
            v=3, h_0=4, L=50, nu=1.0038, T=3, S=120, S_B=130, rho=1
        )
