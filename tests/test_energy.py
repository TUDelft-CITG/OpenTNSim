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
    calculate_viscous_resistance,
    calculate_appendage_resistance,
    karpov,
    calculate_wave_resistance,
    calculate_residual_resistance,
    calculate_total_resistance,
    calculate_total_power_required,
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


@pytest.mark.skip(
    reason="Computation of Cf_Katsui encounters an invalid value in scalar power."
)
def test_calculate_frictional_resistance():
    """Test the calculate_frictional_resistance function."""
    # make a calculation
    R_f, C_f, R_e, Cf_deep, Cf_shallow, Cf_0, Cf_Katsui, V_B, D, a = (
        calculate_frictional_resistance(
            v=3, h_0=4, L=50, nu=1.002e-6, T=3, S=120, S_B=100, rho=1000
        )
    )

    # check the outcome
    # assert R_f == pytest.approx(None, abs=1e-2) --> NaN
    # assert C_f == pytest.approx(2.None, abs=1e-2) --> NaN
    assert R_e == pytest.approx(0.00015, abs=1e-5)
    assert Cf_deep == pytest.approx(0.00266, abs=1e-5)
    assert Cf_shallow == pytest.approx(0.00257, abs=1e-5)
    assert Cf_0 == pytest.approx(0.00221, abs=1e-5)
    # assert Cf_Katsui == pytest.approx(None, abs=1e-2) --> NaN
    assert V_B == pytest.approx(3.41, abs=1e-2)
    assert D == pytest.approx(1.0, abs=1e-2)
    assert a == pytest.approx(0.40, abs=1e-2)


@pytest.mark.parametrize("h_0,T", [(1, 1), (1, 2)])
def test_calculate_frictional_draught_h0_mismatch(h_0, T):
    """Test cases where h_0 - T <= 0"""
    with pytest.raises(Exception):
        _ = calculate_frictional_resistance(
            v=3, h_0=h_0, L=50, nu=1.0038, T=T, S=120, S_B=100, rho=1000
        )


@pytest.mark.skip(
    reason="Current implementation does allow S_B > S, "
    "however this seems an infeasible usecase."
)
def test_calculate_frictional_resistance_SB_gt_S():
    """Test calculate_frictional_resistance with S_B > S."""
    with pytest.raises(Exception):
        _ = calculate_frictional_resistance(
            v=3, h_0=4, L=50, nu=1.0038, T=3, S=120, S_B=130, rho=1000
        )


# %% TESTING calculate_viscous_resistance
def test_calculate_viscous_resistance_1():
    """Test the calculate_viscous_resistance function."""
    # make a calculation
    # c_stern = 0 is used tshould lead to c_14 equal to 1
    c_14, one_k1, R_f_one_k1 = calculate_viscous_resistance(
        c_stern=0, B=5, L=40, T=2, L_R=4, C_P=0.5, R_f=1, delta=100
    )

    # check the outcome
    assert c_14 == 1.0
    assert one_k1 == pytest.approx(1.21, abs=1e-2)
    assert R_f_one_k1 == pytest.approx(1.21, abs=1e-2)


def test_calculate_viscous_resistance_2():
    """Test the calculate_viscous_resistance function."""
    # make a calculation
    c_14, one_k1, R_f_one_k1 = calculate_viscous_resistance(
        c_stern=10, B=5, L=40, T=2, L_R=4, C_P=0.5, R_f=1, delta=100
    )

    # check the outcome
    assert c_14 == pytest.approx(1.011, abs=1e-3)
    assert one_k1 == pytest.approx(1.21, abs=1e-2)
    assert R_f_one_k1 == pytest.approx(1.21, abs=1e-2)


# %% TESTING calculate_appendage_resistance
def test_calculate_appendage_resistance():
    """Test the calculate_appendage_resistance function."""
    # make a calculation
    R_APP = calculate_appendage_resistance(v=3, rho=1000, S_APP=50, one_k2=1, C_f=1)

    assert R_APP == 225.0


# %% TESTING karpov
# TODO: replace None with the expected values for F_rh, V_2, and alpha_xx
# and perform appropriate checks on the function outcomes
@pytest.mark.parametrize(
    "v,T,F_rh,V_2,alpha_xx",
    [
        (3, 9, None, None, None),
        (3, 5, None, None, None),
        (3, 4, None, None, None),
        (3, 2, None, None, None),
        (5, 9, None, None, None),
        (5, 5, None, None, None),
        (5, 4, None, None, None),
        (5, 3.3, None, None, None),
        (5, 2.5, None, None, None),
        (5, 2, None, None, None),
        (5, 1.6, None, None, None),
        (5, 1.4, None, None, None),
        (5, 1.2, None, None, None),
        (5, 1.1, None, None, None),
        (5, 1, None, None, None),
        (7, 1.1, None, None, None),
        (7, 1, None, None, None),
    ],
)
def test_karpov(v, T, F_rh, V_2, alpha_xx):
    """Test the karpov function."""
    # F_rh = v / np.sqrt(g * h_0)
    # functionality distinguishes the following logic
    # - F_rh <= 0.4
    #   - 0 <= h_0 / T < 1.75
    #   - 1.75 <= h_0 / T < 2.25
    #   - 2.25 <= h_0 / T < 2.75
    #   - h_0 / T >= 2.75
    # - F_rh > 0.4
    #   - 0 <= h_0 / T < 1.75
    #   - 1.75 <= h_0 / T < 2.25
    #   - 2.25 <= h_0 / T < 2.75
    #   - 2.75 <= h_0 / T < 3.25
    #   - 3.25 <= h_0 / T < 3.75
    #   - 3.75 <= h_0 / T < 4.5
    #   - 4.5 <= h_0 / T < 5.5
    #   - 5.5 <= h_0 / T < 6.5
    #   - 6.5 <= h_0 / T < 7.5
    #   - 7.5 <= h_0 / T < 8.5
    #   - 8.5 <= h_0 / T < 9.5
    #        - F_rh < 0.6
    #        - F_rh >= 0.6
    #   - h_0 / T >= 9.5
    #        - F_rh < 0.6
    #        - F_rh >= 0.6
    #
    # test cases have been derived from these rules
    # v=3 --> F_rh <= 0.4
    # v=5 --> 0.4 <= F_rh <= 0.6
    # v=7 --> F_rh >= 0.6
    #
    # T=9 --> 0 <= h_0 / T < 1.75
    # T=5 --> 1.75 <= h_0 / T < 2.25
    # T=4 --> 2.25 <= h_0 / T < 2.75
    # T=3.3 --> 2.75 <= h_0 / T < 3.25
    # T=3 --> 3.25 <= h_0 / T < 3.75
    # T=2.5 --> 3.75 <= h_0 / T < 4.5
    # T=2 --> 4.5 <= h_0 / T < 5.5
    # T=1.6 --> 5.5 <= h_0 / T < 6.5
    # T=1.4 --> 6.5 <= h_0 / T < 7.5
    # T=1.2 --> 7.5 <= h_0 / T < 8.5
    # T=1.1 --> 8.5 <= h_0 / T < 9.5
    # T=1 --> h_0 / T >= 9.5

    # make a calculation
    F_rh, V_2, alpha_xx = karpov(v=v, h_0=10, g=9.81, T=T)


# %% TESTING calculate_wave_resistance


# %% TESTING calculate_residual_resistance


# %% TESTING calculate_total_resistance


# %% TESTING calculate_total_power_required
