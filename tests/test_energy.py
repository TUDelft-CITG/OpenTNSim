"""Tests related to the energy module."""

# %% IMPORT DEPENDENCIES
import pytest
from shapely import Point

import opentnsim.fis
from opentnsim.core.vessel_properties import VesselProperties
from opentnsim.energy.mixins import ConsumesEnergy
from opentnsim.energy.mixins import (
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
        raise AssertionError(f"Expected {outcome}, but got {r} for v={v}, h_0={h_0}, T={T}, B={B}, C_B={C_B}, width={width}")


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
    C_M, C_WP, C_P, delta, lcb, L_R, A_T, A_BT, S, S_APP, S_B, T_F, h_B = calculate_properties(
        C_B=0.8, L=20, B=5, T=2, bulbous_bow=True, C_BB=0.7
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
    C_M, C_WP, C_P, delta, lcb, L_R, A_T, A_BT, S, S_APP, S_B, T_F, h_B = calculate_properties(
        C_B=0.8, L=20, B=5, T=2, bulbous_bow=False, C_BB=0.7
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


# @pytest.mark.skip(reason="Computation of Cf_Katsui encounters an invalid value in scalar power.")
def test_calculate_frictional_resistance():
    """Test the calculate_frictional_resistance function.
    Test case comse from running code, not from a paper.
    """
    # make a calculation
    R_f, C_f, R_e, Cf_deep, Cf_shallow, Cf_0, Cf_Katsui, V_B, D, a = calculate_frictional_resistance(
        v=3, h_0=4, L=50, nu=1.002e-6, T=3, S=120, S_B=100, rho=1000
    )

    # check the outcome
    # assert R_f == pytest.approx(None, abs=1e-2) --> NaN
    # assert C_f == pytest.approx(2.None, abs=1e-2) --> NaN
    assert R_e == pytest.approx(149700599, abs=1)
    assert Cf_deep == pytest.approx(0.00195, abs=1e-5)
    assert Cf_shallow == pytest.approx(0.00210, abs=1e-5)
    assert Cf_0 == pytest.approx(0.00196, abs=1e-5)
    # assert Cf_Katsui == pytest.approx(None, abs=1e-2) --> NaN
    assert V_B == pytest.approx(3.41, abs=1e-2)
    assert D == pytest.approx(1.0, abs=1e-2)
    assert a == pytest.approx(0.92, abs=1e-2)


@pytest.mark.parametrize("h_0,T", [(1, 1), (1, 2)])
def test_calculate_frictional_draught_h0_mismatch(h_0, T):
    """Test cases where h_0 - T <= 0"""
    with pytest.raises(Exception):
        _ = calculate_frictional_resistance(v=3, h_0=h_0, L=50, nu=1.0038, T=T, S=120, S_B=100, rho=1000)


@pytest.mark.skip(reason="Current implementation does allow S_B > S, " "however this seems an infeasible usecase.")
def test_calculate_frictional_resistance_SB_gt_S():
    """Test calculate_frictional_resistance with S_B > S."""
    with pytest.raises(Exception):
        _ = calculate_frictional_resistance(v=3, h_0=4, L=50, nu=1.0038, T=3, S=120, S_B=130, rho=1000)


# %% TESTING calculate_viscous_resistance
def test_calculate_viscous_resistance_1():
    """Test the calculate_viscous_resistance function."""
    # make a calculation
    # c_stern = 0 is used tshould lead to c_14 equal to 1
    c_14, one_k1, R_f_one_k1 = calculate_viscous_resistance(c_stern=0, B=5, L=40, T=2, L_R=4, C_P=0.5, R_f=1, delta=100)

    # check the outcome
    assert c_14 == 1.0
    assert one_k1 == pytest.approx(1.21, abs=1e-2)
    assert R_f_one_k1 == pytest.approx(1.21, abs=1e-2)


def test_calculate_viscous_resistance_2():
    """Test the calculate_viscous_resistance function."""
    # make a calculation
    c_14, one_k1, R_f_one_k1 = calculate_viscous_resistance(c_stern=10, B=5, L=40, T=2, L_R=4, C_P=0.5, R_f=1, delta=100)

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
# and perform appropriate checks on the function outcomes
@pytest.mark.parametrize(
    "v,T,F_rh,V_2,alpha_xx",
    [
        (3, 9, 0.3029, 3.1232, 0.9606),
        (3, 5, 0.3029, 3.0159, 0.9947),
        (3, 4, 0.3029, 3.0031, 0.999),
        (3, 2, 0.3029, 3.0, 1),
        (5, 9, 0.5048, 5.4655, 0.9148),
        (5, 5, 0.5048, 5.2393, 0.9543),
        (5, 4, 0.5048, 5.1342, 0.9739),
        (5, 3.3, 0.5048, 5.0322, 0.9936),
        (5, 2.5, 0.5048, 4.9938, 1.0012),
        (5, 2, 0.5048, 4.9338, 1.0134),
        (5, 1.6, 0.5048, 4.9401, 1.0121),
        (5, 1.4, 0.5048, 4.9402, 1.0121),
        (5, 1.2, 0.5048, 4.9294, 1.0143),
        (5, 1.1, 0.5048, 5.0, 1),
        (5, 1, 0.5048, 5.0, 1),
        (7, 1.1, 0.7067, 7.2014, 0.972),
        (7, 1, 0.7067, 7.1782, 0.9752),
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
    _F_rh, _V_2, _alpha_xx = karpov(v=v, h_0=10, g=9.81, T=T)

    # check the outcome
    assert _F_rh == pytest.approx(F_rh, abs=1e-4)
    assert _V_2 == pytest.approx(V_2, abs=1e-4)
    assert _alpha_xx == pytest.approx(alpha_xx, abs=1e-4)


# %% TESTING calculate_wave_resistance
@pytest.mark.parametrize(
    "B,L,delta,C_P,F_rL,i_E,c_1,c_2,c_5,c_7,c_15,c_16,lmbda,m_1,m_2,R_W",
    [
        (
            3,
            30,
            100,
            0.6,
            0.17,
            3.59,
            0.8,
            1,
            -7.89,
            0.1,
            -2.34,
            1.36,
            0.57,
            -1.97,
            -0.03,
            -0.46,
        ),
        (
            10,
            30,
            300,
            0.6,
            0.17,
            32.01,
            28.26,
            1,
            -1.67,
            0.31,
            -3.18,
            1.36,
            0.78,
            -3.21,
            -0.04,
            -0.03,
        ),
        (
            5,
            30,
            150,
            0.6,
            0.17,
            11.94,
            3.68,
            1,
            -4.33,
            0.17,
            -2.69,
            1.36,
            0.69,
            -2.33,
            -0.04,
            -0.33,
        ),
        (
            10,
            150,
            100,
            0.6,
            0.08,
            21.38,
            0.06,
            1,
            -1.67,
            0.07,
            0,
            1.36,
            0.51,
            -1.04,
            0.0,
            -0.0,
        ),
        (
            10,
            50,
            100,
            0.6,
            0.14,
            32.53,
            5.28,
            1,
            -1.67,
            0.2,
            -0.52,
            1.36,
            0.72,
            -2.25,
            -0.0,
            -0.01,
        ),
        (
            10,
            50,
            100,
            0.9,
            0.14,
            60.35,
            13.09,
            1,
            -1.67,
            0.2,
            -0.52,
            1.09,
            1.15,
            -1.98,
            -0.0,
            -0.13,
        ),
    ],
)
def test_calculate_wave_resistance(B, L, delta, C_P, F_rL, i_E, c_1, c_2, c_5, c_7, c_15, c_16, lmbda, m_1, m_2, R_W):
    """Test the calculate_wave_resistance function."""
    # cases represented in the parametrization:
    # - B / L < 0.11, B / L > 0.25, else
    # - (L**3) / delta < 512, (L**3) / delta > 1727, else
    # - C_P < 0.8, else
    # - L / B < 12, else

    # make a calculation
    _F_rL, _i_E, _c_1, _c_2, _c_5, _c_7, _c_15, _c_16, _lmbda, _m_1, _m_2, _R_W = calculate_wave_resistance(
        V_2=3,
        h_0=10,
        g=9.81,
        T=3,
        L=L,
        B=B,
        C_P=C_P,
        C_WP=0.8,
        lcb=0.5,
        L_R=20,
        A_T=50,
        C_M=0.5,
        delta=delta,
        rho=1000,
    )

    # check the outcome
    assert _F_rL == pytest.approx(F_rL, abs=1e-2)
    assert _i_E == pytest.approx(i_E, abs=1e-2)
    assert _c_1 == pytest.approx(c_1, abs=1e-2)
    assert _c_2 == pytest.approx(c_2, abs=1e-2)
    assert _c_5 == pytest.approx(c_5, abs=1e-2)
    assert _c_7 == pytest.approx(c_7, abs=1e-2)
    assert _c_15 == pytest.approx(c_15, abs=1e-2)
    assert _c_16 == pytest.approx(c_16, abs=1e-2)
    assert _lmbda == pytest.approx(lmbda, abs=1e-2)
    assert _m_1 == pytest.approx(m_1, abs=1e-2)
    assert _m_2 == pytest.approx(m_2, abs=1e-2)
    assert _R_W == pytest.approx(R_W, abs=1e-2)


# %% TESTING calculate_residual_resistance
@pytest.mark.parametrize(
    "bulbous_bow,T,F_nT, c_6, R_TR, c_4, c_2, C_A, R_A, F_ni, P_B, R_B, R_res",
    [
        (True, 1, 0.29, 0.19, 42.41, 0.02, 1, 0.0, 0.32, 1.04, -2.24, 0.0, 42.73),
        (False, 5, 0.29, 0.19, 42.41, 0.04, 1, 0.0, 0.29, 1.04, -2.24, 0, 42.7),
    ],
)
def test_calculate_residual_resistance(bulbous_bow, T, F_nT, c_6, R_TR, c_4, c_2, C_A, R_A, F_ni, P_B, R_B, R_res):
    """Test the calculate_residual_resistance function."""
    # cases
    # - bulbous_bow T/F, if False, then R_B=0
    # - T / L < 0.04, else
    # make a calculation
    _F_nT, _c_6, _R_TR, _c_4, _c_2, _C_A, _R_A, _F_ni, _P_B, _R_B, _R_res = calculate_residual_resistance(
        V_2=3,
        g=9.81,
        A_T=50,
        B=5,
        C_WP=0.8,
        rho=1000,
        T=T,
        L=50,
        C_B=0.8,
        S=100,
        T_F=1,
        h_B=1,
        A_BT=4,
        bulbous_bow=bulbous_bow,
    )

    # check the outcome
    assert _F_nT == pytest.approx(F_nT, abs=1e-2)
    assert _c_6 == pytest.approx(c_6, abs=1e-2)
    assert _R_TR == pytest.approx(R_TR, abs=1e-2)
    assert _c_4 == pytest.approx(c_4, abs=1e-2)
    assert _c_2 == pytest.approx(c_2, abs=1e-2)
    assert _C_A == pytest.approx(C_A, abs=1e-2)
    assert _R_A == pytest.approx(R_A, abs=1e-2)
    assert _F_ni == pytest.approx(F_ni, abs=1e-2)
    assert _P_B == pytest.approx(P_B, abs=1e-2)
    assert _R_B == pytest.approx(R_B, abs=1e-2)
    assert _R_res == pytest.approx(R_res, abs=1e-2)


# %% TESTING calculate_total_resistance

# %% TESTING calculate_total_power_required
