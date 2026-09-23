from __future__ import annotations

import pathlib
import logging
import warnings
import functools
import pyproj
import numpy as np
import pandas as pd
import scipy.optimize
from typing import Any


# OpenTNSim
import opentnsim
import opentnsim.strategy

from collections.abc import Mapping

logger = logging.getLogger(__name__)

KARPOV_ALPHA_MIN = 0.5   # chart minimum (conservative)
KARPOV_ALPHA_MAX = 1.0   # alpha_xx > 1 is nonphysical




def load_partial_engine_load_correction_factors():
    """read correction factor from package directory"""

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    correctionfactors_path = data_dir / "Correctionfactors.csv"
    df = pd.read_csv(correctionfactors_path, comment="#")

    return df


def karpov_smooth_curves():
    """read correction factor from package directory"""

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    karpov_smooth_curves_path = data_dir / "KarpovSmoothCurves.csv"
    df = pd.read_csv(karpov_smooth_curves_path, comment="#")

    return df


def find_closest_node(G, point):
    """find the closest node on the graph from a given point"""

    distance = np.full((len(G.nodes)), fill_value=np.nan)
    for ii, n in enumerate(G.nodes):
        distance[ii] = point.distance(G.nodes[n]["geometry"])
    name_node = list(G.nodes)[np.argmin(distance)]
    distance_node = np.min(distance)

    return name_node, distance_node


def _edge_hydraulic_value(edge, key):
    """Read a hydraulic value from an edge or its nested ``Info`` mapping."""

    if edge is None:
        return None
 
    value = edge.get(key, None)
    if value is not None:
        return value
 
    info = edge.get("Info", {})
    if isinstance(info, dict):
        return info.get(key, None)
 
    return None


# Restricted water (channel width). With confinement_mode "drawdown" or "full", ConsumesEnergy adds
# to the chain (Holtrop-Mennen, Zeng, Karpov):
#     R_confinement = c_f [C_F(V + V_R) rho/2 (V + V_R)^2 - C_F(V) rho/2 V^2] S + c_z rho g A_M Z
# The first term is the return-flow friction ("full" only), the second the drawdown term.
# Hydraulics: 1D continuity and Bernoulli (Schijf 1949; van de Kaa 1978, Eqs. 1-3; Spitzer 2021,
# Eqs. 8-13). Valid below the critical speed V_cr and with a positive clearance h - T - Z.
# van de Kaa 1978: 
# Spitzer 2021: https://ascelibrary.org/doi/10.1061/%28ASCE%29WW.1943-5460.0000672


RHO_FRESH_WATER = 1000.0      # water density [kg/m3]
G = 9.81                      # gravity [m/s2]
NU_WATER_15C = 1.139e-6       # kinematic viscosity of fresh water at 15 degC [m2/s]

C_GLOBAL_BARGE = 1.35         # Spitzer Eq. 25, barge set
C_P0_BARGE = 0.67
C_GLOBAL_MOTOR = 1.45         # Spitzer Eq. 21, motorvessel set
C_P0_MOTOR = 0.075
DELTA_CF_SPITZER = 0.0004     # roughness allowance, Spitzer (2021)
DELTA_CF_VAN_DE_KAA = 2.5e-4  # roughness allowance, van de Kaa (1978)
FROUDE_EC_M0 = 0.684          # Spitzer Eq. 19
C_P1_BARGE = 0.007           # Spitzer (2021), Eq. 14 barge set
C_P1_MOTOR = 0.04            # Spitzer (2021), Eq. 14 motor set

BASE_RESISTANCE_MODELS = ("auto", "holtrop", "spitzer")
W_BARGE, T_BARGE = 0.30, 0.20  # van de Kaa (1978) Sec. 4.2 and Eq. 24, loaded push tows (Luthra 1974)
W_MOTOR, T_MOTOR = 0.24, 0.27  # VBD Rep. 788 via Kulczyk and Tabaczek (2014) Table 2, motor vessel, mean of 4 tests/ Kulczyk and Tabaczek (2014): https://www.transnav.eu/Article_Coefficients_of_Propeller-hull_Kulczyk,31,520.html
CONFINEMENT_HULLS = ("barge", "barge_spitzer", "motor")
CONFINEMENT_MODES = ("none", "drawdown", "full")

class HydraulicSolutionError(RuntimeError):
    """No subcritical drawdown solution for this speed and section."""


def _pos(name, value):
    """Value as a positive, finite float."""
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite; received {value!r}.")
    return value


def _nonneg(name, value):
    """Value as a finite float that is not negative."""
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0; received {value!r}.")
    return value


def c_shallow_barge(h, T):
    """Shallow-water factor of a barge convoy, Spitzer Eq. 25. h is the depth alongside the vessel."""
    h = _pos("h", h)
    T = _pos("T", T)
    return 2.125 - 0.75 * (h / T) if h / T < 1.5 else 1.0


def c_shallow_motor(h, T):
    """Shallow-water factor of a motor ship, Spitzer Eq. 21. h is the depth alongside the vessel."""
    h = _pos("h", h)
    T = _pos("T", T)
    return 1.96 - 0.64 * (h / T) if h / T < 1.5 else 1.0


def c_friction_ittc(V_rel, L_friction, nu=NU_WATER_15C, delta_cf=DELTA_CF_SPITZER):
    """C_F, ITTC-1957 line with roughness allowance, Spitzer Eq. 4. V_rel = V + V_R."""
    V_rel = _pos("V_rel", V_rel)
    L_friction = _pos("L_friction", L_friction)
    nu = _pos("nu", nu)
    denominator = np.log10(V_rel * L_friction / nu) - 2.0
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        raise ValueError(f"Invalid ITTC-1957 denominator for V_rel={V_rel:.6g} and L={L_friction:.6g}.")
    return 0.075 / denominator ** 2 + float(delta_cf)


def economic_speed(h_mean, blockage_ratio, g=G):
    """Economic speed V_ec [m/s], Spitzer Eq. 19. h_mean = A_C / W, blockage m = A_M / A_C."""
    h_mean = _pos("h_mean", h_mean)
    m = float(blockage_ratio)
    if not 0.0 <= m < 1.0:
        raise ValueError(f"blockage_ratio must lie in [0, 1); received {m!r}.")
    return FROUDE_EC_M0 * (1.0 - m) ** 1.854 * np.sqrt(g * h_mean)


def critical_speed(h_mean, blockage_ratio, g=G):
    """Schijf critical speed V_cr [m/s], Spitzer Eq. 17."""
    h_mean = _pos("h_mean", h_mean)
    m = float(blockage_ratio)
    if not 0.0 <= m < 1.0:
        raise ValueError(f"blockage_ratio must lie in [0, 1); received {m!r}.")
    return np.sqrt(g * h_mean) * (2.0 * np.sin(np.arcsin(1.0 - m) / 3.0)) ** 1.5


def _solve_A_W(A_C, A_M, W, V, g, alpha):
    """Flow area A_W, the largest real root in (0, A_C - A_M] of the cubic form of Spitzer Eq. 9."""
    k = V * V / (2.0 * g)
    A_free = A_C - A_M
    roots = np.roots([1.0, -(A_free + W * k), 0.0, W * k * alpha * A_C ** 2])
    real = roots[np.abs(roots.imag) < 1e-8 * max(1.0, np.abs(roots.real).max())].real
    valid = real[(real > 0.0) & (real <= A_free * (1.0 + 1e-12))]
    if valid.size == 0:
        raise HydraulicSolutionError("No subcritical root: the speed is at or above the critical regime.")
    return float(min(valid.max(), A_free))


def drawdown_return_flow(V, h, B, T, W, channel_area_m2=None, midship_area_m2=None, mean_depth_m=None, alpha=1.0, g=G):
    """Drawdown Z and return flow V_R, Spitzer Eqs. 8, 9 and 13.

    V is the speed through the water, h the depth alongside the vessel, W the water-surface width.
    A_C defaults to W * h, A_M to B * T, the mean depth to A_C / W.
    """
    V = _nonneg("V", V)
    h = _pos("h", h)
    B = _pos("B", B)
    T = _pos("T", T)
    W = _pos("W", W)
    g = _pos("g", g)
    alpha = _pos("alpha", alpha)
    if h <= T:
        raise ValueError(f"Insufficient depth: h={h:.3f} m must exceed T={T:.3f} m.")
    if W <= B:
        raise ValueError(f"Width W={W:.3f} m must exceed the beam B={B:.3f} m.")

    A_C = W * h if channel_area_m2 is None else _pos("channel_area_m2", channel_area_m2)
    A_M = B * T if midship_area_m2 is None else _pos("midship_area_m2", midship_area_m2)
    h_mean = A_C / W if mean_depth_m is None else _pos("mean_depth_m", mean_depth_m)
    if A_M >= A_C:
        raise ValueError(f"Midship area A_M={A_M:.3f} m2 must be smaller than channel area A_C={A_C:.3f} m2.")
    m = A_M / A_C

    if V == 0.0:
        Z = 0.0
        V_R = 0.0
        A_W = A_C - A_M
        dynamic_ukc = h - T
    else:
        A_W = _solve_A_W(A_C, A_M, W, V, g, alpha)
        Z = (A_C - A_M - A_W) / W
        V_R = V * (np.sqrt(1.0 + 2.0 * g * Z / V ** 2) - 1.0)
        dynamic_ukc = h - T - Z
        if not np.isfinite(V_R) or V_R < 0.0:
            raise HydraulicSolutionError(f"Invalid return velocity V_R={V_R!r}.")
        if dynamic_ukc <= 0.0:
            raise HydraulicSolutionError(f"Non-positive dynamic UKC {dynamic_ukc:.4f} m at h={h:.3f} m, "
                                         f"T={T:.3f} m, Z={Z:.4f} m.")

    return {"Z_m": Z, "V_R_ms": V_R, "blockage_ratio": m, "dynamic_ukc_m": dynamic_ukc,
            "channel_area_m2": A_C, "midship_area_m2": A_M, "available_flow_area_m2": A_W,
            "water_depth_m": h, "mean_depth_m": h_mean}


def default_hull(vessel_type):
    """Coefficient set of a vessel_type: "Barge" gives "barge", every other type "motor"."""
    return "barge" if str(vessel_type).strip().lower() == "barge" else "motor"


def increment_coefficients(hull, h, T):
    """Coefficients c_f, c_z and delta_cf of the increment, with C_Shallow

    - "barge": van de Kaa (1978) Eq. 21, thus c_f = c_z = 1
    - "motor": Spitzer (2021) Eq. 21, thus c_f = 1.45 C_Shallow and c_z = 0.075 c_f
    - "barge_spitzer": Spitzer Eq. 25
    """
    if hull == "barge":
        return {"c_f": 1.0, "c_z": 1.0, "delta_cf": DELTA_CF_VAN_DE_KAA, "C_Shallow": 1.0}
    if hull == "barge_spitzer":
        c_sh = c_shallow_barge(h, T)
        scale = C_GLOBAL_BARGE * c_sh
        return {"c_f": scale, "c_z": C_P0_BARGE * scale, "delta_cf": DELTA_CF_SPITZER, "C_Shallow": c_sh}
    if hull == "motor":
        c_sh = c_shallow_motor(h, T)
        scale = C_GLOBAL_MOTOR * c_sh
        return {"c_f": scale, "c_z": C_P0_MOTOR * scale, "delta_cf": DELTA_CF_SPITZER, "C_Shallow": c_sh}
    raise ValueError(f"Unknown hull {hull!r}; use one of {CONFINEMENT_HULLS}.")


def _section(h, B, T, W, channel_area_m2=None, midship_area_m2=None, mean_depth_m=None):
    """Return A_C, A_M, the mean depth A_C / W and the blockage m, as drawdown_return_flow does."""
    h = _pos("h", h)
    B = _pos("B", B)
    T = _pos("T", T)
    W = _pos("W", W)
    if h <= T:
        raise ValueError(f"Insufficient depth: h={h:.3f} m must exceed T={T:.3f} m.")
    if W <= B:
        raise ValueError(f"Width W={W:.3f} m must exceed the beam B={B:.3f} m.")
    A_C = W * h if channel_area_m2 is None else _pos("channel_area_m2", channel_area_m2)
    A_M = B * T if midship_area_m2 is None else _pos("midship_area_m2", midship_area_m2)
    h_mean = A_C / W if mean_depth_m is None else _pos("mean_depth_m", mean_depth_m)
    if A_M >= A_C:
        raise ValueError(f"Midship area A_M={A_M:.3f} m2 must be smaller than channel area A_C={A_C:.3f} m2.")
    return A_C, A_M, h_mean, A_M / A_C


def hydraulic_speed_limits(h, B, T, W, channel_area_m2=None, midship_area_m2=None,
                           mean_depth_m=None, g=G):
    """V_cr and V_ec of the section, from the mean depth A_C / W and the blockage A_M / A_C."""
    A_C, A_M, h_mean, m = _section(h, B, T, W, channel_area_m2, midship_area_m2, mean_depth_m)
    return {"V_cr_ms": float(critical_speed(h_mean, m, g=g)),
            "V_ec_ms": float(economic_speed(h_mean, m, g=g)),
            "blockage_ratio": m, "mean_depth_m": h_mean,
            "channel_area_m2": A_C, "midship_area_m2": A_M}


def max_speed_for_ukc(h, B, T, W, ukc_min=0.01, channel_area_m2=None, midship_area_m2=None,
                      mean_depth_m=None, alpha=1.0, g=G, v_tol=1e-4):
    """Highest speed with dynamic clearance h - T - Z >= ukc_min, by bisection below 0.999 V_cr."""
    ukc_min = float(ukc_min)
    if h - T <= ukc_min:
        return 0.0
    lim = hydraulic_speed_limits(h, B, T, W, channel_area_m2=channel_area_m2,
                                 midship_area_m2=midship_area_m2, mean_depth_m=mean_depth_m, g=g)
    kw = dict(channel_area_m2=channel_area_m2, midship_area_m2=midship_area_m2,
              mean_depth_m=mean_depth_m, alpha=alpha, g=g)

    def ok(v):
        try:
            return drawdown_return_flow(v, h, B, T, W, **kw)["dynamic_ukc_m"] >= ukc_min
        except HydraulicSolutionError:
            return False

    hi = 0.999 * lim["V_cr_ms"]
    if ok(hi):
        return float(hi)
    lo = 0.0
    while hi - lo > v_tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return float(lo)


def confinement_resistance_increment(V, h, L, B, T, W, S, hull="motor", mode="full",
                                     channel_area_m2=None, midship_area_m2=None, mean_depth_m=None,
                                     alpha=1.0, rho=RHO_FRESH_WATER, g=G, nu=NU_WATER_15C,
                                     delta_cf=None, c_f=None, c_z=None):
    """Resistance increment of a channel of finite width, in N and kN

    - V: speed through the water; L, B, T: length, beam and real draught; W: water-surface width
    - S: wetted surface
    - hull: "barge", "motor" or "barge_spitzer"; mode: "drawdown" or "full"
    - delta_cf, c_f, c_z: overrides of the coefficient set, for sensitivity runs

    The result also carries the hydraulics (Z, V_R, m, dynamic clearance) and the speed limits
    for the event table. A speed at or above V_cr, or a clearance of zero, raises
    HydraulicSolutionError.
    """
    if mode not in ("drawdown", "full"):
        raise ValueError(f"Unknown mode {mode!r}; use 'drawdown' or 'full' (ConsumesEnergy handles 'none').")
    V = _nonneg("V", V)
    L = _pos("L", L)
    S = _pos("S", S)
    rho = _pos("rho", rho)
    g = _pos("g", g)
    nu = _pos("nu", nu)

    coef = increment_coefficients(hull, h, T)
    k_f = coef["c_f"] if c_f is None else float(c_f)
    k_z = coef["c_z"] if c_z is None else float(c_z)
    dcf = coef["delta_cf"] if delta_cf is None else float(delta_cf)

    hyd = drawdown_return_flow(V, h, B, T, W, channel_area_m2=channel_area_m2,
                               midship_area_m2=midship_area_m2, mean_depth_m=mean_depth_m,
                               alpha=alpha, g=g)
    V_R = float(hyd["V_R_ms"])
    Z = float(hyd["Z_m"])
    A_M = float(hyd["midship_area_m2"])
    h_mean = float(hyd["mean_depth_m"])
    m = float(hyd["blockage_ratio"])

    if V == 0.0:
        C_F_channel = C_F_open = float("nan")
        dR_friction = 0.0
        dR_drawdown = 0.0
    else:
        C_F_channel = float(c_friction_ittc(V + V_R, L, nu=nu, delta_cf=dcf))
        C_F_open = float(c_friction_ittc(V, L, nu=nu, delta_cf=dcf))
        dR_friction = k_f * 0.5 * rho * S * (C_F_channel * (V + V_R) ** 2 - C_F_open * V ** 2)
        dR_drawdown = k_z * rho * g * A_M * Z
    if mode == "drawdown":
        dR_friction = 0.0
    dR = dR_friction + dR_drawdown

    V_cr = float(critical_speed(h_mean, m, g=g))
    V_ec = float(economic_speed(h_mean, m, g=g))

    return {"confinement_model": f"confinement_{hull}", "hull": hull, "mode": mode,
            "dR_N": dR, "dR_kN": dR / 1000.0, "dR_friction_N": dR_friction, "dR_drawdown_N": dR_drawdown,
            "c_f": k_f, "c_z": k_z, "delta_cf": dcf, "C_Shallow": coef["C_Shallow"],
            "C_F_channel": C_F_channel, "C_F_open": C_F_open,
            "V_R_ms": V_R, "Z_m": Z, "V_relative_ms": V + V_R, "blockage_ratio": m,
            "dynamic_ukc_m": float(hyd["dynamic_ukc_m"]), "channel_area_m2": float(hyd["channel_area_m2"]),
            "midship_area_m2": A_M, "water_depth_m": float(hyd["water_depth_m"]), "mean_depth_m": h_mean,
            "h_over_T": float(hyd["water_depth_m"]) / float(T),
            "V_cr_ms": V_cr, "V_ec_ms": V_ec, "V_over_V_cr": V / V_cr, "V_over_V_ec": V / V_ec,
            "econ_speed_exceeded": bool(V > V_ec)}


def spitzer_resistance(V, h, L, B, T, hull="barge", W=None, channel_area_m2=None,
                       rho=RHO_FRESH_WATER, g=G, nu=NU_WATER_15C):
    """Spitzer (2021) Eq. 14 in N, alpha = 1, box wetted surface S = LB + 2T(L + B).

    R = C_Global C_Shallow [C_F(V + V_R) rho/2 (V + V_R)^2 S + C_P0 rho g B T Z + C_P1 rho/2 V^2 B T]
    W None gives open water (Z = V_R = 0). R_open is the same equation with Z = V_R = 0.
    """
    if h <= T:
        raise ValueError(f"Insufficient depth: h={h:.3f} m must exceed T={T:.3f} m.")
    if hull == "barge":
        C_G, C_P0, C_P1, C_sh = C_GLOBAL_BARGE, C_P0_BARGE, C_P1_BARGE, c_shallow_barge(h, T)
    else:
        C_G, C_P0, C_P1, C_sh = C_GLOBAL_MOTOR, C_P0_MOTOR, C_P1_MOTOR, c_shallow_motor(h, T)
    S = L * B + 2.0 * T * (L + B)
    hyd = None if W is None else drawdown_return_flow(V, h, B, T, W, channel_area_m2=channel_area_m2, g=g)
    Z, V_R = (0.0, 0.0) if hyd is None else (hyd["Z_m"], hyd["V_R_ms"])

    k = C_G * C_sh
    R_pressure = k * C_P1 * 0.5 * rho * V ** 2 * B * T
    R_friction = k * c_friction_ittc(V + V_R, L, nu=nu) * 0.5 * rho * (V + V_R) ** 2 * S
    R_open = k * c_friction_ittc(V, L, nu=nu) * 0.5 * rho * V ** 2 * S + R_pressure
    R = R_friction + k * C_P0 * rho * g * B * T * Z + R_pressure
    return {"R_N": R, "R_open_N": R_open, "R_friction_N": R_friction, "R_pressure_N": R_pressure,
            "C_Shallow": C_sh, "Z_m": Z, "V_R_ms": V_R, "hydraulics": hyd}


def power2v(vessel, edge, upperbound):
    """Compute vessel speed for a prescribed total engine-power setting.

    The resistance calculation is delegated to ``vessel.calculate_resistance_for_waterway``.
    ``confinement_mode="none"`` (default): Barrass squat depth, then Holtrop/Zeng/Karpov.
    ``"drawdown"`` or ``"full"``: raw depth, Holtrop/Zeng/Karpov and the channel-width increment.
    """

    assert isinstance(vessel, opentnsim.vessel.VesselProperties), "vessel should be an instance of VesselProperties"
    assert vessel.C_B is not None, "C_B cannot be None"
 
    h_raw = _edge_hydraulic_value(edge, "GeneralDepth")
    waterway_width = _edge_hydraulic_value(edge, "GeneralWidth")
    # When the real wetted area is available, the channel hydraulics use h for C_Shallow/UKC and A_C/W for V_cr/V_ec.
    # Absent -> A_C = W*h

    channel_area = _edge_hydraulic_value(edge, "GeneralCrossSectionArea")
    if channel_area is not None and (not np.isfinite(channel_area) or channel_area <= 0):
        channel_area = None
 
    if h_raw is None or not np.isfinite(h_raw) or h_raw <= 0:
        raise ValueError(f"A positive GeneralDepth is required for power2v; received {h_raw!r}.")
 
    # confinement_mode "drawdown" / "full": the limit stays below the critical speed and the dynamic-UKC speed. "none": the input, unchanged.
    upperbound = vessel.limit_power2v_upperbound(upperbound=upperbound, h_0=float(h_raw), width=waterway_width, channel_area=channel_area,)

    if not np.isfinite(upperbound) or upperbound <= 0:
        raise ValueError(f"Invalid power2v upper bound: {upperbound!r}.")


    def seek_v_given_power(v):
        """function to optimize"""
        # TODO: check it this needs to be made more general, now relies on ['Info'] to be present
        # water depth from the edge

        if v <= 0:
            return np.inf
        
        # h_0 = edge["Info"]["GeneralDepth"]
        # try:
        #     h_0 = vessel.calculate_h_squat(v, h_0)
        # except AttributeError:
        #     # no squat available
        #     pass
        # TODO: consider precomputing a range v/h combinations for the ship before the simulation starts
        #vessel.calculate_total_resistance(v, h_0)

        vessel.calculate_resistance_for_waterway(v=float(v), h_0=float(h_raw), width=waterway_width, channel_area=channel_area,)

        vessel.calculate_total_power_required(v=float(v), h_0=vessel.h_resistance)

        if isinstance(vessel.P_tot, complex):
            raise ValueError(f"P_tot is complex: {vessel.P_tot}")
        diff = float(vessel.P_tot_given) - float(vessel.P_tot)   

        logger.debug("optimizing for v=%s, P_tot_given=%s, P_tot=%s, P_given=%s",v, vessel.P_tot_given, vessel.P_tot, getattr(vessel, "P_given", np.nan),)
        return diff**2
 
    fit = scipy.optimize.minimize_scalar(seek_v_given_power, bounds=(1e-6, float(upperbound)), method="bounded", options=dict(xatol=1e-7),)
 
    if not fit.success:
        raise ValueError(fit)
 
    logger.debug("fit: %s", fit)
    return float(fit.x)



class ConsumesEnergy:
    """Mixin class: Something that consumes energy.

    Keyword arguments:

    - P_installed: installed engine power [kW]
    - P_tot_given: Total power set by captain (includes hotel power). When P_tot_given > P_installed; P_tot_given=P_installed.
    - bulbous_bow: inland ships generally do not have a bulbous_bow, set to False (default). If a ship has a bulbous_bow, set to True.
    - L_w: weight class of the ship (depending on carrying capacity) (classes: L1 (=1), L2 (=2), L3 (=3))
    - current_year: current year
    - nu: kinematic viscosity [m^2/s]
    - rho: density of the surrounding water [kg/m^3]
    - g: gravitational accelleration [m/s^2]
    - x: number of propellers [-]
    - eta_o: open water efficiency of propeller [-]
    - eta_r: relative rotative efficiency [-]
    - eta_t: transmission efficiency [-]
    - eta_g: gearing efficiency [-]
    - c_stern: determines shape of the afterbody [-]
    - C_BB: breadth coefficient of bulbous_bow, set to 0.2 according to the paper of Kracht (1970), https://doi.org/10.5957/jsr.1970.14.1.1
    - C_B: block coefficient ('fullness') [-] (default to 0.85)
    - one_k2: appendage resistance factor (1+k2) [-]
    - C_year: construction year of the engine [y]
    - D_s: propeller diameter [m]; None gives 0.7 T, the rule of Segers (2021) Appendix C
    - wake_fraction, thrust_deduction: fixed w and t [-]. None gives 0.30 and 0.20 for vessel_type
        "Barge" (van de Kaa 1978) and 0.24 and 0.27 for motor vessels (VBD Rep. 788)
    - confinement_mode: "none" (default) keeps the original resistance, R_tot = R_base.
      "drawdown" adds the drawdown term, "full" also the return-flow friction. Both need the
      waterway width (edge GeneralWidth) and use GeneralCrossSectionArea when present; both
      evaluate the chain at the raw depth, with Barrass squat as a navigation diagnostic.
    - confinement_hull: "barge", "motor", "barge_spitzer", or None for the set of vessel_type
    - confinement_speed_cap: "critical" (default) or "economic"; confinement_ukc_min: clearance [m]
    - squat_in_resistance: True gives the squat depth to the chain (double count), default False
    - confinement_c_f, confinement_c_z, confinement_delta_cf: coefficient overrides for sensitivity
    """

    # Resistance model name for event tables (the event table also records confinement_mode).
    resistance_model = "holtrop_zeng_karpov"

    @property
    def requires_waterway_width(self):
        """True when the resistance needs the real waterway width (confinement_mode "drawdown" or "full")."""
        return getattr(self, "confinement_mode", "none") != "none"

    def __init__(
        self,
        P_installed,
        L_w,
        C_year,
        current_year=None,  # current_year
        engine_age_seed=None,  # PATCH P8: seed for the Weibull engine-age draw
        bulbous_bow=False,
        P_hotel_perc=0.05,
        P_hotel=None,
        P_tot_given=None,  # the actual power engine setting
        nu=1 * 10 ** (-6),
        rho=1000,
        g=9.81,
        x=2,
        D_s=None,
        eta_o=0.4,
        eta_r=1.00,
        eta_t=0.98,
        eta_g=0.96,
        c_stern=0,
        C_BB=0.2,
        C_B=0.85,
        one_k2=2.5, # following Segers (2021) we assume (1 + k2) to be 2.5 (see below Eq 3.27)
        karpov_correction=True,
        wake_fraction=None,
        thrust_deduction=None,
        confinement_mode="none",
        confinement_hull=None,
        confinement_speed_cap="critical",
        confinement_ukc_min=0.01,
        squat_in_resistance=False,
        confinement_c_f=None,
        confinement_c_z=None,
        confinement_delta_cf=None,
        base_resistance="auto",
        *args,
        **kwargs,
        
    ):
        
        super().__init__(*args, **kwargs)

        """Initialization"""

        self.P_installed = P_installed
        self.bulbous_bow = bulbous_bow

        # Required power for systems on board, "5%" based on De Vos and van Gils (2011): Walstroom versus generator stroom
        self.P_hotel_perc = P_hotel_perc

        if P_hotel is not None:  # if P_hotel is specified use the given value
            self.P_hotel = P_hotel
        else:  # if P_hotel is None calculate it from P_hotel_percentage and P_installed
            self.P_hotel = self.P_hotel_perc * self.P_installed

        self.P_tot_given = P_tot_given
        self.L_w = L_w
        self.year = current_year
        self.engine_age_seed = engine_age_seed
        self.nu = nu
        self.rho = rho
        self.g = g
        self.x = x
        self.D_s = D_s
        self.eta_o = eta_o
        self.eta_r = eta_r
        self.eta_t = eta_t
        self.eta_g = eta_g
        self.c_stern = c_stern
        self.C_BB = C_BB
        self.C_B = C_B
        self.karpov_correction = karpov_correction 
        self.one_k2 = one_k2

        # Optional fixed propulsion factors; None uses the Segers (2021) relations.
        self.wake_fraction = None if wake_fraction is None else float(wake_fraction)
        self.thrust_deduction = None if thrust_deduction is None else float(thrust_deduction)
        
        # Restricted water (channel width); "none" prevents
        if confinement_mode not in CONFINEMENT_MODES:
            raise ValueError(f"confinement_mode must be one of {CONFINEMENT_MODES}; received {confinement_mode!r}.")
        if confinement_speed_cap not in ("critical", "economic"):
            raise ValueError(f"confinement_speed_cap must be 'critical' or 'economic'; received {confinement_speed_cap!r}.")
        self.confinement_mode = confinement_mode
        self.confinement_hull = confinement_hull          # None: the set of vessel_type
        self.confinement_speed_cap = confinement_speed_cap
        self.confinement_ukc_min = float(confinement_ukc_min)
        self.squat_in_resistance = bool(squat_in_resistance)
        self.confinement_c_f = confinement_c_f            # None: the value of the coefficient set
        self.confinement_c_z = confinement_c_z
        self.confinement_delta_cf = confinement_delta_cf
        if base_resistance not in BASE_RESISTANCE_MODELS:
            raise ValueError(f"base_resistance must be one of {BASE_RESISTANCE_MODELS}; received {base_resistance!r}.")
        self.base_resistance = base_resistance
        
        # plugin function that computes velocity based on power
        self.power2v = power2v

        # TODO: C_year is obligatory, so why is this code here?
        if C_year:
            self.C_year = C_year
        else:
            self.C_year = self.calculate_engine_age()

        if self.P_tot_given is not None and self.P_installed is not None:
            if P_tot_given > P_installed:
                self.P_tot_given = self.P_installed


    def calculate_engine_age(self):
        """Calculating the construction year of the engine, dependent on a Weibull function with
        shape factor 'k', and scale factor 'lmb', which are determined by the weight class L_w
        """

        # Determining which shape and scale factor to use, based on the weight class L_w = L1, L2 or L3
        assert self.L_w in [1, 2, 3], "Invalid value L_w, should be 1,2 or 3"
        if self.L_w == 1:  # Weight class L1
            self.k = 1.3
            self.lmb = 20.4
        elif self.L_w == 2:  # Weight class L2
            self.k = 1.12
            self.lmb = 18.5
        elif self.L_w == 3:  # Weight class L3
            self.k = 1.26
            self.lmb = 18.6

        # The age of the engine (PATCH P8: deterministic when seeded; prefer
        # supplying C_year directly from the fleet table)
        if self.year is None:
            raise ValueError(
                "current_year must be set to derive an engine construction year; "
                "prefer supplying C_year directly from the fleet table."
            )
        if self.engine_age_seed is None:
            warnings.warn(
                "Engine age drawn without a seed: diesel emission results will not "
                "be reproducible across runs. Supply C_year or engine_age_seed.",
                UserWarning,
            )
        rng = np.random.default_rng(self.engine_age_seed)
        self.age = int(rng.weibull(self.k) * self.lmb)

        # Construction year of the engine
        self.C_year = self.year - self.age

        logger.debug(f"The construction year of the engine is {self.C_year}")

        return self.C_year


    def calculate_properties(self):
        """Calculate a number of basic vessel properties"""

        # TODO: add properties for seagoing ships with bulbs

        # (Van Koningsveld et al (2023) - Part IV Eq 5.9, 5.10 and below Eq 5.12)
        self.C_M = 1.006 - 0.0056 * self.C_B ** (-3.56)  # Midship section coefficient (Eq 5.9)
        self.C_WP = (1 + 2 * self.C_B) / 3  # Waterplane coefficient (Eq 5.10)
        self.C_P = self.C_B / self.C_M  # Prismatic coefficient (see below Eq 5.12)

        # Segers (2021) (http://resolver.tudelft.nl/uuid:a260bc48-c6ce-4f7c-b14a-e681d2e528e3)
        # Appendix C - Eq C.2
        self.delta = self.C_B * self.L * self.B * self.T  # Water displacement

        # Van Koningsveld et al (2023) - Part IV Table 5.1
        self.lcb = -13.5 + 19.4 * self.C_P  # longitudinal center of buoyancy
        # Van Koningsveld et al (2023) - Part IV Eq 5.13
        self.L_R = self.L * (
            1 - self.C_P + (0.06 * self.C_P * self.lcb) / (4 * self.C_P - 1)
        )  # length parameter reflecting the length of the run

        # Van Koningsveld et al (2023) - below Eq 5.16
        self.A_T = 0.1 * self.B * self.T  # transverse area of the transom
        # calculation for A_BT (cross-sectional area of the bulb at still water level [m^2]) depends on whether a ship has a bulb
        if self.bulbous_bow:
            # TODO: check Holtrop and Mennen for this formulation
            self.A_BT = self.C_BB * self.B * self.T * self.C_M  # calculate A_BT for seagoing ships having a bulb
        else:
            self.A_BT = 0  # most inland ships do not have a bulb. So we assume A_BT=0.

        # Total wet area: S (Van Koningsveld et al (2023) - Eq 5.8)
        assert self.C_M >= 0, f"C_M should be positive: {self.C_M}"
        self.S = self.L * (2 * self.T + self.B) * np.sqrt(self.C_M) * (
            0.453 + 0.4425 * self.C_B - 0.2862 * self.C_M - 0.003467 * (self.B / self.T) + 0.3696 * self.C_WP
        ) + 2.38 * (self.A_BT / self.C_B)

        # Segers (2021) (http://resolver.tudelft.nl/uuid:a260bc48-c6ce-4f7c-b14a-e681d2e528e3)
        # In the explanation under Eq 3.27
        self.S_APP = 0.05 * self.S  # Wet area of appendages
        # Segers (2021) Eq 3.20
        self.S_B = self.L * self.B  # Area of flat bottom

        # TODO: check references for these equations
        self.T_F = self.T  # Forward draught of the vessel [m]
        self.h_B = 0.2 * self.T  # Position of the centre of the transverse area [m]


    def calculate_frictional_resistance(self, v, h_0):
        """Frictional resistance

        - 1st resistance component defined by Holtrop and Mennen (1982)
        - A modification to the original friction line is applied, based on literature of Zeng (2018), to account for shallow water effects
        """
        Th = getattr(self, "T_hydro", None) or self.T
        self.R_e = v * self.L / self.nu  # Reynolds number

        self.D = h_0 - Th  # distance from bottom ship to the bottom of the fairway
        if not self.D > 0:
            raise ValueError(
                f"Under-keel clearance must be > 0 for vessel "
                f"{getattr(self, 'name', getattr(self, 'id', '?'))}: "
                f"D={self.D:.3f} m (h_0={h_0:.3f} m, T={Th:.3f} m)"
            )

        # Friction coefficient based on CFD computations of Zeng et al. (2018), in deep water
        # Van Koningsveld et al (2023) - Eq 5.3
        self.Cf_deep = 0.08169 / ((np.log10(self.R_e) - 1.717) ** 2)
        assert not isinstance(self.Cf_deep, complex), f"Cf_deep should not be complex: {self.Cf_deep}"

        # Friction coefficient based on CFD computations of Zeng et al. (2018), taking into account shallow water effects
        # Van Koningsveld et al (2023) - Eq 5.4
        self.Cf_shallow = (0.08169 / ((np.log10(self.R_e) - 1.717) ** 2)) * (
            1 + (0.003998 / (np.log10(self.R_e) - 4.393)) * (self.D / self.L) ** (-1.083)
        )
        assert not isinstance(self.Cf_shallow, complex), f"Cf_shallow should not be complex: {self.Cf_shallow}"

        # Friction coefficient in deep water according to ITTC-1957 curve
        # Van Koningsveld et al (2023) - Eq 5.6
        self.Cf_0 = 0.075 / ((np.log10(self.R_e) - 2) ** 2)

        # 'a' is the coefficient needed to calculate the Katsui friction coefficient
        # Van Koningsveld et al (2023) - below Eq 5.7
        self.a = 0.042612 * np.log10(self.R_e) + 0.56725
        # Van Koningsveld et al (2023) - Eq 5.7
        self.Cf_Katsui = 0.0066577 / ((np.log10(self.R_e) - 4.3762) ** self.a)

        # The average velocity underneath the ship, taking into account the shallow water effect
        # This calculation is to get V_B, which will be used in the following Cf for shallow water equation:
        if h_0 / Th <= 4:
            self.V_B = 0.4277 * v * np.exp((h_0 / Th ) ** (-0.07625))
        else:
            self.V_B = v

        # cf_shallow and cf_deep cannot be applied directly, since a vessel also has non-horizontal wet surfaces that have to be taken
        # into account. Therefore, the following formula for the final friction coefficient 'C_f' for deep water or shallow water is
        # defined according to Zeng et al. (2018)

        if (h_0 - Th) / self.L > 1:
            # calculate Friction coefficient C_f for deep water:
            # Zeng et al. (2018)
            self.C_f = self.Cf_0 + (self.Cf_deep - self.Cf_Katsui) * (self.S_B / self.S)
            logger.debug("now i am in the deep loop")
        else:
            # calculate Friction coefficient C_f for shallow water:
            # Van Koningsveld et al (2023) - Eq 5.5
            self.C_f = self.Cf_0 + (self.Cf_shallow - self.Cf_Katsui) * (self.S_B / self.S) * (self.V_B / v) ** 2
            logger.debug("now i am in the shallow loop")
        assert not isinstance(self.C_f, complex), f"C_f should not be complex: {self.C_f}"

        # The total frictional resistance R_f [kN]:
        # Van Koningsveld et al (2023) - Eq 5.2
        self.R_f = (0.5 * self.rho * (v**2) * self.C_f *  self.S) / 1000
        assert not isinstance(self.R_f, complex), f"R_f should not be complex: {self.R_f}"


    def calculate_viscous_resistance(self):
        """Viscous resistance

        - 2nd resistance component defined by Holtrop and Mennen (1982)
        - Form factor (1 + k1) has to be multiplied by the frictional resistance R_f, to account for the effect of viscosity"""

        # c_14 accounts for the specific shape of the afterbody
        # TODO: check where this value comes from (Holtrop and Mennen?) (following Segers (2021) we assume c_stern = 0 which leads to c_14 to be 1
        self.c_14 = 1 + 0.0011 * self.c_stern

        # the form factor (1+k1) describes the viscous resistance
        # Van Koningsveld et al (2023) - Eq 5.12
        # TODO: consider to rename self.delta to self.nabla
        self.one_k1 = 0.93 + 0.487 * self.c_14 * ((self.B / self.L) ** 1.068) * ((self.T / self.L) ** 0.461) * (
            (self.L / self.L_R) ** 0.122
        ) * (((self.L**3) / self.delta) ** 0.365) * ((1 - self.C_P) ** (-0.604))

        self.R_f_one_k1 = self.R_f * self.one_k1


    def calculate_appendage_resistance(self, v):
        """Appendage resistance

        - 3rd resistance component defined by Holtrop and Mennen (1982)
        - Appendages (like a rudder, shafts, skeg) result in additional frictional resistance"""

        # Frictional resistance resulting from wetted area of appendages: R_APP [kN]
        # Segers (2021) - Eq 3.27 (http://resolver.tudelft.nl/uuid:a260bc48-c6ce-4f7c-b14a-e681d2e528e3)
        self.R_APP = (0.5 * self.rho * (v**2) * self.S_APP * self.one_k2 * self.C_f) / 1000


    def karpov(self, v, h_0):
        """Intermediate calculation: Karpov

        - The Karpov method computes a velocity correction that accounts for limited water depth (corrected velocity V2,
          expressed as "Vs + delta_V" in the paper), but it also can be used for deeper water depth (h_0 / T >= 9.5).
        - V2 has to be implemented in the wave resistance (R_W) and the residual resistance terms (R_res: R_TR, R_A, R_B)
        """

        # The Froude number used in the Karpov method is the depth related froude number F_rh

        # The different alpha** curves are determined with a sixth power polynomial approximation in Excel
        # A distinction is made between different ranges of Froude numbers, because this resulted in a better approximation of the curve
        Th = getattr(self, "T_hydro", None) or self.T
        assert self.g >= 0, f"g should be positive: {self.g}"
        assert h_0 >= 0, f"h_0 should be positive: {h_0}"
        self.F_rh = v / np.sqrt(self.g * h_0)

        alpha_xx = 1.0
 
        if self.F_rh <= 0.4:
            if 0 <= h_0 /Th < 1.75:
                alpha_xx = (-4 * 10 ** (-12)) * self.F_rh**3 - 0.2143 * self.F_rh**2 - 0.0643 * self.F_rh + 0.9997
            if 1.75 <=  h_0 / Th< 2.25:
                alpha_xx = -0.8333 * self.F_rh**3 + 0.25 * self.F_rh**2 - 0.0167 * self.F_rh + 1
            if 2.25 <= h_0 / Th < 2.75:
                alpha_xx = -1.25 * self.F_rh**4 + 0.5833 * self.F_rh**3 - 0.0375 * self.F_rh**2 - 0.0108 * self.F_rh + 1
            if h_0 / Th >= 2.75:
                alpha_xx = 1
    
        if self.F_rh > 0.4:
            if 0 <= h_0 / Th < 1.75:
                alpha_xx = (
                    -0.9274 * self.F_rh**6 + 9.5953 * self.F_rh**5 - 37.197 * self.F_rh**4
                    + 69.666 * self.F_rh**3 - 65.391 * self.F_rh**2 + 28.025 * self.F_rh - 3.4143
                )
            if 1.75 <= h_0 / Th < 2.25:
                alpha_xx = (
                    2.2152 * self.F_rh**6 - 11.852 * self.F_rh**5 + 21.499 * self.F_rh**4
                    - 12.174 * self.F_rh**3 - 4.7873 * self.F_rh**2 + 5.8662 * self.F_rh - 0.2652
                )
            if 2.25 <= h_0 / Th < 2.75:
                alpha_xx = (
                    1.2205 * self.F_rh**6 - 5.4999 * self.F_rh**5 + 5.7966 * self.F_rh**4
                    + 6.6491 * self.F_rh**3 - 16.123 * self.F_rh**2 + 9.2016 * self.F_rh - 0.6342
                )
            if 2.75 <= h_0 / Th < 3.25:
                alpha_xx = (
                    -0.4085 * self.F_rh**6 + 4.534 * self.F_rh**5 - 18.443 * self.F_rh**4
                    + 35.744 * self.F_rh**3 - 34.381 * self.F_rh**2 + 15.042 * self.F_rh - 1.3807
                )
            if 3.25 <= h_0 / Th < 3.75:
                alpha_xx = (
                    0.4078 * self.F_rh**6 - 0.919 * self.F_rh**5 - 3.8292 * self.F_rh**4
                    + 15.738 * self.F_rh**3 - 19.766 * self.F_rh**2 + 9.7466 * self.F_rh - 0.6409
                )
            if 3.75 <= h_0 / Th < 4.5:
                alpha_xx = (
                    0.3067 * self.F_rh**6 - 0.3404 * self.F_rh**5 - 5.0511 * self.F_rh**4
                    + 16.892 * self.F_rh**3 - 20.265 * self.F_rh**2 + 9.9002 * self.F_rh - 0.6712
                )
            if 4.5 <= h_0 / Th < 5.5:
                alpha_xx = (
                    0.3212 * self.F_rh**6 - 0.3559 * self.F_rh**5 - 5.1056 * self.F_rh**4
                    + 16.926 * self.F_rh**3 - 20.253 * self.F_rh**2 + 10.013 * self.F_rh - 0.7196
                )
            if 5.5 <= h_0 / Th < 6.5:
                alpha_xx = (
                    0.9252 * self.F_rh**6 - 4.2574 * self.F_rh**5 + 5.0363 * self.F_rh**4
                    + 3.3282 * self.F_rh**3 - 10.367 * self.F_rh**2 + 6.3993 * self.F_rh - 0.2074
                )
            if 6.5 <= h_0 / Th < 7.5:
                alpha_xx = (
                    0.8442 * self.F_rh**6 - 4.0261 * self.F_rh**5 + 5.313 * self.F_rh**4
                    + 1.6442 * self.F_rh**3 - 8.1848 * self.F_rh**2 + 5.3209 * self.F_rh - 0.0267
                )
            if 7.5 <= h_0 / Th < 8.5:
                alpha_xx = (
                    0.1211 * self.F_rh**6 + 0.628 * self.F_rh**5 - 6.5106 * self.F_rh**4
                    + 16.7 * self.F_rh**3 - 18.267 * self.F_rh**2 + 8.7077 * self.F_rh - 0.4745
                )
            if 8.5 <= h_0 / Th < 9.5:
                if self.F_rh < 0.6:
                    alpha_xx = 1
                if self.F_rh >= 0.6:
                    alpha_xx = (
                        -6.4069 * self.F_rh**6 + 47.308 * self.F_rh**5 - 141.93 * self.F_rh**4
                        + 220.23 * self.F_rh**3 - 185.05 * self.F_rh**2 + 79.25 * self.F_rh - 12.484
                    )
            if h_0 / Th >= 9.5:
                if self.F_rh < 0.6:
                    alpha_xx = 1
                if self.F_rh >= 0.6:
                    alpha_xx = (
                        -6.0737 * self.F_rh**6 + 44.97 * self.F_rh**5 - 135.21 * self.F_rh**4
                        + 210.13 * self.F_rh**3 - 176.72 * self.F_rh**2 + 75.728 * self.F_rh - 11.893
                    )
    
       
        self.karpov_alpha_raw = float(alpha_xx)
        alpha_clamped = min(KARPOV_ALPHA_MAX, max(float(alpha_xx), KARPOV_ALPHA_MIN))
        self.karpov_clamped = bool(alpha_clamped != self.karpov_alpha_raw)
        self.alpha_xx = alpha_clamped
    
        if not self.karpov_correction:     
            self.alpha_xx = 1.0           


        self.V_2 = v / self.alpha_xx



    def calculate_wave_resistance(self, v, h_0):
        """Wave resistance

        - 4th resistance component defined by Holtrop and Mennen (1982)
        - When the speed or the vessel size increases, the wave making resistance increases
        - In shallow water, the wave resistance shows an asymptotical behaviour by reaching the critical speed
        """

        self.karpov(v, h_0)

        v_eff = self.V_2
        assert self.g >= 0, f"g should be positive: {self.g}"
        assert self.L >= 0, f"L should be positive: {self.L}"


        self.F_rL = v_eff / np.sqrt(self.g * self.L)  # Froude number based on ship's speed to water and its length of waterline

        # parameter c_7 is determined by the B/L ratio
        # Van Koningsveld et al (2023) - Part IV Table 5.1
        if self.B / self.L < 0.11:
            self.c_7 = 0.229577 * (self.B / self.L) ** 0.33333
        elif self.B / self.L > 0.25:
            self.c_7 = 0.5 - 0.0625 * (self.L / self.B)
        else:
            self.c_7 = self.B / self.L

        # 1 - C_P - 0.0225 lcb must stay positive; it reaches zero near C_B = 0.906.
        entrance_base = 1 - self.C_P - 0.0225 * self.lcb
        if not entrance_base > 0:
            raise ValueError(f"Holtrop-Mennen wave resistance is not defined for C_B={self.C_B:.4f}; "
                             f"1 - C_P - 0.0225 lcb = {entrance_base:.4f}. Use a lower C_B, for example 0.88.")
        
        # half angle of entrance in degrees
        # Van Koningsveld et al (2023) - Part IV Table 5.1
        self.i_E = 1 + 89 * np.exp(
            -((self.L / self.B) ** 0.80856)
            * ((1 - self.C_WP) ** 0.30484)
            * ((1 - self.C_P - 0.0225 * self.lcb) ** 0.6367)
            * ((self.L_R / self.B) ** 0.34574)
            * ((100 * self.delta / (self.L**3)) ** 0.16302)
        )

        # Van Koningsveld et al (2023) - Part IV Table 5.1
        self.c_1 = 2223105 * (self.c_7**3.78613) * ((self.T / self.B) ** 1.07961) * (90 - self.i_E) ** (-1.37165)
        self.c_2 = 1  # accounts for the effect of the bulbous bow, which is not present at inland ships
        self.c_5 = 1 - (0.8 * self.A_T) / (self.B * self.T * self.C_M)  # influence of the transom stern on the wave resistance

        # parameter c_15 depoends on the ratio L^3 / delta
        # Van Koningsveld et al (2023) - Part IV Table 5.1
        if (self.L**3) / self.delta < 512:
            self.c_15 = -1.69385
        elif (self.L**3) / self.delta > 1727:
            self.c_15 = 0
        else:
            self.c_15 = -1.69385 + (self.L / (self.delta ** (1 / 3)) - 8) / 2.36

        # parameter c_16 depends on C_P
        # Van Koningsveld et al (2023) - Part IV Table 5.1
        if self.C_P < 0.8:
            self.c_16 = 8.07981 * self.C_P - 13.8673 * (self.C_P**2) + 6.984388 * (self.C_P**3)
        else:
            self.c_16 = 1.73014 - 0.7067 * self.C_P

        if self.L / self.B < 12:
            self.lmbda = 1.446 * self.C_P - 0.03 * (self.L / self.B)
        else:
            self.lmbda = 1.446 * self.C_P - 0.36

        # Van Koningsveld et al (2023) - Part IV Table 5.1
        self.m_1 = (
            0.0140407 * (self.L / self.T) - 1.75254 * ((self.delta) ** (1 / 3) / self.L) - 4.79323 * (self.B / self.L) - self.c_16
        )
        # Van Koningsveld et al (2023) - Part IV Table 5.1
        self.m_2 = self.c_15 * (self.C_P**2) * np.exp((-0.1) * (self.F_rL ** (-2)))

        # Van Koningsveld et al (2023) - Part IV Eq 5.16
        self.R_W = (
            self.c_1
            * self.c_2
            * self.c_5
            * self.delta
            * self.rho
            * self.g
            * np.exp(self.m_1 * (self.F_rL ** (-0.9)) + self.m_2 * np.cos(self.lmbda * (self.F_rL ** (-2))))
            / 1000
        )  # kN


    def calculate_residual_resistance(self, v, h_0):
        """Residual resistance terms

        - Holtrop and Mennen (1982) defined three residual resistance terms:
        - 1) Resistance due to immersed transom (R_TR), Karpov corrected velocity V2 is used
        - 2) Resistance due to model-ship correlation (R_A), Karpov corrected velocity V2 is used
        - 3) Resistance due to the bulbous bow (R_B), Karpov corrected velocity V2 is used
        """

        self.karpov(v, h_0)

        self.F_rL = self.V_2 / np.sqrt(self.g * self.L)

        # Resistance due to immersed transom: R_TR [kN]
        self.F_nT = self.V_2 / np.sqrt(
        2 * self.g * self.A_T / (self.B + self.B * self.C_WP)
        )
        assert not isinstance(self.F_nT, complex), f"residual? froude number should not be complex: {self.F_nT}"

        self.c_6 = 0.2 * (1 - 0.2 * self.F_nT)

        self.R_TR = (0.5 * self.rho * (self.V_2**2) * self.A_T * self.c_6) / 1000

        # Model-ship correlation resistance: R_A [kN]

        if self.T / self.L < 0.04:
            self.c_4 = self.T / self.L
        else:
            self.c_4 = 0.04
        self.c_2 = 1

        self.C_A = (
            0.006 * (self.L + 100) ** (-0.16)
            - 0.00205
            + 0.003 * np.sqrt(self.L / 7.5) * (self.C_B**4) * self.c_2 * (0.04 - self.c_4)
        )
        assert not isinstance(self.C_A, complex), f"C_A number should not be complex: {self.C_A}"

        self.R_A = (0.5 * self.rho * (self.V_2**2) * self.S * self.C_A) / 1000  # kW

        # Resistance due to the bulbous bow (R_B)

        # Froude number based on immersoin of bulbous bow [-]
        self.F_ni = self.V_2 / np.sqrt(self.g * (self.T_F - self.h_B - 0.25 * np.sqrt(self.A_BT) + 0.15 * self.V_2**2))

        self.P_B = (0.56 * np.sqrt(self.A_BT)) / (self.T_F - 1.5 * self.h_B)  # P_B is coefficient for the emergence of bulbous bow
        if self.bulbous_bow:
            self.R_B = (
                (0.11 * np.exp(-3 * self.P_B**2) * self.F_ni**3 * self.A_BT**1.5 * self.rho * self.g) / (1 + self.F_ni**2)
            ) / 1000
        else:
            self.R_B = 0

        self.R_res = self.R_TR + self.R_A + self.R_B


    def calculate_total_resistance(self, v, h_0, width=None, channel_area=None, h_channel=None):
        """Total resistance:

        The total resistance is the sum of all resistance components (Holtrop and Mennen, 1982)
        """
        if self.base_resistance_used() == "spitzer":
            return self._spitzer_total_resistance(v, h_0 if h_channel is None else h_channel,
                                                  width=width, channel_area=channel_area)

        self.resistance_model = "holtrop_zeng_karpov"

        self.calculate_properties()
        self.calculate_frictional_resistance(v, h_0)
        self.calculate_viscous_resistance()
        self.calculate_appendage_resistance(v)
        self.calculate_wave_resistance(v, h_0)
        self.calculate_residual_resistance(v, h_0)

        # The total resistance R_tot [kN] = R_f * (1+k1) + R_APP + R_W + R_TR + R_A
        self.R_base = self.R_f * self.one_k1 + self.R_APP + self.R_W + self.R_TR + self.R_A + self.R_B
        
        # Restricted-water (channel-width) increment [kN]; 0.0 with confinement_mode "none".
        self.R_confinement = self.calculate_confinement_resistance(
            v=v, h_0=h_0 if h_channel is None else h_channel, width=width, channel_area=channel_area,
        )
        self.R_tot = self.R_base + self.R_confinement
        return self.R_tot


    def base_resistance_used(self):
        """"spitzer" or "holtrop". With "auto", vessel_type "Barge" gives "spitzer", every other type "holtrop"."""

        choice = getattr(self, "base_resistance", "auto")
        if choice == "auto":
            return "spitzer" if default_hull(getattr(self, "vessel_type", None)) == "barge" else "holtrop"
        return choice


    def _spitzer_total_resistance(self, v, h, width=None, channel_area=None):
        """Spitzer Eq. 14 at the raw depth h. R_base: open water; R_confinement: the channel part."""
        self.calculate_properties()          # displacement for the propulsion step
        T = self._hydraulic_draught()
        mode = getattr(self, "confinement_mode", "none")
        W = self._required_width(width) if mode != "none" else None
        hull = default_hull(getattr(self, "vessel_type", None))
        res = spitzer_resistance(float(v), float(h), self.L, self.B, T, hull=hull, W=W,
                                 channel_area_m2=channel_area, rho=self.rho, g=self.g, nu=self.nu)
        self.resistance_model = f"spitzer_eq14_{hull}"
        self.F_rL = v / np.sqrt(self.g * self.L)
        self.C_Shallow = res["C_Shallow"]
        self.R_f = res["R_friction_N"] / 1000.0
        self.R_base = res["R_open_N"] / 1000.0
        self.R_tot = res["R_N"] / 1000.0
        self.R_confinement = self.R_tot - self.R_base

        hyd = res["hydraulics"]
        self.confinement_result = None
        if hyd is not None:
            lim = hydraulic_speed_limits(h, self.B, T, W, channel_area_m2=channel_area, g=self.g)
            self.Z, self.V_R, self.V_cr, self.V_ec = res["Z_m"], res["V_R_ms"], lim["V_cr_ms"], lim["V_ec_ms"]
            self.confinement_result = {
                **hyd, "V_relative_ms": v + self.V_R, "V_cr_ms": self.V_cr, "V_ec_ms": self.V_ec,
                "V_over_V_cr": v / self.V_cr, "V_over_V_ec": v / self.V_ec, "econ_speed_exceeded": v > self.V_ec}
        return self.R_tot

    def calculate_confinement_resistance(self, v, h_0, width=None, channel_area=None):
        """Channel-width resistance increment R_confinement in kN
    
        With "none" the increment is zero. With "drawdown" or "full" it comes from
        confinement_resistance_increment, with the wetted surface S of the chain and the real draught.
        The hydraulics (Z, V_R, V_cr, clearance) stay on the vessel for the event table.
        """
        if getattr(self, "confinement_mode", "none") == "none":
            return 0.0
    
        W = self._required_width(width)
        result = confinement_resistance_increment(
            V=float(v),
            h=float(h_0),
            L=self.L,
            B=self.B,
            T=self._hydraulic_draught(),
            W=W,
            S=self.S,                     
            hull=self.confinement_hull_used(),
            mode=self.confinement_mode,
            channel_area_m2=None if channel_area is None else float(channel_area),
            rho=self.rho,
            g=self.g,
            nu=self.nu,
            delta_cf=self.confinement_delta_cf,
            c_f=self.confinement_c_f,
            c_z=self.confinement_c_z,
            )
        self.confinement_result = result
    
        self.R_confinement_friction = result["dR_friction_N"] / 1000.0
        self.R_confinement_drawdown = result["dR_drawdown_N"] / 1000.0
        self.C_Shallow = result["C_Shallow"]
        self.Z = result["Z_m"]
        self.V_R = result["V_R_ms"]
        self.blockage_ratio = result["blockage_ratio"]
        self.dynamic_ukc = result["dynamic_ukc_m"]
        self.mean_depth = result["mean_depth_m"]
        self.V_cr = result["V_cr_ms"]
        self.V_ec = result["V_ec_ms"]
        self.V_over_V_cr = result["V_over_V_cr"]
        self.V_over_V_ec = result["V_over_V_ec"]
        self.econ_speed_exceeded = result["econ_speed_exceeded"]
    
        return result["dR_kN"]

    
    def calculate_total_power_required(self, v, h_0):
        """Total required power:

        - The total required power is the sum of the power for systems on board (P_hotel) + power required for
          propulsion
        - The power required for propulsion depends on the calculated resistance

        Output:
        - P_propulsion: required power for propulsion, equals to P_d (Delivered Horse Power)
        - P_tot: required power for propulsion and hotelling
        - P_given: the power given by the engine to the ship (for propulsion and hotelling), which is the actual power
          the ship uses

        Note:
        PATCH P4: In this version P_propulsion, P_tot and P_given are BRAKE power,
        because P_installed is a brake rating. P_d (Delivered Horse Power) is kept
        as an attribute: the event-table pipeline multiplies conversion-efficiency
        (Marin) SFCs by shaft energy (P_d + P_hotel), which prevents double use of
        the same power efficiencies.
        The details are
        1) The P_b calculation involves gearing efficiency and transmission efficiency already while P_d not.
        2) P_d is the power delivered to propellers.
        3) To estimate the renewable fuel use, we will involve "energy conversion efficiencies" later in the
           calculation.
        The 'energy conversion efficiencies' for renewable fuel powered vessels are commonly measured/given as a whole
        covering the engine power systems, includes different engine (such as fuel cell engine, battery engine, internal
        combustion engine, hybrid engine) efficiencies, and corresponding gearbox efficiencies, AC/DC converter
        efficiencies, excludes the efficiency items of propellers.
        Therefore, to align with the later use of "energy conversion efficiencies" for fuel use estimation and prevent
        double use of some power efficiencies such as gearing efficiency, here we choose P_d as propulsion power.
        """

        # Required power for propulsion
        # Effective Horse Power (EHP), P_e (Van Koningsveld et al (2023) - Part IV Eq 5.17)
        self.P_e = v * self.R_tot

        # Segers (2021) (http://resolver.tudelft.nl/uuid:a260bc48-c6ce-4f7c-b14a-e681d2e528e3)
        # Appendix C
        if self.F_rL < 0.2:
            self.dw = 0    # the velocity correction coefficient is 0 when FrL is smaller than 0.2
        else:
            self.dw = 0.1  # otherwise the velocity correction coefficient is 0.1

        # Segers (2021) Appendix C: D_s = 0.7 T, then Eq. C.1 for w
        self.D_s_used = self.D_s if self.D_s is not None else 0.7 * self.T
        self.w = 0.11 * (0.16 / self.x) * self.C_B * np.sqrt(self.delta ** (1 / 3) / self.D_s_used) - self.dw


        # Measured pair per hull family, since Eq. C.1 gives w near 0.02 for full inland hulls
        barge = default_hull(getattr(self, "vessel_type", None)) == "barge"
        w_pair, t_pair = (W_BARGE, T_BARGE) if barge else (W_MOTOR, T_MOTOR)
        w_fixed = self.wake_fraction if self.wake_fraction is not None else w_pair
        t_fixed = self.thrust_deduction if self.thrust_deduction is not None else (
            t_pair if self.wake_fraction is None else None)
        
        if w_fixed is not None:
            self.w = w_fixed

        if self.x == 1:
            # (Van Koningsveld et al (2023) - Part IV Eq 5.22)
            self.t = 0.6 * self.w * (1 + 0.67 * self.w)  # thrust deduction factor 't'
        else:
            # (Van Koningsveld et al (2023) - Part IV Eq 5.23)
            self.t = 0.8 * self.w * (1 + 0.25 * self.w)

        if t_fixed is not None:
            self.t = t_fixed
        self.eta_h = (1 - self.t) / (1 - self.w)  # hull efficiency eta_h

        # TODO: check below suggestions. They were made to allow for better translation to alternative energy sources. But the changes induced unexpected behaviour.
        # Calculation hydrodynamic efficiency eta_D  according to Simic et al (2013) "On Energy Efficiency of Inland
        # Waterway Self-Propelled Cargo Vessels", https://www.researchgate.net/publication/269103117
        # hydrodynamic efficiency eta_D is a ratio of power used to propel the ship and delivered power
        # relation between eta_D and ship velocity v

        # if h_0 >= 9:
        #     if self.F_rh >= 0.5:
        #         self.eta_D = 0.6
        #     elif 0.325 <= self.F_rh < 0.5:
        #         self.eta_D = 0.7
        #     elif 0.28 <= self.F_rh < 0.325:
        #         self.eta_D = 0.59
        #     elif 0.2 < self.F_rh < 0.28:
        #         self.eta_D = 0.56
        #     elif 0.17 < self.F_rh <= 0.2:
        #         self.eta_D = 0.41
        #     elif 0.15 < self.F_rh <= 0.17:
        #         self.eta_D = 0.35
        #     else:
        #         self.eta_D = 0.29
        #
        # elif 5 <= h_0 < 9:
        #     if self.F_rh > 0.62:
        #         self.eta_D = 0.7
        #     elif 0.58 < self.F_rh < 0.62:
        #         self.eta_D = 0.68
        #     elif 0.57 < self.F_rh <= 0.58:
        #         self.eta_D = 0.7
        #     elif 0.51 < self.F_rh <= 0.57:
        #         self.eta_D = 0.68
        #     elif 0.475 < self.F_rh <= 0.51:
        #         self.eta_D = 0.53
        #     elif 0.45 < self.F_rh <= 0.475:
        #         self.eta_D = 0.4
        #     elif 0.36 < self.F_rh <= 0.45:
        #         self.eta_D = 0.37
        #     elif 0.33 < self.F_rh <= 0.36:
        #         self.eta_D = 0.36
        #     elif 0.3 < self.F_rh <= 0.33:
        #         self.eta_D = 0.35
        #     elif 0.28 < self.F_rh <= 0.3:
        #         self.eta_D = 0.331
        #     else:
        #         self.eta_D = 0.33
        # else:
        #     if self.F_rh > 0.56:
        #         self.eta_D = 0.28
        #     elif 0.4 < self.F_rh <= 0.56:
        #         self.eta_D = 0.275
        #     elif 0.36 < self.F_rh <= 0.4:
        #         self.eta_D = 0.345
        #     elif 0.33 < self.F_rh <= 0.36:
        #         self.eta_D = 0.28
        #     elif 0.3 < self.F_rh <= 0.33:
        #         self.eta_D = 0.27
        #     elif 0.28 < self.F_rh <= 0.3:
        #         self.eta_D = 0.26
        #     else:
        #         self.eta_D = 0.25
        #
        # # Delivered Horse Power (DHP), P_d
        # self.P_d = self.P_e / self.eta_D

        # logger.debug("eta_D = {:.2f}".format(self.eta_D))

        # (Van Koningsveld et al (2023) - Part IV Eq 5.19)
        self.P_d = self.P_e / (self.eta_o * self.eta_r * self.eta_h)

        # Brake Horse Power (BHP), P_b (P_b was used in OpenTNsim version v1.1.2. we do not use it in this version. The reseaon is listed in the doc string above)
        # (Van Koningsveld et al (2023) - Part IV Eq 5.24)
        self.P_b = self.P_d / (self.eta_t * self.eta_g)

        # self.P_propulsion = self.P_d  # propulsion power is defined here as Delivered horse power, the power delivered to propellers
        # PATCH P4: capping runs on the brake side (P_installed is a brake rating);
        # P_d stays available for the shaft-basis fuel accounting in the event table
        self.P_propulsion = self.P_b

        # TODO: consider to facilitate that all engine power can go into propulsion (Auxiliary generator for hotel)
        self.P_tot = self.P_hotel + self.P_propulsion

        # Partial engine load (P_partial): needed in the 'Emission calculations'
        if self.P_tot > self.P_installed:
            self.P_given = self.P_installed
            self.P_partial = 1
        else:
            self.P_given = self.P_tot
            self.P_partial = self.P_tot / self.P_installed

        logger.debug(f'The total power required is {self.P_tot} kW')
        logger.debug(f'The actual total power given is {self.P_given} kW')
        logger.debug(f'The partial load is {self.P_partial}')

        assert not isinstance(self.P_given, complex), f"P_given number should not be complex: {self.P_given}"

        # return these three variables:
        # 1) self.P_propulsion, for the convience of validation.  (propulsion power and fuel used for propulsion),
        # 2) self.P_tot, know the required power, especially when it exceeds installed engine power while sailing shallower and faster
        # 3) self.P_given, the actual power the engine gives for "propulsion + hotel" within its capacity (means installed power). This varible is used for calculating delta_energy of each sailing time step.

        return self.P_given

    def emission_factors_general(self):
        """General emission factors:

        This function computes general emission factors, based on construction year of the engine.
        - Based on literature TNO (2019)

        Please note: later on a correction factor has to be applied to get the total emission factor
        """

        # The general emission factors of CO2, PM10 and NOX are based on the construction year of the engine

        if self.C_year < 1974:
            self.EF_CO2 = 756
            self.EF_PM10 = 0.6
            self.EF_NOX = 10.8
        if 1975 <= self.C_year <= 1979:
            self.EF_CO2 = 730
            self.EF_PM10 = 0.6
            self.EF_NOX = 10.6
        if 1980 <= self.C_year <= 1984:
            self.EF_CO2 = 714
            self.EF_PM10 = 0.6
            self.EF_NOX = 10.4
        if 1985 <= self.C_year <= 1989:
            self.EF_CO2 = 698
            self.EF_PM10 = 0.5
            self.EF_NOX = 10.1
        if 1990 <= self.C_year <= 1994:
            self.EF_CO2 = 698
            self.EF_PM10 = 0.4
            self.EF_NOX = 10.1
        if 1995 <= self.C_year <= 2002:
            self.EF_CO2 = 650
            self.EF_PM10 = 0.3
            self.EF_NOX = 9.4
        if 2003 <= self.C_year <= 2007:
            self.EF_CO2 = 635
            self.EF_PM10 = 0.3
            self.EF_NOX = 9.2
        if 2008 <= self.C_year <= 2019:
            self.EF_CO2 = 635
            self.EF_PM10 = 0.2
            self.EF_NOX = 7
        if self.C_year > 2019:
            if self.L_w == 1:
                self.EF_CO2 = 650
                self.EF_PM10 = 0.1
                self.EF_NOX = 2.1
            else:
                self.EF_CO2 = 603
                self.EF_PM10 = 0.015
                self.EF_NOX = 1.8

        logger.debug(f"The general emission factor of CO2 is {self.EF_CO2} g/kWh")
        logger.debug(f"The general emission factor of PM10 is {self.EF_PM10} g/kWh")
        logger.debug(f"The general emission factor CO2 is {self.EF_NOX} g/kWh")

    def energy_density(self):
        """net energy density of diesel and renewable energy sources. This will be used for calculating SFC later.

        - Edens_xx_mass: net gravimetric energy density, which is the amount of energy stored in a given energy source in mass [kWh/kg].
        - Edens_xx_vol: net volumetric energy density, which is the amount of energy stored in a given energy source in volume [kWh/m3].


        Data source:
        Table 3-2 from Marin report 2019,  Energietransitie emissieloze binnenvaart, vooronderzoek ontwerpaspecten, systeem configuraties.(Energy transition zero-emission inland shipping, preliminary research on design aspects, system configurations

        Note:
        net energy density can be used for calculate fuel consumption in mass and volume, but for required energy source storage space determination, the packaging factors of different energy sources also need to be considered.
        """

        # gravimetric net energy density
        self.Edens_diesel_mass = 11.67 / 1000  # kWh/kg
        self.Edens_LH2_mass = 33.3 / 1000  # kWh/kg
        self.Edens_eLNG_mass = 13.3 / 1000  # kWh/kg
        self.Edens_eMethanol_mass = 5.47 / 1000  # kWh/kg
        self.Edens_eNH3_mass = 5.11 / 1000  # kWh/kg
        self.Edens_Li_NMC_Battery_mass = 0.11 / 1000  # kWh/kg

        # volumetric net energy density
        self.Edens_diesel_vol = 9944  # kWh/m3
        self.Edens_LH2_vol = 2556  # kWh/m3
        self.Edens_eLNG_vol = 5639  # kWh/m3
        self.Edens_eMethanol_vol = 4333  # kWh/m3
        self.Edens_eNH3_vol = 3139  # kWh/m3
        self.Edens_Li_NMC_Battery_vol = 139  # kWh/m3

    def energy_conversion_efficiency(self):
        """energy efficiencies for combinations of different energy source and energy-power conversion systems, including engine and power plant, excluding propellers. This will be used for calculating SFC later.

        - Eeff_FuelCell: the efficiency of the fuel cell energy conversion system on board, includes fuel cells, AC/DC converter, electric motor and gearbox. Generally this value is between 40% - 60%, here we use 45%.
        - Eeff_ICE: the efficiency of the Internal Combustion Engine (ICE) energy conversion system on board, includes ICE and gearbox. This value is approximately 35%.
        - Eeff_Battery: the efficiency of the battery energy conversion system on board. Batteries use 80% capacity to prolong life cycle, and lose efficiency in AC/DC converter, electric motor. Generally this value is between 70% - 95%, here we use 80 %.

        data source:
        Marin report 2019, Energietransitie emissieloze binnenvaart, vooronderzoek ontwerpaspecten, systeem configuraties.(Energy transition zero-emission inland shipping, preliminary research on design aspects, system configurations)
        add other ref

        """
        self.Eeff_FuelCell = 0.45
        self.Eeff_ICE = 0.38
        self.Eeff_Battery = 0.8

    def SFC_general(self):
        """Specific Fuel Consumption (SFC) is calculated by energy density and energy conversion efficiency.
        The SFC calculation equation, SFC = 1 / (energy density * energy conversion efficiency), can be found in the paper of Kim et al (2020)(A Preliminary Study on an Alternative Ship Propulsion System Fueled by Ammonia: Environmental and Economic Assessments, https://doi.org/10.3390/jmse8030183).

        for diesel SFC, there are 3 kinds of general diesel SFC
        - SFC_diesel_ICE_mass, calculated by net diesel gravimetric density and ICE energy-power system efficiency, without considering engine performence variation due to engine ages
        - SFC_diesel_ICE_vol, calculated by net diesel volumetric density and ICE energy-power system efficiency, without considering engine performence variation due to engine ages
        - SFC_diesel_C_year, a group of SFC considering ICE engine performence variation due to engine ages (C_year), based on TNO (2019)

        Please note: later on a correction factor has to be applied to get the total SFC
        """
        # to estimate the requirement of the amount of ZES_batterypacks for different IET scenarios, we include ZES battery capacity per container here.
        # ZES_batterypack capacity > 2000kWh, its average usable energy = 2000 kWh,  mass = 27 ton, vol = 20ft A60 container (6*2.5*2.5 = 37.5 m3) (source: ZES report)
        self.energy_density()
        self.energy_conversion_efficiency()

        self.ZES_batterypack2000kWh = 2000  # kWh/pack,

        # SFC in mass for Fuel Cell engine
        self.SFC_LH2_FuelCell_mass = 1 / (self.Edens_LH2_mass * self.Eeff_FuelCell)  # g/kWh
        self.SFC_eLNG_FuelCell_mass = 1 / (self.Edens_eLNG_mass * self.Eeff_FuelCell)  # g/kWh
        self.SFC_eMethanol_FuelCell_mass = 1 / (self.Edens_eMethanol_mass * self.Eeff_FuelCell)  # g/kWh
        self.SFC_eNH3_FuelCell_mass = 1 / (self.Edens_eNH3_mass * self.Eeff_FuelCell)  # g/kWh

        # SFC in mass for ICE engine
        self.SFC_diesel_ICE_mass = 1 / (self.Edens_diesel_mass * self.Eeff_ICE)  # g/kWh
        self.SFC_eLNG_ICE_mass = 1 / (self.Edens_eLNG_mass * self.Eeff_ICE)  # g/kWh
        self.SFC_eMethanol_ICE_mass = 1 / (self.Edens_eMethanol_mass * self.Eeff_ICE)  # g/kWh
        self.SFC_eNH3_ICE_mass = 1 / (self.Edens_eNH3_mass * self.Eeff_ICE)  # g/kWh

        # SFC in mass and volume for battery electric ships
        self.SFC_Li_NMC_Battery_mass = 1 / (self.Edens_Li_NMC_Battery_mass * self.Eeff_Battery)  # g/kWh
        self.SFC_Li_NMC_Battery_vol = 1 / (self.Edens_Li_NMC_Battery_vol * self.Eeff_Battery)  # m3/kWh
        self.SFC_ZES_battery2000kWh = 1 / (self.ZES_batterypack2000kWh * self.Eeff_Battery)  # kWh

        # SFC in volume for Fuel Cell engine
        self.SFC_LH2_FuelCell_vol = 1 / (self.Edens_LH2_vol * self.Eeff_FuelCell)  # m3/kWh
        self.SFC_eLNG_FuelCell_vol = 1 / (self.Edens_eLNG_vol * self.Eeff_FuelCell)  # m3/kWh
        self.SFC_eMethanol_FuelCell_vol = 1 / (self.Edens_eMethanol_vol * self.Eeff_FuelCell)  # m3/kWh
        self.SFC_eNH3_FuelCell_vol = 1 / (self.Edens_eNH3_vol * self.Eeff_FuelCell)  # m3/kWh

        # SFC in volume for ICE engine
        self.SFC_diesel_ICE_vol = 1 / (self.Edens_diesel_vol * self.Eeff_ICE)  # m3/kWh
        self.SFC_eLNG_ICE_vol = 1 / (self.Edens_eLNG_vol * self.Eeff_ICE)  # m3/kWh
        self.SFC_eMethanol_ICE_vol = 1 / (self.Edens_eMethanol_vol * self.Eeff_ICE)  # m3/kWh
        self.SFC_eNH3_ICE_vol = 1 / (self.Edens_eNH3_vol * self.Eeff_ICE)  # m3/kWh

        # Another source of diesel SFC: The general diesel SFC (g/kWh) which are based on the construction year of the engine (TNO)

        if self.C_year < 1974:
            self.SFC_diesel_C_year = 235
        if 1975 <= self.C_year <= 1979:
            self.SFC_diesel_C_year = 230
        if 1980 <= self.C_year <= 1984:
            self.SFC_diesel_C_year = 225
        if 1985 <= self.C_year <= 1989:
            self.SFC_diesel_C_year = 220
        if 1990 <= self.C_year <= 1994:
            self.SFC_diesel_C_year = 220
        if 1995 <= self.C_year <= 2002:
            self.SFC_diesel_C_year = 205
        if 2003 <= self.C_year <= 2007:
            self.SFC_diesel_C_year = 200
        if 2008 <= self.C_year <= 2019:
            self.SFC_diesel_C_year = 200
        if self.C_year > 2019:
            if self.L_w == 1:
                self.SFC_diesel_C_year = 205
            else:
                self.SFC_diesel_C_year = 190

        logger.debug(f"The general fuel consumption factor for diesel is {self.SFC_diesel_C_year} g/kWh")

    def correction_factors(self, v, h_0, P_partial=None):
        """Partial engine load correction factors (C_partial_load):

        - The correction factors have to be multiplied by the general emission factors (or general SFC), to get the total emission factors (or final SFC)
        - The correction factor takes into account the effect of the partial engine load
        - When the partial engine load is low, the correction factors for ICE engine are higher (ICE engine is less efficient at lower enegine load)
        - the correction factors for emissions and diesel fuel in ICE engine are based on literature TNO (2019)
        - For fuel cell enegines(PEMFC & SOFC), the correction factors are lower when the partial engine load is low (fuel cell enegine is more efficient at lower enegine load)
        - the correction factors for renewable fuels used in fuel cell engine are based on literature Kim et al (2020) (A Preliminary Study on an Alternative Ship Propulsion System Fueled by Ammonia: Environmental and Economic Assessments, https://doi.org/10.3390/jmse8030183)
        """

        # TODO: create correction factors for renewable powered ship, the factor may be 100%
        if P_partial is None:
            self.calculate_total_power_required(v=v, h_0=h_0)  # You need the P_partial values
        else:
            # PATCH P3: evaluate factors at a prescribed operating point (e.g. the
            # hotel load of a stationary event) without recomputing resistance
            self.P_partial = P_partial

        # Import the correction factors table
        # TODO: use package data, not an arbitrary location
        self.C_partial_load = opentnsim.energy.load_partial_engine_load_correction_factors()
        self.C_partial_load_battery = 1  # assume the battery energy consumption is not influenced by different engine load

        for i in range(20):
            # If the partial engine load is smaller or equal to 5%, the correction factors corresponding to P_partial = 5% are assigned.
            if self.P_partial <= self.C_partial_load.iloc[0, 0]:
                self.C_partial_load_CO2 = self.C_partial_load.iloc[0, 5]
                self.C_partial_load_PM10 = self.C_partial_load.iloc[0, 6]
                self.C_partial_load_fuel_ICE = (
                    self.C_partial_load_CO2
                )  # CO2 emission is generated from fuel consumption, so these two
                # correction factors are equal
                self.C_partial_load_PEMFC = self.C_partial_load.iloc[0, 7]
                self.C_partial_load_SOFC = self.C_partial_load.iloc[0, 8]

                # The NOX correction factors are dependend on the construction year of the engine and the weight class
                if self.C_year < 2008:
                    self.C_partial_load_NOX = self.C_partial_load.iloc[0, 1]  # <= CCR-1 class
                if 2008 <= self.C_year <= 2019:
                    self.C_partial_load_NOX = self.C_partial_load.iloc[0, 2]  # CCR-2 / Stage IIIa
                if self.C_year > 2019:
                    if self.L_w == 1:  #
                        self.C_partial_load_NOX = self.C_partial_load.iloc[
                            0, 3
                        ]  # Stage V: IWP/IWA-v/c-3 class (vessels with P <300 kW: assumed to be weight class L1)
                    else:
                        self.C_partial_load_NOX = self.C_partial_load.iloc[
                            0, 4
                        ]  # Stage V:IWP/IWA-v/c-4 class (vessels with P >300 kw: assumed to be weight class L2-L3)

            # If the partial engine load is greater than 5%:
            # It is determined inbetween which two percentages in the table the partial engine load lies
            # The correction factor is determined by means of linear interpolation

            elif self.C_partial_load.iloc[i, 0] < self.P_partial <= self.C_partial_load.iloc[i + 1, 0]:
                self.C_partial_load_CO2 = (
                    (self.P_partial - self.C_partial_load.iloc[i, 0])
                    * (self.C_partial_load.iloc[i + 1, 5] - self.C_partial_load.iloc[i, 5])
                ) / (self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) + self.C_partial_load.iloc[i, 5]
                self.C_partial_load_PM10 = (
                    (self.P_partial - self.C_partial_load.iloc[i, 0])
                    * (self.C_partial_load.iloc[i + 1, 6] - self.C_partial_load.iloc[i, 6])
                ) / (self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) + self.C_partial_load.iloc[i, 6]
                self.C_partial_load_fuel_ICE = (
                    self.C_partial_load_CO2
                )  # CO2 emission is generated from fuel consumption, so these two
                # correction factors are equal
                self.C_partial_load_PEMFC = (
                    (self.P_partial - self.C_partial_load.iloc[i, 0])
                    * (self.C_partial_load.iloc[i + 1, 7] - self.C_partial_load.iloc[i, 7])
                ) / (self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) + self.C_partial_load.iloc[i, 7]
                self.C_partial_load_SOFC = (
                    (self.P_partial - self.C_partial_load.iloc[i, 0])
                    * (self.C_partial_load.iloc[i + 1, 8] - self.C_partial_load.iloc[i, 8])
                ) / (self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) + self.C_partial_load.iloc[i, 8]
                if self.C_year < 2008:
                    self.C_partial_load_NOX = (
                        (self.P_partial - self.C_partial_load.iloc[i, 0])
                        * (self.C_partial_load.iloc[i + 1, 1] - self.C_partial_load.iloc[i, 1])
                    ) / (self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) + self.C_partial_load.iloc[i, 1]
                if 2008 <= self.C_year <= 2019:
                    self.C_partial_load_NOX = (
                        (self.P_partial - self.C_partial_load.iloc[i, 0])
                        * (self.C_partial_load.iloc[i + 1, 2] - self.C_partial_load.iloc[i, 2])
                    ) / (self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) + self.C_partial_load.iloc[i, 2]
                if self.C_year > 2019:
                    if self.L_w == 1:
                        self.C_partial_load_NOX = (
                            (self.P_partial - self.C_partial_load.iloc[i, 0])
                            * (self.C_partial_load.iloc[i + 1, 3] - self.C_partial_load.iloc[i, 3])
                        ) / (self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) + self.C_partial_load.iloc[i, 3]
                    else:
                        self.C_partial_load_NOX = (
                            (self.P_partial - self.C_partial_load.iloc[i, 0])
                            * (self.C_partial_load.iloc[i + 1, 4] - self.C_partial_load.iloc[i, 4])
                        ) / (self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) + self.C_partial_load.iloc[i, 4]

            # If the partial engine load is => 100%, the correction factors corresponding to P_partial = 100% are assigned.
            elif self.P_partial >= self.C_partial_load.iloc[19, 0]:
                self.C_partial_load_CO2 = self.C_partial_load.iloc[19, 5]
                self.C_partial_load_PM10 = self.C_partial_load.iloc[19, 6]
                self.C_partial_load_fuel_ICE = (
                    self.C_partial_load_CO2
                )  # CO2 emission is generated from fuel consumption, so these two
                # correction factors are equal
                self.C_partial_load_PEMFC = self.C_partial_load.iloc[19, 7]
                self.C_partial_load_SOFC = self.C_partial_load.iloc[19, 8]
                # The NOX correction factors are dependend on the construction year of the engine and the weight class
                if self.C_year < 2008:
                    self.C_partial_load_NOX = self.C_partial_load.iloc[19, 1]  # <= CCR-1 class
                if 2008 <= self.C_year <= 2019:
                    self.C_partial_load_NOX = self.C_partial_load.iloc[19, 2]  # CCR-2 / Stage IIIa
                if self.C_year > 2019:
                    if self.L_w == 1:  #
                        self.C_partial_load_NOX = self.C_partial_load.iloc[
                            19, 3
                        ]  # Stage V: IWP/IWA-v/c-3 class (vessels with P <300 kW: assumed to be weight class L1)
                    else:
                        self.C_partial_load_NOX = self.C_partial_load.iloc[
                            19, 4
                        ]  # Stage V:IWP/IWA-v/c-4 class (vessels with P >300 kw: assumed to be weight class L2-L3)

        logger.debug(f"Partial engine load correction factor of CO2 is {self.C_partial_load_CO2}")
        logger.debug(f"Partial engine load correction factor of PM10 is {self.C_partial_load_PM10}")
        logger.debug(f"Partial engine load correction factor of NOX is {self.C_partial_load_NOX}")
        logger.debug(f"Partial engine load correction factor of diesel fuel consumption in ICE is {self.C_partial_load_fuel_ICE}")
        logger.debug(f"Partial engine load correction factor of fuel consumption in PEMFC is {self.C_partial_load_PEMFC}")
        logger.debug(f"Partial engine load correction factor of fuel consumption in SOFC is {self.C_partial_load_SOFC}")
        logger.debug(f"Partial engine load correction factor of energy consumption in battery is {self.C_partial_load_battery}")

    def calculate_emission_factors_total(self, v, h_0, P_partial=None):
        """Total emission factors:

        - The total emission factors can be computed by multiplying the general emission factor by the correction factor
        """

        self.emission_factors_general()  # You need the values of the general emission factors of CO2, PM10, NOX
        self.correction_factors(v=v, h_0=h_0, P_partial=P_partial)  # You need the correction factors of CO2, PM10, NOX

        # The total emission factor is calculated by multiplying the general emission factor (EF_CO2 / EF_PM10 / EF_NOX)
        # By the correction factor (C_partial_load_CO2 / C_partial_load_PM10 / C_partial_load_NOX)

        self.total_factor_CO2 = self.EF_CO2 * self.C_partial_load_CO2
        self.total_factor_PM10 = self.EF_PM10 * self.C_partial_load_PM10
        self.total_factor_NOX = self.EF_NOX * self.C_partial_load_NOX

        logger.debug(f"The total emission factor of CO2 is {self.total_factor_CO2} g/kWh")
        logger.debug(f"The total emission factor of PM10 is {self.total_factor_PM10} g/kWh")
        logger.debug(f"The total emission factor CO2 is {self.total_factor_NOX} g/kWh")

    def calculate_SFC_final(self, v, h_0, P_partial=None):
        """The final SFC is computed by multiplying the general SFC by the partial engine load correction factor.

        The calculation of final SFC below includes
        - the final SFC of LH2, eLNG, eMethanol, eNH3 in mass and volume while using Fuel Cell Engine (PEMFC, SOFC)
        - the final SFC of eLNG, eMethanol, eNH3 in mass and volume while using Internal Combustion Engine
        - the final SFC of diesel in mass and volume while using Internal Combustion Engine
        - the final SFC of battery in mass and volume while use battery-electric power system
        """

        self.SFC_general()  # You need the values of the general SFC
        self.correction_factors(v=v, h_0=h_0, P_partial=P_partial)  # You need the correction factors of SFC

        # final SFC of fuel cell in mass   [g/kWh]
        self.final_SFC_LH2_mass_PEMFC = self.SFC_LH2_FuelCell_mass * self.C_partial_load_PEMFC
        self.final_SFC_LH2_mass_SOFC = self.SFC_LH2_FuelCell_mass * self.C_partial_load_SOFC
        self.final_SFC_eLNG_mass_PEMFC = self.SFC_eLNG_FuelCell_mass * self.C_partial_load_PEMFC
        self.final_SFC_eLNG_mass_SOFC = self.SFC_eLNG_FuelCell_mass * self.C_partial_load_SOFC
        self.final_SFC_eMethanol_mass_PEMFC = self.SFC_eMethanol_FuelCell_mass * self.C_partial_load_PEMFC
        self.final_SFC_eMethanol_mass_SOFC = self.SFC_eMethanol_FuelCell_mass * self.C_partial_load_SOFC
        self.final_SFC_eNH3_mass_PEMFC = self.SFC_eNH3_FuelCell_mass * self.C_partial_load_PEMFC
        self.final_SFC_eNH3_mass_SOFC = self.SFC_eNH3_FuelCell_mass * self.C_partial_load_SOFC

        # final SFC of fuel cell in vol  [m3/kWh]
        self.final_SFC_LH2_vol_PEMFC = self.SFC_LH2_FuelCell_vol * self.C_partial_load_PEMFC
        self.final_SFC_LH2_vol_SOFC = self.SFC_LH2_FuelCell_vol * self.C_partial_load_SOFC
        self.final_SFC_eLNG_vol_PEMFC = self.SFC_eLNG_FuelCell_vol * self.C_partial_load_PEMFC
        self.final_SFC_eLNG_vol_SOFC = self.SFC_eLNG_FuelCell_vol * self.C_partial_load_SOFC
        self.final_SFC_eMethanol_vol_PEMFC = self.SFC_eMethanol_FuelCell_vol * self.C_partial_load_PEMFC
        self.final_SFC_eMethanol_vol_SOFC = self.SFC_eMethanol_FuelCell_vol * self.C_partial_load_SOFC
        self.final_SFC_eNH3_vol_PEMFC = self.SFC_eNH3_FuelCell_vol * self.C_partial_load_PEMFC
        self.final_SFC_eNH3_vol_SOFC = self.SFC_eNH3_FuelCell_vol * self.C_partial_load_SOFC

        # final SFC of ICE in mass [g/kWh]
        self.final_SFC_diesel_C_year_ICE_mass = self.SFC_diesel_C_year * self.C_partial_load_fuel_ICE
        self.final_SFC_diesel_ICE_mass = self.SFC_diesel_ICE_mass * self.C_partial_load_fuel_ICE
        self.final_SFC_eLNG_ICE_mass = self.SFC_eLNG_ICE_mass * self.C_partial_load_fuel_ICE
        self.final_SFC_eMethanol_ICE_mass = self.SFC_eMethanol_ICE_mass * self.C_partial_load_fuel_ICE
        self.final_SFC_eNH3_ICE_mass = self.SFC_eNH3_ICE_mass * self.C_partial_load_fuel_ICE

        # final SFC of ICE in vol  [m3/kWh]
        self.final_SFC_diesel_ICE_vol = self.SFC_diesel_ICE_vol * self.C_partial_load_fuel_ICE
        self.final_SFC_eLNG_ICE_vol = self.SFC_eLNG_ICE_vol * self.C_partial_load_fuel_ICE
        self.final_SFC_eMethanol_ICE_vol = self.SFC_eMethanol_ICE_vol * self.C_partial_load_fuel_ICE
        self.final_SFC_eNH3_ICE_vol = self.SFC_eNH3_ICE_vol * self.C_partial_load_fuel_ICE

        # final SFC of battery in mass and vol
        self.final_SFC_Li_NMC_Battery_mass = self.SFC_Li_NMC_Battery_mass * self.C_partial_load_battery  # g/kWh
        self.final_SFC_Li_NMC_Battery_vol = self.SFC_Li_NMC_Battery_vol * self.C_partial_load_battery  # m3/kWh
        self.final_SFC_Battery2000kWh = self.SFC_ZES_battery2000kWh * self.C_partial_load_battery  # kWh

    def calculate_diesel_use_g_m(self, v):
        """Total diesel fuel use in g/m:

        - The total fuel use in g/m can be computed by total fuel use in g (P_tot * delt_t * self.total_factor_) diveded by the sailing distance (v * delt_t)
        """
        self.diesel_use_g_m = (self.P_given * self.final_SFC_diesel_ICE_mass / v) / 3600  # without considering C_year
        self.diesel_use_g_m_C_year = (self.P_given * self.final_SFC_diesel_C_year_ICE_mass / v) / 3600  # considering C_year


    def calculate_diesel_use_g_s(self):
        """Total diesel fuel use in g/s:

        - The total fuel use in g/s can be computed by total emission in g (P_tot * delta_t * self.total_factor_) diveded by the sailing duration (delt_t)
        """
        self.diesel_use_g_s = self.P_given * self.final_SFC_diesel_ICE_mass / 3600  # without considering C_year
        self.diesel_use_g_s_C_year = self.P_given * self.final_SFC_diesel_C_year_ICE_mass / 3600  # considering C_year


    def calculate_emission_rates_g_m(self, v):
        """CO2, PM10, NOX emission rates in g/m:

        - The CO2, PM10, NOX emission rates in g/m can be computed by total fuel use in g (P_tot * delta_t * self.total_factor_) diveded by the sailing distance (v * delt_t)
        """
        self.emission_g_m_CO2 = self.P_given * self.total_factor_CO2 / v / 3600
        self.emission_g_m_PM10 = self.P_given * self.total_factor_PM10 / v / 3600
        self.emission_g_m_NOX = self.P_given * self.total_factor_NOX / v / 3600


    def calculate_emission_rates_g_s(self):
        """CO2, PM10, NOX emission rates in g/s:

        - The CO2, PM10, NOX emission rates in g/s can be computed by total fuel use in g (P_tot * delta_t * self.total_factor_) diveded by the sailing duration (delt_t)
        """
        self.emission_g_s_CO2 = self.P_given * self.total_factor_CO2 / 3600
        self.emission_g_s_PM10 = self.P_given * self.total_factor_PM10 / 3600
        self.emission_g_s_NOX = self.P_given * self.total_factor_NOX / 3600


    def calculate_max_sinkage(self, v, h_0, width=150):
        """Calculate the maximum sinkage of a moving ship

        the calculation equation is described in Barrass, B. & Derrett, R.'s book (2006), Ship Stability for Masters and Mates,
        chapter 42. https://doi.org/10.1016/B978-0-08-097093-6.00042-6

        some explanation for the variables in the equation:
        - h_0: water depth
        - v: ship velocity relative to the water
        - width: river width, default to 150
        """

        max_sinkage = 0
        if self.h_squat:
            max_sinkage = (self.C_B * ((self.B * self.T) / (width * h_0)) ** 0.81) * ((v * 1.94) ** 2.08) / 20

        return max_sinkage


    def calculate_h_squat(self, v, h_0, width=150):
        """Calculate the water depth in case h_squat is set to True

        The amount of water under the keel is calculated h_0 - T. When h_squat is set to True, we estimate a max_sinkage
        that is subtracted from h_0. This values is returned as h_squat for further calculation.

        """
        h_squat = h_0 - self.calculate_max_sinkage(v, h_0, width=width)

        return h_squat

    def confinement_hull_used(self):
        """Coefficient set of the increment. None: the set of the vessel type ("Barge" -> "barge", else "motor").
        """

        if self.confinement_hull is not None:
            return self.confinement_hull
        return default_hull(getattr(self, "vessel_type", None))

    def _hydraulic_draught(self):
        """Real draught for the channel hydraulics (T_hydro when set, else T)."""
        return getattr(self, "T_hydro", None) or self.T

    def _required_width(self, width):
        if width is None or not np.isfinite(width) or width <= 0:
            raise ValueError(
                f"A positive waterway width (GeneralWidth) is required for confinement_mode "
                f"{getattr(self, 'confinement_mode', None)!r}; received {width!r}. There is no default width in this mode."
            )
        return float(width)


    def limit_power2v_upperbound(self, upperbound, h_0, width=None, channel_area=None):
        """Upper bound of the power2v speed search

        With "none" the input comes back unchanged. With "drawdown" or "full" the bound also stays below
        0.999 V_cr, below the speed of the minimum dynamic clearance and, with speed cap "economic",
        below 0.999 V_ec.
        """
        if getattr(self, "confinement_mode", "none") == "none":
            return float(upperbound)

        W = self._required_width(width)
        T = self._hydraulic_draught()
        limits = hydraulic_speed_limits(float(h_0), self.B, T, W, channel_area_m2=channel_area, g=self.g)
        self.V_cr = limits["V_cr_ms"]
        self.V_ec = limits["V_ec_ms"]

        caps = [float(upperbound), 0.999 * self.V_cr]
        if self.confinement_speed_cap == "economic":
            caps.append(0.999 * self.V_ec)
        self.V_ukc_limit = max_speed_for_ukc(
            float(h_0), self.B, T, W, ukc_min=self.confinement_ukc_min, channel_area_m2=channel_area, g=self.g,
        )
        caps.append(self.V_ukc_limit)
        return float(min(caps))


    def calculate_resistance_for_waterway(self, v, h_0, width=None, channel_area=None):
        """Resistance of one waterway edge in kN, with the depth of the mode

        Callers do not apply squat themselves. With "none" the chain runs at the squat depth.
        With "drawdown" or "full" the chain and the hydraulics run at the raw depth and the width is
        required; Barrass squat stays as the navigation depth h_navigation, because the drawdown term
        already models the sinkage.
        """
        if getattr(self, "confinement_mode", "none") == "none":
            width_for_squat = 150.0 if width is None else float(width)

            self.h_raw = float(h_0)
            self.squat_m = self.calculate_max_sinkage(v=float(v), h_0=self.h_raw, width=width_for_squat,)
            self.h_resistance = self.h_raw - self.squat_m
            self.h_navigation = self.h_resistance

            self.calculate_total_resistance(v=float(v), h_0=self.h_resistance, width=width, channel_area=channel_area, h_channel=self.h_raw,)
            return self.R_tot

        W = self._required_width(width)
        self.h_raw = float(h_0)

        # Barrass squat: navigation diagnostic only (the drawdown term models the sinkage effect).
        self.squat_m = self.calculate_max_sinkage(v=float(v), h_0=self.h_raw, width=W)
        self.h_navigation = self.h_raw - self.squat_m
        self.h_resistance = self.h_navigation if self.squat_in_resistance else self.h_raw

        return self.calculate_total_resistance(
            v=float(v), h_0=self.h_resistance, width=W, channel_area=channel_area, h_channel=self.h_raw,
        )

    

class EnergyCalculation:
    """Add information on energy use and effects on energy use."""

    # ToDo: add other alternatives from Marin's table to have completed renewable energy sources
    # ToDo: add renewable fuel cost from Marin's table, add fuel cell / other engine cost, power plan cost to calculate the cost of ship refit or new ships.

    def __init__(self, FG, vessel, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """Initialization"""
        self.FG = FG
        self.vessel = vessel

        self.energy_use = {
            "time_start": [],
            "time_stop": [],
            "edge_start": [],
            "edge_stop": [],
            "P_tot": [],
            "P_given": [],
            "P_installed": [],
            "total_energy": [],
            "energy_shaft": [],
            "total_diesel_consumption_C_year_ICE_mass": [],
            "total_diesel_consumption_ICE_mass": [],
            "total_diesel_consumption_ICE_vol": [],
            "total_LH2_consumption_PEMFC_mass": [],
            "total_LH2_consumption_SOFC_mass": [],
            "total_LH2_consumption_PEMFC_vol": [],
            "total_LH2_consumption_SOFC_vol": [],
            "total_eLNG_consumption_PEMFC_mass": [],
            "total_eLNG_consumption_SOFC_mass": [],
            "total_eLNG_consumption_PEMFC_vol": [],
            "total_eLNG_consumption_SOFC_vol": [],
            "total_eLNG_consumption_ICE_mass": [],
            "total_eLNG_consumption_ICE_vol": [],
            "total_eMethanol_consumption_PEMFC_mass": [],
            "total_eMethanol_consumption_SOFC_mass": [],
            "total_eMethanol_consumption_PEMFC_vol": [],
            "total_eMethanol_consumption_SOFC_vol": [],
            "total_eMethanol_consumption_ICE_mass": [],
            "total_eMethanol_consumption_ICE_vol": [],
            "total_eNH3_consumption_PEMFC_mass": [],
            "total_eNH3_consumption_SOFC_mass": [],
            "total_eNH3_consumption_PEMFC_vol": [],
            "total_eNH3_consumption_SOFC_vol": [],
            "total_eNH3_consumption_ICE_mass": [],
            "total_eNH3_consumption_ICE_vol": [],
            "total_Li_NMC_Battery_mass": [],
            "total_Li_NMC_Battery_vol": [],
            "total_Battery2000kWh_consumption_num": [],
            "total_emission_CO2": [],
            "total_emission_PM10": [],
            "total_emission_NOX": [],
            "stationary": [],
            "power_capped": [],
            "water depth": [],
            "distance": [],
            "delta_t": [],
            "v_g": [],
            "v_c": [],
            "v_w": [],
        }

        self.co2_footprint = {"total_footprint": 0, "stationary": 0}
        self.mki_footprint = {"total_footprint": 0, "stationary": 0}

    def calculate_energy_consumption(self):
        """Calculation of energy consumption based on total time in system and properties"""
        warnings.warn(
            "EnergyCalculation is retained for comparison only; the event-table "
            "pipeline (logutils.logbook2eventtable + energy_logutils) is the "
            "canonical accounting path.",
            DeprecationWarning,
        )

        def calculate_distance(geom_start, geom_stop):
            """method to calculate the distance in meters between two geometries"""

            wgs84 = pyproj.Geod(ellps="WGS84")

            # distance between two points
            return float(wgs84.inv(geom_start.x, geom_start.y, geom_stop.x, geom_stop.y)[2])

        def calculate_depth(geom_start, geom_stop):
            """method to calculate the depth of the waterway in meters between two geometries"""

            depth = 0

            # The node on the graph of vaarweginformatie.nl closest to geom_start and geom_stop

            node_start = find_closest_node(self.FG, geom_start)[0]
            node_stop = find_closest_node(self.FG, geom_stop)[0]

            # Read from the FG data from vaarweginformatie.nl the General depth of each edge
            #TODO: check it this needs to be made more general, now relies on ['Info'] to be present
            try:  # if node_start != node_stop:
                depth = self.FG.get_edge_data(node_start, node_stop)["Info"]["GeneralDepth"]
            except:
                depth = np.nan  # When there is no data of the depth available of this edge, it gives a message

            h_0 = depth

            # depth of waterway between two points
            return h_0


        # log messages that are related to locking
        # todo: check if this still works with Floors new locking module
        stationary_phase_indicator = [
            "Waiting to enter waiting area stop",   # checked: not sure if still used in locking module
            "Waiting in waiting area stop",         # checked: not sure if still used in locking module
            "Waiting in line-up area stop",         # checked: still used in locking module
            "Passing lock stop",                    # checked: still used in locking module
        ]

        # extract relevant elements from the vessel log
        times = [row["Timestamp"] for row in self.vessel.logbook]
        messages = [row["Message"] for row in self.vessel.logbook]
        geometries = [row["Geometry"] for row in self.vessel.logbook]

        # now walk past each logged event (each 'time interval' in the log corresponds to an event)
        for i in range(len(times) - 1):
            # determine the time associated with the logged event (how long did it last)
            delta_t = (times[i + 1] - times[i]).total_seconds()

            if delta_t != 0:
                # append time information to the variables for the dataframe
                self.energy_use["time_start"].append(times[i])
                self.energy_use["time_stop"].append(times[i + 1])

                # append geometry information to the variables for the dataframe
                self.energy_use["edge_start"].append(geometries[i])
                self.energy_use["edge_stop"].append(geometries[i + 1])

                # calculate the distance travelled and the associated velocity
                message = messages[i]
                if 'from node ' in message and 'to node ' in message:
                    node_start = message.split('from node ')[1].split(' to node')[0]
                    node_stop = message.split('to node ')[1].split(' ')[0]

                    # edge data from environment graph
                    e_data = self.vessel.env.FG.edges[node_start, node_stop]
                    g_edge = e_data.get("geometry", None)

                    # distance along edge (fallback to geometric)
                    distance = e_data.get("length", calculate_distance(geometries[i], geometries[i + 1]))

                    # discharge (if stored on edge)
                    Q = e_data.get("discharge", None)

                    info = e_data.get("Info", {})
                    waterway_width = e_data.get("GeneralWidth", info.get("GeneralWidth"))
                    channel_area = e_data.get("GeneralCrossSectionArea",info.get("GeneralCrossSectionArea"))
                    h_edge = e_data.get("GeneralDepth", info.get("GeneralDepth", np.nan))
                  
                    now_s = pd.Timestamp(times[i]).timestamp()   
                    v_c = float(self.vessel.env.get_current(node_start, node_stop, now_s)) 

                    
                else:
                    distance = calculate_distance(geometries[i], geometries[i + 1])
                    v_c = 0.0
                    e_data = {}
                    waterway_width = None
                    channel_area = None
                    h_edge = np.nan

                v_g = distance / delta_t
                v_w = v_g - v_c


                self.vessel.v_g = v_g
                self.vessel.v_c = v_c
                self.vessel.v_w = v_w

                self.energy_use["v_g"].append(v_g)
                self.energy_use["v_c"].append(v_c)
                self.energy_use["v_w"].append(v_w)
                
                v = v_w

                self.energy_use["distance"].append(distance)

                # calculate the delta t
                self.energy_use["delta_t"].append(delta_t)

                logger.debug("geometries[i]: {0}, geometries[i + 1] {1}".format(geometries[i], geometries[i + 1]))

                # calculate the water depth
                if h_edge is not None and np.isfinite(h_edge):
                    h_0 = float(h_edge)
                else:
                    h_0 = calculate_depth(geometries[i], geometries[i + 1])


                # printstatements to check the output (can be removed later)
                logger.debug("delta_t: {:.4f} s".format(delta_t))
                logger.debug("distance: {:.4f} m".format(distance))
                logger.debug("v_ground: {:.4f} m/s".format(v_g))
                logger.debug("v_water:  {:.4f} m/s".format(v))
                logger.debug("h_0: {:.4f} m".format(h_0))


                # we use the calculated velocity to determine the resistance and power required
                # we can switch between the original water depth and the squat-corrected water
                # depth via calculate_h_squat (h_squat set as True/False on the vessel)
                # PATCH P2/P3: stationary events are stored at hotel power; every event
                # appends the full key set so all lists stay equally long.
                stationary = messages[i + 1] in stationary_phase_indicator

                if stationary:
                    # stationary stage: hotel power only; emission and fuel factors
                    # are evaluated at the hotel operating point
                    energy_delta = self.vessel.P_hotel * delta_t / 3600  # kJ/3600 = kWh
                    P_tot_delta = self.vessel.P_hotel
                    P_given_delta = self.vessel.P_hotel
                    P_installed_delta = self.vessel.P_installed
                    power_capped = False
                    energy_shaft = self.vessel.P_hotel * delta_t / 3600
                    P_partial_hotel = self.vessel.P_hotel / self.vessel.P_installed
                    self.vessel.calculate_emission_factors_total(v=0.0, h_0=h_0, P_partial=P_partial_hotel)
                    self.vessel.calculate_SFC_final(v=0.0, h_0=h_0, P_partial=P_partial_hotel)
                else:
                    # propulsion stage: evaluate resistance and power at the event
                    # water speed (v = v_w, derived above from v_g and v_c)
                    if v <= 0:
                        raise ValueError(
                            "Non-positive water speed on a sailing event: "
                            f"v_w={v:.3f} m/s (v_g={v_g:.3f}, v_c={v_c:.3f})."
                        )
                    self.vessel.calculate_resistance_for_waterway(v=v, h_0=h_0, width=waterway_width, channel_area=channel_area)
                    h_for_model = self.vessel.h_resistance
                    self.vessel.calculate_total_power_required(v=v, h_0=h_for_model)
                    self.vessel.calculate_emission_factors_total(v=v, h_0=h_for_model, P_partial=self.vessel.P_partial)
                    self.vessel.calculate_SFC_final(v=v, h_0=h_for_model, P_partial=self.vessel.P_partial)

                    # PATCH P2: energy actually drawn from the engine (brake power,
                    # capped at installed power); the uncapped requirement remains
                    # visible as P_tot and the capped share as power_capped
                    energy_delta = self.vessel.P_given * delta_t / 3600  # kJ/3600 = kWh
                    P_tot_delta = self.vessel.P_tot  # in kW, required power, may exceed installed engine power
                    P_given_delta = self.vessel.P_given  # in kW, actual given power
                    P_installed_delta = self.vessel.P_installed  # in kW
                    power_capped = bool(self.vessel.P_tot > self.vessel.P_installed)
                    if power_capped:
                        eta_tg = self.vessel.eta_t * self.vessel.eta_g
                        P_shaft = (max(self.vessel.P_installed - self.vessel.P_hotel, 0.0)* eta_tg + self.vessel.P_hotel)
                    else:
                        P_shaft = self.vessel.P_d + self.vessel.P_hotel
                    energy_shaft = P_shaft * delta_t / 3600



                # emissions and fuel per event; the factors were evaluated at the
                # operating point selected above. NOTE: in this legacy class the
                # alternative-carrier quantities stay on the brake-energy basis;
                # the event-table pipeline separates brake and shaft bases.
                emission_delta_CO2 = self.vessel.total_factor_CO2 * energy_delta  # in g
                emission_delta_PM10 = self.vessel.total_factor_PM10 * energy_delta  # in g
                emission_delta_NOX = self.vessel.total_factor_NOX * energy_delta  # in g
                delta_diesel_C_year = self.vessel.final_SFC_diesel_C_year_ICE_mass * energy_delta  # in g
                delta_diesel_ICE_mass = self.vessel.final_SFC_diesel_ICE_mass * energy_delta  # in g
                delta_diesel_ICE_vol = self.vessel.final_SFC_diesel_ICE_vol * energy_delta  # in m3

                delta_LH2_PEMFC_mass = self.vessel.final_SFC_LH2_mass_PEMFC * energy_delta  # in g
                delta_LH2_SOFC_mass = self.vessel.final_SFC_LH2_mass_SOFC * energy_delta  # in g
                delta_LH2_PEMFC_vol = self.vessel.final_SFC_LH2_vol_PEMFC * energy_delta  # in m3
                delta_LH2_SOFC_vol = self.vessel.final_SFC_LH2_vol_SOFC * energy_delta  # in m3

                delta_eLNG_PEMFC_mass = self.vessel.final_SFC_eLNG_mass_PEMFC * energy_delta  # in g
                delta_eLNG_SOFC_mass = self.vessel.final_SFC_eLNG_mass_SOFC * energy_delta  # in g
                delta_eLNG_PEMFC_vol = self.vessel.final_SFC_eLNG_vol_PEMFC * energy_delta  # in m3
                delta_eLNG_SOFC_vol = self.vessel.final_SFC_eLNG_vol_SOFC * energy_delta  # in m3
                delta_eLNG_ICE_mass = self.vessel.final_SFC_eLNG_ICE_mass * energy_delta  # in g
                delta_eLNG_ICE_vol = self.vessel.final_SFC_eLNG_ICE_vol * energy_delta  # in m3

                delta_eMethanol_PEMFC_mass = self.vessel.final_SFC_eMethanol_mass_PEMFC * energy_delta  # in g
                delta_eMethanol_SOFC_mass = self.vessel.final_SFC_eMethanol_mass_SOFC * energy_delta  # in g
                delta_eMethanol_PEMFC_vol = self.vessel.final_SFC_eMethanol_vol_PEMFC * energy_delta  # in m3
                delta_eMethanol_SOFC_vol = self.vessel.final_SFC_eMethanol_vol_SOFC * energy_delta  # in m3
                delta_eMethanol_ICE_mass = self.vessel.final_SFC_eMethanol_ICE_mass * energy_delta  # in g
                delta_eMethanol_ICE_vol = self.vessel.final_SFC_eMethanol_ICE_vol * energy_delta  # in m3

                delta_eNH3_PEMFC_mass = self.vessel.final_SFC_eNH3_mass_PEMFC * energy_delta  # in g
                delta_eNH3_SOFC_mass = self.vessel.final_SFC_eNH3_mass_SOFC * energy_delta  # in g
                delta_eNH3_PEMFC_vol = self.vessel.final_SFC_eNH3_vol_PEMFC * energy_delta  # in m3
                delta_eNH3_SOFC_vol = self.vessel.final_SFC_eNH3_vol_SOFC * energy_delta  # in m3
                delta_eNH3_ICE_mass = self.vessel.final_SFC_eNH3_ICE_mass * energy_delta  # in g
                delta_eNH3_ICE_vol = self.vessel.final_SFC_eNH3_ICE_vol * energy_delta  # in m3

                delta_Li_NMC_Battery_mass = self.vessel.final_SFC_Li_NMC_Battery_mass * energy_delta  # in g
                delta_Li_NMC_Battery_vol = self.vessel.final_SFC_Li_NMC_Battery_vol * energy_delta  # in m3
                delta_Battery2000kWh = self.vessel.final_SFC_Battery2000kWh * energy_delta  # in ZESpack number

                self.energy_use["P_tot"].append(P_tot_delta)
                self.energy_use["P_given"].append(P_given_delta)
                self.energy_use["P_installed"].append(P_installed_delta)
                self.energy_use["total_energy"].append(energy_delta)
                self.energy_use["energy_shaft"].append(energy_shaft)
                self.energy_use["stationary"].append(energy_delta if stationary else 0.0)
                self.energy_use["power_capped"].append(power_capped)
                self.energy_use["total_emission_CO2"].append(emission_delta_CO2)
                self.energy_use["total_emission_PM10"].append(emission_delta_PM10)
                self.energy_use["total_emission_NOX"].append(emission_delta_NOX)
                self.energy_use["total_diesel_consumption_C_year_ICE_mass"].append(delta_diesel_C_year)
                self.energy_use["total_diesel_consumption_ICE_mass"].append(delta_diesel_ICE_mass)
                self.energy_use["total_diesel_consumption_ICE_vol"].append(delta_diesel_ICE_vol)
                self.energy_use["total_LH2_consumption_PEMFC_mass"].append(delta_LH2_PEMFC_mass)
                self.energy_use["total_LH2_consumption_SOFC_mass"].append(delta_LH2_SOFC_mass)
                self.energy_use["total_LH2_consumption_PEMFC_vol"].append(delta_LH2_PEMFC_vol)
                self.energy_use["total_LH2_consumption_SOFC_vol"].append(delta_LH2_SOFC_vol)
                self.energy_use["total_eLNG_consumption_PEMFC_mass"].append(delta_eLNG_PEMFC_mass)
                self.energy_use["total_eLNG_consumption_SOFC_mass"].append(delta_eLNG_SOFC_mass)
                self.energy_use["total_eLNG_consumption_PEMFC_vol"].append(delta_eLNG_PEMFC_vol)
                self.energy_use["total_eLNG_consumption_SOFC_vol"].append(delta_eLNG_SOFC_vol)
                self.energy_use["total_eLNG_consumption_ICE_mass"].append(delta_eLNG_ICE_mass)
                self.energy_use["total_eLNG_consumption_ICE_vol"].append(delta_eLNG_ICE_vol)
                self.energy_use["total_eMethanol_consumption_PEMFC_mass"].append(delta_eMethanol_PEMFC_mass)
                self.energy_use["total_eMethanol_consumption_SOFC_mass"].append(delta_eMethanol_SOFC_mass)
                self.energy_use["total_eMethanol_consumption_PEMFC_vol"].append(delta_eMethanol_PEMFC_vol)
                self.energy_use["total_eMethanol_consumption_SOFC_vol"].append(delta_eMethanol_SOFC_vol)
                self.energy_use["total_eMethanol_consumption_ICE_mass"].append(delta_eMethanol_ICE_mass)
                self.energy_use["total_eMethanol_consumption_ICE_vol"].append(delta_eMethanol_ICE_vol)
                self.energy_use["total_eNH3_consumption_PEMFC_mass"].append(delta_eNH3_PEMFC_mass)
                self.energy_use["total_eNH3_consumption_SOFC_mass"].append(delta_eNH3_SOFC_mass)
                self.energy_use["total_eNH3_consumption_PEMFC_vol"].append(delta_eNH3_PEMFC_vol)
                self.energy_use["total_eNH3_consumption_SOFC_vol"].append(delta_eNH3_SOFC_vol)
                self.energy_use["total_eNH3_consumption_ICE_mass"].append(delta_eNH3_ICE_mass)
                self.energy_use["total_eNH3_consumption_ICE_vol"].append(delta_eNH3_ICE_vol)
                self.energy_use["total_Li_NMC_Battery_mass"].append(delta_Li_NMC_Battery_mass)
                self.energy_use["total_Li_NMC_Battery_vol"].append(delta_Li_NMC_Battery_vol)
                self.energy_use["total_Battery2000kWh_consumption_num"].append(delta_Battery2000kWh)
                self.energy_use["water depth"].append(h_0)


        # TODO: er moet hier een heel aantal dingen beter worden ingevuld
        # - de kruissnelheid is nu nog per default 1 m/s (zie de Movable mixin). Eigenlijk moet in de
        #   vessel database ook nog een speed_loaded en een speed_unloaded worden toegevoegd.
        # - er zou nog eens goed gekeken moeten worden wat er gedaan kan worden rond kustwerken
        # - en er is nog iets mis met de snelheid rond een sluis

        # - add HasCurrent Class or def

