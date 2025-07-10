"""Energy module for OpenTNSim."""

# %% IMPORT DEPENENDENCIES
# generic
import pathlib
import logging
import functools
import pyproj
import numpy as np
import pandas as pd
import scipy.optimize

# OpenTNSim
import opentnsim
import opentnsim.strategy

# logging
logger = logging.getLogger(__name__)


def calculate_distance(geom_start, geom_stop):
    """method to calculate the distance (as the bird flies) in meters between two geometries

    Parameters
    ----------
    geom_start : shapely.geometry.Point
        Starting point geometry. must contain x and y attributes.
    geom_stop : shapely.geometry.Point
        Stopping point geometry. must contain x and y attributes.

    Returns
    -------
    float
        Distance in meters between the two geometries.
    """

    wgs84 = pyproj.Geod(ellps="WGS84")

    # distance between two points
    return float(wgs84.inv(geom_start.x, geom_start.y, geom_stop.x, geom_stop.y)[2])


def calculate_depth(geom_start, geom_stop, FG):
    """method to calculate the depth of the waterway in meters between two geometries.

    Parameters
    ----------
    geom_start : shapely.geometry.Point
        Starting point geometry. Must represent a node in graph FG.
    geom_stop : shapely.geometry.Point
        Stopping point geometry. must represent a node in graph FG.
    FG : networkx.Graph
        The graph containing vaarweginformatie.nl data, with nodes and edges.
        Must contain 'Info' attribute on edges with 'GeneralDepth'.
        Must contain an edge between geom_start and geom_stop.

    Returns
    -------
    float
        The depth of the waterway between the two geometries in meters.

    Raises
    ------
    ValueError
        If geom_start or geom_stop are not nodes in the graph FG.
        If there is no edge between the two nodes in the graph FG.
        If the depth data is not available for the edge between the two nodes.
    """

    depth = 0

    # The node on the graph of vaarweginformatie.nl closest to geom_start and geom_stop

    node_start = find_closest_node(FG, geom_start)[0]
    node_stop = find_closest_node(FG, geom_stop)[0]

    # Read from the FG data from vaarweginformatie.nl the General depth of each edge
    # TODO: check it this needs to be made more general, now relies on ['Info'] to be present
    try:  # if node_start != node_stop:
        depth = FG.get_edge_data(node_start, node_stop)["Info"]["GeneralDepth"]
    except:
        depth = np.nan  # When there is no data of the depth available of this edge, it gives a message

    h_0 = depth

    # depth of waterway between two points
    return h_0


# %% AUXILIARY FUNCTIONS
def load_partial_engine_load_correction_factors():
    """read correction factor from package directory"""

    # Can't get this  to work with pkg_resourcs
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    correctionfactors_path = data_dir / "Correctionfactors.csv"
    df = pd.read_csv(correctionfactors_path, comment="#")

    return df


def karpov_smooth_curves():
    """read correction factor from package directory"""

    # Can't get this  to work with pkg_resourcs
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


def power2v(vessel, edge, upperbound):
    """Compute vessel velocity given an edge and power (P_tot_given)

    bounds is the limits where to look for a solution for the velocity [m/s]
    returns velocity [m/s]
    """

    assert isinstance(vessel, opentnsim.vessel.VesselProperties), "vessel should be an instance of VesselProperties"
    assert vessel.C_B is not None, "C_B cannot be None"

    def seek_v_given_power(v, vessel, edge):
        """function to optimize"""
        # TODO: check it this needs to be made more general, now relies on ['Info'] to be present
        # water depth from the edge
        h_0 = edge["Info"]["GeneralDepth"]
        try:
            h_0 = vessel.calculate_h_squat(v, h_0)
        except AttributeError:
            # no squat available
            pass
        # TODO: consider precomputing a range v/h combinations for the ship before the simulation starts
        vessel.calculate_total_resistance(v, h_0)

        # compute total power given
        P_given = vessel.calculate_total_power_required(v=v, h_0=h_0)
        if isinstance(vessel.P_tot, complex):
            raise ValueError(f"P tot is complex: {vessel.P_tot}")

        # compute difference between power setting by captain (incl hotel) and power needed for velocity (incl hotel)
        diff = vessel.P_tot_given - P_given  # vessel.P_tot
        logger.debug(f"optimizing for v: {v}, P_tot_given: {vessel.P_tot_given}, P_tot {vessel.P_tot}, P_given {P_given}")

        return diff**2

    # fill in some of the parameters that we already know
    fun = functools.partial(seek_v_given_power, vessel=vessel, edge=edge)
    # lookup a minimum
    fit = scipy.optimize.minimize_scalar(fun, bounds=(0, upperbound), method="bounded", options=dict(xatol=0.0000001))

    # check if we found a minimum
    if not fit.success:
        raise ValueError(fit)
    logger.debug(f"fit: {fit}")

    return fit.x


# %% ENERGY FUNCTIONS


def sample_engine_age(L_w):
    """
    Samples the age of the engine based on the weight class of the vessel. The age is
    drawn randomly from a Weibull distribution with parameters k and lmb, which depend
    on the weight class of the ship. The age is then returned in years.

    TODO: add reference to literature on which this is based --> H5 over Performance

    Parameters
    ----------
    L_w : int
        weight class of the ship (depending on carrying capacity), values supported are
        1 (class L1), 2 (class L2) or 3 (class L3).

    Returns
    -------
    age : int
        The calculated age of the engine in years.
    """
    # check params
    if not L_w in [1, 2, 3]:
        raise ValueError("L_w should be 1, 2 or 3")

    # determine the shape (k) and scale factor (lmb) to use based on the weight class
    if L_w == 1:  # Weight class L1
        k = 1.3
        lmb = 20.5
    elif L_w == 2:  # Weight class L2
        k = 1.12
        lmb = 18.5
    elif L_w == 3:  # Weight class L3
        k = 1.26
        lmb = 18.6

    # the engine age
    # TODO: I would not expect a random distribution if the function is called
    age = int(np.random.weibull(k) * lmb)

    return age


def calculate_max_sinkage(v, h_0, T, B, C_B, width):
    """
    Calculate tha maximum sinkage of a moving vessel.

    The calculation equation is described in Barrass, B. & Derrett, R.'s book (2006),
    Ship Stability for Masters and Mates, chapter 42.
    https://doi.org/10.1016/B978-0-08-097093-6.00042-6

    Parameters
    ----------
    v : float
        Velocity of the vessel relative to the water [m/s]
    h_0 : float
        Water depth [m]
    T : float
        Actual draught of the vessel [m]
    B : float
        Breadth of the vessel [m]
    C_B : float
        Block coefficient of the vessel [-]
    width : float
        Width of the fairway [m]

    Returns
    -------
    float
        The maximum sinkage of the vessel [m]
    """
    # checks
    if v < 0:
        raise ValueError("Velocity v should be >= 0")
    if h_0 <= 0:
        raise ValueError("Water depth h_0 should be > 0")
    if T <= 0:
        raise ValueError("Draught T should be > 0")
    if B <= 0:
        raise ValueError("Breadth B should be > 0")
    if width <= 0:
        raise ValueError("Width of the fairway should be > 0")
    if C_B <= 0:
        raise ValueError("Block coefficient C_B should be > 0")

    if B > width:
        raise ValueError(f"Width of the fairway ({width}) should be larger than " f"the breadth of the vessel ({B})")

    # calculate the maximum sinkage
    return (C_B * ((B * T) / (width * h_0)) ** 0.81) * ((v * 1.94) ** 2.08) / 20


def calculate_properties(C_B, L, B, T, bulbous_bow, C_BB):
    """
    Calculate the properties of a vessel based on its block coefficient, length,
    breadth, draught, bulbous bow coefficient, and whether it has a bulbous bow.

    Parameters
    ----------
    C_B : float
        Block coefficient of the vessel [-]
    L : float
        Length of the vessel [m]
    B : float
        Breadth of the vessel [m]
    T : float
        Actual draught of the vessel [m]
    bulbous_bow : bool
        Whether the vessel has a bulbous bow or not
    C_BB : float
        Bulbous bow coefficient of the vessel [-]

    Returns
    -------
    C_M : float
        Midship section coefficient [-]
    C_WP : float
        Waterplane coefficient [-]
    C_P : float
        Prismatic coefficient [-]
    delta : float
        Water displacement of the vessel [m^3]
    lcb : float
        Longitudinal center of buoyancy [m]
    L_R : float
        Length parameter reflecting the length of the run [m]
    A_T : float
        Transverse area of the transom [m^2]
    A_BT : float
        Cross-sectional area of the bulb at still water level [m^2]
    S : float
        Total wetted surface area of the vessel [m^2]
    S_APP : float
        Wet area of appendages [m^2]
    S_B : float
        Area of flat bottom [m^2]
    T_F : float
        Forward draught of the vessel [m]
    h_B : float
        Position of the centre of the transverse area [m]
    D_s : float, optional
        Diameter of the screw [m], if not provided, it is assumed to be 0.7 * T

    """

    # TODO: add properties for seagoing ships with bulbs

    # (Van Koningsveld et al (2023) - Part IV Eq 5.9, 5.10 and below Eq 5.12)
    C_M = 1.006 - 0.0056 * C_B ** (-3.56)  # Midship section coefficient (Eq 5.9)
    C_WP = (1 + 2 * C_B) / 3  # Waterplane coefficient (Eq 5.10)
    C_P = C_B / C_M  # Prismatic coefficient (see below Eq 5.12)

    # Segers (2021) (http://resolver.tudelft.nl/uuid:a260bc48-c6ce-4f7c-b14a-e681d2e528e3)
    # Appendix C - Eq C.2
    delta = C_B * L * B * T  # Water displacement

    # Van Koningsveld et al (2023) - Part IV Table 5.1
    lcb = -13.5 + 19.4 * C_P  # longitudinal center of buoyancy
    # Van Koningsveld et al (2023) - Part IV Eq 5.13
    L_R = L * (
        1 - C_P + ((0.06 * C_P * lcb) / (4 * C_P - 1)) * (19.4 * C_P - 13.5)
    )  # length parameter reflecting the length of the run

    # Van Koningsveld et al (2023) - below Eq 5.16
    A_T = 0.1 * B * T  # transverse area of the transom
    # calculation for A_BT (cross-sectional area of the bulb at still water level [m^2]) depends on whether a ship has a bulb
    if bulbous_bow:
        # TODO: check Holtrop and Mennen for this formulation
        A_BT = C_BB * B * T * C_M  # calculate A_BT for seagoing ships having a bulb
    else:
        A_BT = 0  # most inland ships do not have a bulb. So we assume A_BT=0.

    # Total wet area: S (Van Koningsveld et al (2023) - Eq 5.8)
    assert C_M >= 0, f"C_M should be positive: {C_M}"
    S = L * (2 * T + B) * np.sqrt(C_M) * (0.453 + 0.4425 * C_B - 0.2862 * C_M - 0.003467 * (B / T) + 0.3696 * C_WP) + 2.38 * (
        A_BT / C_B
    )

    # Segers (2021) (http://resolver.tudelft.nl/uuid:a260bc48-c6ce-4f7c-b14a-e681d2e528e3)
    # In the explanation under Eq 3.27
    S_APP = 0.05 * S  # Wet area of appendages
    # Segers (2021) Eq 3.20
    S_B = L * B  # Area of flat bottom

    # TODO: we D_s is a property that should be given, not calculated
    # if D_s is None:
    #     D_s = 0.7 * T  # Diameter of the screw

    # TODO: check references for these equations
    T_F = T  # Forward draught of the vessel [m]
    h_B = 0.2 * T  # Position of the centre of the transverse area [m]

    return C_M, C_WP, C_P, delta, lcb, L_R, A_T, A_BT, S, S_APP, S_B, T_F, h_B


def calculate_frictional_resistance(v, h_0, L, nu, T, S, S_B, rho):
    """
    Calculate the frictional resistance of a vessel.

    This is the first resistance component defined by Holtrop and Mennen (1982). A
    modification to the original friction line is applied, based on literature of
    Zeng (2018), to account for shallow water effects.

    Parameters
    ----------
    v : float
        Velocity of the vessel relative to the water [m/s]
    h_0 : float
        Water depth [m]
    L : float
        Length of the vessel [m]
    nu : float
        Kinematic viscosity of the water [m^2/s]
    T : float
        Actual draught of the vessel [m]
    S : float
        Total wetted surface area of the vessel [m^2]
    S_B : float
        Area of the flat bottom of the vessel [m^2]
    rho : float
        Density of the water [kg/m^3]

    Returns
    -------
    R_f : float
        Frictional resistance of the vessel [kN]
    C_f : float
        Friction coefficient of the vessel [-]
    R_e : float
        Reynolds number of the vessel [-]
    Cf_deep : float
        Friction coefficient in deep water based on CFD computations of
        Zeng et al. (2018) [-]
    Cf_shallow : float
        Friction coefficient in shallow water based on CFD computations of
        Zeng et al. (2018) [-]
    Cf_0 : float
        Friction coefficient in deep water according to ITTC-1957 curve [-]
    Cf_Katsui : float
        Friction coefficient according to Katsui (1978) [-]
    V_B : float
        Average velocity underneath the ship, taking into account the shallow water
        effect [m/s]
    D : float
        Distance from the bottom of the ship to the bottom of the fairway [m]
    a : float
        Coefficient needed to calculate the Katsui friction coefficient [-]
    """
    # TODO: makes sense to store np.log10(R_e) as constant intstead of re-calculating
    # Reynolds number
    R_e = v * L * nu

    # distance from bottom ship to the bottom of the fairway
    D = h_0 - T
    if not D > 0:
        raise ValueError(f"Distance between ship and bottom should be > 0: {D}")

    # Friction coefficient based on CFD computations of Zeng et al. (2018), in deep
    # water --> Van Koningsveld et al (2023) - Eq 5.3
    Cf_deep = 0.08169 / ((np.log10(R_e) - 1.717) ** 2)
    assert not isinstance(Cf_deep, complex), f"Cf_deep should not be complex: {Cf_deep}"

    # Friction coefficient based on CFD computations of Zeng et al. (2018), taking into
    # account shallow water effects --> Van Koningsveld et al (2023) - Eq 5.4
    Cf_shallow = (0.08169 / ((np.log10(R_e) - 1.717) ** 2)) * (1 + (0.003998 / (np.log10(R_e) - 4.393)) * (D / L) ** (-1.083))
    assert not isinstance(Cf_shallow, complex), f"Cf_shallow should not be complex: {Cf_shallow}"

    # Friction coefficient in deep water according to ITTC-1957 curve
    # Van Koningsveld et al (2023) - Eq 5.6
    Cf_0 = 0.075 / ((np.log10(R_e) - 2) ** 2)

    # 'a' is the coefficient needed to calculate the Katsui friction coefficient
    # Van Koningsveld et al (2023) - below Eq 5.7
    a = 0.042612 * np.log10(R_e) + 0.56725

    # Van Koningsveld et al (2023) - Eq 5.7
    # TODO: may lead to "invalid value encountered in scalar power"
    # see https://github.com/orgs/TUDelft-CITG/projects/3/views/1?pane=issue&itemId=118343899&issue=TUDelft-CITG%7COpenTNSim%7C100
    Cf_Katsui = 0.0066577 / (np.log10(R_e) - 4.3762) ** a

    # The average velocity underneath the ship, taking into account the shallow water
    # effect. This calculation is to get V_B, which will be used in the following Cf
    # for shallow water equation:
    if h_0 / T <= 4:
        V_B = 0.4277 * v * np.exp((h_0 / T) ** (-0.07625))
    else:
        V_B = v

    # cf_shallow and cf_deep cannot be applied directly, since a vessel also has
    # non-horizontal wet surfaces that have to be taken into account. Therefore, the
    # following formula for the final friction coefficient 'C_f' for deep water or
    # shallow water is defined according to Zeng et al. (2018)
    if (h_0 - T) / L > 1:
        # calculate Friction coefficient C_f for deep water:
        # Zeng et al. (2018)
        C_f = Cf_0 + (Cf_deep - Cf_Katsui) * (S_B / S)
        logger.debug("now i am in the deep loop")
    else:
        # calculate Friction coefficient C_f for shallow water:
        # Van Koningsveld et al (2023) - Eq 5.5
        C_f = Cf_0 + (Cf_shallow - Cf_Katsui) * (S_B / S) * (V_B / v) ** 2
        logger.debug("now i am in the shallow loop")
    assert not isinstance(C_f, complex), f"C_f should not be complex: {C_f}"

    # The total frictional resistance R_f [kN]:
    # Van Koningsveld et al (2023) - Eq 5.2
    R_f = (0.5 * rho * (v**2) * C_f * S) / 1000
    assert not isinstance(R_f, complex), f"R_f should not be complex: {R_f}"

    return R_f, C_f, R_e, Cf_deep, Cf_shallow, Cf_0, Cf_Katsui, V_B, D, a


def calculate_viscous_resistance(c_stern, B, L, T, L_R, C_P, R_f, delta):
    """
    Calculate the viscous resistance of a vessel.

    This is the second resistance component defined by Holtrop and Mennen (1982). Form
    factor (1 + k1) has to be multiplied by the frictional resistance R_f, to account
    for the effect of viscosity.

    Parameters
    ----------
    c_stern : float
        Determines the shape of the afterbody [-]
    B : float
        Breadth of the vessel [m]
    L : float
        Length of the vessel [m]
    T : float
        Actual draught of the vessel [m]
    L_R : float
        Length parameter reflecting the length of the run [m]
    C_P : float
        Prismatic coefficient of the vessel [-]
    R_f : float
        Frictional resistance of the vessel [kN]
    delta : float
        Water displacement of the vessel [m^3]

    Returns
    -------
    c_14 : float
        Coefficient accounting for the specific shape of the afterbody [-]
    one_k1 : float
        Form factor (1 + k1) describing the viscous resistance [-]
    R_f_one_k1 : float
        Viscous resistance of the vessel, which is the product of the frictional
        resistance R_f and the form factor (1 + k1) [kN]
    """
    # c_14 accounts for the specific shape of the afterbody
    # TODO: check where this value comes from (Holtrop and Mennen?) (following
    # Segers (2021) we assume c_stern = 0 which leads to c_14 to be 1)
    c_14 = 1 + 0.0011 * c_stern

    # the form factor (1+k1) describes the viscous resistance
    # Van Koningsveld et al (2023) - Eq 5.12
    # TODO: consider to rename delta to nabla
    one_k1 = 0.93 + 0.487 * c_14 * ((B / L) ** 1.068) * ((T / L) ** 0.461) * ((L / L_R) ** 0.122) * (
        ((L**3) / delta) ** 0.365
    ) * ((1 - C_P) ** (-0.604))

    R_f_one_k1 = R_f * one_k1

    return c_14, one_k1, R_f_one_k1


def calculate_appendage_resistance(v, rho, S_APP, one_k2, C_f):
    """
    Calculate the frictional resistance resulting from the wetted area of appendages.
    This function computes the appendage resistance (R_APP) in kilonewtons (kN) based
    on the provided parameters, using the formula from Segers (2021) - Eq 3.27.

    Parameters
    ----------
    v : float
        Ship velocity in meters per second (m/s).
    rho : float
        Water density in kilograms per cubic meter (kg/m^3).
    S_APP : float
        Wetted surface area of appendages in square meters (m^2).
    one_k2 : float
        Form factor for appendages (dimensionless).
    C_f : float
        Frictional resistance coefficient (dimensionless).

    Returns
    -------
    float
        Frictional resistance of appendages (R_APP) in kilonewtons (kN).

    References
    ----------
    Segers (2021) - Eq 3.27. http://resolver.tudelft.nl/uuid:a260bc48-c6ce-4f7c-b14a-e681d2e528e3
    """
    # Frictional resistance resulting from wetted area of appendages: R_APP [kN]
    # Segers (2021) - Eq 3.27 (http://resolver.tudelft.nl/uuid:a260bc48-c6ce-4f7c-b14a-e681d2e528e3)
    R_APP = (0.5 * rho * (v**2) * S_APP * one_k2 * C_f) / 1000

    return R_APP


def karpov(v, h_0, g, T):
    """
    Calculate the corrected velocity and alpha coefficient using the Karpov method.
    The Karpov method applies a correction factor (alpha_xx) to the velocity based on
    the Froude number and the ratio of water depth to draft. The correction is
    determined using polynomial approximations for different ranges of the Froude
    number and h_0/T.

    Parameters
    ----------
    v : float
        The measured velocity (m/s).
    h_0 : float
        The water depth (m).
    g : float
        The gravitational acceleration (m/s^2). Must be non-negative.
    T : float
        The draft (m).

    Returns
    -------
    F_rh : float
        The Froude number based on depth (dimensionless).
    V_2 : float
        The corrected velocity (m/s) according to the Karpov method.
    alpha_xx : float
        The correction coefficient applied to the velocity.


    Notes
    -----
    - The function uses a piecewise polynomial fit for the correction coefficient based
      on the Froude number and h_0/T.
    - The Froude number is calculated as v / sqrt(g * h_0).
    """

    # The Froude number used in the Karpov method is the depth related froude number
    # F_rh

    # The different alpha** curves are determined with a sixth power polynomial
    # approximation in Excel
    # A distinction is made between different ranges of Froude numbers, because this
    # resulted in a better approximation of the curve
    assert g >= 0, f"g should be positive: {g}"
    assert h_0 >= 0, f"h_0 should be positive: {h_0}"
    F_rh = v / np.sqrt(g * h_0)

    if F_rh <= 0.4:
        if 0 <= h_0 / T < 1.75:
            alpha_xx = (-4 * 10 ** (-12)) * F_rh**3 - 0.2143 * F_rh**2 - 0.0643 * F_rh + 0.9997
        if 1.75 <= h_0 / T < 2.25:
            alpha_xx = -0.8333 * F_rh**3 + 0.25 * F_rh**2 - 0.0167 * F_rh + 1
        if 2.25 <= h_0 / T < 2.75:
            alpha_xx = -1.25 * F_rh**4 + 0.5833 * F_rh**3 - 0.0375 * F_rh**2 - 0.0108 * F_rh + 1
        if h_0 / T >= 2.75:
            alpha_xx = 1

    if F_rh > 0.4:
        if 0 <= h_0 / T < 1.75:
            alpha_xx = (
                -0.9274 * F_rh**6
                + 9.5953 * F_rh**5
                - 37.197 * F_rh**4
                + 69.666 * F_rh**3
                - 65.391 * F_rh**2
                + 28.025 * F_rh
                - 3.4143
            )
        if 1.75 <= h_0 / T < 2.25:
            alpha_xx = (
                2.2152 * F_rh**6
                - 11.852 * F_rh**5
                + 21.499 * F_rh**4
                - 12.174 * F_rh**3
                - 4.7873 * F_rh**2
                + 5.8662 * F_rh
                - 0.2652
            )
        if 2.25 <= h_0 / T < 2.75:
            alpha_xx = (
                1.2205 * F_rh**6
                - 5.4999 * F_rh**5
                + 5.7966 * F_rh**4
                + 6.6491 * F_rh**3
                - 16.123 * F_rh**2
                + 9.2016 * F_rh
                - 0.6342
            )
        if 2.75 <= h_0 / T < 3.25:
            alpha_xx = (
                -0.4085 * F_rh**6
                + 4.534 * F_rh**5
                - 18.443 * F_rh**4
                + 35.744 * F_rh**3
                - 34.381 * F_rh**2
                + 15.042 * F_rh
                - 1.3807
            )
        if 3.25 <= h_0 / T < 3.75:
            alpha_xx = (
                0.4078 * F_rh**6 - 0.919 * F_rh**5 - 3.8292 * F_rh**4 + 15.738 * F_rh**3 - 19.766 * F_rh**2 + 9.7466 * F_rh - 0.6409
            )
        if 3.75 <= h_0 / T < 4.5:
            alpha_xx = (
                0.3067 * F_rh**6
                - 0.3404 * F_rh**5
                - 5.0511 * F_rh**4
                + 16.892 * F_rh**3
                - 20.265 * F_rh**2
                + 9.9002 * F_rh
                - 0.6712
            )
        if 4.5 <= h_0 / T < 5.5:
            alpha_xx = (
                0.3212 * F_rh**6
                - 0.3559 * F_rh**5
                - 5.1056 * F_rh**4
                + 16.926 * F_rh**3
                - 20.253 * F_rh**2
                + 10.013 * F_rh
                - 0.7196
            )
        if 5.5 <= h_0 / T < 6.5:
            alpha_xx = (
                0.9252 * F_rh**6
                - 4.2574 * F_rh**5
                + 5.0363 * F_rh**4
                + 3.3282 * F_rh**3
                - 10.367 * F_rh**2
                + 6.3993 * F_rh
                - 0.2074
            )
        if 6.5 <= h_0 / T < 7.5:
            alpha_xx = (
                0.8442 * F_rh**6 - 4.0261 * F_rh**5 + 5.313 * F_rh**4 + 1.6442 * F_rh**3 - 8.1848 * F_rh**2 + 5.3209 * F_rh - 0.0267
            )
        if 7.5 <= h_0 / T < 8.5:
            alpha_xx = (
                0.1211 * F_rh**6 + 0.628 * F_rh**5 - 6.5106 * F_rh**4 + 16.7 * F_rh**3 - 18.267 * F_rh**2 + 8.7077 * F_rh - 0.4745
            )

        if 8.5 <= h_0 / T < 9.5:
            if F_rh < 0.6:
                alpha_xx = 1
            if F_rh >= 0.6:
                alpha_xx = (
                    -6.4069 * F_rh**6
                    + 47.308 * F_rh**5
                    - 141.93 * F_rh**4
                    + 220.23 * F_rh**3
                    - 185.05 * F_rh**2
                    + 79.25 * F_rh
                    - 12.484
                )
        if h_0 / T >= 9.5:
            if F_rh < 0.6:
                alpha_xx = 1
            if F_rh >= 0.6:
                alpha_xx = (
                    -6.0727 * F_rh**6
                    + 44.97 * F_rh**5
                    - 135.21 * F_rh**4
                    + 210.13 * F_rh**3
                    - 176.72 * F_rh**2
                    + 75.728 * F_rh
                    - 11.893
                )

    V_2 = v / alpha_xx

    return F_rh, V_2, alpha_xx


def calculate_wave_resistance(v, h_0, g, T, L, B, C_P, C_WP, lcb, L_R, A_T, C_M, delta, rho):
    """
    Calculate the wave resistance and related hydrodynamic coefficients for a ship.

    Parameters
    ----------
    v : float
        Ship's speed relative to water (m/s).
    h_0 : float
        Water depth (m).
    g : float
        Gravitational acceleration (m/s^2).
    T : float
        Ship's draft (m).
    L : float
        Ship's length at waterline (m).
    B : float
        Ship's beam at waterline (m).
    C_P : float
        Prismatic coefficient (dimensionless).
    C_WP : float
        Waterplane area coefficient (dimensionless).
    lcb : float
        Longitudinal center of buoyancy (as a fraction of L, dimensionless).
    L_R : float
        Length of run (m).
    A_T : float
        Transom area (m^2).
    C_M : float
        Midship section coefficient (dimensionless).
    delta : float
        Displacement volume (m^3).
    rho : float
        Water density (kg/m^3).

    Returns
    -------
    F_rL : float
        Froude number based on ship's speed and length.
    i_E : float
        Half angle of entrance (degrees).
    c_1 : float
        Empirical coefficient for wave resistance.
    c_2 : float
        Bulbous bow effect coefficient.
    c_5 : float
        Transom stern influence coefficient.
    c_7 : float
        Coefficient based on B/L ratio.
    c_15 : float
        Coefficient based on L^3/delta ratio.
    c_16 : float
        Coefficient based on prismatic coefficient.
    lmbda : float
        Lambda parameter for wave resistance calculation.
    m_1 : float
        Exponential coefficient for wave resistance.
    m_2 : float
        Cosine coefficient for wave resistance.
    R_W : float
        Calculated wave resistance (kN).

    """
    # checks
    assert g >= 0, f"g should be positive: {g}"
    assert L >= 0, f"L should be positive: {L}"

    F_rL = v / np.sqrt(g * L)  # Froude number based on ship's speed to water and its length of waterline

    # parameter c_7 is determined by the B/L ratio
    # Van Koningsveld et al (2023) - Part IV Table 5.1
    if B / L < 0.11:
        c_7 = 0.229577 * (B / L) ** 0.33333
    if B / L > 0.25:
        c_7 = 0.5 - 0.0625 * (L / B)
    else:
        c_7 = B / L

    # half angle of entrance in degrees
    # Van Koningsveld et al (2023) - Part IV Table 5.1
    i_E = 1 + 89 * np.exp(
        -((L / B) ** 0.80856)
        * ((1 - C_WP) ** 0.30484)
        * ((1 - C_P - 0.0225 * lcb) ** 0.6367)
        * ((L_R / B) ** 0.34574)
        * ((100 * delta / (L**3)) ** 0.16302)
    )

    # Van Koningsveld et al (2023) - Part IV Table 5.1
    c_1 = 2223105 * (c_7**3.78613) * ((T / B) ** 1.07961) * (90 - i_E) ** (-1.37165)
    c_2 = 1  # accounts for the effect of the bulbous bow, which is not present at inland ships
    c_5 = 1 - (0.8 * A_T) / (B * T * C_M)  # influence of the transom stern on the wave resistance

    # parameter c_15 depoends on the ratio L^3 / delta
    # Van Koningsveld et al (2023) - Part IV Table 5.1
    if (L**3) / delta < 512:
        c_15 = -1.69385
    if (L**3) / delta > 1727:
        c_15 = 0
    else:
        c_15 = -1.69385 + (L / (delta ** (1 / 3)) - 8) / 2.36

    # parameter c_16 depends on C_P
    # Van Koningsveld et al (2023) - Part IV Table 5.1
    if C_P < 0.8:
        c_16 = 8.07981 * C_P - 13.8673 * (C_P**2) + 6.984388 * (C_P**3)
    else:
        c_16 = 1.73014 - 0.7067 * C_P

    if L / B < 12:
        lmbda = 1.446 * C_P - 0.03 * (L / B)
    else:
        lmbda = 1.446 * C_P - 0.36

    # Van Koningsveld et al (2023) - Part IV Table 5.1
    m_1 = 0.0140407 * (L / T) - 1.75254 * ((delta) ** (1 / 3) / L) - 4.79323 * (B / L) - c_16
    # Van Koningsveld et al (2023) - Part IV Table 5.1
    m_2 = c_15 * (C_P**2) * np.exp((-0.1) * (F_rL ** (-2)))

    # Van Koningsveld et al (2023) - Part IV Eq 5.16
    R_W = c_1 * c_2 * c_5 * delta * rho * g * np.exp(m_1 * (F_rL ** (-0.9)) + m_2 * np.cos(lmbda * (F_rL ** (-2)))) / 1000  # kN

    return F_rL, i_E, c_1, c_2, c_5, c_7, c_15, c_16, lmbda, m_1, m_2, R_W


def calculate_residual_resistance(V_2, g, A_T, B, C_WP, rho, T, L, C_B, S, T_F, h_B, A_BT, bulbous_bow):
    """
    Calculate the residual resistance of a ship, which includes the resistance due to
    the immersed transom, model-ship correlation resistance, and bulbous bow resistance.

    This function computes the residual resistance components based on the ship's
    speed, transom area, breadth, waterplane coefficient, density, draft, length,
    block coefficient, wetted surface area, and bulbous bow presence.

    Parameters
    ----------
    V_2 : float
        The corrected velocity (m/s) according to the Karpov method.
    g : float
        Gravitational acceleration (m/s^2).
    A_T : float
        Traverse area of the transom (m^2). Van Koningsveld et al (2023) - below Eq 5.16
    B : float
        Breadth of the ship (m).
    C_WP : float
        Waterplane coefficient (dimensionless).
    rho : float
        Density of the water (kg/m^3).
    T : float
        Actual draft of the ship (m).
    L : float
        Length of the ship (m).
    C_B : float
        Block coefficient of the ship (dimensionless).
    S : float
        Wetted surface area of the ship (m^2).
    T_F : float
        Draft at the forefoot (m).
    h_B : float
        Height of the bulbous bow (m).
    A_BT : float
        Area of the bulbous bow transom (m^2).
    bulbous_bow : bool
        Whether the ship has a bulbous bow (True) or not (False).

    Returns
    -------
    F_nT : float
        Froude number based on transom immersion (dimensionless).
    c_6 : float
        Coefficient for resistance due to immersed transom (dimensionless).
    R_TR : float
        Resistance due to immersed transom (kN).
    c_4 : float
        Coefficient for model-ship correlation resistance (dimensionless).
    c_2 : float
        Coefficient for model-ship correlation resistance (dimensionless).
    C_A : float
        Model-ship correlation resistance coefficient (dimensionless).
    R_A : float
        Model-ship correlation resistance (kN).
    F_ni : float
        Froude number based on immersion of bulbous bow (dimensionless).
    P_B : float
        Coefficient for the emergence of bulbous bow (dimensionless).
    R_B : float
        Resistance due to the bulbous bow (kN).
    R_res : float
        Total residual resistance (kN).

    """

    # Resistance due to immersed transom: R_TR [kN]
    F_nT = V_2 / np.sqrt(2 * g * A_T / (B + B * C_WP))  # Froude number based on transom immersion
    assert not isinstance(F_nT, complex), f"residual? froude number should not be complex: {F_nT}"

    c_6 = 0.2 * (1 - 0.2 * F_nT)  # Assuming F_nT < 5, this is the expression for coefficient c_6

    R_TR = (0.5 * rho * (V_2**2) * A_T * c_6) / 1000

    # Model-ship correlation resistance: R_A [kN]

    if T / L < 0.04:
        c_4 = T / L
    else:
        c_4 = 0.04
    c_2 = 1

    C_A = 0.006 * (L + 100) ** (-0.16) - 0.00205 + 0.003 * np.sqrt(L / 7.5) * (C_B**4) * c_2 * (0.04 - c_4)
    assert not isinstance(C_A, complex), f"C_A number should not be complex: {C_A}"

    R_A = (0.5 * rho * (V_2**2) * S * C_A) / 1000  # kW

    # Resistance due to the bulbous bow (R_B)

    # Froude number based on immersoin of bulbous bow [-]
    F_ni = V_2 / np.sqrt(g * (T_F - h_B - 0.25 * np.sqrt(A_BT) + 0.15 * V_2**2))

    P_B = (0.56 * np.sqrt(A_BT)) / (T_F - 1.5 * h_B)  # P_B is coefficient for the emergence of bulbous bow
    if bulbous_bow:
        R_B = ((0.11 * np.exp(-3 * P_B**2) * F_ni**3 * A_BT**1.5 * rho * g) / (1 + F_ni**2)) / 1000
    else:
        R_B = 0

    R_res = R_TR + R_A + R_B

    return F_nT, c_6, R_TR, c_4, c_2, C_A, R_A, F_ni, P_B, R_B, R_res


def calculate_total_resistance(v, g, h_0, C_B, L, B, T, bulbous_bow, C_BB, nu, rho, c_stern, one_k2):
    """
    Calculate the total resistance of a ship, which includes frictional, viscous,
    appendage, wave, and residual resistance components.

    The total resistance
    R_tot [kN] = R_f * (1+k1) + R_APP + R_W + R_TR + R_A

    Parameters
    ----------
    v : float
        Ship's speed relative to water (m/s).
    g : float
        Gravitational acceleration (m/s^2).
    h_0 : float
        Water depth (m).
    C_B : float
        Block coefficient of the ship (dimensionless).
    L : float
        Length of the ship (m).
    B : float
        Breadth of the ship (m).
    T : float
        Actual draft of the ship (m).
    bulbous_bow : bool
        Whether the ship has a bulbous bow (True) or not (False).
    C_BB : float
        Breadth coefficient of bulbous bow (dimensionless).
    nu : float
        Kinematic viscosity of water (m^2/s).
    rho : float
        Density of water (kg/m^3).
    c_stern : float
        Determines the shape of the afterbody (dimensionless).
    one_k2 : float
        Appendage resistance factor (1 + k2) (dimensionless).

    Returns
    -------
    R_tot : float
        Total resistance of the ship (kN).
    """
    # TODO: this function is rhather odd as it calls all other resistance functions,
    # computing lots of unused values (which are set in the corresponding method),
    # hence the function is not yet used in the main class method.

    # vessel properties
    C_M, C_WP, C_P, delta, lcb, L_R, A_T, A_BT, S, S_APP, S_B, T_F, h_B = calculate_properties(C_B, L, B, T, bulbous_bow, C_BB)

    # frictional resistance
    R_f, C_f, _, _, _, _, _, _, _, _ = calculate_frictional_resistance(v, h_0, L, nu, T, S, S_B, rho)

    # viscous resistance
    _, one_k1, _ = calculate_viscous_resistance(c_stern, B, L, T, L_R, C_P, R_f, delta)

    # appendage resistance
    R_APP = calculate_appendage_resistance(v, rho, S_APP, one_k2, C_f)

    # wave resistance
    _, _, _, _, _, _, _, _, _, _, _, R_W = calculate_wave_resistance(v, h_0, g, T, L, B, C_P, C_WP, lcb, L_R, A_T, C_M, delta, rho)

    # residual resistance
    V_2 = v  # TODO: correct? this is how it is done in the original method
    _, _, R_TR, _, _, _, R_A, _, _, R_B, _ = calculate_residual_resistance(
        V_2, g, A_T, B, C_WP, rho, T, L, C_B, S, T_F, h_B, A_BT, bulbous_bow
    )

    # The total resistance R_tot [kN] = R_f * (1+k1) + R_APP + R_W + R_TR + R_A
    R_tot = R_f * one_k1 + R_APP + R_W + R_TR + R_A + R_B

    return R_tot


def calculate_total_power_required(
    v,
    h_0,
    R_tot,
    F_rL,
    x,
    C_B,
    delta,
    D_s,
    eta_o,
    eta_r,
    eta_t,
    eta_g,
    P_hotel,
    P_installed,
):
    """
    Calculate the total power required for a ship based on its speed, resistance,
    and various efficiency factors.

    Parameters
    ----------
    v : float
        Ship's speed relative to water (m/s).
    h_0 : float
        Water depth (m).
    R_tot : float
        Total resistance of the ship (kN).
    F_rL : float
        Froude number based on ship's speed and length (dimensionless).
    x : float
        Propeller design factor (dimensionless).
    C_B : float
        Block coefficient of the ship (dimensionless).
    delta : float
        Displacement volume of the ship (m^3).
    D_s : float
        Ship's draft (m).
    eta_o : float
        Overall efficiency of the propulsion system (dimensionless).
    eta_r : float
        Efficiency of the reduction gear (dimensionless).
    eta_t : float
        Efficiency of the transmission (dimensionless).
    eta_g : float
        Efficiency of the generator (dimensionless).
    P_hotel : float
        Hotel load power (kW).
    P_installed : float
        Installed power of the ship (kW).

    Returns
    -------
    P_e : float
        Required power for propulsion (kW).
    dw : float
        Velocity correction coefficient (dimensionless).
    w : float
        Wake fraction (dimensionless).
    t : float
        Thrust deduction factor (dimensionless).
    eta_h : float
        Hull efficiency (dimensionless).
    P_d : float,
    P_b : float
    P_propulsion : float
        Power required for propulsion (kW).
    P_tot : float
        Total power required for the ship (kW).
    P_given : float
        Power given by the installed power (kW).
    P_partial : float
        Partial power required for propulsion (kW).

    Notes
    -----
    In this version, we define the propulsion power as P_d (Delivered Horse Power)
    rather than P_b (Brake Horse Power). The reason we choose P_d as propulsion
    power is to prevent double use of the same power efficiencies. The details are:

    1. The P_b calculation involves gearing efficiency and transmission efficiency
    already, while P_d does not.
    2. P_d is the power delivered to propellers.
    3. To estimate the renewable fuel use, we will involve 'energy conversion
    efficiencies' later in the calculation.

    The 'energy conversion efficiencies' for renewable fuel powered vessels are
    commonly measured/given as a whole covering the engine power systems, including
    different engines (such as fuel cell engine, battery engine, internal
    combustion engine, hybrid engine) efficiencies, and corresponding gearbox
    efficiencies, AC/DC converter efficiencies, excluding the efficiency items of
    propellers.

    Therefore, to align with the later use of 'energy conversion efficiencies' for
    fuel use estimation and to prevent double use of some power efficiencies, such
    as gearing efficiency, here we choose P_d as propulsion power.

    """
    # Required power for propulsion
    # Effective Horse Power (EHP), P_e (Van Koningsveld et al (2023) - Part IV Eq 5.17)
    P_e = v * R_tot

    # Segers (2021) (http://resolver.tudelft.nl/uuid:a260bc48-c6ce-4f7c-b14a-e681d2e528e3)
    # Appendix C
    if F_rL < 0.2:
        dw = 0  # the velocity correction coefficient is 0 when FrL is smaller than 0.2
    else:
        dw = 0.1  # otherwise the velocity correction coefficient is 0.1

    # Segers (2021) (http://resolver.tudelft.nl/uuid:a260bc48-c6ce-4f7c-b14a-e681d2e528e3)
    # Appendix C - Eq C.1
    w = 0.11 * (0.16 / x) * C_B * np.sqrt((delta ** (1 / 3)) / D_s) - dw  # wake fraction 'w'

    assert not isinstance(w, complex), f"w should not be complex: {w}"

    if x == 1:
        # (Van Koningsveld et al (2023) - Part IV Eq 5.22)
        t = 0.6 * w * (1 + 0.67 * w)  # thrust deduction factor 't'
    else:
        # (Van Koningsveld et al (2023) - Part IV Eq 5.23)
        t = 0.8 * w * (1 + 0.25 * w)

    eta_h = (1 - t) / (1 - w)  # hull efficiency eta_h

    # TODO: check below suggestions. They were made to allow for better translation to alternative energy sources. But the changes induced unexpected behaviour.
    # Calculation hydrodynamic efficiency eta_D  according to Simic et al (2013) "On Energy Efficiency of Inland
    # Waterway Self-Propelled Cargo Vessels", https://www.researchgate.net/publication/269103117
    # hydrodynamic efficiency eta_D is a ratio of power used to propel the ship and delivered power
    # relation between eta_D and ship velocity v

    # if h_0 >= 9:
    #     if F_rh >= 0.5:
    #         eta_D = 0.6
    #     elif 0.325 <= F_rh < 0.5:
    #         eta_D = 0.7
    #     elif 0.28 <= F_rh < 0.325:
    #         eta_D = 0.59
    #     elif 0.2 < F_rh < 0.28:
    #         eta_D = 0.56
    #     elif 0.17 < F_rh <= 0.2:
    #         eta_D = 0.41
    #     elif 0.15 < F_rh <= 0.17:
    #         eta_D = 0.35
    #     else:
    #         eta_D = 0.29
    #
    # elif 5 <= h_0 < 9:
    #     if F_rh > 0.62:
    #         eta_D = 0.7
    #     elif 0.58 < F_rh < 0.62:
    #         eta_D = 0.68
    #     elif 0.57 < F_rh <= 0.58:
    #         eta_D = 0.7
    #     elif 0.51 < F_rh <= 0.57:
    #         eta_D = 0.68
    #     elif 0.475 < F_rh <= 0.51:
    #         eta_D = 0.53
    #     elif 0.45 < F_rh <= 0.475:
    #         eta_D = 0.4
    #     elif 0.36 < F_rh <= 0.45:
    #         eta_D = 0.37
    #     elif 0.33 < F_rh <= 0.36:
    #         eta_D = 0.36
    #     elif 0.3 < F_rh <= 0.33:
    #         eta_D = 0.35
    #     elif 0.28 < F_rh <= 0.3:
    #         eta_D = 0.331
    #     else:
    #         eta_D = 0.33
    # else:
    #     if F_rh > 0.56:
    #         eta_D = 0.28
    #     elif 0.4 < F_rh <= 0.56:
    #         eta_D = 0.275
    #     elif 0.36 < F_rh <= 0.4:
    #         eta_D = 0.345
    #     elif 0.33 < F_rh <= 0.36:
    #         eta_D = 0.28
    #     elif 0.3 < F_rh <= 0.33:
    #         eta_D = 0.27
    #     elif 0.28 < F_rh <= 0.3:
    #         eta_D = 0.26
    #     else:
    #         eta_D = 0.25
    #
    # # Delivered Horse Power (DHP), P_d
    # P_d = P_e / eta_D

    # logger.debug("eta_D = {:.2f}".format(eta_D))

    # (Van Koningsveld et al (2023) - Part IV Eq 5.19)
    P_d = P_e / (eta_o * eta_r * eta_h)

    # Brake Horse Power (BHP), P_b (P_b was used in OpenTNsim version v1.1.2. we do not use it in this version. The reseaon is listed in the doc string above)
    # (Van Koningsveld et al (2023) - Part IV Eq 5.24)
    P_b = P_d / (eta_t * eta_g)

    # P_propulsion = P_d  # propulsion power is defined here as Delivered horse power, the power delivered to propellers
    P_propulsion = P_b  # propulsion power is defined here as Delivered horse power, the power delivered to propellers

    # TODO: consider to facilitate that all engine power can go into propulsion (Auxiliary generator for hotel)
    P_tot = P_hotel + P_propulsion

    # Partial engine load (P_partial): needed in the 'Emission calculations'
    if P_tot > P_installed:
        P_given = P_installed
        P_partial = 1
    else:
        P_given = P_tot
        P_partial = P_tot / P_installed

    logger.debug(f"The total power required is {P_tot} kW")
    logger.debug(f"The actual total power given is {P_given} kW")
    logger.debug(f"The partial load is {P_partial}")

    assert not isinstance(P_given, complex), f"P_given number should not be complex: {P_given}"

    return (
        P_e,
        dw,
        w,
        t,
        eta_h,
        P_d,
        P_b,
        P_propulsion,
        P_tot,
        P_given,
        P_partial,
    )  # , eta_D


# %% CLASSES


class ConsumesEnergy:
    """
    Mixin class: Something that consumes energy.

    Parameters
    ----------
    P_installed : float
         Installed engine power in kilowatts (kW).
    P_tot_given : float
        Total power set by captain (includes hotel power). When
        P_tot_given > P_installed; P_tot_given=P_installed.
    bulbous_bow : bool, optional
        Indicates if the ship has a bulbous bow. Inland ships generally do
        not have a bulbous bow, hence the default is False. If a ship has
        a bulbous bow, set to True.
    L_w : int
        Weight class of the ship depending on carrying capacity. Classes:
        L1 (=1), L2 (=2), L3 (=3).
    current_year : int
        The current year.
    nu : float
        Kinematic viscosity in square meters per second (m^2/s).
    rho : float
        Density of the surrounding water in kilograms per cubic meter
        (kg/m^3).
    g : float
        Gravitational acceleration in meters per second squared (m/s^2).
    x : int
        Number of propellers.
    eta_o : float
        Open water efficiency of the propeller.
    eta_r : float
        Relative rotative efficiency.
    eta_t : float
        Transmission efficiency.
    eta_g : float
        Gearing efficiency.
    c_stern : float
        Determines the shape of the afterbody.
    C_BB : float
        Breadth coefficient of the bulbous bow, set to 0.2 according to the
        paper of Kracht (1970), https://doi.org/10.5957/jsr.1970.14.1.1.
    C_B : float, optional
        Block coefficient ('fullness'), default to 0.85.
    one_k2 : float
        Appendage resistance factor (1+k2).
    C_year : int
        Construction year of the engine.
    """

    def __init__(
        self,
        P_installed,
        L_w,
        C_year=None,
        current_year=None,  # current_year
        bulbous_bow=False,
        P_hotel_perc=0.05,
        P_hotel=None,
        P_tot_given=None,  # the actual power engine setting
        nu=1 * 10 ** (-6),
        rho=1000,
        g=9.81,
        x=2,
        D_s=1.4,
        eta_o=0.4,
        eta_r=1.00,
        eta_t=0.98,
        eta_g=0.96,
        c_stern=0,
        C_BB=0.2,
        C_B=0.85,
        one_k2=2.5,  # following Segers (2021) we assume (1 + k2) to be 2.5 (see below Eq 3.27)
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        """Initialization
        """

        self.P_installed = P_installed
        self.bulbous_bow = bulbous_bow

        # Required power for systems on board, "5%" based on De Vos and van Gils (2011): Walstroom versus generator stroom
        self.P_hotel_perc = P_hotel_perc

        if P_hotel:  # if P_hotel is specified use the given value
            self.P_hotel = P_hotel
        else:  # if P_hotel is None calculate it from P_hotel_percentage and P_installed
            self.P_hotel = self.P_hotel_perc * self.P_installed

        self.P_tot_given = P_tot_given
        self.L_w = L_w
        self.year = current_year
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

        self.one_k2 = one_k2

        # plugin function that computes velocity based on power
        self.power2v = opentnsim.energy.power2v

        # TODO: C_year is obligatory, so why is this code here?
        if C_year:
            self.C_year = C_year
        else:
            self.C_year = self.calculate_engine_age()

        if self.P_tot_given is not None and self.P_installed is not None:
            if P_tot_given > P_installed:
                self.P_tot_given = self.P_installed

    def calculate_engine_age(self):
        """
        Calculate the age of the engine based on the weight class of the ship (L_w).

        The age is drawn randomly from a Weibull distribution with parameters k and lmb,
        which depend on the weight class of the ship. The year of construction is
        computed from the current year of the simulation and the age of the engine.

        Notes
        -----
        Uses `self.L_w` and `self.year` to compute the age and construction year of the
        engine. This method sets attributes `self.age` and `self.C_year`.

        """
        # compute the age of the engine
        self.age = sample_engine_age(self.L_w)

        # compute the construction year of the engine
        if self.year is None:
            raise ValueError("year must be set to calculate the construction year of the engine")
        self.C_year = self.year - self.age
        logger.debug(f"Engine age calculated as {self.age}, hence construction year is {self.C_year}")

        return self.C_year

    def calculate_properties(self):
        """Calculate a number of basic vessel properties"""

        (
            self.C_M,
            self.C_WP,
            self.C_P,
            self.delta,
            self.lcb,
            self.L_R,
            self.A_T,
            self.A_BT,
            self.S,
            self.S_APP,
            self.S_B,
            self.T_F,
            self.h_B,
        ) = calculate_properties(
            C_B=self.C_B,
            L=self.L,
            B=self.L,
            T=self.T,
            bulbous_bow=self.bulbous_bow,
            C_BB=self.C_BB,
        )

    def calculate_frictional_resistance(self, v, h_0):
        """Frictional resistance

        - 1st resistance component defined by Holtrop and Mennen (1982)
        - A modification to the original friction line is applied, based on literature of Zeng (2018), to account for shallow water effects
        """
        (
            self.R_f,
            self.C_f,
            self.R_e,
            self.Cf_deep,
            self.Cf_shallow,
            self.Cf_0,
            self.Cf_Katsui,
            self.V_B,
            self.D,
            self.a,
        ) = calculate_frictional_resistance(
            v=v,
            h_0=h_0,
            L=self.L,
            nu=self.nu,
            T=self.T,
            S=self.S,
            S_B=self.S_B,
            rho=self.rho,
        )

    def calculate_viscous_resistance(self):
        """Viscous resistance

        - 2nd resistance component defined by Holtrop and Mennen (1982)
        - Form factor (1 + k1) has to be multiplied by the frictional resistance R_f, to account for the effect of viscosity
        """

        self.c_14, self.one_k1, self.R_f_one_k1 = calculate_viscous_resistance(
            c_stern=self.c_stern,
            B=self.B,
            L=self.L,
            T=self.T,
            L_R=self.L_R,
            C_P=self.C_P,
            R_f=self.R_f,
            delta=self.delta,
        )

    def calculate_appendage_resistance(self, v):
        """Appendage resistance

        - 3rd resistance component defined by Holtrop and Mennen (1982)
        - Appendages (like a rudder, shafts, skeg) result in additional frictional resistance
        """

        self.R_APP = calculate_appendage_resistance(
            v=v,
            rho=self.rho,
            S_APP=self.S_APP,
            one_k2=self.one_k2,
            C_f=self.C_f,
        )

    def karpov(self, v, h_0):
        """Intermediate calculation: Karpov

        - The Karpov method computes a velocity correction that accounts for limited water depth (corrected velocity V2,
          expressed as "Vs + delta_V" in the paper), but it also can be used for deeper water depth (h_0 / T >= 9.5).
        - V2 has to be implemented in the wave resistance (R_W) and the residual resistance terms (R_res: R_TR, R_A, R_B)
        """

        self.F_rh, self.V_2, self.alpha_xx = karpov(
            v=v,
            h_0=h_0,
            g=self.g,
            T=self.T,
        )

    def calculate_wave_resistance(self, v, h_0):
        """Wave resistance

        - 4th resistance component defined by Holtrop and Mennen (1982)
        - When the speed or the vessel size increases, the wave making resistance increases
        - In shallow water, the wave resistance shows an asymptotical behaviour by reaching the critical speed
        """

        self.karpov(v, h_0)
        # TODO: what is the purpose of executing Karpov here if the attributes set
        # (F_rh, V_2, alpha_xx) are not used in the wave resistance calculation?

        # perform calculation of wave resistance
        (
            self.F_rL,
            self.i_E,
            self.c_1,
            self.c_2,
            self.c_5,
            self.c_7,
            self.c_15,
            self.c_16,
            self.lmbda,
            self.m_1,
            self.m_2,
            self.R_W,
        ) = calculate_wave_resistance(
            v=v,
            h_0=h_0,
            g=self.g,
            T=self.T,
            L=self.L,
            B=self.B,
            C_P=self.C_P,
            C_WP=self.C_WP,
            lcb=self.lcb,
            L_R=self.L_R,
            A_T=self.A_T,
            C_M=self.C_M,
            delta=self.delta,
            rho=self.rho,
        )

    def calculate_residual_resistance(self, v, h_0):
        """Residual resistance terms

        - Holtrop and Mennen (1982) defined three residual resistance terms:
        - 1) Resistance due to immersed transom (R_TR), Karpov corrected velocity V2 is used
        - 2) Resistance due to model-ship correlation (R_A), Karpov corrected velocity V2 is used
        - 3) Resistance due to the bulbous bow (R_B), Karpov corrected velocity V2 is used
        """

        self.karpov(v, h_0)
        self.V_2 = v  # TODO:  why overrule the just computed V_2 from Karpov?

        # compute the residual resistance terms
        (
            self.F_nT,
            self.c_6,
            self.R_TR,
            self.c_4,
            self.c_2,
            self.C_A,
            self.R_A,
            self.F_ni,
            self.P_B,
            self.R_B,
            self.R_res,
        ) = calculate_residual_resistance(
            V_2=self.V_2,
            g=self.g,
            A_T=self.A_T,
            B=self.B,
            C_WP=self.C_WP,
            rho=self.rho,
            T=self.T,
            L=self.L,
            C_B=self.C_B,
            S=self.S,
            T_F=self.T_F,
            h_B=self.h_B,
            A_BT=self.A_BT,
            bulbous_bow=self.bulbous_bow,
        )

    def calculate_total_resistance(self, v, h_0):
        """Total resistance:

        The total resistance is the sum of all resistance components (Holtrop and Mennen, 1982)
        """

        self.calculate_properties()
        self.calculate_frictional_resistance(v, h_0)
        self.calculate_viscous_resistance()
        self.calculate_appendage_resistance(v)
        self.calculate_wave_resistance(v, h_0)
        self.calculate_residual_resistance(v, h_0)

        # The total resistance R_tot [kN] = R_f * (1+k1) + R_APP + R_W + R_TR + R_A
        self.R_tot = self.R_f * self.one_k1 + self.R_APP + self.R_W + self.R_TR + self.R_A + self.R_B

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
        In this version, we define the propulsion power as P_d (Delivered Horse Power) rather than P_b (Brake Horse
        Power). The reason we choose P_d as propulsion power is to prevent double use of the same power efficiencies.
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
        (
            self.P_e,
            self.dw,
            self.w,
            self.t,
            self.eta_h,
            self.P_d,
            self.P_b,
            self.P_propulsion,
            self.P_tot,
            self.P_given,
            self.P_partial,
        ) = calculate_total_power_required(
            v=v,
            h_0=h_0,
            R_tot=self.R_tot,
            F_rL=self.F_rL,
            x=self.x,
            C_B=self.C_B,
            delta=self.delta,
            D_s=self.D_s,
            eta_o=self.eta_o,
            eta_r=self.eta_r,
            eta_t=self.eta_t,
            eta_g=self.eta_g,
            P_hotel=self.P_hotel,
            P_installed=self.P_installed,
        )

        # return these three variables:
        # 1) self.P_propulsion, for the convience of validation.  (propulsion power and fuel used for propulsion),
        # 2) self.P_tot, know the required power, especially when it exceeds installed engine power while sailing shallower and faster
        # 3) self.P_given, the actual power the engine gives for "propulsion + hotel" within its capacity (means installed power). This varible is used for calculating delta_energy of each sailing time step.
        # TODO: return description does not match the docstring and comments

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

    def correction_factors(self, v, h_0):
        """Partial engine load correction factors (C_partial_load):

        - The correction factors have to be multiplied by the general emission factors (or general SFC), to get the total emission factors (or final SFC)
        - The correction factor takes into account the effect of the partial engine load
        - When the partial engine load is low, the correction factors for ICE engine are higher (ICE engine is less efficient at lower enegine load)
        - the correction factors for emissions and diesel fuel in ICE engine are based on literature TNO (2019)
        - For fuel cell enegines(PEMFC & SOFC), the correction factors are lower when the partial engine load is low (fuel cell enegine is more efficient at lower enegine load)
        - the correction factors for renewable fuels used in fuel cell engine are based on literature Kim et al (2020) (A Preliminary Study on an Alternative Ship Propulsion System Fueled by Ammonia: Environmental and Economic Assessments, https://doi.org/10.3390/jmse8030183)
        """
        # TODO: create correction factors for renewable powered ship, the factor may be 100%
        self.calculate_total_power_required(v=v, h_0=h_0)  # You need the P_partial values

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

    def calculate_emission_factors_total(self, v, h_0):
        """Total emission factors:

        - The total emission factors can be computed by multiplying the general emission factor by the correction factor
        """

        self.emission_factors_general()  # You need the values of the general emission factors of CO2, PM10, NOX
        self.correction_factors(v=v, h_0=h_0)  # You need the correction factors of CO2, PM10, NOX

        # The total emission factor is calculated by multiplying the general emission factor (EF_CO2 / EF_PM10 / EF_NOX)
        # By the correction factor (C_partial_load_CO2 / C_partial_load_PM10 / C_partial_load_NOX)

        self.total_factor_CO2 = self.EF_CO2 * self.C_partial_load_CO2
        self.total_factor_PM10 = self.EF_PM10 * self.C_partial_load_PM10
        self.total_factor_NOX = self.EF_NOX * self.C_partial_load_NOX

        logger.debug(f"The total emission factor of CO2 is {self.total_factor_CO2} g/kWh")
        logger.debug(f"The total emission factor of PM10 is {self.total_factor_PM10} g/kWh")
        logger.debug(f"The total emission factor CO2 is {self.total_factor_NOX} g/kWh")

    def calculate_SFC_final(self, v, h_0):
        """The final SFC is computed by multiplying the general SFC by the partial engine load correction factor.

        The calculation of final SFC below includes
        - the final SFC of LH2, eLNG, eMethanol, eNH3 in mass and volume while using Fuel Cell Engine (PEMFC, SOFC)
        - the final SFC of eLNG, eMethanol, eNH3 in mass and volume while using Internal Combustion Engine
        - the final SFC of diesel in mass and volume while using Internal Combustion Engine
        - the final SFC of battery in mass and volume while use battery-electric power system
        """

        self.SFC_general()  # You need the values of the general SFC
        self.correction_factors(v=v, h_0=h_0)  # You need the correction factors of SFC

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
            max_sinkage = calculate_max_sinkage(
                v=v,
                h_0=h_0,
                T=self._T,  # TODO: why _T and not T? moreover: T stems from VesselProperties
                B=self.B,
                C_B=self.C_B,
                width=width,
            )

        return max_sinkage

    def calculate_h_squat(self, v, h_0, width=150):
        """Calculate the water depth in case h_squat is set to True

        The amount of water under the keel is calculated h_0 - T. When h_squat is set to True, we estimate a max_sinkage
        that is subtracted from h_0. This values is returned as h_squat for further calculation.

        """
        h_squat = h_0 - self.calculate_max_sinkage(v, h_0, width=width)

        return h_squat


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
            "water depth": [],
            "distance": [],
            "delta_t": [],
        }

        self.co2_footprint = {"total_footprint": 0, "stationary": 0}
        self.mki_footprint = {"total_footprint": 0, "stationary": 0}

    def calculate_energy_consumption(self):
        """Calculation of energy consumption based on total time in system and properties"""

        # log messages that are related to locking
        # todo: check if this still works with Floors new locking module
        stationary_phase_indicator = [
            "Waiting to enter waiting area stop",  # checked: not sure if still used in locking module
            "Waiting in waiting area stop",  # checked: not sure if still used in locking module
            "Waiting in line-up area stop",  # checked: still used in locking module
            "Passing lock stop",  # checked: still used in locking module
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
                distance = calculate_distance(geometries[i], geometries[i + 1])
                v = distance / delta_t
                self.energy_use["distance"].append(distance)

                # calculate the delta t
                self.energy_use["delta_t"].append(delta_t)

                # calculate the water depth
                h_0 = calculate_depth(geometries[i], geometries[i + 1], self.FG)

                # printstatements to check the output (can be removed later)
                logger.debug("delta_t: {:.4f} s".format(delta_t))
                logger.debug("distance: {:.4f} m".format(distance))
                logger.debug("velocity: {:.4f} m/s".format(v))

                # we use the calculated velocity to determine the resistance and power required
                # we can switch between the 'original water depth' and 'water depth considering ship squatting' for energy calculation, by using the function "calculate_h_squat (h_squat is set as Yes/No)" in the core.py
                h_0 = self.vessel.calculate_h_squat(v, h_0)
                # print(h_0)
                self.vessel.calculate_total_resistance(v, h_0)
                self.vessel.calculate_total_power_required(v=v, h_0=h_0)

                self.vessel.calculate_emission_factors_total(v=v, h_0=h_0)
                self.vessel.calculate_SFC_final(v=v, h_0=h_0)

                if messages[i + 1] in stationary_phase_indicator:  # if we are in a stationary stage only log P_hotel
                    # Energy consumed per time step delta_t in the stationary stage
                    energy_delta = self.vessel.P_hotel * delta_t / 3600  # kJ/3600 = kWh

                    # Emissions CO2, PM10 and NOX, in gram - emitted in the stationary stage per time step delta_t,
                    # consuming 'energy_delta' kWh
                    # TODO: check, as it seems that stationary energy use is now not stored.
                    P_hotel_delta = self.vessel.P_hotel  # in kW
                    P_installed_delta = self.vessel.P_installed  # in kW

                else:  # otherwise log P_tot
                    # Energy consumed per time step delta_t in the propulsion stage
                    # TODO: energy_delta should be P_tot times delta_t (was P_given, but then when the vessel is driven with v a strange cutoff occurs, when it is driven by P_tot_given it should be limited by the available power ... that now works)
                    energy_delta = (
                        self.vessel.P_tot * delta_t / 3600
                    )  # kJ/3600 = kWh, when P_tot >= P_installed, P_given = P_installed; when P_tot < P_installed, P_given = P_tot

                    # Emissions CO2, PM10 and NOX, in gram - emitted in the propulsion stage per time step delta_t,
                    # consuming 'energy_delta' kWh
                    P_tot_delta = self.vessel.P_tot  # in kW, required power, may exceed installed engine power
                    P_given_delta = self.vessel.P_given  # in kW, actual given power
                    P_installed_delta = self.vessel.P_installed  # in kW
                    emission_delta_CO2 = (
                        self.vessel.total_factor_CO2 * energy_delta
                    )  # Energy consumed per time step delta_t in the                                                                                              #stationary phase # in g
                    emission_delta_PM10 = self.vessel.total_factor_PM10 * energy_delta  # in g
                    emission_delta_NOX = self.vessel.total_factor_NOX * energy_delta  # in g
                    # Todo: we need to rename the factor name for fuels, not starting with "emission" , consider seperating it from emission factors
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
                    self.energy_use["stationary"].append(energy_delta)
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
                    # self.energy_use["water depth info from vaarweginformatie.nl"].append(depth)

        # TODO: er moet hier een heel aantal dingen beter worden ingevuld
        # - de kruissnelheid is nu nog per default 1 m/s (zie de Movable mixin). Eigenlijk moet in de
        #   vessel database ook nog een speed_loaded en een speed_unloaded worden toegevoegd.
        # - er zou nog eens goed gekeken moeten worden wat er gedaan kan worden rond kustwerken
        # - en er is nog iets mis met de snelheid rond een sluis

        # - add HasCurrent Class or def
