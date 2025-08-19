"""Module with calculation functions for the energy module
Most functions are called by the ConsumesEnergy mixin.

"""

# %% IMPORT DEPENDENCIES
# generic
import logging
import numpy as np

# logging
logger = logging.getLogger(__name__)


def sample_engine_age(L_w):
    """
    Samples the age of the engine based on the weight class of the vessel. The age is
    drawn randomly from a Weibull distribution with parameters k and lmb, which depend
    on the weight class of the ship. The age is then returned in years.

    Ligterink, N.E., R.N. van Gijlswijk, G. Kadijk, R.J. Vermeulen, A.P. Indrajuana, M. Elstgeest,
    P. van Mensch, J.M. de Ruiter, R.P. Verbeek, J.H.J. Hulskotte, G. Geilenkirchen (PBL) and
    M. Traa (PBL) (2019). "Emissiefactoren wegverkeer - Actualisatie 2019", TNO 2019 R10825v2.
    https://www.pbl.nl/publicaties/emissiefactoren-wegverkeer-actualisatie-2019

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
    # Table 8, in Ligterink et al (2019)
    if L_w == 1:  # Weight class L1
        k = 1.3
        lmb = 20.4
    elif L_w == 2:  # Weight class L2
        k = 1.12
        lmb = 18.5
    elif L_w == 3:  # Weight class L3
        k = 1.26
        lmb = 18.6

    # the engine age
    # returns a randomly sampled estimate of the engine age (in years) for a ship, based on its weight class
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
        Beam of the vessel [m]
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
        raise ValueError("Beam B should be > 0")
    if width <= 0:
        raise ValueError("Width of the fairway should be > 0")
    if C_B <= 0:
        raise ValueError("Block coefficient C_B should be > 0")

    if B > width:
        raise ValueError(f"Width of the fairway ({width}) should be larger than " f"the beam of the vessel ({B})")

    # calculate the maximum sinkage
    return (C_B * ((B * T) / (width * h_0)) ** 0.81) * ((v * 1.94) ** 2.08) / 20


def calculate_properties(C_B, L, B, T, bulbous_bow, C_BB):
    """
    Calculate the properties of a vessel based on its block coefficient, length,
    beam, draught, bulbous bow coefficient, and whether it has a bulbous bow.

    Parameters
    ----------
    C_B : float
        Block coefficient of the vessel [-]
    L : float
        Length of the vessel [m]
    B : float
        Beam of the vessel [m]
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

    # TODO: D_s is a property that should be given, not calculated
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
    # Reynolds number
    R_e = v * L / nu

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
        Beam of the vessel [m]
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
    #  Segers (2021) we assume c_stern = 0 which leads to c_14 to be 1)
    c_14 = 1 + 0.0011 * c_stern

    # the form factor (1+k1) describes the viscous resistance
    # Van Koningsveld et al (2023) - Eq 5.12
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

    # The Froude number used in the Karpov method is the depth related Froude number F_rh

    # The different alpha** curves are determined with a sixth power polynomial approximation in Excel (Segers, 2019)
    # A distinction is made between different ranges of Froude numbers, because this resulted in a better approximation of the curve

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
    elif F_rh > 0.4:
        if 0 <= h_0 / T < 1.75:
            alpha_xx = (
                -0.9274 * F_rh**6 + 9.5953 * F_rh**5 - 37.197 * F_rh**4 + 69.666 * F_rh**3 - 65.391 * F_rh**2 + 28.025 * F_rh - 3.4143
            )
        if 1.75 <= h_0 / T < 2.25:
            alpha_xx = (
                2.2152 * F_rh**6 - 11.852 * F_rh**5 + 21.499 * F_rh**4 - 12.174 * F_rh**3 - 4.7873 * F_rh**2 + 5.8662 * F_rh - 0.2652
            )
        if 2.25 <= h_0 / T < 2.75:
            alpha_xx = (
                1.2205 * F_rh**6 - 5.4999 * F_rh**5 + 5.7966 * F_rh**4 + 6.6491 * F_rh**3 - 16.123 * F_rh**2 + 9.2016 * F_rh - 0.6342
            )
        if 2.75 <= h_0 / T < 3.25:
            alpha_xx = (
                -0.4085 * F_rh**6 + 4.534 * F_rh**5 - 18.443 * F_rh**4 + 35.744 * F_rh**3 - 34.381 * F_rh**2 + 15.042 * F_rh - 1.3807
            )
        if 3.25 <= h_0 / T < 3.75:
            alpha_xx = (
                0.4078 * F_rh**6 - 0.919 * F_rh**5 - 3.8292 * F_rh**4 + 15.738 * F_rh**3 - 19.766 * F_rh**2 + 9.7466 * F_rh - 0.6409
            )
        if 3.75 <= h_0 / T < 4.5:
            alpha_xx = (
                0.3067 * F_rh**6 - 0.3404 * F_rh**5 - 5.0511 * F_rh**4 + 16.892 * F_rh**3 - 20.265 * F_rh**2 + 9.9002 * F_rh - 0.6712
            )
        if 4.5 <= h_0 / T < 5.5:
            alpha_xx = (
                0.3212 * F_rh**6 - 0.3559 * F_rh**5 - 5.1056 * F_rh**4 + 16.926 * F_rh**3 - 20.253 * F_rh**2 + 10.013 * F_rh - 0.7196
            )
        if 5.5 <= h_0 / T < 6.5:
            alpha_xx = (
                0.9252 * F_rh**6 - 4.2574 * F_rh**5 + 5.0363 * F_rh**4 + 3.3282 * F_rh**3 - 10.367 * F_rh**2 + 6.3993 * F_rh - 0.2074
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
            if F_rh <= 0.6:
                alpha_xx = 1
            if F_rh > 0.6:
                alpha_xx = (
                    -6.4069 * F_rh**6 + 47.308 * F_rh**5 - 141.93 * F_rh**4 + 220.23 * F_rh**3 - 185.05 * F_rh**2 + 79.25 * F_rh - 12.484
                )
        if h_0 / T >= 9.5:
            if F_rh <= 0.6:
                alpha_xx = 1
            if F_rh > 0.6:
                alpha_xx = (
                    -6.0737 * F_rh**6 + 44.97 * F_rh**5 - 135.21 * F_rh**4 + 210.13 * F_rh**3 - 176.72 * F_rh**2 + 75.728 * F_rh - 11.893
                )

    V_2 = v / alpha_xx

    return F_rh, V_2, alpha_xx


def calculate_wave_resistance(V_2, h_0, g, T, L, B, C_P, C_WP, lcb, L_R, A_T, C_M, delta, rho):
    """
    Calculate the wave resistance and related hydrodynamic coefficients for a ship.

    Parameters
    ----------
    V_2v : float
        Karpov corrected ship speed relative to water (m/s).
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

    F_rL = V_2 / np.sqrt(g * L)  # Froude number based on ship's speed to water and its length of waterline

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
    # TODO: check if we need to adapt this, for cases where we do want to calculate with bulbous bows
    c_2 = 1  # accounts for the effect of the bulbous bow, which is not present at inland ships
    c_5 = 1 - (0.8 * A_T) / (B * T * C_M)  # influence of the transom stern on the wave resistance

    # parameter c_15 depends on the ratio L^3 / delta
    # Van Koningsveld et al (2023) - Part IV Table 5.1
    if (L**3) / delta < 512:
        c_15 = -1.69385
    elif (L**3) / delta > 1727:
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
    # Segers (2019) distinguishes multiple Froude classes (Section 3.2.5 and Appendix C - C.2).
    # for all reasonable combinations of ship lengths and speeds, inland ships always fall in the
    # F_n,V_2 < 0.4 class
    R_W = c_1 * c_2 * c_5 * delta * rho * g * np.exp(m_1 * (F_rL ** (-0.9)) + m_2 * np.cos(lmbda * (F_rL ** (-2)))) / 1000  # kN

    return F_rL, i_E, c_1, c_2, c_5, c_7, c_15, c_16, lmbda, m_1, m_2, R_W


def calculate_residual_resistance(V_2, g, A_T, B, C_WP, rho, T, L, C_B, S, T_F, h_B, A_BT, bulbous_bow):
    """
    Calculate the residual resistance of a ship, which includes the resistance due to
    the immersed transom, model-ship correlation resistance, and bulbous bow resistance.

    This function computes the residual resistance components based on the ship's
    speed, transom area, beam, waterplane coefficient, density, draft, length,
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
        Beam of the ship (m).
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
        Beam of the ship (m).
    T : float
        Actual draft of the ship (m).
    bulbous_bow : bool
        Whether the ship has a bulbous bow (True) or not (False).
    C_BB : float
        Beam coefficient of bulbous bow (dimensionless).
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

    # TODO: this function is rather odd as it calls all other resistance functions,
    #  computing lots of unused values (which are set in the corresponding method),
    #  hence the function is not yet used in the main class method.

    # vessel properties
    C_M, C_WP, C_P, delta, lcb, L_R, A_T, A_BT, S, S_APP, S_B, T_F, h_B = calculate_properties(C_B, L, B, T, bulbous_bow, C_BB)

    # frictional resistance
    R_f, C_f, _, _, _, _, _, _, _, _ = calculate_frictional_resistance(v, h_0, L, nu, T, S, S_B, rho)

    # viscous resistance
    _, one_k1, _ = calculate_viscous_resistance(c_stern, B, L, T, L_R, C_P, R_f, delta)

    # appendage resistance
    R_APP = calculate_appendage_resistance(v, rho, S_APP, one_k2, C_f)

    # calculate the Karpov corrected velocity
    F_rh, V_2, alpha_xx = karpov(v, h_0, g, T)
    print('Original v = {} m/s, Karpov corrected V_2 = {} m/s'.format(v, V_2))

    # wave resistance
    _, _, _, _, _, _, _, _, _, _, _, R_W = calculate_wave_resistance(V_2, h_0, g, T, L, B, C_P, C_WP, lcb, L_R, A_T, C_M, delta, rho)

    # residual resistance
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
        Number of propellers (dimensionless).
    C_B : float
        Block coefficient of the ship (dimensionless).
    delta : float
        Displacement volume of the ship (m^3).
    D_s : float
        Propeller diameter (m).
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
        Power delivered (kW)
    P_b : float
        Power at the brake (kW)
    P_propulsion : float
        Power required for propulsion (kW).
    P_tot : float
        Total power required for the ship (propulsion + hotel power) (kW).
    P_given : float
        Power given by the captain (kW).
    P_partial : float
        Partial engine load (P_tot / P_installed) (dimensionless).

    Notes
    -----
    The P_b calculation involves gearing efficiency and transmission efficiency
    already, while P_d does not. P_d is the power delivered to propellers. To estimate
    the renewable fuel use, 'energy conversion efficiencies' are taken into consideration.

    The 'energy conversion efficiencies' for renewable fuel powered vessels are
    commonly measured/given as a whole covering the engine power systems, including
    different engines (such as fuel cell engine, battery engine, internal
    combustion engine, hybrid engine) efficiencies, and corresponding gearbox
    efficiencies, AC/DC converter efficiencies, excluding the efficiency items of
    propellers.

    It is therefore, important carefully align with the later use of 'energy conversion
    efficiencies' for fuel use estimation and to prevent double use of some power efficiencies,
    such as gearing efficiency. It is important to carefully consider if it is best to use
    P_d or P_b as power starting point.

    """

    # Effective Horse Power (EHP), 'P_e' - power associated with the vessel's speed and its resistance
    # (Van Koningsveld et al (2023) - Part IV Eq 5.17)
    P_e = v * R_tot

    # velocity correction coefficient, 'dw'
    # Segers (2021) (http://resolver.tudelft.nl/uuid:a260bc48-c6ce-4f7c-b14a-e681d2e528e3)
    # Appendix C
    if F_rL < 0.2:
        dw = 0  # the velocity correction coefficient is 0 when FrL is smaller than 0.2
    else:
        dw = 0.1  # otherwise the velocity correction coefficient is 0.1

    # wake fraction, 'w'
    # Segers (2021) (http://resolver.tudelft.nl/uuid:a260bc48-c6ce-4f7c-b14a-e681d2e528e3)
    # Appendix C - Eq C.1
    w = 0.11 * (0.16 / x) * C_B * np.sqrt((delta ** (1 / 3)) / D_s) - dw  # wake fraction 'w'

    assert not isinstance(w, complex), f"w should not be complex: {w}"

    # thrust deduction factor, 't'
    if x == 1:
        # if the ship has 1 propeller
        # (Van Koningsveld et al (2023) - Part IV Eq 5.22)
        t = 0.6 * w * (1 + 0.67 * w)
    else:
        # (Van Koningsveld et al (2023) - Part IV Eq 5.23)
        t = 0.8 * w * (1 + 0.25 * w)

    # hull efficiency 'eta_h'
    eta_h = (1 - t) / (1 - w)  # hull efficiency eta_h

    # TODO: check below suggestions.
    #   They were made to allow for better translation to alternative energy sources.
    #   But the changes induced unexpected behaviour.

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

    # Delivered Horse Power (DHP), 'P_d'
    # (Van Koningsveld et al (2023) - Part IV Eq 5.19)
    P_d = P_e / (eta_o * eta_r * eta_h)

    # Brake Horse Power (BHP), 'P_b'
    # (Van Koningsveld et al (2023) - Part IV Eq 5.24)
    P_b = P_d / (eta_t * eta_g)

    # TODO: consider how to integrate the suggestion to use P_propulsion = P_d
    #  When working with alternative energy carriers it may be better to use
    #  Delivered horse power, the power delivered to propellers, rather than
    #  Power at the Brake. This may avoid double counting of efficiencies.

    # Propulsion Power, 'P_propulsion'
    P_propulsion = P_b  # propulsion power is defined here as Power at the Brake.

    # Total Power, 'P_tot'
    P_tot = P_hotel + P_propulsion

    # Partial engine load, 'P_partial': used in the 'Emission calculations'
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
