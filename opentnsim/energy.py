import datetime, time
import pathlib
import logging
import uuid
import functools
import itertools
import json
import pyproj
import shapely.geometry
import numpy as np
import pandas as pd
import scipy.optimize
import simpy
import tqdm

# package(s) for data handling

# OpenTNSim
import opentnsim
import opentnsim.strategy
import opentnsim.graph_module

# Used for mathematical functions
import math

# Used for making the graph to visualize our problem
import networkx as nx

logger = logging.getLogger(__name__)


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


def get_upperbound_for_power2v(vessel, width, depth, bounds=(0, 20)):
    """for a waterway section with a given width and depth, compute a maximum installed-
    power-allowed velocity, considering squat. This velocity is set as upperbound in the
    power2v function in energy.py "upperbound" is the maximum value in velocity searching
    range.
    """

    def get_grounding_v(vessel, width, depth, bounds):
        def seek_v_given_z(v, vessel, width, depth):
            # calculate sinkage
            z_computed = (vessel.C_B * ((vessel.B * vessel._T) / (width * depth)) ** 0.81) * ((v * 1.94) ** 2.08) / 20

            # calculate available underkeel clearance (vessel in rest)
            z_given = depth - vessel._T

            # compute difference between the sinkage and the space available for sinkage
            diff = z_given - z_computed

            return diff**2

        # goalseek to minimize
        fun = functools.partial(seek_v_given_z, vessel=vessel, width=width, depth=depth)
        fit = scipy.optimize.minimize_scalar(fun, bounds=bounds, method="bounded")

        # check if we found a minimum
        if not fit.success:
            raise ValueError(fit)

        # the value of fit.x within the bound (0,20) is the velocity we find where the diff**2 reach a minimum (zero).
        grounding_v = fit.x

        print("grounding velocity {:.2f} m/s".format(grounding_v))

        return grounding_v

    # create a large velocity[m/s] range for both inland shipping and seagoing shipping
    grounding_v = get_grounding_v(vessel, width, depth, bounds)
    velocity = np.linspace(0.01, grounding_v, 1000)
    task = list(itertools.product(velocity[0:-1]))

    # prepare a list of dictionaries for pandas
    rows = []
    for item in task:
        row = {"velocity": item[0]}
        rows.append(row)

    # convert simulations to dataframe, so that we can apply a function and monitor progress
    task_df = pd.DataFrame(rows)

    # creat a results empty list to collect the below results
    results = []
    for i, row in tqdm.tqdm(task_df.iterrows(), disable=True):
        h_0 = depth
        velocity = row["velocity"]

        # calculate squat and the waterdepth after squat
        z_computed = (vessel.C_B * ((vessel.B * vessel._T) / (150 * h_0)) ** 0.81) * ((velocity * 1.94) ** 2.08) / 20
        h_0 = depth - z_computed

        # for the squatted water depth calculate resistance and power
        # vessel.calculate_properties()
        # vessel.calculate_frictional_resistance(v=velocity, h_0=h_0)
        vessel.calculate_total_resistance(v=velocity, h_0=h_0)
        P_tot = vessel.calculate_total_power_required(v=velocity)

        # prepare a row
        result = {}
        result.update(row)
        result["Powerallowed_v"] = velocity
        result["P_tot"] = P_tot
        result["P_installed"] = vessel.P_installed

        # update resulst dict
        results.append(result)

    results_df = pd.DataFrame(results)

    selected = results_df.query("P_tot < P_installed")
    upperbound = max(selected["Powerallowed_v"])
    print("upperbound velocity {:.2f} m/s".format(upperbound))
    return upperbound


def power2v(vessel, edge, upperbound):
    """Compute vessel velocity given an edge and power (P_tot_given)

    bounds is the limits where to look for a solution for the velocity [m/s]
    returns velocity [m/s]
    """

    assert isinstance(vessel, opentnsim.core.VesselProperties), "vessel should be an instance of VesselProperties"

    assert vessel.C_B is not None, "C_B cannot be None"

    # upperbound = get_upperbound_for_power2v()
    # bounds > 10 gave an issue...
    # TODO: check what the origin of this is.
    def seek_v_given_power(v, vessel, edge):
        """function to optimize"""
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

        # compute difference between power setting by captain and power needed for velocity
        diff = vessel.P_tot_given - vessel.P_tot
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
    """

    def __init__(
        self,
        P_installed,
        L_w,
        C_year,
        current_year=None,  # current_year
        bulbous_bow=False,
        P_hotel_perc=0.05,
        P_hotel=None,
        P_tot_given=None,  # the actual power engine setting
        nu=1 * 10 ** (-6),
        rho=1000,
        g=9.81,
        x=2,
        eta_o=0.4,
        eta_r=1.00,
        eta_t=0.98,
        eta_g=0.96,
        c_stern=0,
        C_BB=0.2,
        C_B=0.85,
        one_k2=2.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        """Initialization
        """

        self.P_installed = P_installed
        self.bulbous_bow = bulbous_bow
        self.P_hotel_perc = P_hotel_perc
        if P_hotel:  # if P_hotel is specified as None calculate it from P_hotel_percentage
            self.P_hotel = P_hotel
        else:  # otherwise use the given value
            self.P_hotel = self.P_hotel_perc * self.P_installed
        self.P_tot_given = P_tot_given
        self.L_w = L_w
        self.year = current_year
        self.nu = nu
        self.rho = rho
        self.g = g
        self.x = x
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

        # # TODO: check assumption when combining move with energy
        # if self.P_tot_given is not None and self.v is not None:
        #     raise ValueError("please specify v or P_tot_given, but not both")

    # The engine age and construction year of the engine is computed with the function below.
    # The construction year of the engine is used in the emission functions (1) emission_factors_general and (2) correction_factors

    def calculate_engine_age(self):
        """Calculating the construction year of the engine, dependend on a Weibull function with
        shape factor 'k', and scale factor 'lmb', which are determined by the weight class L_w
        """

        # Determining which shape and scale factor to use, based on the weight class L_w = L1, L2 or L3
        assert self.L_w in [1, 2, 3], "Invalid value L_w, should be 1,2 or 3"
        if self.L_w == 1:  # Weight class L1
            self.k = 1.3
            self.lmb = 20.5
        elif self.L_w == 2:  # Weight class L2
            self.k = 1.12
            self.lmb = 18.5
        elif self.L_w == 3:  # Weight class L3
            self.k = 1.26
            self.lmb = 18.6

        # The age of the engine
        # TODO: I would not expect a random distribution if the function is cal
        self.age = int(np.random.weibull(self.k) * self.lmb)

        # Construction year of the engine
        self.C_year = self.year - self.age

        logger.debug(f"The construction year of the engine is {self.C_year}")
        return self.C_year

    def calculate_properties(self):
        """Calculate a number of basic vessel properties"""

        # TO DO: add properties for seagoing ships with bulbs

        self.C_M = 1.006 - 0.0056 * self.C_B ** (-3.56)  # Midship section coefficient
        self.C_WP = (1 + 2 * self.C_B) / 3  # Waterplane coefficient
        self.C_P = self.C_B / self.C_M  # Prismatic coefficient

        self.delta = self.C_B * self.L * self.B * self.T  # Water displacement

        self.lcb = -13.5 + 19.4 * self.C_P  # longitudinal center of buoyancy
        self.L_R = self.L * (
            1 - self.C_P + (0.06 * self.C_P * self.lcb) / (4 * self.C_P - 1)
        )  # length parameter reflecting the length of the run

        self.A_T = 0.2 * self.B * self.T  # transverse area of the transom
        # calculation for A_BT (cross-sectional area of the bulb at still water level [m^2]) depends on whether a ship has a bulb
        if self.bulbous_bow:
            self.A_BT = self.C_BB * self.B * self.T * self.C_M  # calculate A_BT for seagoing ships having a bulb
        else:
            self.A_BT = 0  # most inland ships do not have a bulb. So we assume A_BT=0.

        # Total wet area: S
        assert self.C_M >= 0, f"C_M should be positive: {self.C_M}"
        self.S = self.L * (2 * self.T + self.B) * np.sqrt(self.C_M) * (
            0.453 + 0.4425 * self.C_B - 0.2862 * self.C_M - 0.003467 * (self.B / self.T) + 0.3696 * self.C_WP
        ) + 2.38 * (self.A_BT / self.C_B)

        self.S_APP = 0.05 * self.S  # Wet area of appendages
        self.S_B = self.L * self.B  # Area of flat bottom

        self.D_s = 0.7 * self.T  # Diameter of the screw
        self.T_F = self.T  # Forward draught of the vessel [m]
        self.h_B = 0.2 * self.T  # Position of the centre of the transverse area [m]

    def calculate_frictional_resistance(self, v, h_0):
        """Frictional resistance

        - 1st resistance component defined by Holtrop and Mennen (1982)
        - A modification to the original friction line is applied, based on literature of Zeng (2018), to account for shallow water effects
        """

        self.R_e = v * self.L / self.nu  # Reynolds number

        self.D = h_0 - self.T  # distance from bottom ship to the bottom of the fairway
        assert self.D > 0, f"D should be > 0: {self.D}"

        # Friction coefficient based on CFD computations of Zeng et al. (2018), in deep water
        self.Cf_deep = 0.08169 / ((np.log10(self.R_e) - 1.717) ** 2)
        assert not isinstance(self.Cf_deep, complex), f"Cf_deep should not be complex: {self.Cf_deep}"

        # Friction coefficient based on CFD computations of Zeng et al. (2018), taking into account shallow water effects
        self.Cf_shallow = (0.08169 / ((np.log10(self.R_e) - 1.717) ** 2)) * (
            1 + (0.003998 / (np.log10(self.R_e) - 4.393)) * (self.D / self.L) ** (-1.083)
        )
        assert not isinstance(self.Cf_shallow, complex), f"Cf_shallow should not be complex: {self.Cf_shallow}"

        # Friction coefficient in deep water according to ITTC-1957 curve
        self.Cf_0 = 0.075 / ((np.log10(self.R_e) - 2) ** 2)

        # 'a' is the coefficient needed to calculate the Katsui friction coefficient
        self.a = 0.042612 * np.log10(self.R_e) + 0.56725
        self.Cf_Katsui = 0.0066577 / ((np.log10(self.R_e) - 4.3762) ** self.a)

        # The average velocity underneath the ship, taking into account the shallow water effect
        # This calculation is to get V_B, which will be used in the following Cf for shallow water equation:
        if h_0 / self.T <= 4:
            self.V_B = 0.4277 * v * np.exp((h_0 / self.T) ** (-0.07625))
        else:
            self.V_B = v

        # cf_shallow and cf_deep cannot be applied directly, since a vessel also has non-horizontal wet surfaces that have to be taken
        # into account. Therefore, the following formula for the final friction coefficient 'C_f' for deep water or shallow water is
        # defined according to Zeng et al. (2018)

        # calculate Friction coefficient C_f for deep water:

        if (h_0 - self.T) / self.L > 1:
            self.C_f = self.Cf_0 + (self.Cf_deep - self.Cf_Katsui) * (self.S_B / self.S)
            logger.debug("now i am in the deep loop")
        else:
            # calculate Friction coefficient C_f for shallow water:
            self.C_f = self.Cf_0 + (self.Cf_shallow - self.Cf_Katsui) * (self.S_B / self.S) * (self.V_B / v) ** 2
            logger.debug("now i am in the shallow loop")
        assert not isinstance(self.C_f, complex), f"C_f should not be complex: {self.C_f}"

        # The total frictional resistance R_f [kN]:
        self.R_f = (self.C_f * 0.5 * self.rho * (v**2) * self.S) / 1000
        assert not isinstance(self.R_f, complex), f"R_f should not be complex: {self.R_f}"

        return self.R_f

    def calculate_viscous_resistance(self):
        """Viscous resistance

        - 2nd resistance component defined by Holtrop and Mennen (1982)
        - Form factor (1 + k1) has to be multiplied by the frictional resistance R_f, to account for the effect of viscosity"""

        # c_14 accounts for the specific shape of the afterbody
        self.c_14 = 1 + 0.0011 * self.c_stern

        # the form factor (1+k1) describes the viscous resistance
        self.one_k1 = 0.93 + 0.487 * self.c_14 * ((self.B / self.L) ** 1.068) * ((self.T / self.L) ** 0.461) * (
            (self.L / self.L_R) ** 0.122
        ) * (((self.L**3) / self.delta) ** 0.365) * ((1 - self.C_P) ** (-0.604))
        self.R_f_one_k1 = self.R_f * self.one_k1
        return self.R_f_one_k1

    def calculate_appendage_resistance(self, v):
        """Appendage resistance

        - 3rd resistance component defined by Holtrop and Mennen (1982)
        - Appendages (like a rudder, shafts, skeg) result in additional frictional resistance"""

        # Frictional resistance resulting from wetted area of appendages: R_APP [kN]
        self.R_APP = (0.5 * self.rho * (v**2) * self.S_APP * self.one_k2 * self.C_f) / 1000

        return self.R_APP

    def karpov(self, v, h_0):
        """Intermediate calculation: Karpov

        - The Karpov method computes a velocity correction that accounts for limited water depth (corrected velocity V2,
          expressed as "Vs + delta_V" in the paper), but it also can be used for deeper water depth (h_0 / T >= 9.5).
        - V2 has to be implemented in the wave resistance (R_W) and the residual resistance terms (R_res: R_TR, R_A, R_B)
        """

        # The Froude number used in the Karpov method is the depth related froude number F_rh

        # The different alpha** curves are determined with a sixth power polynomial approximation in Excel
        # A distinction is made between different ranges of Froude numbers, because this resulted in a better approximation of the curve
        assert self.g >= 0, f"g should be positive: {self.g}"
        assert h_0 >= 0, f"g should be positive: {h_0}"
        self.F_rh = v / np.sqrt(self.g * h_0)

        if self.F_rh <= 0.4:
            if 0 <= h_0 / self.T < 1.75:
                self.alpha_xx = (-4 * 10 ** (-12)) * self.F_rh**3 - 0.2143 * self.F_rh**2 - 0.0643 * self.F_rh + 0.9997
            if 1.75 <= h_0 / self.T < 2.25:
                self.alpha_xx = -0.8333 * self.F_rh**3 + 0.25 * self.F_rh**2 - 0.0167 * self.F_rh + 1
            if 2.25 <= h_0 / self.T < 2.75:
                self.alpha_xx = -1.25 * self.F_rh**4 + 0.5833 * self.F_rh**3 - 0.0375 * self.F_rh**2 - 0.0108 * self.F_rh + 1
            if h_0 / self.T >= 2.75:
                self.alpha_xx = 1

        if self.F_rh > 0.4:
            if 0 <= h_0 / self.T < 1.75:
                self.alpha_xx = (
                    -0.9274 * self.F_rh**6
                    + 9.5953 * self.F_rh**5
                    - 37.197 * self.F_rh**4
                    + 69.666 * self.F_rh**3
                    - 65.391 * self.F_rh**2
                    + 28.025 * self.F_rh
                    - 3.4143
                )
            if 1.75 <= h_0 / self.T < 2.25:
                self.alpha_xx = (
                    2.2152 * self.F_rh**6
                    - 11.852 * self.F_rh**5
                    + 21.499 * self.F_rh**4
                    - 12.174 * self.F_rh**3
                    - 4.7873 * self.F_rh**2
                    + 5.8662 * self.F_rh
                    - 0.2652
                )
            if 2.25 <= h_0 / self.T < 2.75:
                self.alpha_xx = (
                    1.2205 * self.F_rh**6
                    - 5.4999 * self.F_rh**5
                    + 5.7966 * self.F_rh**4
                    + 6.6491 * self.F_rh**3
                    - 16.123 * self.F_rh**2
                    + 9.2016 * self.F_rh
                    - 0.6342
                )
            if 2.75 <= h_0 / self.T < 3.25:
                self.alpha_xx = (
                    -0.4085 * self.F_rh**6
                    + 4.534 * self.F_rh**5
                    - 18.443 * self.F_rh**4
                    + 35.744 * self.F_rh**3
                    - 34.381 * self.F_rh**2
                    + 15.042 * self.F_rh
                    - 1.3807
                )
            if 3.25 <= h_0 / self.T < 3.75:
                self.alpha_xx = (
                    0.4078 * self.F_rh**6
                    - 0.919 * self.F_rh**5
                    - 3.8292 * self.F_rh**4
                    + 15.738 * self.F_rh**3
                    - 19.766 * self.F_rh**2
                    + 9.7466 * self.F_rh
                    - 0.6409
                )
            if 3.75 <= h_0 / self.T < 4.5:
                self.alpha_xx = (
                    0.3067 * self.F_rh**6
                    - 0.3404 * self.F_rh**5
                    - 5.0511 * self.F_rh**4
                    + 16.892 * self.F_rh**3
                    - 20.265 * self.F_rh**2
                    + 9.9002 * self.F_rh
                    - 0.6712
                )
            if 4.5 <= h_0 / self.T < 5.5:
                self.alpha_xx = (
                    0.3212 * self.F_rh**6
                    - 0.3559 * self.F_rh**5
                    - 5.1056 * self.F_rh**4
                    + 16.926 * self.F_rh**3
                    - 20.253 * self.F_rh**2
                    + 10.013 * self.F_rh
                    - 0.7196
                )
            if 5.5 <= h_0 / self.T < 6.5:
                self.alpha_xx = (
                    0.9252 * self.F_rh**6
                    - 4.2574 * self.F_rh**5
                    + 5.0363 * self.F_rh**4
                    + 3.3282 * self.F_rh**3
                    - 10.367 * self.F_rh**2
                    + 6.3993 * self.F_rh
                    - 0.2074
                )
            if 6.5 <= h_0 / self.T < 7.5:
                self.alpha_xx = (
                    0.8442 * self.F_rh**6
                    - 4.0261 * self.F_rh**5
                    + 5.313 * self.F_rh**4
                    + 1.6442 * self.F_rh**3
                    - 8.1848 * self.F_rh**2
                    + 5.3209 * self.F_rh
                    - 0.0267
                )
            if 7.5 <= h_0 / self.T < 8.5:
                self.alpha_xx = (
                    0.1211 * self.F_rh**6
                    + 0.628 * self.F_rh**5
                    - 6.5106 * self.F_rh**4
                    + 16.7 * self.F_rh**3
                    - 18.267 * self.F_rh**2
                    + 8.7077 * self.F_rh
                    - 0.4745
                )

            if 8.5 <= h_0 / self.T < 9.5:
                if self.F_rh < 0.6:
                    self.alpha_xx = 1
                if self.F_rh >= 0.6:
                    self.alpha_xx = (
                        -6.4069 * self.F_rh**6
                        + 47.308 * self.F_rh**5
                        - 141.93 * self.F_rh**4
                        + 220.23 * self.F_rh**3
                        - 185.05 * self.F_rh**2
                        + 79.25 * self.F_rh
                        - 12.484
                    )
            if h_0 / self.T >= 9.5:
                if self.F_rh < 0.6:
                    self.alpha_xx = 1
                if self.F_rh >= 0.6:
                    self.alpha_xx = (
                        -6.0727 * self.F_rh**6
                        + 44.97 * self.F_rh**5
                        - 135.21 * self.F_rh**4
                        + 210.13 * self.F_rh**3
                        - 176.72 * self.F_rh**2
                        + 75.728 * self.F_rh
                        - 11.893
                    )

        self.V_2 = v / self.alpha_xx

    def calculate_wave_resistance(self, v, h_0):
        """Wave resistance

        - 4th resistance component defined by Holtrop and Mennen (1982)
        - When the speed or the vessel size increases, the wave making resistance increases
        - In shallow water, the wave resistance shows an asymptotical behaviour by reaching the critical speed
        """

        self.karpov(v, h_0)

        assert self.g >= 0, f"g should be positive: {self.g}"
        assert self.L >= 0, f"L should be positive: {self.L}"
        # self.F_rL = self.V_2 / np.sqrt(self.g * self.L)  # Froude number based on ship's speed to water and its length of waterline
        self.F_rL = v / np.sqrt(self.g * self.L)  # Froude number based on ship's speed to water and its length of waterline
        # parameter c_7 is determined by the B/L ratio
        if self.B / self.L < 0.11:
            self.c_7 = 0.229577 * (self.B / self.L) ** 0.33333
        if self.B / self.L > 0.25:
            self.c_7 = 0.5 - 0.0625 * (self.L / self.B)
        else:
            self.c_7 = self.B / self.L

        # half angle of entrance in degrees
        self.i_E = 1 + 89 * np.exp(
            -((self.L / self.B) ** 0.80856)
            * ((1 - self.C_WP) ** 0.30484)
            * ((1 - self.C_P - 0.0225 * self.lcb) ** 0.6367)
            * ((self.L_R / self.B) ** 0.34574)
            * ((100 * self.delta / (self.L**3)) ** 0.16302)
        )

        self.c_1 = 2223105 * (self.c_7**3.78613) * ((self.T / self.B) ** 1.07961) * (90 - self.i_E) ** (-1.37165)
        self.c_2 = 1  # accounts for the effect of the bulbous bow, which is not present at inland ships
        self.c_5 = 1 - (0.8 * self.A_T) / (self.B * self.T * self.C_M)  # influence of the transom stern on the wave resistance

        # parameter c_15 depoends on the ratio L^3 / delta
        if (self.L**3) / self.delta < 512:
            self.c_15 = -1.69385
        if (self.L**3) / self.delta > 1727:
            self.c_15 = 0
        else:
            self.c_15 = -1.69385 + (self.L / (self.delta ** (1 / 3)) - 8) / 2.36

        # parameter c_16 depends on C_P
        if self.C_P < 0.8:
            self.c_16 = 8.07981 * self.C_P - 13.8673 * (self.C_P**2) + 6.984388 * (self.C_P**3)
        else:
            self.c_16 = 1.73014 - 0.7067 * self.C_P

        if self.L / self.B < 12:
            self.lmbda = 1.446 * self.C_P - 0.03 * (self.L / self.B)
        else:
            self.lmbda = 1.446 * self.C_P - 0.36

        self.m_1 = (
            0.0140407 * (self.L / self.T) - 1.75254 * ((self.delta) ** (1 / 3) / self.L) - 4.79323 * (self.B / self.L) - self.c_16
        )
        self.m_2 = self.c_15 * (self.C_P**2) * np.exp((-0.1) * (self.F_rL ** (-2)))

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

        return self.R_W

    def calculate_residual_resistance(self, v, h_0):
        """Residual resistance terms

        - Holtrop and Mennen (1982) defined three residual resistance terms:
        - 1) Resistance due to immersed transom (R_TR), Karpov corrected velocity V2 is used
        - 2) Resistance due to model-ship correlation (R_A), Karpov corrected velocity V2 is used
        - 3) Resistance due to the bulbous bow (R_B), Karpov corrected velocity V2 is used
        """

        self.karpov(v, h_0)

        self.V_2 = v
        # Resistance due to immersed transom: R_TR [kN]
        self.F_nT = self.V_2 / np.sqrt(
            2 * self.g * self.A_T / (self.B + self.B * self.C_WP)
        )  # Froude number based on transom immersion
        assert not isinstance(self.F_nT, complex), f"residual? froude number should not be complex: {self.F_nT}"

        self.c_6 = 0.2 * (1 - 0.2 * self.F_nT)  # Assuming F_nT < 5, this is the expression for coefficient c_6

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

        return self.R_res

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

        return self.R_tot

    def calculate_total_power_required(self, v, h_0):
        """Total required power:

        - The total required power is the sum of the power for systems on board (P_hotel) + power required for propulsion
        - The power required for propulsion depends on the calculated resistance

        Output:
        - P_propulsion: required power for propulsion, equals to P_d (Delivered Horse Power)
        - P_tot: required power for propulsion and hotelling
        - P_given: the power given by the engine to the ship (for propulsion and hotelling), which is the actual power the ship uses

        Note:
        In this version, we define the propulsion power as P_d (Delivered Horse Power) ratehr than P_b (Brake Horse Power). The reason we choose P_d as propulsion power is to prevent double use of the same power efficiencies.
        The details are 1) The P_b calculation involves gearing efficiency and transmission efficiency already while P_d not. 2) P_d is the power delivered to propellers. 3) To estimate the reneable fuel use, we will involve "energy conversion efficicies" later in the calculation. The 'energy conversion efficicies' for renewable fuel powered vessels are commonly measured/given as a whole covering the engine power systems, includes different engine (such as fuel cell engine, battery engine, internal combustion engine, hybird engine) efficiencies, and corresponding gearbox efficiencies, AC/DC converter efficiencies, excludes the efficiency items of propellers.
        Therefore, to algin with the later use of "energy conversion efficicies" for fuel use estimation and prevent double use of some power efficiencies such as gearing efficiency, here we choose P_d as propulsion power.
        """

        # Required power for systems on board, "5%" based on De Vos and van Gils (2011):Walstrom versus generators troom
        # self.P_hotel = 0.05 * self.P_installed

        # Required power for propulsion
        # Effective Horse Power (EHP), P_e
        self.P_e = v * self.R_tot

        #         # Calculation hull efficiency
        #         dw = np.zeros(101)  # velocity correction coefficient
        #         counter = 0

        #         if self.F_rL < 0.2:
        #             self.dw = 0
        #         else:
        #             self.dw = 0.1

        #         self.w = (
        #             0.11
        #             * (0.16 / self.x)
        #             * self.C_B
        #             * np.sqrt((self.delta ** (1 / 3)) / self.D_s)
        #             - self.dw
        #         )  # wake fraction 'w'

        #         assert not isinstance(self.w, complex), f"w should not be complex: {self.w}"

        #         if self.x == 1:
        #             self.t = 0.6 * self.w * (1 + 0.67 * self.w)  # thrust deduction factor 't'
        #         else:
        #             self.t = 0.8 * self.w * (1 + 0.25 * self.w)

        #         self.eta_h = (1 - self.t) / (1 - self.w)  # hull efficiency eta_h

        # Calculation hydrodynamic efficiency eta_D  according to Simic et al (2013) "On Energy Efficiency of Inland Waterway Self-Propelled Cargo Vessels", https://www.researchgate.net/publication/269103117
        # hydrodynamic efficiency eta_D is a ratio of power used to propel the ship and delivered power
        # relation between eta_D and ship velocity v

        if h_0 >= 9:
            if self.F_rh >= 0.5:
                self.eta_D = 0.6
            elif 0.325 <= self.F_rh < 0.5:
                self.eta_D = 0.7
            elif 0.28 <= self.F_rh < 0.325:
                self.eta_D = 0.59
            elif 0.2 < self.F_rh < 0.28:
                self.eta_D = 0.56
            elif 0.17 < self.F_rh <= 0.2:
                self.eta_D = 0.41
            elif 0.15 < self.F_rh <= 0.17:
                self.eta_D = 0.35
            else:
                self.eta_D = 0.29

        elif 5 <= h_0 < 9:
            if self.F_rh > 0.62:
                self.eta_D = 0.7
            elif 0.58 < self.F_rh < 0.62:
                self.eta_D = 0.68
            elif 0.57 < self.F_rh <= 0.58:
                self.eta_D = 0.7
            elif 0.51 < self.F_rh <= 0.57:
                self.eta_D = 0.68
            elif 0.475 < self.F_rh <= 0.51:
                self.eta_D = 0.53
            elif 0.45 < self.F_rh <= 0.475:
                self.eta_D = 0.4
            elif 0.36 < self.F_rh <= 0.45:
                self.eta_D = 0.37
            elif 0.33 < self.F_rh <= 0.36:
                self.eta_D = 0.36
            elif 0.3 < self.F_rh <= 0.33:
                self.eta_D = 0.35
            elif 0.28 < self.F_rh <= 0.3:
                self.eta_D = 0.331
            else:
                self.eta_D = 0.33
        else:
            if self.F_rh > 0.56:
                self.eta_D = 0.28
            elif 0.4 < self.F_rh <= 0.56:
                self.eta_D = 0.275
            elif 0.36 < self.F_rh <= 0.4:
                self.eta_D = 0.345
            elif 0.33 < self.F_rh <= 0.36:
                self.eta_D = 0.28
            elif 0.3 < self.F_rh <= 0.33:
                self.eta_D = 0.27
            elif 0.28 < self.F_rh <= 0.3:
                self.eta_D = 0.26
            else:
                self.eta_D = 0.25
        # Delivered Horse Power (DHP), P_d
        self.P_d = self.P_e / self.eta_D

        logger.debug("eta_D = {:.2f}".format(self.eta_D))
        # self.P_d = self.P_e / (self.eta_o * self.eta_r * self.eta_h)

        # Brake Horse Power (BHP), P_b (P_b was used in OpenTNsim version v1.1.2. we do not use it in this version. The reseaon is listed in the doc string above)
        # self.P_b = self.P_d / (self.eta_t * self.eta_g)

        self.P_propulsion = self.P_d  # propulsion power is defined here as Delivered horse power, the power delivered to propellers

        self.P_tot = self.P_hotel + self.P_propulsion

        # Partial engine load (P_partial): needed in the 'Emission calculations'
        if self.P_tot > self.P_installed:
            self.P_given = self.P_installed
            self.P_partial = 1
        else:
            self.P_given = self.P_tot
            self.P_partial = self.P_tot / self.P_installed

        # logger.debug(f'The total power required is {self.P_tot} kW')
        # logger.debug(f'The actual total power given is {self.P_given} kW')
        # logger.debug(f'The partial load is {self.P_partial}')

        assert not isinstance(self.P_given, complex), f"P_given number should not be complex: {self.P_given}"

        # return these three varible:
        # 1) self.P_propulsion, for the convience of validation.  (propulsion power and fuel used for propulsion),
        # 2) self.P_tot, know the required power, especially when it exceeds installed engine power while sailing shallower and faster
        # 3) self.P_given, the actual power the engine gives for "propulsion + hotel" within its capacity (means installed power). This varible is used for calculating delta_energy of each sailing time step.

        return self.P_propulsion, self.P_tot, self.P_given

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

        return self.diesel_use_g_m, self.diesel_use_g_m_C_year

    def calculate_diesel_use_g_s(self):
        """Total diesel fuel use in g/s:

        - The total fuel use in g/s can be computed by total emission in g (P_tot * delt_t * self.total_factor_) diveded by the sailing duration (delt_t)
        """
        self.diesel_use_g_s = self.P_given * self.final_SFC_diesel_ICE_mass / 3600  # without considering C_year
        self.diesel_use_g_s_C_year = self.P_given * self.final_SFC_diesel_C_year_ICE_mass / 3600  # considering C_year

        return self.diesel_use_g_s, self.diesel_use_g_s_C_year

    # TO DO: Add functions here to calculate renewable energy source use rate in g/m, g/s

    def calculate_emission_rates_g_m(self, v):
        """CO2, PM10, NOX emission rates in g/m:

        - The CO2, PM10, NOX emission rates in g/m can be computed by total fuel use in g (P_tot * delt_t * self.total_factor_) diveded by the sailing distance (v * delt_t)
        """
        self.emission_g_m_CO2 = self.P_given * self.total_factor_CO2 / v / 3600
        self.emission_g_m_PM10 = self.P_given * self.total_factor_PM10 / v / 3600
        self.emission_g_m_NOX = self.P_given * self.total_factor_NOX / v / 3600

        return self.emission_g_m_CO2, self.emission_g_m_PM10, self.emission_g_m_NOX

    def calculate_emission_rates_g_s(self):
        """CO2, PM10, NOX emission rates in g/s:

        - The CO2, PM10, NOX emission rates in g/s can be computed by total fuel use in g (P_tot * delt_t * self.total_factor_) diveded by the sailing duration (delt_t)
        """
        self.emission_g_s_CO2 = self.P_given * self.total_factor_CO2 / 3600
        self.emission_g_s_PM10 = self.P_given * self.total_factor_PM10 / 3600
        self.emission_g_s_NOX = self.P_given * self.total_factor_NOX / 3600

        return self.emission_g_s_CO2, self.emission_g_s_PM10, self.emission_g_s_NOX

    def calculate_max_sinkage(self, v, h_0):
        """Calculate the maximum sinkage of a moving ship

        the calculation equation is described in Barrass, B. & Derrett, R.'s book (2006), Ship Stability for Masters and Mates,
        chapter 42. https://doi.org/10.1016/B978-0-08-097093-6.00042-6

        some explanation for the variables in the equation:
        - h_0: water depth
        - v: ship velocity relative to the water
        - 150: Here we use the standard width 150 m as the waterway width

        """

        max_sinkage = (self.C_B * ((self.B * self._T) / (150 * h_0)) ** 0.81) * ((v * 1.94) ** 2.08) / 20

        return max_sinkage

    def calculate_h_squat(self, v, h_0):
        if self.h_squat:
            h_squat = h_0 - self.calculate_max_sinkage(v, h_0)

        else:
            h_squat = h_0

        return h_squat


class EnergyCalculation:
    """Add information on energy use and effects on energy use."""

    # to do: add other alternatives from Marin's table to have completed renewable energy sources
    # to do: add renewable fuel cost from Marin's table, add fuel cell / other engine cost, power plan cost to calculate the cost of ship refit or new ships.

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
            "Waiting to enter waiting area stop",
            "Waiting in waiting area stop",
            "Waiting in line-up area stop",
            "Passing lock stop",
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
                h_0 = calculate_depth(geometries[i], geometries[i + 1])

                # printstatements to check the output (can be removed later)
                logger.debug("delta_t: {:.4f} s".format(delta_t))
                logger.debug("distance: {:.4f} m".format(distance))
                logger.debug("velocity: {:.4f} m/s".format(v))

                # we use the calculated velocity to determine the resistance and power required
                # we can switch between the 'original water depth' and 'water depth considering ship squatting' for energy calculation, by using the function "calculate_h_squat (h_squat is set as Yes/No)" in the core.py
                h_0 = self.vessel.calculate_h_squat(v, h_0)
                print(h_0)
                self.vessel.calculate_total_resistance(v, h_0)
                self.vessel.calculate_total_power_required(v=v, h_0=h_0)

                self.vessel.calculate_emission_factors_total(v=v, h_0=h_0)
                self.vessel.calculate_SFC_final(v=v, h_0=h_0)

                if messages[i + 1] in stationary_phase_indicator:  # if we are in a stationary stage only log P_hotel
                    # Energy consumed per time step delta_t in the stationary stage
                    energy_delta = self.vessel.P_hotel * delta_t / 3600  # kJ/3600 = kWh

                    # Emissions CO2, PM10 and NOX, in gram - emitted in the stationary stage per time step delta_t,
                    # consuming 'energy_delta' kWh
                    P_hotel_delta = self.vessel.P_hotel  # in kW
                    P_installed_delta = self.vessel.P_installed  # in kW

                else:  # otherwise log P_tot
                    # Energy consumed per time step delta_t in the propulsion stage
                    energy_delta = (
                        self.vessel.P_given * delta_t / 3600
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
                    # To do: we need to rename the factor name for fuels, not starting with "emission" , consider seperating it from emission factors
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

    def plot(self):
        import folium

        df = pd.DataFrame.from_dict(self.energy_use)

        m = folium.Map(location=[51.7, 4.4], zoom_start=12)

        line = []
        for index, row in df.iterrows():
            line.append((row["edge_start"].y, row["edge_start"].x))

        folium.PolyLine(line, weight=4).add_to(m)

        return m
