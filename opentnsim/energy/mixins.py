"""Mixins for Energy

This script contains the classes
    - ConsumesEnergy: a mixin for objects that consume energy
    - EnergyCalculations: a class to perform energy calculations and save results in a logbook. This is not a mixin, but must be called after the simulation.
"""

# %% IMPORT DEPENENDENCIES
# generic
import pathlib
import logging
import numpy as np
import pandas as pd

# OpenTNSim
import opentnsim
from opentnsim.graph.mixins import calculate_distance, calculate_depth
from opentnsim.energy.algorithms import power2v
from opentnsim.energy.calculations import (
    sample_engine_age,
    calculate_properties,
    calculate_frictional_resistance,
    calculate_viscous_resistance,
    calculate_appendage_resistance,
    karpov,
    calculate_wave_resistance,
    calculate_residual_resistance,
    calculate_total_power_required,
    calculate_max_sinkage,
    calculate_total_resistance,
)


# logging
logger = logging.getLogger(__name__)


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
    karpov_correction : bool, optional
        If True, apply Karpovs correction for velocity under the vessel, if False,
        use the speed to water.
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
        karpov_correction=False,
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
        self.karpov_correction = karpov_correction

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
        self.power2v = opentnsim.energy.algorithms.power2v

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
            raise ValueError(
                "year must be set to calculate the construction year of the engine"
            )
        self.C_year = self.year - self.age
        logger.debug(
            f"Engine age calculated as {self.age}, hence construction year is {self.C_year}"
        )

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
            B=self.B,
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

    def calculate_wave_resistance(self, V_2, h_0):
        """Wave resistance

        - 4th resistance component defined by Holtrop and Mennen (1982)
        - When the speed or the vessel size increases, the wave making resistance increases
        - In shallow water, the wave resistance shows an asymptotical behaviour by reaching the critical speed
        """

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
            V_2=V_2,
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

    def calculate_residual_resistance(self, V_2, h_0):
        """Residual resistance terms

        - Holtrop and Mennen (1982) defined three residual resistance terms:
        - 1) Resistance due to immersed transom (R_TR), Karpov corrected velocity V2 is used
        - 2) Resistance due to model-ship correlation (R_A), Karpov corrected velocity V2 is used
        - 3) Resistance due to the bulbous bow (R_B), Karpov corrected velocity V2 is used
        """

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
            V_2=V_2,
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
        # TODO: we doen nu hier alle stappen en combineren dan alles, waarom roepen we hier niet
        #  calculate_total_resistance uit de calculations.py aan?

        self.calculate_properties()
        self.calculate_frictional_resistance(v, h_0)
        self.calculate_viscous_resistance()
        self.calculate_appendage_resistance(v)

        self.karpov(v, h_0)
        # print('Original v = {} m/s, Karpov corrected V_2 = {} m/s'.format(v, self.V_2))
        if self.karpov_correction:
            self.calculate_wave_resistance(self.V_2, h_0)
            self.calculate_residual_resistance(self.V_2, h_0)
        else:
            self.calculate_wave_resistance(v, h_0)
            self.calculate_residual_resistance(v, h_0)

        # The total resistance R_tot [kN] = R_f * (1+k1) + R_APP + R_W + R_TR + R_A
        self.R_tot = (
            self.R_f * self.one_k1
            + self.R_APP
            + self.R_W
            + self.R_TR
            + self.R_A
            + self.R_B
        )

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
        Table 3-2 from Marin report 2019,  Energietransitie emissieloze binnenvaart, vooronderzoek ontwerpaspecten,
        systeem configuraties.(Energy transition zero-emission inland shipping, preliminary research on design aspects,
        system configurations

        Note:
        net energy density can be used for calculate fuel consumption in mass and volume, but for required energy
        source storage space determination, the packaging factors of different energy sources also need to be
        considered.
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

        # ToDo: Check - in the function 'energy_density' the 'Edens*' are in kWh/kg, but the units in the comments here,
        #  suggest that the 'SFC*' are in g/kWh (wonder if that then should not be kg/kWh)
        # to estimate the requirement of the amount of ZES_batterypacks for different IET scenarios, we include ZES battery capacity per container here.
        # ZES_batterypack capacity > 2000kWh, its average usable energy = 2000 kWh,  mass = 27 ton, vol = 20ft A60 container (6*2.5*2.5 = 37.5 m3) (source: ZES report)
        self.energy_density()
        self.energy_conversion_efficiency()

        self.ZES_batterypack2000kWh = 2000  # kWh/pack,

        # SFC in mass for Fuel Cell engine
        self.SFC_LH2_FuelCell_mass = 1 / (
            self.Edens_LH2_mass * self.Eeff_FuelCell
        )  # g/kWh
        self.SFC_eLNG_FuelCell_mass = 1 / (
            self.Edens_eLNG_mass * self.Eeff_FuelCell
        )  # g/kWh
        self.SFC_eMethanol_FuelCell_mass = 1 / (
            self.Edens_eMethanol_mass * self.Eeff_FuelCell
        )  # g/kWh
        self.SFC_eNH3_FuelCell_mass = 1 / (
            self.Edens_eNH3_mass * self.Eeff_FuelCell
        )  # g/kWh

        # SFC in mass for ICE engine
        self.SFC_diesel_ICE_mass = 1 / (self.Edens_diesel_mass * self.Eeff_ICE)  # g/kWh
        self.SFC_eLNG_ICE_mass = 1 / (self.Edens_eLNG_mass * self.Eeff_ICE)  # g/kWh
        self.SFC_eMethanol_ICE_mass = 1 / (
            self.Edens_eMethanol_mass * self.Eeff_ICE
        )  # g/kWh
        self.SFC_eNH3_ICE_mass = 1 / (self.Edens_eNH3_mass * self.Eeff_ICE)  # g/kWh

        # SFC in mass and volume for battery electric ships
        self.SFC_Li_NMC_Battery_mass = 1 / (
            self.Edens_Li_NMC_Battery_mass * self.Eeff_Battery
        )  # g/kWh
        self.SFC_Li_NMC_Battery_vol = 1 / (
            self.Edens_Li_NMC_Battery_vol * self.Eeff_Battery
        )  # m3/kWh
        self.SFC_ZES_battery2000kWh = 1 / (
            self.ZES_batterypack2000kWh * self.Eeff_Battery
        )  # kWh

        # SFC in volume for Fuel Cell engine
        self.SFC_LH2_FuelCell_vol = 1 / (
            self.Edens_LH2_vol * self.Eeff_FuelCell
        )  # m3/kWh
        self.SFC_eLNG_FuelCell_vol = 1 / (
            self.Edens_eLNG_vol * self.Eeff_FuelCell
        )  # m3/kWh
        self.SFC_eMethanol_FuelCell_vol = 1 / (
            self.Edens_eMethanol_vol * self.Eeff_FuelCell
        )  # m3/kWh
        self.SFC_eNH3_FuelCell_vol = 1 / (
            self.Edens_eNH3_vol * self.Eeff_FuelCell
        )  # m3/kWh

        # SFC in volume for ICE engine
        self.SFC_diesel_ICE_vol = 1 / (self.Edens_diesel_vol * self.Eeff_ICE)  # m3/kWh
        self.SFC_eLNG_ICE_vol = 1 / (self.Edens_eLNG_vol * self.Eeff_ICE)  # m3/kWh
        self.SFC_eMethanol_ICE_vol = 1 / (
            self.Edens_eMethanol_vol * self.Eeff_ICE
        )  # m3/kWh
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

        logger.debug(
            f"The general fuel consumption factor for diesel is {self.SFC_diesel_C_year} g/kWh"
        )

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
        self.calculate_total_power_required(
            v=v, h_0=h_0
        )  # You need the P_partial values

        # Import the correction factors table
        # TODO: use package data, not an arbitrary location
        self.C_partial_load = load_partial_engine_load_correction_factors()
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
                    self.C_partial_load_NOX = self.C_partial_load.iloc[
                        0, 1
                    ]  # <= CCR-1 class
                if 2008 <= self.C_year <= 2019:
                    self.C_partial_load_NOX = self.C_partial_load.iloc[
                        0, 2
                    ]  # CCR-2 / Stage IIIa
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

            elif (
                self.C_partial_load.iloc[i, 0]
                < self.P_partial
                <= self.C_partial_load.iloc[i + 1, 0]
            ):
                self.C_partial_load_CO2 = (
                    (self.P_partial - self.C_partial_load.iloc[i, 0])
                    * (
                        self.C_partial_load.iloc[i + 1, 5]
                        - self.C_partial_load.iloc[i, 5]
                    )
                ) / (
                    self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]
                ) + self.C_partial_load.iloc[
                    i, 5
                ]
                self.C_partial_load_PM10 = (
                    (self.P_partial - self.C_partial_load.iloc[i, 0])
                    * (
                        self.C_partial_load.iloc[i + 1, 6]
                        - self.C_partial_load.iloc[i, 6]
                    )
                ) / (
                    self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]
                ) + self.C_partial_load.iloc[
                    i, 6
                ]
                self.C_partial_load_fuel_ICE = (
                    self.C_partial_load_CO2
                )  # CO2 emission is generated from fuel consumption, so these two
                # correction factors are equal
                self.C_partial_load_PEMFC = (
                    (self.P_partial - self.C_partial_load.iloc[i, 0])
                    * (
                        self.C_partial_load.iloc[i + 1, 7]
                        - self.C_partial_load.iloc[i, 7]
                    )
                ) / (
                    self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]
                ) + self.C_partial_load.iloc[
                    i, 7
                ]
                self.C_partial_load_SOFC = (
                    (self.P_partial - self.C_partial_load.iloc[i, 0])
                    * (
                        self.C_partial_load.iloc[i + 1, 8]
                        - self.C_partial_load.iloc[i, 8]
                    )
                ) / (
                    self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]
                ) + self.C_partial_load.iloc[
                    i, 8
                ]
                if self.C_year < 2008:
                    self.C_partial_load_NOX = (
                        (self.P_partial - self.C_partial_load.iloc[i, 0])
                        * (
                            self.C_partial_load.iloc[i + 1, 1]
                            - self.C_partial_load.iloc[i, 1]
                        )
                    ) / (
                        self.C_partial_load.iloc[i + 1, 0]
                        - self.C_partial_load.iloc[i, 0]
                    ) + self.C_partial_load.iloc[
                        i, 1
                    ]
                if 2008 <= self.C_year <= 2019:
                    self.C_partial_load_NOX = (
                        (self.P_partial - self.C_partial_load.iloc[i, 0])
                        * (
                            self.C_partial_load.iloc[i + 1, 2]
                            - self.C_partial_load.iloc[i, 2]
                        )
                    ) / (
                        self.C_partial_load.iloc[i + 1, 0]
                        - self.C_partial_load.iloc[i, 0]
                    ) + self.C_partial_load.iloc[
                        i, 2
                    ]
                if self.C_year > 2019:
                    if self.L_w == 1:
                        self.C_partial_load_NOX = (
                            (self.P_partial - self.C_partial_load.iloc[i, 0])
                            * (
                                self.C_partial_load.iloc[i + 1, 3]
                                - self.C_partial_load.iloc[i, 3]
                            )
                        ) / (
                            self.C_partial_load.iloc[i + 1, 0]
                            - self.C_partial_load.iloc[i, 0]
                        ) + self.C_partial_load.iloc[
                            i, 3
                        ]
                    else:
                        self.C_partial_load_NOX = (
                            (self.P_partial - self.C_partial_load.iloc[i, 0])
                            * (
                                self.C_partial_load.iloc[i + 1, 4]
                                - self.C_partial_load.iloc[i, 4]
                            )
                        ) / (
                            self.C_partial_load.iloc[i + 1, 0]
                            - self.C_partial_load.iloc[i, 0]
                        ) + self.C_partial_load.iloc[
                            i, 4
                        ]

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
                    self.C_partial_load_NOX = self.C_partial_load.iloc[
                        19, 1
                    ]  # <= CCR-1 class
                if 2008 <= self.C_year <= 2019:
                    self.C_partial_load_NOX = self.C_partial_load.iloc[
                        19, 2
                    ]  # CCR-2 / Stage IIIa
                if self.C_year > 2019:
                    if self.L_w == 1:  #
                        self.C_partial_load_NOX = self.C_partial_load.iloc[
                            19, 3
                        ]  # Stage V: IWP/IWA-v/c-3 class (vessels with P <300 kW: assumed to be weight class L1)
                    else:
                        self.C_partial_load_NOX = self.C_partial_load.iloc[
                            19, 4
                        ]  # Stage V:IWP/IWA-v/c-4 class (vessels with P >300 kw: assumed to be weight class L2-L3)

        logger.debug(
            f"Partial engine load correction factor of CO2 is {self.C_partial_load_CO2}"
        )
        logger.debug(
            f"Partial engine load correction factor of PM10 is {self.C_partial_load_PM10}"
        )
        logger.debug(
            f"Partial engine load correction factor of NOX is {self.C_partial_load_NOX}"
        )
        logger.debug(
            f"Partial engine load correction factor of diesel fuel consumption in ICE is {self.C_partial_load_fuel_ICE}"
        )
        logger.debug(
            f"Partial engine load correction factor of fuel consumption in PEMFC is {self.C_partial_load_PEMFC}"
        )
        logger.debug(
            f"Partial engine load correction factor of fuel consumption in SOFC is {self.C_partial_load_SOFC}"
        )
        logger.debug(
            f"Partial engine load correction factor of energy consumption in battery is {self.C_partial_load_battery}"
        )

    def calculate_emission_factors_total(self, v, h_0):
        """Total emission factors:

        - The total emission factors can be computed by multiplying the general emission factor by the correction factor
        """

        self.emission_factors_general()  # You need the values of the general emission factors of CO2, PM10, NOX
        self.correction_factors(
            v=v, h_0=h_0
        )  # You need the correction factors of CO2, PM10, NOX

        # The total emission factor is calculated by multiplying the general emission factor (EF_CO2 / EF_PM10 / EF_NOX)
        # by the correction factor (C_partial_load_CO2 / C_partial_load_PM10 / C_partial_load_NOX)

        self.total_factor_CO2 = self.EF_CO2 * self.C_partial_load_CO2
        self.total_factor_PM10 = self.EF_PM10 * self.C_partial_load_PM10
        self.total_factor_NOX = self.EF_NOX * self.C_partial_load_NOX

        logger.debug(
            f"The total emission factor of CO2 is {self.total_factor_CO2} g/kWh"
        )
        logger.debug(
            f"The total emission factor of PM10 is {self.total_factor_PM10} g/kWh"
        )
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
        self.final_SFC_LH2_mass_PEMFC = (
            self.SFC_LH2_FuelCell_mass * self.C_partial_load_PEMFC
        )
        self.final_SFC_LH2_mass_SOFC = (
            self.SFC_LH2_FuelCell_mass * self.C_partial_load_SOFC
        )
        self.final_SFC_eLNG_mass_PEMFC = (
            self.SFC_eLNG_FuelCell_mass * self.C_partial_load_PEMFC
        )
        self.final_SFC_eLNG_mass_SOFC = (
            self.SFC_eLNG_FuelCell_mass * self.C_partial_load_SOFC
        )
        self.final_SFC_eMethanol_mass_PEMFC = (
            self.SFC_eMethanol_FuelCell_mass * self.C_partial_load_PEMFC
        )
        self.final_SFC_eMethanol_mass_SOFC = (
            self.SFC_eMethanol_FuelCell_mass * self.C_partial_load_SOFC
        )
        self.final_SFC_eNH3_mass_PEMFC = (
            self.SFC_eNH3_FuelCell_mass * self.C_partial_load_PEMFC
        )
        self.final_SFC_eNH3_mass_SOFC = (
            self.SFC_eNH3_FuelCell_mass * self.C_partial_load_SOFC
        )

        # final SFC of fuel cell in vol  [m3/kWh]
        self.final_SFC_LH2_vol_PEMFC = (
            self.SFC_LH2_FuelCell_vol * self.C_partial_load_PEMFC
        )
        self.final_SFC_LH2_vol_SOFC = (
            self.SFC_LH2_FuelCell_vol * self.C_partial_load_SOFC
        )
        self.final_SFC_eLNG_vol_PEMFC = (
            self.SFC_eLNG_FuelCell_vol * self.C_partial_load_PEMFC
        )
        self.final_SFC_eLNG_vol_SOFC = (
            self.SFC_eLNG_FuelCell_vol * self.C_partial_load_SOFC
        )
        self.final_SFC_eMethanol_vol_PEMFC = (
            self.SFC_eMethanol_FuelCell_vol * self.C_partial_load_PEMFC
        )
        self.final_SFC_eMethanol_vol_SOFC = (
            self.SFC_eMethanol_FuelCell_vol * self.C_partial_load_SOFC
        )
        self.final_SFC_eNH3_vol_PEMFC = (
            self.SFC_eNH3_FuelCell_vol * self.C_partial_load_PEMFC
        )
        self.final_SFC_eNH3_vol_SOFC = (
            self.SFC_eNH3_FuelCell_vol * self.C_partial_load_SOFC
        )

        # final SFC of ICE in mass [g/kWh]
        self.final_SFC_diesel_C_year_ICE_mass = (
            self.SFC_diesel_C_year * self.C_partial_load_fuel_ICE
        )
        self.final_SFC_diesel_ICE_mass = (
            self.SFC_diesel_ICE_mass * self.C_partial_load_fuel_ICE
        )
        self.final_SFC_eLNG_ICE_mass = (
            self.SFC_eLNG_ICE_mass * self.C_partial_load_fuel_ICE
        )
        self.final_SFC_eMethanol_ICE_mass = (
            self.SFC_eMethanol_ICE_mass * self.C_partial_load_fuel_ICE
        )
        self.final_SFC_eNH3_ICE_mass = (
            self.SFC_eNH3_ICE_mass * self.C_partial_load_fuel_ICE
        )

        # final SFC of ICE in vol  [m3/kWh]
        self.final_SFC_diesel_ICE_vol = (
            self.SFC_diesel_ICE_vol * self.C_partial_load_fuel_ICE
        )
        self.final_SFC_eLNG_ICE_vol = (
            self.SFC_eLNG_ICE_vol * self.C_partial_load_fuel_ICE
        )
        self.final_SFC_eMethanol_ICE_vol = (
            self.SFC_eMethanol_ICE_vol * self.C_partial_load_fuel_ICE
        )
        self.final_SFC_eNH3_ICE_vol = (
            self.SFC_eNH3_ICE_vol * self.C_partial_load_fuel_ICE
        )

        # final SFC of battery in mass and vol
        self.final_SFC_Li_NMC_Battery_mass = (
            self.SFC_Li_NMC_Battery_mass * self.C_partial_load_battery
        )  # g/kWh
        self.final_SFC_Li_NMC_Battery_vol = (
            self.SFC_Li_NMC_Battery_vol * self.C_partial_load_battery
        )  # m3/kWh
        self.final_SFC_Battery2000kWh = (
            self.SFC_ZES_battery2000kWh * self.C_partial_load_battery
        )  # kWh

    def calculate_diesel_use_g_m(self, v):
        """Total diesel fuel use in g/m:

        - The total fuel use in g/m can be computed by total fuel use in g (P_tot * delta_t * self.total_factor_) divided by the sailing distance (v * delt_t)
        """
        self.diesel_use_g_m = (
            self.P_given * self.final_SFC_diesel_ICE_mass / v
        ) / 3600  # without considering C_year
        self.diesel_use_g_m_C_year = (
            self.P_given * self.final_SFC_diesel_C_year_ICE_mass / v
        ) / 3600  # considering C_year

    def calculate_diesel_use_g_s(self):
        """Total diesel fuel use in g/s:

        - The total fuel use in g/s can be computed by total emission in g (P_tot * delta_t * self.total_factor_) diveded by the sailing duration (delt_t)
        """
        self.diesel_use_g_s = (
            self.P_given * self.final_SFC_diesel_ICE_mass / 3600
        )  # without considering C_year
        self.diesel_use_g_s_C_year = (
            self.P_given * self.final_SFC_diesel_C_year_ICE_mass / 3600
        )  # considering C_year

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

        The amount of water under the keel is calculated h_0 - T. When h_squat is set
        to True, we estimate a max_sinkage that is subtracted from h_0. This values is
        returned as h_squat for further calculation.

        """
        h_squat = h_0 - self.calculate_max_sinkage(v, h_0, width=width)

        return h_squat


class EnergyCalculation:
    """Add information on energy use and effects on energy use."""

    # ToDo: add other alternatives from Marin's table to have completed renewable energy sources
    # ToDo: add renewable fuel cost from Marin's table, add fuel cell / other engine cost, power plan cost to calculate the cost of ship refit or new ships.

    def __init__(self, graph, vessel, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """Initialization"""
        self.graph = graph
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
                v = (
                    distance / delta_t
                )  # TODO: this is probably wrong. You don't want the speed over ground here, but the speed to water
                self.energy_use["distance"].append(distance)

                # calculate the delta t
                self.energy_use["delta_t"].append(delta_t)

                # calculate the water depth
                h_0 = calculate_depth(geometries[i], geometries[i + 1], self.graph)

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

                if (
                    messages[i + 1] in stationary_phase_indicator
                ):  # if we are in a stationary stage only log P_hotel
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
                    P_tot_delta = (
                        self.vessel.P_tot
                    )  # in kW, required power, may exceed installed engine power
                    P_given_delta = self.vessel.P_given  # in kW, actual given power
                    P_installed_delta = self.vessel.P_installed  # in kW
                    emission_delta_CO2 = (
                        self.vessel.total_factor_CO2 * energy_delta
                    )  # Energy consumed per time step delta_t in the                                                                                              #stationary phase # in g
                    emission_delta_PM10 = (
                        self.vessel.total_factor_PM10 * energy_delta
                    )  # in g
                    emission_delta_NOX = (
                        self.vessel.total_factor_NOX * energy_delta
                    )  # in g
                    # Todo: we need to rename the factor name for fuels, not starting with "emission", consider seperating it from emission factors
                    delta_diesel_C_year = (
                        self.vessel.final_SFC_diesel_C_year_ICE_mass * energy_delta
                    )  # in g
                    delta_diesel_ICE_mass = (
                        self.vessel.final_SFC_diesel_ICE_mass * energy_delta
                    )  # in g
                    delta_diesel_ICE_vol = (
                        self.vessel.final_SFC_diesel_ICE_vol * energy_delta
                    )  # in m3

                    delta_LH2_PEMFC_mass = (
                        self.vessel.final_SFC_LH2_mass_PEMFC * energy_delta
                    )  # in g
                    delta_LH2_SOFC_mass = (
                        self.vessel.final_SFC_LH2_mass_SOFC * energy_delta
                    )  # in g
                    delta_LH2_PEMFC_vol = (
                        self.vessel.final_SFC_LH2_vol_PEMFC * energy_delta
                    )  # in m3
                    delta_LH2_SOFC_vol = (
                        self.vessel.final_SFC_LH2_vol_SOFC * energy_delta
                    )  # in m3

                    delta_eLNG_PEMFC_mass = (
                        self.vessel.final_SFC_eLNG_mass_PEMFC * energy_delta
                    )  # in g
                    delta_eLNG_SOFC_mass = (
                        self.vessel.final_SFC_eLNG_mass_SOFC * energy_delta
                    )  # in g
                    delta_eLNG_PEMFC_vol = (
                        self.vessel.final_SFC_eLNG_vol_PEMFC * energy_delta
                    )  # in m3
                    delta_eLNG_SOFC_vol = (
                        self.vessel.final_SFC_eLNG_vol_SOFC * energy_delta
                    )  # in m3
                    delta_eLNG_ICE_mass = (
                        self.vessel.final_SFC_eLNG_ICE_mass * energy_delta
                    )  # in g
                    delta_eLNG_ICE_vol = (
                        self.vessel.final_SFC_eLNG_ICE_vol * energy_delta
                    )  # in m3

                    delta_eMethanol_PEMFC_mass = (
                        self.vessel.final_SFC_eMethanol_mass_PEMFC * energy_delta
                    )  # in g
                    delta_eMethanol_SOFC_mass = (
                        self.vessel.final_SFC_eMethanol_mass_SOFC * energy_delta
                    )  # in g
                    delta_eMethanol_PEMFC_vol = (
                        self.vessel.final_SFC_eMethanol_vol_PEMFC * energy_delta
                    )  # in m3
                    delta_eMethanol_SOFC_vol = (
                        self.vessel.final_SFC_eMethanol_vol_SOFC * energy_delta
                    )  # in m3
                    delta_eMethanol_ICE_mass = (
                        self.vessel.final_SFC_eMethanol_ICE_mass * energy_delta
                    )  # in g
                    delta_eMethanol_ICE_vol = (
                        self.vessel.final_SFC_eMethanol_ICE_vol * energy_delta
                    )  # in m3

                    delta_eNH3_PEMFC_mass = (
                        self.vessel.final_SFC_eNH3_mass_PEMFC * energy_delta
                    )  # in g
                    delta_eNH3_SOFC_mass = (
                        self.vessel.final_SFC_eNH3_mass_SOFC * energy_delta
                    )  # in g
                    delta_eNH3_PEMFC_vol = (
                        self.vessel.final_SFC_eNH3_vol_PEMFC * energy_delta
                    )  # in m3
                    delta_eNH3_SOFC_vol = (
                        self.vessel.final_SFC_eNH3_vol_SOFC * energy_delta
                    )  # in m3
                    delta_eNH3_ICE_mass = (
                        self.vessel.final_SFC_eNH3_ICE_mass * energy_delta
                    )  # in g
                    delta_eNH3_ICE_vol = (
                        self.vessel.final_SFC_eNH3_ICE_vol * energy_delta
                    )  # in m3

                    delta_Li_NMC_Battery_mass = (
                        self.vessel.final_SFC_Li_NMC_Battery_mass * energy_delta
                    )  # in g
                    delta_Li_NMC_Battery_vol = (
                        self.vessel.final_SFC_Li_NMC_Battery_vol * energy_delta
                    )  # in m3
                    delta_Battery2000kWh = (
                        self.vessel.final_SFC_Battery2000kWh * energy_delta
                    )  # in ZESpack number

                    self.energy_use["P_tot"].append(P_tot_delta)
                    self.energy_use["P_given"].append(P_given_delta)
                    self.energy_use["P_installed"].append(P_installed_delta)
                    self.energy_use["total_energy"].append(energy_delta)
                    self.energy_use["stationary"].append(energy_delta)
                    self.energy_use["total_emission_CO2"].append(emission_delta_CO2)
                    self.energy_use["total_emission_PM10"].append(emission_delta_PM10)
                    self.energy_use["total_emission_NOX"].append(emission_delta_NOX)
                    self.energy_use["total_diesel_consumption_C_year_ICE_mass"].append(
                        delta_diesel_C_year
                    )
                    self.energy_use["total_diesel_consumption_ICE_mass"].append(
                        delta_diesel_ICE_mass
                    )
                    self.energy_use["total_diesel_consumption_ICE_vol"].append(
                        delta_diesel_ICE_vol
                    )
                    self.energy_use["total_LH2_consumption_PEMFC_mass"].append(
                        delta_LH2_PEMFC_mass
                    )
                    self.energy_use["total_LH2_consumption_SOFC_mass"].append(
                        delta_LH2_SOFC_mass
                    )
                    self.energy_use["total_LH2_consumption_PEMFC_vol"].append(
                        delta_LH2_PEMFC_vol
                    )
                    self.energy_use["total_LH2_consumption_SOFC_vol"].append(
                        delta_LH2_SOFC_vol
                    )
                    self.energy_use["total_eLNG_consumption_PEMFC_mass"].append(
                        delta_eLNG_PEMFC_mass
                    )
                    self.energy_use["total_eLNG_consumption_SOFC_mass"].append(
                        delta_eLNG_SOFC_mass
                    )
                    self.energy_use["total_eLNG_consumption_PEMFC_vol"].append(
                        delta_eLNG_PEMFC_vol
                    )
                    self.energy_use["total_eLNG_consumption_SOFC_vol"].append(
                        delta_eLNG_SOFC_vol
                    )
                    self.energy_use["total_eLNG_consumption_ICE_mass"].append(
                        delta_eLNG_ICE_mass
                    )
                    self.energy_use["total_eLNG_consumption_ICE_vol"].append(
                        delta_eLNG_ICE_vol
                    )
                    self.energy_use["total_eMethanol_consumption_PEMFC_mass"].append(
                        delta_eMethanol_PEMFC_mass
                    )
                    self.energy_use["total_eMethanol_consumption_SOFC_mass"].append(
                        delta_eMethanol_SOFC_mass
                    )
                    self.energy_use["total_eMethanol_consumption_PEMFC_vol"].append(
                        delta_eMethanol_PEMFC_vol
                    )
                    self.energy_use["total_eMethanol_consumption_SOFC_vol"].append(
                        delta_eMethanol_SOFC_vol
                    )
                    self.energy_use["total_eMethanol_consumption_ICE_mass"].append(
                        delta_eMethanol_ICE_mass
                    )
                    self.energy_use["total_eMethanol_consumption_ICE_vol"].append(
                        delta_eMethanol_ICE_vol
                    )
                    self.energy_use["total_eNH3_consumption_PEMFC_mass"].append(
                        delta_eNH3_PEMFC_mass
                    )
                    self.energy_use["total_eNH3_consumption_SOFC_mass"].append(
                        delta_eNH3_SOFC_mass
                    )
                    self.energy_use["total_eNH3_consumption_PEMFC_vol"].append(
                        delta_eNH3_PEMFC_vol
                    )
                    self.energy_use["total_eNH3_consumption_SOFC_vol"].append(
                        delta_eNH3_SOFC_vol
                    )
                    self.energy_use["total_eNH3_consumption_ICE_mass"].append(
                        delta_eNH3_ICE_mass
                    )
                    self.energy_use["total_eNH3_consumption_ICE_vol"].append(
                        delta_eNH3_ICE_vol
                    )
                    self.energy_use["total_Li_NMC_Battery_mass"].append(
                        delta_Li_NMC_Battery_mass
                    )
                    self.energy_use["total_Li_NMC_Battery_vol"].append(
                        delta_Li_NMC_Battery_vol
                    )
                    self.energy_use["total_Battery2000kWh_consumption_num"].append(
                        delta_Battery2000kWh
                    )

                    self.energy_use["water depth"].append(h_0)
                    # self.energy_use["water depth info from vaarweginformatie.nl"].append(depth)

        # TODO: er moet hier een heel aantal dingen beter worden ingevuld
        # - de kruissnelheid is nu nog per default 1 m/s (zie de Movable mixin). Eigenlijk moet in de
        #   vessel database ook nog een speed_loaded en een speed_unloaded worden toegevoegd.
        # - er zou nog eens goed gekeken moeten worden wat er gedaan kan worden rond kustwerken
        # - en er is nog iets mis met de snelheid rond een sluis

        # - add HasCurrent Class or def
        # - De EnergyCalculation class heeft nu een logboek met veel variabelen. Welke variabelen worden bijgehouden hangt af van een if-statement. De lijsten zijn dus niet allemaal even lang.
