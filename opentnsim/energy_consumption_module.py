# package(s) related to the simulation
import numpy as np
import pandas as pd

class ConsumesEnergy:
    """Mixin class: Something that consumes energy.

    P_installed: installed engine power [kW]
    L_w: weight class of the ship (depending on carrying capacity) (classes: L1 (=1), L2 (=2), L3 (=3))
    C_b: block coefficient ('fullness') [-]
    nu: kinematic viscosity [m^2/s]
    rho: density of the surrounding water [kg/m^3]
    g: gravitational accelleration [m/s^2]
    x: number of propellors [-]
    eta_0: open water efficiency of propellor [-]
    eta_r: relative rotative efficiency [-]
    eta_t: transmission efficiency [-]
    eta_g: gearing efficiency [-]
    c_stern: determines shape of the afterbody [-]
    one_k2: appendage resistance factor [-]
    c_year: construction year of the engine [y]
    """

    def __init__(
            self,
            P_installed,
            L_w,
            C_b,
            nu=1 * 10 ** (-6),  # kinematic viscosity
            rho=1000,
            g=9.81,
            x=2,  # number of propellors
            eta_0=0.6,
            eta_r=1.00,
            eta_t=0.98,
            eta_g=0.96,
            c_stern=0,
            one_k2=2.5,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        """Initialization"""
        self.P_installed = P_installed
        self.L_w = L_w
        self.C_b = C_b
        self.nu = nu
        self.rho = rho
        self.g = g
        self.x = x
        self.eta_0 = eta_0
        self.eta_r = eta_r
        self.eta_t = eta_t
        self.eta_g = eta_g
        self.c_stern = c_stern
        self.one_k2 = one_k2
        self.c_year = self.calculate_engine_age()  # The construction year of the engine is now generated once, instead of for each time step

    # The engine age and construction year of the engine is computed with the function below.
    # The construction year of the engine is used in the emission functions (1) emission_factors_general and (2) correction_factors

    def calculate_engine_age(self):
        """Calculating the construction year of the engine, dependend on a Weibull function with
        shape factor 'k', and scale factor 'lmb', which are determined by the weight class L_w"""

        # Determining which shape and scale factor to use, based on the weight class L_w = L1, L2 or L3
        if self.L_w == 1:  # Weight class L1
            self.k = 1.3
            self.lmb = 20.5
        if self.L_w == 2:  # Weight class L2
            self.k = 1.12
            self.lmb = 18.5
        if self.L_w == 3:  # Weight class L3
            self.k = 1.26
            self.lmb = 18.6

        # The age of the engine
        self.age = int(np.random.weibull(self.k) * self.lmb)

        # Current year (TO DO: fix hardcoded year)
        # self.year = datetime.date.year
        self.year = 2021

        # Construction year of the engine
        self.c_year = self.year - self.age

        print('The construction year of the engine is', self.c_year)
        return self.c_year

    def calculate_properties(self):
        """Calculate a number of basic vessel properties"""
        self.C_M = 1.006 - 0.0056 * self.C_b ** (-3.56)  # Midship section coefficient
        self.C_wp = (1 + 2 * self.C_b) / 3  # Waterplane coefficient
        self.C_p = self.C_b / self.C_M  # Prismatic coefficient

        self.delta = self.C_b * self.L * self.B * self.T  # Water displacement

        self.lcb = -13.5 + 19.4 * self.C_p  # longitudinal center of buoyancy
        self.L_R = self.L * (1 - self.C_p + (0.06 * self.C_p * self.lcb) / (
                    4 * self.C_p - 1))  # parameter reflecting the length of the run

        self.A_T = 0.2 * self.B * self.T  # transverse area of the transom

        # Total wet area
        self.S_T = self.L * (2 * self.T + self.B) * np.sqrt(self.C_M) * (
                    0.453 + 0.4425 * self.C_b - 0.2862 * self.C_M - 0.003467 * (
                        self.B / self.T) + 0.3696 * self.C_wp)  # + 2.38 * (self.A_BT / self.C_b)

        self.S_APP = 0.05 * self.S_T  # Wet area of appendages
        self.S_B = self.L * self.B  # Area of flat bottom

        self.D_s = 0.7 * self.T  # Diameter of the screw

    def calculate_frictional_resistance(self, V_0, h):
        """1) Frictional resistance

        - 1st resistance component defined by Holtrop and Mennen (1982)
        - A modification to the original friction line is applied, based on literature of Zeng (2018), to account for shallow water effects """

        self.R_e = V_0 * self.L / self.nu  # Reynolds number
        self.D = h - self.T  # distance from bottom ship to the bottom of the fairway

        # Friction coefficient in deep water
        self.Cf_0 = 0.075 / ((np.log10(self.R_e) - 2) ** 2)

        # Friction coefficient proposed, taking into account shallow water effects
        self.Cf_proposed = (0.08169 / ((np.log10(self.R_e) - 1.717) ** 2)) * (
                    1 + (0.003998 / (np.log10(self.R_e) - 4.393)) * (self.D / self.L) ** (-1.083))

        # 'a' is the coefficient needed to calculate the Katsui friction coefficient
        self.a = 0.042612 * np.log10(self.R_e) + 0.56725
        self.Cf_katsui = 0.0066577 / ((np.log10(self.R_e) - 4.3762) ** self.a)

        # The average velocity underneath the ship, taking into account the shallow water effect

        if h / self.T <= 4:
            self.V_B = 0.4277 * V_0 * np.exp((h / self.T) ** (-0.07625))
        else:
            self.V_B = V_0

        # cf_proposed cannot be applied directly, since a vessel also has non-horizontal wet surfaces that have to be taken
        # into account. Therefore, the following formula for the final friction coefficient 'C_f' is defined:
        self.C_f = self.Cf_0 + (self.Cf_proposed - self.Cf_katsui) * (self.S_B / self.S_T) * (self.V_B / V_0) ** 2

        # The total frictional resistance R_f [kN]:
        self.R_f = (self.C_f * 0.5 * self.rho * (V_0 ** 2) * self.S_T) / 1000

    def calculate_viscous_resistance(self):
        """2) Viscous resistance

        - 2nd resistance component defined by Holtrop and Mennen (1982)
        - Form factor (1 + k1) has to be multiplied by the frictional resistance R_f, to account for the effect of viscosity"""

        # c_14 accounts for the specific shape of the afterbody
        self.c_14 = 1 + 0.0011 * self.c_stern

        # the form factor (1+k1) describes the viscous resistance
        self.one_k1 = 0.93 + 0.487 * self.c_14 * ((self.B / self.L) ** 1.068) * ((self.T / self.L) ** 0.461) * (
                    (self.L / self.L_R) ** 0.122) * (((self.L ** 3) / self.delta) ** 0.365) * (
                                  (1 - self.C_p) ** (-0.604))

    def calculate_appendage_resistance(self, V_0):
        """3) Appendage resistance

        - 3rd resistance component defined by Holtrop and Mennen (1982)
        - Appendages (like a rudder, shafts, skeg) result in additional frictional resistance"""

        # Frictional resistance resulting from wetted area of appendages: R_APP [kN]
        self.R_APP = (0.5 * self.rho * (V_0 ** 2) * self.S_APP * self.one_k2 * self.C_f) / 1000

    def karpov(self, V_0, h):
        """Intermediate calculation: Karpov

        - The Karpov method computes a velocity correction that accounts for limited water depth (corrected velocity V2)
        - V2 has to be implemented in the wave resistance and the residual resistance terms"""

        # The Froude number used in the Karpov method is the depth related froude number F_nh

        # The different alpha** curves are determined with a sixth power polynomial approximation in Excel
        # A distinction is made between different ranges of Froude numbers, because this resulted in a better approximation of the curve
        self.F_nh = V_0 / np.sqrt(self.g * h)

        if self.F_nh <= 0.4:

            if 0 <= h / self.T < 1.75:
                self.alpha_xx = (-4 * 10 ** (
                    -12)) * self.F_nh ** 3 - 0.2143 * self.F_nh ** 2 - 0.0643 * self.F_nh + 0.9997
            if 1.75 <= h / self.T < 2.25:
                self.alpha_xx = -0.8333 * self.F_nh ** 3 + 0.25 * self.F_nh ** 2 - 0.0167 * self.F_nh + 1
            if 2.25 <= h / self.T < 2.75:
                self.alpha_xx = -1.25 * self.F_nh ** 4 + 0.5833 * self.F_nh ** 3 - 0.0375 * self.F_nh ** 2 - 0.0108 * self.F_nh + 1
            if h / self.T >= 2.75:
                self.alpha_xx = 1

        if self.F_nh > 0.4:
            if 0 <= h / self.T < 1.75:
                self.alpha_xx = -0.9274 * self.F_nh ** 6 + 9.5953 * self.F_nh ** 5 - 37.197 * self.F_nh ** 4 + 69.666 * self.F_nh ** 3 - 65.391 * self.F_nh ** 2 + 28.025 * self.F_nh - 3.4143
            if 1.75 <= h / self.T < 2.25:
                self.alpha_xx = 2.2152 * self.F_nh ** 6 - 11.852 * self.F_nh ** 5 + 21.499 * self.F_nh ** 4 - 12.174 * self.F_nh ** 3 - 4.7873 * self.F_nh ** 2 + 5.8662 * self.F_nh - 0.2652
            if 2.25 <= h / self.T < 2.75:
                self.alpha_xx = 1.2205 * self.F_nh ** 6 - 5.4999 * self.F_nh ** 5 + 5.7966 * self.F_nh ** 4 + 6.6491 * self.F_nh ** 3 - 16.123 * self.F_nh ** 2 + 9.2016 * self.F_nh - 0.6342
            if 2.75 <= h / self.T < 3.25:
                self.alpha_xx = -0.4085 * self.F_nh ** 6 + 4.534 * self.F_nh ** 5 - 18.443 * self.F_nh ** 4 + 35.744 * self.F_nh ** 3 - 34.381 * self.F_nh ** 2 + 15.042 * self.F_nh - 1.3807
            if 3.25 <= h / self.T < 3.75:
                self.alpha_xx = 0.4078 * self.F_nh ** 6 - 0.919 * self.F_nh ** 5 - 3.8292 * self.F_nh ** 4 + 15.738 * self.F_nh ** 3 - 19.766 * self.F_nh ** 2 + 9.7466 * self.F_nh - 0.6409
            if 3.75 <= h / self.T < 4.5:
                self.alpha_xx = 0.3067 * self.F_nh ** 6 - 0.3404 * self.F_nh ** 5 - 5.0511 * self.F_nh ** 4 + 16.892 * self.F_nh ** 3 - 20.265 * self.F_nh ** 2 + 9.9002 * self.F_nh - 0.6712
            if 4.5 <= h / self.T < 5.5:
                self.alpha_xx = 0.3212 * self.F_nh ** 6 - 0.3559 * self.F_nh ** 5 - 5.1056 * self.F_nh ** 4 + 16.926 * self.F_nh ** 3 - 20.253 * self.F_nh ** 2 + 10.013 * self.F_nh - 0.7196
            if 5.5 <= h / self.T < 6.5:
                self.alpha_xx = 0.9252 * self.F_nh ** 6 - 4.2574 * self.F_nh ** 5 + 5.0363 * self.F_nh ** 4 + 3.3282 * self.F_nh ** 3 - 10.367 * self.F_nh ** 2 + 6.3993 * self.F_nh - 0.2074
            if 6.5 <= h / self.T < 7.5:
                self.alpha_xx = 0.8442 * self.F_nh ** 6 - 4.0261 * self.F_nh ** 5 + 5.313 * self.F_nh ** 4 + 1.6442 * self.F_nh ** 3 - 8.1848 * self.F_nh ** 2 + 5.3209 * self.F_nh - 0.0267
            if 7.5 <= h / self.T < 8.5:
                self.alpha_xx = 0.1211 * self.F_nh ** 6 + 0.628 * self.F_nh ** 5 - 6.5106 * self.F_nh ** 4 + 16.7 * self.F_nh ** 3 - 18.267 * self.F_nh ** 2 + 8.7077 * self.F_nh - 0.4745

            if 8.5 <= h / self.T < 9.5:
                if self.F_nh < 0.6:
                    self.alpha_xx = 1
                if self.F_nh >= 0.6:
                    self.alpha_xx = -6.4069 * self.F_nh ** 6 + 47.308 * self.F_nh ** 5 - 141.93 * self.F_nh ** 4 + 220.23 * self.F_nh ** 3 - 185.05 * self.F_nh ** 2 + 79.25 * self.F_nh - 12.484
            if h / self.T >= 9.5:
                if self.F_nh < 0.6:
                    self.alpha_xx = 1
                if self.F_nh >= 0.6:
                    self.alpha_xx = -6.0727 * self.F_nh ** 6 + 44.97 * self.F_nh ** 5 - 135.21 * self.F_nh ** 4 + 210.13 * self.F_nh ** 3 - 176.72 * self.F_nh ** 2 + 75.728 * self.F_nh - 11.893

        self.V_2 = V_0 / self.alpha_xx

    def calculate_wave_resistance(self, V_0, h):
        """4) Wave resistance

        - 4th resistance component defined by Holtrop and Mennen (1982)
        - When the speed or the vessel size increases, the wave making resistance increases
        - In shallow water, the wave resistance shows an asymptotical behaviour by reaching the critical speed"""

        self.karpov(V_0, h)

        self.F_n = self.V_2 / np.sqrt(self.g * self.L)  # Froude number

        # parameter c_7 is determined by the B/L ratio
        if self.B / self.L < 0.11:
            self.c_7 = 0.229577 * (self.B / self.L) ** 0.33333
        if self.B / self.L > 0.25:
            self.c_7 = 0.5 - 0.0625 * (self.L / self.B)
        else:
            self.c_7 = self.B / self.L

        # half angle of entrance in degrees
        self.i_E = 1 + 89 * np.exp(-((self.L / self.B) ** 0.80856) * ((1 - self.C_wp) ** 0.30484) * (
                    (1 - self.C_p - 0.0225 * self.lcb) ** 0.6367) * ((self.L_R / self.B) ** 0.34574) * (
                                               (100 * self.delta / (self.L ** 3)) ** 0.16302))

        self.c_1 = 2223105 * (self.c_7 ** 3.78613) * ((self.T / self.B) ** 1.07961) * (90 - self.i_E) ** (-1.37165)
        self.c_2 = 1  # accounts for the effect of the bulbous bow, which is not present at inland ships
        self.c_5 = 1 - (0.8 * self.A_T) / (
                    self.B * self.T * self.C_M)  # influence of the transom stern on the wave resistance

        # parameter c_15 depoends on the ratio L^3 / delta
        if (self.L ** 3) / self.delta < 512:
            self.c_15 = -1.69385
        if (self.L ** 3) / self.delta > 1727:
            self.c_15 = 0
        else:
            self.c_15 = -1.69385 + (self.L / (self.delta ** (1 / 3)) - 8) / 2.36

        # parameter c_16 depends on C_p
        if self.C_p < 0.8:
            self.c_16 = 8.07981 * self.C_p - 13.8673 * (self.C_p ** 2) + 6.984388 * (self.C_p ** 3)
        else:
            self.c_16 = 1.73014 - 0.7067

        self.m_1 = 0.0140407 * (self.L / self.T) - 1.75254 * ((self.delta) ** (1 / 3) / self.L) - 4.79323 * (
                    self.B / self.L) - self.c_16

        self.m_4 = 0.4 * self.c_15 * np.exp(-0.034 * (self.F_n ** (-3.29)))

        if self.L / self.B < 12:
            self.lmbda = 1.446 * self.C_p - 0.03 * (self.L / self.B)
        else:
            self.lmbda = 1.446 * self.C_p - 0.036

        # parameters needed for RW_2
        self.c_17 = 6919.3 * (self.C_M ** (-1.3346)) * ((self.delta / (self.L ** 3)) ** 2.00977) * (
                    (self.L / self.B - 2) ** 1.40692)
        self.m_3 = -7.2035 * ((self.B / self.L) ** 0.326869) * ((self.T / self.B) ** 0.605375)

        ######### When Fn < 0.4
        self.RW_1 = self.c_1 * self.c_2 * self.c_5 * self.delta * self.rho * self.g * np.exp(
            self.m_1 * (self.F_n ** (-0.9)) + self.m_4 * np.cos(self.lmbda * (self.F_n ** (-2))))

        ######## When Fn > 0.5
        self.RW_2 = self.c_17 * self.c_2 * self.c_5 * self.delta * self.rho * self.g * np.exp(
            self.m_3 * (self.F_n ** (-0.9)) + self.m_4 * np.cos(self.lmbda * (self.F_n ** (-2))))

        if self.F_n < 0.4:
            self.R_W = self.RW_1 / 1000  # kN
        if self.F_n > 0.55:
            self.R_W = self.RW_2 / 1000  # kN
        else:
            self.R_W = (self.RW_1 + ((10 * self.F_n - 4) * (self.RW_2 - self.RW_1)) / 1.5) / 1000  # kN

    def calculate_residual_resistance(self, V_0, h):
        """5) Residual resistance terms

        - Holtrop and Mennen (1982) defined three residual resistance terms:
        - 1) Resistance due to the bulbous bow (not incorporated since inland ships in general don't have a bulb)
        - 2) Resistance due to immersed transom
        - 3) Resistance due to model-ship correlation """

        self.karpov(V_0, h)

        # Resistance due to immersed transom: R_TR [kN]
        self.F_nt = self.V_2 / np.sqrt(
            2 * self.g * self.A_T / (self.B + self.B * self.C_wp))  # Froude number based on transom immersion
        self.c_6 = 0.2 * (1 - 0.2 * self.F_nt)  # Assuming F_nt < 5, this is the expression for coefficient c_6

        self.R_TR = (0.5 * self.rho * (self.V_2 ** 2) * self.A_T * self.c_6) / 1000

        # Model-ship correlation resistance: R_A [kN]

        if self.T / self.L < 0.04:
            self.c_4 = self.T / self.L
        else:
            self.c_4 = 0.04
        self.c_2 = 1

        self.C_A = 0.006 * (self.L + 100) ** (-0.16) - 0.00205 + 0.003 * np.sqrt(self.L / 7.5) * (
                    self.C_b ** 4) * self.c_2 * (0.04 - self.c_4)

        ####### Holtrop and Mennen in the document of Sarris, 2003 #######
        self.R_A = (0.5 * self.rho * (self.V_2 ** 2) * self.S_T * self.C_A) / 1000  # kW

    def calculate_total_resistance(self, V_0, h):
        """Total resistance:

        The total resistance is the sum of all resistance components (Holtrop and Mennen, 1982) """

        self.calculate_properties()
        self.calculate_frictional_resistance(V_0, h)
        self.calculate_viscous_resistance()
        self.calculate_appendage_resistance(V_0)
        self.calculate_wave_resistance(V_0, h)
        self.calculate_residual_resistance(V_0, h)

        # The total resistance R_tot [kN] = R_f * (1+k1) + R_APP + R_W + R_TR + R_A
        self.R_tot = self.R_f * self.one_k1 + self.R_APP + self.R_W + self.R_TR + self.R_A

    def calculate_total_power_required(self):
        """Total required power:

        - The total required power is the sum of the power for systems on board (P_hotel) + power required for propulsion (P_BHP)
        - The P_BHP depends on the calculated resistance"""

        # ---- Required power for systems on board
        self.P_hotel = 0.05 * self.P_installed

        # ---- Required power for propulsion

        # Effective Horse Power (EHP)
        self.P_EHP = self.V_B * self.R_tot

        # Calculation hull efficiency
        dw = np.zeros(101)  # velocity correction coefficient
        counter = 0

        if self.F_n < 0.2:
            self.dw = 0
        else:
            self.dw = 0.1

        self.w = 0.11 * (0.16 / self.x) * self.C_b * np.sqrt(
            (self.delta ** (1 / 3)) / self.D_s) - self.dw  # wake fraction 'w'

        if self.x == 1:
            self.t = 0.6 * self.w * (1 + 0.67 * self.w)  # thrust deduction factor 't'
        else:
            self.t = 0.8 * self.w * (1 + 0.25 * self.w)

        self.eta_h = (1 - self.t) / (1 - self.w)  # hull efficiency eta_h

        # Delivered Horse Power (DHP)

        self.P_DHP = self.P_EHP / (self.eta_0 * self.eta_r * self.eta_h)

        # Brake Horse Power (BHP)
        self.P_BHP = self.P_DHP / (self.eta_t * self.eta_g)

        self.P_tot = self.P_hotel + self.P_BHP

        # Partial engine load (P_partial): needed in the 'Emission calculations'
        if self.P_tot > self.P_installed:
            self.P_partial = 1
        else:
            self.P_partial = self.P_tot / self.P_installed

        print('The total power required is', self.P_tot, 'kW')
        print('The partial load is', self.P_partial, 'kW')

    def emission_factors_general(self):
        """General emission factors:

        This function computes general emission factors, based on construction year of the engine.
        - Based on literature TNO (2019)

        Please note: later on a correction factor has to be applied to get the total emission factor"""

        # The general emission factors of CO2, PM10 and NOX are based on the construction year of the engine

        if self.c_year < 1974:
            self.EM_CO2 = 756
            self.EM_PM10 = 0.6
            self.EM_NOX = 10.8
        if 1975 <= self.c_year <= 1979:
            self.EM_CO2 = 730
            self.EM_PM10 = 0.6
            self.EM_NOX = 10.6
        if 1980 <= self.c_year <= 1984:
            self.EM_CO2 = 714
            self.EM_PM10 = 0.6
            self.EM_NOX = 10.4
        if 1985 <= self.c_year <= 1989:
            self.EM_CO2 = 698
            self.EM_PM10 = 0.5
            self.EM_NOX = 10.1
        if 1990 <= self.c_year <= 1994:
            self.EM_CO2 = 698
            self.EM_PM10 = 0.4
            self.EM_NOX = 10.1
        if 1995 <= self.c_year <= 2002:
            self.EM_CO2 = 650
            self.EM_PM10 = 0.3
            self.EM_NOX = 9.4
        if 2003 <= self.c_year <= 2007:
            self.EM_CO2 = 635
            self.EM_PM10 = 0.3
            self.EM_NOX = 9.2
        if 2008 <= self.c_year <= 2019:
            self.EM_CO2 = 635
            self.EM_PM10 = 0.2
            self.EM_NOX = 7
        if self.c_year > 2019:
            if self.L_w == 1:
                self.EM_CO2 = 650
                self.EM_PM10 = 0.1
                self.EM_NOX = 2.9
            else:
                self.EM_CO2 = 603
                self.EM_PM10 = 0.015
                self.EM_NOX = 2.4

        print('The general emission factor of CO2 is', self.EM_CO2, 'g/kWh')
        print('The general emission factor of PM10 is', self.EM_PM10, 'g/kWh')
        print('The general emission factor CO2 is', self.EM_NOX, 'g/kWh')

    def correction_factors(self):
        """Correction factors:

        - The correction factors have to be multiplied by the general emission factors, to get the total emission factors
        - The correction factor takes into account the effect of the partial engine load
        - When the partial engine load is low, the correction factors are higher (engine is less efficient)
        - Based on literature TNO (2019)"""

        self.calculate_total_power_required()  # You need the P_partial values

        # Import the correction factors table
        self.corf = pd.read_excel(r'correctionfactors.xlsx')

        for i in range(20):
            # If the partial engine load is smaller or equal to 5%, the correction factors corresponding to P_partial = 5% are assigned.
            if self.P_partial <= self.corf.iloc[0, 0]:
                self.corf_CO2 = self.corf.iloc[0, 5]
                self.corf_PM10 = self.corf.iloc[0, 6]

                # The NOX correction factors are dependend on the construction year of the engine and the weight class
                if self.c_year < 2008:
                    self.corf_NOX = self.corf.iloc[0, 1]  # <= CCR-1 class
                if 2008 <= self.c_year <= 2019:
                    self.corf_NOX = self.corf.iloc[0, 2]  # CCR-2 / Stage IIIa
                if self.c_year > 2019:
                    if self.L_w == 1:  #
                        self.corf_NOX = self.corf.iloc[
                            0, 3]  # Stage V: IWP/IWA-v/c-3 class (vessels with P <300 kW: assumed to be weight class L1)
                    else:
                        self.corf_NOX = self.corf.iloc[
                            0, 4]  # Stage V:IWP/IWA-v/c-4 class (vessels with P >300 kw: assumed to be weight class L2-L3)

            # If the partial engine load is greater than 5%:
            # It is determined inbetween which two percentages in the table the partial engine load lies
            # The correction factor is determined by means of linear interpolation

            elif self.corf.iloc[i, 0] < self.P_partial <= self.corf.iloc[i + 1, 0]:
                self.corf_CO2 = ((self.P_partial - self.corf.iloc[i, 0]) * (
                            self.corf.iloc[i + 1, 5] - self.corf.iloc[i, 5])) / (
                                            self.corf.iloc[i + 1, 0] - self.corf.iloc[i, 0]) + self.corf.iloc[i, 5]
                self.corf_PM10 = ((self.P_partial - self.corf.iloc[i, 0]) * (
                            self.corf.iloc[i + 1, 6] - self.corf.iloc[i, 6])) / (
                                             self.corf.iloc[i + 1, 0] - self.corf.iloc[i, 0]) + self.corf.iloc[i, 6]

                if self.c_year < 2008:
                    self.corf_NOX = ((self.P_partial - self.corf.iloc[i, 0]) * (
                                self.corf.iloc[i + 1, 1] - self.corf.iloc[i, 1])) / (
                                                self.corf.iloc[i + 1, 0] - self.corf.iloc[i, 0]) + self.corf.iloc[i, 1]
                if 2008 <= self.c_year <= 2019:
                    self.corf_NOX = ((self.P_partial - self.corf.iloc[i, 0]) * (
                                self.corf.iloc[i + 1, 2] - self.corf.iloc[i, 2])) / (
                                                self.corf.iloc[i + 1, 0] - self.corf.iloc[i, 0]) + self.corf.iloc[i, 2]
                if self.c_year > 2019:
                    if self.L_w == 1:
                        self.corf_NOX = ((self.P_partial - self.corf.iloc[i, 0]) * (
                                    self.corf.iloc[i + 1, 3] - self.corf.iloc[i, 3])) / (
                                                    self.corf.iloc[i + 1, 0] - self.corf.iloc[i, 0]) + self.corf.iloc[
                                            i, 3]
                    else:
                        self.corf_NOX = ((self.P_partial - self.corf.iloc[i, 0]) * (
                                    self.corf.iloc[i + 1, 4] - self.corf.iloc[i, 4])) / (
                                                    self.corf.iloc[i + 1, 0] - self.corf.iloc[i, 0]) + self.corf.iloc[
                                            i, 4]

            # If the partial engine load is => 100%, the correction factors corresponding to P_partial = 100% are assigned.
            elif self.P_partial >= self.corf.iloc[19, 0]:
                self.corf_CO2 = self.corf.iloc[19, 5]
                self.corf_PM10 = self.corf.iloc[19, 6]

                # The NOX correction factors are dependend on the construction year of the engine and the weight class
                if self.c_year < 2008:
                    self.corf_NOX = self.corf.iloc[19, 1]  # <= CCR-1 class
                if 2008 <= self.c_year <= 2019:
                    self.corf_NOX = self.corf.iloc[19, 2]  # CCR-2 / Stage IIIa
                if self.c_year > 2019:
                    if self.L_w == 1:  #
                        self.corf_NOX = self.corf.iloc[
                            19, 3]  # Stage V: IWP/IWA-v/c-3 class (vessels with P <300 kW: assumed to be weight class L1)
                    else:
                        self.corf_NOX = self.corf.iloc[
                            19, 4]  # Stage V:IWP/IWA-v/c-4 class (vessels with P >300 kw: assumed to be weight class L2-L3)

        print('Correction factor of CO2 is', self.corf_CO2)
        print('Correction factor of PM10 is', self.corf_PM10)
        print('Correction factor of NOX is', self.corf_NOX)

    def calculate_emission_factors_total(self):
        """Total emission factors:

        - The total emission factors can be computed by multiplying the general emission factor by the correction factor"""

        print('The construction year of the engine is', self.c_year)
        # self.calculate_engine_age() #You need the values of c_year

        self.emission_factors_general()  # You need the values of the general emission factors of CO2, PM10, NOX
        self.correction_factors()  # You need the correction factors of CO2, PM10, NOX

        # The total emission factor is calculated by multiplying the general emission factor (EM_CO2 / EM_PM10 / EM_NOX)
        # By the correction factor (corf_CO2 / corf_PM10 / corf_NOX)

        self.Emf_CO2 = self.EM_CO2 * self.corf_CO2
        self.Emf_PM10 = self.EM_PM10 * self.corf_PM10
        self.Emf_NOX = self.EM_NOX * self.corf_NOX

        print('The total emission factor of CO2 is', self.Emf_CO2, 'g/kWh')
        print('The total emission factor of PM10 is', self.Emf_PM10, 'g/kWh')
        print('The total emission factor CO2 is', self.Emf_NOX, 'g/kWh')