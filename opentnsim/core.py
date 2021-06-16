"""Main module."""

# package(s) related to time, space and id
import json
import logging
import uuid

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy
import random
import networkx as nx
import numpy as np
import math
import pandas as pd

# spatial libraries
import pyproj
import shapely.geometry

# additional packages
import datetime, time

logger = logging.getLogger(__name__)


class SimpyObject:
    """General object which can be extended by any class requiring a simpy environment
    env: a simpy Environment
    """

    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env


class HasResource(SimpyObject):
    """Something that has a resource limitation, a resource request must be granted before the object can be used.

    nr_resources: nr of requests that can be handled simultaneously"""

    def __init__(self, nr_resources=1, priority=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.resource = (
            simpy.PriorityResource(self.env, capacity=nr_resources)
            if priority
            else simpy.Resource(self.env, capacity=nr_resources)
        )


class Identifiable:
    """Mixin class: Something that has a name and id

    name: a name
    id: a unique id generated with uuid"""

    def __init__(self, name, id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.name = name
        # generate some id, in this case based on m
        self.id = id if id else str(uuid.uuid1())


class Locatable:
    """Mixin class: Something with a geometry (geojson format)

    geometry: can be a point as well as a polygon"""

    def __init__(self, geometry, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.geometry = geometry
        self.node = None


class Neighbours:
    """Can be added to a locatable object (list)
    travel_to: list of locatables to which can be travelled"""

    def ___init(self, travel_to, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.neighbours = travel_to

class HasLength(SimpyObject): #used by IsLock and IsLineUpArea to regulate number of vessels in each lock cycle and calculate repsective position in lock chamber/line-up area
    """Mixin class: Something with a storage capacity
    capacity: amount the container can hold
    level: amount the container holds initially
    total_requested: a counter that helps to prevent over requesting"""

    def __init__(self, length, remaining_length=0, total_requested=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.length = simpy.Container(self.env, capacity = length, init=remaining_length)
        self.pos_length = simpy.Container(self.env, capacity = length, init=remaining_length)

class HasContainer(SimpyObject):
    """Mixin class: Something with a storage capacity
    capacity: amount the container can hold
    level: amount the container holds initially
    container: a simpy object that can hold stuff
    total_requested: a counter that helps to prevent over requesting"""

    def __init__(self, capacity, level=0, total_requested=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.container = simpy.Container(self.env, capacity, init=level)
        self.total_requested = total_requested

    @property
    def is_loaded(self):
        return True if self.container.level > 0 else False

    @property
    def filling_degree(self):
        return self.container.level / self.container.capacity


class Log(SimpyObject):
    """Mixin class: Something that has logging capability

    log: log message [format: 'start activity' or 'stop activity']
    t: timestamp
    value: a value can be logged as well
    geometry: value from locatable (lat, lon)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.log = {"Message": [], "Timestamp": [], "Value": [], "Geometry": []}

    def log_entry(self, log, t, value, geometry_log):
        """Log"""
        self.log["Message"].append(log)
        self.log["Timestamp"].append(datetime.datetime.fromtimestamp(t))
        self.log["Value"].append(value)
        self.log["Geometry"].append(geometry_log)

    def get_log_as_json(self):
        json = []
        for msg, t, value, geometry_log in zip(
            self.log["Message"],
            self.log["Timestamp"],
            self.log["Value"],
            self.log["Geometry"],
        ):
            json.append(
                dict(message=msg, time=t, value=value, geometry_log=geometry_log)
            )
        return json

class ExtraMetadata:
    """store all leftover keyword arguments as metadata property (use as last mixin)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        # store all other properties as metadata
        self.metadata = kwargs

class VesselProperties:
    """Mixin class: Something that has vessel properties
    This mixin is updated to better accommodate the ConsumesEnergy mixin

    type: can contain info on vessel type (avv class, cemt_class or other)
    B: vessel width
    L: vessel length
    H_e: vessel height unloaded
    H_f: vessel height loaded
    T_e: draught unloaded
    T_f: draught loaded

    Add information on possible restrictions to the vessels, i.e. height, width, etc.
    """

    def __init__(
            self,
            type,
            B,
            L,
            H_e,
            H_f,
            T_e,
            T_f,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        """Initialization"""
        self.type = type

        self.B = B
        self.L = L

        self.H_e = H_e
        self.H_f = H_e

        self.T_e = T_e
        self.T_f = T_f

    @property
    def H(self):
        """ Calculate current height based on filling degree """

        return (
                self.filling_degree * (self.H_f - self.H_e)
                + self.H_e
        )

    @property
    def T(self):
        """ Calculate current draught based on filling degree

        Here we should implement the rules from Van Dorsser et al
        https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships
        """

        return (
                self.filling_degree * (self.T_f - self.T_e)
                + self.T_e
        )

    def get_route(
            self,
            origin,
            destination,
            graph=None,
            minWidth=None,
            minHeight=None,
            minDepth=None,
            randomSeed=4,
    ):
        """ Calculate a path based on vessel restrictions """

        graph = graph if graph else self.env.FG
        minWidth = minWidth if minWidth else 1.1 * self.B
        minHeight = minWidth if minHeight else 1.1 * self.H
        minDepth = minWidth if minDepth else 1.1 * self.T

        # Check if information on restrictions is added to the edges
        random.seed(randomSeed)
        edge = random.choice(list(graph.edges(data=True)))
        edge_attrs = list(edge[2].keys())

        # IMPROVE THIS TO CHECK ALL EDGES AND COMBINATIONS OF RESTRICTIONS

        if all(item in edge_attrs for item in ["Width", "Height", "Depth"]):
            edges = []
            nodes = []

            for edge in graph.edges(data=True):
                if (
                        edge[2]["Width"] >= minWidth
                        and edge[2]["Height"] >= minHeight
                        and edge[2]["Depth"] >= minDepth
                ):
                    edges.append(edge)

                    nodes.append(graph.nodes[edge[0]])
                    nodes.append(graph.nodes[edge[1]])

            subGraph = graph.__class__()

            for node in nodes:
                subGraph.add_node(
                    node["name"],
                    name=node["name"],
                    geometry=node["geometry"],
                    position=(node["geometry"].x, node["geometry"].y),
                )

            for edge in edges:
                subGraph.add_edge(edge[0], edge[1], attr_dict=edge[2])

            try:
                return nx.dijkstra_path(subGraph, origin, destination)
                # return nx.bidirectional_dijkstra(subGraph, origin, destination)
            except:
                raise ValueError(
                    "No path was found with the given boundary conditions."
                )

        # If not, return shortest path
        else:
            return nx.dijkstra_path(graph, origin, destination)
    
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


class Routeable:
    """Mixin class: Something with a route (networkx format)

    route: a networkx path"""

    def __init__(self, route, complete_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.route = route
        self.complete_path = complete_path

class IsLockWaitingArea(HasResource, Identifiable, Log):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(
        self,
        node, #a string which indicates the location of the start of the waiting area
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        """Initialization"""

        waiting_area_resources = 100
        self.waiting_area = {
            node: simpy.PriorityResource(self.env, capacity=waiting_area_resources),
        }

class IsLockLineUpArea(HasResource, HasLength, Identifiable, Log):
    """Mixin class: Something has line-up area object properties as part of the lock complex [in SI-units]:
            creates a line-up area with the following resources:
                - enter_line_up_area: resource used when entering the line-up area (assures one-by-one entry of the line-up area by vessels)
                - line_up_area: resource with unlimited capacity used to formally request access to the line-up area
                - converting_while_in_line_up_area: resource used when requesting for an empty conversion of the lock chamber
                - pass_line_up_area: resource used to pass the second encountered line-up area"""

    def __init__(
        self,
        node, #a string which indicates the location of the start of the line-up area
        lineup_length, #a float which contains the length of the line-up area
        *args,
        **kwargs
    ):
        super().__init__(length = lineup_length, remaining_length = lineup_length, *args, **kwargs)
        
        """Initialization"""
        # Lay-Out
        self.enter_line_up_area = { #used to regulate one by one entering of line-up area, so capacity must be 1
            node: simpy.PriorityResource(self.env, capacity=1),
        }

        self.line_up_area = { #line-up area itself, infinite capacity, as this is regulated by the HasLength, so capacity = inf
            node: simpy.PriorityResource(self.env, capacity=100),
        }

        self.converting_while_in_line_up_area = { #used to minimize the number of empty convertion requests by one by multiple waiting vessels, so capacity must be 1
            node: simpy.PriorityResource(self.env, capacity=1),
        }

        self.pass_line_up_area = { #used to prevent vessel from entering the lock before all previously locked vessels have passed the line-up area one by one, so capacity must be 1
            node: simpy.PriorityResource(self.env, capacity=1),
        }
       
class IsLock(HasResource, HasLength, Identifiable, Log):
    """Mixin class: Something which has lock chamber object properties as part of a lock complex [in SI-units] """

    def __init__(
        self,
        node_1, #a string which indicates the location of the first pair of lock doors
        node_2, #a string which indicates the center of the lock chamber
        node_3, #a string which indicates the location of the second pair of lock doors
        lock_length, #a float which contains the length of the lock chamber
        lock_width, #a float which contains the width of the lock chamber
        lock_depth, #a float which contains the depth of the lock chamber
        doors_open, #a float which contains the time it takes to open the doors
        doors_close, #a float which contains the time it takes to close the doors
        wlev_dif, #a float or list of floats which resembles the water level difference over the lock
        disch_coeff, #a float which contains the discharge coefficient of filling system
        opening_area, #a float which contains the cross-sectional area of filling system
        opening_depth, #a float which contains the depth at which filling system is located
        simulation_start, #a datetime which contains the simulation start time
        operating_time,
        grav_acc = 9.81, #a float which contains the gravitational acceleration
        *args,
        **kwargs
    ):

        """Initialization"""
        # Properties
        self.lock_length = lock_length
        self.lock_width = lock_width
        self.lock_depth = lock_depth
        self.wlev_dif = wlev_dif
        self.disch_coeff = disch_coeff
        self.grav_acc = grav_acc
        self.opening_area = opening_area
        self.opening_depth = opening_depth
        self.operating_time = operating_time
        self.simulation_start = simulation_start.timestamp()
        
        super().__init__(length = lock_length, remaining_length = lock_length, *args, **kwargs)
        
        self.doors_1 = {
            node_1: simpy.PriorityResource(self.env, capacity = 1), #Only one ship can pass at a time: capacity = 1, request can have priority
        }
        self.doors_2 = {
            node_3: simpy.PriorityResource(self.env, capacity = 1), #Only one ship can pass at a time: capacity = 1, request can have priority
        }

        # Operating
        self.doors_open = doors_open
        self.doors_close = doors_close

        # Water level
        assert node_1 != node_3

        self.node_1 = node_1
        self.node_3 = node_3
        self.water_level = random.choice([node_1, node_3])
        
    def operation_time(self, environment):
        """ Function which calculates the operation time:
                based on the constant or nearest in the signal of the water level difference
                
            Input:
                - environment: see init function"""
        
        if type(self.wlev_dif) == list: #picks the wlev_dif from measurement signal closest to the discrete time
            operating_time = (2*self.lock_width*self.lock_length*abs(self.wlev_dif[1][np.abs(self.wlev_dif[0]-(environment.now-self.simulation_start)).argmin()]))/(self.disch_coeff*self.opening_area*math.sqrt(2*self.grav_acc*self.opening_depth))

        elif type(self.wlev_dif) == float or type(self.wlev_dif) == int: #constant water level difference
            operating_time = (2*self.lock_width*self.lock_length*abs(self.wlev_dif))/(self.disch_coeff*self.opening_area*math.sqrt(2*self.grav_acc*self.opening_depth))

        return operating_time

    def convert_chamber(self, environment, new_level, number_of_vessels):
        """ Function which converts the lock chamber and logs this event.
        
            Input:
                - environment: see init function
                - new_level: a string which represents the node and indicates the side at which the lock is currently levelled
                - number_of_vessels: the total number of vessels which are levelled simultaneously"""

        # Close the doors
        self.log_entry("Lock doors closing start", environment.now, number_of_vessels, self.water_level)
        yield environment.timeout(self.doors_close)
        self.log_entry("Lock doors closing stop", environment.now, number_of_vessels, self.water_level)

        # Convert the chamber
        self.log_entry(
            "Lock chamber converting start", environment.now, number_of_vessels, self.water_level
        )

        # Water level will shift
        self.change_water_level(new_level)
        yield environment.timeout(self.operation_time(environment))
        self.log_entry(
            "Lock chamber converting stop", environment.now, number_of_vessels, self.water_level
        )
        # Open the doors
        self.log_entry("Lock doors opening start", environment.now, number_of_vessels, self.water_level)
        yield environment.timeout(self.doors_open)
        self.log_entry("Lock doors opening stop", environment.now, number_of_vessels, self.water_level)

    def change_water_level(self, side):
        """ Function which changes the water level in the lock chamber and priorities in queue """

        self.water_level = side

        for request in self.resource.queue:
            request.priority = -1 if request.priority == 0 else 0

            if request.priority == -1:
                self.resource.queue.insert(
                    0, self.resource.queue.pop(self.resource.queue.index(request))
                )
            else:
                self.resource.queue.insert(
                    -1, self.resource.queue.pop(self.resource.queue.index(request))
                )

class PassLock():
    """ Mixin class: a collection of functions which are used to pass a lock complex consisting of a waiting area, line-up areas, and lock chambers"""
    
    def approach_waiting_area(vessel,node_waiting_area):
        """ Processes vessels which are approaching a waiting area of a lock complex: 
                if the waiting area is full, vessels will be waiting outside the waiting area for a spot, otherwise if the vessel
                fits within the waiting area the vessels will proceed to the waiting area. 
        
            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_waiting_area: a string which includes the name of the node the waiting area is located in the network """
            
        #Imports the properties of the waiting area
        waiting_area = vessel.env.FG.nodes[node_waiting_area]["Waiting area"][0]
        
        #Identifies the index of the node of the waiting area within the route of the vessel
        index_node_waiting_area = vessel.route.index(node_waiting_area)
        
        #Checks whether the waiting area is the first encountered waiting area of the lock complex
        for node_lineup_area in vessel.route[index_node_waiting_area:]:
            if 'Line-up area' not in vessel.env.FG.nodes[node_lineup_area].keys():
                continue    
            
            #Imports the properties of the line-up areas
            lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
            for lineup_area in lineup_areas:
                if waiting_area.name.split('_')[0] != lineup_area.name.split('_')[0]:
                    continue
                
                #Determines the current time
                wait_for_waiting_area = vessel.env.now
                
                #Requests access to the waiting area
                vessel.access_waiting_area = waiting_area.waiting_area[node_waiting_area].request()
                yield vessel.access_waiting_area
    
                #Calculates and reports the waiting time for entering the waiting area
                if wait_for_waiting_area != vessel.env.now:
                    waiting = vessel.env.now - wait_for_waiting_area
                    vessel.log_entry("Waiting to enter waiting area start", wait_for_waiting_area, 0, 
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node_waiting_area)-1]],)
                    vessel.log_entry("Waiting to enter waiting area stop", vessel.env.now, waiting, 
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node_waiting_area)-1]],)
                break
            break
    
    def leave_waiting_area(vessel,node_waiting_area):
        """ Processes vessels which are waiting in the waiting area of the lock complex and requesting access to preceding the line-up area: 
                if there area multiple parallel lock chambers, the chamber with the least expected total waiting time is chosen,
                after which access is requested to enter the line-up area corresponding with the assigned lock chain series. 
                
            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_waiting_area: a string which includes the name of the node the waiting area is located in the network """
        
        #Imports the properties of the waiting area
        waiting_area = vessel.env.FG.nodes[node_waiting_area]["Waiting area"][0]
        
        #Identifies the index of the node of the waiting area within the route of the vessel
        index_node_waiting_area = vessel.route.index(node_waiting_area)
        
        #Checks whether the waiting area is the first encountered waiting area of the lock complex
        for node_lineup_area in vessel.route[index_node_waiting_area:]: 
            if 'Line-up area' not in vessel.env.FG.nodes[node_lineup_area].keys():
                continue 
            
            #Imports the properties of the line-up areas
            lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
            for lineup_area in lineup_areas:
                if waiting_area.name.split('_')[0] != lineup_area.name.split('_')[0]:
                    continue
                
                #Imports the location of the lock chamber of the lock complex
                for node_lock in vessel.route[index_node_waiting_area:]:
                    if 'Lock' not in vessel.env.FG.nodes[node_lock].keys():
                        continue
                    locks = vessel.env.FG.nodes[node_lock]["Lock"]
                    break
                
                def choose_lock_chamber(vessel,lock,lock_position,series_number,lineup_areas,lock_queue_length):
                    """ Assigns the lock chamber with the least expected total waiting time to the vessel in case of parallell lock chambers. The
                            expected total waiting time is calculated through quantifying the total length of the queued vessels. If a vessel does
                            not fit in a lockage, it will create a new lockage cycle by requesting the full length capacity of the line-up area. When
                            this request is granted, the vessel will immediately release the obsolete length, such that more vessels can go with the
                            next lockage.
                        This function is evaluated in the leave_waiting_area function.
                    
                        Input:
                            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                            - lock: an object within the network which is generated with the IsLock mixin class
                            - lock_position: a string which includes the name of the node the lock chamber is located in the network
                            - series_number: an integer number which indicates which lock series is evaluated (should not be greater than
                                             the maximum number of parallel lock chambers)
                            - lineup_areas: the collection of line-up areas in the node at which the line-up areas are located (the total
                                            amount of line-up areas should not exceed the amount of lock series)
                            - lock_queue_length: an empty list at which the resistance in total queue length (~ total waiting time) is
                                                 appended. """
                    
                    #Imports the properties of the evaluated line-up area
                    lineup_area = lineup_areas[series_number]
                    
                    #Assesses the total queue length within this lock series 
                    #- if the queue for the line-up area is empty, a name is set if the vessel fits in the lock chamber and line-up right away, otherwise the queue is calculated
                    if lineup_area.length.get_queue == []:
                        if (vessel.L < lock.length.level and vessel.L < lineup_area.length.level and lock.water_level == vessel.route[vessel.route.index(lock_position)-1]):
                            vessel.lock_name = lock.name
                        elif vessel.L < lineup_area.length.level:
                            lock_queue_length.append(lineup_area.length.level)
                        else:
                            lock_queue_length.append(lineup_area.length.capacity)
                    
                    #- else, if the vessel does not fit in the line-up area, the total length of the queued is calculated added with the full length capacity of the line-up area
                    else:               
                        line_up_queue_length = lineup_area.length.capacity
                        for q in range(len(lineup_area.length.get_queue)):
                            line_up_queue_length += lineup_area.length.get_queue[q].amount
                        lock_queue_length.append(line_up_queue_length)
                                                    
                def access_lineup_area(vessel,lineup_area):
                    """ Processes the request of vessels to access the line-up area by claiming a position (which equals the length of
                            the vessel) along its jetty.
                        This function is evaluated in the leave_waiting_area function
                        
                        Input:
                            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                            - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                                           assigned to the vessel as the lock series with the least expected total waiting time """
                    
                    def create_new_lock_cycle_and_request_access():
                        """ Creates a new locking cycle (wave of vessels assigned to same lockage) by requesting the full length
                                capacity of the line-up area, assigning the vessel to this request and returning the obsolete length
                                when the request is granted.
                            This function is used in the access_lineup_area function within the leave_waiting_area function 
                            
                            No input required """
                        
                        access_lineup_length = lineup_area.length.get(lineup_area.length.capacity)
                        lineup_area.length.get_queue[-1].length = vessel.L
                        yield access_lineup_length
                        lineup_area.length.put(lineup_area.length.capacity-vessel.L)
                    
                    def request_access_lock_cycle(total_length_waiting_vessels = 0):
                        """ Processes the request of a vessel to enter a lock cycle (wave of vessels assigned to same lockage), depending
                                on the governing conditions regarding the current situation in the line-up area.
                            This function is used in the access_lineup_area function within the leave_waiting_area function 
                            
                            No input required """
                               
                        #- If the line-up area has no queue, the vessel will access the lock cycle
                        if lineup_area.length.get_queue == []:                    
                            access_lineup_length = lineup_area.length.get(vessel.L)
                            
                        #Else, if there are already preceding vessels waiting in a queue, the vessel will request access to a lock cycle
                        else:
                            
                            #Determines the vessel which has started the last lock cycle
                            for q in reversed(range(len(lineup_area.length.get_queue))):
                                if lineup_area.length.get_queue[q].amount == lineup_area.length.capacity:
                                    break
                                
                            #Calculates the total length of vessels assigned to this lock cycle
                            for q2 in range(q,len(lineup_area.length.get_queue)):
                                total_length_waiting_vessels += lineup_area.length.get_queue[q2].length
                
                            #If the vessels does not fit in this lock cycle, it will start a new lock cycle
                            if vessel.L > lineup_area.length.capacity - total_length_waiting_vessels:  
                                yield from create_new_lock_cycle_and_request_access()                                                                 
                            
                            #Else, if the vessel does fit in this last lock cycle, it will request a place in this cycle
                            else:
                                access_lineup_length = lineup_area.length.get(vessel.L)
                                
                                #Assigns the length of the vessel to this request
                                lineup_area.length.get_queue[-1].length = vessel.L
                                
                                yield access_lineup_length
                                
                    #Requesting procedure for access to line-up area
                    #- If the vessels fits in the line-up area right away
                    if vessel.L < lineup_area.length.level:
                        yield from request_access_lock_cycle()
                        
                    #- Else, if the vessel does not fit in the line-up area right away
                    else:
                        if lineup_area.length.get_queue == []:
                            yield from create_new_lock_cycle_and_request_access()
                        else:
                            yield from request_access_lock_cycle()
                
                #Determines the current time
                wait_for_lineup_area = vessel.env.now
                
                #Assigning the lock chain series with least expected waiting time to the vessel
                vessel.lock_name = []     
                lock_queue_length = []                                                   
                for count,lock in enumerate(locks):
                    choose_lock_chamber(vessel,lock,node_lock,count,lineup_areas,lock_queue_length)
                
                #If the function did not yet assign a lock chain series
                if vessel.lock_name == []:
                    vessel.lock_name = lineup_areas[lock_queue_length.index(min(lock_queue_length))].name
    
                #Request access line-up area which is assigned to the vessel
                if lineup_area.name != vessel.lock_name:
                    continue
                
                yield from access_lineup_area(vessel,lineup_area)
    
                #Release of vessel's occupation of the waiting area
                waiting_area.waiting_area[node_waiting_area].release(vessel.access_waiting_area)
                
                #Calculation of location in line-up area as a distance in [m] from start line-up jetty
                #- If the line-up area is not empty
                if len(lineup_area.line_up_area[node_lineup_area].users) != 0:
                    vessel.lineup_dist = (lineup_area.line_up_area[node_lineup_area].users[-1].lineup_dist - 
                                          0.5*lineup_area.line_up_area[node_lineup_area].users[-1].length - 
                                          0.5*vessel.L)
                #- Else, if the line-up area is empty
                else:
                    vessel.lineup_dist = lineup_area.length.capacity - 0.5*vessel.L
    
                #Calculation of the (lat,lon)-coordinates of the assigned position in the line-up area
                vessel.wgs84 = pyproj.Geod(ellps="WGS84")
                [lineup_area_start_lat, 
                 lineup_area_start_lon, 
                 lineup_area_stop_lat, 
                 lineup_area_stop_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x, 
                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)+1]]['geometry'].x, 
                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)+1]]['geometry'].y]                  
                fwd_azimuth,_,_ = vessel.wgs84.inv(lineup_area_start_lat, lineup_area_start_lon, lineup_area_stop_lat, lineup_area_stop_lon)
                [vessel.lineup_pos_lat,
                 vessel.lineup_pos_lon,_] = pyproj.Geod(ellps="WGS84").fwd(vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x,
                                                                           vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                                                           fwd_azimuth,vessel.lineup_dist)
                
                #Formal request of the vessel to access the line-up area assigned to the vessel (always granted)                                                  
                vessel.access_lineup_area = lineup_area.line_up_area[node_lineup_area].request()
                
                #Some attributes are assigned to the vessel's formal request to access the line-up area
                lineup_area.line_up_area[node_lineup_area].users[-1].length = vessel.L
                lineup_area.line_up_area[node_lineup_area].users[-1].id = vessel.id
                [lineup_area.line_up_area[node_lineup_area].users[-1].lineup_pos_lat,
                 lineup_area.line_up_area[node_lineup_area].users[-1].lineup_pos_lon] = [vessel.lineup_pos_lat, vessel.lineup_pos_lon]
                lineup_area.line_up_area[node_lineup_area].users[-1].lineup_dist = vessel.lineup_dist
                lineup_area.line_up_area[node_lineup_area].users[-1].n = len(lineup_area.line_up_area[node_lineup_area].users) #(the number of vessels in the line-up area at the moment)
                lineup_area.line_up_area[node_lineup_area].users[-1].v = 0.5*vessel.v
                lineup_area.line_up_area[node_lineup_area].users[-1].wait_for_next_cycle = False #(a boolean which indicates if the vessel has to wait for a next lock cycle)
                lineup_area.line_up_area[node_lineup_area].users[-1].waited_in_waiting_area = False #(a boolean which indicates if the vessel had to wait in the waiting area)
                
                #Request of entering the line-up area to assure that vessels will enter the line-up area one-by-one                                                                            
                vessel.enter_lineup_length = lineup_area.enter_line_up_area[node_lineup_area].request()
                yield vessel.enter_lineup_length
                    
                #Calculates and reports the total waiting time in the waiting area
                if wait_for_lineup_area != vessel.env.now: 
                    waiting = vessel.env.now - wait_for_lineup_area
                    vessel.log_entry("Waiting in waiting area start", wait_for_lineup_area, 0, 
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[node_waiting_area])
                    vessel.log_entry("Waiting in waiting area stop", vessel.env.now, waiting, 
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[node_waiting_area])
                    
                    #Speed reduction in the approach to the line-up area, as the vessel had to lay still in the waiting area
                    vessel.v = 0.5*vessel.v
                    
                    #Changes boolean of the vessel which indicates that it had to wait in the waiting area
                    for line_up_user in range(len(lineup_area.line_up_area[node_lineup_area].users)):
                        if lineup_area.line_up_area[node_lineup_area].users[line_up_user].id == vessel.id:
                            lineup_area.line_up_area[node_lineup_area].users[line_up_user].waited_in_waiting_area = True
                            break
                break
            break
    
    def approach_lineup_area(vessel,node_lineup_area):
        """ Processes vessels which are approaching the line-up area of the lock complex: 
                determines whether the assigned position in the line-up area (distance in [m]) should be changed as the preceding vessel(s), 
                which was/were waiting in the line-up area, has/have of is/are already accessed/accessing the lock.
                
            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """
        
        #Imports the properties of the line-up area the vessel is assigned to
        lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name:
                continue
            
            #Identifies the index of the node of the waiting area within the route of the vessel
            index_node_lineup_area = vessel.route.index(node_lineup_area)
            
            #Checks whether the line-up area is the first encountered line-up area of the lock complex
            for node_lock in vessel.route[index_node_lineup_area:]:
                if 'Lock' in vessel.env.FG.nodes[node_lock].keys():
                    locks = vessel.env.FG.nodes[node_lock]["Lock"] 
                    
                    for lock in locks:
                        if lock.name != vessel.lock_name:
                            continue
                        
                        def change_lineup_dist(vessel, lineup_area, lock, lineup_dist, q):
                            """ Determines whether the assigned position in the line-up area (distance in [m]) should be changed as the preceding vessel(s), 
                                    which was/were waiting in the line-up area, has/have of is/are already accessed/accessing the lock.
                                This function is used in the approach_lineup_area function. 
                                
                                Input: 
                                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                                    - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                                                   assigned to the vessel as the lock series with the least expected total waiting time
                                    - lock: an object within the network which is generated with the IsLock mixin class and
                                            assigned to the vessel as the lock series with the least expected total waiting time
                                    - lineup_dist: the initial position of the vessel in the line-up area as the distance from the origin of the jetty in [m]
                                    - q: an integer number which represents the assigned position of the vessel in the line-up area, only the vessel which is
                                         the new first in line (q=0) will be processed"""
                            
                            if q == 0 and lineup_area.line_up_area[node_lineup_area].users[q].n != (lineup_area.line_up_area[node_lineup_area].users[q].n-len(lock.resource.users)):
                                lineup_dist = lock.length.capacity - 0.5*vessel.L
                            return lineup_dist
                        
                        #Checks the need to change the position of the vessel within the line-up area 
                        for q in range(len(lineup_area.line_up_area[node_lineup_area].users)):
                            if lineup_area.line_up_area[node_lineup_area].users[q].id == vessel.id:
                                
                                #Imports information about the current lock cycle
                                direction = vessel.route[vessel.route.index(node_lock)-1] == lock.node_1
                                lock_door_1_user_priority = 0
                                lock_door_2_user_priority = 0
                                lock_door_1_users = lock.doors_1[lock.node_1].users
                                lock_door_2_users = lock.doors_2[lock.node_3].users
                                
                                if direction and lock_door_2_users != []:
                                    lock_door_2_user_priority = lock.doors_2[lock.node_3].users[0].priority
                                 
                                elif not direction and lock_door_1_users != []:
                                    lock_door_1_user_priority = lock.doors_1[lock.node_1].users[0].priority
                                
                                #Decision if position should be changed
                                if direction and lock_door_2_user_priority == -1:
                                    vessel.lineup_dist = change_lineup_dist(vessel, lineup_area, lock, vessel.lineup_dist, q)
                                
                                elif not direction and lock_door_1_user_priority == -1: 
                                    vessel.lineup_dist = change_lineup_dist(vessel, lineup_area, lock, vessel.lineup_dist, q)
                                
                                #Calculation of (lat,lon)-coordinates based on (newly) assigned position (line-up distance in [m])
                                [lineup_area_start_lat, 
                                 lineup_area_start_lon, 
                                 lineup_area_stop_lat, 
                                 lineup_area_stop_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x, 
                                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)+1]]['geometry'].x, 
                                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)+1]]['geometry'].y]
                                                          
                                fwd_azimuth,_,_ = pyproj.Geod(ellps="WGS84").inv(lineup_area_start_lat, 
                                                                                 lineup_area_start_lon, 
                                                                                 lineup_area_stop_lat, 
                                                                                 lineup_area_stop_lon)
                                [vessel.lineup_pos_lat,
                                 vessel.lineup_pos_lon,_] = pyproj.Geod(ellps="WGS84").fwd(vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x,
                                                                                           vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                                                                           fwd_azimuth,vessel.lineup_dist)
                                
                                #Changes the positional attributes of the vessel as user of the line-up area accordingly
                                lineup_area.line_up_area[node_lineup_area].users[q].lineup_pos_lat = vessel.lineup_pos_lat
                                lineup_area.line_up_area[node_lineup_area].users[q].lineup_pos_lon = vessel.lineup_pos_lon
                                lineup_area.line_up_area[node_lineup_area].users[q].lineup_dist = vessel.lineup_dist
                                break
                        break
                break
            break
            
    def leave_lineup_area(vessel,node_lineup_area):
        """ Processes vessels which are waiting in the line-up area of the lock complex: 
                requesting access to the lock chamber given the governing phase in the lock cycle of the lock chamber and calculates the
                position within the lock chamber.
                
            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """
        
        #Imports the properties of the line-up area the vessel is assigned to
        lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name:
                continue
            
            #Identifies the index of the node of the line-up area within the route of the vessel
            index_node_lineup_area = vessel.route.index(node_lineup_area)
            
            #Checks whether the line-up area is the first encountered line-up area of the lock complex
            for node_lock in vessel.route[index_node_lineup_area:]: #(loops over line-up area)
                if 'Lock' not in vessel.env.FG.nodes[node_lock].keys():
                    continue
                
                #Imports the properties of the lock chamber the vessel is assigned to
                locks = vessel.env.FG.nodes[node_lock]["Lock"]
                for lock in locks:
                    if lock.name != vessel.lock_name:
                        continue
                
                    #Alters vessel's approach speed to lock chamber, if vessel didn't have to wait in the waiting area
                    for line_up_user in range(len(lineup_area.line_up_area[node_lineup_area].users)):
                        if (lineup_area.line_up_area[node_lineup_area].users[line_up_user].id == vessel.id and not
                            lineup_area.line_up_area[node_lineup_area].users[line_up_user].waited_in_waiting_area):
                            vessel.v = 0.5*vessel.v
                            break
                    
                    #Imports position of the vessel in the line-up area
                    position_in_lineup_area = shapely.geometry.Point(vessel.lineup_pos_lat,vessel.lineup_pos_lon)
                    
                    #Vessel releases its request to enter the line-up area, made in the waiting area
                    lineup_area.enter_line_up_area[node_lineup_area].release(vessel.enter_lineup_length)
    
                    #Determines current time
                    wait_for_lock_entry = vessel.env.now
        
                    #Checks whether the lock chamber is the first encountered lock chamber of the lock complex
                    for node_opposing_lineup_area in vessel.route[(index_node_lineup_area+1):]:
                        if 'Line-up area' not in vessel.env.FG.nodes[node_opposing_lineup_area].keys():
                            continue
                        
                        #Imports the properties of the opposing line-up area the vessel is assigned to
                        opposing_lineup_areas = vessel.env.FG.nodes[node_opposing_lineup_area]["Line-up area"]
                        for opposing_lineup_area in opposing_lineup_areas:
                            if opposing_lineup_area.name == vessel.lock_name:
                                break
                        break
                    
                    def access_lock_chamber(vessel,lineup_area,node_lineup_area,lock,node_lock,opposing_lineup_area,node_opposing_lineup_area,door1,door2):
                        """ Processes vessels which are waiting in the line-up area of the lock complex: 
                                determines the current phase within the lock cycle and adjusts the request of the vessel accordingly
                            This function is used in the leave_lineup_area function.
                            
                            Input:
                                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                                - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                                               assigned to the vessel as the lock series with the least expected total waiting time
                                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network
                                - lock: an object within the network which is generated with the IsLock mixin class and
                                        assigned to the vessel as the lock series with the least expected total waiting time
                                - node_lock: a string which includes the name of the node at which the lock chamber is located in the network
                                - opposing_lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                                                        assigned to the vessel to be entered when leaving the lock chamber
                                - node_opposing_lineup_area: a string which includes the name of the node at which the line-up area is located on the 
                                                             opposite side of the lock chamber in the network
                                - door1: an object created in the IsLock class which resembles the set of lock doors which is first encountered by
                                         the vessel, which should be supscripted to, using a string which includes the name of the node at which this 
                                         lock door is located in the network and was specified as input in the IsLock class
                                - door2: an object created in the IsLock class which resembles the set of lock doors which is last encountered by
                                         the vessel, which should be supscripted to, using a string which includes the name of the node at which this 
                                         lock door is located in the network and was specified as input in the IsLock class """    
                        
                        def wait_for_requested_empty_conversion_by_oppositely_directed_vessel():
                            """ Vessel will wait for the lock chamber to be converted without vessels, as it was requested by
                                    the vessel(s) waiting on the other side of the lock chamber. This is programmed by a vessel's
                                    request of the converting while_in_line_up_area-resource of the opposing line-up area with 
                                    capacity = 1, yielding a timeout, and immediately releasing the request.
                                    
                                No input required. """
                            
                            vessel.waiting_during_converting = opposing_lineup_area.converting_while_in_line_up_area[node_opposing_lineup_area].request()
                            yield vessel.waiting_during_converting
                            opposing_lineup_area.converting_while_in_line_up_area[node_opposing_lineup_area].release(vessel.waiting_during_converting)
                    
                        def request_approach_lock_chamber(timeout_required = True):
                            """ Vessel will request if it can enter the lock by requesting access to the first set of lock doors. This
                                    request always has priority = 0, as vessels can only pass these doors when the doors are open (not 
                                    claimed by a priority = -1 request). The capacity of the doors equals one, to prevent ships from 
                                    entering simultaneously. The function yields a timeout. This can be switched off if it is assured
                                    the vessel can approach immediately.
                                    
                                Input:
                                    - timeout_required: a boolean which defines whether the requesting vessel receives a timeout. """
                                
                            vessel.access_lock_door1 = door1.request()
                            if timeout_required:
                                yield vessel.access_lock_door1
                            
                        def request_empty_lock_conversion(hold_request = False):
                            """ Vessel will request the lock chamber to be converted without vessels to his side of the lock chamber. This
                                    is programmed by requesting the converting_while_in_line_up_area resource of the line-up area the vessels is
                                    currently located in. If there was already a request by another vessel waiting in the same line-up area, this
                                    original request can be holded.
    
                                Input:
                                    - hold_request: a boolean which defines where an earlier request for the same empty lock conversion which 
                                                    was made by another ship should be holded"""
                            
                            vessel.waiting_during_converting = lineup_area.converting_while_in_line_up_area[node_lineup_area].request()
                            yield vessel.waiting_during_converting
                            if not hold_request:
                                yield from lock.convert_chamber(vessel.env, vessel.route[vessel.route.index(node_lock)-1], 0)
                            lineup_area.converting_while_in_line_up_area[node_lineup_area].release(vessel.waiting_during_converting)
                                
                        def secure_lock_cycle(hold_request = False, timeout_required = True, priority = -1):
                            """ Vessel will indicate the direction of the next lock cycle by requesting access to the second pair of lock
                                    doors. Therefore, this request by default has priority = -1, which indicates the direction of the lockage
                                    as requests to access the same doors by vessels on the opposite side of the lock will be queued. A timeout
                                    is yielded. This can be switched off if it is assured the vessel can approach immediately. Furthermore, if 
                                    there was already a request by another vessel waiting in the same line-up area, this original request can be 
                                    holded. Lastly, the function is also used with priority = 0 in order to let vessels wait for the next lockage.
    
                                Input:
                                    - hold_request: a boolean which defines where an earlier request for the same empty lock conversion which 
                                                    was made by another ship should be holded
                                    - timeout_required: a boolean which defines whether the requesting vessel receives a timeout.
                                    - priority: an integer [-1,0] which indicates the priority of the request: either with (-1) or without (0)
                                                priority. """
                            
                            vessel.access_lock_door2 = door2.request(priority = priority)
                            if hold_request:
                                door2.release(door2.users[0])
                            if timeout_required:
                                yield vessel.access_lock_door2
                            door2.users[0].id = vessel.id
                        
                        def wait_for_next_lockage():
                            """ Vessels will wait for the next lockage by requesting access to the second pair of lock doors without priority. If
                                    granted, the request will immediately be released.
                                
                                No input required. """
                            
                            yield from secure_lock_cycle(priority = 0)
                            door2.release(vessel.access_lock_door2)

                        def wait_for_next_cycle():
                            """ Vessels will wait for the next lock cycle by requesting to pass the line-up area on the opposite side of the lock
                                    chamber without priority. A timeout is yielded untill all vessels in the lock have passed this area.
                                
                                No input required. """
                                
                            vessel.wait_for_vessels_to_pass = opposing_lineup_area.pass_line_up_area[node_opposing_lineup_area].request()
                            yield vessel.wait_for_vessels_to_pass
                            opposing_lineup_area.pass_line_up_area[node_opposing_lineup_area].release(vessel.wait_for_vessels_to_pass)
                        
                        #Determines current moment within the lock cycle
                        lock_door_2_user_priority = 0
                        if door2.users != []:
                            lock_door_2_user_priority = door2.users[0].priority
                            
                        #Request procedure of the lock doors, which is dependent on the current moment within the lock cycle:
                        #- If there is a lock cycle being prepared or going on in the same direction of the vessel
                        if lock_door_2_user_priority == -1:
                            
                            #If vessel does not fit in next lock cycle
                            if lock.resource.users != [] and (vessel.L > (lock.resource.users[-1].lock_dist-0.5*lock.resource.users[-1].length) or lock.resource.users[-1].converting):
                                yield from wait_for_next_cycle()
                                
                            #Determines whether an empty conversion is needed or already requested by another vessel going the same way
                            if lineup_area.converting_while_in_line_up_area[node_lineup_area].users != []:
                                yield from request_empty_lock_conversion(hold_request = True)
                                
                            elif (len(door2.users) == 0 and vessel.route[vessel.route.index(node_lock)-1] != lock.water_level):
                                yield from request_empty_lock_conversion()
                            
                            #Request to enter the lock
                            yield from request_approach_lock_chamber()
                            
                            #Request to start the lock cycle
                            if door2.users != [] and door2.users[0].priority == -1:
                                yield from secure_lock_cycle(hold_request = True)
                            else:
                                yield from secure_lock_cycle()
                        
                        #- If there is a lock cycle being prepared or going on to the direction of the vessel or if the lock chamber is empty
                        else:
                            
                            #If the lock is already converting empty to the other side as requested by a vessel on the opposite side of the lock chamber
                            if opposing_lineup_area.converting_while_in_line_up_area[node_opposing_lineup_area].users != []:
                                yield from wait_for_requested_empty_conversion_by_oppositely_directed_vessel()
                                
                            #Determining (new) situation
                            #- If the lock chamber is empty
                            if lock.resource.users == []:
                                yield from request_approach_lock_chamber(timeout_required=False)
                                yield from secure_lock_cycle(timeout_required=False)
                                
                                #Determines if an empty lockage is required
                                if vessel.route[vessel.route.index(node_lock)-1] != lock.water_level:
                                    yield from request_empty_lock_conversion()
                                
                            #- Else, if the lock chamber is occupied
                            else:
                                yield from wait_for_next_lockage()                                   
                                yield from request_approach_lock_chamber()
                                
                                #Request to start the lock cycle
                                if door2.users != [] and door2.users[0].priority == -1:
                                    yield from secure_lock_cycle(hold_request = True)
                                else:
                                    yield from secure_lock_cycle()
                           
                        #Formal request access to lock chamber and calculate position within the lock chamber
                        lock.length.get(vessel.L)
                        vessel.access_lock = lock.resource.request()
                        lock.pos_length.get(vessel.L)
                        vessel.lock_dist = lock.pos_length.level + 0.5*vessel.L #(distance from first set of doors in [m])
                        
                        #Assign attributes to granted request
                        lock.resource.users[-1].id = vessel.id
                        lock.resource.users[-1].length = vessel.L
                        lock.resource.users[-1].lock_dist = vessel.lock_dist
                        lock.resource.users[-1].converting = False #(boolean which indicates if the lock is already converting)
        
                    #Request access to lock chamber
                    direction = vessel.route[vessel.route.index(node_lock)-1] == lock.node_1 #(determines vessel's direction)
                    if direction:
                        yield from access_lock_chamber(vessel,lineup_area,node_lineup_area,lock,node_lock,opposing_lineup_area,node_opposing_lineup_area,
                                                       lock.doors_1[lock.node_1],lock.doors_2[lock.node_3])
                        
                    if not direction:
                        yield from access_lock_chamber(vessel,lineup_area,node_lineup_area,lock,node_lock,opposing_lineup_area,node_opposing_lineup_area,
                                                       lock.doors_2[lock.node_3],lock.doors_1[lock.node_1])
                    
                    #Calculates and reports the total waiting time in the line-up area
                    if wait_for_lock_entry != vessel.env.now:
                        waiting = vessel.env.now - wait_for_lock_entry
                        vessel.log_entry("Waiting in line-up area start", wait_for_lock_entry, 0, position_in_lineup_area)
                        vessel.log_entry("Waiting in line-up area stop", vessel.env.now, waiting, position_in_lineup_area)
                    
                    #Calculation of (lat,lon)-coordinates of assigned position in lock chamber
                    vessel.wgs84 = pyproj.Geod(ellps="WGS84")
                    [doors_node_lineup_area_lat, 
                     doors_node_lineup_area_lon, 
                     doors_destination_lat, 
                     doors_destination_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)-1]]['geometry'].x, 
                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)-1]]['geometry'].y,
                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)+1]]['geometry'].x, 
                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)+1]]['geometry'].y]
                    fwd_azimuth,_,distance = vessel.wgs84.inv(doors_node_lineup_area_lat, doors_node_lineup_area_lon, doors_destination_lat, doors_destination_lon)
                    [vessel.lock_pos_lat,vessel.lock_pos_lon,_] = vessel.wgs84.fwd(vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)-1]]['geometry'].x,
                                                                                   vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)-1]]['geometry'].y,
                                                                                   fwd_azimuth,vessel.lock_dist)
                    break
                break
            break
     
    def left_lineup_area(vessel,node_lineup_area):
        """ Processes vessels which have left the line-up area of the lock complex and are approaching the lock chamber: 
                releases their requests of the line-up area
                
            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """
                
        #Imports the properties of the line-up area the vessel was assigned to 
        lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name:
                continue
            
            #Identifies the index of the node of the line-up area within the route of the vessel
            index_node_lineup_area = vessel.route.index(node_lineup_area)
            
            #Checks whether the line-up area is the first encountered line-up area of the lock complex
            for node_lock in vessel.route[index_node_lineup_area:]:
                if 'Lock' not in vessel.env.FG.nodes[node_lock].keys():
                    continue
                
                if vessel.env.FG.nodes[node_lock]["Lock"][0].name.split('_')[0] != vessel.lock_name.split('_')[0]:
                    continue
               
                #Releases the vessel's formal request to access the line-up area and releases its occupied length of the line-up area
                lineup_area.line_up_area[node_lineup_area].release(vessel.access_lineup_area)
                lineup_area.length.put(vessel.L)
                break
            break
        
    def leave_lock(vessel,node_lock):
        """ Processes vessels which are waiting in the lock chamber to be levelled and after levelling:
                checks if vessels which have entered the lock chamber have to wait for the other vessels to enter the lock chamber and
                requests conversion of the lock chamber.
                
            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lock: a string which includes the name of the node at which the lock chamber is located in the network """
        
        #Imports the properties of the lock chamber the vessel is assigned to 
        locks = vessel.env.FG.nodes[node_lock]["Lock"]
        for lock in locks:
            if lock.name != vessel.lock_name:
                continue
            position_in_lock = shapely.geometry.Point(vessel.lock_pos_lat,vessel.lock_pos_lon) #alters node_lock in accordance with given position in lock
            first_user_in_lineup_length = lock.length.capacity
            
            #Identifies the index of the node of the lock chamber within the route of the vessel
            index_node_lock = vessel.route.index(node_lock)

            #Checks whether the lock chamber is the first encountered lock chamber of the lock complex
            for node_opposing_lineup_area in reversed(vessel.route[index_node_lock:]):
                if "Line-up area" not in vessel.env.FG.nodes[node_opposing_lineup_area].keys():
                    continue
                
                #Imports the properties of the opposing line-up area the vessel is assigned to
                opposing_lineup_areas = vessel.env.FG.nodes[node_opposing_lineup_area]["Line-up area"]
                for opposing_lineup_area in opposing_lineup_areas:
                    if opposing_lineup_area.name != vessel.lock_name:
                        continue
                    break
                break
            
            #Request access to pass the next line-up area after the lock chamber has levelled, so that vessels will leave the lock chamber one-by-one
            vessel.departure_lock = opposing_lineup_area.pass_line_up_area[node_opposing_lineup_area].request(priority = -1)    
               
            #Releases the vessel's request of their first encountered set of lock doors 
            direction = vessel.route[vessel.route.index(node_lock)-1] == lock.node_1
            if direction:
                yield lock.doors_1[lock.node_1].release(vessel.access_lock_door1)
            elif not direction:
                yield lock.doors_2[lock.node_3].release(vessel.access_lock_door1)

            #Imports the properties of the line-up area the vessel was assigned to 
            for node_lineup_area in reversed(vessel.route[:(index_node_lock-1)]):
                if "Line-up area" not in vessel.env.FG.nodes[node_lineup_area].keys():
                    continue
                lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
                for lineup_area in lineup_areas:
                    if lineup_area.name != vessel.lock_name:
                        continue
                    lineup_users = lineup_area.line_up_area[node_lineup_area].users
                    if lineup_users != []:
                        first_user_in_lineup_length = lineup_area.line_up_area[node_lineup_area].users[0].length
    
                    #Determines current time and reports this to vessel's log as start time of lock passage
                    start_time_in_lock = vessel.env.now
                    vessel.log_entry("Passing lock start", vessel.env.now, 0, position_in_lock)
                    
                    def waiting_for_other_lock_users(vessel,lock,node_lock,lineup_area,node_lineup_area,door1):
                        """ Function which yields a timeout to vessels as they have to wait for the other vessels to enter the lock.
                                the timeout is calculated by subsequently requesting and releasing the previously passed line-up area 
                                and lock doors again without priority, such that the all the vessels within the line-up area can enter
                                the lock before the levelling will start.
                            This function is used in the leave_lock function.
                            
                            Input:
                                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                                - lock: an object within the network which is generated with the IsLock mixin class and is asigned to the
                                        vessel as the lock chamber in the lock chain series with the least expected total waiting time
                                - node_lock: a string which includes the name of the node at which the lock chamber is located in the network
                                - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and is assigned
                                               to the vessel as the line-up area in the lock chain series with the least expected total waiting time
                                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network
                                - door1: an object created in the IsLock class which resembles the set of lock doors which is first encountered by
                                         the vessel, which should be supscripted to, using a string which includes the name of the node at which this 
                                         lock door is located in the network and was specified as input in the IsLock class """
            
                        vessel.access_line_up_area = lineup_area.enter_line_up_area[node_lineup_area].request()
                        yield vessel.access_line_up_area
                        lineup_area.enter_line_up_area[node_lineup_area].release(vessel.access_line_up_area)
                        vessel.access_lock_door1 = door1.request() #therefore it requests first the entering of line-up area and then the lock doors again
                        yield vessel.access_lock_door1
                        door1.release(vessel.access_lock_door1)
                    
                    #Determines if accessed vessel has to wait on accessing vessels
                    if direction and first_user_in_lineup_length < lock.length.level:
                        yield from waiting_for_other_lock_users(vessel,lock,node_lock,lineup_area,node_lineup_area,lock.doors_1[lock.node_1])
                    if not direction and first_user_in_lineup_length < lock.length.level:
                        yield from waiting_for_other_lock_users(vessel,lock,node_lock,lineup_area,node_lineup_area,lock.doors_2[lock.node_3])
    
                    #Determines if the vessel explicitely has to request the conversion of the lock chamber (only the first entered vessel) or can go with a previously made request
                    if lock.resource.users[0].id == vessel.id: 
                        lock.resource.users[0].converting = True
                        number_of_vessels = len(lock.resource.users)
                        yield from lock.convert_chamber(vessel.env, vessel.route[vessel.route.index(node_lock)+1],number_of_vessels)
                    else:
                        for lock_user in range(len(lock.resource.users)):
                            if lock.resource.users[lock_user].id != vessel.id:
                                continue
                            lock.resource.users[lock_user].converting = True
                            yield vessel.env.timeout(lock.doors_close + lock.operation_time(vessel.env) + lock.doors_open)
                            break
                
                #Yield request to leave the lock chamber
                yield vessel.departure_lock
                
                #Calculates and reports the total locking time
                vessel.log_entry("Passing lock stop", vessel.env.now, vessel.env.now-start_time_in_lock, position_in_lock,)
                
                #Adjusts position of the vessel in line-up area, apart from the first line-up area also used to position the vessel in the next line-up area, to the origin of the next line-up area
                [vessel.lineup_pos_lat,vessel.lineup_pos_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_opposing_lineup_area)]]['geometry'].x, 
                                                                 vessel.env.FG.nodes[vessel.route[vessel.route.index(node_opposing_lineup_area)]]['geometry'].y]
                break
            break
        
    def leave_opposite_lineup_area(vessel,node_lineup_area):
        """ Processes vessels which have left the lock chamber after levelling and are now in the next line-up area in order to leave the lock complex through the next waiting area:
                release of their requests for accessing their second encountered line-up area and lock chamber.
                
            Input:
                - vessel: an identity which is Identifiable, Movable,and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """
               
        #Imports the properties of the line-up area the vessel is assigned to 
        lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name: #assure lock chain = assigned chain
                continue
            index_node_lineup_area = vessel.route.index(node_lineup_area)
            
            #Checks whether the line-up area is the second encountered line-up area of the lock complex
            for node_lock in reversed(vessel.route[:(index_node_lineup_area-1)]):
                if 'Lock' not in vessel.env.FG.nodes[node_lock].keys():
                    continue
                
                #Imports the properties of the lock chamber the vessel was assigned to 
                locks = vessel.env.FG.nodes[node_lock]["Lock"]
                for lock in locks:
                    if lock.name != vessel.lock_name:
                        continue
                    
                    #Releases the vessel's request of their second encountered set of lock doors 
                    direction = vessel.route[vessel.route.index(node_lock)+1] == lock.node_3 
                    if direction and lock.doors_2[lock.node_3].users[0].id == vessel.id:
                        lock.doors_2[lock.node_3].release(vessel.access_lock_door2)
                    if not direction and lock.doors_1[lock.node_1].users[0].id == vessel.id:
                        lock.doors_1[lock.node_1].release(vessel.access_lock_door2)                   
        
                    #Releases the vessel's request to enter the second line-up area
                    lineup_area.pass_line_up_area[node_lineup_area].release(vessel.departure_lock)
                    
                    #Releases the vessel's formal request of the lock chamber and returns its occupied length in the lock chamber
                    lock.resource.release(vessel.access_lock)
                    departure_lock_length = lock.length.put(vessel.L) #put length back in lock
                    departure_lock_pos_length = lock.pos_length.put(vessel.L) #put position length back in lock
                    yield departure_lock_length
                    yield departure_lock_pos_length
                    break
                break
            break
    
class Movable(Locatable, Routeable, Log):
    """Mixin class: Something can move

    Used for object that can move with a fixed speed

    geometry: point used to track its current location
    v: speed"""

    def __init__(self, v=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.v = v
        self.wgs84 = pyproj.Geod(ellps="WGS84")

    def move(self):
        """determine distance between origin and destination, and
        yield the time it takes to travel it
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].
        """
        self.distance = 0

        # Check if vessel is at correct location - if not, move to location
        if (
            self.geometry
            != nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]
        ):
            orig = self.geometry
            dest = nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]

            print("Origin", orig)
            print("Destination", dest)

            self.distance += self.wgs84.inv(
                shapely.geometry.asShape(orig).x,
                shapely.geometry.asShape(orig).y,
                shapely.geometry.asShape(dest).x,
                shapely.geometry.asShape(dest).y,
            )[2]

            yield self.env.timeout(self.distance / self.current_speed)
            self.log_entry("Sailing to start", self.env.now, self.distance, dest)


        # Move over the path and log every step
        for node in enumerate(self.route):
            self.node = node[1]

            if node[0] + 2 <= len(self.route):
                origin = self.route[node[0]]
                destination = self.route[node[0] + 1]

            #if "Terminal" in self.env.FG.nodes[destination].keys():
            #    for event in pass_terminal(self,origin,destination):
            #        yield event
            
            #PassLock
            if "Waiting area" in self.env.FG.nodes[destination].keys():         #if waiting area is located at next node 
                print(self.id,self.env.now,'hello world')
                yield from PassLock.approach_waiting_area(self, destination)
                
            if "Waiting area" in self.env.FG.nodes[origin].keys():              #if vessel is in waiting area 
                yield from PassLock.leave_waiting_area(self, origin)

            if "Line-up area" in self.env.FG.nodes[destination].keys(): #if vessel is approaching the line-up area
                PassLock.approach_lineup_area(self, destination)

            if "Line-up area" in self.env.FG.nodes[origin].keys(): #if vessel is located in the line-up
                lineup_areas = self.env.FG.nodes[origin]["Line-up area"]
                for lineup_area in lineup_areas:
                    if lineup_area.name != self.lock_name: #picks the assigned parallel lock chain
                        continue    
                    
                    index_node_lineup_area = self.route.index(origin)
                    for node_lock in self.route[index_node_lineup_area:]:
                        if 'Lock' in self.env.FG.nodes[node_lock].keys():
                            yield from PassLock.leave_lineup_area(self,origin)
                            break
                        
                        elif 'Waiting area' in self.env.FG.nodes[node_lock].keys(): #if vessel is leaving the lock complex
                            yield from PassLock.leave_opposite_lineup_area(self,origin)
                            break

            if "Line-up area" in self.env.FG.nodes[self.route[node[0]-1]].keys(): #if line-up in previous node
                PassLock.left_lineup_area(self,self.route[node[0]-1])

            if "Lock" in self.env.FG.nodes[origin].keys(): #if vessel in lock
                yield from PassLock.leave_lock(self,origin)
                yield from self.pass_edge(origin, destination)
                self.v = 4*self.v    

            else:
                # print('I am going to go to the next node {}'.format(destination))
                yield from self.pass_edge(origin, destination)

            if node[0] + 2 == len(self.route):
                break

        # self.geometry = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        logger.debug("  distance: " + "%4.2f" % self.distance + " m")
        logger.debug("  sailing:  " + "%4.2f" % self.current_speed + " m/s")
        logger.debug(
            "  duration: "
            + "%4.2f" % ((self.distance / self.current_speed) / 3600)
            + " hrs"
        )

    def pass_edge(self, origin, destination):
        edge = self.env.FG.edges[origin, destination]
        orig = nx.get_node_attributes(self.env.FG, "geometry")[origin]
        dest = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        if "Lock" in self.env.FG.nodes[origin].keys():
            orig = shapely.geometry.Point(self.lock_pos_lat,self.lock_pos_lon)

        if "Lock" in self.env.FG.nodes[destination].keys():
            dest = shapely.geometry.Point(self.lock_pos_lat,self.lock_pos_lon)

        if "Line-up area" in self.env.FG.nodes[origin].keys():
            orig = shapely.geometry.Point(self.lineup_pos_lat,self.lineup_pos_lon)

        if "Line-up area" in self.env.FG.nodes[destination].keys():
            dest = shapely.geometry.Point(self.lineup_pos_lat,self.lineup_pos_lon)

        if 'geometry' in edge:
            edge_route = np.array(edge['geometry'])

            # check if edge is in the sailing direction, otherwise flip it
            distance_from_start = self.wgs84.inv(
                    orig.x,
                    orig.y,
                    edge_route[0][0],
                    edge_route[0][1],
                )[2]
            distance_from_stop = self.wgs84.inv(
                    orig.x,
                    orig.y,
                    edge_route[-1][0],
                    edge_route[-1][1],
                )[2]
            if distance_from_start>distance_from_stop:
                # when the distance from the starting point is greater than from the end point
                edge_route = np.flipud(np.array(edge['geometry']))

            for index, pt in enumerate(edge_route[:-1]):
                sub_orig = shapely.geometry.Point(edge_route[index][0], edge_route[index][1])
                sub_dest = shapely.geometry.Point(edge_route[index+1][0], edge_route[index+1][1])

                distance = self.wgs84.inv(
                    shapely.geometry.asShape(sub_orig).x,
                    shapely.geometry.asShape(sub_orig).y,
                    shapely.geometry.asShape(sub_dest).x,
                    shapely.geometry.asShape(sub_dest).y,
                )[2]
                self.distance += distance
                self.log_entry("Sailing from node {} to node {} sub edge {} start".format(origin, destination, index), self.env.now, 0, sub_orig,)
                yield self.env.timeout(distance / self.current_speed)
                self.log_entry("Sailing from node {} to node {} sub edge {} stop".format(origin, destination, index), self.env.now, 0, sub_dest,)
            self.geometry = dest
            # print('   My new origin is {}'.format(destination))
        else:
            distance = self.wgs84.inv(
                shapely.geometry.asShape(orig).x,
                shapely.geometry.asShape(orig).y,
                shapely.geometry.asShape(dest).x,
                shapely.geometry.asShape(dest).y,
            )[2]

            self.distance += distance
            arrival = self.env.now

            # Act based on resources
            if "Resources" in edge.keys():
                with self.env.FG.edges[origin, destination]["Resources"].request() as request:
                    yield request

                    if arrival != self.env.now:
                        self.log_entry("Waiting to pass edge {} - {} start".format(origin, destination), arrival, 0, orig,)
                        self.log_entry("Waiting to pass edge {} - {} stop".format(origin, destination), self.env.now, 0, orig,)

                    self.log_entry("Sailing from node {} to node {} start".format(origin, destination), self.env.now, 0, orig,)
                    yield self.env.timeout(distance / self.current_speed)
                    self.log_entry("Sailing from node {} to node {} stop".format(origin, destination), self.env.now, 0, dest,)

            else:
                self.log_entry("Sailing from node {} to node {} start".format(origin, destination), self.env.now, 0, orig,)
                yield self.env.timeout(distance / self.current_speed)
                self.log_entry("Sailing from node {} to node {} stop".format(origin, destination), self.env.now, 0, dest,)

    @property
    def current_speed(self):
        return self.v


class ContainerDependentMovable(Movable, HasContainer):
    """ContainerDependentMovable class
    Used for objects that move with a speed dependent on the container level
    compute_v: a function, given the fraction the container is filled (in [0,1]), returns the current speed"""

    def __init__(self, compute_v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.compute_v = compute_v
        self.wgs84 = pyproj.Geod(ellps="WGS84")

    @property
    def current_speed(self):
        return self.compute_v(self.container.level / self.container.capacity)
