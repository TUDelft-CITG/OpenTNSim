"""Main module."""

# package(s) related to time, space and id
import json
import logging
import uuid
import pathlib
import datetime
import time

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

import opentnsim.energy
import opentnsim.graph_module

logger = logging.getLogger(__name__)



class SimpyObject:
    """General object which can be extended by any class requiring a simpy environment

    - env: a simpy Environment
    """

    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env


class HasResource(SimpyObject):
    """Something that has a resource limitation, a resource request must be granted before the object can be used.

    - nr_resources: nr of requests that can be handled simultaneously
    """

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

    - name: a name
    - id: a unique id generated with uuid
    """

    def __init__(self, name, id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.name = name
        # generate some id, in this case based on m
        self.id = id if id else str(uuid.uuid1())


class Locatable:
    """Mixin class: Something with a geometry (geojson format)

    - geometry: can be a point as well as a polygon
    """

    def __init__(self, geometry, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.geometry = geometry
        self.node = None


class Neighbours:
    """Can be added to a locatable object (list)

    - travel_to: list of locatables to which can be travelled
    """

    def ___init(self, travel_to, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.neighbours = travel_to

class HasLength(SimpyObject):
    """Mixin class: Something with a storage capacity

    capacity: amount the container can hold
    level: amount the container holds initially
    total_requested: a counter that helps to prevent over requesting
    """

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


class VesselProperties:
    """Mixin class: Something that has vessel properties
    This mixin is updated to better accommodate the ConsumesEnergy mixin

    - type: can contain info on vessel type (avv class, cemt_class or other)
    - B: vessel width
    - L: vessel length
    - h_min: vessel minimum water depth, can also be extracted from the network edges if they have the property ['Info']['GeneralDepth']
    - T: actual draught
    - C_B: block coefficient ('fullness') [-]
    - safety_margin : the water area above the waterway bed reserved to prevent ship grounding due to ship squatting during sailing, the value of safety margin depends on waterway bed material and ship types. For tanker vessel with rocky bed the safety margin is recommended as 0.3 m based on Van Dorsser et al. The value setting for safety margin depends on the risk attitude of the ship captain and shipping companies.
    - h_squat: the water depth considering ship squatting while the ship moving
    - payload: cargo load [ton], the actual draught can be determined by knowing payload based on van Dorsser et al's method.(https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)
    - vessel_type: vessel type can be selected from "Container","Dry_SH","Dry_DH","Barge","Tanker". ("Dry_SH" means dry bulk single hull, "Dry_DH" means dry bulk double hull), based on van Dorsser et al's paper.(https://www.researchgate.net/publication/344340126_The_effect_of_low_water_on_loading_capacity_of_inland_ships)
    Alternatively you can specify draught based on filling degree
    - H_e: vessel height unloaded
    - H_f: vessel height loaded
    - T_e: draught unloaded
    - T_f: draught loaded

    """
        # TODO: add blockage factor S to vessel properties

    def __init__(
            self,
            type,
            B,
            L,
            h_min=None,
            T=None,
            C_B=None,
            H_e=None,
            H_f=None,
            T_e=None,
            T_f=None,
            safety_margin=None,
            h_squat=None,
            payload=None,
            vessel_type=None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        """Initialization
        """
        self.type = type
        self.B = B
        self.L = L
        # hidden because these can also computed on the fly
        self._T = T
        self._h_min = h_min
        self.C_B = C_B
        # alternative  options
        self.H_e = H_e
        self.H_f = H_f
        self.T_e = T_e
        self.T_f = T_f
        self.safety_margin = safety_margin
        self.h_squat = h_squat
        self.payload = payload
        self.vessel_type = vessel_type

    @property
    def T(self):
        """Compute the actual draught

        There are 3 ways to get actual draught
        - by directly providing actual draught values in the notebook
        - Or by providing ship draughts in fully loaded state and empty state, the actual draught will be computed based on filling degree
        - Or by giving vessel type with its payload (van Dorsser et al, 2020)

        """
        if self._T is not None:
            # if we were passed a T value, use tha one
            T = self._T
        elif self.T_f is not None and self.T_e is not None:
            # base draught on filling degree
            T = self.filling_degree * (self.T_f - self.T_e) + self.T_e
        elif self.payload is not None and self.vessel_type is not None:
            T = opentnsim.strategy.Payload2T(self, Payload_strategy = self.payload, vessel_type = self.vessel_type, bounds=(0, 40))  # this need to be tested

        return T

    @property
    def h_min(self):
        if self._h_min is not None:
            h_min = self._h_min
        else:
            h_min = opentnsim.graph_module.get_minimum_depth(graph=self.env.FG, route=self.route)

        return h_min


    def calculate_max_sinkage(self, v, h_0):
        """Calculate the maximum sinkage of a moving ship

        the calculation equation is described in Barrass, B. & Derrett, R.'s book (2006), Ship Stability for Masters and Mates, chapter 42. https://doi.org/10.1016/B978-0-08-097093-6.00042-6

        some explanation for the variables in the equation:
        - h_0: water depth
        - v: ship velocity relative to the water
        - 150: Here we use the standard width 150 m as the waterway width

        """

        max_sinkage = (self.C_B * ((self.B * self._T) / (150 * h_0)) ** 0.81) * (v ** 2.08) / 20

        return max_sinkage

    def calculate_h_squat(self, v, h_0):
        if self.h_squat is "No":
            h_squat = h_0
        elif self.h_squat is "Yes":
            h_squat = h_0 - self.calculate_max_sinkage(v, h_0)

        return h_squat


    @property
    def H(self):
        """ Calculate current height based on filling degree
        """

        return (
                self.filling_degree * (self.H_f - self.H_e)
                + self.H_e
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
        """ Calculate a path based on vessel restrictions
        """

        graph = graph if graph else self.env.FG
        minWidth = minWidth if minWidth else 1.1 * self.B
        minHeight = minHeight if minHeight else 1.1 * self.H
        minDepth = minDepth if minDepth else 1.1 * self.T

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

    Keyword arguments:

    - P_installed: installed engine power [kW]
    - P_tot_given: Total power set by captain (includes hotel power). When P_tot_given > P_installed; P_tot_given=P_installed.
    - bulbous_bow: inland ships generally do not have a bulbous_bow,set to none. If a ship has a bulbous_bow, set to 1.
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
    - one_k2: appendage resistance factor (1+k2) [-]
    - C_year: construction year of the engine [y]
    """

    def __init__(
            self,
            P_installed,
            L_w,
            C_year,
            current_year=None,  # current_year
            bulbous_bow=None,
            P_tot_given=None,  # the actual power engine setting
            nu=1 * 10 ** (-6),
            rho=1000,
            g=9.81,
            x=2,
            eta_o=0.6,
            eta_r=1.00,
            eta_t=0.98,
            eta_g=0.96,
            c_stern=0,
            C_BB=0.2,
            one_k2=2.5,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        """Initialization
        """

        self.P_installed = P_installed
        self.bulbous_bow=bulbous_bow
        self.P_tot_given=P_tot_given
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
        self.one_k2 = one_k2


        # plugin function that computes velocity based on power
        self.power2v = opentnsim.energy.power2v

        if C_year:
            self.C_year= C_year
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
        assert self.L_w in [1,2,3],'Invalid value L_w, should be 1,2 or 3'
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
        self.age = int(np.random.weibull(self.k) * self.lmb)

        # Construction year of the engine
        self.C_year = self.year - self.age

        logger.debug(f'The construction year of the engine is {self.C_year}')
        return self.C_year

    def calculate_properties(self):
        """Calculate a number of basic vessel properties
        """

    # TO DO: add properties for seagoing ships with bulbs

        self.C_M = 1.006 - 0.0056 * self.C_B ** (-3.56)  # Midship section coefficient
        self.C_WP = (1 + 2 * self.C_B) / 3  # Waterplane coefficient
        self.C_P = self.C_B / self.C_M  # Prismatic coefficient

        self.delta = self.C_B * self.L * self.B * self.T  # Water displacement

        self.lcb = -13.5 + 19.4 * self.C_P  # longitudinal center of buoyancy
        self.L_R = self.L * (1 - self.C_P + (0.06 * self.C_P * self.lcb) / (
                    4 * self.C_P - 1))  # length parameter reflecting the length of the run

        self.A_T = 0.2 * self.B * self.T  # transverse area of the transom
        # calculation for A_BT (cross-sectional area of the bulb at still water level [m^2]) depends on whether a ship has a bulb
        if self.bulbous_bow is None:
            self.A_BT = 0     # most inland ships do not have a bulb. So we assume A_BT=0.
        else:
            self.A_BT = self.C_BB * self.B * self.T * self.C_M  # calculate A_BT for seagoing ships having a bulb

        # Total wet area: S
        assert self.C_M >= 0, f'C_M should be positive: {self.C_M}'
        self.S = self.L * (2 * self.T + self.B) * np.sqrt(self.C_M) * (
                    0.453 + 0.4425 * self.C_B - 0.2862 * self.C_M - 0.003467 * (
                        self.B / self.T) + 0.3696 * self.C_WP)  + 2.38 * (self.A_BT / self.C_B)

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
        assert self.D > 0,  f'D should be > 0: {self.D}'

        # Friction coefficient based on CFD computations of Zeng et al. (2018), in deep water
        self.Cf_deep = 0.08169 / ((np.log10(self.R_e) - 1.717) ** 2)
        assert not isinstance(self.Cf_deep, complex),  f'Cf_deep should not be complex: {self.Cf_deep}'

        # Friction coefficient based on CFD computations of Zeng et al. (2018), taking into account shallow water effects
        self.Cf_shallow = (0.08169 / ((np.log10(self.R_e) - 1.717) ** 2)) * (
                    1 + (0.003998 / (np.log10(self.R_e) - 4.393)) * (self.D / self.L) ** (-1.083))
        assert not isinstance(self.Cf_shallow, complex),  f'Cf_shallow should not be complex: {self.Cf_shallow}'

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
            logger.debug(f'now i am in the deep loop')
        else:

            # calculate Friction coefficient C_f for shallow water:
            self.C_f = self.Cf_0 + (self.Cf_shallow - self.Cf_Katsui) * (self.S_B / self.S) * (self.V_B / v) ** 2
            logger.debug(f'now i am in the shallow loop')
        assert not isinstance(self.C_f, complex),  f'C_f should not be complex: {self.C_f}'

        # The total frictional resistance R_f [kN]:
        self.R_f = (self.C_f * 0.5 * self.rho * (v ** 2) * self.S) / 1000
        assert not isinstance(self.R_f, complex),  f'R_f should not be complex: {self.R_f}'

        return self.R_f

    def calculate_viscous_resistance(self):
        """Viscous resistance

        - 2nd resistance component defined by Holtrop and Mennen (1982)
        - Form factor (1 + k1) has to be multiplied by the frictional resistance R_f, to account for the effect of viscosity"""

        # c_14 accounts for the specific shape of the afterbody
        self.c_14 = 1 + 0.0011 * self.c_stern

        # the form factor (1+k1) describes the viscous resistance
        self.one_k1 = 0.93 + 0.487 * self.c_14 * ((self.B / self.L) ** 1.068) * ((self.T / self.L) ** 0.461) * (
                    (self.L / self.L_R) ** 0.122) * (((self.L ** 3) / self.delta) ** 0.365) * (
                                  (1 - self.C_P) ** (-0.604))
        self.R_f_one_k1 = self.R_f * self.one_k1
        return self.R_f_one_k1

    def calculate_appendage_resistance(self, v):
        """Appendage resistance

        - 3rd resistance component defined by Holtrop and Mennen (1982)
        - Appendages (like a rudder, shafts, skeg) result in additional frictional resistance"""

        # Frictional resistance resulting from wetted area of appendages: R_APP [kN]
        self.R_APP = (0.5 * self.rho * (v ** 2) * self.S_APP * self.one_k2 * self.C_f) / 1000

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
        assert self.g >= 0, f'g should be positive: {self.g}'
        assert h_0 >= 0, f'g should be positive: {h_0}'
        self.F_rh = v / np.sqrt(self.g * h_0)

        if self.F_rh <= 0.4:

            if 0 <= h_0 / self.T < 1.75:
                self.alpha_xx = (-4 * 10 ** (
                    -12)) * self.F_rh ** 3 - 0.2143 * self.F_rh ** 2 - 0.0643 * self.F_rh + 0.9997
            if 1.75 <= h_0 / self.T < 2.25:
                self.alpha_xx = -0.8333 * self.F_rh ** 3 + 0.25 * self.F_rh ** 2 - 0.0167 * self.F_rh + 1
            if 2.25 <= h_0 / self.T < 2.75:
                self.alpha_xx = -1.25 * self.F_rh ** 4 + 0.5833 * self.F_rh ** 3 - 0.0375 * self.F_rh ** 2 - 0.0108 * self.F_rh + 1
            if h_0 / self.T >= 2.75:
                self.alpha_xx = 1

        if self.F_rh > 0.4:
            if 0 <= h_0 / self.T < 1.75:
                self.alpha_xx = (-0.9274 * self.F_rh ** 6 + 9.5953 * self.F_rh ** 5 - 37.197 * self.F_rh ** 4 +
                69.666 * self.F_rh ** 3 - 65.391 * self.F_rh ** 2 + 28.025 * self.F_rh - 3.4143)
            if 1.75 <= h_0 / self.T < 2.25:
                self.alpha_xx = (2.2152 * self.F_rh ** 6 - 11.852 * self.F_rh ** 5 + 21.499 * self.F_rh ** 4 -
                12.174 * self.F_rh ** 3 - 4.7873 * self.F_rh ** 2 + 5.8662 * self.F_rh - 0.2652)
            if 2.25 <= h_0 / self.T < 2.75:
                self.alpha_xx = (1.2205 * self.F_rh ** 6 - 5.4999 * self.F_rh ** 5 + 5.7966 * self.F_rh ** 4 +
                6.6491 * self.F_rh ** 3 - 16.123 * self.F_rh ** 2 + 9.2016 * self.F_rh - 0.6342)
            if 2.75 <= h_0 / self.T < 3.25:
                self.alpha_xx = (-0.4085 * self.F_rh ** 6 + 4.534 * self.F_rh ** 5 - 18.443 * self.F_rh ** 4 +
                35.744 * self.F_rh ** 3 - 34.381 * self.F_rh ** 2 + 15.042 * self.F_rh - 1.3807)
            if 3.25 <= h_0 / self.T < 3.75:
                self.alpha_xx = (0.4078 * self.F_rh ** 6 - 0.919 * self.F_rh ** 5 - 3.8292 * self.F_rh ** 4 +
                15.738 * self.F_rh ** 3 - 19.766 * self.F_rh ** 2 + 9.7466 * self.F_rh - 0.6409)
            if 3.75 <= h_0 / self.T < 4.5:
                self.alpha_xx = (0.3067 * self.F_rh ** 6 - 0.3404 * self.F_rh ** 5 - 5.0511 * self.F_rh ** 4 +
                16.892 * self.F_rh ** 3 - 20.265 * self.F_rh ** 2 + 9.9002 * self.F_rh - 0.6712)
            if 4.5 <= h_0 / self.T < 5.5:
                self.alpha_xx = (0.3212 * self.F_rh ** 6 - 0.3559 * self.F_rh ** 5 - 5.1056 * self.F_rh ** 4 +
                16.926 * self.F_rh ** 3 - 20.253 * self.F_rh ** 2 + 10.013 * self.F_rh - 0.7196)
            if 5.5 <= h_0 / self.T < 6.5:
                self.alpha_xx = (0.9252 * self.F_rh ** 6 - 4.2574 * self.F_rh ** 5 + 5.0363 * self.F_rh ** 4 +
                3.3282 * self.F_rh ** 3 - 10.367 * self.F_rh ** 2 + 6.3993 * self.F_rh - 0.2074)
            if 6.5 <= h_0 / self.T < 7.5:
                self.alpha_xx = (0.8442 * self.F_rh ** 6 - 4.0261 * self.F_rh ** 5 + 5.313 * self.F_rh ** 4 +
                1.6442 * self.F_rh ** 3 - 8.1848 * self.F_rh ** 2 + 5.3209 * self.F_rh - 0.0267)
            if 7.5 <= h_0 / self.T < 8.5:
                self.alpha_xx = (0.1211 * self.F_rh ** 6 + 0.628 * self.F_rh ** 5 - 6.5106 * self.F_rh ** 4 +
                16.7 * self.F_rh ** 3 - 18.267 * self.F_rh ** 2 + 8.7077 * self.F_rh - 0.4745)

            if 8.5 <= h_0 / self.T < 9.5:
                if self.F_rh < 0.6:
                    self.alpha_xx = 1
                if self.F_rh >= 0.6:
                    self.alpha_xx = (-6.4069 * self.F_rh ** 6 + 47.308 * self.F_rh ** 5 - 141.93 * self.F_rh ** 4 +
                    220.23 * self.F_rh ** 3 - 185.05 * self.F_rh ** 2 + 79.25 * self.F_rh - 12.484)
            if h_0 / self.T >= 9.5:
                if self.F_rh < 0.6:
                    self.alpha_xx = 1
                if self.F_rh >= 0.6:
                    self.alpha_xx = (-6.0727 * self.F_rh ** 6 + 44.97 * self.F_rh ** 5 - 135.21 * self.F_rh ** 4 +
                    210.13 * self.F_rh ** 3 - 176.72 * self.F_rh ** 2 + 75.728 * self.F_rh - 11.893)

        self.V_2 = v / self.alpha_xx

    def calculate_wave_resistance(self, v, h_0):
        """Wave resistance

        - 4th resistance component defined by Holtrop and Mennen (1982)
        - When the speed or the vessel size increases, the wave making resistance increases
        - In shallow water, the wave resistance shows an asymptotical behaviour by reaching the critical speed
        """

        self.karpov(v, h_0)

        assert self.g >= 0, f'g should be positive: {self.g}'
        assert self.L >= 0, f'L should be positive: {self.L}'
        self.F_rL = self.V_2 / np.sqrt(self.g * self.L)  # Froude number based on ship's speed to water and its length of waterline

        # parameter c_7 is determined by the B/L ratio
        if self.B / self.L < 0.11:
            self.c_7 = 0.229577 * (self.B / self.L) ** 0.33333
        if self.B / self.L > 0.25:
            self.c_7 = 0.5 - 0.0625 * (self.L / self.B)
        else:
            self.c_7 = self.B / self.L

        # half angle of entrance in degrees
        self.i_E = 1 + 89 * np.exp(-((self.L / self.B) ** 0.80856) * ((1 - self.C_WP) ** 0.30484) * (
                    (1 - self.C_P - 0.0225 * self.lcb) ** 0.6367) * ((self.L_R / self.B) ** 0.34574) * (
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

        # parameter c_16 depends on C_P
        if self.C_P < 0.8:
            self.c_16 = 8.07981 * self.C_P - 13.8673 * (self.C_P ** 2) + 6.984388 * (self.C_P ** 3)
        else:
            self.c_16 = 1.73014 - 0.7067 * self.C_P

        if self.L / self.B < 12:
            self.lmbda = 1.446 * self.C_P - 0.03 * (self.L / self.B)
        else:
            self.lmbda = 1.446 * self.C_P - 0.36

        self.m_1 = 0.0140407 * (self.L / self.T) - 1.75254 * ((self.delta) ** (1 / 3) / self.L) - 4.79323 * (
                    self.B / self.L) - self.c_16
        self.m_2 = self.c_15 * (self.C_P**2) *np.exp((-0.1)* (self.F_rL**(-2)))

        self.R_W = self.c_1 * self.c_2 * self.c_5 * self.delta * self.rho * self.g * np.exp(self.m_1 * (self.F_rL**(-0.9)) +
                   self.m_2 * np.cos(self.lmbda * (self.F_rL ** (-2)))) / 1000 # kN

        return self.R_W



    def calculate_residual_resistance(self, v, h_0):
        """Residual resistance terms

        - Holtrop and Mennen (1982) defined three residual resistance terms:
        - 1) Resistance due to immersed transom (R_TR), Karpov corrected velocity V2 is used
        - 2) Resistance due to model-ship correlation (R_A), Karpov corrected velocity V2 is used
        - 3) Resistance due to the bulbous bow (R_B), Karpov corrected velocity V2 is used
        """

        self.karpov(v, h_0)

        # Resistance due to immersed transom: R_TR [kN]
        self.F_nT = self.V_2 / np.sqrt(
            2 * self.g * self.A_T / (self.B + self.B * self.C_WP))  # Froude number based on transom immersion
        assert not isinstance(self.F_nT, complex),  f'residual? froude number should not be complex: {self.F_nT}'


        self.c_6 = 0.2 * (1 - 0.2 * self.F_nT)  # Assuming F_nT < 5, this is the expression for coefficient c_6

        self.R_TR = (0.5 * self.rho * (self.V_2 ** 2) * self.A_T * self.c_6) / 1000

        # Model-ship correlation resistance: R_A [kN]

        if self.T / self.L < 0.04:
            self.c_4 = self.T / self.L
        else:
            self.c_4 = 0.04
        self.c_2 = 1

        self.C_A = 0.006 * (self.L + 100) ** (-0.16) - 0.00205 + 0.003 * np.sqrt(self.L / 7.5) * (
                    self.C_B ** 4) * self.c_2 * (0.04 - self.c_4)
        assert not isinstance(self.C_A, complex),  f'C_A number should not be complex: {self.C_A}'


        self.R_A = (0.5 * self.rho * (self.V_2 ** 2) * self.S * self.C_A) / 1000  # kW

        # Resistance due to the bulbous bow (R_B)

        # Froude number based on immersoin of bulbous bow [-]
        self.F_ni = (self.V_2 / np.sqrt( self.g * (self.T_F - self.h_B - 0.25 * np.sqrt(self.A_BT) + 0.15 * self.V_2**2)))

        self.P_B = (0.56 * np.sqrt(self.A_BT)) / (self.T_F - 1.5 * self.h_B) #P_B is coefficient for the emergence of bulbous bow
        if self.bulbous_bow is None:
            self.R_B = 0
        else:
            self.R_B = ((0.11 * np.exp(-3 * self.P_B**2) * self.F_ni**3 * self.A_BT**1.5 * self.rho * self.g) / (1+ self.F_ni**2)) / 1000

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

    def calculate_total_power_required(self, v):
        """Total required power:

        - The total required power is the sum of the power for systems on board (P_hotel) + power required for propulsion (P_b)
        - The P_b depends on the calculated resistance
        """

        # Required power for systems on board, "5%" based on De Vos and van Gils (2011):Walstrom versus generators troom
        self.P_hotel = 0.05 * self.P_installed

        # Required power for propulsion
        # Effective Horse Power (EHP), P_e
        self.P_e = v * self.R_tot

        # Calculation hull efficiency
        dw = np.zeros(101)  # velocity correction coefficient
        counter = 0

        if self.F_rL < 0.2:
            self.dw = 0
        else:
            self.dw = 0.1

        self.w = 0.11 * (0.16 / self.x) * self.C_B * np.sqrt(
            (self.delta ** (1 / 3)) / self.D_s) - self.dw  # wake fraction 'w'

        assert not isinstance(self.w, complex),  f'w should not be complex: {self.w}'


        if self.x == 1:
            self.t = 0.6 * self.w * (1 + 0.67 * self.w)  # thrust deduction factor 't'
        else:
            self.t = 0.8 * self.w * (1 + 0.25 * self.w)

        self.eta_h = (1 - self.t) / (1 - self.w)  # hull efficiency eta_h

        # Delivered Horse Power (DHP), P_d

        self.P_d = self.P_e / (self.eta_o * self.eta_r * self.eta_h)

        # Brake Horse Power (BHP), P_b
        self.P_b = self.P_d / (self.eta_t * self.eta_g)
        self.P_propulsion = self.P_b    # propulsion power is brake horse power

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

        assert not isinstance(self.P_given, complex),  f'P_given number should not be complex: {self.P_given}'

        # return to the power given by the engine to the ship (for hotelling and propulsion), which is the actual power the ship uses
        return self.P_given



    def emission_factors_general(self):
        """General emission factors:

        This function computes general emission factors, based on construction year of the engine.
        - Based on literature TNO (2019)

        Please note: later on a correction factor has to be applied to get the total emission factor
        """

        # The general emission factors of CO2, PM10 and NOX, and SFC are based on the construction year of the engine

        if self.C_year < 1974:
            self.EF_CO2 = 756
            self.EF_PM10 = 0.6
            self.EF_NOX = 10.8
            self.SFC = 235
        if 1975 <= self.C_year <= 1979:
            self.EF_CO2 = 730
            self.EF_PM10 = 0.6
            self.EF_NOX = 10.6
            self.SFC = 230
        if 1980 <= self.C_year <= 1984:
            self.EF_CO2 = 714
            self.EF_PM10 = 0.6
            self.EF_NOX = 10.4
            self.SFC = 225
        if 1985 <= self.C_year <= 1989:
            self.EF_CO2 = 698
            self.EF_PM10 = 0.5
            self.EF_NOX = 10.1
            self.SFC = 220
        if 1990 <= self.C_year <= 1994:
            self.EF_CO2 = 698
            self.EF_PM10 = 0.4
            self.EF_NOX = 10.1
            self.SFC = 220
        if 1995 <= self.C_year <= 2002:
            self.EF_CO2 = 650
            self.EF_PM10 = 0.3
            self.EF_NOX = 9.4
            self.SFC = 205
        if 2003 <= self.C_year <= 2007:
            self.EF_CO2 = 635
            self.EF_PM10 = 0.3
            self.EF_NOX = 9.2
            self.SFC = 200
        if 2008 <= self.C_year <= 2019:
            self.EF_CO2 = 635
            self.EF_PM10 = 0.2
            self.EF_NOX = 7
            self.SFC = 200
        if self.C_year > 2019:
            if self.L_w == 1:
                self.EF_CO2 = 650
                self.EF_PM10 = 0.1
                self.EF_NOX = 2.1
                self.SFC = 205
            else:
                self.EF_CO2 = 603
                self.EF_PM10 = 0.015
                self.EF_NOX = 1.8
                self.SFC = 190

        logger.debug(f'The general emission factor of CO2 is {self.EF_CO2} g/kWh')
        logger.debug(f'The general emission factor of PM10 is {self.EF_PM10} g/kWh')
        logger.debug(f'The general emission factor CO2 is {self.EF_NOX} g/kWh')
        logger.debug(f'The general fuel consumption factor for diesel is {self.SFC} g/kWh')

    def correction_factors(self, v):
        """ Partial engine load correction factors (C_partial_load):

        - The correction factors have to be multiplied by the general emission factors, to get the total emission factors
        - The correction factor takes into account the effect of the partial engine load
        - When the partial engine load is low, the correction factors are higher (engine is less efficient)
        - Based on literature TNO (2019)
        """
        #TODO: implement the case where v=None
        self.calculate_total_power_required(v=v)  # You need the P_partial values

        # Import the correction factors table
        # TODO: use package data, not an arbitrary location
        self.C_partial_load = opentnsim.energy.load_partial_engine_load_correction_factors()

        for i in range(20):
            # If the partial engine load is smaller or equal to 5%, the correction factors corresponding to P_partial = 5% are assigned.
            if self.P_partial <= self.C_partial_load.iloc[0, 0]:
                self.C_partial_load_CO2 = self.C_partial_load.iloc[0, 5]
                self.C_partial_load_PM10 = self.C_partial_load.iloc[0, 6]
                self.C_partial_load_fuel = self.C_partial_load_CO2 # CO2 emission is generated from fuel consumption, so these two
                                                                   # correction factors are equal

                # The NOX correction factors are dependend on the construction year of the engine and the weight class
                if self.C_year < 2008:
                    self.C_partial_load_NOX = self.C_partial_load.iloc[0, 1]  # <= CCR-1 class
                if 2008 <= self.C_year <= 2019:
                    self.C_partial_load_NOX = self.C_partial_load.iloc[0, 2]  # CCR-2 / Stage IIIa
                if self.C_year > 2019:
                    if self.L_w == 1:  #
                        self.C_partial_load_NOX = self.C_partial_load.iloc[
                            0, 3]  # Stage V: IWP/IWA-v/c-3 class (vessels with P <300 kW: assumed to be weight class L1)
                    else:
                        self.C_partial_load_NOX = self.C_partial_load.iloc[
                            0, 4]  # Stage V:IWP/IWA-v/c-4 class (vessels with P >300 kw: assumed to be weight class L2-L3)

            # If the partial engine load is greater than 5%:
            # It is determined inbetween which two percentages in the table the partial engine load lies
            # The correction factor is determined by means of linear interpolation

            elif self.C_partial_load.iloc[i, 0] < self.P_partial <= self.C_partial_load.iloc[i + 1, 0]:
                self.C_partial_load_CO2 = ((self.P_partial - self.C_partial_load.iloc[i, 0]) * (
                            self.C_partial_load.iloc[i + 1, 5] - self.C_partial_load.iloc[i, 5])) / (
                                self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) + self.C_partial_load.iloc[i, 5]
                self.C_partial_load_PM10 = ((self.P_partial - self.C_partial_load.iloc[i, 0]) * (
                            self.C_partial_load.iloc[i + 1, 6] - self.C_partial_load.iloc[i, 6])) / (
                                self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) + self.C_partial_load.iloc[i, 6]
                self.C_partial_load_fuel = self.C_partial_load_CO2 # CO2 emission is generated from fuel consumption, so these two
                                                                   # correction factors are equal

                if self.C_year < 2008:
                    self.C_partial_load_NOX = ((self.P_partial - self.C_partial_load.iloc[i, 0]) * (
                                self.C_partial_load.iloc[i + 1, 1] - self.C_partial_load.iloc[i, 1])) / (
                                                self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) +                                                                             self.C_partial_load.iloc[i, 1]
                if 2008 <= self.C_year <= 2019:
                    self.C_partial_load_NOX = ((self.P_partial - self.C_partial_load.iloc[i, 0]) * (
                                self.C_partial_load.iloc[i + 1, 2] - self.C_partial_load.iloc[i, 2])) / (
                                                self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) +                                                                             self.C_partial_load.iloc[i, 2]
                if self.C_year > 2019:
                    if self.L_w == 1:
                        self.C_partial_load_NOX = ((self.P_partial - self.C_partial_load.iloc[i, 0]) * (
                                    self.C_partial_load.iloc[i + 1, 3] - self.C_partial_load.iloc[i, 3])) / (
                                                self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) +                                                                             self.C_partial_load.iloc[i, 3]
                    else:
                        self.C_partial_load_NOX = ((self.P_partial - self.C_partial_load.iloc[i, 0]) * (
                                    self.C_partial_load.iloc[i + 1, 4] - self.C_partial_load.iloc[i, 4])) / (
                                                self.C_partial_load.iloc[i + 1, 0] - self.C_partial_load.iloc[i, 0]) +                                                                             self.C_partial_load.iloc[i, 4]

            # If the partial engine load is => 100%, the correction factors corresponding to P_partial = 100% are assigned.
            elif self.P_partial >= self.C_partial_load.iloc[19, 0]:
                self.C_partial_load_CO2 = self.C_partial_load.iloc[19, 5]
                self.C_partial_load_PM10 = self.C_partial_load.iloc[19, 6]
                self.C_partial_load_fuel = self.C_partial_load_CO2 # CO2 emission is generated from fuel consumption, so these two
                                                                   # correction factors are equal

                # The NOX correction factors are dependend on the construction year of the engine and the weight class
                if self.C_year < 2008:
                    self.C_partial_load_NOX = self.C_partial_load.iloc[19, 1]  # <= CCR-1 class
                if 2008 <= self.C_year <= 2019:
                    self.C_partial_load_NOX = self.C_partial_load.iloc[19, 2]  # CCR-2 / Stage IIIa
                if self.C_year > 2019:
                    if self.L_w == 1:  #
                        self.C_partial_load_NOX = self.C_partial_load.iloc[
                            19, 3]  # Stage V: IWP/IWA-v/c-3 class (vessels with P <300 kW: assumed to be weight class L1)
                    else:
                        self.C_partial_load_NOX = self.C_partial_load.iloc[
                            19, 4]  # Stage V:IWP/IWA-v/c-4 class (vessels with P >300 kw: assumed to be weight class L2-L3)

        logger.debug(f'Partial engine load correction factor of CO2 is {self.C_partial_load_CO2}')
        logger.debug(f'Partial engine load correction factor of PM10 is {self.C_partial_load_PM10}')
        logger.debug(f'Partial engine load correction factor of NOX is {self.C_partial_load_NOX}')
        logger.debug(f'Partial engine load correction factor of diesel fuel consumption is {self.C_partial_load_fuel}')

    def calculate_emission_factors_total(self, v):
        """Total emission factors:

        - The total emission factors can be computed by multiplying the general emission factor by the correction factor
        """

        self.emission_factors_general()  # You need the values of the general emission factors of CO2, PM10, NOX
        self.correction_factors(v=v)  # You need the correction factors of CO2, PM10, NOX

        # The total emission factor is calculated by multiplying the general emission factor (EF_CO2 / EF_PM10 / EF_NOX)
        # By the correction factor (C_partial_load_CO2 / C_partial_load_PM10 / C_partial_load_NOX)

        self.total_factor_CO2 = self.EF_CO2 * self.C_partial_load_CO2
        self.total_factor_PM10 = self.EF_PM10 * self.C_partial_load_PM10
        self.total_factor_NOX = self.EF_NOX * self.C_partial_load_NOX
        self.total_factor_FU = self.SFC * self.C_partial_load_fuel

        logger.debug(f'The total emission factor of CO2 is {self.total_factor_CO2} g/kWh')
        logger.debug(f'The total emission factor of PM10 is {self.total_factor_PM10} g/kWh')
        logger.debug(f'The total emission factor CO2 is {self.total_factor_NOX} g/kWh')
        logger.debug(f'The total fuel use factor for diesel is {self.total_factor_FU} g/kWh')



    def calculate_fuel_use_g_m(self,v):
        """Total fuel use in g/m:

        - The total fuel use in g/m can be computed by total fuel use in g (P_tot * delt_t * self.total_factor_FU) diveded by the sailing distance (v * delt_t)
        """
        self.fuel_use_g_m = (self.P_given * self.total_factor_FU / v ) / 3600
        return self.fuel_use_g_m


    def calculate_fuel_use_g_s(self):
        """Total fuel use in g/s:

       - The total fuel use in g/s can be computed by total emission in g (P_tot * delt_t * self.total_factor_FU) diveded by the sailing duration (delt_t)
       """
        self.fuel_use_g_m = self.P_given * self.total_factor_FU / 3600
        return self.fuel_use_g_s


    def calculate_emission_rates_g_m(self,v):
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



class Routeable:
    """Mixin class: Something with a route (networkx format)

    - route: a networkx path
    """

    def __init__(self, route, complete_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.route = route
        self.complete_path = complete_path

class IsLockWaitingArea(HasResource, Identifiable, Log):
    """Mixin class: Something has lock object properties

    - properties in meters
    - operation in seconds
    """

    def __init__(
        self,
        node,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        """Initialization
        """

        waiting_area_resources = 100
        self.waiting_area = {
            node: simpy.PriorityResource(self.env, capacity=waiting_area_resources),
        }

        #departure_resources = 4
        #self.departure = {
        #    node: simpy.PriorityResource(self.env, capacity=departure_resources),
        #}

class IsLockLineUpArea(HasResource, HasLength, Identifiable, Log):
    """Mixin class: Something has lock object properties
    - properties in meters
    - operation in seconds
    """

    def __init__(
        self,
        node,
        lineup_length,
        *args,
        **kwargs
    ):
        super().__init__(length = lineup_length, remaining_length = lineup_length, *args, **kwargs)
        """Initialization"""

        self.lock_queue_length = 0

        # Lay-Out
        self.enter_line_up_area = {
            node: simpy.PriorityResource(self.env, capacity=1),
        }

        self.line_up_area = {
            node: simpy.PriorityResource(self.env, capacity=100),
        }

        self.converting_while_in_line_up_area = {
            node: simpy.PriorityResource(self.env, capacity=1),
        }

        self.pass_line_up_area = {
            node: simpy.PriorityResource(self.env, capacity=1),
        }

class HasLockDoors(SimpyObject):

     def __init__(
        self,
        node_1,
        node_3,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        """Initialization
        """

        self.doors_1 = {
            node_1: simpy.PriorityResource(self.env, capacity = 1),
        }
        self.doors_2 = {
            node_3: simpy.PriorityResource(self.env, capacity = 1),
        }

class IsLock(HasResource, HasLength, HasLockDoors, Identifiable, Log):
    """Mixin class: Something has lock object properties
    - properties in meters
    - operation in seconds
    """

    def __init__(
        self,
        node_1,
        node_2,
        node_3,
        lock_length,
        lock_width,
        lock_depth,
        doors_open,
        doors_close,
        wlev_dif,
        disch_coeff,
        grav_acc,
        opening_area,
        opening_depth,
        simulation_start,
        operating_time,
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
        self.simulation_start = simulation_start.timestamp()
        self.operating_time = operating_time

        # Operating
        self.doors_open = doors_open
        self.doors_close = doors_close

        # Water level
        assert node_1 != node_3

        self.node_1 = node_1
        self.node_3 = node_3
        self.water_level = random.choice([node_1, node_3])

        super().__init__(length = lock_length, remaining_length = lock_length, node_1 = node_1, node_3 = node_3, *args, **kwargs)

    def operation_time(self, environment):
        if type(self.wlev_dif) == list:
            operating_time = (2*self.lock_width*self.lock_length*abs(self.wlev_dif[1][np.abs(self.wlev_dif[0]-(environment.now-self.simulation_start)).argmin()]))/(self.disch_coeff*self.opening_area*math.sqrt(2*self.grav_acc*self.opening_depth))

        elif type(self.wlev_dif) == float or type(self.wlev_dif) == int:
            operating_time = (2*self.lock_width*self.lock_length*abs(self.wlev_dif))/(self.disch_coeff*self.opening_area*math.sqrt(2*self.grav_acc*self.opening_depth))
        assert not isinstance(operating_time, complex),  f'operating_time number should not be complex: {operating_time}'


        return operating_time

    def convert_chamber(self, environment, new_level, number_of_vessels):
        """ Convert the water level """

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
        """ Change water level and priorities in queue """

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

class Movable(Locatable, Routeable, Log):
    """Mixin class: Something can move

    Used for object that can move with a fixed speed

    - geometry: point used to track its current location
    - v: speed
    """

    def __init__(self, v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.v = v
        self.edge_functions = []
        self.wgs84 = pyproj.Geod(ellps="WGS84")


    def move(self):
        """determine distance between origin and destination, and
        yield the time it takes to travel it
        Assumption is that self.path is in the right order - vessel moves from route[0] to route[-1].
        """
        self.distance = 0
        speed = self.v

        # Check if vessel is at correct location - if not, move to location
        if (
            self.geometry
            != nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]
        ):
            orig = self.geometry
            dest = nx.get_node_attributes(self.env.FG, "geometry")[self.route[0]]

            logger.debug("Origin: {orig}")
            logger.debug("Destination: {dest}")

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


            if "Waiting area" in self.env.FG.nodes[destination].keys():
                locks = self.env.FG.nodes[destination]["Waiting area"]
                for lock in locks:
                    loc = self.route.index(destination)
                    for r in self.route[loc:]:
                        if 'Line-up area' in self.env.FG.nodes[r].keys():
                            wait_for_waiting_area = self.env.now
                            access_waiting_area = lock.waiting_area[destination].request()
                            yield access_waiting_area

                            if wait_for_waiting_area != self.env.now:
                                waiting = self.env.now - wait_for_waiting_area
                                self.log_entry("Waiting to enter waiting area start", wait_for_waiting_area, 0, nx.get_node_attributes(self.env.FG, "geometry")[origin],)
                                self.log_entry("Waiting to enter waiting area stop", self.env.now, waiting, nx.get_node_attributes(self.env.FG, "geometry")[origin],)

            if "Waiting area" in self.env.FG.nodes[origin].keys():
                locks = self.env.FG.nodes[origin]["Waiting area"]
                for lock in locks:
                    loc = self.route.index(origin)
                    for r in self.route[loc:]:
                        if 'Line-up area' in self.env.FG.nodes[r].keys():
                            locks2 = self.env.FG.nodes[r]["Line-up area"]
                            for r2 in self.route[loc:]:
                                if 'Lock' in self.env.FG.nodes[r2].keys():
                                    locks3 = self.env.FG.nodes[r2]["Lock"]
                                    break

                            self.lock_name = []
                            for lock3 in locks3:
                                if lock3.water_level == self.route[self.route.index(r2)-1]:
                                    for lock2 in locks2:
                                        if lock2.name == lock3.name:
                                            if lock2.lock_queue_length == 0:
                                                self.lock_name = lock3.name
                                        break

                            lock_queue_length = [];
                            if self.lock_name == []:
                                for lock2 in locks2:
                                    lock_queue_length.append(lock2.lock_queue_length)

                                self.lock_name = locks2[lock_queue_length.index(min(lock_queue_length))].name

                            for lock2 in locks2:
                                if lock2.name == self.lock_name:
                                    lock2.lock_queue_length += 1

                            for lock2 in locks2:
                                if lock2.name == self.lock_name:
                                    self.v = 0.5*speed
                                    break

                            wait_for_lineup_area = self.env.now
                            lock.waiting_area[origin].release(access_waiting_area)

                            if self.route[self.route.index(r2)-1] == lock3.node_1:
                                if lock3.doors_2[lock3.node_3].users != [] and lock3.doors_2[lock3.node_3].users[0].priority == -1:
                                    if self.L < lock2.length.level + lock3.length.level:
                                        access_lineup_length = lock2.length.get(self.L)
                                    elif self.L < lock2.length.level:
                                        if lock2.length.level == lock2.length.capacity:
                                            access_lineup_length = lock2.length.get(self.L)
                                        elif lock2.line_up_area[r].users != [] and lock3.length.level < lock2.line_up_area[r].users[0].length:
                                            access_lineup_length = lock2.length.get(self.L)
                                        else:
                                            if lock2.length.get_queue == []:
                                                access_lineup_length = lock2.length.get(lock2.length.capacity)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length
                                                correct_lineup_length = lock2.length.put(lock2.length.capacity-self.L)
                                            else:
                                                total_length_waiting_vessels = 0
                                                for q in reversed(range(len(lock2.length.get_queue))):
                                                    if lock2.length.get_queue[q].amount == lock2.length.capacity:
                                                        break
                                                for q2 in range(q,len(lock2.length.get_queue)):
                                                    total_length_waiting_vessels += lock2.length.get_queue[q2].length

                                                if self.L > lock2.length.capacity - total_length_waiting_vessels:
                                                    access_lineup_length = lock2.length.get(lock2.length.capacity)
                                                    lock2.length.get_queue[-1].length = self.L
                                                    yield access_lineup_length
                                                    correct_lineup_length = lock2.length.put(lock2.length.capacity-self.L)
                                                else:
                                                    access_lineup_length = lock2.length.get(self.L)
                                                    lock2.length.get_queue[-1].length = self.L
                                                    yield access_lineup_length
                                    else:
                                        if lock2.length.get_queue == []:
                                            access_lineup_length = lock2.length.get(lock2.length.capacity)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(lock2.length.capacity-self.L)
                                        else:
                                            total_length_waiting_vessels = 0
                                            for q in reversed(range(len(lock2.length.get_queue))):
                                                if lock2.length.get_queue[q].amount == lock2.length.capacity:
                                                    break
                                            for q2 in range(q,len(lock2.length.get_queue)):
                                                total_length_waiting_vessels += lock2.length.get_queue[q2].length

                                            if self.L > lock2.length.capacity - total_length_waiting_vessels:
                                                access_lineup_length = lock2.length.get(lock2.length.capacity)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length
                                                correct_lineup_length = lock2.length.put(lock2.length.capacity-self.L)
                                            else:
                                                access_lineup_length = lock2.length.get(self.L)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length

                                else:
                                    if lock2.length.level == lock2.length.capacity:
                                        access_lineup_length = lock2.length.get(self.L)
                                    elif lock2.line_up_area[r].users != [] and self.L < lock2.line_up_area[r].users[-1].lineup_dist-0.5*lock2.line_up_area[r].users[-1].length:
                                        access_lineup_length = lock2.length.get(self.L)
                                    else:
                                        if lock2.length.get_queue == []:
                                            access_lineup_length = lock2.length.get(lock2.length.capacity)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(lock2.length.capacity-self.L)
                                        else:
                                            total_length_waiting_vessels = 0
                                            for q in reversed(range(len(lock2.length.get_queue))):
                                                if lock2.length.get_queue[q].amount == lock2.length.capacity:
                                                    break
                                            for q2 in range(q,len(lock2.length.get_queue)):
                                                total_length_waiting_vessels += lock2.length.get_queue[q2].length

                                            if self.L > lock2.length.capacity - total_length_waiting_vessels:
                                                access_lineup_length = lock2.length.get(lock2.length.capacity)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length
                                                correct_lineup_length = lock2.length.put(lock2.length.capacity-self.L)
                                            else:
                                                access_lineup_length = lock2.length.get(self.L)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length

                            elif self.route[self.route.index(r2)-1] == lock3.node_3:
                                if lock3.doors_1[lock3.node_1].users != [] and lock3.doors_1[lock3.node_1].users[0].priority == -1:
                                    if self.L < lock2.length.level + lock3.length.level:
                                        access_lineup_length = lock2.length.get(self.L)
                                    elif self.L < lock2.length.level:
                                        if lock2.length.level == lock2.length.capacity:
                                            access_lineup_length = lock2.length.get(self.L)
                                        elif lock2.line_up_area[r].users != [] and lock3.length.level < lock2.line_up_area[r].users[0].length:
                                            access_lineup_length = lock2.length.get(self.L)
                                        else:
                                            if lock2.length.get_queue == []:
                                                access_lineup_length = lock2.length.get(lock2.length.capacity)
                                                yield access_lineup_length
                                                correct_lineup_length = lock2.length.put(lock2.length.capacity-self.L)
                                                yield correct_lineup_length
                                            else:
                                                total_length_waiting_vessels = 0
                                                for q in reversed(range(len(lock2.length.get_queue))):
                                                    if lock2.length.get_queue[q].amount == lock2.length.capacity:
                                                        break
                                                for q2 in range(q,len(lock2.length.get_queue)):
                                                    total_length_waiting_vessels += lock2.length.get_queue[q2].length

                                                if self.L > lock2.length.capacity - total_length_waiting_vessels:
                                                    access_lineup_length = lock2.length.get(lock2.length.capacity)
                                                    lock2.length.get_queue[-1].length = self.L
                                                    yield access_lineup_length
                                                    correct_lineup_length = lock2.length.put(lock2.length.capacity-self.L)
                                                else:
                                                    access_lineup_length = lock2.length.get(self.L)
                                                    lock2.length.get_queue[-1].length = self.L
                                                    yield access_lineup_length
                                    else:
                                        if lock2.length.get_queue == []:
                                            access_lineup_length = lock2.length.get(lock2.length.capacity)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(lock2.length.capacity-self.L)
                                        else:
                                            total_length_waiting_vessels = 0
                                            for q in reversed(range(len(lock2.length.get_queue))):
                                                if lock2.length.get_queue[q].amount == lock2.length.capacity:
                                                    break
                                            for q2 in range(q,len(lock2.length.get_queue)):
                                                total_length_waiting_vessels += lock2.length.get_queue[q2].length

                                            if self.L > lock2.length.capacity - total_length_waiting_vessels:
                                                access_lineup_length = lock2.length.get(lock2.length.capacity)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length
                                                correct_lineup_length = lock2.length.put(lock2.length.capacity-self.L)
                                            else:
                                                access_lineup_length = lock2.length.get(self.L)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length
                                else:
                                    if lock2.length.level == lock2.length.capacity:
                                        access_lineup_length = lock2.length.get(self.L)
                                    elif lock2.line_up_area[r].users != [] and self.L < lock2.line_up_area[r].users[-1].lineup_dist-0.5*lock2.line_up_area[r].users[-1].length:
                                        access_lineup_length = lock2.length.get(self.L)
                                    else:
                                        if lock2.length.get_queue == []:
                                            access_lineup_length = lock2.length.get(lock2.length.capacity)
                                            lock2.length.get_queue[-1].length = self.L
                                            yield access_lineup_length
                                            correct_lineup_length = lock2.length.put(lock2.length.capacity-self.L)
                                        else:
                                            total_length_waiting_vessels = 0
                                            for q in reversed(range(len(lock2.length.get_queue))):
                                                if lock2.length.get_queue[q].amount == lock2.length.capacity:
                                                    break
                                            for q2 in range(q,len(lock2.length.get_queue)):
                                                total_length_waiting_vessels += lock2.length.get_queue[q2].length

                                            if self.L > lock2.length.capacity - total_length_waiting_vessels:
                                                access_lineup_length = lock2.length.get(lock2.length.capacity)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length
                                                correct_lineup_length = lock2.length.put(lock2.length.capacity-self.L)
                                            else:
                                                access_lineup_length = lock2.length.get(self.L)
                                                lock2.length.get_queue[-1].length = self.L
                                                yield access_lineup_length

                            if len(lock2.line_up_area[r].users) != 0:
                                self.lineup_dist = lock2.line_up_area[r].users[-1].lineup_dist - 0.5*lock2.line_up_area[r].users[-1].length - 0.5*self.L
                            else:
                                self.lineup_dist = lock2.length.capacity - 0.5*self.L

                            self.wgs84 = pyproj.Geod(ellps="WGS84")
                            [lineup_area_start_lat, lineup_area_start_lon, lineup_area_stop_lat, lineup_area_stop_lon] = [self.env.FG.nodes[self.route[self.route.index(r)]]['geometry'].x, self.env.FG.nodes[self.route[self.route.index(r)]]['geometry'].y,
                                                                                                                          self.env.FG.nodes[self.route[self.route.index(r)+1]]['geometry'].x, self.env.FG.nodes[self.route[self.route.index(r)+1]]['geometry'].y]
                            fwd_azimuth,_,_ = self.wgs84.inv(lineup_area_start_lat, lineup_area_start_lon, lineup_area_stop_lat, lineup_area_stop_lon)
                            [self.lineup_pos_lat,self.lineup_pos_lon,_] = self.wgs84.fwd(self.env.FG.nodes[self.route[self.route.index(r)]]['geometry'].x,
                                                                                         self.env.FG.nodes[self.route[self.route.index(r)]]['geometry'].y,
                                                                                         fwd_azimuth,self.lineup_dist)

                            access_lineup_area = lock2.line_up_area[r].request()
                            lock2.line_up_area[r].users[-1].length = self.L
                            lock2.line_up_area[r].users[-1].id = self.id
                            lock2.line_up_area[r].users[-1].lineup_pos_lat = self.lineup_pos_lat
                            lock2.line_up_area[r].users[-1].lineup_pos_lon = self.lineup_pos_lon
                            lock2.line_up_area[r].users[-1].lineup_dist = self.lineup_dist
                            lock2.line_up_area[r].users[-1].n = len(lock2.line_up_area[r].users)
                            lock2.line_up_area[r].users[-1].v = 0.25*speed
                            lock2.line_up_area[r].users[-1].wait_for_next_cycle = False
                            yield access_lineup_area

                            enter_lineup_length = lock2.enter_line_up_area[r].request()
                            yield enter_lineup_length
                            lock2.enter_line_up_area[r].users[0].id = self.id

                            if wait_for_lineup_area != self.env.now:
                                self.v = 0.25*speed
                                waiting = self.env.now - wait_for_lineup_area
                                self.log_entry("Waiting in waiting area start", wait_for_lineup_area, 0, nx.get_node_attributes(self.env.FG, "geometry")[origin])
                                self.log_entry("Waiting in waiting area stop", self.env.now, waiting, nx.get_node_attributes(self.env.FG, "geometry")[origin])
                            break

            if "Line-up area" in self.env.FG.nodes[destination].keys():
                locks = self.env.FG.nodes[destination]["Line-up area"]
                for lock in locks:
                    if lock.name == self.lock_name:
                        loc = self.route.index(destination)
                        orig = shapely.geometry.Point(self.lineup_pos_lat,self.lineup_pos_lon)
                        for r in self.route[loc:]:
                            if 'Lock' in self.env.FG.nodes[r].keys():
                                locks = self.env.FG.nodes[r]["Lock"]
                                for lock2 in locks:
                                    for q in range(len(lock.line_up_area[destination].users)):
                                        if lock.line_up_area[destination].users[q].id == self.id:
                                            if self.route[self.route.index(r)-1] == lock2.node_1:
                                                if lock2.doors_2[lock2.node_3].users != [] and lock2.doors_2[lock2.node_3].users[0].priority == -1:
                                                    if q <= 1 and lock.line_up_area[destination].users[q].n != lock.line_up_area[destination].users[q].n-len(lock2.resource.users):
                                                        self.lineup_dist = lock.length.capacity - 0.5*self.L
                                            elif self.route[self.route.index(r)-1] == lock2.node_3:
                                                if lock2.doors_1[lock2.node_1].users != [] and lock2.doors_1[lock2.node_1].users[0].priority == -1:
                                                    if q <= 1 and lock.line_up_area[destination].users[q].n != lock.line_up_area[destination].users[q].n-len(lock2.resource.users):
                                                        self.lineup_dist = lock.length.capacity - 0.5*self.L
                                            [self.lineup_pos_lat,self.lineup_pos_lon,_] = self.wgs84.fwd(self.env.FG.nodes[self.route[self.route.index(destination)]]['geometry'].x,
                                                                                                         self.env.FG.nodes[self.route[self.route.index(destination)]]['geometry'].y,
                                                                                                         fwd_azimuth,self.lineup_dist)
                                            lock.line_up_area[destination].users[q].lineup_pos_lat = self.lineup_pos_lat
                                            lock.line_up_area[destination].users[q].lineup_pos_lon = self.lineup_pos_lon
                                            lock.line_up_area[destination].users[q].lineup_dist = self.lineup_dist
                                            break

            if "Line-up area" in self.env.FG.nodes[origin].keys():
                locks = self.env.FG.nodes[origin]["Line-up area"]
                for lock in locks:
                    if lock.name == self.lock_name:
                        loc = self.route.index(origin)
                        orig = shapely.geometry.Point(self.lineup_pos_lat,self.lineup_pos_lon)
                        for r in self.route[loc:]:
                            if 'Lock' in self.env.FG.nodes[r].keys():
                                locks = self.env.FG.nodes[r]["Lock"]
                                lock.enter_line_up_area[origin].release(enter_lineup_length)
                                for q in range(len(lock.line_up_area[origin].users)):
                                    if lock.line_up_area[origin].users[q].id == self.id:
                                        if q > 0:
                                            _,_,distance = self.wgs84.inv(orig.x,
                                                                          orig.y,
                                                                          lock.line_up_area[origin].users[0].lineup_pos_lat,
                                                                          lock.line_up_area[origin].users[0].lineup_pos_lon)
                                            yield self.env.timeout(distance/self.v)
                                            break

                                for lock2 in locks:
                                    if lock2.name == self.lock_name:
                                        self.v = 0.25*speed
                                        wait_for_lock_entry = self.env.now

                                        for r2 in self.route[(loc+1):]:
                                            if 'Line-up area' in self.env.FG.nodes[r2].keys():
                                                locks = self.env.FG.nodes[r2]["Line-up area"]
                                                for lock3 in locks:
                                                    if lock3.name == self.lock_name:
                                                        break
                                                break

                                        if self.route[self.route.index(r)-1] == lock2.node_1:
                                            if len(lock2.doors_2[lock2.node_3].users) != 0:
                                                if lock2.doors_2[lock2.node_3].users[0].priority == -1:
                                                    if self.L > (lock2.resource.users[-1].lock_dist-0.5*lock2.resource.users[-1].length) or lock2.resource.users[-1].converting == True:
                                                        access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority = -1)
                                                        yield access_lock_door2
                                                        lock2.doors_2[lock2.node_3].release(access_lock_door2)

                                                        wait_for_next_cycle = lock3.pass_line_up_area[r2].request()
                                                        yield wait_for_next_cycle
                                                        lock3.pass_line_up_area[r2].release(wait_for_next_cycle)

                                                    if lock.converting_while_in_line_up_area[origin].users != []:
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[origin].request(priority = -1)
                                                        yield waiting_during_converting
                                                        lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)

                                                    elif (len(lock2.doors_1[lock2.node_1].users) == 0 or (len(lock2.doors_1[lock2.node_1].users) != 0 and lock2.doors_1[lock2.node_1].users[0].priority != -1)) and self.route[self.route.index(r)-1] != lock2.water_level:
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[origin].request()
                                                        yield waiting_during_converting
                                                        yield from lock2.convert_chamber(self.env, self.route[self.route.index(r)-1], 0)
                                                        lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)

                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                    yield access_lock_door1

                                                    if lock2.doors_2[lock2.node_3].users != [] and lock2.doors_2[lock2.node_3].users[0].priority == -1:
                                                        access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority = -1)
                                                        lock2.doors_2[lock2.node_3].release(lock2.doors_2[lock2.node_3].users[0])
                                                        yield access_lock_door2
                                                        lock2.doors_2[lock2.node_3].users[0].id = self.id
                                                    else:
                                                        access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority = -1)
                                                        yield access_lock_door2
                                                        lock2.doors_2[lock2.node_3].users[0].id = self.id

                                                else:
                                                    if lock3.converting_while_in_line_up_area[r2].users != []:
                                                        waiting_during_converting = lock3.converting_while_in_line_up_area[r2].request()
                                                        yield waiting_during_converting
                                                        lock3.converting_while_in_line_up_area[r2].release(waiting_during_converting)

                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                    yield access_lock_door1

                                                    if lock2.doors_2[lock2.node_3].users != [] and lock2.doors_2[lock2.node_3].users[0].priority == -1:
                                                        access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority = -1)
                                                        lock2.doors_2[lock2.node_3].release(lock2.doors_2[lock2.node_3].users[0])
                                                        yield access_lock_door2
                                                        lock2.doors_2[lock2.node_3].users[0].id = self.id
                                                    else:
                                                        access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority = -1)
                                                        yield access_lock_door2
                                                        lock2.doors_2[lock2.node_3].users[0].id = self.id
                                            else:
                                                if lock2.doors_2[lock2.node_3].users != [] and lock2.doors_2[lock2.node_3].users[0].priority == -1:
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                    yield access_lock_door1
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority = -1)
                                                    lock2.doors_2[lock2.node_3].release(lock2.doors_2[lock2.node_3].users[0])
                                                    yield access_lock_door2
                                                    lock2.doors_2[lock2.node_3].users[0].id = self.id

                                                elif lock2.doors_2[lock2.node_3].users != [] and lock2.doors_2[lock2.node_3].users[0].priority == 0:
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                    yield access_lock_door1
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority = -1)
                                                    yield access_lock_door2
                                                    lock2.doors_2[lock2.node_3].users[0].id = self.id

                                                else:
                                                    if lock.converting_while_in_line_up_area[origin].users != []:
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[origin].request(priority = -1)
                                                        yield waiting_during_converting
                                                        lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)
                                                        access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                        yield access_lock_door1

                                                    elif (len(lock2.doors_1[lock2.node_1].users) == 0 or (len(lock2.doors_1[lock2.node_1].users) != 0 and lock2.doors_1[lock2.node_1].users[0].priority != -1)) and self.route[self.route.index(r)-1] != lock2.water_level:
                                                        access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[origin].request()
                                                        yield waiting_during_converting
                                                        yield from lock2.convert_chamber(self.env, self.route[self.route.index(r)-1], 0)
                                                        lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)

                                                    elif len(lock2.doors_1[lock2.node_1].users) != 0 and lock2.doors_1[lock2.node_1].users[0].priority == -1:
                                                        access_lock_door1 = lock2.doors_1[lock2.node_1].request()
                                                        yield access_lock_door1

                                                    else:
                                                        access_lock_door1 = lock2.doors_1[lock2.node_1].request()

                                                    if lock2.doors_2[lock2.node_3].users != [] and lock2.doors_2[lock2.node_3].users[0].priority == -1:
                                                        access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority = -1)
                                                        lock2.doors_2[lock2.node_3].release(lock2.doors_2[lock2.node_3].users[0])
                                                        yield access_lock_door2
                                                        lock2.doors_2[lock2.node_3].users[0].id = self.id
                                                    else:
                                                        access_lock_door2 = lock2.doors_2[lock2.node_3].request(priority = -1)
                                                        yield access_lock_door2
                                                        lock2.doors_2[lock2.node_3].users[0].id = self.id

                                        elif self.route[self.route.index(r)-1] == lock2.node_3:
                                            if len(lock2.doors_1[lock2.node_1].users) != 0:
                                                if lock2.doors_1[lock2.node_1].users[0].priority == -1:
                                                    if self.L > (lock2.resource.users[-1].lock_dist-0.5*lock2.resource.users[-1].length) or lock2.resource.users[-1].converting == True:
                                                        access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority = -1)
                                                        yield access_lock_door1
                                                        lock2.doors_1[lock2.node_1].release(access_lock_door1)

                                                        wait_for_next_cycle = lock3.pass_line_up_area[r2].request()
                                                        yield wait_for_next_cycle
                                                        lock3.pass_line_up_area[r2].release(wait_for_next_cycle)

                                                    if lock.converting_while_in_line_up_area[origin].users != []:
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[origin].request(priority = -1)
                                                        yield waiting_during_converting
                                                        lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)

                                                    elif (len(lock2.doors_2[lock2.node_3].users) == 0 or (len(lock2.doors_2[lock2.node_3].users) != 0 and lock2.doors_2[lock2.node_3].users[0].priority != -1)) and self.route[self.route.index(r)-1] != lock2.water_level:
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[origin].request()
                                                        yield waiting_during_converting
                                                        yield from lock2.convert_chamber(self.env, self.route[self.route.index(r)-1], 0)
                                                        lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)

                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                    yield access_lock_door2

                                                    if lock2.doors_1[lock2.node_1].users != [] and lock2.doors_1[lock2.node_1].users[0].priority == -1:
                                                        access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority = -1)
                                                        lock2.doors_1[lock2.node_1].release(lock2.doors_1[lock2.node_1].users[0])
                                                        yield access_lock_door1
                                                        lock2.doors_1[lock2.node_1].users[0].id = self.id
                                                    else:
                                                        access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority = -1)
                                                        yield access_lock_door1
                                                        lock2.doors_1[lock2.node_1].users[0].id = self.id

                                                else:
                                                    if lock3.converting_while_in_line_up_area[r2].users != []:
                                                        waiting_during_converting = lock3.converting_while_in_line_up_area[r2].request()
                                                        yield waiting_during_converting
                                                        lock3.converting_while_in_line_up_area[r2].release(waiting_during_converting)

                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                    yield access_lock_door2

                                                    if lock2.doors_1[lock2.node_1].users != [] and lock2.doors_1[lock2.node_1].users[0].priority == -1:
                                                        access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority = -1)
                                                        lock2.doors_1[lock2.node_1].release(lock2.doors_1[lock2.node_1].users[0])
                                                        yield access_lock_door1
                                                        lock2.doors_1[lock2.node_1].users[0].id = self.id
                                                    else:
                                                        access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority = -1)
                                                        yield access_lock_door1
                                                        lock2.doors_1[lock2.node_1].users[0].id = self.id
                                            else:
                                                if lock2.doors_1[lock2.node_1].users != [] and lock2.doors_1[lock2.node_1].users[0].priority == -1:
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                    yield access_lock_door2
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority = -1)
                                                    lock2.doors_1[lock2.node_1].release(lock2.doors_1[lock2.node_1].users[0])
                                                    yield access_lock_door1
                                                    lock2.doors_1[lock2.node_1].users[0].id = self.id

                                                elif lock2.doors_1[lock2.node_1].users != [] and lock2.doors_1[lock2.node_1].users[0].priority == 0:
                                                    access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                    yield access_lock_door2
                                                    access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority = -1)
                                                    yield access_lock_door1
                                                    lock2.doors_1[lock2.node_1].users[0].id = self.id

                                                else:
                                                    if lock.converting_while_in_line_up_area[origin].users != []:
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[origin].request(priority = -1)
                                                        yield waiting_during_converting
                                                        lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)
                                                        access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                        yield access_lock_door2

                                                    elif (len(lock2.doors_2[lock2.node_3].users) == 0 or (len(lock2.doors_2[lock2.node_3].users) != 0 and lock2.doors_2[lock2.node_3].users[0].priority != -1)) and self.route[self.route.index(r)-1] != lock2.water_level:
                                                        access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                        waiting_during_converting = lock.converting_while_in_line_up_area[origin].request()
                                                        yield waiting_during_converting
                                                        yield from lock2.convert_chamber(self.env, self.route[self.route.index(r)-1], 0)
                                                        lock.converting_while_in_line_up_area[origin].release(waiting_during_converting)

                                                    elif len(lock2.doors_2[lock2.node_3].users) != 0 and lock2.doors_2[lock2.node_3].users[0].priority == -1:
                                                        access_lock_door2 = lock2.doors_2[lock2.node_3].request()
                                                        yield access_lock_door2

                                                    else:
                                                        access_lock_door2 = lock2.doors_2[lock2.node_3].request()

                                                    if lock2.doors_1[lock2.node_1].users != [] and lock2.doors_1[lock2.node_1].users[0].priority == -1:
                                                        access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority = -1)
                                                        lock2.doors_1[lock2.node_1].release(lock2.doors_1[lock2.node_1].users[0])
                                                        yield access_lock_door1
                                                        lock2.doors_1[lock2.node_1].users[0].id = self.id
                                                    else:
                                                        access_lock_door1 = lock2.doors_1[lock2.node_1].request(priority = -1)
                                                        yield access_lock_door1
                                                        lock2.doors_1[lock2.node_1].users[0].id = self.id

                                        access_lock_length = lock2.length.get(self.L)
                                        access_lock = lock2.resource.request()

                                        access_lock_pos_length = lock2.pos_length.get(self.L)
                                        self.lock_dist = lock2.pos_length.level + 0.5*self.L
                                        yield access_lock_pos_length

                                        lock2.resource.users[-1].id = self.id
                                        lock2.resource.users[-1].length = self.L
                                        lock2.resource.users[-1].lock_dist = self.lock_dist
                                        lock2.resource.users[-1].converting = False
                                        if self.route[self.route.index(r)-1] == lock2.node_1:
                                            lock2.resource.users[-1].dir = 1.0
                                        else:
                                            lock2.resource.users[-1].dir = 2.0

                                        if wait_for_lock_entry != self.env.now:
                                            waiting = self.env.now - wait_for_lock_entry
                                            self.log_entry("Waiting in line-up area start", wait_for_lock_entry, 0, orig)
                                            self.log_entry("Waiting in line-up area stop", self.env.now, waiting, orig)

                                        self.wgs84 = pyproj.Geod(ellps="WGS84")
                                        [doors_origin_lat, doors_origin_lon, doors_destination_lat, doors_destination_lon] = [self.env.FG.nodes[self.route[self.route.index(r)-1]]['geometry'].x, self.env.FG.nodes[self.route[self.route.index(r)-1]]['geometry'].y,
                                                                                                                               self.env.FG.nodes[self.route[self.route.index(r)+1]]['geometry'].x, self.env.FG.nodes[self.route[self.route.index(r)+1]]['geometry'].y]
                                        fwd_azimuth,_,distance = self.wgs84.inv(doors_origin_lat, doors_origin_lon, doors_destination_lat, doors_destination_lon)
                                        [self.lock_pos_lat,self.lock_pos_lon,_] = self.wgs84.fwd(self.env.FG.nodes[self.route[self.route.index(r)-1]]['geometry'].x,
                                                                                                 self.env.FG.nodes[self.route[self.route.index(r)-1]]['geometry'].y,
                                                                                                 fwd_azimuth,self.lock_dist)

                                        for r4 in reversed(self.route[:(loc-1)]):
                                            if 'Line-up area' in self.env.FG.nodes[r4].keys():
                                                locks = self.env.FG.nodes[r4]["Line-up area"]
                                                for lock4 in locks:
                                                    if lock4.name == self.lock_name:
                                                        lock4.lock_queue_length -= 1
                                break

                            elif 'Waiting area' in self.env.FG.nodes[r].keys():
                                for r2 in reversed(self.route[:(loc-1)]):
                                    if 'Lock' in self.env.FG.nodes[r2].keys():
                                        locks = self.env.FG.nodes[r2]["Lock"]
                                        for lock2 in locks:
                                            if lock2.name == self.lock_name:
                                                if self.route[self.route.index(r2)+1] == lock2.node_3 and len(lock2.doors_2[lock2.node_3].users) != 0 and lock2.doors_2[lock2.node_3].users[0].id == self.id:
                                                    lock2.doors_2[lock2.node_3].release(access_lock_door2)
                                                elif self.route[self.route.index(r2)+1] == lock2.node_1 and len(lock2.doors_1[lock2.node_1].users) != 0 and lock2.doors_1[lock2.node_1].users[0].id == self.id:
                                                    lock2.doors_1[lock2.node_1].release(access_lock_door1)

                                                lock.pass_line_up_area[origin].release(departure_lock)
                                                lock2.resource.release(access_lock)
                                                departure_lock_length = lock2.length.put(self.L)
                                                departure_lock_pos_length = lock2.pos_length.put(self.L)
                                                yield departure_lock_length
                                                yield departure_lock_pos_length
                                        break

            if "Line-up area" in self.env.FG.nodes[self.route[node[0]-1]].keys():
                locks = self.env.FG.nodes[self.route[node[0]-1]]["Line-up area"]
                for lock in locks:
                    if lock.name == self.lock_name:
                        loc = self.route.index(origin)
                        for r in self.route[loc:]:
                            if 'Lock' in self.env.FG.nodes[r].keys():
                                locks = self.env.FG.nodes[r]["Lock"]
                                lock.line_up_area[self.route[node[0]-1]].release(access_lineup_area)
                                departure_lineup_length = lock.length.put(self.L)
                                yield departure_lineup_length

            if "Lock" in self.env.FG.nodes[origin].keys():
                locks = self.env.FG.nodes[origin]["Lock"]
                for lock in locks:
                    if lock.name == self.lock_name:
                        if self.route[self.route.index(origin)-1] == lock.node_1:
                            lock.doors_1[lock.node_1].release(access_lock_door1)
                        elif self.route[self.route.index(origin)-1] == lock.node_3:
                            lock.doors_2[lock.node_3].release(access_lock_door2)
                        orig = shapely.geometry.Point(self.lock_pos_lat,self.lock_pos_lon)
                        loc = self.route.index(origin)
                        for r2 in reversed(self.route[loc:]):
                            if "Line-up area" in self.env.FG.nodes[r2].keys():
                                locks = self.env.FG.nodes[r2]["Line-up area"]
                                for lock3 in locks:
                                    if lock3.name == self.lock_name:
                                        departure_lock = lock3.pass_line_up_area[r2].request(priority = -1)
                                        break
                                break

                        for r in reversed(self.route[:(loc-1)]):
                            if "Line-up area" in self.env.FG.nodes[r].keys():
                                locks = self.env.FG.nodes[r]["Line-up area"]
                                for lock2 in locks:
                                    if lock2.name == self.lock_name:
                                        for q2 in range(0,len(lock.resource.users)):
                                            if lock.resource.users[q2].id == self.id:
                                                break

                                        start_time_in_lock = self.env.now
                                        self.log_entry("Passing lock start", self.env.now, 0, orig)

                                        if len(lock2.line_up_area[r].users) != 0 and lock2.line_up_area[r].users[0].length < lock.length.level:
                                            if self.route[self.route.index(origin)-1] == lock.node_1:
                                                access_line_up_area = lock2.enter_line_up_area[r].request()
                                                yield access_line_up_area
                                                lock2.enter_line_up_area[r].release(access_line_up_area)
                                                access_lock_door1 = lock.doors_1[lock.node_1].request()
                                                yield access_lock_door1
                                                lock.doors_1[lock.node_1].release(access_lock_door1)

                                            elif self.route[self.route.index(origin)-1] == lock.node_3:
                                                access_line_up_area = lock2.enter_line_up_area[r].request()
                                                yield access_line_up_area
                                                lock2.enter_line_up_area[r].release(access_line_up_area)
                                                access_lock_door2 = lock.doors_2[lock.node_3].request()
                                                yield access_lock_door2
                                                lock.doors_2[lock.node_3].release(access_lock_door2)

                                        if lock.resource.users[0].id == self.id:
                                            lock.resource.users[0].converting = True
                                            number_of_vessels = len(lock.resource.users)
                                            yield from lock.convert_chamber(self.env, destination,number_of_vessels)
                                        else:
                                            for u in range(len(lock.resource.users)):
                                                if lock.resource.users[u].id == self.id:
                                                    lock.resource.users[u].converting = True
                                                    yield self.env.timeout(lock.doors_close + lock.operation_time(self.env) + lock.doors_open)
                                                    break

                        yield departure_lock

                        self.log_entry("Passing lock stop", self.env.now, self.env.now-start_time_in_lock, orig,)
                        [self.lineup_pos_lat,self.lineup_pos_lon] = [self.env.FG.nodes[self.route[self.route.index(r2)]]['geometry'].x, self.env.FG.nodes[self.route[self.route.index(r2)]]['geometry'].y]
                        yield from self.pass_edge(origin, destination)
                        self.v = speed

            else:
                # print('I am going to go to the next node {}'.format(destination))
                yield from self.pass_edge(origin, destination)

            if node[0] + 2 == len(self.route):
                break

        # self.geometry = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        logger.debug("  distance: " + "%4.2f" % self.distance + " m")
        if self.current_speed is not None:
            logger.debug("  sailing:  " + "%4.2f" % self.current_speed + " m/s")
            logger.debug(
                "  duration: "
                + "%4.2f" % ((self.distance / self.current_speed) / 3600)
                + " hrs"
            )
        else:
            logger.debug("  current_speed:  not set")

    def pass_edge(self, origin, destination):
        edge = self.env.FG.edges[origin, destination]
        orig = nx.get_node_attributes(self.env.FG, "geometry")[origin]
        dest = nx.get_node_attributes(self.env.FG, "geometry")[destination]

        for edge_function in self.edge_functions:
            edge_function(orig, dest, edge)

        if "Lock" in self.env.FG.nodes[origin].keys():
            orig = shapely.geometry.Point(self.lock_pos_lat,self.lock_pos_lon)

        if "Lock" in self.env.FG.nodes[destination].keys():
            dest = shapely.geometry.Point(self.lock_pos_lat,self.lock_pos_lon)

        if "Line-up area" in self.env.FG.nodes[origin].keys():
            orig = shapely.geometry.Point(self.lineup_pos_lat,self.lineup_pos_lon)

        if "Line-up area" in self.env.FG.nodes[destination].keys():
            dest = shapely.geometry.Point(self.lineup_pos_lat,self.lineup_pos_lon)

        if 'geometry' in edge:
            edge_route = np.array(edge['geometry'].coords)

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
                edge_route = np.flipud(np.array(edge['geometry'].coords))

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
                shapely.geometry.shape(orig).x,
                shapely.geometry.shape(orig).y,
                shapely.geometry.shape(dest).x,
                shapely.geometry.shape(dest).y,
            )[2]

            self.distance += distance

            value = 0

            # remember when we arrived at the edge
            arrival = self.env.now

            v = self.current_speed

            # This is the case if we are sailing on power
            if getattr(self, 'P_tot_given', None) is not None:
                edge = self.env.FG.edges[origin, destination]
                # use power2v on self so that you can override it from outside
                v = self.power2v(self, edge)
                # use computed power
                value = self.P_given

            # determine time to pass edge
            timeout = distance / v


            # Wait for edge resources to become available
            if "Resources" in edge.keys():
                with self.env.FG.edges[origin, destination]["Resources"].request() as request:
                    yield request
                    # we had to wait, log it
                    if arrival != self.env.now:
                        self.log_entry("Waiting to pass edge {} - {} start".format(origin, destination), arrival, value, orig,)
                        self.log_entry("Waiting to pass edge {} - {} stop".format(origin, destination), self.env.now, value, orig,)

            # default velocity based on current speed.
            self.log_entry("Sailing from node {} to node {} start".format(origin, destination), self.env.now, value, orig,)
            yield self.env.timeout(timeout)
            self.log_entry("Sailing from node {} to node {} stop".format(origin, destination), self.env.now, value, dest,)

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


class ExtraMetadata:
    """store all leftover keyword arguments as metadata property (use as last mixin)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        # store all other properties as metadata
        self.metadata = kwargs
