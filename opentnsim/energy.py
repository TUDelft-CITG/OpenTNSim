import datetime,time
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
    """read correction factor from package directory
    """

    # Can't get this  to work with pkg_resourcs
    data_dir = pathlib.Path(__file__).parent.parent / 'data'
    correctionfactors_path = data_dir / 'Correctionfactors.csv'
    df = pd.read_csv(correctionfactors_path, comment='#')
    return df

def karpov_smooth_curves():
    """read correction factor from package directory
    """

    # Can't get this  to work with pkg_resourcs
    data_dir = pathlib.Path(__file__).parent.parent / 'data'
    karpov_smooth_curves_path = data_dir / 'KarpovSmoothCurves.csv'
    df = pd.read_csv(karpov_smooth_curves_path, comment='#')
    return df



def find_closest_node(G, point):
    """find the closest node on the graph from a given point
    """

    distance = np.full((len(G.nodes)), fill_value=np.nan)
    for ii, n in enumerate(G.nodes):
        distance[ii] = point.distance(G.nodes[n]['geometry'])
    name_node = list(G.nodes)[np.argmin(distance)]
    distance_node = np.min(distance)

    return name_node, distance_node


def get_upperbound_for_power2v(vessel, width, depth, bounds=(0,20)):
    """ for a waterway section with a given width and depth, compute a maximum installed-
    power-allowed velocity, considering squat. This velocity is set as upperbound in the 
    power2v function in energy.py "upperbound" is the maximum value in velocity searching 
    range.
    """
    
    def get_grounding_v(vessel, width, depth, bounds):
        
        def seek_v_given_z(v, vessel, width, depth):
            # calculate sinkage
            z_computed = (vessel.C_B * ((vessel.B * vessel._T) / (width * depth)) ** 0.81) * ((v*1.94) ** 2.08) / 20
            
            # calculate available underkeel clearance (vessel in rest)
            z_given = depth - vessel._T
            
            # compute difference between the sinkage and the space available for sinkage
            diff = z_given - z_computed

            return diff ** 2
        
        # goalseek to minimize
        fun = functools.partial(seek_v_given_z, vessel=vessel, width=width, depth=depth)
        fit = scipy.optimize.minimize_scalar(fun, bounds=bounds, method='bounded')
        
        # check if we found a minimum
        if not fit.success:
            raise ValueError(fit)

        # the value of fit.x within the bound (0,20) is the velocity we find where the diff**2 reach a minimum (zero).
        grounding_v =  fit.x
        
        print('grounding velocity {:.2f} m/s'.format(grounding_v))
        
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
        velocity = row['velocity']
        
        # calculate squat and the waterdepth after squat
        z_computed = (vessel.C_B * ((vessel.B * vessel._T) / (150 * h_0)) ** 0.81) * ((velocity*1.94) ** 2.08) / 20
        h_0 = depth - z_computed
        
        # for the squatted water depth calculate resistance and power
        # vessel.calculate_properties()
        # vessel.calculate_frictional_resistance(v=velocity, h_0=h_0)
        vessel.calculate_total_resistance(v=velocity, h_0=h_0)
        P_tot = vessel.calculate_total_power_required(v=velocity)
        
        # prepare a row
        result = {}
        result.update(row)
        result['Powerallowed_v'] = velocity
        result['P_tot'] = P_tot
        result['P_installed'] = vessel.P_installed
        
        # update resulst dict
        results.append(result)
    
    results_df = pd.DataFrame(results)

    selected = results_df.query('P_tot < P_installed')
    upperbound = max(selected['Powerallowed_v'])
    print('upperbound velocity {:.2f} m/s'.format(upperbound))
    return upperbound




def power2v(vessel, edge, upperbound):
    """Compute vessel velocity given an edge and power (P_tot_given)

    bounds is the limits where to look for a solution for the velocity [m/s]
    returns velocity [m/s]
    """
    # upperbound = get_upperbound_for_power2v()
    # bounds > 10 gave an issue...
    # TODO: check what the origin of this is.
    def seek_v_given_power(v, vessel, edge):
        """function to optimize"""
        # water depth from the edge
        h_0 = edge['Info']['GeneralDepth']
        h_0 = vessel.calculate_h_squat(v, h_0)
        # TODO: consider precomputing a range v/h combinations for the ship before the simulation starts
        vessel.calculate_total_resistance(v, h_0)
        # compute total power given
        P_given = vessel.calculate_total_power_required(v=v)
        if isinstance(vessel.P_tot, complex):
            raise ValueError(f"P tot is complex: {vessel.P_tot}")

        # compute difference between power setting by captain and power needed for velocity
        diff = vessel.P_tot_given - vessel.P_tot
        logger.debug(f'optimizing for v: {v}, P_tot_given: {vessel.P_tot_given}, P_tot {vessel.P_tot}, P_given {P_given}')
        return diff ** 2

    # fill in some of the parameters that we already know
    fun = functools.partial(seek_v_given_power, vessel=vessel, edge=edge)
    # lookup a minimum
    fit = scipy.optimize.minimize_scalar(fun, bounds=(0, upperbound), method='bounded', options=dict(xatol=0.0000001))

    # check if we found a minimum
    if not fit.success:
        raise ValueError(fit)
    logger.debug(f"fit: {fit}")
    return fit.x



class EnergyCalculation:
    """Add information on energy use and effects on energy use.
    """

    def __init__(self, FG, vessel, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """Initialization"""
        self.FG = FG
        self.vessel = vessel

        self.energy_use = {"time_start": [],
                           "time_stop": [],
                           "edge_start": [],
                           "edge_stop": [],
                           "P_tot": [],
                           "P_installed": [],
                           "total_energy": [],
                           "total_fuel_consumption": [],
                           "total_emission_CO2": [],
                           "total_emission_PM10": [],
                           "total_emission_NOX": [],
                           "stationary": [],
                           "water depth": [],
                           "distance": [],
                           "delta_t": []}

        self.co2_footprint = {"total_footprint": 0, "stationary": 0}
        self.mki_footprint = {"total_footprint": 0, "stationary": 0}

    def calculate_energy_consumption(self):
        """Calculation of energy consumption based on total time in system and properties
        """

        def calculate_distance(geom_start, geom_stop):
            """method to calculate the distance in meters between two geometries
            """

            wgs84 = pyproj.Geod(ellps='WGS84')

            # distance between two points
            return float(wgs84.inv(geom_start.x, geom_start.y,
                                 geom_stop.x,  geom_stop.y) [2])

        def calculate_depth(geom_start, geom_stop):
            """method to calculate the depth of the waterway in meters between two geometries
            """

            depth = 0

            #The node on the graph of vaarweginformatie.nl closest to geom_start and geom_stop

            node_start = find_closest_node(self.FG, geom_start)[0]
            node_stop = find_closest_node(self.FG, geom_stop)[0]

            #Read from the FG data from vaarweginformatie.nl the General depth of each edge
            try:#if node_start != node_stop:
                depth = self.FG.get_edge_data(node_start, node_stop)["Info"]["GeneralDepth"]
            except:
                depth = np.nan     #When there is no data of the depth available of this edge, it gives a message

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
        times = self.vessel.log["Timestamp"]
        messages = self.vessel.log["Message"]
        geometries = self.vessel.log["Geometry"]

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
                logger.debug('delta_t: {:.4f} s'. format(delta_t))
                logger.debug('distance: {:.4f} m'. format(distance))
                logger.debug('velocity: {:.4f} m/s'. format(v))

                # we use the calculated velocity to determine the resistance and power required
                # we can switch between the 'original water depth' and 'water depth considering ship squatting' for energy calculation, by using the function "calculate_h_squat (h_squat is set as Yes/No)" in the core.py
                h_0 = self.vessel.calculate_h_squat(v, h_0)                              
                self.vessel.calculate_total_resistance(v, h_0)
                self.vessel.calculate_total_power_required(v=v)

                self.vessel.calculate_emission_factors_total(v=v)

                if messages[i + 1] in stationary_phase_indicator:  # if we are in a stationary stage only log P_hotel
                    #Energy consumed per time step delta_t in the stationary stage
                    energy_delta = self.vessel.P_hotel * delta_t / 3600  # kJ/3600 = kWh

                    # Emissions CO2, PM10 and NOX, in gram - emitted in the stationary stage per time step delta_t,
                    # consuming 'energy_delta' kWh
                    P_hotel_delta = self.vessel.P_hotel   # in kW
                    P_installed_delta = self.vessel.P_installed   # in kW
                    emission_delta_CO2 = self.vessel.total_factor_CO2 * energy_delta # in g
                    emission_delta_PM10 = self.vessel.total_factor_PM10 * energy_delta # in g
                    emission_delta_NOX = self.vessel.total_factor_NOX * energy_delta # in g
                    emission_delta_fuel = self.vessel.total_factor_FU * energy_delta # in g

                    self.energy_use["P_tot"].append(P_hotel_delta)
                    self.energy_use["P_installed"].append(P_installed_delta)
                    self.energy_use["total_energy"].append(energy_delta)
                    self.energy_use["stationary"].append(energy_delta)
                    self.energy_use["total_emission_CO2"].append(emission_delta_CO2)
                    self.energy_use["total_emission_PM10"].append(emission_delta_PM10)
                    self.energy_use["total_emission_NOX"].append(emission_delta_NOX)
                    self.energy_use["total_fuel_consumption"].append(emission_delta_fuel)

                    if not np.isnan(h_0):
                        self.energy_use["water depth"].append(h_0)
                    else:
                        self.energy_use["water depth"].append(self.energy_use["water depth"].iloc[i])

                else:  # otherwise log P_tot
                    #Energy consumed per time step delta_t in the propulsion stage
                    energy_delta = self.vessel.P_tot * delta_t / 3600  # kJ/3600 = kWh

                    # Emissions CO2, PM10 and NOX, in gram - emitted in the propulsion stage per time step delta_t,
                    # consuming 'energy_delta' kWh
                    P_tot_delta = self.vessel.P_tot   # in kW
                    P_installed_delta = self.vessel.P_installed   # in kW
                    emission_delta_CO2 = self.vessel.total_factor_CO2 * energy_delta #Energy consumed per time step delta_t in the                                                                                              #stationary phase # in g
                    emission_delta_PM10 = self.vessel.total_factor_PM10 * energy_delta # in g
                    emission_delta_NOX = self.vessel.total_factor_NOX * energy_delta # in g
                    emission_delta_fuel=self.vessel.total_factor_FU * energy_delta # in g

                    self.energy_use["P_tot"].append(P_tot_delta)
                    self.energy_use["P_installed"].append(P_installed_delta)
                    self.energy_use["total_energy"].append(energy_delta)
                    self.energy_use["stationary"].append(0)
                    self.energy_use["total_emission_CO2"].append(emission_delta_CO2)
                    self.energy_use["total_emission_PM10"].append(emission_delta_PM10)
                    self.energy_use["total_emission_NOX"].append(emission_delta_NOX)
                    self.energy_use["total_fuel_consumption"].append(emission_delta_fuel)
                    self.energy_use["water depth"].append(h_0)
                    #self.energy_use["water depth info from vaarweginformatie.nl"].append(depth)


        # TODO: er moet hier een heel aantal dingen beter worden ingevuld
        # - de kruissnelheid is nu nog per default 1 m/s (zie de Movable mixin). Eigenlijk moet in de
        #   vessel database ook nog een speed_loaded en een speed_unloaded worden toegevoegd.
        # - er zou nog eens goed gekeken moeten worden wat er gedaan kan worden rond kustwerken
        # - en er is nog iets mis met de snelheid rond een sluis

        # - add HasCurrent Class or def
        # - add HasSquat

    def plot(self):

        import folium

        df = pd.DataFrame.from_dict(self.energy_use)

        m = folium.Map(location=[51.7, 4.4], zoom_start = 12)

        line = []
        for index, row in df.iterrows():
            line.append((row["edge_start"].y, row["edge_start"].x))

        folium.PolyLine(line, weight = 4).add_to(m)

        return m
