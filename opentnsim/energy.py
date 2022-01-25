import pathlib
import logging
import functools

import pyproj
import numpy as np
import pandas as pd
import scipy.optimize

logger = logging.getLogger(__name__)

def correction_factors():
    """read correction factor from package directory"""
    # Can't get this  to work with pkg_resourcs
    data_dir = pathlib.Path(__file__).parent.parent / 'data'
    correctionfactors_path = data_dir / 'Correctionfactors.csv'
    df = pd.read_csv(correctionfactors_path, comment='#')
    return df

def karpov_smooth_curves():
    """read correction factor from package directory"""
    # Can't get this  to work with pkg_resourcs
    data_dir = pathlib.Path(__file__).parent.parent / 'data'
    karpov_smooth_curves_path = data_dir / 'KarpovSmoothCurves.csv'
    df = pd.read_csv(karpov_smooth_curves_path, comment='#')
    return df



def find_closest_node(G, point):
    """find the closest node on the graph from a given point"""

    distance = np.full((len(G.nodes)), fill_value=np.nan)
    for ii, n in enumerate(G.nodes):
        distance[ii] = point.distance(G.nodes[n]['geometry'])
    name_node = list(G.nodes)[np.argmin(distance)]
    distance_node = np.min(distance)

    return name_node, distance_node


def power2v(vessel, edge, bounds=(0, 10)):
    """Compute vessel velocity given an edge and power (P_tot_given)
    bounds is the limits where to look for a solution for the velocity [m/s]
    returns velocity [m/s]
    """
    # bounds > 10 gave an issue...
    # TODO: check what the origin of this is.
    def seek_v_given_power(v, vessel, edge):
        """function to optimize"""
        logger.debug(f'optimizing for v: {v}, P_tot_given: {vessel.P_tot_given}')
        # water depth from the edge
        h_0 = edge['Info']['GeneralDepth']
        # TODO: consider precomputing a range v/h combinations for the ship before the simulation starts
        vessel.calculate_total_resistance(v, h_0)
        vessel.calculate_total_power_required()
        diff = vessel.P_given - vessel.P_tot_given
        return diff ** 2

    # fill in some of the parameters that we already know
    fun = functools.partial(seek_v_given_power, vessel=vessel, edge=edge)
    # lookup a minimum
    fit = scipy.optimize.minimize_scalar(fun, bounds=bounds, method='bounded')

    # check if we found a minimum
    if not fit.success:
        raise ValueError(fit)
    logger.debug(f"fit: {fit}")
    return fit.x



class EnergyCalculation:
    """
    Add information on energy use and effects on energy use.
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
        """Calculation of energy consumption based on total time in system and properties"""

        def calculate_distance(geom_start, geom_stop):
            """method to calculate the distance in meters between two geometries"""
            wgs84 = pyproj.Geod(ellps='WGS84')

            # distance between two points
            return float(wgs84.inv(geom_start.x, geom_start.y,
                                 geom_stop.x,  geom_stop.y) [2])

        def calculate_depth(geom_start, geom_stop):
            """method to calculate the depth of the waterway in meters between two geometries"""
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
                self.vessel.calculate_total_resistance(v, h_0)
                self.vessel.calculate_total_power_required()

                self.vessel.calculate_emission_factors_total()

                if messages[i + 1] in stationary_phase_indicator:  # if we are in a stationary stage only log P_hotel
                    #Energy consumed per time step delta_t in the stationary stage
                    energy_delta = self.vessel.P_hotel * delta_t / 3600  # kJ/3600 = kWh

                    #Emissions CO2, PM10 and NOX, in gram - emitted in the stationary stage per time step delta_t, consuming 'energy_delta' kWh
                    emission_delta_CO2 = self.vessel.Emf_CO2 * energy_delta # in g
                    emission_delta_PM10 = self.vessel.Emf_PM10 * energy_delta # in g
                    emission_delta_NOX = self.vessel.Emf_NOX * energy_delta # in g
                    emission_delta_fuel=self.vessel.fuel_consumption* energy_delta/1000 # in kg

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

                    #Emissions CO2, PM10 and NOX, in gram - emitted in the propulsion stage per time step delta_t, consuming 'energy_delta' kWh
                    emission_delta_CO2 = self.vessel.Emf_CO2 * energy_delta #Energy consumed per time step delta_t in the stationary phase # in g
                    emission_delta_PM10 = self.vessel.Emf_PM10 * energy_delta # in g
                    emission_delta_NOX = self.vessel.Emf_NOX * energy_delta # in g
                    emission_delta_fuel=self.vessel.fuel_consumption* energy_delta/1000 # in kg

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


    def plot(self):

        import folium

        df = pd.DataFrame.from_dict(self.energy_use)

        m = folium.Map(location=[51.7, 4.4], zoom_start = 12)

        line = []
        for index, row in df.iterrows():
            line.append((row["edge_start"].y, row["edge_start"].x))

        folium.PolyLine(line, weight = 4).add_to(m)

        return m
