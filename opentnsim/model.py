"""Vessel generator."""

# package(s) related to time, space and id
import json
import logging
import uuid

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy
import networkx as nx
import numpy as np

# spatial libraries
import pyproj
import shapely.geometry
import pytz

# additional packages
import datetime, time
import pandas as pd
import random
import xarray as xr

# import core from self
import opentnsim.core as core
import opentnsim.vessel_traffic_service as vessel_traffic_service

logger = logging.getLogger(__name__)

class VesselGenerator:
    """
    A class to generate vessels from a database
    """

    def __init__(self, vessel_type, vessel_database, loaded=None, random_seed=3):
        """ Initialization """

        self.vessel_type = vessel_type
        self.vessel_database = vessel_database
        self.loaded = loaded
        random.seed(random_seed)

    def generate(self, environment, vessel_name, fleet_distribution=None, scenario=None):
        """ Generate a vessel """

        if fleet_distribution == None:
            vessel_info = self.vessel_database.sample(
                n=1, random_state=int(1000 * random.random())
            )
        else:
            vessel_info = self.vessel_database.sample(
                n=1, weights=fleet_distribution,random_state=int(1000 * random.random())
            )
        vessel_data = {}

        vessel_data["env"] = environment
        vessel_data["name"] = vessel_name

        if scenario:
            vessel_info = vessel_info[vessel_info["scenario"] == scenario]

        for key in vessel_info:
            if key == "vessel_id":
                vessel_data["id"] = vessel_info[key].values[0]
            else:
                vessel_data[key] = vessel_info[key].values[0]

        if self.loaded == True:
            vessel_data["level"] = vessel_data["capacity"]
        elif self.loaded == "Random":
            if random.random() < 0.5:
                vessel_data["level"] = vessel_data["capacity"]
            else:
                vessel_data["level"] = 0
        vessel_data["route"] = None
        vessel_data["geometry"] = None
        self.vessel_type(**vessel_data)
        return self.vessel_type(**vessel_data)

    def arrival_process(
        self,
        environment,
        origin,
        destination,
        arrival_distribution,
        scenario,
        arrival_process,
        fleet_distribution,
    ):
        """ 
        Make arrival process
        
        environment:            simpy environment
        arrival_distribution:   specify the distribution from which vessels are generated, int or list
        arrival_process:        process of arrivals
        """

        # Create an array with inter-arrival times -- average number of seconds between arrivals
        if type(arrival_distribution) == int or type(arrival_distribution) == float:
            self.inter_arrival_times = [3600 / arrival_distribution] * 24

        elif type(arrival_distribution) == list and len(arrival_distribution) == 24:
            self.inter_arrival_times = [3600 / n for n in arrival_distribution]

        elif type(arrival_distribution) == list:
            raise ValueError(
                "List should contain an average number of vessels per hour for an entire day: 24 entries."
            )

        else:
            raise ValueError(
                "Specify an arrival distribution: type Integer or type List."
            )

        while True:

            # Check simulation time
            inter_arrival = self.inter_arrival_times[
                datetime.datetime.fromtimestamp(environment.now).hour
            ]

            # In the case of a Markovian arrival process
            if arrival_process == "Markovian":

                # Make a timestep based on the poisson process
                yield environment.timeout(random.expovariate(1 / inter_arrival))

            elif arrival_process == "Uniform":

                # Make a timestep based on uniform arrivals
                yield environment.timeout(inter_arrival)

            else:
                raise ValueError(
                    "No other arrival processes are yet defined. You can add them to transport_network_analysis/vessel_generator.py."
                )

            # Create a vessel
            vessel = self.generate(environment, "Vessel", fleet_distribution, scenario)
            vessel.output = {}
            #core.Output.vessel_dependent_output(vessel)
            vessel.env = environment
            vessel.route = nx.dijkstra_path(environment.FG, origin, destination)
            vessel.geometry = nx.get_node_attributes(environment.FG, "geometry")[
                vessel.route[0]
            ]
            environment.vessels.append(vessel)
            # Move on path
            process = environment.process(vessel.move())
            vessel.process = process

class Hydrodata:
    def __init__(self,hydrodynamic_data):
        self.hydrodynamic_data = hydrodynamic_data
        return

class Simulation(core.Identifiable):
    """
    A class to generate vessels from a database
    """

    def __init__(self,
                 graph,
                 simulation_start=datetime.datetime.now(),
                 simulation_duration=None,
                 simulation_stop=None,
                 hydrodynamic_start_time=datetime.datetime.now(),
                 hydrodynamic_data_path=None,
                 vessel_speed_data_path=None, scenario=None):
        """ 
        Initialization 
        
        scenario: scenario with vessels - should be coupled to the database
        """
        self.env = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))
        self.env.FG = graph
        self.env.simulation_start = simulation_start
        self.env.simulation_stop = simulation_stop
        self.simulation_duration = simulation_duration
        if self.env.simulation_stop:
            self.simulation_duration = self.env.simulation_stop-self.env.simulation_start
        else:
            self.env.simulation_stop = self.env.simulation_start+self.simulation_duration
        self.env.routes = pd.DataFrame.from_dict(
            {
                "Origin": [],
                "Destination": [],
                "Width": [],
                "Height": [],
                "Depth": [],
                "Route": [],
            }
        )
        self.scenario = scenario

        self.env.vessels = []
        self.output = {}

        self.env.vessel_traffic_service = vessel_traffic_service.VesselTrafficService(FG=graph,
                                                                                      hydrodynamic_start_time = hydrodynamic_start_time,
                                                                                      hydrodynamic_information_path = hydrodynamic_data_path,
                                                                                      vessel_speed_data_path=vessel_speed_data_path)


    def create_vessel_speed_data_file(self, data):
        '''
        Function to create a vessel speed dataframe file.
        '''
        speed_df = pd.DataFrame(columns=['Speed', 'Distance', 'Time'],
                                index=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []],
                                                    names=["start_node", "stop_node", "k_edge"]))
        return speed_df


    def add_vessels(
        self,
        origin=None,
        destination=None,
        vessel=None,
        vessel_generator=None,
        fleet_distribution=None,
        arrival_distribution=1,
        arrival_process="Markovian",
    ):
        """ 
        Make arrival process
        
        environment:            simpy environment
        arrival_distribution:   specify the distribution from which vessels are generated, int or list
        arrival_process:        process of arrivals
        """

        if vessel_generator == None:
            self.env.vessels.append(vessel)
            process = self.env.process(vessel.move())
            vessel.process = process
            if 'metadata' in dir(vessel) and 'arrival_time' not in vessel.metadata.keys():
                vessel.metadata['arrival_time'] = self.env.simulation_start

        else:
            self.env.process(
                vessel_generator.arrival_process(
                    self.env,
                    origin,
                    destination,
                    arrival_distribution,
                    self.scenario,
                    arrival_process,
                    fleet_distribution,
                )
            )

    def run(self):
        """ 
        Run the simulation 
        
        duration:               specify the duration of the simulation in seconds
        """
        self.env.run(until=self.env.now + self.simulation_duration.total_seconds())
