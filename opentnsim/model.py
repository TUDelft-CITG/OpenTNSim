"""Vessel generator."""

# packkage(s) for documentation, debugging, saving and loading
import logging

# package(s) related to time, space and id
import uuid
import numbers
import datetime, time

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy
import networkx as nx

# package(s) for data handling
import pandas as pd
import random

# import core from self
import opentnsim.core as core

logger = logging.getLogger(__name__)


class VesselGenerator:
    """
    A class to generate vessels from a database

    Parameters
    ----------
    vessel_type: class of mixins.
        The type of vessel to be generated.
    vessel_database: ??
        The database from which the vessel is generated.
        Make sure all needed attributes for vessel_type are available in the database.
    loaded: optional
        whether or not the vessel is loaded.
        If True, the vessel is loaded.
        If "Random", the vessel is randomly loaded or not (50% chance fully loaded, 50% chance empty).
        If not specified, the vessel is empty.
    random_seed: int, optional
        The random seed for generating vessels. The default is 3.
    """

    def __init__(self, vessel_type, vessel_database, loaded=None, random_seed=3):
        """Initialization"""

        self.vessel_type = vessel_type
        self.vessel_database = vessel_database
        self.loaded = loaded

        random.seed(random_seed)

    def generate(self, environment, vessel_name: str, scenario=None):
        """Get a random vessel from self.database

        Parameters
        ----------
        environment: simpy environment
            The environment in which the vessel is generated.
        vessel_name: str
            The name that is assigned to the generated vessel.
        scenario: str, optional
            The scenario of the generated vessel. If given, the vessel with this scenario is selected from the database.
        """

        vessel_info = self.vessel_database.sample(n=1, random_state=int(1000 * random.random()))
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

        # add a unique id to the id, so that our ships are unique even when resampled
        vessel_data["id"] = f'{vessel_data["id"]}-{uuid.uuid4()}'

        return self.vessel_type(**vessel_data)

    def arrival_process(
        self,
        environment,
        origin,
        destination,
        arrival_distribution,
        scenario,
        arrival_process,
    ):
        """
        Make arrival process in the simulation environment. Vessels with a route and between origin and destination are generated according to the arrival distribution.
        The route is calculated using the dijkstra algorithm.

        Parameters
        ----------
        environment:  simpy environment
            The environment in which the vessel is generated.
        origin:       str
            The origin of the vessel.
        destination:  str
            The destination of the vessel.
        arrival_distribution: int or list
            The amount of vessels that enter the simulation per hour.
            If int, it is the average number of vessels per hour over the entire day.
            If list, it is the average number of vessels per hour for each hour of the day. List must have length of 24 (one entry for each hour).
        scenario:     str
            The scenario that is assigned to the generated vessel.
        arrival_process:  str
            process of arrivals. choose between "Markovian" or "Uniform".
        """

        # Create an array with inter-arrival times -- average number of seconds between arrivals
        if isinstance(arrival_distribution, numbers.Number):
            self.inter_arrival_times = [3600 / arrival_distribution] * 24

        elif isinstance(arrival_distribution, list) and len(arrival_distribution) == 24:
            self.inter_arrival_times = [3600 / n for n in arrival_distribution]

        elif isinstance(arrival_distribution, list):
            raise ValueError("List should contain an average number of vessels per hour for an entire day: 24 entries.")

        else:
            raise ValueError("Specify an arrival distribution: type Integer or type List.")

        while True:
            # Check simulation time
            inter_arrival = self.inter_arrival_times[datetime.datetime.fromtimestamp(environment.now).hour]

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
            vessel = self.generate(environment, "Vessel", scenario)
            vessel.env = environment
            vessel.route = nx.dijkstra_path(environment.FG, origin, destination)
            vessel.geometry = nx.get_node_attributes(environment.FG, "geometry")[vessel.route[0]]

            environment.vessels.append(vessel)
            # Move on path
            environment.process(vessel.move())


class Simulation(core.Identifiable):
    """
    A class to generate vessels from a database

    Parameters
    ----------
    simulation_start: datetime
        The start time of the simulation.
    graph: networkx graph
        The graph that is used for the simulation.
    scenario:
        scenario with vessels - should be coupled to the database
    """

    def __init__(self, simulation_start, graph, scenario=None):
        """
        Initialization


        """
        self.environment = simpy.Environment(initial_time=time.mktime(simulation_start.timetuple()))
        self.environment.FG = graph
        self.environment.routes = pd.DataFrame.from_dict(
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

        self.environment.vessels = []

    def add_vessels(
        self,
        origin=None,
        destination=None,
        vessel=None,
        vessel_generator: VesselGenerator = None,
        arrival_distribution=1,
        arrival_process="Markovian",
    ):
        """
        Make arrival process in the environment. Add one vessel  on one input vessel, or with the vessel generator

        Parameters
        ----------
        vessel_generator:      VesselGenerator
            The vessel generator that is used to generate vessels. Optional. If not specified, the vessel should be specified.
            If specified, the vessel should be None, but origin, destination, arrival_distribution and arrival_process should be specified.
        origin: str
            The origin of the vessel. Must be specified if vessel_generator is specified.
        destination: str
            The destination of the vessel. Must be specified if vessel_generator is specified.
        arrival_distribution: int or list
            The amount of vessels that enter the simulation per hour. Must be specified if vessel_generator is specified.
            If int, it is the average number of vessels per hour over the entire day.
            If list, it is the average number of vessels per hour for each hour of the day. List must have length of 24 (one entry for each hour).
        arrival_process: str
            The process of arrivals. Must be specified if vessel_generator is specified.
            Choose between "Markovian" or "Uniform".
        vessel: Vessel
            A vessel object with a route between origin and destination. Optional.
            If specified, the vessel_generator should be None, and origin, destination, arrival_distribution and arrival_process are ignored.
        """

        if vessel_generator == None:
            self.environment.vessels.append(vessel)
            self.environment.process(vessel.move())

        else:
            self.environment.process(
                vessel_generator.arrival_process(
                    self.environment,
                    origin,
                    destination,
                    arrival_distribution,
                    self.scenario,
                    arrival_process,
                )
            )

    def run(self, duration=24 * 60 * 60):
        """
        Run the simulation

        duration:   float
            specify the duration of the simulation in seconds
        """

        self.environment.run(until=self.environment.now + duration)
