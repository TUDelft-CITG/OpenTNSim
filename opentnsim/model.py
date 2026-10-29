"""Vessel generator."""

# packkage(s) for documentation, debugging, saving and loading
import datetime
import logging

# package(s) for data handling
import random
import pandas as pd


# package(s) related to time, space and id
import time
import datetime
import math

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy
import networkx as nx

# import core from self
from opentnsim.core import Identifiable, Movable
from opentnsim.energy.mixins import ConsumesEnergy
from opentnsim.energy.utils import check_if_route_has_depth

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

    def generate(self, environment, vessel_name: str, fleet_distribution=None, scenario=None):
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

        if fleet_distribution == None:
            vessel_info = self.vessel_database.sample(n=1, random_state=int(1000 * random.random()))
        else:
            vessel_info = self.vessel_database.sample(n=1, weights=fleet_distribution, random_state=int(1000 * random.random()))
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

        if self.loaded is True:
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
        Make arrival process in the simulation environment. Vessels with a route
        and between origin and destination are generated according to the arrival distribution.
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
            If list, it is the average number of vessels per hour for each hour of the day.
            List must have length of 24 (one entry for each hour).
        scenario:     str
            The scenario that is assigned to the generated vessel.
        arrival_process:  str
            process of arrivals. choose between "Markovian" or "Uniform".
        """

        # Create an array with inter-arrival times -- average number of seconds between arrivals
        if type(arrival_distribution) == int or type(arrival_distribution) == float:
            self.inter_arrival_times = [3600 / arrival_distribution] * 24

        elif type(arrival_distribution) == list and len(arrival_distribution) == 24:
            self.inter_arrival_times = [3600 / n for n in arrival_distribution]

        elif type(arrival_distribution) == list:
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
                    "No other arrival processes are yet defined. You can add them to "
                    "transport_network_analysis/vessel_generator.py."
                )

            # Create a vessel
            vessel = self.generate(environment, "Vessel", fleet_distribution, scenario)
            vessel.output = {}
            # core.Output.vessel_dependent_output(vessel)
            vessel.env = environment
            vessel.route = nx.dijkstra_path(environment.graph, origin, destination)
            vessel.geometry = nx.get_node_attributes(environment.graph, "geometry")[vessel.route[0]]
            environment.vessels.append(vessel)
            # Move on path
            process = environment.process(vessel.move())
            vessel.process = process


class SimulationEnvironment(simpy.Environment):
    """SimPy environment containing the state of a vessel simulation."""

    def __init__(
        self,
        graph=None,
        simulation_start=None,
        simulation_duration=None,
        simulation_stop=None,
    ):
        simulation_start = (
            simulation_start or datetime.datetime.now()
        )

        super().__init__(
            initial_time=time.mktime(
                simulation_start.timetuple()
            )
        )

        self.simulation_start = simulation_start
        self.simulation_stop = simulation_stop
        self.simulation_duration = simulation_duration

        if simulation_stop is not None:
            self.simulation_duration = (
                simulation_stop - simulation_start
            )
        elif simulation_duration is not None:
            self.simulation_stop = (
                simulation_start + simulation_duration
            )

        if (
            self.simulation_duration is not None
            and self.simulation_duration.total_seconds() <= 0
        ):
            raise ValueError(
                "Simulation duration must be positive."
            )

        self.graph = None
        if graph is not None:
            self.add_graph(graph)

        self.routes = pd.DataFrame(
            columns=[
                "Origin",
                "Destination",
                "Width",
                "Height",
                "Depth",
                "Route",
            ]
        )

    def add_graph(self, graph: nx.Graph) -> None:
        """Validate and attach a graph to the environment."""

        required_edge_info = {"geometry", "length_m"}

        for u, v, data in graph.edges(data=True):
            missing = required_edge_info - data.keys()

            if missing:
                raise ValueError(
                    f"Edge ({u}, {v}) is missing attributes: "
                    f"{sorted(missing)}"
                )

            length = data["length_m"]
            if not isinstance(length, (int, float)) or length < 0:
                raise ValueError(
                    f"Edge ({u}, {v}) has an invalid length_m: "
                    f"{length!r}"
                )

        self.graph = graph


class Simulation:
    """Coordinate vessel setup and simulation execution."""

    def __init__(
        self,
        graph=None,
        simulation_start=None,
        simulation_duration=None,
        simulation_stop=None,
        scenario=None,
    ):
        simulation_start = (
            simulation_start or datetime.datetime.now()
        )

        self.env = SimulationEnvironment(
            simulation_start=simulation_start,
            simulation_duration=simulation_duration,
            simulation_stop=simulation_stop,
        )

        if graph is not None:
            self.add_graph(graph)

        self.scenario = scenario
        self.output = {}

    def add_graph(self, graph: nx.Graph) -> None:
        """Validate and attach a graph to the simulation."""
        self.env.add_graph(graph)

    def add_vessel(self, vessel, mission):
        """Register a vessel and schedule its mission."""
        if not isinstance(vessel, Identifiable):
            raise TypeError("vessel must inherit from Identifiable.")

        if not isinstance(vessel, Movable):
            raise TypeError("vessel must inherit from Movable.")

        if isinstance(vessel, ConsumesEnergy):
            try:
                check_if_route_has_depth(vessel.env.graph, vessel.edge_route)
            except ValueError as e:
                raise ValueError(f"Vessel ({vessel.name}) ConsumesEnergy, but route lacks GeneralDepth at following edges: {e}")

        if vessel.env is not self.env:
            raise ValueError("Vessel must belong to this simulation's environment.")

        vessel.mission = self.env.process(mission(self.env, vessel))

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
        """Add an individual vessel or a vessel arrival process."""

        if vessel_generator is None:
            if vessel is None:
                raise ValueError(
                    "Provide either a vessel or a vessel_generator."
                )

            process = self.env.process(vessel.move())
            vessel.process = process

            if (
                hasattr(vessel, "metadata")
                and "arrival_time" not in vessel.metadata
            ):
                vessel.metadata["arrival_time"] = (
                    self.env.simulation_start
                )

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
        """Run the simulation until its stop time or until events are exhausted."""

        if self.env.simulation_stop is not None:
            until = self.env.simulation_stop.timestamp()
        elif self.env.simulation_duration is not None:
            until = (
                self.env.now
                + self.env.simulation_duration.total_seconds()
            )
        else:
            self.env.run()
            return

        self.env.run(until=until)
