"""Vessel generator."""

# package(s) related to time, space and id
import json
import logging
import uuid

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy
import networkx as nx

# spatial libraries
import pyproj
import shapely.geometry

# additional packages
import datetime, time
import random

logger = logging.getLogger(__name__)


class VesselGenerator:
    """
    A class to generate vessels from a database
    """

    def __init__(self, vessel_type, vessel_database, random_seed = 4, *args, **kwargs):
        """ Initialization """

        self.vessel_type = vessel_type
        self.vessel_database = vessel_database

        random.seed(random_seed)
    
    def generate(self, environment, name, path = None, scenario = None):
        """ Generate a vessel """

        vessel_info = self.vessel_database.sample(n = 1, random_state = int(1000 * random.random()))
        vessel_data = {}

        vessel_data["env"] = environment
        vessel_data["name"] = name

        if scenario:
            vessel_info = vessel_info[vessel_info["scenario"] == scenario]

        for key in vessel_info:
            if key == "vessel_id":
                vessel_data["id"] = vessel_info[key].values[0]
            else:
                vessel_data[key] = vessel_info[key].values[0]
        
        if path:
            vessel_data["route"] = path
            vessel_data["geometry"] = nx.get_node_attributes(environment.FG, "geometry")[path[0]]
        
        return self.vessel_type(**vessel_data)