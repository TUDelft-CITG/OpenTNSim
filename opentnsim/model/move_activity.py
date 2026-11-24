"""
--- Add description for the MoveActivity class. ---
"""

# %% DEPENDENCIES
# generic
import shapely
import pandas as pd
import networkx as nx
import datetime as dt
import geopandas as gpd

# openclsim imports
from openclsim.core.log import LogState
from openclsim.model import GenericActivity
from opentnsim.core import Movable


# %% NETWORK MOVE ACTIVITY CLASS
class NetworkMoveActivity(GenericActivity):
    def __init__(self, mover, route, *args, **kwargs):
        """
        Initialize a network move activity.

        Parameters
        ----------
        mover : opentnsim.core.Movable
            The entity that will move along the route.
        route : list
            A list of nodes representing the route to follow. Must be the string names
            of the nodes in the graph. The graph is expected to be stored in the
            simulation environment as `env.graph`.
        """
        # mover must be an opentnsim Movable instance
        if not isinstance(mover, Movable):
            raise ValueError("mover must be an instance of opentnsim.core.Movable")

        # make sure to set the proper logger
        self.mover._logger = "v1"

        # store the mover and route
        self.mover = mover
        self.route = route

        # initialize the base class
        super().__init__(*args, **kwargs)

    def main_process_function(self, activity_log, env):
        """
        Main process function for the network move activity.

        Parameters
        ----------
        activity_log : object
            The activity log associated with this activity.
        env : simpy.Environment
            The simulation environment.
        """

        # prepare the activity data
        start_time = env.now
        args_data = {
            "env": env,
            "activity_log": activity_log,
            "activity": self,
        }
        yield from self.pre_process(args_data)

        # copy the activity id and set route
        self.mover.activity_id = activity_log.id
        self.mover.route = self.route

        # move the entity along the route until it reaches the final node's geometry
        start_mover = env.now
        while True:
            yield from self.mover.move()
            if (
                self.mover.geometry
                == nx.get_node_attributes(env.graph, "geometry")[self.mover.route[-1]]
            ):
                break  # actively stop the openclsim activity

        args_data["start_preprocessing"] = start_time
        args_data["start_activity"] = start_mover
        yield from self.post_process(**args_data)
