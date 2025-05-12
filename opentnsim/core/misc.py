"""
Miscellaneous mixin classes for object in OpenTNSim.

The following classes are provided:
- Neighbours
- ExtraMetadata

"""
# packkage(s) for documentation, debugging, saving and loading
import logging
import warnings
import deprecated
from typing import Union

# math packages
import random
import numpy as np

# spatial libraries
import pyproj
import shapely
import shapely.geometry
from shapely import Geometry
import networkx as nx

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy

# Use OpenCLSim objects for core objects (identifiable is imported for later use.)
from openclsim.core import Identifiable, Locatable, SimpyObject, Log
import opentnsim.graph_module

# get logger
logger = logging.getLogger(__name__)


class Neighbours:
    """Mixin class: Can be added to a locatable object (list)

    Parameters
    ----------
    travel_to: list
            list of locatables to which can be travelled
    args, kwargs:
        passed to SimpyObject. Must at least contain parameter env: simpy.Environment.

    Attributes
    -----------
    neighbours: list
        list of locatables to which can be travelled.
    """

    def ___init(self, travel_to: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.neighbours = travel_to



class ExtraMetadata:
    """Mixin class: store all leftover keyword arguments as metadata property (use as last mixin)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        # store all other properties as metadata
        self.metadata = kwargs