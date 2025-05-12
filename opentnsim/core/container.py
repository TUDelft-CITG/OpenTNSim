"""
Mixin classes related to containers.

The following classes are provided:
- HasContainer

"""
# packkage(s) for documentation, debugging, saving and loading
import logging

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation
import simpy

# Use OpenCLSim objects for core objects (identifiable is imported for later use.)
from openclsim.core import SimpyObject

# get logger
logger = logging.getLogger(__name__)


class HasContainer(SimpyObject):
    """Mixin class: Something with a container, modelled as a storage capacity

    Parameters
    -----------
    capacity: float
        the capacity of the container, which may either be continuous (like water) or discrete (like apples)
    level: int, default=0
        level of the container at the beginning of the simulation
    total_requested: int, default=0
        total amount that has been requested at the beginning of the simulation
    args, kwargs:
        passed to SimpyObject. Must at least contain parameter env: simpy.Environment.

    Attributes
    -----------
    container: simpy.Container
        the container that is used to limit the amount that can be requested.
    total_requested: int
        total amount that has been requested.
    """

    def __init__(self, capacity: float, level: float = 0, total_requested: float = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.container = simpy.Container(self.env, capacity, init=level)
        self.total_requested = total_requested

    @property
    def is_loaded(self):
        """Return if the container is loaded"""
        return True if self.container.level > 0 else False

    @property
    def filling_degree(self):
        """return the filling degree of the container"""
        return self.container.level / self.container.capacity

    @property
    def max_load(self):
        """return the maximum cargo to load"""
        # independent of trip
        return self.container.capacity - self.container.level