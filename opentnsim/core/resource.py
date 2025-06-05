"""
Mixin classes related to resources.

The following classes are provided:
- HasResource

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


class HasResource(SimpyObject):
    """Mixin class: Something that has a resource limitation, a resource request must be granted before the object can be used.

    Parameters
    -----------
    nr_resources: int, default=1
        nr of requests that can be handled simultaneously, optional, default=1
    priority: bool, default=False
        if True, prioritized resources can be handled. optional, default=False.
    args, kwargs:
        passed to SimpyObject. Must at least contain parameter env: simpy.Environment.

    Attributes
    -----------
    resource: simpy.Resource or simpy.PriorityResource
        the resource that is used to limit the nr of requests that can be handled simultaneously.
    env: simpy.Environment
        the simpy environment that is used to run the simulation.
    """

    def __init__(self, nr_resources: int = 1, priority: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.resource = (
            simpy.PriorityResource(self.env, capacity=nr_resources) if priority else simpy.Resource(self.env, capacity=nr_resources)
        )