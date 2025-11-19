"""
Mixin classes related to resources.

The following classes are provided:
- HasResource

"""
# package(s) for documentation, debugging, saving and loading
import logging

# package(s) related to the simulation
import simpy
from simpy import FilterStore

# use OpenCLSim objects for core objects (identifiable is imported for later use)
from openclsim.core import SimpyObject

# get logger
logger = logging.getLogger(__name__)

class HasLength(SimpyObject):
    """Mixin class: Something with a length. The length is modelled as a storage capacity

    Parameters
    -----------
    length: float
        length that can be requested
    remaining_length: float, default=0
        length that is still available at the beginning of the simulation.
    args, kwargs:
        passed to SimpyObject. Must at least contain parameter env: simpy.Environment.

    Attributes
    -----------
    length: simpy.Container
        the container that is used to limit the length that can be requested.
    pos_length: simpy.Container
        the container that is used to limit the length that can be requested.
    """

    def __init__(self, length: float, remaining_length: float = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialization"""
        self.resource = simpy.Container(self.env, capacity=length, init=remaining_length)
        #self.pos_length = simpy.Container(self.env, capacity=length, init=remaining_length)

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

class PriorityFilterStore(FilterStore):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_with_priority(self, vessel, filter, priority=0):
        vessels_in_waiting_area_old = self.get_queue.copy()
        request = self.get(filter)
        request.priority = priority
        request.obj = vessel
        if priority and vessels_in_waiting_area_old:
            for number_in_line,waiting_vessels in enumerate(vessels_in_waiting_area_old):
                if not waiting_vessels.priority:
                    break
            self.get_queue.insert(number_in_line, self.get_queue.pop())
        return request