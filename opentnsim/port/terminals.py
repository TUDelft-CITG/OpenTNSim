from opentnsim.core import  HasResource, Identifiable, Locatable, Log, HasLength, HasResource
from opentnsim.output import HasOutput
from simpy import FilterStore

import numpy as np

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

class IsTerminal(Identifiable, Locatable, HasOutput, Log):

    def __init__(self, env, berths, *args,**kwargs):
        """ Creates a terminal

        Input
        -----
        berths : a list of IsBerth, IsJetty and/or IsQuay objects
        """
        self.berths = PriorityFilterStore(env=env)
        for berth in berths:
            self.berths.put(berth)
        super().__init__(env=env,*args, **kwargs)


    def request_berth(self, vessel):
        request = self.resource.get_with_priority(vessel,
                                                  (lambda request: (request.depth > vessel.T) and (request.length > vessel.L)),
                                                  priority=0)
        return request

class IsJetty(HasResource, Identifiable, Log):
    def __init__(self, length, depth, capacity=1, *args, **kwargs):
        super().__init__(nr_resources=capacity, *args, **kwargs)
        self.length = length
        self.depth = depth
        self.capacity = capacity

class IsQuay(HasLength, Identifiable, Log):
    def __init__(self, length, depth, *args, **kwargs):
        super().__init__(length=length, *args, **kwargs)
        self.length = length
        self.depth = depth
        self.capacity = np.inf


