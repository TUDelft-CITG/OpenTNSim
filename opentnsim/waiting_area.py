from opentnsim.core import  HasResource, Identifiable, Log, Locatable
from opentnsim.graph.mixins import OnNode

class IsWaitingArea(OnNode, HasResource, Log, Identifiable, Locatable):
    def __init__(self, node, capacity, *args,**kwargs):
        super().__init__(node, nr_resources=capacity, *args, **kwargs)

    def register_waiting_area(self):
        self.env.graph.nodes[self.node]['Waiting Area'] = self