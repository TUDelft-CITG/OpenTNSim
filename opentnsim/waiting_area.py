from opentnsim.core import  HasResource, Identifiable, Log, Locatable
from opentnsim.graph import OnNode

class IsWaitingArea(HasResource, OnNode, Log, Identifiable, Locatable):
    def __init__(self,capacity,*args,**kwargs):
        super().__init__(nr_resources = capacity,*args, **kwargs)
        self.env.graph.nodes[self.node]['Waiting Area'] = self