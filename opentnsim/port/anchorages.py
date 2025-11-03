from opentnsim.core import HasResource, Identifiable, Log, Locatable
from opentnsim.output import HasOutput

class IsAnchorage(HasResource, Locatable, Identifiable, Log, HasOutput):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(self,capacity,*args,**kwargs):
        super().__init__(nr_resources = capacity,*args, **kwargs)