class IsAnchorage(core.HasResource,core.Identifiable, core.Log, output.HasOutput):
    """Mixin class: Something has waiting area object properties as part of the lock complex [in SI-units]:
            creates a waiting area with a waiting_area resource which is requested when a vessels wants to enter the area with limited capacity"""

    def __init__(self,capacity,*args,**kwargs):
        super().__init__(capacity = capacity,*args, **kwargs)