from enum import Enum

class vessel_characteristics(Enum):
    min_ge_Length = ['minLength', '>=']
    min_gt_Length = ['minLength', '>']
    max_le_Length = ['maxLength', '<=']
    max_lt_Length = ['maxLength', '<']
    min_ge_Draught = ['minDraught', '>=']
    min_gt_Draught = ['minDraught', '>']
    max_le_Draught = ['maxDraught', '<=']
    max_lt_Draught = ['maxDraught', '<']
    min_ge_Beam = ['minBeam', '>=']
    min_gt_Beam = ['minBeam', '>']
    max_le_Beam = ['maxBeam', '<=']
    max_lt_Beam = ['maxBeam', '<']
    min_ge_UKC = ['minUKC', '>=']
    min_gt_UKC = ['minUKC', '>']
    max_le_UKC = ['maxUKC', '<=']
    max_lt_UKC = ['maxUKC', '<']
    type = ['Type', '==']
    terminal = ['Terminal','.isin(']
    visited_terminal = ['Previous terminal','.isin(']
    bound_from = ['Bound from','==']
    bound_to = ['Bound to','==']

class vessel_direction(Enum):
    inbound = 'inbound'
    outbound = 'outbound'

class vessel_type(Enum):
    GeneralCargo = 'GeneralCargo'
    LiquidBulk = 'LiquidBulk'
    Container = 'Container'
    DryBulk = 'DryBulk'
    MultiPurpose = 'MultiPurpose'
    Reefer = 'Reefer'
    RoRo = 'RoRo'
    Barge = 'Barge'