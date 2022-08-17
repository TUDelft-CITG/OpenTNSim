from enum import Enum
from dataclasses import dataclass

class horizontal_tidal_window_method(Enum):
    critical_cross_current = 'Critical cross-current'
    point_based = 'Point-based'

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
    Type = ['Type', '==']

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

class accessibility(Enum):
    non_accessible = 0
    accessible = -1

class tidal_period(Enum):
    Flood = 'Flood'
    Ebb = 'Ebb'

class current_velocity_type(Enum):
    CurrentVelocity = 'Current velocity'
    LongitudinalCurrent = 'Longitudinal current'
    CrossCurrent = 'Cross-current'

@dataclass
class vessel_specifications:
    vessel_characteristics: dict  # {item of vessel_characteristics class: user-defined value,...}
    vessel_method: str  # string containing the operators between the vessel characteristics (symbolized by x): e.g. '(x and x) or x'
    vessel_direction: str  # item of vessel_direction class

    def characteristic_dicts(self):
        characteristic_dicts = {}
        for characteristic in self.vessel_characteristics:
            characteristic_dict = {
                characteristic.value[0]: [characteristic.value[1], self.vessel_characteristics[characteristic]]}
            characteristic_dicts = characteristic_dicts | characteristic_dict
        return characteristic_dicts

@dataclass
class horizontal_tidal_window_specifications:
    window_method: str  # item of window_method class
    current_velocity_values: dict  # {tidal_period.Flood.value: user-defined value or item from accessibility class,...}
    current_velocity_ranges: dict = dict  # if window_method is point-based: {tidal_period.Ebb.value: user-defined value,...}

@dataclass
class vertical_tidal_window_specifications:
    ukc_s: dict  # {tidal_period.Flood.value: user-defined value or item from accessibility class,...}
    ukc_p: dict  # {tidal_period.Flood.value: user-defined value or item from accessibility class,...}
    fwa: dict  # {tidal_period.Flood.value: user-defined value or item from accessibility class,...}

@dataclass
class vertical_tidal_window_input:
    vessel_specifications: vessel_specifications  # class
    window_specifications: vertical_tidal_window_specifications  # class

@dataclass
class horizontal_tidal_window_input:
    vessel_specifications: vessel_specifications  # class
    window_specifications: horizontal_tidal_window_specifications  # class
    condition: dict  # {'Origin':node, 'Destination': node}
    data: list  # Calculated input: [node,]