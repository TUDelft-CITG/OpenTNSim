from enum import Enum
from dataclasses import dataclass,field,fields

import opentnsim.vessel_traffic_service as vessel_traffic_service
import numpy as np
import pandas as pd
import typing
import itertools

class vessel_characteristic_conditions(Enum):
    min_ge_Length = ['min_ge_Length', '>=']
    min_gt_Length = ['min_gt_Length', '>']
    max_le_Length = ['max_le_Length', '<=']
    max_lt_Length = ['max_lt_Length', '<']
    min_ge_Draught = ['min_ge_Draught', '>=']
    min_gt_Draught = ['min_gt_Draught', '>']
    max_le_Draught = ['max_le_Draught', '<=']
    max_lt_Draught = ['max_lt_Draught', '<']
    min_ge_UKC = ['min_ge_UKC', '>=']
    min_gt_UKC = ['min_gt_UKC', '>']
    max_le_UKC = ['max_le_UKC', '<=']
    max_lt_UKC = ['max_lt_UKC', '<']
    type = ['type', '==']
    terminal = ['terminal','.isin(']
    visited_terminal = ['visited_terminal','.isin(']
    bound_from = ['bound_from','==']
    bound_to = ['bound_to','==']


class horizontal_tidal_window_method(Enum):
    maximum = 'Maximum'
    point_based = 'Point-based'

class accessibility(Enum):
    inaccessible = -999
    accessible = 999

class tidal_period(Enum):
    Flood = 'Flood'
    Ebb = 'Ebb'

@dataclass
class vessel_specifications:
    vessel_characteristics: dict  # {item of vessel_characteristics class: user-defined value,...}
    vessel_method: str  # string containing the operators between the vessel characteristics (symbolized by x): e.g. '(x and x) or x'

    def __post_init__(self):
        if not self.vessel_characteristics:
            self.vessel_characteristics = {}
        if not self.vessel_method:
            self.vessel_method = ''

    def characteristic_dicts(self):
        characteristic_dicts = {}
        for characteristic in self.vessel_characteristics:
            characteristic_dict = {
                characteristic.value[0]: [characteristic.value[1], self.vessel_characteristics[characteristic]]}
            characteristic_dicts = {**characteristic_dicts,**characteristic_dict}
        return characteristic_dicts

@dataclass
class horizontal_tidal_window_specifications:
    window_method: str  # item of window_method class
    current_velocity_values: typing.Union[pd.DataFrame,dict]

@dataclass
class vertical_tidal_window_specifications:
    ukc_s: list = field(default_factory=list)  # {tidal_period.Flood.value: user-defined value or item from accessibility class,...}
    ukc_p: list = field(default_factory=list)  # {tidal_period.Flood.value: user-defined value or item from accessibility class,...}
    ukc_r: list = field(default_factory=list)  # {tidal_period.Flood.value: user-defined value or item from accessibility class,...}
    fwa: list = field(default_factory=list)  # {tidal_period.Flood.value: user-defined value or item from accessibility class,...}

    def __post_init__(self):
        if not self.ukc_s:
            self.ukc_s = 0.0
        if not self.ukc_p:
            self.ukc_p = 0.0
        if not self.ukc_r:
            self.ukc_r = [0.0,0.0]
        if not self.fwa:
            self.ukc_s = 0.0

@dataclass
class vertical_tidal_window_input:
    vessel_specifications: vessel_specifications  # class
    window_specifications: vertical_tidal_window_specifications  # class

@dataclass
class horizontal_tidal_window_input:
    vessel_specifications: vessel_specifications  # class
    window_specifications: horizontal_tidal_window_specifications  # class
    data: list  # Calculated input: [node,]

class NetworkProperties:
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def append_vertical_tidal_restriction_to_network(self,network,node,vertical_tidal_window_input):
        """ Function: appends vertical tidal restrictions to the node of the network

            Input:
                - network: a graph constructed with the DiGraph class of the networkx package
                - node: the name string of the node in the given network
                - vertical_tidal_window_input: assembly of specific information that defines the restriction (see specific input classes in the notebook)
        """
        # Loops over the number of types of restrictions that may hold for different classes of vessels
        columns = vessel_characteristic_conditions.__members__.keys()
        column_operators = [values.value[1] for values in vessel_characteristic_conditions.__members__.values()]

        specification_df = pd.DataFrame(columns=columns)
        if 'Vertical tidal restriction' not in network.nodes[node]:
            network.nodes[node]['Vertical tidal restriction'] = {}
        if 'Specifications' not in network.nodes[node]['Vertical tidal restriction']:
            network.nodes[node]['Vertical tidal restriction']['Specifications'] = specification_df
        else:
            specification_df = network.nodes[node]['Vertical tidal restriction']['Specifications']

        #Loops over the number of types of restrictions that may hold for different classes of vessels
        for input_data in vertical_tidal_window_input:
            window_specifications = input_data.window_specifications

            # Unravels the boolean operators between the restrictions and appends it to a list
            vessel_method_list = []
            index = len(specification_df)

            sign = input_data.vessel_specifications.vessel_method
            combinations = sign.split(' or ')
            new_combination_indexes = []
            new_combination_index = index
            for combination in combinations:
                new_combination_index += combination.count('x')
                new_combination_indexes.append(new_combination_index)
            # Unpacks the data for the different vessel criteria and appends it to a list
            vessel_characteristics = input_data.vessel_specifications.characteristic_dicts()
            values = {key: value[1] for key, value in vessel_characteristics.items()}
            values = {key: value if isinstance(value, list) else [value] for key, value in values.items()}
            value_keys = values.keys()
            values = itertools.product(*(values[Name] for Name in values))
            new_values = []
            for combination in values:
                new_values.append({key: value for key, value in zip(value_keys, combination)})

            for i, combination in enumerate(new_values):
                if not combination:
                    specification_df.loc[index + i, 'Data'] = node
                    specification_df.loc[index + i, 'Restriction'] = window_specifications
                for restriction_type,restriction in combination.items():
                    specification_df.loc[index + i, 'Data'] = node
                    specification_df.loc[index + i, 'Restriction'] = window_specifications
                    specification_df.loc[index + i, restriction_type] = restriction
                    if index in new_combination_indexes:
                        index += 1

        for column, operator in zip(columns, column_operators):
            if operator in ['>=', '>']:
                specification_df.loc[specification_df[specification_df[column].isna()].index, column] = -999.
            elif operator in ['<=', '<']:
                specification_df.loc[specification_df[specification_df[column].isna()].index, column] = 999.

        network.nodes[node]['Vertical tidal restriction']['Specifications'] = specification_df

    def append_horizontal_tidal_restriction_to_network(self,network,node,horizontal_tidal_window_input):
        """ Function: appends horizontal tidal restrictions to the node of the network

            Input:
                - network: a graph constructed with the DiGraph class of the networkx package
                - node: the name string of the node in the given network
                - horizontal_tidal_window_input: assembly of specific information that defines the restriction (see specific input classes in the notebook)
        """
        # Loops over the number of types of restrictions that may hold for different classes of vessels
        columns = vessel_characteristic_conditions.__members__.keys()
        column_operators = [values.value[1] for values in vessel_characteristic_conditions.__members__.values()]

        specification_df = pd.DataFrame(columns=columns)
        if 'Horizontal tidal restriction' not in network.nodes[node]:
            network.nodes[node]['Horizontal tidal restriction'] = {}
        if 'Specifications' not in network.nodes[node]['Horizontal tidal restriction']:
            network.nodes[node]['Horizontal tidal restriction']['Specifications'] = specification_df
        else:
            specification_df = network.nodes[node]['Horizontal tidal restriction']['Specifications']

        for input_data in horizontal_tidal_window_input:
            # Unpacks the data for flood and ebb and appends it to a list
            window_specifications = input_data.window_specifications
            current_velocity_data = input_data.data

            # Unravels the boolean operators between the restrictions and appends it to a list
            vessel_method_list = []
            index = len(specification_df)

            sign = input_data.vessel_specifications.vessel_method
            combinations = sign.split(' or ')
            new_combination_indexes = []
            new_combination_index = index
            for combination in combinations:
                new_combination_index += combination.count('x')
                new_combination_indexes.append(new_combination_index)
            # Unpacks the data for the different vessel criteria and appends it to a list
            vessel_characteristics = input_data.vessel_specifications.characteristic_dicts()
            values = {key: value[1] for key, value in vessel_characteristics.items()}
            values = {key: value if isinstance(value, list) else [value] for key, value in values.items()}
            value_keys = values.keys()
            values = itertools.product(*(values[Name] for Name in values))
            new_values = []
            for combination in values:
                new_values.append({key:value for key,value in zip(value_keys,combination)})

            for i, combination in enumerate(new_values):
                for restriction_type,restriction in combination.items():
                    specification_df.loc[index+i, 'Data'] = current_velocity_data
                    specification_df.loc[index+i, 'Restriction'] = window_specifications
                    specification_df.loc[index+i, restriction_type] = restriction
                    if index in new_combination_indexes:
                        index += 1

        specification_df = specification_df.reset_index(drop=True)
        for column,operator in zip(columns,column_operators):
            if operator in ['>=','>']:
                specification_df.loc[specification_df[specification_df[column].isna()].index,column] = -999.
            elif operator in ['<=','<']:
                specification_df.loc[specification_df[specification_df[column].isna()].index,column] = 999.

        network.nodes[node]['Horizontal tidal restriction']['Specifications'] = specification_df
