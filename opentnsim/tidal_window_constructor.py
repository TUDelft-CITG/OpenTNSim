from enum import Enum
from dataclasses import dataclass,field,fields

import opentnsim.vessel_traffic_service as vessel_traffic_service
import numpy as np
import pandas as pd
import typing

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
    vessel_direction: str  # item of vessel_direction class

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
        #Specifies two parameters in the dictionary with a corresponding data structure of lists
        if 'Type' not in network.nodes[node]['Info']['Vertical tidal restriction']:
            network.nodes[node]['Info']['Vertical tidal restriction']['Type'] = [[], [], [], []]
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'] = [[], [], [], [], [], []]

        #Loops over the number of types of restrictions that may hold for different classes of vessels
        for input_data in vertical_tidal_window_input:
            for field in fields(input_data.window_specifications):
                data = getattr(input_data.window_specifications, field.name)
                if data:
                    globals()[field.name] = data
                else:
                    globals()[field.name] = 0
                    if field.name == 'ukc_r':
                        globals()[field.name] = [0,0]

            # Appends the specific data regarding the type of the restriction to data structure
            network.nodes[node]['Info']['Vertical tidal restriction']['Type'][0].append(ukc_s)
            network.nodes[node]['Info']['Vertical tidal restriction']['Type'][1].append(ukc_p)
            network.nodes[node]['Info']['Vertical tidal restriction']['Type'][2].append(ukc_r)
            network.nodes[node]['Info']['Vertical tidal restriction']['Type'][3].append(fwa)

            # Unpacks the data for the different vessel criteria and appends it to a list
            vessel_characteristics_type = []
            vessel_characteristics_spec = []
            vessel_characteristics_value = []
            vessel_characteristics = input_data.vessel_specifications.characteristic_dicts()
            for info in vessel_characteristics:
                vessel_characteristics_type.append(info)
                vessel_characteristics_spec.append(vessel_characteristics[info][0])
                vessel_characteristics_value.append(vessel_characteristics[info][1])

            # Unravels the boolean operators between the restrictions and appends it to a list
            vessel_method_list = []
            sign_list = input_data.vessel_specifications.vessel_method.split()
            for sign in sign_list:
                if sign[0] != '(' and sign[-1] != ')' and sign != 'x':
                    vessel_method_list.append(sign)

            # Appends the specific data regarding the properties for which the restriction holds to data structure
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][0].append(vessel_characteristics_type)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][1].append(vessel_characteristics_value)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][2].append(input_data.vessel_specifications.vessel_direction)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][3].append(vessel_characteristics_spec)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][3].append(vessel_characteristics_spec)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][4].append(vessel_method_list)
            network.nodes[node]['Info']['Vertical tidal restriction']['Specification'][5].append([])

    def append_horizontal_tidal_restriction_to_network(self,network,node,horizontal_tidal_window_input):
        """ Function: appends horizontal tidal restrictions to the node of the network

            Input:
                - network: a graph constructed with the DiGraph class of the networkx package
                - node: the name string of the node in the given network
                - horizontal_tidal_window_input: assembly of specific information that defines the restriction (see specific input classes in the notebook)
        """
        # Loops over the number of types of restrictions that may hold for different classes of vessels
        if 'Data' not in network.nodes[node]['Info']['Horizontal tidal restriction']:
            network.nodes[node]['Info']['Horizontal tidal restriction']['Data'] = []
            network.nodes[node]['Info']['Horizontal tidal restriction']['Limit'] = []
            network.nodes[node]['Info']['Horizontal tidal restriction']['Type'] = []
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'] = [[], [], [], [], [], []]

        for input_data in horizontal_tidal_window_input:
            # Unpacks the data for flood and ebb and appends it to a list
            window_specifications = input_data.window_specifications
            current_velocity_data = input_data.data
            network.nodes[node]['Info']['Horizontal tidal restriction']['Limit'].append({})
            if window_specifications.window_method == 'Maximum':
                max_flood_current = input_data.window_specifications.current_velocity_values['Flood']
                max_ebb_current = input_data.window_specifications.current_velocity_values['Ebb']
                # tidal_periods = input_data.data['Horizontal tidal periods'].values
                # flood_start_times = [np.datetime64(time[0]) for time in tidal_periods if time[1] == 'Flood Start']
                # ebb_start_times = [np.datetime64(time[0]) for time in tidal_periods if time[1] == 'Ebb Start']
                # cross_current_limit_dataframe = pd.DataFrame(columns=['Limit', 'Tide'])
                # for flood_start, ebb_start in zip(flood_start_times, ebb_start_times):
                #     cross_current_limit_dataframe.at[flood_start - np.timedelta64(1, 'ns'), 'Limit'] = max_ebb_current
                #     cross_current_limit_dataframe.at[flood_start - np.timedelta64(1, 'ns'), 'Tide'] = 'Ebb'
                #     cross_current_limit_dataframe.at[flood_start, 'Limit'] = max_flood_current
                #     cross_current_limit_dataframe.at[flood_start, 'Tide'] = 'Flood'
                #     cross_current_limit_dataframe.at[ebb_start - np.timedelta64(1, 'ns'), 'Limit'] = max_flood_current
                #     cross_current_limit_dataframe.at[ebb_start - np.timedelta64(1, 'ns'), 'Tide'] = 'Flood'
                #     cross_current_limit_dataframe.at[ebb_start, 'Limit'] = max_ebb_current
                #     cross_current_limit_dataframe.at[ebb_start, 'Tide'] = 'Ebb'
                # cross_current_limit_dataframe['Limit'] = cross_current_limit_dataframe.Limit.astype(float)
                network.nodes[node]['Info']['Horizontal tidal restriction']['Limit'][-1]['Flood'] = max_flood_current
                network.nodes[node]['Info']['Horizontal tidal restriction']['Limit'][-1]['Ebb'] = max_ebb_current

            elif window_specifications.window_method == 'Point-based':
                network.nodes[node]['Info']['Horizontal tidal restriction']['Limit'][-1]['Flood'] = input_data.window_specifications.current_velocity_values['Flood']
                network.nodes[node]['Info']['Horizontal tidal restriction']['Limit'][-1]['Ebb'] = input_data.window_specifications.current_velocity_values['Ebb']

            # Appends the specific data regarding the type of the restriction to data structure
            network.nodes[node]['Info']['Horizontal tidal restriction']['Data'].append(current_velocity_data)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Type'].append(window_specifications)

            # Unpacks the data for the different vessel criteria and appends it to a list
            vessel_characteristics_type = []
            vessel_characteristics_spec = []
            vessel_characteristics_value = []
            vessel_characteristics = input_data.vessel_specifications.characteristic_dicts()
            for info in vessel_characteristics:
                vessel_characteristics_type.append(info)
                vessel_characteristics_spec.append(vessel_characteristics[info][0])
                vessel_characteristics_value.append(vessel_characteristics[info][1])

            # Unravels the boolean operators between the restrictions and appends it to a list
            vessel_method_list = []
            sign_list = input_data.vessel_specifications.vessel_method.split()
            for sign in sign_list:
                if sign[0] != '(' and sign[-1] != ')' and sign != 'x':
                    vessel_method_list.append(sign)

            # Appends the specific data regarding the properties for which the restriction holds to data structure
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][0].append(vessel_characteristics_type)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][1].append(vessel_characteristics_value)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][2].append(input_data.vessel_specifications.vessel_direction)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][3].append(vessel_characteristics_spec)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][4].append(vessel_method_list)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][5].append([input_data.condition['Origin'], input_data.condition['Destination']])