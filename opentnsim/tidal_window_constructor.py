from enum import Enum
from dataclasses import dataclass,field,fields

import opentnsim.vessel_traffic_service as vessel_traffic_service

class horizontal_tidal_window_method(Enum):
    critical_cross_current = 'Critical cross-current'
    point_based = 'Point-based'

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
    def append_vertical_tidal_restriction_to_network(network,node,vertical_tidal_window_input):
        """ Function: appends vertical tidal restrictions to the node of the network

            Input:
                - network: a graph constructed with the DiGraph class of the networkx package
                - node: the name string of the node in the given network
                - vertical_tidal_window_input: assembly of specific information that defines the restriction (see specific input classes in the notebook)
        """
        #Specifies two parameters in the dictionary with a corresponding data structure of lists
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

    def append_horizontal_tidal_restriction_to_network(network,env,node,horizontal_tidal_window_input):
        """ Function: appends horizontal tidal restrictions to the node of the network

            Input:
                - network: a graph constructed with the DiGraph class of the networkx package
                - node: the name string of the node in the given network
                - horizontal_tidal_window_input: assembly of specific information that defines the restriction (see specific input classes in the notebook)
        """
        # Specifies two parameters in the dictionary with a corresponding data structure of lists
        network.nodes[node]['Info']['Horizontal tidal restriction']['Type'] = [[], [], []]
        network.nodes[node]['Info']['Horizontal tidal restriction']['Data'] = {}
        network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'] = [[], [], [], [], [], []]

        # Loops over the number of types of restrictions that may hold for different classes of vessels
        for input_data in enumerate(horizontal_tidal_window_input):
            # Unpacks the data for flood and ebb and appends it to a list
            current_velocity_value = []
            for info in input_data[1].window_specifications.current_velocity_values:
                current_velocity_value.append(input_data[1].window_specifications.current_velocity_values[info])

            # Dependent on the type of restriction, additional information should be unpacked and appended to a list
            current_velocity_range = []
            # - if no necessary:
            if input_data[1].window_specifications.current_velocity_ranges == dict:
                current_velocity_range = []
            # - elif necessary:
            else:
                for info in input_data[1].window_specifications.current_velocity_ranges:
                    current_velocity_range.append(input_data[1].window_specifications.current_velocity_ranges[info])

            # Appends the specific data regarding the type of the restriction to data structure
            network.nodes[node]['Info']['Horizontal tidal restriction']['Type'][0].append(input_data[1].window_specifications.window_method)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Type'][1].append(current_velocity_value)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Type'][2].append(current_velocity_range)

            # Unpacks the data for the different vessel criteria and appends it to a list
            vessel_characteristics_type = []
            vessel_characteristics_spec = []
            vessel_characteristics_value = []
            vessel_characteristics = input_data[1].vessel_specifications.characteristic_dicts()
            for info in vessel_characteristics:
                vessel_characteristics_type.append(info)
                vessel_characteristics_spec.append(vessel_characteristics[info][0])
                vessel_characteristics_value.append(vessel_characteristics[info][1])

            # Unravels the boolean operators between the restrictions and appends it to a list
            vessel_method_list = []
            sign_list = input_data[1].vessel_specifications.vessel_method.split()
            for sign in sign_list:
                if sign[0] != '(' and sign[-1] != ')' and sign != 'x':
                    vessel_method_list.append(sign)

            # Appends the specific data regarding the properties for which the restriction holds to data structure
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][0].append(vessel_characteristics_type)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][1].append(vessel_characteristics_value)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][2].append(input_data[1].vessel_specifications.vessel_direction)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][3].append(vessel_characteristics_spec)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][4].append(vessel_method_list)
            network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][5].append([input_data[1].condition['Origin'], input_data[1].condition['Destination']])
            # Unpacks the specific current velocity data used for the restriction and appends it to the data structure
            # - if pre-calculated data from the model should be used:

            if input_data[1].data[0] in list(network.nodes):
                for n in network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][5][input_data[0]]:
                    node_index = list(env.vessel_traffic_service.hydrodynamic_information['Stations'].values).index(input_data[1].data[0])
                    network.nodes[node]['Info']['Horizontal tidal restriction']['Data'][n] = list(env.vessel_traffic_service.hydrodynamic_information['Current velocity'][node_index].values)
            else:
                for n in network.nodes[node]['Info']['Horizontal tidal restriction']['Specification'][5][input_data[0]]:
                    network.nodes[node]['Info']['Horizontal tidal restriction']['Data'][n] = list(input_data[1].data[1].values)