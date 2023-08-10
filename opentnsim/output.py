import numpy as np
import pandas as pd
from copy import deepcopy
import datetime
import pytz

class HasOutput:
    def __init__(self, *args, **kwargs):
        self.output = {}
        super().__init__(*args, **kwargs)

        if self.__class__.__name__ == 'Vessel':
            #Route-dependent output
            self.output['origin'] = ''
            self.output['destination'] = ''
            self.output['route'] = []
            self.output['bound'] = ''
            self.output['anchorage'] = ''
            self.output['turning_basin'] = ''
            self.output['terminal'] = ''
            self.output['berth'] = ''
            self.output['length'] = np.NaN
            self.output['beam'] = np.NaN
            self.output['draught'] = np.NaN
            self.output['sailing_distance'] = np.NaN
            self.output['sailing_time'] = np.NaN
            self.output['waiting_times_in_anchorages'] = []
            self.output['waiting_times_at_terminals'] = []
            self.output['turning_times'] = []
            self.output['(un)loading_times'] = []

            #Edge-dependent output
            self.output['current_node'] = ''
            self.output['next_node'] = ''
            self.output['speed'] = np.NaN
            self.output['heading'] = np.NaN
            self.output['water_level'] = np.NaN
            self.output['MBL'] = np.NaN
            self.output['net_ukc'] = np.NaN
            self.output['gross_ukc'] = np.NaN
            self.output['ship_related_ukc_factors'] = {}
            self.output['limiting current velocity'] = np.NaN

            #Historic output
            self.output['vessel_arrival'] = deepcopy(self.metadata['arrival_time'])
            self.output['sailed_routes'] = []
            self.output['visited_anchorages'] = []
            self.output['visited_turning_basins'] = []
            self.output['visited_terminals'] = []
            self.output['visited_berths'] = []

        elif self.__class__.__name__ == 'IsTerminal' or self.__class__.__name__ == 'IsJettyTerminal':
            for berth in [berth.name for berth in self.resource.items]:
                #Visit-dependent output
                self.output[berth] = {}
                self.output[berth]['vessel_information'] = {}
                self.output[berth]['vessel_arrival'] = pd.NaT
                self.output[berth]['vessel_departure'] = pd.NaT
                self.output[berth]['vessel_berthing_times'] = np.NaN
                self.output[berth]['vessel_(un)loading_time'] = np.NaN
                self.output[berth]['vessel_waiting_time'] = np.NaN
                self.output[berth]['visited_vessels'] = []

        elif self.__class__.__name__ == 'IsAnchorage':
            self.output['vessel_information'] = {}
            self.output['vessel_arrival'] = pd.NaT
            self.output['vessel_departure'] = pd.NaT
            self.output['vessel_waiting_time'] = np.NaN
            self.output['visited_vessels'] = []

        elif self.__class__.__name__ == 'IsTurningBasin':
            self.output['vessel_information'] = {}
            self.output['vessel_arrival'] = pd.NaT
            self.output['vessel_departure'] = pd.NaT
            self.output['vessel_turning_time'] = np.NaN
            self.output['visited_vessels'] = []

    def update_route_status_report(self,move_stop=False):
        if not move_stop:
            self.output['length'] = self.L
            self.output['beam'] = self.B
            self.output['draught'] = self.T
            self.output['route'] = self.route
            self.output['bound'] = self.bound
            self.output['sailed_routes'].append(self.route)
            self.output['origin'] = self.route[0]
            self.output['destination'] = self.route[-1]
            self.output['sailing_distance'] = 0.
            self.output['sailing_time'] = 0.
            for node in self.route:
                if 'Anchorage' in self.env.FG.nodes[node]:
                    self.output['anchorage'] = self.env.FG.nodes[node]['Anchorage'][0].name
                if 'Turning basin' in self.env.FG.nodes[node] and self.bound == 'inbound' and self.env.FG.nodes[node]['Turning Basin'][0].length <= self.L:
                    self.output['turning_basin'] = self.env.FG.nodes[node]['Turning Basin'][0].name
            if self.metadata['terminal_of_call'].size:
                self.output['terminal'] = self.metadata['terminal_of_call'][0]
            if self.metadata['berth_of_call'].size:
                self.output['berth'] = self.metadata['berth_of_call'][0]

        # if move_stop and self.log['Status']:
        #     correction = deepcopy(self.log['Status'])[-1]['sailing_distance']
        #     if 'anchorage' in self.output['sailed_routes'][-1]:
        #         correction = self.env.vessel_traffic_service.provide_sailing_distance_over_route(self, self.route_after_anchorage[:-1])['Distance'].sum()
        #     for index, info in enumerate(list(reversed(self.log['Status']))):
        #         index = len(self.log['Status']) - index - 1
        #         if info['route'] != self.output['sailed_routes'][-1]:
        #             break
        #         if info['bound'] == 'inbound':
        #             self.log['Status'][index]['sailing_distance'] -= correction
        #         else:
        #             self.log['Status'][index]['sailing_distance'] = -info['sailing_distance']

    def update_turing_basin_status_report(self,turning_basin,turning_stop=False):
        turning_basin.output['vessel_information'] = deepcopy(self.output)
        if not turning_stop:
            turning_basin.output['vessel_arrival'] = self.env.now
            turning_basin.output['vessel_departure'] = self.env.now
            turning_basin.output['vessel_turning_time'] = pd.Timedelta(0, 's')
        if turning_stop:
            self.output['visited_turning_basins'].append(turning_basin.name)
            turning_basin.output['visited_vessels'].append(self.name)
            turning_basin.output['vessel_departure'] = self.env.now
            turning_basin.output['vessel_turning_time'] = deepcopy(turning_basin.output)['vessel_departure'] - deepcopy(turning_basin.output)['vessel_arrival']
            self.output['turning_times'].append(deepcopy(turning_basin.output)['vessel_turning_time'])
        return

    def update_waiting_status(self,priority=False,availability=False,tidal_window=False,waiting_stop=False):
        if not waiting_stop:
            self.output['waiting_start'] = self.env.now
            if 'waiting_time' not in self.output.keys():
                self.output['waiting_time'] = {'Priority':pd.Timedelta(0,'s'),'Availability': pd.Timedelta(0, 's'), 'Tidal window': pd.Timedelta(0, 's')}
        else:
            if availability:
                self.output['waiting_time']['Availability'] = pd.Timedelta(int(self.env.now-self.output['waiting_start']),'s')
            elif tidal_window:
                self.output['waiting_time']['Tidal window'] = pd.Timedelta(int(self.env.now-self.output['waiting_start']),'s')
            elif priority:
                self.output['waiting_time']['Priority'] = pd.Timedelta(int(self.env.now - self.output['waiting_start']), 's')
        return

    def update_anchorage_status_report(self,anchorage,departure=False):
        anchorage.output['vessel_information'] = deepcopy(self.output)
        if not departure:
            anchorage.output['vessel_arrival'] = self.env.now
            anchorage.output['vessel_departure'] = self.env.now
            anchorage.output['vessel_waiting_time'] = deepcopy(self.output)['waiting_time']
        if departure:
            self.output['visited_anchorages'].append(anchorage.name)
            anchorage.output['visited_vessels'].append(self.name)
            anchorage.output['vessel_departure'] = self.env.now
            anchorage.output['vessel_waiting_time'] = deepcopy(self.output)['waiting_time']
            self.output['waiting_times_in_anchorages'].append(deepcopy(self.output)['waiting_time'])
            try:
                del(self.output['waiting_start'])
                del(self.output['waiting_time'])
            except:
                pass
        return

    def update_terminal_berth_status_report(self,terminal,berth,departure=False):
        terminal.output[berth.name]['vessel_information'] = deepcopy(self.output)
        if not departure:
            terminal.output[berth.name]['vessel_arrival'] = self.env.now
            terminal.output[berth.name]['vessel_berthing_times'] = self.metadata['t_berthing']
            if 'waiting_time' in self.output.keys():
                self.output['waiting_times_at_terminals'].append([deepcopy(self.output)['waiting_time']])
                del (self.output['waiting_time'])
            else:
                self.output['waiting_times_at_terminals'].append([{'Priority': pd.Timedelta(0, 's'), 'Availability': pd.Timedelta(0, 's'), 'Tidal window': pd.Timedelta(0, 's')}])
            terminal.output[berth.name]['vessel_waiting_time'] = deepcopy(self.output)['waiting_times_at_terminals'][-1]
            self.output['(un)loading_times'].append(self.metadata['t_(un)loading'][0])
            terminal.output[berth.name]['vessel_berthing_times'] = self.metadata['t_berthing']
            terminal.output[berth.name]['vessel_(un)loading_time'] = self.metadata['t_(un)loading'][0]
        if departure:
            self.output['visited_terminals'].append(terminal.name)
            self.output['visited_berths'].append(berth.name)
            terminal.output[berth.name]['visited_vessels'].append(self.name)
            terminal.output[berth.name]['vessel_departure'] = self.env.now
            if 'waiting_time' in self.output.keys():
                self.output['waiting_times_at_terminals'][-1].append(deepcopy(self.output)['waiting_time'])
            else:
                self.output['waiting_times_at_terminals'][-1].append({'Priority': pd.Timedelta(0, 's'), 'Availability': pd.Timedelta(0, 's'), 'Tidal window': pd.Timedelta(0, 's')})
            terminal.output[berth.name]['vessel_waiting_time'] = deepcopy(self.output)['waiting_times_at_terminals'][-1]
            try:
                del(self.output['waiting_start'])
                del(self.output['waiting_time'])
            except:
                pass
        return

    def update_sailing_status_report(self,current_node,next_node,edge):
        self.output['current_node'] = current_node
        self.output['next_node'] = next_node
        if "Terminal" in self.env.FG.edges[edge].keys() and self.metadata['terminal_of_call'].size and self.metadata['terminal_of_call'][0] in self.env.FG.edges[edge]['Terminal'].keys():
            self.output['speed'] = 0.
        else:
            self.output['speed'] = self.env.vessel_traffic_service.provide_speed(self,edge[:2])
        if self.bound == 'outbound':
            self.output['heading'] = 180 - self.env.vessel_traffic_service.provide_heading(self,edge)
        else:
            self.output['heading'] = self.env.vessel_traffic_service.provide_heading(self,edge)
        if current_node == edge[1]:
            self.output['sailing_distance'] += self.env.FG.edges[edge]['Info']['length']
            self.output['sailing_time'] += self.env.FG.edges[edge]['Info']['length']/self.output['speed']
        self.output['MBL'],self.output['water_level'],_ = self.env.vessel_traffic_service.provide_water_depth(self,current_node)

        #Rule-dependent vessel output
        if 'Vertical tidal restriction' in self.env.FG.nodes[current_node]['Info'].keys():
            self.output['net_ukc'],self.output['gross_ukc'],_,_,self.output['ship_related_ukc_factors'],_ = self.env.vessel_traffic_service.provide_ukc_clearance(self,current_node)

        if 'Horizontal tidal restriction' in self.env.FG.nodes[current_node]['Info'].keys():
            time_index = np.absolute(self.env.vessel_traffic_service.hydrodynamic_information.TIME.values - pd.Timestamp(datetime.datetime.fromtimestamp(self.env.now,tz=pytz.utc)).to_datetime64()).argmin()
            _,self.output['limiting current velocity'] = self.env.vessel_traffic_service.provide_governing_current_velocity(self,current_node,time_index,time_index+1)

        return deepcopy(self.output)