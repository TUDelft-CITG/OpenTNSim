import numpy as np

class Output():
    """ mixing class: collection of unfinished functions that should store the output to the vessels and the network in the future development of OpenTNSim"""

    def general_output(sim):
        sim.output['Avg_turnaround_time'] = []
        sim.output['Avg_sailing_time'] = []
        sim.output['Avg_waiting_time'] = []
        sim.output['Avg_service_time'] = []
        sim.output['Total_throughput'] = []

    def vessel_dependent_output(vessel):
        vessel.output['route'] = []
        vessel.output['speed'] = []
        vessel.output['sailing_time'] = []
        vessel.output['freeboard'] = []
        vessel.output['throughput'] = []
        vessel.output['n_encouters'] = []
        vessel.output['n_overtaking'] = []
        vessel.output['n_overtaken'] = []
        vessel.output['terminals_called'] = []
        vessel.output['anchorages_used'] = []
        vessel.output['waiting_time'] = []
        vessel.output['service_time'] = []
        vessel.output['total_sailing_time'] = []
        vessel.output['total_waiting_time'] = []
        vessel.output['total_service_time'] = []
        vessel.output['turnaround_time'] = []

    def node_dependent_output(network):
        for node in network.nodes:
            if 'Anchorage' in network.nodes[node].keys():
                network.nodes[node]['Anchorage'][0].output['Anchorage_time'] = []
                network.nodes[node]['Anchorage'][0].output['Anchorage_availability'] = []
                network.nodes[node]['Anchorage'][0].output['Anchorage_waiting_time'] = []
                network.nodes[node]['Anchorage'][0].output['Anchorage_occupancy'] = []

            if 'Turning basin' in network.nodes[node].keys():
                network.nodes[node]['Turning basin'][0].output['Turning_basin_time'] = []
                network.nodes[node]['Turning basin'][0].output['Turning_basin_availability'] = []
                network.nodes[node]['Turning basin'][0].output['Turning_basin_waiting_time'] = []
                network.nodes[node]['Turning basin'][0].output['Turning_basin_occupancy'] = []

    def edge_dependent_output(network):
        for edge in network.edges:
            network.edges[edge]['Output'] = {'v_traffic': [],
                                             'h_req_traffic': [],
                                             'throughput_traffic': [],
                                             'n_encounters': [],
                                             'n_overtakes': [],
                                             't_passages': [],
                                             'n_passages': []}

            if 'Terminal' in network.edges[edge].keys():
                if network.edges[edge]['Terminal'][0].type == 'jetty':
                    units = len(network.edges[edge]['Terminal'][0].jetty_lengths)
                else:
                    units = 1
                network.edges[edge]['Terminal'][0].output['Berth_vessels_served'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Berth_arrival_times'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Berth_departure_times'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Berth_net_throughput'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Berth_waiting_time'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Berth_service_time'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Berth_productivity'] = [[] for i in range(units)]
                network.edges[edge]['Terminal'][0].output['Mean_berth_productivity'] = []
                network.edges[edge]['Terminal'][0].output['Berth_interarrival_times'] = []
                network.edges[edge]['Terminal'][0].output['Berth_interdeparture_times'] = []
                network.edges[edge]['Terminal'][0].output['Mean_berth_occupancy'] = []
                network.edges[edge]['Terminal'][0].output['Terminal_availability'] = []
                network.edges[edge]['Terminal'][0].output['Terminal_waiting_time'] = []
                network.edges[edge]['Terminal'][0].output['Terminal_service_time'] = []
                network.edges[edge]['Terminal'][0].output['Terminal_occupancy'] = []
                network.edges[edge]['Terminal'][0].output['Terminal_vessels_served'] = []
                network.edges[edge]['Terminal'][0].output['Terminal_net_throughput'] = []

    def real_time_terminal_output(terminal,berth,vessel):
        if network.edges[edge]['Terminal'][0].type == 'jetty':
            terminal_type = 'Berth'
        elif network.edges[edge]['Terminal'][0].type == 'terminal':
            terminal_type = 'Terminal'
        for berth in terminal.berths:
            terminal.output[terminal_type+'_vessels_served'][berth].extend(vessel)
            terminal.output[terminal_type+'_vessels_lengths'][berth].extend(vessel.L)
            terminal.output[terminal_type+'_arrival_times'][berth].extend(vessel.arrival_time)
            terminal.output[terminal_type+'_departure_times'][berth].extend(vessel.departure_time)
            terminal.output[terminal_type+'_net_throughput'][berth].extend(vessel.throughput_departure - vessel.throughput_arrival)
            terminal.output[terminal_type+'_waiting_time'][berth].extend(vessel.waiting_time)
            terminal.output[terminal_type+'_service_time'][berth].extend(vessel.service_time)
            terminal.output[terminal_type+'_turnaround_time'][berth].extend(vessel.service_time + vessel.waiting_time)
            terminal.output[terminal_type+'_productivity'][berth].extend(vessel.service_time / (vessel.waiting_time + vessel.service_time))

    def post_process_terminal_output(terminal, berth, simulation_duration):
        if terminal.type == 'jetty':
            for berth in terminal.berths:
                terminal.output['Mean_berth_productivity'][berth] = np.sum(terminal.output['Berth_sevice_time']) / (np.sum(terminal.output['Berth_waiting_time']) + np.sum(terminal.output['Berth_service_time']))
                terminal.output['Berth_interarrival_times'][berth]= [terminal.output['Berth_arrival_times'][berth][t]-terminal.output['Berth_arrival_times'][berth][t-1] for t in range(len(terminal.output['Berth_arrival_times'][berth])) if t>0]
                terminal.output['Berth_interdeparture_times'][berth] = [terminal.output['Berth_departure_times'][berth][t] - terminal.output['Berth_departure_times'][berth][t - 1] for t in range(len(terminal.output['Berth_departure_times'][berth])) if t > 0]
                terminal.output['Berth_occupancy'][berth] = np.sum([terminal.output['Berth_departure_times'][berth][t]- terminal.output['Berth_arrival_times'][berth][t] for t in range(len(terminal.output['Berth_vessels_served'][berth]))])/simulation_duration
                terminal.output['Berth_availability'][berth] = 1-terminal.output['Mean_berth_occupancy'][berth]
                terminal.output['Effective_berth_occupancy'][berth] = np.sum([terminal.output['Berth_vessels_lengths'][berth][t]*terminal.output['Berth_productivity'][berth][t]*terminal.output['Berth_turnaround_time'][berth][t] for t in range(len(terminal.output['Terminal_vessels_served'][berth]))]) / (terminal.length*simulation_duration)
                terminal.output['Potential_berth_availability'][berth] = 1-terminal.output['Mean_effective_berth_occupancy'][berth]
                terminal.output['Mean_berth_waiting_time'][berth] = np.mean(terminal.output['Berth_waiting_time'][berth])
                terminal.output['Mean_berth_service_time'][berth] = np.mean(terminal.output['Berth_service_time'][berth])
                terminal.output['Mean_berth_turnaround_time'][berth] = np.mean(terminal.output['Berth_turnaround_time'][berth])
                terminal.output['Berth_total_net_throughput'][berth] = np.sum(terminal.output['Berth_net_throughput'][berth])
            terminal.output['Terminal_occupancy'] = np.mean(terminal.output['Mean_berth_occupancy'])
            terminal.output['Terminal_availability'] = 1-np.mean(terminal.output['Mean_berth_occupancy'])
            terminal.output['Effective_terminal_occupancy'] = np.mean(terminal.output['Mean_effective_berth_occupancy'])
            terminal.output['Potential_terminal_availability'] = 1-np.mean(terminal.output['Mean_effective_berth_occupancy'])
            terminal.output['Terminal_waiting_time'] = np.mean(terminal.output['Berth_waiting_time'])
            terminal.output['Terminal_service_time'] = np.mean(terminal.output['Berth_service_time'])
            terminal.output['Terminal_turnaround_time'] = np.mean(terminal.output['Berth_turnaround_time'])
            terminal.output['Terminal_total_net_throughput'] = np.sum(terminal.output['Berth_net_throughput'])

        elif network.edges[edge]['Terminal'][0].type == 'terminal':
            terminal.output['Mean_terminal_productivity'] = np.sum(terminal.output['Berth_sevice_time'][berth]) / (np.sum(terminal.output['Berth_waiting_time'][berth]) + np.sum(terminal.output['Berth_sevice_time'][berth]))
            terminal.output['Terminal_interarrival_times'] = [terminal.output['Berth_arrival_times'][berth][t] - terminal.output['Berth_arrival_times'][berth][t - 1] for t in range(len(terminal.output['Berth_arrival_times'][berth])) if t > 0]
            terminal.output['Terminal_interdeparture_times'] = [terminal.output['Berth_departure_times'][berth][t] - terminal.output['Berth_departure_times'][berth][t - 1] for t in range(len(terminal.output['Berth_departure_times'][berth])) if t > 0]
            terminal.output['Quay_occupancy'] = [terminal.output['Terminal_vessels_lengths'][berth][t]*terminal.output['Terminal_turnaround_time'][berth][t] for t in range(len(terminal.output['Terminal_vessels_served'][berth]))] / (terminal.length*simulation_duration)
            terminal.output['Effective_quay_occupancy'] = np.sum([terminal.output['Terminal_vessels_lengths'][berth][t]*terminal.output['Terminal_productivity'][berth][t]*terminal.output['Terminal_turnaround_time'][berth][t] for t in range(len(terminal.output['Terminal_vessels_served'][berth]))]) / (terminal.length*simulation_duration)
            terminal.output['Terminal_occupancy'] = [] #information about the number of cranes per vessels required (not yet included in the model)
            terminal.output['Terminal_availability'] = [] #idem
            terminal.output['Effective_terminal_occupancy'] = [] #idem
            terminal.output['Potential_terminal_availability'] = [] #idem
            terminal.output['Terminal_waiting_time'] = np.mean(terminal.output['Terminal_waiting_time'])
            terminal.output['Terminal_service_time'] = np.mean(terminal.output['Terminal_service_time'])
            terminal.output['Terminal_turnaround_time'] = np.mean(terminal.output['Terminal_turnaround_time'])
            terminal.output['Terminal_total_net_throughput'] = np.sum(terminal.output['Terminal_net_throughput'])