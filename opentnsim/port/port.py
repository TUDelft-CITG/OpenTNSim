from opentnsim.core import SimpyObject, Identifiable, Movable, Log
from opentnsim.graph import OnNode
from opentnsim.tidal_accessibility import check_if_route_contains_restrictions

import datetime
import pandas as pd
import numpy as np
import networkx as nx
import simpy
import warnings

import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

def merge_figures(fig1, fig2):
    new_fig, new_ax = plt.subplots()

    for fig in (fig1, fig2):
        for ax in fig.axes:
            for line in ax.get_lines():
                new_ax.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    label=line.get_label())

    new_ax.legend()
    return new_fig, new_ax


class IsPartofPort:
    def __init__(self, port, *args, **kwargs):
        if not isinstance(port,IsPort):
            raise ValueError("'port' should be an IsPort-object")
        self.port = port
        super().__init__(*args, **kwargs)


class HasPortAccess(Movable, Log):
    def __init__(self, bound, *args, **kwargs):
        self.bound = bound
        self.routes_sailed = []
        super().__init__(*args, **kwargs)
        self.on_pass_node_functions.append(self.request_port_entry)
        self.env.vessels.append(self)


    def determine_sailing_time(self):
        route = self.route
        sailing_information = self.env.vessel_traffic_service.provide_sailing_time(self, route)
        sailing_time = sailing_information["Time"].cumsum().values[0]
        return sailing_time


    def request_port_entry(self, origin, at_terminal = False, leaving_port = False, parallel_process = None, loading = False):
        if not at_terminal:
            if 'Port Entry' not in self.env.graph.nodes[origin].keys():
                return
            port = self.env.graph.nodes[origin]['Port Entry'].port
            if not self.terminal.port == port:
                return
            elif 'port_accessed' in dir(self) and self.port_accessed == port:
                return
        else:
            port = self.terminal.port

        berth = self.select_berth(origin)
        yield from port.communicate_port_accessibility_info(self, origin, berth, leaving_port=leaving_port, parallel_process = parallel_process, loading = loading)


    def select_berth(self, origin, leaving_port = False):
        berth = None
        if not leaving_port:
            if hasattr(self,'berth'):
                return self.berth
            available_berth_time_slots = self.terminal.determine_berth_availability(self, origin)
            berth = self.terminal.select_berth_for_vessel(available_berth_time_slots)
        return berth


    def request_port_exit(self, origin, parallel_process = None, loading = False):
        try:
            yield from self.request_port_entry(origin, at_terminal = True, leaving_port = True, parallel_process=parallel_process, loading = loading)
        except simpy.Interrupt:
            return


    def generate_logbook_with_directed_distances(self):
        first_index = 0
        df = pd.DataFrame(self.logbook)
        corrected_df = pd.DataFrame()
        for index, route in enumerate(self.routes_sailed):
            mask = df.index > first_index
            mask2 = df[mask].Message.apply(lambda x: route[-1] in x and 'stop' in x)
            last_index = df[mask][mask2].iloc[0].name
            df_route = df[(df.index >= first_index) & (df.index <= last_index)]
            maximum_sailed_distance = df_route.Value.max()
            if index == 1:
                df_route.loc[:, "delta_distance"] = df_route["Value"].diff()
                df_route.loc[:, "delta_distance"] = np.where(df_route["delta_distance"] >= 0,
                                                             df_route["delta_distance"],
                                                             (maximum_sailed_distance - df_route["Value"].shift()) + df_route["Value"])

                df_route.loc[:, "delta_distance"] = df_route["delta_distance"].ffill()
                df_route.loc[:, 'Value'] = df_route.Value.shift(1) - df_route.delta_distance
            first_index = last_index + 1
            corrected_df = pd.concat([corrected_df, df_route])
        corrected_df = corrected_df.ffill()
        return corrected_df


    def plot_time_distance_diagram(self):
        df = self.generate_logbook_with_directed_distances()
        fig, ax  = plt.subplots()
        ax.plot(df.Value, df.Timestamp, label=self.name)
        plt.close()
        return fig


class IsPort(SimpyObject, Identifiable):
    def __init__(self, port_entry_nodes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.port_entry_nodes = port_entry_nodes
        if not len(self.port_entry_nodes):
            raise ValueError('The port will not be accessible for vessels. It needs port entry_nodes')
        for port_entry_node in self.port_entry_nodes:
            IsPortEntry(env=self.env,node=port_entry_node,port=self)
        self.anchorage_areas = []
        self.terminals = []
        self.env.vessels = []
        if 'ports' not in dir(self.env):
            self.env.ports = []
        self.env.ports.append(self)


    def plot_vessels(self, vessels = None):
        fig, ax = plt.subplots()
        plt.close()
        if vessels is None:
            vessels = self.env.vessels

        for vessel in vessels:
            fig_vessel = vessel.plot_time_distance_diagram()
            fig, ax = merge_figures(fig,fig_vessel)
            plt.close()

        handles, labels = ax.get_legend_handles_labels()

        ax.legend(handles, labels)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.925),
            frameon = False,
            borderaxespad=0)

        return fig

    def communicate_vessel_to_hold_position(self, vessel, origin, parallel_process, leaving_port=False, loading = False):
        while not parallel_process.processed:
            port_availability_df = self.get_accessibility_info(vessel, origin, leaving_port=leaving_port)
            port_availability_df['Combined'] = port_availability_df.all(axis=1)
            with pd.option_context("future.no_silent_downcasting", True):
                port_availability_df = port_availability_df.ffill()

            current_time = datetime.datetime.fromtimestamp(vessel.env.now)
            future_events = port_availability_df[port_availability_df.index > current_time]
            waiting_time = 3600.
            if not future_events.empty:
                future_event = future_events.iloc[0]
                if future_event.Combined and len(future_events) > 1:
                    vessel.berth.update_planning(vessel, new_release_time = future_event.name)
                waiting_time = future_event.name - current_time
            yield vessel.env.timeout(waiting_time.total_seconds())


    def request_updated_resource_plannings(self, resources_requested = {}):
        for resource in resources_requested.items():
            pass


    def communicate_port_accessibility_info(self, vessel, origin, berth = None, leaving_port = False, parallel_process = None, loading = False):
        if not parallel_process is None:
            yield from self.communicate_vessel_to_hold_position(vessel, origin, parallel_process,leaving_port=leaving_port, loading = loading)

        port_availability_df = self.get_accessibility_info(vessel, origin, berth, leaving_port=leaving_port)
        waiting_events = self.determine_vessel_waiting_events(vessel, port_availability_df, leaving_port)
        if waiting_events is None:
            self.communicate_trip_not_possible(vessel, leaving_port)

        if leaving_port:
            if len(waiting_events):
                yield from self.communicate_vessel_to_wait(vessel, origin, waiting_events)

        else:
            #Assign vessel to berth
            vessel.terminal.assign_berth_to_vessel(vessel, origin, berth)

            #Tides (already pre-calculated)
            #Terminal (planning should be updated and can be updated)

            #Move vessel to the anchorage area if required
            if len(waiting_events):
                total_waiting_time = sum(waiting_events.values())
                required_to_sail_to_anchorage_area = self.determine_if_vessel_needs_to_sail_to_the_anchorage_area(vessel, origin, total_waiting_time)
                if required_to_sail_to_anchorage_area:
                    yield from self.communicate_vessel_to_sail_to_anchorage(vessel, origin)

                yield from self.communicate_vessel_to_wait(vessel, origin, waiting_events)

        yield from self.communicate_vessel_to_continue_trip(vessel)


    def communicate_vessel_to_wait(self, vessel, origin, waiting_events):
        waiting_event = self.env.event()
        waiting_event.succeed()

        for waiting_event_reason, waiting_event_time in waiting_events.items():
            vessel.log_entry_v0(f"Waiting for {waiting_event_reason} start",
                                vessel.env.now,
                                vessel.distance,
                                vessel.env.graph.nodes[origin]["geometry"])
            yield vessel.env.timeout(waiting_event_time)
            vessel.log_entry_v0(f"Waiting for {waiting_event_reason} stop",
                                vessel.env.now,
                                vessel.distance,
                                vessel.env.graph.nodes[origin]["geometry"])

        yield waiting_event


    def determine_if_vessel_needs_to_sail_to_the_anchorage_area(self, vessel, origin, waiting_time):
        sail_to_anchorage_area = False
        vessel_traffic_service = self.env.vessel_traffic_service
        nearest_anchorage_area = vessel.find_nearest_anchorage_area(origin)
        route_to_anchorage_area = nx.dijkstra_path(self.env.graph, origin, nearest_anchorage_area.node)
        route_after_anchorage_area = nx.dijkstra_path(self.env.graph, nearest_anchorage_area.node, vessel.route[-1])
        sailing_time_to_terminal = vessel_traffic_service.provide_sailing_time(vessel, vessel.route)["Time"].sum()
        sailing_time_to_anchorage_area = vessel.determine_sailing_time_to_anchorage_area(route_to_anchorage_area)
        new_sailing_time_to_terminal = vessel_traffic_service.provide_sailing_time(vessel, route_after_anchorage_area)["Time"].sum()
        delay_of_sailing_to_terminal = new_sailing_time_to_terminal - sailing_time_to_terminal + sailing_time_to_anchorage_area
        if delay_of_sailing_to_terminal <= waiting_time:
            sail_to_anchorage_area = True
        return sail_to_anchorage_area


    def determine_vessel_waiting_events(self, vessel, port_availability_df, leaving_port = False):
        port_availability_df['Combined'] = port_availability_df.all(axis=1)
        with pd.option_context("future.no_silent_downcasting", True):
            port_availability_df = port_availability_df.ffill()

        def get_waiting_time_reason(lst):
            if not lst:
                return ""  # No False columns
            elif len(lst) == 1:
                return lst[0]
            else:
                return ", ".join(lst[:-1]) + " and " + lst[-1]

        current_time = datetime.datetime.fromtimestamp(vessel.env.now)
        previous_events = port_availability_df[port_availability_df.index <= current_time]
        future_events = port_availability_df[port_availability_df.index > current_time]
        last_previous_event_index = previous_events.index.max()
        if pd.isna(last_previous_event_index):
            previous_event = port_availability_df.iloc[0:0]
        else:
            previous_event = port_availability_df.loc[[last_previous_event_index]]
        port_availability_df = pd.concat([previous_event, future_events])
        port_availability_df.index.values[0] = current_time

        cols_to_check = port_availability_df.columns.drop('Combined')
        port_availability_df['Reason'] = port_availability_df[cols_to_check].apply(
            lambda row: get_waiting_time_reason(list(row[row.eq(False)].index)),
            axis=1)

        port_available_df = port_availability_df[port_availability_df['Combined'] == True]
        if port_available_df.empty:
            return None

        waiting_time_end = port_available_df.iloc[0].name
        waiting_events = port_availability_df.loc[:waiting_time_end]
        waiting_reasons = waiting_events['Reason'][:-1]
        waiting_times = (waiting_events.index.to_series().shift(-1) - waiting_events.index).apply(lambda x: x.total_seconds())

        waiting_events = {}
        for waiting_reason, waiting_time in zip(waiting_reasons,waiting_times):
            waiting_events[waiting_reason] = waiting_time

        vessel.port_accessed = self
        return waiting_events


    def communicate_vessel_to_sail_to_anchorage(self, vessel, origin):
        yield from vessel.sail_to_anchorage(origin)


    def communicate_vessel_to_wait_for_berth_availability(self, vessel, origin, waiting_time):
        yield from vessel.wait_for_berth_availability(origin, waiting_time)


    def communicate_vessel_to_wait_for_tidal_window(self, vessel, origin, waiting_time):
        yield from vessel.wait_for_tidal_window(origin, waiting_time)


    def communicate_vessel_to_continue_trip(self, vessel):
        vessel.routes_sailed.append(vessel.route)
        yield from []


    def communicate_trip_not_possible(self, vessel, leaving_port):
        if leaving_port:
            warnings.warn(f"The port is not accessible for the outbound trip of vessel with id: {vessel.id}")
        else:
            warnings.warn(f"The port is not accessible for the inbound trip of vessel with id: {vessel.id}")
        raise simpy.exceptions.Interrupt('Vessel trip not possible.')


    def plan_new_route_for_vessel(self, vessel):
        new_route = None
        origin = vessel.route[-1]
        if vessel.next_destination is not None:
            destination = vessel.next_destination
            new_route = nx.dijkstra_path(vessel.env.graph,origin,destination)
        elif len(vessel.next_terminals):
            next_terminal = vessel.next_terminals[-1]
            vessel.next_terminals = vessel.next_terminals[1:]
            berth = vessel.request_terminal_access(vessel, origin)
        return new_route


    def get_accessibility_info(self, vessel, origin, berth = None, leaving_port = False):
        df_tidal_availability = self.get_tidal_availability_info(vessel)
        df_terminal_availability = self.get_terminal_availability_info(vessel, origin, berth, leaving_port)

        #Combine the dataframes
        port_availability_df = pd.concat([df_tidal_availability,df_terminal_availability],axis=1)
        port_availability_df = port_availability_df.sort_index()
        with pd.option_context("future.no_silent_downcasting", True):
            port_availability_df = port_availability_df.ffill()
            port_availability_df = port_availability_df.ffill()
        return port_availability_df


    def get_terminal_availability_info(self, vessel, origin, berth = None, leaving_port = False):
        df_terminal_availability = pd.DataFrame()
        if not leaving_port:
            df_terminal_availability = vessel.terminal.provide_terminal_availability_info(vessel, origin, berth)
        return df_terminal_availability


    def get_tidal_availability_info(self, vessel):
        tide_bound = check_if_route_contains_restrictions(vessel)
        route = vessel.route
        time_start = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now))
        sailing_time = vessel.determine_sailing_time()
        sailing_time = np.max([pd.Timedelta(seconds=sailing_time), pd.Timedelta(hours=48)])
        time_end = np.datetime64(datetime.datetime.fromtimestamp(vessel.env.now) + sailing_time)
        df_tidal_availability = pd.DataFrame(columns=['Accessibility'])
        if tide_bound:
            df_tidal_availability = vessel.env.vessel_traffic_service.provide_tidal_windows(vessel, route, time_start, time_end)[0]
        df_tidal_availability['Tide'] = df_tidal_availability['Accessibility'] == 'Accessible'
        return df_tidal_availability[['Tide']]


class IsPortEntry(SimpyObject, OnNode, IsPartofPort):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env.graph.nodes[self.node]['Port Entry'] = self