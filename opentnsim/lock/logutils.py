import pandas as pd
import networkx as nx
import numpy as np
from opentnsim.graph.calculations import calculate_distance_over_network_to_location
from opentnsim.lock.utils import _get_vessels_that_passed_the_lock_chamber
import re

def get_levelling_cycles(lock_chamber):
    """
    Identifies which vessels were present during each lock leveling event.

    Parameters:
    - lock: An object with a `.logbook` attribute (list of dicts with 'Message' and 'Timestamp').
    - vessels: List of vessel objects, each with a `.logbook` attribute and optional `.name`.

    Returns:
    - List of dicts with keys: 'leveling_start', 'leveling_stop', 'vessels_present'
    """
    # Convert lock logbook to DataFrame
    lock_df = pd.DataFrame(lock_chamber.logbook)

    # Extract leveling start/stop events
    gate_closings = lock_df[lock_df["Message"] == "Lock gate closing start"]
    leveling_starts = lock_df[lock_df["Message"] == "Lock levelling start"]
    leveling_stops = lock_df[lock_df["Message"] == "Lock levelling stop"]
    gate_openings = lock_df[lock_df["Message"] == "Lock gate opening stop"]

    # Ensure matching pairs
    levelling_cycles = pd.DataFrame({
        "direction": leveling_starts["Timestamp"].values,
        "gate_closed": gate_closings["Timestamp"].values,
        "leveling_start": leveling_starts["Timestamp"].values,
        "leveling_stop": leveling_stops["Timestamp"].values,
        "gate_opened": gate_openings["Timestamp"].values,
    })

    # Add directions
    directions = lock_df[lock_df["Message"] == "Lock levelling start"]['Geometry'] == lock_chamber.edge[0]
    directions = (~directions).astype(int)
    levelling_cycles['direction'] = directions.values
    return levelling_cycles


def get_index_of_logbook_when_vessel_passes_registration_node(lock_complex, vessel, levelling_idx, start = True):
    vessel_df = pd.DataFrame(vessel.logbook)
    pattern = r"^Sailing from node (\S+) to node (\S+) (start|stop)$"
    msg_type = 'start'
    group_nr_node = 1
    search_range = range(levelling_idx, -1, -1)
    if not start:
        group_nr_node = 2
        msg_type = 'stop'
        search_range = range(levelling_idx, len(vessel_df))

    registration_nodes = set(map(str, lock_complex.registration_nodes))
    index = None
    for i in search_range:
        msg = vessel_df.loc[i, "Message"]
        m = re.search(pattern, msg)
        if m and m.group(3) == msg_type:
            node = m.group(group_nr_node)
            for registration_node in registration_nodes:
                if node == registration_node:
                    index = i
                    break
            if index is not None:
                break
    return index, node


def get_events(vessel, start_idx, stop_idx, route):
    vessel_df = pd.DataFrame(vessel.logbook)
    node_start, node_stop = route[0], route[-1]

    segment = vessel_df.loc[start_idx:stop_idx].copy()
    segment["Event"] = segment["Message"].str.replace(" start| stop", "", regex=True)
    segment["Duration"] = segment["Timestamp"].shift(-1) - segment["Timestamp"]
    segment["Start_Location"] = segment["Geometry"]
    segment["Stop_Location"] = segment["Geometry"].shift(-1)
    segment = segment[segment["Duration"] > pd.Timedelta(0)]
    segment["Start_Distance"] = segment.apply(
        lambda x: calculate_distance_over_network_to_location(
            vessel.env.graph,
            node_start,
            node_stop,
            x.Start_Location)[0],
        axis=1
    )
    segment["Stop_Distance"] = segment.apply(
        lambda x: calculate_distance_over_network_to_location(
            vessel.env.graph,
            node_start,
            node_stop,
            x.Stop_Location)[0],
        axis=1
    )
    segment["Distance"] = segment["Stop_Distance"] - segment["Start_Distance"]
    segment["Speed"] = segment["Distance"]/segment["Duration"].apply(lambda x: x.total_seconds())
    segment = segment.drop(['Message','Timestamp','Value','Geometry'],axis=1)
    return segment


def get_waiting_and_idle_events(vessel, lock_passing_start_idx, levelling_idx, lock_passing_stop_idx, route):
    event_before_levelling = get_events(vessel, lock_passing_start_idx, levelling_idx, route)
    event_after_levelling = get_events(vessel, levelling_idx+1, lock_passing_stop_idx, route)
    events = pd.concat([event_before_levelling, event_after_levelling])
    return events


def get_vessels_per_cycle(lock_chamber):
    vessels = _get_vessels_that_passed_the_lock_chamber(lock_chamber)
    levelling_cycles = get_levelling_cycles(lock_chamber)
    vessels_per_cycle = []
    for cycle_nr, level_event in levelling_cycles.iterrows():
        for vessel in vessels:
            vessel_df = pd.DataFrame(vessel.logbook)
            vessel_df["Timestamp"] = pd.to_datetime(vessel_df["Timestamp"])

            # Find levelling start/stop pairs
            levelling_starts = vessel_df[vessel_df["Message"] == "Waiting for lock levelling start"]
            vessel_in_lock_cycle = False
            for levelling_idx, _ in levelling_starts.iterrows():
                levelling_start = vessel_df.loc[levelling_idx]["Timestamp"]
                levelling_stop = vessel_df.loc[levelling_idx + 1]["Timestamp"]
                if levelling_start <= level_event["leveling_stop"] and levelling_stop >= level_event["leveling_start"]:
                    vessel_in_lock_cycle = True
                    break

            if not vessel_in_lock_cycle:
                continue

            vessel_info = {'cycle_nr': cycle_nr, 'vessel_id': vessel.id}
            vessels_per_cycle.append(vessel_info)
    vessels_per_cycle = pd.DataFrame(vessels_per_cycle)
    return vessels_per_cycle


def get_vessel_delays(lock_chamber):
    lock_complex = lock_chamber.lock_complex
    vessels = _get_vessels_that_passed_the_lock_chamber(lock_chamber)
    vessel_speed_edge = vessels[0]._compute_velocity_on_edge(*lock_chamber.edge)
    levelling_cycles = get_levelling_cycles(lock_chamber)
    vessels_per_cyle = []
    cycle_nr = 0
    for cycle_nr, level_event in levelling_cycles.iterrows():
        for vessel in vessels:
            vessel_df = pd.DataFrame(vessel.logbook)
            vessel_df["Timestamp"] = pd.to_datetime(vessel_df["Timestamp"])

            # Find levelling start/stop pairs
            levelling_starts = vessel_df[vessel_df["Message"] == "Waiting for lock levelling start"]

            vessel_in_lock_cycle = False
            levelling_idx = None
            levelling_start = None
            levelling_stop = None
            for levelling_idx, _ in levelling_starts.iterrows():
                levelling_start = vessel_df.loc[levelling_idx]["Timestamp"]
                levelling_stop = vessel_df.loc[levelling_idx + 1]["Timestamp"]
                if levelling_start <= level_event["leveling_stop"] and levelling_stop >= level_event["leveling_start"]:
                    vessel_in_lock_cycle = True
                    break

            if not vessel_in_lock_cycle:
                continue

            lock_passing_start_idx, node_start = get_index_of_logbook_when_vessel_passes_registration_node(
                lock_complex,
                vessel,
                levelling_idx,
                True
            )
            lock_passing_stop_idx, node_stop = get_index_of_logbook_when_vessel_passes_registration_node(
                lock_complex,
                vessel,
                levelling_idx,
                False
            )

            route = nx.dijkstra_path(vessel.env.graph, node_start, node_stop)

            locking_events = get_waiting_and_idle_events(vessel, lock_passing_start_idx, levelling_idx,
                                                         lock_passing_stop_idx, route)
            levelling_event = get_events(vessel, levelling_idx, levelling_idx + 1, route)
            locking_events = pd.concat([locking_events, levelling_event])
            locking_events = locking_events.sort_index()
            locking_events['Normal_duration'] = locking_events['Distance']/ vessel_speed_edge
            locking_events['Normal_duration'] = locking_events['Normal_duration'].apply(
                lambda x: pd.Timedelta(seconds=x)
            )
            locking_events['Delay'] = locking_events['Duration'] - locking_events['Normal_duration']
            locking_events = locking_events[locking_events['Delay'] > pd.Timedelta(seconds=0)]

            pre_levelling_events = locking_events.loc[:levelling_idx - 1]  # everything before levelling
            post_levelling_events = locking_events.loc[levelling_idx + 1:]  # everything after levelling

            waiting_at_traffic_in_waiting_area = pre_levelling_events.loc[
                pre_levelling_events["Event"].str.contains("for other vessel for lock operation"), "Delay"].sum()
            waiting_for_lock_in_waiting_area = pre_levelling_events.loc[
                pre_levelling_events["Event"].str.contains("Waiting for lock operation"), "Delay"].sum()
            sailing_to_lock_gate = pre_levelling_events.loc[
                pre_levelling_events["Event"].str.contains("Sailing to first"), "Delay"].sum()
            sailing_to_lock_position = pre_levelling_events.loc[
                pre_levelling_events["Event"].str.contains("Sailing to position"), "Delay"].sum()
            waiting_for_other_vessels_to_sail_in = pre_levelling_events.loc[
                pre_levelling_events["Event"].str.contains("for other vessels to enter"), "Delay"].sum()
            doors_closing = pre_levelling_events.loc[
                pre_levelling_events["Event"].str.contains("Waiting for lock gate closing"), "Delay"].sum()
            levelling = levelling_stop - levelling_start
            doors_opening= post_levelling_events.loc[
                post_levelling_events["Event"].str.contains("Waiting for lock gate opening"), "Delay"].sum()
            waiting_for_other_vessels_to_sail_out = post_levelling_events.loc[
                post_levelling_events["Event"].str.contains("for other vessels to leave"), "Delay"].sum()
            leaving_lock_chamber = post_levelling_events.loc[
                post_levelling_events["Event"].str.contains("Sailing to second"), "Delay"].sum()
            leaving_lock_complex = post_levelling_events.loc[
                post_levelling_events["Event"].str.contains("Sailing to lock complex exit"), "Delay"].sum()
            total_delay = waiting_at_traffic_in_waiting_area + waiting_for_lock_in_waiting_area + \
                          sailing_to_lock_gate + sailing_to_lock_position + waiting_for_other_vessels_to_sail_in + \
                          doors_closing + levelling + doors_opening + waiting_for_other_vessels_to_sail_out + \
                          leaving_lock_chamber + leaving_lock_complex

            vessel_info = {
                "operation_nr": cycle_nr,
                "vessel_id": vessel.id,
                "total_delay": pd.Timedelta(seconds=round(total_delay.total_seconds())),
                "waiting time in waiting_area for other vessels (%)":
                    np.round(waiting_at_traffic_in_waiting_area / total_delay * 100, 2),
                "waiting time in waiting_area for available operation (%)":
                    np.round(waiting_for_lock_in_waiting_area / total_delay * 100, 2),
                "delay due to sailing to lock gate (%)":
                    np.round(sailing_to_lock_gate / total_delay * 100, 2),
                "delay due to sailing to position in lock (%)":
                    np.round(sailing_to_lock_position / total_delay * 100, 2),
                "waiting time for other vessels to sail into lock (%)":
                    np.round(waiting_for_other_vessels_to_sail_in / total_delay * 100, 2),
                "waiting time for closing doors (%)":
                    np.round(doors_closing / total_delay * 100, 2),
                "waiting time for levelling (%)":
                    np.round(levelling / total_delay * 100, 2),
                "waiting time for opening doors (%)":
                    np.round(doors_opening / total_delay * 100, 2),
                "waiting time for other vessels to sail out of lock (%)":
                    np.round(waiting_for_other_vessels_to_sail_out / total_delay * 100, 2),
                "delay due to sailing out of lock (%)":
                    np.round(leaving_lock_chamber / total_delay * 100, 2),
                "delay due to sailing away from lock (%)":
                    np.round(leaving_lock_complex / total_delay * 100, 2),
            }
            vessels_per_cyle.append(vessel_info)

    vessel_delays = pd.DataFrame(vessels_per_cyle)
    if vessel_delays.empty:
        return pd.DataFrame(), pd.Series(), pd.Series()

    total_delay = vessel_delays.total_delay.sum()
    waiting_time_in_waiting_area = np.round(
        ((vessel_delays["waiting time in waiting_area for other vessels (%)"] +
          vessel_delays["waiting time in waiting_area for available operation (%)"])*vessel_delays["total_delay"]).sum() / total_delay, 2)
    sailing_delay_to_lock = np.round(
        ((vessel_delays["delay due to sailing to lock gate (%)"] +
          vessel_delays["delay due to sailing to position in lock (%)"])*vessel_delays["total_delay"]).sum() / total_delay, 2)
    waiting_time_in_lock = np.round(
        ((vessel_delays["waiting time for other vessels to sail into lock (%)"] +
          vessel_delays["waiting time for closing doors (%)"] +
          vessel_delays["waiting time for levelling (%)"] +
          vessel_delays["waiting time for opening doors (%)"] +
          vessel_delays["waiting time for other vessels to sail out of lock (%)"])*vessel_delays["total_delay"]).sum() / total_delay, 2)
    sailing_delay_from_lock = np.round(
        ((vessel_delays["delay due to sailing out of lock (%)"] +
          vessel_delays["delay due to sailing away from lock (%)"])*vessel_delays["total_delay"]).sum() / total_delay, 2)

    vessel_delay_locations = {
        "nr_operations":cycle_nr,
        "nr_vessels": len(vessel_delays),
        "min_delay": pd.Timedelta(seconds=round(vessel_delays.total_delay.min().total_seconds())),
        "average_delay": pd.Timedelta(seconds=round(vessel_delays.total_delay.mean().total_seconds())),
        "max_delay": pd.Timedelta(seconds=round(vessel_delays.total_delay.max().total_seconds())),
        "total_delay": total_delay,
        "waiting_area (%)": waiting_time_in_waiting_area,
        "sailing_to_lock (%)": sailing_delay_to_lock,
        "in_lock (%)": waiting_time_in_lock,
        "sailing_from_lock (%)": sailing_delay_from_lock,
    }

    vessel_delays_causes = {
        "nr_operations":cycle_nr,
        "nr_vessels": len(vessel_delays),
        "min_delay": pd.Timedelta(seconds=round(vessel_delays.total_delay.min().total_seconds())),
        "average_delay": pd.Timedelta(seconds=round(vessel_delays.total_delay.mean().total_seconds())),
        "max_delay": pd.Timedelta(seconds=round(vessel_delays.total_delay.max().total_seconds())),
        "total_delay": total_delay,
        "congestion (%)": np.round(
            (vessel_delays["waiting time in waiting_area for available operation (%)"]*
             vessel_delays["total_delay"]).sum() / total_delay, 2),
        "obstruction (%)": np.round(
            ((vessel_delays["delay due to sailing to lock gate (%)"] +
              vessel_delays["delay due to sailing to position in lock (%)"] +
              vessel_delays["delay due to sailing out of lock (%)"] +
              vessel_delays["delay due to sailing away from lock (%)"])
             *vessel_delays["total_delay"]).sum() / total_delay, 2),
        "traffic (%)": np.round(
            ((vessel_delays["waiting time in waiting_area for other vessels (%)"] +
              vessel_delays["waiting time for other vessels to sail into lock (%)"] +
              vessel_delays["waiting time for other vessels to sail out of lock (%)"])*
             vessel_delays["total_delay"]).sum() / total_delay, 2),
        "operation of lock (%)": np.round(
            ((vessel_delays["waiting time for closing doors (%)"] +
              vessel_delays["waiting time for levelling (%)"] +
              vessel_delays["waiting time for opening doors (%)"])
             *vessel_delays["total_delay"]).sum() / total_delay, 2),
    }

    return vessel_delays, pd.Series(vessel_delay_locations), pd.Series(vessel_delays_causes)


def calculate_cycle_looptimes(lock_chamber):
    vessels = _get_vessels_that_passed_the_lock_chamber(lock_chamber)
    levelling_cycles = get_levelling_cycles(lock_chamber)

    # Create a lookup for vessel logbooks
    vessel_logs = {}
    for vessel in vessels:
        id = getattr(vessel, "id")
        vessel_logs[id] = vessel.logbook

    results = []
    for i, cycle in enumerate(levelling_cycles):
        if i == 0:
            results.append({"cycle": i + 1, "looptime_seconds": 0})
            continue

        prev_vessels = levelling_cycles[i - 1]["vessels_present"]
        curr_vessels = cycle["vessels_present"]

        # Get latest exit time from previous cycle
        if len(prev_vessels):
            prev_exit_times = [
                event["Timestamp"]
                for v in prev_vessels
                for event in vessel_logs.get(v, [])
                if event["Message"] == "Sailing to second lock gate stop"
            ]
            last_exit = max(prev_exit_times) if prev_exit_times else None
        else:
            last_exit = levelling_cycles[i-1]["gate_opened"]

        # Get earliest entry time from current cycle
        if len(curr_vessels):
            curr_entry_times = [
                event["Timestamp"]
                for v in curr_vessels
                for event in vessel_logs.get(v, [])
                if event["Message"] == "Sailing to first lock gate stop"
            ]
            first_entry = min(curr_entry_times) if curr_entry_times else None
        else:
            first_entry = levelling_cycles[i]["gate_closed"]

        # Calculate looptime
        looptime = (first_entry - last_exit).total_seconds() if last_exit and first_entry else None

        results.append({
            "cycle": i + 1,
            "looptime_seconds": looptime
        })

    return pd.DataFrame(results)


def calculate_cycle_information(lock_chamber):
    vessels = _get_vessels_that_passed_the_lock_chamber(lock_chamber)
    levelling_cycles = get_levelling_cycles(lock_chamber)
    levelling_cycles['vessels_present'] = [[] for _ in range(len(levelling_cycles))]
    vessels_per_cycle = get_vessels_per_cycle(lock_chamber)
    if vessels_per_cycle.empty:
        return pd.DataFrame()
    for cycle_nr, _ in levelling_cycles.iterrows():
        vessels_in_cycle = vessels_per_cycle[vessels_per_cycle.cycle_nr == cycle_nr]
        if not vessels_in_cycle.empty:
            vessels_present = list(vessels_in_cycle.vessel_id.values)
            levelling_cycles.at[cycle_nr, 'vessels_present'] = vessels_present

    def get_duration(df, start_msg, stop_msg):
        starts = df[df["Message"] == start_msg]["Timestamp"].reset_index(drop=True)
        stops = df[df["Message"] == stop_msg]["Timestamp"].reset_index(drop=True)
        return [(stop - start).total_seconds() for start, stop in zip(starts, stops)]

    def get_time_range(log, start_msg, stop_msg):
        df = pd.DataFrame(log)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        starts = df[df["Message"] == start_msg]["Timestamp"]
        stops = df[df["Message"] == stop_msg]["Timestamp"]
        if not starts.empty and not stops.empty:
            return starts.iloc[0], stops.iloc[-1]
        return None, None

    lock_df = pd.DataFrame(lock_chamber.logbook)
    lock_df["Timestamp"] = pd.to_datetime(lock_df["Timestamp"])

    # Extract per-cycle lock durations
    T_close_list = get_duration(lock_df, "Lock gate closing start", "Lock gate closing stop")
    T_waterlevel_list = get_duration(lock_df, "Lock levelling start", "Lock levelling stop")
    T_open_list = get_duration(lock_df, "Lock gate opening start", "Lock gate opening stop")

    vessel_logs = {getattr(v, "id"): v.logbook for i, v in enumerate(vessels)}

    results = []
    for index, levelling_cycle in levelling_cycles.iterrows():
        direction = levelling_cycle.direction
        up_cycle = levelling_cycles.loc[index]
        loc = levelling_cycles.index.get_loc(index)
        last_index = levelling_cycles.index[loc - 1]
        try:
            next_index = levelling_cycles.index[loc + 1]
        except:
            break
        down_cycle = levelling_cycles.loc[next_index]

        up_vessels = up_cycle["vessels_present"]
        down_vessels = down_cycle["vessels_present"]

        # t_l_up
        if index == 0:
            t_l_up = 0
        else:
            prev_down_vessels = levelling_cycles.loc[last_index]["vessels_present"]
            last_exit_prev = max([
                get_time_range(vessel_logs[v], "Sailing to second lock gate start",
                               "Sailing to second lock gate stop")[1]
                for v in prev_down_vessels if v in vessel_logs
            ], default=None)
            first_entry_up = min([
                get_time_range(vessel_logs[v], "Sailing to first lock gate stop", "Sailing to first lock gate stop")[
                    0]
                for v in up_vessels if v in vessel_logs
            ], default=None)
            t_l_up = (first_entry_up - last_exit_prev).total_seconds() if first_entry_up and last_exit_prev else 0

        # t_l_down
        last_exit_up = max([
            get_time_range(vessel_logs[v], "Sailing to second lock gate start", "Sailing to second lock gate stop")[1]
            for v in up_vessels if v in vessel_logs
        ], default=None)
        first_entry_down = min([
            get_time_range(vessel_logs[v], "Sailing to first lock gate stop", "Sailing to first lock gate stop")[0]
            for v in down_vessels if v in vessel_logs
        ], default=None)
        t_l_down = (first_entry_down - last_exit_up).total_seconds() if first_entry_down and last_exit_up else 0

        # Entry and exit durations using time range
        entry_times_up = [
            get_time_range(vessel_logs[v], "Sailing to position in lock start", "Sailing to position in lock stop") for
            v in up_vessels if v in vessel_logs]
        exit_times_up = [
            get_time_range(vessel_logs[v], "Sailing to second lock gate start", "Sailing to second lock gate stop")
            for v in up_vessels if v in vessel_logs]
        entry_times_down = [
            get_time_range(vessel_logs[v], "Sailing to position in lock start", "Sailing to position in lock stop") for
            v in down_vessels if v in vessel_logs]
        exit_times_down = [
            get_time_range(vessel_logs[v], "Sailing to second lock gate start", "Sailing to second lock gate stop")
            for v in down_vessels if v in vessel_logs]

        # Sum of entering times (up), - Part III, Ch 3, Eq. 3.2
        entry_start_up = min([t[0] for t in entry_times_up if t[0] is not None], default=None)
        entry_stop_up = max([t[1] for t in entry_times_up if t[1] is not None], default=None)
        sum_t_i_up = (entry_stop_up - entry_start_up).total_seconds() if entry_start_up and entry_stop_up else 0

        # Sum of exiting times (up), - Part III, Ch 3, Eq. 3.4
        exit_start_up = min([t[0] for t in exit_times_up if t[0] is not None], default=None)
        exit_stop_up = max([t[1] for t in exit_times_up if t[1] is not None], default=None)
        sum_t_u_up = (exit_stop_up - exit_start_up).total_seconds() if exit_start_up and exit_stop_up else 0

        # Sum of entering times (down), - Part III, Ch 3, Eq. 3.2
        entry_start_down = min([t[0] for t in entry_times_down if t[0] is not None], default=None)
        entry_stop_down = max([t[1] for t in entry_times_down if t[1] is not None], default=None)
        sum_t_i_down = (
                    entry_stop_down - entry_start_down).total_seconds() if entry_start_down and entry_stop_down else 0

        # Sum of exiting times (down), - Part III, Ch 3, Eq. 3.4
        exit_start_down = min([t[0] for t in exit_times_down if t[0] is not None], default=None)
        exit_stop_down = max([t[1] for t in exit_times_down if t[1] is not None], default=None)
        sum_t_u_down = (exit_stop_down - exit_start_down).total_seconds() if exit_start_down and exit_stop_down else 0

        # Identify the index of the op and down cycles
        cycle_index_up = index
        cycle_index_down = next_index

        # Operation components (up) - Part III, Ch 3, Eq. 3.3
        T_close_up = T_close_list[cycle_index_up] if cycle_index_up < len(T_close_list) else 0
        T_waterlevel_up = T_waterlevel_list[cycle_index_up] if cycle_index_up < len(T_waterlevel_list) else 0
        T_open_up = T_open_list[cycle_index_up] if cycle_index_up < len(T_open_list) else 0

        # Operation components (down) - Part III, Ch 3, Eq. 3.3
        T_close_down = T_close_list[cycle_index_down] if cycle_index_down < len(T_close_list) else 0
        T_waterlevel_down = T_waterlevel_list[cycle_index_down] if cycle_index_down < len(T_waterlevel_list) else 0
        T_open_down = T_open_list[cycle_index_down] if cycle_index_down < len(T_open_list) else 0

        # Part III, Ch 3, Eq. 3.1
        Tc_seconds = (
                t_l_up + sum_t_i_up + T_close_up + T_waterlevel_up + T_open_up + sum_t_u_up +
                t_l_down + sum_t_i_down + T_close_down + T_waterlevel_down + T_open_down + sum_t_u_down
        )

        # Part III, Ch 3, Eq. 3.6
        # NB: n_max is here equivalent to 2 * n_max, since it counts both the up and down vessels
        n_max = len(up_vessels) + len(down_vessels)
        I_s = (n_max / (Tc_seconds / 3600)) if Tc_seconds else None


        number_of_upstream_vessels = len(up_vessels)
        number_of_downstream_vessels = len(down_vessels)
        if direction:
            number_of_upstream_vessels = len(down_vessels)
            number_of_downstream_vessels = len(up_vessels)

        t_stop = exit_stop_down
        t_start = exit_stop_down - pd.Timedelta(seconds=Tc_seconds)
        if pd.isna(t_stop):
            t_start = entry_start_up
            t_stop = entry_start_up + pd.Timedelta(seconds=Tc_seconds)

        results.append({
            "Start time of cycle":t_start,
            "Stop time of cycle":t_stop,
            "Direction": direction,
            "Loop time start side": pd.Timedelta(seconds=round(t_l_up)),
            "Sailing-in time start side": pd.Timedelta(seconds=round(sum_t_i_up)),
            "Closing gate time start side": pd.Timedelta(seconds=round(T_close_up)),
            "Levelling time to opposing side": pd.Timedelta(seconds=round(T_waterlevel_up)),
            "Opening gate time opposing side": pd.Timedelta(seconds=round(T_open_up)),
            "Sailing-out time opposing side": pd.Timedelta(seconds=round(sum_t_u_up)),
            "Loop time opposing side": pd.Timedelta(seconds=round(t_l_down)),
            "Sailing-in time opposing side": pd.Timedelta(seconds=round(sum_t_i_down)),
            "Closing gate time opposing side": pd.Timedelta(seconds=round(T_close_down)),
            "Levelling time to start side": pd.Timedelta(seconds=round(T_waterlevel_down)),
            "Opening gate time start side": pd.Timedelta(seconds=round(T_open_down)),
            "Sailing-out time start side": pd.Timedelta(seconds=round(sum_t_u_down)),
            "Cycle duration": pd.Timedelta(seconds=round(Tc_seconds)),
            "Number of upstream vessels": number_of_upstream_vessels,
            "Number of downstream vessels": number_of_downstream_vessels,
            "Upstream vessel_ids": up_vessels,
            "Downstream vessel_ids": down_vessels,
            "Intensity (I_s)": I_s
        })

    return pd.DataFrame(results)

