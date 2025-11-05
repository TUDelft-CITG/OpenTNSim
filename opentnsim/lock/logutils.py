import pandas as pd


def get_vessels_during_leveling(lock, vessels: list) -> list:
    """
    Identifies which vessels were present during each lock leveling event.

    Parameters:
    - lock: An object with a `.logbook` attribute (list of dicts with 'Message' and 'Timestamp').
    - vessels: List of vessel objects, each with a `.logbook` attribute and optional `.name`.

    Returns:
    - List of dicts with keys: 'leveling_start', 'leveling_stop', 'vessels_present'
    """
    # Convert lock logbook to DataFrame
    lock_df = pd.DataFrame(lock.logbook)

    # Extract leveling start/stop events
    leveling_starts = lock_df[lock_df["Message"] == "Lock chamber converting start"]
    leveling_stops = lock_df[lock_df["Message"] == "Lock chamber converting stop"]

    # Ensure matching pairs
    leveling_events = pd.DataFrame({
        "leveling_start": leveling_starts["Timestamp"].values,
        "leveling_stop": leveling_stops["Timestamp"].values
    })

    leveling_cycles = []
    for _, level_event in leveling_events.iterrows():
        vessels_present = []
        for vessel in vessels:
            vessel_df = pd.DataFrame(vessel.logbook)
            vessel_df["Timestamp"] = pd.to_datetime(vessel_df["Timestamp"])
            name = getattr(vessel, "name", f"Vessel_{vessels.index(vessel)+1}")

            # Find levelling start/stop pairs
            levelling_starts = vessel_df[vessel_df["Message"] == "Levelling start"]
            levelling_stops = vessel_df[vessel_df["Message"] == "Levelling stop"]

            for i in range(min(len(levelling_starts), len(levelling_stops))):
                start = levelling_starts.iloc[i]["Timestamp"]
                stop = levelling_stops.iloc[i]["Timestamp"]
                if start <= level_event["leveling_stop"] and stop >= level_event["leveling_start"]:
                    vessels_present.append(name)
                    break

        leveling_cycles.append({
            "leveling_start": level_event["leveling_start"],
            "leveling_stop": level_event["leveling_stop"],
            "vessels_present": vessels_present
        })

    return leveling_cycles

def calculate_cycle_looptimes(leveling_cycles: list, vessels: list) -> pd.DataFrame:
    """
    Calculates the looptime for each locking cycle.

    Looptime is defined as the time between:
    - the last vessel exiting the lock in the previous cycle ('Sailing to lock complex exit start')
    - the first vessel entering the lock in the current cycle ('Sailing to first lock doors stop')

    Parameters:
    - leveling_cycles: List of dicts from get_vessels_during_leveling, each with 'vessels_present'.
    - vessels: List of vessel objects, each with a `.logbook` attribute and optional `.name`.

    Returns:
    - DataFrame with columns: 'cycle', 'looptime_seconds'
    """
    # Create a lookup for vessel logbooks
    vessel_logs = {}
    for vessel in vessels:
        name = getattr(vessel, "name", f"Vessel_{vessels.index(vessel)+1}")
        vessel_logs[name] = vessel.logbook

    results = []
    for i, cycle in enumerate(leveling_cycles):
        if i == 0:
            results.append({
                "cycle": i + 1,
                "looptime_seconds": 0
            })
            continue

        prev_vessels = leveling_cycles[i - 1]["vessels_present"]
        curr_vessels = cycle["vessels_present"]

        # Get latest exit time from previous cycle
        prev_exit_times = [
            event["Timestamp"]
            for v in prev_vessels
            for event in vessel_logs.get(v, [])
            if event["Message"] == "Sailing to lock complex exit start"
        ]
        last_exit = max(prev_exit_times) if prev_exit_times else None

        # Get earliest entry time from current cycle
        curr_entry_times = [
            event["Timestamp"]
            for v in curr_vessels
            for event in vessel_logs.get(v, [])
            if event["Message"] == "Sailing to first lock doors stop"
        ]
        first_entry = min(curr_entry_times) if curr_entry_times else None

        # Calculate looptime
        looptime = (first_entry - last_exit).total_seconds() if last_exit and first_entry else None

        results.append({
            "cycle": i + 1,
            "looptime_seconds": looptime
        })

    return pd.DataFrame(results)


def calculate_detailed_cycle_time(lock, vessels, leveling_cycles):
    """
    Calculates detailed timing metrics for full lock cycles (up + down) using logbook data.

    Each full cycle consists of two consecutive leveling events: one upward and one downward.
    The function computes:
    - Looptimes before each phase (t_l_up, t_l_down)
    - Entry and exit durations for vessels based on first and last movement timestamps
    - Lock operation durations (door opening/closing, water level adjustment) per cycle
    - Total cycle time (Tc)
    - Locking system intensity (I_s = 2 * n_max / (Tc / 3600))

    Parameters
    ----------
    lock : object
        Lock object with a `.logbook` attribute containing a list of dicts with 'Message' and 'Timestamp'.
    vessels : list
        List of vessel objects, each with a `.logbook` attribute and optional `.name`.
    leveling_cycles : list
        Output from `get_vessels_during_leveling`, containing dicts with 'leveling_start', 'leveling_stop', and 'vessels_present'.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per full lock cycle and the following columns:
        - t_l_up: Looptime before upcycle
        - sum_t_i_up: Time between first vessel's entry start and last vessel's entry stop (upcycle)
        - T_close_up, T_waterlevel_up, T_open_up: Lock operation durations (upcycle)
        - sum_t_u_up: Time between first vessel's exit start and last vessel's exit stop (upcycle)

        - t_l_down: Looptime before downcycle
        - sum_t_i_down: Time between first vessel's entry start and last vessel's entry stop (downcycle)
        - T_close_down, T_waterlevel_down, T_open_down: Lock operation durations (downcycle)
        - sum_t_u_down: Time between first vessel's exit start and last vessel's exit stop (downcycle)

        - Tc_seconds: Total time for the full cycle (up + down)
        - up_vessels: List of vessels in upcycle
        - down_vessels: List of vessels in downcycle
        - I_s: Locking system intensity (vessels per hour)

    Notes
    -----
    - Lock operation durations are extracted per cycle from the lock's logbook using start/stop message pairs.
    - Vessel entry and exit durations are calculated using the earliest and latest timestamps of relevant movement messages.
    - The function assumes `leveling_cycles` are chronologically ordered and alternate between up and down phases.
    """

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

    lock_df = pd.DataFrame(lock.logbook)
    lock_df["Timestamp"] = pd.to_datetime(lock_df["Timestamp"])

    # Extract per-cycle lock durations
    T_close_list = get_duration(lock_df, "Lock doors closing start", "Lock doors closing stop")
    T_waterlevel_list = get_duration(lock_df, "Lock chamber converting start", "Lock chamber converting stop")
    T_open_list = get_duration(lock_df, "Lock doors opening start", "Lock doors opening stop")

    vessel_logs = {
        getattr(v, "name", f"Vessel_{i + 1}"): v.logbook
        for i, v in enumerate(vessels)
    }

    results = []
    for i in range(0, len(leveling_cycles) - 1, 2):
        up_cycle = leveling_cycles[i]
        down_cycle = leveling_cycles[i + 1]

        up_vessels = up_cycle["vessels_present"]
        down_vessels = down_cycle["vessels_present"]

        # t_l_up
        if i == 0:
            t_l_up = 0
        else:
            prev_down_vessels = leveling_cycles[i - 1]["vessels_present"]
            last_exit_prev = max([
                get_time_range(vessel_logs[v], "Sailing to second lock doors start",
                               "Sailing to second lock doors stop")[1]
                for v in prev_down_vessels if v in vessel_logs
            ], default=None)
            first_entry_up = min([
                get_time_range(vessel_logs[v], "Sailing to first lock doors stop", "Sailing to first lock doors stop")[
                    0]
                for v in up_vessels if v in vessel_logs
            ], default=None)
            t_l_up = (first_entry_up - last_exit_prev).total_seconds() if first_entry_up and last_exit_prev else 0

        # t_l_down
        last_exit_up = max([
            get_time_range(vessel_logs[v], "Sailing to second lock doors start", "Sailing to second lock doors stop")[1]
            for v in up_vessels if v in vessel_logs
        ], default=None)
        first_entry_down = min([
            get_time_range(vessel_logs[v], "Sailing to first lock doors stop", "Sailing to first lock doors stop")[0]
            for v in down_vessels if v in vessel_logs
        ], default=None)
        t_l_down = (first_entry_down - last_exit_up).total_seconds() if first_entry_down and last_exit_up else 0

        # Entry and exit durations using time range
        entry_start_up, entry_stop_up = None, None
        exit_start_up, exit_stop_up = None, None
        entry_start_down, entry_stop_down = None, None
        exit_start_down, exit_stop_down = None, None

        entry_times_up = [
            get_time_range(vessel_logs[v], "Sailing to position in lock start", "Sailing to position in lock stop") for
            v in up_vessels if v in vessel_logs]
        exit_times_up = [
            get_time_range(vessel_logs[v], "Sailing to second lock doors start", "Sailing to second lock doors stop")
            for v in up_vessels if v in vessel_logs]
        entry_times_down = [
            get_time_range(vessel_logs[v], "Sailing to position in lock start", "Sailing to position in lock stop") for
            v in down_vessels if v in vessel_logs]
        exit_times_down = [
            get_time_range(vessel_logs[v], "Sailing to second lock doors start", "Sailing to second lock doors stop")
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
        cycle_index_up = i
        cycle_index_down = i + 1

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

        results.append({
            "t_l_up": t_l_up,
            "sum_t_i_up": sum_t_i_up,
            "T_close_up": T_close_up,
            "T_waterlevel_up": T_waterlevel_up,
            "T_open_up": T_open_up,
            "sum_t_u_up": sum_t_u_up,
            "t_l_down": t_l_down,
            "sum_t_i_down": sum_t_i_down,
            "T_close_down": T_close_down,
            "T_waterlevel_down": T_waterlevel_down,
            "T_open_down": T_open_down,
            "sum_t_u_down": sum_t_u_down,
            "Tc_seconds": Tc_seconds,
            "up_vessels": up_vessels,
            "down_vessels": down_vessels,
            "I_s": I_s
        })

    return pd.DataFrame(results)