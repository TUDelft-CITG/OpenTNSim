
import pandas as pd

def isolate_events(logbook):
    df = logbook.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)

    def classify(message):
        message = message.lower()

        if "waiting" in message:
            return "Waiting"
        elif "sailing" in message:
            return "Sailing"
        elif "deberthing" in message or "de berthing" in message:
            return "Deberthing"
        elif "berthing" in message:
            return "Berthing"
        elif "loading" in message:
            return "Loading"
        return None

    df["Event"] = df["Message"].map(classify)
    df["Action"] = df["Message"].str.extract(
        r"\b(start|stop)\s*$", expand=False
    )
    df["Base"] = df["Message"].str.replace(
        r"\s+(start|stop)\s*$", "", regex=True
    )

    # Pair repeated event descriptions by occurrence order
    starts = df[df["Action"] == "start"].copy()
    stops = df[df["Action"] == "stop"].copy()

    starts["Occurrence"] = starts.groupby("Base").cumcount()
    stops["Occurrence"] = stops.groupby("Base").cumcount()

    intervals = starts[
        ["Base", "Timestamp", "Event", "Occurrence"]
    ].merge(
        stops[["Base", "Timestamp", "Occurrence"]],
        on=["Base", "Occurrence"],
        suffixes=("_start", "_stop"),
        how="inner",
        validate="one_to_one"
    )

    intervals = intervals.rename(columns={
        "Timestamp_start": "Start",
        "Timestamp_stop": "Stop"
    })

    intervals = intervals.dropna(subset=["Event"])
    intervals = intervals[intervals["Stop"] >= intervals["Start"]]
    intervals = intervals.sort_values(
        ["Start", "Stop"], kind="stable"
    ).reset_index(drop=True)

    if intervals.empty:
        return pd.DataFrame(
            columns=["Event", "Start", "Stop", "Duration"]
        )

    # Resolve overlapping intervals into a chronological timeline
    boundaries = sorted(
        set(intervals["Start"]).union(set(intervals["Stop"]))
    )

    segments = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        stop = boundaries[i + 1]

        active = intervals[
            (intervals["Start"] <= start) &
            (intervals["Stop"] >= stop)
        ]

        if active.empty:
            continue

        active = active.copy()
        active["IsWaiting"] = active["Event"].eq("Waiting")

        chosen = active.sort_values(
            ["Start", "IsWaiting"],
            ascending=[False, True],
            kind="stable"
        ).iloc[0]

        event = chosen["Event"]

        if (
            segments
            and segments[-1]["Event"] == event
            and segments[-1]["Stop"] == start
        ):
            segments[-1]["Stop"] = stop
        else:
            segments.append({
                "Event": event,
                "Start": start,
                "Stop": stop
            })

    result = pd.DataFrame(segments)

    if result.empty:
        return pd.DataFrame(
            columns=["Event", "Start", "Stop", "Duration"]
        )

    # Classify waiting locations using the surrounding activities
    result["Previous_Event"] = result["Event"].shift(1)
    result["Next_Event"] = result["Event"].shift(-1)

    waiting_at_berth = (
        result["Event"].eq("Waiting")
        & result["Previous_Event"].eq("Loading")
        & result["Next_Event"].eq("Deberthing")
    )

    result.loc[waiting_at_berth, "Event"] = "Waiting at berth"

    result.loc[
        result["Event"].eq("Waiting"),
        "Event"
    ] = "Waiting at anchorage"

    result = result.drop(
        columns=["Previous_Event", "Next_Event"]
    )

    result["Duration"] = result["Stop"] - result["Start"]

    return result.reset_index(drop=True)
