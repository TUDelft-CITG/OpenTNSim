"""Test of the durations of the lock cycle events with all time resolutions of pandas timedeltas (pandas 3 changed the default)."""

import datetime
import warnings

import numpy as np
import pandas as pd
import pytest

from opentnsim.lock.calculations import calculate_cycle_event_durations

# the columns of the cycle information (calculate_cycle_information) with a duration
EVENT_COLUMNS = [
    "Loop time start side",
    "Loop time opposing side",
    "Sailing-in time start side",
    "Sailing-in time opposing side",
    "Closing gate time start side",
    "Closing gate time opposing side",
    "Levelling time to opposing side",
    "Levelling time to start side",
    "Opening gate time start side",
    "Opening gate time opposing side",
    "Sailing-out time start side",
    "Sailing-out time opposing side",
]


def cycle_information(unit):
    """Two lock cycles, one that starts downstream and one that starts upstream, with durations as timedelta64[unit].

    The duration of each event is 10 s (cycle 0, direction 1) or 30 s (cycle 1, direction 0).
    """
    seconds = {0: 10, 1: 30}
    data = {
        "Direction first operation": [1, 0],
        "Number of downstream vessels": [2, 1],
        "Intensity (I_s)": [0.5, 0.7],
        "Cycle duration": [pd.Timedelta(seconds=400), pd.Timedelta(seconds=600)],
    }
    for column in EVENT_COLUMNS:
        data[column] = [pd.Timedelta(seconds=seconds[0]), pd.Timedelta(seconds=seconds[1])]
    cycles = pd.DataFrame(data)
    timedelta_columns = EVENT_COLUMNS + ["Cycle duration"]
    cycles[timedelta_columns] = cycles[timedelta_columns].astype(f"timedelta64[{unit}]")
    return cycles


def durations(unit):
    return calculate_cycle_event_durations(cycle_information(unit))[0]


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_durations_do_not_depend_on_the_resolution_of_the_timedeltas(unit):
    expected = durations("ns")
    pd.testing.assert_series_equal(durations(unit), expected)


def test_average_durations_are_timedeltas():
    results = durations("us")
    # cycle 0 starts downstream (10 s), cycle 1 starts upstream (30 s)
    assert results["Average duration loop time downstream"] == pd.Timedelta(seconds=10)
    assert results["Average duration loop time upstream"] == pd.Timedelta(seconds=30)
    assert results["Average cycle time"] == pd.Timedelta(seconds=500)


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_no_timedelta_is_passed_to_to_timedelta_with_a_unit(monkeypatch, unit):
    """pandas 3 warns: "The 'unit' keyword is only used when the Timedelta input is an integer or float, not Timedelta"."""
    to_timedelta = pd.to_timedelta
    timedelta_types = (pd.Timedelta, datetime.timedelta, np.timedelta64)
    offending = []

    def checked_to_timedelta(arg, *args, **kwargs):
        if "unit" in kwargs and isinstance(arg, timedelta_types):
            offending.append(arg)
        return to_timedelta(arg, *args, **kwargs)

    monkeypatch.setattr(pd, "to_timedelta", checked_to_timedelta)
    with warnings.catch_warnings():
        # the warning of pandas 3 itself, as an error
        warnings.filterwarnings("error", message=".*'unit' keyword.*")
        durations(unit)
    assert offending == []
