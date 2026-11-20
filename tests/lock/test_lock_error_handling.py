"""Tests of the situations that were handled by a bare `except:` in the lock module (issue #153).

The tests check that the intended fallbacks still work, now with explicit checks or specific exceptions.
"""

import math
import types

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from opentnsim.environment.mixins.hydrodynamics import HydrodynamicDataManager
from opentnsim.lock import calculations
from opentnsim.lock.calculations import (
    _get_ship_volume_in_levelling,
    calculate_lock_operation_times,
    calculate_lock_salinity_and_saltmass,
    calculate_sailing_time_to_waiting_area,
)
from opentnsim.lock.mixins.chamber import _has_hydrodynamic_data
from opentnsim.lock.visualizations import _as_axes_list, _first_axes


# %% hydrodynamic data of a lock chamber (lock/mixins/chamber.py)
@pytest.fixture
def hydromanager():
    """The (singleton) hydrodynamic data manager with data for two stations, restored after the test."""
    manager = HydrodynamicDataManager()
    original = manager.hydrodynamic_data
    time = pd.date_range("2025-01-01", periods=5, freq="h")
    manager.hydrodynamic_data = xr.Dataset(
        {"Water level": (("STATION", "TIME"), np.array([[1.0] * 5, [np.nan] * 5]))},
        coords={"STATION": ["lock_node", "node_without_values"], "TIME": time},
    )
    yield manager
    manager.hydrodynamic_data = original


TIME = pd.Timestamp("2025-01-01 02:00")


def test_has_hydrodynamic_data(hydromanager):
    assert _has_hydrodynamic_data(hydromanager, TIME, "lock_node", "Water level")


def test_has_no_hydrodynamic_data_without_data(hydromanager):
    hydromanager.hydrodynamic_data = None
    assert not _has_hydrodynamic_data(hydromanager, TIME, "lock_node", "Water level")


def test_has_no_hydrodynamic_data_for_unknown_node(hydromanager):
    assert not _has_hydrodynamic_data(hydromanager, TIME, "unknown_node", "Water level")


def test_has_no_hydrodynamic_data_for_missing_property(hydromanager):
    # the data has no salinity
    assert not _has_hydrodynamic_data(hydromanager, TIME, "lock_node", "Salinity")


def test_has_no_hydrodynamic_data_for_nan_values(hydromanager):
    assert not _has_hydrodynamic_data(hydromanager, TIME, "node_without_values", "Water level")


def test_unexpected_errors_in_hydrodynamic_data_are_not_hidden(hydromanager, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("bug in the interpolation")

    monkeypatch.setattr(hydromanager, "_get_hydrodynamic_data_value", broken)
    with pytest.raises(RuntimeError):
        _has_hydrodynamic_data(hydromanager, TIME, "lock_node", "Water level")


# %% volume of the vessels during a levelling (calculate_ZSF_eventttable)
PHASE = pd.Series({"time_start": pd.Timestamp("2025-01-01 10:00"), "time_stop": pd.Timestamp("2025-01-01 11:00")})


def test_ship_volume_without_levelling_cycles():
    # get_levelling_cycles returns a dataframe without levelling columns if no operation has been logged
    df_operations = pd.DataFrame({"cycle": []})
    assert _get_ship_volume_in_levelling(df_operations, PHASE) == 0.0


def test_ship_volume_of_levelling_in_phase():
    df_operations = pd.DataFrame(
        {
            "leveling_start": [pd.Timestamp("2025-01-01 08:00"), pd.Timestamp("2025-01-01 10:10")],
            "leveling_stop": [pd.Timestamp("2025-01-01 08:20"), pd.Timestamp("2025-01-01 10:30")],
            "volume_of_vessels_in_lock": [111.0, 222.0],
        }
    )
    assert _get_ship_volume_in_levelling(df_operations, PHASE) == 222.0


def test_ship_volume_without_levelling_in_phase():
    df_operations = pd.DataFrame(
        {
            "leveling_start": [pd.Timestamp("2025-01-01 08:00")],
            "leveling_stop": [pd.Timestamp("2025-01-01 08:20")],
            "volume_of_vessels_in_lock": [111.0],
        }
    )
    assert _get_ship_volume_in_levelling(df_operations, PHASE) == 0.0


# %% average sailing speed to the waiting area (calculate_sailing_time_to_waiting_area)
@pytest.mark.parametrize(
    "sailing_time, distance, expected_speed",
    [
        (pd.Timedelta(seconds=50), np.float64(100.0), 2.0),
        # no sailing time: speed 0 (also for numpy floats, which do not raise ZeroDivisionError)
        (pd.Timedelta(seconds=0), np.float64(100.0), 0.0),
        (pd.Timedelta(seconds=0), 100.0, 0.0),
    ],
)
def test_average_sailing_speed_to_waiting_area(monkeypatch, sailing_time, distance, expected_speed):
    monkeypatch.setattr(calculations, "_get_edge_route_to_waiting_area", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        calculations, "get_sailing_information_on_edge_to_distance_on_another_edge", lambda *args, **kwargs: (sailing_time, distance)
    )
    waiting_area = types.SimpleNamespace(distance_from_edge_start=0.0)
    time, sailing_distance, speed = calculate_sailing_time_to_waiting_area(waiting_area, vessel=types.SimpleNamespace())
    assert time == sailing_time
    assert sailing_distance == distance
    assert speed == expected_speed


# %% operation times of a lock (calculate_lock_operation_times)
START = pd.Timestamp("2025-01-01 10:00")
PLANNED = {
    "time_gate_closing_stop": pd.Timestamp("2025-01-01 10:35"),
    "time_levelling_start": pd.Timestamp("2025-01-01 10:35"),
    "time_levelling_stop": pd.Timestamp("2025-01-01 10:55"),
    "time_gate_opening_start": pd.Timestamp("2025-01-01 10:55"),
    "time_gate_opening_stop": pd.Timestamp("2025-01-01 11:00"),
}


def _lock_chamber_with_operation(time_gate_closing_start):
    operation_planning = pd.DataFrame(
        [{"lock_chamber": "chamber", "operation_index": 0, "time_gate_closing_start": time_gate_closing_start, **PLANNED}]
    )
    return types.SimpleNamespace(
        name="chamber",
        lock_complex=types.SimpleNamespace(vessel_planning=pd.DataFrame(), operation_planning=operation_planning),
        gate_closing_time=60.0,
        gate_opening_time=120.0,
        close_gate_before_vessel_is_laying_still=False,
    )


@pytest.fixture
def levelling(monkeypatch):
    monkeypatch.setattr(calculations, "calculate_levelling_time", lambda *args, **kwargs: (600.0, None, None))
    monkeypatch.setattr(calculations, "_get_water_levels_before_and_after_levelling", lambda *args, **kwargs: (0.0, 0.0))


@pytest.mark.parametrize("planned_closing_start", [pd.NaT, np.nan, None])
def test_operation_times_without_planned_gate_closing(levelling, planned_closing_start):
    lock_chamber = _lock_chamber_with_operation(planned_closing_start)
    times = calculate_lock_operation_times(lock_chamber, 0, START)
    # a new gate closing is planned from the start time
    assert times["time_gate_closing_start"] == START
    assert times["time_gate_closing_stop"] == START + pd.Timedelta(seconds=60)
    assert times["time_levelling_stop"] == START + pd.Timedelta(seconds=60 + 600)


def test_operation_times_joins_a_planned_operation(levelling):
    lock_chamber = _lock_chamber_with_operation(pd.Timestamp("2025-01-01 10:30"))
    times = calculate_lock_operation_times(lock_chamber, 0, START)
    assert times["time_gate_closing_start"] == pd.Timestamp("2025-01-01 10:30")
    assert times["time_gate_opening_stop"] == PLANNED["time_gate_opening_stop"]


def test_operation_times_after_the_planned_gate_closing(levelling):
    lock_chamber = _lock_chamber_with_operation(pd.Timestamp("2025-01-01 09:30"))
    times = calculate_lock_operation_times(lock_chamber, 0, START)
    assert times["time_gate_closing_start"] == START


# %% salinity in the lock with the gate open (calculate_lock_salinity_and_saltmass)
N_TIMES = 100
OPEN_START = 10
OPEN_STOP = 40
T_OPEN = (OPEN_STOP - OPEN_START) * 60.0
LOCK_VOLUME = (2.0 + 10.0) * 100.0 * 20.0
SALINITY_SEA = 30.0


def _salinity_with_gate_open_at_sea(saltmass_start, saltmass_stop):
    time = pd.date_range("2025-01-01", periods=N_TIMES, freq="60s")
    lock_chamber = types.SimpleNamespace(
        lock_depth=10.0,
        lock_length=100.0,
        lock_width=20.0,
        time=time,
        time_step=60.0,
        salinity=np.zeros(N_TIMES),
        saltmass=np.zeros(N_TIMES),
    )
    results = pd.DataFrame(
        [
            {
                "routine": "Gate open at sea",
                "time_start": time[OPEN_START],
                "time_stop": time[OPEN_STOP],
                "head_lock_start": 2.0,
                "salinity_sea": SALINITY_SEA,
                "salinity_lake": 0.0,
                "salinity_lock_a": 0.0,
                "salinity_lock_start": 0.0,
                "salinity_lock_c": SALINITY_SEA,
                "saltmass_lock_a": saltmass_start,
                "saltmass_lock_b": saltmass_stop,
                "saltmass_lock_start": saltmass_start,
                "saltmass_lock_c": saltmass_stop,
            }
        ]
    )
    calculate_lock_salinity_and_saltmass(lock_chamber, ZSF_results=results)
    return lock_chamber.salinity[OPEN_START:OPEN_STOP]


def _expected_salinity(t_exchange):
    t = np.arange(0.0, T_OPEN, 60.0)
    return np.tanh(t / t_exchange) * SALINITY_SEA


def test_exchange_time_from_the_exchanged_salt_mass():
    fully_exchanged = LOCK_VOLUME * SALINITY_SEA
    salinity = _salinity_with_gate_open_at_sea(saltmass_start=0.0, saltmass_stop=0.5 * fully_exchanged)
    assert salinity == pytest.approx(_expected_salinity(T_OPEN / math.atanh(0.5)))


@pytest.mark.parametrize(
    "saltmass_start, exchanged_fraction",
    [
        (0.0, 0.0),  # nothing exchanged: atanh(0) = 0 -> ZeroDivisionError
        (0.0, 1.0),  # fully exchanged: atanh(1) is not defined -> ValueError
        (0.0, 1.2),  # more than fully exchanged: atanh is not defined -> ValueError
        (LOCK_VOLUME * SALINITY_SEA, 0.5),  # lock already at the final salt mass: no exchange fraction (NaN)
    ],
)
def test_exchange_time_defaults_to_half_an_hour(saltmass_start, exchanged_fraction):
    fully_exchanged = LOCK_VOLUME * SALINITY_SEA
    salinity = _salinity_with_gate_open_at_sea(saltmass_start, saltmass_stop=exchanged_fraction * fully_exchanged)
    expected = _expected_salinity(1800.0) + saltmass_start / LOCK_VOLUME
    assert salinity == pytest.approx(expected)


# %% axes of the lock plots (lock/visualizations.py)
def test_single_axes():
    fig, ax = plt.subplots()
    assert _first_axes(ax) is ax
    assert _as_axes_list(ax) == [ax]
    plt.close(fig)


def test_multiple_axes():
    fig, axes = plt.subplots(1, 3)
    assert _first_axes(axes) is axes[0]
    assert _as_axes_list(axes) == list(axes)
    assert _first_axes(list(axes)) is axes[0]
    plt.close(fig)
