"""Energy and fuel attributes for OpenTNSim event tables.

The three resistance modes of ``ConsumesEnergy``:

* ``"none"``: squat depth, then Holtrop/Zeng/Karpov, the original model.
* ``"drawdown"`` / ``"full"``: raw depth, the chain plus the channel-width increment. The real
  waterway width is required, without the 150 m fallback.

The event table keeps three distinct energy bases:

* ``energy_required (kWh)``: uncapped brake-energy requirement.
* ``total_energy (kWh)``: brake energy actually supplied, capped at installed
  engine power. This is used with TNO diesel and emission factors.
* ``energy_shaft (kWh)``: delivered propulsion energy plus hotel load, capped
  consistently. This is used with the alternative-carrier SFCs that already
  include conversion-system efficiencies.

"""

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

import opentnsim.mixins as graph_module

logger = logging.getLogger(__name__)


# Attributes copied from the vessel after evaluating the operating point.
_FACTOR_COLUMNS = {
    "total_factor_CO2": "factor_CO2 (g/kWh)",
    "total_factor_PM10": "factor_PM10 (g/kWh)",
    "total_factor_NOX": "factor_NOX (g/kWh)",
    "final_SFC_diesel_C_year_ICE_mass": "SFC_diesel_C_year (g/kWh)",
    "final_SFC_LH2_mass_PEMFC": "SFC_LH2_PEMFC (g/kWh)",
    "final_SFC_LH2_vol_PEMFC": "SFC_LH2_PEMFC (m3/kWh)",
    "final_SFC_Li_NMC_Battery_mass": "SFC_Li_NMC (g/kWh)",
    "final_SFC_Battery2000kWh": "SFC_Battery2000kWh (packs/kWh)",
}


# Columns written by add_energy_attributes_to_eventtable. Initialising them
# prevents stale values and makes stationary/missing-hydraulic rows explicit.
_EVENT_DEFAULTS = {
    # Event state and kinematics
    "stationary": pd.NA,
    "current (m/s)": np.nan,
    "v_g (m/s)": np.nan,
    "v_w (m/s)": np.nan,
    "resistance model": None,
    "confinement mode": None,
    # Waterway and depth treatment
    "waterdepth (m)": np.nan,  # legacy alias: resistance-model depth
    "waterdepth raw (m)": np.nan,
    "waterdepth resistance (m)": np.nan,
    "waterdepth navigation (m)": np.nan,
    "waterway width (m)": np.nan,
    "squat (m)": np.nan,
    # Resistance and hydrodynamics
    "R_total (kN)": np.nan,
    "R_friction (kN)": np.nan,
    "R_friction_form_corrected (kN)": np.nan,
    "R_appendage (kN)": np.nan,
    "R_wave (kN)": np.nan,
    "R_transom (kN)": np.nan,
    "R_correlation (kN)": np.nan,
    "R_bulbous_bow (kN)": np.nan,
    "R_residual (kN)": np.nan,
    # channel-width increment: R_total = R_base + R_confinement
    "R_base (kN)": np.nan,
    "R_confinement (kN)": np.nan,
    "R_confinement_friction (kN)": np.nan,
    "R_confinement_drawdown (kN)": np.nan,
    "Reynolds number": np.nan,
    "friction coefficient": np.nan,
    "shallow-water coefficient": np.nan,
    "Froude depth": np.nan,
    "Froude length": np.nan,
    "Karpov alpha": np.nan,
    "Karpov clamped": pd.NA,
    # channel hydraulics, modes "drawdown" and "full"
    "drawdown Z (m)": np.nan,
    "return velocity (m/s)": np.nan,
    "friction relative velocity (m/s)": np.nan,
    "blockage ratio": np.nan,
    "dynamic UKC (m)": np.nan,
    "economic speed (m/s)": np.nan,
    "V/V_ec": np.nan,
    "critical speed (m/s)": np.nan,
    "V/V_cr": np.nan,
    "channel area (m2)": np.nan,
    "mean depth (m)": np.nan,
    "economic speed exceeded": pd.NA,
    # Power and energy
    "engine age (year)": np.nan,
    "P_effective (kW)": np.nan,
    "P_delivered (kW)": np.nan,
    "P_brake_propulsion (kW)": np.nan,
    "P_hotel (kW)": np.nan,
    "P_tot (kW)": np.nan,
    "P_given (kW)": np.nan,
    "P_installed (kW)": np.nan,
    "P_partial (-)": np.nan,
    "power_capped": pd.NA,
    "energy_required (kWh)": np.nan,
    "total_energy (kWh)": np.nan,
    "energy_shaft (kWh)": np.nan,
}


def _is_missing(value) -> bool:
    """Return True for scalar None/NaN/NA values."""

    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.ndim(missing) == 0 else False


def _finite_float(value):
    """Return a finite float, or None when the value is missing/non-finite."""

    if _is_missing(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _get_env_graph(obj):
    """Return the environment graph, accepting either ``.graph`` or ``.FG``."""

    env = getattr(obj, "env", None)
    if env is None:
        return None
    graph = getattr(env, "graph", None)
    if graph is None:
        graph = getattr(env, "FG", None)
    return graph


def _get_edge_data(graph, node_u, node_v):
    """Return one edge-attribute mapping for a graph or multigraph."""

    if graph is None or _is_missing(node_u) or _is_missing(node_v):
        return None

    edge_data = graph.get_edge_data(node_u, node_v)
    if edge_data is None:
        return None

    if graph.is_multigraph():
        # Each value is an edge-attribute mapping. The event table currently
        # carries no edge key, so use the first parallel edge deterministically.
        try:
            first_key = sorted(edge_data.keys(), key=str)[0]
            edge_data = edge_data[first_key]
        except (AttributeError, IndexError, KeyError, TypeError):
            return None

    return edge_data if isinstance(edge_data, Mapping) else None


def _mapping_value(mapping, *keys):
    """Read the first present value from an edge mapping or nested ``Info``."""

    if not isinstance(mapping, Mapping):
        return None

    for key in keys:
        value = mapping.get(key, None)
        if not _is_missing(value):
            return value

    info = mapping.get("Info", {})
    if isinstance(info, Mapping):
        for key in keys:
            value = info.get(key, None)
            if not _is_missing(value):
                return value

    return None


def _event_hydraulics(row, graph, node_u, node_v):
    """Return ``(raw_depth, width, channel_area, edge_data)`` for one event.

    Direct edge attributes are preferred. Depth falls back to the existing
    geometry-based OpenTNSim helper. Width has no reliable geometry fallback
    and therefore remains None when it is absent from the edge.

    ``channel_area`` is the real wetted cross-section A_C of the section. Rhine
    reaches are not rectangular, so when A_C is present Spitzer uses the depth
    alongside the vessel for C_Shallow/UKC and A_C/W for the limiting economic
    speed. When A_C is absent the resistance model falls back to A_C = W*h, i.e.
    the rectangular approximation.
    """


    edge_data = _get_edge_data(graph, node_u, node_v)

    depth = _finite_float(_mapping_value(edge_data, "GeneralDepth", "Depth"))
    width = _finite_float(_mapping_value(edge_data, "GeneralWidth", "general_width"))
    channel_area = _finite_float(_mapping_value(edge_data,"GeneralCrossSectionArea","CrossSectionArea","general_cross_section_area",))

    if depth is None:
        start_location = row.get("start location", None)
        stop_location = row.get("stop location", None)
        if not _is_missing(start_location) and not _is_missing(stop_location):
            try:
                depth = _finite_float(graph_module.calculate_depth(start_location,stop_location,graph,))
            except Exception:
                logger.debug("Geometry-based depth lookup failed for edge (%s, %s)",node_u,node_v,exc_info=True,)

    return depth, width, channel_area, edge_data


def _get_event_current(obj, node_u, node_v, t_seconds, *, edge_data=None, fallback=0.0,):
    """Return signed current in m/s, positive along the sailing direction.

    Lookup order:
    1. ``env.get_current(u, v, t)`` for time-resolved forcing.
    2. Edge attributes ``current_ms`` or ``Current``.
    3. The event-table fallback value, normally zero.
    """

    if _is_missing(node_u) or _is_missing(node_v):
        return float(fallback)

    env = getattr(obj, "env", None)
    if env is not None and hasattr(env, "get_current"):
        try:
            current = float(env.get_current(node_u, node_v, t_seconds))
            if np.isfinite(current):
                return current
        except Exception:
            logger.debug("env.get_current failed for edge (%s, %s)", node_u, node_v, exc_info=True,)

    if edge_data is None:
        edge_data = _get_edge_data(_get_env_graph(obj), node_u, node_v)

    current = _finite_float(_mapping_value(edge_data, "current_ms", "Current"))
    return current if current is not None else float(fallback)


def _is_sailing_event(row):
    """Return True for a sailing activity with positive distance and duration."""

    name = str(row.get("activity name", ""))
    distance = _finite_float(row.get("distance (m)", None))
    duration = _finite_float(row.get("duration (s)", None))

    return (name.startswith("Sailing from node") and distance is not None and distance > 0.0 and duration is not None and duration > 0.0)


def _ensure_event_columns(df):
    """Create output columns that are absent from the input event table."""

    for column, default in _EVENT_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default

    for column in _FACTOR_COLUMNS.values():
        if column not in df.columns:
            df[column] = np.nan


def _clear_hydrodynamic_columns(df, index):
    """Clear sailing-only outputs to avoid carrying stale object attributes."""

    protected = {
        "stationary",
        "current (m/s)",
        "v_g (m/s)",
        "v_w (m/s)",
        "engine age (year)",
        "P_hotel (kW)",
        "P_tot (kW)",
        "P_given (kW)",
        "P_installed (kW)",
        "P_partial (-)",
        "power_capped",
        "energy_required (kWh)",
        "total_energy (kWh)",
        "energy_shaft (kWh)",
    }
    for column, default in _EVENT_DEFAULTS.items():
        if column not in protected:
            df.at[index, column] = default


def _store_resistance_diagnostics(df, index, obj):
    """Resistance results and channel hydraulics of one event row."""

    df.at[index, "resistance model"] = getattr(obj, "resistance_model", "holtrop_zeng_karpov")
    df.at[index, "confinement mode"] = getattr(obj, "confinement_mode", "none")

    df.at[index, "R_total (kN)"] = getattr(obj, "R_tot", np.nan)
    df.at[index, "R_friction (kN)"] = getattr(obj, "R_f", np.nan)
    df.at[index, "R_friction_form_corrected (kN)"] = getattr( obj, "R_f_one_k1", np.nan)
    df.at[index, "R_appendage (kN)"] = getattr(obj, "R_APP", np.nan)
    df.at[index, "R_wave (kN)"] = getattr(obj, "R_W", np.nan)
    df.at[index, "R_transom (kN)"] = getattr(obj, "R_TR", np.nan)
    df.at[index, "R_correlation (kN)"] = getattr(obj, "R_A", np.nan)
    df.at[index, "R_bulbous_bow (kN)"] = getattr(obj, "R_B", np.nan)
    df.at[index, "R_residual (kN)"] = getattr(obj, "R_res", np.nan)
    df.at[index, "R_base (kN)"] = getattr(obj, "R_base", np.nan)
    df.at[index, "R_confinement (kN)"] = getattr(obj, "R_confinement", np.nan)
    df.at[index, "R_confinement_friction (kN)"] = getattr(obj, "R_confinement_friction", np.nan)
    df.at[index, "R_confinement_drawdown (kN)"] = getattr(obj, "R_confinement_drawdown", np.nan)

    df.at[index, "Reynolds number"] = getattr(obj, "R_e", np.nan)
    df.at[index, "friction coefficient"] = getattr(obj, "C_f", np.nan)
    df.at[index, "shallow-water coefficient"] = getattr(obj, "C_Shallow", np.nan)
    df.at[index, "Froude depth"] = getattr(obj, "F_rh", np.nan)
    df.at[index, "Froude length"] = getattr(obj, "F_rL", np.nan)
    df.at[index, "Karpov alpha"] = getattr(obj, "alpha_xx", np.nan)
    df.at[index, "Karpov clamped"] = getattr(obj, "karpov_clamped", pd.NA)

    # channel hydraulics of the increment
    result = getattr(obj, "confinement_result", None) if getattr(obj, "requires_waterway_width", False) else None
    if isinstance(result, Mapping):
        df.at[index, "drawdown Z (m)"] = result.get("Z_m", np.nan)
        df.at[index, "return velocity (m/s)"] = result.get("V_R_ms", np.nan)
        df.at[index, "friction relative velocity (m/s)"] = result.get("V_relative_ms", np.nan)
        df.at[index, "blockage ratio"] = result.get("blockage_ratio", np.nan)
        df.at[index, "dynamic UKC (m)"] = result.get("dynamic_ukc_m", np.nan)
        df.at[index, "economic speed (m/s)"] = result.get("V_ec_ms", np.nan)
        df.at[index, "V/V_ec"] = result.get("V_over_V_ec", np.nan)
        df.at[index, "economic speed exceeded"] = result.get("econ_speed_exceeded", pd.NA)
        df.at[index, "critical speed (m/s)"] = result.get("V_cr_ms", np.nan)
        df.at[index, "V/V_cr"] = result.get("V_over_V_cr", np.nan)
        df.at[index, "channel area (m2)"] = result.get("channel_area_m2", np.nan)
        df.at[index, "mean depth (m)"] = result.get("mean_depth_m", np.nan)


def _shaft_power_for_accounting(obj, power_capped):
    """Return delivered propulsion power plus hotel load in kW.

    ``P_installed`` and ``P_given`` are on the brake-power basis. When capped,
    hotel power is reserved first and the remaining brake power reaches the
    propulsion shaft through ``eta_t * eta_g``.
    """

    if not power_capped:
        return float(obj.P_d + obj.P_hotel)

    eta_tg = float(obj.eta_t * obj.eta_g)
    brake_available_for_propulsion = max(float(obj.P_installed) - float(obj.P_hotel),0.0,)
    return brake_available_for_propulsion * eta_tg + float(obj.P_hotel)


# %% ADD ENERGY ATTRIBUTES INTO EVENT TABLE
def add_energy_attributes_to_eventtable(df, objs):
    """Add resistance, power, energy, and operating-point factors to events.

    Sailing events are evaluated at their event-specific speed through water,
    ``v_w = v_g - v_c``. The model-specific depth treatment is delegated to
    ``calculate_resistance_for_waterway``:

    * confinement_mode "none": raw depth -> Barrass squat -> Holtrop/Zeng/Karpov;
    * "drawdown" / "full": raw depth -> Holtrop/Zeng/Karpov + channel-width increment,
      with real waterway width.

    Stationary events use hotel power only. Missing sailing-event depth leaves
    energy fields as NaN. Missing width is permitted with confinement_mode "none",
    whose wrapper retains the historical 150 m squat fallback, but is an error with
    "drawdown" / "full", because the width is a required input of the increment.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    _ensure_event_columns(df)

    objects_by_id = {obj.id: obj for obj in objs}
    n_skipped_depth = 0
    n_width_fallback = 0

    for index, row in df.iterrows():
        object_id = row.get("object id", None)
        obj = objects_by_id.get(object_id)
        if obj is None:
            raise KeyError(f"No simulation object with id {object_id!r} was found in objs; the event table and object list do not match.")

        if not hasattr(obj, "calculate_resistance_for_waterway"):
            raise AttributeError(f"Object {object_id!r} does not provide calculate_resistance_for_waterway(). Install the revised energy.py before using this energy_logutils.py.")

        installed_power = _finite_float(getattr(obj, "P_installed", None))
        hotel_power = _finite_float(getattr(obj, "P_hotel", None))
        if installed_power is None or installed_power <= 0.0:
            raise ValueError(f"Object {object_id!r} has invalid P_installed="f"{getattr(obj, 'P_installed', None)!r}.")
        if hotel_power is None or hotel_power < 0.0:
            raise ValueError(f"Object {object_id!r} has invalid P_hotel="f"{getattr(obj, 'P_hotel', None)!r}.")
        if hotel_power > installed_power:
            raise ValueError(f"Object {object_id!r} has P_hotel={hotel_power:.3f} kW greater "f"than P_installed={installed_power:.3f} kW.")

        sailing = _is_sailing_event(row)
        df.at[index, "stationary"] = not sailing
        df.at[index, "engine age (year)"] = getattr(obj, "C_year", np.nan)
        df.at[index, "P_installed (kW)"] = installed_power
        df.at[index, "P_hotel (kW)"] = hotel_power

        duration_s = _finite_float(row.get("duration (s)", None))
        if duration_s is None or duration_s < 0.0:
            raise ValueError(f"Invalid duration for event {index}: {row.get('duration (s)', None)!r}.")
        dt_h = duration_s / 3600.0

        if sailing:
            graph = _get_env_graph(obj)
            if graph is None:
                raise ValueError(f"Object {object_id!r} has no environment graph (.graph or .FG); ""sailing-event hydraulics cannot be evaluated.")

            node_u = row.get("start node", None)
            node_v = row.get("stop node", None)
            h_raw, waterway_width, channel_area, edge_data = _event_hydraulics(row,graph,node_u,node_v,)

            distance_m = _finite_float(row.get("distance (m)", None))
            if distance_m is None or distance_m <= 0.0 or duration_s <= 0.0:
                raise ValueError(f"Invalid sailing kinematics for event {index}: "f"distance={distance_m!r}, duration={duration_s!r}.")

            v_g = distance_m / duration_s
            t_seconds = pd.Timestamp(row["start time"]).timestamp()

            fallback_current = _finite_float(row.get("current (m/s)", None))
            if fallback_current is None:
                fallback_current = 0.0

            v_c = _get_event_current(obj, node_u, node_v, t_seconds, edge_data=edge_data, fallback=fallback_current,)
            v_w = v_g - v_c

            if not np.isfinite(v_w) or v_w <= 0.0:
                raise ValueError(f"Non-positive water speed on sailing event {index} "
                    f"(object {object_id!r}, edge {node_u!r} -> {node_v!r}): "
                    f"v_g={v_g:.3f} m/s and v_c={v_c:.3f} m/s. Check the "
                    "current sign convention; positive current must point along "
                    "the sailing direction.")

            df.at[index, "v_g (m/s)"] = v_g
            df.at[index, "v_w (m/s)"] = v_w
            df.at[index, "current (m/s)"] = v_c
            df.at[index, "waterdepth raw (m)"] = (np.nan if h_raw is None else h_raw)
            df.at[index, "waterway width (m)"] = (np.nan if waterway_width is None else waterway_width)

            if h_raw is None or h_raw <= 0.0:
                n_skipped_depth += 1
                logger.warning("No positive depth for event %s (object %r, edge %r -> %r); energy and resistance fields remain NaN.",index,object_id,node_u,node_v,)
                continue

            needs_width = bool(getattr(obj, "requires_waterway_width", False))
            if needs_width and (waterway_width is None or waterway_width <= 0.0):
                raise ValueError(f"Missing positive GeneralWidth for confinement_mode "
                    f"{getattr(obj, 'confinement_mode', None)!r} event {index} "
                    f"(object {object_id!r}, edge {node_u!r} -> {node_v!r}). "
                    "The channel-width increment cannot be evaluated with a silent width fallback.")
            
            if not needs_width and waterway_width is None:
                n_width_fallback += 1

            try:
                obj.calculate_resistance_for_waterway(v=v_w, h_0=h_raw, width=waterway_width, channel_area=channel_area,)
                h_for_model = float(obj.h_resistance)
                obj.calculate_total_power_required(v=v_w, h_0=h_for_model,)
            except Exception:
                logger.exception("Energy evaluation failed for event %s (object %r, edge %r -> %r, "
                    "v_w=%.3f m/s, depth=%.3f m, width=%r).", index, object_id, node_u, node_v, v_w, h_raw, waterway_width,)
                raise

            # Supplying P_partial prevents correction_factors() from calculating
            # power a second time at the same event.
            obj.calculate_emission_factors_total(v=v_w, h_0=h_for_model, P_partial=obj.P_partial,)
            obj.calculate_SFC_final(v=v_w, h_0=h_for_model, P_partial=obj.P_partial,)

            P_tot = float(obj.P_tot)
            P_given = float(obj.P_given)
            power_capped = bool(P_tot > float(obj.P_installed))
            P_shaft = _shaft_power_for_accounting(obj, power_capped)

            df.at[index, "waterdepth (m)"] = h_for_model
            df.at[index, "waterdepth resistance (m)"] = h_for_model
            df.at[index, "waterdepth navigation (m)"] = getattr(obj, "h_navigation", np.nan)
            df.at[index, "squat (m)"] = getattr(obj, "squat_m", np.nan)

            _store_resistance_diagnostics(df, index, obj)

            df.at[index, "P_effective (kW)"] = getattr(obj, "P_e", np.nan)
            df.at[index, "P_delivered (kW)"] = getattr(obj, "P_d", np.nan)
            df.at[index, "P_brake_propulsion (kW)"] = getattr(obj, "P_b", np.nan)

        else:
            _clear_hydrodynamic_columns(df, index)

            v_g = 0.0
            v_c = 0.0
            v_w = 0.0
            P_tot = float(obj.P_hotel)
            P_given = float(obj.P_hotel)
            P_shaft = float(obj.P_hotel)
            power_capped = False

            P_partial_hotel = min(P_tot / installed_power, 1.0)

            obj.calculate_emission_factors_total(v=0.0, h_0=np.nan, P_partial=P_partial_hotel,)
            obj.calculate_SFC_final(v=0.0, h_0=np.nan, P_partial=P_partial_hotel,)

            df.at[index, "current (m/s)"] = v_c
            df.at[index, "v_g (m/s)"] = v_g
            df.at[index, "v_w (m/s)"] = v_w
            df.at[index, "resistance model"] = getattr(obj, "resistance_model", "holtrop_zeng_karpov")
            df.at[index, "confinement mode"] = getattr(obj, "confinement_mode", "none")

        df.at[index, "P_tot (kW)"] = P_tot
        df.at[index, "P_given (kW)"] = P_given
        df.at[index, "P_partial (-)"] = min(P_given / installed_power, 1.0)
        df.at[index, "power_capped"] = power_capped

        df.at[index, "energy_required (kWh)"] = P_tot * dt_h
        df.at[index, "total_energy (kWh)"] = P_given * dt_h
        df.at[index, "energy_shaft (kWh)"] = P_shaft * dt_h

        for attribute, column in _FACTOR_COLUMNS.items():
            df.at[index, column] = getattr(obj, attribute, np.nan)

    if n_skipped_depth:
        logger.warning(
            "add_energy_attributes_to_eventtable: %d sailing events had no "
            "positive depth and remain without energy attributes.",n_skipped_depth,)
        
    if n_width_fallback:
        logger.warning("add_energy_attributes_to_eventtable: %d sailing events "
            "had no GeneralWidth. Their squat wrapper (confinement_mode \"none\") used its "
            "historical 150 m fallback. Events with confinement_mode \"drawdown\" / \"full\" never use this fallback.",
            n_width_fallback,)

    return df


# %% ADD FUEL ATTRIBUTES INTO EVENT TABLE
def add_fuel_attributes_to_event_table(df, objs=None):
    """Add fuel consumption and emissions to an energy-enriched event table.

    Basis convention
    ----------------
    * TNO diesel SFC and CO2/PM10/NOX factors multiply capped brake energy,
      ``total_energy (kWh)``.
    * Alternative-carrier SFCs based on energy density and conversion-system
      efficiency multiply ``energy_shaft (kWh)`` to avoid counting drivetrain
      losses twice.

    ``objs`` is retained for backward API compatibility and is not used because
    all operating-point factors are stored per event by
    :func:`add_energy_attributes_to_eventtable`.
    """

    if "total_energy (kWh)" not in df.columns:
        raise ValueError("DataFrame must contain 'total_energy (kWh)'. Call add_energy_attributes_to_eventtable() first.")

    for index, row in df.iterrows():
        e_brake = _finite_float(row.get("total_energy (kWh)", None))
        e_shaft = _finite_float(row.get("energy_shaft (kWh)", None))

        if e_brake is None:
            continue

        distance = _finite_float(row.get("distance (m)", None))
        duration = _finite_float(row.get("duration (s)", None))
        safe_distance = distance if distance is not None and distance > 0 else np.nan
        safe_duration = duration if duration is not None and duration > 0 else np.nan

        sfc_diesel = row.get("SFC_diesel_C_year (g/kWh)", np.nan)
        factor_co2 = row.get("factor_CO2 (g/kWh)", np.nan)
        factor_pm10 = row.get("factor_PM10 (g/kWh)", np.nan)
        factor_nox = row.get("factor_NOX (g/kWh)", np.nan)

        diesel = e_brake * sfc_diesel
        df.at[index, "diesel_consumption (g)"] = diesel
        df.at[index, "diesel_consumption_m (g/m)"] = diesel / safe_distance
        df.at[index, "diesel_consumption_s (g/s)"] = diesel / safe_duration

        co2 = e_brake * factor_co2
        pm10 = e_brake * factor_pm10
        nox = e_brake * factor_nox

        df.at[index, "CO2_emission_total (g)"] = co2
        df.at[index, "PM10_emission_total (g)"] = pm10
        df.at[index, "NOX_emission_total (g)"] = nox

        df.at[index, "CO2_emission_per_m (g/m)"] = co2 / safe_distance
        df.at[index, "PM10_emission_per_m (g/m)"] = pm10 / safe_distance
        df.at[index, "NOX_emission_per_m (g/m)"] = nox / safe_distance

        df.at[index, "CO2_emission_per_s (g/s)"] = co2 / safe_duration
        df.at[index, "PM10_emission_per_s (g/s)"] = pm10 / safe_duration
        df.at[index, "NOX_emission_per_s (g/s)"] = nox / safe_duration

        if e_shaft is not None:
            df.at[index, "LH2_PEMFC_consumption (g)"] = e_shaft * row.get("SFC_LH2_PEMFC (g/kWh)", np.nan)
            df.at[index, "LH2_PEMFC_consumption (m3)"] = e_shaft * row.get("SFC_LH2_PEMFC (m3/kWh)", np.nan)
            df.at[index, "Li_NMC_battery (g)"] = e_shaft * row.get("SFC_Li_NMC (g/kWh)", np.nan)
            df.at[index, "Battery2000kWh (packs)"] = e_shaft * row.get("SFC_Battery2000kWh (packs/kWh)", np.nan)

    return df