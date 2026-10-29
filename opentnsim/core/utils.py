"""
Utilities for OpenTNSim

This file also includes the networkx read shapefile functions that used to be in networkx.
These originate from:
https://github.com/networkx/networkx/blob/9256ef670730b741105a9264197353004bd6719f/networkx/readwrite/nx_shp.py

Generates a networkx.DiGraph from point and line shapefiles.

"The Esri Shapefile or simply a shapefile is a popular geospatial vector
data format for geographic information systems software. It is developed
and regulated by Esri as a (mostly) open specification for data
interoperability among Esri and other software products."
See https://en.wikipedia.org/wiki/Shapefile for additional information.
"""

from typing import Any, Dict, List, Tuple, Type

# packkage(s) for documentation, debugging, saving and loading
import pathlib
import inspect
from IPython.display import display

# time libraries
import datetime

# data libraries
import numpy as np
import pandas as pd
import itertools
import math

# OpenTNSim
import opentnsim


def inherit_docstring(cls):
    for name, func in cls.__dict__.items():
        if callable(func) and not func.__doc__:
            parent_func = getattr(super(cls, cls), name, None)
            if parent_func:
                func.__doc__ = parent_func.__doc__
    return cls


def find_notebook_path():
    """Lookup the path where the notebooks are located. Returns a pathlib.Path object."""
    opentnsim_path = pathlib.Path(opentnsim.__file__)
    # check if the path looks normal
    assert "opentnsim" in str(opentnsim_path), "we can't find the opentnsim path: {opentnsim_path} (opentnsim not in path name)"
    # src_dir/opentnsim/__init__.py -> ../.. -> src_dir
    notebook_path = opentnsim_path.parent.parent / "notebooks"
    return notebook_path


def time_to_numpy(t_start):
    """Convert time to np.datetime64

    Parameters
    ----------
    t_start : float, datetime.datetime, pd.Timestamp
        the time to be converted
    Returns
    -------
    t_start : np.datetime64
        the converted time
    """
    if isinstance(t_start, float):
        t_start = np.datetime64(datetime.datetime.fromtimestamp(t_start))
    elif isinstance(t_start, datetime.datetime):
        t_start = np.datetime64(t_start)
    elif isinstance(t_start, pd.Timestamp):
        t_start = np.array([t_start], dtype=np.datetime64)[0]
    return t_start


def inspect_object(Class, candidate_kwargs=None, show_parameter_table=True):

    documenting = not candidate_kwargs

    def highlight_status(row):
        colors = {
            "missing": "#ffcccc",
            "missing (one alternative)": "#ffcccc",
            "optional (unused)": "#ffe5b4",
            "optional (used)": "#e8f5e9",
            "required": "#6fa6cf",
            "required (one alternative)": "#6fa6cf",
            "optional": "#afcee8",
            "added": "#c8e6c9",
        }
        color = colors.get(row["status"], "#ffffff")
        return [f"background-color: {color}"] * len(row)

    candidate_kwargs = candidate_kwargs or {}

    # Collect alternative requirement groups from the class hierarchy.
    # Each tuple is a group of parameters that must all be supplied.
    # At least one group must be satisfied.
    alternative_requirements = []

    for cls in Class.mro():
        alternative_requirements.extend(
            cls.__dict__.get("__parameter_requirements__", [])
        )

    # Collect constructor parameters.
    rows = []

    for cls in Class.mro():
        if cls is object or "__init__" not in cls.__dict__:
            continue

        sig = inspect.signature(cls.__init__)

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            rows.append({
                "parameter": name,
                "class": cls.__name__,
                "required": param.default is inspect.Parameter.empty,
                "provided": name in candidate_kwargs,
            })

    df = pd.DataFrame(rows).drop_duplicates("parameter")

    if df.empty:
        if show_parameter_table:
            display(df)
            return
        return df, False

    # Check alternative requirements.
    provided = set(candidate_kwargs)

    alternatives_satisfied = (
        not alternative_requirements
        or any(
            all(parameter in provided for parameter in group)
            for group in alternative_requirements
        )
    )

    # Add a row for an unsatisfied alternative requirement.
    df["status"] = " "
    if alternative_requirements and not alternatives_satisfied:
        alternatives_label = " OR ".join(
            " + ".join(group)
            for group in alternative_requirements
        )

        df = pd.concat(
            [
                df,
                pd.DataFrame([{
                    "parameter": f"{alternatives_label}",
                    "class": Class.__name__,
                    "required": True,
                    "provided": False,
                    "status": "combination",
                }]),
            ],
            ignore_index=True,
        )

    # Assign statuses based on the mode.
    df["missing"] = df["required"] & ~df["provided"]
    df.loc[
        df["missing"] & (df["status"] != "combination"), "status"
    ] = "missing"
    df.loc[
        df["required"] & df["provided"], "status"
    ] = "added"
    df.loc[
        ~df["required"] & df["provided"], "status"
    ] = "optional (used)"
    df.loc[
        ~df["required"] & ~df["provided"], "status"
    ] = "optional (unused)"

    # Apply alternative requirement status.
    combination_mask = df["status"].str.startswith(
        "combi"
    )

    if alternative_requirements:
        if documenting:
            df.loc[
                combination_mask, "status"
            ] = "required (one alternative)"
        elif not alternatives_satisfied:
            df.loc[
                combination_mask, "status"
            ] = "missing (one alternative)"
        else:
            # The combination requirement is satisfied.
            df.loc[combination_mask, "status"] = "added"

    # In documentation mode, show required/optional parameters.
    if documenting:
        df.loc[df["status"] == "missing", "status"] = "required"
        df.loc[
            df["status"] == "optional (unused)", "status"
        ] = "optional"

    # In validation mode, report missing individual and combination requirements.
    missing_parameters = (
        not documenting
        and (
            df["status"].isin(
                ["missing", "requirement missing"]
            ).any()
        )
    )

    df = df.sort_values(
        by=["class", "parameter"],
        ascending=[True, True],
    ).reset_index(drop=True)

    df = df.drop(
        ["missing", "required", "provided"],
        axis=1,
    )

    if documenting:
        status_order = {
            "required": 0,
            "required (one alternative)": 1,
            "optional": 2,
        }
    else:
        status_order = {
            "added": 0,
            "optional (used)": 1,
            "missing": 2,
            "requirement missing": 3,
            "optional (unused)": 4,
        }

    df["_sort_order"] = df["status"].map(status_order).fillna(99)

    df = (
        df.sort_values(
            by=["_sort_order", "class", "parameter"],
            ascending=[True, True, True],
        )
        .drop(columns="_sort_order")
        .reset_index(drop=True)
    )

    if show_parameter_table:
        display(df.style.apply(highlight_status, axis=1))
    else:
        return df, missing_parameters


def check_class_is_vesselclass(vessel_class):
    required_mixins = ["Identifiable", "Movable"]
    present_mixins = [cls.__name__ for cls in vessel_class.mro()]
    missing_mixins = [m for m in required_mixins if m not in present_mixins]

    if missing_mixins:
        raise TypeError(f"vessel_class must include the following mixins: {missing_mixins}")


def get_all_init_params(vessel_class):
    """
    Return a set of all parameters from __init__ in the MRO
    (excluding self, *args, **kwargs)
    """
    allowed = set()
    for cls in vessel_class.mro():
        if cls is object:
            continue
        if "__init__" not in cls.__dict__:
            continue
        sig = inspect.signature(cls.__init__)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                continue
            allowed.add(name)
    return allowed


def generate_vessels_from_distribution(
    env,
    vessel_class,
    vessel_parameters,
    mean_arrival_rate,
    number_of_vessels,
    origin,
    destination,
    seed=None,
    start_time=None,
    use_fleet=False,
    fleet_composition=None,
    fleet_data=None
):
    if start_time is None:
        start_time = env.simulation_start

    rng = np.random.default_rng(seed)
    arrival_time = start_time

    try:
        number_of_earlier_vessels = len(env.vessels)
    except:
        number_of_earlier_vessels = 0

    vessels = []

    if use_fleet:
        if fleet_composition is None or fleet_data is None:
            raise ValueError("fleet_composition and fleet_data must be provided when use_fleet=True")

        fleet_types = list(fleet_composition.keys())
        fleet_probs = np.array(list(fleet_composition.values()), dtype=float)
        fleet_probs = fleet_probs / fleet_probs.sum()

        fleet_specs = fleet_data.set_index("Class")

    for i in range(number_of_vessels):

        arrival_time += pd.Timedelta(
            minutes=rng.exponential(mean_arrival_rate)
        )

        params = {
            "name": f"Vessel {number_of_earlier_vessels + i}",
            "arrival_time": arrival_time
        }

        if not use_fleet:
            params.update(vessel_parameters)
        else:
            vessel_class = rng.choice(fleet_types, p=fleet_probs)
            class_row = fleet_specs.loc[vessel_class]

            resolved_params = {}

            for out_key, source_key in vessel_parameters.items():
                if source_key == "Class":
                    resolved_params[out_key] = vessel_class
                elif source_key in class_row.index:
                    resolved_params[out_key] = class_row[source_key]
                else:
                    resolved_params[out_key] = source_key

            params.update(resolved_params)

        v = vessel_parameters['v']
        vessel = create_vessel(
            env,
            vessel_class,
            origin,
            destination,
            **params
        )

        vessels.append(vessel)

    return vessels


def highlight_row_if_required(row):
    color = 'background-color: #c6efce' if bool(row['required']) else ''
    return [color] * len(row)


def describe_inits_by_mro(cls: Type[Any]) -> Dict[str, Any]:
    """
    Inspect each base in MRO and capture its __init__ parameters (excluding *args/**kwargs).
    Returns a dict with 'per_base' (mixin -> rows) and 'union' (first-seen param in MRO).
    """
    per_base: Dict[str, List[Dict[str, Any]]] = {}
    union: Dict[str, Dict[str, Any]] = {}

    for base in cls.__mro__:
        if base is object:
            continue
        init = getattr(base, "__init__", object.__init__)
        if init is object.__init__:
            continue

        sig = inspect.signature(init)
        rows: List[Dict[str, Any]] = []  # <-- fixed bracket

        for pname, p in sig.parameters.items():
            if pname == "self":
                continue
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):  # skip *args/**kwargs
                continue

            rows.append({
                "name": pname,
                "required": (p.default is inspect._empty),
                "default": None if p.default is inspect._empty else p.default,
                "annotation": None if p.annotation is inspect._empty else p.annotation,
                "kind": p.kind.name,
                "declared_in": base.__name__,
            })

            # Union: first occurrence in MRO wins if duplicates
            if pname not in union:
                union[pname] = rows[-1].copy()

        per_base[base.__name__] = rows

    return {"per_base": per_base, "union": union}


def params_for_class(cls: Type[Any], *, skip_composed: bool = True) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Return (required_param_names, optional_param_names, param_annotations_by_name) for a class
    by aggregating across its mixins in MRO order. If skip_composed=True, skip params declared
    on cls itself (e.g., 'SystemElement') to avoid duplicates with 'Identifiable'.
    """
    info = describe_inits_by_mro(cls)

    ann_map: Dict[str, Any] = {}
    required: List[str] = []
    optional: List[str] = []

    # Ordered bases in MRO
    mro_bases = [b for b in cls.__mro__ if b is not object]
    if skip_composed:
        mro_bases = [b for b in mro_bases if b.__name__ != cls.__name__]

    for base in mro_bases:
        rows = info["per_base"].get(base.__name__, [])
        for r in rows:
            pname = r["name"]
            if pname not in ann_map and r.get("annotation", None) is not None:
                ann_map[pname] = r["annotation"]
            if r["required"]:
                if pname not in required:
                    required.append(pname)
            else:
                if pname not in optional:
                    optional.append(pname)

    return required, optional, ann_map


def inits_to_dataframe(cls, include_types: bool = False, include_kind: bool = False) -> pd.DataFrame:
    info = describe_inits_by_mro(cls)

    records = []
    skip_class_name = cls.__name__  # e.g., "SystemElement"

    for mixin, rows in info["per_base"].items():
        if mixin == skip_class_name:
            # Skip entries declared in the composed class
            continue

        for r in rows:
            rec = {
                "mixin": mixin,
                "inputname": r["name"],
                "required": r["required"],
                "default": r["default"],
            }
            if include_types:
                ann = r.get("annotation", None)
                rec["type"] = getattr(ann, "__name__", str(ann)) if ann is not None else None
            if include_kind:
                rec["kind"] = r.get("kind", None)
            records.append(rec)

    df = pd.DataFrame.from_records(records)

    # Sort by MRO order (excluding the composed class), then by parameter name
    mro_order = [b.__name__ for b in cls.__mro__ if b is not object and b.__name__ != skip_class_name]
    order_map = {name: i for i, name in enumerate(mro_order)}
    df["__mixin_order"] = df["mixin"].map(order_map)
    df.sort_values(["__mixin_order", "inputname"], inplace=True, kind="stable")
    df.drop(columns="__mixin_order", inplace=True)

    return df


def create_object(name, classes):
    for perm in itertools.permutations(classes):
        try:
            object = type(name, perm, {})
            return object
        except TypeError:
            pass
    raise TypeError("No valid base class order exists")


def create_vessel(
    env,
    vessel_class,
    origin = None,
    destination = None,
    route = None,
    show_unused_optional_parameters=False,
    **kwargs,
):
    # Check vessel class
    check_class_is_vesselclass(vessel_class)

    # Automatically generated constructor arguments take precedence
    auto_kwargs = {
        "env": env,
        "origin": origin,
        "destination": destination,
        "route": route,
    }
    candidate_kwargs = {**kwargs, **auto_kwargs}

    # Inspect constructor parameters
    df, missing_parameters = inspect_object(
        vessel_class,
        candidate_kwargs,
        False,
    )

    if show_unused_optional_parameters or missing_parameters:
        display(df)

    # Separate constructor arguments from extra metadata
    allowed = get_all_init_params(vessel_class)

    filtered_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in allowed
    }
    extra_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key not in allowed
    }

    # Construct vessel
    vessel = vessel_class(**filtered_kwargs)

    # Store unused parameters as metadata when supported
    if "ExtraMetadata" in {
        cls.__name__ for cls in type(vessel).mro()
    }:
        vessel.metadata.update(extra_kwargs)

    if show_unused_optional_parameters:
        return vessel, df

    return vessel