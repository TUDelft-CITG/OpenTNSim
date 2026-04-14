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

from typing import Any, Dict, List, Tuple, Type, get_origin, get_args

# packkage(s) for documentation, debugging, saving and loading
import pathlib
import inspect
from IPython.display import display

# spatial libraries
import networkx as nx

# time libraries
import datetime

# data libraries
import numpy as np
import pandas as pd
import itertools

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


def inspect_object(Class, candidate_kwargs={}, show_parameter_table = False):

    def highlight_status(row):
        if row["status"] == "missing":
            return ["background-color: #ffcccc"] * len(row)
        if row["status"] == "optional (unused)":
            return ["background-color: #ffe5b4"] * len(row)
        if row["status"] == "optional (used)":
            return ["background-color: #e8f5e9"] * len(row)
        return ["background-color: #c8e6c9"] * len(row)

    rows = []
    for cls in Class.mro():
        if cls is object:
            continue

        if "__init__" not in cls.__dict__:
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

            rows.append(
                {
                    "parameter": name,
                    "class": cls.__name__,
                    "required": param.default is inspect.Parameter.empty,
                    "provided": name in candidate_kwargs,
                }
            )

    df = pd.DataFrame(rows).drop_duplicates("parameter")
    df["missing"] = df["required"] & (~df["provided"])
    df["status"] = " "
    df.loc[df["missing"], "status"] = "missing"
    df.loc[(df["required"]) & (df["provided"]), "status"] = "added"
    df.loc[(~df["required"]) & (df["provided"]), "status"] = "optional (used)"
    df.loc[(~df["required"]) & (~df["provided"]), "status"] = "optional (unused)"
    df = df.sort_values(
        by=["missing", "required", "class", "parameter"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    missing_parameters = not df[df.status == 'missing'].empty
    df = df.drop(["missing", "required", "provided"], axis=1)
    df = df.style.apply(highlight_status, axis=1)
    if show_parameter_table:
        display(df)
    else:
        return df, missing_parameters


def check_class_is_vesselclass(VesselClass):
    required_mixins = ["Identifiable", "Movable"]
    present_mixins = [cls.__name__ for cls in VesselClass.mro()]
    missing_mixins = [m for m in required_mixins if m not in present_mixins]

    if missing_mixins:
        raise TypeError(f"VesselClass must include the following mixins: {missing_mixins}")


def get_all_init_params(VesselClass):
    """
    Return a set of all parameters from __init__ in the MRO
    (excluding self, *args, **kwargs)
    """
    allowed = set()
    for cls in VesselClass.mro():
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


def create_vessel(
        VesselClass,
        env,
        start_node,
        end_node,
        show_unused_optional_parameters = False,
        **kwargs,
):
    # check class
    check_class_is_vesselclass(VesselClass)

    route = nx.dijkstra_path(env.graph, start_node, end_node)
    geometry = env.graph.nodes[start_node]["geometry"]

    # check input
    auto_kwargs = {"env": env,
                   "route": route,
                   "geometry": geometry, }
    candidate_kwargs = {**auto_kwargs, **kwargs}
    df, missing_parameters = inspect_object(VesselClass, candidate_kwargs)
    if show_unused_optional_parameters or missing_parameters:
        display(df)

    # construct vessel
    allowed = get_all_init_params(VesselClass)
    extra_kwargs = {k: v for k, v in candidate_kwargs.items() if k not in allowed}
    filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in allowed}
    vessel = VesselClass(**filtered_kwargs)
    if "ExtraMetadata" in [cls.__name__ for cls in type(vessel).mro()]:
        vessel.metadata.update(extra_kwargs)

    if show_unused_optional_parameters:
        return vessel, df
    return vessel


def generate_vessels_from_distribution(env,
                                       VesselClass,
                                       vessel_parameters,
                                       mean_arrival_rate,
                                       number_of_vessels,
                                       start_node,
                                       end_node,
                                       seed=None,
                                       start_time=None):
    if start_time is None:
        start_time = env.epoch
    arrival_time = start_time

    rng = np.random.default_rng()
    if seed is not None:
        rng = np.random.default_rng(seed)

    vessels = []
    try:
        number_of_earlier_vessels = len(env.vessels)
    except:
        number_of_earlier_vessels = 0

    for i in range(number_of_vessels):
        arrival_time += pd.Timedelta(minutes=rng.exponential(mean_arrival_rate))

        params = {
            'name': f"Vessel {number_of_earlier_vessels + i}",
            'arrival_time': arrival_time,
            **vessel_parameters
        }

        vessel = create_vessel(
            VesselClass,
            env,
            start_node=start_node,
            end_node=end_node,
            **params
        )

        vessels.append(vessel)

    env.vessels[vessel.id] = vessel
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
