"""Component to inspect a class constructed from mixins."""

import pandas as pd

import inspect
from typing import Any, Dict, List, Tuple, Type, Optional, get_origin, get_args, Union

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
