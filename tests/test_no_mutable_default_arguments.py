"""Functions in the port, lock and environment modules must not use mutable default arguments (issue #153).

A list, dict, set, array or dataframe as default value is created once and shared by all calls (and by all objects
created with that call), so changes made by one caller show up in all others. Use `None` and create the value in the function.
"""

import importlib
import inspect

import numpy as np
import pandas as pd
import pytest

MODULES = [
    "opentnsim.environment.mixins.hydrodynamics",
    "opentnsim.environment.utils",
    "opentnsim.lock.calculations",
    "opentnsim.lock.logutils",
    "opentnsim.lock.mixins.chamber",
    "opentnsim.lock.mixins.complex",
    "opentnsim.lock.mixins.master",
    "opentnsim.lock.mixins.operator",
    "opentnsim.lock.utils",
    "opentnsim.lock.visualizations",
    "opentnsim.port.calculations",
    "opentnsim.port.logutils",
    "opentnsim.port.mixins.anchorage_area",
    "opentnsim.port.mixins.berth",
    "opentnsim.port.mixins.port",
    "opentnsim.port.mixins.rules",
    "opentnsim.port.mixins.terminal",
    "opentnsim.port.mixins.tidal_accessibility",
    "opentnsim.port.mixins.waterway",
    "opentnsim.port.utils",
    "opentnsim.port.visualizations",
]

MUTABLE_TYPES = (list, dict, set, bytearray, np.ndarray, pd.DataFrame, pd.Series)


def _functions_defined_in(module):
    """All functions and methods that are defined in the module itself (not imported)."""
    for name, member in inspect.getmembers(module):
        if inspect.isfunction(member) and member.__module__ == module.__name__:
            yield name, member
        elif inspect.isclass(member) and member.__module__ == module.__name__:
            for method_name, method in inspect.getmembers(member, inspect.isfunction):
                if method.__module__ == module.__name__:
                    yield f"{name}.{method_name}", method


@pytest.mark.parametrize("module_name", MODULES)
def test_no_mutable_default_arguments(module_name):
    module = importlib.import_module(module_name)
    offenders = []
    for name, function in _functions_defined_in(module):
        for parameter in inspect.signature(function).parameters.values():
            if isinstance(parameter.default, MUTABLE_TYPES):
                offenders.append(f"{name}({parameter.name}={type(parameter.default).__name__})")
    assert not offenders, f"mutable default arguments in {module_name}: {offenders}"
