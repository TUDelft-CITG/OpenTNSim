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

# packkage(s) for documentation, debugging, saving and loading
import pathlib
import warnings

# spatial libraries
import networkx as nx

# time libraries
import datetime

# data libraries
import numpy as np
import pandas as pd

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


# // END-NOSCAN
