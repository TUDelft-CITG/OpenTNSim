# -*- coding: utf-8 -*-
from importlib.metadata import version, PackageNotFoundError

import opentnsim.core as core  # now contains mixin class consumes energy, also the modified vessel properties
import opentnsim.energy as energy
import opentnsim.graph.mixins as mixins
import opentnsim.lock as lock
import opentnsim.model as model
import opentnsim.plot as plot
import opentnsim.strategy as strategy


"""Top-level package for OpenTNSim."""

__author__ = """Mark van Koningsveld"""
__email__ = "M.vanKoningsveld@tudelft.nl"

try:
    __version__ = version("opentnsim")
except PackageNotFoundError:
    # Package is not installed yet (e.g., running tests from source without an install)
    # This value will be replaced by the build system when the package is installed
    __version__ = "uninstalled" 
