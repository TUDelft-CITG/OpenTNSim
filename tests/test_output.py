#!/usr/bin/env python3

import pytest
import simpy
import opentnsim.output


def test_output():
    Vessel = type("Vessel", (opentnsim.output.HasOutput,), {})
    vessel = Vessel()

    assert hasattr(vessel, "output"), "Vessel should have an output attribute"
