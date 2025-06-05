#!/usr/bin/env python3

import opentnsim.output


def test_output():
    Vessel = type("Vessel", (opentnsim.output.HasOutput,), {})
    vessel = Vessel()

    assert hasattr(vessel, "output"), "Vessel should have an output attribute"
