#!/usr/bin/env python3

import opentnsim.energy


def test_load_partial_engine_load_correction_factors():
    df = opentnsim.energy.load_partial_engine_load_correction_factors()
    assert df.shape and df.shape[0] > 0


def test_karpov_smooth_curves():
    df = opentnsim.energy.karpov_smooth_curves()
    assert df.shape and df.shape[0] > 0
