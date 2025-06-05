#!/usr/bin/env python3

import opentnsim.fis


def test_network_v2():
    """check if we can load the network"""
    _ = opentnsim.fis.load_network(version="0.2")


def test_network_v3():
    """check if we can load the network"""
    _ = opentnsim.fis.load_network(version="0.3")
