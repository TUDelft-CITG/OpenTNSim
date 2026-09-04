#!/usr/bin/env python3

from opentnsim.graph.mixins import FIS


def test_fis_network_v2():
    """check if we can load the network"""
    _ = FIS.load_network(network="fis", version="0.2")


def test_fis_network_v3():
    """check if we can load the network"""
    _ = FIS.load_network(network="fis", version="0.3")

def test_euris_network_v1():
    """check if we can load the network"""
    _ = FIS.load_network(network="euris", version="0.1")
