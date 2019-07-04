# Importing libraries
# import pytest library for testing
import pytest

# tranport network analysis package
import opentnsim.core as core
import opentnsim.graph_module as graph_module

# import spatial libraries
import pyproj
import shapely.geometry

# import graph package and simulation package
import networkx
import simpy


# Creating the test objects
# Make the vessel
@pytest.fixture()
def vessel():
    # Make a vessel class out of available mix-ins.
    TransportResource = type('TransportResource', 
                             (core.Identifiable, 
                              core.ContainerDependentMovable, 
                              core.HasResource, 
                              core.Routeable), 
                            {})

    # ContainerDependentMovable requires a function as input for "compute_v", so we define it as follows
    # No matter what v_empty and v_full will be, the vessel velocity will always be 1 meter per second
    def compute_v_provider(v_empty, v_full):
        return lambda x: 1

    # Create a dict with all required settings
    data_vessel = {"env": None,
                   "name": "Vessel number 1",
                   "route": None,
                   "geometry": shapely.geometry.Point(0, 0),  # lon, lat
                   "capacity": 1_000,
                   "compute_v": compute_v_provider(v_empty=1, v_full=1)}

    # Create the transport processing resource using the dict as keyword value pairs
    return TransportResource(**data_vessel)


# Actual testing starts here
def test_type(vessel):
    # Test if vessel is initialized correctly
    assert type(vessel.__dict__) == dict
    
    # Test if the vessel resource is initialized with a Simpy Resource
    assert type(vessel.__dict__["resource"]) == simpy.resources.resource.Resource

    # Test if the vessel container is initialized with a Simpy container
    assert type(vessel.__dict__["container"]) == simpy.resources.container.Container
    
    # Test if the vessel location is a shapely point
    assert type(vessel.__dict__["geometry"]) == shapely.geometry.point.Point

    # Test if the vessel is initialized with an ID
    assert type(vessel.__dict__["id"]) == str