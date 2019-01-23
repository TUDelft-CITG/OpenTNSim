# Importing libraries
# import pytest library for testing
import pytest

# tranport network analysis package
import transport_network_analysis.core as core
import transport_network_analysis.graph_module as graph_module

# import spatial libraries
import pyproj
import shapely.geometry

# import graph package and simulation package
import networkx
import simpy


# Creating the test objects
# Distance between point 1 and 3 is approximately 1000 meters
# Point 2 is lies precisely between point 1 and point 3
point_1 = shapely.geometry.Point(4.49540, 51.905505)
point_2 = shapely.geometry.Point(4.48935, 51.907995)
point_3 = shapely.geometry.Point(4.48330, 51.910485)

# Make the graph
@pytest.fixture()
def graph():
    graph = graph_module.Graph()
    graph.graph = graph.graph.to_directed()

    node_1 = {"Name": "Node 1", "Geometry": shapely.geometry.Point(4.49540, 51.905505)}
    node_2 = {"Name": "Node 2", "Geometry": shapely.geometry.Point(4.48935, 51.907995)}
    node_3 = {"Name": "Node 3", "Geometry": shapely.geometry.Point(4.48330, 51.910485)}

    nodes = [node_1, node_2, node_3]

    for node in nodes:
        graph.graph.add_node(node["Name"], 
                            Geometry = node["Geometry"], 
                            Position = (node["Geometry"].x, node["Geometry"].y))

    edges = [[node_1, node_2], [node_2, node_3]]
    for edge in edges:
        graph.graph.add_edge(edge[0]["Name"], edge[1]["Name"], weight = 1)

    return graph.graph

# Make the vessel
@pytest.fixture()
def vessel():
    # Make a vessel class out of available mix-ins.
    TransportResource = type('TransportResource', 
                             (core.Identifiable, 
                              core.Log, 
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
                   "geometry": shapely.geometry.Point(4.49540, 51.905505),  # Vessel starts at point 1
                   "capacity": 1_000,
                   "compute_v": compute_v_provider(v_empty=1, v_full=1)}

    # Create the transport processing resource using the dict as keyword value pairs
    return TransportResource(**data_vessel)


# Actual testing starts here
def test_simulation_1(graph, vessel):
    # Start simpy environment
    env = simpy.Environment()

    # Add graph to environment
    env.FG = graph

    # Find the shortest past between node 1 and node 2 
    # There is only one option, passing one edge
    path = networkx.dijkstra_path(graph, "Node 1", "Node 2")

    # Add environment and path to vessel route
    vessel.env = env
    vessel.route = path

    # Create process for the simulation
    def start(env, vessel):
        while True:
            vessel.log_entry("Start sailing", env.now, "", vessel.geometry)
            yield from vessel.move()
            vessel.log_entry("Stop sailing", env.now, "", vessel.geometry)
            
            if vessel.geometry == networkx.get_node_attributes(graph, "Geometry")[vessel.route[-1]]:
                break
    
    # Run simulation
    env.process(start(env, vessel))
    env.run()

    # If simulation time is equal to the distance the test has passed
    wgs84 = pyproj.Geod(ellps='WGS84')
    distance = 0

    for i, _ in enumerate(path):
        point_1 = networkx.get_node_attributes(graph, "Geometry")[path[i]]
        point_2 = networkx.get_node_attributes(graph, "Geometry")[path[i + 1]]
        distance += wgs84.inv(point_1.x, point_1.y, point_2.x, point_2.y)[2]

        if i == len(path) - 2:
            break

    assert distance == env.now

# Actual testing starts here
def test_simulation_2(graph, vessel):
    # Start simpy environment
    env = simpy.Environment()

    # Add graph to environment
    env.FG = graph

    # Find the shortest past between node 1 and node 2 
    # There is only one option, passing one edge
    path = networkx.dijkstra_path(graph, "Node 1", "Node 3")

    # Add environment and path to vessel route
    vessel.env = env
    vessel.route = path

    # Create process for the simulation
    def start(env, vessel):
        while True:
            vessel.log_entry("Start sailing", env.now, "", vessel.geometry)
            yield from vessel.move()
            vessel.log_entry("Stop sailing", env.now, "", vessel.geometry)
            
            if vessel.geometry == networkx.get_node_attributes(graph, "Geometry")[vessel.route[-1]]:
                break
    
    # Run simulation
    env.process(start(env, vessel))
    env.run()

    # If simulation time is equal to the distance the test has passed
    wgs84 = pyproj.Geod(ellps='WGS84')
    distance = 0

    for i, _ in enumerate(path):
        point_1 = networkx.get_node_attributes(graph, "Geometry")[path[i]]
        point_2 = networkx.get_node_attributes(graph, "Geometry")[path[i + 1]]
        distance += wgs84.inv(point_1.x, point_1.y, point_2.x, point_2.y)[2]

        if i == len(path) - 2:
            break

    assert distance == env.now