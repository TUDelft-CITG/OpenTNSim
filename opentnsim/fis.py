import io

import requests
import networkx as nx
import shapely.geometry


def load_network():
    """load the pickle version of the fairway information system network"""
    url = "https://zenodo.org/record/4578289/files/network_digital_twin_v0.2.pickle"
    resp = requests.get(url)
    # convert the response to a file
    f = io.BytesIO(resp.content)

    # read the graph
    graph = nx.read_gpickle(f)

    # convert the edges and nodes geometry to shapely objects
    for e in graph.edges:
        edge = graph.edges[e]
        edge["geometry"] = shapely.geometry.asShape(edge["geometry"])
    for n in graph.nodes:
        node = graph.nodes[n]
        node["geometry"] = shapely.geometry.asShape(node["geometry"])

    return graph
