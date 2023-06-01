import io
import pickle

import requests
import requests_cache
import shapely.geometry

# inject caching
requests_cache.install_cache("fis_cache")

urls = {
    "0.2": "https://zenodo.org/record/4578289/files/network_digital_twin_v0.2.pickle",
    "0.3": "https://zenodo.org/record/6673604/files/network_digital_twin_v0.3.pickle",
}


def load_network(version="0.3"):
    """load the pickle version of the fairway information system network"""
    url = urls[version]
    resp = requests.get(url)
    # convert the response to a file
    f = io.BytesIO(resp.content)

    # read the graph
    graph = pickle.load(f)

    # convert the edges and nodes geometry to shapely objects
    for e in graph.edges:
        edge = graph.edges[e]
        edge["geometry"] = shapely.geometry.shape(edge["geometry"])
    for n in graph.nodes:
        node = graph.nodes[n]
        node["geometry"] = shapely.geometry.shape(node["geometry"])

    return graph
