import networkx as nx
import shapely
import warnings

import opentnsim

class NetworkWarning(Warning):
    pass


def find_notebook_path():
    """Lookup the path where the notebooks are located. Returns a pathlib.Path object."""
    opentnsim_path = pathlib.Path(opentnsim.__file__)
    # check if the path looks normal
    assert 'opentnsim' in str(opentnsim_path), "we can't find the opentnsim path: {opentnsim_path} (opentnsim not in path name)"
    # src_dir/opentnsim/__init__.py -> ../.. -> src_dir
    src_path = opentnsim_path.parent.parent
    notebook_path = opentnsim_path.parent.parent / "notebooks"
    return notebook_path


def network_check(graph):
    """Assertions about the graphs used in OpenTNSim"""
    node_type = (str, shapely.Point)
    ok = True

    if not isinstance(graph, nx.Graph):
        warnings.warn("graph should be of type nx.Graph", NetworkWarning)
        ok = False
    if not len(graph.nodes) >= 2:
        warnings.warn("there should be at least 2 nodes in the graph", NetworkWarning)
        ok = False
    if not len(graph.edges) >= 1:
        warnings.warn("there should be at least 1 edge in the graph", NetworkWarning)
        ok = False
    if not all(isinstance(n, node_type) for n in graph.nodes.keys()):
        warnings.warn("all keys should be str or Point", NetworkWarning)
        ok = False
    # check all edges
    for e in (graph.edges.keys()):
        if not len(e) == 2:
            warnings.warn(f"edge keys should be tuples of length 2, {e} was not", NetworkWarning)
            ok = False
            # stop checking
            break

        source, target = e
        if not isinstance(source, node_type):
            warnings.warn(f"edges should be of a tuple of Points or str, {e} was not ", NetworkWarning)
            ok = False
            # stop checking
            break
    for e, edge in graph.edges.items():
        if not 'geometry' in edge:
            warnings.warn(f"edges should have of geometry attribute, {e} did not ", NetworkWarning)
            ok = False
        elif not isinstance(edge['geometry'], (str, shapely.Geometry)):
            warnings.warn(f"edges geometry should be of type string or Geometry, {edge['geometry']} was not.", NetworkWarning)
            ok = False
    return ok
