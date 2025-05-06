"""
Mixin classes for routable objects.

The following classes are provided:
- Routable
- Routeable (deprecated naming convention, use Routable instead)

"""
# packkage(s) for documentation, debugging, saving and loading
import logging
import warnings
import deprecated

# spatial libraries
import networkx as nx

# you need these dependencies (you can get these from anaconda)
# package(s) related to the simulation

# Use OpenCLSim objects for core objects (identifiable is imported for later use.)
from openclsim.core import SimpyObject

# get logger
logger = logging.getLogger(__name__)





class Routable(SimpyObject):
    """Mixin class: Something with a route (networkx format)

    Parameters
    ----------
    route: list
        list of node-IDs
    complete_path: list, optional
        ???
    args, kwargs:
        passed to SimpyObject. Must at least contain parameter env: simpy.Environment.

    Attributes
    -----------
    route: list
        list of node-IDs
    complete_path: list, optional
        ???
    position_on_route: int
        index of position on the route
    """

    def __init__(self, route, complete_path=None, *args, **kwargs):
        """Initialization"""
        super().__init__(*args, **kwargs)
        env = kwargs.get("env")
        # if env is given and env is not None
        if env is not None:
            has_fg = hasattr(env, "FG")
            has_graph = hasattr(env, "graph")
            if has_fg and not has_graph:
                warnings.warn(".FG attribute has been renamed to .graph, please update your code", DeprecationWarning)
            assert (
                has_fg or has_graph
            ), "Routable expects `.graph` (a networkx graph) to be present as an attribute on the environment"
        super().__init__(*args, **kwargs)
        self.route = route
        # start at start of route
        self.position_on_route = 0
        self.complete_path = complete_path

    @property
    def graph(self):
        """
        Return the graph of the underlying environment.

        If it's multigraph cast to corresponding type
        If you want the multidigraph use the HasMultiGraph mixin

        """
        graph = None
        if hasattr(self.env, "graph"):
            graph = self.env.graph
        elif hasattr(self.env, "FG"):
            graph = self.env.FG
        else:
            raise ValueError("Routable expects .graph to be present on env")

        if isinstance(graph, nx.MultiDiGraph):
            return nx.DiGraph(graph)
        elif isinstance(graph, nx.MultiGraph):
            return nx.Graph(graph)
        return graph


@deprecated.deprecated(reason="Use Routable instead of Routeable")
class Routeable(Routable):
    """Old name for Mixin class: renamed to Routable."""