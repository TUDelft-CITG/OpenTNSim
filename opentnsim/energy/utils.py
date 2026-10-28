import pathlib
import pandas as pd

def load_partial_engine_load_correction_factors():
    """read correction factor from package directory"""

    # Can't get this  to work with pkg_resourcs
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    correctionfactors_path = data_dir / "Correctionfactors.csv"
    df = pd.read_csv(correctionfactors_path, comment="#")

    return df


def karpov_smooth_curves():
    """read correction factor from package directory"""

    # Can't get this  to work with pkg_resourcs
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    karpov_smooth_curves_path = data_dir / "KarpovSmoothCurves.csv"
    df = pd.read_csv(karpov_smooth_curves_path, comment="#")

    return df


def check_if_route_has_depth(graph, edge_route):
    edges_with_missing_depths = []
    for edge in edge_route:
        if "GeneralDepth" not in graph.edges[edge]:
            edges_with_missing_depths.append(edge)
    if edges_with_missing_depths:
        raise ValueError(f"{edges_with_missing_depths}")
