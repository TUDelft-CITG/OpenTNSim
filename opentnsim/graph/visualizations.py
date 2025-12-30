# -*- coding: utf-8 -*-

"""Graph module."""
# packkage(s) for documentation, debugging, saving and loading
import plotly.graph_objects as go

# OpenTNSim
from opentnsim.graph.calculations import calculate_distance
from plotly.offline import init_notebook_mode, iplot

def plot_graph(graph, static: bool = False):
    """method to plot a graph

    Parameters
    ----------
    graph : networkx.Graph
        A graph object.
    static : bool, optional
        If True, returns a static Plotly figure object.
        If False, displays the figure

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Object that contains a graph figure.
    """

    # Labels
    labels = {node: node for node in graph.nodes()}
    edge_labels = {(u, v): f"{d['weight']} km" for u, v, d in graph.edges(data=True)}

    # positions
    positions = {node: (graph.nodes[node]["geometry"].x, graph.nodes[node]["geometry"].y) for node in graph.nodes}

    # Edge labels in meters
    edge_labels = {}
    for u, v in graph.edges():
        origin = graph.nodes[u]['geometry']
        destination = graph.nodes[v]['geometry']
        distance_m = calculate_distance(origin, destination)
        edge_labels[(u, v)] = f"{int(distance_m)} m"

    # Edge traces and arrow annotations
    edge_traces = []
    arrow_annotations = []
    for u, v in graph.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(width=2, color='red'),
            mode='lines',
            hoverinfo='none'
        ))
        arrow_annotations.append(go.layout.Annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,  # Closed arrowhead
            arrowsize=2,
            arrowwidth=1,
            arrowcolor='red'
        ))

    # Node trace
    node_trace = go.Scatter(
        x=[positions[node][0] for node in graph.nodes()],
        y=[positions[node][1] for node in graph.nodes()],
        mode='markers+text',
        marker=dict(color='darkblue', size=20),
        text=[labels[node] for node in graph.nodes()],
        textposition='middle center',
        textfont=dict(color='white', size=15),
        hoverinfo='text'
    )

    # Edge label annotations
    edge_label_annotations = []
    for (u, v), label in edge_labels.items():
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        x_mid = (x0 + x1) / 2
        y_mid = (y0 + y1) / 2
        edge_label_annotations.append(go.layout.Annotation(
            x=x_mid, y=y_mid,
            text=label,
            showarrow=False,
            font=dict(color='black', size=15)
        ))

    # Combine annotations
    annotations = arrow_annotations + edge_label_annotations

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Directed Geographic Network Graph (WGS84 Projection) with Edge Lengths in Meters",
        xaxis=dict(title="Longitude", showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title="Latitude", showgrid=False, zeroline=False, showticklabels=False),
        annotations=annotations,
        showlegend=False,
        plot_bgcolor='white',
    )

    if static is False:
        # Initialize notebook mode for Plotly
        init_notebook_mode(connected=True)
        # Display the figure in a Jupyter notebook
        iplot(fig)
    else:
        return fig
