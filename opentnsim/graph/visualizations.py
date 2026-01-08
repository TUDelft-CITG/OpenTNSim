# -*- coding: utf-8 -*-

"""Graph module."""
# packkage(s) for documentation, debugging, saving and loading
import plotly.graph_objects as go
import folium
import pandas as pd

# OpenTNSim
from opentnsim.graph.calculations import calculate_distance
from plotly.offline import init_notebook_mode, iplot
from opentnsim.port.visualizations import plot_anchorage_areas, plot_turning_basins, plot_berths

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


def plot_graph_folium(graph, longitude, latitude, zoom_start=5, berths = None, turning_basins = None, anchorage_areas = None):
    m = folium.Map(location=[latitude, longitude], zoom_start=zoom_start, tiles="cartodbpositron")

    if isinstance(anchorage_areas,pd.DataFrame):
        plot_anchorage_areas(anchorage_areas, m=m)
    if isinstance(turning_basins,pd.DataFrame):
        plot_turning_basins(turning_basins, m=m)
    if isinstance(berths,pd.DataFrame):
        plot_berths(berths, m=m)

    for edge in graph.edges(data=True):
        points_x = list(edge[2]["geometry"].coords.xy[0])
        points_y = list(edge[2]["geometry"].coords.xy[1])
        line = []
        for i, _ in enumerate(points_x):
            line.append((points_y[i], points_x[i]))

        else:
            popup = folium.Popup(width=500, height=300)
            folium.PolyLine(line, weight=3, color='violet', tooltip=[edge[0], edge[1]],
                            popup=[edge[0], edge[1]]).add_to(m)

    for node in graph.nodes(data=True):
        points_x = list(node[1]["geometry"].coords.xy[0])
        points_y = list(node[1]["geometry"].coords.xy[1])

        point = []
        for i, _ in enumerate(points_x):
            point.append((points_y[i], points_x[i]))
        else:
            if 'terminal' in node[1]:
                terminal = node[1]['terminal']
                folium.Circle(point[0], radius=5, color='black', fill=False, fill_opacity=1, tooltip=terminal,
                              popup=node[0]).add_to(m)
            else:
                folium.Circle(point[0], radius=5, color='black', fill=False, fill_opacity=1, tooltip=node[0],
                              popup=node[0]).add_to(m)

    return m
