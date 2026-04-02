# -*- coding: utf-8 -*-

"""Graph module."""
# packkage(s) for documentation, debugging, saving and loading
import plotly.graph_objects as go
import folium
import pandas as pd
import geopandas as gpd
import networkx as nx

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

    # positions
    positions = {node: (graph.nodes[node]["geometry"].x, graph.nodes[node]["geometry"].y) for node in graph.nodes}

    # Edge labels in meters
    edge_labels = {}
    for edge in graph.edges:
        u, v = edge[:2]
        origin = graph.nodes[u]['geometry']
        destination = graph.nodes[v]['geometry']
        distance_m = calculate_distance(origin, destination)
        edge_labels[edge] = f"{int(distance_m)} m"

    # Edge traces and arrow annotations
    edge_traces = []
    arrow_annotations = []
    for edge in graph.edges:
        u, v = edge[:2]
        geom = graph.edges[edge].get("geometry")

        if geom is not None:
            x, y = geom.xy
            x = list(x)
            y = list(y)
            x0, y0 = geom.coords[-2]  # second last point
            x1, y1 = geom.coords[-1]  # last point
        else:
            # fallback if no geometry exists
            x = [positions[u][0], positions[v][0]]
            y = [positions[u][1], positions[v][1]]
            x0, y0 = positions[u]
            x1, y1 = positions[v]

        edge_traces.append(go.Scatter(
            x=x,
            y=y,
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
    for (edge), label in edge_labels.items():
        u, v = edge[:2]
        geom = graph.edges[edge].get("geometry")

        if geom is not None:
            midpoint = geom.interpolate(0.5, normalized=True)
            x_mid, y_mid = midpoint.x, midpoint.y
        else:
            # fallback to straight line midpoint
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


def visualize_geometry_point_in_folium_plot(m, geometry, size=25, color='black', label=''):
    point_x = geometry.coords.xy[0][0]
    point_y = geometry.coords.xy[1][0]
    folium.Circle((point_y, point_x), color=color, fill_color=color, radius=size, popup=label, tooltip=label).add_to(m)


def visualize_geometry_polygon_in_folium_plot(m, geometry, label = None):
    points_x = list(geometry.exterior.coords.xy[0])
    points_y = list(geometry.exterior.coords.xy[1])
    polyline = []
    for i, _ in enumerate(points_x):
        polyline.append((points_y[i], points_x[i]))
    if label is not None:
        folium.Polygon(polyline, tooltip=label, popup=label).add_to(m)
    else:
        folium.Polygon(polyline).add_to(m)


def visualize_node_in_folium_plot(m, graph, node, size=25, color='black', label=''):
    node_info = graph.nodes[node]
    point_x = node_info["geometry"].coords.xy[0][0]
    point_y = node_info["geometry"].coords.xy[1][0]
    folium.Circle((point_y, point_x), color=color, fill_color=color, radius=size, popup=label, tooltip=label).add_to(m)


def visualize_edge_in_folium_plot(m, graph, edge, color = 'violet', weight = 3,
                                  label = None, popup_width = 500, popup_height = 300):
    edge_info = graph.edges[edge]
    points_x = list(edge_info["geometry"].coords.xy[0])
    points_y = list(edge_info["geometry"].coords.xy[1])
    line = []
    for i, _ in enumerate(points_x):
        line.append((points_y[i], points_x[i]))

    else:
        popup = folium.Popup(width=popup_width, height=popup_height)
        if label is None:
            label = edge
        folium.PolyLine(line, weight=weight, color=color, tooltip=label, popup=label).add_to(m)


def plot_graph_folium(graph, longitude, latitude, zoom_start=5,
                     berths=None, turning_basins=None, anchorage_areas=None):

    m = folium.Map(location=[latitude, longitude],
                   zoom_start=zoom_start,
                   tiles="cartodbpositron")

    if isinstance(anchorage_areas, pd.DataFrame):
        plot_anchorage_areas(anchorage_areas, m=m)

    if isinstance(turning_basins, pd.DataFrame):
        plot_turning_basins(turning_basins, m=m)

    if isinstance(berths, pd.DataFrame):
        plot_berths(berths, m=m)

    edges = []
    graph_loop = graph.edges(data=True)
    multi_graph = False
    if isinstance(graph, nx.MultiGraph) or isinstance(graph, nx.MultiDiGraph):
        graph_loop = graph.edges(keys=True, data=True)
        multi_graph = True
    for edge_info in graph_loop:
        u, v = edge_info[:2]
        data = edge_info[-1]
        if multi_graph:
            k = edge_info[2]
        geom = data["geometry"]
        edge_info = {"geometry": data["geometry"], "u": u,"v": v}
        fields = ["u", "v"]
        if isinstance(graph, nx.MultiGraph) or isinstance(graph, nx.MultiDiGraph):
            edge_info['k'] = k
            fields = ["u", "v", "k"]
        edges.append(edge_info)

    gdf_edges = gpd.GeoDataFrame(edges,geometry="geometry", crs='EPSG:4326')

    folium.GeoJson(
        gdf_edges,
        style_function=lambda x: {"color": "violet", "weight": 2},
        popup=folium.GeoJsonPopup(fields=fields),
        tooltip=folium.GeoJsonTooltip(fields=fields),
    ).add_to(m)

    nodes = []
    for node, data in graph.nodes(data=True):
        nodes.append({
            "geometry": data["geometry"],
            "name": node
        })

    gdf_nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs='EPSG:4326')

    folium.GeoJson(
        gdf_nodes,
        marker=folium.CircleMarker(radius=3, color='purple', fill=True, fill_opacity=1.0, opacity=1.0),
        popup=folium.GeoJsonPopup(fields=["name"]),
        tooltip=folium.GeoJsonTooltip(fields=["name"]),
    ).add_to(m)

    return m


def create_real_world_graph(graph, lat_start=52.24, lon_start=5.75, zoom_start=6):
    # Create a map centered between the two points
    m = folium.Map(location=[lat_start, lon_start], zoom_start=zoom_start, tiles="cartodb positron")

    for edge in graph.edges(data=True):
        points_x = list(edge[2]["geometry"].coords.xy[0])
        points_y = list(edge[2]["geometry"].coords.xy[1])

        line = []
        for i, _ in enumerate(points_x):
            line.append((points_y[i], points_x[i]))

        folium.PolyLine(line, color="darkgrey", weight=3, popup=edge[2]["Name"]).add_to(m)

    for node in graph.nodes(data=True):
        point = list(node[1]["geometry"].coords.xy)
        folium.CircleMarker(location=[point[1][0], point[0][0]], color='grey', fill_color="grey", fill=True, radius=2,
                            popup=node[0]).add_to(m)

    return m
