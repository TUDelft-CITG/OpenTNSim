# -*- coding: utf-8 -*-

"""Graph module."""
# packkage(s) for documentation, debugging, saving and loading
import plotly.graph_objects as go
import folium
import pandas as pd

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

def plot_berths(berths, m = None, longitude = 0, latitude = 0, zoom_start = 5):
    if not m:
        m = folium.Map(location=[latitude, longitude], zoom_start=zoom_start, tiles="cartodbpositron")
    style = {'fillColor': '#000000', 'color': '#000000', 'opacity': 0}
    tooltip_berths = folium.GeoJsonTooltip(fields=["Port", "Company", "Berth"],
                                           aliases=["City:", "Port/Company:", "Berth:"],
                                           localize=True,
                                           sticky=False,
                                           labels=True,
                                           style="""
                                        background-color: #F0EFEF;
                                        border: 2px solid black;
                                        border-radius: 3px;
                                        box-shadow: 3px;
                                    """,
                                           max_width=800, )
    popup_berths = folium.GeoJsonPopup(
        fields=["Port", "Company", "Berth", "Berth_type", "Cargo_type", "Maximum_vessel_length",
                "Maximum_vessel_draught"],
        aliases=["City:", "Port/Company:", "Berth:", "Berth type:", "Cargo type:", "Maximum vessel length:",
                 "Maximum vessel draught"],
        localize=True,
        labels=True,
        style="""
                                        background-color: #F0EFEF;
                                        border: 2px solid black;
                                        border-radius: 3px;
                                        box-shadow: 3px;
                                    """,
        max_width=800, )
    folium.GeoJson(berths.set_crs('EPSG:4326'), style_function=lambda x: style, tooltip=tooltip_berths,
                   popup=popup_berths).add_to(m)

def plot_anchorage_areas(anchorage_areas, m = None, longitude = 0, latitude = 0, zoom_start = 5):
    if not m:
        m = folium.Map(location=[latitude, longitude], zoom_start=zoom_start, tiles="cartodbpositron")
    style_anchorage = {'fillColor': '#4CFF00', 'color': '#4CFF00', 'fillOpacity': 0, 'dashArray': '4'}
    tooltip_anchorage = folium.GeoJsonTooltip(fields=["Name", "Port"],
                                              aliases=["Name:", "Port:"],
                                              localize=True,
                                              sticky=False,
                                              labels=True,
                                              style="""
                                        background-color: #F0EFEF;
                                        border: 2px solid black;
                                        border-radius: 3px;
                                        box-shadow: 3px;
                                    """,
                                              max_width=800, )
    popup_anchorage = folium.GeoJsonPopup(fields=["Name", "Port", "Detail"],
                                          aliases=["Name:", "Port:", "Detail:"],
                                          localize=True,
                                          labels=True,
                                          style="""
                                        background-color: #F0EFEF;
                                        border: 2px solid black;
                                        border-radius: 3px;
                                        box-shadow: 3px;
                                    """,
                                          max_width=800, )
    folium.GeoJson(anchorage_areas.set_crs('EPSG:4326'), style_function=lambda x: style_anchorage,
                   tooltip=tooltip_anchorage, popup=popup_anchorage).add_to(m)

def plot_turning_basins(turning_basins, m = None, longitude = 0, latitude = 0, zoom_start = 5):
    if not m:
        m = folium.Map(location=[latitude, longitude], zoom_start=zoom_start, tiles="cartodbpositron")

    style_turning_basin = {'fillColor': '#FF0000', 'color': '#FF0000', 'fillOpacity':0,'dashArray':'4'}
    tooltip_turning_basin = folium.GeoJsonTooltip(fields=["Name", "Port"],
                                    aliases=["Name:","Port:"],
                                    localize=True,
                                    sticky=False,
                                    labels=True,
                                    style="""
                                        background-color: #F0EFEF;
                                        border: 2px solid black;
                                        border-radius: 3px;
                                        box-shadow: 3px;
                                    """,
                                    max_width=800,)
    popup_turning_basin = folium.GeoJsonPopup(fields=["Name", "Port", "Diameter"],
                                aliases=["Name:","Port:","Diameter:"],
                                localize=True,
                                labels=True,
                                style="""
                                        background-color: #F0EFEF;
                                        border: 2px solid black;
                                        border-radius: 3px;
                                        box-shadow: 3px;
                                    """,
                                max_width=800,)

    folium.GeoJson(turning_basins.set_crs('EPSG:4326'),style_function=lambda x:style_turning_basin,
                   tooltip=tooltip_turning_basin,popup=popup_turning_basin).add_to(m)


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
