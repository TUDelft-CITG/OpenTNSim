"""
Core utilities related to plotting.
"""

# packages for time, space and id
import datetime
from shapely.geometry import LineString, Point

# packages for data handling
import numpy as np
import pandas as pd

# plotting libraries
import matplotlib.pyplot as plt
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot

# spatial libraries
import networkx as nx
from simplekml import Kml, Style


def create_and_plot_vessel_activities(vessels, activities, colors):
    """create a plot of the planning of vessels

    Parameters
    ----------
    vessels : list of objects
        A list containing vessel objects to plot. vessel need to have a logbook (mixin class Log)
    activities : list of strings
        The activities to plot. Must be denoted in logbook as '<activity> start' and '<activity> stop'
    colors : list of colors
        The colors to use for the activities. must be the same length as activities
    """

    def get_segments(series, activity, y_val):
        """extract 'start' and 'stop' of activities from log"""
        x = []
        y = []
        for i, v in series.iteritems():
            if v == activity + " start":
                start = i
            if v == activity + " stop":
                x.extend((start, start, i, i, i))
                y.extend((y_val, y_val, y_val, y_val, None))
        return x, y

    # organise logdata into 'dataframes'
    dataframes = []
    for vessel in vessels:
        rename = {"Value": "log_value", "Message": "log_string"}
        df = pd.DataFrame(vessel.logbook)
        df = df.set_index("Timestamp").rename(columns=rename)

        dataframes.append(df)
    df = dataframes[0]

    # prepare traces for each of the activities
    traces = []
    for i, activity in enumerate(activities):
        x_combined = []
        y_combined = []
        for k, df in enumerate(dataframes):
            y_val = vessels[k].name
            x, y = get_segments(df["log_string"], activity=activity, y_val=y_val)
            x_combined.extend(x)
            y_combined.extend(y)
        traces.append(
            go.Scatter(
                name=activity,
                x=x_combined,
                y=y_combined,
                mode="lines",
                hoverinfo="y+name",
                line=dict(color=colors[i], width=10),
                connectgaps=False,
            )
        )

    # prepare layout of figure
    layout = go.Layout(
        title="Vessel planning",
        hovermode="closest",
        legend=dict(x=0, y=-0.2, orientation="h"),
        xaxis=dict(
            title="Time",
            titlefont=dict(family="Courier New, monospace", size=18, color="#7f7f7f"),
            range=[0, df.index[-1]],
        ),
        yaxis=dict(
            title="Vessels",
            titlefont=dict(family="Courier New, monospace", size=18, color="#7f7f7f"),
        ),
    )

    # plot figure
    init_notebook_mode(connected=True)
    fig = go.Figure(data=traces, layout=layout)
    return iplot(fig, filename="news-source")


def create_vessels_kml(
    vessels,
    fname="vessel_movements.kml",
    icon="http://maps.google.com/mapfiles/kml/shapes/sailing.png",
    size=1,
    scale=1,
    stepsize=120,
):
    """Create a kml visualisation of vessels and save in defined file.

    Parameters
    ----------
    vessels: list of vessels
        Vessels need logs that contain geometries in lat, lon as a function of time. (mixin class Log)
    fname: str
        The name of the kml file to be created. Default is 'vessel_movements.kml'
    icon: str
        The icon to be used for the vessels. Default is 'http://maps.google.com/mapfiles/kml/shapes/sailing.png'
    size: float
        The size of the label. Default is 1
    scale: float
        The scale of the icon. Default is 1
    stepsize: int
        The stepsize for the interpolation of the geometry. Default is 120 seconds (2 minutes)
    """

    # create a kml file containing the visualisation
    kml = Kml()
    fol = kml.newfolder(name="Vessels")

    shared_style = Style()
    shared_style.labelstyle.color = "ffffffff"  # White
    shared_style.labelstyle.scale = size
    shared_style.iconstyle.color = "ffff0000"  # Blue
    shared_style.iconstyle.scale = scale
    shared_style.iconstyle.icon.href = icon

    # each timestep will be represented as a single point
    # todo: create a tmpvessel to log info. Do not attach it to the original vessel log!
    for vessel in vessels:
        tmp_vessel = {
            "Geometry - x": [],
            "Geometry - y": [],
            "timestamps_t": [],
            "timestamps_x": [],
        }
        geom_x = []
        geom_y = []

        vessel_log = pd.DataFrame(vessel.logbook)

        for geom in vessel_log["Geometry"]:
            geom_x.append(geom.x)
            geom_y.append(geom.y)

        tmp_vessel["Geometry - x"] = geom_x
        tmp_vessel["Geometry - y"] = geom_y

        time_stamp_min = min(vessel_log["Timestamp"]).timestamp()
        time_stamp_max = max(vessel_log["Timestamp"]).timestamp()

        steps = int(np.floor((time_stamp_max - time_stamp_min) / stepsize))
        timestamps_t = np.linspace(time_stamp_min, time_stamp_max, steps)

        times = []
        for row in vessel.logbook:
            t = row["Timestamp"]
            times.append(t.timestamp())

        tmp_vessel["timestamps_t"] = timestamps_t
        tmp_vessel["timestamps_x"] = np.interp(timestamps_t, times, tmp_vessel["Geometry - x"])
        tmp_vessel["timestamps_y"] = np.interp(timestamps_t, times, tmp_vessel["Geometry - y"])

        log_index = -1
        for log_index, value in enumerate(tmp_vessel["timestamps_t"][:-1]):
            begin = datetime.datetime.fromtimestamp(tmp_vessel["timestamps_t"][log_index])
            end = datetime.datetime.fromtimestamp(tmp_vessel["timestamps_t"][log_index + 1])

            pnt = fol.newpoint(
                name=vessel.name,
                coords=[
                    (
                        tmp_vessel["timestamps_x"][log_index],
                        tmp_vessel["timestamps_y"][log_index],
                    )
                ],
            )
            pnt.timespan.begin = begin.isoformat()
            pnt.timespan.end = end.isoformat()
            pnt.style = shared_style

        # include last point as well
        begin = datetime.datetime.fromtimestamp(tmp_vessel["timestamps_t"][log_index + 1])
        # end = datetime.datetime.fromtimestamp(vessel.log["timestamps_t"][log_index + 1])

        pnt = fol.newpoint(
            name=vessel.name,
            coords=[
                (
                    tmp_vessel["timestamps_x"][log_index + 1],
                    tmp_vessel["timestamps_y"][log_index + 1],
                )
            ],
        )
        pnt.timespan.begin = begin.isoformat()
        # pnt.timespan.end = end.isoformat()
        pnt.style = shared_style

    kml.save(fname)


def create_site_kml(
    sites,
    fname="site_development.kml",
    icon="http://maps.google.com/mapfiles/kml/shapes/square.png",
    size=1,
    scale=3,
):
    """Create a kml visualisation of vessels.

    Parameters
    ----------
    env : simpy.Environment
        The environment object containing the simulation time. Env variable needs to contain epoch to enable conversion of
        the simulation time to real time.
    sites: list of sites
        Sites need logs that contain geometries in lat, lon as a function of time. (mixin class Log)
    fname: str
        The name of the kml file to be created. Default is 'site_development.kml'
    icon: str
        The icon to be used for the vessels. Default is 'http://maps.google.com/mapfiles/kml/shapes/square.png'
    size: float
        The size of the label. Default is 1
    scale: float
        The scale of the icon. Default is 1
    """

    # create a kml file containing the visualisation
    kml = Kml()
    fol = kml.newfolder(name="Sites")

    # each timestep will be represented as a single point
    for site in sites:
        for log_index, value in enumerate(site.log["Timestamp"][:-1]):
            style = Style()
            style.labelstyle.color = "ffffffff"  # White
            style.labelstyle.scale = size
            style.iconstyle.color = "ff00ffff"  # Yellow
            style.iconstyle.scale = scale * (site.log["Value"][log_index] / site.container.capacity)
            style.iconstyle.icon.href = icon

            begin = site.log["Timestamp"][log_index]
            end = site.log["Timestamp"][log_index + 1]

            pnt = fol.newpoint(
                name=site.name,
                coords=[
                    (
                        site.log["Geometry"][log_index].x,
                        site.log["Geometry"][log_index].y,
                    )
                ],
            )
            pnt.timespan.begin = begin.isoformat()
            pnt.timespan.end = end.isoformat()
            pnt.style = style

        # include last point as well
        style = Style()
        style.labelstyle.color = "ffffffff"  # White
        style.labelstyle.scale = 1
        style.iconstyle.color = "ff00ffff"  # Yellow
        style.iconstyle.scale = scale * (site.log["Value"][log_index + 1] / site.container.capacity)
        style.iconstyle.icon.href = icon

        begin = site.log["Timestamp"][log_index + 1]
        # end = site.log["Timestamp"][log_index + 1]

        pnt = fol.newpoint(
            name=site.name,
            coords=[
                (
                    site.log["Geometry"][log_index + 1].x,
                    site.log["Geometry"][log_index + 1].y,
                )
            ],
        )
        pnt.timespan.begin = begin.isoformat()
        # pnt.timespan.end = end.isoformat()
        pnt.style = style

    kml.save(fname)


def create_graph_kml(
    env,
    fname="graph.kml",
    icon="http://maps.google.com/mapfiles/kml/shapes/donut.png",
    size=0.5,
    scale=0.5,
    width=5,
):
    """Create a kml visualisation of graph and save in defined file.

    Parameters
    ----------
    env : simpy.Environment
        The environment object containing the simulation time. Env variable needs to contain the graph.
    fname: str
        The name of the kml file to be created. Default is 'graph.kml'
    icon: str
        The icon to be used for the vessels. Default is 'http://maps.google.com/mapfiles/kml/shapes/donut.png'
    size: float
        The size of the label. Default is 0.5
    scale: float
        The scale of the icon. Default is 0.5
    width: float
        The width of the line. Default is 5
    """

    # create a kml file containing the visualisation
    kml = Kml()
    fol = kml.newfolder(name="Vessels")

    shared_style = Style()
    shared_style.labelstyle.color = "ffffffff"  # White
    shared_style.labelstyle.scale = size
    shared_style.iconstyle.color = "ffffffff"  # White
    shared_style.iconstyle.scale = scale
    shared_style.iconstyle.icon.href = icon
    shared_style.linestyle.color = "ff0055ff"  # Red
    shared_style.linestyle.width = width

    nodes = list(env.graph.nodes)

    # each timestep will be represented as a single point
    for log_index, value in enumerate(list(env.graph.nodes)[0 : -1 - 1]):
        pnt = fol.newpoint(
            name="",
            coords=[
                (
                    nx.get_node_attributes(env.graph, "geometry")[nodes[log_index]].x,
                    nx.get_node_attributes(env.graph, "geometry")[nodes[log_index]].y,
                )
            ],
        )
        pnt.style = shared_style

    edges = list(env.graph.edges)
    for log_index, value in enumerate(list(env.graph.edges)[0 : -1 - 1]):
        lne = fol.newlinestring(
            name="",
            coords=[
                (
                    nx.get_node_attributes(env.graph, "geometry")[edges[log_index][0]].x,
                    nx.get_node_attributes(env.graph, "geometry")[edges[log_index][0]].y,
                ),
                (
                    nx.get_node_attributes(env.graph, "geometry")[edges[log_index][1]].x,
                    nx.get_node_attributes(env.graph, "geometry")[edges[log_index][1]].y,
                ),
            ],
        )
        lne.style = shared_style

    kml.save(fname)


def generate_vessel_gantt_chart(df_eventtable: pd.DataFrame, static: bool = False):
    """
    Method to generate a Gantt chart from a vessel activity log DataFrame.

    This method visualizes the activity timeline of vessels by combining vessel
    names and activity types into a single label, and plotting them using Plotly
    Express's timeline chart.

    Parameters
    ----------
    df_eventtable : pandas.DataFrame
        DataFrame containing columns 'object name', 'activity name', 'start time',
        and 'stop time' representing vessel activity logs.
    static : bool, optional
        If True, returns a static Plotly figure object.
        If False, displays the figure

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A Plotly figure object representing the Gantt chart of vessel activities.
    """

    df = df_eventtable.copy()
    object_order = df_eventtable["object name"].drop_duplicates().tolist()

    # Create segment id when main activity changes
    df["main_segment"] = (
        df.groupby("object id")["main activity name"]
        .apply(lambda s: (s != s.shift()).cumsum())
        .reset_index(level=0, drop=True)
    )

    # Create a dataframe for main activities by taking the min start time and max stop time for each segment of main activity per vessel
    df_main = (
        df.groupby(
            ["object id", "object name", "main activity name", "main_segment"],
            as_index=False
        )
        .agg(
            {
                "start time": "min",
                "stop time": "max"
            }
        )
    )

    # Add vessel name to activity label
    df_main["activity label"] = (
        df_main["object name"] + " - " + df_main["main activity name"]
    )

    # Create dataframe for sub-activities, only if they exist, and add vessel name to activity label
    df_sub = df[
        df["subactivity name"].notna() &
        (df["subactivity name"] != "")
    ].copy()

    df_sub["activity label"] = (
        df_sub["object name"] + " - " + df_sub["subactivity name"]
    )

    # Combine main and sub activities for plotting
    df_plot = pd.concat([df_main, df_sub], ignore_index=True)
    df_plot = df_plot.sort_values(["object name", "start time"])

    # Create the Gantt chart
    fig = px.timeline(
        df_plot,
        x_start="start time",
        x_end="stop time",
        y="activity label",
        color="object name",
        category_orders={"object name": object_order},
        title="Gantt chart of logged events",
    )

    # Reverse the Y-axis to match Gantt chart style
    fig.update_yaxes(autorange="reversed")

    # Customize layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Activity (with Vessel)",
        legend_title="Vessel",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    if static is False:
        # Initialize notebook mode for Plotly
        init_notebook_mode(connected=True)
        # Display the figure in a Jupyter notebook
        iplot(fig)
    else:
        return fig


def plot_gdf_eventtable(gdf, attr=None, colorscale="Reds", padding=0.05):
    traces = []
    edges_gdf = gdf[gdf.geometry.apply(lambda x: isinstance(x, LineString))]
    nodes_gdf = gdf[gdf.geometry.apply(lambda x: isinstance(x, Point))]

    # --- edges ---
    if attr is not None and attr in edges_gdf.columns:
        edge_vals = edges_gdf[attr].fillna(0).values
        e_min, e_max = edge_vals.min(), edge_vals.max()
        edge_norm = (edge_vals - e_min) / (e_max - e_min + 1e-9)
        edge_colors = [pc.sample_colorscale(pc.get_colorscale(colorscale), nv)[0] for nv in edge_norm]
    else:
        edge_colors = ["red"] * len(edges_gdf)

    for geom, val, color in zip(edges_gdf.geometry, edges_gdf[attr] if attr in edges_gdf.columns else [None]*len(edges_gdf), edge_colors):
        x, y = zip(*geom.coords)
        hover_text = f"{attr}: {val}" if attr else ""
        traces.append(go.Scatter(x=x, y=y, mode="lines",
                                 line=dict(color=color, width=3),
                                 hovertext=hover_text))

    # --- nodes ---
    if not nodes_gdf.empty:
        if attr is not None and attr in nodes_gdf.columns:
            node_vals = nodes_gdf[attr].fillna(0).values
            n_min, n_max = node_vals.min(), node_vals.max()
            node_norm = (node_vals - n_min) / (n_max - n_min + 1e-9)
            node_colors = [pc.sample_colorscale(pc.get_colorscale(colorscale), nv)[0] for nv in node_norm]
        else:
            node_colors = ["blue"] * len(nodes_gdf)

        for geom, val, color in zip(nodes_gdf.geometry, nodes_gdf[attr] if attr in nodes_gdf.columns else [None]*len(nodes_gdf), node_colors):
            hover_text = f"{attr}: {val}" if attr else ""
            traces.append(go.Scatter(x=[geom.x], y=[geom.y], mode="markers",
                                     marker=dict(size=12, color=color),
                                     hovertext=hover_text))

    # --- compute axis limits with padding ---
    all_x, all_y = [], []
    for geom in gdf.geometry:
        if isinstance(geom, Point):
            all_x.append(geom.x)
            all_y.append(geom.y)
        elif isinstance(geom, LineString):
            xs, ys = zip(*geom.coords)
            all_x.extend(xs)
            all_y.extend(ys)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # add padding
    x_pad = (x_max - x_min) * padding if x_max != x_min else 1
    y_pad = (y_max - y_min) * padding if y_max != y_min else 1
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]

    # --- figure ---
    fig = go.Figure(traces)
    fig.update_layout(
        width=800,
        height=400,
        title=f"Directed Geographic Network Graph ({attr or 'Edges'})",
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=False, zeroline=False, range=x_range,
            title="Longitude", showticklabels=False
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, range=y_range,
            title="Latitude", showticklabels=False
        ),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig
