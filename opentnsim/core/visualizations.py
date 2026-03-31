"""
Core utilities related to plotting.
"""

# %% IMPORT DEPENDENCIES
# generic
import pandas as pd
import plotly.express as px
from shapely.geometry import LineString, Point
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.offline import init_notebook_mode, iplot


def generate_vessel_gantt_chart(df_eventtable: pd.DataFrame, static: bool = False):

    df = df_eventtable.copy()
    object_order = df_eventtable["object name"].drop_duplicates().tolist()

    # Create segment id when main activity changes
    df["main_segment"] = (
        df.groupby("object id")["main activity name"]
        .apply(lambda s: (s != s.shift()).cumsum())
        .reset_index(level=0, drop=True)
    )

    # ---- MAIN BLOCKS ----
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

    df_main["activity label"] = (
        df_main["object name"] + " - " + df_main["main activity name"]
    )

    # ---- SUBACTIVITIES ----
    df_sub = df[
        df["subactivity name"].notna() &
        (df["subactivity name"] != "")
    ].copy()

    df_sub["activity label"] = (
        df_sub["object name"] + " - " + df_sub["subactivity name"]
    )

    # ---- COMBINE ----
    df_plot = pd.concat([df_main, df_sub], ignore_index=True)
    df_plot = df_plot.sort_values(["object name", "start time"])

    # ---- PLOT ----
    fig = px.timeline(
        df_plot,
        x_start="start time",
        x_end="stop time",
        y="activity label",
        color="object name",
        category_orders={"object name": object_order},
        title="Gantt chart of logged events",
    )

    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Activity (with Vessel)",
        legend_title="Vessel",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    if static is False:
        init_notebook_mode(connected=True)
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
