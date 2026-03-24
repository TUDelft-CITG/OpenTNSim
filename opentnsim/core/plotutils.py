"""
Core utilities related to plotting.
"""

# %% IMPORT DEPENDENCIES
# generic
import pandas as pd
import plotly.express as px

from plotly.offline import init_notebook_mode, iplot


# %% GENERATE GANTT CHART FROM EVENT TABLE
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
