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

    # Add vessel name to activity label
    df_eventtable["activity label"] = (
        df_eventtable["object name"] + " - " + df_eventtable["activity name"]
    )

    # Create the Gantt chart
    fig = px.timeline(
        df_eventtable,
        x_start="start time",
        x_end="stop time",
        y="activity label",
        color="object name",
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
