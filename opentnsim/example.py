import io
import datetime
import simpy

import requests
import numpy as np
import networkx as nx
import shapely.geometry
import geopandas as gpd
import pandas as pd
import scipy.stats

import ipyleaflet
import ipywidgets
import matplotlib.cm
import matplotlib.pyplot as plt

import opentnsim.core

# Dynamic object types, based on opentnsim.core
Anchorage = type(
    "Anchorage",
    (
        opentnsim.core.Identifiable,
        opentnsim.core.Log,
        opentnsim.core.Locatable,
        opentnsim.core.HasResource,
        opentnsim.core.ExtraMetadata,
    ),
    {},
)
Quay = type(
    "Quay",
    (
        opentnsim.core.Identifiable,
        opentnsim.core.Log,
        opentnsim.core.Locatable,
        opentnsim.core.HasContainer,
        opentnsim.core.ExtraMetadata,
    ),
    {},
)
Jetty = type(
    "Jetty",
    (
        opentnsim.core.Identifiable,
        opentnsim.core.Log,
        opentnsim.core.Locatable,
        opentnsim.core.HasResource,
        opentnsim.core.ExtraMetadata,
    ),
    {},
)
Vessel = type(
    "Vessel",
    (
        opentnsim.core.Identifiable,
        opentnsim.core.VesselProperties,
        opentnsim.core.Movable,
        opentnsim.core.ExtraMetadata,
    ),
    {},
)


def load_example_locations(graph, anchorage_size=30):
    rows = [
        # terminals have a capacity in quay meters
        {
            "n": "8863919",
            "e": ("8864185", "8863919"),
            "berth_type": "Quay",
            "type": "Container",
            "name": "Container1",
            "channel": "Europa Haven",
            "terminal": "APM",
            "quay_length": 1600,
        },
        {
            "n": "8865973",
            "e": ("8867633", "8865973"),
            "berth_type": "Quay",
            "type": "Container",
            "name": "Container2",
            "channel": "Amazonehaven",
            "terminal": "Hutchison Ports ECT Delta",
            "quay_length": 2000,
        },
        # {"n": "8866119", "e": ("8865518", "8866119"), "berth_type": "Jetty", "type": "Bulk", "name": "Agri Bulk", "terminal": "EBS", "quay_length": 220, "capacity": 4},
        {
            "n": "8868049",
            "e": ("8861217", "8868049"),
            "berth_type": "Quay",
            "type": "Bulk",
            "name": "Bulk",
            "channel": "Mississippi Haven",
            "terminal": "EMO",
            "quay_length": 1350,
        },
        # {"n": "8861397", "e": ("8860727", "8861397"), "berth_type": "Quay", "type": "Cargo", "name": "Cargo", "terminal": "Waalhaven Noordzijde", "quay_length": quay_length},
        {
            "n": "8868178",
            "e": ["8866999", "8868178"],
            "berth_type": "Jetty",
            "type": "Liquid",
            "name": "Liquid",
            "terminal": "Vopak Terminal Botlek",
            "channel": "3e Petroleumhaven",
            "capacity": 8,
        },
        # Liquid bbotlek
        # anchorage has a capacity in number of spots
        {
            "n": "8867050",
            "e": ("8867050", "8863654"),
            "berth_type": "Anchorage",
            "type": "Anchorage",
            "name": "Anchorage",
            "capacity": anchorage_size,
        },
        {
            "n": "8863061",
            "e": ("8863061", "13174263"),
            "berth_type": None,
            "type": "Origin",
            "name": "Origin",
        },
    ]
    geometry = [graph.edges[row["e"]]["geometry"] for row in rows]
    locations = gpd.GeoDataFrame(rows, geometry=geometry)
    # Index the rows by type
    locations = locations.set_index("name")

    origin = locations.loc["Origin"]
    anchorage = locations.loc["Anchorage"]

    # ad the routes from Origin to location.
    locations["route"] = locations.apply(
        lambda row: (
            # make sure we pass the anchorage area (slight detour)
            nx.shortest_path(graph, origin["n"], anchorage["n"], weight="length_m")
            + nx.shortest_path(graph, anchorage["n"], row["n"], weight="length_m")[1:]
        ),
        axis=1,
    )
    return locations


def plot_locations(graph, locations, center, zoom):
    """create a leaflet plot showing the graph and locations and the routes"""

    columns = [
        "terminal",
        "type",
        "berth_type",
        "channel",
        "quay_length",
        "capacity",
        "geometry",
    ]

    map = ipyleaflet.Map(
        basemap=ipyleaflet.basemaps.OpenStreetMap.BlackAndWhite,
        center=center,
        zoom=zoom,
    )

    for terminal, color in zip(
        ["Container1", "Container2", "Bulk", "Liquid"], matplotlib.cm.Set1.colors
    ):
        coords = []
        route = locations.loc[terminal]["route"]

        for n_a, n_b in zip(route[:-1], route[1:]):
            edge = graph.edges[(n_a, n_b)]
            coords_e = [latlon[::-1] for latlon in edge["geometry"].coords]
            if edge["StartJunctionId"] != n_a:
                # invert
                coords_e = coords_e[::-1]
            coords.extend(coords_e)

        ant_path = ipyleaflet.AntPath(
            locations=coords,
            dash_array=[1, 10],
            delay=2000,
            color=matplotlib.colors.rgb2hex(color) + "33",
            pulse_color=matplotlib.colors.rgb2hex(color) + "88",
        )
        map.add_layer(ant_path)

    # highlight
    locations_data = ipyleaflet.GeoData(
        geo_dataframe=locations[columns],
        style={
            "color": "#ccf",
            "opacity": 3,
            "weight": 4,
            "dashArray": "2",
            "fillOpacity": 0.6,
        },
        hover_style={"color": "yellow"},
        name="Terminals",
    )

    map.add_layer(locations_data)

    for i, row in locations.iterrows():
        marker = ipyleaflet.Marker(
            location=(row.geometry.centroid.y, row.geometry.centroid.x)
        )
        message = ipywidgets.HTML()
        # format series as html

        message.value = pd.DataFrame(row[columns])._repr_html_()
        popup = ipyleaflet.Popup(
            location=center,
            child=message,
            close_button=False,
            auto_close=False,
            close_on_escape_key=False,
        )
        marker.popup = popup
        map.add_layer(marker)

    return map


def add_locations_and_graph_to_env(env, locations, graph):
    """
    Make a copy of the graph and location and add the locations to the graph. Store the object in the locations.
    Use the environment (env) to instantiate the objects.
    """
    graph = graph.copy()
    locations = locations.copy()

    row = locations.loc["Anchorage"]
    anchorage = Anchorage(
        env=env,
        name="Anchorage",
        nr_resources=row["capacity"],
        geometry=row["geometry"],
    )
    locations.at["Anchorage", "Berth"] = anchorage
    graph.edges[row["e"]]["Anchorage"] = anchorage
    # add the quays
    for name, row in locations.loc[
        ["Container1", "Container2", "Bulk", "Liquid"]
    ].iterrows():
        #  Create a Quay with an initial available length of quay_length
        berth = None
        if row.berth_type == "Quay":
            berth = Quay(
                env=env,
                name=name,
                capacity=row["quay_length"],
                level=row["quay_length"],
                geometry=row["geometry"],
            )
        elif row.berth_type == "Jetty":
            berth = Jetty(
                env=env,
                name=name,
                nr_resources=row["capacity"],
                geometry=row["geometry"],
            )
        else:
            raise ValueError(row)
        # store as the more generic berth
        graph.edges[row["e"]]["Berth"] = berth
        locations.at[name, "Berth"] = berth
    return locations, graph


def vessel_sample(
    vessels, locations, graph, n_ships, arrival_rate=1, seed=42, env=None
):
    """Generate new vessels that arrive at the Origin area."""

    # sample and copy so that we have can do in place operations without interfering
    # with the original dataframe
    vessels = vessels.copy()
    sample = vessels.sample(n=n_ships, replace=True, random_state=seed)
    # initialize random number seed
    M = scipy.stats.expon(scale=1 / arrival_rate)
    sample["IAT"] = M.rvs(n_ships, random_state=seed)
    sample["AT"] = sample["IAT"].cumsum()

    # create vessel objects, needed for the simulation
    sample["vessel"] = sample.apply(
        row2vessel, axis=1, env=env, locations=locations, graph=graph
    )

    # compute a fresh index
    sample = sample.reset_index(drop=True).sort_values("AT")

    # Compute a unique id based on the index (row.name) and the name (row['name'])
    sample["id"] = sample.apply(lambda row: f"{row['name'].strip()}-{row.name}", axis=1)

    # use this id as a new index
    return sample.set_index("id")


def row2vessel(row, env, locations, graph):
    """convert a vessel row to a OpenTNSim object"""
    # lookup the start location
    origin = locations.loc["Origin"]

    geometry = graph.nodes[origin["n"]]["geometry"]
    row = row.copy(deep=True)
    row["geometry"] = geometry
    terminal = locations.loc[row["terminal"]]
    route = terminal["route"]
    vessel = Vessel(env=env, route=route, **row, unload_duration=3600 * 8)
    return vessel


# Define the vessel move process
# TODO: merge this with move process
def move(env, graph, vessel):
    """Let's move a vessel over the network."""

    # let's log what we are doing
    vessel.log_entry("Start sailing", env.now, "", vessel.geometry)

    route = vessel.route

    # check that we are at the start of our route
    current_location = vessel.geometry
    start_location = graph.nodes[route[0]]["geometry"]
    assert (
        current_location == start_location
    ), f"Ship is at {current_location}, not at {start_location}"

    anchorage_info = {}

    # loop over all node pairs in the route
    for i, e in enumerate(zip(route[:-1], route[1:])):
        # lookup the corresponding edge
        edge = graph.edges[e]
        # pass_anchorage
        if "Anchorage" in edge:
            # pass along the remaining track
            yield from opentnsim.edges.pass_anchorage(
                env, vessel, edge, graph=graph, info=anchorage_info, route=route[i:]
            )

            # No spot at the anchorage
            if anchorage_info["out_of_system"]:
                # We're done, we can return
                return
        # Now we actualy start sailing
        # vessel.log_entry("Passing edge", env.now, "", edge['geometry'])

        # compute how long it takes
        duration = edge["length_m"] / vessel.v
        # move vessel to new location
        vessel.geometry = graph.nodes[e[-1]]["geometry"]
        # progress time
        yield vessel.env.timeout(duration)

        # pass_berth
        if "Berth" in edge:
            yield from opentnsim.edges.pass_berth(
                env, vessel, edge, anchorage_info=anchorage_info
            )

    vessel.log_entry("Stop sailing", env.now, "", vessel.geometry)


def run(env, sample, graph):
    for i, row in sample.iterrows():
        yield env.timeout(row["vessel"].metadata["IAT"])
        env.process(opentnsim.example.move(env, graph, row["vessel"]))


def locations_plot(results):
    """Plot the results of a simulation"""

    locations = results["locations"]
    fig, axes = plt.subplots(nrows=5, figsize=(8, 10), sharex=True)

    ax = axes[0]
    berth = locations.loc["Anchorage"].Berth
    log = pd.DataFrame(berth.log)
    log["count"] = log["Value"].apply(lambda x: x["count"])
    ax.plot(log["Timestamp"], log["count"])
    ax.set_title(f'Anchorage, {results["moved_out"]} ships not handled')
    ax.set_ylabel("Spots in use")

    xlim = ax.get_xlim()
    ax.hlines(berth.resource.capacity, *xlim, "r")
    ax.hlines(0, *ax.get_xlim(), "g")
    ax.set_xlim(xlim)

    for ax, (name, row) in zip(
        axes[1:], locations.loc[["Container1", "Container2", "Bulk"]].iterrows()
    ):
        berth = row["Berth"]
        log = pd.DataFrame(berth.log)
        log["level"] = log["Value"].apply(lambda x: x["level"])
        ax.plot(log["Timestamp"], log["level"])
        ax.set_title(name)
        ax.set_ylabel("Length available")
        xlim = ax.get_xlim()
        ax.hlines(berth.container.capacity, *xlim, "g")
        ax.hlines(0, *ax.get_xlim(), "r")
        ax.set_xlim(xlim)

    ax = axes[4]
    berth = locations.loc["Liquid"].Berth
    log = pd.DataFrame(berth.log)
    log["count"] = log["Value"].apply(lambda x: x["count"])
    ax.plot(log["Timestamp"], log["count"])
    ax.set_title("Liquid")
    ax.set_ylabel("Spots in use")

    xlim = ax.get_xlim()
    ax.hlines(berth.resource.capacity, *xlim, "r")
    ax.hlines(0, *ax.get_xlim(), "g")
    ax.set_xlim(xlim)

    inter_arrival_period = (1 / results['arrival_rate']) / 3600

    title = f'OpenTNSim results, $n_{{ships}}$ {results["n_ships"]}, arrival rate {results["arrival_rate"]}, inter arrival period: {inter_arrival_period:.2}h, duration: {results["duration"]}'
    fig.suptitle(title, fontsize=16)

    return fig, ax


def simulate(locations, vessels, graph, arrival_rate, n_ships, seed=42):
    # examples: Give 4 suggestions for improvement
    simulation_start = datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc)
    env = simpy.Environment(initial_time=simulation_start.timestamp())

    # Add the locations to the graph and create Berth and Anchorage objects.
    # Location now has an extra property Berth
    # Edges can now have an extra attribute Berth or Anchorage
    locations, graph = opentnsim.example.add_locations_and_graph_to_env(
        env, locations, graph
    )
    sample = opentnsim.example.vessel_sample(
        vessels,
        locations=locations,
        graph=graph,
        n_ships=n_ships,
        arrival_rate=arrival_rate,
        seed=seed,
        env=env,
    )
    # wait for arrival and move across the network
    env.process(opentnsim.example.run(env, sample=sample, graph=graph))
    # run the simulation
    env.run()

    # compute the end date
    simulation_end = datetime.datetime.fromtimestamp(env.now, tz=datetime.timezone.utc)

    moved_out = (
        sample.vessel.apply(lambda vessel: pd.DataFrame(vessel.log).iloc[-1].Message)
        .str.startswith("Moving out")
        .sum()
    )

    results = {
        "duration": simulation_end - simulation_start,
        "locations": locations,
        "sample": sample,
        "moved_out": moved_out,
        "n_ships": n_ships,
        "arrival_rate": arrival_rate,
    }
    return results
