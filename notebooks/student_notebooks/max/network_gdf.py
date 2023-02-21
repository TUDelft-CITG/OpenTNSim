#!/usr/bin/env python3
import math
import datetime
import time

import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
import shapely.geometry
import matplotlib.pyplot as plt

import simpy
import opentnsim.core as core
import pyproj
import pickle

def create_shape():
    """ Use this function to create the shape of
    the desired network in a (x,y) coordinate system. 
    Here, we'll make a 12 sided polygon. 
    Function returns coordinates in (x,y) meters"""
    
    # Define number of nodes
    n_nodes = 12

    # Set up 12 values from 0 - 2pi for the circle shape and define radius
    radians = np.linspace(0, np.pi * 2, num = n_nodes, endpoint=False)
    radius = 8 # meters

    # Create coordinates of each node
    x = radius*np.cos(radians)
    y = radius*np.sin(radians)

    return x, y

def RD_coordinates():
    """Translate the (x,y) coordinates of the shape into RD-coordinates.
    RD-coordinates are chosen because it is also an cartesian system.
    Set the origin of your (x,y) system equal to a desired origin in 
    the RD-system and find the coordinates
    Function returns coordinates in RD-coords"""

    #The origin in (x,y) is the middle of the circle so determine where middle of the circle will be in RD
    origin_RD = (85279, 446400)
    origin_xy = (0, 0)
    
    # Determine difference of origins for translation

    x_trans, y_trans = origin_RD[0]-origin_xy[0], origin_RD[1] -origin_xy[1]

    # Translate
    x_RD = create_shape()[0] + x_trans
    y_RD = create_shape()[1] + y_trans
    
    return x_RD, y_RD


def nodes_edges():
    """Create nodes and edges from the RD-coordinates"""

    edges = []
    nodes = []

    x, y = create_shape()
    x_RD, y_RD = RD_coordinates()

    # Loop over all the nodes
    for i, (x_i, y_i, x_RD_i, y_RD_i) in enumerate(zip(x, y, x_RD, y_RD)):
        start = i
        # Are we at the end node?
        # Then we need to close the loop
        if i == (len(x)-1):
            end = 0
        else:
            end = i + 1

        # and edges consists of a start node and an end node
        edge = {
            "source": start,
            "target": end,
            "e": (start, end),
        }
        node = {
            "x": x_i,
            "y": y_i,
            "n": i,
            "x_RD": x_RD_i,
            "y_RD":y_RD_i,
            "geometry": shapely.geometry.Point(x_RD_i, y_RD_i),
        }
        edges.append(edge)
        nodes.append(node)

    return edges, nodes

def geoDataFrames():
    """Creates a geoDataFrame from the edges and nodes and translates coordinates to lat lon
    return two geoDataframes of nodes and edges"""
    edges, nodes = nodes_edges()
    
    # Create normal DataFrame first
    edges_df = pd.DataFrame(edges)
    nodes_df = pd.DataFrame(nodes)

    # create two temporary columns with the start end end geometry
    edges_df['source_geometry'] = nodes_df['geometry'][edges_df['source']].reset_index(drop=True)
    edges_df['target_geometry'] = nodes_df['geometry'][edges_df['target']].reset_index(drop=True)
    # for each edge combine start and end geometry into a linestring (line)
    edges_df['geometry'] = edges_df.apply(
        lambda row: shapely.geometry.LineString([row['source_geometry'], row['target_geometry']]), 
        axis=1
    )
    edges_df = edges_df.drop(columns=['source_geometry', 'target_geometry'])

    nodes_gdf = gpd.GeoDataFrame(nodes_df)
    edges_gdf = gpd.GeoDataFrame(edges_df)

    #Used RD-coordinates (epsg:28992) to define nodes so set this for geopandas to know
    nodes_gdf.crs = 'epsg:28992'
    edges_gdf.crs = 'epsg:28992'

    #Change to lat lon (epsg:4326)
    nodes_gdf = nodes_gdf.to_crs('epsg:4326')
    edges_gdf = edges_gdf.to_crs('epsg:4326')
    
    return edges_gdf, nodes_gdf

def network_FG():
    """Create a FG (Fairway Graph) from geoDataFrames"""

    edges_gdf, nodes_gdf = geoDataFrames()

    FG = nx.from_pandas_edgelist(edges_gdf, edge_attr=True)
    for e, edge in FG.edges.items():
        # make all edges 6m deep
        edge['Info'] = {"GeneralDepth": 6}
    
    # Update all nodes with info from the nodes table
    nodes_gdf.apply(lambda row: FG.nodes[row.n].update(row), axis= 1)

    # Optional: create an easy-to-export (bniary) file of FG and save in cd
    """with open('FG_pond.gpickle', 'wb') as f:
        pickle.dump(Fg, f)"""

    return FG









    
    

    
    





