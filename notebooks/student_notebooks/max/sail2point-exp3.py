#!/usr/bin/env python3
import pickle
import math
import datetime
import time

import rospy

import pyproj

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
import shapely.geometry
import pandas as pd
import matplotlib.pyplot as plt
import opentnsim.core
import opentnsim.energy
import simpy

from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix

from network_gdf import network_FG, load_experiment3
import tactics 

FG = network_FG(as_numbers=False)

STRATEGY = 'strategy_duration'  #Choose kpi on which you want to sort 'strategy_duration', 'strategy_fuel', 'strategy_mca'

WAYPOINT_THRESHOLD = 3 #meters

def distance(lon1, lat1, lon2, lat2):
        wgs84 = pyproj.Geod(ellps="WGS84")
        return wgs84.inv(
                lon1,
                lat1,
                lon2,
                lat2, radians=False)[2]

def heading(lon1, lat1, lon2, lat2):
        wgs84 = pyproj.Geod(ellps="WGS84")
        heading = wgs84.inv(
                lon1,
                lat1,
                lon2,
                lat2, radians=False)[1]
        heading_rad = heading/180*math.pi
        heading_0_2pi = np.mod(heading_rad, np.pi * 2)
        return heading_0_2pi

def run_simulation(geometry, route, graph, engine_order=0.8):
    Vessel = type(
        'Vessel', 
        (
            opentnsim.core.Identifiable, 
            opentnsim.core.Movable, 
            opentnsim.core.Routeable, 
            opentnsim.core.VesselProperties,
            opentnsim.energy.ConsumesEnergy,
            opentnsim.core.ExtraMetadata
        ), 
        {}
    )

    max_v = 5.0
    P_installed = 1750
    
    data_vessel = {
        "env": None,
        "name": "NausBot",
        "route": None,
        "geometry": geometry,
        "type": "Va",
        "B": 11.4,
        "L": 110,
        'P_installed': P_installed, 
        'L_w': 3, 
        "T": 3.5,
        'C_year': 1997,
        "v": engine_order * max_v
    
    } 
    vessel = Vessel(**data_vessel)
    
        

    # start simpy environment (specify the start time and add the graph to the environment)
    simulation_start = datetime.datetime.now()
    env = simpy.Environment(initial_time = time.mktime(simulation_start.timetuple()))
    env.FG = graph

    # add environment to the vessel, and specify the vessels route and current location (beginning of the path)
    vessel.env = env
    vessel.route = route
    
    # specify the process that needs to be executed
    env.process(vessel.move())

    # start the simulation
    env.run()

    #Determine trip time
    start_time = simulation_start.timestamp()
        

    end = vessel.log["Timestamp"][-1]

    end_time = end.timestamp()

    energycalculation = opentnsim.energy.EnergyCalculation(graph, vessel)
    energycalculation.calculate_energy_consumption()

    # create dataframe from energy calculation computation
    energy_df = pd.DataFrame.from_dict(energycalculation.energy_use)

    return end_time-start_time, end, energy_df, vessel



        
class Operator():
    def __init__(self, id, controlled_vessel, observed_vessel, alternatives_df):
        self.id = id
        self.controlled_vessel = controlled_vessel
        self.observed_vessel = observed_vessel
        self.kpi_df = None
        self.alternatives_df = alternatives_df
        
        self.berth_loc =  [4.3719054436847, 52.00161375312554]

        self.pub_route = rospy.Publisher(f"/{controlled_vessel.id}/route", String, queue_size=10)
        
        self.sub_observed_vessel = rospy.Subscriber(f"/{self.observed_vessel.id}/state/geopos", NavSatFix, callback = self.check_berth_availability)
        self.berth_available = True 

        #rospy.Timer(rospy.Duration(5), self.publish_route)
        rospy.Timer(rospy.Duration(5), self.compute_kpi)
        #self.publish_tactic()

    def check_berth_availability(self, msg: NavSatFix()):
    
        dist = distance(self.berth_loc[0], self.berth_loc[1], msg.longitude, msg.latitude)
        print(dist)
        # Add conditional statement that involves the berth availability: 'if pos is within polygon'
        if dist <= 1:
            self.berth_available = False
        else:
            self.berth_available = True
        print("BERTH AVAILABILITY:", self.berth_available)

    def compute_kpi(self, event):
        print('computing kpi')
        geometry = self.controlled_vessel.geometry
        # remaining waypoints for vessel
        waypoints = self.controlled_vessel.waypoints 
        graph = FG
        self.kpi_df = tactics.add_kpi(
            alternatives_df=self.alternatives_df,
            berth_available=self.berth_available, 
            graph=FG, 
            geometry=geometry, 
            visited_nodes=self.controlled_vessel.visited_nodes, 
            waypoints=waypoints
        )
        self.kpi_df.sort_values(STRATEGY, inplace=True)
        time_remaining = self.kpi_df.loc[:,'duration'].iloc[0]
        print(self.kpi_df)
        print(f'Duration of alternative with least emissions = {time_remaining} seconds')
        self.publish_route()
        self.publish_velocity()

    def publish_velocity(self):
        engine_order = self.kpi_df.loc[:, 'engine_order'].iloc[0]
        print('Engine order = ', engine_order)
        self.controlled_vessel.engine_order = engine_order
        self.controlled_vessel.publish_velocity()

    def publish_route(self):
        msg = String()
        route = self.kpi_df.loc[:, 'route'].iloc[0]
        if 'remaining_route' in self.kpi_df.columns:
            route = self.kpi_df.loc[:, 'remaining_route'].iloc[0]
        msg.data =str(route)
        self.pub_route.publish(msg)
        self.controlled_vessel.route = route


class Vessel():
    def __init__(self, id, route=None, waypoints=None):
        self.id = id
        # an ordered set of nodes that we must pass
        self.waypoints = waypoints
        # an ordered set of nodes that we are told to sail through
        self.route = route  

        self.visited_waypoints = []
        self.visited_nodes = []
        self.pos = NavSatFix()
        self.sub = rospy.Subscriber(f"/{self.id}/state/geopos", NavSatFix, callback=self.update_pos)
        self.pub_currentwaypoint = rospy.Publisher(f"/{self.id}/next_node", NavSatFix, queue_size=10)
        self.pub_headingref = rospy.Publisher(f"/{self.id}/reference/yaw", Float32, queue_size=10)

        self.engine_order = 0.8 
        self.max_velocity = 0.5
        
        self.pub_velocity = rospy.Publisher(f"/{self.id}/reference/velocity", Float32MultiArray, queue_size=10)   
        self.current_waypoint = self.waypoints[0] if self.waypoints else None
        rospy.sleep(0.1)
        self.publish_velocity()
        
    @property
    def next_n(self):
        if not self.route:
            return None
        return self.route[0]

    @property
    def next_node_geometry(self):
        next_n = self.next_n
        if next_n is None:
            return None
        next_node = FG.nodes[next_n]
        next_node_geometry = next_node['geometry']
        return next_node_geometry

    @property
    def velocity(self):
        return self.engine_order * self.max_velocity
    
    def publish_velocity(self):
        msg = Float32MultiArray()
        msg.data = [self.velocity, 0, 0]
        self.pub_velocity.publish(msg)
        

    def update_pos(self, msg: NavSatFix()):
        self.pos = msg
        rospy.loginfo(f'Vessel : {self.id}, Lat {self.pos.latitude},Lon {self.pos.longitude}')
        self.update_route()
        self.publish_current_waypoint()
        self.publish_heading_ref()
        # self.visited

    def update_route(self):
        """check if we are at a node, pop the node from the route and the waypoints"""
        if self.route is None:
            return
        if len(self.route)<1:
            return
        geometry = self.geometry
        dist_to_next_node = distance(self.next_node_geometry.x, self.next_node_geometry.y, geometry.x, geometry.y)
        print(f'Distance to next node {dist_to_next_node}')
        if dist_to_next_node < WAYPOINT_THRESHOLD:
            # remove the first node from the route and remember so that we can
            # check if it was also a waypoint
            visited_node = self.route.pop(0)
            self.visited_nodes.append(visited_node)
            rospy.loginfo(f' Visited node: {visited_node}')
            # safeguards for empty waypoints list
            if self.waypoints is None:
                return
            if len(self.waypoints) < 1:
                return
            if visited_node == self.waypoints[0]:
                visited_waypoint = self.waypoints.pop(0)
                self.visited_waypoints.append(visited_waypoint)

    def publish_current_waypoint(self):
        """Publish current waypoint (as understood by ras, it is the next node geometry) to a topic"""
        if self.next_node_geometry is None:
            rospy.loginfo(f'Vessel {self.id} does not have a next node geometry')
            return 
        #Define msg type
        pub_msg = NavSatFix()

        pub_msg.longitude = self.next_node_geometry.x
        pub_msg.latitude =  self.next_node_geometry.y

        self.pub_currentwaypoint.publish(pub_msg)
    
    def publish_heading_ref(self):
        """Publish heading_ref to a topic"""
        if self.next_node_geometry is None:
            rospy.loginfo(f'Vessel {self.id} does not have a next node geometry')
            return 
        pub_msg = Float32()

        self.heading_ref = heading(self.next_node_geometry.x, self.next_node_geometry.y , self.pos.longitude, self.pos.latitude)

        pub_msg.data = self.heading_ref
        rospy.loginfo(f"Heading_ref: {pub_msg}")

        self.pub_headingref.publish(pub_msg)


    @property
    def geometry(self):
        geometry = shapely.geometry.Point(self.pos.longitude, self.pos.latitude)
        return geometry
  
def main():
    node_name = 'Exp3'
    VESSEL_ID_1 = "RAS_TN_DB" #controlled

    route = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    VESSEL_ID_2 = "RAS_TN_GR" #what topic to subscribe to
    rospy.init_node(f"{node_name}", anonymous=False, log_level=rospy.INFO)
    

    vessel_1 = Vessel(id=VESSEL_ID_1, route=route, waypoints=['A', 'C', 'F', 'I', 'L'])
    vessel_2 = Vessel(id=VESSEL_ID_2, route=None, waypoints=None)

    alternatives_df = tactics.generate_all_alternatives(FG)
    
    operator_1 = Operator("Operator", controlled_vessel=vessel_1, observed_vessel=vessel_2, alternatives_df=alternatives_df)
    print('Hello')
    rospy.spin()

if __name__ == '__main__':
    main()

    


