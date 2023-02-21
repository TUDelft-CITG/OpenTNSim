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
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix

from network_gdf import network_FG

FG = network_FG()

WAYPOINT_THRESHOLD = 3 #m


def read_network():
    filename = '/home/gijn/catkin_ws/src/my_robot_controller/scripts/FG_pond.gpickle'
    with open(filename, 'rb') as f:
       FG = pickle.load(f)
    return FG


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
    def __init__(self, id, controlled_vessel, observed_vessel):
        self.id = id
        self.controlled_vessel = controlled_vessel
        self.observed_vessel = observed_vessel

        self.berth_loc =  [4.371807932569391, 52.001592043587344]

        self.pub_tactic = rospy.Publisher(f"/{controlled_vessel.id}/tactic", String, queue_size=10)
        print(self.observed_vessel.id)
        self.sub_observed_vessel = rospy.Subscriber(f"/{self.observed_vessel.id}/geopos_est", NavSatFix, callback = self.check_berth_availability)
        self.berth_available = True 

        rospy.Timer(rospy.Duration(1), self.publish_tactic)
        rospy.Timer(rospy.Duration(5), self.run_simulation)
        #self.publish_tactic()

    def check_berth_availability(self, msg: NavSatFix()):
        self.pos = msg
        dist = distance(self.berth_loc[0], self.berth_loc[1], self.pos.longitude, self.pos.latitude)
        print(dist)
        # Add conditional statement that involves the berth availability: 'if pos is within polygon'
        if dist <= 8:
            self.berth_available = False
        else:
            self.berth_available = True
            
    def publish_tactic(self, event):
        if not self.berth_available:
            tactic = "GS"
        else:
            tactic = "GR"
        self.pub_tactic.publish(tactic)

    def run_simulation(self, event):
        geometry = self.controlled_vessel.geometry
        graph = FG
        
        duration, end, energy_df, vessel = run_simulation(geometry=geometry, route=self.controlled_vessel.route, graph=graph)
        print(duration)

class Vessel():
    def __init__(self, id, route=None):
        self.id = id
        self.route = route 
        self.pos = NavSatFix()
        self.sub = rospy.Subscriber(f"/{self.id}/geopos_est", NavSatFix, callback=self.update_pos)
    
    def update_pos(self, msg: NavSatFix()):
        self.pos = msg
        rospy.loginfo(f'Vessel : {self.id}, Lat {self.pos.latitude},Lon {self.pos.longitude}')
        self.update_route()
        #self.pose_update_control_func()

    def update_route(self):
        if self.route is None:
            return
        if len(self.route)<1:
            return
        next_node = self.route[0]
        node_geometry = FG.nodes[next_node]['geometry']
        geometry = self.geometry
        dist_to_next_node = distance(node_geometry.x, node_geometry.y, geometry.x, geometry.y)
        print(f'Distance to next node {dist_to_next_node}')
        if dist_to_next_node < WAYPOINT_THRESHOLD:
            visited_node = self.route.pop(0)
            rospy.loginfo(f' Visited node: {visited_node}')

        


    @property
    def geometry(self):
        geometry = shapely.geometry.Point(self.pos.longitude, self.pos.latitude)
        return geometry

class Sail2point():
    def __init__(self):
        VESSEL_ID = "RAS_TN_DB"
        self.waypoint_criteria  = 3# meters
        self.pos = NavSatFix()
        self.FG = FG
        
        self.pub_currentwaypoint = rospy.Publisher(f"/{VESSEL_ID}/current_waypoint", NavSatFix, queue_size=10)
        self.pub_headingref = rospy.Publisher(f"/{VESSEL_ID}/heading_ref", Float32, queue_size=10)

        # Get geometry of nodes = Point (lon, lat) and store as waypoint
        self.waypoints = nx.get_node_attributes(self.FG, "geometry")

        self.lap_counter = 0
        self.currentpoint = 0
        self.currentwaypoint = self.waypoints[0]
        print(self.waypoints[0])



    
    def update_pos(self, msg: NavSatFix()):
        self.pos.latitude  = msg.latitude
        self.pos.longitude = msg.longitude
        rospy.loginfo(f'Lat {self.pos.latitude},Lon {self.pos.longitude}')
        self.pose_update_control_func()
        
    def pose_update_control_func(self):
        """Calculate distance to the next waypoint and check if it needs to be
        moved to the next one in the list"""

        # Calculate distance to the next waypoint
        dist = distance(shapely.geometry.shape(self.waypoints[self.currentpoint]).x, shapely.geometry.shape(self.waypoints[self.currentpoint]).y , self.pos.longitude, self.pos.latitude)
        rospy.loginfo(f'Sailing to node {self.currentpoint}. Distance to next waypoint = {dist:.3f} meters')
        #check if advance criteria is met
        if dist <= WAYPOINT_THRESHOLD:
            # Go to next waypoint
            # Length waypoints -1 because it's a circle and the end node is the start node again
            if self.currentpoint >= len(self.waypoints)-1:
                self.lap_counter = +1
                self.currentpoint = 1
            else:
                self.currentpoint += 1
        self.publishCurrentWaypoint()
        self.publishHeadingRef()


    def publishCurrentWaypoint(self):
        """Publish current waypoint to a topic"""

        #Define msg type
        pub_msg = NavSatFix()

        self.currentwaypoint = self.waypoints[self.currentpoint]

        pub_msg.longitude = shapely.geometry.shape(self.currentwaypoint).x
        pub_msg.latitude =  shapely.geometry.shape(self.currentwaypoint).y

        self.pub_currentwaypoint.publish(pub_msg)
    
    def publishHeadingRef(self):
        """Publish heading_ref to a topic"""

        pub_msg = Float32()

        self.heading_ref = heading(shapely.geometry.shape(self.waypoints[self.currentpoint]).x, shapely.geometry.shape(self.waypoints[self.currentpoint]).y , self.pos.longitude, self.pos.latitude)

        pub_msg.data = self.heading_ref
        rospy.loginfo(f"Heading_ref: {pub_msg}")

        self.pub_headingref.publish(pub_msg)
    

def main():
    node_name = 'Exp3'
    VESSEL_ID_1 = "RAS_TN_DB" #controlled

    route = [0,1,2,3,4,5,6,7,8,9,10,11]
    VESSEL_ID_2 = "RAS_TN_DG" #what topic to subscribe to
    rospy.init_node(f"{node_name}", anonymous=False, log_level= rospy.INFO)


    vessel_1 = Vessel(id=VESSEL_ID_1, route=route)
    vessel_2 = Vessel(id=VESSEL_ID_2, route=None)
    

    # initialize ROS node and subscribers
    

    
    
    #operator_1.publish_tactic()
    #waypoints = []
    #waypoints.append(("Back of the boat", 52.00162378864698, 4.371862804893368))
    #waypoints.append(("Middle of the boat", 52.001605624642195, 4.371874204281952))
    #waypoints.append(("Front of the boat", 52.00158126835151, 4.371891638640965))

    #print(read_network())
    
    #s2p = Sail2point()
    #rospy.Subscriber(f"/{VESSEL_ID_1}/geopos_est", NavSatFix, callback=vessel_1.update_pos)
    #rospy.Subscriber(f"/{VESSEL_ID_2}/geopos_est", NavSatFix, callback=vessel_2.update_pos)
    #rospy.Subscriber(f"/{VESSEL_ID_1}/geopos_est", NavSatFix, callback=operator_1.update_pos)
    operator_1 = Operator("Operator", controlled_vessel=vessel_1, observed_vessel=vessel_2)
    print('Hello')
    rospy.spin()

if __name__ == '__main__':
    main()

    


