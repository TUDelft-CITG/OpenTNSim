#!/usr/bin/env python3
import pickle
import math
import matplotlib.pyplot as plt

import rospy

import pyproj

import numpy as np
import networkx as nx
import geopandas as gpd
import shapely.geometry
import pandas as pd
import matplotlib.pyplot as plt

from std_msgs.msg import Float32
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix

from network_gdf import network_FG

FG = network_FG()



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
        
class Operator():
    def __init__(self, id, controlled_vessel, observed_vessel):
        self.id = id
        self.controlled_vessel = controlled_vessel
        self.observed_vessel = observed_vessel
        self.pub_tactic = rospy.Publisher(f"/{controlled_vessel.id}/tactic", String, queue_size=10)

        self.rate = rospy.Rate(0.5) 
        self.publish_tactic()

    # def update_pos(self, msg: NavSatFix()):
    #     self.pos = msg
    #     dist = distance(self.berth_loc[0], self.berth_loc[1], self.pos.longitude, self.pos.latitude)
    #     print(dist)
    #     # Add conditional statement that involves the berth availability: 'if pos is within polygon'
    #     if dist <= 6:
    #         rospy.loginfo('Observed ship is at berth so run simulation and send tactic')
    #         self.publish_tactic()

    def publish_tactic(self):
        while not rospy.is_shutdown():
            self.pub_tactic.publish("GS")
            self.rate.sleep()
        
        #self.berth_loc =  [4.371807932569391, 52.001592043587344]
        #self.timer =  rospy.Timer(rospy.Duration(2), self.publish_tactic)

    

    
        

class Vessel():
    def __init__(self, id):
        self.id = id
        self.pos = NavSatFix()
    
    def update_pos(self, msg: NavSatFix()):
        self.pos = msg
        rospy.loginfo(f'Vessel : {self.id}, Lat {self.pos.latitude},Lon {self.pos.longitude}')
        #self.pose_update_control_func()

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
        if dist <= self.waypoint_criteria:
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
    VESSEL_ID_1 = "RAS_TN_DB" #what topic to subscribe to
    VESSEL_ID_2 = "RAS_TN_DG" #what topic to subscribe to

    vessel_1 = Vessel(id=VESSEL_ID_1)
    vessel_2 = Vessel(id=VESSEL_ID_2)
    

    # initialize ROS node and subscribers
    rospy.init_node(f"{node_name}", anonymous=False, log_level= rospy.INFO)

    
    
    #operator_1.publish_tactic()
    #waypoints = []
    #waypoints.append(("Back of the boat", 52.00162378864698, 4.371862804893368))
    #waypoints.append(("Middle of the boat", 52.001605624642195, 4.371874204281952))
    #waypoints.append(("Front of the boat", 52.00158126835151, 4.371891638640965))

    #print(read_network())
    
    #s2p = Sail2point()
    rospy.Subscriber(f"/{VESSEL_ID_1}/geopos_est", NavSatFix, callback=vessel_1.update_pos)
    rospy.Subscriber(f"/{VESSEL_ID_2}/geopos_est", NavSatFix, callback=vessel_2.update_pos)
    #rospy.Subscriber(f"/{VESSEL_ID_1}/geopos_est", NavSatFix, callback=operator_1.update_pos)
    operator_1 = Operator("Operator", vessel_2, vessel_1)
    rospy.spin()

if __name__ == '__main__':
    main()

    


