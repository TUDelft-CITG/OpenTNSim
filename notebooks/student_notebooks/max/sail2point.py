#!/usr/bin/env python3
import pickle
import math
import matplotlib.pyplot as plt

import rospy

import pyproj

import networkx as nx
import geopandas as gpd
import shapely.geometry
import pandas as pd
import matplotlib.pyplot as plt

from std_msgs.msg import Float32
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
                lat2, radians=False)[0]
        return heading/180*math.pi

class Sail2point():
    def __init__(self):
        VESSEL_ID = "RAS_TN_GR"
        self.waypoint_criteria  = 3# meters
        self.pos = NavSatFix()
        self.FG = FG

        self.pub_currentwaypoint = rospy.Publisher(f"/{VESSEL_ID}/current_waypoint", NavSatFix, queue_size=10)
        self.pub_headingref = rospy.Publisher(f"/{VESSEL_ID}/heading_ref_frank", Float32, queue_size=10)

        # Get geometry of nodes = Point (lon, lat) and store as waypoint
        self.waypoints = nx.get_node_attributes(self.FG, "geometry")

        self.currentpoint = 0
        self.currentwaypoint = self.waypoints[0]



    
    def update_pos(self, msg: NavSatFix()):
        self.pos.latitude  = msg.latitude
        self.pos.longitude = msg.longitude
        print(self.pos.latitude, self.pos.longitude)
        self.pose_update_control_func()
        
    def pose_update_control_func(self):
        """Calculate distance to the next waypoint and check if it needs to be
        moved to the next one in the list"""

        # Calculate distance to the next waypoint
        dist = distance(shapely.geometry.shape(self.waypoints[self.currentpoint]).x, shapely.geometry.shape(self.waypoints[self.currentpoint]).y , self.pos.longitude, self.pos.latitude)
        rospy.loginfo(f'Distance to next waypoint = {dist:.3f} meters')
        #check if advance criteria is met
        if dist <= self.waypoint_criteria:
            # Go to next waypoint
            if self.currentpoint >= len(self.waypoints):
                self.currentpoint = 1
            else:
                self.currentpoint += 1
        print(f'Index of current waypoint is {self.currentpoint}')

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

        self.pub_headingref.publish(pub_msg)
    

def main():
    VESSEL_ID = "RAS_TN_GR" #what topic to subscribe to

    # initialize ROS node and subscribers
    rospy.init_node(f"{VESSEL_ID}_sail_2_point", anonymous=False, log_level= rospy.INFO)

    #waypoints = []
    #waypoints.append(("Back of the boat", 52.00162378864698, 4.371862804893368))
    #waypoints.append(("Middle of the boat", 52.001605624642195, 4.371874204281952))
    #waypoints.append(("Front of the boat", 52.00158126835151, 4.371891638640965))

    #print(read_network())
    
    s2p = Sail2point()
    rospy.Subscriber(f"/{VESSEL_ID}/geopos_est", NavSatFix, callback=s2p.update_pos)
    rospy.spin()

if __name__ == '__main__':
    main()

    


