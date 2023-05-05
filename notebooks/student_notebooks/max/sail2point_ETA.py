#!/usr/bin/env python3
import pickle
import math
import datetime, time

import matplotlib.pyplot as plt
import numpy as np

import rospy
import simpy
import pyproj
import opentnsim.core

import networkx as nx
import geopandas as gpd
import shapely.geometry
import pandas as pd


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
                lat2, radians=False)[1]
        return heading/180*math.pi

class Sail2point():
    def __init__(self):
        VESSEL_ID = "titoneri1"
        self.waypoint_criteria  = 0.75# meters
        self.pos = NavSatFix()
        self.FG = FG
        self.path_og = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] #hard-coded: should be dynamic
        self.pub_currentwaypoint = rospy.Publisher(f"/{VESSEL_ID}/current_waypoint", NavSatFix, queue_size=10)
        self.pub_headingref = rospy.Publisher(f"/{VESSEL_ID}/heading_ref", Float32, queue_size=10)

        # Get geometry of nodes = Point (lon, lat) and store as waypoint
        self.waypoints = nx.get_node_attributes(self.FG, "geometry")
        self.counter = datetime.datetime.now()
        self.currentpoint = 0
        self.currentwaypoint = self.waypoints[0]
        rospy.loginfo(f'ETA of trip = {self.run_simulation()[1]} and it will take {self.run_simulation()[0]:.1f} seconds')
        



    
    def update_pos(self, msg: NavSatFix()):
        self.pos.latitude  = msg.latitude
        self.pos.longitude = msg.longitude
        #rospy.loginfo(f'Lat {self.pos.latitude},Lon {self.pos.longitude}')
        self.pose_update_control_func()
        
    def pose_update_control_func(self):
        """Calculate distance to the next waypoint and check if it needs to be
        moved to the next one in the list"""

        # Calculate distance to the next waypoint
        dist = distance(shapely.geometry.shape(self.waypoints[self.currentpoint]).x, shapely.geometry.shape(self.waypoints[self.currentpoint]).y , self.pos.longitude, self.pos.latitude)
        #rospy.loginfo(f'Sailing to node {self.currentpoint}. Distance to next waypoint = {dist:.3f} meters')
        #check if advance criteria is met
        if dist <= self.waypoint_criteria:
            duration_sailing = datetime.datetime.now() - self.counter
            self.counter = datetime.datetime.now()
            # Go to next waypoint
            # Length waypoints -1 because it's a circle and the end node is the start node again
            if self.currentpoint >= len(self.waypoints)-1:
                self.currentpoint = 1
            else:
                self.currentpoint += 1
            
            #Calculate new ETA when waypoint criteria is met
            rospy.loginfo(f'Registered at node {self.currentpoint}.')
            rospy.loginfo(f'Sailing took {duration_sailing.total_seconds():.2f} seconds')
            rospy.loginfo(f'ETA of trip = {self.run_simulation()[1]} in {self.run_simulation()[0]:.1f} seconds')

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
        #rospy.loginfo(f"Heading_ref: {pub_msg}")

        self.pub_headingref.publish(pub_msg)
    
    def run_simulation(self):
        Vessel = type('Vessel', 
                            (opentnsim.core.Identifiable, opentnsim.core.Movable, 
                            opentnsim.core.HasResource, opentnsim.core.Routeable, opentnsim.core.HasContainer), {})
        
        data_vessel = {"env": None,
                        "name": "NausBot",
                        "route": None,
                        "geometry": self.waypoints[self.currentpoint],  # lon, lat
                        "capacity": 1_000,
                        "v": 0.7} #approximate speed of Nausbot
        vessel = Vessel(**data_vessel)
        
        #Assuming the path is always an ascending order of numbers till the last node
        path = np.arange(self.path_og[self.currentpoint], self.path_og[-1])

        # start simpy environment (specify the start time and add the graph to the environment)
        simulation_start = datetime.datetime.now()
        env = simpy.Environment(initial_time = time.mktime(simulation_start.timetuple()))
        env.FG = self.FG

        # add environment to the vessel, and specify the vessels route and current location (beginning of the path)
        vessel.env = env
        vessel.route = path
             

        # specify the process that needs to be executed
        env.process(vessel.move())

        # start the simulation
        env.run()
        
        #Determine trip time
        start_time = simulation_start.timestamp()

        end = vessel.log["Timestamp"][-1]

        end_time = end.timestamp()
        
        
        return end_time-start_time, end

def main():
    VESSEL_ID = "titoneri1" #what topic to subscribe to

    # initialize ROS node and subscribers
    rospy.init_node(f"{VESSEL_ID}_sail_2_point", anonymous=False, log_level= rospy.INFO)

    #waypoints = []
    #waypoints.append(("Back of the boat", 52.00162378864698, 4.371862804893368))
    #waypoints.append(("Middle of the boat", 52.001605624642195, 4.371874204281952))
    #waypoints.append(("Front of the boat", 52.00158126835151, 4.371891638640965))

    #print(read_network())
    
    s2p = Sail2point()
    rospy.Subscriber(f"/{VESSEL_ID}/geoPos_est", NavSatFix, callback=s2p.update_pos)
    rospy.spin()

if __name__ == '__main__':
    main()

    


