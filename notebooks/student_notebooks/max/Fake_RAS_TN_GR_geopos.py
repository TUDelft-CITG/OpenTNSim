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


def main():
    rospy.init_node(f"fake", anonymous=False, log_level=rospy.INFO)
    counter = 0
    geopos1= [4.371807932569391, 52.001592043587344]
    geopos2 = [4.372, 52.002]
    r = rospy.Rate(0.1) # 10hz
    while not rospy.is_shutdown():
        if (counter % 10) < 5:
            geopos = geopos1
        else:
            geopos = geopos2
        pub_state_geopos_GR = rospy.Publisher(f"/RAS_TN_GR/state/geopos", NavSatFix, queue_size=10)
        msg = NavSatFix()
        msg.longitude = geopos[0]
        msg.latitude = geopos[1]
        pub_state_geopos_GR.publish(msg)
        rospy.loginfo(msg)
        counter += 1
        r.sleep()
    




if __name__ == '__main__':
    main()


