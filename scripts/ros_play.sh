#!/bin/bash
. /ros_entrypoint.sh
cat /ros_entrypoint.sh
roslaunch rosbridge_server rosbridge_websocket.launch &
rosbag play 9_nov_2022_demo1.bag
