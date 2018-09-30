#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core code for package"""

import math
import networkx as nx

def Distance(Node1, Node2, Print=False):
    # calculate distance between two points
    distance = ((Node2[0]-Node1[0])**2+(Node2[1]-Node1[1])**2)**0.5

    if Print:
        print('distance :', distance)
    return distance

def Angle(Node1, Node2, Print=False):
    # calculate angle going from point 1 to point 2 (NB: atan2(y,x))
    angle = math.atan2(Node2[1]-Node1[1],Node2[0]-Node1[0])*(180.0 / math.pi) 

    if Print:
        print('angle :', angle)
    return angle

def Move(Node1, Node2, move, Print=False):
    # move towards Node2
    distance = Distance(Node1, Node2)
    angle = Angle(Node1, Node2)
    new_node = (Node1[0]+math.cos(angle*math.pi/180)*move, Node1[1]+math.sin(angle*math.pi/180)*move)
    if Print:
        print('new_node :', new_node)
    return new_node

def Move_on_path(w_G, vessel_pos, path, to_node_id, move, Print=False):
    # 1. vessel_pos is your boat
    # 2. you need to know the next node in the path
    # 3. calculate the distance between Node1 and the next node
    to_node_pos = nx.get_node_attributes(w_G, 'pos')[path[to_node_id]]
    distance_to_next_node = Distance(vessel_pos, to_node_pos, True)
    print('distance_to_next_node: ', distance_to_next_node)
    
    # If move <= distance to next node, make move
    if move < distance_to_next_node:
            print('move: ', move)
            vessel_pos_new = Move(vessel_pos, to_node_pos, move, True)
    else:
        if to_node_id == len(path)-1:
            print('arrived')
            vessel_pos_new = to_node_pos 
        else:
            # reposition vessel at next node and calculate remaining move distance
            vessel_pos = nx.get_node_attributes(w_G, 'pos')[path[to_node_id]]
            # up the to_node_id with 1 and find new to_node_pos
            to_node_id+=1
            to_node_pos = nx.get_node_attributes(w_G, 'pos')[path[to_node_id]]
            # calculate distance to next node
            print('move: ', move)
            move = move-distance_to_next_node
            print('move: ', move)
            vessel_pos_new = Move(vessel_pos, to_node_pos, move, True)
        
    # If move > distance to next node, move to next node and set move to move-distance, try again
    if Print:
        print('vessel_pos_new:', vessel_pos_new)
        print('to_node_id :', to_node_id)
        print('')
        
    return vessel_pos_new, to_node_id