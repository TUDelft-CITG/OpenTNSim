# package(s) related to the simulation
import networkx as nx

# spatial libraries
import pyproj
import shapely.geometry

class PassLock():
    """ Mixin class: a collection of functions which are used to pass a lock complex consisting of a waiting area, line-up areas, and lock chambers"""
    @staticmethod
    def approach_waiting_area(vessel,node_waiting_area):
        """ Processes vessels which are approaching a waiting area of a lock complex: 
                if the waiting area is full, vessels will be waiting outside the waiting area for a spot, otherwise if the vessel
                fits within the waiting area the vessels will proceed to the waiting area. 

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_waiting_area: a string which includes the name of the node the waiting area is located in the network """

        #Imports the properties of the waiting area
        waiting_area = vessel.env.FG.nodes[node_waiting_area]["Waiting area"][0]

        #Identifies the index of the node of the waiting area within the route of the vessel
        index_node_waiting_area = vessel.route.index(node_waiting_area)

        #Checks whether the waiting area is the first encountered waiting area of the lock complex
        for node_lineup_area in vessel.route[index_node_waiting_area:]:
            if 'Line-up area' not in vessel.env.FG.nodes[node_lineup_area].keys():
                continue

            #Imports the properties of the line-up areas
            lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
            for lineup_area in lineup_areas:
                if waiting_area.name.split('_')[0] != lineup_area.name.split('_')[0]:
                    continue

                #Determines the current time
                wait_for_waiting_area = vessel.env.now

                #Requests access to the waiting area
                vessel.access_waiting_area = waiting_area.waiting_area[node_waiting_area].request()
                yield vessel.access_waiting_area

                #Calculates and reports the waiting time for entering the waiting area
                if wait_for_waiting_area != vessel.env.now:
                    waiting = vessel.env.now - wait_for_waiting_area
                    vessel.log_entry("Waiting to enter waiting area start", wait_for_waiting_area, 0,
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node_waiting_area)-1]],)
                    vessel.log_entry("Waiting to enter waiting area stop", vessel.env.now, waiting,
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[vessel.route[vessel.route.index(node_waiting_area)-1]],)
                break
            break

    def leave_waiting_area(vessel,node_waiting_area):
        """ Processes vessels which are waiting in the waiting area of the lock complex and requesting access to preceding the line-up area:
                if there area multiple parallel lock chambers, the chamber with the least expected total waiting time is chosen,
                after which access is requested to enter the line-up area corresponding with the assigned lock chain series.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_waiting_area: a string which includes the name of the node the waiting area is located in the network """

        #Imports the properties of the waiting area
        waiting_area = vessel.env.FG.nodes[node_waiting_area]["Waiting area"][0]

        #Identifies the index of the node of the waiting area within the route of the vessel
        index_node_waiting_area = vessel.route.index(node_waiting_area)

        #Checks whether the waiting area is the first encountered waiting area of the lock complex
        for node_lineup_area in vessel.route[index_node_waiting_area:]:
            if 'Line-up area' not in vessel.env.FG.nodes[node_lineup_area].keys():
                continue

            #Imports the properties of the line-up areas
            lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
            for lineup_area in lineup_areas:
                if waiting_area.name.split('_')[0] != lineup_area.name.split('_')[0]:
                    continue

                #Imports the location of the lock chamber of the lock complex
                for node_lock in vessel.route[index_node_waiting_area:]:
                    if 'Lock' not in vessel.env.FG.nodes[node_lock].keys():
                        continue
                    locks = vessel.env.FG.nodes[node_lock]["Lock"]
                    break

                def choose_lock_chamber(vessel,lock,lock_position,series_number,lineup_areas,lock_queue_length):
                    """ Assigns the lock chamber with the least expected total waiting time to the vessel in case of parallell lock chambers. The
                            expected total waiting time is calculated through quantifying the total length of the queued vessels. If a vessel does
                            not fit in a lockage, it will create a new lockage cycle by requesting the full length capacity of the line-up area. When
                            this request is granted, the vessel will immediately release the obsolete length, such that more vessels can go with the
                            next lockage.
                        This function is evaluated in the leave_waiting_area function.

                        Input:
                            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                            - lock: an object within the network which is generated with the IsLock mixin class
                            - lock_position: a string which includes the name of the node the lock chamber is located in the network
                            - series_number: an integer number which indicates which lock series is evaluated (should not be greater than
                                             the maximum number of parallel lock chambers)
                            - lineup_areas: the collection of line-up areas in the node at which the line-up areas are located (the total
                                            amount of line-up areas should not exceed the amount of lock series)
                            - lock_queue_length: an empty list at which the resistance in total queue length (~ total waiting time) is
                                                 appended. """

                    #Imports the properties of the evaluated line-up area
                    lineup_area = lineup_areas[series_number]

                    #Assesses the total queue length within this lock series
                    #- if the queue for the line-up area is empty, a name is set if the vessel fits in the lock chamber and line-up right away, otherwise the queue is calculated
                    if lineup_area.length.get_queue == []:
                        if (vessel.L < lock.length.level and vessel.L < lineup_area.length.level and lock.water_level == vessel.route[vessel.route.index(lock_position)-1]):
                            vessel.lock_name = lock.name
                        elif vessel.L < lineup_area.length.level:
                            lock_queue_length.append(lineup_area.length.level)
                        else:
                            lock_queue_length.append(lineup_area.length.capacity)

                    #- else, if the vessel does not fit in the line-up area, the total length of the queued is calculated added with the full length capacity of the line-up area
                    else:
                        line_up_queue_length = lineup_area.length.capacity
                        for q in range(len(lineup_area.length.get_queue)):
                            line_up_queue_length += lineup_area.length.get_queue[q].amount
                        lock_queue_length.append(line_up_queue_length)

                def access_lineup_area(vessel,lineup_area):
                    """ Processes the request of vessels to access the line-up area by claiming a position (which equals the length of
                            the vessel) along its jetty.
                        This function is evaluated in the leave_waiting_area function

                        Input:
                            - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                            - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                                           assigned to the vessel as the lock series with the least expected total waiting time """

                    def create_new_lock_cycle_and_request_access():
                        """ Creates a new locking cycle (wave of vessels assigned to same lockage) by requesting the full length
                                capacity of the line-up area, assigning the vessel to this request and returning the obsolete length
                                when the request is granted.
                            This function is used in the access_lineup_area function within the leave_waiting_area function

                            No input required """

                        access_lineup_length = lineup_area.length.get(lineup_area.length.capacity)
                        lineup_area.length.get_queue[-1].length = vessel.L
                        yield access_lineup_length
                        lineup_area.length.put(lineup_area.length.capacity-vessel.L)

                    def request_access_lock_cycle(total_length_waiting_vessels = 0):
                        """ Processes the request of a vessel to enter a lock cycle (wave of vessels assigned to same lockage), depending
                                on the governing conditions regarding the current situation in the line-up area.
                            This function is used in the access_lineup_area function within the leave_waiting_area function

                            No input required """

                        #- If the line-up area has no queue, the vessel will access the lock cycle
                        if lineup_area.length.get_queue == []:
                            access_lineup_length = lineup_area.length.get(vessel.L)

                        #Else, if there are already preceding vessels waiting in a queue, the vessel will request access to a lock cycle
                        else:
                            #Determines the vessel which has started the last lock cycle
                            for q in reversed(range(len(lineup_area.length.get_queue))):
                                if lineup_area.length.get_queue[q].amount == lineup_area.length.capacity:
                                    break

                            #Calculates the total length of vessels assigned to this lock cycle
                            for q2 in range(q,len(lineup_area.length.get_queue)):
                                total_length_waiting_vessels += lineup_area.length.get_queue[q2].length

                            #If the vessels does not fit in this lock cycle, it will start a new lock cycle
                            if vessel.L > lineup_area.length.capacity - total_length_waiting_vessels:
                                yield from create_new_lock_cycle_and_request_access()

                            #Else, if the vessel does fit in this last lock cycle, it will request a place in this cycle
                            else:
                                access_lineup_length = lineup_area.length.get(vessel.L)

                                #Assigns the length of the vessel to this request
                                lineup_area.length.get_queue[-1].length = vessel.L

                                yield access_lineup_length

                    #Requesting procedure for access to line-up area
                    #- If there area vessels in the line-up area
                    if lineup_area.line_up_area[node_lineup_area].users != []:
                        # - If the vessels fits in the lock cycle right away
                        if vessel.L < (lineup_area.line_up_area[node_lineup_area].users[-1].lineup_dist-0.5*lineup_area.line_up_area[node_lineup_area].users[-1].length):
                            yield from request_access_lock_cycle()

                        #- Else, if the vessel does not fit in the lock cyle right away
                        else:
                            if lineup_area.length.get_queue == []:
                                yield from create_new_lock_cycle_and_request_access()
                            else:
                                yield from request_access_lock_cycle()

                    #- Else, if there are no vessels yet in the line-up area
                    else:
                        # - If the vessels fits in the lock cycle right away
                        if vessel.L < lineup_area.length.level:
                            yield from request_access_lock_cycle()

                        # - Else, if the vessel does not fit in the lock cyle right away
                        else:
                            if lineup_area.length.get_queue == []:
                                yield from create_new_lock_cycle_and_request_access()
                            else:
                                yield from request_access_lock_cycle()

                #Determines the current time
                wait_for_lineup_area = vessel.env.now

                #Assigning the lock chain series with least expected waiting time to the vessel
                vessel.lock_name = []
                lock_queue_length = []
                for count,lock in enumerate(locks):
                    choose_lock_chamber(vessel,lock,node_lock,count,lineup_areas,lock_queue_length)

                #If the function did not yet assign a lock chain series
                if vessel.lock_name == []:
                    vessel.lock_name = lineup_areas[lock_queue_length.index(min(lock_queue_length))].name

                #Request access line-up area which is assigned to the vessel
                if lineup_area.name != vessel.lock_name:
                    continue

                yield from access_lineup_area(vessel,lineup_area)

                #Release of vessel's occupation of the waiting area
                waiting_area.waiting_area[node_waiting_area].release(vessel.access_waiting_area)

                #Calculation of location in line-up area as a distance in [m] from start line-up jetty
                #- If the line-up area is not empty
                if len(lineup_area.line_up_area[node_lineup_area].users) != 0:
                    vessel.lineup_dist = (lineup_area.line_up_area[node_lineup_area].users[-1].lineup_dist -
                                          0.5*lineup_area.line_up_area[node_lineup_area].users[-1].length -
                                          0.5*vessel.L)
                #- Else, if the line-up area is empty
                else:
                    vessel.lineup_dist = lineup_area.length.capacity - 0.5*vessel.L

                #Calculation of the (lat,lon)-coordinates of the assigned position in the line-up area
                vessel.wgs84 = pyproj.Geod(ellps="WGS84")
                [lineup_area_start_lat,
                 lineup_area_start_lon,
                 lineup_area_stop_lat,
                 lineup_area_stop_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x,
                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)+1]]['geometry'].x,
                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)+1]]['geometry'].y]
                fwd_azimuth,_,_ = vessel.wgs84.inv(lineup_area_start_lat, lineup_area_start_lon, lineup_area_stop_lat, lineup_area_stop_lon)
                [vessel.lineup_pos_lat,
                 vessel.lineup_pos_lon,_] = pyproj.Geod(ellps="WGS84").fwd(vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x,
                                                                           vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                                                           fwd_azimuth,vessel.lineup_dist)

                #Formal request of the vessel to access the line-up area assigned to the vessel (always granted)
                vessel.access_lineup_area = lineup_area.line_up_area[node_lineup_area].request()

                #Some attributes are assigned to the vessel's formal request to access the line-up area
                lineup_area.line_up_area[node_lineup_area].users[-1].length = vessel.L
                lineup_area.line_up_area[node_lineup_area].users[-1].id = vessel.id
                [lineup_area.line_up_area[node_lineup_area].users[-1].lineup_pos_lat,
                 lineup_area.line_up_area[node_lineup_area].users[-1].lineup_pos_lon] = [vessel.lineup_pos_lat, vessel.lineup_pos_lon]
                lineup_area.line_up_area[node_lineup_area].users[-1].lineup_dist = vessel.lineup_dist
                lineup_area.line_up_area[node_lineup_area].users[-1].n = len(lineup_area.line_up_area[node_lineup_area].users) #(the number of vessels in the line-up area at the moment)
                lineup_area.line_up_area[node_lineup_area].users[-1].v = 0.5*vessel.v
                lineup_area.line_up_area[node_lineup_area].users[-1].wait_for_next_cycle = False #(a boolean which indicates if the vessel has to wait for a next lock cycle)
                lineup_area.line_up_area[node_lineup_area].users[-1].waited_in_waiting_area = False #(a boolean which indicates if the vessel had to wait in the waiting area)

                #Request of entering the line-up area to assure that vessels will enter the line-up area one-by-one
                vessel.enter_lineup_length = lineup_area.enter_line_up_area[node_lineup_area].request()
                yield vessel.enter_lineup_length

                #Speed reduction in the approach to the line-up area
                vessel.v = 0.5*vessel.v

                #Calculates and reports the total waiting time in the waiting area
                if wait_for_lineup_area != vessel.env.now:
                    waiting = vessel.env.now - wait_for_lineup_area
                    vessel.log_entry("Waiting in waiting area start", wait_for_lineup_area, 0,
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[node_waiting_area])
                    vessel.log_entry("Waiting in waiting area stop", vessel.env.now, waiting,
                                     nx.get_node_attributes(vessel.env.FG, "geometry")[node_waiting_area])

                    #Speed reduction in the approach to the line-up area, as the vessel had to lay still in the waiting area
                    vessel.v = 0.5*vessel.v

                    #Changes boolean of the vessel which indicates that it had to wait in the waiting area
                    for line_up_user in range(len(lineup_area.line_up_area[node_lineup_area].users)):
                        if lineup_area.line_up_area[node_lineup_area].users[line_up_user].id == vessel.id:
                            lineup_area.line_up_area[node_lineup_area].users[line_up_user].waited_in_waiting_area = True
                            break
                break
            break

    def approach_lineup_area(vessel,node_lineup_area):
        """ Processes vessels which are approaching the line-up area of the lock complex:
                determines whether the assigned position in the line-up area (distance in [m]) should be changed as the preceding vessel(s),
                which was/were waiting in the line-up area, has/have of is/are already accessed/accessing the lock.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """

        #Imports the properties of the line-up area the vessel is assigned to
        lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
        [vessel.lineup_pos_lat,vessel.lineup_pos_lon] = [vessel.env.FG.nodes[node_lineup_area]['geometry'].x,vessel.env.FG.nodes[node_lineup_area]['geometry'].y]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name:
                continue

            #Identifies the index of the node of the waiting area within the route of the vessel
            index_node_lineup_area = vessel.route.index(node_lineup_area)
            yield vessel.env.timeout(0)

            #Checks whether the line-up area is the first encountered line-up area of the lock complex
            for node_lock in vessel.route[index_node_lineup_area:]:
                if 'Lock' in vessel.env.FG.nodes[node_lock].keys():
                    locks = vessel.env.FG.nodes[node_lock]["Lock"]

                    for lock in locks:
                        if lock.name != vessel.lock_name:
                            continue

                        def change_lineup_dist(vessel, lineup_area, lock, lineup_dist, q):
                            """ Determines whether the assigned position in the line-up area (distance in [m]) should be changed as the preceding vessel(s),
                                    which was/were waiting in the line-up area, has/have of is/are already accessed/accessing the lock.
                                This function is used in the approach_lineup_area function.

                                Input:
                                    - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                                    - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                                                   assigned to the vessel as the lock series with the least expected total waiting time
                                    - lock: an object within the network which is generated with the IsLock mixin class and
                                            assigned to the vessel as the lock series with the least expected total waiting time
                                    - lineup_dist: the initial position of the vessel in the line-up area as the distance from the origin of the jetty in [m]
                                    - q: an integer number which represents the assigned position of the vessel in the line-up area, only the vessel which is
                                         the new first in line (q=0) will be processed"""

                            if q == 0 and lineup_area.line_up_area[node_lineup_area].users[q].n != (lineup_area.line_up_area[node_lineup_area].users[q].n-len(lock.resource.users)):
                                lineup_dist = lock.length.capacity - 0.5*vessel.L
                            return lineup_dist

                        #Checks the need to change the position of the vessel within the line-up area
                        for q in range(len(lineup_area.line_up_area[node_lineup_area].users)):
                            if lineup_area.line_up_area[node_lineup_area].users[q].id == vessel.id:

                                #Imports information about the current lock cycle
                                direction = vessel.route[vessel.route.index(node_lock)-1] == lock.node_1
                                lock_door_1_user_priority = 0
                                lock_door_2_user_priority = 0
                                lock_door_1_users = lock.doors_1[lock.node_1].users
                                lock_door_2_users = lock.doors_2[lock.node_3].users

                                if direction and lock_door_2_users != []:
                                    lock_door_2_user_priority = lock.doors_2[lock.node_3].users[0].priority

                                elif not direction and lock_door_1_users != []:
                                    lock_door_1_user_priority = lock.doors_1[lock.node_1].users[0].priority

                                #Decision if position should be changed
                                if direction and lock_door_2_user_priority == -1:
                                    vessel.lineup_dist = change_lineup_dist(vessel, lineup_area, lock, vessel.lineup_dist, q)

                                elif not direction and lock_door_1_user_priority == -1:
                                    vessel.lineup_dist = change_lineup_dist(vessel, lineup_area, lock, vessel.lineup_dist, q)

                                #Calculation of (lat,lon)-coordinates based on (newly) assigned position (line-up distance in [m])
                                [lineup_area_start_lat,
                                 lineup_area_start_lon,
                                 lineup_area_stop_lat,
                                 lineup_area_stop_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x,
                                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)+1]]['geometry'].x,
                                                          vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)+1]]['geometry'].y]

                                fwd_azimuth,_,_ = pyproj.Geod(ellps="WGS84").inv(lineup_area_start_lat,
                                                                                 lineup_area_start_lon,
                                                                                 lineup_area_stop_lat,
                                                                                 lineup_area_stop_lon)
                                [vessel.lineup_pos_lat,
                                 vessel.lineup_pos_lon,_] = pyproj.Geod(ellps="WGS84").fwd(vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].x,
                                                                                           vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lineup_area)]]['geometry'].y,
                                                                                           fwd_azimuth,vessel.lineup_dist)

                                #Changes the positional attributes of the vessel as user of the line-up area accordingly
                                lineup_area.line_up_area[node_lineup_area].users[q].lineup_pos_lat = vessel.lineup_pos_lat
                                lineup_area.line_up_area[node_lineup_area].users[q].lineup_pos_lon = vessel.lineup_pos_lon
                                lineup_area.line_up_area[node_lineup_area].users[q].lineup_dist = vessel.lineup_dist
                                break
                        break
                    break
                else:
                    continue
            break

    def leave_lineup_area(vessel,node_lineup_area):
        """ Processes vessels which are waiting in the line-up area of the lock complex:
                requesting access to the lock chamber given the governing phase in the lock cycle of the lock chamber and calculates the
                position within the lock chamber.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """

        #Imports the properties of the line-up area the vessel is assigned to
        lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name:
                continue

            #Identifies the index of the node of the line-up area within the route of the vessel
            index_node_lineup_area = vessel.route.index(node_lineup_area)

            #Checks whether the line-up area is the first encountered line-up area of the lock complex
            for node_lock in vessel.route[index_node_lineup_area:]: #(loops over line-up area)
                if 'Lock' not in vessel.env.FG.nodes[node_lock].keys():
                    continue

                #Imports the properties of the lock chamber the vessel is assigned to
                locks = vessel.env.FG.nodes[node_lock]["Lock"]
                for lock in locks:
                    if lock.name != vessel.lock_name:
                        continue

                    #Alters vessel's approach speed to lock chamber, if vessel didn't have to wait in the waiting area
                    for line_up_user in range(len(lineup_area.line_up_area[node_lineup_area].users)):
                        if (lineup_area.line_up_area[node_lineup_area].users[line_up_user].id == vessel.id and not
                            lineup_area.line_up_area[node_lineup_area].users[line_up_user].waited_in_waiting_area):
                            vessel.v = 0.5*vessel.v
                            break

                    #Imports position of the vessel in the line-up area
                    position_in_lineup_area = shapely.geometry.Point(vessel.lineup_pos_lat,vessel.lineup_pos_lon)

                    #Vessel releases its request to enter the line-up area, made in the waiting area
                    lineup_area.enter_line_up_area[node_lineup_area].release(vessel.enter_lineup_length)

                    #Determines current time
                    wait_for_lock_entry = vessel.env.now

                    #Checks whether the lock chamber is the first encountered lock chamber of the lock complex
                    for node_opposing_lineup_area in vessel.route[(index_node_lineup_area+1):]:
                        if 'Line-up area' not in vessel.env.FG.nodes[node_opposing_lineup_area].keys():
                            continue

                        #Imports the properties of the opposing line-up area the vessel is assigned to
                        opposing_lineup_areas = vessel.env.FG.nodes[node_opposing_lineup_area]["Line-up area"]
                        for opposing_lineup_area in opposing_lineup_areas:
                            if opposing_lineup_area.name == vessel.lock_name:
                                break
                        break

                    def access_lock_chamber(vessel,lineup_area,node_lineup_area,lock,node_lock,opposing_lineup_area,node_opposing_lineup_area,door1,door2):
                        """ Processes vessels which are waiting in the line-up area of the lock complex:
                                determines the current phase within the lock cycle and adjusts the request of the vessel accordingly
                            This function is used in the leave_lineup_area function.

                            Input:
                                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                                - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                                               assigned to the vessel as the lock series with the least expected total waiting time
                                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network
                                - lock: an object within the network which is generated with the IsLock mixin class and
                                        assigned to the vessel as the lock series with the least expected total waiting time
                                - node_lock: a string which includes the name of the node at which the lock chamber is located in the network
                                - opposing_lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and
                                                        assigned to the vessel to be entered when leaving the lock chamber
                                - node_opposing_lineup_area: a string which includes the name of the node at which the line-up area is located on the
                                                             opposite side of the lock chamber in the network
                                - door1: an object created in the IsLock class which resembles the set of lock doors which is first encountered by
                                         the vessel, which should be supscripted to, using a string which includes the name of the node at which this
                                         lock door is located in the network and was specified as input in the IsLock class
                                - door2: an object created in the IsLock class which resembles the set of lock doors which is last encountered by
                                         the vessel, which should be supscripted to, using a string which includes the name of the node at which this
                                         lock door is located in the network and was specified as input in the IsLock class """

                        def wait_for_requested_empty_conversion_by_oppositely_directed_vessel():
                            """ Vessel will wait for the lock chamber to be converted without vessels, as it was requested by
                                    the vessel(s) waiting on the other side of the lock chamber. This is programmed by a vessel's
                                    request of the converting while_in_line_up_area-resource of the opposing line-up area with
                                    capacity = 1, yielding a timeout, and immediately releasing the request.

                                No input required. """

                            vessel.waiting_during_converting = opposing_lineup_area.converting_while_in_line_up_area[node_opposing_lineup_area].request()
                            yield vessel.waiting_during_converting
                            opposing_lineup_area.converting_while_in_line_up_area[node_opposing_lineup_area].release(vessel.waiting_during_converting)

                        def request_approach_lock_chamber(timeout_required=True,priority=0):
                            """ Vessel will request if it can enter the lock by requesting access to the first set of lock doors. This
                                    request always has priority = 0, as vessels can only pass these doors when the doors are open (not
                                    claimed by a priority = -1 request). The capacity of the doors equals one, to prevent ships from
                                    entering simultaneously. The function yields a timeout. This can be switched off if it is assured
                                    the vessel can approach immediately.

                                Input:
                                    - timeout_required: a boolean which defines whether the requesting vessel receives a timeout. """

                            vessel.access_lock_door1 = door1.request(priority=priority)
                            if timeout_required:
                                yield vessel.access_lock_door1

                        def request_empty_lock_conversion(hold_request = False):
                            """ Vessel will request the lock chamber to be converted without vessels to his side of the lock chamber. This
                                    is programmed by requesting the converting_while_in_line_up_area resource of the line-up area the vessels is
                                    currently located in. If there was already a request by another vessel waiting in the same line-up area, this
                                    original request can be holded.

                                Input:
                                    - hold_request: a boolean which defines where an earlier request for the same empty lock conversion which
                                                    was made by another ship should be holded"""

                            vessel.waiting_during_converting = lineup_area.converting_while_in_line_up_area[node_lineup_area].request()
                            yield vessel.waiting_during_converting
                            if not hold_request:
                                yield from lock.convert_chamber(vessel.env, vessel.route[vessel.route.index(node_lock)-1], 0)
                            lineup_area.converting_while_in_line_up_area[node_lineup_area].release(vessel.waiting_during_converting)

                        def secure_lock_cycle(hold_request = False, timeout_required = True, priority = -1):
                            """ Vessel will indicate the direction of the next lock cycle by requesting access to the second pair of lock
                                    doors. Therefore, this request by default has priority = -1, which indicates the direction of the lockage
                                    as requests to access the same doors by vessels on the opposite side of the lock will be queued. A timeout
                                    is yielded. This can be switched off if it is assured the vessel can approach immediately. Furthermore, if
                                    there was already a request by another vessel waiting in the same line-up area, this original request can be
                                    holded. Lastly, the function is also used with priority = 0 in order to let vessels wait for the next lockage.

                                Input:
                                    - hold_request: a boolean which defines where an earlier request for the same empty lock conversion which
                                                    was made by another ship should be holded
                                    - timeout_required: a boolean which defines whether the requesting vessel receives a timeout.
                                    - priority: an integer [-1,0] which indicates the priority of the request: either with (-1) or without (0)
                                                priority. """

                            vessel.access_lock_door2 = door2.request(priority = priority)
                            if hold_request:
                                door2.release(door2.users[0])
                            if timeout_required:
                                yield vessel.access_lock_door2
                            door2.users[0].id = vessel.id

                        def wait_for_next_lockage():
                            """ Vessels will wait for the next lockage by requesting access to the second pair of lock doors without priority. If
                                    granted, the request will immediately be released.

                                No input required. """

                            yield from secure_lock_cycle(priority = 0)
                            door2.release(vessel.access_lock_door2)

                        #Determines current moment within the lock cycle
                        lock_door_2_user_priority = 0
                        if door2.users != []:
                            lock_door_2_user_priority = door2.users[0].priority

                        #Request procedure of the lock doors, which is dependent on the current moment within the lock cycle:
                        #- If there is a lock cycle being prepared or going on in the same direction of the vessel
                        if lock_door_2_user_priority == -1:
                            #If vessel does not fit in next lock cycle or locking has already started
                            if lock.resource.users != [] and (vessel.L > (lock.resource.users[-1].lock_dist-0.5*lock.resource.users[-1].length) or lock.resource.users[-1].converting):
                                yield from wait_for_next_lockage()

                            #Determines whether an empty conversion is needed or already requested by another vessel going the same way
                            if lineup_area.converting_while_in_line_up_area[node_lineup_area].users != []:
                                yield from request_empty_lock_conversion(hold_request = True)

                            elif len(door2.users) == 0 and len(door1.users) == 0 and vessel.route[vessel.route.index(node_lock)-1] != lock.water_level:
                                yield from request_empty_lock_conversion()

                            # Request to start the lock cycle
                            if not lock.resource.users[-1].converting:
                                yield from request_approach_lock_chamber()
                            else:
                                yield from request_approach_lock_chamber(priority=-1)

                            if door2.users != [] and door2.users[0].priority == -1:
                                yield from secure_lock_cycle(hold_request=True)
                            else:
                                yield from secure_lock_cycle(timeout_required=False)

                        #- If there is a lock cycle being prepared or going on to the direction of the vessel or if the lock chamber is empty
                        else:
                            #If the lock is already converting empty to the other side as requested by a vessel on the opposite side of the lock chamber
                            if opposing_lineup_area.converting_while_in_line_up_area[node_opposing_lineup_area].users != []:
                                yield from wait_for_requested_empty_conversion_by_oppositely_directed_vessel()

                            elif lineup_area.converting_while_in_line_up_area[node_lineup_area].users != []:
                                yield from request_empty_lock_conversion(hold_request=True)

                            #Determining (new) situation
                            #- If the lock chamber is empty
                            if lock.resource.users == []:
                                if door2.users != [] and door2.users[0].priority == -1:
                                    yield from request_approach_lock_chamber()
                                    yield from secure_lock_cycle(hold_request=True)
                                else:
                                    yield from request_approach_lock_chamber(timeout_required=False)
                                    yield from secure_lock_cycle(timeout_required=False)

                                #Determines if an empty lockage is required
                                if vessel.route[vessel.route.index(node_lock)-1] != lock.water_level:
                                    yield from request_empty_lock_conversion()

                            #- Else, if the lock chamber is occupied
                            else:
                                #Request to start the lock cycle
                                if not lock.resource.users[-1].converting:
                                    yield from request_approach_lock_chamber()
                                else:
                                    yield from request_approach_lock_chamber(priority=-1)

                                if door2.users != [] and door2.users[0].priority == -1:
                                    yield from secure_lock_cycle(hold_request=True)
                                else:
                                    yield from secure_lock_cycle(timeout_required=False)

                        #Formal request access to lock chamber and calculate position within the lock chamber
                        lock.length.get(vessel.L)
                        vessel.access_lock = lock.resource.request()
                        lock.pos_length.get(vessel.L)
                        vessel.lock_dist = lock.pos_length.level + 0.5*vessel.L #(distance from first set of doors in [m])

                        #Assign attributes to granted request
                        lock.resource.users[-1].id = vessel.id + '_' + str(vessel.lock_dist)
                        lock.resource.users[-1].length = vessel.L
                        lock.resource.users[-1].lock_dist = vessel.lock_dist
                        lock.resource.users[-1].converting = False #(boolean which indicates if the lock is already converting)

                    #Request access to lock chamber
                    direction = vessel.route[vessel.route.index(node_lock)-1] == lock.node_1 #(determines vessel's direction)
                    if direction:
                        yield from access_lock_chamber(vessel,lineup_area,node_lineup_area,lock,node_lock,opposing_lineup_area,node_opposing_lineup_area,
                                                       lock.doors_1[lock.node_1],lock.doors_2[lock.node_3])
                    if not direction:
                        yield from access_lock_chamber(vessel,lineup_area,node_lineup_area,lock,node_lock,opposing_lineup_area,node_opposing_lineup_area,
                                                       lock.doors_2[lock.node_3],lock.doors_1[lock.node_1])

                    #Releases the vessel's formal request to access the line-up area and releases its occupied length of the line-up area
                    lineup_area.line_up_area[node_lineup_area].release(vessel.access_lineup_area)
                    lineup_area.length.put(vessel.L)

                    #Calculates and reports the total waiting time in the line-up area
                    if wait_for_lock_entry != vessel.env.now:
                        waiting = vessel.env.now - wait_for_lock_entry
                        vessel.log_entry("Waiting in line-up area start", wait_for_lock_entry, 0, position_in_lineup_area)
                        vessel.log_entry("Waiting in line-up area stop", vessel.env.now, waiting, position_in_lineup_area)

                    #Calculation of (lat,lon)-coordinates of assigned position in lock chamber
                    vessel.wgs84 = pyproj.Geod(ellps="WGS84")
                    [doors_node_lineup_area_lat,
                     doors_node_lineup_area_lon,
                     doors_destination_lat,
                     doors_destination_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)-1]]['geometry'].x,
                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)-1]]['geometry'].y,
                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)+1]]['geometry'].x,
                                               vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)+1]]['geometry'].y]
                    fwd_azimuth,_,distance = vessel.wgs84.inv(doors_node_lineup_area_lat, doors_node_lineup_area_lon, doors_destination_lat, doors_destination_lon)
                    [vessel.lock_pos_lat,vessel.lock_pos_lon,_] = vessel.wgs84.fwd(vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)-1]]['geometry'].x,
                                                                                   vessel.env.FG.nodes[vessel.route[vessel.route.index(node_lock)-1]]['geometry'].y,
                                                                                   fwd_azimuth,vessel.lock_dist)
                    break
                break
            break

    def leave_lock(vessel,node_lock):
        """ Processes vessels which are waiting in the lock chamber to be levelled and after levelling:
                checks if vessels which have entered the lock chamber have to wait for the other vessels to enter the lock chamber and
                requests conversion of the lock chamber.

            Input:
                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                - node_lock: a string which includes the name of the node at which the lock chamber is located in the network """

        #Imports the properties of the lock chamber the vessel is assigned to
        locks = vessel.env.FG.nodes[node_lock]["Lock"]
        for lock in locks:
            if lock.name != vessel.lock_name:
                continue

            position_in_lock = shapely.geometry.Point(vessel.lock_pos_lat,vessel.lock_pos_lon) #alters node_lock in accordance with given position in lock
            first_user_in_lineup_length = lock.length.capacity

            #Identifies the index of the node of the lock chamber within the route of the vessel
            index_node_lock = vessel.route.index(node_lock)

            #Checks whether the lock chamber is the first encountered lock chamber of the lock complex
            for node_opposing_lineup_area in vessel.route[index_node_lock:]:
                if "Line-up area" not in vessel.env.FG.nodes[node_opposing_lineup_area].keys():
                    continue

                #Imports the properties of the opposing line-up area the vessel is assigned to
                opposing_lineup_areas = vessel.env.FG.nodes[node_opposing_lineup_area]["Line-up area"]
                for opposing_lineup_area in opposing_lineup_areas:
                    if opposing_lineup_area.name != vessel.lock_name:
                        continue
                    break
                break

            #Request access to pass the next line-up area after the lock chamber has levelled, so that vessels will leave the lock chamber one-by-one
            vessel.departure_lock = opposing_lineup_area.pass_line_up_area[node_opposing_lineup_area].request(priority = -1)

            #Releases the vessel's request of their first encountered set of lock doors
            direction = vessel.route[vessel.route.index(node_lock)-1] == lock.node_1
            if direction:
                lock.doors_1[lock.node_1].release(vessel.access_lock_door1)
            elif not direction:
                lock.doors_2[lock.node_3].release(vessel.access_lock_door1)

            #Imports the properties of the line-up area the vessel was assigned to

            for node_lineup_area in reversed(vessel.route[:(index_node_lock-1)]):
                if "Line-up area" not in vessel.env.FG.nodes[node_lineup_area].keys():
                    continue
                lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
                for lineup_area in lineup_areas:
                    if lineup_area.name != vessel.lock_name:
                        continue
                    lineup_users = lineup_area.line_up_area[node_lineup_area].users
                    if lineup_users != []:
                        first_user_in_lineup_length = lineup_area.line_up_area[node_lineup_area].users[0].length

                    #Determines current time and reports this to vessel's log as start time of lock passage
                    start_time_in_lock = vessel.env.now
                    vessel.log_entry("Passing lock start", vessel.env.now, 0, position_in_lock)

                    def waiting_for_other_lock_users(vessel,lock,node_lock,lineup_area,node_lineup_area,door1):
                        """ Function which yields a timeout to vessels as they have to wait for the other vessels to enter the lock.
                                the timeout is calculated by subsequently requesting and releasing the previously passed line-up area
                                and lock doors again without priority, such that the all the vessels within the line-up area can enter
                                the lock before the levelling will start.
                            This function is used in the leave_lock function.

                            Input:
                                - vessel: an identity which is Identifiable, Movable, and Routable, and has VesselProperties
                                - lock: an object within the network which is generated with the IsLock mixin class and is asigned to the
                                        vessel as the lock chamber in the lock chain series with the least expected total waiting time
                                - node_lock: a string which includes the name of the node at which the lock chamber is located in the network
                                - lineup_area: an object within the network which is generated with the IsLineUpArea mixin class and is assigned
                                               to the vessel as the line-up area in the lock chain series with the least expected total waiting time
                                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network
                                - door1: an object created in the IsLock class which resembles the set of lock doors which is first encountered by
                                         the vessel, which should be supscripted to, using a string which includes the name of the node at which this
                                         lock door is located in the network and was specified as input in the IsLock class """

                        vessel.access_line_up_area = lineup_area.enter_line_up_area[node_lineup_area].request()
                        yield vessel.access_line_up_area
                        lineup_area.enter_line_up_area[node_lineup_area].release(vessel.access_line_up_area)
                        vessel.access_lock_door1 = door1.request() #therefore it requests first the entering of line-up area and then the lock doors again
                        yield vessel.access_lock_door1
                        door1.release(vessel.access_lock_door1)

                    #Determines if accessed vessel has to wait on accessing vessels
                    if direction and first_user_in_lineup_length < lock.length.level:
                        yield from waiting_for_other_lock_users(vessel,lock,node_lock,lineup_area,node_lineup_area,lock.doors_1[lock.node_1])
                    if not direction and first_user_in_lineup_length < lock.length.level:
                        yield from waiting_for_other_lock_users(vessel,lock,node_lock,lineup_area,node_lineup_area,lock.doors_2[lock.node_3])

                    #Determines if the vessel explicitely has to request the conversion of the lock chamber (only the last entered vessel) or can go with a previously made request
                    if lock.resource.users[-1].id == vessel.id + '_' + str(vessel.lock_dist):
                        lock.resource.users[-1].converting = True
                        number_of_vessels = len(lock.resource.users)
                        yield from lock.convert_chamber(vessel.env, vessel.route[vessel.route.index(node_lock)+1],number_of_vessels)
                    else:
                        for lock_user in range(len(lock.resource.users)):
                            if lock.resource.users[lock_user].id != vessel.id + '_' + str(vessel.lock_dist):
                                continue
                            lock.resource.users[lock_user].converting = True
                            yield vessel.env.timeout(lock.doors_close + lock.operation_time(vessel.env) + lock.doors_open)
                            break

                #Yield request to leave the lock chamber
                yield vessel.departure_lock

                #Calculates and reports the total locking time
                vessel.log_entry("Passing lock stop", vessel.env.now, vessel.env.now-start_time_in_lock, position_in_lock,)

                #Adjusts position of the vessel in line-up area, apart from the first line-up area also used to position the vessel in the next line-up area, to the origin of the next line-up area
                [vessel.lineup_pos_lat,vessel.lineup_pos_lon] = [vessel.env.FG.nodes[vessel.route[vessel.route.index(node_opposing_lineup_area)]]['geometry'].x,
                                                                 vessel.env.FG.nodes[vessel.route[vessel.route.index(node_opposing_lineup_area)]]['geometry'].y]
                break
            break

    def leave_opposite_lineup_area(vessel,node_lineup_area):
        """ Processes vessels which have left the lock chamber after levelling and are now in the next line-up area in order to leave the lock complex through the next waiting area:
                release of their requests for accessing their second encountered line-up area and lock chamber.

            Input:
                - vessel: an identity which is Identifiable, Movable,and Routable, and has VesselProperties
                - node_lineup_area: a string which includes the name of the node at which the line-up area is located in the network """

        #Imports the properties of the line-up area the vessel is assigned to
        lineup_areas = vessel.env.FG.nodes[node_lineup_area]["Line-up area"]
        for lineup_area in lineup_areas:
            if lineup_area.name != vessel.lock_name: #assure lock chain = assigned chain
                continue
            index_node_lineup_area = vessel.route.index(node_lineup_area)

            #Checks whether the line-up area is the second encountered line-up area of the lock complex
            for node_lock in reversed(vessel.route[:(index_node_lineup_area-1)]):
                if 'Lock' not in vessel.env.FG.nodes[node_lock].keys():
                    continue

                #Imports the properties of the lock chamber the vessel was assigned to
                locks = vessel.env.FG.nodes[node_lock]["Lock"]
                for lock in locks:
                    if lock.name != vessel.lock_name:
                        continue

                    #Releases the vessel's request of their second encountered set of lock doors
                    direction = vessel.route[vessel.route.index(node_lock)+1] == lock.node_3
                    if direction and lock.doors_2[lock.node_3].users[0].id == vessel.id:
                        lock.doors_2[lock.node_3].release(vessel.access_lock_door2)
                    if not direction and lock.doors_1[lock.node_1].users[0].id == vessel.id:
                        lock.doors_1[lock.node_1].release(vessel.access_lock_door2)

                    #Releases the vessel's request to enter the second line-up area
                    opposing_lineup_area = lineup_area
                    node_opposing_lineup_area = node_lineup_area
                    opposing_lineup_area.pass_line_up_area[node_opposing_lineup_area].release(vessel.departure_lock)

                    #Releases the vessel's formal request of the lock chamber and returns its occupied length in the lock chamber
                    lock.resource.release(vessel.access_lock)
                    departure_lock_length = lock.length.put(vessel.L) #put length back in lock
                    departure_lock_pos_length = lock.pos_length.put(vessel.L) #put position length back in lock
                    yield departure_lock_length
                    yield departure_lock_pos_length
                    break
                break
            break