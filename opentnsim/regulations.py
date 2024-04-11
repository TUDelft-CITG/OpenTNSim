import pandas as pd
from opentnsim import lock
import networkx as nx

class IsOneWayTrafficRegulation:
    def __init__(self, env, name, start_node_regulation, end_node_regulation, node_waiting_area_I, node_waiting_area_II, conditions, priority_rules, *args, **kwargs):
        self.env = env
        self.name = name
        self.start_node_regulation = node_doors1 = start_node_regulation
        self.end_node_regulation = node_doors2 = end_node_regulation
        self.node_waiting_area_start_I = node_waiting_area_I
        self.node_waiting_area_start_II = node_waiting_area_II
        self.conditions = conditions
        self.priority_rules = priority_rules
        super().__init__(*args, **kwargs)

        self.node_waiting_area_end_I = nx.dijkstra_path(self.env.FG,self.node_waiting_area_start_I,self.start_node_regulation)[1]
        self.node_waiting_area_end_II = nx.dijkstra_path(self.env.FG,self.node_waiting_area_start_II,self.end_node_regulation)[1]
        self.distance_to_regulation_I = self.env.vessel_traffic_service.provide_trajectory(self.env,self.node_waiting_area_start_I,self.start_node_regulation).length
        self.distance_to_regulation_II = self.env.vessel_traffic_service.provide_trajectory(self.env,self.node_waiting_area_start_II,self.end_node_regulation).length

        self.regulation = lock.IsLock(env=env,
                                      name=name,
                                      node_doors1 = node_doors1,
                                      node_doors2 = node_doors2,
                                      lock_length = self.env.FG.edges[start_node_regulation,end_node_regulation,0]['Info']['length'],
                                      lock_width = 999,
                                      lock_depth = 999,
                                      distance_doors1_from_first_waiting_area = self.distance_to_regulation_I,
                                      distance_doors2_from_second_waiting_area = self.distance_to_regulation_II,
                                      detector_nodes=[self.node_waiting_area_start_I, self.node_waiting_area_start_II],
                                      speed_reduction_factor=1.0,
                                      levelling_time=0,
                                      mandatory_time_gap_between_entering_vessels = 0,
                                      used_as_one_way_traffic_regulation = True,
                                      conditions = self.conditions,
                                      priority_rules = self.priority_rules)

        self.approach_I = lock.IsLockLineUpArea(env=self.env,
                                                name=self.name,
                                                start_node=self.node_waiting_area_start_I,
                                                end_node=self.node_waiting_area_end_I,
                                                lineup_length = 0,
                                                effective_lineup_length=100000,
                                                distance_to_lock_doors=self.distance_to_regulation_I,
                                                speed_reduction_factor=1.0,
                                                passing_allowed = True)
        self.approach_II = lock.IsLockLineUpArea(env=self.env,
                                                 name=self.name,
                                                 start_node=self.node_waiting_area_start_II,
                                                 end_node=self.node_waiting_area_end_II,
                                                 lineup_length=0,
                                                 effective_lineup_length = 100000,
                                                 distance_to_lock_doors=self.distance_to_regulation_II,
                                                 speed_reduction_factor=1.0,
                                                 passing_allowed = True)
        self.waiting_I = lock.IsLockWaitingArea(env=self.env,
                                                name=self.name,
                                                node=self.node_waiting_area_start_I,
                                                distance_from_node=self.distance_to_regulation_I)
        self.waiting_II = lock.IsLockWaitingArea(env=self.env,
                                                 name=self.name,
                                                 node=self.node_waiting_area_start_II,
                                                 distance_from_node=self.distance_to_regulation_II)

class HasOneWayTrafficRegulation(lock.HasLock,lock.HasLineUpArea,lock.HasWaitingArea):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ConditionalRule:
    def __init__(self,parameter,comparison,value,*args,**kwargs):
        self.parameter = parameter
        self.comparison = comparison
        self.value = value
        super().__init__(*args, **kwargs)

    def evaluate(self,parameter):
        df = pd.DataFrame([[parameter,self.value]],columns=['parameter','value'])
        condition = df.eval('parameter ' + self.comparison + ' value')
        return condition