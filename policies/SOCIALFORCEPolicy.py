import os
import math
import sys
import torch
import numpy as np


from policies import socialforce

import copy
import argparse

# Filter list by Boolean list 
# Using itertools.compress 
from itertools import compress

class SOCIALFORCEPolicy(object):
    def __init__(self):

        self.obs_seq_len = 8

        self.is_init = False


            

    def init(self):

        self.is_init = True

    def predict(self, history, agent_index, goal, pref_speed=1.0):


        goal= np.array(goal)

        combined_history_x = history[:,:,0]
        combined_history_y = history[:,:,1]

        self.n_agents = len(combined_history_x)
        

        combined_history_x = np.array( combined_history_x )
        combined_history_y = np.array( combined_history_y )


        
        observation_array = [] #observation array for social force, consist of N row of agents, each row = vector (x, y, v_x, v_y, d_x, d_y, [tau])
        



        #initialize the observation vector because when starts, social force seems to require a starting vel for agents to move
        for a in range(self.n_agents):

            #if is agent itself, use goal
            if a==agent_index:      
                pos_difference = goal - np.array( [ combined_history_x[a][-1] ,   combined_history_y[a][-1]  ] )
                dist_next_waypoint = ( pos_difference / (np.linalg.norm( pos_difference ,ord=1)+0.0000001)  ) * ( pref_speed )

                vel_next_waypoint  = dist_next_waypoint

                observation_array.append( [ combined_history_x[a][-1], combined_history_y[a][-1], vel_next_waypoint[0], vel_next_waypoint[1], goal[0], goal[1]   ]  )

            #if is other agent, assume same speed to reach next position as goal
            else:
                
                pos_difference = np.array( [ combined_history_x[a][-1] ,   combined_history_y[a][-1]  ] ) - np.array( [ combined_history_x[a][-2] ,   combined_history_y[a][-2]  ] )
                dist_next_waypoint = ( pos_difference / (np.linalg.norm( pos_difference ,ord=1)+0.0000001)  ) * ( pref_speed )

                vel_next_waypoint  = dist_next_waypoint

                observation_array.append( [ combined_history_x[a][-1], combined_history_y[a][-1], vel_next_waypoint[0], vel_next_waypoint[1], combined_history_x[a][-1]+pos_difference[0], combined_history_y[a][-1]+pos_difference[1]   ]  )


        #print("goal")
        #print(agents[agent_index].goal_global_frame)

                
        
        initial_state = np.array( observation_array )

        print(initial_state)
        s=None
        #s = socialforce.Simulator(initial_state, delta_t=0.1)
        s = socialforce.Simulator(initial_state, delta_t=0.4)
        states = np.stack([s.step().state.copy() for _ in range(1)]) #step one time only

        #print("states")
        #print(states)

        next_waypoint_x = states[:, agent_index, 0][0]
        next_waypoint_y = states[:, agent_index, 1][0]
        
        next_waypoint_vel_x = states[:, agent_index, 2][0]
        next_waypoint_vel_y = states[:, agent_index, 3][0]

        self.next_waypoint = np.array( [ next_waypoint_x , next_waypoint_y ] )
        
     
        return self.next_waypoint

        #agents[agent_index].set_state( next_waypoint_x  , next_waypoint_y, next_waypoint_vel_x, next_waypoint_vel_y )
        
        #resultant_speed_global_frame         = agents[agent_index].speed_global_frame
        #resultant_delta_heading_global_frame = agents[agent_index].delta_heading_global_frame
        
        ###########################################################POSITION APPROACH##########################################################################
##        print("position")
##        print(agents[agent_index].pos_global_frame)
##        next_waypoint_x = states[:, agent_index, 0][0]
##        next_waypoint_y = states[:, agent_index, 1][0]
##
##        next_waypoint = np.array( [ next_waypoint_x, next_waypoint_y  ] )
##        print("next_waypoint")
##        print(next_waypoint)
##
##
##        
##        pos_difference = next_waypoint -  agents[agent_index].pos_global_frame    
##        dist_next_waypoint = ( pos_difference / (np.linalg.norm( pos_difference ,ord=1)+0.0000001)  ) * ( agents[agent_index].pref_speed * 0.1)
##
##        position_x = agents[agent_index].pos_global_frame[0] + dist_next_waypoint[0]
##        position_y = agents[agent_index].pos_global_frame[1] + dist_next_waypoint[1]
##        agents[agent_index].set_state( position_x , position_y )
##        
##        resultant_speed_global_frame         = agents[agent_index].speed_global_frame
##        resultant_delta_heading_global_frame = agents[agent_index].delta_heading_global_frame

        #Although documentation and code comment mentioned that action is consisted with  [heading delta, speed]
        #But in reality, the format of action is [speed, heading_delta]
        ###########################################################################################################################################

