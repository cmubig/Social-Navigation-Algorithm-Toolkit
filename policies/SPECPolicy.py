import os
import math
import sys
import torch

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#sys.path.insert(1, "../envs/policies/SPEC/sgan/")

#print(os.getcwd())

sys.path.append("./policies/SPEC/sgan/")

import scnn.model as model


import glob
import torch.distributions.multivariate_normal as torchdist





#print(os.getcwd())

import copy
import argparse

# Filter list by Boolean list 
# Using itertools.compress 
from itertools import compress

class parameters():
    def __init__(self):

        '''
        parser.add_argument('--dataset_name', default='zara1', type=str)
        parser.add_argument('--delim', default='\t')
        parser.add_argument('--loader_num_workers', default=4, type=int)
        parser.add_argument('--min_ped', default=2, type=int)
        parser.add_argument('--hist_len', default=8, type=int)
        parser.add_argument('--fut_len', default=12, type=int)
        parser.add_argument('--loadNpy', default=1, type=int)
        parser.add_argument('--untracked_ratio', default=1.0, type=float)
        '''
        self.dataset_name        = 'zara1'
        self.delim               = '\t'
        self.loader_num_workers  = 4
        self.min_ped             = 2
        self.hist_len            = 8
        self.fut_len             = 12
        self.loadNpy             = 1
        self.untracked_ratio     = 1.0
        '''
        # Network design
        parser.add_argument('--l2d', default=1, type=int)
        parser.add_argument('--tanh', default=0, type=int)
        parser.add_argument('--n_ch', default=2, type=int)
        parser.add_argument('--use_max', default=1, type=int)
        parser.add_argument('--targ_ker_num', default=[], type=list) # [7,28]
        parser.add_argument('--targ_ker_size', default=[], type=list)
        parser.add_argument('--targ_pool_size', default=[2,2], type=list)
        parser.add_argument('--cont_ker_num', default=[-1], type=list) # [17,72]
        parser.add_argument('--cont_ker_size', default=[2,2], type=list)
        parser.add_argument('--cont_pool_size', default=[2,1], type=list)
        parser.add_argument('--n_fc', default=-1, type=int)
        parser.add_argument('--fc_width', default=[20], type=list) # 280,200,120,80
        parser.add_argument('--output_size', default=5, type=int)
        parser.add_argument('--neighbor', default=1, type=int)
        parser.add_argument('--drop_rate', default=0.0, type=float)
        parser.add_argument('--lock_l2d', default=0, type=int)
        '''
        self.l2d                =1
        self.tanh               =0
        self.n_ch               =2
        self.use_max            =1
        self.targ_ker_num       =[]
        self.targ_ker_size      =[]
        self.targ_pool_size     =[2,2]
        self.cont_ker_num       =[-1]
        self.cont_ker_size      =[2,2]
        self.cont_pool_size     =[2,1]
        self.n_fc               =-1
        self.fc_width           =[20]
        self.output_size        =5
        self.neighbor           =1
        self.drop_rate          =0.0
        self.lock_l2d           =0
        '''
        # Training
        parser.add_argument('--loadModel', default='', type=str)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--n_epoch', default=1000, type=int)
        parser.add_argument('--n_iteration', default=300, type=int)
        parser.add_argument('--lr', default=0.0005, type=float)
        parser.add_argument('--start', default=0, type=int)
        '''
        self.loadModel     = ''
        self.batch_size    = 64
        self.n_epoch       = 1000
        self.n_iteration   = 300
        self.lr            = 0.0005
        self.start         = 0
        
        '''
        # Validation and Output
        parser.add_argument('--batch_size_val', default=2, type=int)
        parser.add_argument('--batch_size_tst', default=2, type=int)
        parser.add_argument('--n_batch_val', default=6, type=int)
        parser.add_argument('--n_batch_tst', default=4, type=int)
        parser.add_argument('--val_freq', default=1, type=int)
        parser.add_argument('--n_guess', default=2, type=int)
        parser.add_argument('--n_sample', default=20, type=int)
        parser.add_argument('--coef', default=1.000000001, type=float)
        '''
        self.batch_size_val  = 2
        self.batch_size_tst  = 2
        self.n_batch_val     = 6
        self.n_batch_tst     = 4
        self.val_freq        = 1
        self.n_guess         = 2
        self.n_sample        = 20
        self.coef            = 1.000000001

        print(os.getcwd())
        
        
class SPECPolicy(object):
    def __init__(self):

        self.dt = 0.4 #Config.DT
        self.obs_seq_len = 8
        self.near_goal_threshold = 0.5
        
        self.is_init = False

        self.model = np.load("./policies/SPEC/sgan/univ_best_1.npy",allow_pickle=True)[0]


        self.args = parameters()
            

    def init(self):
        
        self.is_init = True

    def predict(self, history, agent_index):

        combined_history_x = history[:,:,0]
        combined_history_y = history[:,:,1]

        self.n_agents = len(combined_history_x)
        

        combined_history_x = np.array( combined_history_x )
        combined_history_y = np.array( combined_history_y )
        
        observation_x_input = combined_history_x
        observation_y_input = combined_history_y

        
        if observation_x_input.shape[0]==1: return np.array([0,0])    


        ####FOR Observation input, its shape is [20, num_agents, 3(agent_id,x,y) ]
        #####HOWEVER, the target agent have to be the last within the timestamp array, e.g:  for 1,2,3 agents, if agent 2 is target agent, then for timestamp x, the array is [ [1,x,y],[3,x,y],[2,x,y] ]

        ####Basically, remove the target agent from the array and append to the end of the array

        observation_input = []
        
        for time_ind in range(self.obs_seq_len):
            temp = []
            for agent_ind in range(self.n_agents):
                temp.append([ observation_x_input[agent_ind][time_ind], observation_y_input[agent_ind][time_ind] ])
            observation_input.append( temp  )
            
        
        data = torch.from_numpy(np.transpose(np.array( observation_input ).astype(np.float32), (1, 2, 0)))
        #fut = model.LocPredictor(self.args).predictTraj(data)
        fut = self.model.predictTraj(data.to("cuda"))
        prediction = np.transpose(fut.detach().cpu().numpy() , ( 0,2,1 ) )
        #print("FULL observation_input")
        #print(np.transpose(np.array( observation_input ).astype(np.float32), (1, 2, 0)))
        #print("FULL prediction")
        #print(prediction)
        #print("observation_input")
        #print(np.transpose(np.array( observation_input ).astype(np.float32), (1, 2, 0))[0])

        return prediction[agent_index]
        
        '''
        prediction_index = 5 #3 better in 10x10 #2 original test
        self.next_waypoint = prediction[agent_index][prediction_index] #agents[agent_index].pos_global_frame + prediction[agent_index][prediction_index]
        '''
        #print(next_waypoint)

        '''
        goal_direction = self.next_waypoint - agents[agent_index].pos_global_frame
        self.dist_to_goal = math.sqrt(goal_direction[0]**2 + goal_direction[1]**2)
        if self.dist_to_goal > 1e-8:
            ref_prll = goal_direction / agents[agent_index].dist_to_goal
        else:
            ref_prll = goal_direction
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])  # rotate by 90 deg

        ref_prll_angle_global_frame = np.arctan2(ref_prll[1],
                                                 ref_prll[0])
        heading_ego_frame = wrap( agents[agent_index].heading_global_frame -
                                      ref_prll_angle_global_frame)

    

        vel_global_frame = (( (prediction[agent_index][prediction_index])/(prediction_index+1))/4) / agents[agent_index].dt_nominal

        speed_global_frame = np.linalg.norm(vel_global_frame) 
        print("calc speed")
        print(speed_global_frame)
        #if speed_global_frame > agents[agent_index].pref_speed: speed_global_frame = agents[agent_index].pref_speed

        if speed_global_frame > 1.5: speed_global_frame = 1.5
        if speed_global_frame < 0.5: speed_global_frame = 0.5

        #But in reality, the format of action is [speed, heading_delta]

        action = np.array([speed_global_frame, -heading_ego_frame])
        print("action")
        print(action)
       
        return action
        '''

