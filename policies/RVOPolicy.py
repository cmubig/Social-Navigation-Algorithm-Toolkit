import numpy as np

import rvo2


# Filter list by Boolean list 
# Using itertools.compress 
from itertools import compress

class RVOPolicy(object):
    def __init__(self):

        self.DT = 0.4 #Config.DT
        self.neighbor_dist = np.inf
        self.max_neighbors = 30

        self.RVO_COLLAB_COEFF = 0.5

        self.has_fixed_speed = False
        self.heading_noise = False

        self.max_delta_heading = np.pi/6
        
        # TODO share this parameter with environment
        self.time_horizon = 20.0 # NOTE: bjorn used 1.0 in training for corl19
        # Initialize RVO simulator
        self.sim = rvo2.PyRVOSimulator(timeStep=self.DT, neighborDist=self.neighbor_dist, 
            maxNeighbors=self.max_neighbors, timeHorizon=self.time_horizon, 
            timeHorizonObst=self.time_horizon, radius=0.0, 
            maxSpeed=0.0)

        self.is_init = False

        self.use_non_coop_policy = True

    def init(self):
        pass
    

    def setup(self):
        self.sim = None
        self.sim = rvo2.PyRVOSimulator(timeStep=self.DT, neighborDist=self.neighbor_dist, 
            maxNeighbors=self.max_neighbors, timeHorizon=self.time_horizon, 
            timeHorizonObst=self.time_horizon, radius=0.0, 
            maxSpeed=0.0)
        state_dim = 2
        self.pos_agents = np.empty((self.n_agents, state_dim))
        self.vel_agents = np.empty((self.n_agents, state_dim))
        self.goal_agents = np.empty((self.n_agents, state_dim))
        self.pref_vel_agents = np.empty((self.n_agents, state_dim))
        self.pref_speed_agents = np.empty((self.n_agents))
        
        self.rvo_agents = [None]*self.n_agents

        # Init simulation
        for a in range(self.n_agents):
            self.rvo_agents[a] = self.sim.addAgent((0,0))
        
        self.is_init = True

    def predict(self, history, agent_index, goal, pref_speed=1.0, dt=None, radius=0.2):
        if dt is not None: self.DT = dt
        goal= np.array(goal)

        combined_history_x = history[:,:,0]
        combined_history_y = history[:,:,1]

        self.n_agents = len(combined_history_x)
        

        combined_history_x = np.array( combined_history_x )
        combined_history_y = np.array( combined_history_y )

        self.setup()
        
        # Share all agent positions and preferred velocities from environment with RVO simulator
        for a in range(self.n_agents):
            # Copy current agent positions, goal and preferred speeds into np arrays
            self.pos_agents[a,:]  = np.array( [ combined_history_x[a][-1], combined_history_y[a][-1] ]  )
            self.vel_agents[a,:]  = (np.array( [ combined_history_x[a][-1], combined_history_y[a][-1] ]  ) - np.array( [ combined_history_x[a][-2], combined_history_y[a][-2] ]  ) ) 
            
            if a == agent_index:
                self.goal_agents[a,:] = goal
                self.pref_speed_agents[a] = pref_speed
            else:
                self.goal_agents[a,:] = np.array( [ combined_history_x[a][-1], combined_history_y[a][-1] ]  ) + self.vel_agents[a,:]
                self.pref_speed_agents[a] = np.linalg.norm(self.vel_agents[a,:])
                
            

            # Calculate preferred velocity
            # Assumes non RVO agents are acting like RVO agents
            self.pref_vel_agents[a,:] = self.goal_agents[a,:] - self.pos_agents[a,:]
            self.pref_vel_agents[a,:] = self.pref_speed_agents[a] / np.linalg.norm(self.pref_vel_agents[a,:]) * self.pref_vel_agents[a,:]

            # Set agent positions and velocities in RVO simulator
            self.sim.setAgentMaxSpeed(self.rvo_agents[a], pref_speed)
            self.sim.setAgentRadius(self.rvo_agents[a], (1+5e-2)* radius)
            self.sim.setAgentPosition(self.rvo_agents[a], tuple(self.pos_agents[a,:]))
            self.sim.setAgentVelocity(self.rvo_agents[a], tuple(self.vel_agents[a,:]))
            self.sim.setAgentPrefVelocity(self.rvo_agents[a], tuple(self.pref_vel_agents[a,:]))

##            print("self.pos_agents[a,:])")
##            print(self.pos_agents[a,:])
##            print("tuple(self.vel_agents[a,:])")
##            print(tuple(self.vel_agents[a,:]))
##            print("tuple(self.pref_vel_agents[a,:])")
##            print(tuple(self.pref_vel_agents[a,:]))


        # Execute one step in the RVO simulator
        self.sim.doStep()

        # Calculate desired change of heading
        self.new_rvo_pos = self.sim.getAgentPosition(self.rvo_agents[agent_index])[:]

        return np.array(self.new_rvo_pos)


        '''
        deltaPos = self.new_rvo_pos - self.pos_agents[agent_index,:]
        p1 = deltaPos
        p2 = np.array([1,0]) # Angle zero is parallel to x-axis
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        new_heading_global_frame = (ang1 - ang2) % (2 * np.pi)
        delta_heading = wrap(new_heading_global_frame - agents[agent_index].heading_global_frame)
            
        # Calculate desired speed
        pref_speed = 1/self.dt * np.linalg.norm(deltaPos)

        # Limit the turning rate: stop and turn in place if exceeds
        if abs(delta_heading) > self.max_delta_heading:
            delta_heading = np.sign(delta_heading)*self.max_delta_heading
            pref_speed = 0.

        # Ignore speed
        if self.has_fixed_speed:
            pref_speed = self.max_speed

        # Add noise
        if self.heading_noise:
            delta_heading = delta_heading + np.random.normal(0,0.5)

        action = np.array([pref_speed, delta_heading])

##        print("="*40)
##        print("obs")
##        print(type(obs))
##        print(obs)
##        print("="*40)
##        print("action")
##        print(type(action))
##        print(action)
##        print("="*40)
        
        return action
        '''
