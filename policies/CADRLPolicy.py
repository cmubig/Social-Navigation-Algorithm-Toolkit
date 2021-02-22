import numpy as np
import os

from policies.CADRL.scripts.multi import nn_navigation_value_multi as nn_nav
from policies import util
from policies.util import wrap, find_nearest
import math

# Filter list by Boolean list 
# Using itertools.compress 
from itertools import compress


class new_agent(object):
    def __init__(self):
        self.pos_global_frame     = None
        self.goal_global_frame    = None
        self.vel_global_frame     = None
        self.heading_global_frame = None
        self.radius               = None
        self.turning_dir          = None
        self.pref_speed           = None
        self.past_global_velocities = None

    def get_ref(self):
        """ Using current and goal position of agent in global frame, compute coordinate axes of ego frame.

        Ego frame is defined as: origin at center of agent, x-axis pointing from agent's center to agent's goal (right-hand rule, z axis upward).
        This is a useful representation for goal-conditioned tasks, since many configurations of agent-pos-and-goal in the global frame map to the same ego setup. 

        Returns:
        2-element tuple containing

        - **ref_prll** (*np array*): (2,) with vector corresponding to ego-x-axis (pointing from agent_position->goal)
        - **ref_orth** (*np array*): (2,) with vector corresponding to ego-y-axis (orthogonal to ref_prll)

        """
        goal_direction = self.goal_global_frame - self.pos_global_frame
        self.dist_to_goal = math.sqrt(goal_direction[0]**2 + goal_direction[1]**2)
        if self.dist_to_goal > 1e-8:
            ref_prll = goal_direction / self.dist_to_goal
        else:
            ref_prll = goal_direction
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])  # rotate by 90 deg

        #return ref_prll, ref_orth
        self.ref_prll = ref_prll
        self.ref_orth = ref_orth

class CADRLPolicy(object):
    """ Re-purposed from: Socially Aware Motion Planning with Deep Reinforcement Learning

    Loads a pre-traned SA-CADRL 4-agent network (with no social norm preference LHS/RHS).
    Some methods to convert the gym agent representation to the numpy arrays used in the old code.

    """
    def __init__(self):

        num_agents = 4
        file_dir = os.path.dirname(os.path.realpath(__file__)) + '/CADRL/scripts/multi'

        # load value_net
        # mode = 'rotate_constr'; passing_side = 'right'; iteration = 1300
        mode = 'no_constr'; passing_side = 'none'; iteration = 1000
        filename="%d_agents_policy_iter_"%num_agents + str(iteration) + ".p"
        self.value_net = nn_nav.load_NN_navigation_value(file_dir, num_agents, mode, passing_side, filename=filename, ifPrint=False)


        self.SENSING_HORIZON = np.inf
        self.MAX_NUM_OTHER_AGENTS_OBSERVED = 30
        self.DT = 0.4

    def init(self):
        pass

    

    def predict(self, history, agent_index, goal, pref_speed=1.0, radius =0.2):
        goal = np.array(goal)

        combined_history_x = history[:,:,0]
        combined_history_y = history[:,:,1]

        self.n_agents = len(combined_history_x)
        

        combined_history_x = np.array( combined_history_x )
        combined_history_y = np.array( combined_history_y )


        agents = []

        
        

        for a in range(self.n_agents):
            pos_difference = np.array( [ combined_history_x[a][-1] ,   combined_history_y[a][-1]  ] ) - np.array( [ combined_history_x[a][-2] ,   combined_history_y[a][-2]  ] )
            dist_next_waypoint = ( pos_difference / (np.linalg.norm( pos_difference ,ord=1)+0.0000001)  ) * ( pref_speed ) * self.DT

            vel_next_waypoint  = dist_next_waypoint

            agent = new_agent()

            agent.pos_global_frame = np.array( [  combined_history_x[a][-1], combined_history_y[a][-1]  ]  )
            agent.vel_global_frame = np.array( [  vel_next_waypoint[0], vel_next_waypoint[1]  ]  )

            if a == agent_index:
                agent.goal_global_frame = goal
            else:
                agent.goal_global_frame = np.array( [  combined_history_x[a][-1]+pos_difference[0], combined_history_y[a][-1]+pos_difference[1]  ]  )

            deltaPos = pos_difference #self.new_rvo_pos - self.pos_agents[agent_index,:]
            p1 = deltaPos
            p2 = np.array([1,0]) # Angle zero is parallel to x-axis
            ang1 = np.arctan2(*p1[::-1])
            ang2 = np.arctan2(*p2[::-1])
            new_heading_global_frame = (ang1 - ang2) % (2 * np.pi)
            agent.heading_global_frame = new_heading_global_frame
  
            agent.pref_speed = pref_speed
            agent.radius = radius

            agent.turning_dir = 1 #rad/s

            agent.get_ref()

            agent.past_global_velocities= np.array(    [[combined_history_x[a][-2]-combined_history_x[a][-3],combined_history_y[a][-2]-combined_history_y[a][-3]],[combined_history_x[a][-1]-combined_history_x[a][-2],combined_history_y[a][-1]-combined_history_y[a][-2]] ] )
            agent.past_global_velocities= agent.past_global_velocities * self.DT
            # turning dir: needed for cadrl value fn
##            if abs(self.agent.turning_dir) < 1e-5:
##                self.agent.turning_dir = 0.11 * np.sign(agent.heading_global_frame)
##            elif self.agent.turning_dir * selected_heading < 0:
##                self.agent.turning_dir = max(-np.pi, min(np.pi, -self.agent.turning_dir + agent.heading_global_frame))
##            else:
##                self.agent.turning_dir = np.sign(self.agent.turning_dir) * max(0.0, abs(self.agent.turning_dir)-0.1)

            agents.append( agent )

        host_agent, agent_state, other_agents_state, other_agents_actions = self.parse_agents(agents, agent_index)
        action = self.query_and_rescale_action(host_agent, agent_state, other_agents_state, other_agents_actions)
        #return action

        selected_speed = action[0]
        selected_heading = wrap(action[1] + agents[agent_index].heading_global_frame)

        dx = selected_speed * np.cos(selected_heading) * self.DT
        dy = selected_speed * np.sin(selected_heading) * self.DT

        return agents[agent_index].pos_global_frame + np.array([dx, dy])  

    def find_next_action(self, obs, agents, agent_index ):
        """ Converts environment's agents representation to CADRL format, then queries NN

        Args:
            obs (dict): ignored
            agents (list): of :class:`~gym_collision_avoidance.envs.agent.Agent` objects
            i (int): index of agents list corresponding to this agent

        Returns:
            commanded [heading delta, speed]

        """
        
        host_agent, agent_state, other_agents_state, other_agents_actions = self.parse_agents(agents, agent_index)
        action = self.query_and_rescale_action(host_agent, agent_state, other_agents_state, other_agents_actions)
        return action

    def find_next_action_and_value(self, obs, agents, i):
        """ Same as find_next_action but also queries value fn """
        host_agent, agent_state, other_agents_state, other_agents_actions = self.parse_agents(agents, i)
        action = self.query_and_rescale_action(host_agent, agent_state, other_agents_state, other_agents_actions)
        value = self.value_net.find_states_values(agent_state, other_agents_state)
        return action, value

    def parse_agents(self, agents, i):
        """ Convert from gym env representation of agents to CADRL's representation.

        Args:
            obs (dict): ignored
            agents (list): of :class:`~gym_collision_avoidance.envs.agent.Agent` objects
            i (int): index of agents list corresponding to this agent

        Returns:
            host_agent (:class:`~gym_collision_avoidance.envs.agent.Agent`): this agent
            agent_state (np array): CADRL representation of this agent's state
            other_agents_state (np array): CADRL repr. of other agents' states
            other_agents_actions (np array): CADRL repr. of other agents' current actions

        """
        host_agent = agents[i]
        other_agents = agents[:i]+agents[i+1:]
        agent_state = self.convert_host_agent_to_cadrl_state(host_agent)
        other_agents_state, other_agents_actions = self.convert_other_agents_to_cadrl_state(host_agent, other_agents)
        return host_agent, agent_state, other_agents_state, other_agents_actions

    def query_and_rescale_action(self, host_agent, agent_state, other_agents_state, other_agents_actions):
        """ If there's nobody around, just go straight to goal, otherwise query DNN and make heading action an offset from current heading

        """
        if len(other_agents_state) > 0:
            action = self.value_net.find_next_action(agent_state, other_agents_state, other_agents_actions)
            # action[0] /= host_agent.pref_speed
            action[1] = util.wrap(action[1]-host_agent.heading_global_frame)
        else:
            #action = np.array([1.0, -self.heading_ego_frame])
            action = np.array([1.0, -host_agent.heading_ego_frame])
        return action

    def convert_host_agent_to_cadrl_state(self, agent):
        """ Convert this repo's state representation format into the legacy cadrl format for the host agent 

        Args:
            agent (:class:`~gym_collision_avoidance.envs.agent.Agent`): this agent

        Returns:
            10-element (np array) describing current state

        """

        # rel pos, rel vel, size
        x = agent.pos_global_frame[0]; y = agent.pos_global_frame[1]
        v_x = agent.vel_global_frame[0]; v_y = agent.vel_global_frame[1]
        radius = agent.radius; turning_dir = agent.turning_dir
        heading_angle = agent.heading_global_frame
        pref_speed = agent.pref_speed
        goal_x = agent.goal_global_frame[0]; goal_y = agent.goal_global_frame[1]
        
        agent_state = np.array([x, y, v_x, v_y, heading_angle, pref_speed, \
            goal_x, goal_y, radius, turning_dir])

        return agent_state

    def convert_other_agents_to_cadrl_state(self, host_agent, other_agents):
        """ Convert this repo's state representation format into the legacy cadrl format
        for the other agents in the environment.

        Filtering other agents' velocities was crucial to replicate SA-CADRL results

        Args:
            host_agent (:class:`~gym_collision_avoidance.envs.agent.Agent`): this agent
            other_agents (list): of all the other :class:`~gym_collision_avoidance.envs.agent.Agent` objects

        Returns:
            - (3 x 10) np array (this cadrl can handle 3 other agents), each has 10-element state vector
            - (3 x 2) np array of other agents' filtered velocities
        """        
        # if len(other_agents) > 3:
        #     print("CADRL ISN'T DESIGNED TO HANDLE > 4 AGENTS")

        # This is a hack that CADRL was not trained to handle (only trained on 4 agents)
        other_agent_dists = []
        for i, other_agent in enumerate(other_agents):
            # project other elements onto the new reference frame
            rel_pos_to_other_global_frame = other_agent.pos_global_frame - host_agent.pos_global_frame
            p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_orth)
            dist_between_agent_centers = np.linalg.norm(rel_pos_to_other_global_frame)
            dist_2_other = dist_between_agent_centers - host_agent.radius - other_agent.radius
            if dist_between_agent_centers > self.SENSING_HORIZON:
                # print "Agent too far away"
                continue
            other_agent_dists.append([i,round(dist_2_other,2),p_orthog_ego_frame])
        sorted_dists = sorted(other_agent_dists, key = lambda x: (-x[1], x[2]))
        sorted_inds = [x[0] for x in sorted_dists]
        clipped_sorted_inds = sorted_inds[-min(self.MAX_NUM_OTHER_AGENTS_OBSERVED,3):]
        clipped_sorted_agents = [other_agents[i] for i in clipped_sorted_inds]

        agents = clipped_sorted_agents

        other_agents_state = []
        other_agents_actions = []
        for agent in agents:
            x = agent.pos_global_frame[0]; y = agent.pos_global_frame[1]
            v_x = agent.vel_global_frame[0]; v_y = agent.vel_global_frame[1]
            radius = agent.radius; turning_dir = agent.turning_dir
            # helper fields: # TODO: why are these here? these are hidden states - CADRL uses the raw agent states to convert to local representation internally
            heading_angle = agent.heading_global_frame
            pref_speed = agent.pref_speed
            goal_x = agent.goal_global_frame[0]; goal_y = agent.goal_global_frame[1]

            # experimental - filter velocities and pass as other_agents_actions
            # if np.shape(agent.global_state_history)[0] > 3:
            if True:
                past_vel = agent.past_global_velocities[-2:,:]
                dt_past_vec = self.DT*np.ones((2))
                filtered_actions_theta = util.filter_vel(dt_past_vec, past_vel)
                other_agents_actions.append(filtered_actions_theta)
            else:
                other_agents_actions = None

            other_agent_state = np.array([x, y, v_x, v_y, heading_angle, pref_speed, \
                    goal_x, goal_y, radius, turning_dir])
            other_agents_state.append(other_agent_state)
        return other_agents_state, other_agents_actions
