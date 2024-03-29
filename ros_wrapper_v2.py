import rospy

import sys
import os
print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))





from std_msgs.msg import String, Float64
from geometry_msgs.msg import Point, Vector3, PointStamped, Pose, PoseStamped
from nav_msgs.msg import Odometry
from hdl_people_tracking.msg import Track, TrackArray

import numpy as np

from policies.CADRLPolicy import CADRLPolicy
policy = CADRLPolicy()
policy.init()


class social_navigator(object):
        
        def __init__(self, node_name): 
       
                self.node_name = node_name
                rospy.init_node(node_name)
                #rospy.Rate(100)#30HZ
                rospy.loginfo("Starting node " + str(node_name))
                rospy.on_shutdown(self.cleanup)

                self.waypoint_pub = rospy.Publisher('/way_point',  PointStamped, queue_size=10)
                self.peds_sub     = rospy.Subscriber("/hdl_people_tracking/tracks", TrackArray, self.track_callback, queue_size=1)

                self.goal_sub     = rospy.Subscriber("/nav_goal", PointStamped,  self.goal_callback, queue_size=1)

                self.old_timestamp = None
                self.old_odom_msg  = None
                self.goal = None

        def goal_callback(self, data):

            goal_x = data.point.x
            goal_y = data.point.y

            self.goal = [goal_x,goal_y]
            


        def track_callback(self, data):

            if self.old_timestamp is None:
                self.old_timestamp = data.header.stamp
                self.old_odom_msg  = rospy.wait_for_message('/integrated_to_init', Odometry)
                return

            self.data = data

            

            


        def cleanup(self):
                print("Shutting down social_navigator node.")
                rospy.signal_shutdown("social_navigator Shutdown")
                

   

node_name = "social_navigator"
social_navigator_node = social_navigator(node_name)

while not rospy.is_shutdown():
    try:
            ped_position_list = [ track.pos for track in social_navigator_node.data.tracks ]
            ped_velocity_list = [ track.vel for track in social_navigator_node.data.tracks ]

            ped_len = len(social_navigator_node.data.tracks)

            ped_history_list = []

            social_navigator_node.odom_msg = rospy.wait_for_message('/integrated_to_init', Odometry)

            dt = (rospy.Time.now() - social_navigator_node.old_timestamp).to_sec()

            #create the history for ped position
            for ped_id in range(ped_len):
                pos_now_x = ped_position_list[ped_id].x
                pos_now_y = ped_position_list[ped_id].y              

                pos_old_x = ped_position_list[ped_id].x - ped_velocity_list[ped_id].x * dt
                pos_old_y = ped_position_list[ped_id].y - ped_velocity_list[ped_id].y * dt

                pos_older_x = ped_position_list[ped_id].x - ped_velocity_list[ped_id].x * (2*dt)
                pos_older_y = ped_position_list[ped_id].y - ped_velocity_list[ped_id].y * (2*dt)

                ped_data = [[0,0], [0,0], [0,0], [0,0], [0,0],[ pos_older_x , pos_older_y], [ pos_old_x , pos_old_y ], [ pos_now_x , pos_now_y ]]

                ped_history_list.append(ped_data)

            #create the history for robot position
            robot_pos_now_x    = social_navigator_node.odom_msg.pose.pose.position.x
            robot_pos_now_y    = social_navigator_node.odom_msg.pose.pose.position.y

            robot_pos_old_x    = social_navigator_node.old_odom_msg.pose.pose.position.x  
            robot_pos_old_y    = social_navigator_node.old_odom_msg.pose.pose.position.y
            
            robot_pos_older_x  = social_navigator_node.old_odom_msg.pose.pose.position.x - (robot_pos_now_x - robot_pos_old_x)
            robot_pos_older_y  = social_navigator_node.old_odom_msg.pose.pose.position.y - (robot_pos_now_y - robot_pos_old_y)

            robot_data = [[0,0], [0,0], [0,0], [0,0], [0,0], [ robot_pos_older_x , robot_pos_older_y], [ robot_pos_old_x , robot_pos_old_y ], [ robot_pos_now_x , robot_pos_now_y ]]

            #add/insert robot history to the ped history [ robot, agent_0, agent_1, agent_2 ]
            ped_history_list.insert( 0 , robot_data)
            
            social_navigator_node.timestamp = rospy.Time.now()
            social_navigator_node.old_odom_msg = social_navigator_node.odom_msg

            if social_navigator_node.goal is None: continue

            #start feeding the data to algorithms
            data = np.array( ped_history_list )
            goal = np.array(social_navigator_node.goal)
            result = policy.predict(data, 0, goal, pref_speed= 1.0, dt=0.4) #dt

            
            waypoint = PointStamped()
            waypoint.header.stamp  = rospy.Time.now()
            waypoint.point.x = result[0]
            waypoint.point.y = result[1]

            social_navigator_node.waypoint_pub( waypoint )
    except:
            print("ERROR")

    




