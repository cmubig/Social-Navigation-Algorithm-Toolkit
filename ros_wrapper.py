import rospy

import sys
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
                rospy.Rate(100)#30HZ
                rospy.loginfo("Starting node " + str(node_name))
                rospy.on_shutdown(self.cleanup)

                self.waypoint_pub = rospy.Publisher('/way_point',  PointStamped, queue_size=10)
                self.peds_sub     = rospy.Subscriber("/tracks", TrackArray, self.track_callback, queue_size=1)

                self.goal_sub     = rospy.Subscriber("/goal", PointStamped,  self.goal_callback, queue_size=1)

                self.timestamp = None
                self.old_odom_msg  = None
                self.goal = None

        def goal_callback(self, data):

            goal_x = data.pose.position.x
            goal_y = data.pose.position.y

            self.goal = [goal_x,goal_y]
            


        def track_callback(self, data):

            if self.timestamp is None:
                self.old_timestamp = data.header.stamp
                self.old_odom_msg  = rospy.wait_for_message('/integrated_to_init', Odometry)
                return

            

            ped_position_list = [ track.pos for track in data.tracks ]
            ped_velocity_list = [ track.vel for track in data.tracks ]

            ped_len = len(data.tracks)

            ped_history_list = []

            self.odom_msg = rospy.wait_for_message('/integrated_to_init', Odometry)

            dt = (rospy.Time.now() - self.old_timestamp).to_sec()

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
            robot_pos_now_x    = self.odom_msg.pose.pose.position.x
            robot_pos_now_y    = self.odom_msg.pose.pose.position.y

            robot_pos_old_x    = self.old_odom_msg.pose.pose.position.x  
            robot_pos_old_y    = self.old_odom_msg.pose.pose.position.y
            
            robot_pos_older_x  = self.old_odom_msg.pose.pose.position.x - (robot_pos_now_x - robot_pos_old_x)
            robot_pos_older_y  = self.old_odom_msg.pose.pose.position.y - (robot_pos_now_y - robot_pos_old_y)

            robot_data = [[0,0], [0,0], [0,0], [0,0], [0,0], [ robot_pos_older_x , robot_pos_older_y], [ robot_pos_old_x , robot_pos_old_y ], [ robot_pos_now_x , robot_pos_now_y ]]

            #add/insert robot history to the ped history [ robot, agent_0, agent_1, agent_2 ]
            ped_history_list.insert( 0 , robot_data)
            
            self.timestamp = rospy.Time.now()
            self.old_odom_msg = self.odom_msg

            if self.goal is None: return

            #start feeding the data to algorithms
            data = np.array( ped_history_list )
            goal = np.array(self.goal)
            result = policy.predict(data, 0, goal, pref_speed= 1.0, dt=dt)

            
            waypoint = PoseStamped()
            waypoint.header.stamp  = rospy.Time.now()
            waypoint.pose.position.x = result[0]
            waypoint.pose.position.y = result[1]

            self.waypoint_pub( waypoint )


        def cleanup(self):
                print("Shutting down social_navigator node.")
                rospy.signal_shutdown("social_navigator Shutdown")
                

def main(args):    
        try:
                node_name = "social_navigator"
                social_navigator(node_name)
                rospy.spin()
        
        except KeyboardInterrupt:
                print("Shutting down social_navigator node.")

if __name__ == '__main__':
        main(sys.argv)



