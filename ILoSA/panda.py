"""
Authors: Giovanni Franzese & Anna Mészáros, May 2022
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
#!/usr/bin/env python
import rospy
import math
import numpy as np
import time
import quaternion
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import Point, WrenchStamped, PoseStamped, Vector3
from franka_gripper.msg import GraspActionGoal, HomingActionGoal, StopActionGoal, MoveActionGoal
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from sys import exit

from pynput.keyboard import Listener, Key

class Panda():

    def __init__(self):
        super(Panda,self).__init__()
        self.K_ori  = 30.0
        self.K_cart = 600.0
        self.K_null = 10.0

        self.start = True
        self.end = False
        
        self.move_command=MoveActionGoal()
        self.grasp_command = GraspActionGoal()
        self.home_command = HomingActionGoal()
        self.stop_command = StopActionGoal()
        self.gripper_width = 0
        self.move_command.goal.speed=1
        self.grasp_command.goal.epsilon.inner = 0.1
        self.grasp_command.goal.epsilon.outer = 0.1
        self.grasp_command.goal.speed = 0.1
        self.grasp_command.goal.force = 5
        self.grasp_command.goal.width = 1
        # Start keyboard listener
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        # This function runs on the background and checks if a keyboard key was pressed
        if key == Key.esc:
            self.end = True

    def ee_pose_callback(self, data):
        self.cart_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.cart_ori = np.array([data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z])

    # joint angle subscriber
    def joint_callback(self, data):
        self.joint_pos = np.array(data.position[0:7])


    # gripper state subscriber
    def gripper_callback(self, data):
        self.gripper_pos = np.array(data.position[7]+data.position[8])


    # spacemouse joystick subscriber
    def teleop_callback(self, data):
        self.feedback = np.array([data.x, data.y, data.z])


    # spacemouse buttons subscriber
    def btns_callback(self, data):
        self.left_btn = data.buttons[0]
        self.right_btn = data.buttons[1]

    def move_gripper(self,width):
        self.move_command.goal.width=width
        self.move_pub.publish(self.move_command)

    def grasp_gripper(self, width):
        if width < 0.07 and self.grasp_command.goal.width != 0:
            self.grasp_command.goal.width = 0
            self.grasp_pub.publish(self.grasp_command)

        elif width > 0.07 and self.grasp_command.goal.width != 1:
            self.grasp_command.goal.width = 1
            self.grasp_pub.publish(self.grasp_command)

    def home_gripper(self):
        self.homing_pub.publish(self.home_command)

    def stop_gripper(self):
        self.stop_pub.publish(self.stop_command)    

    def connect_ROS(self):

        rospy.Subscriber("/cartesian_pose", PoseStamped, self.ee_pose_callback)
        rospy.Subscriber("/spacenav/offset", Vector3, self.teleop_callback)
        rospy.Subscriber("/spacenav/joy", Joy, self.btns_callback)
        rospy.Subscriber("/joint_states", JointState, self.joint_callback)
        rospy.Subscriber("/joint_states", JointState, self.gripper_callback)

        self.goal_pub  = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=0)
        self.stiff_pub = rospy.Publisher('/stiffness', Float32MultiArray, queue_size=0)
        self.configuration_pub = rospy.Publisher("/equilibrium_configuration",Float32MultiArray, queue_size=0)
        self.grasp_pub = rospy.Publisher("/franka_gripper/grasp/goal", GraspActionGoal,
                                           queue_size=0)
        self.move_pub = rospy.Publisher("/franka_gripper/move/goal", MoveActionGoal,
                                           queue_size=0)
        self.homing_pub = rospy.Publisher("/franka_gripper/homing/goal", HomingActionGoal,
                                          queue_size=0)
        self.stop_pub = rospy.Publisher("/franka_gripper/stop/goal", StopActionGoal,
                                          queue_size=0)
        self.stiff_ori_pub = rospy.Publisher('/stiffness_rotation', PoseStamped, queue_size=0)
        self.ellipse_pub = rospy.Publisher('/stiffness_ellipsoid', Marker, queue_size=0)

    def set_stiffness(self,pos_stiff,rot_stiff,null_stiff):
        stiff_des = Float32MultiArray()
        stiff_des.data = np.array([pos_stiff[0], pos_stiff[1], pos_stiff[2], rot_stiff[0], rot_stiff[1], rot_stiff[2], null_stiff[0]]).astype(np.float32)
        self.stiff_pub.publish(stiff_des)    

    def set_attractor(self,pos,quat):
        goal = PoseStamped()
        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = pos[0]
        goal.pose.position.y = pos[1]
        goal.pose.position.z = pos[2]

        goal.pose.orientation.w = quat[0]
        goal.pose.orientation.x = quat[1]
        goal.pose.orientation.y = quat[2]
        goal.pose.orientation.z = quat[3]
        

        self.goal_pub.publish(goal)

    def set_stiffness_ori(self,quat):

        goal = PoseStamped()
        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = 0.0
        goal.pose.position.y = 0.0
        goal.pose.position.z = 0.0

        goal.pose.orientation.w = quat[0]
        goal.pose.orientation.x = quat[1]
        goal.pose.orientation.y = quat[2]
        goal.pose.orientation.z = quat[3]
        

        self.stiff_ori_pub.publish(goal)

    def visualize_stiffness_ellipsoid(self,stiff, quat):
            marker = Marker()
            marker.header.frame_id = "panda_hand"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            # stiff=stiff/1000
            stiff = [x / 1000 for x in stiff]
            marker.scale = Vector3(stiff[0], stiff[1], stiff[2])  # Dimensions of the ellipsoid
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            # Set the position and orientation of the ellipsoid
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = quat[1]
            marker.pose.orientation.y = quat[2]
            marker.pose.orientation.z = quat[3]
            marker.pose.orientation.w = quat[0]


            # Publish the marker
            self.ellipse_pub.publish(marker)    


    def set_configuration(self,joint):
        joint_des=Float32MultiArray()
        joint_des.data= np.array(joint).astype(np.float32)
        self.configuration_pub.publish(joint_des)


    def go_to_pose(self, goal_pose):

        r=rospy.Rate(100)
        # the goal pose should be of type PoseStamped. E.g. goal_pose=PoseStampled()
        start = self.cart_pos
        start_ori=self.cart_ori
        goal_=np.array([goal_pose.pose.position.x, goal_pose.pose.position.y, goal_pose.pose.position.z])
        # interpolate from start to goal with attractor distance of approx 1 mm
        squared_dist = np.sum(np.subtract(start, goal_)**2, axis=0)
        dist = np.sqrt(squared_dist)
        # e("dist", dist)
        interp_dist = 0.001  # [m]
        step_num_lin = math.floor(dist / interp_dist)

        
        # print("num of steps linear", step_num_lin)
        
        
        q_start=np.quaternion(start_ori[0], start_ori[1], start_ori[2], start_ori[3])
        # print("q_start", q_start)
        q_goal=np.quaternion(goal_pose.pose.orientation.w, goal_pose.pose.orientation.x, goal_pose.pose.orientation.y, goal_pose.pose.orientation.z)
        inner_prod=q_start.x*q_goal.x+q_start.y*q_goal.y+q_start.z*q_goal.z+q_start.w*q_goal.w
        if inner_prod < 0:
            q_start.x=-q_start.x
            q_start.y=-q_start.y
            q_start.z=-q_start.z
            q_start.w=-q_start.w
        inner_prod=q_start.x*q_goal.x+q_start.y*q_goal.y+q_start.z*q_goal.z+q_start.w*q_goal.w
        theta= np.arccos(np.abs(inner_prod))
        # print(theta)
        interp_dist_polar = 0.001 
        step_num_polar = math.floor(theta / interp_dist_polar)

        
        # print("num of steps polar", step_num_polar)
        
        step_num=np.max([step_num_polar,step_num_lin])
        
        # print("num of steps max", step_num)
        x = np.linspace(start[0], goal_pose.pose.position.x, step_num)
        y = np.linspace(start[1], goal_pose.pose.position.y, step_num)
        z = np.linspace(start[2], goal_pose.pose.position.z, step_num)
        
        goal = PoseStamped()
        
        goal.pose.position.x = x[0]
        goal.pose.position.y = y[0]
        goal.pose.position.z = z[0]
        
        
        quat=np.slerp_vectorized(q_start, q_goal, 0.0)
        goal.pose.orientation.x = quat.x
        goal.pose.orientation.y = quat.y
        goal.pose.orientation.z = quat.z
        goal.pose.orientation.w = quat.w

        self.goal_pub.publish(goal)

        pos_stiff=[self.K_cart, self.K_cart, self.K_cart]
        rot_stiff=[self.K_ori, self.K_ori, self.K_ori]
        null_stiff=[0]
        self.set_stiffness(pos_stiff, rot_stiff, null_stiff)
        self.set_stiffness(pos_stiff, rot_stiff, null_stiff)

        goal = PoseStamped()
        for i in range(step_num):
            now = time.time()         
            goal.header.seq = 1
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = "map"

            goal.pose.position.x = x[i]
            goal.pose.position.y = y[i]
            goal.pose.position.z = z[i]
            quat=np.slerp_vectorized(q_start, q_goal, i/step_num)
            #print("quat", quat) 
            goal.pose.orientation.x = quat.x
            goal.pose.orientation.y = quat.y
            goal.pose.orientation.z = quat.z
            goal.pose.orientation.w = quat.w
            self.goal_pub.publish(goal)
            r.sleep()   

        
    def Kinesthetic_Demonstration(self, trigger=0.005): 
        self.Passive()

        self.end = False
        init_pos = self.cart_pos
        vel = 0
        print("Move robot to start recording.")
        while vel < trigger:
            vel = math.sqrt((self.cart_pos[0]-init_pos[0])**2 + (self.cart_pos[1]-init_pos[1])**2 + (self.cart_pos[2]-init_pos[2])**2)

        print("Recording started. Press Esc to stop.")

        self.recorded_traj = self.cart_pos.reshape(1,3)
        self.recorded_ori  = self.cart_ori.reshape(1,4)
        self.recorded_joint= self.joint_pos.reshape(1,7)
        self.recorded_gripper= self.gripper_pos.reshape(1,1)
        while not self.end:

            self.recorded_traj = np.vstack([self.recorded_traj, self.cart_pos])
            self.recorded_ori  = np.vstack([self.recorded_ori,  self.cart_ori])
            self.recorded_joint = np.vstack([self.recorded_joint, self.joint_pos])
            self.recorded_gripper = np.vstack([self.recorded_gripper, self.gripper_pos])

            self.r_rec.sleep()
        
            

    def Passive(self):
        pos_stiff=[0.0,0.0,0.0]
        rot_stiff=[0.0 , 0.0, 0.0] 
        null_stiff=[0.0]
        self.set_stiffness(pos_stiff, rot_stiff, null_stiff)
