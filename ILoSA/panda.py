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
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import Point, WrenchStamped, PoseStamped, Vector3
from std_msgs.msg import Float32MultiArray

from pynput.keyboard import Listener, KeyCode

class Panda():

    def __init__(self):
        self.K_ori  = 30.0
        self.K_cart = 600.0
        self.K_null = 10.0

        self.start = True
        self.end = False

        # Start keyboard listener
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        # This function runs on the background and checks if a keyboard key was pressed
        if key == KeyCode.from_char('e'):
            self.end = True

    def ee_pose_callback(self, data):
        self.cart_pos = [data.pose.position.x, data.pose.position.y, data.pose.position.z]
        self.cart_ori = [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w]

    # joint angle subscriber
    def joint_callback(self, data):
        self.joint_pos = data.position[0:7]


    # gripper state subscriber
    def gripper_callback(self, data):
        self.gripper_pos = data.position[7:9]


    # spacemouse joystick subscriber
    def teleop_callback(self, data):
        self.feedback = [data.x, data.y, data.z]


    # spacemouse buttons subscriber
    def btns_callback(self, data):
        self.left_btn = data.buttons[0]
        self.right_btn = data.buttons[1]

    def connect_ROS(self):
        rospy.init_node('ILoSA', anonymous=True)
        r=rospy.Rate(self.control_freq)

        rospy.Subscriber("/cartesian_pose", PoseStamped, self.ee_pose_callback)
        rospy.Subscriber("/spacenav/offset", Vector3, self.teleop_callback)
        rospy.Subscriber("/spacenav/joy", Joy, self.btns_callback)
        rospy.Subscriber("/joint_states", JointState, self.joint_callback)
        rospy.Subscriber("/joint_states", JointState, self.gripper_callback)

        self.goal_pub  = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=0)
        self.stiff_pub = rospy.Publisher('/stiffness', Float32MultiArray, queue_size=0)
        self.configuration_pub = rospy.Publisher("/equilibrium_configuration",Float32MultiArray, queue_size=0)

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

        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]

        self.goal_pub.publish(goal)

    def set_configuration(self,joint):
        joint_des=Float32MultiArray()
        joint_des.data= np.array(joint).astype(np.float32)
        self.configuration_pub.publish(joint_des)


        
    def go_to_3d(self,goal_):
        start = self.cart_pos
        r=rospy.Rate(self.control_freq)
        # interpolate from start to goal with attractor distance of approx 1 mm
        squared_dist = np.sum(np.subtract(start, goal_)**2, axis=0)
        dist = np.sqrt(squared_dist)
        interp_dist = 0.001  # [m]
        step_num = math.floor(dist / interp_dist)

        x = np.linspace(start[0], goal_[0], step_num)
        y = np.linspace(start[1], goal_[1], step_num)
        z = np.linspace(start[2], goal_[2], step_num)
        
        position=[x[0],y[0],z[0]]
        orientation=[1,0,0,0]
        self.set_attractor(position, orientation)

        pos_stiff=[self.K_cart, self.K_cart, self.K_cart]
        rot_stiff=[self.K_ori, self.K_ori, self.K_ori]
        null_stiff=[0]
        self.set_stiffness(pos_stiff, rot_stiff, null_stiff)

        # send attractors to controller
        for i in range(step_num):
            position=[x[i],y[i],z[i]]
            orientation=[1,0,0,0]
            self.set_attractor(position,orientation)
            r.sleep()

    def Kinesthetic_Demonstration(self, trigger=0.005): 
        r=rospy.Rate(self.rec_freq)
        self.Passive()

        self.end = False
        init_pos = self.cart_pos
        vel = 0
        print("Move robot to start recording.")
        while vel < trigger:
            vel = math.sqrt((self.cart_pos[0]-init_pos[0])**2 + (self.cart_pos[1]-init_pos[1])**2 + (self.cart_pos[2]-init_pos[2])**2)

        print("Recording started. Press e to stop.")

        self.recorded_traj = self.cart_pos
        self.recorded_joint= self.joint_pos
        while not self.end:

            self.recorded_traj = np.c_[self.recorded_traj, self.cart_pos]
            self.recorded_joint = np.c_[self.recorded_joint, self.joint_pos]
            r.sleep()
            

    def Passive(self):
        pos_stiff=[0.0,0.0,0.0]
        rot_stiff=[self.K_ori , self.K_ori , self.K_ori ] 
        null_stiff=[0.0]
        self.set_stiffness(pos_stiff, rot_stiff, null_stiff)
