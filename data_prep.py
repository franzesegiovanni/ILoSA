#!/usr/bin/env python
"""
Authors: Giovanni Franzese & Anna Mészáros, May 2022
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import rospy
import math
import numpy as np
import time
import scipy
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation
import pathlib


# Resample and save movement data from demonstration
# step: number of timesteps between elements used for computing the attractor distance; can be adjusted for regulating the velocity
def resample(recorded_traj, step):
    resampled_delta = np.subtract(recorded_traj[:, step:], recorded_traj[:, :-step])
    for i in range(step):
        resampled_delta = np.concatenate((resampled_delta, (np.subtract(recorded_traj[:, -1], recorded_traj[:, -step+i])).reshape(3,1)),axis=1)

    return resampled_delta


# Extract gripper width
def extract_grip_width(gripper_traj):
    recorded_width = np.add(gripper_traj[:, 0], gripper_traj[:, 1])
    return recorded_width



# Extract euler angles from quaternion
def extract_euler(quat):
    
    rot = Rotation.from_quat(quat[0])
    eul_ang = np.array(rot.as_euler('xyz')).reshape(1,3)
    
    for j in range(1,len(quat)):
        rot = Rotation.from_quat(quat[j])
        eul_ang = np.concatenate((eul_ang, np.array(rot.as_euler('xyz')).reshape(1,3)), axis=0)
        
    sin_eul = np.sin(eul_ang)
    cos_eul = np.cos(eul_ang)

    ang_data = np.c_[sin_eul, cos_eul]
    return ang_data

def slerp_sat(q1, q2, theta_max_perc): 
    '''
    This function goes to q2 from q1 but with set maximum theta
    '''
    theta_max=theta_max_perc*np.pi/2
    # if np.shape(q1)!=(1,4) or np.shape(q2)!=(1,4):
    #     print("Wrong dimensions of q1 or q2")
    q1=q1.reshape(4)
    q2=q2.reshape(4)
    q1=q1/np.sqrt(np.sum(q1**2))
    q2=q2/np.sqrt(np.sum(q2**2))
    inner=np.inner(q1,q2)
    # if inner<0:
    #     q2=-q2
    theta= np.arccos(np.abs(inner)) 
    q_slerp=np.copy(q2)
    # print("Theta",theta)
    if theta>theta_max:
        #print('out_of_bounds')
        #if 
        q_slerp[0]=(np.sin(theta-theta_max)*q1[0]+np.sin(theta_max)*q2[0])/np.sin(theta)
        q_slerp[1]=(np.sin(theta-theta_max)*q1[1]+np.sin(theta_max)*q2[1])/np.sin(theta)
        q_slerp[2]=(np.sin(theta-theta_max)*q1[2]+np.sin(theta_max)*q2[2])/np.sin(theta)
        q_slerp[3]=(np.sin(theta-theta_max)*q1[3]+np.sin(theta_max)*q2[3])/np.sin(theta)
    return q_slerp 