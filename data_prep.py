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

# Learn more about quaternions:
#  https://personal.utdallas.edu/~sxb027100/dock/quaternion.html#:~:text=order%20of%20multiplication.-,The%20inverse%20of%20a%20quaternion%20refers%20to%20the%20multiplicative%20inverse,for%20any%20non%2Dzero%20quaternion.
def quaternion_divide(q1, q2):
    """Divide quaternions q1/q2 = q1 * q2.inverse.
    Be careful (!) q=[w,x,y,z] """
    q1=q1/np.sqrt(np.sum(q1**2, axis=1)).reshape(-1,1)
    q2=q2/np.sqrt(np.sum(q2**2, axis=1)).reshape(-1,1)
    mask=np.diag(np.inner(q1,q2))<0
    q2[mask,:]=-q2[mask,:]
    q2norm = np.sqrt(q2[:,0]**2 + q2[:,1]**2 + q2[:,2]**2 + q2[:,3]**2)
    q2=q2/q2norm.reshape(-1,1)
    a = (+q1[:,0]*q2[:,0] + q1[:,1]*q2[:,1] + q1[:,2]*q2[:,2] + q1[:,3]*q2[:,3])# / q2norm
    b = (-q1[:,0]*q2[:,1] + q1[:,1]*q2[:,0] - q1[:,2]*q2[:,3] + q1[:,3]*q2[:,2])# / q2norm
    c = (-q1[:,0]*q2[:,2] + q1[:,1]*q2[:,3] + q1[:,2]*q2[:,0] - q1[:,3]*q2[:,1])# / q2norm
    d = (-q1[:,0]*q2[:,3] - q1[:,1]*q2[:,2] + q1[:,2]*q2[:,1] + q1[:,3]*q2[:,0])# / q2norm
    qout=np.copy(q1)
    qout[:,0]=a
    qout[:,1]=b
    qout[:,2]=c
    qout[:,3]=d

    return qout

def quaternion_product(q1, q2):
    """Multiply quaternions q1 * q2. If you are fitting a DS, q1 is the delta and q2 is the current quaternion .
    Be careful (!) q=[w,x,y,z] """
    q1=q1/np.sqrt(np.sum(q1**2, axis=1)).reshape(-1,1)
    q2=q2/np.sqrt(np.sum(q2**2, axis=1)).reshape(-1,1)
    mask=np.diag(np.inner(q1,q2))<0
    q2[mask,:]=-q2[mask,:]
    q2norm = np.sqrt(q2[:,0]**2 + q2[:,1]**2 + q2[:,2]**2 + q2[:,3]**2)
    q2=q2/q2norm.reshape(-1,1)
    a = (q1[:,0]*q2[:,0]  - q1[:,1]*q2[:,1] - q1[:,2]*q2[:,2] - q1[:,3]*q2[:,3])
    b = (q1[:,0]*q2[:,1]  + q1[:,1]*q2[:,0] + q1[:,2]*q2[:,3] - q1[:,3]*q2[:,2])
    c = (q1[:,0]*q2[:,2]  - q1[:,1]*q2[:,3] + q1[:,2]*q2[:,0] + q1[:,3]*q2[:,1])
    d = (q1[:,0]*q2[:,3]  + q1[:,1]*q2[:,2] - q1[:,2]*q2[:,1] + q1[:,3]*q2[:,0])
    qout=np.copy(q1)
    qout[:,0]=a
    qout[:,1]=b
    qout[:,2]=c
    qout[:,3]=d

    return qout   

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
    if inner<0:
        q2=-q2
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

