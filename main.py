"""
Authors: Giovanni Franzese & Anna Mészáros, May 2022
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
#%%
from ILoSA import ILoSA
import time
from geometry_msgs.msg import PoseStamped
import rospy
#%%
if __name__ == '__main__':
    rospy.init_node('ILoSA', anonymous=True)
    ILoSA=ILoSA()
    ILoSA.connect_ROS()
    time.sleep(5)
    ILoSA.home_gripper()
    #%% 
    print("Recording of Nullspace contraints")
    ILoSA.Record_NullSpace()
    #%%     
    time.sleep(1)
    print("Reset to the starting cartesian position")
    ILoSA.go_to_pose(ILoSA.nullspace_traj[:, 0])    
    #%%
    time.sleep(1)
    print("Record of the cartesian trajectory")
    ILoSA.Record_Demonstration()     

    #%%
    time.sleep(1)
    print("Save the data") 
    ILoSA.save()
    #%%
    time.sleep(1)
    print("Load the data") 
    ILoSA.load()    
    #%% 
    time.sleep(1)
    print("Train the Gaussian Process Models")
    ILoSA.Train_GPs()
    ILoSA.save_models()
    ILoSA.find_alpha()
    #%%
    time.sleep(1)
    print("Reset to the starting cartesian position")
    start = PoseStamped()
    ILoSA.home_gripper()

    start.pose.position.x = ILoSA.training_traj[0,0]
    start.pose.position.y = ILoSA.training_traj[0,1]
    start.pose.position.z = ILoSA.training_traj[0,2]
    
    start.pose.orientation.w = ILoSA.training_ori[0,0] 
    start.pose.orientation.x = ILoSA.training_ori[0,1] 
    start.pose.orientation.y = ILoSA.training_ori[0,2] 
    start.pose.orientation.z = ILoSA.training_ori[0,3] 
    ILoSA.go_to_pose(start)
    #%% 
    time.sleep(1)
    print("Interactive Control Starting")
    ILoSA.Interactive_Control(verboose=False)

# %%
