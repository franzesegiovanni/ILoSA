"""
Authors: Giovanni Franzese & Anna Mészáros, May 2022
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
#%%
from ILoSA import ILoSA
import time
import rospy
#%%
if __name__ == '__main__':
    rospy.init_node('ILoSA', anonymous=True)
    ILoSA=ILoSA()
    ILoSA.connect_ROS()
    time.sleep(5)
    # ILoSA.home_gripper()
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
    ILoSA.go_to_start()
    #%% 
    time.sleep(1)
    print("Interactive Control Starting")
    ILoSA.Interactive_Control()