"""
Authors: Giovanni Franzese & Anna Mészáros, May 2022
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
#%%
import ILoSA
#import ILoSA  
import time
#%%
if __name__ == '__main__':
    ILoSA=ILoSA()
    ILoSA.connect_ROS()
    time.sleep(5)
    #%% 
    print("Recording of Nullspace contraints")
    ILoSA.Record_NullSpace()
    #%%     
    time.sleep(1)
    print("Reset to the starting cartesian position")
    ILoSA.go_to_3d(ILoSA.nullspace_traj[:, 0])    
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
    #%%  
    time.sleep(1)
    print("Reset to the starting cartesian position")
    ILoSA.go_to_3d(ILoSA.training_traj[:, 0])

    #%% 
    time.sleep(1)
    print("Interactive Control Starting")
    ILoSA.Interactive_Control(verboose=False)

# %%
