"""
Authors: Giovanni Franzese & Anna Mészáros, May 2022
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
#%%
# Stuff from the ILoSA package
from ILoSA.ILoSA import ILoSA
from ILoSA.user_interfaces import KBUI # keyboard user interface

# ROS stff
from geometry_msgs.msg import PoseStamped
import rospy

# Put together the user interface and ILoSA
class My_ILoSA(ILoSA, KBUI):
    def __init__(self):
        super(My_ILoSA, self).__init__()
#%%

if __name__ == '__main__':
    rospy.init_node('ILoSA', anonymous=True)
    ILoSA=My_ILoSA()
    ILoSA.connect_ROS()
    rospy.sleep(5)
    ILoSA.home_gripper()
    #%% 
    print("Recording of Nullspace contraints")
    ILoSA.Record_NullSpace()
    #%%
    rospy.sleep(1)
    print("Record of the cartesian trajectory")
    ILoSA.Record_Demonstration()     
    
    #%% 
    rospy.sleep(1)
    print("Reset to the starting cartesian position")
    pose = ILoSA.training_traj[0]
    ori = ILoSA.training_ori[0]
    p = PoseStamped()
    p.pose.position.x = pose[0]
    p.pose.position.y = pose[1]
    p.pose.position.z = pose[2]
    p.pose.orientation.w = ori[0]
    p.pose.orientation.x = ori[1]
    p.pose.orientation.y = ori[2]
    p.pose.orientation.z = ori[3]    
    ILoSA.go_to_pose(p)    
    
    #%%
    rospy.sleep(1)
    print("Save the data") 
    ILoSA.save()
    #%%
    rospy.sleep(1)
    print("Load the data") 
    ILoSA.load()    
    #%% 
    rospy.sleep(1)
    print("Train the Gaussian Process Models")
    ILoSA.Train_GPs()
    ILoSA.save_models()
    ILoSA.find_alpha()
    #%%
    rospy.sleep(1)
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
    rospy.sleep(1)
    print("Interactive Control Starting")
    ILoSA.Interactive_Control(verboose=False)

    # %%
