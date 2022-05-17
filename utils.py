"""
Authors: Giovanni Franzese & Anna Mészáros, May 2022
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import numpy as np
def Interpret_1D(feedback, delta_, K_, delta_lim, K_mean):
    # this function interprest the feedback as an increase of delta or stiffness in the selected axis
    if np.abs(delta_+feedback) > delta_lim or K_>K_mean:
        dK_inc = K_ * (np.abs(delta_ + feedback)/delta_lim -1 )
        delta_inc = 0 
    else:
        delta_inc = feedback
        dK_inc=0

    return float(delta_inc), float(dK_inc)


# Interpret user input for altering the applied force
def Interpret_3D(feedback, delta, K, delta_lim, K_mean):

    # x-axis
    if np.abs(feedback[0]*delta_lim)>0:
        delta_inc_x, dK_inc_x = Interpret_1D(feedback[0]*delta_lim, delta[0], K[0], delta_lim, K_mean)
    else:
        delta_inc_x = 0
        dK_inc_x = 0
    # y-axis
    if np.abs(feedback[1]*delta_lim)>0:
        delta_inc_y, dK_inc_y = Interpret_1D(feedback[1]*delta_lim, delta[1], K[1], delta_lim, K_mean)
    else:
        delta_inc_y = 0
        dK_inc_y = 0
    # z-axis
    if np.abs(feedback[2]*delta_lim)>0:
        delta_inc_z, dK_inc_z = Interpret_1D(feedback[2]*delta_lim, delta[2], K[2], delta_lim, K_mean)
    else:
        delta_inc_z = 0
        dK_inc_z = 0

    delta_inc = np.array([delta_inc_x, delta_inc_y, delta_inc_z])
    dK_inc = np.array([dK_inc_x, dK_inc_y, dK_inc_z])

    return delta_inc, dK_inc

def Force2Impedance(Delta_x, K, f_stable, delta_lim):
    # This function is converting the total force vector in attractor and stiffness
    f_total= f_stable+K*Delta_x 
    #print('f_total')
    #print(f_total)
    #print('K*DeltaX')
    #print(K*Delta_x)
    if (f_total[0]>(K[0]*delta_lim)):
        Delta_x[0]=delta_lim
        K[0]=f_total[0]/Delta_x[0]
    else:
        Delta_x[0]=Delta_x[0]+f_stable[0]/K[0]

    if (f_total[1]>(K[1]*delta_lim)):
        Delta_x[1]=delta_lim
        K[1]=f_total[1]/Delta_x[1]
    else:
        Delta_x[1]=Delta_x[1]+f_stable[1]/K[1]

    if (f_total[2]>(K[2]*delta_lim)):
        Delta_x[2]=delta_lim
        K[2]=f_total[2]/Delta_x[2]
    else:
        Delta_x[2]=Delta_x[2]+f_stable[2]/K[2]
        
    return Delta_x, K

