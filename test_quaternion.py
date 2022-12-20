#%%
from kernel import Matern_Quaternion
import numpy as np
# %%
x1=np.array([[0.5,0.5,0.0,0.0], [0.0,1.0,0.0,0.0]]).reshape(-1,4)
x2=np.array([[-1.0, 0.0,0.0,0.0], [-0.7,0.3,0.0,0.0]]).reshape(-1,4)
k=Matern_Quaternion(1*np.ones(1), nu=1.2)
print("Covariance Matrix", k )
# %%
from data_prep import quaternion_divide, slerp_sat, quaternion_product
q_div=quaternion_divide(x1,x2)
print("Division between two arrays of quaternions", q_div )

print("Product between two arrays of quaternions", quaternion_product(q_div,x2))
#%%
print("Test on the saturated Slerp")
q1=np.array([1.0,0.0,0.0,0.0])
q2=np.array([0.0,1.0,0.0,0.0])
print("test to move from", q1,"to", q2, "with saturation", 20 , "percent")
q_slerp= slerp_sat(q1,q2, 0.2)
print("", q_slerp)
# %% check 
import quaternion
q1=np.quaternion(1,0,0,0)
q2=np.quaternion(0.5,0.5,0,0)
q2=q2/np.sqrt(q2.norm())

print("Check that my function does divisions between quaternion correctly. This is necessary to be used when computing difference between two arrays of quaternions to then fit a dynamical system")
print("Package says:",q1/q2)

print("My function says:", quaternion_divide(np.array([1,0,0,0]).reshape(1,4),np.array([0.5,0.5,0,0]).reshape(1,4)))

#%% Let's fit a Dynamical System 
import os
import matplotlib.pyplot as plt
os.getcwd()
test=np.load(os.getcwd()+'/data/data.npz')
data=np.transpose(test['recorded_ori'])
data=data[::10,:]
plt.plot(data)
Q_diff=quaternion_divide(data[1:,:],data[0:-1,:])
Q_prod=quaternion_product(Q_diff,data[0:-1,:])
plt.plot(Q_prod,'o')
# plt.legend(['w','x', 'y', 'z'])
plt.show()