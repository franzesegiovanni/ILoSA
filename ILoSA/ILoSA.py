"""
Authors: Giovanni Franzese & Anna Mészáros, May 2022
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
#!/usr/bin/env python
import numpy as np
import pandas as pd
from ILoSA.gaussian_process import * 
from ILoSA.panda import * 
from ILoSA.utils import * 
from ILoSA.data_prep import * 
import pickle
# class for storing different data types into one variable
class Struct:
    pass

class ILoSA(Panda):
    def __init__(self, *args, **kwargs):
        super(ILoSA,self).__init__(*args, **kwargs)
        self.rec_freq = 10  # [Hz]

        # stiffness parameters
        self.K_min = 0.0
        self.K_max = 2000.0
        self.K_mean = 600
        self.dK_min = 0.0
        self.dK_max = self.K_max-self.K_mean
        self.K_null=5
        # maximum attractor distance along each axis
        self.attractor_lim = 0.04
        self.scaling_factor=1
        self.scaling_factor_ns=1
        # user-provided teleoperation input
        self.feedback = [0, 0, 0]
        # uncertainty threshold at which new points are added
        self.theta = 0.4
        # uncertainty threshold at which stiffness is automatically reduced
        self.theta_stiffness = 0.99
        self.theta_nullspace= 0
        # training data initialisation
        self.training_traj = []
        self.training_delta = []
        self.training_dK = []
        self.nullspace_traj=[]
        self.nullspace_joints=[]
        
        # maximum force of the gradient
        self.max_grad_force = 20

        self.NullSpaceControl=None

    def Record_NullSpace(self):
        self.Kinesthetic_Demonstration()
        print('Recording ended.')
        save_demo = self.save_demo_user_input() 
        print('save demo: ', save_demo)
        '''
        if save_demo.lower()=='y':
            if len(self.nullspace_traj)==0:
                self.nullspace_traj=np.zeros((3,1))
                self.nullspace_joints=np.zeros((7,1))
                self.nullspace_traj=np.concatenate((self.nullspace_traj,self.recorded_traj.transpose() ), axis=1)
                self.nullspace_joints=np.concatenate((self.nullspace_joints,self.recorded_joint.transpose() ), axis=1)
                self.nullspace_traj=np.delete(self.nullspace_traj, 0,1)
                self.nullspace_joints=np.delete(self.nullspace_joints,0,1)
            else:
                self.nullspace_traj=np.concatenate((self.nullspace_traj,self.recorded_traj ), axis=1)
                self.nullspace_joints=np.concatenate((self.nullspace_joints,self.recorded_joint ), axis=1)

            print("Demo Saved")
        else:
            print("Demo Discarded")
        '''

    def Record_Demonstration(self):
        self.Kinesthetic_Demonstration()
        print('Recording ended.')
        save_demo = input("Do you want to keep this demonstration? [y/n] \n")
        if save_demo.lower()=='y':
            if len(self.training_traj)==0:
                self.training_traj=np.zeros((1,3))
                self.training_delta=np.zeros((1,3))
                self.training_dK=np.zeros((1,3))
                self.training_ori=np.zeros((1,4))
                self.training_gripper=np.zeros((1,1))
                
                self.recorded_traj=self.recorded_traj.transpose()
                self.recorded_ori=self.recorded_ori.transpose()
                self.recorded_joint=self.recorded_joint.transpose()
                self.recorded_gripper=self.recorded_gripper.transpose()

                self.training_traj=np.concatenate((self.training_traj,self.recorded_traj ), axis=0)
                self.training_ori=np.concatenate((self.training_ori,self.recorded_ori), axis=0)
                self.training_gripper=np.concatenate((self.training_gripper,self.recorded_gripper), axis=0)

                delta_x=resample(self.recorded_traj, step=2)
                self.training_delta=np.concatenate((self.training_delta,delta_x), axis=0)

                self.training_dK=np.concatenate((self.training_dK,np.zeros(np.shape(self.recorded_traj))), axis=0)
                
                
                self.training_traj=np.delete(self.training_traj, 0,axis=0)
                self.training_ori=np.delete(self.training_ori, 0,axis=0)
                self.training_delta=np.delete(self.training_delta,0,axis=0)
                self.training_dK=np.delete(self.training_dK,0,axis=0)
                self.training_gripper=np.delete(self.training_gripper,0, axis=0)

            else:
                self.recorded_traj=self.recorded_traj.transpose()
                self.recorded_ori=self.recorded_ori.transpose()
                self.recorded_joint=self.recorded_joint.transpose()
                self.recorded_gripper=self.recorded_gripper.transpose()
                
                self.training_traj=np.concatenate((self.training_traj,self.recorded_traj ), axis=0)
                self.training_ori=np.concatenate((self.training_ori,self.recorded_ori), axis=0)
                self.training_gripper=np.concatenate((self.training_gripper,self.recorded_gripper), axis=0)

                delta_x=resample(self.recorded_traj, step=2)
                self.training_delta=np.concatenate((self.training_delta,delta_x), axis=0)
                self.training_dK=np.concatenate((self.training_dK,np.zeros(np.shape(self.recorded_traj))), axis=0)
            print("Demo Saved")
        else:
            print("Demo Discarded")

    def Clear_Training_Data(self):
        self.training_traj = []
        self.training_ori  = []
        self.training_delta = []
        self.training_dK = []

    def save(self, data='last'):
        np.savez(str(pathlib.Path().resolve())+'/data/'+str(data)+'.npz', 
        nullspace_traj = self.nullspace_traj, 
        nullspace_joints = self.nullspace_joints, 
        training_traj = self.training_traj,
        training_ori = self.training_ori,
        training_delta = self.training_delta,
        training_gripper = self.training_gripper,
        training_dK = self.training_dK)
        print(np.shape(self.training_ori))
        print(np.shape(self.training_traj))  

    def load(self, file='last'):
        data =np.load(str(pathlib.Path().resolve())+'/data/'+str(file)+'.npz')

        self.nullspace_traj = data['nullspace_traj']
        self.nullspace_joints = data['nullspace_joints']
        self.training_traj = data['training_traj']
        self.training_ori = data['training_ori']
        self.training_delta = data['training_delta']
        self.training_dK = data['training_dK']
        self.training_gripper = data['training_gripper'] 
        self.nullspace_traj = self.nullspace_traj
        self.nullspace_joints = self.nullspace_joints
        self.training_traj = self.training_traj
        # print(np.shape(self.training_traj))
        self.training_ori = self.training_ori
        # print(np.shape(self.training_ori))
        self.training_delta = self.training_delta
        self.training_dK = self.training_dK

    def Train_GPs(self):
        if len(self.training_traj)>0 and len(self.training_delta)>0:
            print("Training of Delta")
            kernel = C(constant_value = 0.01, constant_value_bounds=[0.0005, self.attractor_lim]) * RBF(length_scale=[0.1, 0.1, 0.1], length_scale_bounds=[0.025, 0.2]) + WhiteKernel(0.00025, [0.0001, 0.0005]) 
            self.Delta=InteractiveGP(X=self.training_traj, Y=self.training_delta, y_lim=[-self.attractor_lim, self.attractor_lim], kernel=kernel, n_restarts_optimizer=20)
            self.Delta.fit()
            with open('models/delta.pkl','wb') as delta:
                pickle.dump(self.Delta,delta)

        else:
            raise TypeError("There are no data for learning a trajectory dynamical system")
        with open('models/delta.pkl','wb') as delta:
            pickle.dump(self.Delta,delta)

        if len(self.training_traj)>0 and len(self.training_dK)>0:
            print("Training of Stiffness")
            self.Stiffness=InteractiveGP(X=self.training_traj, Y=self.training_dK, y_lim=[self.K_min, self.K_max], kernel=self.Delta.kernel_, n_restarts_optimizer=0) 
            self.Stiffness.fit()
            with open('models/stiffness.pkl','wb') as stiffness:
                pickle.dump(self.Stiffness,stiffness)
        else:
            raise TypeError("There are no data for learning a stiffness dynamical system")

        if len(self.nullspace_traj)>0 and len(self.nullspace_joints)>0:
            print("Training of Nullspace")
            kernel = C(constant_value = 0.1, constant_value_bounds=[0.0005, self.attractor_lim]) * RBF(length_scale=[0.1, 0.1, 0.1], length_scale_bounds=[0.025, 0.1]) + WhiteKernel(0.00025, [0.0001, 0.0005]) 
            self.NullSpaceControl=InteractiveGP(X=self.nullspace_traj, Y=self.nullspace_joints, y_lim=[-self.attractor_lim, self.attractor_lim], kernel=kernel, n_restarts_optimizer=20)
            self.NullSpaceControl.fit()
            with open('models/nullspace.pkl','wb') as nullspace:
                pickle.dump(self.NullSpaceControl,nullspace)
        else: 
            print('No Null Space Control Policy Learned')    
    
    def save_models(self):
        with open('models/delta.pkl','wb') as delta:
            pickle.dump(self.Delta,delta)
        with open('models/stiffness.pkl','wb') as stiffness:
            pickle.dump(self.Stiffness,stiffness)
        if self.NullSpaceControl:
            with open('models/nullspace.pkl','wb') as nullspace:
                pickle.dump(self.NullSpaceControl,nullspace)

    def load_models(self):
        try:
            with open('models/delta.pkl', 'rb') as delta:
                self.Delta = pickle.load(delta)
        except:
            print("No delta model saved")
        try:
            with open('models/stiffness.pkl', 'rb') as stiffness:
                self.Stiffness = pickle.load(stiffness)
        except:
            print("No stiffness model saved")
        try:
            with open('models/nullspace.pkl', 'rb') as nullspace:
                self.NullSpace = pickle.load(nullspace)
        except:
            print("No NullSpace model saved")
    
    def find_alpha(self):
        alpha=np.zeros(len(self.Delta.X))
        for i in range(len(self.Delta.X)):         
            pos = self.Delta.X[i,:]+self.Delta.length_scales 
            dSigma_dx, dSigma_dy, dSigma_dz = self.Delta.var_gradient(pos.reshape(1,-1))                                                                                                                                                                
            alpha[i]=self.max_grad_force/ np.sqrt(dSigma_dx**2+dSigma_dy**2+dSigma_dz**2)
            self.alpha=np.min(alpha)

    def Interactive_Control(self, verboose=False):
        r = rospy.Rate(self.control_freq)
        self.find_alpha()
        
        self.wait_for_user_input()

        while not self.end:
            # read the actual position of the robot

            cart_pos=np.array(self.cart_pos).reshape(1,-1)
            # GP predictions Delta_x
            [self.delta, self.sigma, index_max_k_star]=self.Delta.predict(cart_pos)

            # GP prediction K stiffness
            [self.dK, _, _]=self.Stiffness.predict(cart_pos, return_std=False)
  
            self.delta = np.clip(self.delta[0], -self.attractor_lim, self.attractor_lim)

            self.dK = np.clip(self.dK[0], self.dK_min, self.dK_max)

            dSigma_dx, dSigma_dy, dSigma_dz = self.Delta.var_gradient(cart_pos)
            
            f_stable=-self.alpha*np.array([dSigma_dx, dSigma_dy, dSigma_dz])

            self.K_tot = np.clip(np.add(self.dK, self.K_mean), self.K_min, self.K_max)

            
            if any(np.abs(np.array(self.feedback)) > 0.05): # this avoids to activate the feedback on noise joystick

                print("Received Feedback")
                delta_inc, dK_inc = Interpret_3D(feedback=self.feedback, delta=self.delta, K=self.K_tot, delta_lim=self.attractor_lim, K_mean=self.K_mean)
                print('delta_inc')
                print(delta_inc)
                print("dK_inc")
                print(dK_inc)
                is_uncertain=self.Delta.is_uncertain(theta=self.theta)
                self.Delta.update_with_k(x=cart_pos, mu=self.delta, epsilon_mu=delta_inc, is_uncertain=is_uncertain)
                self.Stiffness.update_with_k(x=cart_pos, mu=self.dK, epsilon_mu=dK_inc, is_uncertain=is_uncertain)
        
            
            self.delta, self.K_tot = Force2Impedance(self.delta, self.K_tot, f_stable, self.attractor_lim)
            self.K_tot=[self.K_tot]
            K_ori_scaling=self.K_ori
            self.scaling_factor = (1- self.sigma / self.Delta.max_var) / (1 - self.theta_stiffness)
            if self.sigma / self.Delta.max_var > self.theta_stiffness: 
                self.K_tot=self.K_tot*self.scaling_factor
                K_ori_scaling= self.K_ori*self.scaling_factor
            x_new = cart_pos[0][0] + self.delta[0]  
            y_new = cart_pos[0][1] + self.delta[1]  
            z_new = cart_pos[0][2] + self.delta[2]  

            quat_goal=self.training_ori[index_max_k_star,:]
            gripper_goal=self.training_gripper[index_max_k_star,0]

            quat_goal=slerp_sat(self.cart_ori, quat_goal, 0.1)

            pos_goal=[x_new, y_new, z_new]
            self.set_attractor(pos_goal,quat_goal)
            self.move_gripper(gripper_goal)
            #self.grasp_gripper(gripper_goal)
            null_stiff = [0]

            if self.NullSpaceControl:
                [self.equilibrium_configuration, self.sigma_null_space]=self.NullSpaceControl.predict(np.array(pos_goal).reshape(1,-1))
                self.scaling_factor_ns = (1-self.sigma_null_space / self.NullSpaceControl.max_var) / (1 - self.theta_nullspace)
                
                if self.sigma_null_space / self.NullSpaceControl.max_var > self.theta_nullspace:
                    self.null_stiff=self.K_null*self.scaling_factor_ns
                else:
                    self.null_stiff=self.K_null      
                self.set_configuration(self.equilibrium_configuration[0])
                null_stiff = [self.null_stiff]
            
            pos_stiff = [self.K_tot[0][0],self.K_tot[0][1],self.K_tot[0][2]]
            rot_stiff = [K_ori_scaling,K_ori_scaling,K_ori_scaling]#[self.K_ori, self.K_ori, self.K_ori]
            self.set_stiffness(pos_stiff, rot_stiff, null_stiff)
            if verboose :
                print("Delta")
                print(self.delta)
                print("Stabilization field")
                print(f_stable)
                print("Scaling_factor_cartesian:" + str(self.scaling_factor))
                print("Scaling_factor_nullspace:" + str(self.scaling_factor_ns))   
            r.sleep()

if __name__ == '__main__':
    rospy.sleep(1)
    ilosa = ILoSA()
    rospy.init_node('ILoSA', anonymous=False)
    ilosa.connect_ROS()
    rospy.on_shutdown(ilosa.disconnect_ROS)

    while not rospy.is_shutdown():
        rospy.sleep(1)